"""Activation extraction for K-means initialization.

This module provides :class:`ActivationExtractor`, which wraps a
:class:`SAETrainingPipeline` to extract activations from a trained model
and save them as chunked ``.pt`` files for subsequent K-means training.

The heavy lifting is delegated to the inference-based extractor already
present in ``scripts/kmeans/core/extract_activations_for_kmeans.py``; this
class adds a higher-level, config-driven interface suitable for use inside
:class:`~src.core.sae.kmeans.pipeline.KMeansPipeline`.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
import torch
import yaml

from .utils import sanitize_layer_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so script-level imports work
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class ActivationExtractor:
    """Config-driven activation extraction for K-means initialisation.

    Parameters
    ----------
    config : dict
        Full training config (the same YAML dict passed to
        :class:`SAETrainingPipeline`).
    output_dir : str | Path
        Root directory to write per-layer activation chunks.
    layers : list[str]
        Layer names to extract.
    target_tokens : int
        Total token budget for the *primary* layer.
    primary_layer : str | None
        Reference layer used to compute target inference count.  Defaults
        to the first entry in *layers*.
    flush_every_tokens : int
        Buffer size (in tokens) before flushing to disk.
    checkpoint_every : int
        Save a JSON checkpoint every *N* inferences.
    auto_probe : bool
        If ``True``, run a probe pass to measure tokens-per-inference.
    num_probe_batches : int
        Number of batches for the auto-probe pass.
    rank : int
        DDP rank.
    world_size : int
        DDP world size.
    """

    def __init__(
        self,
        config: dict,
        output_dir: str | Path,
        layers: list[str],
        target_tokens: int = 10_000_000,
        primary_layer: str | None = None,
        flush_every_tokens: int = 16384,
        checkpoint_every: int = 100,
        auto_probe: bool = True,
        num_probe_batches: int = 10,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.layers = list(layers)
        self.target_tokens = target_tokens
        self.primary_layer = primary_layer or self.layers[0]
        self.flush_every_tokens = flush_every_tokens
        self.checkpoint_every = checkpoint_every
        self.auto_probe = auto_probe
        self.num_probe_batches = num_probe_batches
        self.rank = rank
        self.world_size = world_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise model, dataset, and activation store.

        Must be called before :meth:`extract`.
        """
        from src.core.sae.train.runner import SAETrainingPipeline

        # SAETrainingPipeline expects a file path, so serialize dict configs
        # to a temporary YAML file.
        if isinstance(self.config, dict):
            self._tmp_config = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False,
            )
            yaml.safe_dump(self.config, self._tmp_config)
            self._tmp_config.close()
            config_path = self._tmp_config.name
        else:
            config_path = self.config
            self._tmp_config = None

        self._pipeline = SAETrainingPipeline(
            config_path=config_path,
            rank=self.rank,
            world_size=self.world_size,
        )

        self._pipeline.model = self._pipeline._load_model()
        self._pipeline._load_dataset()

        # Disable hook-point validation for extraction (we specify layers manually)
        self._pipeline.config["sae"]["validate_hook_points"] = {"enabled": False}

        self._pipeline._create_activation_store()
        store = self._pipeline.activation_store

        # Set expanded hook points manually (skip discovery)
        store.expanded_hook_points = list(self.layers)
        store._probe_ran = True

        # Initialise queues
        self._ensure_queues(store)

        # Layer ownership for DDP
        if self.world_size > 1:
            owners = {ln: i % self.world_size for i, ln in enumerate(self.layers)}
            store.set_layer_owners(owners)

        logger.info(
            "[Rank %d] ActivationExtractor.setup() complete (%d layers)",
            self.rank,
            len(self.layers),
        )

    def extract(self) -> Path:
        """Run the extraction loop and return *output_dir*.

        Returns
        -------
        Path
            The directory containing per-layer activation chunks.
        """
        # Import the script-level extractor implementation
        from scripts.kmeans.core.extract_activations_for_kmeans import (
            InferenceBasedExtractor,
            auto_calculate_subsample_rates,
            probe_tokens_per_inference,
        )

        store = self._pipeline.activation_store
        per_layer_cfg = self.config.get("sae", {}).get("training", {}).get("per_layer", {})

        # Determine subsample rates from config
        subsample_rates: dict[str, float] = {}
        for ln in self.layers:
            rate = per_layer_cfg.get(ln, {}).get("random_subsample_rate", 1.0)
            subsample_rates[ln] = rate

        # Auto-probe if requested
        tokens_per_inference: dict[str, float] | None = None
        if self.auto_probe:
            # Temporarily set subsample=1.0 for probing
            for ln in self.layers:
                store.per_layer.setdefault(ln, {})["random_subsample_rate"] = 1.0

            tokens_per_inference = probe_tokens_per_inference(
                store,
                self.layers,
                num_probe_batches=self.num_probe_batches,
                rank=self.rank,
                world_size=self.world_size,
            )

            # If no manual subsample rates, auto-calculate
            primary_sub = per_layer_cfg.get(self.primary_layer, {}).get(
                "random_subsample_rate", 0.0625
            )
            auto_rates = auto_calculate_subsample_rates(
                tokens_per_inference,
                primary_layer=self.primary_layer,
                primary_subsample=primary_sub,
            )
            subsample_rates.update(auto_rates)

            # Apply calculated rates back to store
            for ln, rate in subsample_rates.items():
                store.per_layer.setdefault(ln, {})["random_subsample_rate"] = rate

        # Build the inference-based extractor
        extractor = InferenceBasedExtractor(
            activation_store=store,
            output_dir=self.output_dir,
            primary_layer=self.primary_layer,
            target_tokens_primary=self.target_tokens,
            layers=self.layers,
            subsample_rates=subsample_rates,
            flush_every_tokens=self.flush_every_tokens,
            rank=self.rank,
            world_size=self.world_size,
            file_format="pt",
        )

        if tokens_per_inference:
            for ln, tpi in tokens_per_inference.items():
                extractor.layer_state.setdefault(ln, {})["tokens_per_inference"] = tpi
            extractor.target_inferences = extractor.calculate_target_inferences(
                tokens_per_inference
            )

        # Prefill
        store.collect_round(n_batches=2)

        # Run
        extractor.extract(save_checkpoint_every=self.checkpoint_every)
        logger.info(
            "[Rank %d] Extraction complete -> %s", self.rank, self.output_dir
        )
        return self.output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release resources (GPU memory, temp files, etc.)."""
        # Free GPU-heavy resources (model, activation store)
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            if hasattr(pipeline, "model"):
                del pipeline.model
            if hasattr(pipeline, "activation_store"):
                del pipeline.activation_store
            del self._pipeline

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                "[Rank %d] GPU memory released after extraction.", self.rank
            )

        if getattr(self, "_tmp_config", None) is not None:
            try:
                os.unlink(self._tmp_config.name)
            except OSError:
                pass
            self._tmp_config = None

    def _ensure_queues(self, store) -> None:
        """Create TokenBlockQueue instances for all layers if missing."""
        from src.core.sae.activation_stores.universal_activation_store import (
            TokenBlockQueue,
        )

        for lname in self.layers:
            if lname not in store.queues:
                cap_blocks = store.in_memory_blocks_per_layer
                spill_here = store.spill_dir if store.spill_to_disk else None
                allow_gpu = not store.buffer_on_cpu
                store.queues[lname] = TokenBlockQueue(
                    block_size_tokens=store.block_size_tokens,
                    spill_dir=spill_here,
                    in_memory_blocks_cap=cap_blocks,
                    lname=lname,
                    allow_gpu=allow_gpu,
                )


__all__ = ["ActivationExtractor"]
