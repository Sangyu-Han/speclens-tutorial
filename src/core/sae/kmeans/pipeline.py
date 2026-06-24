"""End-to-end K-means initialization pipeline for RA-SAE.

Orchestrates the full workflow:
    1. **Extraction** -- collect activations from a trained model.
    2. **K-means training** -- fit Faiss K-means on each layer.
    3. **Centroid saving** -- write ``centroids.pt`` files that the training
       runner can load during SAE initialisation.

Each phase can be run independently::

    pipe = KMeansPipeline("configs/sam2_sav_ra-ar_train.yaml")

    # Phase 1 only (DDP, torchrun)
    pipe.run_extraction()

    # Phase 2 only (single process, Faiss multi-GPU)
    pipe.run_kmeans()

    # Both phases
    pipe.run()

See also :mod:`scripts.run_kmeans` for a CLI entry-point.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import yaml

from .extractor import ActivationExtractor
from .trainer import KMeansTrainer
from .utils import sanitize_layer_name, centroids_path

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

__all__ = ["KMeansPipeline"]


class KMeansPipeline:
    """Orchestrate activation extraction, K-means training, and centroid saving.

    Parameters
    ----------
    config_path_or_dict : str | Path | dict
        Path to a YAML config file **or** a pre-loaded config dictionary.
        The config must contain ``sae.layers`` and ``sae.training`` sections.
    rank : int
        DDP rank (default ``0`` for single-GPU).
    world_size : int
        DDP world size (default ``1``).
    """

    def __init__(
        self,
        config_path_or_dict: str | Path | dict,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.config = self._load_config(config_path_or_dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_extraction(
        self,
        layers: list[str] | None = None,
        target_tokens: int | None = None,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Phase 1: Extract activations from the model.

        All DDP ranks participate in extraction.  GPU memory is released
        when this method returns.

        Parameters
        ----------
        layers : list[str] | None
            Override for layers to process.  If ``None``, resolved from config.
        target_tokens : int | None
            Override for total token budget.  If ``None``, resolved from config.
        output_dir : str | Path | None
            Override for the activation data directory.

        Returns
        -------
        Path
            The directory containing per-layer activation chunks.
        """
        layers = self._resolve_layers(layers)
        kmeans_cfg = self._kmeans_cfg()

        target_tokens = target_tokens or int(
            kmeans_cfg.get("target_tokens", 10_000_000)
        )
        data_dir = Path(
            output_dir or kmeans_cfg.get("data_dir", "outputs/kmeans_activations")
        )
        primary_layer: str = kmeans_cfg.get("primary_layer") or layers[0]
        flush_every: int = int(kmeans_cfg.get("flush_every_tokens", 16384))
        checkpoint_every: int = int(kmeans_cfg.get("checkpoint_every", 100))
        auto_probe: bool = bool(kmeans_cfg.get("auto_probe", True))

        logger.info("=" * 72)
        logger.info("Phase 1: Extracting activations")
        logger.info("  Layers         : %d", len(layers))
        logger.info("  Target tokens  : %s", f"{target_tokens:,}")
        logger.info("  Data dir       : %s", data_dir)
        logger.info("  Primary layer  : %s", primary_layer)
        logger.info("=" * 72)

        extractor = ActivationExtractor(
            config=self.config,
            output_dir=data_dir,
            layers=layers,
            target_tokens=target_tokens,
            primary_layer=primary_layer,
            flush_every_tokens=flush_every,
            checkpoint_every=checkpoint_every,
            auto_probe=auto_probe,
            rank=self.rank,
            world_size=self.world_size,
        )
        extractor.setup()
        try:
            extractor.extract()
        finally:
            extractor.cleanup()

        logger.info("Phase 1 complete -> %s", data_dir)
        return data_dir

    def run_kmeans(
        self,
        layers: list[str] | None = None,
        data_dir: str | Path | None = None,
        centroids_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Phase 2: Train K-means on extracted activations.

        Does **not** require DDP — runs as a single process.
        Faiss ``gpu=True`` internally uses all visible GPUs.

        Parameters
        ----------
        layers : list[str] | None
            Override for layers to process.  If ``None``, resolved from config.
        data_dir : str | Path | None
            Directory containing per-layer activation chunks (output of
            :meth:`run_extraction`).
        centroids_dir : str | Path | None
            Directory to save per-layer ``centroids.pt`` files.

        Returns
        -------
        dict[str, Path]
            Mapping from layer name to the ``centroids.pt`` file that was saved.
        """
        layers = self._resolve_layers(layers)
        kmeans_cfg = self._kmeans_cfg()

        data_dir = Path(
            data_dir or kmeans_cfg.get("data_dir", "outputs/kmeans_activations")
        )
        centroids_dir = Path(
            centroids_dir or kmeans_cfg.get("centroids_dir", "outputs/kmeans_centers")
        )
        n_init: int = int(kmeans_cfg.get("n_init", 2))
        max_iter: int = int(kmeans_cfg.get("max_iter", 20))
        max_samples: int = int(kmeans_cfg.get("max_samples", 10_000_000))
        seed: int = int(
            self.config.get("sae", {}).get("training", {}).get("seed", 42)
        )

        logger.info("=" * 72)
        logger.info("Phase 2: Training K-means")
        logger.info("  Layers         : %d", len(layers))
        logger.info("  Data dir       : %s", data_dir)
        logger.info("  Centroids dir  : %s", centroids_dir)
        logger.info("  n_init=%d  max_iter=%d  max_samples=%s",
                     n_init, max_iter, f"{max_samples:,}")
        logger.info("=" * 72)

        results: dict[str, Path] = {}

        for layer_name in layers:
            layer_safe = sanitize_layer_name(layer_name)
            layer_data_dir = data_dir / layer_safe

            if not layer_data_dir.exists():
                logger.warning(
                    "Skipping %s: data directory %s does not exist.",
                    layer_name,
                    layer_data_dir,
                )
                continue

            # Peek at first chunk to determine act_size
            act_size = self._peek_act_size(layer_data_dir)
            if act_size is None:
                logger.warning(
                    "Skipping %s: could not determine act_size from %s.",
                    layer_name,
                    layer_data_dir,
                )
                continue

            n_clusters = self._resolve_n_clusters(layer_name, act_size)

            logger.info(
                "  Training K-means for %s: n_clusters=%d, act_size=%d",
                layer_name,
                n_clusters,
                act_size,
            )

            trainer = KMeansTrainer(
                data_dir=layer_data_dir,
                n_clusters=n_clusters,
                n_init=n_init,
                max_iter=max_iter,
                max_samples=max_samples,
                seed=seed,
            )
            trained_centroids, _ = trainer.train()

            # Save centroids
            out_path = centroids_path(centroids_dir, layer_name)
            trainer.save(trained_centroids, act_size, out_path)
            results[layer_name] = out_path

            logger.info(
                "  -> Centroids saved: %s (%d clusters)",
                out_path,
                n_clusters,
            )

        logger.info("=" * 72)
        logger.info("Phase 2 complete.  %d / %d layers processed.",
                     len(results), len(layers))
        for layer_name, path in results.items():
            logger.info("  %s -> %s", layer_name, path)
        logger.info("=" * 72)

        return results

    def run(
        self,
        layers: list[str] | None = None,
        target_tokens: int | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Run the full pipeline: extract -> train K-means -> save centroids.

        Phase 1 (extraction) uses all DDP ranks.  After extraction, GPU
        memory is freed on all ranks.  Phase 2 (K-means) runs on rank 0
        only — Faiss ``gpu=True`` leverages all visible GPUs.

        Parameters
        ----------
        layers : list[str] | None
            Override for layers to process.  If ``None``, resolved from config.
        target_tokens : int | None
            Override for total token budget.  If ``None``, resolved from config.
        output_dir : str | Path | None
            Override for the activation data directory.  If ``None``, resolved
            from config.

        Returns
        -------
        dict[str, Path]
            Mapping from layer name to the ``centroids.pt`` file that was saved.
        """
        layers = self._resolve_layers(layers)
        kmeans_cfg = self._kmeans_cfg()
        data_dir = Path(
            output_dir or kmeans_cfg.get("data_dir", "outputs/kmeans_activations")
        )

        # Phase 1 -- Extraction (all ranks)
        self.run_extraction(
            layers=layers,
            target_tokens=target_tokens,
            output_dir=data_dir,
        )

        # Sync all ranks after extraction (GPU memory already freed by
        # extractor.cleanup → torch.cuda.empty_cache).
        if self.world_size > 1:
            import torch.distributed as dist
            dist.barrier()
            logger.info(
                "[Rank %d] All ranks synced after extraction.", self.rank
            )

        # Phase 2 -- K-means (rank 0 only, Faiss uses all visible GPUs)
        results: dict[str, Path] = {}
        if self.rank == 0:
            results = self.run_kmeans(layers=layers, data_dir=data_dir)

        # Sync so all ranks wait for rank 0's K-means to finish
        if self.world_size > 1:
            dist.barrier()

        return results

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_n_clusters(self, layer_name: str, act_size: int) -> int:
        """Determine the number of K-means clusters for a layer.

        Resolution order:
            1. ``per_layer[layer_name]["kmeans_n_clusters"]`` if explicitly set.
            2. ``expansion_factor * act_size`` (per-layer override or global).

        Parameters
        ----------
        layer_name : str
            Fully-qualified layer name.
        act_size : int
            Activation dimensionality.

        Returns
        -------
        int
            Number of clusters.
        """
        tr = self.config.get("sae", {}).get("training", {})
        per_layer = tr.get("per_layer", {}).get(layer_name, {})

        # Explicit override
        explicit = per_layer.get("kmeans_n_clusters")
        if explicit is not None:
            return int(explicit)

        # Derive from expansion factor
        expansion = per_layer.get(
            "expansion_factor",
            tr.get("expansion_factor", 8),
        )
        return int(expansion) * act_size

    def _resolve_layers(self, override: list[str] | None = None) -> list[str]:
        """Get the list of layers to process.

        Parameters
        ----------
        override : list[str] | None
            If provided, returned as-is.  Otherwise reads
            ``config["sae"]["layers"]``.

        Returns
        -------
        list[str]
        """
        if override:
            return list(override)
        layers = self.config.get("sae", {}).get("layers", [])
        if not layers:
            raise ValueError(
                "No layers specified.  Pass them via `layers=` argument or "
                "set `sae.layers` in the config."
            )
        return list(layers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kmeans_cfg(self) -> dict:
        """Return the ``sae.training.kmeans_init`` sub-dict (or empty)."""
        return (
            self.config.get("sae", {})
            .get("training", {})
            .get("kmeans_init", {})
        )

    @staticmethod
    def _load_config(config_path_or_dict: str | Path | dict) -> dict:
        """Load and validate the configuration.

        Accepts either a filesystem path to a YAML file or an already-parsed
        dictionary.
        """
        if isinstance(config_path_or_dict, dict):
            return config_path_or_dict

        path = Path(config_path_or_dict)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh)

        if cfg is None or not isinstance(cfg, dict):
            raise ValueError(f"Invalid config (must be a YAML dict): {path}")

        return cfg

    @staticmethod
    def _peek_act_size(layer_data_dir: Path) -> int | None:
        """Read the first chunk file to determine activation dimensionality."""
        import torch

        pt_files = sorted(layer_data_dir.glob("chunk_*.pt"))
        if pt_files:
            chunk = torch.load(pt_files[0], map_location="cpu", weights_only=False)
            return int(chunk.shape[-1])

        parquet_files = sorted(layer_data_dir.glob("chunk_*.parquet"))
        if parquet_files:
            try:
                import pyarrow.parquet as pq
                import pyarrow as pa

                schema = pq.read_schema(parquet_files[0])
                field = schema.field("activations")
                if pa.types.is_fixed_size_list(field.type):
                    return field.type.list_size
                # Fallback: read one row
                table = pq.read_table(parquet_files[0]).slice(0, 1)
                return len(table.column("activations")[0].as_py())
            except Exception as exc:
                logger.warning("Could not read act_size from parquet: %s", exc)
                return None

        return None
