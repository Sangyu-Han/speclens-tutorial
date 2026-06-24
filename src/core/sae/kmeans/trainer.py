"""K-means trainer for SAE centroid initialization.

This module provides :class:`KMeansTrainer`, a thin wrapper around
:class:`FaissKMeansTrainer` (from ``scripts/kmeans/core/train_kmeans_centers.py``)
that offers a config-driven interface consumable by
:class:`~src.core.sae.kmeans.pipeline.KMeansPipeline`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class KMeansTrainer:
    """Config-driven K-means centroid training for a single layer.

    Parameters
    ----------
    data_dir : str | Path
        Directory that contains chunked activation files (``chunk_*.pt``
        or ``chunk_*.parquet``) for the target layer.
    n_clusters : int
        Number of centroids to learn.
    n_init : int
        Number of K-means restarts (``nredo`` in Faiss).
    max_iter : int
        Maximum iterations per restart.
    max_samples : int
        Cap on training samples loaded into memory.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        data_dir: str | Path,
        n_clusters: int,
        n_init: int = 2,
        max_iter: int = 20,
        max_samples: int = 10_000_000,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.max_samples = max_samples
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> tuple[torch.Tensor, int]:
        """Train K-means and return ``(centroids, act_size)``.

        Returns
        -------
        centroids : torch.Tensor
            Centroid matrix of shape ``[n_clusters, act_size]``.
        act_size : int
            Activation dimensionality (columns of *centroids*).
        """
        from scripts.kmeans.core.train_kmeans_centers import FaissKMeansTrainer

        faiss_trainer = FaissKMeansTrainer(
            data_dir=str(self.data_dir),
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            seed=self.seed,
        )
        centroids, act_size = faiss_trainer.train(max_samples=self.max_samples)
        logger.info(
            "K-means training complete: n_clusters=%d, act_size=%d",
            self.n_clusters,
            act_size,
        )
        return centroids, act_size

    def save(
        self,
        centroids: torch.Tensor,
        act_size: int,
        output_path: str | Path,
    ) -> Path:
        """Save centroids checkpoint to *output_path*.

        The checkpoint is a dict with keys ``centroids``, ``n_clusters``,
        ``act_size``, and ``config``.

        Parameters
        ----------
        centroids : torch.Tensor
            Centroid matrix ``[n_clusters, act_size]``.
        act_size : int
            Activation dimensionality.
        output_path : str | Path
            Where to save the ``.pt`` file.

        Returns
        -------
        Path
            The resolved output path.
        """
        import time

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "centroids": centroids,
            "n_clusters": self.n_clusters,
            "act_size": act_size,
            "trained_at": time.time(),
            "config": {
                "n_init": self.n_init,
                "max_iter": self.max_iter,
                "max_samples": self.max_samples,
                "seed": self.seed,
            },
        }
        torch.save(save_dict, output_path)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("Centroids saved to %s (%.1f MB)", output_path, file_size_mb)
        return output_path

    def load_activations(self) -> tuple[torch.Tensor, int]:
        """Load activation chunks from *data_dir* using the Faiss trainer loader.

        This is a convenience method that delegates to the Faiss trainer's
        loading logic (which handles both ``.pt`` and ``.parquet`` formats).

        Returns
        -------
        data : torch.Tensor
            Loaded activations ``[n_samples, act_size]``.
        act_size : int
            Activation dimensionality.
        """
        from scripts.kmeans.core.train_kmeans_centers import FaissKMeansTrainer

        faiss_trainer = FaissKMeansTrainer(
            data_dir=str(self.data_dir),
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            seed=self.seed,
        )
        fmt = faiss_trainer.file_format
        if fmt == "pt":
            return faiss_trainer._load_pt(max_samples=self.max_samples)
        else:
            return faiss_trainer._load_parquet(max_samples=self.max_samples)


__all__ = ["KMeansTrainer"]
