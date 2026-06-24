"""Shared utilities for K-means initialization pipeline.

Provides filesystem-safe layer name sanitization, centroid path resolution,
and centroid loading functions used by both the K-means pipeline and the
SAE training runner.

Convention: ``/`` -> ``_``, ``:`` -> ``__`` (matches the extraction script
in ``scripts/kmeans/core/extract_activations_for_kmeans.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

__all__ = ["sanitize_layer_name", "centroids_path", "load_centroids"]


def sanitize_layer_name(layer: str) -> str:
    """Convert a layer name to a filesystem-safe directory/file name.

    Rules (matching the extraction script convention):
        * ``/`` is replaced with ``_``
        * ``:`` is replaced with ``__``
        * ``@`` is kept as-is (safe on all major filesystems)

    Args:
        layer: Full layer name, e.g. ``"model.image_encoder.trunk@3"``.

    Returns:
        Sanitized string, e.g. ``"model.image_encoder.trunk@3"``.
    """
    return layer.replace("/", "_").replace(":", "__")


def centroids_path(base_dir: str | Path, layer_name: str) -> Path:
    """Return the canonical path to a centroid file for a given layer.

    The path follows the convention used by the orchestration shell script::

        {base_dir}/{sanitize_layer_name(layer_name)}/centroids.pt

    Args:
        base_dir: Root directory containing per-layer centroid subdirectories.
        layer_name: Full layer name (will be sanitized).

    Returns:
        ``Path`` pointing to the expected ``centroids.pt`` file.
    """
    return Path(base_dir) / sanitize_layer_name(layer_name) / "centroids.pt"


def load_centroids(path: str | Path) -> torch.Tensor:
    """Load centroid tensor from a ``centroids.pt`` checkpoint.

    The checkpoint is expected to contain at least a ``"centroids"`` key
    mapping to a ``[n_clusters, act_size]`` tensor.

    Args:
        path: Path to the ``centroids.pt`` file.

    Returns:
        ``torch.Tensor`` of shape ``[n_clusters, act_size]``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the checkpoint lacks a ``"centroids"`` key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Centroid file not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "centroids" not in ckpt:
        raise KeyError(
            f"Centroid checkpoint at {path} does not contain a 'centroids' key. "
            f"Available keys: {list(ckpt.keys())}"
        )

    centroids = ckpt["centroids"]
    logger.debug(
        "Loaded centroids from %s: shape=%s, dtype=%s",
        path, centroids.shape, centroids.dtype,
    )
    return centroids
