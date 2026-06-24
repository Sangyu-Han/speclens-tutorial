from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import cm as mpl_cm


_BERLIN_CMAP: Optional[LinearSegmentedColormap] = None


def get_berlin_cmap() -> LinearSegmentedColormap:
    """Return a lazily-initialised diverging colormap similar to "Berlin"."""
    global _BERLIN_CMAP
    if _BERLIN_CMAP is None:
        berlin_colors = [
            "#1f004b",
            "#2c2e9f",
            "#285bc4",
            "#2b82d1",
            "#5ab4d4",
            "#9bd8d7",
            "#e0f3f8",
            "#f4d4bf",
            "#e99f82",
            "#d95c4d",
            "#af3030",
            "#7f0000",
        ]
        _BERLIN_CMAP = LinearSegmentedColormap.from_list("berlin_custom", berlin_colors, N=256)
    return _BERLIN_CMAP


def apply_diverging_cmap(array: np.ndarray) -> np.ndarray:
    """Map a 2D array in [-1, 1] style ranges to RGB values via a diverging colormap."""
    if array.size == 0:
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    arr = array.astype(np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if math.isclose(vmin, vmax, rel_tol=1e-6, abs_tol=1e-6):
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    vmin = min(vmin, 0.0)
    vmax = max(vmax, 0.0)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax if vmax > 0 else 0.0)
    cmap = get_berlin_cmap()
    heat_rgba = cmap(norm(arr))
    heat_rgb = np.clip(heat_rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return heat_rgb


def apply_named_cmap(array: np.ndarray, cmap_name: str) -> np.ndarray:
    if not cmap_name or cmap_name.lower() in {"berlin", "berlin_custom"}:
        return apply_diverging_cmap(array)
    if array.size == 0:
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    arr = array.astype(np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if math.isclose(vmin, vmax, rel_tol=1e-6, abs_tol=1e-6):
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    arr_norm = (arr - vmin) / (vmax - vmin + 1e-8)
    arr_norm = np.power(np.clip(arr_norm, 0.0, 1.0), 0.5)
    cmap = mpl_cm.get_cmap(cmap_name)
    heat_rgba = cmap(arr_norm)
    heat_rgb = np.clip(heat_rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return heat_rgb


def blend_heatmap(base_rgb: np.ndarray, heat_tensor: torch.Tensor, alpha: float, cmap_name: str = "berlin") -> np.ndarray:
    """Overlay a heat tensor onto an RGB image using the requested colormap."""
    alpha = float(max(0.0, min(1.0, alpha)))
    heat_rgb = apply_named_cmap(heat_tensor.detach().cpu().numpy(), cmap_name)
    if base_rgb.shape[:2] != heat_rgb.shape[:2]:
        from PIL import Image

        heat_rgb = np.array(
            Image.fromarray(heat_rgb).resize((base_rgb.shape[1], base_rgb.shape[0]), Image.BILINEAR)
        )
    blended = alpha * heat_rgb.astype(np.float32) + (1.0 - alpha) * base_rgb.astype(np.float32)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
