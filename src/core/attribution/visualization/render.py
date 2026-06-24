from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from src.core.attribution.visualization.panels import (
    save_anchor_heatmap_overlay,
    save_feature_activation_overlay,
    save_output_contribution_overlay,
)


def render_anchor_heatmaps(
    *,
    out_dir: Path,
    sid: int,
    frames: Sequence[torch.Tensor],
    heatmaps: Dict[str, torch.Tensor],
    overlay_kwargs: Dict[str, float],
    prompt_points: Optional[Sequence[tuple[int, int]]] = None,
    use_abs: bool = False,
    feature_points: Optional[Sequence[tuple[int, int]]] = None,
    feature_point_frame_idx: Optional[int] = None,
    patch_bboxes: Optional[Dict[str, Optional[Sequence[tuple[int, int, int, int]]]]] = None,
) -> None:
    prompt_pts = list(prompt_points or [])
    overlay_alpha = float(overlay_kwargs.get("alpha", 0.4))
    overlay_cmap = overlay_kwargs.get("anchor_cmap", overlay_kwargs.get("cmap", "hot"))
    for name, tensor in heatmaps.items():
        heat_stack = tensor.detach().cpu() if torch.is_tensor(tensor) else torch.tensor(tensor)
        boxes = patch_bboxes.get(name) if patch_bboxes else None
        try:
            save_anchor_heatmap_overlay(
                out_dir=out_dir,
                sid=sid,
                anchor_name=name,
                score_suffix="",
                heat_stack=heat_stack,
                frames=frames,
                prompt_points=prompt_pts,
                overlay_alpha=overlay_alpha,
                overlay_cmap=overlay_cmap,
                use_abs_overlay=use_abs,
                feature_points=list(feature_points) if feature_points else None,
                feature_point_frame_idx=feature_point_frame_idx,
                patch_bboxes=boxes,
            )
        except Exception:
            continue


def render_feature_activation_map(
    *,
    out_dir: Path,
    sid: int,
    frames: Sequence[torch.Tensor],
    feature_map: torch.Tensor,
    overlay_kwargs: Dict[str, float],
    file_stub: str,
) -> None:
    feature_alpha = float(overlay_kwargs.get("feature_map_alpha", overlay_kwargs.get("alpha", 0.4)))
    feature_min_abs = float(overlay_kwargs.get("feature_map_min_abs", 0.0))
    feature_cmap = overlay_kwargs.get("feature_map_cmap", overlay_kwargs.get("cmap", "plasma"))
    try:
        save_feature_activation_overlay(
            out_dir=out_dir,
            sid=sid,
            score_suffix="",
            map_stack=feature_map.cpu() if torch.is_tensor(feature_map) else torch.tensor(feature_map),
            frames=frames,
            prompt_points=[],
            overlay_alpha=feature_alpha,
            overlay_cmap=feature_cmap,
            file_stub=file_stub,
            min_abs=feature_min_abs,
        )
    except Exception:
        return


def render_output_contribution_overlays(
    *,
    out_dir: Path,
    sid: int,
    frames: Sequence[torch.Tensor],
    heatmaps: Dict[str, torch.Tensor],
    contributions: Dict[str, Dict[str, torch.Tensor]],
    overlay_kwargs: Dict[str, float],
    prompt_points: Optional[Sequence[tuple[int, int]]] = None,
    feature_points: Optional[Sequence[tuple[int, int]]] = None,
    feature_point_frame_idx: Optional[int] = None,
    score_suffix: str = "",
    sigmoid_specs: Optional[Sequence[str]] = None,
) -> None:
    overlay_alpha = float(overlay_kwargs.get("alpha", 0.4))
    overlay_cmap = overlay_kwargs.get("mask_cmap", overlay_kwargs.get("output_cmap", "bwr"))
    prompt_pts = list(prompt_points or [])
    sigmoid_set = set(sigmoid_specs or [])
    for name, heat_stack in heatmaps.items():
        payload = contributions.get(name)
        if not payload:
            continue
        alpha_entry = payload.get("alpha1")
        if isinstance(alpha_entry, dict):
            alpha_tensor = alpha_entry.get(name)
        else:
            alpha_tensor = alpha_entry
        if alpha_tensor is None:
            baseline_entry = payload.get("baseline")
            if isinstance(baseline_entry, dict):
                alpha_tensor = baseline_entry.get(name)
            else:
                alpha_tensor = baseline_entry
        if alpha_tensor is None:
            continue
        tensor = heat_stack.detach().cpu() if torch.is_tensor(heat_stack) else torch.tensor(heat_stack)
        try:
            save_output_contribution_overlay(
                out_dir=out_dir,
                sid=sid,
                score_suffix=score_suffix,
                heat_stack=tensor,
                target_tensor=alpha_tensor,
                prompt_points=prompt_pts,
                overlay_alpha=overlay_alpha,
                overlay_cmap=overlay_cmap,
                use_abs_overlay=False,
                feature_points=list(feature_points) if feature_points else None,
                feature_point_frame_idx=feature_point_frame_idx,
                apply_sigmoid=name in sigmoid_set,
                file_stub=name,
            )
        except Exception:
            continue


__all__ = [
    "render_anchor_heatmaps",
    "render_feature_activation_map",
    "render_output_contribution_overlays",
]
