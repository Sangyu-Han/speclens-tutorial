from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from matplotlib import cm as mpl_cm

from src.core.attribution.visualization.color_maps import blend_heatmap

PromptPoint = Tuple[int, int]

# RGB 배열 위에 패치 테두리를 그린다.
def _draw_patch_outline(image: np.ndarray, left: int, top: int, width: int, height: int, color: Tuple[int, int, int] = (0, 0, 0), thickness: int = 3) -> None:
    import cv2
    right = min(image.shape[1] - 1, left + width)
    bottom = min(image.shape[0] - 1, top + height)
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness=thickness)


def _draw_red_dot(image: np.ndarray, x: int, y: int, radius: int = 6) -> None:
    import cv2
    cv2.circle(image, (x, y), radius, (255, 0, 0), thickness=-1)
    cv2.circle(image, (x, y), radius + 4, (255, 255, 255), thickness=2)

def _draw_prompt_points(
    image: np.ndarray,
    prompt_points: Sequence[PromptPoint],
    *,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 10,
) -> None:
    if not prompt_points:
        return
    import cv2

    for (px, py) in prompt_points:
        cv2.circle(image, (px, py), int(radius), color, thickness=-1)
        cv2.circle(image, (px, py), int(radius) + 4, (255, 255, 255), thickness=2)


def save_anchor_heatmap_overlay(
    *,
    out_dir: Path,
    sid: int,
    anchor_name: str,
    score_suffix: str,
    heat_stack: torch.Tensor,
    frames: Sequence[torch.Tensor],
    prompt_points: Sequence[PromptPoint],
    prompt_color: Tuple[int, int, int] = (0, 0, 255),
    overlay_alpha: float,
    overlay_cmap: str,
    use_abs_overlay: bool,
    feature_points: Optional[List[Tuple[int,int]]] = None,             # [NEW - kept for backwards compatibility]
    feature_point_frame_idx: Optional[int] = None,
    patch_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> None:
    from src.utils.render import make_ERF_figure, torch_to_image

    overlays = []
    for t in range(heat_stack.size(0)):
        img = torch_to_image(frames[t])
        if use_abs_overlay:
            fig = make_ERF_figure(
                heat_stack[t],
                img,
                token_idx=None,
                alpha=overlay_alpha,
                cmap=overlay_cmap,
            )
        else:
            fig = blend_heatmap(img, heat_stack[t], overlay_alpha, overlay_cmap)
        if t == 0:
            _draw_prompt_points(fig, prompt_points, color=prompt_color)
        # Backwards compatibility: optional red dots
        if feature_points:
            if (feature_point_frame_idx is None and t == 0) or \
                (feature_point_frame_idx is not None and int(t) == int(feature_point_frame_idx)):
                for (x, y) in feature_points:
                    _draw_red_dot(fig, x, y)
        if patch_bboxes and ((feature_point_frame_idx is None and t == 0) or (feature_point_frame_idx is not None and int(t) == int(feature_point_frame_idx))):
            for (left, top, width, height) in patch_bboxes:
                _draw_patch_outline(fig, left, top, width, height)
        overlays.append(fig)
    if not overlays:
        return
    panel = np.concatenate(overlays, axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(out_dir / f"sid{sid}_panel__{anchor_name.replace('.', '_').replace('@', '_')}{score_suffix}.jpeg")


def save_output_contribution_overlay(
    *,
    out_dir: Path,
    sid: int,
    score_suffix: str,
    heat_stack: torch.Tensor,
    target_tensor: torch.Tensor,
    prompt_points: Sequence[PromptPoint],
    prompt_color: Tuple[int, int, int] = (0, 0, 255),
    overlay_alpha: float,
    overlay_cmap: str,
    use_abs_overlay: bool,
    overlay_on_base: bool = True,
    feature_points: Optional[List[Tuple[int,int]]] = None,             # [NEW]
    feature_point_frame_idx: Optional[int] = None,
    patch_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    apply_sigmoid: bool = False,
    file_stub: str = "output_contribution",
) -> None:

    base_tensor = target_tensor.detach()
    if apply_sigmoid:
        base_tensor = torch.sigmoid(base_tensor)
    if base_tensor.dim() == 2:
        base_tensor = base_tensor.unsqueeze(0).unsqueeze(0)
    if base_tensor.dim() == 3:
        base_tensor = base_tensor.unsqueeze(1)
    if base_tensor.dim() != 4:
        raise RuntimeError(f"[output_overlay] Unexpected base tensor shape: {tuple(base_tensor.shape)}")
    base_np = base_tensor.cpu().numpy()
    heat_tensor = heat_stack
    if isinstance(heat_tensor, torch.Tensor):
        heat_tensor = heat_tensor.detach().cpu()
    if isinstance(heat_tensor, np.ndarray):
        heat_np = np.asarray(heat_tensor)
    else:
        heat_np = np.asarray(heat_tensor)
    if heat_np.ndim == 3:
        heat_np = heat_np[:, None, ...]
    if heat_np.ndim != 4:
        raise RuntimeError(f"[output_overlay] Unexpected heatmap tensor shape: {heat_np.shape}")

    num_frames, num_variants = base_np.shape[:2]
    if heat_np.shape[0] != num_frames:
        raise RuntimeError("Mismatch between heatmap frames and base tensor frames")
    if heat_np.shape[1] not in {1, num_variants}:
        raise RuntimeError("Heatmap variant dimension must be 1 or match base tensor variants")

    def _apply_signed_cmap(array: np.ndarray, cmap_name: str, *, signed: bool) -> np.ndarray:
        cmap = mpl_cm.get_cmap(cmap_name)
        if signed:
            arr = np.clip(array, -1.0, 1.0)
            norm = (arr + 1.0) * 0.5
        else:
            arr = np.clip(array, 0.0, 1.0)
            norm = arr
        heat_rgba = cmap(norm)
        return np.clip(heat_rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)

    rows: List[np.ndarray] = []
    for variant in range(num_variants):
        tiles: List[np.ndarray] = []
        for t in range(num_frames):
            base_map = base_np[t, variant]
            heat_index = variant if heat_np.shape[1] == num_variants else 0
            heat_map = heat_np[t, heat_index]
            signed = not use_abs_overlay
            if use_abs_overlay:
                heat_map = np.abs(heat_map)
            heat_rgb = _apply_signed_cmap(heat_map, overlay_cmap, signed=signed)
            if overlay_on_base:
                if base_map.ndim == 2:
                    base_rgb = np.repeat(np.clip(base_map, 0.0, 1.0)[..., None], 3, axis=2)
                else:
                    arr = base_map
                    if arr.ndim == 3:
                        arr = arr.transpose(1, 2, 0)
                    arr_min = arr.min()
                    arr_max = arr.max()
                    if arr_max - arr_min > 1e-8:
                        arr = (arr - arr_min) / (arr_max - arr_min)
                    else:
                        arr = np.zeros_like(arr)
                    if arr.shape[-1] == 1:
                        arr = np.repeat(arr, 3, axis=2)
                    elif arr.shape[-1] > 3:
                        arr = arr[..., :3]
                    base_rgb = np.clip(arr, 0.0, 1.0)
                mask_rgb = (base_rgb * 255.0).astype(np.uint8)
                if mask_rgb.shape[:2] != heat_rgb.shape[:2]:
                    heat_rgb = np.array(
                        Image.fromarray(heat_rgb).resize((mask_rgb.shape[1], mask_rgb.shape[0]), Image.BILINEAR)
                    )
                mask_f = mask_rgb.astype(np.float32)
                heat_f = heat_rgb.astype(np.float32)
                overlay_f = (1.0 - overlay_alpha) * mask_f + overlay_alpha * heat_f
                fig = np.clip(overlay_f, 0.0, 255.0).astype(np.uint8)
            else:
                fig = heat_rgb
            if t == 0:
                _draw_prompt_points(fig, prompt_points, color=prompt_color)
            # [NEW] feature_points (빨간 점)
            if feature_points:
                if (feature_point_frame_idx is None and t == 0) or \
                   (feature_point_frame_idx is not None and int(t) == int(feature_point_frame_idx)):
                    for (x, y) in feature_points:
                        _draw_red_dot(fig, x, y)
            if patch_bboxes and ((feature_point_frame_idx is None and t == 0) or (feature_point_frame_idx is not None and int(t) == int(feature_point_frame_idx))):
                for (left, top, width, height) in patch_bboxes:
                    _draw_patch_outline(fig, left, top, width, height)
            tiles.append(fig)
        if tiles:
            rows.append(np.concatenate(tiles, axis=1))
    if not rows:
        return
    panel = np.concatenate(rows, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    stub = file_stub or "output_contribution"
    Image.fromarray(panel).save(out_dir / f"sid{sid}_panel__{stub}{score_suffix}.jpeg")


def save_mask_logits_panels(
    *,
    out_dir: Path,
    sid: int,
    lane_idx: int,
    score_suffix: str,
    mask_logits: torch.Tensor,
    frames: Sequence[torch.Tensor],
) -> None:
    from src.utils.render import torch_to_image

    out_dir.mkdir(parents=True, exist_ok=True)

    mask_tensor = mask_logits.detach()
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(1)
    num_frames = mask_tensor.shape[0]
    num_variants = mask_tensor.shape[1]

    masks_np = torch.sigmoid(mask_tensor).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    rows_sigmoid: List[np.ndarray] = []
    for variant in range(num_variants):
        tiles = [masks_np[t, variant] for t in range(num_frames)]
        rows_sigmoid.append(np.concatenate(tiles, axis=1))
    if rows_sigmoid:
        panel_sigmoid = np.concatenate(rows_sigmoid, axis=0)
        Image.fromarray(np.ascontiguousarray(panel_sigmoid), mode="L").save(
            out_dir / f"sid{sid}_lane{lane_idx}_mask_panel{score_suffix}.jpeg"
        )

    logits_np = mask_tensor.cpu().numpy()
    rows_raw: List[np.ndarray] = []
    for variant in range(num_variants):
        variant_frames = logits_np[:, variant]
        vmin = float(variant_frames.min())
        vmax = float(variant_frames.max())
        if np.isclose(vmax, vmin, rtol=1e-6, atol=1e-6):
            scaled = np.full_like(variant_frames, 127, dtype=np.uint8)
        else:
            scaled = ((variant_frames - vmin) / (vmax - vmin + 1e-8) * 255.0)
            scaled = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
        tiles_raw = [scaled[t] for t in range(num_frames)]
        rows_raw.append(np.concatenate(tiles_raw, axis=1))
    if rows_raw:
        panel_raw = np.concatenate(rows_raw, axis=0)
        Image.fromarray(np.ascontiguousarray(panel_raw), mode="L").save(
            out_dir / f"sid{sid}_lane{lane_idx}_mask_panel_raw{score_suffix}.jpeg"
        )

    frame_tiles: List[np.ndarray] = []
    for t in range(min(num_frames, len(frames))):
        img = torch_to_image(frames[t])
        frame_tiles.append((img * 255.0).clip(0, 255).astype(np.uint8))
    if frame_tiles:
        frame_panel = np.concatenate(frame_tiles, axis=1)
        Image.fromarray(np.ascontiguousarray(frame_panel)).save(
            out_dir / f"sid{sid}_lane{lane_idx}_input_panel{score_suffix}.jpeg"
        )


# [NEW] 전 좌표 SAE unit 활성 맵(naive feature activation map) 오버레이 저장
def save_feature_activation_overlay(
    *,
    out_dir: Path,
    sid: int,
    score_suffix: str,
    map_stack: torch.Tensor,                    # [T, Hm, Wm] (CPU or CUDA)
    frames: Sequence[torch.Tensor],
    prompt_points: Sequence[PromptPoint],
    prompt_color: Tuple[int, int, int] = (0, 0, 255),
    overlay_alpha: float,
    overlay_cmap: str,
    file_stub: str = "feature_map",
    min_abs: float = 0.05,
) -> None:
    from src.utils.render import make_ERF_figure, torch_to_image
    maps = map_stack.detach().cpu()
    overlays: List[np.ndarray] = [] 
    for t in range(min(maps.size(0), len(frames))):
        img = torch_to_image(frames[t])         # float RGB [0,1]
        map_t = maps[t].float()
        mag = map_t.abs()
        max_val = float(mag.max())
        if max_val > 1e-6:
            map_norm = mag / max_val
        else:
            map_norm = torch.zeros_like(mag)
        if min_abs > 0:
            map_norm = map_norm.masked_fill(map_norm < float(min_abs), 0.0)
        fig = make_ERF_figure(
            map_norm,
            img,
            alpha=overlay_alpha,
            cmap=overlay_cmap,
        )
        if t == 0:
            _draw_prompt_points(fig, prompt_points, color=prompt_color)

        overlays.append(fig)
    if not overlays:
        return
    panel = np.concatenate(overlays, axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(out_dir / f"sid{sid}_panel__{file_stub}{score_suffix}.jpeg")
