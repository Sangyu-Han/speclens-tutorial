from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.hooks import RemovableHandle
import torch.autograd.forward_ad as forward_ad

from .activation_tape import ActivationTape
from .interventions import InterventionFn, identity_intervention
from .specs import (
    OverrideSpec,
    SelectionFn,
    infer_grid_shape,
    normalise_indices,
)
import math

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_anchor_contexts(anchors: Iterable[Any], forward_fn: Callable[[], Any]) -> None:
    """
    Populate SAE contexts if any anchor is missing cached activations.

    This is a lightweight guard to avoid repeated boilerplate in attribution
    utilities that need anchor.branch.sae_context()[attr_name] to exist before
    setting requires_grad tensors.
    """
    missing = False
    for anchor in anchors:
        branch = getattr(anchor, "branch", anchor)
        attr = getattr(anchor, "attr_name", None)
        if attr is None:
            continue
        ctx_getter = getattr(branch, "sae_context", None)
        if ctx_getter is None:
            continue
        ctx = ctx_getter()
        if ctx.get(attr) is None:
            missing = True
            break
    if missing:
        with torch.no_grad():
            forward_fn()

def _maybe_detach(tensor: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if not enabled:
        return tensor
    detached = tensor.detach().clone()
    detached.requires_grad_(tensor.requires_grad)
    return detached


def _extract_grid_from_meta(meta: Optional[Dict]) -> Optional[Tuple[int, int]]:
    if not meta:
        return None
    for key in ("grid_shape", "spatial_shape", "grid_hw"):
        if key in meta:
            value = meta[key]
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return int(value[0]), int(value[1])
    prefix = meta.get("prefix_shape") if isinstance(meta, dict) else None
    if isinstance(prefix, (tuple, list)) and len(prefix) >= 2:
        return int(prefix[-2]), int(prefix[-1])
    return None


@dataclass
class TensorLayout:
    tensor: torch.Tensor
    reshape_meta: Optional[Dict]

    def __post_init__(self) -> None:
        self.view, self._restore_fn, self.grid_shape = self._build_view(self.tensor, self.reshape_meta)

    def materialize(self, view: torch.Tensor) -> torch.Tensor:
        return self._restore_fn(view)

    @property
    def positions(self) -> int:
        return int(self.view.shape[1])

    @property
    def features(self) -> int:
        return int(self.view.shape[-1])

    def _build_view(
        self,
        tensor: torch.Tensor,
        reshape_meta: Optional[Dict],
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], Optional[Tuple[int, int]]]:
        grid_shape = _extract_grid_from_meta(reshape_meta)
        prefix_shape = tuple(int(v) for v in ((reshape_meta or {}).get("prefix_shape") or ()))
        if tensor.dim() == 2 and prefix_shape and len(prefix_shape) >= 1:
            lanes = prefix_shape[0]
            tail = prefix_shape[1:]
            positions = int(math.prod(tail)) if tail else 1
            channels = tensor.shape[-1]
            view = tensor.reshape(lanes, positions, channels)
            def _restore(v: torch.Tensor) -> torch.Tensor:
                return v.reshape(lanes * positions, channels)
            grid = grid_shape
            if grid is None and len(tail) >= 2:
                grid = (int(tail[-2]), int(tail[-1]))
            return view, _restore, grid
        if tensor.dim() == 4:
            lanes, channels, height, width = tensor.shape
            perm = tensor.permute(0, 2, 3, 1).contiguous()
            view = perm.reshape(lanes, height * width, channels)

            def _restore(v: torch.Tensor) -> torch.Tensor:
                shaped = v.reshape(lanes, height, width, channels)
                return shaped.permute(0, 3, 1, 2).contiguous()

            grid = grid_shape or (height, width)
            return view, _restore, grid
        if tensor.dim() == 3:
            lanes, positions, channels = tensor.shape

            def _restore(v: torch.Tensor) -> torch.Tensor:
                return v.reshape(lanes, positions, channels)

            grid = grid_shape
            return tensor.reshape(lanes, positions, channels), _restore, grid
        if tensor.dim() == 2:
            lanes, channels = tensor.shape

            def _restore(v: torch.Tensor) -> torch.Tensor:
                return v.reshape(lanes, channels)

            view = tensor.reshape(lanes, channels)
            return view, _restore, grid_shape
        raise RuntimeError(f"Unsupported tensor rank {tensor.dim()} for activation override")


def _positions_from_spec(
    spec: OverrideSpec,
    layout: TensorLayout,
    *,
    default_all: bool,
) -> Optional[torch.Tensor]:
    if spec.position_indices is not None:
        return normalise_indices(spec.position_indices, size=layout.positions, name="position")
    if spec.token_indices is not None:
        return normalise_indices(spec.token_indices, size=layout.positions, name="token")
    if spec.spatial_y is not None and spec.spatial_x is not None:
        grid = infer_grid_shape(explicit=layout.grid_shape, positions=layout.positions)
        if grid is None:
            raise RuntimeError("Cannot resolve spatial indices without a grid shape.")
        h, w = grid
        y_idx = normalise_indices(spec.spatial_y, size=h, name="spatial_y")
        x_idx = normalise_indices(spec.spatial_x, size=w, name="spatial_x")
        if y_idx is None or x_idx is None:
            raise RuntimeError("Spatial indices must specify both y and x")
        combos = torch.cartesian_prod(y_idx, x_idx)
        return (combos[:, 0] * w + combos[:, 1]).to(dtype=torch.long)
    if default_all:
        return torch.arange(layout.positions, dtype=torch.long)
    return None


class ActivationControllerBase:
    """
    Base controller that records activations through an ActivationTape and
    applies interventions on demand.
    """

    mode: str = "base"

    def __init__(
        self,
        *,
        spec: OverrideSpec,
        intervention: Optional[InterventionFn] = None,
        detach_overrides: bool = False,
    ) -> None:
        self.spec = spec
        self._intervention = intervention or identity_intervention
        self._detach_overrides = bool(detach_overrides)
        self._pre = ActivationTape()
        self._post = ActivationTape()

    @property
    def pre_tape(self) -> ActivationTape:
        return self._pre

    @property
    def post_tape(self) -> ActivationTape:
        return self._post

    def record_pre(self, frame_idx: int, tensor: torch.Tensor) -> None:
        self._pre.append(frame_idx, tensor)

    def record_post(self, frame_idx: int, tensor: torch.Tensor) -> None:
        self._post.append(frame_idx, tensor)

    def pre_stack(self, *, detach: bool = False) -> torch.Tensor:
        return self._pre.as_stack(detach=detach)

    def post_stack(self, *, detach: bool = False) -> torch.Tensor:
        return self._post.as_stack(detach=detach)

    def release_tapes(self) -> None:
        self._pre.clear()
        self._post.clear()

    def clear(self) -> None:
        self._pre.clear()
        self._post.clear()

    def should_override(self, frame_idx: int) -> bool:
        return True

    def override(
        self,
        frame_idx: int,
        tensor: torch.Tensor,
        *,
        reshape_meta: Optional[Dict] = None,
    ) -> torch.Tensor:
        if not self.should_override(frame_idx):
            return tensor
        return self._run_override(tensor, reshape_meta=reshape_meta)

    def _prepare_subset(self, subset: torch.Tensor) -> torch.Tensor:
        return _maybe_detach(subset, enabled=self._detach_overrides)

    def _apply_selector(
        self,
        tensor: torch.Tensor,
        reshape_meta: Optional[Dict],
        selector: SelectionFn,
    ) -> torch.Tensor:
        view, restore = selector(tensor, reshape_meta=reshape_meta)
        working = self._prepare_subset(view)
        updated = self._intervention(working)
        if updated.shape != view.shape:
            raise RuntimeError(
                f"Intervention returned shape {tuple(updated.shape)} "
                f"but expected {tuple(view.shape)}"
            )
        return restore(updated)

    def _run_override(
        self,
        tensor: torch.Tensor,
        *,
        reshape_meta: Optional[Dict],
    ) -> torch.Tensor:
        if self.spec.selector is not None:
            return self._apply_selector(tensor, reshape_meta, self.spec.selector)
        layout = TensorLayout(tensor, reshape_meta)
        view = layout.view.clone()
        updated_view = self._override_view(view, layout)
        return layout.materialize(updated_view)

    def _override_view(self, view: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
        raise NotImplementedError


class SingleActivationController(ActivationControllerBase):
    """
    Overrides specific coordinates (lane/token/unit) per frame.
    """

    mode: str = "single"

    def _override_view(self, view: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
        lane_indices = normalise_indices(
            self.spec.lane_idx, size=view.shape[0], name="lane"
        )
        if lane_indices is None:
            lane_indices = torch.tensor([0], dtype=torch.long)
        position_idx = _positions_from_spec(self.spec, layout, default_all=False)
        if position_idx is None:
            position_idx = torch.tensor([0], dtype=torch.long)
        unit_idx = normalise_indices(
            self.spec.unit_indices,
            size=layout.features,
            name="unit",
        )
        for lane in lane_indices.tolist():
            for pos in position_idx.tolist():
                vector = view[lane, pos]
                view[lane, pos] = self._apply_on_vector(vector, unit_idx)
        return view

    def _apply_on_vector(
        self,
        vector: torch.Tensor,
        unit_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if unit_idx is None:
            working = self._prepare_subset(vector)
            updated = self._intervention(working)
            if updated.shape != vector.shape:
                raise RuntimeError("Intervention must return the same shape as the target vector")
            return updated
        if unit_idx.device != vector.device:
            unit_idx = unit_idx.to(device=vector.device)
        selection = vector.index_select(-1, unit_idx)
        working = self._prepare_subset(selection)
        updated = self._intervention(working)
        if updated.shape != selection.shape:
            raise RuntimeError("Intervention must preserve the selected unit shape")
        base = vector.clone()
        base.index_copy_(-1, unit_idx, updated)
        return base


class FrameActivationController(ActivationControllerBase):
    """
    Overrides an entire lane/frame when the requested frame index matches.
    """

    mode: str = "frame"

    def should_override(self, frame_idx: int) -> bool:
        target = self.spec.target_frame_idx
        return target is None or frame_idx == target

    def _override_view(self, view: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
        lane_indices = normalise_indices(
            self.spec.lane_idx,
            size=view.shape[0],
            name="lane",
        )
        if lane_indices is None:
            lane_indices = torch.arange(view.shape[0], dtype=torch.long)
        for lane in lane_indices.tolist():
            lane_slice = view[lane]
            working = self._prepare_subset(lane_slice)
            updated = self._intervention(working)
            if updated.shape != lane_slice.shape:
                raise RuntimeError("Intervention must preserve lane slice shape")
            view[lane] = updated
        return view


class AllFrameActivationController(ActivationControllerBase):
    """
    Overrides every frame regardless of index.
    """

    mode: str = "all_frames"

    def _override_view(self, view: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
        # 1. 전체 Override Tensor를 가져옵니다.
        #    Loop 밖에서 view 전체를 넘겨야 _intervention 내부의 shape check를 통과합니다.
        #    replacement는 view와 동일한 shape(L, P, F)을 가집니다.
        replacement = self._intervention(view)

        # 만약 replacement가 view와 동일하다면(수정 없음), 바로 리턴
        if replacement is view:
            return view

        # 2. 적용할 인덱스 계산 (Lanes, Positions, Units)
        lane_indices = normalise_indices(
            self.spec.lane_idx,
            size=view.shape[0],
            name="lane",
            default_all=True,
        )
        if lane_indices is None:
            lane_indices = torch.arange(view.shape[0], device=view.device)

        # Positions: 보통 'all'이지만 spec에 따라 다를 수 있음
        positions = _positions_from_spec(self.spec, layout, default_all=True)
        if positions is None:  # Should default to all if None returned
            positions = torch.arange(view.shape[1], device=view.device)

        # Units: SAE Feature Index
        unit_idx = normalise_indices(
            self.spec.unit_indices,
            size=layout.features,
            name="unit",
        )

        # device 통일 (indices only)
        if lane_indices.device != view.device:
            lane_indices = lane_indices.to(view.device)
        if positions.device != view.device:
            positions = positions.to(view.device)
        if unit_idx is not None and unit_idx.device != view.device:
            unit_idx = unit_idx.to(view.device)

        # 3. Forward-AD safe update: build boolean mask (non-dual) and use torch.where
        lane_mask = torch.zeros(view.shape[0], dtype=torch.bool, device=view.device)
        lane_mask.index_fill_(0, lane_indices, True)
        pos_mask = torch.zeros(view.shape[1], dtype=torch.bool, device=view.device)
        pos_mask.index_fill_(0, positions, True)

        if unit_idx is not None:
            unit_mask = torch.zeros(view.shape[2], dtype=torch.bool, device=view.device)
            unit_mask.index_fill_(0, unit_idx, True)
            mask = lane_mask[:, None, None] & pos_mask[None, :, None] & unit_mask[None, None, :]
        else:
            mask = lane_mask[:, None, None] & pos_mask[None, :, None]

        # Avoid in-place advanced indexing to preserve forward-mode tangents.
        return torch.where(mask, replacement, view)



__all__ = [
    "ActivationControllerBase",
    "AllFrameActivationController",
    "FrameActivationController",
    "SingleActivationController",
    "FeatureOverrideBase",
    "FeatureOverrideController",
    "AllTokensFeatureOverrideController",
    "FrameFeatureOverrideController",
    "AllFramesFeatureOverrideController",
    "ControllerAutoResetHandle",
    "install_controller_autoreset_hooks",
]


class FeatureOverrideBase(ActivationControllerBase):
    """Base class for SAE feature override controllers."""

    mode: str = "single"

    def disable(self) -> None:
        raise NotImplementedError

    def enable(self) -> None:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    def clear_override(self) -> None:
        raise NotImplementedError

    def set_override(self, vec: torch.Tensor) -> None:
        raise NotImplementedError

    def set_override_all(self, acts: torch.Tensor) -> None:
        raise NotImplementedError

    def baseline_act(self) -> torch.Tensor:
        raise NotImplementedError

    def baseline_act_for_frame(self, frame_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def last_encoded_act(self, frame_idx: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError

    def last_encoded_all(self) -> torch.Tensor:
        raise NotImplementedError

    def live_encoded_act(self) -> torch.Tensor:
        raise NotImplementedError

    def live_encoded_for_frame(self, frame_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def live_encoded_all(self) -> torch.Tensor:
        raise NotImplementedError

    def frame_order(self) -> List[int]:
        raise NotImplementedError

    def prepare_for_forward(self) -> None:
        raise NotImplementedError

    def set_debug_dir(self, path: Optional[Path]) -> None:
        raise NotImplementedError

    def release_cached_activations(self) -> None:
        raise NotImplementedError


class _FeatureOverrideMixin(FeatureOverrideBase):
    """Shared implementation for SAE feature overrides built on controllers."""

    def __init__(
        self,
        *,
        spec: OverrideSpec,
        frame_getter: Optional[Callable[[], int]],
        mode: str,
    ) -> None:
        self.mode = mode
        self._frame_getter = frame_getter
        # Allow consumers to opt out of automatic pre/post resets (e.g., when
        # tapes are read immediately after a forward). Default is opt-in.
        self.auto_reset_ok: bool = True
        self._frame_counter = 0
        self._frame_order: List[int] = []
        self._frame_index: Dict[int, int] = {}
        self._override_single: Optional[torch.Tensor] = None
        self._override_stack: Optional[torch.Tensor] = None
        self._anchored_override = False
        self._baseline_single: Optional[torch.Tensor] = None
        self._baseline_by_frame: Dict[int, torch.Tensor] = {}
        self._last_single: Optional[torch.Tensor] = None
        self._last_by_frame: Dict[int, torch.Tensor] = {}
        self._live_single: Optional[torch.Tensor] = None
        self._live_by_frame: Dict[int, torch.Tensor] = {}
        self._debug_dir: Optional[Path] = None
        self._debug_counter = 0
        self._disabled = False

        def _intervention(vec: torch.Tensor) -> torch.Tensor:
            return self._apply_override(vec)

        super().__init__(spec=spec, intervention=_intervention, detach_overrides=False)  # type: ignore[arg-type]

    # -- ActivationControllerBase hooks ---------------------------------
    def clear(self) -> None:
        super().clear()
        self._frame_order.clear()
        self._frame_index.clear()
        self._live_by_frame.clear()

    # -- FeatureOverrideBase API ----------------------------------------
    def disable(self) -> None:
        self._disabled = True
        self.clear_override()

    def enable(self) -> None:
        self._disabled = False

    def is_available(self) -> bool:
        return not self._disabled

    def clear_override(self) -> None:
        self._override_single = None
        self._override_stack = None

    def set_override(self, vec: torch.Tensor) -> None:
        self._override_single = vec

    def set_override_all(self, acts: torch.Tensor) -> None:
        self._override_stack = acts

    def set_anchored_override(self, enabled: bool) -> None:
        self._anchored_override = bool(enabled)

    def baseline_act(self) -> torch.Tensor:
        if self._baseline_single is None:
            raise RuntimeError("Baseline activation is not available yet")
        return self._baseline_single

    def baseline_act_for_frame(self, frame_idx: int) -> torch.Tensor:
        tensor = self._baseline_by_frame.get(frame_idx)
        if tensor is None:
            raise RuntimeError(f"No baseline activation for frame {frame_idx}")
        return tensor

    def last_encoded_act(self, frame_idx: Optional[int] = None) -> torch.Tensor:
        if frame_idx is None:
            if self._last_single is None:
                raise RuntimeError("No encoded activation recorded yet")
            return self._last_single
        tensor = self._last_by_frame.get(frame_idx)
        if tensor is None:
            raise RuntimeError(f"No encoded activations for frame {frame_idx}")
        return tensor

    def last_encoded_all(self) -> torch.Tensor:
        if not self._frame_order:
            raise RuntimeError("No encoded activations recorded.")
        rows = []
        for frame_idx in self._frame_order:
            tensor = self._last_by_frame.get(frame_idx)
            if tensor is None:
                raise RuntimeError(f"Missing encoded activation for frame {frame_idx}")
            rows.append(tensor)
        return torch.stack(rows, dim=0)

    def live_encoded_act(self) -> torch.Tensor:
        if self._live_single is None:
            raise RuntimeError("Live encoded activations are unavailable")
        return self._live_single

    def live_encoded_for_frame(self, frame_idx: int) -> torch.Tensor:
        tensor = self._live_by_frame.get(frame_idx)
        if tensor is None:
            raise RuntimeError(f"No live activations for frame {frame_idx}")
        return tensor

    def live_encoded_all(self) -> torch.Tensor:
        if not self._frame_order:
            raise RuntimeError("No live activations recorded.")
        rows = []
        for frame_idx in self._frame_order:
            tensor = self._live_by_frame.get(frame_idx)
            if tensor is None:
                raise RuntimeError(f"Missing live activation for frame {frame_idx}")
            rows.append(tensor)
        return torch.stack(rows, dim=0)

    def frame_order(self) -> List[int]:
        return list(self._frame_order)

    def prepare_for_forward(self) -> None:
        # Reset per-forward frame bookkeeping so indices start at 0 each run and
        # drop stale tape entries from prior forwards to avoid accumulation.
        self._pre.clear()
        self._post.clear()
        self._frame_counter = 0
        self._frame_order.clear()
        self._frame_index.clear()
        self._live_by_frame.clear()
        self._live_single = None
        self._last_single = None
        self._last_by_frame.clear()
        self._baseline_single = None
        self._baseline_by_frame.clear()

    def set_debug_dir(self, path: Optional[Path]) -> None:
        self._debug_dir = Path(path) if path else None
        self._debug_counter = 0

    def release_cached_activations(self) -> None:
        self._pre.clear()
        self._post.clear()
        self._live_single = None
        self._live_by_frame.clear()
        self._baseline_single = None
        self._baseline_by_frame.clear()
        self._last_single = None
        self._last_by_frame.clear()
        self._frame_order.clear()
        self._frame_index.clear()

    # -- internal helpers -----------------------------------------------
    def _record_frame(self, frame_idx: int) -> None:
        if frame_idx not in self._frame_index:
            self._frame_index[frame_idx] = len(self._frame_order)
            self._frame_order.append(frame_idx)

    def _current_frame(self) -> int:
        getter = self._frame_getter
        if getter is not None:
            try:
                return int(getter())
            except Exception:
                pass
        idx = self._frame_counter
        self._frame_counter += 1
        return idx

    def _apply_override(self, vec: torch.Tensor) -> torch.Tensor:
        if self._disabled:
            return vec
        frame_idx = self._current_frame()
        self._record_frame(frame_idx)
        det = vec.detach()
        self._live_single = vec
        self._live_by_frame[frame_idx] = vec
        self._last_single = det
        self._last_by_frame[frame_idx] = det
        if self.mode == "all_frames":
            self._baseline_by_frame.setdefault(frame_idx, det)
        else:
            if self._baseline_single is None:
                self._baseline_single = det
        override = self._select_override(frame_idx)
        if override is None:
            return vec
        if self._anchored_override:
            if not forward_ad._is_fwd_grad_enabled():
                return vec
            try:
                vec_primal, _ = forward_ad.unpack_dual(vec)
            except Exception:
                vec_primal = vec
            try:
                _primal, tangent = forward_ad.unpack_dual(override)
            except Exception:
                return vec
            if tangent is None:
                return vec_primal
            tangent = tangent.to(device=vec_primal.device, dtype=vec_primal.dtype)
            if tangent.shape != vec_primal.shape:
                if tangent.numel() == vec_primal.numel():
                    tangent = tangent.reshape(vec_primal.shape)
                else:
                    raise RuntimeError(
                        f"Override tangent shape {tuple(tangent.shape)} does not match subset {tuple(vec_primal.shape)}"
                    )
            return forward_ad.make_dual(vec_primal, tangent)
        if forward_ad._is_fwd_grad_enabled():
            if override.device != vec.device or override.dtype != vec.dtype:
                raise RuntimeError(
                    "[forward-AD] Override dtype/device must match target tensor. "
                    "Move override to the target dtype/device before enabling forward-AD."
                )
        elif override.device != vec.device or override.dtype != vec.dtype:
            override = override.to(device=vec.device, dtype=vec.dtype)
        if override.shape != vec.shape:
            if override.numel() == vec.numel():
                override = override.reshape(vec.shape)
            else:
                raise RuntimeError(
                    f"Override tensor shape {tuple(override.shape)} does not match subset {tuple(vec.shape)}"
                )
        return override

    def _select_override(self, frame_idx: int) -> Optional[torch.Tensor]:
        if self._override_stack is not None:
            idx = self._frame_index.get(frame_idx)
            if idx is None or idx < 0 or idx >= self._override_stack.shape[0]:
                return None
            return self._override_stack[idx]
        return self._override_single


class FeatureOverrideController(_FeatureOverrideMixin, SingleActivationController):
    """Override controller that targets a single coordinate."""

    def __init__(
        self,
        *,
        spec: OverrideSpec,
        frame_getter: Optional[Callable[[], int]] = None,
    ) -> None:
        super().__init__(spec=spec, frame_getter=frame_getter, mode="single")


class AllTokensFeatureOverrideController(_FeatureOverrideMixin, AllFrameActivationController):
    """Override controller that targets all tokens within a lane/frame."""

    def __init__(
        self,
        *,
        spec: OverrideSpec,
        frame_getter: Optional[Callable[[], int]] = None,
    ) -> None:
        super().__init__(spec=spec, frame_getter=frame_getter, mode="all_tokens")


class FrameFeatureOverrideController(_FeatureOverrideMixin, FrameActivationController):
    """Override controller that swaps entire frames on demand."""

    def __init__(
        self,
        *,
        spec: OverrideSpec,
        frame_getter: Optional[Callable[[], int]] = None,
    ) -> None:
        super().__init__(spec=spec, frame_getter=frame_getter, mode="frame")


class AllFramesFeatureOverrideController(_FeatureOverrideMixin, AllFrameActivationController):
    """Override controller that applies replacements across every frame."""

    def __init__(
        self,
        *,
        spec: OverrideSpec,
        frame_getter: Optional[Callable[[], int]] = None,
    ) -> None:
        super().__init__(spec=spec, frame_getter=frame_getter, mode="all_frames")

    def set_override(self, vec: torch.Tensor) -> None:
        raise RuntimeError("AllFramesFeatureOverrideController requires set_override_all()")

#TODO : 모델 설계를 한번 점검해야함. model -> anchor 관리 handle -> wrapper 등록 -> capture anchor 등록. 그리고 capture anchor도 override를 할 수 있게 설계를 해야할거같아.
class ControllerAutoResetHandle:
    """Manage automatic controller/anchor resets tied to a model's forward hooks."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        controllers: List[ActivationControllerBase],
        branches: List[Any],
        pre_handle: RemovableHandle,
        post_handle: Optional[RemovableHandle],
    ) -> None:
        self._model = model
        self._controllers = controllers
        self._branches = branches
        self._pre_handle = pre_handle
        self._post_handle = post_handle

    def reset_now(self, *, clear_override: bool = True) -> None:
        for branch in self._branches:
            if branch is None:
                continue
            try:
                branch.clear_context()
            except Exception:
                pass
        for ctrl in self._controllers:
            if ctrl is None or not getattr(ctrl, "auto_reset_ok", True):
                continue
            try:
                ctrl.prepare_for_forward()
                ctrl.release_cached_activations()
                if clear_override:
                    ctrl.clear_override()
            except Exception:
                pass

    def remove(self) -> None:
        state = getattr(self._model, "_sae_autoreset_state", None)
        if isinstance(state, dict):
            state_controllers = state.get("controllers", [])
            state_branches = state.get("branches", [])
            remaining = [c for c in state_controllers if c not in self._controllers]
            remaining_br = [b for b in state_branches if b not in self._branches]
            state["controllers"] = remaining
            state["branches"] = remaining_br
            if not remaining and not remaining_br:
                try:
                    if self._pre_handle is not None:
                        self._pre_handle.remove()
                except Exception:
                    pass
                try:
                    if self._post_handle is not None:
                        self._post_handle.remove()
                except Exception:
                    pass
                try:
                    delattr(self._model, "_sae_autoreset_state")
                except Exception:
                    pass
        else:
            try:
                if self._pre_handle is not None:
                    self._pre_handle.remove()
            except Exception:
                pass
            try:
                if self._post_handle is not None:
                    self._post_handle.remove()
            except Exception:
                pass


def install_controller_autoreset_hooks(
    model: torch.nn.Module,
    controllers: Iterable[Optional[ActivationControllerBase]],
    *,
    branches: Optional[Iterable[Any]] = None,
    enable_post_hook: bool = False,
) -> Optional[ControllerAutoResetHandle]:
    """
    Register a single pair of forward pre/post hooks on `model` that prepare and
    optionally clear overrides for the provided controllers. Subsequent calls
    reuse the same hooks and expand the controller/branch set.

    Args:
        model: module whose forward demarcates a "run" boundary (typically the full model).
        controllers: iterable of controller instances to reset.
        branches: iterable of SAE branch modules whose contexts should be cleared
            once per model forward (e.g., _PhysicalSAEBranch).
        enable_post_hook: if True, also clear overrides/caches right after forward
            returns. Leave False when code needs to read tapes immediately after forward.
    """
    ctrls = [c for c in controllers if c is not None]
    brs = [b for b in (branches or []) if b is not None]
    if not ctrls and not brs:
        return None

    state = getattr(model, "_sae_autoreset_state", None)
    if state is None or not isinstance(state, dict):
        state = {"controllers": [], "branches": [], "pre": None, "post": None}
        setattr(model, "_sae_autoreset_state", state)

    # Deduplicate controller list
    for ctrl in ctrls:
        if ctrl not in state["controllers"]:
            state["controllers"].append(ctrl)

    for br in brs:
        if br not in state["branches"]:
            state["branches"].append(br)

    def _pre(_module, _inputs):
        for br in state.get("branches", []):
            if br is None:
                continue
            try:
                br.clear_context()
            except Exception:
                pass
        for ctrl in state["controllers"]:
            if ctrl is None or not getattr(ctrl, "auto_reset_ok", True):
                continue
            ctrl.prepare_for_forward()

    def _post(_module, _inputs, _output):
        if not state.get("enable_post", False):
            return
        for ctrl in state["controllers"]:
            if ctrl is None or not getattr(ctrl, "auto_reset_ok", True):
                continue
            ctrl.release_cached_activations()
            ctrl.clear_override()

    if state.get("pre") is None:
        state["pre"] = model.register_forward_pre_hook(_pre)
    if state.get("post") is None:
        state["post"] = model.register_forward_hook(_post)
    # Update post flag
    state["enable_post"] = bool(state.get("enable_post", False) or enable_post_hook)

    return ControllerAutoResetHandle(
        model,
        controllers=list(ctrls),
        branches=list(brs),
        pre_handle=state["pre"],
        post_handle=state["post"],
    )
