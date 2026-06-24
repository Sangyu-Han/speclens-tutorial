from __future__ import annotations

import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.core.attribution.backends.gradients import BACKENDS as GRADIENT_BACKENDS, ObjectiveTensor
from src.core.attribution.feature_contributions.integrated_gradients import (
    FeatureContributionPathCallbacks,
    FeatureContributionPathState,
    compute_feature_contribution,
)
from src.core.attribution.feature_contributions.weight_utils import build_unit_weight_builder
from src.core.attribution.sae.constants import (
    SAE_LAYER_ATTRIBUTE_TENSORS,
    SAE_LAYER_DEFAULT_ATTR,
    SAE_LAYER_METHOD,
    normalise_sae_attr,
    resolve_sae_request,
)
from src.core.hooks.spec import parse_spec
from src.core.runtime.capture import LayerCapture, MultiAnchorCapture
from src.core.runtime.controllers import (
    ActivationControllerBase,
    AllFramesFeatureOverrideController,
    AllTokensFeatureOverrideController,
    FeatureOverrideBase,
    FeatureOverrideController,
    FrameFeatureOverrideController,
)
from src.core.runtime.specs import OverrideSpec
from src.core.runtime.wrappers import wrap_target_layer_with_sae
from src.utils.utils import resolve_module


def _build_feature_override_controller(
    mode: str,
    spec: OverrideSpec,
    frame_getter: Optional[Callable[[], int]],
) -> FeatureOverrideBase:
    key = (mode or "single").lower()
    controllers = {
        "single": FeatureOverrideController,
        "frame": FrameFeatureOverrideController,
        "all_frames": AllFramesFeatureOverrideController,
        "all_tokens": AllTokensFeatureOverrideController,
    }
    if key not in controllers:
        raise ValueError(
            f"Unsupported override mode '{mode}'. "
            f"Available: {sorted(controllers.keys())}"
        )
    return controllers[key](spec=spec, frame_getter=frame_getter)


def _split_pre_specs(specs: Iterable[str]) -> Tuple[List[str], List[str]]:
    pre_specs: List[str] = []
    others: List[str] = []
    for raw in specs or []:
        _base, method, _branch, _alias, _attr = parse_spec(raw)
        if method == "pre":
            pre_specs.append(raw)
        else:
            others.append(raw)
    return pre_specs, others


def _build_weight_mask(
    tensor: torch.Tensor,
    *,
    unit_index: int,
    aggregation: str,
) -> torch.Tensor:
    if tensor.dim() == 0:
        raise RuntimeError("Objective tensor for SAE units must have rank >= 1.")
    if tensor.shape[-1] <= unit_index:
        raise ValueError(
            f"Unit index {unit_index} exceeds latent dimension {tensor.shape[-1]}"
        )
    weight = torch.zeros_like(tensor)
    index = (slice(None),) * (tensor.dim() - 1) + (unit_index,)
    weight[index] = 1.0
    agg = (aggregation or "sum").lower()
    if agg == "sum":
        return weight
    if agg == "mean":
        denom = weight[index].numel()
        if denom > 0:
            weight[index] = weight[index] / float(denom)
        return weight
    raise ValueError(f"Unsupported objective aggregation '{aggregation}'")


def _normalise_method_key(key: str, available: Iterable[str]) -> str:
    normalised = (key or "").lower()
    if normalised in {"integrated_gradients", "ig_conductance"}:
        # keep ig_conductance for forward conductance support
        pass
    elif normalised in {"ig", "grad", "input_x_grad"}:
        pass
    else:
        normalised = normalised.replace("-", "_")
    if normalised not in available:
        raise ValueError(f"Unsupported method '{key}'. Available: {sorted(set(available))}")
    return normalised


@dataclass
class RuntimeTarget:
    layer: str
    unit: int
    override_mode: str = "all_frames"
    objective_aggregation: str = "sum"


@dataclass
class AnchorConfig:
    capture: Sequence[str]
    ig_active: Sequence[str] = ()
    stop_grad: Sequence[str] = ()


@dataclass
class BackwardConfig:
    enabled: bool = True
    method: str = "ig"
    ig_steps: int = 32
    baseline: str = "zeros"


@dataclass
class ForwardConfig:
    enabled: bool = False
    method: str = "ig_conductance"
    ig_steps: int = 32
    baseline: str = "zeros"


class AttributionRuntime:
    """
    High-level runtime wrapper that wires SAE overrides, multi-anchor capture,
    and attribution backends for transformer-style models (SAMv2, CLIP, etc.).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        adapter: Optional[object],
        forward_fn: Callable[[], None],
        sae_module: torch.nn.Module,
        target: RuntimeTarget,
        allow_missing_anchor_grad: bool = False,
    ) -> None:
        if forward_fn is None:
            raise ValueError("forward_fn callback must be provided.")
        self.model = model
        self.adapter = adapter
        self.forward_fn = forward_fn
        self.target = target
        self.unit_index = int(target.unit)
        self._frame_getter = getattr(adapter, "current_frame_idx", None)
        self._resolve_module = partial(resolve_module, self.model)
        self._stop_grad: set[str] = set()
        self._ig_active: set[str] = set()
        self._backward_weight_multiplier: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self._forward_weight_multiplier: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self._force_sdpa_math = True
        self._allow_missing_anchor_grad = bool(allow_missing_anchor_grad)
        self._perturb_fn: Optional[Callable[[], None]] = None
        self._anchor_override_modules: Dict[str, Any] = {}

        spec = OverrideSpec(
            lane_idx=None,
            unit_indices=[self.unit_index],
        )
        self.controller = _build_feature_override_controller(
            target.override_mode,
            spec,
            self._frame_getter,
        )
        self._wrap_target_layer(target.layer, sae_module)
        self.anchor_capture = MultiAnchorCapture(frame_getter=self._frame_getter)

    # ------------------------------------------------------------------
    # Target switching
    # ------------------------------------------------------------------
    def set_target_unit(self, unit: int) -> None:
        """
        Update the active target unit (latent index) for subsequent attributions.
        """
        self.unit_index = int(unit)
        try:
            if hasattr(self.controller, "spec"):
                self.controller.spec.unit_indices = [self.unit_index]
        except Exception:
            pass

    def set_target(self, layer: str, unit: int, sae_module: Optional[torch.nn.Module] = None) -> None:
        """
        Rebind the objective/override controller to a new layer/unit.
        """
        if layer == self.target.layer and int(unit) == self.target.unit:
            self.set_target_unit(unit)
            return
        # restore previous target wrapper
        if self._restore_target is not None:
            try:
                self._restore_target()
            except Exception:
                pass
        self.target = RuntimeTarget(
            layer=layer,
            unit=int(unit),
            override_mode=self.target.override_mode,
            objective_aggregation=self.target.objective_aggregation,
        )
        self.unit_index = int(unit)
        spec = OverrideSpec(
            lane_idx=None,
            unit_indices=[self.unit_index],
        )
        self.controller = _build_feature_override_controller(
            self.target.override_mode,
            spec,
            self._frame_getter,
        )
        if sae_module is None:
            raise RuntimeError("sae_module is required when retargeting to a new layer")
        self._wrap_target_layer(layer, sae_module)

    def _wrap_target_layer(self, layer: str, sae_module: torch.nn.Module) -> None:
        capture = LayerCapture(layer)
        module = self._resolve_module(capture.base)
        self._restore_target, self._target_sae_branch = wrap_target_layer_with_sae(
            module,
            capture=capture,
            sae=sae_module,
            controller=self.controller,
            frame_getter=self._frame_getter,
        )

    # ------------------------------------------------------------------
    # Anchors / hooks
    # ------------------------------------------------------------------
    def configure_anchors(self, cfg: AnchorConfig) -> None:
        # IG alpha interpolation must be applied at every ig_active site, even if
        # that site is not an attribution anchor. Merge the two lists so the
        # capture machinery can wrap those modules/pre-hooks and participate in
        # the computation graph.
        merged_specs = list(dict.fromkeys(list(cfg.capture or []) + list(cfg.ig_active or [])))
        pre_specs, other_specs = _split_pre_specs(merged_specs)
        if other_specs:
            self.anchor_capture.register_from_specs(other_specs, resolve_module_fn=self._resolve_module)
        if pre_specs:
            self.anchor_capture.register_preinput_from_specs(pre_specs, resolve_module_fn=self._resolve_module)
        self._ig_active = set(cfg.ig_active or [])
        self.anchor_capture.ig_set_active(self._ig_active)
        self._stop_grad = set(cfg.stop_grad or [])
        try:
            self._anchor_override_modules = self.anchor_capture.get_overrideable_modules()
        except Exception:
            self._anchor_override_modules = {}

    def set_backward_weight_multiplier(
        self,
        fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ) -> None:
        """
        Optional hook that multiplies the backward objective mask with a custom tensor.

        The callable receives the latent tensor (pre_stack output) and must return a tensor
        broadcastable to the same shape.
        """
        self._backward_weight_multiplier = fn

    def set_forward_weight_multiplier(
        self,
        fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ) -> None:
        """
        Optional hook that multiplies forward-mode weight builders with a custom tensor.
        """
        self._forward_weight_multiplier = fn

    def set_perturb_fn(self, fn: Optional[Callable[[], None]]) -> None:
        """
        Optional callable invoked before every forward pass (useful for noisy methods like SmoothGrad).
        """
        self._perturb_fn = fn

    def set_override_lane(self, lane_idx: Optional[int]) -> None:
        if not hasattr(self.controller, "spec"):
            return
        if lane_idx is None:
            self.controller.spec.lane_idx = None
        else:
            self.controller.spec.lane_idx = [int(lane_idx)]


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _before_forward(self) -> None: # override를 clear하지 않음. override 기록 -> override 적용 -> before_forward -> forward -> override clear 순으로 해야함.
        self.controller.clear()
        self.controller.prepare_for_forward()
        self.anchor_capture.release_step_refs()
        try:
            self.anchor_capture.reset_frame_counter()
        except Exception:
            pass
        # advance frame counter once per forward call
        try:
            self.anchor_capture.next_frame()
        except Exception:
            pass

    def _sdpa_context(self, force_math: Optional[bool] = None):
        use_math = self._force_sdpa_math if force_math is None else bool(force_math)
        if not use_math:
            return nullcontext()
        return sdpa_kernel(backends=[SDPBackend.MATH])

    def _run_forward(self, *, require_grad: bool = True, force_sdpa_math: Optional[bool] = None) -> None:
        self._before_forward()
        """
       require_grad 의미:
          - True  : "바깥 grad 모드를 존중" (여기서 굳이 enable_grad() 하지 않음)
          - False : 이 호출 동안은 강제로 no_grad 로 실행 (값만 필요할 때)

        이렇게 해야 외부에서 torch.no_grad()로 감싼 호출이 제대로 먹히고,
        forward-contribution 같이 '값만 필요'한 경로에서 그래프 생성을 막을 수 있음.
        """
        ctx = self._sdpa_context(force_sdpa_math)
        if self._perturb_fn is not None:
            try:
                self._perturb_fn()
            except Exception:
                pass
        if not require_grad:
            with ctx:
                with torch.no_grad():
                    self.forward_fn()
        else:
            # 기본 grad 모드(전역 상태)를 그대로 사용
            with ctx:
                self.forward_fn()

    def _release_after_step(self) -> None:
        self.anchor_capture.release_step_refs()
        try:
            self.controller.release_cached_activations()
        except Exception:
            pass
        # [FIX 4] Backward IG 루프에서도 Target SAE Context를 비워줘야 함
        # 이걸 안 하면 Backward Graph의 잔재가 다음 Step까지 남음
        self._clear_target_sae_context()
    def _collect_anchor_tensors(self) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        try:
            tensors = self.anchor_capture.get_tensor_lists(detach=False)
        except Exception:
            raise
        if not tensors:
            try:
                print('get_tensor_lists 실패, get_tensors 시도 - debugging 필요.')
                tensors = self.anchor_capture.get_tensors()
            except Exception:
                tensors = {}
        if os.getenv("ATTR_MEM_DEBUG", "0") == "1":
            try:
                head = ", ".join(list(tensors.keys())[:5])
            except Exception:
                head = ""
            print(
                f"[feature_contrib][anchors] captured={len(tensors)} stop_grad={bool(self._stop_grad)} {head}"
            )
        # IG 스텝마다 테이프를 비워 스택이 다음 스텝으로 누적되지 않도록 함
        # (여기서는 스택 전체를 반환해 recurrent 모델의 모든 호출을 반영)
        stacks = tensors
        if not self._stop_grad:
            self.anchor_capture.clear_tapes()
            self._clear_target_sae_context()
            return stacks
        detached: Dict[str, torch.Tensor] = {}
        for name, tensor in stacks.items():
            if name in self._stop_grad:
                detached[name] = tensor.detach()
            else:
                detached[name] = tensor
        self.anchor_capture.clear_tapes()
        self._clear_target_sae_context()
        return detached

    def _build_objective_getter(self) -> Callable[[], ObjectiveTensor]:
        def _getter() -> ObjectiveTensor:
            tensor = self.controller.pre_stack(detach=False) 
            weight = _build_weight_mask(
                tensor,
                unit_index=self.unit_index,
                aggregation=self.target.objective_aggregation,
            )
            if self._backward_weight_multiplier is not None:
                extra = self._backward_weight_multiplier(tensor)
                if not torch.is_tensor(extra):
                    raise TypeError("backward weight multiplier must return a tensor")
                weight = weight * extra.to(device=weight.device, dtype=weight.dtype)
            return ObjectiveTensor(tensor=tensor, weight=weight)

        return _getter

    def _prepare_anchor_baselines(self, baseline: str) -> Dict[str, torch.Tensor]:
        mode = (baseline or "zeros").lower()
        if mode in {"zero", "zeros", "none"}:
            return {}
        if mode in {"record", "recorded", "current"}:
            self.anchor_capture.record_baselines()
            return self.anchor_capture.get_baseline_tensors()
        raise ValueError(f"Unsupported anchor baseline mode '{baseline}'")

    # ------------------------------------------------------------------
    # Backward attribution
    # ------------------------------------------------------------------
    def run_backward(self, cfg: BackwardConfig) -> Optional[Dict[str, torch.Tensor]]:
        if not cfg.enabled:
            return None
        method = _normalise_method_key(cfg.method, GRADIENT_BACKENDS.keys())
        use_cached_targets = method in {"ig_cached", "ig_target"}
        target_cache_ready = False

        if method == "ig_target":
            self.anchor_capture.clear_ig_targets()
            # Capture the endpoint activations once so non-target units can stay fixed.
            self._run_forward(require_grad=False, force_sdpa_math=False)
            base_map = self._prepare_anchor_baselines(cfg.baseline)
            self.anchor_capture.record_ig_targets()
            target_cache_ready = True
            self.anchor_capture.ig_prepare_target_baselines(self.unit_index, base_map)
            anchor_baselines = self.anchor_capture.get_baseline_tensors()
        else:
            if use_cached_targets:
                self.anchor_capture.clear_ig_targets()
                # Populate anchor captures once so IG can reuse the recorded target activations.
                self._run_forward(require_grad=False, force_sdpa_math=False)
            anchor_baselines = self._prepare_anchor_baselines(cfg.baseline)
        objective_getter = self._build_objective_getter()
        anchor_getter = self._collect_anchor_tensors

        if method in {"ig", "ig_conductance", "ig_cached", "ig_target"}:
            self.anchor_capture.enable_ig_override()
            self.anchor_capture.ig_set_active(self._ig_active)
            if use_cached_targets:
                self.anchor_capture.ig_use_cached_targets(True)
                if not target_cache_ready:
                    self.anchor_capture.record_ig_targets()
            backend = GRADIENT_BACKENDS[method](
                anchor_tensors_getter=anchor_getter,
                objective_getter=objective_getter,
                steps=max(1, cfg.ig_steps),
                set_alpha=self.anchor_capture.set_alpha,
                do_forward=lambda require_grad=True: self._run_forward(require_grad=require_grad, force_sdpa_math=False),
                release_step_refs=self._release_after_step,
                anchor_baselines=anchor_baselines,
                allow_missing_grad=self._allow_missing_anchor_grad,
            )
            try:
                return backend()
            finally:
                if use_cached_targets:
                    self.anchor_capture.clear_ig_targets()
                    self.anchor_capture.ig_use_cached_targets(False)
                self.anchor_capture.disable_ig_override()

        if method == "ig_legacy":
            anchor_modules = dict(self._anchor_override_modules or {})
            backend = GRADIENT_BACKENDS[method](
                anchor_modules_getter=lambda: anchor_modules,
                anchor_tensor_lists_getter=lambda: self.anchor_capture.get_tensor_lists(detach=False),
                anchor_tensor_records_getter=lambda: self.anchor_capture.get_tensor_records(detach=False),
                objective_getter=objective_getter,
                steps=max(1, cfg.ig_steps),
                do_forward=lambda: self._run_forward(require_grad=True, force_sdpa_math=False),
                anchor_baselines=anchor_baselines,
                allow_missing_grad=self._allow_missing_anchor_grad,
            )
            return backend()

        if method in {"smoothgrad", "vargrad", "gradient_shap"}:
            samples = max(1, cfg.ig_steps)
            kwargs = {
                "anchor_tensors_getter": anchor_getter,
                "objective_getter": objective_getter,
                "do_forward": lambda: self._run_forward(require_grad=True, force_sdpa_math=False),
                "samples": samples,
                "allow_missing_grad": self._allow_missing_anchor_grad,
            }
            if method == "gradient_shap":
                kwargs["anchor_baselines"] = anchor_baselines
            backend = GRADIENT_BACKENDS[method](**kwargs)
            return backend()

        # grad / input_x_grad: ensure tapes/controller are populated with a fresh forward pass
        # before calling autograd.grad. IG backends already run their own forwards inside.
        self._run_forward(require_grad=True, force_sdpa_math=False)
        kwargs = {
            "anchor_tensors_getter": anchor_getter,
            "objective_getter": objective_getter,
            "allow_missing_grad": self._allow_missing_anchor_grad,
        }
        if method in {"input_x_grad", "ig_conductance"}:
            kwargs["anchor_baselines"] = anchor_baselines
        backend = GRADIENT_BACKENDS[method](**kwargs)
        return backend()

    # ------------------------------------------------------------------
    # Forward (JVP) contribution
    # ------------------------------------------------------------------
    def _collect_target_latent(self) -> torch.Tensor:
        # ensure we have a live stack to initialise the path
        try:
            latent = self.controller.post_stack(detach=True)
        except Exception:
            self._run_forward(require_grad=False)
            latent = self.controller.post_stack(detach=True)
        finally:
            try:
                self.controller.release_cached_activations()
            except Exception:
                pass
            self._clear_target_sae_context()
        return latent

    def _build_forward_baseline(self, target_latent: torch.Tensor, baseline: str) -> torch.Tensor:
        mode = (baseline or "zeros").lower()
        if mode in {"zero", "zeros"}:
            return torch.zeros_like(target_latent)
        if mode in {"current", "data"}:
            return target_latent.detach().clone()
        raise ValueError(f"Unsupported forward baseline '{baseline}'")

    def _resolve_ig_anchor_attr(self) -> str:
        parsed = parse_spec(self.target.layer)
        method = None
        attr = None
        try:
            method, attr = resolve_sae_request(parsed.method, parsed.attr)
        except Exception:
            method, attr = None, None
        if method == SAE_LAYER_METHOD and attr:
            return SAE_LAYER_ATTRIBUTE_TENSORS.get(attr, "acts")
        if parsed.attr:
            try:
                attr_key = normalise_sae_attr(parsed.attr)
            except Exception:
                attr_key = None
            if attr_key:
                return SAE_LAYER_ATTRIBUTE_TENSORS.get(attr_key, "acts")
        return SAE_LAYER_ATTRIBUTE_TENSORS.get(SAE_LAYER_DEFAULT_ATTR, "acts")

    def _build_forward_callbacks(
        self,
        *,
        baseline: torch.Tensor,
        target_latent: torch.Tensor,
        method: str,
        target_mask: Optional[torch.Tensor] = None,
        ig_anchor_attr: Optional[str] = None,
        ig_anchor_baseline: Optional[torch.Tensor] = None,
        ig_anchor_skip: bool = False,
    ) -> FeatureContributionPathCallbacks:
        base = baseline.detach().clone()
        tgt = target_latent.detach().clone()
        mask = target_mask.detach() if target_mask is not None else None

        path_state = FeatureContributionPathState(baseline=base, target=tgt)
        if mask is not None:
            # Move only the masked unit(s); keep all other units fixed at the target value.
            # u(alpha) = target*(1 - mask) + (baseline + delta*alpha*mask)
            def _interp(alpha: float) -> torch.Tensor:
                alpha_f = float(alpha)
                delta = tgt - base
                return (tgt * (1.0 - mask)) + (base + delta * alpha_f * mask)
            path_state.interpolate = _interp

        def _run_forward(alpha: float, _latent: torch.Tensor) -> Dict[str, torch.Tensor]:
            del alpha, _latent
            self._run_forward(require_grad=True)
            return self._collect_anchor_tensors()

        def _extract(alpha: float, _latent: torch.Tensor) -> Dict[str, torch.Tensor]:
            del alpha, _latent
            self._run_forward(require_grad=True)
            return self._collect_anchor_tensors()

        branch = getattr(self, "_target_sae_branch", None) if ig_anchor_attr else None
        setter = getattr(branch, "set_ig_attr_alpha", None) if branch is not None else None
        clearer = getattr(branch, "clear_ig_attr_alpha", None) if branch is not None else None
        anchor_attr = str(ig_anchor_attr) if ig_anchor_attr else None
        baselines = None
        if ig_anchor_baseline is not None and anchor_attr:
            baselines = {anchor_attr: ig_anchor_baseline}

        def _apply_anchor_alpha(alpha: float) -> None:
            if anchor_attr is None:
                return
            if callable(clearer) and ig_anchor_skip:
                clearer()
                return
            if callable(setter):
                try:
                    setter(alpha=float(alpha), active_attrs={anchor_attr}, baselines=baselines, logical_base=None)
                except TypeError:
                    try:
                        setter(alpha=float(alpha), active_attrs={anchor_attr}, baselines=baselines)
                    except Exception:
                        pass

        if os.getenv("ATTR_SIGN_DEBUG", "0") == "1":
            print(
                f"[sign-debug] _build_forward_callbacks: "
                f"weight_multiplier={'SET' if self._forward_weight_multiplier is not None else 'NONE'} "
                f"method={method} unit={self.unit_index}"
            )
        weight_builder = build_unit_weight_builder(
            method=method,
            unit=self.unit_index,
            weight_multiplier=self._forward_weight_multiplier,
        )
        return FeatureContributionPathCallbacks(
            prepare_path=lambda: path_state,
            run_alpha=_run_forward,
            extract_target=_extract,
            alpha_pre_hook=lambda _alpha: (
                _apply_anchor_alpha(_alpha),
                self.anchor_capture.release_step_refs(),
                self._clear_target_sae_context(),
            ),
            alpha_post_hook=lambda _alpha: (
                callable(clearer) and clearer(),
                self.anchor_capture.release_step_refs(),
                self._clear_target_sae_context(),
            ),
            weight_builder=weight_builder,
        )

    def run_forward_contribution(self, cfg: ForwardConfig) -> Optional[Dict[str, torch.Tensor]]:
        if not cfg.enabled:
            return None
        noise_methods = {"smoothgrad", "vargrad", "gradient_shap"}
        raw_method = (cfg.method or "ig").lower()
        anchored_aliases = {"ig_anchor", "ig_anchored", "anchored_ig", "ig-anchor", "ig-anchored"}
        anchored_requested = raw_method in anchored_aliases
        method = "ig" if anchored_requested else raw_method
        method = _normalise_method_key(
            method,
            {"ig", "ig_target", "ig_conductance", "grad", "input_x_grad", *noise_methods},
        )
        spec_obj = getattr(self.controller, "spec", None)
        saved_spec = None
        widen_override = method in {"ig", "ig_conductance", "gradient_shap"}
        if spec_obj is not None and widen_override:
            # Forward contribution path overrides must sweep the full latent vector; the
            # per-unit mask is applied via the JVP weight builder, not by slicing the
            # override itself. Restricting unit/token selections here skews the path.
            saved_spec = (
                spec_obj.unit_indices,
                spec_obj.token_indices,
                spec_obj.position_indices,
                spec_obj.spatial_y,
                spec_obj.spatial_x,
            )
            spec_obj.unit_indices = None
            spec_obj.token_indices = None
            spec_obj.position_indices = None
            spec_obj.spatial_y = None
            spec_obj.spatial_x = None
        anchored = False
        anchored_setter = getattr(self.controller, "set_anchored_override", None)
        if anchored_requested and callable(anchored_setter) and getattr(self, "_target_sae_branch", None) is not None:
            anchored = True
            anchored_setter(True)
        ig_anchor_attr = self._resolve_ig_anchor_attr() if anchored else None
        baseline_mode = (cfg.baseline or "zeros").lower()
        ig_anchor_skip = anchored and baseline_mode in {"current", "data"}
        samples = max(1, cfg.ig_steps)
        try:
            if method in noise_methods:
                def _zeros_like_nested(obj: Any) -> Any:
                    if torch.is_tensor(obj):
                        return torch.zeros_like(obj).cpu()
                    if isinstance(obj, dict):
                        return {k: _zeros_like_nested(v) for k, v in obj.items()}
                    raise TypeError(f"Unsupported contribution structure type: {type(obj)}")

                def _add_nested(dst: Any, src: Any) -> None:
                    if torch.is_tensor(dst):
                        dst.add_(src.detach().cpu())
                    elif isinstance(dst, dict):
                        if not isinstance(src, dict):
                            raise TypeError("Mismatched contribution structures")
                        for k, v in dst.items():
                            if k in src:
                                _add_nested(v, src[k])
                    else:
                        raise TypeError(f"Unsupported contribution structure type: {type(dst)}")

                def _pow2_nested(obj: Any) -> Any:
                    if torch.is_tensor(obj):
                        return obj.pow(2)
                    if isinstance(obj, dict):
                        return {k: _pow2_nested(v) for k, v in obj.items()}
                    raise TypeError(f"Unsupported contribution structure type: {type(obj)}")

                def _div_nested(obj: Any, denom: float) -> Any:
                    if torch.is_tensor(obj):
                        return obj / float(denom)
                    if isinstance(obj, dict):
                        return {k: _div_nested(v, denom) for k, v in obj.items()}
                    raise TypeError(f"Unsupported contribution structure type: {type(obj)}")

                base_method = "grad" if method in {"smoothgrad", "vargrad"} else ("ig" if method != "ig_target" else "ig_target")
                accum: Dict[str, Any] = {}
                sumsqs: Optional[Dict[str, Any]] = {} if method == "vargrad" else None
                for _ in range(samples):
                    target_latent = self._collect_target_latent()
                    baseline = self._build_forward_baseline(target_latent, cfg.baseline)
                    use_anchor = anchored and base_method == "ig"
                    target_mask = _build_weight_mask(target_latent, unit_index=self.unit_index, aggregation="sum") if base_method == "ig_target" else None
                    callbacks = self._build_forward_callbacks(
                        baseline=baseline,
                        target_latent=target_latent,
                        method=base_method,
                        target_mask=target_mask,
                        ig_anchor_attr=ig_anchor_attr if use_anchor else None,
                        ig_anchor_baseline=None,
                        ig_anchor_skip=ig_anchor_skip if use_anchor else False,
                    )
                    del target_latent, baseline, target_mask
                    out = compute_feature_contribution(
                        enabled=True,
                        steps=max(1, cfg.ig_steps),
                        unit_index=self.unit_index,
                        feature_override=self.controller,
                        callbacks=callbacks,
                        method=base_method,
                    )
                    if out is None:
                        continue
                    if not accum:
                        accum = {k: _zeros_like_nested(v) for k, v in out.items()}
                        if sumsqs is not None:
                            sumsqs = {k: _zeros_like_nested(v) for k, v in out.items()}
                    for k, v in out.items():
                        _add_nested(accum[k], v)
                        if sumsqs is not None:
                            _add_nested(sumsqs[k], _pow2_nested(v))
                if not accum:
                    return {}
                mean = {k: _div_nested(v, float(samples)) for k, v in accum.items()}
                if sumsqs is None:
                    return mean
                var: Dict[str, Any] = {}
                for k in mean.keys():
                    var[k] = _div_nested(sumsqs[k], float(samples))
                    if torch.is_tensor(var[k]) and torch.is_tensor(mean[k]):
                        var[k] = var[k] - mean[k].pow(2)
                    elif isinstance(var[k], dict):
                        def _sub_nested(a: Any, b: Any) -> Any:
                            if torch.is_tensor(a) and torch.is_tensor(b):
                                return a - b
                            if isinstance(a, dict) and isinstance(b, dict):
                                return {kk: _sub_nested(a[kk], b[kk]) for kk in a.keys()}
                            raise TypeError("Mismatched contribution structures")
                        var[k] = _sub_nested(var[k], _pow2_nested(mean[k]))
                    else:
                        raise TypeError("Unsupported contribution structure type for variance computation")
                return var

            target_latent = self._collect_target_latent()
            if torch.cuda.is_available() and os.getenv("ATTR_MEM_DEBUG") == "1":
                print(f"[feature_contrib][after_collect_latent] cuda_alloc={torch.cuda.memory_allocated()/(1024**3):.3f}GB")
            baseline = self._build_forward_baseline(target_latent, cfg.baseline)
            use_anchor = anchored and method == "ig"
            target_mask = _build_weight_mask(target_latent, unit_index=self.unit_index, aggregation="sum") if method == "ig_target" else None
            callbacks = self._build_forward_callbacks(
                baseline=baseline,
                target_latent=target_latent,
                method=method,
                target_mask=target_mask,
                ig_anchor_attr=ig_anchor_attr if use_anchor else None,
                ig_anchor_baseline=None,
                ig_anchor_skip=ig_anchor_skip if use_anchor else False,
            )
            # Free caller-scope (N, dict_size) tensors — they were cloned inside callbacks.
            del target_latent, baseline, target_mask
            if torch.cuda.is_available():
                import gc; gc.collect()
                torch.cuda.empty_cache()
            return compute_feature_contribution(
                enabled=True,
                steps=max(1, cfg.ig_steps),
                unit_index=self.unit_index,
                feature_override=self.controller,
                callbacks=callbacks,
                method=method,
            )
        finally:
            if saved_spec is not None:
                (
                    spec_obj.unit_indices,
                    spec_obj.token_indices,
                    spec_obj.position_indices,
                    spec_obj.spatial_y,
                    spec_obj.spatial_x,
                ) = saved_spec
            if anchored and callable(anchored_setter):
                anchored_setter(False)
            if anchored:
                try:
                    branch = getattr(self, "_target_sae_branch", None)
                    if branch is not None and hasattr(branch, "clear_ig_attr_alpha"):
                        branch.clear_ig_attr_alpha()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        try:
            self.anchor_capture.clear_ig_targets()
            self.anchor_capture.clear_baselines()
            self.anchor_capture.clear_tapes()
            self.anchor_capture.clear()
        except Exception:
            pass
        try:
            if isinstance(self.controller, ActivationControllerBase):
                self.controller.release_cached_activations()
                self.controller.clear_override()
                self.controller.clear()
        except Exception:
            pass
        try:
            self._clear_target_sae_context()
        except Exception:
            pass
        self._backward_weight_multiplier = None
        self._forward_weight_multiplier = None
        self._perturb_fn = None
        if self._restore_target is not None:
            try:
                self._restore_target()
            finally:
                self._restore_target = None
        self._target_sae_branch = None

    def _clear_target_sae_context(self) -> None:
        branch = getattr(self, "_target_sae_branch", None)
        if branch is None:
            return
        try:
            branch.clear_context()
        except Exception:
            pass


Sam2AttributionRuntime = AttributionRuntime


__all__ = [
    "RuntimeTarget",
    "AnchorConfig",
    "BackwardConfig",
    "ForwardConfig",
    "AttributionRuntime",
    "Sam2AttributionRuntime",
]
