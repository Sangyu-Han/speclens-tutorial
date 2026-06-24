from __future__ import annotations

import os
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.autograd.functional import jvp as autograd_jvp

from src.core.runtime.controllers import FeatureOverrideBase


AlphaRunner = Callable[[float, torch.Tensor], Optional[Any]]
TargetExtractor = Callable[[float, torch.Tensor], torch.Tensor]
AlphaStepHook = Callable[[float], None]
ReverseModeCheck = Callable[[float, torch.Tensor], None]


@dataclass
class FeatureContributionPathState:
    baseline: torch.Tensor
    target: torch.Tensor
    interpolate: Optional[Callable[[float], torch.Tensor]] = None

    def latent_at(self, alpha: float) -> torch.Tensor:
        fn = self.interpolate
        if fn is None:
            return self.baseline + (self.target - self.baseline) * float(alpha)
        vec = fn(float(alpha))
        if not torch.is_tensor(vec):
            raise RuntimeError("Path interpolation callback must return a tensor")
        return vec


@dataclass
class FeatureContributionPathCallbacks:
    prepare_path: Callable[[], FeatureContributionPathState]
    run_alpha: AlphaRunner
    extract_target: TargetExtractor
    alpha_pre_hook: Optional[AlphaStepHook] = None
    alpha_post_hook: Optional[AlphaStepHook] = None
    reverse_mode_check: Optional[ReverseModeCheck] = None
    weight_builder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


@dataclass
class ContributionComputationState:
    baseline_vec: torch.Tensor
    target_vec: torch.Tensor
    delta_vec: torch.Tensor
    delta_flat: torch.Tensor
    entries: List["TargetEntry"]
    target_base_flat: torch.Tensor
    target_a0_flat: torch.Tensor
    target_a1_flat: torch.Tensor
    target_a0_map: "OrderedDict[str, torch.Tensor]"
    target_a1_map: "OrderedDict[str, torch.Tensor]"
    feat_a0_flat: torch.Tensor
    feat_a1_flat: torch.Tensor


@dataclass
class TargetEntry:
    name: str
    shape: torch.Size
    start: int
    end: int


def _normalise_target_bundle(obj: Any) -> "OrderedDict[str, torch.Tensor]":
    if torch.is_tensor(obj):
        return OrderedDict([("output", obj)])
    if isinstance(obj, OrderedDict):
        out = OrderedDict()
        for k, v in obj.items():
            key = str(k)
            if torch.is_tensor(v):
                out[key] = v
            elif isinstance(v, (list, tuple)):
                tensors = [t for t in v if torch.is_tensor(t)]
                if not tensors:
                    raise TypeError(f"Target bundle element '{key}' must contain tensors")
                out[key] = torch.stack(tensors, dim=0)
            else:
                raise TypeError(f"Target bundle element '{key}' must be tensor or list/tuple of tensors")
        return out
    if isinstance(obj, dict):
        out = OrderedDict()
        for k, v in obj.items():
            key = str(k)
            if torch.is_tensor(v):
                out[key] = v
            elif isinstance(v, (list, tuple)):
                tensors = [t for t in v if torch.is_tensor(t)]
                if not tensors:
                    raise TypeError(f"Target bundle element '{key}' must contain tensors")
                out[key] = torch.stack(tensors, dim=0)
            else:
                raise TypeError(f"Target bundle element '{key}' must be tensor or list/tuple of tensors")
        return out
    if isinstance(obj, (list, tuple)):
        bundle = OrderedDict()
        for idx, value in enumerate(obj):
            if torch.is_tensor(value):
                bundle[str(idx)] = value
            elif isinstance(value, (list, tuple)):
                tensors = [t for t in value if torch.is_tensor(t)]
                if not tensors:
                    raise TypeError(f"Target bundle element '{idx}' must contain tensors")
                bundle[str(idx)] = torch.stack(tensors, dim=0)
            else:
                raise TypeError("Target bundle elements must be tensors or list/tuple of tensors")
        if not bundle:
            raise RuntimeError("Target bundle cannot be empty")
        return bundle
    raise TypeError(f"Unsupported target bundle type: {type(obj)}")


def _flatten_target_bundle(bundle: "OrderedDict[str, torch.Tensor]") -> Tuple[torch.Tensor, List[TargetEntry]]:
    entries: List[TargetEntry] = []
    flat_segments: List[torch.Tensor] = []
    offset = 0
    for name, tensor in bundle.items():
        if not torch.is_tensor(tensor):
            raise TypeError("Target tensors must be torch.Tensor instances")
        flat = tensor.reshape(-1)
        length = flat.numel()
        entries.append(TargetEntry(name=name, shape=torch.Size(tensor.shape), start=offset, end=offset + length))
        flat_segments.append(flat)
        offset += length
    if not flat_segments:
        raise RuntimeError("Target bundle produced no tensors to flatten")
    flat_cat = torch.cat(flat_segments, dim=0)
    return flat_cat, entries


def _reconstruct_target_bundle(entries: List[TargetEntry], flat: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for entry in entries:
        seg = flat[entry.start : entry.end].view(entry.shape)
        out[entry.name] = seg
    return out


def compute_feature_contribution(
    *,
    enabled: bool,
    steps: int,
    unit_index: int,
    feature_override: FeatureOverrideBase,
    callbacks: FeatureContributionPathCallbacks,
    method: str = "ig",
    weight_multiplier: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Dispatch feature contribution computation to the requested method."""
    method_key = _normalise_method_key(method)
    return _compute_feature_contribution_core(
        method_key=method_key,
        enabled=enabled,
        steps=steps,
        unit_index=unit_index,
        feature_override=feature_override,
        callbacks=callbacks,
        weight_multiplier=weight_multiplier,
    )


def compute_feature_contribution_via_ig(
    *,
    enabled: bool,
    steps: int,
    unit_index: int,
    feature_override: FeatureOverrideBase,
    callbacks: FeatureContributionPathCallbacks,
) -> Optional[Dict[str, torch.Tensor]]:
    """Integrated Gradients feature contribution."""
    return _compute_feature_contribution_core(
        method_key="ig",
        enabled=enabled,
        steps=steps,
        unit_index=unit_index,
        feature_override=feature_override,
        callbacks=callbacks,
        weight_multiplier=None,
    )


def compute_feature_contribution_via_ig_conductance(
    *,
    enabled: bool,
    steps: int,
    unit_index: int,
    feature_override: FeatureOverrideBase,
    callbacks: FeatureContributionPathCallbacks,
) -> Optional[Dict[str, torch.Tensor]]:
    """Integrated Gradients with conductance-style weighting."""
    return _compute_feature_contribution_core(
        method_key="ig_conductance",
        enabled=enabled,
        steps=steps,
        unit_index=unit_index,
        feature_override=feature_override,
        callbacks=callbacks,
    )


def compute_feature_contribution_via_grad(
    *,
    enabled: bool,
    steps: int,
    unit_index: int,
    feature_override: FeatureOverrideBase,
    callbacks: FeatureContributionPathCallbacks,
) -> Optional[Dict[str, torch.Tensor]]:
    """Single-step gradients feature contribution."""
    return _compute_feature_contribution_core(
        method_key="grad",
        enabled=enabled,
        steps=steps,
        unit_index=unit_index,
        feature_override=feature_override,
        callbacks=callbacks,
    )


def compute_feature_contribution_via_input_x_grad(
    *,
    enabled: bool,
    steps: int,
    unit_index: int,
    feature_override: FeatureOverrideBase,
    callbacks: FeatureContributionPathCallbacks,
) -> Optional[Dict[str, torch.Tensor]]:
    """Input × Grad feature contribution."""
    return _compute_feature_contribution_core(
        method_key="input_x_grad",
        enabled=enabled,
        steps=steps,
        unit_index=unit_index,
        feature_override=feature_override,
        callbacks=callbacks,
    )


def _normalise_method_key(method: str) -> str:
    key = (method or "ig").lower()
    if key in {"gradient"}:
        key = "grad"
    elif key in {"ixg", "input*grad", "input-grad", "inputxgrad"}:
        key = "input_x_grad"
    allowed_methods = {"ig", "ig_target", "ig_conductance", "grad", "input_x_grad"}
    if key not in allowed_methods:
        raise ValueError(f"Unsupported feature contribution method '{method}'. Supported: {sorted(allowed_methods)}")
    return key


def _compute_feature_contribution_core(
    *,
    method_key: str,
    enabled: bool,
    steps: int,
    unit_index: int,
    feature_override: FeatureOverrideBase,
    callbacks: FeatureContributionPathCallbacks,
    weight_multiplier: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> Optional[Dict[str, torch.Tensor]]:
    if not enabled or not feature_override.is_available():
        return None
    if steps <= 0:
        return None

    override_mode = getattr(feature_override, "mode", "single")

    def _flatten_latents(vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        if vec.dim() == 1:
            return vec.unsqueeze(0), vec.shape
        if vec.dim() == 2:
            return vec, vec.shape
        if vec.dim() >= 3:
            leading = int(torch.tensor(vec.shape[:-1]).prod().item())
            latent = vec.shape[-1]
            return vec.reshape(leading, latent), vec.shape
        raise RuntimeError(f"Unsupported latent tensor rank {vec.dim()}")

    path_state = callbacks.prepare_path()
    if not isinstance(path_state, FeatureContributionPathState):
        raise TypeError("prepare_path must return a FeatureContributionPathState instance")

    # Keep target_vec on GPU (needed directly by grad/input_x_grad methods).
    # Move baseline_vec, delta_vec, delta_flat to CPU to free GPU memory —
    # they're only used for interpolation and column extraction with .to().
    _latent_device = path_state.baseline.device
    target_vec = path_state.target.detach().clone()
    baseline_vec = path_state.baseline.detach().cpu()
    delta_vec = (path_state.target - path_state.baseline).detach().cpu()
    delta_flat, _ = _flatten_latents(delta_vec)

    def _latent_at(alpha: float) -> torch.Tensor:
        vec = path_state.latent_at(float(alpha))
        if not torch.is_tensor(vec):
            raise RuntimeError("Path interpolation must return a tensor")
        return vec

    def _apply_override(vec: Optional[torch.Tensor]) -> None:
        if vec is None:
            return
        if override_mode in {"all_frames", "all_tokens"}:
            setter = getattr(feature_override, "set_override_all", None)
            if setter is None:
                raise RuntimeError(f"{override_mode} override requires set_override_all on controller")
            setter(vec)
        else:
            feature_override.set_override(vec)

    def _call_alpha_hook(hook: Optional[AlphaStepHook], alpha: float) -> None:
        if hook is not None:
            hook(alpha)

    def _log_mem(label: str) -> None:
        if os.getenv("ATTR_MEM_DEBUG", "0") != "1":
            return
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            print(
                f"[feature_contrib][{label}] cuda_alloc={allocated / (1024 ** 3):.3f}GB"
            )

    def _extract_pair(alpha: float) -> Tuple[Any, torch.Tensor]:
        latent_vec = _latent_at(alpha)
        feature_override.clear_override()
        feature_override.prepare_for_forward()
        _log_mem(f"alpha={alpha:.3f}/pre")
        _call_alpha_hook(callbacks.alpha_pre_hook, alpha)
        try:
            _apply_override(latent_vec)
            # baseline/endpoint(alpha=0,1) 에서는 값만 필요하고 그래프는 필요 없음.
            # 여기서 no_grad 로 감싸서 불필요한 그래프 생성을 막는다.
            with torch.no_grad():
                target_tensor = callbacks.run_alpha(alpha, latent_vec)
                extracted = (
                    target_tensor
                    if target_tensor is not None
                    else callbacks.extract_target(alpha, latent_vec)
                )

            # 컨트롤러가 기록한 latent stack 은 detach=True 로 CPU 스냅샷만 사용한다.
            feat = feature_override.post_stack(detach=True)
        finally:
            try:
                feature_override.release_cached_activations()
            except Exception:
                pass
            feature_override.clear_override()
            _call_alpha_hook(callbacks.alpha_post_hook, alpha)
            _log_mem(f"alpha={alpha:.3f}/post")
        if extracted is None:
            raise RuntimeError("Target tensor missing during feature contribution extraction")
        if not torch.is_tensor(feat):
            raise RuntimeError("Controller did not record post-stack activations for feature contribution.")
        feature_override.clear()
        return extracted, feat

    def _detach_to_cpu(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: _detach_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_detach_to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_detach_to_cpu(v) for v in obj)
        return obj

    feature_override.clear_override()
    tensor_a0_raw, feat_a0 = _extract_pair(0.0) 
    tensor_a0_raw = _detach_to_cpu(tensor_a0_raw)
    feat_a0 = feat_a0.detach().cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log_mem("alpha0/after_empty_cache")
    tensor_a1_raw, feat_a1 = _extract_pair(1.0)
    tensor_a1_raw = _detach_to_cpu(tensor_a1_raw)
    feat_a1 = feat_a1.detach().cpu()
    # path_state tensors are no longer needed on GPU after both extract_pairs.
    # Move to CPU to free (N, dict_size) GPU memory before the JVP loop.
    if torch.is_tensor(path_state.baseline):
        path_state.baseline = path_state.baseline.cpu()
    if torch.is_tensor(path_state.target):
        path_state.target = path_state.target.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log_mem("alpha1/after_pathstate_cleanup")
    if feat_a0 is None or feat_a1 is None:
        return None

    def _detach_bundle_to_cpu(bundle: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, tensor in bundle.items():
            if torch.is_tensor(tensor):
                out[name] = tensor.detach().cpu()
            elif isinstance(tensor, (list, tuple)):
                tensors = [t for t in tensor if torch.is_tensor(t)]
                if not tensors:
                    raise TypeError(f"Target bundle element '{name}' is an empty list/tuple or contains no tensors")
                stacked = torch.stack(tensors, dim=0)
                out[name] = stacked.detach().cpu()
            else:
                raise TypeError(f"Target bundle element '{name}' must be a tensor or list/tuple of tensors")
        return out

    tensor_a0_det = _detach_bundle_to_cpu(_normalise_target_bundle(tensor_a0_raw))
    tensor_a1_det = _detach_bundle_to_cpu(_normalise_target_bundle(tensor_a1_raw))
    flat_a0, entries_a0 = _flatten_target_bundle(tensor_a0_det)
    flat_a1, entries_a1 = _flatten_target_bundle(tensor_a1_det)
    if len(entries_a0) != len(entries_a1):
        raise RuntimeError("Baseline/target bundles have mismatched entry counts for output IG")
    for e0, e1 in zip(entries_a0, entries_a1):
        if e0.name != e1.name or e0.shape != e1.shape:
            raise RuntimeError("Baseline/target bundles are structurally inconsistent for output IG")
    feat_a0_flat, _ = _flatten_latents(feat_a0.detach().cpu())
    feat_a1_flat, _ = _flatten_latents(feat_a1.detach().cpu())

    if os.getenv("ATTR_SIGN_DEBUG", "0") == "1":
        print(f"[sign-debug] entries ({len(entries_a0)}): " + ", ".join(
            f"{e.name}:{tuple(e.shape)}[{e.start}:{e.end}]" for e in entries_a0
        ))

    state = ContributionComputationState(
        baseline_vec=baseline_vec,
        target_vec=target_vec,
        delta_vec=delta_vec,
        delta_flat=delta_flat,
        entries=entries_a0,
        target_base_flat=flat_a0.detach().clone(),
        target_a0_flat=flat_a0.detach().clone(),
        target_a1_flat=flat_a1.detach().clone(),
        target_a0_map=tensor_a0_det,
        target_a1_map=tensor_a1_det,
        feat_a0_flat=feat_a0_flat,
        feat_a1_flat=feat_a1_flat,
    )

    unit_idx = int(unit_index)

    def _evaluate_step(alpha: float, latent_vec: torch.Tensor, allow_reverse_check: bool) -> Optional[torch.Tensor]:
        feature_override.clear_override()
        feature_override.prepare_for_forward()
        _call_alpha_hook(callbacks.alpha_pre_hook, alpha)
        with torch.no_grad(): # jvp는 standard autograd 불필요.
            try:
                # Derive u_alpha directly from latent_vec instead of running a
                # redundant first forward pass.  post_stack() == latent_vec because
                # the controller replaces SAE output with the override tensor, so
                # the extra forward only served to capture what we already know.
                # Crucially, that extra forward polluted SAM2's stateful memory
                # bank, causing 0-80% JVP error depending on the sample.
                u_alpha_flat, u_alpha_shape = _flatten_latents(latent_vec.detach())
                u_vector = u_alpha_flat.reshape(-1)

                def _f(vec_flat: torch.Tensor) -> torch.Tensor:
                    override_tensor = vec_flat.view(u_alpha_shape)
                    feature_override.clear_override()
                    feature_override.prepare_for_forward()
                    _apply_override(override_tensor)
                    try:
                        target_tensor = callbacks.extract_target(alpha, override_tensor) # where computation graph is built
                        if target_tensor is None:
                            raise RuntimeError("Target tensor missing during path eval")
                        bundle = _normalise_target_bundle(target_tensor)
                        flat, _ = _flatten_target_bundle(bundle)
                        return flat
                    finally:
                        feature_override.clear_override()

                use_delta_dir = method_key in {"ig", "ig_target", "input_x_grad"}

                # ── Build per-token column values for unit_idx ──
                # The JVP tangent is sparse: only column unit_idx is non-zero.
                # Compute column values compactly (1D) and create the full
                # matrix only once at the end to minimise GPU memory.
                _col_index = (slice(None),) * (u_alpha_flat.dim() - 1) + (unit_idx,)
                col_vals = None

                if callbacks.weight_builder is not None:
                    _wb_result = callbacks.weight_builder(u_alpha_flat)
                    if not torch.is_tensor(_wb_result):
                        raise TypeError("weight_builder must return a tensor")
                    if _wb_result.shape == u_alpha_flat.shape:
                        col_vals = _wb_result[_col_index].clone()
                        del _wb_result
                    else:
                        col_vals = _wb_result.reshape(-1)
                        del _wb_result

                if col_vals is None:
                    col_vals = (
                        u_alpha_flat.detach()[_col_index]
                        if method_key == "ig_conductance"
                        else torch.ones(
                            u_alpha_flat.shape[:-1],
                            device=u_alpha_flat.device,
                            dtype=u_alpha_flat.dtype,
                        )
                    )

                if weight_multiplier is not None:
                    extra = weight_multiplier(u_alpha_flat)
                    if not torch.is_tensor(extra):
                        raise TypeError("weight multiplier must return a tensor")
                    if extra.shape == u_alpha_flat.shape:
                        col_vals = col_vals * extra[_col_index].to(
                            device=col_vals.device, dtype=col_vals.dtype
                        )
                    else:
                        col_vals = col_vals * extra.reshape(col_vals.shape).to(
                            device=col_vals.device, dtype=col_vals.dtype
                        )
                    del extra

                if use_delta_dir:
                    delta_col = state.delta_flat[_col_index].to(
                        device=col_vals.device, dtype=col_vals.dtype
                    )
                    col_vals = col_vals * delta_col
                    del delta_col

                # Single full (N, D) allocation for JVP tangent
                dvec_matrix = torch.zeros_like(u_alpha_flat)
                dvec_matrix[_col_index] = col_vals
                if os.getenv("ATTR_SIGN_DEBUG", "0") == "1":
                    _cv_nz = int((col_vals != 0).sum().item())
                    _cv_tot = int(col_vals.numel())
                    print(
                        f"[sign-debug] tangent col_vals: nonzero={_cv_nz}/{_cv_tot} "
                        f"sum={float(col_vals.sum().item()):.4f} "
                        f"dvec_nnz={int((dvec_matrix != 0).sum().item())}"
                    )
                del col_vals

                _, g = autograd_jvp(
                    _f,
                    (u_vector,),
                    (dvec_matrix.reshape(-1),),
                    create_graph=False,
                    strict=True,
                )
                del dvec_matrix

                if os.getenv("ATTR_SIGN_DEBUG", "0") == "1":
                    print(
                        f"[sign-debug] JVP g: shape={tuple(g.shape)} sum={float(g.sum().item()):.4f} "
                        f"nnz={int((g != 0).sum().item())}/{int(g.numel())}"
                    )

                if callbacks.reverse_mode_check is not None and allow_reverse_check and os.getenv("ATTR_REVMODE_CHECK", "0") == "1":
                    callbacks.reverse_mode_check(alpha, u_alpha_flat.sum(dim=0))
                return g
            finally:
                try:
                    feature_override.release_cached_activations()
                except Exception:
                    pass
                feature_override.clear_override()
                _call_alpha_hook(callbacks.alpha_post_hook, alpha)

    method_impl = _METHOD_IMPLEMENTATIONS[method_key]
    attr_flat = method_impl(state, steps, _evaluate_step)

    if attr_flat is None:
        return None

    attr_map = _reconstruct_target_bundle(state.entries, attr_flat)
    feature_override.clear_override()
    return {
        "attr": OrderedDict((name, tensor.detach().cpu()) for name, tensor in attr_map.items()),
        "baseline": OrderedDict((name, tensor.detach().cpu()) for name, tensor in state.target_a0_map.items()),
        "alpha0": OrderedDict((name, tensor.detach().cpu()) for name, tensor in state.target_a0_map.items()),
        "alpha1": OrderedDict((name, tensor.detach().cpu()) for name, tensor in state.target_a1_map.items()),
        "feature_baseline": state.feat_a0_flat.sum(dim=0).detach().cpu(),
        "feature_target": state.feat_a1_flat.sum(dim=0).detach().cpu(),
    }


def _compute_via_integrated_gradients(
    state: ContributionComputationState,
    steps: int,
    evaluate_step: Callable[[float, torch.Tensor, bool], Optional[torch.Tensor]],
    *,
    allow_conductance: bool,
) -> Optional[torch.Tensor]:
    attr_flat = torch.zeros_like(state.target_base_flat)
    steps = max(1, int(steps))
    _device = state.target_vec.device
    for k in range(steps):
        alpha = (k + 1) / steps
        # baseline_vec and delta_vec are on CPU; compute interpolation on CPU
        # and transfer to GPU only the result (one tensor instead of keeping
        # three full copies on GPU).
        latent_vec = (state.baseline_vec + state.delta_vec * float(alpha)).to(_device)
        grad_vec = evaluate_step(alpha, latent_vec, True)
        if grad_vec is None:
            continue
        grad_vec_cpu = grad_vec.detach().cpu()
        
        # GPU 텐서 레퍼런스 즉시 제거 (VRAM 확보)
        del grad_vec
        attr_flat.add_(grad_vec_cpu)
        del latent_vec, grad_vec_cpu
    if attr_flat is None:
        return None
    return attr_flat / steps


def _compute_via_gradient(
    state: ContributionComputationState,
    evaluate_step: Callable[[float, torch.Tensor, bool], Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    return evaluate_step(1.0, state.target_vec, False)


def _compute_via_input_x_grad(
    state: ContributionComputationState,
    evaluate_step: Callable[[float, torch.Tensor, bool], Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    grad_vec = evaluate_step(1.0, state.target_vec, False)
    if grad_vec is None:
        return None

    grad_vec_cpu = grad_vec.detach().cpu()
    del grad_vec
    return grad_vec_cpu


def _run_ig(state: ContributionComputationState, steps: int, evaluate_step: Callable[[float, torch.Tensor, bool], Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    return _compute_via_integrated_gradients(state, steps, evaluate_step, allow_conductance=False)


def _run_ig_conductance(state: ContributionComputationState, steps: int, evaluate_step: Callable[[float, torch.Tensor, bool], Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    return _compute_via_integrated_gradients(state, steps, evaluate_step, allow_conductance=True)


def _run_grad(state: ContributionComputationState, steps: int, evaluate_step: Callable[[float, torch.Tensor, bool], Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    del steps
    return _compute_via_gradient(state, evaluate_step)


def _run_input_x_grad(state: ContributionComputationState, steps: int, evaluate_step: Callable[[float, torch.Tensor, bool], Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    del steps
    return _compute_via_input_x_grad(state, evaluate_step)


_METHOD_IMPLEMENTATIONS = {
    "ig": _run_ig,
    "ig_target": _run_ig,
    "ig_conductance": _run_ig_conductance,
    "grad": _run_grad,
    "input_x_grad": _run_input_x_grad,
}
