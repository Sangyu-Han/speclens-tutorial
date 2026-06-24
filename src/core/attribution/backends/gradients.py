from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel  # PyTorch >= 2.2
except ImportError:
    # PyTorch 2.1 compatibility
    from torch.backends.cuda import sdp_kernel as _sdp_kernel_old
    import contextlib, enum

    class SDPBackend(enum.Enum):
        MATH = 0
        FLASH_ATTENTION = 1
        EFFICIENT_ATTENTION = 2
        CUDNN_ATTENTION = 3

    @contextlib.contextmanager
    def sdpa_kernel(backends):
        use_math = SDPBackend.MATH in backends
        use_flash = SDPBackend.FLASH_ATTENTION in backends
        use_eff = SDPBackend.EFFICIENT_ATTENTION in backends
        with _sdp_kernel_old(enable_flash=use_flash, enable_math=use_math, enable_mem_efficient=use_eff):
            yield
@dataclass
class ObjectiveTensor:
    tensor: torch.Tensor
    weight: Optional[torch.Tensor] = None


@dataclass
class _PreparedObjective:
    target: torch.Tensor
    grad_output: Optional[torch.Tensor]
    scalar_value: torch.Tensor

# ---------------- registry -----------------
ObjectiveLike = Union[torch.Tensor, ObjectiveTensor]

BACKENDS: Dict[str, Callable[..., Callable[[], Dict[str, torch.Tensor]]]] = {}
_MISSING_GRAD_WARNED: Set[str] = set()

def _attr_from_anchor_key(key: str) -> str:
    """Extract attribute name (normalized) from anchor key strings."""
    if "#" in key:
        raw = key.split("#", 1)[1].lower()
    else:
        raw = ""
    if raw in {"latent", "acts", "activation"} or raw == "":
        return "acts"
    if raw in {"error_coeff", "error", "residual_coeff"}:
        return "error_coeff"
    if raw in {"residual", "sae_error"}:
        return "residual"
    return raw or "acts"


def _init_accumulators(
    template: Dict[str, torch.Tensor],
    *,
    with_sumsq: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    totals: Dict[str, torch.Tensor] = {}
    sumsqs: Optional[Dict[str, torch.Tensor]] = {} if with_sumsq else None
    for k, g in template.items():
        zero = torch.zeros_like(g).cpu()
        totals[k] = zero.clone()
        if sumsqs is not None:
            sumsqs[k] = zero.clone()
    return totals, sumsqs


def _maybe_baseline_for_anchor(
    anchor_val: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    base: Optional[torch.Tensor],
) -> torch.Tensor:
    if isinstance(anchor_val, (list, tuple)):
        tensors = [t for t in anchor_val if torch.is_tensor(t)]
        if not tensors:
            raise RuntimeError("Anchor list is empty; cannot build baseline")
        stacked = torch.stack(tensors, dim=0)
        if base is not None:
            base = base.to(device=stacked.device, dtype=stacked.dtype)
            return base
        return torch.zeros_like(stacked)
    if base is not None:
        return base.to(device=anchor_val.device, dtype=anchor_val.dtype)
    return torch.zeros_like(anchor_val)

def register(name: str):
    def deco(fn):
        BACKENDS[name] = fn
        return fn
    return deco

def _prepare_objective(
    value: Union[torch.Tensor, ObjectiveTensor]
) -> _PreparedObjective:
    if isinstance(value, ObjectiveTensor):
        tensor = value.tensor
        grad_output = value.weight
    else:
        tensor = value
        grad_output = None
    if not torch.is_tensor(tensor):
        raise TypeError("objective_getter must return a torch.Tensor or ObjectiveTensor")
    if tensor.ndim == 0:
        prepared = _PreparedObjective(target=tensor, grad_output=None, scalar_value=tensor)
        if os.getenv("ATTR_DEBUG", "0") == "1":
            print("[objective] scalar", float(tensor.detach().cpu()))
        return prepared
    grad = grad_output
    if grad is None:
        grad = torch.ones_like(tensor)
    elif not torch.is_tensor(grad):
        raise TypeError("ObjectiveTensor.weight must be a torch.Tensor")
    elif grad.shape != tensor.shape:
        raise RuntimeError(
            f"ObjectiveTensor.weight shape {tuple(grad.shape)} does not match tensor shape {tuple(tensor.shape)}"
        )
    grad = grad.detach()
    scalar = (tensor * grad).sum()
    if os.getenv("ATTR_DEBUG", "0") == "1":
        print("[objective] tensor_norm", float(tensor.detach().abs().sum().cpu()))
    return _PreparedObjective(target=tensor, grad_output=grad, scalar_value=scalar)


def _grads_wrt_multi(
    objective: Union[torch.Tensor, ObjectiveTensor],
    anchors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return _grads_wrt_multi_allow_missing(objective, anchors, allow_missing_grad=False)


def _grads_wrt_multi_allow_missing(
    objective: Union[torch.Tensor, ObjectiveTensor],
    anchors: Dict[str, torch.Tensor],
    *,
    allow_missing_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    prepared = _prepare_objective(objective)
    return _grads_wrt_multi_prepared(prepared, anchors, allow_missing_grad=allow_missing_grad)


def _grads_wrt_multi_prepared(
    prepared: _PreparedObjective,
    anchors: Dict[str, torch.Tensor],
    *,
    allow_missing_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    if not anchors:
        return {}
    # 디버깅 도우미(필요 시 주석 처리)
    debug_attrs = os.getenv("ATTR_DEBUG", "0") == "1"
    expanded_keys: list[str] = []
    expanded_ins: list[torch.Tensor] = []
    list_map: Dict[str, list[torch.Tensor]] = {}
    missing: Dict[str, List[int] | None] = {}

    for k, v in anchors.items():
        if torch.is_tensor(v):
            if debug_attrs:
                print(f"[grad_anchor] {k} requires_grad={v.requires_grad}")
            if not v.requires_grad:
                if allow_missing_grad:
                    missing[k] = None
                    continue
                raise RuntimeError(f"[attr] anchor '{k}' does not require grad (dtype={getattr(v,'dtype',None)})")
            expanded_keys.append(k)
            expanded_ins.append(v)
        elif isinstance(v, (list, tuple)):
            tensors = [t for t in v if torch.is_tensor(t)]
            if not tensors:
                continue
            missing_idx: List[int] = []
            for idx, t in enumerate(tensors):
                if debug_attrs:
                    print(f"[grad_anchor] {k}[{idx}] requires_grad={t.requires_grad}")
                if not t.requires_grad:
                    if allow_missing_grad:
                        missing_idx.append(idx)
                        continue
                    raise RuntimeError(f"[attr] anchor '{k}[{idx}]' does not require grad (dtype={getattr(t,'dtype',None)})")
                expanded_keys.append(f"{k}::{idx}")
                expanded_ins.append(t)
            list_map[k] = tensors
            if missing_idx:
                missing[k] = missing_idx
        else:
            raise TypeError(f"[attr] anchor '{k}' must be Tensor or list/tuple of Tensors")

    outs = torch.autograd.grad(
        prepared.target,
        expanded_ins,
        grad_outputs=prepared.grad_output,
        retain_graph=False,
        create_graph=False,
        allow_unused=True, # 일부 anchor 가 실제로 사용되지 않을 수 있음 - debugging을 위해선 False로
    )

    # rebuild per-anchor outputs (stack lists back)
    out_dict: Dict[str, torch.Tensor] = {}
    idx = 0
    missing_notes: List[str] = []
    for k, v in anchors.items():
        if torch.is_tensor(v):
            if k in missing:
                out_dict[k] = torch.zeros_like(v)
                missing_notes.append(k)
                continue
            g = outs[idx]
            out_dict[k] = g if g is not None else torch.zeros_like(v)
            idx += 1
        elif isinstance(v, (list, tuple)):
            tensors = [t for t in v if torch.is_tensor(t)]
            if not tensors:
                continue
            grads: list[torch.Tensor] = []
            missing_idx = set(missing.get(k, []) or [])
            for j, t in enumerate(tensors):
                if j in missing_idx:
                    grads.append(torch.zeros_like(t))
                    continue
                g = outs[idx]
                grads.append(g if g is not None else torch.zeros_like(t))
                idx += 1
            # stack along step dimension
            out_dict[k] = torch.stack(grads, dim=0)
            if missing_idx:
                missing_notes.append(f"{k}[{sorted(missing_idx)}]")
    if missing_notes and allow_missing_grad:
        msg = ", ".join(missing_notes)
        if msg not in _MISSING_GRAD_WARNED:
            print(f"[attr][warn] missing grad for anchors: {msg} (returning zeros)")
            _MISSING_GRAD_WARNED.add(msg)
    return out_dict

# --------------- saliency ------------------
@register("grad")
def build_saliency_backend(
    *,
    anchor_tensors_getter: Callable[[], Dict[str, torch.Tensor]],
    objective_getter: Callable[[], ObjectiveLike],
    allow_missing_grad: bool = False,
):
    def run():
        anchors = anchor_tensors_getter()
        return _grads_wrt_multi_allow_missing(objective_getter(), anchors, allow_missing_grad=allow_missing_grad)
    return run

# ----------- input x grad ------------------
@register("input_x_grad")
def build_ixg_backend(
    *,
    anchor_tensors_getter: Callable[[], Dict[str, torch.Tensor]],
    objective_getter: Callable[[], ObjectiveLike],
    anchor_baselines: Optional[Dict[str, torch.Tensor]] = None,
    allow_missing_grad: bool = False,
):
    def run():
        anchors = anchor_tensors_getter()
        g = _grads_wrt_multi_allow_missing(objective_getter(), anchors, allow_missing_grad=allow_missing_grad)
        out: Dict[str, torch.Tensor] = {}
        baselines = anchor_baselines or {}
        for k, gk in g.items():
            anchor_val = anchors[k]
            base = baselines.get(k)
            if isinstance(anchor_val, (list, tuple)):
                tensors = [t for t in anchor_val if torch.is_tensor(t)]
                if not tensors:
                    continue
                stacked = torch.stack(tensors, dim=0)
                if base is not None:
                    base = base.to(device=stacked.device, dtype=stacked.dtype)
                    diff = stacked - base
                else:
                    diff = stacked
                out[k] = diff * gk
            else:
                if base is not None:
                    base = base.to(device=anchor_val.device, dtype=anchor_val.dtype)
                    diff = anchor_val - base
                else:
                    diff = anchor_val
                out[k] = diff * gk
        return out
    return run

# -------- integrated gradients -------------
@register("ig")
def build_ig_backend(
    *,
    anchor_tensors_getter: Callable[[], Dict[str, torch.Tensor]],
    objective_getter: Callable[[], ObjectiveLike],
    steps: int,
    set_alpha: Callable[[float], None],
    do_forward: Callable[[], None],
    release_step_refs: Optional[Callable[[], None]] = None,
    force_sdpa_math: bool = False,
    anchor_baselines: Optional[Dict[str, torch.Tensor]] = None,
    allow_missing_grad: bool = False,
):
    
    steps = max(1, int(steps))
    def run():
        totals_cpu: Dict[str, torch.Tensor] = {}
        ig_debug = os.getenv("ATTR_IG_DEBUG", "0") == "1"
        def _sample(val):
            if torch.is_tensor(val):
                return val.detach().flatten()[0].item() if val.numel() > 0 else 0.0
            if isinstance(val, (list, tuple)) and val:
                t = val[0]
                if torch.is_tensor(t):
                    return t.detach().flatten()[0].item() if t.numel() > 0 else 0.0
            return None
        for k in range(1, steps + 1):
            alpha = float(k) / float(steps)
            if ig_debug:
                print(f"[ig-debug] step={k}/{steps} alpha={alpha:.4f}")
            set_alpha(alpha)
            # forward must run AFTER setting alpha so the anchor module outputs the path point
            if force_sdpa_math:
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    do_forward()
            else:
                do_forward()
            anchors = anchor_tensors_getter()
            if ig_debug:
                for name, tensor in anchors.items():
                    shape = tuple(int(x) for x in tensor.shape) if torch.is_tensor(tensor) else f"list(len={len(tensor)})"
                    print(f"[ig-debug] anchor[{name}] shape={shape} sample={_sample(tensor)}")
            prepared = _prepare_objective(objective_getter())
            g = _grads_wrt_multi_prepared(prepared, anchors, allow_missing_grad=allow_missing_grad)
            if ig_debug:
                for name, grad in g.items():
                    shape = tuple(int(x) for x in grad.shape)
                    print(f"[ig-debug] grad[{name}] shape={shape} sample={_sample(grad)}")
            if not totals_cpu:
                totals_cpu = {
                    key: torch.zeros_like(grad).cpu()
                    for key, grad in g.items()
                }
            for key, grad in g.items():
                totals_cpu[key].add_(grad.detach().cpu())
            del g, prepared
            # ⬇ anchor 원본 텐서 참조 제거 → 다음 step 전 메모리 회수 가능
            if release_step_refs is not None:
                release_step_refs()
            # torch.cuda.empty_cache()
        set_alpha(1.0)  # restore
        # one final forward at alpha=1 to get the endpoint anchor
        try:
            do_forward(require_grad=False)
        except TypeError:
            do_forward()
        anchors = anchor_tensors_getter()
        out: Dict[str, torch.Tensor] = {}
        baselines = anchor_baselines or {}
        for k, a in anchors.items():
            baseline = baselines.get(k)
            if isinstance(a, (list, tuple)):
                tensors = [t for t in a if torch.is_tensor(t)]
                if not tensors:
                    continue
                stacked = torch.stack(tensors, dim=0)
                if baseline is not None:
                    baseline = baseline.to(device=stacked.device, dtype=stacked.dtype)
                else:
                    baseline = torch.zeros_like(stacked)
                total_cpu = totals_cpu.get(k)
                if total_cpu is None:
                    total = baseline
                else:
                    total = total_cpu.to(device=stacked.device, dtype=stacked.dtype)
                out[k] = (stacked - baseline) * (total / steps)
            else:
                if baseline is not None:
                    baseline = baseline.to(device=a.device, dtype=a.dtype)
                else:
                    baseline = torch.zeros_like(a)
                total_cpu = totals_cpu.get(k)
                if total_cpu is None:
                    total = baseline
                else:
                    total = total_cpu.to(device=a.device, dtype=a.dtype)
                out[k] = (a - baseline) * (total / steps)
        return out
    return run


@register("ig_legacy")
def build_ig_legacy_backend(
    *,
    anchor_modules_getter: Callable[[], Dict[str, Any]],
    anchor_tensor_lists_getter: Optional[Callable[[], Dict[str, torch.Tensor]]] = None,
    anchor_tensor_records_getter: Optional[Callable[[], Dict[str, list[Any]]]] = None,
    objective_getter: Callable[[], ObjectiveLike],
    steps: int,
    do_forward: Callable[[], None],
    anchor_baselines: Optional[Dict[str, torch.Tensor]] = None,
    allow_missing_grad: bool = False,
):
    """
    Legacy-style IG that overrides SAE branch anchors directly (no ig_active needed).
    Each anchor is integrated independently by swapping its activations along the
    baseline→input path while other anchors remain untouched.
    """
    steps = max(1, int(steps))

    def run():
        modules = anchor_modules_getter() or {}
        if not modules:
            return {}

        base_stacks: Dict[str, torch.Tensor] = {}
        baseline_stacks: Dict[str, torch.Tensor] = {}
        frame_maps: Dict[str, List[int]] = {}

        def _get_base_and_baseline(name: str, mod: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            # Prefer pre-stacked tensors if provided (recurrent models).
            if anchor_tensor_lists_getter is not None:
                try:
                    stacks = anchor_tensor_lists_getter() or {}
                    base_stack = stacks.get(name)
                    if torch.is_tensor(base_stack):
                        base_stacks[name] = base_stack
                        baseline_stack = (anchor_baselines or {}).get(name)
                        if baseline_stack is None:
                            baseline_stack = torch.zeros_like(base_stack)
                        baseline_stacks[name] = baseline_stack
                        return base_stack, baseline_stack
                except Exception:
                    pass
            if anchor_tensor_records_getter is not None:
                try:
                    records_map = anchor_tensor_records_getter() or {}
                    recs = records_map.get(name) or []
                    if recs:
                        frame_idx_list: List[int] = []
                        tensors: List[torch.Tensor] = []
                        for rec in recs:
                            frame_idx_list.append(int(getattr(rec, "frame_idx", len(frame_idx_list))))
                            t = rec.tensor if hasattr(rec, "tensor") else None
                            if torch.is_tensor(t):
                                tensors.append(t)
                        if tensors:
                            base_stack = torch.stack(tensors, dim=0)
                            frame_maps[name] = frame_idx_list[: len(tensors)]
                            baseline_val = (anchor_baselines or {}).get(name)
                            if baseline_val is None:
                                baseline_stack = torch.zeros_like(base_stack)
                            elif torch.is_tensor(baseline_val) and baseline_val.shape == base_stack.shape:
                                baseline_stack = baseline_val
                            elif torch.is_tensor(baseline_val) and baseline_val.dim() == base_stack.dim() - 1:
                                baseline_stack = baseline_val.unsqueeze(0).expand_as(base_stack)
                            else:
                                baseline_stack = torch.zeros_like(base_stack)
                            base_stacks[name] = base_stack
                            baseline_stacks[name] = baseline_stack
                            return base_stack, baseline_stack
                except Exception:
                    pass

            ctx_getter = getattr(mod, "sae_context", None)
            if ctx_getter is None:
                return None
            ctx = ctx_getter() or {}
            attr = _attr_from_anchor_key(name)
            base = ctx.get(attr)
            if not torch.is_tensor(base):
                return None
            baseline_map = anchor_baselines or {}
            baseline = baseline_map.get(name)
            if baseline is None:
                baseline = torch.zeros_like(base)
            return base, baseline

        out: Dict[str, torch.Tensor] = {}
        # Populate contexts once before integrating anchors.
        with torch.no_grad():
            do_forward()

        for name, mod in modules.items():
            if not hasattr(mod, "set_anchor_override") or not hasattr(mod, "clear_anchor_overrides"):
                continue
            base_pair = _get_base_and_baseline(name, mod)
            if base_pair is None:
                continue
            base, baseline = base_pair
            sae_param = None
            sae_obj = getattr(mod, "sae", None)
            try:
                sae_param = next(sae_obj.parameters())
            except Exception:
                sae_param = None
            target_device = sae_param.device if sae_param is not None else base.device
            target_dtype = sae_param.dtype if sae_param is not None else base.dtype
            base = base.to(device=target_device, dtype=target_dtype)
            baseline = baseline.to(device=target_device, dtype=target_dtype)
            diff = base - baseline
            total = torch.zeros_like(base)
            attr = _attr_from_anchor_key(name)
            frame_getter = getattr(mod, "_current_frame_idx", None)

            def _select_frame(stack: torch.Tensor) -> torch.Tensor:
                if stack.dim() == base.dim() and stack.shape == base.shape:
                    # per-call tensor (no frames)
                    return stack
                if stack.dim() < 2:
                    return stack
                idx = 0
                if callable(frame_getter):
                    try:
                        idx = int(frame_getter())
                    except Exception:
                        idx = 0
                if name in frame_maps:
                    frames = frame_maps[name]
                    if idx in frames:
                        sel = frames.index(idx)
                        return stack[sel]
                idx = max(0, min(idx, stack.shape[0] - 1))
                return stack[idx]

            for k in range(1, steps + 1):
                alpha = float(k) / float(steps)
                scaled = baseline + diff * alpha
                scaled = scaled.detach().clone().requires_grad_(True)
                mod.clear_anchor_overrides()
                mod.set_anchor_override(attr, lambda _t, s=scaled: _select_frame(s))
                do_forward()
                obj = objective_getter()
                grad = torch.autograd.grad(
                    obj,
                    scaled,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad is None:
                    if allow_missing_grad:
                        grad = torch.zeros_like(scaled)
                    else:
                        raise RuntimeError(f"[ig_legacy] missing grad for anchor '{name}'")
                total = total + grad * diff
                mod.clear_anchor_overrides()
            out[name] = total / float(steps)
            try:
                mod.clear_context()
            except Exception:
                pass
        for mod in modules.values():
            try:
                mod.clear_anchor_overrides()
            except Exception:
                pass
        return out

    return run


@register("ig_cached")
def build_ig_cached_backend(
    *,
    anchor_tensors_getter: Callable[[], Dict[str, torch.Tensor]],
    objective_getter: Callable[[], ObjectiveLike],
    steps: int,
    set_alpha: Callable[[float], None],
    do_forward: Callable[[], None],
    release_step_refs: Optional[Callable[[], None]] = None,
    force_sdpa_math: bool = False,
    anchor_baselines: Optional[Dict[str, torch.Tensor]] = None,
    allow_missing_grad: bool = False,
):
    """
    Alias of IG that works with cached IG targets in the capture layer.
    MultiAnchorCapture can be switched to use pre-recorded activations
    via ig_use_cached_targets(True) before invoking this backend.
    """
    return build_ig_backend(
        anchor_tensors_getter=anchor_tensors_getter,
        objective_getter=objective_getter,
        steps=steps,
        set_alpha=set_alpha,
        do_forward=do_forward,
        release_step_refs=release_step_refs,
        force_sdpa_math=force_sdpa_math,
        anchor_baselines=anchor_baselines,
        allow_missing_grad=allow_missing_grad,
    )

@register("ig_target")
def build_ig_target_backend(
    *,
    anchor_tensors_getter: Callable[[], Dict[str, torch.Tensor]],
    objective_getter: Callable[[], ObjectiveLike],
    steps: int,
    set_alpha: Callable[[float], None],
    do_forward: Callable[..., None],
    release_step_refs: Optional[Callable[[], None]] = None,
    force_sdpa_math: bool = False,
    anchor_baselines: Optional[Dict[str, torch.Tensor]] = None,
    allow_missing_grad: bool = False,
    **_: Any,
):
    """
    Targeted IG variant: alias of IG with the intent that only the target unit moves
    along the path. The forward runtime handles the path masking; backward reuse of
    IG keeps compatibility with existing captures.
    """
    return build_ig_backend(
        anchor_tensors_getter=anchor_tensors_getter,
        objective_getter=objective_getter,
        steps=steps,
        set_alpha=set_alpha,
        do_forward=do_forward,
        release_step_refs=release_step_refs,
        force_sdpa_math=force_sdpa_math,
        anchor_baselines=anchor_baselines,
        allow_missing_grad=allow_missing_grad,
    )

# -------- IG with conductance-like weighting (u * grad) ------------- # u * grad 말고 input x grad 적분이 더 나을거같은데? u를 곱해주면 inhibitory concept 반영도가 떨어질듯듯
@register("ig_conductance") 
def build_ig_conductance_backend(
    *,
    anchor_tensors_getter: Callable[[], Dict[str, torch.Tensor]],
    objective_getter: Callable[[], ObjectiveLike],
    steps: int,
    set_alpha: Callable[[float], None],
    do_forward: Callable[..., None],
    release_step_refs: Optional[Callable[[], None]] = None,
    force_sdpa_math: bool = False,
    anchor_baselines: Optional[Dict[str, torch.Tensor]] = None,
    allow_missing_grad: bool = False,
):
    """
    Input×Grad를 alpha 경로에서 적분한 변형 IG.
    각 스텝에서 anchor(alpha)와 baseline 차이를 곱한 뒤 기울기를 곱해 누적합니다.
    baseline은 anchor_baselines에 주어진 값을 사용(없으면 0).
    """
    steps = max(1, int(steps))

    def run():
        totals_cpu: Dict[str, torch.Tensor] = {}
        baselines = anchor_baselines or {}
        for k in range(1, steps + 1):
            alpha = float(k) / float(steps)
            set_alpha(alpha)
            if force_sdpa_math:
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    do_forward()
            else:
                do_forward()

            prepared = _prepare_objective(objective_getter())
            anchors = anchor_tensors_getter()
            g = _grads_wrt_multi_prepared(prepared, anchors, allow_missing_grad=allow_missing_grad)
            for key, grad in g.items():
                anchor_val = anchors[key]
                base = baselines.get(key)
                if isinstance(anchor_val, (list, tuple)):
                    tensors = [t for t in anchor_val if torch.is_tensor(t)]
                    if not tensors:
                        continue
                    stacked = torch.stack(tensors, dim=0)
                    if base is not None:
                        base = base.to(device=stacked.device, dtype=stacked.dtype)
                    else:
                        base = torch.zeros_like(stacked)
                    diff = stacked - base
                else:
                    if base is not None:
                        base = base.to(device=anchor_val.device, dtype=anchor_val.dtype)
                    else:
                        base = torch.zeros_like(anchor_val)
                    diff = anchor_val - base

                contrib = (diff * grad).detach().cpu()
                if key not in totals_cpu:
                    totals_cpu[key] = torch.zeros_like(contrib)
                totals_cpu[key].add_(contrib)

            del g, anchors, prepared
            if release_step_refs is not None:
                release_step_refs()

        set_alpha(1.0)
        try:
            do_forward(require_grad=False)
        except TypeError:
            do_forward()
        anchors = anchor_tensors_getter()
        out: Dict[str, torch.Tensor] = {}
        for k, a in anchors.items():
            total = totals_cpu.get(k)
            if total is None:
                if isinstance(a, (list, tuple)):
                    tensors = [t for t in a if torch.is_tensor(t)]
                    if not tensors:
                        continue
                    zeros = torch.zeros_like(torch.stack(tensors, dim=0))
                else:
                    zeros = torch.zeros_like(a)
                total = zeros.cpu()
            if isinstance(a, (list, tuple)):
                tensors = [t for t in a if torch.is_tensor(t)]
                if not tensors:
                    continue
                stacked = torch.stack(tensors, dim=0)
                out[k] = total.to(device=stacked.device, dtype=stacked.dtype) / steps
            else:
                out[k] = total.to(device=a.device, dtype=a.dtype) / steps
        return out
    return run

# -------- SmoothGrad (mean of noisy grads) -------------
@register("smoothgrad")
def build_smoothgrad_backend(
    *,
    anchor_tensors_getter: Callable[[], Dict[str, torch.Tensor]],
    objective_getter: Callable[[], ObjectiveLike],
    do_forward: Callable[[], None],
    samples: int,
    allow_missing_grad: bool = False,
    compute_variance: bool = False,
):
    samples = max(1, int(samples))

    def run():
        totals_cpu: Dict[str, torch.Tensor] = {}
        sumsqs_cpu: Optional[Dict[str, torch.Tensor]] = {} if compute_variance else None
        for _ in range(samples):
            do_forward()
            grads = _grads_wrt_multi_allow_missing(objective_getter(), anchor_tensors_getter(), allow_missing_grad=allow_missing_grad)
            if not totals_cpu:
                totals_cpu, sumsqs_cpu = _init_accumulators(grads, with_sumsq=compute_variance)
            for k, g in grads.items():
                g_cpu = g.detach().cpu()
                totals_cpu[k].add_(g_cpu)
                if sumsqs_cpu is not None:
                    sumsqs_cpu[k].add_(g_cpu.pow(2))
        if not totals_cpu:
            return {}
        mean = {k: v / float(samples) for k, v in totals_cpu.items()}
        if not compute_variance or sumsqs_cpu is None:
            return mean
        var = {}
        for k in mean.keys():
            var[k] = (sumsqs_cpu[k] / float(samples)) - mean[k].pow(2)
        return var

    return run


# -------- VarGrad (variance of noisy grads) -------------
@register("vargrad")
def build_vargrad_backend(
    *,
    anchor_tensors_getter: Callable[[], Dict[str, torch.Tensor]],
    objective_getter: Callable[[], ObjectiveLike],
    do_forward: Callable[[], None],
    samples: int,
    allow_missing_grad: bool = False,
):
    return build_smoothgrad_backend(
        anchor_tensors_getter=anchor_tensors_getter,
        objective_getter=objective_getter,
        do_forward=do_forward,
        samples=samples,
        allow_missing_grad=allow_missing_grad,
        compute_variance=True,
    )


# -------- AnnealedConcrete (optimization-based ERF) -------------
@register("annealed_concrete")
def build_annealed_concrete_backend(
    *,
    objective_getter: Callable[[], ObjectiveLike],
    do_forward_masked: Callable[[torch.Tensor], None],
    distribution_getter: Optional[Callable[[], torch.Tensor]] = None,
    acts_orig_spatial: Optional[torch.Tensor] = None,
    loss_mode: str = "recovery",
    pearson_weight: float = 0.5,
    irrelevance: Optional[torch.Tensor] = None,
    irrelevance_weight: float = 1.0,
    tv_weight: float = 0.0,
    n_patches: int = 196,
    lam: float = 0.02,
    steps: int = 25,
    lr: float = 0.15,
    lr_end: Optional[float] = None,
    tau_start: float = 3.0,
    tau_end: float = 0.05,
    init_prob: float = 0.9,
    init_log_alphas: Optional[torch.Tensor] = None,
    optimizer_name: Optional[str] = None,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    cautious_rescale: bool = True,
    hc_beta: float = 2.0 / 3.0,
    hc_gamma: float = -0.1,
    hc_zeta: float = 1.1,
    seed: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    **_: Any,
) -> Callable[[], Dict[str, torch.Tensor]]:
    """
    Optimization-based ERF via Hard-Concrete mask relaxation.

    loss_mode:
      "recovery"  : (1 - act/act_orig) + lam*L0
      "pearson"   : (1 - pearson(acts_spatial, acts_orig_spatial)) + lam*L0
                    Requires distribution_getter + acts_orig_spatial.
      "combined"  : recovery + pearson_weight * pearson_spatial + lam*L0
                    Requires distribution_getter + acts_orig_spatial.

    optimizer_name:
      None                   : default to cautious Adam + cosine for soft insertion/deletion,
                               plain Adam for the legacy HC losses.
      "adam"                 : torch.optim.Adam with fixed lr.
      "cautious_adam"        : Adam moments + cautious sign mask, fixed lr.
      "cautious_adam_cosine" : cautious Adam with cosine lr decay from lr to lr_end.

    distribution_getter(): called after do_forward_masked; returns [n_patches]
        SAE activations for the target feature (in grad graph).
    acts_orig_spatial: [n_patches] precomputed original spatial activations (no grad).
    """
    import math

    def _logit(p: float) -> float:
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    def _hc_l0(log_alpha: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(log_alpha - hc_beta * math.log(-hc_gamma / hc_zeta))

    def _pearson_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """1 - pearson(x, y).  Differentiable w.r.t. x; y is treated as fixed."""
        xm = x - x.mean()
        ym = y - y.mean()
        cos = (xm * ym).sum() / (xm.norm() * ym.norm() + 1e-8)
        return 1.0 - cos

    def run() -> Dict[str, torch.Tensor]:
        dev = device
        if dev is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(dev)

        torch.manual_seed(seed)

        # Reference forward (z=1 everywhere, no mask)
        with torch.no_grad():
            ones = torch.ones(n_patches, device=dev)
            do_forward_masked(ones)
            act_orig_val = _prepare_objective(objective_getter()).scalar_value.detach()
            zeros = torch.zeros(n_patches, device=dev)
            do_forward_masked(zeros)
            act_base_val = _prepare_objective(objective_getter()).scalar_value.detach()
            acts_orig = (
                acts_orig_spatial.to(dev).detach()
                if acts_orig_spatial is not None
                else None
            )

        def _recovery_loss(act_masked: torch.Tensor) -> torch.Tensor:
            denom = act_orig_val - act_base_val
            denom_safe = torch.where(
                denom.abs() >= 1e-8,
                denom,
                torch.where(denom >= 0, torch.full_like(denom, 1e-8), torch.full_like(denom, -1e-8)),
            )
            recovery = (act_masked - act_base_val) / denom_safe
            return 1.0 - recovery

        if init_log_alphas is not None:
            log_alphas = init_log_alphas.detach().to(dev).clone().requires_grad_(True)
        else:
            log_alphas = torch.full(
                (n_patches,), _logit(init_prob),
                device=dev, requires_grad=True,
            )

        resolved_optimizer = optimizer_name
        if resolved_optimizer is None:
            if loss_mode in {"soft_ins_auc", "soft_del_auc"}:
                resolved_optimizer = "cautious_adam_cosine"
            else:
                resolved_optimizer = "adam"
        valid_optimizers = {"adam", "cautious_adam", "cautious_adam_cosine"}
        if resolved_optimizer not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer_name={resolved_optimizer!r}; "
                f"expected one of {sorted(valid_optimizers)}"
            )

        final_lr = lr_end
        if final_lr is None:
            final_lr = 0.01 if resolved_optimizer == "cautious_adam_cosine" else lr

        opt = None
        m = None
        v = None
        if resolved_optimizer == "adam":
            opt = torch.optim.Adam(
                [log_alphas],
                lr=lr,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
            )
        else:
            m = torch.zeros_like(log_alphas)
            v = torch.zeros_like(log_alphas)

        irr = irrelevance.to(dev).detach() if irrelevance is not None else None

        grid_h = grid_w = int(n_patches ** 0.5)  # 14 for 196 patches

        def _tv_loss(z: torch.Tensor) -> torch.Tensor:
            g = z.view(grid_h, grid_w)
            return (g[:, :-1] - g[:, 1:]).abs().sum() + (g[:-1, :] - g[1:, :]).abs().sum()

        eps = 1e-6
        for step in range(steps):
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            elif log_alphas.grad is not None:
                log_alphas.grad.zero_()

            frac = step / max(steps - 1, 1)
            tau = tau_start * (tau_end / tau_start) ** frac

            if loss_mode == "soft_ins_auc":
                # Direct soft-insertion AUC optimization.
                # At each step, sample a budget b ~ Uniform(0, N) and compute the
                # soft-insertion weights w_i = clamp(p_i * b, 1) where p_i is the
                # normalized probability mass derived from log_alphas.
                # Loss = 1 - baseline-corrected recovery.
                # This is differentiable w.r.t. log_alphas through p → w → h_inj → act.
                # No HC sampling needed here — skip the z computation entirely.
                probs = torch.sigmoid(log_alphas)              # [n_patches]
                p = probs / (probs.sum() + 1e-8)               # normalized, sums to 1
                b_sample = torch.rand(1, device=dev).item() * n_patches
                w = (p * b_sample).clamp(max=1.0)              # soft-ins weights
                irr_penalty = (
                    irrelevance_weight * (torch.sigmoid(log_alphas) * irr).sum()
                    if irr is not None else 0.0
                )
                tv_penalty = tv_weight * _tv_loss(probs) if tv_weight > 0.0 else 0.0
                do_forward_masked(w)
                act_masked = _prepare_objective(objective_getter()).scalar_value
                loss = _recovery_loss(act_masked) + irr_penalty + tv_penalty

            elif loss_mode == "soft_del_auc":
                # Deletion-direction soft AUC optimization.
                # probs = sigmoid(log_alphas) = "importance / keep-probability" of each patch.
                # Deletion mass: q_i = (1 - probs_i) / sum(1 - probs) — normalized removal weights.
                # At each step, sample budget b_del ~ Uniform(0, N) and compute:
                #   del_w_i = clamp(q_i * b_del, 1)  = how much each patch is removed
                #   keep_w_i = 1 - del_w_i           = effective keep weight
                # Loss = 1 - baseline-corrected recovery.
                # Gradient: for a critical patch i, removing it hurts → ∂loss/∂probs_i > 0
                #   → optimizer increases probs_i → q_i decreases → patch is protected.
                # For irrelevant patches: removing them doesn't hurt → probs_i stays low.
                # This yields a calibrated sparse distribution where important patches have
                # high probs (matching their true marginal contribution from the full-set view).
                probs = torch.sigmoid(log_alphas)              # [n_patches]
                q = (1.0 - probs) / ((1.0 - probs).sum() + 1e-8)  # deletion mass
                b_del = torch.rand(1, device=dev).item() * n_patches
                del_w = (q * b_del).clamp(max=1.0)            # how much each patch is deleted
                keep_w = 1.0 - del_w                           # effective keep weight
                irr_penalty = (
                    irrelevance_weight * (probs * irr).sum()
                    if irr is not None else 0.0
                )
                tv_penalty = tv_weight * _tv_loss(probs) if tv_weight > 0.0 else 0.0
                do_forward_masked(keep_w)
                act_masked = _prepare_objective(objective_getter()).scalar_value
                loss = _recovery_loss(act_masked) + irr_penalty + tv_penalty

            else:
                u = torch.rand_like(log_alphas).clamp(eps, 1 - eps)
                z = torch.sigmoid((log_alphas + torch.log(u) - torch.log(1 - u)) / tau)

                do_forward_masked(z)
                l0 = _hc_l0(log_alphas).sum()

                irr_penalty = (
                    irrelevance_weight * (torch.sigmoid(log_alphas) * irr).sum()
                    if irr is not None else 0.0
                )

                tv_penalty = tv_weight * _tv_loss(z) if tv_weight > 0.0 else 0.0

                if loss_mode == "recovery":
                    act_masked = _prepare_objective(objective_getter()).scalar_value
                    loss = _recovery_loss(act_masked) + lam * l0 + irr_penalty + tv_penalty

                elif loss_mode == "pearson":
                    assert distribution_getter is not None and acts_orig is not None
                    dist_masked = distribution_getter()  # [n_patches], in grad graph
                    loss = _pearson_loss(dist_masked, acts_orig) + lam * l0 + irr_penalty + tv_penalty

                elif loss_mode == "combined":
                    assert distribution_getter is not None and acts_orig is not None
                    act_masked = _prepare_objective(objective_getter()).scalar_value
                    dist_masked = distribution_getter()
                    recovery = _recovery_loss(act_masked)
                    pearson  = _pearson_loss(dist_masked, acts_orig)
                    loss = recovery + pearson_weight * pearson + lam * l0 + irr_penalty + tv_penalty

                else:
                    raise ValueError(f"Unknown loss_mode: {loss_mode!r}")

            loss.backward()

            if opt is not None:
                opt.step()
            else:
                assert m is not None and v is not None
                cur_lr = lr
                if resolved_optimizer == "cautious_adam_cosine":
                    cur_lr = final_lr + 0.5 * (lr - final_lr) * (1 + math.cos(math.pi * frac))

                with torch.no_grad():
                    g = log_alphas.grad.detach()
                    t = step + 1
                    m.mul_(adam_beta1).add_(g, alpha=1.0 - adam_beta1)
                    v.mul_(adam_beta2).addcmul_(g, g, value=1.0 - adam_beta2)
                    m_hat = m / (1.0 - adam_beta1 ** t)
                    v_hat = v / (1.0 - adam_beta2 ** t)
                    adam_dir = m_hat / (v_hat.sqrt() + adam_eps)
                    mask = (adam_dir * g > 0).to(dtype=log_alphas.dtype)
                    if cautious_rescale:
                        n_active = mask.sum().clamp(min=1.0)
                        mask = mask * (float(n_patches) / n_active)
                    log_alphas.add_(adam_dir * mask, alpha=-cur_lr)
                    log_alphas.grad = None

        return {"log_alphas": log_alphas.detach().cpu()}

    return run


# -------- GradientSHAP (noisy input x grad) -------------
@register("gradient_shap")
def build_gradient_shap_backend(
    *,
    anchor_tensors_getter: Callable[[], Dict[str, torch.Tensor]],
    objective_getter: Callable[[], ObjectiveLike],
    do_forward: Callable[[], None],
    samples: int,
    anchor_baselines: Optional[Dict[str, torch.Tensor]] = None,
    allow_missing_grad: bool = False,
):
    samples = max(1, int(samples))

    def run():
        totals_cpu: Dict[str, torch.Tensor] = {}
        baselines = anchor_baselines or {}
        for _ in range(samples):
            do_forward()
            anchors = anchor_tensors_getter()
            grads = _grads_wrt_multi_allow_missing(objective_getter(), anchors, allow_missing_grad=allow_missing_grad)
            if not totals_cpu:
                totals_cpu, _ = _init_accumulators(grads, with_sumsq=False)
            for k, g in grads.items():
                anchor_val = anchors[k]
                base = _maybe_baseline_for_anchor(anchor_val, baselines.get(k))
                if isinstance(anchor_val, (list, tuple)):
                    tensors = [t for t in anchor_val if torch.is_tensor(t)]
                    if not tensors:
                        continue
                    stacked = torch.stack(tensors, dim=0)
                    contrib = (stacked - base) * g
                else:
                    contrib = (anchor_val - base) * g
                totals_cpu[k].add_(contrib.detach().cpu())
        if not totals_cpu:
            return {}
        return {k: v / float(samples) for k, v in totals_cpu.items()}

    return run
