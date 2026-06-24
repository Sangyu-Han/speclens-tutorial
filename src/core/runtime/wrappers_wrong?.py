from __future__ import annotations

import types
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.autograd.forward_ad as forward_ad
import torch.nn as nn

from src.core.attribution.utils import restore_tokens_like
from src.core.hooks.spec import SEP, parse_spec, sanitize_id
from src.core.runtime.capture import LayerCapture
from src.core.sae.activation_stores.hook_helper import flatten_tensor_for_sae
from src.core.attribution.sae.constants import (
    SAE_ANCHOR_KIND_MAP,
    SAE_LAYER_ATTRIBUTE_TENSORS,
    SAE_LAYER_METHOD,
    resolve_sae_request,
)
from src.core.runtime.controllers import ActivationControllerBase
from src.utils.utils import resolve_module


class _SAEAnchorStore:
    __slots__ = ("live",)

    def __init__(self) -> None:
        self.live: Dict[str, Any] = {}

    def clear(self) -> None:
        self.live.clear()

    def set(self, key: str, value: Any) -> None:
        if value is None:
            return
        self.live[key] = value

    def get(self, key: str) -> Optional[Any]:
        return self.live.get(key)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.live)


def _install_sae_anchor_methods(module: torch.nn.Module, store: _SAEAnchorStore):
    registered: List[Tuple[str, Optional[Any]]] = []
    for method_name, attr_key in SAE_ANCHOR_KIND_MAP.items():
        prev_attr = getattr(module, method_name, None)
        if isinstance(prev_attr, torch.nn.Module):
            continue

        def _sae_anchor_method(self, *args, __key=attr_key, __method=method_name):
            tensor = store.get(__key)
            if tensor is None:
                raise RuntimeError(
                    f"SAE anchor '{__method}' on module '{module.__class__.__name__}' was "
                    "requested before a forward pass populated it."
                )
            return tensor

        setattr(module, method_name, types.MethodType(_sae_anchor_method, module))
        registered.append((method_name, prev_attr))
    return registered


def get_sae_anchor_store(module: torch.nn.Module) -> Optional[_SAEAnchorStore]:
    return getattr(module, "_sae_anchor_store", None)


def get_sae_anchor_map(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    store = get_sae_anchor_store(module)
    return store.snapshot() if store is not None else {}


def clear_sae_anchor_store(module: torch.nn.Module) -> None:
    store = get_sae_anchor_store(module)
    if store is not None:
        store.clear()


def _split_module_and_branch(raw: str) -> Tuple[str, Tuple[Any, ...]]:
    """
    Separate the module path from any embedded branch selectors encoded via '@'.
    Example: 'model.memory_encoder@vision_pos_enc' -> ('model.memory_encoder', ('vision_pos_enc',))
    """
    if not raw:
        return raw, ()
    tokens = [tok for tok in raw.split(SEP) if tok != ""]
    if not tokens:
        return raw, ()
    module_name = tokens[0]
    branches: List[Any] = []
    for token in tokens[1:]:
        if token.lstrip("-").isdigit():
            try:
                branches.append(int(token))
                continue
            except ValueError:
                pass
        branches.append(token)
    return module_name, tuple(branches)


def _normalise_branch_tokens(
    base_tokens: Sequence[Any],
    extra: Optional[Any],
) -> Tuple[Any, ...]:
    tokens = list(base_tokens)
    if extra is not None:
        tokens.append(extra)
    return tuple(tokens)


def _format_sae_attr_name(kind: Optional[str], branch_tokens: Sequence[Any]) -> Optional[str]:
    if not kind:
        return None
    suffix = ""
    if branch_tokens:
        suffix = "_" + "_".join(sanitize_id(str(token)) for token in branch_tokens)
    return f"{kind}{suffix}"


class _PhysicalSAEBranch(nn.Module):
    """
    Thin nn.Module wrapper that executes a trained SAE, stores intermediate tensors,
    and exposes a predictable API for downstream attribution code.
    """

    def __init__(
        self,
        *,
        name: str,
        sae: torch.nn.Module,
        anchor_store: _SAEAnchorStore,
        controller: Optional[ActivationControllerBase] = None,
        owner_module: Optional[torch.nn.Module] = None,
        attr_prefix: Optional[str] = None,
        exposed_attrs: Optional[Dict[str, str]] = None,
        frame_getter: Optional[Callable[[], int]] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.sae = sae
        self.controller = controller
        self._anchor_store = anchor_store
        self._sae_anchor_store = anchor_store
        self._context: Dict[str, torch.Tensor] = {}
        self._owner_module = owner_module
        self._attr_prefix = attr_prefix
        self._published_attrs: List[str] = []
        self._last_exposed: Dict[str, torch.Tensor] = {}
        self._exposed_attr_map = dict(exposed_attrs or SAE_LAYER_ATTRIBUTE_TENSORS)
        self._frame_getter = frame_getter
        self._anchor_overrides: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        self._ig_alpha: Optional[float] = None
        self._ig_attr_targets: set[str] = set()
        self._ig_attr_baselines: Dict[str, torch.Tensor] = {}

    def _current_frame_idx(self) -> int:
        getter = self._frame_getter
        if getter is None:
            return 0
        try:
            return int(getter())
        except Exception:
            return 0

    def clear_context(self) -> None:
        self._anchor_store.clear()
        self._context.clear()
        self._clear_owner_attrs()

    def sae_context(self) -> Dict[str, torch.Tensor]:
        return dict(self._context)

    # ------------------------------------------------------------------
    # Anchor overrides (for deletion/insertion experiments)
    # ------------------------------------------------------------------
    def set_anchor_override(self, name: str, fn: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        if fn is None:
            self._anchor_overrides.pop(name, None)
        else:
            self._anchor_overrides[str(name)] = fn

    def clear_anchor_overrides(self) -> None:
        self._anchor_overrides.clear()

    def _apply_anchor_override(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        fn = self._anchor_overrides.get(name)
        if fn is None:
            return tensor
        out = fn(tensor)
        if not torch.is_tensor(out):
            raise RuntimeError(f"Anchor override for '{name}' must return a tensor")
        return out

    # ------------------------------------------------------------------
    # IG attribute interpolation (applied before SAE decode/error split)
    # ------------------------------------------------------------------
    def set_ig_attr_alpha(
        self,
        *,
        alpha: Optional[float],
        active_attrs: Iterable[str],
        baselines: Optional[Dict[str, torch.Tensor]] = None,
        logical_base: Optional[str] = None,
    ) -> None:
        del logical_base
        targets = {str(attr) for attr in active_attrs if attr}
        if not targets or alpha is None:
            self.clear_ig_attr_alpha()
            return
        self._ig_alpha = float(alpha)
        self._ig_attr_targets = targets
        self._ig_attr_baselines = {k: v for k, v in (baselines or {}).items() if torch.is_tensor(v)}

    def clear_ig_attr_alpha(self) -> None:
        self._ig_alpha = None
        self._ig_attr_targets.clear()
        self._ig_attr_baselines.clear()

    def _apply_ig_alpha_attr(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        if self._ig_alpha is None or key not in self._ig_attr_targets:
            return tensor
        alpha = float(self._ig_alpha)
        baseline = self._ig_attr_baselines.get(key)
        if baseline is not None:
            if baseline.device != tensor.device or baseline.dtype != tensor.dtype:
                baseline = baseline.to(device=tensor.device, dtype=tensor.dtype)
            v = baseline + (tensor - baseline) * alpha
        else:
            v = tensor * alpha
        if tensor.requires_grad:
            return v
        base_v = v.detach() if tensor.grad_fn is not None else v
        return base_v + torch.zeros_like(base_v, requires_grad=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        store = self._anchor_store
        store.clear()
        sae = self.sae
        sae_param = next(sae.parameters(), None)
        sae_device = sae_param.device if sae_param is not None else tensor.device
        sae_dtype = sae_param.dtype if sae_param is not None else tensor.dtype

        tensor_for_sae = tensor
        if getattr(sae, "is_dummy_identity", False) and sae_param is None:
            sae.ensure_shape(tensor.shape[-1], tensor.device, tensor.dtype)
            sae_device = tensor.device
            sae_dtype = tensor.dtype

        if not forward_ad._is_fwd_grad_enabled():
            if tensor.dtype != sae_dtype or tensor.device != sae_device:
                tensor_for_sae = tensor.to(device=sae_device, dtype=sae_dtype)
        else:
            if tensor.dtype != sae_dtype or tensor.device != sae_device:
                raise RuntimeError(
                    "[forward-AD] SAE dtype/device must match target layer. "
                    "Move SAE to the target layer's dtype/device before wrapping."
                )

        tokens, reshape_meta = flatten_tensor_for_sae(tensor_for_sae)
        if tokens is None:
            self._context.clear()
            return tensor

        acts = sae.encode(tokens)
        acts_pre = getattr(sae, "_last_pre_acts", None)
        controller = self.controller
        frame_idx = self._current_frame_idx()
        if controller is not None:
            controller.record_pre(frame_idx, acts)
            acts = controller.override(frame_idx, acts, reshape_meta=reshape_meta)
            controller.record_post(frame_idx, acts)

        overrides = self._anchor_overrides
        has_act_override = "acts" in overrides
        has_err_override = "error_coeff" in overrides

        recon_tokens_base = sae.decode(acts)
        recon_base = restore_tokens_like(recon_tokens_base, tensor_for_sae)
        if recon_base is None:
            self._context.clear()
            return tensor

        sae_error = tensor_for_sae - recon_base
        sae_error_base = sae_error.detach()
        recon_l2 = float((sae_error.float().pow(2).mean()).item())
        if os.getenv("SAE_ERROR_DEBUG", "0") == "1":
            print(f"[sae_wrap][{self.name}] recon_l2={recon_l2:.6f}")

        diff_tokens = tokens - recon_tokens_base
        error_const_tokens = diff_tokens.detach()
        error_norm_detached = torch.linalg.vector_norm(error_const_tokens, dim=-1, keepdim=True)
        safe_norm = error_norm_detached.clamp_min(1e-12)
        error_direction = torch.where(
            error_norm_detached > 0,
            error_const_tokens / safe_norm,
            torch.zeros_like(error_const_tokens),
        )
        coeff_requires_grad = torch.is_grad_enabled() or forward_ad._is_fwd_grad_enabled()
        error_norm_live = torch.linalg.vector_norm(diff_tokens, dim=-1, keepdim=True)
        error_coeff_base = error_norm_live.detach()
        if coeff_requires_grad:
            error_coeff_live = error_coeff_base + torch.zeros_like(error_coeff_base, requires_grad=True)
        else:
            error_coeff_live = error_coeff_base

        residual_const = restore_tokens_like(error_const_tokens, tensor_for_sae)
        if residual_const is None:
            residual_const = tensor_for_sae - recon_base

        if forward_ad._is_fwd_grad_enabled():
            def _cast(v: torch.Tensor) -> torch.Tensor:
                return v
        else:
            def _cast(v: torch.Tensor) -> torch.Tensor:
                return v.to(device=tensor.device, dtype=tensor.dtype)
            residual_const = _cast(residual_const)

        error_coeff_alpha = self._apply_ig_alpha_attr("error_coeff", error_coeff_live)
        error_coeff_out = self._apply_anchor_override("error_coeff", error_coeff_alpha) if has_err_override else error_coeff_alpha

        error_proxy_tokens = error_direction * error_coeff_out
        error_proxy = restore_tokens_like(error_proxy_tokens, tensor_for_sae)
        if error_proxy is None:
            raise RuntimeError("Failed to reshape SAE error proxy back to the target tensor shape")
        if forward_ad._is_fwd_grad_enabled():
            error_proxy_cast = error_proxy
        else:
            error_proxy_cast = _cast(error_proxy)

        acts_alpha = self._apply_ig_alpha_attr("acts", acts)
        acts_for_decode = self._apply_anchor_override("acts", acts_alpha) if has_act_override else acts_alpha
        recon_tokens = sae.decode(acts_for_decode)
        recon_masked = restore_tokens_like(recon_tokens, tensor_for_sae)
        if recon_masked is None:
            self._context.clear()
            return tensor

        recon_cast = _cast(recon_masked)
        final_output = recon_cast + error_proxy_cast
        recon_out = recon_cast
        original_error = tensor - recon_cast

        sae_error_out = self._apply_anchor_override("sae_error", sae_error) if "sae_error" in overrides else sae_error
        residual_live = self._apply_anchor_override("residual", residual_const) if "residual" in overrides else residual_const

        store.set("sae_input", tensor_for_sae)
        store.set("tokens", tokens)
        acts_pre_live = acts_pre if acts_pre is not None else acts_alpha
        acts_pre_live = self._apply_ig_alpha_attr("acts_pre", acts_pre_live)
        store.set("acts_pre", acts_pre_live)
        store.set("acts", acts_for_decode)
        store.set("recon_tokens", recon_tokens)
        store.set("recon", recon_masked)
        store.set("sae_error_base", sae_error_base)
        store.set("sae_error", sae_error_out)
        store.set("error_coeff_base", error_coeff_base.detach())
        store.set("error_coeff", error_coeff_out)
        store.set("residual_const", residual_const.detach())
        store.set("residual", residual_live)
        store.set("recon_cast", recon_out)
        store.set("output", final_output)
        store.set("original", tensor)
        store.set("original_error", original_error)
        store.set("recon_l2_loss", sae_error.new_tensor(recon_l2))
        store.set("reshape_meta", reshape_meta)

        self._context = store.snapshot()
        self._publish_owner_attrs()
        return final_output

    def _clear_owner_attrs(self) -> None:
        owner = self._owner_module
        if owner is None:
            self._published_attrs.clear()
            self._last_exposed.clear()
            return
        for name in self._published_attrs:
            try:
                delattr(owner, name)
            except AttributeError:
                pass
        self._published_attrs.clear()
        self._last_exposed.clear()

    def _publish_owner_attrs(self) -> None:
        owner = self._owner_module
        prefix = self._attr_prefix
        if owner is None or not prefix:
            self._last_exposed.clear()
            return
        self._clear_owner_attrs()
        exposures: Dict[str, torch.Tensor] = {}
        for alias, store_key in self._exposed_attr_map.items():
            tensor = self._context.get(store_key)
            if not torch.is_tensor(tensor):
                continue
            exposures[alias] = tensor
            attr_name = f"{prefix}_{alias}"
            setattr(owner, attr_name, tensor)
            self._published_attrs.append(attr_name)
        self._last_exposed = exposures

    def exposed_tensors(self, names: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
        if names is None:
            return dict(self._last_exposed)
        selected: Dict[str, torch.Tensor] = {}
        for name in names:
            tensor = self._last_exposed.get(name)
            if torch.is_tensor(tensor):
                selected[name] = tensor
        return selected


class _SAEAttachment:
    __slots__ = ("base", "allowed", "module", "attr_name", "prev_attr")

    def __init__(
        self,
        *,
        base: str,
        branch: Optional[Any],
        module: _PhysicalSAEBranch,
        attr_name: Optional[str] = None,
        prev_attr: Optional[Any] = None,
    ) -> None:
        self.base = base
        self.allowed = (
            [base] if branch is None else [f"{base}{SEP}{branch}"]
        )
        self.module = module
        self.attr_name = attr_name
        self.prev_attr = prev_attr

    def _allow(self, name: str) -> bool:
        for candidate in self.allowed:
            if name == candidate or name.startswith(candidate + SEP) or candidate.startswith(name + SEP):
                return True
        return False

    def _rewrite(self, prefix: str, obj: Any) -> Any:
        if not self._allow(prefix):
            return obj
        if torch.is_tensor(obj):
            return self.module(obj)
        if isinstance(obj, tuple):
            return tuple(self._rewrite(f"{prefix}{SEP}{idx}", it) for idx, it in enumerate(obj))
        if isinstance(obj, list):
            return [self._rewrite(f"{prefix}{SEP}{idx}", it) for idx, it in enumerate(obj)]
        if isinstance(obj, dict):
            return {k: self._rewrite(f"{prefix}{SEP}{k}", it) for k, it in obj.items()}
        return obj

    def apply(self, out: Any) -> Any:
        return self._rewrite(self.base, out)


class SAEAnchorView:
    """Lightweight view that exposes SAE anchor tensors as attributes."""

    _ALIASES = {"recon": "recon_cast"}

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def _resolve(self, name: str) -> torch.Tensor:
        store = get_sae_anchor_store(self._module)
        if store is None:
            raise RuntimeError("SAE anchor store is not attached to this module.")
        key = self._ALIASES.get(name, name)
        tensor = store.get(key)
        if tensor is None:
            raise AttributeError(f"SAE anchor '{key}' is not available on this module.")
        return tensor

    def __getattr__(self, name: str) -> torch.Tensor:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._resolve(name)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return get_sae_anchor_map(self._module)


class SAEHookHandle:
    """Helper that provides convenient access to the latest SAE anchor tensors."""

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def last(self) -> Optional[SAEAnchorView]:
        store = get_sae_anchor_store(self._module)
        if store is None or not store.live:
            return None
        return SAEAnchorView(self._module)

    def clear(self) -> None:
        clear_sae_anchor_store(self._module)


def wrap_target_layer_with_sae(
    module: torch.nn.Module,
    *,
    capture: LayerCapture,
    sae: torch.nn.Module,
    controller: Optional[ActivationControllerBase] = None,
    frame_getter: Optional[Callable[[], int]] = None,
) -> Callable[[], None]:
    """Patch a module's forward to decode via an SAE and optionally attach a controller."""

    layer_name = capture.full
    base = capture.base
    branch = capture.branch
    _module_name, base_branch_tokens = _split_module_and_branch(base)
    branch_tokens = _normalise_branch_tokens(base_branch_tokens, branch)
    _spec_base, method, _spec_branch, _alias, attr_suffix = parse_spec(layer_name)
    canonical_method, _ = resolve_sae_request(method, attr_suffix)
    attr_kind = canonical_method or method

    anchor_store = _SAEAnchorStore()
    physical_module = _PhysicalSAEBranch(
        name=layer_name,
        sae=sae,
        anchor_store=anchor_store,
        controller=controller,
        owner_module=module,
        attr_prefix=_format_sae_attr_name(attr_kind, branch_tokens),
        exposed_attrs=SAE_LAYER_ATTRIBUTE_TENSORS,
        frame_getter=frame_getter,
    )

    attachments = getattr(module, "_sae_attachments", None)
    if attachments is None:
        attachments = {}
        setattr(module, "_sae_attachments", attachments)

    attr_name = _format_sae_attr_name(attr_kind, branch_tokens)
    prev_attr = None
    if attr_name:
        existing = getattr(module, attr_name, None)
        if existing is None:
            setattr(module, attr_name, physical_module)
        elif isinstance(existing, _PhysicalSAEBranch):
            prev_attr = existing
        else:
            raise RuntimeError(
                f"Cannot attach SAE attribute '{attr_name}' to module '{module.__class__.__name__}': "
                "a conflicting attribute already exists. Ensure physical SAE modules are wired directly "
                "instead of relying on legacy virtual specs."
            )

    attachment_key = layer_name
    attachment = _SAEAttachment(
        base=base,
        branch=branch,
        module=physical_module,
        attr_name=attr_name,
        prev_attr=prev_attr,
    )
    attachments[attachment_key] = attachment

    if not hasattr(module, "_sae_forward_orig"):
        orig_forward = module.forward

        def _wrapped(self, *args, **kwargs):
            out = orig_forward(*args, **kwargs)
            active = getattr(self, "_sae_attachments", None)
            if not active:
                return out
            result = out
            for att in active.values():
                result = att.apply(result)
            return result

        module.forward = types.MethodType(_wrapped, module)
        setattr(module, "_sae_forward_orig", orig_forward)

    if not hasattr(module, "_sae_anchor_methods_prev"):
        registered_anchor_methods = _install_sae_anchor_methods(module, anchor_store)
        setattr(module, "_sae_anchor_methods_prev", registered_anchor_methods)

    setattr(module, "_sae_anchor_store", anchor_store)

    def _restore() -> None:
        attachments = getattr(module, "_sae_attachments", None)
        if attachments is not None:
            attachments.pop(attachment_key, None)

        if attr_name:
            if prev_attr is None:
                try:
                    delattr(module, attr_name)
                except AttributeError:
                    pass
            else:
                setattr(module, attr_name, prev_attr)

        if attachments:
            return

        orig_forward = getattr(module, "_sae_forward_orig", None)
        if orig_forward is not None:
            module.forward = orig_forward
            delattr(module, "_sae_forward_orig")

        registered_anchor_methods = getattr(module, "_sae_anchor_methods_prev", None)
        if registered_anchor_methods is not None:
            for name, prev in registered_anchor_methods:
                try:
                    if prev is None:
                        delattr(module, name)
                    else:
                        setattr(module, name, prev)
                except AttributeError:
                    pass
            delattr(module, "_sae_anchor_methods_prev")

        try:
            delattr(module, "_sae_anchor_store")
        except AttributeError:
            pass

        try:
            delattr(module, "_sae_attachments")
        except AttributeError:
            pass

    return _restore, physical_module


def install_sae_wrappers_for_specs(
    *,
    model: torch.nn.Module,
    specs: Iterable[str],
    sae_resolver: Callable[[str], torch.nn.Module],
    frame_getter: Optional[Callable[[], int]] = None,
) -> List[Callable[[], None]]:
    """
    Ensure every SAE-layer spec in `specs` has a physical wrapper attached to `model`.

    Args:
        model: target model (potentially wrapped in DDP) to patch.
        specs: iterable of tensor specs (e.g. "...::sae_layer#latent").
        sae_resolver: callable that returns a torch.nn.Module SAE for a given spec.
        frame_getter: optional callable returning current frame index.

    Returns:
        List of restoration handles. Call each handle to remove the wrapper.
    """
    handles: List[Callable[[], None]] = []
    installed: set[str] = set()
    for spec in specs:
        spec = str(spec or "").strip()
        if not spec:
            continue
        parsed = parse_spec(spec)
        canonical_method, _ = resolve_sae_request(parsed.method, parsed.attr)
        if canonical_method != SAE_LAYER_METHOD:
            continue
        key = parsed.base_with_branch
        if key in installed:
            continue
        sae_module = sae_resolver(spec)
        if sae_module is None:
            raise RuntimeError(f"SAE resolver returned None for spec '{spec}'")
        capture = LayerCapture(spec)
        owner_module = resolve_module(model, capture.base)
        handle, _ = wrap_target_layer_with_sae(
            owner_module,
            capture=capture,
            sae=sae_module,
            controller=None,
            frame_getter=frame_getter,
        )
        handles.append(handle)
        installed.add(key)
    return handles
