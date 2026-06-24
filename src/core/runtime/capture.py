from __future__ import annotations

import copy
import os
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from src.core.hooks.spec import (
    METHOD_SEP,
    SEP,
    canonical_name,
    parse_spec,
    sanitize_id,
    split_layer_and_branch,
    walk_tensors,
)
from src.core.runtime.activation_tape import ActivationTape
from src.core.attribution.sae.constants import ( # 얘는 옮기자. 
    SAE_LAYER_ATTRIBUTE_TENSORS,
    SAE_LAYER_METHOD,
    resolve_sae_request,
)
from src.utils.utils import resolve_module


_ig_debug_seen: dict[str, int] = {}
_IG_DEBUG_SUPPRESS_AFTER = 8


def _ig_debug(msg: str) -> None:
    if os.getenv("ATTR_IG_DEBUG", "0") != "1":
        return
    count = _ig_debug_seen.get(msg, 0)
    if count >= _IG_DEBUG_SUPPRESS_AFTER:
        if count == _IG_DEBUG_SUPPRESS_AFTER:
            print(f"[ig-debug] suppressing further repeats of: {msg}")
        _ig_debug_seen[msg] = count + 1
        return
    _ig_debug_seen[msg] = count + 1
    print(msg)


def _split_module_and_branch(raw: str) -> Tuple[str, Tuple[Any, ...]]:
    if not raw:
        return raw, ()
    tokens = [tok for tok in raw.split(SEP) if tok != ""]
    if not tokens:
        return raw, ()
    module_name = tokens[0]
    branches: list[Any] = []
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


def _format_attr_prefix(kind: Optional[str], branch_tokens: Sequence[Any]) -> Optional[str]:
    if not kind:
        return None
    suffix = ""
    if branch_tokens:
        suffix = "_" + "_".join(sanitize_id(str(token)) for token in branch_tokens)
    return f"{kind}{suffix}"


def _physical_base_name(logical: Optional[str]) -> str:
    base = (logical or "").strip()
    if not base or SEP not in base:
        return base
    return base.split(SEP, 1)[0]


class LayerCapture:
    """
    Hook helper that records tensors for a target layer/method.

    Captured tensors are stored inside an ActivationTape so that recurrent runs
    (multiple frames) can later be stacked without breaking the computation graph.
    """

    def __init__(self, layer_name: str):
        self.full = layer_name
        self.base, self.method, self.branch, _alias, self.attr = parse_spec(layer_name)
        self.by_frame: Dict[int, torch.Tensor] = {}
        self.last: Optional[torch.Tensor] = None
        self._orig_method = None
        self._attr_accessor: Optional[str] = self._resolve_attr_accessor()
        self._tape = ActivationTape()

    def hook(self, adapter, *, frame_var_name: str = "_frame_idx_var"):
        if self.attr is not None:
            return self._hook_attribute(adapter, frame_var_name=frame_var_name)
        assert self.method is None, "method specs must use hook_method"
        base = self.base
        allowed = [f"{base}{SEP}{self.branch}"] if self.branch is not None else [base]

        def _fn(_module, _inp, out):
            self._capture_from_output(base, allowed, out, adapter, frame_var_name)

        return _fn

    def hook_method(self, module, adapter, *, frame_var_name: str = "_frame_idx_var"):
        if self.attr is not None:
            return module.register_forward_hook(self._hook_attribute(adapter, frame_var_name=frame_var_name))
        assert self.method is not None
        probe_prefix = f"{self.base}{METHOD_SEP}{self.method}"
        allowed = [f"{probe_prefix}{SEP}{self.branch}"] if self.branch is not None else [probe_prefix]

        try:
            bound = getattr(module, self.method)
        except AttributeError:
            return module.register_forward_hook(
                self._forward_fallback(probe_prefix, allowed, adapter, frame_var_name)
            )
        orig = bound.__func__ if hasattr(bound, "__func__") else bound

        def _wrapped(self_obj, *args, **kwargs):
            out = orig(self_obj, *args, **kwargs)
            self._capture_from_output(probe_prefix, allowed, out, adapter, frame_var_name)
            return out

        setattr(module, self.method, types.MethodType(_wrapped, module))
        self._orig_method = bound

        class _Rem:
            def remove(_self):
                try:
                    setattr(module, self.method, bound)
                except Exception:
                    pass

        return _Rem()

    def _forward_fallback(self, prefix, allowed, adapter, frame_var_name):
        def _hook(_module, _inp, out):
            self._capture_from_output(prefix, allowed, out, adapter, frame_var_name)

        return _hook

    def _capture_from_output(self, prefix, allowed, out, adapter, frame_var_name):
        pairs = walk_tensors(prefix, out, allowed_prefixes=allowed)
        ten = pairs[0][1] if pairs else None
        if ten is not None:
            self._record_tensor(ten, adapter, frame_var_name)

    def _hook_attribute(self, adapter, *, frame_var_name: str = "_frame_idx_var"):
        attr_name = self._attr_accessor
        if not attr_name:
            raise RuntimeError(f"LayerCapture cannot resolve attribute accessor for spec '{self.full}'")

        def _hook(_module, _inp, out):
            tensor = getattr(_module, attr_name, None)
            if torch.is_tensor(tensor):
                self._record_tensor(tensor, adapter, frame_var_name)
            return out

        return _hook

    def _resolve_attr_accessor(self) -> Optional[str]:
        if self.attr is None:
            return None
        if self.method:
            module_name, base_branch_tokens = _split_module_and_branch(self.base)
            branch_tokens = _normalise_branch_tokens(base_branch_tokens, self.branch)
            prefix = _format_attr_prefix(self.method, branch_tokens)
            if not prefix:
                return None
            return f"{prefix}_{self.attr}"
        return self.attr

    def _record_tensor(self, tensor: torch.Tensor, adapter, frame_var_name: str) -> None:
        self.last = tensor
        try:
            fidx = int(getattr(adapter, frame_var_name).get())
        except Exception:
            fidx = -1
        self.by_frame[fidx] = tensor
        self._tape.append(fidx, tensor)

    def stack(self, *, detach: bool = False) -> torch.Tensor:
        if len(self._tape) == 0:
            raise RuntimeError("LayerCapture stack is empty")
        return self._tape.as_stack(detach=detach)

    def release_step_refs(self) -> None:
        self.last = None
        self.by_frame.clear()
        self._tape.clear()

    def __repr__(self) -> str:
        branch = None if self.branch is None else self.branch
        last = self.last
        if torch.is_tensor(last):
            last_str = f"Tensor(shape={tuple(int(x) for x in last.shape)})"
        else:
            last_str = type(last).__name__
        return (
            "LayerCapture("
            f"full='{self.full}', base='{self.base}', method={self.method!r}, "
            f"branch={branch}, last={last_str}, frames={len(self.by_frame)})"
        )
class MultiAnchorCapture:
    """
    Capture helper that manages multiple anchor specs simultaneously.

    Each anchor maintains an ActivationTape so recurrent models can expose
    stacked tensors (frames × tensor_shape) without detaching from the graph.
    - _Mod   : module forward output hook (base = module name)
    - _Meth  : module method hook (base = "module::method")
    - _PreIn : pre-input hook (base = module name, captures inputs before forward)
    All three store per-call tensors in tapes (list-like) so repeated calls/recurrent
    models still return lists of tensors to the caller (get_tensor_lists).
    """

    class _Mod:
        __slots__ = (
            "module",
            "base",
            "logical_base",
            "allowed",
            "hook",
            "last",
            "ig_targets",
            "orig_forward",
            "ig_alpha",
            "baselines",
            "rename",
            "expose",
            "context_snapshot",
            "display_base",
            "sae_attrs",
            "tapes",
            "frame_counter",
        )

        def __init__(
            self,
            module,
            base,
            allowed,
            rename,
            expose: bool = True,
            display_base: Optional[str] = None,
            sae_attrs: Optional[Iterable[str]] = None,
            logical_base: Optional[str] = None,
        ):
            self.module = module
            self.base = base
            self.logical_base = logical_base or display_base or base
            self.allowed = tuple(allowed) if allowed is not None else None
            self.hook = None
            self.last: Dict[str, torch.Tensor] = {}
            self.ig_targets: Dict[str, torch.Tensor] = {}
            self.orig_forward = None
            self.ig_alpha: Optional[float] = None
            self.baselines: Dict[str, torch.Tensor] = {}
            self.rename: Optional[str] = rename
            self.expose = expose
            self.context_snapshot: Optional[Dict[str, torch.Tensor]] = None
            self.display_base = display_base
            self.sae_attrs = sae_attrs
            self.tapes: Dict[str, ActivationTape] = {}
            self.frame_counter = [0]

    class _Meth:
        __slots__ = ("module", "base", "method", "allowed", "orig_method", "last", "ig_targets", "ig_alpha", "baselines", "rename", "tapes", "frame_counter")

        def __init__(self, module, base, method, allowed, rename):
            self.module = module
            self.base = base
            self.method = method
            self.allowed = tuple(allowed) if allowed is not None else None
            self.orig_method = None
            self.last: Dict[str, torch.Tensor] = {}
            self.ig_targets: Dict[str, torch.Tensor] = {}
            self.ig_alpha: Optional[float] = None
            self.baselines: Dict[str, torch.Tensor] = {}
            self.rename = rename
            self.tapes: Dict[str, ActivationTape] = {}
            self.frame_counter = [0]

    class _PreIn:
        __slots__ = ("module", "base", "allowed", "hook", "last", "ig_targets", "ig_alpha", "baselines", "rename", "call_idx", "tapes", "frame_counter", "context_snapshot")

        def __init__(self, module, base, allowed, rename):
            self.module = module
            self.base = base
            self.allowed = tuple(allowed) if allowed is not None else None
            self.hook = None
            self.last: Dict[str, torch.Tensor] = {}
            self.ig_targets: Dict[str, torch.Tensor] = {}
            self.ig_alpha: Optional[float] = None
            self.baselines: Dict[str, torch.Tensor] = {}
            self.rename = rename
            self.call_idx = -1
            self.tapes: Dict[str, ActivationTape] = {}
            self.frame_counter = [0]
            self.context_snapshot: Optional[Dict[str, torch.Tensor]] = None

    def __init__(self, *, frame_getter: Optional[Callable[[], int]] = None) -> None:
        self._mods: List[MultiAnchorCapture._Mod] = []
        self._meths: List[MultiAnchorCapture._Meth] = []
        self._preins: List[MultiAnchorCapture._PreIn] = []
        self._store_probes: List[MultiAnchorCapture._StoreProbe] = []
        self._contexts: Dict[str, Dict[str, torch.Tensor]] = {}
        self._ig_active_prefixes: set[str] = set()
        self._ig_active_attrs: Dict[str, set[str]] = {}
        self._ig_use_cached_targets: bool = False
        self._frame_getter = frame_getter
        self._frame_counter = -1
        self._current_frame_tag = -1

    def _current_frame_idx(self) -> int:
        return self._current_frame_tag

    def next_frame(self) -> None:
        """
        Advance frame counter once per forward pass.
        All hooks in the same forward will reuse the same tag.
        """
        if self._frame_getter is not None:
            try:
                self._current_frame_tag = int(self._frame_getter())
                return
            except Exception:
                pass
        self._frame_counter += 1
        self._current_frame_tag = self._frame_counter

    def reset_frame_counter(self) -> None:
        """Reset frame tags so the next forward starts counting from 0."""
        self._frame_counter = -1
        self._current_frame_tag = -1

    class _StoreProbe:
        __slots__ = ("module", "layer", "mode", "attr", "rename", "probe_prefix", "attr_accessor")

        def __init__(self, module, layer, mode, attr_name, rename, probe_prefix, attr_accessor):
            self.module = module
            self.layer = layer
            self.mode = mode
            self.attr = attr_name
            self.rename = rename
            self.probe_prefix = probe_prefix
            self.attr_accessor = attr_accessor

        def tensor(self) -> Optional[torch.Tensor]:
            if self.mode != "attr":
                return None
            if not self.attr_accessor:
                return None
            tensor = getattr(self.module, self.attr_accessor, None)
            if torch.is_tensor(tensor):
                return tensor
            return None

    def register_from_specs(self, specs: Iterable[str], *, resolve_module_fn: Callable[[str], torch.nn.Module] = resolve_module):
        """
        Register forward-output anchors from spec strings.
        - plain module name: capture forward output
        - SAE layer method: capture SAE tensors (latent/error etc.)
        - attr-only spec: attach a store-probe to read module attributes
        """
        by_mod: Dict[str, Dict[str, Any]] = {}
        by_meth: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for s in specs:
            base, method, branch, alias, attr = parse_spec(s)
            base_full = base
            base_resolve, base_branch_hint = split_layer_and_branch(base_full)
            canonical_method, canonical_attr = resolve_sae_request(method, attr)
            if canonical_method == SAE_LAYER_METHOD:
                module_base, base_branch_tokens = _split_module_and_branch(base_full)
                branch_tokens = list(base_branch_tokens)
                if branch is not None:
                    branch_tokens.append(branch)
                if not module_base:
                    raise RuntimeError(
                        f"Cannot resolve physical SAE module for spec '{base_full}::{method}'. "
                        "Ensure the base path points to a real module."
                    )
                attr_name = _format_attr_prefix(canonical_method, branch_tokens)
                physical_attr = f"{module_base}.{attr_name}" if attr_name else module_base
                mod = resolve_module_fn(physical_attr)
                logical = f"{base_full}{METHOD_SEP}{SAE_LAYER_METHOD}"
                logical_display = logical
                ent = MultiAnchorCapture._Mod(
                    mod,
                    physical_attr,
                    allowed=[physical_attr],
                    rename=alias,
                    display_base=logical_display,
                    sae_attrs=set([canonical_attr] if canonical_attr else SAE_LAYER_ATTRIBUTE_TENSORS.keys()),
                    logical_base=logical_display,
                )
                self._attach_module_capture(ent, logical_display)
                continue
            if method is None and attr:
                mod = resolve_module(base_resolve or base_full)
                probe_prefix = canonical_name(base_full, None, branch, alias, attr)
                probe = MultiAnchorCapture._StoreProbe(
                    mod,
                    base_full,
                    "attr",
                    attr,
                    alias,
                    probe_prefix=probe_prefix,
                    attr_accessor=attr,
                )
                self._store_probes.append(probe)
                continue
            branch_value = branch
            if isinstance(branch_value, tuple):
                branch_value = branch_value[0] if len(branch_value) == 1 else SEP.join(str(x) for x in branch_value)
            if method is None:
                capture_base = _physical_base_name(base_resolve or base_full)
                g = by_mod.setdefault(
                    base_full,
                    {
                        "branches": set(),
                        "alias": None,
                        "resolve": capture_base,
                        "capture_base": capture_base,
                        "logical_base": base_full,
                    },
                )
            else:
                g = by_meth.setdefault(
                    (base_full, method),
                    {"branches": set(), "alias": None, "resolve": base_resolve or base_full},
                )
            if branch_value is None:
                g["branches"] = None
            else:
                if g["branches"] is not None:
                    g["branches"].add(branch_value)
            if alias:
                g["alias"] = alias

        for base, g in by_mod.items():
            logical_base = g.get("logical_base", base)
            capture_base = g.get("capture_base", g.get("resolve", logical_base))
            resolve_target = g.get("resolve", capture_base) or capture_base
            mod = resolve_module_fn(resolve_target)
            if g["branches"] is None:
                allowed = [logical_base]
            else:
                allowed = [f"{logical_base}{SEP}{b}" for b in sorted(g["branches"], key=lambda x: str(x))]
            ent = MultiAnchorCapture._Mod(
                mod,
                capture_base,
                allowed,
                g["alias"],
                logical_base=logical_base,
            )
            if os.getenv("ATTR_MEM_DEBUG", "0") == "1":
                print(
                    f"[anchor_capture] register mod base={logical_base} allowed={allowed} capture_base={capture_base}"
                )
            self._attach_module_capture(ent, capture_base)

        for (base, method), g in by_meth.items():
            mod = resolve_module_fn(g.get("resolve", base))
            probe_prefix = f"{base}{METHOD_SEP}{method}"
            if g["branches"] is None:
                allowed = [probe_prefix]
            else:
                allowed = [f"{probe_prefix}{SEP}{b}" for b in sorted(g["branches"], key=lambda x: str(x))]
            ent = MultiAnchorCapture._Meth(mod, base, method, allowed, g["alias"])
            self._attach_method_capture(ent, probe_prefix)

    def register_preinput_from_specs(self, specs: Iterable[str], *, resolve_module_fn: Callable[[str], torch.nn.Module] = resolve_module):
        """Register pre-input hooks (capture module inputs before forward)."""
        def _parse_pre_spec(s: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
            alias = None
            for sep in [" as ", "=>", "="]:
                if sep in s:
                    core, alias = s.split(sep, 1)
                    s = core.strip()
                    alias = alias.strip()
                    break
            base, method, branch = s, None, None
            if METHOD_SEP in s:
                base, rest = s.split(METHOD_SEP, 1)
                base = base.strip()
                if SEP in rest:
                    method, branch = rest.split(SEP, 1)
                    method = method.strip()
                    branch = branch.strip()
                else:
                    method = rest.strip()
            else:
                raise ValueError(f"[pre] invalid spec (missing '{METHOD_SEP}'): {s}")
            if method != "pre":
                raise ValueError(f"[pre] method must be 'pre', got '{method}' in '{s}'")
            if branch is not None and branch.startswith("kw:"):
                branch = branch[len("kw:"):]
            return base, method, branch, alias

        for s in specs:
            base, method, branch, alias = _parse_pre_spec(s)
            base_resolve, _ = split_layer_and_branch(base)
            mod = resolve_module_fn(base_resolve or base)
            if branch is None:
                allowed = [base]
            else:
                allowed = [f"{base}{SEP}{branch}"]
            ent = MultiAnchorCapture._PreIn(mod, base, allowed, alias)
            self._attach_preinput_capture(ent, base)

    def _record_to_tape(self, tapes: Dict[str, ActivationTape], name: str, tensor: torch.Tensor, frame_idx: int):
        """Append a tensor (with frame index) to the per-anchor ActivationTape."""
        tape = tapes.get(name)
        if tape is None:
            tape = ActivationTape()
            tapes[name] = tape
        tape.append(frame_idx, tensor)

    def _attach_module_capture(self, ent: "_Mod", logical_base: str) -> None:
        def _hook(_m, _inp, out, __base=logical_base, __ent=ent):
            frame_idx = self._current_frame_idx()
            debug_all = os.getenv("ATTR_MEM_DEBUG_DUMP", "0") == "1"
            prefixes = None if debug_all else __ent.allowed
            pairs = walk_tensors(__base, out, allowed_prefixes=prefixes)
            __ent.last.clear()
            logical_prefix = __ent.display_base or __base
            for nm, t in pairs:
                logical_name = nm
                if nm.startswith(__base):
                    logical_name = logical_prefix + nm[len(__base):]
                __ent.last[logical_name] = t
                self._record_to_tape(__ent.tapes, logical_name, t, frame_idx)
                if os.getenv("ATTR_MEM_DEBUG", "0") == "1":
                    print(f"[anchor_capture] record {logical_name} shape={tuple(int(x) for x in t.shape)}")
            if debug_all and not pairs:
                print(f"[anchor_capture] no tensors under base={__base}")
            store = getattr(__ent.module, "_sae_anchor_store", None)
            ctx_snapshot = None
            if store is not None:
                try:
                    ctx_snapshot = store.snapshot()
                except Exception:
                    ctx_snapshot = None
            __ent.context_snapshot = ctx_snapshot
            if __ent.sae_attrs:
                exposed_getter = getattr(__ent.module, "exposed_tensors", None)
                if callable(exposed_getter):
                    try:
                        tensors = exposed_getter(__ent.sae_attrs)
                    except Exception:
                        tensors = {}
                    for alias_name, tensor in tensors.items():
                        if not torch.is_tensor(tensor):
                            continue
                        logical_base = __ent.display_base or __base
                        key = f"{logical_base}#{alias_name}"
                        __ent.last[key] = tensor
                        self._record_to_tape(__ent.tapes, key, tensor, frame_idx)
            elif ctx_snapshot and ".sae_" in __base:
                for extra_key in ("error_coeff",):
                    tensor = ctx_snapshot.get(extra_key)
                    if not torch.is_tensor(tensor):
                        continue
                    logical_base = __ent.display_base or __base
                    alias_name = f"{logical_base}#{extra_key}"
                    __ent.last[alias_name] = tensor
                    self._record_to_tape(__ent.tapes, alias_name, tensor, frame_idx)

        ent.hook = ent.module.register_forward_hook(_hook)
        self._mods.append(ent)

    def _attach_method_capture(self, ent: "_Meth", probe_prefix: str) -> None:
        bound = getattr(ent.module, ent.method)
        _orig = bound.__func__ if hasattr(bound, "__func__") else bound
        _probe = probe_prefix
        _allowed = ent.allowed

        def _wrapped(self_obj, *args, __orig=_orig, __probe=_probe, __allowed=_allowed, __ent=ent, **kwargs):
            frame_idx = self._current_frame_idx()
            out = __orig(self_obj, *args, **kwargs)
            pairs = walk_tensors(__probe, out, allowed_prefixes=__allowed)
            __ent.last.clear()
            for nm, t in pairs:
                __ent.last[nm] = t
                self._record_to_tape(__ent.tapes, nm, t, frame_idx)
            return out

        setattr(ent.module, ent.method, types.MethodType(_wrapped, ent.module))
        ent.orig_method = bound
        self._meths.append(ent)

    def _attach_preinput_capture(self, ent: "_PreIn", base: str) -> None:
        """Attach a pre-input hook to capture module inputs before forward."""
        def _pre_hook(m, args, kwargs, __ent=ent, __mac=self):
            __ent.call_idx += 1
            frame_idx = self._current_frame_idx()
            call_tag = f"call_{__ent.call_idx:03d}"
            __ent.last.clear()

            def _ensure_requires_grad(leaf: torch.Tensor) -> torch.Tensor:
                if not torch.is_tensor(leaf):
                    return leaf
                if leaf.requires_grad:
                    return leaf
                base_leaf = leaf.detach() if leaf.grad_fn is not None else leaf
                return base_leaf + torch.zeros_like(base_leaf, requires_grad=True)

            apply_alpha = __ent.ig_alpha is not None and __mac._is_active_for_ig(__ent.base)
            alpha = float(__ent.ig_alpha) if apply_alpha else None
            printed = [False]

            def _mk(name, leaf):
                if not torch.is_tensor(leaf):
                    return leaf
                if not apply_alpha:
                    return _ensure_requires_grad(leaf)

                # Prefer a recorded baseline if present (supports record/current baselines)
                baseline = __ent.baselines.get(name)
                if baseline is None and name.startswith(__ent.base):
                    # pre-hooks add a call tag to recorded names; try that key as well
                    suffix = name[len(__ent.base):]
                    call_name = f"{__ent.base}{SEP}{call_tag}{suffix}"
                    baseline = __ent.baselines.get(call_name)
                base = baseline if torch.is_tensor(baseline) else None

                if torch.is_tensor(baseline):
                    if base.device != leaf.device or base.dtype != leaf.dtype:
                        base = base.to(device=leaf.device, dtype=leaf.dtype)
                    v = base + (leaf - base) * alpha
                else:
                    v = leaf * alpha
                if apply_alpha and not printed[0] and os.getenv("ATTR_IG_DEBUG", "0") == "1":
                    leaf0 = leaf.detach().flatten()[0].item() if leaf.numel() > 0 else 0.0
                    base0 = base.detach().flatten()[0].item() if torch.is_tensor(baseline) and baseline.numel() > 0 else None
                    _ig_debug(f"[ig-debug] apply_alpha(pre) base={__ent.base} name={name} alpha={alpha:.4f} leaf0={leaf0:.4f} base0={base0}")
                    printed[0] = True

                if leaf.requires_grad:
                    return v
                base_v = v.detach() if leaf.grad_fn is not None else v
                return base_v + torch.zeros_like(base_v, requires_grad=True)

            new_args = list(args)
            if isinstance(new_args, list):
                new_args = _reconstruct_with(__ent.base, new_args, __ent.allowed, _mk)
            new_kwargs = _reconstruct_with(__ent.base, kwargs, __ent.allowed, _mk)
            # Record the transformed (alpha-applied) tensors so anchors match the
            # actual forward inputs seen by the module.
            pairs = walk_tensors(__ent.base, (new_args, new_kwargs), allowed_prefixes=__ent.allowed)
            for nm, t in pairs:
                logical = nm
                if nm.startswith(__ent.base):
                    logical = __ent.base + SEP + call_tag + nm[len(__ent.base):]
                __ent.last[logical] = t
                self._record_to_tape(__ent.tapes, logical, t, frame_idx)
            return tuple(new_args), new_kwargs

        ent.hook = ent.module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        self._preins.append(ent)

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        self._contexts.clear()
        for ent in self._mods:
            if not ent.expose:
                continue
            for k, v in ent.last.items():
                rename_base = ent.logical_base or ent.base
                if ent.display_base and k.startswith(ent.display_base):
                    rename_base = ent.display_base
                user_k = self._rename_key(k, ent.rename, rename_base)
                out[user_k] = v
                if ent.context_snapshot:
                    ctx_converted: Dict[str, Any] = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        for ent in self._meths:
            probe_prefix = f"{ent.base}{METHOD_SEP}{ent.method}"
            for k, v in ent.last.items():
                user_k = self._rename_key(k, ent.rename, probe_prefix)
                out[user_k] = v
        for ent in self._preins:
            for k, v in ent.last.items():
                user_k = self._rename_key(k, ent.rename, ent.base)
                out[user_k] = v
        for probe in self._store_probes:
            tensor = probe.tensor()
            if tensor is None:
                continue
            user_k = self._rename_key(probe.probe_prefix, probe.rename, probe.probe_prefix)
            out[user_k] = tensor
        return out

    def get_stacked_tensors(self, *, detach: bool = False, squeeze_single: bool = False) -> Dict[str, torch.Tensor]:
        stacks: Dict[str, torch.Tensor] = {}
        self._contexts.clear()
        for ent in self._mods:
            for name, tape in ent.tapes.items():
                if ent.display_base and name.startswith(ent.display_base):
                    rename_base = ent.display_base
                else:
                    rename_base = ent.logical_base or ent.base
                user_k = self._rename_key(name, ent.rename, rename_base)
                try:
                    stacks[user_k] = tape.as_stack(detach=detach, squeeze_single=squeeze_single)
                except RuntimeError:
                    # empty tape: skip
                    continue
                if ent.context_snapshot:
                    ctx_converted: Dict[str, Any] = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        for ent in self._meths:
            prefix = f"{ent.base}{METHOD_SEP}{ent.method}"
            for name, tape in ent.tapes.items():
                user_k = self._rename_key(name, ent.rename, prefix)
                try:
                    stacks[user_k] = tape.as_stack(detach=detach, squeeze_single=squeeze_single)
                except RuntimeError:
                    continue
                if ent.context_snapshot:
                    ctx_converted = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        for ent in self._preins:
            for name, tape in ent.tapes.items():
                user_k = self._rename_key(name, ent.rename, ent.base)
                try:
                    stacks[user_k] = tape.as_stack(detach=detach, squeeze_single=squeeze_single)
                except RuntimeError:
                    continue
                if ent.context_snapshot:
                    ctx_converted = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        return stacks

    def get_tensor_lists(self, *, detach: bool = False) -> Dict[str, list[torch.Tensor]]:
        """
        Return raw per-step tensors without stacking, preserving computation graph.
        """
        lists: Dict[str, list[torch.Tensor]] = {}
        self._contexts.clear()
        for ent in self._mods:
            for name, tape in ent.tapes.items():
                if ent.display_base and name.startswith(ent.display_base):
                    rename_base = ent.display_base
                else:
                    rename_base = ent.logical_base or ent.base
                user_k = self._rename_key(name, ent.rename, rename_base)
                tensors: list[torch.Tensor] = []
                for rec in tape:
                    tensors.append(rec.tensor.detach() if detach else rec.tensor)
                if tensors:
                    lists[user_k] = tensors
                if ent.context_snapshot:
                    ctx_converted: Dict[str, Any] = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        for ent in self._meths:
            prefix = f"{ent.base}{METHOD_SEP}{ent.method}"
            for name, tape in ent.tapes.items():
                user_k = self._rename_key(name, ent.rename, prefix)
                tensors = [rec.tensor.detach() if detach else rec.tensor for rec in tape]
                if tensors:
                    lists[user_k] = tensors
                if ent.context_snapshot:
                    ctx_converted = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        for ent in self._preins:
            for name, tape in ent.tapes.items():
                user_k = self._rename_key(name, ent.rename, ent.base)
                tensors = [rec.tensor.detach() if detach else rec.tensor for rec in tape]
                if tensors:
                    lists[user_k] = tensors
                if getattr(ent, "context_snapshot", None):
                    ctx_converted = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        return lists

    def get_tensor_records(self, *, detach: bool = False) -> Dict[str, list[ActivationRecord]]:
        """
        Return raw per-step ActivationRecord entries (frame_idx + tensor), preserving computation graph.
        """
        records: Dict[str, list[ActivationRecord]] = {}
        self._contexts.clear()
        for ent in self._mods:
            for name, tape in ent.tapes.items():
                rename_base = ent.display_base if ent.display_base and name.startswith(ent.display_base) else (ent.logical_base or ent.base)
                user_k = self._rename_key(name, ent.rename, rename_base)
                recs: list[ActivationRecord] = []
                for rec in tape:
                    tensor = rec.tensor.detach() if detach else rec.tensor
                    recs.append(ActivationRecord(frame_idx=rec.frame_idx, tensor=tensor))
                if recs:
                    records[user_k] = recs
                if ent.context_snapshot:
                    ctx_converted: Dict[str, Any] = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        for ent in self._meths:
            prefix = f"{ent.base}{METHOD_SEP}{ent.method}"
            for name, tape in ent.tapes.items():
                user_k = self._rename_key(name, ent.rename, prefix)
                recs = [ActivationRecord(frame_idx=rec.frame_idx, tensor=rec.tensor.detach() if detach else rec.tensor) for rec in tape]
                if recs:
                    records[user_k] = recs
                if ent.context_snapshot:
                    ctx_converted = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        for ent in self._preins:
            for name, tape in ent.tapes.items():
                user_k = self._rename_key(name, ent.rename, ent.base)
                recs = [ActivationRecord(frame_idx=rec.frame_idx, tensor=rec.tensor.detach() if detach else rec.tensor) for rec in tape]
                if recs:
                    records[user_k] = recs
                if getattr(ent, "context_snapshot", None):
                    ctx_converted = {}
                    for ctx_key, ctx_tensor in ent.context_snapshot.items():
                        if torch.is_tensor(ctx_tensor) or isinstance(ctx_tensor, dict):
                            ctx_converted[ctx_key] = ctx_tensor
                    if ctx_converted:
                        self._contexts[user_k] = ctx_converted
        return records

    def get_contexts(self) -> Dict[str, Dict[str, torch.Tensor]]:
        contexts: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, ctx in self._contexts.items():
            contexts[name] = {key: tensor for key, tensor in ctx.items()}
        return contexts

    def record_ig_targets(self) -> None:
        def _clone_map(src: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            cloned: Dict[str, torch.Tensor] = {}
            for k, v in src.items():
                if torch.is_tensor(v):
                    cloned[k] = v.detach().clone()
            return cloned

        for ent in self._mods:
            ent.ig_targets = _clone_map(ent.last)
        for ent in self._meths:
            ent.ig_targets = _clone_map(ent.last)
        for ent in self._preins:
            ent.ig_targets = _clone_map(ent.last)

    def clear_ig_targets(self) -> None:
        for ent in self._mods:
            ent.ig_targets.clear()
        for ent in self._meths:
            ent.ig_targets.clear()
        for ent in self._preins:
            ent.ig_targets.clear()

    def ig_use_cached_targets(self, enabled: bool) -> None:
        self._ig_use_cached_targets = bool(enabled)

    def record_baselines(self) -> None:
        def _clone_map(src: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            cloned: Dict[str, torch.Tensor] = {}
            for k, v in src.items():
                if torch.is_tensor(v):
                    cloned[k] = v.detach().clone()
            return cloned

        for ent in self._mods:
            ent.baselines = _clone_map(ent.last)
        for ent in self._meths:
            ent.baselines = _clone_map(ent.last)
        for ent in self._preins:
            ent.baselines = _clone_map(ent.last)

    def ig_prepare_target_baselines(
        self,
        unit_index: int,
        base_map: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Build IG baselines where only the target unit moves (others stay at target value).
        """
        base_map = base_map or {}
        self.clear_baselines()

        def _unit_mask(tensor: torch.Tensor) -> Optional[torch.Tensor]:
            if not torch.is_tensor(tensor):
                return None
            if tensor.dim() == 0:
                return None
            if tensor.shape[-1] <= unit_index:
                return None
            mask = torch.zeros_like(tensor)
            idx = (slice(None),) * (mask.dim() - 1) + (unit_index,)
            mask[idx] = 1.0
            return mask

        def _coerce_base(user_key: str, like: torch.Tensor) -> torch.Tensor:
            base = base_map.get(user_key)
            if not torch.is_tensor(base):
                return torch.zeros_like(like)
            base_t = base.to(device=like.device, dtype=like.dtype)
            if base_t.shape == like.shape:
                return base_t
            if base_t.dim() == like.dim() - 1 and base_t.shape == like.shape[1:]:
                return base_t.unsqueeze(0).expand_as(like)
            if base_t.numel() == like.numel():
                try:
                    return base_t.reshape(like.shape)
                except Exception:
                    pass
            try:
                return base_t.expand_as(like)
            except Exception:
                return torch.zeros_like(like)

        def _apply_baseline(ent, items, rename_base: str) -> None:
            for name, tensor in items:
                if not torch.is_tensor(tensor):
                    continue
                user_k = self._rename_key(name, getattr(ent, "rename", None), rename_base)
                has_base = user_k in base_map
                mask = _unit_mask(tensor)
                if mask is None:
                    if has_base:
                        base_only = _coerce_base(user_k, tensor)
                        ent.baselines[name] = base_only.detach().clone()
                    continue
                base = _coerce_base(user_k, tensor)
                mask = mask.to(device=tensor.device, dtype=tensor.dtype)
                source = tensor.detach()
                baseline = (source * (1.0 - mask)) + (base * mask)
                ent.baselines[name] = baseline.detach().clone()

        for ent in self._mods:
            logical_prefix = ent.display_base or ent.logical_base or ent.base
            _apply_baseline(ent, ent.last.items(), logical_prefix)
        for ent in self._meths:
            probe_prefix = f"{ent.base}{METHOD_SEP}{ent.method}"
            _apply_baseline(ent, ent.last.items(), probe_prefix)
        for ent in self._preins:
            _apply_baseline(ent, ent.last.items(), ent.base)

    def get_baseline_tensors(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for ent in self._mods:
            if not ent.expose:
                continue
            for k, v in ent.baselines.items():
                if ent.display_base and k.startswith(ent.display_base):
                    rename_base = ent.display_base
                else:
                    rename_base = ent.logical_base or ent.base
                user_k = self._rename_key(k, ent.rename, rename_base)
                out[user_k] = v
        for ent in self._meths:
            probe_prefix = f"{ent.base}{METHOD_SEP}{ent.method}"
            for k, v in ent.baselines.items():
                user_k = self._rename_key(k, ent.rename, probe_prefix)
                out[user_k] = v
        for ent in self._preins:
            for k, v in ent.baselines.items():
                user_k = self._rename_key(k, ent.rename, ent.base)
                out[user_k] = v
        return out

    def get_overrideable_modules(self) -> Dict[str, Any]:
        """
        Map logical anchor names to modules that support anchor overrides.
        Keys include the logical base and (for SAE attrs) logical_base#attr.
        """
        mods: Dict[str, Any] = {}
        for ent in self._mods:
            mod = ent.module
            if not hasattr(mod, "set_anchor_override") or not hasattr(mod, "clear_anchor_overrides"):
                continue
            base = ent.display_base or ent.logical_base or ent.base
            if base:
                mods[str(base)] = mod
            if ent.sae_attrs:
                for attr in ent.sae_attrs:
                    mods[f"{base}#{attr}"] = mod
        return mods

    def clear_baselines(self) -> None:
        for ent in self._mods:
            ent.baselines.clear()
        for ent in self._meths:
            ent.baselines.clear()
        for ent in self._preins:
            ent.baselines.clear()

    def remove_hooks(self):
        for ent in self._mods:
            try:
                if ent.hook is not None:
                    ent.hook.remove()
            except Exception:
                pass
            ent.hook = None
        for ent in self._preins:
            try:
                if ent.hook is not None:
                    ent.hook.remove()
            except Exception:
                pass
            ent.hook = None

    def clear(self):
        self.remove_hooks()
        for ent in self._mods:
            ent.last.clear()
            ent.ig_targets.clear()
            ent.baselines.clear()
            ent.context_snapshot = None
            for tape in ent.tapes.values():
                tape.clear()
        for ent in self._meths:
            ent.last.clear()
            ent.ig_targets.clear()
            ent.baselines.clear()
            for tape in ent.tapes.values():
                tape.clear()
        for ent in self._preins:
            ent.last.clear()
            ent.ig_targets.clear()
            ent.baselines.clear()
            for tape in ent.tapes.values():
                tape.clear()
        self._contexts.clear()
        self._ig_use_cached_targets = False

    def clear_tapes(self) -> None:
        for ent in self._mods:
            for tape in ent.tapes.values():
                tape.clear()
        for ent in self._meths:
            for tape in ent.tapes.values():
                tape.clear()
        for ent in self._preins:
            for tape in ent.tapes.values():
                tape.clear()

    def release_step_refs(self):
        for ent in self._mods:
            ent.last.clear()
            ent.context_snapshot = None
            for tape in ent.tapes.values():
                tape.clear()
            store = getattr(ent.module, "_sae_anchor_store", None)
            if store is not None:
                try:
                    store.clear()
                except Exception:
                    pass
            attachments = getattr(ent.module, "_sae_attachments", None)
            if attachments:
                for att in attachments.values():
                    mod = getattr(att, "module", None)
                    if mod is not None:
                        try:
                            mod.clear_context()
                        except Exception:
                            pass
        for ent in self._meths:
            ent.last.clear()
            for tape in ent.tapes.values():
                tape.clear()
        for ent in self._preins:
            ent.last.clear()
            ent.call_idx = -1
            for tape in ent.tapes.values():
                tape.clear()
        self._contexts.clear()

    def ig_set_active(self, names: Optional[set[str]]):
        cleaned: set[str] = set()
        attr_map: Dict[str, set[str]] = {}
        for name in names or []:
            base = str(name or "").strip()
            if not base:
                continue
            parsed = None
            try:
                parsed = parse_spec(base)
            except Exception:
                parsed = None
            if parsed is not None:
                try:
                    canonical_method, canonical_attr = resolve_sae_request(parsed.method, parsed.attr)
                except Exception:
                    canonical_method, canonical_attr = None, None
                if canonical_method == SAE_LAYER_METHOD and canonical_attr:
                    logical = canonical_name(parsed.base, canonical_method, None, None, None)
                    attr_set = attr_map.setdefault(logical, set())
                    attr_set.add(canonical_attr)
                    base = logical
            if "#" in base:
                base = base.split("#", 1)[0]
            cleaned.add(base)
        self._ig_active_prefixes = cleaned
        self._ig_active_attrs = attr_map

    def _is_active_for_ig(self, probe_prefix: str) -> bool:
        if not self._ig_active_prefixes:
            return True
        prefix = probe_prefix.split("#", 1)[0] if "#" in probe_prefix else probe_prefix
        if prefix in self._ig_active_prefixes:
            _ig_debug(f"[ig-debug] active (exact) {probe_prefix}")
            return True
        # 허용 목록이 더 짧거나(모듈 전체) 더 길 때(브랜치/메서드 포함)도 매칭하도록 prefix 비교
        for name in self._ig_active_prefixes:
            if prefix.startswith(name) or name.startswith(prefix):
                _ig_debug(f"[ig-debug] active (prefix) probe={probe_prefix} rule={name}")
                return True
        _ig_debug(f"[ig-debug] inactive probe={probe_prefix}")
        return False

    def enable_ig_override(self):
        for ent in self._mods:
            if ent.orig_forward is not None:
                continue
            bound = ent.module.forward
            _orig = bound.__func__ if hasattr(bound, "__func__") else bound
            _base = ent.base
            _logical_name = ent.logical_base or _base
            _allowed = ent.allowed

            def _wrapped(
                self_obj,
                *args,
                __orig=_orig,
                __ent=ent,
                __base=_base,
                __logical=_logical_name,
                __allowed=_allowed,
                __mac=self,
                **kwargs,
            ):
                out = __orig(self_obj, *args, **kwargs)
                apply_alpha = __ent.ig_alpha is not None and __mac._is_active_for_ig(__logical)
                if not apply_alpha:
                    return out
                alpha = float(__ent.ig_alpha)
                printed = [False]

                def _mk(name, leaf, __ent=__ent):
                    cached = __ent.ig_targets.get(name) if __mac._ig_use_cached_targets else None
                    leaf_val = cached if torch.is_tensor(cached) else leaf
                    baseline = __ent.baselines.get(name)
                    base = baseline if torch.is_tensor(baseline) else None
                    if torch.is_tensor(baseline):
                        if base.device != leaf.device or base.dtype != leaf.dtype:
                            base = base.to(device=leaf.device, dtype=leaf.dtype)
                    if torch.is_tensor(leaf_val):
                        if leaf_val.device != leaf.device or leaf_val.dtype != leaf.dtype:
                            leaf_val = leaf_val.to(device=leaf.device, dtype=leaf.dtype)
                    if base is not None:
                        v = base + (leaf_val - base) * alpha
                    else:
                        v = leaf_val * alpha
                    if apply_alpha and not printed[0] and os.getenv("ATTR_IG_DEBUG", "0") == "1":
                        leaf0 = leaf_val.detach().flatten()[0].item() if torch.is_tensor(leaf_val) and leaf_val.numel() > 0 else 0.0
                        base0 = base.detach().flatten()[0].item() if torch.is_tensor(base) and base.numel() > 0 else None
                        _ig_debug(f"[ig-debug] apply_alpha(mod) base={__logical} name={name} alpha={alpha:.4f} leaf0={leaf0:.4f} base0={base0}")
                        printed[0] = True
                    if v.requires_grad:
                        return v
                    base_v = v.detach() if v.grad_fn is not None else v
                    return base_v + torch.zeros_like(base_v, requires_grad=True)

                return _reconstruct_with(__base, out, __allowed, _mk)

            ent.module.forward = types.MethodType(_wrapped, ent.module)
            ent.orig_forward = bound
            ent.ig_alpha = 0.0
        for ent in self._meths:
            ent.ig_alpha = 0.0
        for ent in self._preins:
            ent.ig_alpha = 0.0

    def set_alpha(self, a: float):
        for ent in self._mods:
            ent.ig_alpha = float(a)
            self._sync_sae_ig_state(ent, float(a))
        for ent in self._meths:
            ent.ig_alpha = float(a)
        for ent in self._preins:
            ent.ig_alpha = float(a)
        _ig_debug(f"[ig-debug] set_alpha={a:.4f} active={len(self._ig_active_prefixes)}")

    def _sync_sae_ig_state(self, ent: "_Mod", alpha: float) -> None:
        module = getattr(ent, "module", None)
        setter = getattr(module, "set_ig_attr_alpha", None)
        clearer = getattr(module, "clear_ig_attr_alpha", None)
        if setter is None and clearer is None:
            return
        logical = ent.logical_base or ent.base
        if not self._is_active_for_ig(logical):
            if callable(clearer):
                try:
                    clearer()
                except Exception:
                    pass
            return
        attrs = self._ig_active_attrs.get(logical)
        if not attrs:
            if callable(clearer):
                try:
                    clearer()
                except Exception:
                    pass
            return
        store_keys: set[str] = set()
        baselines: Dict[str, torch.Tensor] = {}
        for attr in attrs:
            store_key = SAE_LAYER_ATTRIBUTE_TENSORS.get(attr)
            if not store_key:
                continue
            store_keys.add(store_key)
            baseline_key = f"{logical}#{attr}"
            baseline = ent.baselines.get(baseline_key)
            if baseline is not None:
                baselines[store_key] = baseline
        if not store_keys:
            if callable(clearer):
                try:
                    clearer()
                except Exception:
                    pass
            return
        try:
            setter(alpha=float(alpha), active_attrs=store_keys, baselines=baselines, logical_base=logical)
        except TypeError:
            try:
                setter(alpha=float(alpha), active_attrs=store_keys, baselines=baselines)
            except Exception:
                pass
        except Exception:
            pass

    def disable_ig_override(self):
        for ent in self._mods:
            if ent.orig_forward is not None:
                ent.module.forward = ent.orig_forward
            ent.orig_forward = None
            ent.baselines.clear()
            ent.ig_alpha = None
        for ent in self._meths:
            if ent.orig_method is not None:
                setattr(ent.module, ent.method, ent.orig_method)
            ent.orig_method = None
            ent.baselines.clear()
            ent.ig_alpha = None
        for ent in self._preins:
            ent.baselines.clear()
            ent.ig_alpha = None
        for ent in self._mods:
            clearer = getattr(ent.module, "clear_ig_attr_alpha", None)
            if callable(clearer):
                try:
                    clearer()
                except Exception:
                    pass

    def _rename_key(self, physical: str, rename_prefix: Optional[str], base_or_probe: str) -> str:
        if rename_prefix and physical.startswith(base_or_probe):
            new = rename_prefix + physical[len(base_or_probe):]
            dup = f"{rename_prefix}{SEP}{rename_prefix}"
            if new.startswith(dup):
                new = rename_prefix + new[len(dup):]
            parts = new.split(SEP)
            if len(parts) >= 3 and parts[0] == rename_prefix and parts[1].startswith("call_") and parts[2] == rename_prefix:
                parts.pop(2)
                new = SEP.join(parts)
            return new
        return physical


def _reconstruct_with(prefix: str, obj: Any, allowed: Optional[Iterable[str]], repl) -> Any:
    if torch.is_tensor(obj):
        name = prefix
        if allowed is None or any(a == name or a.startswith(name + SEP) or name.startswith(a + SEP) for a in allowed):
            return repl(name, obj)
        return obj
    if isinstance(obj, (list, tuple)):
        xs = []
        for i, it in enumerate(obj):
            child = f"{prefix}{SEP}{i}"
            x = _reconstruct_with(child, it, allowed, repl)
            xs.append(x)
        return tuple(xs) if isinstance(obj, tuple) else xs
    if isinstance(obj, dict):
        out = {}
        for k, it in obj.items():
            child = f"{prefix}{SEP}{k}"
            out[k] = _reconstruct_with(child, it, allowed, repl)
        return out
    if allowed is not None:
        prefix_sep = prefix + SEP
        candidates = [
            cand
            for cand in allowed
            if cand != prefix and cand.startswith(prefix_sep)
        ]
        modified = False
        obj_copy = obj
        for candidate in candidates:
            remainder = candidate[len(prefix_sep):]
            if not remainder:
                continue
            token = remainder.split(SEP, 1)[0]
            if token.isdigit():
                continue
            if not hasattr(obj, token):
                continue
            attr_val = getattr(obj, token)
            if callable(attr_val):
                continue
            child = f"{prefix}{SEP}{token}"
            new_val = _reconstruct_with(child, attr_val, allowed, repl)
            if new_val is attr_val:
                continue
            if not modified:
                try:
                    obj_copy = copy.copy(obj)
                except Exception:
                    obj_copy = obj
                modified = True
            try:
                setattr(obj_copy, token, new_val)
            except Exception:
                pass
        if modified:
            return obj_copy
    return obj


__all__ = ["LayerCapture", "MultiAnchorCapture"]
