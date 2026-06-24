from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Callable

import torch

from src.core.indexing.registry_utils import sanitize_layer_name
from src.core.runtime.attribution_runtime import (
    AnchorConfig,
    AttributionRuntime,
    BackwardConfig,
    ForwardConfig,
    RuntimeTarget,
)
from src.core.runtime.wrappers import install_sae_wrappers_for_specs
from src.core.sae.activation_stores.hook_helper import reshape_flat_sae_tensor


def _ensure_list(value: Optional[Iterable[str]]) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


class CircuitRuntime:
    """
    Edge-level attribution runtime used by FeatureTreeBuilder.

    This class is model-agnostic; packs are responsible for providing a model,
    adapter, forward_fn, SAE resolver, and wrapper specs.
    """

    def __init__(
        self,
        *,
        runtime_cfg: Dict[str, Any],
        model: torch.nn.Module,
        adapter: Any,
        forward_fn: Callable[[], None],
        sae_resolver: Callable[[str], torch.nn.Module],
        wrap_specs: Sequence[str] = (),
        device: Optional[torch.device | str] = None,
        allow_missing_anchor_grad: bool = False,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.forward_fn = forward_fn
        self._sae_resolver = sae_resolver
        self.device = torch.device(device or getattr(adapter, "device", "cpu"))
        self.cfg = runtime_cfg

        self._extra_wrap_handles = install_sae_wrappers_for_specs(
            model=self.model,
            specs=wrap_specs,
            sae_resolver=self._sae_resolver,
            frame_getter=getattr(adapter, "current_frame_idx", None),
        )

        target_cfg = runtime_cfg.get("target") or {}
        self.target = RuntimeTarget(
            layer=target_cfg["layer"],
            unit=int(target_cfg["unit"]),
            override_mode=target_cfg.get("override_mode", "all_tokens"),
            objective_aggregation=target_cfg.get("objective_aggregation", "sum"),
        )
        self.backward_cfg = BackwardConfig(
            enabled=bool(runtime_cfg.get("backward", {}).get("enabled", True)),
            method=runtime_cfg.get("backward", {}).get("method", "ig"),
            ig_steps=int(runtime_cfg.get("backward", {}).get("ig_steps", 32)),
            baseline=runtime_cfg.get("backward", {}).get("baseline", "zeros"),
        )
        self.forward_cfg = ForwardConfig(
            enabled=bool(runtime_cfg.get("forward", {}).get("enabled", False)),
            method=runtime_cfg.get("forward", {}).get("method", "ig"),
            ig_steps=int(runtime_cfg.get("forward", {}).get("ig_steps", 32)),
            baseline=runtime_cfg.get("forward", {}).get("baseline", "zeros"),
        )

        self.runtime = AttributionRuntime(
            model=self.model,
            adapter=self.adapter,
            forward_fn=self.forward_fn,
            sae_module=self._sae_resolver(self.target.layer),
            target=self.target,
            allow_missing_anchor_grad=allow_missing_anchor_grad,
        )

    # ------------------------------------------------------------------ helpers
    def _build_weight_multiplier(
        self,
        weight: Optional[torch.Tensor],
        target_feature_indices: Optional[torch.Tensor],
    ):
        if weight is None:
            return None
        if not torch.is_tensor(weight):
            weight = torch.tensor(weight, dtype=torch.float32)
        weight = weight.to(device=self.device, dtype=torch.float32)
        idx = target_feature_indices.flatten().long() if target_feature_indices is not None else None
        if idx is not None and idx.numel() == 0:
            return None

        def _multiplier(latent: torch.Tensor) -> torch.Tensor:
            def _broadcast_to_shape(t: torch.Tensor, shape: torch.Size, *, last_dim: int) -> Optional[torch.Tensor]:
                view = t
                if view.dim() == 0:
                    view = view.view(1)
                if view.dim() > len(shape) and view.dim() > 1 and view.shape[-1] in {1, last_dim}:
                    leading = int(torch.tensor(view.shape[:-1]).prod().item())
                    view = view.reshape(leading, view.shape[-1])
                if view.dim() > len(shape):
                    return None
                while view.dim() < len(shape):
                    view = view.unsqueeze(0)
                try:
                    return torch.broadcast_to(view, shape)
                except Exception:
                    return None

            w = weight.to(device=latent.device, dtype=latent.dtype)
            feat = latent.shape[-1]
            mask = torch.zeros_like(latent)
            if idx is None:
                w_view = w
                if w_view.dim() > 0 and w_view.shape[-1] not in {1, feat}:
                    flat = w_view.flatten()
                    if flat.numel() < feat:
                        pad = torch.zeros(feat - flat.numel(), device=flat.device, dtype=flat.dtype)
                        flat = torch.cat([flat, pad], dim=0)
                    else:
                        flat = flat[:feat]
                    w_view = flat
                broadcasted = _broadcast_to_shape(w_view, latent.shape, last_dim=feat)
                if broadcasted is None:
                    flat = w_view.flatten()
                    if flat.numel() < feat:
                        pad = torch.zeros(feat - flat.numel(), device=flat.device, dtype=flat.dtype)
                        flat = torch.cat([flat, pad], dim=0)
                    else:
                        flat = flat[:feat]
                    while flat.dim() < latent.dim():
                        flat = flat.unsqueeze(0)
                    broadcasted = flat.expand_as(latent)
                return broadcasted

            index = idx.to(device=latent.device)
            valid = index < latent.shape[-1]
            if not torch.all(valid):
                index = index[valid]
            if index.numel() == 0:
                return mask

            w_view = w
            if w_view.dim() > 0 and w_view.shape[-1] not in {1, feat} and w_view.numel() >= feat:
                w_view = w_view.flatten()[:feat]

            feature_broadcast = _broadcast_to_shape(w_view, latent.shape, last_dim=feat)
            if feature_broadcast is not None and feature_broadcast.shape[-1] == feat:
                weight_vals = feature_broadcast.index_select(-1, index)
            else:
                per_index_shape = (*latent.shape[:-1], index.numel())
                per_index = _broadcast_to_shape(w, per_index_shape, last_dim=index.numel())
                if per_index is None:
                    flat = w.flatten()
                    if flat.numel() == 1 and index.numel() > 1:
                        flat = flat.expand(index.numel())
                    elif flat.numel() < index.numel():
                        pad = torch.zeros(index.numel() - flat.numel(), device=flat.device, dtype=flat.dtype)
                        flat = torch.cat([flat, pad], dim=0)
                    else:
                        flat = flat[: index.numel()]
                    weight_vals = flat
                    while weight_vals.dim() < latent.dim():
                        weight_vals = weight_vals.unsqueeze(0)
                    weight_vals = weight_vals.expand(*latent.shape[:-1], -1)
                else:
                    weight_vals = per_index

            while weight_vals.dim() < latent.dim():
                weight_vals = weight_vals.unsqueeze(0)
            weight_vals = weight_vals.expand(*latent.shape[:-1], -1)
            mask.index_copy_(-1, index, weight_vals)
            return mask

        return _multiplier

    def _extract_attr_map(self, output: Optional[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not output:
            return {}
        if isinstance(output, dict) and "attr" in output and isinstance(output["attr"], dict):
            return output["attr"]
        return output if isinstance(output, dict) else {}

    def _select_attr_tensor(
        self,
        attr_map: Dict[str, torch.Tensor],
        preferred: List[str],
    ) -> tuple[Optional[torch.Tensor], Optional[str]]:
        for name in preferred:
            if not name:
                continue
            if name in attr_map:
                return attr_map[name], name
            alt = sanitize_layer_name(name)
            if alt in attr_map:
                return attr_map[alt], alt
        for key, tensor in attr_map.items():
            if torch.is_tensor(tensor):
                return tensor, key
        return None, None

    def _restore_with_meta(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        contexts = self.runtime.anchor_capture.get_contexts()
        ctx = contexts.get(key, {})
        reshape_meta = ctx.get("reshape_meta") if isinstance(ctx, dict) else None
        if reshape_meta is not None:
            try:
                tensor = reshape_flat_sae_tensor(tensor, reshape_meta)
            except Exception:
                pass
        return tensor

    def _gather_attr_tensors(
        self,
        attr_map: Dict[str, torch.Tensor],
        modules: List[str],
    ) -> Dict[str, torch.Tensor]:
        wanted_norm = {sanitize_layer_name(m) for m in modules if m}
        out: Dict[str, torch.Tensor] = {}
        for key, tensor in attr_map.items():
            if not torch.is_tensor(tensor):
                continue
            norm = sanitize_layer_name(key)
            if norm not in wanted_norm and key not in modules:
                continue
            out[key] = tensor
        return out

    def _normalise_attr_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.view(1, 1, -1)
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        if tensor.dim() >= 3:
            return tensor
        raise RuntimeError(f"Unsupported attribution tensor rank {tensor.dim()}")

    # ------------------------------------------------------------------ public API
    def attribute_edge_multi(
        self,
        edge_spec,
        layer_module_name: str,
        target_feature_indices: Optional[torch.Tensor] = None,
        weight_vector: Optional[torch.Tensor] = None,
        parent_module_name: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if target_feature_indices is not None and target_feature_indices.numel() > 0:
            feat_idx = int(target_feature_indices.flatten()[0].item())
            sae_module = self._sae_resolver(layer_module_name)
            self.runtime.set_target(layer_module_name, feat_idx, sae_module=sae_module)
        else:
            self.runtime.set_target(self.target.layer, self.target.unit, sae_module=self._sae_resolver(self.target.layer))

        try:
            self.runtime.anchor_capture.clear()
        except Exception:
            pass
        anchor_cfg = AnchorConfig(
            capture=list({*(edge_spec.anchor_modules or []), *( [parent_module_name] if parent_module_name else []), layer_module_name}),
            ig_active=_ensure_list(getattr(edge_spec, "anchor_ig_active", None)),
            stop_grad=_ensure_list(getattr(edge_spec, "stop_grad", None)),
        )
        self.runtime.configure_anchors(anchor_cfg)

        weight_multiplier = None
        if weight_vector is not None:
            weight_multiplier = self._build_weight_multiplier(weight_vector, target_feature_indices)
        elif sanitize_layer_name(layer_module_name) == sanitize_layer_name(self.target.layer):
            target_for_mask = target_feature_indices
            mask_weight = torch.ones_like(target_for_mask, dtype=torch.float32) if target_for_mask is not None else None
            weight_multiplier = self._build_weight_multiplier(mask_weight, target_for_mask)
        self.runtime.set_backward_weight_multiplier(weight_multiplier)
        self.runtime.set_forward_weight_multiplier(weight_multiplier)

        method = backend or edge_spec.backend or (self.backward_cfg.method if self.backward_cfg.enabled else self.forward_cfg.method)
        use_forward = False
        if self.forward_cfg.enabled and (edge_spec.direction == "forward" or not self.backward_cfg.enabled):
            use_forward = True

        try:
            if use_forward:
                forward_cfg = ForwardConfig(
                    enabled=True,
                    method=method,
                    ig_steps=self.forward_cfg.ig_steps,
                    baseline=self.forward_cfg.baseline,
                )
                output = self.runtime.run_forward_contribution(forward_cfg)
            else:
                backward_cfg = BackwardConfig(
                    enabled=True,
                    method=method,
                    ig_steps=self.backward_cfg.ig_steps,
                    baseline=self.backward_cfg.baseline,
                )
                output = self.runtime.run_backward(backward_cfg)
        finally:
            self.runtime.set_backward_weight_multiplier(None)
            self.runtime.set_forward_weight_multiplier(None)

        attr_map = self._extract_attr_map(output)
        modules = list({*(edge_spec.anchor_modules or []), parent_module_name, layer_module_name})
        gathered = self._gather_attr_tensors(attr_map, modules)
        out: Dict[str, torch.Tensor] = {}
        for key, tensor in gathered.items():
            if isinstance(tensor, (list, tuple)):
                tensors = [t for t in tensor if torch.is_tensor(t)]
                if not tensors:
                    continue
                stacked = torch.stack(tensors, dim=0)
                out[key] = self._normalise_attr_tensor(stacked)
            else:
                out[key] = self._normalise_attr_tensor(tensor)
        return out

    def attribute_edge(
        self,
        edge_spec,
        layer_module_name: str,
        target_feature_indices: Optional[torch.Tensor],
        weight_vector: Optional[torch.Tensor] = None,
        parent_module_name: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        if target_feature_indices is not None and target_feature_indices.numel() > 0:
            feat_idx = int(target_feature_indices.flatten()[0].item())
            sae_module = self._sae_resolver(layer_module_name)
            self.runtime.set_target(layer_module_name, feat_idx, sae_module=sae_module)
        else:
            self.runtime.set_target(self.target.layer, self.target.unit, sae_module=self._sae_resolver(self.target.layer))

        try:
            self.runtime.anchor_capture.clear()
        except Exception:
            pass
        anchor_cfg = AnchorConfig(
            capture=list({*(edge_spec.anchor_modules or []), *( [parent_module_name] if parent_module_name else [])}),
            ig_active=_ensure_list(getattr(edge_spec, "anchor_ig_active", None)),
            stop_grad=_ensure_list(getattr(edge_spec, "stop_grad", None)),
        )
        self.runtime.configure_anchors(anchor_cfg)

        weight_multiplier = None
        if weight_vector is not None:
            weight_multiplier = self._build_weight_multiplier(weight_vector, None)
        elif sanitize_layer_name(layer_module_name) == sanitize_layer_name(self.target.layer):
            target_for_mask = target_feature_indices
            mask_weight = torch.ones_like(target_for_mask, dtype=torch.float32) if target_for_mask is not None else None
            weight_multiplier = self._build_weight_multiplier(mask_weight, target_for_mask)
        self.runtime.set_backward_weight_multiplier(weight_multiplier)
        self.runtime.set_forward_weight_multiplier(weight_multiplier)

        method = backend or edge_spec.backend or (self.backward_cfg.method if self.backward_cfg.enabled else self.forward_cfg.method)
        use_forward = False
        if self.forward_cfg.enabled and (edge_spec.direction == "forward" or not self.backward_cfg.enabled):
            use_forward = True

        output: Optional[Dict[str, Any]]
        try:
            if use_forward:
                forward_cfg = ForwardConfig(
                    enabled=True,
                    method=method,
                    ig_steps=self.forward_cfg.ig_steps,
                    baseline=self.forward_cfg.baseline,
                )
                output = self.runtime.run_forward_contribution(forward_cfg)
            else:
                backward_cfg = BackwardConfig(
                    enabled=True,
                    method=method,
                    ig_steps=self.backward_cfg.ig_steps,
                    baseline=self.backward_cfg.baseline,
                )
                output = self.runtime.run_backward(backward_cfg)
        finally:
            self.runtime.set_backward_weight_multiplier(None)
            self.runtime.set_forward_weight_multiplier(None)

        attr_map = self._extract_attr_map(output)
        modules = list({*(edge_spec.anchor_modules or []), parent_module_name, layer_module_name})
        gathered = self._gather_attr_tensors(attr_map, modules)
        tensor = None
        if gathered:
            tensor = next(iter(gathered.values()))
        else:
            tensor, _ = self._select_attr_tensor(attr_map, [parent_module_name, layer_module_name])
            if tensor is not None:
                tensor = tensor
        if tensor is None:
            return None
        return self._normalise_attr_tensor(tensor)

    def cleanup(self) -> None:
        try:
            self.runtime.cleanup()
        finally:
            for handle in reversed(self._extra_wrap_handles):
                try:
                    handle()
                except Exception:
                    pass


__all__ = ["CircuitRuntime"]
