# circuits/generic_edge_runtime.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import torch

from src.core.runtime.attribution_runtime import AnchorConfig, BackwardConfig, ForwardConfig


def _ensure_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


class GenericAttrRuntime:
    """
    Lightweight EdgeAttributor implementation that wraps an AttributionRuntime.
    It mirrors ClipAttrRuntime but leaves model/sae construction to the caller.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        target_layer: str,
        target_unit: int,
        sae_resolver: Callable[[str], torch.nn.Module],
        backward_cfg: Optional[BackwardConfig] = None,
        forward_cfg: Optional[ForwardConfig] = None,
    ) -> None:
        self.runtime = runtime
        self.target_layer = target_layer
        self.target_unit = target_unit
        self._sae_resolver = sae_resolver
        self.backward_cfg = backward_cfg or BackwardConfig(
            enabled=True, method="ig", ig_steps=32, baseline="zeros"
        )
        self.forward_cfg = forward_cfg or ForwardConfig(
            enabled=False, method="ig", ig_steps=32, baseline="zeros"
        )
        self.runtime.set_target(self.target_layer, self.target_unit, sae_module=self._sae_resolver(self.target_layer))

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
        weight = weight.to(dtype=torch.float32)
        idx = target_feature_indices.flatten().long() if target_feature_indices is not None else None
        if idx is not None and idx.numel() == 0:
            return None

        def _multiplier(latent: torch.Tensor) -> torch.Tensor:
            w = weight.to(device=latent.device, dtype=latent.dtype)
            try:
                if w.dim() < latent.dim():
                    while w.dim() < latent.dim():
                        w = w.unsqueeze(0)
                return torch.broadcast_to(w, latent.shape)
            except Exception:
                pass

            mask = torch.zeros_like(latent)
            if idx is None:
                feat = latent.shape[-1]
                flat = w.flatten()[:feat]
                if flat.numel() < feat:
                    pad = torch.zeros(feat - flat.numel(), device=flat.device, dtype=flat.dtype)
                    flat = torch.cat([flat, pad], dim=0)
                view = flat
                while view.dim() < latent.dim():
                    view = view.unsqueeze(0)
                mask = view.expand(latent.shape)
            else:
                index = idx.to(device=latent.device)
                valid = index < latent.shape[-1]
                if not torch.all(valid):
                    index = index[valid]
                    w = w.flatten()[: index.numel()]
                if index.numel() == 0:
                    return mask
                expanded = w.view(*([1] * (latent.dim() - 1)), -1).expand_as(latent)
                mask.index_copy_(-1, index, expanded.index_select(-1, index))
            return mask

        return _multiplier

    def _extract_attr_map(self, output: Optional[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not output:
            return {}
        if isinstance(output, dict) and "attr" in output and isinstance(output["attr"], dict):
            return output["attr"]
        return output if isinstance(output, dict) else {}

    def _normalise_attr_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.view(1, 1, -1)
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        if tensor.dim() >= 3:
            return tensor
        raise RuntimeError(f"Unsupported attribution tensor rank {tensor.dim()}")

    def _gather_attr_tensors(
        self,
        attr_map: Dict[str, torch.Tensor],
        modules: list[str],
    ) -> Dict[str, torch.Tensor]:
        wanted = {m for m in modules if m}
        out: Dict[str, torch.Tensor] = {}
        for key, tensor in attr_map.items():
            if not torch.is_tensor(tensor):
                continue
            if key not in wanted:
                continue
            out[key] = tensor
        return out

    # ------------------------------------------------------------------ public API
    def attribute_edge_multi(
        self,
        edge_spec,
        layer_module_name: str,
        backend: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        self.runtime.set_target(self.target_layer, self.target_unit, sae_module=self._sae_resolver(self.target_layer))
        try:
            self.runtime.anchor_capture.clear()
        except Exception:
            pass
        anchor_cfg = AnchorConfig(
            capture=list({*(edge_spec.anchor_modules or []), layer_module_name}),
            ig_active=_ensure_list(getattr(edge_spec, "anchor_ig_active", None)),
            stop_grad=_ensure_list(getattr(edge_spec, "stop_grad", None)),
        )
        self.runtime.configure_anchors(anchor_cfg)
        method = backend or edge_spec.backend or (self.backward_cfg.method if self.backward_cfg.enabled else self.forward_cfg.method)
        backward_cfg = BackwardConfig(
            enabled=True,
            method=method,
            ig_steps=self.backward_cfg.ig_steps,
            baseline=self.backward_cfg.baseline,
        )
        output = self.runtime.run_backward(backward_cfg)
        attr_map = self._extract_attr_map(output)
        modules = list({*(edge_spec.anchor_modules or []), layer_module_name})
        gathered = self._gather_attr_tensors(attr_map, modules)
        return {k: self._normalise_attr_tensor(v.detach().cpu()) for k, v in gathered.items()}

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
            self.runtime.set_target(layer_module_name, feat_idx, sae_module=self._sae_resolver(layer_module_name))
        else:
            self.runtime.set_target(self.target_layer, self.target_unit, sae_module=self._sae_resolver(self.target_layer))

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
        elif layer_module_name == self.target_layer:
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
        if tensor is None:
            return None
        return self._normalise_attr_tensor(tensor.detach().cpu())

    def cleanup(self) -> None:
        try:
            self.runtime.cleanup()
        except Exception:
            pass
