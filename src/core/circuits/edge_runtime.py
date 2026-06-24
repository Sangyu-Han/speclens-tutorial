# circuits/edge_runtime.py
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Protocol
import torch

from .topology import CircuitEdgeSpec
class EdgeAttributor(Protocol):
    """
    core/circuits/runtime/* 에서 제공하는 runtime 클래스가
    이런 인터페이스를 구현한다고 가정.
    """

    def attribute_edge(
        self,
        edge_spec: CircuitEdgeSpec,
        layer_module_name: str,
        target_feature_indices: Optional[torch.Tensor],
        weight_vector: Optional[torch.Tensor] = None,
        parent_module_name: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> torch.Tensor:
        """
        반환: attribution tensor of shape (batch, token, parent_features)
        - layer_module_name: 실제 PyTorch 모듈 이름(예: "model.blocks.9::sae_layer#latent")
        - target_feature_indices: child layer의 feature index 리스트 (1D tensor)
        - weight_vector: grad/JVP 직전에 곱할 weight (target dim과 브로드캐스트 가능)
        - parent_module_name: parent layer 모듈 이름(있으면 anchor 선택 시 우선 사용)

        이 함수 내부 구현은 runtime 구현체(예: ClipCircuitRuntime) 쪽에 있음.
        """
        ...

    def attribute_edge_multi(
        self,
        edge_spec: CircuitEdgeSpec,
        layer_module_name: str,
        target_feature_indices: Optional[torch.Tensor] = None,
        weight_vector: Optional[torch.Tensor] = None,
        parent_module_name: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Optional: return attribution tensors for all captured anchors (module-name keyed).
        """
        ...


@dataclass
class EdgeAttrRuntime:
    """
    FeatureTreeBuilder 가 사용하는 래퍼.
    """
    runtime: EdgeAttributor
    layer_to_module: Dict[str, str]
    module_to_layer: Dict[str, str] = None

    def __post_init__(self):
        self.module_to_layer = {}
        for layer, module in self.layer_to_module.items():
            self.module_to_layer[module] = layer
            if "#latent" in module:
                err_mod = module.replace("#latent", "#error_coeff")
                self.module_to_layer.setdefault(err_mod, layer)

    def _module_of_layer(self, layer: str) -> str:
        try:
            return self.layer_to_module[layer]
        except KeyError:
            raise KeyError(f"module name for layer '{layer}' not found")

    def compute_edge_attribution(
        self,
        edge_spec: CircuitEdgeSpec,
        target_layer: str,
        target_feature_indices: torch.Tensor,
        weight_vector: Optional[torch.Tensor] = None,
        source_layer: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> torch.Tensor:
        """
        target layer의 특정 feature 들(=target units)에 대한
        source layer feature attribution을 계산.

        반환 shape: (batch, token, parent_features)
        """
        module_name = self._module_of_layer(target_layer)
        parent_module_name = None
        if source_layer is not None:
            try:
                parent_module_name = self._module_of_layer(source_layer)
            except KeyError:
                parent_module_name = None
        attr = self.runtime.attribute_edge(
            edge_spec=edge_spec,
            layer_module_name=module_name,
            target_feature_indices=target_feature_indices,
            weight_vector=weight_vector,
            parent_module_name=parent_module_name,
            backend=backend or edge_spec.backend,
        )
        return attr

    def compute_edge_attribution_split(
        self,
        edge_spec: CircuitEdgeSpec,
        target_layer: str,
        target_feature_indices: torch.Tensor,
        source_layer: Optional[str] = None,
        weight_vector: Optional[torch.Tensor] = None,
        backend: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (latent_attr, error_attr_optional)
        - prefer multi-anchor capture (latent + error) in a single backward pass
        - fallback to per-edge attr when multi-capture not available
        """
        latent_mod = self._module_of_layer(source_layer) if source_layer is not None else None
        err_mod = latent_mod.replace("#latent", "#error_coeff") if latent_mod and "#latent" in latent_mod else None

        # try single-pass multi-anchor capture
        if hasattr(self.runtime, "attribute_edge_multi"):
            attr_map = self.runtime.attribute_edge_multi(
                edge_spec=edge_spec,
                layer_module_name=self._module_of_layer(target_layer),
                target_feature_indices=target_feature_indices,
                weight_vector=weight_vector,
                parent_module_name=latent_mod,
                backend=backend or edge_spec.backend,
            )
            latent_attr = attr_map.get(latent_mod) if latent_mod else None
            err_attr = attr_map.get(err_mod) if err_mod else None
            if latent_attr is not None or err_attr is not None:
                return latent_attr, err_attr

        # fallback: per-edge attribution + optional error capture
        latent = self.compute_edge_attribution(
            edge_spec=edge_spec,
            target_layer=target_layer,
            target_feature_indices=target_feature_indices,
            weight_vector=weight_vector,
            source_layer=source_layer,
            backend=backend,
        )
        err_attr = None
        if hasattr(self.runtime, "attribute_edge_multi"):
            attr_map = self.runtime.attribute_edge_multi(
                edge_spec=edge_spec,
                layer_module_name=self._module_of_layer(target_layer),
                target_feature_indices=None,
                weight_vector=None,
                parent_module_name=latent_mod,
                backend=backend or edge_spec.backend,
            )
            if err_mod and err_mod in attr_map:
                err_attr = attr_map[err_mod]
        return latent, err_attr

    def compute_target_attribution_map(
        self,
        edge_spec: CircuitEdgeSpec,
        backend: Optional[str] = None,
        target_feature_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(self.runtime, "attribute_edge_multi"):
            return {}
        module_name = self._module_of_layer(edge_spec.src)
        target_indices = target_feature_indices
        if target_indices is None:
            target_obj = getattr(self.runtime, "target", None)
            if target_obj is not None:
                unit = getattr(target_obj, "unit", None)
                if unit is not None:
                    try:
                        target_indices = torch.tensor([int(unit)], dtype=torch.long)
                    except Exception:
                        target_indices = None
        attr_map = self.runtime.attribute_edge_multi(
            edge_spec=edge_spec,
            layer_module_name=module_name,
            target_feature_indices=target_indices,
            weight_vector=None,
            parent_module_name=None,
            backend=backend or edge_spec.backend,
        )
        # drop error_coeff modules for weighting
        filtered: Dict[str, torch.Tensor] = {}
        for k, v in attr_map.items():
            if "#error_coeff" in k:
                continue
            filtered[k] = v
        return filtered

    def compute_target_attribution(
        self,
        edge_spec: CircuitEdgeSpec,
        source_layer: str,
        backend: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        target -> parent layer attribution (feature-level).
        """
        try:
            module_name = self._module_of_layer(source_layer)
        except KeyError:
            return None
        return self.runtime.attribute_edge(
            edge_spec=edge_spec,
            layer_module_name=module_name,
            target_feature_indices=None,
            weight_vector=None,
            parent_module_name=module_name,
            backend=backend or edge_spec.backend,
        )
