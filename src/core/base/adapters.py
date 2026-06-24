import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

from src.core.base.layout import LayoutSpec, infer_layout, build_provenance_from_layout

class ModelAdapter(ABC):
    """모델별 I/O 및 forward 인터페이스를 통일하기 위한 어댑터 베이스."""
    collate_fn: Optional[Callable] = None
    current_meta: Optional[Dict[str, Any]] = None

    def get_hook_points(self) -> List[str]:
        return []

    def create_dummy_input(self, batch_size: int, device) -> Any:
        raise NotImplementedError

    @abstractmethod
    def preprocess_input(self, raw_batch: Any) -> Any:
        ...

    @abstractmethod
    def forward(self, batch: Any) -> None:
        ...

    def get_provenance_spec(self) -> dict:
        """
        어댑터가 생성할 provenance 스키마를 정의.
        반환 예:
          {"cols": ("sample_id","frame_idx","y","x")}              # 4열
          {"cols": ("sample_id","frame_idx","y","x","prompt_id","uid")}  # 6열
        num_cols는 cols 길이로 유도하며, 명시적으로 지정해도 됨({"num_cols": 6}).
        """
        # 기본값(프로프트 없는 일반 모델): 3열 — most models don't need frame_idx.
        # Models that do (e.g. SAM2) already override this method.
        cols = ("sample_id", "y", "x")
        return {"cols": cols, "num_cols": len(cols)}

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    # ---- LayoutSpec integration --------------------------------- #

    def get_layout_spec(
        self, act_name: str, tensor: torch.Tensor
    ) -> Optional[LayoutSpec]:
        """Return LayoutSpec for activation. Override for per-layer specs."""
        overrides = getattr(self, "_layout_overrides", {})
        spec = overrides.get(act_name)
        if spec is not None:
            return spec
        return infer_layout(tensor)

    def build_token_provenance(
        self,
        *,
        act_name: str,
        raw_output: Any,
        flattened_tokens: torch.Tensor,
        fidx_hint: Optional[Union[int, torch.Tensor]] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Generic provenance builder using LayoutSpec."""
        # Handle dict outputs (Mask2Former-style)
        tensor = raw_output
        if not torch.is_tensor(tensor):
            if isinstance(tensor, dict):
                for key in ("pred_logits", "pred_masks", "outputs"):
                    if key in tensor and torch.is_tensor(tensor[key]):
                        tensor = tensor[key]
                        break
            if not torch.is_tensor(tensor):
                N = flattened_tokens.shape[0]
                return self._fallback_provenance(N)

        spec = self.get_layout_spec(act_name, tensor)
        if spec is None:
            return self._fallback_provenance(flattened_tokens.shape[0])

        return build_provenance_from_layout(
            spec=spec,
            raw_output=tensor,
            sample_ids=self._get_sample_ids(),
            fidx_hint=fidx_hint,
            device=getattr(self, "device", torch.device("cpu")),
        )

    def _get_sample_ids(self) -> torch.Tensor:
        """Resolve current-batch sample IDs from adapter state."""
        sids = getattr(self, "_current_sample_ids", None)
        if sids is not None:
            return sids
        meta = getattr(self, "current_meta", None)
        if meta is not None:
            if hasattr(meta, "sample_ids") and torch.is_tensor(meta.sample_ids):
                return meta.sample_ids.to(torch.long)
            if isinstance(meta, dict):
                s = meta.get("sample_ids")
                if torch.is_tensor(s):
                    return s.to(torch.long)
        return torch.zeros(1, dtype=torch.long)

    def _fallback_provenance(self, N: int) -> torch.Tensor:
        """Return a zero provenance tensor when LayoutSpec is unavailable."""
        spec = self.get_provenance_spec()
        num_cols = int(
            spec.get("num_cols", len(spec.get("cols", ("sample_id", "y", "x"))))
        )
        return torch.zeros(N, max(num_cols, 1), dtype=torch.long)

    def __repr__(self) -> str:
        hook_points = []
        try:
            hook_points = list(self.get_hook_points())
        except Exception:
            hook_points = []
        if hook_points:
            preview = ", ".join(hook_points[:3])
            if len(hook_points) > 3:
                preview += ", ..."
            hook_repr = f"[{preview}]"
        else:
            hook_repr = "[]"
        model_obj = getattr(self, "model", None)
        model_name = type(model_obj).__name__ if model_obj is not None else None
        device = getattr(self, "device", None)
        collate = getattr(self, "collate_fn", None)
        if callable(collate):
            collate_name = getattr(collate, "__name__", type(collate).__name__)
        else:
            collate_name = collate
        return (
            f"{self.__class__.__name__}(model={model_name}, device={device}, "
            f"hook_points={hook_repr}, collate_fn={collate_name})"
        )


class GenericModelAdapter(ModelAdapter):
    """일반 PyTorch 모듈용 범용 어댑터 예시."""
    def __init__(self, model: nn.Module, hook_points: Optional[List[str]] = None, device: Optional[Union[str, torch.device]] = None):
        self.model = model.eval()
        self._hook_points = hook_points or []
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

    def get_hook_points(self) -> List[str]:
        if self._hook_points:
            return self._hook_points
        pts = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.TransformerEncoderLayer)):
                pts.append(name)
        return pts[:10]

    def preprocess_input(self, data: Any) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        if isinstance(data, (list, tuple)):
            return torch.stack([d.to(self.device, non_blocking=True) for d in data])
        raise ValueError(f"Unsupported data type: {type(data)}")

    def create_dummy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        first = next(self.model.parameters())
        if first.ndim == 4:
            in_ch = first.shape[1]
            return torch.randn(batch_size, in_ch, 224, 224, device=device)
        if first.ndim == 2:
            in_feat = first.shape[1]
            return torch.randn(batch_size, in_feat, device=device)
        return torch.randn(batch_size, 1024, device=device)

    @torch.no_grad()
    def forward(self, batch: Any) -> None:
        _ = self.model(batch)


# ============================== #
#          Factory API           #
# ============================== #

from src.core.sae.activation_stores.universal_activation_store import UniversalActivationStore

def create_activation_store(
    model: nn.Module,
    cfg: Dict,
    adapter: Optional[ModelAdapter] = None,
    dataset=None,
    sampler=None
) -> UniversalActivationStore:
    if adapter is None:
        adapter = GenericModelAdapter(model, cfg.get("hook_points"))
    return UniversalActivationStore(model, cfg, adapter, dataset, sampler)
