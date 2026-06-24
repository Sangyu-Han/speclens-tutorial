from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.core.base.adapters import ModelAdapter
from src.core.sae.activation_stores.universal_activation_store import UniversalActivationStore


@dataclass
class ResNetBatchMeta:
    sample_ids: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    paths: Sequence[str]


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class ResNetVisionAdapter(ModelAdapter):
    """Adapter to run ResNet vision encoders inside the universal activation store."""

    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None):
        self.model = model.eval()
        unwrapped = _unwrap_module(model)
        if device is None:
            device = next(unwrapped.parameters()).device
        self.device = torch.device(device)
        self._current_sample_ids: Optional[torch.Tensor] = None
        self.current_meta: Optional[Dict[str, Any]] = None

    def get_hook_points(self) -> List[str]:
        base: List[str] = []
        model = _unwrap_module(self.model)

        if hasattr(model, "conv1"):
            base.append("conv1")
        if hasattr(model, "layer1"):
            try:
                n_blocks = len(model.layer1)  # type: ignore[arg-type]
            except TypeError:
                n_blocks = 0
            for idx in range(n_blocks):
                base.append(f"layer1.{idx}")
        if hasattr(model, "layer2"):
            try:
                n_blocks = len(model.layer2)  # type: ignore[arg-type]
            except TypeError:
                n_blocks = 0
            for idx in range(n_blocks):
                base.append(f"layer2.{idx}")
        if hasattr(model, "layer3"):
            try:
                n_blocks = len(model.layer3)  # type: ignore[arg-type]
            except TypeError:
                n_blocks = 0
            for idx in range(n_blocks):
                base.append(f"layer3.{idx}")
        if hasattr(model, "layer4"):
            try:
                n_blocks = len(model.layer4)  # type: ignore[arg-type]
            except TypeError:
                n_blocks = 0
            for idx in range(n_blocks):
                base.append(f"layer4.{idx}")
        if hasattr(model, "avgpool"):
            base.append("avgpool")
        if hasattr(model, "global_pool"):
            base.append("global_pool")
        if hasattr(model, "fc"):
            base.append("fc")

        return base

    def preprocess_input(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        pixel_values = raw_batch["pixel_values"].to(self.device, non_blocking=True)
        labels = raw_batch.get("label")
        if labels is None:
            labels = raw_batch.get("labels")
        if torch.is_tensor(labels):
            labels = labels.to(self.device, non_blocking=True)
        sample_ids = raw_batch.get("sample_id")
        if sample_ids is None:
            sample_ids = raw_batch.get("sample_ids")
        if torch.is_tensor(sample_ids):
            self._current_sample_ids = sample_ids.detach().to(torch.long).cpu()
        else:
            self._current_sample_ids = None

        paths = raw_batch.get("path") or raw_batch.get("paths") or []
        self.current_meta = {
            "sample_ids": sample_ids.detach().cpu() if torch.is_tensor(sample_ids) else sample_ids,
            "labels": labels.detach().cpu() if torch.is_tensor(labels) else labels,
            "paths": list(paths),
        }
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "sample_ids": sample_ids,
            "paths": paths,
        }

    def forward(self, batch: Dict[str, Any]) -> None:
        pixel_values = batch["pixel_values"]
        if torch.is_grad_enabled():
            _ = self.model(pixel_values)
        else:
            with torch.no_grad():
                _ = self.model(pixel_values)

    def get_provenance_spec(self) -> Dict[str, Any]:
        cols = ("sample_id", "y", "x")
        return {"cols": cols, "num_cols": len(cols)}


def create_resnet_store(
    model: nn.Module,
    cfg: Dict[str, Any],
    dataset=None,
    sampler=None,
    collate_fn: Optional[Any] = None,
    on_batch_generated: Optional[Any] = None,
) -> UniversalActivationStore:
    adapter = ResNetVisionAdapter(model, device=cfg.get("device"))
    if collate_fn is not None:
        adapter.collate_fn = collate_fn
    return UniversalActivationStore(
        model,
        cfg,
        adapter,
        dataset,
        sampler,
        on_batch_generated=on_batch_generated,
    )


__all__ = ["ResNetVisionAdapter", "create_resnet_store", "ResNetBatchMeta"]
