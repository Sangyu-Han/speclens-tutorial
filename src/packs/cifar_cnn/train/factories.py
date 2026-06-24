"""Pack factories for cifar_cnn (mirrors src.packs.resnet.train.factories).

Wires our CIFAR CNN + CIFAR-100 dataset into the shared SAE-training / indexing
pipeline. Adapter and activation store are reused from the resnet pack.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.packs.cifar_cnn.dataset.builders import build_cifar100_dataset
from src.packs.cifar_cnn.models.adapters import create_cifar_store
from src.packs.cifar_cnn.models.model_loaders import load_cifar_cnn_model

LOGGER = logging.getLogger(__name__)


def _prepend_sys_path(path_like: str | Path) -> None:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def setup_env(config: Optional[Dict[str, Any]] = None, *, project_root: Optional[Path] = None) -> None:
    cfg = config or {}
    for rel in cfg.get("sys_paths") or []:
        if project_root and not Path(rel).is_absolute():
            _prepend_sys_path(project_root / rel)
        else:
            _prepend_sys_path(rel)


def load_model(
    model_cfg: Dict[str, Any],
    *positional,
    device: Optional[torch.device] = None,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    if device is None:
        if not positional:
            raise TypeError("'device' must be provided either positionally or as a keyword argument")
        device, *positional = positional
    return load_cifar_cnn_model(
        model_cfg, device=device, rank=rank, world_size=world_size, full_config=full_config
    )


def build_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    rank: int,
    world_size: int,
    device: torch.device | None = None,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
):
    return build_cifar100_dataset(dataset_cfg, rank=rank, world_size=world_size, full_config=full_config)


def create_store(*, model, cfg: Dict[str, Any], dataset, sampler, collate_fn, on_batch_generated=None, **_):
    return create_cifar_store(
        model=model,
        cfg=cfg,
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        on_batch_generated=on_batch_generated,
    )


__all__ = ["setup_env", "load_model", "build_dataset", "create_store"]
