from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.packs.resnet.dataset.builders import build_imagefolder_dataset
from src.packs.resnet.models.adapters import create_resnet_store
from src.packs.resnet.models.model_loaders import load_resnet_model

LOGGER = logging.getLogger(__name__)


def _prepend_sys_path(path_like: str | Path) -> None:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def setup_env(config: Optional[Dict[str, Any]] = None, *, project_root: Optional[Path] = None) -> None:
    cfg = config or {}
    paths = cfg.get("sys_paths") or []
    for rel in paths:
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
    """Create a ResNet model using either keyword or legacy positional arguments."""

    if device is None:
        if not positional:
            raise TypeError("'device' must be provided either as a positional or keyword argument")
        device, *positional = positional

    if positional:
        kwargs.setdefault("logger", positional[0])

    return load_resnet_model(
        model_cfg,
        device=device,
        rank=rank,
        world_size=world_size,
        full_config=full_config,
        **kwargs,
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
    return build_imagefolder_dataset(
        dataset_cfg,
        rank=rank,
        world_size=world_size,
        full_config=full_config,
    )


def create_store(
    *,
    model,
    cfg: Dict[str, Any],
    dataset,
    sampler,
    collate_fn,
    on_batch_generated=None,
    **_,
):
    return create_resnet_store(
        model=model,
        cfg=cfg,
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        on_batch_generated=on_batch_generated,
    )


__all__ = ["setup_env", "load_model", "build_dataset", "create_store"]
