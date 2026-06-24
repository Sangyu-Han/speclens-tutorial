"""Loader for the small CIFAR CNN (mirrors resnet pack's load_resnet_model).

Unlike the resnet pack (timm), this builds our own ``CifarResNet`` and loads the
checkpoint written by ``train_cnn.py``. Architecture is taken from the
checkpoint's ``arch`` blob unless overridden in the config.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.packs.cifar_cnn.models.model import CifarResNet

LOGGER = logging.getLogger(__name__)


def load_cifar_cnn_model(
    model_cfg: Dict[str, Any],
    *,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
) -> nn.Module:
    arch: Dict[str, Any] = dict(model_cfg.get("arch") or {})
    state = None

    ckpt_path = model_cfg.get("ckpt") or model_cfg.get("checkpoint")
    if ckpt_path:
        path = Path(ckpt_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"CIFAR CNN checkpoint not found: {path}")
        blob = torch.load(str(path), map_location="cpu")
        if isinstance(blob, dict) and "state_dict" in blob:
            state = blob["state_dict"]
            arch = {**(blob.get("arch") or {}), **arch}  # config overrides checkpoint
        else:
            state = blob

    model = CifarResNet(**arch) if arch else CifarResNet()
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=bool(model_cfg.get("strict_load", True)))
        if rank == 0:
            LOGGER.info("[cifar_cnn] loaded ckpt missing=%d unexpected=%d", len(missing), len(unexpected))

    model.eval().to(device)
    dtype = model_cfg.get("dtype")
    if dtype:
        model = model.to(getattr(torch, dtype))
    return model


__all__ = ["load_cifar_cnn_model"]
