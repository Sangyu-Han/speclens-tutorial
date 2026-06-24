"""Activation-store wiring for the cifar_cnn pack.

The CIFAR CNN is built with resnet-style module names, so the resnet pack's
``ResNetVisionAdapter`` (which discovers conv1/layer1..4/global_pool/fc and emits
spatial (sample_id, y, x) provenance) works unchanged. We just re-export it.
"""
from __future__ import annotations

from src.packs.resnet.models.adapters import ResNetVisionAdapter, create_resnet_store

# Alias under a pack-local name for clarity in configs/imports.
CifarCNNVisionAdapter = ResNetVisionAdapter
create_cifar_store = create_resnet_store

__all__ = ["CifarCNNVisionAdapter", "create_cifar_store", "ResNetVisionAdapter", "create_resnet_store"]
