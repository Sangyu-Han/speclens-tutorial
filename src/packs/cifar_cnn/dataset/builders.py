"""CIFAR-100 dataset builders for the cifar_cnn pack.

Returns the same record schema as the resnet pack
(``pixel_values``/``label``/``sample_id``/``path``) and reuses
``resnet_collate_fn`` so the universal activation store / indexing pipeline work
unchanged. ``path`` is a synthetic id (CIFAR has no files); recover the raw
image by ``sample_id`` from a no-transform CIFAR100 instance.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from src.packs.resnet.dataset.builders import resnet_collate_fn

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class PatchCorner:
    """Stamp a magenta corner patch (in normalized space) with probability `prob`.
    Off by default (patch_size=0); used to train patch-aware SAEs on the shortcut model."""

    def __init__(self, size: int, prob: float, mean, std):
        self.size = int(size); self.prob = float(prob)
        self.val = ((torch.tensor([1.0, 0.0, 1.0]) - torch.tensor(mean)) / torch.tensor(std))[:, None, None]

    def __call__(self, x):
        if self.size > 0 and random.random() < self.prob:
            x = x.clone(); x[:, :self.size, :self.size] = self.val
        return x


def build_cifar_transform(cfg: Dict[str, Any], *, is_train: bool) -> transforms.Compose:
    mean = tuple(cfg.get("mean", CIFAR100_MEAN))
    std = tuple(cfg.get("std", CIFAR100_STD))
    if is_train:
        aug = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        aug = []
    aug += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    ps = int(cfg.get("patch_size", 0))
    if ps > 0:
        aug.append(PatchCorner(ps, cfg.get("patch_prob", 0.5), mean, std))
    return transforms.Compose(aug)


class IndexedCIFAR100(datasets.CIFAR100):
    """CIFAR100 that yields the SpecLens record dict."""

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, target = super().__getitem__(index)
        split = "train" if self.train else "test"
        return {
            "pixel_values": img,
            "label": int(target),
            "sample_id": int(index),
            "path": f"cifar100/{split}/{index}",
        }


def build_cifar100_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    rank: int,
    world_size: int,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
) -> Dict[str, Any]:
    root = Path(dataset_cfg.get("root", "./data")).expanduser()
    split = str(dataset_cfg.get("split", "train"))
    is_train = bool(dataset_cfg.get("is_train", split == "train"))
    transform = build_cifar_transform(dataset_cfg, is_train=is_train)
    dataset = IndexedCIFAR100(
        root=str(root),
        train=(split == "train"),
        download=bool(dataset_cfg.get("download", True)),
        transform=transform,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=bool(dataset_cfg.get("shuffle", is_train)),
        drop_last=bool(dataset_cfg.get("drop_last", False)),
    )
    return {"dataset": dataset, "collate_fn": resnet_collate_fn, "sampler": sampler}


def build_indexing_dataset(dataset_cfg: Dict[str, Any], *, world_size: int, rank: int, **_):
    cfg = dict(dataset_cfg)
    cfg.setdefault("is_train", False)
    info = build_cifar100_dataset(cfg, rank=rank, world_size=world_size)
    return info["dataset"], info["sampler"]


def build_collate_fn(_dataset: Any, **_) -> Callable[[Iterable[Dict[str, Any]]], Dict[str, Any]]:
    return resnet_collate_fn


__all__ = [
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "build_cifar_transform",
    "IndexedCIFAR100",
    "build_cifar100_dataset",
    "build_indexing_dataset",
    "build_collate_fn",
]
