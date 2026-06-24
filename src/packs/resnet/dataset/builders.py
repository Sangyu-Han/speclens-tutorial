from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def _resolve_interpolation(mode: str | None) -> InterpolationMode:
    if not mode:
        return InterpolationMode.BICUBIC
    key = mode.upper()
    if hasattr(InterpolationMode, key):
        return getattr(InterpolationMode, key)
    return InterpolationMode.BICUBIC


def _build_transform(cfg: Dict[str, Any]) -> transforms.Compose:
    image_size = int(cfg.get("image_size", 224))
    mean = tuple(cfg.get("mean", DEFAULT_MEAN))
    std = tuple(cfg.get("std", DEFAULT_STD))
    interpolation = _resolve_interpolation(cfg.get("interpolation", "bicubic"))
    is_train = bool(cfg.get("is_train", True))

    if is_train:
        aug = [
            transforms.RandomResizedCrop(image_size, interpolation=interpolation, antialias=True),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        aug = [
            transforms.Resize(int(image_size * 256 / 224), interpolation=interpolation, antialias=True),
            transforms.CenterCrop(image_size),
        ]
    aug.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(aug)


def build_resnet_transform(cfg: Dict[str, Any], *, is_train: Optional[bool] = None) -> transforms.Compose:
    cfg_copy = dict(cfg)
    if is_train is not None:
        cfg_copy["is_train"] = bool(is_train)
    return _build_transform(cfg_copy)


class IndexedImageFolder(datasets.ImageFolder):
    """ImageFolder that keeps the sample index and path in each record."""

    def __init__(self, root: str | Path, *, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return {
            "pixel_values": image,
            "label": int(target) if target is not None else None,
            "sample_id": int(index),
            "path": path,
        }


def resnet_collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = list(batch)
    pixel_values = torch.stack([item["pixel_values"] for item in items])
    labels: Optional[torch.Tensor]
    if items and items[0]["label"] is not None:
        labels = torch.tensor([int(item["label"]) for item in items], dtype=torch.long)
    else:
        labels = None
    sample_ids = torch.tensor([int(item["sample_id"]) for item in items], dtype=torch.long)
    paths = [item["path"] for item in items]
    return {
        "pixel_values": pixel_values,
        "label": labels,
        "sample_id": sample_ids,
        "path": paths,
    }


def build_imagefolder_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    rank: int,
    world_size: int,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
) -> Dict[str, Any]:
    root = Path(dataset_cfg["root"]).expanduser()
    split = dataset_cfg.get("split")
    if split:
        root = root / split
    if not root.exists():
        raise FileNotFoundError(f"ImageFolder root does not exist: {root}")

    transform = _build_transform(dataset_cfg)
    dataset = IndexedImageFolder(str(root), transform=transform)

    shuffle = bool(dataset_cfg.get("shuffle", True))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=bool(dataset_cfg.get("drop_last", False)),
    )

    return {
        "dataset": dataset,
        "collate_fn": resnet_collate_fn,
        "sampler": sampler,
    }


def build_indexing_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    world_size: int,
    rank: int,
    **_,
):
    cfg = dict(dataset_cfg)
    cfg.setdefault("is_train", False)
    dataset_info = build_imagefolder_dataset(
        cfg,
        rank=rank,
        world_size=world_size,
    )
    return dataset_info["dataset"], dataset_info["sampler"]


def build_collate_fn(_dataset: Any) -> Callable[[Iterable[Dict[str, Any]]], Dict[str, Any]]:
    return resnet_collate_fn


__all__ = [
    "DEFAULT_MEAN",
    "DEFAULT_STD",
    "build_resnet_transform",
    "IndexedImageFolder",
    "resnet_collate_fn",
    "build_imagefolder_dataset",
    "build_indexing_dataset",
    "build_collate_fn",
]
