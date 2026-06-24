#!/usr/bin/env python3
"""Unified CLI for SAE training across packs."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path

# Ensure THIS project's src/ is first on sys.path, overriding any editable installs
# from other projects (e.g. General_SAE_project) that may shadow our src package.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.sae.train.runner import SAETrainingPipeline, main_worker

PACK_DEFAULT_CONFIGS = {
    "sam2": "configs/sam2_sav_batchtopk_train_batchtopk.yaml",
    "clip": "configs/clip_imagenet_train.yaml",
    "siglip": "configs/siglip_imagenet_train.yaml",
    "dinov2": "configs/dinov2_imagenet_train.yaml",
    "dinov3": "configs/dinov3_imagenet_train.yaml",
    "resnet": "configs/resnet18_imagenet_train.yaml",
    "mask2former": "configs/mask2former_sav_train.yaml",
}
DEFAULT_PACK = "mask2former"


def _resolve_config(args: argparse.Namespace) -> str:
    if args.config:
        return args.config
    pack = args.pack or DEFAULT_PACK
    if pack not in PACK_DEFAULT_CONFIGS:
        raise ValueError(f"Unknown pack '{pack}'. Available: {sorted(PACK_DEFAULT_CONFIGS)}")
    return PACK_DEFAULT_CONFIGS[pack]


def _list_packs() -> None:
    print("Available pack defaults:")
    for name, path in sorted(PACK_DEFAULT_CONFIGS.items()):
        print(f"  {name:>8}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SAE with config (Universal Activation Store)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a training config YAML")
    parser.add_argument(
        "--pack",
        type=str,
        choices=sorted(PACK_DEFAULT_CONFIGS.keys()),
        default=None,
        help="Use the default config for the specified pack",
    )
    parser.add_argument(
        "--list-packs",
        action="store_true",
        help="List supported pack defaults and exit",
    )
    args, extras = parser.parse_known_args()

    if args.list_packs:
        _list_packs()
        return

    config_path = _resolve_config(args)
    if extras:
        logging.warning("Ignoring unparsed extras: %s", " ".join(extras))

    is_ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if is_ddp:
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)
        main_worker(rank, world_size, Namespace(config=config_path))
    else:
        print(f"🚀 Running SAE training (config={config_path}) in single-process mode.")
        pipeline = SAETrainingPipeline(config_path, rank=0, world_size=1)
        try:
            pipeline.train()
        finally:
            pipeline.cleanup()


if __name__ == "__main__":
    main()
