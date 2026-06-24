#!/usr/bin/env python3
"""Unified CLI for SAE decile indexing."""
from __future__ import annotations

import argparse
import logging
from typing import Dict

from src.core.indexing.index_runner import run_indexing

PACK_DEFAULT_CONFIGS: Dict[str, str] = {
    "sam2": "configs/sam2_sav_feature_index.yaml",
    "clip": "configs/clip_imagenet_index.yaml",
    "dinov2": "configs/dinov2_imagenet_index.yaml",
    "dinov3": "configs/dinov3_imagenet_index.yaml",
    "mask2former": "configs/mask2former_sav_index.yaml",
}
DEFAULT_PACK = "sam2"


def _resolve_config(args: argparse.Namespace) -> str:
    if args.config:
        return args.config
    pack = args.pack or DEFAULT_PACK
    if pack not in PACK_DEFAULT_CONFIGS:
        raise ValueError(
            f"Unknown pack '{pack}'. Available: {sorted(PACK_DEFAULT_CONFIGS.keys())}"
        )
    return PACK_DEFAULT_CONFIGS[pack]


def _list_packs() -> None:
    print("Available pack defaults:")
    for name, path in sorted(PACK_DEFAULT_CONFIGS.items()):
        print(f"  {name:>8}: {path}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Run SAE decile indexing")
    parser.add_argument("--config", type=str, default=None, help="Path to indexing config YAML")
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
    parser.add_argument(
        "--l2-warn-threshold",
        type=float,
        default=None,
        help="Emit a warning whenever batch reconstruction L2 exceeds this value (default: use config or disabled)",
    )
    args, extras = parser.parse_known_args()

    if args.list_packs:
        _list_packs()
        return

    config_path = _resolve_config(args)
    if extras:
        logging.warning("Ignoring unparsed extras: %s", " ".join(extras))

    run_indexing(config_path, l2_warn_threshold=args.l2_warn_threshold)


if __name__ == "__main__":
    main()
