from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict

from src.core.indexing.registry_utils import load_obj

DEFAULT_LEDGER_CLASS = "src.packs.clip.offline.offline_meta_ledger:ClipJSONLedger"


def resolve_offline_part_modulus(cfg: Dict[str, Any]) -> int:
    offline_cfg = cfg.get("offline_meta") or {}
    indexing_cfg = cfg.get("indexing") or {}
    raw = offline_cfg.get("part_modulus", indexing_cfg.get("partition_modulus", 128))
    try:
        return max(1, int(raw))
    except Exception:
        return 128


def build_offline_ledger(cfg: Dict[str, Any], *, root_override: str | Path | None = None):
    """
    Instantiate the offline meta ledger specified in the config.
    - honors offline_meta.part_modulus (falls back to indexing.partition_modulus)
    - passes optional kwargs only when supported by the ledger __init__
    """
    offline_cfg = cfg.get("offline_meta") or {}
    ledger_path = offline_cfg.get("ledger_class", DEFAULT_LEDGER_CLASS)
    ledger_cls = load_obj(ledger_path)

    root_raw = root_override or offline_cfg.get("root_dir") or cfg.get("indexing", {}).get("offline_meta_root", "")
    if not root_raw:
        raise ValueError("offline_meta_root is required to build the offline ledger")
    root = Path(root_raw)

    part_mod = resolve_offline_part_modulus(cfg)
    kwargs: Dict[str, Any] = {"root_dir": root, "part_modulus": part_mod}

    sig = inspect.signature(ledger_cls)
    for name in ("run_id", "partition_by_run_id", "compression", "filename_prefix"):
        if name in offline_cfg and name in sig.parameters:
            kwargs[name] = offline_cfg[name]

    return ledger_cls(**kwargs)


__all__ = ["build_offline_ledger", "resolve_offline_part_modulus", "DEFAULT_LEDGER_CLASS"]
