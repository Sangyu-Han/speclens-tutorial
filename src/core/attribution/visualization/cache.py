from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image
from src.core.indexing.registry_utils import sanitize_layer_name


def _tile_thumbnail(images: List[Path], dest: Path, tile_size: int = 224) -> Optional[str]:
    if not images:
        return None
    cols = 2
    rows = (len(images) + cols - 1) // cols
    thumb_width = cols * tile_size
    thumb_height = rows * tile_size
    canvas = Image.new("RGB", (thumb_width, thumb_height), (0, 0, 0))
    for idx, img_path in enumerate(images[: cols * rows]):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        img = img.resize((tile_size, tile_size))
        row = idx // cols
        col = idx % cols
        canvas.paste(img, (col * tile_size, row * tile_size))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return str(dest)


def write_viz_manifest(
    *,
    layer: str,
    unit: int,
    cache_root: Path,
    entries: Iterable[dict],
    max_samples: int = 4,
) -> None:
    entries = list(entries)
    if not entries:
        return
    layer_sanitised = sanitize_layer_name(layer)
    target_dir = cache_root / layer_sanitised / f"unit_{unit}"
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        cache_root_resolved = cache_root.resolve()
    except Exception:
        cache_root_resolved = cache_root

    def _reuse_or_copy(src: Path, dest: Path) -> Optional[Path]:
        if not src.exists():
            return None
        try:
            if cache_root_resolved in src.resolve().parents or src.resolve() == cache_root_resolved:
                return src
        except Exception:
            pass
        try:
            shutil.copy2(src, dest)
            return dest
        except Exception:
            return None

    sorted_entries = sorted(entries, key=lambda e: abs(e.get("score") or 0.0), reverse=True)
    chosen = []
    copied_paths: List[Path] = []
    for idx, info in enumerate(sorted_entries[:max_samples]):
        source = Path(info["path"])
        if not source.exists():
            continue
        dest = target_dir / f"{idx:02d}_{source.name}"
        panel_path = _reuse_or_copy(source, dest)
        if panel_path is None:
            continue
        record = dict(info)
        record["path"] = str(panel_path)
        input_src = info.get("input_feature_map")
        if input_src:
            input_path = Path(input_src)
            if input_path.exists():
                input_dest = target_dir / f"{idx:02d}_input_feature_map_{input_path.name}"
                chosen_input = _reuse_or_copy(input_path, input_dest)
                if chosen_input is not None:
                    record["input_feature_map"] = str(chosen_input)
                else:
                    record.setdefault("input_feature_map", str(panel_path))
        else:
            record.setdefault("input_feature_map", str(panel_path))

        anchor_records: List[dict] = []
        for a_idx, anchor in enumerate(info.get("anchor_heatmaps", []) or []):
            anchor_path = Path(anchor.get("path", ""))
            if not anchor_path.exists():
                continue
            dest_path = target_dir / f"{idx:02d}_anchor_{a_idx:02d}_{anchor_path.name}"
            chosen_anchor = _reuse_or_copy(anchor_path, dest_path)
            if chosen_anchor is None:
                continue
            anchor_records.append({
                "name": anchor.get("name"),
                "path": str(chosen_anchor),
            })
        if anchor_records:
            record["anchor_heatmaps"] = anchor_records

        contrib_records: List[dict] = []
        for c_idx, contrib in enumerate(info.get("output_contributions", []) or []):
            contrib_path = Path(contrib.get("path", ""))
            if not contrib_path.exists():
                continue
            dest_path = target_dir / f"{idx:02d}_output_{c_idx:02d}_{contrib_path.name}"
            chosen_out = _reuse_or_copy(contrib_path, dest_path)
            if chosen_out is None:
                continue
            contrib_records.append({
                "name": contrib.get("name"),
                "path": str(chosen_out),
            })
        if contrib_records:
            record["output_contributions"] = contrib_records
        chosen.append(record)
        copied_paths.append(panel_path)
    if not chosen:
        return
    thumb_path = target_dir / "thumbnail.jpeg"
    thumb_str = _tile_thumbnail(copied_paths, thumb_path) or ""
    primary = chosen[0]
    manifest = {
        "layer": layer,
        "unit": unit,
        "panels": chosen,
        "thumbnail": thumb_str,
        "samples": len(chosen),
        "input_feature_map": primary.get("input_feature_map"),
        "anchor_heatmaps": primary.get("anchor_heatmaps", []),
        "output_contributions": primary.get("output_contributions", []),
    }
    with (target_dir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
