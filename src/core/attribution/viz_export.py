from __future__ import annotations

"""
Lightweight visualization exporter (manifest + overlays) built on the new core runtime.

This mirrors the legacy export behaviour enough to feed Sankey thumbnails:
  - runs forward to grab SAE unit activations and renders a feature-activation overlay
  - optionally runs backward IG/grad to patch_embed and saves an attribution map (not included in manifest)
  - writes manifest.json via write_viz_manifest (thumbnail + panels)

Supported packs: CLIP vision (timm) only for now.
"""

import math
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from contextlib import nullcontext

import torch
import yaml
from PIL import Image
import pyarrow as pa

from src.core.attribution.visualization.cache import write_viz_manifest
from src.core.attribution.visualization.render import render_feature_activation_map
from src.core.attribution.visualization.panels import save_output_contribution_overlay
from src.core.runtime.attribution_runtime import (
    AnchorConfig,
    AttributionRuntime,
    BackwardConfig,
    ForwardConfig,
    RuntimeTarget,
)
from src.core.indexing.offline_meta import build_offline_ledger
from src.core.indexing.registry_utils import sanitize_layer_name
from src.core.indexing.decile_parquet_ledger import DecileParquetLedger
from src.packs.clip.dataset.builders import build_clip_transform
from src.packs.clip.models.adapters import CLIPVisionAdapter
from src.utils.utils import load_obj

try:
    from src.packs.clip.models.libragrad import apply_libragrad, enable_sae_libragrad
except Exception:  # pragma: no cover - optional dependency
    apply_libragrad = None
    enable_sae_libragrad = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_list(val: Optional[Iterable[str]]) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    return [str(v) for v in val]


def _default_cache_root(cfg: Dict[str, Any]) -> Path:
    tree_cfg = cfg.get("tree") or {}
    viz_cfg = tree_cfg.get("visualization") or {}
    root = viz_cfg.get("cache_root") or cfg.get("output_root") or "./outputs/node_viz"
    return Path(root)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _canonical_layer(layer: str) -> str:
    return layer.split("::", 1)[0] if "::" in layer else layer

def _lookup_offline_paths(ledger: Any, sample_ids: Sequence[int]) -> Dict[int, str]:
    if ledger is None:
        return {}
    lookup = getattr(ledger, "lookup", None)
    if lookup is None:
        return {}
    try:
        return lookup(sample_ids)
    except Exception:
        return {}


def _load_sae(index_cfg: Dict[str, Any], layer: str, device: torch.device) -> torch.nn.Module:
    from src.packs.clip.circuit_runtime import _load_sae_for_layer  # reuse tested helper

    sae_cfg = index_cfg.get("sae") or {}
    sae_root = Path(sae_cfg.get("output", {}).get("save_path", "")) if isinstance(sae_cfg, dict) else None
    if sae_root is None or not sae_root:
        # fallback to helper which resolves from factory block
        return _load_sae_for_layer(index_cfg["sae"], layer, device)
    # use the same helper for robustness (handles :: branches)
    return _load_sae_for_layer(index_cfg["sae"], layer, device)


class _LibragradContext:
    """Lightweight context to apply/revert libragrad patches for forward-mode attribution."""

    def __init__(
        self,
        model: torch.nn.Module,
        sae: Optional[torch.nn.Module],
        gamma: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        self.model = model
        self.sae = sae
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self._restore = None

    def __enter__(self):
        if apply_libragrad is None:
            return self
        self._restore = apply_libragrad(self.model, gamma=self.gamma, alpha=self.alpha, beta=self.beta)
        try:
            if enable_sae_libragrad is not None and self.sae is not None:
                enable_sae_libragrad(self.sae)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._restore is not None:
            try:
                self._restore()
            except Exception:
                pass
        return False


def _build_image_batch(cfg_index: Dict[str, Any], image_path: Path) -> Dict[str, Any]:
    dataset_cfg = dict(cfg_index.get("dataset", {}))
    transform = build_clip_transform(dataset_cfg, is_train=False)
    pil = Image.open(image_path).convert("RGB")
    pixel_values = transform(pil).unsqueeze(0)
    sample_id = torch.tensor([0], dtype=torch.long)
    batch = {
        "pixel_values": pixel_values,
        "label": None,
        "sample_id": sample_id,
        "path": [str(image_path)],
    }
    return batch


def _extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    if isinstance(output, dict):
        for key in ("logits", "preds", "output"):
            tensor = output.get(key)
            if torch.is_tensor(tensor):
                return tensor
    raise TypeError(f"Unable to extract logits from output type {type(output)}")


def _reshape_token_map(vec: torch.Tensor) -> torch.Tensor:
    """
    vec: (tokens,) -> (H, W) square grid (best-effort)
    - drop cls token when (tokens-1) is a perfect square (ViT style)
    - otherwise use the largest perfect square <= tokens
    """
    if vec.dim() != 1:
        vec = vec.flatten()
    tokens_total = vec.numel()
    if tokens_total == 0:
        return torch.zeros(1, 1, device=vec.device)

    drop_cls = False
    if tokens_total > 1:
        root_drop = int(math.isqrt(tokens_total - 1))
        if root_drop * root_drop == tokens_total - 1:
            drop_cls = True
    vec_use = vec[1:] if drop_cls else vec
    tokens_use = vec_use.numel()
    root = int(math.isqrt(tokens_use))
    usable = root * root
    if usable == 0:
        return torch.zeros(1, 1, device=vec.device)
    vec_sq = vec_use[:usable]
    return vec_sq.view(root, root)


def _render_feature_map(
    *,
    out_dir: Path,
    sid: int,
    frames: Sequence[torch.Tensor],
    map_2d: torch.Tensor,
    overlay_kwargs: Dict[str, Any],
    file_stub: str,
) -> Path:
    map_stack = map_2d.unsqueeze(0)  # [1, H, W]
    render_feature_activation_map(
        out_dir=out_dir,
        sid=sid,
        frames=frames,
        feature_map=map_stack,
        overlay_kwargs=overlay_kwargs,
        file_stub=file_stub,
    )
    return out_dir / f"sid{sid}_panel__{file_stub}.jpeg"


def _prepare_attr_map_2d(map_2d: torch.Tensor, *, min_abs: float = 0.0) -> torch.Tensor:
    """
    Apply log scaling + masking to attribution maps for visualization.
    """
    m = map_2d.abs()
    if m.numel() == 0:
        return m
    m = torch.log1p(m)
    max_val = float(m.max())
    if max_val > 1e-8:
        m = m / max_val
    if min_abs > 0:
        m = m.masked_fill(m < float(min_abs), 0.0)
    return m


def _reduce_contribution_vector(tensor: torch.Tensor) -> torch.Tensor:
    """
    Collapse a contribution tensor to a 1D token vector (keeps sign).
    """
    data = tensor.detach().cpu()
    while data.dim() > 3 and data.shape[0] == 1:
        data = data.squeeze(0)
    if data.dim() == 4 and data.shape[0] == 1:
        data = data[0]
    if data.dim() == 3 and data.shape[-1] > 1:
        data = data.pow(2).mean(-1).sqrt()
    if data.dim() == 2 and data.shape[0] == 1:
        data = data[0]
    if data.dim() == 2:
        data = data.mean(dim=0)
    if data.dim() != 1:
        data = data.reshape(-1)
    return data


def _choose_contribution_entry(attr_map: Dict[str, torch.Tensor]) -> tuple[Optional[str], Optional[torch.Tensor]]:
    if not attr_map:
        return None, None
    keys = sorted(attr_map.keys(), key=lambda k: 0 if ("patch" in k or "embed" in k or "decoder" in k) else 1)
    for key in keys:
        tensor = attr_map.get(key)
        if torch.is_tensor(tensor):
            return key, tensor
    return None, None


def _topk_signed_values(vec: torch.Tensor, k: int) -> tuple[List[float], List[float]]:
    flat = vec.reshape(-1)
    if flat.numel() == 0:
        return [], []
    k = max(1, min(int(k), flat.numel()))
    pos_vals, _ = torch.topk(flat, k=k, largest=True, sorted=True)
    neg_vals, _ = torch.topk(-flat, k=k, largest=True, sorted=True)
    pos_list = [float(v.item()) for v in pos_vals]
    neg_list = [float(-v.item()) for v in neg_vals]
    return pos_list, neg_list


def _is_class_contribution(key: Optional[str], tensor: Optional[torch.Tensor]) -> bool:
    if tensor is None or not torch.is_tensor(tensor):
        return False
    key_l = (key or "").lower()
    if any(tok in key_l for tok in ("head", "logit", "classifier")):
        return True
    return False


def _reduce_class_contribution_vector(tensor: torch.Tensor) -> torch.Tensor:
    """
    Collapse contribution tensor to a 1D class vector (keeps sign).
    """
    data = tensor.detach().cpu()
    if data.dim() == 0:
        return data.reshape(1)
    if data.dim() == 1:
        return data
    if data.dim() >= 2:
        data = data.reshape(-1, data.shape[-1])
        data = data.sum(dim=0)
    return data.reshape(-1)


def _topk_class_contrib(vec: torch.Tensor, k: int) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    if not torch.is_tensor(vec):
        return [], []
    flat = vec.reshape(-1)
    if flat.numel() == 0:
        return [], []
    k = max(1, min(int(k), flat.numel()))
    pos_vals, pos_idx = torch.topk(flat, k=k, largest=True, sorted=True)
    neg_vals, neg_idx = torch.topk(-flat, k=k, largest=True, sorted=True)
    pos_list = [{"class_index": int(i.item()), "value": float(v.item())} for v, i in zip(pos_vals, pos_idx)]
    neg_list = [{"class_index": int(i.item()), "value": float(-v.item())} for v, i in zip(neg_vals, neg_idx)]
    return pos_list, neg_list


def _collect_class_delta_via_override(
    runtime: AttributionRuntime,
    batch: Dict[str, Any],
    baseline_mode: str,
) -> Optional[torch.Tensor]:
    """
    Compute class-logit deltas by toggling the target latent from a baseline to the
    recorded activation (activation-patch style).
    """
    try:
        target_latent = runtime._collect_target_latent()
        baseline_latent = runtime._build_forward_baseline(target_latent, baseline_mode)
    except Exception:
        return None

    def _run_with_latent(latent: torch.Tensor) -> Optional[torch.Tensor]:
        setter = getattr(runtime.controller, "set_override_all", None)
        if setter is None:
            return None
        runtime.controller.clear_override()
        runtime.controller.prepare_for_forward()
        setter(latent)
        try:
            with torch.no_grad():
                out = runtime.model(batch["pixel_values"])
        finally:
            runtime.controller.clear_override()
        try:
            logits = _extract_logits(out)
        except Exception:
            return None
        return logits.detach().cpu()

    base_logits = _run_with_latent(baseline_latent)
    full_logits = _run_with_latent(target_latent)
    if base_logits is None or full_logits is None:
        return None
    while base_logits.dim() > 2 or full_logits.dim() > 2:
        base_logits = base_logits.mean(dim=0)
        full_logits = full_logits.mean(dim=0)
    if base_logits.dim() == 2 and base_logits.shape[0] == 1:
        base_logits = base_logits[0]
    if full_logits.dim() == 2 and full_logits.shape[0] == 1:
        full_logits = full_logits[0]
    return full_logits - base_logits


def _render_output_contribution_map(
    *,
    out_dir: Path,
    sid: int,
    base_map_2d: torch.Tensor,
    contrib_map_2d: torch.Tensor,
    overlay_kwargs: Dict[str, Any],
    file_stub: str,
) -> Path:
    heat_stack = contrib_map_2d.unsqueeze(0).unsqueeze(0)
    target_tensor = base_map_2d.unsqueeze(0).unsqueeze(0).abs()
    alpha = float(overlay_kwargs.get("output_contribution_alpha", overlay_kwargs.get("alpha", 0.4)))
    cmap = overlay_kwargs.get("output_contribution_cmap", overlay_kwargs.get("mask_cmap", "bwr"))
    save_output_contribution_overlay(
        out_dir=out_dir,
        sid=sid,
        score_suffix="",
        heat_stack=heat_stack,
        target_tensor=target_tensor,
        prompt_points=[],
        overlay_alpha=alpha,
        overlay_cmap=cmap,
        use_abs_overlay=False,
        file_stub=file_stub,
    )
    return out_dir / f"sid{sid}_panel__{file_stub}.jpeg"


@dataclass
class VizSample:
    image_path: Path
    sample_id: int
    run_id: str
    caption: str = ""
    score: float = 0.0
    source: str = "manual"
    bucket: Optional[str] = None  # optional grouping for external instances


def _collect_ledger_samples(
    *,
    cfg_index: Dict[str, Any],
    target_layer: str,
    unit: int,
    max_deciles: Optional[int],
    max_rows_per_decile: Optional[int],
    topn_per_decile: int,
    offline_ledger: Any,
) -> List[VizSample]:
    """
    Fetch top rows from decile parquet ledger for a layer/unit and convert to VizSample list.
    """
    ledger_root = Path(cfg_index.get("indexing", {}).get("out_dir", ""))
    ledger = DecileParquetLedger(ledger_root)
    layer_key = _canonical_layer(target_layer)
    num_deciles = int(cfg_index.get("indexing", {}).get("num_deciles", 10))
    limit_deciles = min(num_deciles, int(max_deciles)) if max_deciles else num_deciles
    limit_rows = max_rows_per_decile if max_rows_per_decile is not None else topn_per_decile

    rows: List[dict] = []
    processed_deciles = 0
    for dec in range(num_deciles):
        if processed_deciles >= limit_deciles:
            break
        tbl: pa.Table = ledger.topn_for(layer=layer_key, unit=int(unit), decile=dec, n=topn_per_decile)
        if tbl is None or tbl.num_rows == 0:
            continue
        processed_deciles += 1
        count = 0
        for idx in range(tbl.num_rows):
            if limit_rows is not None and count >= limit_rows:
                break
            sample_id = int(tbl.column("sample_id")[idx].as_py())
            score = float(tbl.column("score")[idx].as_py())
            rank = int(tbl.column("rank_in_decile")[idx].as_py())
            rows.append({"decile": dec, "rank": rank, "sample_id": sample_id, "score": score})
            count += 1
            if len(rows) >= limit_deciles * topn_per_decile:
                break

    if not rows:
        return []

    # dedupe by sample_id, keep first occurrence order
    deduped: List[dict] = []
    seen: set[int] = set()
    for r in rows:
        sid = r["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(r)

    # resolve paths via offline meta
    path_map = _lookup_offline_paths(offline_ledger, [r["sample_id"] for r in deduped])

    samples: List[VizSample] = []
    for r in deduped:
        sid = r["sample_id"]
        img_path = path_map.get(sid)
        if not img_path:
            continue
        caption = f"decile {r['decile']} rank {r['rank']}"
        samples.append(
            VizSample(
                image_path=Path(img_path),
                sample_id=sid,
                run_id=f"decile_{r['decile']:02d}_rank_{r['rank']:03d}",
                caption=caption,
                score=r.get("score", 0.0),
                source="ledger",
            )
        )
    return samples


# ---------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------


_VIZ_MODEL_CACHE: Dict[tuple[str, str], tuple[torch.nn.Module, CLIPVisionAdapter]] = {}


def _get_cached_model(index_cfg: Dict[str, Any], *, device: torch.device) -> tuple[torch.nn.Module, CLIPVisionAdapter]:
    """
    Avoid re-loading the CLIP model + adapter for every layer when auto-viz
    needs to export multiple manifests in a single run.
    """
    key = (str(index_cfg), str(device))
    if key in _VIZ_MODEL_CACHE:
        return _VIZ_MODEL_CACHE[key]
    model_loader = load_obj(index_cfg["model"]["loader"])
    model = model_loader(index_cfg["model"], device=device).eval()
    adapter = CLIPVisionAdapter(model, device=device)
    _VIZ_MODEL_CACHE[key] = (model, adapter)
    return model, adapter


def export_viz_for_units(
    *,
    config_path: str | Path,
    units: Sequence[int],
    layer: Optional[str] = None,
    image_paths: Optional[Sequence[str]] = None,
    sample_ids: Optional[Sequence[int]] = None,
    cache_root: Optional[str | Path] = None,
    viz_max_samples: int = 4,
    contribution_methods: Optional[Sequence[str]] = None,
    contribution_steps: int = 32,
    contribution_baseline: str = "zeros",
    contribution_topk: int = 5,
    contribution_target_classes: Optional[Sequence[int]] = None,
    unit_dirnames: Optional[Dict[int, str]] = None,
    save_contribution_json: bool = True,
    contribution_captures: Optional[Sequence[str]] = None,
    contribution_ig_active: Optional[Sequence[str]] = None,
    contribution_stop_grad: Optional[Sequence[str]] = None,
    max_deciles: Optional[int] = None,
    max_rows_per_decile: Optional[int] = None,
    topn_per_decile: Optional[int] = None,
    use_libragrad: bool = False,
    libragrad_gamma: Optional[float] = None,
    libragrad_alpha: Optional[float] = None,
    libragrad_beta: Optional[float] = None,
) -> Dict[int, Path]:
    """
    Generate SAE activation overlays + manifests for the given units.
    - Uses a viz/attribution config (similar to sam2_attr_index_v2.yaml).
    - image_paths: explicit images to run; if absent, tries sample_ids; otherwise, pulls top-k decile rows from the ledger.
    - contribution_target_classes: optional list of class indices to always record (even if not in top-k).
    """
    cfg = _load_yaml(Path(config_path))
    viz_indexing = cfg.get("indexing") or {}
    index_cfg_path = viz_indexing.get("config")
    if not index_cfg_path:
        raise ValueError("indexing config path is required.")
    cfg_index = _load_yaml(Path(index_cfg_path))

    device = torch.device(cfg_index.get("model", {}).get("device", "cuda"))
    model, adapter = _get_cached_model(cfg_index, device=device)
    libragrad_restore = None
    if use_libragrad and apply_libragrad is not None:
        try:
            libragrad_restore = apply_libragrad(
                model,
                gamma=libragrad_gamma,
                alpha=libragrad_alpha,
                beta=libragrad_beta,
            )
        except Exception:
            libragrad_restore = None

    target_layer = layer or viz_indexing.get("layer")
    if not target_layer:
        raise ValueError("target layer is required (provide via config.indexing.layer or --layer).")
    base_unit = int(viz_indexing.get("unit", 0))

    sae = _load_sae(cfg_index, target_layer, device)
    if use_libragrad and enable_sae_libragrad is not None:
        try:
            enable_sae_libragrad(sae)
        except Exception:
            pass
    offline_ledger = build_offline_ledger(cfg_index)

    # anchor (backward) settings
    backward_block = cfg.get("backward", {})
    backward_anchors = backward_block.get("backward_anchors", backward_block.get("anchors", {}))
    anchor_cfg = AnchorConfig(
        capture=_ensure_list((backward_anchors or {}).get("module")),
        ig_active=_ensure_list((backward_anchors or {}).get("ig_active")),
        stop_grad=_ensure_list((backward_anchors or {}).get("stop_grad")),
    )
    backward_cfg = BackwardConfig(
        enabled=True,
        method=backward_block.get("method", "ig"),
        ig_steps=int(backward_block.get("ig_steps", 16)),
        baseline=str(backward_block.get("baseline", "zeros")),
    )
    target_override_mode = backward_block.get("override_mode", "all_tokens")
    objective_aggregation = backward_block.get("objective_aggregation", "sum")

    # overlay defaults
    heat_cfg = cfg.get("heatmap", {})
    overlay_kwargs = {
        "alpha": float(heat_cfg.get("overlay_alpha", 0.35)),  # slightly more transparent
        "feature_map_alpha": float(heat_cfg.get("feature_map_alpha", heat_cfg.get("overlay_alpha", 0.35))),
        "feature_map_cmap": heat_cfg.get("feature_map_cmap", heat_cfg.get("overlay_cmap", "plasma")),
        "feature_map_min_abs": float(heat_cfg.get("feature_map_min_abs", 0.02)),
        "attr_map_cmap": heat_cfg.get("attr_map_cmap", "viridis"),
        "attr_map_alpha": float(heat_cfg.get("attr_map_alpha", heat_cfg.get("overlay_alpha", 0.35))),
        "attr_map_min_abs": float(heat_cfg.get("attr_map_min_abs", 0.0)),
    }
    contrib_cfg = cfg.get("contribution") or {}
    contrib_methods_cfg = _ensure_list(contrib_cfg.get("methods"))
    if contribution_methods is None:
        contrib_methods = [str(m).strip() for m in contrib_methods_cfg]
        if not contrib_methods and contrib_cfg.get("enabled", True) is not False:
            contrib_methods = ["ig"]
    else:
        contrib_methods = [str(m).strip() for m in contribution_methods]
    contrib_methods = [m for m in contrib_methods if m]
    contrib_steps = int(contrib_cfg.get("ig_steps", contribution_steps))
    contrib_baseline = str(contrib_cfg.get("baseline", contribution_baseline))
    contrib_topk = int(contrib_cfg.get("topk", contribution_topk))
    save_contrib_json = bool(save_contribution_json)
    contrib_anchors_cfg = contrib_cfg.get("anchors") or contrib_cfg.get("forward_anchors") or contrib_cfg.get("anchor") or {}
    contrib_capture = _ensure_list(contribution_captures or contrib_anchors_cfg.get("module") or contrib_anchors_cfg.get("capture"))
    contrib_ig_active = _ensure_list(contribution_ig_active or contrib_anchors_cfg.get("ig_active"))
    contrib_stop_grad = _ensure_list(contribution_stop_grad or contrib_anchors_cfg.get("stop_grad"))
    contrib_anchor_cfg = AnchorConfig(
        capture=contrib_capture or anchor_cfg.capture,
        ig_active=contrib_ig_active or anchor_cfg.ig_active,
        stop_grad=contrib_stop_grad or anchor_cfg.stop_grad,
    )

    topk_decile = int(viz_indexing.get("topn_per_decile") or cfg_index.get("indexing", {}).get("top_k_per_decile", 5))
    if topn_per_decile is not None:
        topk_decile = int(topn_per_decile)
    max_deciles_cfg = viz_indexing.get("max_deciles")
    max_rows_per_decile_cfg = viz_indexing.get("max_rows_per_decile")
    max_deciles_val = max_deciles if max_deciles is not None else max_deciles_cfg
    max_rows_val = max_rows_per_decile if max_rows_per_decile is not None else max_rows_per_decile_cfg

    cache_root_path = Path(cache_root) if cache_root else _default_cache_root(cfg)
    cache_root_path.mkdir(parents=True, exist_ok=True)

    manifests: Dict[int, Path] = {}
    unit_list = list(dict.fromkeys(int(u) for u in (units or [base_unit])))

    def _unit_dir(unit: int) -> str:
        if unit_dirnames and unit in unit_dirnames:
            return sanitize_layer_name(unit_dirnames[unit])
        return f"unit_{unit}"

    target_class_list = [int(c) for c in (contribution_target_classes or [])]

    for unit in unit_list:
        decile_samples = _collect_ledger_samples(
            cfg_index=cfg_index,
            target_layer=target_layer,
            unit=unit,
            max_deciles=max_deciles_val,
            max_rows_per_decile=max_rows_val,
            topn_per_decile=topk_decile,
            offline_ledger=offline_ledger,
        )
        manual_samples: List[VizSample] = []
        if image_paths:
            for idx, path in enumerate(image_paths):
                slug = Path(path).stem.replace(" ", "_")
                manual_samples.append(
                    VizSample(
                        image_path=Path(path).expanduser(),
                        sample_id=idx,
                        run_id=f"img_{idx:03d}",
                        caption=Path(path).name,
                        source="external_image",
                        bucket=slug,
                    )
                )
        if sample_ids:
            path_map = _lookup_offline_paths(offline_ledger, sample_ids)
            for sid in sample_ids:
                p = path_map.get(int(sid))
                if not p:
                    continue
                manual_samples.append(
                    VizSample(
                        image_path=Path(p),
                        sample_id=int(sid),
                        run_id=f"sid_{int(sid)}",
                        caption=f"sample {sid}",
                        source="sample_id",
                    )
                )

        decile_keep: List[VizSample] = []
        target_decile = max(1, viz_max_samples)
        for sample in decile_samples:
            if not sample.image_path.exists():
                continue
            decile_keep.append(sample)
            if len(decile_keep) >= target_decile:
                break
        manual_keep = manual_samples[: max(0, viz_max_samples)]
        samples: List[VizSample] = decile_keep + manual_keep

        entries: List[dict] = []
        contribution_stats: Dict[str, List[dict]] = defaultdict(list)
        class_contrib_records: Dict[str, List[dict]] = defaultdict(list)
        for sample in samples: # 이거 나중에 batchfy할것. 한번에 할 수 있음..
            if not sample.image_path.exists():
                continue
            try:
                batch_cpu = _build_image_batch(cfg_index, sample.image_path)
            except Exception:
                continue
            batch = adapter.preprocess_input(batch_cpu)
            frame_for_render = batch["pixel_values"].detach().cpu()

            def _forward():
                adapter.forward(batch)

            runtime = AttributionRuntime(
                model=model,
                adapter=adapter,
                forward_fn=_forward,
                sae_module=sae,
                target=RuntimeTarget(
                    layer=target_layer,
                    unit=unit,
                    override_mode=target_override_mode,
                    objective_aggregation=objective_aggregation,
                ),
            )
            runtime.configure_anchors(anchor_cfg)

            # forward pass -> feature activation map
            runtime._run_forward(require_grad=False)
            latent = runtime.controller.post_stack(detach=True)
            if latent is None or latent.dim() < 2:
                continue
            act_vec = latent[0].reshape(-1, latent.shape[-1])[:, int(unit)].flatten()
            map_2d = _reshape_token_map(act_vec)

            base_dir = cache_root_path / "_runs" / sanitize_layer_name(target_layer) / f"unit_{unit}"
            if sample.source == "external_image" and sample.bucket:
                base_dir = cache_root_path / "_external_runs" / sample.bucket / sanitize_layer_name(target_layer) / f"unit_{unit}"
            base_dir = base_dir.parent / _unit_dir(unit)
            out_dir = base_dir / sample.run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            feature_path = _render_feature_map(
                out_dir=out_dir,
                sid=sample.sample_id,
                frames=[frame_for_render],
                map_2d=map_2d,
                overlay_kwargs=overlay_kwargs,
                file_stub="feature_map",
            )

            # backward attr map to patch_embed (optional, not included in manifest)
            try:
                attr_map = runtime.run_backward(backward_cfg) or {}
                patch_key = None
                for k in attr_map.keys():
                    if "patch_embed" in k:
                        patch_key = k
                        break
                if patch_key: # 여기 너무 clip 특화임. 추후 여러 모델에 적용할때 일반화할것.. 일단 tensor의 첫 차원은 무조건 step 차원이니 collapse. 근데 모듈마다 이거 설정이 다를텐데.. 어쩐다? 가령 samv2 이미지 인코더는 (steps,frames,C,H,W임임)
                    tensor = attr_map[patch_key]
                    if tensor.dim() > 3:
                        # collapse steps/batch dims
                        tensor = tensor.sum(dim=0) # 원래라면 프레임마다 개별 표시 해줘야함. 근데 clip이니까 그냥 넘어가자. 
                    if tensor.dim() == 3: # (batch, token_num, dim)
                        tensor = tensor.pow(2).mean(-1).sqrt()
                    elif tensor.dim() == 2:
                        pass
                else:
                    tensor = tensor.flatten()
                attr_vec = tensor.flatten()
                attr_map_2d = _reshape_token_map(attr_vec)
                attr_map_2d = _prepare_attr_map_2d(
                    attr_map_2d,
                    min_abs=overlay_kwargs.get("attr_map_min_abs", 0.0),
                )
                attr_cmap = overlay_kwargs.get("attr_map_cmap", "viridis")
                attr_alpha = overlay_kwargs.get("attr_map_alpha", overlay_kwargs.get("feature_map_alpha", 0.4))
                _render_feature_map(
                    out_dir=out_dir,
                    sid=sample.sample_id,
                        frames=[frame_for_render],
                        map_2d=attr_map_2d,
                        overlay_kwargs=dict(
                            overlay_kwargs,
                            feature_map_cmap=attr_cmap,
                            feature_map_alpha=attr_alpha,
                        ),
                        file_stub="attr_patch_embed",
                    )
            except Exception:
                pass

            contrib_stats_for_entry: Dict[str, Any] = {}
            if contrib_methods and map_2d.numel() > 0:
                try:
                    runtime.anchor_capture.clear()
                except Exception:
                    pass
                runtime.configure_anchors(contrib_anchor_cfg)
                try:
                    runtime._ig_active = set(contrib_anchor_cfg.ig_active or ())
                    runtime.anchor_capture.ig_set_active(runtime._ig_active)
                except Exception:
                    pass
                for method in contrib_methods:
                    method = str(method).strip()
                    if not method:
                        continue
                    contrib_key = None
                    contrib_tensor: Optional[torch.Tensor] = None
                    class_mode = False

                    use_libragrad = method.startswith("libragrad_")
                    method_key = method.split("libragrad_", 1)[1] if use_libragrad else method

                    if method_key == "activation_patch_delta":
                        contrib_tensor = _collect_class_delta_via_override(runtime, batch, contrib_baseline)
                        contrib_key = "logits"
                        class_mode = True
                    else:
                        forward_cfg = ForwardConfig(
                            enabled=True,
                            method=method_key,
                            ig_steps=contrib_steps,
                            baseline=contrib_baseline,
                        )
                        ctx_mgr = (
                            _LibragradContext(
                                model,
                                sae,
                                gamma=libragrad_gamma,
                                alpha=libragrad_alpha,
                                beta=libragrad_beta,
                            )
                            if use_libragrad
                            else nullcontext()
                        )
                        try:
                            with ctx_mgr:
                                contrib_out = runtime.run_forward_contribution(forward_cfg) or {}
                        except Exception:
                            continue
                        attr_payload = contrib_out.get("attr") or {}
                        contrib_key, contrib_tensor = _choose_contribution_entry(attr_payload)
                        class_mode = _is_class_contribution(contrib_key, contrib_tensor)

                    if contrib_tensor is None:
                        continue

                    if class_mode:
                        contrib_vec = _reduce_class_contribution_vector(contrib_tensor)
                        pos_topk, neg_topk = _topk_class_contrib(contrib_vec, contrib_topk)
                        pos_vals = [c["value"] for c in pos_topk]
                        neg_vals = [c["value"] for c in neg_topk]
                        contrib_stats_for_entry[method] = {
                            "anchor": contrib_key,
                            "pos_topk": pos_topk,
                            "neg_topk": neg_topk,
                            "pos_mean": float(sum(pos_vals) / len(pos_vals)) if pos_vals else 0.0,
                            "neg_mean": float(sum(neg_vals) / len(neg_vals)) if neg_vals else 0.0,
                            "mode": "class",
                        }
                        if target_class_list:
                            target_vals: List[Dict[str, float]] = []
                            for cls_idx in target_class_list:
                                cls_int = int(cls_idx)
                                if cls_int < 0 or cls_int >= contrib_vec.numel():
                                    continue
                                target_vals.append({"class_index": cls_int, "value": float(contrib_vec[cls_int].item())})
                            if target_vals:
                                class_contrib_records[method].append(
                                    {
                                        "sample_id": sample.sample_id,
                                        "run_id": sample.run_id,
                                        "image": str(sample.image_path),
                                        "anchor": contrib_key,
                                        "classes": target_vals,
                                        "mode": "class",
                                    }
                                )
                    else:
                        contrib_vec = _reduce_contribution_vector(contrib_tensor)
                        contrib_map_2d = _reshape_token_map(contrib_vec)
                        pos_topk, neg_topk = _topk_signed_values(contrib_map_2d, contrib_topk)
                        contrib_stats_for_entry[method] = {
                            "anchor": contrib_key,
                            "pos_topk": pos_topk,
                            "neg_topk": neg_topk,
                            "pos_mean": float(sum(pos_topk) / len(pos_topk)) if pos_topk else 0.0,
                            "neg_mean": float(sum(neg_topk) / len(neg_topk)) if neg_topk else 0.0,
                            "mode": "token",
                        }

                    contribution_stats[method].append(
                        {
                            "sample_id": sample.sample_id,
                            "run_id": sample.run_id,
                            "image": str(sample.image_path),
                            **contrib_stats_for_entry[method],
                        }
                    )

            entry_score = sample.score
            if (entry_score is None or entry_score == 0.0) and act_vec.numel() > 0:
                entry_score = float(act_vec.abs().max().item())
            entry = {
                "path": str(feature_path),
                "caption": sample.caption or sample.image_path.name,
                "score": entry_score,
                "sample_id": sample.sample_id,
                "input_feature_map": str(feature_path),
                "source": sample.source,
            }
            if contrib_stats_for_entry:
                entry["output_contribution_stats"] = contrib_stats_for_entry
            entries.append(entry)

            try:
                runtime.cleanup()
            except Exception:
                pass

        if entries:
            write_viz_manifest(
                layer=target_layer,
                unit=unit,
                cache_root=cache_root_path,
                entries=entries,
                max_samples=max(len(entries), int(max(1, viz_max_samples))),
            )
            manifest_path = cache_root_path / sanitize_layer_name(target_layer) / _unit_dir(unit) / "manifest.json"
            manifests[unit] = manifest_path
        if contribution_stats and save_contrib_json:
            summary = {
                "layer": target_layer,
                "unit": unit,
                "topk": contrib_topk,
                "methods": {},
            }

            def _aggregate_class_means(stats_list: List[dict], key: str, k: int, reverse: bool) -> List[Dict[str, float]]:
                accum: Dict[int, List[float]] = {}
                for rec in stats_list:
                    entries = rec.get(key, [])
                    if not isinstance(entries, (list, tuple)):
                        continue
                    for item in entries:
                        if not isinstance(item, dict):
                            continue
                        cls_idx = item.get("class_index", item.get("class", item.get("idx", None)))
                        if cls_idx is None:
                            continue
                        cls_int = int(cls_idx)
                        accum.setdefault(cls_int, []).append(float(item.get("value", 0.0)))
                aggregated: List[Dict[str, float]] = [
                    {"class_index": cls, "value": float(sum(vals) / len(vals))}
                    for cls, vals in accum.items()
                    if vals
                ]
                aggregated.sort(key=lambda x: x["value"], reverse=reverse)
                return aggregated[: max(1, k)] if aggregated else []

            for method, stats in contribution_stats.items():
                pos_means_scalar = [float(s.get("pos_mean", 0.0)) for s in stats if isinstance(s, dict)]
                neg_means_scalar = [float(s.get("neg_mean", 0.0)) for s in stats if isinstance(s, dict)]
                mean_pos_classes = _aggregate_class_means(stats, "pos_topk", contrib_topk, True)
                mean_neg_classes = _aggregate_class_means(stats, "neg_topk", contrib_topk, False)
                summary["methods"][method] = {
                    "per_sample": stats,
                    "mean_pos_topk": mean_pos_classes if mean_pos_classes else (float(sum(pos_means_scalar) / len(pos_means_scalar)) if pos_means_scalar else 0.0),
                    "mean_neg_topk": mean_neg_classes if mean_neg_classes else (float(sum(neg_means_scalar) / len(neg_means_scalar)) if neg_means_scalar else 0.0),
                }
            summary_path = cache_root_path / sanitize_layer_name(target_layer) / _unit_dir(unit) / "output_contribution_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2, ensure_ascii=False)

        if class_contrib_records and save_contrib_json and target_class_list:
            class_summary = {
                "layer": target_layer,
                "unit": unit,
                "classes": target_class_list,
                "methods": {method: {"per_sample": stats} for method, stats in class_contrib_records.items()},
            }
            class_summary_path = cache_root_path / sanitize_layer_name(target_layer) / _unit_dir(unit) / "output_contribution_class_scores.json"
            class_summary_path.parent.mkdir(parents=True, exist_ok=True)
            with class_summary_path.open("w", encoding="utf-8") as fp:
                json.dump(class_summary, fp, indent=2, ensure_ascii=False)

    if libragrad_restore is not None:
        try:
            libragrad_restore()
        except Exception:
            pass
    return manifests


__all__ = ["export_viz_for_units"]
