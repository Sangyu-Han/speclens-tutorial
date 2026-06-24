from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import yaml

from src.core.indexing.decile_aggregator import DecileTopKParquet, RunFingerprint
from src.core.indexing.decile_parquet_ledger import DecileParquetLedger
from src.core.indexing.topn_aggregator import TopNAggregator
from src.core.indexing.offline_meta import build_offline_ledger, resolve_offline_part_modulus
from src.core.indexing.registry_utils import ensure_dir, load_obj, sanitize_layer_name


logger = logging.getLogger("sae_index")


@dataclass
class ResumeMeta:
    resumed: bool
    loader_pos: Optional[Dict[str, int]]
    steps: int


def _cp_paths(cp_dir: Path, out_prefix: str, layer_name: Optional[str] = None) -> Path:
    if layer_name is None:
        return cp_dir / f"{out_prefix}.__global__.pt"
    return cp_dir / f"{out_prefix}.{_safe_writer_state_name(layer_name)}.pt"


def _lv_key(lname: str, vname: str) -> str:
    return lname if vname == "default" else f"{lname}@@{vname}"


def _lv_parse(key: str) -> tuple[str, str]:
    if "@@" in key:
        lname, vname = key.split("@@", 1)
        return lname, vname
    return key, "default"


def _safe_writer_state_name(key: str) -> str:
    return sanitize_layer_name(key).replace("@@", "__")


def _safe_variant_name(vname: str) -> str:
    return sanitize_layer_name(vname).replace("@@", "__")


def _parse_step_from_path(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _configured_variant_names(cfg: Dict[str, Any]) -> list[str]:
    explicit = cfg.get("indexing", {}).get("variants")
    if explicit is None:
        explicit = cfg.get("sae", {}).get("training", {}).get("variants")
    if not explicit:
        return []

    names: list[str] = []
    for entry in explicit:
        if isinstance(entry, dict):
            name = str(entry.get("name", "default"))
        else:
            name = str(entry)
        if name not in names:
            names.append(name)
    return names


def _variant_names_for_layer(cfg: Dict[str, Any], sae_root: Path, lname: str) -> list[str]:
    configured = _configured_variant_names(cfg)
    if configured:
        return configured

    layer_dir = sae_root / sanitize_layer_name(lname)
    if not layer_dir.exists():
        return ["default"]

    names: list[str] = []
    if any(layer_dir.glob("*.pt")):
        names.append("default")

    for child in sorted(layer_dir.iterdir()):
        if child.is_dir() and any(child.glob("*.pt")) and child.name not in names:
            names.append(child.name)

    return names or ["default"]


def _sae_ckpt_dir_for(sae_root: Path, lname: str, vname: str) -> Path:
    base = sae_root / sanitize_layer_name(lname)
    return base if vname == "default" else base / vname


def _find_latest_sae_ckpt_path(sae_root: Path, lname: str, vname: str) -> Path | None:
    ckpt_dir = _sae_ckpt_dir_for(sae_root, lname, vname)
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: (_parse_step_from_path(p), p.stat().st_mtime))
    return candidates[-1]


def _variant_out_dir(base_out_dir: Path, vname: str, multi_variant: bool) -> Path:
    if not multi_variant and vname == "default":
        return base_out_dir
    return base_out_dir / "variants" / _safe_variant_name(vname)


def _module_device(module: torch.nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return torch.device("cpu")


def load_writer_state_if_exists(writer, cp_path: Path) -> None:
    if not cp_path.exists():
        return
    try:
        sd = torch.load(cp_path, map_location="cpu")
        if isinstance(sd, dict) and "writer" in sd:
            writer.load_state_dict(sd["writer"])
        else:
            writer.load_state_dict(sd)
    except Exception:
        pass


def read_resume_meta(cp_dir: Path, out_prefix: str, my_layers: List[str], device: torch.device) -> ResumeMeta:
    resumed, pos, steps = False, None, 0

    gcp = _cp_paths(cp_dir, out_prefix, None)
    if gcp.exists():
        try:
            gsd = torch.load(gcp, map_location="cpu")
            pos = gsd.get("loader_pos", None)
            steps = int(gsd.get("global_steps", 0))
            resumed = pos is not None
        except Exception:
            pass

    if not resumed:
        for ln in my_layers:
            lcp = _cp_paths(cp_dir, out_prefix, ln)
            if not lcp.exists():
                continue
            try:
                sd = torch.load(lcp, map_location="cpu")
                inner = sd.get("writer", sd) if isinstance(sd, dict) else {}
                pos = sd.get("loader_pos", None)
                if pos is None:
                    pos = {
                        "epoch": int(inner.get("run_epoch", 0)),
                        "b_in_epoch": int(inner.get("run_b_in_epoch", 0)),
                    }
                steps = int(sd.get("global_steps", inner.get("global_steps", 0)))
                resumed = True
                break
            except Exception:
                continue

    if resumed and pos is not None:
        ep = torch.tensor(
            [pos.get("epoch", 0), pos.get("b_in_epoch", 0)],
            device=device,
            dtype=torch.long,
        )
        if dist.is_initialized():
            dist.all_reduce(ep, op=dist.ReduceOp.MAX)
        pos = {"epoch": int(ep[0].item()), "b_in_epoch": int(ep[1].item())}

    return ResumeMeta(resumed, pos, int(steps))


def fast_forward_if_needed(store, resume: ResumeMeta, rank: int) -> None:
    if resume.resumed and resume.loader_pos:
        store.fast_forward_loader(
            epoch=resume.loader_pos["epoch"],
            b_in_epoch=resume.loader_pos["b_in_epoch"],
        )
        if rank == 0:
            logger.info(
                "[resume] fast-forward dataloader to epoch=%d, b_in_epoch=%d",
                resume.loader_pos["epoch"],
                resume.loader_pos["b_in_epoch"],
            )


def init_progress_with_resume_or_warmup(
    store, resume: ResumeMeta, device: torch.device, local: int, rank: int, pbar: tqdm
) -> Tuple[int, int]:
    if resume.resumed:
        if rank == 0:
            pbar.n = min(resume.steps, pbar.total)
            pbar.refresh()
        return 0, 0

    warm_local = int(store.collect_round(n_batches=1) or 0)
    wmin = torch.tensor([warm_local], device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(wmin, op=dist.ReduceOp.MIN)
    w = int(wmin.item())
    if rank == 0 and w > 0:
        pbar.update(w)
    return w, w


def save_layer_checkpoint(
    out_prefix: str,
    cp_dir: Path,
    ln: str,
    writer,
    loader_pos: Dict[str, int],
    global_steps: int,
) -> None:
    writer.run_epoch = int(loader_pos.get("epoch", 0))
    writer.run_b_in_epoch = int(loader_pos.get("b_in_epoch", 0))
    writer.global_steps = int(global_steps)
    payload = {
        "version": 2,
        "writer": writer.state_dict(),
        "loader_pos": loader_pos,
        "global_steps": int(global_steps),
    }
    cp_path = _cp_paths(cp_dir, out_prefix, ln)
    torch.save(payload, cp_path)


def save_global_checkpoint(
    out_prefix: str,
    cp_dir: Path,
    loader_pos: Dict[str, int],
    global_steps: int,
    rank: int,
) -> None:
    if rank != 0:
        return
    gcp = _cp_paths(cp_dir, out_prefix, None)
    torch.save(
        {"version": 1, "loader_pos": loader_pos, "global_steps": int(global_steps)},
        gcp,
    )


def _make_provenance_key_fn(
    *, prov_cols: Sequence[str], fields: Sequence[str]
) -> Callable[[tuple], tuple]:
    prov_cols = tuple(prov_cols)
    if not prov_cols:
        raise ValueError("Cannot build provenance dedupe key without provenance columns")
    indices: List[int] = []
    for name in fields:
        if name not in prov_cols:
            raise ValueError(
                f"Requested dedupe field '{name}' not found in provenance columns {prov_cols}"
            )
        indices.append(prov_cols.index(name))
    prov_len = len(prov_cols)
    offset = 1

    def _key(item: tuple) -> tuple:
        prov_vals = item[offset : offset + prov_len]
        return tuple(prov_vals[idx] for idx in indices)

    return _key


def ddp_setup() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", rank=rank, world_size=world)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world)
    return rank, world, local


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def sha256_of(path: str | Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return ""


def _write_feature_frequency(
    freq_dir: Path,
    layer_name: str,
    freq_data: Dict[int, int],
    total_samples: int,
) -> None:
    """Write feature frequency metadata to parquet: feature_freq/{layer}.parquet"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    safe_name = sanitize_layer_name(layer_name)
    units = sorted(freq_data.keys())
    if not units:
        return

    total = max(total_samples, 1)
    tbl = pa.table({
        "unit": pa.array(units, type=pa.int32()),
        "num_unique_samples": pa.array(
            [freq_data[u] for u in units], type=pa.int64()
        ),
        "freq_pct": pa.array(
            [100.0 * freq_data[u] / total for u in units], type=pa.float32()
        ),
        "layer": pa.array([layer_name] * len(units), type=pa.string()),
        "total_samples": pa.array([total_samples] * len(units), type=pa.int64()),
    })
    out_path = freq_dir / f"{safe_name}.parquet"
    pq.write_table(tbl, str(out_path), compression="zstd")


def run_indexing(config_path: str, *, l2_warn_threshold: Optional[float] = None) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg_warn = cfg.get("indexing", {}).get("l2_warn_threshold")
    warn_threshold = l2_warn_threshold
    if warn_threshold is None and cfg_warn is not None:
        try:
            warn_threshold = float(cfg_warn)
        except (TypeError, ValueError):
            warn_threshold = None
    if warn_threshold is not None and warn_threshold <= 0.0:
        warn_threshold = None

    part_mod = int(cfg["indexing"].get("partition_modulus", 128))
    offline_part_mod = resolve_offline_part_modulus(cfg)
    logger.info("[index] partition_modulus (M) = %d", part_mod)
    if offline_part_mod != part_mod:
        logger.info("[index] offline_meta.part_modulus = %d", offline_part_mod)

    seed = int(cfg.get("indexing", {}).get("seed", 12345))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    rank, world, local = ddp_setup()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    ds_builder = load_obj(cfg["dataset"]["builder"])
    model_loader = load_obj(cfg["model"]["loader"])
    store_factory = load_obj(cfg["store_cfg"]["factory"])
    collate_builder = load_obj(cfg["dataset"]["collate_builder"])

    out_dir = Path(cfg["indexing"]["out_dir"])
    ensure_dir(out_dir)
    cp_dir = Path(cfg.get("indexing", {}).get("checkpoint_dir", str(out_dir / ".state")))
    ensure_dir(cp_dir)
    run_id = cfg["indexing"].get("run_id") or f"run_{int(time.time())}"

    try:
        logger.info("[index] Loading model …")
        model = model_loader(cfg["model"], device, logger)
        model = DDP(
            model,
            device_ids=[local] if device.type == "cuda" else None,
            output_device=local if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
        logger.info("[index] Model loaded & wrapped in DDP (rank=%d/%d)", rank, world)

        ds, dist_sampler = ds_builder(cfg["dataset"], world_size=world, rank=rank)
        collate_fn = collate_builder(ds, prompt_policy=cfg["dataset"].get("prompt_policy"))
        offline_ledger = build_offline_ledger(cfg)
        store_cfg = dict(cfg.get("store_cfg", {}))
        store_cfg.setdefault("device", str(device))
        store_cfg.setdefault("provenance", {"enabled": True})
        store_cfg.setdefault("sync", {"mode": "owner_stream"})
        store_cfg.setdefault("defaults", {"stride": 8, "stride_start": 0})
        store_cfg["hook_points"] = cfg["sae"]["layers"]
        store_cfg.setdefault("model_batch_size", int(cfg["dataset"]["batch_size"]))

        logger.info("[index] Creating activation store …")
        store = store_factory(
            model=model,
            cfg=store_cfg,
            dataset=ds,
            sampler=dist_sampler,
            collate_fn=collate_fn,
            on_batch_generated=offline_ledger.write_from_batch,
        )
        logger.info("[index] Discovering hook points (forward pass) …")
        store.discover_hook_points()
        logger.info("[index] Hook points discovered: %d", len(store.expanded_hook_points))

        layers_local = sorted(store.expanded_hook_points)
        if dist.is_initialized():
            obj = [layers_local if rank == 0 else []]
            dist.broadcast_object_list(obj, src=0)
            layers = obj[0]
        else:
            layers = layers_local

        owners = {ln: (i % world) for i, ln in enumerate(layers)}
        store.set_layer_owners(owners)
        my_layers = [ln for ln in layers if owners.get(ln, -1) == rank]

        sae_root = Path(cfg["sae"]["output"]["save_path"])
        sae_type_fallback = cfg["sae"].get("sae_type", "batch-topk")
        sae_models: Dict[str, torch.nn.Module] = {}
        dict_sizes: Dict[str, int] = {}
        sae_ckpt_for_writer: Dict[str, str] = {}
        layer_to_variants: Dict[str, List[str]] = {}
        get_act_size = getattr(store, "get_activation_size", None)
        create_sae = load_obj(cfg["sae"]["factory"])
        configured_variants = _configured_variant_names(cfg)
        offload_inactive_variants = bool(
            cfg.get("indexing", {}).get(
                "offload_inactive_variants",
                cfg.get("sae", {}).get("training", {}).get(
                    "offload_inactive_variants", False
                ),
            )
        )

        logger.info("[index] Loading SAE checkpoints for %d layers …", len(my_layers))
        for ln in my_layers:
            loaded_vnames: List[str] = []
            for vname in _variant_names_for_layer(cfg, sae_root, ln):
                path = _find_latest_sae_ckpt_path(sae_root, ln, vname)
                if path is None:
                    if configured_variants:
                        logger.warning(
                            "[index] Missing SAE checkpoint for %s variant=%s under %s",
                            ln,
                            vname,
                            _sae_ckpt_dir_for(sae_root, ln, vname),
                        )
                    continue
                try:
                    pkg = torch.load(path, map_location="cpu")
                except Exception as e:
                    logger.warning(
                        "[index] Failed to load checkpoint %s: %s — skipping %s variant=%s",
                        path, e, ln, vname,
                    )
                    continue
                act_size = int(pkg.get("act_size", 0)) or (
                    get_act_size(ln) if get_act_size else 0
                )
                sae_cfg = pkg.get("sae_config", {}) or {}
                sae_cfg.update({"act_size": act_size, "device": str(device)})
                sae_type = sae_cfg.get("sae_type") or sae_type_fallback
                sae = create_sae(sae_type, sae_cfg)
                try:
                    sae.load_state_dict(pkg.get("sae_state", {}), strict=True)
                except Exception as e:
                    logger.warning(
                        "[index] Failed to load SAE state for %s variant=%s: %s",
                        ln,
                        vname,
                        e,
                    )
                    continue
                sae.eval()
                if offload_inactive_variants:
                    sae.cpu()
                else:
                    sae.to(device)

                lv = _lv_key(ln, vname)
                sae_models[lv] = sae
                dict_sizes[lv] = int(
                    sae_cfg.get(
                        "dict_size",
                        int(sae_cfg.get("expansion_factor", 8)) * act_size,
                    )
                )
                sae_ckpt_for_writer[lv] = str(path)
                loaded_vnames.append(vname)
                logger.info(
                    "[index]   ✓ %s variant=%s (type=%s, act=%d, dict=%d)",
                    ln,
                    vname,
                    sae_type,
                    act_size,
                    dict_sizes[lv],
                )
            if loaded_vnames:
                layer_to_variants[ln] = loaded_vnames
            else:
                logger.warning("[index] No SAE checkpoints loaded for %s — skipping", ln)

        loaded_variants = sorted(
            {vname for vnames in layer_to_variants.values() for vname in vnames}
        )
        multi_variant_layout = bool(
            loaded_variants and (
                len(loaded_variants) > 1 or any(vname != "default" for vname in loaded_variants)
            )
        )
        logger.info(
            "[index] Loaded %d SAE models across %d layers. variants=%s",
            len(sae_models),
            len(layer_to_variants),
            loaded_variants or ["default"],
        )
        if multi_variant_layout:
            logger.info(
                "[index] Multi-variant outputs enabled: parquet roots under %s/variants/<variant>",
                out_dir,
            )

        model_yaml = str(cfg["model"].get("yaml", ""))
        model_ckpt = str(cfg["model"].get("ckpt") or "")
        fp_common = {
            "model_name": cfg["model"].get("name", "model"),
            "model_yaml": model_yaml,
            "model_ckpt": model_ckpt,
            "model_ckpt_sha": sha256_of(model_ckpt) if model_ckpt else "",
            "dataset_name": cfg.get("dataset", {}).get("name", "dataset"),
            "run_id": run_id,
        }

        model_bs = int(cfg["dataset"]["batch_size"])
        per_rank_samples = len(dist_sampler)
        global_batches_total = ceil(per_rank_samples / model_bs)
        max_steps = int(cfg["indexing"].get("max_steps", 0))
        if max_steps > 0:
            global_batches_total = min(global_batches_total, max_steps)
            logger.info("[index] max_steps=%d → capping total batches to %d", max_steps, global_batches_total)
        pbar = tqdm(
            total=global_batches_total,
            disable=(rank != 0),
            dynamic_ncols=True,
            desc="Indexing (global)",
            mininterval=0.2,
            smoothing=0.1,
        )

        global_steps_accum = 0
        cp_acc = 0

        cp_every_steps = int(cfg.get("indexing", {}).get("checkpoint_every_steps", 50))
        writers: Dict[str, Any] = {}  # DecileTopKParquet or TopNAggregator
        variant_output_dirs: Dict[str, Path] = {}
        variant_ledgers: Dict[str, DecileParquetLedger] = {}

        prov_cols = tuple(getattr(store, "prov_cols", ()))
        dedupe_key_fn: Optional[Callable[[tuple], tuple]] = None
        dedupe_cfg = cfg["indexing"].get("dedupe")
        if dedupe_cfg:
            factory_path = dedupe_cfg.get("factory")
            if factory_path:
                dedupe_factory = load_obj(factory_path)
                dedupe_key_fn = dedupe_factory(
                    prov_cols=prov_cols,
                    config=dedupe_cfg,
                )
            else:
                fields = dedupe_cfg.get("fields") or dedupe_cfg.get("provenance_fields")
                if fields:
                    dedupe_key_fn = _make_provenance_key_fn(prov_cols=prov_cols, fields=fields)
        elif cfg["indexing"].get("dedupe_provenance_fields"):
            dedupe_key_fn = _make_provenance_key_fn(
                prov_cols=prov_cols,
                fields=cfg["indexing"]["dedupe_provenance_fields"],
            )

        index_mode = cfg["indexing"].get("mode", "decile")
        track_frequency = cfg["indexing"].get("track_frequency", False)

        def _ledger_for_variant(vname: str) -> DecileParquetLedger:
            if vname not in variant_ledgers:
                root = _variant_out_dir(out_dir, vname, multi_variant_layout)
                ensure_dir(root)
                variant_output_dirs[vname] = root
                variant_ledgers[vname] = DecileParquetLedger(root_dir=root, M_part=part_mod)
            return variant_ledgers[vname]

        # Only create writers for layers/variants that have a loaded SAE model
        for ln in my_layers:
            vnames = layer_to_variants.get(ln, [])
            if not vnames:
                logger.warning("[index] Skipping writer for %s (no SAE loaded)", ln)
                continue
            for vname in vnames:
                lv = _lv_key(ln, vname)
                fp = RunFingerprint(
                    **fp_common,
                    sae_ckpt=sae_ckpt_for_writer.get(lv, ""),
                    sae_ckpt_sha=sha256_of(sae_ckpt_for_writer.get(lv, ""))
                    if sae_ckpt_for_writer.get(lv)
                    else "",
                )
                ledger = _ledger_for_variant(vname)
                if index_mode == "topn":
                    w = TopNAggregator(
                        dict_size=dict_sizes[lv],
                        top_n=int(cfg["indexing"].get("top_n", 300)),
                        layer_name=ln,
                        fp=fp,
                        ledger=ledger,
                        prov_cols=prov_cols,
                        track_frequency=track_frequency,
                        dedupe_key_fn=dedupe_key_fn,
                        slack=int(cfg["indexing"].get("topk_slack", 4)),
                        rank=rank,
                    )
                else:
                    maxima0 = torch.zeros(dict_sizes[lv], dtype=torch.float32, device="cpu")
                    w = DecileTopKParquet(
                        dict_size=dict_sizes[lv],
                        num_deciles=int(cfg["indexing"]["num_deciles"]),
                        k=int(cfg["indexing"]["top_k_per_decile"]),
                        rand_k=int(cfg["indexing"]["random_k_per_feature"]),
                        maxima=maxima0,
                        layer_name=ln,
                        fp=fp,
                        ledger=ledger,
                        prov_cols=prov_cols,
                        dedupe_key_fn=dedupe_key_fn,
                        boundary=cfg["indexing"].get("boundary", "max_range"),
                        fixed_cutoffs=cfg["indexing"].get("global_cutoffs"),
                        slack=int(cfg["indexing"].get("topk_slack", 4)),
                        rank=rank,
                    )
                load_writer_state_if_exists(
                    w,
                    _cp_paths(cp_dir, cfg["indexing"]["out_prefix"], lv),
                )
                writers[lv] = w

        # Narrow my_layers to only those with SAE + writer
        _all_my_layers = list(my_layers)  # before filtering (for queue drain)
        my_layers = [ln for ln in my_layers if layer_to_variants.get(ln)]
        _drain_layers = [ln for ln in _all_my_layers if not layer_to_variants.get(ln)]
        if _drain_layers:
            logger.warning(
                "[index] Layers owned by rank %d but without SAE (will drain queues): %s",
                rank, _drain_layers,
            )
        logger.info("[index] Active layers for rank %d: %s", rank, my_layers)
        my_writer_keys = [
            _lv_key(ln, vname)
            for ln in my_layers
            for vname in layer_to_variants.get(ln, [])
            if _lv_key(ln, vname) in writers
        ]
        if offload_inactive_variants and my_writer_keys:
            logger.info(
                "[index] offload_inactive_variants=True: keeping only one SAE variant on %s at a time",
                device,
            )

        resume = read_resume_meta(cp_dir, cfg["indexing"]["out_prefix"], my_writer_keys, device)
        fast_forward_if_needed(store, resume, rank)

        global_steps_accum = int(resume.steps) if resume.resumed else 0
        cp_acc = global_steps_accum % max(1, cp_every_steps) if resume.resumed else 0

        warm_local, warm_world = init_progress_with_resume_or_warmup(
            store, resume, device, local, rank, pbar
        )
        if not resume.resumed:
            global_steps_accum += int(warm_world)
            cp_acc += int(warm_world)

        gate_k = int(cfg["indexing"].get("collect_gate_k", world))
        stop_collect = False
        drain_pbar = None
        drain_prev_left = 0

        logger.info("[index] Starting main indexing loop (mode=%s, batches=%d, rank=%d)",
                    cfg["indexing"].get("mode", "decile"), global_batches_total, rank)

        active_sae_lv: Optional[str] = None

        def _activate_variant_sae(lv: str) -> torch.nn.Module:
            nonlocal active_sae_lv
            sae = sae_models[lv]
            if not offload_inactive_variants:
                return sae
            if active_sae_lv != lv:
                if active_sae_lv is not None:
                    prev = sae_models.get(active_sae_lv)
                    if prev is not None and _module_device(prev).type != "cpu":
                        prev.cpu()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                if _module_device(sae) != device:
                    sae.to(device)
                active_sae_lv = lv
            return sae

        while True:
            any_data_local = False

            # Drain queues for layers without SAE to prevent OOM
            for ln in _drain_layers:
                store.next_batch_with_provenance(
                    ln, batch_size=999999, allow_partial=True,
                )

            for ln in my_layers:
                layer_cfg = store._cfg_for_layer(ln)
                bs = int(layer_cfg.get("batch_size", 8192))
                acts, prov = store.next_batch_with_provenance(
                    ln,
                    batch_size=bs,
                    allow_partial=stop_collect,
                )
                if acts is None or prov is None:
                    continue

                any_data_local = True
                vnames = layer_to_variants.get(ln, [])
                if not vnames:
                    continue
                provc = prov.detach().cpu().long()
                stride_step = int(layer_cfg.get("stride", 1))
                for vname in vnames:
                    lv = _lv_key(ln, vname)
                    writer = writers.get(lv)
                    if writer is None or lv not in sae_models:
                        continue

                    sae = _activate_variant_sae(lv)
                    recon = None
                    with torch.no_grad():
                        if warn_threshold is not None:
                            forward_out = sae(acts)
                            if isinstance(forward_out, dict):
                                out = forward_out.get("feature_acts")
                                recon = forward_out.get("sae_out")
                                if out is None:
                                    continue
                            else:
                                out = forward_out
                        else:
                            out = sae.encode(acts) if hasattr(sae, "encode") else sae(acts)
                    out_float = out.detach().float()
                    batch_max = out_float.amax(dim=0)
                    outc = out_float.cpu()
                    writer.update(outc, provc, stride_step=stride_step, batch_max=batch_max)
                    if warn_threshold is not None and recon is not None:
                        try:
                            diff = (recon - acts).float()
                            l2_val = float(torch.mean(diff.pow(2)).item())
                            if l2_val > warn_threshold and rank == 0:
                                logger.warning(
                                    "[index] High reconstruction L2 (layer=%s, variant=%s): %.4f > %.4f",
                                    ln,
                                    vname,
                                    l2_val,
                                    warn_threshold,
                                )
                        except Exception as exc:
                            logger.debug(
                                "[index] Failed to compute L2 warning for layer %s variant=%s: %s",
                                ln,
                                vname,
                                exc,
                            )

            starved_local = 0 if any_data_local else 1
            t_starved = torch.tensor([starved_local], device=device, dtype=torch.int)
            if dist.is_initialized():
                dist.all_reduce(t_starved, op=dist.ReduceOp.SUM)
            starved_world = int(t_starved.item())

            steps_added_local = 0
            if not stop_collect and starved_world >= gate_k:
                if dist.is_initialized():
                    dist.barrier(device_ids=[local] if torch.cuda.is_available() else None)
                steps_added_local = int(store.collect_round(n_batches=1) or 0)
                if dist.is_initialized():
                    dist.barrier(device_ids=[local] if torch.cuda.is_available() else None)
            t_steps = torch.tensor([steps_added_local], device=device, dtype=torch.long)
            if dist.is_initialized():
                dist.all_reduce(t_steps, op=dist.ReduceOp.MIN)
            steps_added = int(t_steps.item())
            if rank == 0 and steps_added > 0:
                pbar.update(steps_added)
            cp_acc += steps_added
            global_steps_accum += steps_added

            if cp_every_steps > 0 and cp_acc >= cp_every_steps:
                pos = store.get_loader_position()
                for lv in my_writer_keys:
                    save_layer_checkpoint(
                        cfg["indexing"]["out_prefix"],
                        cp_dir,
                        lv,
                        writers[lv],
                        pos,
                        global_steps_accum,
                    )
                save_global_checkpoint(
                    cfg["indexing"]["out_prefix"],
                    cp_dir,
                    pos,
                    global_steps_accum,
                    rank,
                )
                cp_acc = 0

            if (not stop_collect) and (global_steps_accum >= pbar.total):
                stop_collect = True
                store.set_freeze_owner_layers(True)
                try:
                    store.collect_policy = "owner_only"
                except Exception:
                    pass

            if stop_collect:
                tks_local = 0
                layers_local = 0
                if getattr(store, "_pending_copy_events", None):
                    layers_local = 1
                for _ln in my_layers:
                    not_empty = False
                    q = store.queues.get(_ln)
                    if q is not None and getattr(q, "ntoks", 0) > 0:
                        tks_local += int(getattr(q, "ntoks", 0))
                        not_empty = True
                    if store.enable_provenance and (_ln in store.prov_queues):
                        pq = store.prov_queues[_ln]
                        if getattr(pq, "ntoks", 0) > 0:
                            tks_local += int(getattr(pq, "ntoks", 0))
                            not_empty = True
                    if store.activations.get(_ln):
                        not_empty = True
                    if store.enable_provenance and store._prov_accum.get(_ln):
                        not_empty = True
                    if not_empty:
                        layers_local += 1

                t_tks = torch.tensor([tks_local], device=device, dtype=torch.long)
                t_layers = torch.tensor([layers_local], device=device, dtype=torch.long)
                if dist.is_initialized():
                    dist.all_reduce(t_tks, op=dist.ReduceOp.SUM)
                    dist.all_reduce(t_layers, op=dist.ReduceOp.SUM)
                tks_world = int(t_tks.item())
                layers_world = int(t_layers.item())

                if rank == 0:
                    if drain_pbar is None:
                        drain_mode = "tokens" if tks_world > 0 else "layers"
                        drain_total = tks_world if drain_mode == "tokens" else max(layers_world, 1)
                        drain_prev_left = drain_total
                        drain_pbar = tqdm(
                            total=drain_total,
                            disable=False,
                            dynamic_ncols=True,
                            desc=f"Draining queues ({drain_mode})",
                            mininterval=0.2,
                            smoothing=0.1,
                            position=1,
                            leave=True,
                        )
                    cur_left = (
                        tks_world if drain_pbar.desc.endswith("(tokens)") else layers_world
                    )
                    delta = max(0, drain_prev_left - cur_left)
                    if delta > 0:
                        drain_pbar.update(delta)
                        drain_prev_left = cur_left
                    drain_pbar.set_postfix_str(
                        f"layers_left={layers_world}, tokens_left={tks_world}"
                    )

                def _local_empty() -> bool:
                    if getattr(store, "_pending_copy_events", None):
                        return False
                    for _ln in my_layers:
                        q = store.queues.get(_ln)
                        if q is not None and getattr(q, "ntoks", 0) > 0:
                            return False
                        if store.enable_provenance and (
                            (_ln in store.prov_queues)
                            and getattr(store.prov_queues[_ln], "ntoks", 0) > 0
                        ):
                            return False
                        if store.activations.get(_ln):
                            return False
                        if store.enable_provenance and store._prov_accum.get(_ln):
                            return False
                    return True

                t_empty = torch.tensor([1 if _local_empty() else 0], device=device, dtype=torch.int)
                if dist.is_initialized():
                    dist.all_reduce(t_empty, op=dist.ReduceOp.MIN)
                all_empty = int(t_empty.item()) == 1

                store.set_freeze_owner_layers(True)

                if all_empty:
                    if rank == 0 and drain_pbar is not None:
                        if drain_pbar.n < drain_pbar.total:
                            drain_pbar.update(drain_pbar.total - drain_pbar.n)
                        drain_pbar.close()
                        drain_pbar = None
                    break

        if getattr(store, "sync_mode", "") != "owner_stream" and dist.is_initialized():
            for lv in my_writer_keys:
                device_mx = writers[lv].maxima.to(device, non_blocking=True)
                dist.all_reduce(device_mx, op=dist.ReduceOp.MAX)
                writers[lv].maxima = device_mx.cpu()

        if rank == 0:
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()

        for lv in my_writer_keys:
            writers[lv].finalize_and_write(progress_cb=None)
            cp_path = _cp_paths(cp_dir, cfg["indexing"]["out_prefix"], lv)
            try:
                cp_path.unlink(missing_ok=True)
            except Exception:
                pass

        # Write feature frequency metadata (topn mode only)
        if index_mode == "topn" and track_frequency:
            for lv in my_writer_keys:
                w = writers[lv]
                if not isinstance(w, TopNAggregator):
                    continue
                ln, vname = _lv_parse(lv)
                freq_dir = variant_output_dirs.get(
                    vname,
                    _variant_out_dir(out_dir, vname, multi_variant_layout),
                ) / "feature_freq"
                ensure_dir(freq_dir)
                _write_feature_frequency(
                    freq_dir, ln, w.get_feature_frequencies(),
                    w.get_total_samples_seen(),
                )
                if rank == 0:
                    logger.info(
                        "[index] Feature frequency written for %s variant=%s: "
                        "%d active features / %d total, %d total samples",
                        ln,
                        vname,
                        len(w.get_feature_frequencies()),
                        w.D,
                        w.get_total_samples_seen(),
                    )

        if dist.is_initialized():
            dist.barrier()
    finally:
        cleanup_ddp()


__all__ = [
    "run_indexing",
    "ddp_setup",
    "cleanup_ddp",
    "ResumeMeta",
    "load_writer_state_if_exists",
    "read_resume_meta",
    "fast_forward_if_needed",
    "init_progress_with_resume_or_warmup",
    "save_layer_checkpoint",
    "save_global_checkpoint",
]
