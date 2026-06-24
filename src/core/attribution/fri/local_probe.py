from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def rank_score(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(-arr, kind="mergesort")
    out = np.empty_like(arr, dtype=np.float32)
    out[order] = np.linspace(1.0, 0.0, arr.size, endpoint=False, dtype=np.float32)
    return out


def pos01(scores: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1)), 0.0)
    mx = float(arr.max()) if arr.size else 0.0
    return (arr / mx).astype(np.float32) if mx > 1e-12 else np.zeros_like(arr)


def deletion_drops(
    *,
    full_value: float,
    deleted_values: np.ndarray,
    mode: str = "relative",
    eps: float = 1e-8,
) -> np.ndarray:
    vals = np.asarray(deleted_values, dtype=np.float32).reshape(-1)
    if mode == "relative":
        denom = max(abs(float(full_value)), float(eps))
        drops = (float(full_value) - vals) / denom
    elif mode == "absolute":
        drops = float(full_value) - vals
    else:
        raise ValueError(f"Unknown deletion drop mode: {mode!r}")
    return np.maximum(np.nan_to_num(drops, nan=0.0, posinf=0.0, neginf=0.0), 0.0).astype(np.float32)


@dataclass(frozen=True)
class LocalProbeConfig:
    probe_calls: int = 32
    radii: tuple[int, ...] = (1, 2, 1, 3)
    coverage_radius: int = 0
    low_drop: float = 0.015
    mid_drop: float = 0.035
    sparse_drop_mean: float = 0.004


@dataclass(frozen=True)
class LocalProbeMasks:
    masks: np.ndarray
    groups: tuple[np.ndarray, ...]
    centers: tuple[int, ...]
    radii: tuple[int, ...]


@dataclass(frozen=True)
class LocalProbeReadouts:
    scores: dict[str, np.ndarray]
    diagnostics: dict[str, float | int | list[int]]


def make_local_probe_masks(
    prior: np.ndarray,
    *,
    n_patches: int,
    grid_size: int,
    config: LocalProbeConfig,
) -> LocalProbeMasks:
    prior_arr = np.asarray(prior, dtype=np.float32).reshape(-1)
    if prior_arr.size != int(n_patches):
        raise ValueError(f"prior has {prior_arr.size} patches, expected {int(n_patches)}")

    order = np.argsort(-prior_arr, kind="mergesort")
    centers: list[int] = []
    occupied = np.zeros((int(grid_size), int(grid_size)), dtype=bool)
    cr = max(0, int(config.coverage_radius))
    for idx in order:
        y, x = divmod(int(idx), int(grid_size))
        y0, y1 = max(0, y - cr), min(int(grid_size), y + cr + 1)
        x0, x1 = max(0, x - cr), min(int(grid_size), x + cr + 1)
        if occupied[y0:y1, x0:x1].all():
            continue
        centers.append(int(idx))
        occupied[y0:y1, x0:x1] = True
        if len(centers) >= int(config.probe_calls):
            break

    radius_pattern = tuple(int(r) for r in config.radii) or (1, 2, 1, 3)
    masks = np.ones((len(centers), int(n_patches)), dtype=np.float32)
    groups: list[np.ndarray] = []
    for i, idx in enumerate(centers):
        y, x = divmod(int(idx), int(grid_size))
        radius = max(0, int(radius_pattern[i % len(radius_pattern)]))
        y0, y1 = max(0, y - radius), min(int(grid_size), y + radius + 1)
        x0, x1 = max(0, x - radius), min(int(grid_size), x + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        group = (yy.reshape(-1) * int(grid_size) + xx.reshape(-1)).astype(np.int64)
        groups.append(group)
        masks[i, group] = 0.0

    return LocalProbeMasks(
        masks=masks,
        groups=tuple(groups),
        centers=tuple(centers),
        radii=radius_pattern,
    )


def score_local_probe_readouts(
    *,
    prior: np.ndarray,
    drops: np.ndarray,
    groups: Sequence[np.ndarray],
    content_scores: np.ndarray | None = None,
) -> LocalProbeReadouts:
    prior_arr = np.asarray(prior, dtype=np.float32).reshape(-1)
    drops_arr = np.maximum(
        np.nan_to_num(np.asarray(drops, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    n_patches = prior_arr.size
    credit = np.zeros(n_patches, dtype=np.float64)
    cover = np.zeros(n_patches, dtype=np.float64)
    max_credit = np.zeros(n_patches, dtype=np.float64)
    for drop, group in zip(drops_arr, groups):
        group_arr = np.asarray(group, dtype=np.int64).reshape(-1)
        if group_arr.size == 0:
            continue
        val = float(drop) / np.sqrt(float(group_arr.size))
        credit[group_arr] += val
        cover[group_arr] += 1.0
        max_credit[group_arr] = np.maximum(max_credit[group_arr], val)

    mean_credit = np.where(cover > 0, credit / np.maximum(cover, 1.0), 0.0).astype(np.float32)
    max_credit = max_credit.astype(np.float32)
    local_rank = (0.60 * rank_score(max_credit) + 0.40 * rank_score(mean_credit)).astype(np.float32)
    prior_rank = rank_score(prior_arr)
    prior_mag = pos01(prior_arr)
    local_x_prior = np.sqrt(pos01(local_rank) * prior_mag).astype(np.float32)
    readouts = {
        "local_mean": mean_credit,
        "local_max": max_credit,
        "local_rank": local_rank,
        "local_blend": (0.65 * local_rank + 0.35 * prior_rank).astype(np.float32),
        "local_gate": (prior_mag * np.power(0.10 + 0.90 * local_rank, 1.5)).astype(np.float32),
        "local_gate_g2": (prior_mag * np.power(0.10 + 0.90 * local_rank, 2.0)).astype(np.float32),
        "local_gate_g3": (prior_mag * np.power(0.10 + 0.90 * local_rank, 3.0)).astype(np.float32),
        "local_x_prior": local_x_prior,
    }
    if content_scores is not None:
        content_rank = rank_score(np.asarray(content_scores, dtype=np.float32).reshape(-1))
        content_gate = np.power(0.10 + 0.90 * content_rank, 1.5).astype(np.float32)
        content_gate_g2 = np.power(0.10 + 0.90 * content_rank, 2.5).astype(np.float32)
        readouts["content_rank"] = content_rank
        readouts["local_x_prior_content"] = (local_x_prior * content_gate).astype(np.float32)
        readouts["local_x_prior_content_g2"] = (local_x_prior * content_gate_g2).astype(np.float32)
        readouts["local_gate_g2_content"] = (readouts["local_gate_g2"] * content_gate).astype(np.float32)
        readouts["local_gate_g2_content_g2"] = (readouts["local_gate_g2"] * content_gate_g2).astype(np.float32)

    diagnostics = {
        "drop_mean": float(drops_arr.mean()) if drops_arr.size else 0.0,
        "drop_max": float(drops_arr.max()) if drops_arr.size else 0.0,
    }
    return LocalProbeReadouts(
        scores={k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in readouts.items()},
        diagnostics=diagnostics,
    )


def local_probe_readout_from_values(
    *,
    prior: np.ndarray,
    full_value: float,
    deleted_values: np.ndarray,
    groups: Sequence[np.ndarray],
    fallback_scores: Mapping[str, np.ndarray],
    readout: str = "adaptive_sparse_content",
    config: LocalProbeConfig | None = None,
    drop_mode: str = "relative",
    content_scores: np.ndarray | None = None,
    low_fallback: str = "final",
    mid_fallback: str = "soft",
) -> tuple[np.ndarray, LocalProbeReadouts]:
    cfg = config or LocalProbeConfig()
    drops = deletion_drops(
        full_value=float(full_value),
        deleted_values=np.asarray(deleted_values, dtype=np.float32),
        mode=str(drop_mode),
    )
    readouts = score_local_probe_readouts(
        prior=prior,
        drops=drops,
        groups=groups,
        content_scores=content_scores,
    )
    selected = select_local_probe_readout(
        readout=str(readout),
        local_scores=readouts.scores,
        fallback_scores=fallback_scores,
        diagnostics=readouts.diagnostics,
        config=cfg,
        low_fallback=str(low_fallback),
        mid_fallback=str(mid_fallback),
    )
    return selected.astype(np.float32), readouts


def select_local_probe_readout(
    *,
    readout: str,
    local_scores: Mapping[str, np.ndarray],
    fallback_scores: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, float],
    config: LocalProbeConfig,
    low_fallback: str = "final",
    mid_fallback: str = "soft",
) -> np.ndarray:
    def fallback(name: str) -> np.ndarray:
        if name in fallback_scores:
            return np.asarray(fallback_scores[name], dtype=np.float32)
        return np.asarray(fallback_scores["final"], dtype=np.float32)

    def local(name: str) -> np.ndarray:
        if name in local_scores:
            return np.asarray(local_scores[name], dtype=np.float32)
        return np.asarray(local_scores["local_gate"], dtype=np.float32)

    drop_max = float(diagnostics.get("drop_max", 0.0))
    drop_mean = float(diagnostics.get("drop_mean", 0.0))

    if readout == "local_gate":
        return local("local_gate")
    if readout == "local_gate_g2":
        return local("local_gate_g2")
    if readout == "local_gate_g3":
        return local("local_gate_g3")
    if readout == "local_blend":
        return local("local_blend")

    if readout == "adaptive_sparse_content":
        if drop_max < float(config.low_drop):
            return fallback(low_fallback)
        if drop_max < float(config.mid_drop) and drop_mean < float(config.sparse_drop_mean):
            return local("local_x_prior_content_g2")
        if drop_max < float(config.mid_drop):
            return fallback(mid_fallback)
        return local("local_gate_g2")

    base_name = {
        "adaptive_gate_g2": "local_gate_g2",
        "adaptive_gate_g3": "local_gate_g3",
        "adaptive_blend": "local_blend",
        "adaptive_x_prior": "local_x_prior",
        "adaptive_midsafe_content": "local_x_prior_content_g2",
    }.get(readout, "local_gate")
    if drop_max < float(config.low_drop):
        return fallback(low_fallback)
    if drop_max < float(config.mid_drop):
        if readout == "adaptive_midsafe_content":
            return local("local_x_prior_content_g2")
        return fallback(mid_fallback)
    return local(base_name)
