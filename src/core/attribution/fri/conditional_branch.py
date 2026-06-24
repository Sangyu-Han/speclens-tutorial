from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


MaskValueFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PrefixShapeGateConfig:
    early_k: int = 32
    late_k: int = 64
    late_drop_min: float = 0.90
    early_over_late_max: float = 0.10


@dataclass(frozen=True)
class PrefixShapeProbe:
    budgets: tuple[int, ...]
    values: tuple[float, ...]
    ratios: tuple[float, ...]
    full_value: float
    auc_ratio: float
    last_ratio: float
    max_ratio: float
    argmax_budget: int
    argmax_frac: float


@dataclass(frozen=True)
class PrefixShapeDecision:
    selected: bool
    early_drop: float
    late_drop: float
    early_over_late: float


@dataclass(frozen=True)
class ConditionalRandomBranchConfig:
    head_k: int = 10
    pool_k: int = 196
    prefix_k: int = 32
    group_size: int = 16
    n_groups: int = 32
    mix_alpha: float = 0.25
    seed: int = 20260612


@dataclass(frozen=True)
class BandReplacementConfig:
    kind: str = "fill"
    start: int = 56
    end: int = 64


@dataclass(frozen=True)
class ConditionalRandomBranchResult:
    scores: np.ndarray
    order: np.ndarray
    groups: tuple[tuple[int, ...], ...]
    design: np.ndarray
    effects: np.ndarray
    values: np.ndarray
    head: tuple[int, ...]
    pool: tuple[int, ...]


def stable_case_seed(base_seed: int, key: str) -> int:
    digest = hashlib.blake2s(str(key).encode("utf-8"), digest_size=4).digest()
    key_seed = int.from_bytes(digest, byteorder="little", signed=False)
    return int((int(base_seed) + key_seed) % (2**31 - 1))


def order_of(scores: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.nan_to_num(np.asarray(scores, dtype=np.float64).reshape(-1), nan=0.0), 0.0)
    return np.argsort(-arr, kind="mergesort")


def rank_scores_from_order(order: Sequence[int], n_tokens: int | None = None) -> np.ndarray:
    order_arr = np.asarray(order, dtype=np.int64).reshape(-1)
    n = int(n_tokens or order_arr.size)
    out = np.zeros(n, dtype=np.float32)
    mass = np.linspace(1.0, 0.0, n, endpoint=False, dtype=np.float32)
    out[order_arr[:n]] = mass[: min(n, order_arr.size)]
    return out


def rank_values_like(scores: np.ndarray) -> np.ndarray:
    vals = np.maximum(np.nan_to_num(np.asarray(scores, dtype=np.float64).reshape(-1), nan=0.0), 0.0)
    vals = np.sort(vals)[::-1]
    if float(vals.max()) <= 1e-12:
        vals = np.linspace(1.0, 0.0, vals.size, endpoint=False, dtype=np.float64)
    eps = max(float(vals[0]), 1.0) * 1e-7
    return vals + eps * (vals.size - np.arange(vals.size))


def scores_from_order_like(order: Sequence[int], reference_scores: np.ndarray) -> np.ndarray:
    order_arr = np.asarray(order, dtype=np.int64).reshape(-1)
    n = int(np.asarray(reference_scores).reshape(-1).size)
    out = np.zeros(n, dtype=np.float64)
    out[order_arr[:n]] = rank_values_like(reference_scores)[: min(n, order_arr.size)]
    return out.astype(np.float32)


def complete_order(prefix: Sequence[int], fallback_order: Sequence[int]) -> np.ndarray:
    used: set[int] = set()
    out: list[int] = []
    for idx in prefix:
        idx_i = int(idx)
        if idx_i in used:
            continue
        out.append(idx_i)
        used.add(idx_i)
    for idx in fallback_order:
        idx_i = int(idx)
        if idx_i not in used:
            out.append(idx_i)
            used.add(idx_i)
    return np.asarray(out, dtype=np.int64)


def fill_band_order(
    base_order: Sequence[int],
    candidate_order: Sequence[int],
    *,
    keep: int,
    end: int,
) -> np.ndarray:
    base = np.asarray(base_order, dtype=np.int64).reshape(-1)
    cand = np.asarray(candidate_order, dtype=np.int64).reshape(-1)
    n = int(base.size)
    keep_i = int(max(0, min(int(keep), n)))
    end_i = int(max(keep_i, min(int(end), n)))
    prefix = [int(i) for i in base[:keep_i]]
    used = set(prefix)
    fill: list[int] = []
    for idx in cand:
        item = int(idx)
        if item not in used:
            fill.append(item)
            used.add(item)
        if len(prefix) + len(fill) >= end_i:
            break
    rest = [int(i) for i in base if int(i) not in used]
    return np.asarray(prefix + fill + rest, dtype=np.int64)


def rerank_band_order(
    base_order: Sequence[int],
    candidate_order: Sequence[int],
    *,
    start: int,
    end: int,
) -> np.ndarray:
    base = np.asarray(base_order, dtype=np.int64).reshape(-1)
    cand = np.asarray(candidate_order, dtype=np.int64).reshape(-1)
    n = int(base.size)
    start_i = int(max(0, min(int(start), n)))
    end_i = int(max(start_i, min(int(end), n)))
    band = [int(i) for i in base[start_i:end_i]]
    cand_rank = {int(idx): rank for rank, idx in enumerate(cand)}
    band_sorted = sorted(band, key=lambda idx: cand_rank.get(int(idx), n + int(idx)))
    return np.asarray(
        [int(i) for i in base[:start_i]] + band_sorted + [int(i) for i in base[end_i:]],
        dtype=np.int64,
    )


def band_replacement_order(
    *,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    selected: bool,
    config: BandReplacementConfig = BandReplacementConfig(),
) -> np.ndarray:
    base_order = order_of(base_scores)
    if not bool(selected):
        return base_order
    candidate_order = order_of(candidate_scores)
    if str(config.kind) == "fill":
        return fill_band_order(base_order, candidate_order, keep=int(config.start), end=int(config.end))
    if str(config.kind) == "rerank":
        return rerank_band_order(base_order, candidate_order, start=int(config.start), end=int(config.end))
    raise ValueError(f"unknown band replacement kind: {config.kind}")


def band_replacement_scores(
    *,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    selected: bool,
    config: BandReplacementConfig = BandReplacementConfig(),
) -> np.ndarray:
    order = band_replacement_order(
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        selected=bool(selected),
        config=config,
    )
    return scores_from_order_like(order, base_scores)


def fill_prefix(
    *,
    head: Sequence[int],
    group: Sequence[int],
    fill_order: Sequence[int],
    prefix_k: int,
) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for seq in (head, group, fill_order):
        for idx in seq:
            idx_i = int(idx)
            if idx_i in seen:
                continue
            out.append(idx_i)
            seen.add(idx_i)
            if len(out) >= int(prefix_k):
                return out
    return out


def balanced_groups(
    *,
    pool: Sequence[int],
    group_size: int,
    n_groups: int,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    pool_list = [int(x) for x in pool]
    m = len(pool_list)
    group_size = int(min(max(group_size, 1), m))
    n_groups = int(max(n_groups, 1))
    groups: list[tuple[int, ...]] = []
    design = np.zeros((n_groups, m), dtype=np.float32)
    counts = np.zeros(m, dtype=np.int64)
    for gi in range(n_groups):
        jitter = rng.random(m) * 1e-3
        priority = counts.astype(np.float64) + jitter
        chosen_local = np.argsort(priority, kind="mergesort")[:group_size]
        rng.shuffle(chosen_local)
        design[gi, chosen_local] = 1.0
        counts[chosen_local] += 1
        groups.append(tuple(pool_list[int(i)] for i in chosen_local))
    return tuple(groups), design


def conditional_effects(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    design_arr = np.asarray(design, dtype=np.float32)
    response_arr = np.asarray(response, dtype=np.float64).reshape(-1)
    effects = np.zeros(design_arr.shape[1], dtype=np.float64)
    for j in range(design_arr.shape[1]):
        on = design_arr[:, j] > 0.5
        off = ~on
        if int(on.sum()) == 0 or int(off.sum()) == 0:
            continue
        effects[j] = float(response_arr[on].mean() - response_arr[off].mean())
    return effects


def conditional_deletion_effects(design: np.ndarray, deleted_values: np.ndarray) -> np.ndarray:
    """Return effects where positive means group inclusion lowers the value.

    If a diagnostic response is `constant - deleted_value`, the constant
    cancels in the conditional on/off difference.  Therefore this function uses
    `-deleted_value` directly and needs no extra baseline/full forward.
    """

    return conditional_effects(design, -np.asarray(deleted_values, dtype=np.float64))


def norm01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    lo = float(arr.min()) if arr.size else 0.0
    hi = float(arr.max()) if arr.size else 0.0
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def mix_order_from_effects(
    *,
    head: Sequence[int],
    pool: Sequence[int],
    effects: np.ndarray,
    fallback_order: Sequence[int],
    mix_alpha: float,
) -> np.ndarray:
    pool_list = [int(x) for x in pool]
    fallback = np.asarray(fallback_order, dtype=np.int64).reshape(-1)
    n = int(fallback.size)
    rank = {int(idx): r for r, idx in enumerate(fallback)}
    effects_n = norm01(np.asarray(effects, dtype=np.float64).reshape(-1))
    local_index = {int(idx): i for i, idx in enumerate(pool_list)}
    blended = sorted(
        pool_list,
        key=lambda idx: (
            -(
                float(mix_alpha) * effects_n[local_index[int(idx)]]
                + (1.0 - float(mix_alpha)) * (1.0 - rank.get(int(idx), n) / max(n - 1, 1))
            ),
            rank.get(int(idx), n),
        ),
    )
    return complete_order([int(x) for x in head] + blended, fallback)


def masks_for_conditional_groups(
    *,
    head: Sequence[int],
    groups: Sequence[Sequence[int]],
    fill_order: Sequence[int],
    prefix_k: int,
    n_tokens: int,
) -> np.ndarray:
    masks = np.ones((len(groups), int(n_tokens)), dtype=np.float32)
    for gi, group in enumerate(groups):
        prefix = fill_prefix(
            head=head,
            group=group,
            fill_order=fill_order,
            prefix_k=int(prefix_k),
        )
        masks[gi, prefix] = 0.0
    return masks


def run_conditional_random_branch(
    *,
    value_fn: MaskValueFn,
    base_scores: np.ndarray,
    seed_scores: np.ndarray,
    case_key: str,
    config: ConditionalRandomBranchConfig = ConditionalRandomBranchConfig(),
) -> ConditionalRandomBranchResult:
    seed_order = order_of(seed_scores)
    base_order = order_of(base_scores)
    head = tuple(int(x) for x in seed_order[: int(config.head_k)])
    head_set = set(head)
    pool = tuple(int(x) for x in base_order[: int(config.pool_k)] if int(x) not in head_set)
    case_seed = stable_case_seed(int(config.seed), str(case_key))
    rng_case = np.random.default_rng(case_seed)
    rng = np.random.default_rng(int(rng_case.integers(0, 2**31 - 1)))
    groups, design = balanced_groups(
        pool=pool,
        group_size=int(config.group_size),
        n_groups=int(config.n_groups),
        rng=rng,
    )
    masks = masks_for_conditional_groups(
        head=head,
        groups=groups,
        fill_order=seed_order,
        prefix_k=int(config.prefix_k),
        n_tokens=int(seed_order.size),
    )
    values = np.asarray(value_fn(masks), dtype=np.float64).reshape(-1)
    effects = conditional_deletion_effects(design, values)
    order = mix_order_from_effects(
        head=head,
        pool=pool,
        effects=effects,
        fallback_order=seed_order,
        mix_alpha=float(config.mix_alpha),
    )
    return ConditionalRandomBranchResult(
        scores=rank_scores_from_order(order, n_tokens=int(seed_order.size)),
        order=order,
        groups=groups,
        design=design,
        effects=effects,
        values=values,
        head=head,
        pool=pool,
    )


def probe_prefix_shape(
    *,
    value_fn: MaskValueFn,
    scores: np.ndarray,
    budgets: Sequence[int],
) -> PrefixShapeProbe:
    scores_order = order_of(scores)
    n_tokens = int(scores_order.size)
    budget_list = tuple(sorted(set([0, *[int(k) for k in budgets]])))
    masks = np.ones((len(budget_list), n_tokens), dtype=np.float32)
    for row, budget in enumerate(budget_list):
        if int(budget) > 0:
            masks[row, scores_order[: int(budget)]] = 0.0
    values = tuple(float(x) for x in np.asarray(value_fn(masks), dtype=np.float64).reshape(-1))
    full_idx = budget_list.index(0)
    full_value = float(values[full_idx])
    denom = max(abs(full_value), 1e-8)
    ratios = tuple(float(v / denom) for v in values)
    xs = np.asarray(budget_list, dtype=np.float64) / float(n_tokens)
    auc_ratio = float(np.trapz(np.asarray(ratios, dtype=np.float64), xs) / max(float(xs[-1] - xs[0]), 1e-8))
    argmax = int(np.argmax(np.asarray(ratios, dtype=np.float64)))
    return PrefixShapeProbe(
        budgets=budget_list,
        values=values,
        ratios=ratios,
        full_value=full_value,
        auc_ratio=auc_ratio,
        last_ratio=float(ratios[-1]),
        max_ratio=float(max(ratios)),
        argmax_budget=int(budget_list[argmax]),
        argmax_frac=float(int(budget_list[argmax]) / n_tokens),
    )


def ratio_at(probe: PrefixShapeProbe, budget: int) -> float:
    budgets = [int(x) for x in probe.budgets]
    idx = min(range(len(budgets)), key=lambda i: abs(budgets[i] - int(budget)))
    return float(probe.ratios[idx])


def prefix_shape_decision(
    probe: PrefixShapeProbe,
    config: PrefixShapeGateConfig = PrefixShapeGateConfig(),
) -> PrefixShapeDecision:
    early_drop = 1.0 - ratio_at(probe, int(config.early_k))
    late_drop = 1.0 - ratio_at(probe, int(config.late_k))
    early_over_late = float(early_drop / max(late_drop, 1e-8))
    selected = bool(
        late_drop >= float(config.late_drop_min)
        and early_over_late <= float(config.early_over_late_max)
    )
    return PrefixShapeDecision(
        selected=selected,
        early_drop=float(early_drop),
        late_drop=float(late_drop),
        early_over_late=early_over_late,
    )
