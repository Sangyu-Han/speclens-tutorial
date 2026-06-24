"""HG-FRI: FRI mask solve + input-representation gradient-field channel.

Deletion-oriented extension of FRI developed in the 2026-06 deletion research
(outputs/class_fri/peel_research/NOTES.md). Components, all model-agnostic
(mask -> objective oracle + autograd on the oracle inputs), no 2D-grid prior,
per-instance, ~64-68 fwd/bwd model states total:

  1. 29-step alternating soft-ins/del mask solve (fri_alt_prob dynamics) with
     a similarity-kernel smoother built from the BASELINE token embeddings
     (the model's own positional code; replaces the banned grid-TV).
  2. Free harvest: at every visited state with density >= 0.7, the same
     backward pass exposes the input-representation gradient field via a
     zero-valued probe leaf. ||∂f/∂h_i|| accumulated over these near-full
     states is an evidence-union membership map (`hgd`) that recovers
     redundant evidence copies the L1-suppressed mask drops. The del-step
     recovery gradient (analytic reg subtraction, no extra backward) gives
     the del-support channel (`ds`).
  3. Group value test on the lift set (top-k of n01(hgd)*(1-n01(final))):
     r0 = f(full - lift)/f(full).  r0 > 1+eps -> competitor mass -> demote
     (+1 refinement split); 1.15 < r0 <= 1+eps -> sufficiency probe
     f(lift alone): < 0.15 -> anti-evidence -> demote.
  4. Gated frontier probe: one fwd+bwd at full-minus-(gated lift) adds the
     next evidence layer to the field.
  5. Readout: lam_ad * n01(final) + n01(hgd_fr) + w_ds * n01(ds) - penalty
     on demoted patches; lam_ad in {0.7, 1.0, 1.5} from the r0 redundancy
     probe.

Validated on outputs/class_fri/softplus_failure_scan_n100.csv (seed 42/43):
hdel 0.338/0.358 (soft FRI w/ grid-TV: 0.346; inflow: 0.317), hins 0.97
(inflow 0.899); per-case hdel wins vs inflow 30-34/100, both-metric 16-27.
Zebra/elephant: hdel 0.39 vs inflow 0.59; ZEB mean rank 114 vs inflow 68.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class HGFRIConfig:
    solve_steps: int = 29
    lr: float = 0.45
    lr_end: float = 0.01
    irr_weight: float = 0.05
    l1_weight: float = 0.003
    deletion_weight: float = 0.5
    init_prob: float = 0.5
    ktv_weight: float = 0.01
    ktv_temp: float = 0.05
    ktv_knn: int = 8
    harvest_dens_min: float = 0.7
    lift_k: int = 32
    gate_eps: float = 0.6
    ambig_lo: float = 1.15
    ambig_suff_max: float = 0.15
    # rider split inside the redundancy band (r0 ~ 1): next lift bands that
    # show NO conditional necessity at the post-lift frontier are riders.
    # DEFAULT OFF: conditional necessity is redundancy-blind one level deeper
    # (evidence spread over >2 bands gets demoted) — net harmful on subset.
    rider_tests: int = 0           # 0 disables the rider branch
    rider_r0_lo: float = 0.92
    rider_drop_min: float = 0.10   # required extra recovery drop to count as evidence
    # asymmetric ds-gating of the hgd channel: damp lifts with no del-support
    ds_gate_floor: float = 1.0     # 1.0 = off; e.g. 0.3 -> hgd *= (floor + (1-floor)*n01(ds))
    # minimal-firing-coalition census (boundary-tracking budgets):
    #   half the ins steps track the firing boundary (fired -> shrink budget,
    #   unfired -> grow); per-patch census ratio
    #   P(saturated | small fired state) / P(saturated | fired state)
    #   separates evidence (≈1) from background riders (≈0); patches that
    #   never saturate are neutral (1).
    budget_policy: str = "iid"       # iid | boundary
    census_gate_floor: float = 1.0   # 1.0 = off; e.g. 0.4 -> damp by ratio
    census_small_frac: float = 0.35  # 'small' fired state: budget < frac*N
    census_apply: str = "hgd"        # hgd | all (also final & ds terms)
    demote_factor: float = 0.05
    ds_weight: float = 0.5
    frontier_weight: float = 2.0
    lam_redundant: float = 0.7
    lam_core: float = 1.5
    lam_fixed: Optional[float] = None  # override adaptive head weight when set
    hg_floor_q: float = 0.0  # per-state noise floor: subtract this quantile, clamp 0
    # readout-only cleanup (gate logic always uses unfloored fields):
    readout_floor_q: float = 0.0     # per-state quantile floor for the readout pooling
    smooth_alpha: float = 0.0        # kernel-diffusion strength on readout channels
    smooth_iters: int = 2
    smooth_kernel: str = "bilateral"  # bilateral (pos x content, edge-preserving) | pos
    smooth_content_temp: float = 0.3
    # SAE-style firing threshold on the recovery objective:
    #   below tau the (hard) objective is flat -> no model gradient flows;
    #   only sufficiently-assembled coalitions shape the mask / ds channel.
    fire_tau: float = 0.4            # 0 = off; threshold in recovery space
    fire_temp: float = 0.0           # 0 = hard ReLU; >0 = sigmoid((rec-tau)/T)
    fire_phase: str = "ins"          # both | ins | del | ds (gate only ds accumulation)
    fire_leak: float = 0.0           # leaky slope below tau (0 = hard dead zone)
    fire_adapt_budget: bool = False  # raise ins budget floor after dead states
    # decision-style firing: state fires only when target is top-1 (margin>0)
    fire_signal: str = "rec"         # rec | margin | both (margin AND rec>tau)
    fire_fallback_steps: int = 6     # ins steps with no margin-fire -> drop margin gate
    # budget recycling: dead ins states (hard rec-fire, rec<=tau) have exactly
    # zero model gradient -> skip the model backward (analytic reg grad, same
    # trajectory), pay 1 state instead of 2, run more steps in the same budget.
    solve_state_budget: int = 0      # 0 = legacy fixed solve_steps; else state budget
    budget_pairing: str = "iid"      # iid | antithetic (u, 1-u pairs per phase)
    lr_schedule: str = "budget"      # budget (frac of states) | nominal (frac of solve_steps; recycled steps run at lr_end)
    # deterministic full-state field stabilizer (variance reduction for hgd):
    full_field_votes: float = 0.0    # weight (in per-state votes) of the caller-passed full-state field
    seed: int = 42


class HGFRIOracle(Protocol):
    """Contract the runtime must provide. All tensors live on one device."""

    n_tokens: int
    device: torch.device
    dtype: torch.dtype

    def values(self, masks: torch.Tensor) -> torch.Tensor:
        """Batched objective values for binary/soft masks [B, n] -> [B]. No grad."""
        ...

    def value_with_probe(self, state_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Objective at mask `state_w` with grad; returns (scalar, h_probe leaf).

        h_probe is a zeros-like-[1, n, C] leaf ADDED to the mixed input
        representation so its .grad after backward equals ∂objective/∂h.
        """
        ...

    def token_baseline(self) -> torch.Tensor:
        """Baseline token representations [n, C] (used for the kernel)."""
        ...

    def token_content(self) -> torch.Tensor:
        """Content token representations [n, C] (h - baseline; bilateral smoother).

        Optional — implementations may omit it (pos-only smoothing is used then).
        """
        ...


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def _n01(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return v / (v.max() + 1e-12)


def mp_reorder(scores: np.ndarray, dirs: np.ndarray, k: int = 64) -> np.ndarray:
    """Matching-pursuit dedup: reorder the top-k of `scores` by greedy
    credit x residual-direction-norm (Gram-Schmidt deflation over `dirs`).

    Agent-validated (AGENT_EXPLORE_REPORT.md, A_mpC_k64): spreads redundant
    same-direction evidence through the head ranking; top-k SET unchanged.
    """
    n = len(scores)
    order = np.argsort(-scores)
    pool = list(order[:k])
    credit = scores - scores.min()
    d = np.asarray(dirs, np.float64)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    res = {i: d[i].copy() for i in pool}
    picked: list[int] = []
    remaining = set(pool)
    while remaining:
        best, best_eff = None, -1.0
        for i in remaining:
            eff = credit[i] * np.linalg.norm(res[i])
            if eff > best_eff:
                best, best_eff = i, eff
        picked.append(best)
        remaining.discard(best)
        b = res[best]
        nb = np.linalg.norm(b)
        if nb > 1e-9:
            b = b / nb
            for i in remaining:
                res[i] = res[i] - np.dot(res[i], b) * b
    out = np.zeros(n, np.float64)
    rank = 0
    for i in picked:
        out[i] = n - rank
        rank += 1
    for i in order[k:]:
        out[i] = n - rank
        rank += 1
    return out.astype(np.float32)


def _make_kmat(oracle: HGFRIOracle, cfg: HGFRIConfig) -> Optional[torch.Tensor]:
    if cfg.ktv_weight <= 0:
        return None
    with torch.no_grad():
        emb = oracle.token_baseline()
        hn = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S = (hn @ hn.T).float()
        K = torch.exp((S - 1.0) / max(cfg.ktv_temp, 1e-6))
        K.fill_diagonal_(0.0)
        vals, idx = torch.topk(K, k=int(cfg.ktv_knn), dim=1)
        Ks = torch.zeros_like(K)
        Ks.scatter_(1, idx, vals)
        Ks = 0.5 * (Ks + Ks.T)
        n = oracle.n_tokens
        n_grid_edges = 2.0 * math.sqrt(n) * (math.sqrt(n) - 1.0)
        Ks = Ks * (n_grid_edges / Ks.sum().clamp(min=1e-8))
    return Ks.to(oracle.dtype)


def run_hgfri(
    oracle: HGFRIOracle,
    *,
    irrelevance: torch.Tensor,
    config: HGFRIConfig | None = None,
    full_field: Optional[np.ndarray] = None,
) -> dict:
    """Run HG-FRI. Returns dict with 'scores' (np.float32 [n]) + diagnostics.

    `irrelevance` is the inverse-grad irrelevance vector (legacy FRI; pass
    zeros to disable along with irr_weight=0).
    """
    cfg = config or HGFRIConfig()
    n = int(oracle.n_tokens)
    dev, dtype = oracle.device, oracle.dtype
    irr = irrelevance.detach().reshape(-1).to(device=dev, dtype=dtype)
    kmat = _make_kmat(oracle, cfg)

    with torch.no_grad():
        full_obj = oracle.values(torch.ones(1, n, device=dev, dtype=dtype))[0]
        base_obj = oracle.values(torch.zeros(1, n, device=dev, dtype=dtype))[0]
    scale = float((full_obj - base_obj).abs().clamp(min=1e-8).item())

    def _rec(v: torch.Tensor) -> torch.Tensor:
        denom = full_obj - base_obj
        denom = denom if denom.abs() >= 1e-8 else torch.full_like(denom, 1e-8)
        return (v - base_obj) / denom

    # ---- 1+2: solve with harvest -------------------------------------------
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(cfg.seed))
    la = torch.full((n,), _logit(cfg.init_prob), device=dev, dtype=dtype)
    m_v = torch.zeros(n, device=dev, dtype=dtype)
    v_v = torch.zeros(n, device=dev, dtype=dtype)
    hg_rows: list[np.ndarray] = []
    gvec_sum: Optional[np.ndarray] = None  # summed field VECTORS (for field-MP dirs)
    ds = torch.zeros(n, device=dev, dtype=dtype)
    rec_log = {"ins": [], "del": []}
    b_floor = 0.0  # adaptive ins-budget floor (fire_adapt_budget)
    use_margin = cfg.fire_tau >= 0 and cfg.fire_signal in ("margin", "both") \
        and cfg.fire_phase in ("both", "ins")
    if use_margin and not hasattr(oracle, "value_with_probe_ext"):
        raise RuntimeError("fire_signal=margin requires oracle.value_with_probe_ext")
    margin_disabled = False
    ins_seen = 0
    margin_fires = 0

    def _reg_grad(sig: torch.Tensor) -> torch.Tensor:
        rg = (cfg.irr_weight * irr + cfg.l1_weight) * sig * (1.0 - sig)
        if kmat is not None:
            sgn_diff = torch.sign(sig.view(-1, 1) - sig.view(1, -1))
            rg = rg + cfg.ktv_weight * (kmat * sgn_diff).sum(dim=1) * sig * (1.0 - sig)
        return rg

    state_budget = (
        int(cfg.solve_state_budget) if cfg.solve_state_budget > 0
        else 2 * int(cfg.solve_steps)
    )
    lazy_ok = (cfg.solve_state_budget > 0 and cfg.fire_tau > 0
               and cfg.fire_temp <= 0 and cfg.fire_leak <= 0
               and cfg.fire_signal == "rec")
    states_used = 0
    step = 0
    n_lazy = 0
    u_queues = {"ins": [], "del": []}
    # census accumulators (boundary policy)
    b_fire = 0.5 * n
    ins_idx = 0
    sat_fired = np.zeros(n)
    sat_small_fired = np.zeros(n)
    n_fired_states = 0
    n_small_fired = 0

    while states_used + 2 <= state_budget:
        if cfg.lr_schedule == "nominal":
            frac = min(step / max(int(cfg.solve_steps) - 1, 1), 1.0)
        else:
            frac = states_used / max(state_budget - 2, 1)
        cur_lr = cfg.lr_end + 0.5 * (cfg.lr - cfg.lr_end) * (1 + math.cos(math.pi * frac))
        la_req = la.clone().requires_grad_(True)
        probs = torch.sigmoid(la_req)
        p = probs / (probs.sum() + 1e-8)
        is_del = step % 2 == 1
        phase_name = "del" if is_del else "ins"
        if cfg.budget_pairing == "antithetic":
            qu = u_queues[phase_name]
            if qu:
                u = qu.pop()
            else:
                u = float(torch.rand(1, generator=gen, device=dev).item())
                qu.append(1.0 - u)
        else:
            u = float(torch.rand(1, generator=gen, device=dev).item())
        boundary_step = (cfg.budget_policy == "boundary" and not is_del
                         and ins_idx % 2 == 1)
        if boundary_step:
            budget = b_fire * (0.8 + 0.4 * u)
        elif cfg.fire_adapt_budget and not is_del:
            budget = b_floor + u * (n - b_floor)
        else:
            budget = u * n
        if not is_del:
            ins_idx += 1
        w = (p * budget).clamp(max=1.0)
        state_w = (1.0 - w) if is_del else w

        margin_val = None
        if use_margin and not is_del and not margin_disabled:
            value, margin_t, h_probe = oracle.value_with_probe_ext(state_w)
            margin_val = float(margin_t.detach().item())
        else:
            value, h_probe = oracle.value_with_probe(state_w)
        rec = _rec(value)
        rec_val = float(rec.detach().item())
        rec_log[phase_name].append(rec_val)
        if not is_del:
            ins_fired = rec_val > max(cfg.fire_tau, 0.4)
            if cfg.budget_policy == "boundary":
                if ins_fired:
                    b_fire = max(b_fire * 0.75, 0.04 * n)
                else:
                    b_fire = min(b_fire * 1.3, float(n))
            if ins_fired:
                sat = (w.detach() > 0.99).float().cpu().numpy()
                sat_fired += sat
                n_fired_states += 1
                if budget < cfg.census_small_frac * n:
                    sat_small_fired += sat
                    n_small_fired += 1
        fire_here = (cfg.fire_tau > 0 or use_margin) and cfg.fire_phase in ("both", phase_name)
        if fire_here and cfg.fire_adapt_budget and not is_del:
            if rec_val <= cfg.fire_tau:
                b_floor = max(b_floor, 0.7 * budget)
            else:
                b_floor *= 0.8

        lazy_dead = (lazy_ok and fire_here and not is_del and margin_val is None
                     and rec_val <= cfg.fire_tau)
        if lazy_dead:
            # hard-gated dead state: model gradient is exactly zero ->
            # analytic regularizer gradient only, forward-only cost (1 state)
            g_total = _reg_grad(probs.detach())
            states_used += 1
            n_lazy += 1
        else:
            if fire_here:
                if margin_val is not None:
                    ins_seen += 1
                    m_fired = margin_val > 0.0
                    margin_fires += int(m_fired)
                    if cfg.fire_signal == "margin":
                        gated = rec * float(m_fired)
                    else:  # both: margin AND rec threshold
                        gated = (rec - cfg.fire_tau).clamp(min=0.0) * float(m_fired)
                    obj = cfg.fire_leak * rec + (1.0 - cfg.fire_leak) * gated \
                        if cfg.fire_leak > 0 else gated
                    if ins_seen >= int(cfg.fire_fallback_steps) and margin_fires == 0:
                        margin_disabled = True  # dead-image safeguard
                elif cfg.fire_tau > 0:
                    if cfg.fire_temp > 0:
                        obj = torch.sigmoid((rec - cfg.fire_tau) / cfg.fire_temp)
                    else:
                        obj = (rec - cfg.fire_tau).clamp(min=0.0)
                        if cfg.fire_leak > 0:
                            obj = cfg.fire_leak * rec + (1.0 - cfg.fire_leak) * obj
                else:
                    obj = rec
            else:
                obj = rec
            rec_term = cfg.deletion_weight * obj if is_del else (1.0 - obj)
            loss = rec_term + cfg.irr_weight * (probs * irr).sum() + cfg.l1_weight * probs.sum()
            if kmat is not None:
                diff = (probs.view(-1, 1) - probs.view(1, -1)).abs()
                loss = loss + cfg.ktv_weight * (kmat * diff).sum() / 2.0
            loss.backward()
            states_used += 2
            g_total = la_req.grad.detach()
            if is_del:
                ds_ok = not (cfg.fire_tau > 0 and cfg.fire_phase == "ds"
                             and rec_val <= cfg.fire_tau)
                if ds_ok:
                    ds += (-(g_total - _reg_grad(probs.detach()))).clamp(min=0.0)
            dens = float(state_w.detach().mean().item())
            if dens >= cfg.harvest_dens_min and h_probe.grad is not None:
                g = h_probe.grad.detach()[0] * scale
                hg_rows.append(g.norm(dim=-1).cpu().numpy())
                gvec_sum = gvec_sum + g.cpu().numpy() if gvec_sum is not None else g.cpu().numpy()

        t = step + 1
        m_v = beta1 * m_v + (1 - beta1) * g_total
        v_v = beta2 * v_v + (1 - beta2) * g_total * g_total
        adam_dir = (m_v / (1 - beta1**t)) / ((v_v / (1 - beta2**t)).sqrt() + eps_adam)
        cmask = (adam_dir * g_total > 0).to(dtype)
        cmask = cmask * (n / cmask.sum().clamp(min=1.0))
        la = la - cur_lr * adam_dir * cmask
        step += 1

    final = torch.sigmoid(la).detach().cpu().numpy()

    def _pool(rows: list[np.ndarray], floor_q: float) -> np.ndarray:
        out = np.zeros(n)
        for row in rows:
            if floor_q > 0:
                row = np.maximum(row - np.quantile(row, floor_q), 0.0)
            out += row / (row.sum() + 1e-12)
        return out

    hgd = _pool(hg_rows, cfg.hg_floor_q)               # gate/lift channel
    if full_field is not None and cfg.full_field_votes > 0:
        ff = np.asarray(full_field, np.float64).reshape(-1)
        hgd = hgd + cfg.full_field_votes * (ff / (ff.sum() + 1e-12))
    hgd_read = (
        _pool(hg_rows, cfg.readout_floor_q) if cfg.readout_floor_q > 0 else hgd
    )                                                   # readout channel

    # ---- 3: group value test ------------------------------------------------
    lift = _n01(hgd) * (1.0 - _n01(final))
    G = np.argsort(-lift)[: int(cfg.lift_k)]
    p_full = float(full_obj.item())

    def _value_of(mask_np: np.ndarray) -> float:
        m = torch.as_tensor(mask_np, device=dev, dtype=dtype).unsqueeze(0)
        with torch.no_grad():
            return float(oracle.values(m)[0].item())

    m_del = np.ones(n, np.float32)
    m_del[G] = 0.0
    r0 = _value_of(m_del) / max(p_full, 1e-8)
    demote = np.zeros(n, bool)
    branch = "none"
    if r0 > 1.0 + cfg.gate_eps:
        branch = "competitor"
        demote[G] = True
        # monotone-ascent competitor peeling: extend the demoted set with the
        # next lift group only while its ADDITIONAL deletion raises the prob
        # further (raise = more competitor; drop = the residual lift is
        # evidence -> stop before demoting it)
        level = r0
        tests_left = 2
        while tests_left > 0:
            lift2 = lift.copy()
            lift2[demote] = 0.0
            G2 = np.argsort(-lift2)[: int(cfg.lift_k)]
            m2 = np.ones(n, np.float32)
            m2[np.where(demote)[0]] = 0.0
            m2[G2] = 0.0
            ratio_u = _value_of(m2) / max(p_full, 1e-8)
            tests_left -= 1
            if ratio_u > level + 0.05:
                demote[G2] = True
                level = ratio_u
            else:
                break
    elif r0 > cfg.ambig_lo:
        m_ins = np.zeros(n, np.float32)
        m_ins[G] = 1.0
        suff = _value_of(m_ins) / max(p_full, 1e-8)
        if suff < cfg.ambig_suff_max:
            branch = "anti-evidence"
            demote[G] = True
        else:
            branch = "evidence-kept"
    elif r0 >= cfg.rider_r0_lo and cfg.rider_tests > 0:
        # redundancy band (deleting the lift G changes ~nothing). Probe the
        # NEXT lift bands for CONDITIONAL necessity at the post-lift frontier:
        # ratio_c = p(full - G - C)/p_full vs level = p(full - G)/p_full.
        #   drops further  -> C is backup evidence (copies become necessary
        #                     once the first copy is gone) -> keep
        #   flat or raises -> C never becomes necessary -> context rider /
        #                     competitor leak -> demote (monotone-descent
        #                     peeling, up to rider_tests states)
        level = r0
        excluded = np.zeros(n, bool)
        excluded[G] = True
        tests_left = int(cfg.rider_tests)
        while tests_left > 0:
            lift2 = lift.copy()
            lift2[excluded | demote] = 0.0
            C = np.argsort(-lift2)[: int(cfg.lift_k)]
            if lift2[C].max() <= 0:
                break
            m2 = np.ones(n, np.float32)
            m2[np.where(excluded | demote)[0]] = 0.0
            m2[C] = 0.0
            ratio_c = _value_of(m2) / max(p_full, 1e-8)
            tests_left -= 1
            if ratio_c > level - cfg.rider_drop_min:
                branch = "rider"
                demote[C] = True   # no conditional drop -> never necessary
            else:
                branch = "redundant-kept" if branch == "none" else branch
                excluded[C] = True  # real backup evidence; move past it
                level = ratio_c

    gate = np.where(demote, cfg.demote_factor, 1.0)
    hgd_gated = hgd * gate

    # ---- 4: gated frontier probe --------------------------------------------
    lift_g = _n01(hgd_gated) * (1.0 - _n01(final))
    Gf = np.argsort(-lift_g)[: int(cfg.lift_k)]
    mf = np.ones(n, np.float32)
    mf[Gf] = 0.0
    state_w = torch.as_tensor(mf, device=dev, dtype=dtype)
    value, h_probe = oracle.value_with_probe(state_w)
    value.backward()
    gfr_vec = h_probe.grad.detach()[0].cpu().numpy()
    if gvec_sum is not None:
        gvec_sum = gvec_sum + gfr_vec
    else:
        gvec_sum = gfr_vec
    gfr = np.linalg.norm(gfr_vec, axis=-1)
    fq = max(cfg.hg_floor_q, cfg.readout_floor_q)
    if fq > 0:
        gfr = np.maximum(gfr - np.quantile(gfr, fq), 0.0)
    hgd_fr = hgd_read * gate + cfg.frontier_weight * (gfr / (gfr.sum() + 1e-12)) * gate

    # ---- 5: readout -----------------------------------------------------------
    if cfg.lam_fixed is not None:
        lam = float(cfg.lam_fixed)
    elif branch == "rider":
        lam = 1.0  # redundancy assumption refuted -> neutral head weight
    elif 0.92 <= r0 <= 1.08:
        lam = cfg.lam_redundant
    elif r0 < 0.92:
        lam = cfg.lam_core
    else:
        lam = 1.0
    ds_np = _n01(ds.cpu().numpy()) * gate
    pen = demote.astype(float)
    if cfg.ds_gate_floor < 1.0:
        hgd_fr = hgd_fr * (cfg.ds_gate_floor + (1.0 - cfg.ds_gate_floor) * ds_np)

    # census ratio gate (minimal-firing-coalition membership)
    census_ratio = np.ones(n)
    if n_fired_states > 0:
        p_sat_fired = sat_fired / n_fired_states
        p_sat_small = sat_small_fired / max(n_small_fired, 1)
        informative = sat_fired > 0
        if n_small_fired > 0:
            census_ratio[informative] = np.clip(
                p_sat_small[informative] / (p_sat_fired[informative] + 1e-9), 0.0, 1.0)
    census_gate = np.ones(n)
    if cfg.census_gate_floor < 1.0 and n_small_fired > 0:
        census_gate = cfg.census_gate_floor + (1.0 - cfg.census_gate_floor) * census_ratio
        hgd_fr = hgd_fr * census_gate

    smooth_krow: Optional[np.ndarray] = None
    if cfg.smooth_alpha > 0:
        with torch.no_grad():
            pos = oracle.token_baseline()
            pn = pos / pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Ksm = torch.exp(((pn @ pn.T).float() - 1.0) / max(cfg.ktv_temp, 1e-6))
            if cfg.smooth_kernel == "bilateral" and hasattr(oracle, "token_content"):
                c = oracle.token_content()
                cn = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                Ksm = Ksm * torch.exp(
                    ((cn @ cn.T).float() - 1.0) / max(cfg.smooth_content_temp, 1e-6)
                )
            Ksm.fill_diagonal_(0.0)
            vals, idx = torch.topk(Ksm, k=int(cfg.ktv_knn), dim=1)
            Kk = torch.zeros_like(Ksm)
            Kk.scatter_(1, idx, vals)
            Kk = 0.5 * (Kk + Kk.T)
            smooth_krow = (Kk / Kk.sum(dim=1, keepdim=True).clamp(min=1e-8)).cpu().numpy()

    def _smooth(v: np.ndarray) -> np.ndarray:
        if smooth_krow is None:
            return v
        s = v.astype(np.float64)
        for _ in range(int(cfg.smooth_iters)):
            s = (1.0 - cfg.smooth_alpha) * s + cfg.smooth_alpha * (smooth_krow @ s)
        return s

    final_term = _n01(final)
    ds_term = ds_np
    if cfg.census_apply == "all" and cfg.census_gate_floor < 1.0 and n_small_fired > 0:
        final_term = final_term * census_gate
        ds_term = ds_term * census_gate
    scores = (
        lam * _smooth(final_term)
        + _smooth(_n01(hgd_fr))
        + cfg.ds_weight * _smooth(ds_term)
        - (lam + cfg.ds_weight) * pen
    )
    fired = {
        ph: (float(np.mean(np.asarray(v) > cfg.fire_tau)) if v else 1.0)
        for ph, v in rec_log.items()
    }
    return {
        "scores": scores.astype(np.float32),
        "final": final.astype(np.float32),
        "hgd": hgd.astype(np.float32),
        "r0": float(r0),
        "branch": branch,
        "n_demoted": int(demote.sum()),
        "lam": float(lam),
        "fired_frac": fired,
        "n_steps": int(step),
        "n_lazy": int(n_lazy),
        "solve_states": int(states_used),
        "gvec": gvec_sum,  # [n, C] summed field vectors (field-MP directions)
        "census_ratio": census_ratio.astype(np.float32),
        "n_fired_states": int(n_fired_states),
        "n_small_fired": int(n_small_fired),
    }
