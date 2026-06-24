from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from ..base import BaseAutoencoder
from ..registry import register
from .jumprelu import JumpReLUSAE, _JumpReLUFn

_EPSILON = 1e-6


def _l2(v: torch.Tensor, dims=None) -> torch.Tensor:
    if dims is None:
        return v.square().sum().sqrt()
    return v.square().sum(dims).sqrt()


def _l1(v: torch.Tensor, dims=None) -> torch.Tensor:
    if dims is None:
        return torch.abs(v).sum()
    return torch.abs(v).sum(dims)


def _l1_l2_ratio(x: torch.Tensor, dims: int | tuple = -1) -> torch.Tensor:
    l1_norm = _l1(x, dims)
    l2_norm = _l2(x, dims) + _EPSILON
    return l1_norm / l2_norm


def _hoyer(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 2, "Input tensor must be 2D"
    d_sqrt = torch.sqrt(torch.tensor(x.shape[1], device=x.device, dtype=x.dtype))
    l1_l2 = _l1_l2_ratio(x, 1)
    return (d_sqrt - l1_l2) / (d_sqrt - 1)


def _kappa_4(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 2, "Input tensor must be 2D"
    x4 = (x**4).sum(1)
    x2_2 = x.square().sum(1).square()
    return x4 / (x2_2 + _EPSILON)


_LOSS_ALIASES = {
    "mse": "mse",
    "l2": "mse",
    "mse_l1": "mse_l1",
    "l1": "mse_l1",
    "mse_hoyer": "mse_hoyer",
    "hoyer": "mse_hoyer",
    "mse_kappa_4": "mse_kappa_4",
    "kappa_4": "mse_kappa_4",
    "kappa4": "mse_kappa_4",
    "mse_elastic": "mse_elastic",
    "elastic": "mse_elastic",
    "elastic_net": "mse_elastic",
    "top_k_auxiliary_loss": "top_k_auxiliary_loss",
    "topk_auxiliary_loss": "top_k_auxiliary_loss",
    "topk_aux": "top_k_auxiliary_loss",
    "auxk": "top_k_auxiliary_loss",
}


def _resolve_loss_name(cfg: Dict[str, Any], default: str = "mse_l1") -> str:
    name = cfg.get("loss_name", cfg.get("loss_type", cfg.get("criterion", default)))
    return str(name).lower()


def _get_l1_penalty(cfg: Dict[str, Any]) -> float:
    for key in ("loss_penalty", "l1_coeff", "l1"):
        if cfg.get(key) is not None:
            return float(cfg[key])
    return 1.0


def _get_aux_penalty(cfg: Dict[str, Any]) -> float:
    for key in ("loss_penalty", "aux_penalty", "aux_frac"):
        if cfg.get(key) is not None:
            return float(cfg[key])
    return 0.1


def _get_elastic_alpha(cfg: Dict[str, Any]) -> float:
    for key in ("loss_alpha", "elastic_alpha", "alpha"):
        if cfg.get(key) is not None:
            return float(cfg[key])
    return 0.5


def _zero_like(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, device=x.device, dtype=x.dtype)


def _compute_overcomplete_loss(
    loss_name: str,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    cfg: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    name = _LOSS_ALIASES.get(loss_name, loss_name)

    mse = (x - x_hat).square().mean()
    zero = _zero_like(mse)

    if name == "mse":
        return mse, mse, zero, zero

    if name == "mse_l1":
        penalty = _get_l1_penalty(cfg)
        reg = torch.mean(_l1(codes)) * penalty
        return mse + reg, mse, reg, zero

    if name == "mse_hoyer":
        penalty = _get_l1_penalty(cfg)
        reg = _hoyer(codes).mean() * penalty
        return mse + reg, mse, reg, zero

    if name == "mse_kappa_4":
        penalty = _get_l1_penalty(cfg)
        reg = _kappa_4(codes).mean() * penalty
        return mse + reg, mse, reg, zero

    if name == "mse_elastic":
        alpha = _get_elastic_alpha(cfg)
        l1_term = codes.abs().mean()
        l2_term = dictionary.square().mean()
        loss = mse + (1.0 - alpha) * l1_term + alpha * l2_term
        return loss, mse, (1.0 - alpha) * l1_term, alpha * l2_term

    if name == "top_k_auxiliary_loss":
        penalty = _get_aux_penalty(cfg)
        residual = (x - x_hat).detach()
        pre = torch.relu(pre_codes)
        pre = pre - codes
        # Use k_aux from config; fall back to dict_size // 2
        k = int(cfg.get("k_aux", cfg.get("top_k_aux", pre.shape[1] // 2)))
        k = min(k, pre.shape[1])  # clamp to dict_size
        if k <= 0:
            return mse, mse, zero, zero
        auxiliary_topk = torch.topk(pre, k=k, dim=1)
        pre = torch.zeros_like(codes).scatter(-1, auxiliary_topk.indices, auxiliary_topk.values)
        residual_hat = pre @ dictionary
        aux_mse = (residual - residual_hat).square().mean()
        aux_term = penalty * aux_mse
        return mse + aux_term, mse, zero, aux_term

    raise ValueError(f"Unknown overcomplete loss_name: {loss_name}")


class RelaxedArchetypalDictionary(nn.Module):
    """
    Relaxed Archetypal Dictionary (Fel et al., 2025).

    Each dictionary atom is a convex combination of provided points with
    a relaxation term bounded by `delta` and an optional learnable multiplier.
    """

    def __init__(
        self,
        in_dimensions: int,
        nb_concepts: int,
        points: torch.Tensor,
        delta: float = 1.0,
        use_multiplier: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_dimensions = int(in_dimensions)
        self.nb_concepts = int(nb_concepts)
        self.delta = float(delta)

        if points is None:
            raise ValueError("`points` must be provided for RelaxedArchetypalDictionary.")
        points = torch.as_tensor(points)
        if dtype is None:
            dtype = points.dtype
        if device is None:
            device = points.device
        points = points.to(device=device, dtype=dtype)

        if points.dim() != 2 or points.size(1) != self.in_dimensions:
            raise ValueError(
                f"`points` must be a 2D tensor of shape [num_points, {self.in_dimensions}]. "
                f"Got {tuple(points.shape)}."
            )

        self.register_buffer("C", points)
        self.nb_candidates = int(self.C.shape[0])

        # W: row-stochastic weights over candidate points
        self.W = nn.Parameter(
            torch.eye(self.nb_concepts, self.nb_candidates, device=device, dtype=dtype)
        )
        # Relaxation term
        self.Relax = nn.Parameter(
            torch.zeros(self.nb_concepts, self.in_dimensions, device=device, dtype=dtype)
        )

        if use_multiplier:
            self.multiplier = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        else:
            self.register_buffer(
                "multiplier", torch.tensor(0.0, device=device, dtype=dtype)
            )

        self._fused_dictionary: Optional[torch.Tensor] = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        dictionary = self.get_dictionary()
        return z @ dictionary

    def get_dictionary(self) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                # Project W to the simplex (row-stochastic, non-negative)
                W = torch.relu(self.W)
                W /= (W.sum(dim=-1, keepdim=True) + 1e-8)
                self.W.data.copy_(W)

                # Enforce ||Relax|| <= delta per row
                norm_relax = self.Relax.norm(dim=-1, keepdim=True)
                scale = torch.clamp(self.delta / norm_relax, max=1.0)
                self.Relax.data.mul_(scale)

            D = self.W @ self.C + self.Relax
            return D * torch.exp(self.multiplier)

        assert self._fused_dictionary is not None, "Dictionary is not initialized."
        return self._fused_dictionary

    def train(self, mode: bool = True):
        if not mode:
            self._fused_dictionary = self.get_dictionary().detach()
        return super().train(mode)


@register("ra-topk")
@register("ra-ar")
class RATopKSAE(BaseAutoencoder):
    """
    Relaxed Archetypal TopK SAE (official RA-SAE behaviour).

    Uses a RelaxedArchetypalDictionary for decoding and Top-K sparsification
    for encoding, matching the official implementation in overcomplete.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

        # Respect config; default False for backward compat with overcomplete.
        self.config["input_unit_norm"] = cfg.get("input_unit_norm", False)

        act_size = int(cfg["act_size"])
        dict_size = int(cfg["dict_size"])

        self.top_k = int(cfg.get("top_k", cfg.get("k", max(dict_size // 10, 1))))
        if self.top_k > dict_size:
            raise ValueError(f"top_k must be <= dict_size (got {self.top_k} > {dict_size}).")
        self.k_aux = int(cfg.get("top_k_aux", cfg.get("k_aux", 512)))
        self.aux_frac = float(cfg.get("aux_frac", cfg.get("aux_penalty", 0.1)))
        self.l1_coeff = float(cfg.get("l1_coeff", cfg.get("l1", 0.0)))
        self.loss_name = _resolve_loss_name(cfg, default="mse_l1")

        self.delta = float(cfg.get("delta", 1.0))
        self.use_multiplier = bool(cfg.get("use_multiplier", True))

        points = cfg.get("points", None)
        if points is None:
            raise ValueError(
                "RATopKSAE requires `points` in cfg (tensor [num_points, act_size])."
            )

        device = self.W_enc.device
        dtype = self.W_enc.dtype
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=act_size,
            nb_concepts=dict_size,
            points=points,
            delta=self.delta,
            use_multiplier=self.use_multiplier,
            device=device,
            dtype=dtype,
        )

        # Official overcomplete has no decoder bias / no separate W_dec.
        # Freeze the inherited parameters so they stay zero and the optimizer skips them.
        self.b_dec.requires_grad_(False)
        self.W_dec.requires_grad_(False)

    def get_dictionary(self) -> torch.Tensor:
        return self.dictionary.get_dictionary()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])

        pre_codes = x_flat @ self.W_enc + self.b_enc
        codes = torch.relu(pre_codes)
        tk = torch.topk(codes, self.top_k, dim=-1)
        acts = torch.zeros_like(codes).scatter(-1, tk.indices, tk.values)

        if len(orig_shape) == 3:
            acts = acts.view(orig_shape[0], orig_shape[1], -1)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        orig_shape = acts.shape
        acts_flat = acts.reshape(-1, acts.shape[-1])
        dictionary = self.dictionary.get_dictionary()
        recon = acts_flat @ dictionary

        if len(orig_shape) == 3:
            recon = recon.view(orig_shape[0], orig_shape[1], -1)

        x_mean = getattr(self, "x_mean", torch.zeros_like(recon[:1]))
        x_std = getattr(self, "x_std", torch.ones_like(recon[:1]))
        return self.postprocess_output(recon, x_mean, x_std)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])

        pre_codes = x_flat @ self.W_enc + self.b_enc
        codes = torch.relu(pre_codes)
        tk = torch.topk(codes, self.top_k, dim=-1)
        acts = torch.zeros_like(codes).scatter(-1, tk.indices, tk.values)

        dictionary = self.dictionary.get_dictionary()
        recon_flat = acts @ dictionary
        recon = recon_flat.view_as(x_proc)

        self.update_inactive_features(acts)

        loss, l2_loss, l1_loss, aux_loss = _compute_overcomplete_loss(
            self.loss_name, x_flat, recon_flat, pre_codes, acts, dictionary, self.config
        )
        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_norm = acts.abs().sum(-1).mean()

        # Reconstruction quality metrics
        with torch.no_grad():
            x_f = x_flat.float()
            r_f = recon_flat.float()
            var_x = x_f.var()
            var_resid = (x_f - r_f).var()
            explained_var = 1.0 - (var_resid / (var_x + 1e-8))
            # Normalized MSE (FVU): MSE / Var(x)
            nmse = (x_f - r_f).square().mean() / (var_x + 1e-8)
            # Relative L2: ||x - x_hat|| / ||x|| per sample, averaged
            rel_l2 = ((x_f - r_f).norm(dim=-1) / (x_f.norm(dim=-1) + 1e-8)).mean()

        sae_out = self.postprocess_output(recon, x_mean, x_std)
        if len(orig_shape) == 3:
            acts_out = acts.view(orig_shape[0], orig_shape[1], -1)
        else:
            acts_out = acts

        num_dead_features = (
            self.num_batches_not_active > self.config.get("n_batches_to_dead", 20)
        ).sum()

        return {
            "sae_out": sae_out,
            "feature_acts": acts_out,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "explained_var": explained_var,
            "nmse": nmse,
            "relative_l2": rel_l2,
        }

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """No-op: RA dictionary replaces W_dec normalization."""
        return


@register("ra-batchtopk")
@register("ra-unitcentroid-batchtopk")
class RABatchTopKSAE(RATopKSAE):
    """
    Relaxed Archetypal SAE with *global batch* Top-K encoding.

    Difference from RATopKSAE:
    - TopK is global across the batch (selects B*k values total),
      NOT per-sample. This guarantees average k_eff = k.
    - TopK is applied on pre_codes (before relu), then relu is applied
      to the selected values only.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        device = self.W_enc.device
        dtype = self.W_enc.dtype
        self.register_buffer(
            "threshold", torch.tensor(0.0, device=device, dtype=dtype)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])

        pre_codes = x_flat @ self.W_enc + self.b_enc
        codes = torch.relu(pre_codes)

        if self.training:
            # ReLU first, then global batch Top-(B*k) on non-negative values.
            # All selected values are guaranteed positive → k_eff = k.
            B = x_flat.size(0)
            flat = codes.reshape(-1)
            k_total = int(min(self.top_k * B, flat.numel()))
            if k_total > 0:
                topk = torch.topk(flat, k_total, sorted=False)
                acts_flat = torch.zeros_like(flat)
                acts_flat.scatter_(0, topk.indices, topk.values)
                acts = acts_flat.view_as(codes)
            else:
                acts = torch.zeros_like(codes)
        else:
            acts = codes * (codes > self.threshold).float()

        if len(orig_shape) == 3:
            acts = acts.view(orig_shape[0], orig_shape[1], -1)
        return acts

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])

        pre_codes = x_flat @ self.W_enc + self.b_enc
        codes = torch.relu(pre_codes)

        # ReLU first, then global batch Top-(B*k)
        B = x_flat.size(0)
        flat = codes.reshape(-1)
        k_total = int(min(self.top_k * B, flat.numel()))
        if k_total > 0:
            topk = torch.topk(flat, k_total, sorted=False)
            acts_flat = torch.zeros_like(flat)
            acts_flat.scatter_(0, topk.indices, topk.values)
            acts = acts_flat.view_as(codes)
        else:
            acts = torch.zeros_like(codes)

        # Update threshold for eval mode
        with torch.no_grad():
            if k_total > 0:
                kth_min = topk.values.min()
                self.threshold.lerp_(kth_min, 0.05)

        dictionary = self.dictionary.get_dictionary()
        recon_flat = acts @ dictionary
        recon = recon_flat.view_as(x_proc)

        self.update_inactive_features(acts)
        self.update_feature_freq(acts)

        # Cache batch stats for bias feature folding
        with torch.no_grad():
            self._batch_act_mean = acts.mean(dim=0).detach()
            batch_std = acts.std(dim=0)
            self._batch_act_cv = batch_std / (self._batch_act_mean.abs() + 1e-8)

        # --- Reconstruction loss (MSE or MSE+NMSE blend) ---
        mse = (x_flat - recon_flat).square().mean()
        nmse_w = self.config.get("nmse_weight", 0.0)
        if nmse_w > 0:
            x_var = x_flat.detach().var() + 1e-8
            nmse_loss = mse / x_var
            l2_loss = (1.0 - nmse_w) * mse + nmse_w * nmse_loss
        else:
            l2_loss = mse
        l1_coeff = self.config.get("l1_coeff", 0.0)
        l0_norm = (acts > 0).float().sum(-1).mean()
        l1_norm = acts.abs().sum(-1).mean()
        l1_loss = l1_coeff * l1_norm

        # --- Dead-feature-only auxiliary loss (BatchTopK style) ---
        aux_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
        dead = self.num_batches_not_active >= self.config.get("n_batches_to_dead", 20)
        if dead.any():
            resid = (x_flat - recon_flat).detach()
            num_dead = int(dead.sum().item())
            k_aux = min(int(self.config.get("k_aux", 256)), num_dead)
            if k_aux > 0:
                # Use pre_codes (before relu) for dead features only
                dead_pre = pre_codes[:, dead]
                top_aux = torch.topk(dead_pre, k_aux, dim=-1, sorted=False)
                aux_mask = torch.zeros_like(dead_pre).scatter(
                    -1, top_aux.indices, 1.0
                )
                aux_acts = dead_pre * aux_mask
                # Reconstruct residual using dead feature dictionary atoms
                recon_aux = aux_acts @ dictionary[dead]
                aux_penalty = self.config.get("aux_penalty", 1 / 32)
                aux_loss = aux_penalty * (recon_aux - resid).square().mean()

        # --- Per-batch reanimation loss (overcomplete style) ---
        # Gently push pre_codes of batch-level dead features toward positive
        # to help them win top-k slots in future steps.
        reanim_coeff = self.config.get("reanim_coeff", 1e-3)
        if reanim_coeff > 0:
            batch_dead = ((acts > 0).sum(dim=0) == 0).float().detach()
            reanim_loss = (pre_codes * batch_dead[None, :]).mean()
            loss = l2_loss + l1_loss + aux_loss - reanim_coeff * reanim_loss
        else:
            loss = l2_loss + l1_loss + aux_loss

        # --- Spatial variance penalty (debias high-freq features) ---
        spatial_var_coeff = self.config.get("spatial_var_penalty", 0.0)
        spatial_var_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
        n_high_freq = 0
        if spatial_var_coeff > 0:
            freq_threshold = self.config.get("spatial_var_freq_threshold", 0.9)
            high_freq = (self.feature_freq_ema > freq_threshold)
            n_high_freq = int(high_freq.sum().item())
            if high_freq.any():
                bias_acts = acts[:, high_freq]
                var_per_feat = bias_acts.var(dim=0)  # variance across tokens
                spatial_var_loss = spatial_var_coeff * var_per_feat.mean()
                loss = loss + spatial_var_loss

        # --- Frequency penalty (suppress high-freq bias features) ---
        freq_coeff = float(self.config.get("freq_penalty_coeff", 0.0))
        freq_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
        if freq_coeff > 0:
            freq_weight = self.feature_freq_ema.detach()
            freq_loss = freq_coeff * (freq_weight * acts).sum(-1).mean()
            loss = loss + freq_loss

        # --- Bias explained variance penalty ---
        bev_coeff = float(self.config.get("bev_penalty", 0.0))
        bev_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
        if bev_coeff > 0:
            bev_freq_thr = self.config.get("spatial_var_freq_threshold", 0.9)
            bev_high = (self.feature_freq_ema > bev_freq_thr).detach()
            if bev_high.any():
                bev_acts = acts.clone()
                bev_acts[:, ~bev_high] = 0
                bev_recon = bev_acts @ dictionary.detach()
                bev_var_x = x_flat.float().var().detach()
                bev_var_resid = (x_flat.float() - bev_recon.float()).var()
                raw_bev = 1.0 - bev_var_resid / (bev_var_x + 1e-8)
                bev_loss = bev_coeff * torch.clamp(raw_bev, min=0.0)
                loss = loss + bev_loss

        with torch.no_grad():
            x_f = x_flat.float()
            r_f = recon_flat.float()
            var_x = x_f.var()
            var_resid = (x_f - r_f).var()
            explained_var = 1.0 - (var_resid / (var_x + 1e-8))
            nmse = (x_f - r_f).square().mean() / (var_x + 1e-8)
            rel_l2 = ((x_f - r_f).norm(dim=-1) / (x_f.norm(dim=-1) + 1e-8)).mean()

            # How much variance is explained by bias features alone
            freq_thr_metric = self.config.get("spatial_var_freq_threshold", 0.9)
            high_freq_m = (self.feature_freq_ema > freq_thr_metric)
            if high_freq_m.any():
                bias_only_acts = acts.clone()
                bias_only_acts[:, ~high_freq_m] = 0
                bias_recon = bias_only_acts @ dictionary
                bias_var_resid = (x_f - bias_recon.float()).var()
                bias_explained_var = 1.0 - (bias_var_resid / (var_x + 1e-8))
            else:
                bias_explained_var = torch.tensor(0.0)

        # Add b_norm back for output (Convention A: decode + b_norm)
        if hasattr(self, "b_norm"):
            recon_out = (recon_flat + self.b_norm).view_as(x_proc)
        else:
            recon_out = recon
        sae_out = self.postprocess_output(recon_out, x_mean, x_std)
        if len(orig_shape) == 3:
            acts_out = acts.view(orig_shape[0], orig_shape[1], -1)
        else:
            acts_out = acts

        num_dead_features = (
            self.num_batches_not_active > self.config.get("n_batches_to_dead", 20)
        ).sum()

        return {
            "sae_out": sae_out,
            "feature_acts": acts_out,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "spatial_var_loss": spatial_var_loss,
            "freq_loss": freq_loss,
            "bev_loss": bev_loss,
            "bias_explained_var": bias_explained_var,
            "n_high_freq": n_high_freq,
            "explained_var": explained_var,
            "nmse": nmse,
            "relative_l2": rel_l2,
        }

    @torch.no_grad()
    def fold_bias_features(
        self,
        freq_threshold: float = 0.95,
        cv_threshold: float = 0.02,
    ) -> int:
        """Fold constant-activation (bias) features into b_norm and resample."""
        if not hasattr(self, "b_norm"):
            return 0
        if not hasattr(self, "_batch_act_mean"):
            return 0

        freq = self.feature_freq_ema
        mean_act = self._batch_act_mean
        cv = self._batch_act_cv

        constant = (freq > freq_threshold) & (cv < cv_threshold) & (mean_act > 0)
        if not constant.any():
            return 0

        dictionary = self.dictionary.get_dictionary()
        n_folded = 0

        for idx in constant.nonzero().squeeze(-1):
            i = idx.item()
            a_i = mean_act[i].item()
            d_i = dictionary[i]

            # 1) Absorb into b_norm
            contribution = a_i * d_i
            self.b_norm.data.add_(contribution)

            # 2) Compensate b_enc so pre_code for other features stays same
            self.b_enc.data.add_(contribution @ self.W_enc)

            # 3) Re-initialise encoder column (Kaiming)
            fan_in = self.W_enc.shape[0]
            std = (2.0 / fan_in) ** 0.5
            self.W_enc.data[:, i].normal_(0, std)
            self.b_enc.data[i] = 0.0

            # 4) Reset tracking — aux will revive if it stays dead
            self.num_batches_not_active[i] = 0
            self.feature_freq_ema[i] = 0.0

            n_folded += 1

        return n_folded


@register("ra-jump")
@register("ra-jumprelu")
class RAJumpSAE(JumpReLUSAE):
    """
    Relaxed Archetypal JumpReLU SAE (official RA-SAE behaviour).

    Uses a RelaxedArchetypalDictionary for decoding and JumpReLU sparsification
    for encoding, matching the official implementation in overcomplete.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

        # Match overcomplete defaults for RA-SAE.
        self.config["input_unit_norm"] = False

        act_size = int(cfg["act_size"])
        dict_size = int(cfg["dict_size"])

        self.delta = float(cfg.get("delta", 1.0))
        self.use_multiplier = bool(cfg.get("use_multiplier", True))
        self.loss_name = _resolve_loss_name(cfg, default="mse_l1")

        points = cfg.get("points", None)
        if points is None:
            raise ValueError(
                "RAJumpSAE requires `points` in cfg (tensor [num_points, act_size])."
            )

        device = self.W_enc.device
        dtype = self.W_enc.dtype
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=act_size,
            nb_concepts=dict_size,
            points=points,
            delta=self.delta,
            use_multiplier=self.use_multiplier,
            device=device,
            dtype=dtype,
        )

        # Official overcomplete has no decoder bias / no separate W_dec.
        self.b_dec.requires_grad_(False)
        self.W_dec.requires_grad_(False)

    def get_dictionary(self) -> torch.Tensor:
        return self.dictionary.get_dictionary()

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        orig_shape = acts.shape
        acts_flat = acts.reshape(-1, acts.shape[-1])
        dictionary = self.dictionary.get_dictionary()
        recon = acts_flat @ dictionary
        recon = self._activation_norm_out(recon)

        if len(orig_shape) == 3:
            recon = recon.view(orig_shape[0], orig_shape[1], -1)

        x_mean = getattr(self, "x_mean", torch.zeros_like(recon[:1]))
        x_std = getattr(self, "x_std", torch.ones_like(recon[:1]))
        return self.postprocess_output(recon, x_mean, x_std)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        orig_shape, x_norm, x_proc, x_mean, x_std = self._preprocess_flat(x)

        pre = x_proc @ self.W_enc + self.b_enc
        acts = _JumpReLUFn.apply(pre, self.threshold, self.bandwidth)

        dictionary = self.dictionary.get_dictionary()
        recon_flat = acts @ dictionary
        recon_flat = self._activation_norm_out(recon_flat)
        recon_out = recon_flat
        if len(orig_shape) == 3:
            recon_out = recon_flat.view(orig_shape[0], orig_shape[1], -1)

        self.update_inactive_features(acts)
        self.update_feature_freq(acts)

        # --- Reconstruction loss ---
        mse = (x_norm - recon_flat).square().mean()
        l1_coeff = self.config.get("l1_coeff", 0.0)
        l1_norm = acts.abs().sum(-1).mean()
        l1_loss = l1_coeff * l1_norm if l1_coeff > 0 else _zero_like(mse)

        # --- JumpReLU L0 sparsity loss ---
        l0_loss = self._compute_sparsity_loss(acts, pre)

        # --- Dead-feature-only auxiliary loss (BatchTopK style) ---
        aux_loss = torch.tensor(0.0, device=x_norm.device, dtype=x_norm.dtype)
        dead = self.num_batches_not_active >= self.config.get("n_batches_to_dead", 20)
        if dead.any():
            resid = (x_norm - recon_flat).detach()
            num_dead = int(dead.sum().item())
            k_aux = min(int(self.config.get("k_aux", 256)), num_dead)
            if k_aux > 0:
                dead_pre = pre[:, dead]
                top_aux = torch.topk(dead_pre, k_aux, dim=-1, sorted=False)
                aux_mask = torch.zeros_like(dead_pre).scatter(-1, top_aux.indices, 1.0)
                aux_acts = dead_pre * aux_mask
                recon_aux = aux_acts @ dictionary[dead]
                aux_penalty = self.config.get("aux_penalty", 0.1)
                aux_loss = aux_penalty * (recon_aux - resid).square().mean()

        # --- Optional: frequency penalty (suppress bias features) ---
        freq_coeff = float(self.config.get("freq_penalty_coeff", 0.0))
        freq_loss = torch.tensor(0.0, device=x_norm.device, dtype=x_norm.dtype)
        if freq_coeff > 0:
            freq_weight = self.feature_freq_ema.detach()
            freq_loss = freq_coeff * (freq_weight * acts).sum(-1).mean()

        loss = mse + l1_loss + l0_loss + aux_loss + freq_loss

        # --- Quality metrics ---
        with torch.no_grad():
            x_f = x_norm.float()
            r_f = recon_flat.float()
            var_x = x_f.var()
            var_resid = (x_f - r_f).var()
            explained_var = 1.0 - (var_resid / (var_x + 1e-8))
            nmse = (x_f - r_f).square().mean() / (var_x + 1e-8)
            rel_l2 = ((x_f - r_f).norm(dim=-1) / (x_f.norm(dim=-1) + 1e-8)).mean()

        num_dead_features = (
            self.num_batches_not_active > self.config.get("n_batches_to_dead", 20)
        ).sum()

        sae_out = self.postprocess_output(recon_out, x_mean, x_std)
        return {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": mse,
            "l1_loss": l1_loss,
            "l0_loss": l0_loss,
            "aux_loss": aux_loss,
            "freq_loss": freq_loss,
            "l1_norm": l1_norm,
            "l0_norm": (acts > 0).float().sum(-1).mean(),
            "threshold": self.threshold,
            "explained_var": explained_var,
            "nmse": nmse,
            "relative_l2": rel_l2,
        }

    def _preprocess_flat(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)

        x_flat = x_proc.reshape(-1, x_proc.shape[-1])
        x_norm = self._activation_norm_in(x_flat)
        # Overcomplete-style: do not subtract b_dec from inputs.
        return orig_shape, x_norm, x_norm, x_mean, x_std

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """No-op: RA dictionary replaces W_dec normalization."""
        return


class RAArchetypalSAE(RATopKSAE):
    """Backward-compatible alias for RATopKSAE (registry name: ra-ar)."""

    pass
