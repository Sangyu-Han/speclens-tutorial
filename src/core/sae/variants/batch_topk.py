# src/sae/variants/batch_topk.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ..base import BaseAutoencoder
from ..registry import register

@register("batch-topk")
class BatchTopKSAE(BaseAutoencoder):
    """
    Global-batch Top-K SAE.

    Training: select global Top-(B·k) activations per *batch* (k = per-sample target),
              and update a threshold to track the (B·k)-th largest activation.
    Eval:     gate with the learned threshold (frozen) so expected sparsity ≈ k.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.k        = int(cfg["k"])
        self.k_aux    = int(cfg.get("k_aux", 512))
        self.aux_frac = cfg.get("aux_frac", 1/32)
        # eval-time threshold (registered buffer; keep in-place updates)
        self.register_buffer("threshold", torch.tensor(0.0))
        self._gating_mode: str = "hard"
        self._gating_negative_slope: float = 0.0
        self._gating_temperature: float = 1.0
        self._last_pre_acts: Optional[torch.Tensor] = None

    # ---------- helpers ----------
    def _topk_mask(self, acts: torch.Tensor) -> torch.Tensor:
        """Return boolean mask for global Top-(B·k) over the whole batch."""
        flat = acts.reshape(-1)
        B = acts.size(0)
        k_total = int(min(self.k * B, flat.numel()))
        if k_total <= 0:
            return torch.zeros_like(acts, dtype=torch.bool)
        topk = torch.topk(flat, k_total, sorted=False)
        mask_flat = torch.zeros_like(flat, dtype=torch.bool)
        mask_flat[topk.indices] = True
        return mask_flat.view_as(acts)

    def configure_visualization_gating(
        self,
        *,
        mode: str = "hard",
        negative_slope: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Configure gating behaviour for feature-visualisation use cases."""
        mode = mode.lower()
        if mode not in {"hard", "ste", "annealed", "dict"}:
            raise ValueError(
                f"Unsupported gating mode '{mode}' (expected 'hard', 'ste', 'annealed', or 'dict')"
            )
        self._gating_mode = mode
        if negative_slope is not None:
            self._gating_negative_slope = max(0.0, float(negative_slope))
        if temperature is not None:
            self._gating_temperature = max(1e-6, float(temperature))

    def set_gating_temperature(self, temperature: float) -> None:
        """Adjust the temperature used by the annealed sigmoid gate."""
        self._gating_temperature = max(1e-6, float(temperature))

    @torch.no_grad()
    def _update_threshold_from_topk(self, relu: torch.Tensor, lr: float = 0.05) -> None:
        """
        During training, set EMA target to the (B·k)-th largest activation
        (minimum among the Top-(B·k)), then update threshold in-place.
        """
        flat = relu.reshape(-1)
        B = relu.size(0)
        k_total = int(min(self.k * B, flat.numel()))
        if k_total <= 0:
            return
        kth_min = torch.topk(flat, k_total, sorted=False).values.min()
        # in-place EMA to preserve buffer semantics
        self.threshold.lerp_(kth_min, lr)

    # ---------- encode/decode ----------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return sparse feature activations (gated ReLU).
        - If self.training: apply global batch Top-(B·k) mask (no threshold update here).
        - If not training:  apply fixed threshold.
        Handles 2D [B, D] and 3D [T, B, D] inputs by flattening over the leading dims.
        """
        self._last_pre_acts = None
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])

        if self.config.get("input_global_center_norm", False):
            pre = x_flat @ self.W_enc + self.b_enc
        else:
            pre = (x_flat - self.b_dec) @ self.W_enc + self.b_enc       # [B*, N]
        pre_out = pre
        if self.training:
            # Global Top-(B·k) over the *current* flattened batch
            B_eff = pre.size(0)
            flat = pre.reshape(-1)
            k_total = int(min(self.k * B_eff, flat.numel()))
            if k_total > 0:
                topk = torch.topk(flat, k_total, sorted=False)
                acts_flat = torch.zeros_like(flat)
                acts_flat.scatter_(0, topk.indices, topk.values.relu())
                acts = acts_flat.view_as(pre)
            else:
                acts = torch.zeros_like(pre)
        else:
            activations = pre
            if self._gating_mode == "dict":
                acts = activations
            elif self._gating_mode == "annealed":
                threshold = self.threshold.to(device=activations.device, dtype=activations.dtype)
                temp = max(self._gating_temperature, 1e-6)
                gate = torch.sigmoid((activations - threshold) / temp)
                acts = activations * gate
            else:
                mask = activations > self.threshold
                acts = activations * mask
                if self._gating_mode == "ste":
                    acts = acts + (activations - activations.detach())
                acts = acts.relu()

        # reshape back if input was 3D
        if len(orig_shape) == 3:
            pre_out = pre_out.view(orig_shape[0], orig_shape[1], -1)
            acts = acts.view(orig_shape[0], orig_shape[1], -1)
        self._last_pre_acts = pre_out
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Map sparse features back to input space and apply postprocess_output.
        Uses stored x_mean/x_std if available; otherwise falls back to zeros/ones.
        Handles 2D [B, N] and 3D [T, B, N] feature tensors.
        """
        orig_shape = acts.shape
        acts_flat = acts.reshape(-1, acts.shape[-1])

        _gcenter = self.config.get("input_global_center_norm", False)
        if _gcenter:
            recon = acts_flat @ self.W_dec
            if hasattr(self, "b_norm"):
                recon = recon + self.b_norm
        else:
            recon = acts_flat @ self.W_dec + self.b_dec

        # Try to use the most recent normalization stats if the base class keeps them.
        x_mean = getattr(self, 'x_mean', torch.zeros_like(recon[:1]))
        x_std  = getattr(self, 'x_std',  torch.ones_like(recon[:1]))
        out = self.postprocess_output(recon, x_mean, x_std)

        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)
        return out

    # ---------- forward ----------
    def forward(self, x):
        self._last_pre_acts = None
        _gcenter = self.config.get("input_global_center_norm", False)
        x, m, s = self.preprocess_input(x)
        self._cache_input_stats(m, s)
        # When input_global_center_norm=True, preprocess already removed b_dec,
        # so do NOT subtract it again in the encoder.
        if _gcenter:
            pre = x @ self.W_enc + self.b_enc
        else:
            pre = (x - self.b_dec) @ self.W_enc + self.b_enc              # [B, N]
        self._last_pre_acts = pre
        activations = pre

        if self.training:
            mask = self._topk_mask(pre)
            acts = pre * mask
            acts = acts.relu()
        else:
            if self._gating_mode == "dict":
                acts = activations
            elif self._gating_mode == "annealed":
                threshold = self.threshold.to(device=activations.device, dtype=activations.dtype)
                temp = max(self._gating_temperature, 1e-6)
                gate = torch.sigmoid((activations - threshold) / temp)
                acts = activations * gate
            else:
                mask = activations > self.threshold
                acts = activations * mask
                if self._gating_mode == "ste":
                    acts = acts + (activations - activations.detach())
                acts = acts.relu()

        # Reconstruction stays in preprocessed space; b_dec added back via postprocess.
        if _gcenter:
            recon = acts @ self.W_dec
        else:
            recon = acts @ self.W_dec + self.b_dec

        if self.training and self._gating_mode != "dict":
            self._update_threshold_from_topk(acts)            # train-only
            self.update_inactive_features(acts)               # train-only
            self.update_feature_freq(acts)                    # for high_frac logging

        # Add b_norm back before postprocessing (mirrors RA convention).
        if _gcenter and hasattr(self, "b_norm"):
            recon_out = recon + self.b_norm
            x_for_post = x + self.b_norm
        else:
            recon_out = recon
            x_for_post = x

        # --- extra stats for logging (optional) ---
        with torch.no_grad():
            var_x = x.float().var()
            var_resid = (recon.float() - x.float()).var()
            explained_var = 1.0 - (var_resid / (var_x + 1e-8))
            # relative_l2 in original input space
            sae_out_f = self.postprocess_output(recon_out.float(), m, s)
            x_orig_f  = self.postprocess_output(x_for_post.float(), m, s)
            per_sample_resid_sq = (x_orig_f - sae_out_f).pow(2).sum(-1)
            per_sample_x_sq     = x_orig_f.pow(2).sum(-1)
            relative_l2 = (per_sample_resid_sq / (per_sample_x_sq + 1e-8)).mean()
            k_eff_now = (acts > 0).float().sum(dim=-1).mean()
            pos_mean_now = (
                acts[acts > 0].float().mean()
                if (acts > 0).any()
                else torch.tensor(0.0, device=x.device)
            )

        l2  = (recon.float() - x.float()).pow(2).mean()
        l1n = acts.abs().sum(-1).mean()
        l1  = self.config["l1_coeff"] * l1n

        # --- auxiliary loss for dead columns --------------
        aux = torch.tensor(0.0, device=x.device)
        dead = self.num_batches_not_active >= self.config["n_batches_to_dead"]
        if dead.any():
            resid = x.float() - recon.float()
            num_dead = int(dead.sum().item())
            k_aux = min(self.k_aux, num_dead)
            if k_aux > 0:
                top_aux = torch.topk(activations[:, dead], k_aux, dim=-1, sorted=False)
                aux_mask = torch.zeros_like(activations[:, dead]).scatter(-1, top_aux.indices, 1.0)
                aux_acts = activations[:, dead] * aux_mask
                recon_aux = aux_acts @ self.W_dec[dead]
                aux = self.aux_frac * (recon_aux - resid).pow(2).mean()

        loss = l2 + l1 + aux

        # --- Track batch activation stats for fold ---
        if self.training:
            with torch.no_grad():
                batch_std = acts.std(dim=0)
                self._batch_act_mean = acts.mean(dim=0).detach()
                self._batch_act_cv = batch_std / (self._batch_act_mean.abs() + 1e-8)

        # --- Reanimation loss ---
        reanim_coeff = float(self.config.get("reanim_coeff", 0.0))
        if reanim_coeff > 0:
            batch_dead = ((acts > 0).sum(dim=0) == 0).float().detach()
            reanim_loss = (pre * batch_dead[None, :]).mean()
            loss = loss - reanim_coeff * reanim_loss

        # --- Frequency penalty ---
        freq_coeff = float(self.config.get("freq_penalty_coeff", 0.0))
        freq_loss = torch.tensor(0.0, device=x.device)
        if freq_coeff > 0:
            freq_weight = self.feature_freq_ema.detach()
            freq_loss = freq_coeff * (freq_weight * acts).sum(-1).mean()
            loss = loss + freq_loss

        # --- Spatial variance penalty ---
        spatial_var_coeff = float(self.config.get("spatial_var_penalty", 0.0))
        spatial_var_loss = torch.tensor(0.0, device=x.device)
        n_high_freq = 0
        if spatial_var_coeff > 0:
            freq_threshold = float(self.config.get("spatial_var_freq_threshold", 0.9))
            high_freq = (self.feature_freq_ema > freq_threshold)
            n_high_freq = int(high_freq.sum().item())
            if high_freq.any():
                bias_acts = acts[:, high_freq]
                var_per_feat = bias_acts.var(dim=0)
                spatial_var_loss = spatial_var_coeff * var_per_feat.mean()
                loss = loss + spatial_var_loss

        # --- Orthogonality penalty (proactive anti-alignment prevention) ---
        # Penalises W_enc columns that are anti-aligned with b_dec (or b_norm in gcenter).
        # This fires from the very first gradient step (as soon as b_dec accumulates direction),
        # preventing the anti-alignment that creates constant boosts BEFORE they form.
        # Scale-invariant: uses normalized directions, so coefficient is easy to tune.
        # penalty = relu(-cos(center, W_enc[:,i])).mean()  over all features
        orth_coeff = float(self.config.get("orth_penalty_coeff", 0.0))
        orth_loss = torch.tensor(0.0, device=x.device)
        if orth_coeff > 0:
            if _gcenter and hasattr(self, "b_norm"):
                center_orth = self.b_norm
            else:
                center_orth = self.b_dec
            c_norm = center_orth.norm()
            if c_norm > 1e-6:
                c_n = center_orth / c_norm                               # [D]
                W_n = F.normalize(self.W_enc, dim=0)                     # [D, N]
                orth_loss = orth_coeff * F.relu(-(c_n @ W_n)).mean()
                loss = loss + orth_loss

        # --- Constant boost penalty ---
        # Directly penalizes the input-independent (constant) pre-activation component.
        # const_boost[i] = b_enc[i] - center @ W_enc[:, i]
        # where center = b_norm (gcenter mode) or b_dec (standard mode)
        # Positive const_boost means feature i fires on every token regardless of content.
        # Gradient flows through b_enc and W_enc, pulling them apart from the center direction.
        cb_coeff = float(self.config.get("const_boost_penalty", 0.0))
        cb_loss = torch.tensor(0.0, device=x.device)
        if cb_coeff > 0:
            if _gcenter and hasattr(self, "b_norm"):
                center = self.b_norm
            else:
                center = self.b_dec
            const_boost = self.b_enc - center @ self.W_enc   # [N]
            cb_loss = cb_coeff * const_boost.clamp(min=0).mean()
            loss = loss + cb_loss

        # --- Bias explained variance penalty ---
        bev_coeff = float(self.config.get("bev_penalty", 0.0))
        bev_loss = torch.tensor(0.0, device=x.device)
        if bev_coeff > 0:
            bev_freq_thr = float(self.config.get("spatial_var_freq_threshold", 0.9))
            bev_high = (self.feature_freq_ema > bev_freq_thr).detach()
            if bev_high.any():
                bev_acts = acts.clone()
                bev_acts[:, ~bev_high] = 0
                bev_recon = bev_acts @ self.W_dec
                bev_var_x = x.float().var().detach()
                bev_var_resid = (x.float() - bev_recon.float()).var()
                raw_bev = 1.0 - bev_var_resid / (bev_var_x + 1e-8)
                bev_loss = bev_coeff * torch.clamp(raw_bev, min=0.0)
                loss = loss + bev_loss

        return {
            "sae_out": self.postprocess_output(recon_out, m, s),
            "feature_acts": acts,
            "loss": loss,
            "l1_loss": l1,
            "l2_loss": l2,
            "aux_loss": aux,
            "spatial_var_loss": spatial_var_loss,
            "freq_loss": freq_loss,
            "bev_loss": bev_loss,
            "cb_loss": cb_loss,
            "orth_loss": orth_loss,
            "n_high_freq": n_high_freq,
            "threshold": self.threshold,
            "explained_var": explained_var,
            "relative_l2": relative_l2,
            "k_eff": k_eff_now,
            "pos_act_mean": pos_mean_now,
        }

    def fold_bias_features(
        self,
        freq_threshold: float = 0.95,
        cv_threshold: float = 0.02,
    ) -> int:
        """Fold constant-activation (bias) features into b_dec (or b_norm) and resample."""
        if not hasattr(self, "_batch_act_mean"):
            return 0

        freq = self.feature_freq_ema
        mean_act = self._batch_act_mean
        cv = self._batch_act_cv

        constant = (freq > freq_threshold) & (cv < cv_threshold) & (mean_act > 0)
        if not constant.any():
            return 0

        n_folded = 0
        for idx in constant.nonzero().squeeze(-1):
            i = idx.item()
            a_i = mean_act[i].item()
            d_i = self.W_dec[i]
            contribution = a_i * d_i

            # Absorb into b_norm if available (input_global_center_norm), else b_dec
            if hasattr(self, "b_norm"):
                self.b_norm.data.add_(contribution)
            else:
                self.b_dec.data.add_(contribution)

            # Compensate b_enc so pre-acts for other features are unchanged
            self.b_enc.data.add_(contribution @ self.W_enc)

            # Re-initialise encoder column (Kaiming)
            fan_in = self.W_enc.shape[0]
            std = (2.0 / fan_in) ** 0.5
            self.W_enc.data[:, i].normal_(0, std)
            self.b_enc.data[i] = 0.0

            # Reset tracking
            self.num_batches_not_active[i] = 0
            self.feature_freq_ema[i] = 0.0

            n_folded += 1

        return n_folded
