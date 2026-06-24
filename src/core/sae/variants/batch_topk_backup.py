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
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])

        pre  = (x_flat - self.b_dec) @ self.W_enc          # [B*, N]
        relu = F.relu(pre)

        if self.training:
            # Global Top-(B·k) over the *current* flattened batch
            # We need B = batch size along first dim of x_flat
            B_eff = relu.size(0)
            flat = relu.reshape(-1)
            k_total = int(min(self.k * B_eff, flat.numel()))
            if k_total > 0:
                topk = torch.topk(flat, k_total, sorted=False)
                mask_flat = torch.zeros_like(flat, dtype=torch.bool)
                mask_flat[topk.indices] = True
                mask = mask_flat.view_as(relu)
            else:
                mask = torch.zeros_like(relu, dtype=torch.bool)
        else:
            mask = relu > self.threshold

        acts = relu * mask

        # reshape back if input was 3D
        if len(orig_shape) == 3:
            acts = acts.view(orig_shape[0], orig_shape[1], -1)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Map sparse features back to input space and apply postprocess_output.
        Uses stored x_mean/x_std if available; otherwise falls back to zeros/ones.
        Handles 2D [B, N] and 3D [T, B, N] feature tensors.
        """
        orig_shape = acts.shape
        acts_flat = acts.reshape(-1, acts.shape[-1])

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
        x, m, s = self.preprocess_input(x)
        self._cache_input_stats(m, s)
        pre  = (x - self.b_dec) @ self.W_enc                  # [B, N]
        relu = F.relu(pre)

        if self.training:
            mask = self._topk_mask(relu)
        else:
            mask = relu > self.threshold

        acts  = relu * mask
        recon = acts @ self.W_dec + self.b_dec

        if self.training:
            self._update_threshold_from_topk(relu)            # train-only
            self.update_inactive_features(acts)               # train-only

        # --- extra stats for logging (optional) ---
        with torch.no_grad():
            var_x = x.float().var()
            var_resid = (recon.float() - x.float()).var()
            explained_var = 1.0 - (var_resid / (var_x + 1e-8))
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
                top_aux = torch.topk(relu[:, dead], k_aux, dim=-1, sorted=False)
                aux_mask = torch.zeros_like(relu[:, dead]).scatter(-1, top_aux.indices, 1.0)
                aux_acts = relu[:, dead] * aux_mask
                recon_aux = aux_acts @ self.W_dec[dead]
                aux = self.aux_frac * (recon_aux - resid).pow(2).mean()

        loss = l2 + l1 + aux
        return {
            "sae_out": self.postprocess_output(recon, m, s),
            "feature_acts": acts,
            "loss": loss,
            "l1_loss": l1,
            "l2_loss": l2,
            "aux_loss": aux,
            "threshold": self.threshold,
            "explained_var": explained_var,
            "k_eff": k_eff_now,
            "pos_act_mean": pos_mean_now,
        }
