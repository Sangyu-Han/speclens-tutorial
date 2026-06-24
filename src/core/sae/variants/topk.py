from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ..base import BaseAutoencoder
from ..registry import register

@register("topk")
class TopKSAE(BaseAutoencoder):
    """TopK SAE with auxiliary loss for dead feature recovery."""
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc

        # TopK selection on pre-activations, then ReLU the kept values (SAELens-style)
        tk = torch.topk(pre, self.config["top_k"], dim=-1)
        acts_topk = torch.zeros_like(pre).scatter(-1, tk.indices, tk.values.relu())

        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        self.update_inactive_features(acts_topk)
        return self.get_loss_dict(x, x_reconstruct, pre, acts_topk, x_mean, x_std)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        tk = torch.topk(pre, self.config["top_k"], dim=-1)
        return torch.zeros_like(pre).scatter(-1, tk.indices, tk.values.relu())

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        recon = acts @ self.W_dec + self.b_dec
        x_mean = getattr(self, "x_mean", torch.zeros_like(recon[:1]))
        x_std = getattr(self, "x_std", torch.ones_like(recon[:1]))
        return self.postprocess_output(recon, x_mean, x_std)

    def get_loss_dict(self, x: torch.Tensor, x_reconstruct: torch.Tensor, pre: torch.Tensor,
                      acts_topk: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> Dict[str, torch.Tensor]:
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.config["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, pre)
        loss = l2_loss + l1_loss + aux_loss

        num_dead_features = (self.num_batches_not_active > self.config.get("n_batches_to_dead", 20)).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        # relative_l2: ||x_orig - sae_out||^2 / ||x_orig||^2  (SAELens-style, in original space)
        with torch.no_grad():
            x_orig = self.postprocess_output(x.float(), x_mean, x_std)
            per_sample_resid_sq = (x_orig - sae_out.float()).pow(2).sum(-1)
            per_sample_x_sq = x_orig.pow(2).sum(-1)
            relative_l2 = (per_sample_resid_sq / (per_sample_x_sq + 1e-8)).mean()

        return {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "relative_l2": relative_l2,
        }

    def get_auxiliary_loss(self, x: torch.Tensor, x_reconstruct: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss for dead feature recovery."""
        dead_features = self.num_batches_not_active >= self.config.get("n_batches_to_dead", 20)
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                pre[:, dead_features],
                min(self.config.get("top_k_aux", 512), dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(pre[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values.relu()
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.config.get("aux_penalty", 0.03125)
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        return torch.tensor(0, dtype=x.dtype, device=x.device)
