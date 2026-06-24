from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import torch.nn.functional as F
from ..base import BaseAutoencoder
from ..registry import register

@register("vanilla")
class VanillaSAE(BaseAutoencoder):
    """Standard sparse AE (L1 penalty)."""
    def forward(self, x):
        x, m, s = self.preprocess_input(x)
        acts = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        recon = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)

        l2 = (recon.float() - x.float()).pow(2).mean()
        l1n = acts.abs().sum(-1).mean()
        l1 = self.config["l1_coeff"] * l1n
        loss = l2 + l1
        return {
            "sae_out": self.postprocess_output(recon, m, s),
            "feature_acts": acts,
            "loss": loss,
            "l1_loss": l1,
            "l2_loss": l2,
            "l1_norm": l1n,
            "l0_norm": (acts>0).float().sum(-1).mean(),
        }
