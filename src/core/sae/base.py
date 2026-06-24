# =========================  src/sae/base.py  =========================
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from .registry import register

class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models. Inspired by matryoshka_sae."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.config = cfg
        torch.manual_seed(cfg.get("seed", 42))

        # Core parameters
        act_size = cfg["act_size"]
        dict_size = cfg["dict_size"]
        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = cfg.get("dtype", torch.float32)
        
        # Handle string dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        # Initialize weights
        self.b_dec = nn.Parameter(torch.zeros(act_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(act_size, dict_size))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(dict_size, act_size))
        )
        
        # Tie weights and normalize decoder
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        
        # Dead feature tracking
        self.num_batches_not_active = torch.zeros(dict_size, device=device)

        # Feature frequency tracking (EMA)
        self.register_buffer("feature_freq_ema", torch.zeros(dict_size, device=device))
        self.freq_ema_decay = float(cfg.get("freq_ema_decay", 0.999))

        # b_norm: pre-encoder bias in normalized space (OpenAI Convention A)
        # Captures mean direction on the unit sphere so features only encode deviations.
        if cfg.get("input_global_center_norm", False):
            self.b_norm = nn.Parameter(torch.zeros(act_size))

        self.to(dtype).to(device)

        self.training = cfg.get("is_training", False)
        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None

    def _cache_input_stats(
        self,
        x_mean: Optional[torch.Tensor],
        x_std: Optional[torch.Tensor],
    ) -> None:
        def _detach(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            return value.detach()

        self.x_mean = _detach(x_mean)
        self.x_std = _detach(x_std)

    def preprocess_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess input with optional unit normalization (matryoshka_sae style)."""
        if self.config.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True).clone().detach()  # matryoshka_sae uses dim=-1
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True).clone().detach()
            x = x / (x_std + 1e-5)  # matryoshka_sae uses 1e-5
            return x, x_mean, x_std
        elif self.config.get("input_global_center_norm", False):
            x_centered = x - self.b_dec
            norms = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8).detach()
            x_normed = x_centered / norms
            # Convention A: subtract b_norm so SAE encodes deviations from sphere mean
            if hasattr(self, "b_norm"):
                x_normed = x_normed - self.b_norm
            return x_normed, norms, None
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        """Reverse preprocessing on output (matryoshka_sae style)."""
        if self.config.get("input_global_center_norm", False) and x_mean is not None:
            return x_reconstruct * x_mean + self.b_dec
        if self.config.get("input_unit_norm", False) and x_mean is not None and x_std is not None:
            return x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Normalize decoder weights to unit norm (matryoshka_sae style)."""
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        if self.W_dec.grad is not None:
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
                -1, keepdim=True
            ) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    @torch.no_grad()
    def update_inactive_features(self, acts: torch.Tensor):
        """Track inactive features for dead neuron revival (matryoshka_sae style)."""
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    @torch.no_grad()
    def update_feature_freq(self, acts: torch.Tensor):
        """Update EMA of per-feature activation frequency."""
        batch_freq = (acts > 0).float().mean(dim=0)
        self.feature_freq_ema.mul_(self.freq_ema_decay).add_(
            batch_freq, alpha=1.0 - self.freq_ema_decay
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature activations."""
        raise NotImplementedError

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations to reconstruction."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning loss dict."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        Allow loading checkpoints that omit b_enc or b_norm.
        Missing parameters default to zeros of the correct shape.
        """
        state_dict = dict(state_dict)
        if "b_enc" not in state_dict:
            state_dict["b_enc"] = torch.zeros_like(self.b_enc)
        if hasattr(self, "b_norm") and "b_norm" not in state_dict:
            state_dict["b_norm"] = torch.zeros_like(self.b_norm)
        return super().load_state_dict(state_dict, strict=strict)
