from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from ..base import BaseAutoencoder
from ..registry import register


def _rectangle(x: torch.Tensor) -> torch.Tensor:
    """Indicator used for straight-through threshold gradients."""
    return ((x > -0.5) & (x < 0.5)).to(x)


class _StepFn(torch.autograd.Function):
    """Hard step with threshold gradient for L0-style losses."""

    @staticmethod
    def forward(
        x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(ctx: Any, inputs, output) -> None:  # type: ignore[override]
        x, threshold, bandwidth = inputs
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = float(bandwidth)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = torch.sum(
            -(1.0 / bandwidth)
            * _rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return None, threshold_grad, None


class _JumpReLUFn(torch.autograd.Function):
    """
    JumpReLU with ReLU pre-activations, matching SAELens inference behaviour:
      y = relu(x) * 1[x > threshold]
    Backward propagates STE-style grads to the threshold (bandwidth controls smoothness).
    """

    @staticmethod
    def forward(
        x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        relu = F.relu(x)
        return relu * (x > threshold).to(relu)

    @staticmethod
    def setup_context(ctx: Any, inputs, output) -> None:  # type: ignore[override]
        x, threshold, bandwidth = inputs
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = float(bandwidth)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = grad_output * ((x > threshold) & (x > 0))
        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * _rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


@register("jumprelu")
class JumpReLUSAE(BaseAutoencoder):
    """
    JumpReLU SAE compatible with SAELens weights/behaviour.
    - Activation: relu(pre) * (pre > threshold)
    - Supports SAELens-style activation normalisation and optional L0 sparsity loss.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        device = self.W_enc.device
        dtype = self.W_enc.dtype
        dict_size = int(cfg["dict_size"])

        init_threshold = float(
            cfg.get(
                "threshold_init",
                cfg.get("jumprelu_init_threshold", cfg.get("threshold", 0.0)),
            )
        )
        self.threshold = nn.Parameter(
            torch.full((dict_size,), init_threshold, device=device, dtype=dtype)
        )

        # Sparsity/threshold settings
        self.bandwidth = float(cfg.get("jumprelu_bandwidth", cfg.get("bandwidth", 0.05)))
        self.sparsity_loss_mode: str = str(
            cfg.get("jumprelu_sparsity_loss_mode", cfg.get("sparsity_loss_mode", "step"))
        ).lower()
        self.l0_coeff = float(cfg.get("l0_coefficient", cfg.get("l0_coeff", 0.0)))
        self.pre_act_loss_coeff: Optional[float] = cfg.get(
            "pre_act_loss_coefficient", None
        )
        self.tanh_scale = float(cfg.get("jumprelu_tanh_scale", cfg.get("tanh_scale", 4.0)))
        self.l1_coeff = float(cfg.get("l1_coeff", cfg.get("l1", 0.0)))

        # Input handling
        self.apply_b_dec_to_input = bool(cfg.get("apply_b_dec_to_input", True))
        self.activation_norm_mode = str(
            cfg.get("normalize_activations", cfg.get("activation_norm", "none"))
        ).lower()

        # Caches for decode/postprocess
        self._norm_coeff: Optional[torch.Tensor] = None
        self._ln_mu: Optional[torch.Tensor] = None
        self._ln_std: Optional[torch.Tensor] = None

    # ---------------- activation normalisation (SAELens style) ----------------
    def _activation_norm_in(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_norm_mode == "constant_norm_rescale":
            coeff = (x.size(-1) ** 0.5) / (x.norm(dim=-1, keepdim=True) + 1e-8)
            self._norm_coeff = coeff
            return x * coeff
        if self.activation_norm_mode == "layer_norm":
            mu = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            self._ln_mu = mu
            self._ln_std = std
            return (x - mu) / (std + 1e-5)
        return x

    def _activation_norm_out(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_norm_mode == "constant_norm_rescale":
            coeff = self._norm_coeff
            if coeff is not None:
                x = x / coeff
                self._norm_coeff = None
            return x
        if self.activation_norm_mode == "layer_norm":
            if self._ln_mu is not None and self._ln_std is not None:
                x = x * self._ln_std + self._ln_mu
            self._ln_mu = None
            self._ln_std = None
            return x
        return x

    # ---------------- encode/decode ----------------
    def _preprocess_flat(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)

        x_flat = x_proc.reshape(-1, x_proc.shape[-1])
        x_norm = self._activation_norm_in(x_flat)
        bias_term = self.b_dec * (1.0 if self.apply_b_dec_to_input else 0.0)
        return orig_shape, x_norm, x_norm - bias_term, x_mean, x_std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape, x_norm, x_proc, x_mean, x_std = self._preprocess_flat(x)
        pre = x_proc @ self.W_enc + self.b_enc
        acts = _JumpReLUFn.apply(pre, self.threshold, self.bandwidth)
        if len(orig_shape) == 3:
            acts = acts.view(orig_shape[0], orig_shape[1], -1)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        orig_shape = acts.shape
        acts_flat = acts.reshape(-1, acts.shape[-1])
        recon = acts_flat @ self.W_dec + self.b_dec
        recon = self._activation_norm_out(recon)

        if len(orig_shape) == 3:
            recon = recon.view(orig_shape[0], orig_shape[1], -1)

        x_mean = getattr(self, "x_mean", torch.zeros_like(recon[:1]))
        x_std = getattr(self, "x_std", torch.ones_like(recon[:1]))
        return self.postprocess_output(recon, x_mean, x_std)

    # ---------------- forward / loss ----------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        orig_shape, x_norm, x_proc, x_mean, x_std = self._preprocess_flat(x)

        pre = x_proc @ self.W_enc + self.b_enc
        acts = _JumpReLUFn.apply(pre, self.threshold, self.bandwidth)

        recon = acts @ self.W_dec + self.b_dec
        recon = self._activation_norm_out(recon)
        if len(orig_shape) == 3:
            recon = recon.view(orig_shape[0], orig_shape[1], -1)

        self.update_inactive_features(acts)

        l2_loss = (recon.float() - x_norm.reshape_as(recon).float()).pow(2).mean()
        l1_norm = acts.abs().sum(-1).mean()
        l1_loss = self.l1_coeff * l1_norm if self.l1_coeff != 0.0 else torch.tensor(
            0.0, device=x.device
        )
        l0_loss = self._compute_sparsity_loss(acts, pre)
        pre_act_loss = self._compute_pre_act_loss(pre)

        loss = l2_loss + l1_loss + l0_loss + pre_act_loss

        sae_out = self.postprocess_output(recon, x_mean, x_std)
        return {
            "sae_out": sae_out,
            "feature_acts": acts,
            "loss": loss,
            "l2_loss": l2_loss,
            "l1_loss": l1_loss,
            "l0_loss": l0_loss,
            "pre_act_loss": pre_act_loss,
            "l1_norm": l1_norm,
            "l0_norm": (acts > 0).float().sum(-1).mean(),
            "threshold": self.threshold,
        }

    # ---------------- losses ----------------
    def _compute_sparsity_loss(
        self, acts: torch.Tensor, pre: torch.Tensor
    ) -> torch.Tensor:
        if self.l0_coeff == 0.0:
            return torch.tensor(0.0, device=acts.device)
        if self.sparsity_loss_mode == "step":
            l0 = _StepFn.apply(pre, self.threshold, self.bandwidth).sum(dim=-1)
            return self.l0_coeff * l0.mean()
        if self.sparsity_loss_mode == "tanh":
            w_norm = self.W_dec.norm(dim=1)
            per_item = torch.tanh(self.tanh_scale * acts * w_norm)
            return self.l0_coeff * per_item.sum(dim=-1).mean()
        raise ValueError(
            f"Unknown jumprelu sparsity loss mode: {self.sparsity_loss_mode}"
        )

    def _compute_pre_act_loss(self, pre: torch.Tensor) -> torch.Tensor:
        if self.pre_act_loss_coeff is None:
            return torch.tensor(0.0, device=pre.device)
        dead_mask = self.num_batches_not_active >= self.config.get(
            "n_batches_to_dead", 20
        )
        if not dead_mask.any():
            return torch.tensor(0.0, device=pre.device)
        w_norm = self.W_dec.norm(dim=1)
        loss = (
            (self.threshold - pre).relu()
            * dead_mask.to(pre.dtype)
            * w_norm.unsqueeze(0)
        ).sum(dim=-1)
        return self.pre_act_loss_coeff * loss.mean()

    # ---------------- weight normalisation ----------------
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Normalise decoder rows and project decoder grads (threshold/encoder untouched)."""
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.W_dec.grad is not None:
            grad_proj = (self.W_dec.grad * (self.W_dec / norms)).sum(
                -1, keepdim=True
            ) * (self.W_dec / norms)
            self.W_dec.grad -= grad_proj
        self.W_dec.data = self.W_dec / norms
