from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch


_NOISE_MODES = {"proportional", "fixed", "std"}


def _normalise_noise_mode(noise_mode: str) -> str:
    mode = (noise_mode or "proportional").strip().lower()
    if mode not in _NOISE_MODES:
        raise ValueError(f"noise_mode must be one of {_NOISE_MODES}, got '{noise_mode}'")
    return mode


def precompute_sae_noise_scales(
    acts_by_spec: Mapping[str, torch.Tensor],
    noise_mode: str,
) -> Dict[str, float]:
    mode = _normalise_noise_mode(noise_mode)
    if mode != "std":
        return {}
    scales: Dict[str, float] = {}
    for spec, base in acts_by_spec.items():
        if torch.is_tensor(base):
            scales[spec] = float(base.float().std().detach().cpu().item())
    return scales


def add_sae_noise(
    base: torch.Tensor,
    *,
    noise_std: float,
    noise_mode: str = "proportional",
    std_scale: Optional[float] = None,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not torch.is_tensor(base):
        raise TypeError("base must be a torch.Tensor")
    mode = _normalise_noise_mode(noise_mode)
    if noise_std <= 0:
        return base
    mask = active_mask if active_mask is not None else (base > 0)
    if not torch.is_tensor(mask) or not mask.any():
        return base
    if mode == "fixed":
        noise = torch.randn_like(base) * float(noise_std)
    elif mode == "std":
        scale = float(std_scale) if std_scale is not None else float(base.float().std().detach().cpu().item())
        if scale <= 0:
            return base
        noise = torch.randn(base.shape[-1], device=base.device, dtype=base.dtype) * (float(noise_std) * scale)
        while noise.dim() < base.dim():
            noise = noise.unsqueeze(0)
        noise = noise.expand_as(base)
    else:
        noise = torch.randn_like(base) * (float(noise_std) * base.abs())
    noisy = base + noise
    return torch.where(mask, noisy, base)


__all__ = ["add_sae_noise", "precompute_sae_noise_scales"]
