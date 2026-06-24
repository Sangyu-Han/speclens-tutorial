from __future__ import annotations

from typing import Callable, Optional, Protocol, Union

import torch


class InterventionFn(Protocol):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        ...


def identity_intervention(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def zero_intervention(tensor: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(tensor)


def scale_intervention(
    tensor: torch.Tensor,
    *,
    scale: Union[float, torch.Tensor],
    bias: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    scale_tensor = torch.as_tensor(scale, dtype=tensor.dtype, device=tensor.device)
    out = tensor * scale_tensor
    if bias is not None:
        bias_tensor = torch.as_tensor(bias, dtype=tensor.dtype, device=tensor.device)
        out = out + bias_tensor
    return out


def build_scale_intervention(
    *,
    scale: Union[float, torch.Tensor],
    bias: Optional[Union[float, torch.Tensor]] = None,
) -> InterventionFn:
    def _fn(tensor: torch.Tensor) -> torch.Tensor:
        return scale_intervention(tensor, scale=scale, bias=bias)

    return _fn


def build_replace_intervention(replacement: torch.Tensor) -> InterventionFn:
    replacement = replacement.clone()

    def _fn(tensor: torch.Tensor) -> torch.Tensor:
        target = replacement.to(dtype=tensor.dtype, device=tensor.device)
        if target.shape == tensor.shape:
            return target
        raise RuntimeError(
            f"Replacement tensor shape {tuple(target.shape)} "
            f"does not match target view {tuple(tensor.shape)}"
        )

    return _fn
