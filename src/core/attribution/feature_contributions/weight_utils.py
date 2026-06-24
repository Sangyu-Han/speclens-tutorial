from __future__ import annotations

from typing import Callable, Optional

import torch


def build_unit_weight_builder(
    *,
    method: str,
    unit: int,
    weight_multiplier: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Construct a weight builder for forward feature contribution paths.

    Returns a function that, given latent activations, produces per-token
    column values (1-D) for the unit index.  The caller constructs the full
    JVP tangent matrix from these column values, keeping memory usage low.

    - IG/grad/input_x_grad: column values are 1.0 (delta scaling by caller).
    - ig_conductance: column values are the latent activation at the unit.
    """
    method_key = (method or "ig").lower()
    use_conductance = method_key in {"ig_conductance", "conductance"}

    def _builder(latent: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(latent):
            raise TypeError("weight_builder expects a tensor input")
        last_dim = latent.shape[-1]
        if last_dim <= unit:
            raise RuntimeError(
                f"Weight builder latent dim {last_dim} is smaller than unit index {unit}"
            )
        index = (slice(None),) * (latent.dim() - 1) + (unit,)
        # Compute per-token column values (small 1-D tensor, NOT a full matrix)
        col_val = latent.detach()[index] if use_conductance else torch.ones(
            latent.shape[:-1], device=latent.device, dtype=latent.dtype,
        )
        if weight_multiplier is not None:
            extra = weight_multiplier(latent)
            if not torch.is_tensor(extra):
                raise TypeError("weight_multiplier must return a tensor")
            # Extract per-token multiplier values for the unit column.
            # Supports both full (N, D) matrices and compact 1D per-token vectors.
            if extra.shape == latent.shape:
                col_mult = extra[index]
            else:
                col_mult = extra.reshape(col_val.shape)
            col_val = col_val * col_mult.to(device=col_val.device, dtype=col_val.dtype)
            del extra, col_mult
        return col_val

    return _builder


__all__ = ["build_unit_weight_builder"]
