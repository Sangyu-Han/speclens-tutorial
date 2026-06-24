from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Protocol, Sequence, Tuple, Union

import torch

IndexLike = Union[int, Sequence[int], torch.Tensor]


class SelectionFn(Protocol):
    """
    Optional pluggable selector supplied by advanced controllers.

    It must return a tuple containing:
    - the selected tensor view to override
    - a callable that restores the modified view back into the full tensor
    """

    def __call__(
        self,
        tensor: torch.Tensor,
        *,
        reshape_meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        ...


@dataclass
class OverrideSpec:
    """
    Describes which portion of a tensor should be overridden.

    The controller operates on (lanes, positions, features) logical layouts.
    Positions can be token indices or spatial coordinates; features correspond
    to latent units/channels.
    """

    lane_idx: Optional[IndexLike] = 0
    token_indices: Optional[IndexLike] = None
    spatial_y: Optional[IndexLike] = None
    spatial_x: Optional[IndexLike] = None
    unit_indices: Optional[IndexLike] = None
    position_indices: Optional[IndexLike] = None
    target_frame_idx: Optional[int] = None
    selector: Optional[SelectionFn] = None

    def clone_with_frame(self, frame_idx: Optional[int]) -> "OverrideSpec":
        spec = OverrideSpec(
            lane_idx=self.lane_idx,
            token_indices=self.token_indices,
            spatial_y=self.spatial_y,
            spatial_x=self.spatial_x,
            unit_indices=self.unit_indices,
            position_indices=self.position_indices,
            target_frame_idx=frame_idx,
            selector=self.selector,
        )
        return spec


def _to_long_tensor(indices: IndexLike, *, size: int, name: str) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        values = indices.to(dtype=torch.long).view(-1)
    elif isinstance(indices, Sequence):
        values = torch.tensor(list(indices), dtype=torch.long)
    else:
        values = torch.tensor([int(indices)], dtype=torch.long)
    if values.numel() == 0:
        raise ValueError(f"{name} is empty")
    clipped = values.clamp(min=0, max=size - 1)
    return clipped


def normalise_indices(
    indices: Optional[IndexLike],
    *,
    size: int,
    name: str,
    default_all: bool = False,
) -> Optional[torch.Tensor]:
    if indices is None:
        if default_all:
            return torch.arange(size, dtype=torch.long)
        return None
    return _to_long_tensor(indices, size=size, name=name)


def infer_grid_shape(
    *,
    explicit: Optional[Tuple[int, int]],
    positions: int,
) -> Optional[Tuple[int, int]]:
    if explicit:
        return explicit
    side = int(round(float(positions) ** 0.5))
    if side * side == positions:
        return (side, side)
    return None
