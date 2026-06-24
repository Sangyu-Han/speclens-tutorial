from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


@dataclass(frozen=True)
class ActivationRecord:
    frame_idx: int
    tensor: torch.Tensor

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        shape = tuple(int(x) for x in self.tensor.shape)
        dev = str(self.tensor.device)
        req = self.tensor.requires_grad
        return f"ActivationRecord(frame_idx={self.frame_idx}, shape={shape}, device={dev}, requires_grad={req})"


class ActivationTape:
    """
    Keeps a chronological list of activations while preserving the computation graph.

    The tape is intentionally minimal: it does not detach tensors or perform any
    aggregation unless explicitly requested.  This allows downstream code to build
    objectives over entire frame stacks (e.g. frames×units vectors) without losing
    gradients.
    """

    def __init__(self) -> None:
        self._records: List[ActivationRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def append(self, frame_idx: int, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError("ActivationTape only accepts torch.Tensor entries")
        self._records.append(ActivationRecord(frame_idx=frame_idx, tensor=tensor))

    def latest(self) -> torch.Tensor:
        if not self._records:
            raise RuntimeError("ActivationTape is empty; nothing to retrieve")
        return self._records[-1].tensor

    def frames(self) -> Sequence[ActivationRecord]:
        return tuple(self._records)

    def clear(self) -> None:
        self._records.clear()

    def frame_count(self) -> int:
        return len(self._records)

    def as_stack(self, *, detach: bool = False, squeeze_single: bool = False) -> torch.Tensor:
        """
        Returns tensors stacked along a new time dimension.

        The stack preserves gradients by default; set detach=True when a
        gradient-free snapshot is desired (e.g., serialisation).
        """
        if not self._records:
            raise RuntimeError("ActivationTape is empty; cannot stack.")
        tensors: Iterable[torch.Tensor]
        if detach:
            tensors = [rec.tensor.detach() for rec in self._records]
        else:
            tensors = [rec.tensor for rec in self._records]
        stack = torch.stack(list(tensors), dim=0)
        if squeeze_single and stack.shape[0] == 1:
            return stack.squeeze(0)
        return stack

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        if not self._records:
            return "ActivationTape(len=0)"
        head = self._records[0]
        tail = self._records[-1]
        return f"ActivationTape(len={len(self._records)}, first={head}, last={tail})"
