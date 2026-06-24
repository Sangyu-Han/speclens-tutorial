from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.core.hooks.spec import parse_spec
from src.core.indexing.registry_utils import sanitize_layer_name
import torch.distributed as dist


def canonical_layer_key(spec: str) -> str:
    """Return a stable key for a layer spec (base + branch, sanitized for filenames)."""
    parsed = parse_spec(spec)
    base = parsed.base_with_branch or parsed.base
    return sanitize_layer_name(base)


@dataclass
class BaselineStat:
    mean: torch.Tensor
    count: int

    def to_payload(self) -> Dict[str, Any]:
        return {"mean": self.mean.detach().cpu(), "count": int(self.count)}


class ActivationBaselineCache:
    """
    Lightweight container for cached activation baselines.

    layers: {canonical_layer_key → {attr_name → BaselineStat}}
    """

    def __init__(
        self,
        layers: Optional[Dict[str, Dict[str, BaselineStat]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.layers = layers or {}
        self.meta = meta or {}

    # ---------------------- Serialization ----------------------
    def state_dict(self) -> Dict[str, Any]:
        layer_payload: Dict[str, Dict[str, Any]] = {}
        for key, attrs in self.layers.items():
            payload = {}
            for attr_name, stat in attrs.items():
                payload[attr_name] = stat.to_payload()
            layer_payload[key] = payload
        return {"version": 1, "layers": layer_payload, "meta": dict(self.meta)}

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)

    @classmethod
    def load(cls, path: Path | str) -> "ActivationBaselineCache":
        pkg = torch.load(Path(path), map_location="cpu")
        raw_layers = pkg.get("layers", pkg if isinstance(pkg, dict) else {})
        meta = pkg.get("meta", {}) if isinstance(pkg, dict) else {}
        layers: Dict[str, Dict[str, BaselineStat]] = {}
        if isinstance(raw_layers, dict):
            for layer_key, attrs in raw_layers.items():
                if not isinstance(attrs, dict):
                    continue
                attr_map: Dict[str, BaselineStat] = {}
                for attr_name, payload in attrs.items():
                    mean = None
                    count = -1
                    if torch.is_tensor(payload):
                        mean = payload
                        count = -1
                    elif isinstance(payload, dict):
                        if "mean" in payload and torch.is_tensor(payload["mean"]):
                            mean = payload["mean"]
                        elif "tensor" in payload and torch.is_tensor(payload["tensor"]):
                            mean = payload["tensor"]
                        elif "value" in payload and torch.is_tensor(payload["value"]):
                            mean = payload["value"]
                        raw_count = payload.get("count", payload.get("tokens", -1))
                        try:
                            count = int(raw_count)
                        except Exception:
                            count = -1
                    if torch.is_tensor(mean):
                        attr_map[str(attr_name)] = BaselineStat(mean=mean.detach().cpu(), count=int(count))
                if attr_map:
                    layers[str(layer_key)] = attr_map
        return cls(layers=layers, meta=meta)

    # ---------------------- Access helpers ----------------------
    def get(self, spec: str, attr_name: str) -> Optional[torch.Tensor]:
        key = canonical_layer_key(spec)
        attrs = self.layers.get(key)
        if not attrs:
            return None
        entry = attrs.get(attr_name)
        return entry.mean if entry is not None else None

    def broadcast(self, spec: str, attr_name: str, ref: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the baseline for (spec, attr) broadcast to the shape/dtype/device of ref."""
        base = self.get(spec, attr_name)
        if base is None or not torch.is_tensor(base):
            return None
        out = base.to(device=ref.device, dtype=ref.dtype)
        while out.dim() < ref.dim():
            out = out.unsqueeze(0)
        return out


class RunningMeanAccumulator:
    """Per-layer running mean (on CPU) for arbitrary layer/attr pairs."""

    def __init__(self) -> None:
        self._totals: Dict[str, Dict[str, torch.Tensor]] = {}
        self._counts: Dict[str, Dict[str, int]] = {}

    def update(self, spec: str, attr_name: str, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            return
        key = canonical_layer_key(spec)
        flat = tensor.detach()
        if flat.dim() == 1:
            flat = flat.unsqueeze(0)
        flat = flat.reshape(-1, flat.shape[-1]).to(torch.float64).cpu()
        total = flat.sum(dim=0)

        attr_totals = self._totals.setdefault(key, {})
        attr_counts = self._counts.setdefault(key, {})
        if attr_name not in attr_totals:
            attr_totals[attr_name] = total
            attr_counts[attr_name] = int(flat.shape[0])
        else:
            attr_totals[attr_name] = attr_totals[attr_name] + total
            attr_counts[attr_name] = int(attr_counts.get(attr_name, 0) + flat.shape[0])

    def export_state(self) -> Dict[str, Dict[str, Tuple[torch.Tensor, int]]]:
        state: Dict[str, Dict[str, Tuple[torch.Tensor, int]]] = {}
        for layer, attrs in self._totals.items():
            payload: Dict[str, Tuple[torch.Tensor, int]] = {}
            for attr, total in attrs.items():
                cnt = int(self._counts.get(layer, {}).get(attr, 0))
                payload[attr] = (total.clone(), cnt)
            if payload:
                state[layer] = payload
        return state

    @staticmethod
    def merge_states(states: Sequence[Dict[str, Dict[str, Tuple[torch.Tensor, int]]]]) -> "RunningMeanAccumulator":
        merged = RunningMeanAccumulator()
        for state in states:
            if not isinstance(state, dict):
                continue
            for layer, attrs in state.items():
                for attr, payload in (attrs or {}).items():
                    if not isinstance(payload, tuple) or len(payload) != 2:
                        continue
                    total, cnt = payload
                    if not torch.is_tensor(total):
                        continue
                    merged._totals.setdefault(layer, {})[attr] = merged._totals.get(layer, {}).get(attr, torch.zeros_like(total)) + total.cpu()
                    merged._counts.setdefault(layer, {})[attr] = int(merged._counts.get(layer, {}).get(attr, 0) + int(cnt))
        return merged

    @staticmethod
    def from_state(state: Dict[str, Dict[str, Tuple[torch.Tensor, int]]]) -> "RunningMeanAccumulator":
        return RunningMeanAccumulator.merge_states([state])

    def reduce_distributed(self) -> None:
        """All-reduce totals/counts across initialized process group."""
        if not dist.is_available() or not dist.is_initialized():
            return
        world = dist.get_world_size()
        # Gather states to rank0, merge, then broadcast back.
        state = self.export_state()
        gathered: List[Dict[str, Dict[str, Tuple[torch.Tensor, int]]]] = [None for _ in range(world)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, state)
        merged = RunningMeanAccumulator.merge_states(gathered)
        # Broadcast merged back to all ranks so they share identical stats.
        buf = merged.export_state()
        bcast_list = [buf if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(bcast_list, src=0)
        final_state = bcast_list[0] if bcast_list else {}
        final_acc = RunningMeanAccumulator.from_state(final_state)
        self._totals = final_acc._totals
        self._counts = final_acc._counts

    def to_cache(self, meta: Optional[Dict[str, Any]] = None) -> ActivationBaselineCache:
        layers: Dict[str, Dict[str, BaselineStat]] = {}
        for key, attr_totals in self._totals.items():
            counts = self._counts.get(key, {})
            attr_stats: Dict[str, BaselineStat] = {}
            for attr_name, total in attr_totals.items():
                cnt = int(counts.get(attr_name, 0))
                if cnt <= 0:
                    continue
                mean = (total / float(cnt)).to(torch.float32)
                attr_stats[attr_name] = BaselineStat(mean=mean, count=cnt)
            if attr_stats:
                layers[key] = attr_stats
        return ActivationBaselineCache(layers=layers, meta=meta or {})
