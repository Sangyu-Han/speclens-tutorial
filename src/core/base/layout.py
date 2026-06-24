"""
LayoutSpec – shape-based provenance engine for SAE activation stores.

Phase 1: replaces per-model build_token_provenance with a generic,
shape-driven implementation. Each activation tensor's semantic dimensions
are described by a tuple of DimRole values; provenance columns are derived
automatically.

Example
-------
    # 4D conv feature map (B, C, H, W)
    spec = LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))
    prov = build_provenance_from_layout(spec, tensor, sample_ids)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

# ============================================================ #
#  DimRole enum                                                 #
# ============================================================ #

class DimRole(Enum):
    """Semantic role for each tensor dimension."""
    BATCH   = auto()   # → sample_id
    TIME    = auto()   # → frame_idx
    LANE    = auto()   # → uid (multi-object)
    HEIGHT  = auto()   # → y
    WIDTH   = auto()   # → x
    TOKEN   = auto()   # → y=-1, x=token_idx
    FEATURE = auto()   # encoding dim, not provenance


# Default column names per role
_DEFAULT_COL_NAMES: Dict[DimRole, str] = {
    DimRole.BATCH:  "sample_id",
    DimRole.TIME:   "frame_idx",
    DimRole.LANE:   "uid",
    DimRole.HEIGHT: "y",
    DimRole.WIDTH:  "x",
    DimRole.TOKEN:  "token_idx",  # TOKEN expands to (y, x) in provenance
}

# ============================================================ #
#  LayoutSpec dataclass                                         #
# ============================================================ #

@dataclass(frozen=True)
class LayoutSpec:
    """Declarative description of a tensor's dimension layout.

    Parameters
    ----------
    dims : Tuple[DimRole, ...]
        One role per tensor dimension, in order.
    enrich_fn : callable, optional
        ``enrich_fn(columns_dict, dim_indices_dict)`` may add/modify
        columns in-place (e.g. mapping lane index → uid).
    col_name_overrides : dict, optional
        Override default column names, e.g. ``{DimRole.TIME: "t"}``.
    """
    dims: Tuple[DimRole, ...]
    enrich_fn: Optional[Callable] = None
    col_name_overrides: Optional[Dict[DimRole, str]] = field(default=None)

    # --- derived helpers ---

    @property
    def feature_axis(self) -> int:
        """Index of the FEATURE dimension."""
        for i, d in enumerate(self.dims):
            if d is DimRole.FEATURE:
                return i
        raise ValueError("LayoutSpec has no FEATURE dimension")

    @property
    def prefix_roles(self) -> Tuple[DimRole, ...]:
        """All non-FEATURE roles, preserving order."""
        return tuple(d for d in self.dims if d is not DimRole.FEATURE)

    def _col_name(self, role: DimRole) -> str:
        if self.col_name_overrides and role in self.col_name_overrides:
            return self.col_name_overrides[role]
        return _DEFAULT_COL_NAMES.get(role, role.name.lower())

    def provenance_columns(self) -> Tuple[str, ...]:
        """Derive provenance column names from roles (in order, no duplicates).

        TOKEN expands to ``("y", "x")`` so that the provenance format
        stays compatible with spatial activations.
        """
        seen: set = set()
        cols: List[str] = []
        for role in self.dims:
            if role is DimRole.FEATURE:
                continue
            if role is DimRole.TOKEN:
                for name in ("y", "x"):
                    if name not in seen:
                        cols.append(name)
                        seen.add(name)
            else:
                name = self._col_name(role)
                if name not in seen:
                    cols.append(name)
                    seen.add(name)
        return tuple(cols)


# ============================================================ #
#  Inference / parsing helpers                                  #
# ============================================================ #

def infer_layout(tensor: torch.Tensor) -> Optional[LayoutSpec]:
    """Guess LayoutSpec from tensor ndim using common conventions.

    Returns ``None`` for 5-D+ tensors (needs explicit config).
    """
    ndim = tensor.ndim
    if ndim == 2:
        # (BATCH, FEATURE)
        return LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE))
    if ndim == 3:
        # (BATCH, TOKEN, FEATURE)
        return LayoutSpec(dims=(DimRole.BATCH, DimRole.TOKEN, DimRole.FEATURE))
    if ndim == 4:
        # (BATCH, FEATURE, HEIGHT, WIDTH) – PyTorch NCHW convention
        return LayoutSpec(dims=(DimRole.BATCH, DimRole.FEATURE, DimRole.HEIGHT, DimRole.WIDTH))
    return None  # 5D+ → explicit config required


# Alias table for parse_layout_spec
_ROLE_ALIASES: Dict[str, DimRole] = {
    "b":       DimRole.BATCH,
    "batch":   DimRole.BATCH,
    "t":       DimRole.TIME,
    "time":    DimRole.TIME,
    "frame":   DimRole.TIME,
    "lane":    DimRole.LANE,
    "h":       DimRole.HEIGHT,
    "height":  DimRole.HEIGHT,
    "y":       DimRole.HEIGHT,
    "w":       DimRole.WIDTH,
    "width":   DimRole.WIDTH,
    "x":       DimRole.WIDTH,
    "tok":     DimRole.TOKEN,
    "token":   DimRole.TOKEN,
    "seq":     DimRole.TOKEN,
    "c":       DimRole.FEATURE,
    "channel": DimRole.FEATURE,
    "feat":    DimRole.FEATURE,
    "feature": DimRole.FEATURE,
}


def parse_layout_spec(roles: List[str]) -> LayoutSpec:
    """Parse a YAML-style role list into a LayoutSpec.

    Example
    -------
    >>> parse_layout_spec(["batch", "time", "feature", "h", "w"])
    LayoutSpec(dims=(BATCH, TIME, FEATURE, HEIGHT, WIDTH))
    """
    parsed: List[DimRole] = []
    for r in roles:
        key = r.strip().lower()
        if key not in _ROLE_ALIASES:
            raise ValueError(
                f"Unknown DimRole alias '{r}'. "
                f"Valid aliases: {sorted(_ROLE_ALIASES.keys())}"
            )
        parsed.append(_ROLE_ALIASES[key])
    return LayoutSpec(dims=tuple(parsed))


# ============================================================ #
#  Core provenance builder                                      #
# ============================================================ #

def build_provenance_from_layout(
    spec: LayoutSpec,
    raw_output: torch.Tensor,
    sample_ids: Optional[torch.Tensor] = None,
    fidx_hint: Optional[Union[int, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a (N, num_cols) int64 CPU provenance tensor from LayoutSpec.

    Parameters
    ----------
    spec : LayoutSpec
        Dimension layout of *raw_output*.
    raw_output : torch.Tensor
        The activation tensor **before** flattening (original shape).
    sample_ids : torch.Tensor, optional
        Per-batch-element sample IDs.  Shape ``(B,)`` where ``B`` matches the
        BATCH dimension of *raw_output*.  Falls back to ``arange(B)`` when
        ``None``.
    fidx_hint : int | Tensor, optional
        Frame index used when TIME is not in *spec.dims* but the provenance
        schema expects ``frame_idx``.
    device : torch.device, optional
        Ignored (provenance is always CPU int64), kept for API compat.
    """
    shape = raw_output.shape

    if len(spec.dims) != raw_output.ndim:
        raise ValueError(
            f"LayoutSpec has {len(spec.dims)} dims but tensor has {raw_output.ndim} dims "
            f"(spec={spec.dims}, shape={tuple(shape)})"
        )

    # --- prefix_shape: shape with feature_axis removed ---
    feat_ax = spec.feature_axis
    prefix_roles = spec.prefix_roles
    prefix_shape = tuple(
        int(shape[i]) for i in range(len(spec.dims)) if i != feat_ax
    )
    assert len(prefix_roles) == len(prefix_shape), (
        f"prefix_roles ({len(prefix_roles)}) != prefix_shape ({len(prefix_shape)})"
    )

    N = 1
    for s in prefix_shape:
        N *= s

    # --- column names ---
    col_names = spec.provenance_columns()
    num_cols = len(col_names)
    prov = torch.zeros(N, max(num_cols, 1), dtype=torch.long)

    if N == 0:
        return prov

    # --- prepare sample_ids ---
    batch_dim_idx: Optional[int] = None
    for pi, role in enumerate(prefix_roles):
        if role is DimRole.BATCH:
            batch_dim_idx = pi
            break

    B = prefix_shape[batch_dim_idx] if batch_dim_idx is not None else 1
    if sample_ids is None:
        sample_ids = torch.arange(B, dtype=torch.long)
    else:
        sample_ids = sample_ids.to(torch.long).cpu()
        if sample_ids.numel() < B:
            # pad with zeros if sample_ids is shorter than batch
            padded = torch.zeros(B, dtype=torch.long)
            padded[:sample_ids.numel()] = sample_ids
            sample_ids = padded

    # --- compute multi-index for every token ---
    # Build strides for the prefix dimensions
    strides = []
    stride = 1
    for s in reversed(prefix_shape):
        strides.append(stride)
        stride *= s
    strides.reverse()

    # Column index lookup
    col_idx: Dict[str, int] = {name: i for i, name in enumerate(col_names)}

    # Vectorised multi-index computation
    flat_indices = torch.arange(N, dtype=torch.long)

    # For each prefix dim, compute coordinate via divmod
    dim_coords: Dict[int, torch.Tensor] = {}
    remaining = flat_indices.clone()
    for pi in range(len(prefix_shape)):
        dim_coords[pi] = remaining // strides[pi]
        remaining = remaining % strides[pi]

    # --- map dim roles → provenance columns ---
    # We also collect a dim_indices_dict for enrich_fn
    dim_indices_dict: Dict[DimRole, torch.Tensor] = {}
    columns_dict: Dict[str, torch.Tensor] = {}

    for pi, role in enumerate(prefix_roles):
        coord = dim_coords[pi]
        dim_indices_dict[role] = coord

        if role is DimRole.BATCH:
            cname = spec._col_name(role)
            if cname in col_idx:
                mapped = sample_ids[coord]
                prov[:, col_idx[cname]] = mapped
                columns_dict[cname] = mapped

        elif role is DimRole.HEIGHT:
            cname = spec._col_name(role)
            if cname in col_idx:
                prov[:, col_idx[cname]] = coord
                columns_dict[cname] = coord

        elif role is DimRole.WIDTH:
            cname = spec._col_name(role)
            if cname in col_idx:
                prov[:, col_idx[cname]] = coord
                columns_dict[cname] = coord

        elif role is DimRole.TOKEN:
            # TOKEN → y = -1, x = token_idx
            if "y" in col_idx:
                val_y = torch.full((N,), -1, dtype=torch.long)
                prov[:, col_idx["y"]] = val_y
                columns_dict["y"] = val_y
            if "x" in col_idx:
                prov[:, col_idx["x"]] = coord
                columns_dict["x"] = coord

        elif role is DimRole.TIME:
            cname = spec._col_name(role)
            if cname in col_idx:
                prov[:, col_idx[cname]] = coord
                columns_dict[cname] = coord

        elif role is DimRole.LANE:
            cname = spec._col_name(role)
            if cname in col_idx:
                prov[:, col_idx[cname]] = coord
                columns_dict[cname] = coord

    # --- fill fidx_hint if TIME is not in dims but frame_idx is in columns ---
    if DimRole.TIME not in set(prefix_roles):
        for cname in ("frame_idx",):
            if cname in col_idx and cname not in columns_dict:
                if fidx_hint is not None:
                    if torch.is_tensor(fidx_hint):
                        val = int(fidx_hint.item()) if fidx_hint.numel() == 1 else 0
                    else:
                        val = int(fidx_hint)
                    prov[:, col_idx[cname]] = val
                    columns_dict[cname] = torch.full((N,), val, dtype=torch.long)

    # --- enrich_fn callback ---
    if spec.enrich_fn is not None:
        spec.enrich_fn(columns_dict, dim_indices_dict)
        # Write back any changes from enrich_fn
        for cname, vals in columns_dict.items():
            if cname in col_idx:
                prov[:, col_idx[cname]] = vals

    return prov
