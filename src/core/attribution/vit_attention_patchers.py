from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.packs.clip.models.attnlrp import (
    AttnLRPAttention,
    AttnLRPGELU,
    AttnLRPLayerNorm,
    AttnLRPQuickGELU,
)
from src.packs.clip.models.libragrad import (
    FullGradGELU,
    FullGradLayerNorm,
    FullGradNormalize,
    FullGradQuickGELU,
    LibragradAttention,
)


def is_qkv_attention_module(module: nn.Module) -> bool:
    required = ("qkv", "attn_drop", "proj", "proj_drop", "num_heads")
    return all(hasattr(module, attr) for attr in required)


def ensure_attention_compat(module: nn.Module) -> nn.Module:
    if not hasattr(module, "head_dim"):
        qkv = getattr(module, "qkv", None)
        num_heads = getattr(module, "num_heads", None)
        if qkv is not None and num_heads:
            try:
                out_dim = int(qkv.weight.shape[0])
                setattr(module, "head_dim", out_dim // (3 * int(num_heads)))
            except Exception:
                pass
    if not hasattr(module, "scale") and hasattr(module, "head_dim"):
        try:
            setattr(module, "scale", float(getattr(module, "head_dim")) ** -0.5)
        except Exception:
            pass
    return module


def _is_quick_gelu(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    return name in {"quickgelu", "quick_gelu"}


def _replace_module(
    owner: nn.Module,
    name: str,
    new_mod: nn.Module,
    *,
    record: List[Tuple[nn.Module, str, nn.Module]],
) -> None:
    old = getattr(owner, name, None)
    if old is None or old is new_mod:
        return
    if isinstance(old, type(new_mod)):
        return
    setattr(owner, name, new_mod)
    if isinstance(old, nn.Module):
        new_mod.train(old.training)
    record.append((owner, name, old))


def apply_generic_attnlrp(model: nn.Module) -> Callable[[], None]:
    if getattr(model, "_generic_attnlrp_enabled", False):
        return lambda: None

    replaced: List[Tuple[nn.Module, str, nn.Module]] = []

    def _walk(mod: nn.Module) -> None:
        for child_name, child in mod.named_children():
            if isinstance(child, nn.LayerNorm):
                _replace_module(mod, child_name, AttnLRPLayerNorm.from_layer(child), record=replaced)
                continue
            if is_qkv_attention_module(child) and not isinstance(child, AttnLRPAttention):
                _replace_module(mod, child_name, AttnLRPAttention(ensure_attention_compat(child)), record=replaced)
                continue
            if isinstance(child, nn.GELU):
                _replace_module(mod, child_name, AttnLRPGELU(child), record=replaced)
                continue
            if _is_quick_gelu(child):
                _replace_module(mod, child_name, AttnLRPQuickGELU(), record=replaced)
                continue
            _walk(child)

    _walk(model)
    setattr(model, "_generic_attnlrp_enabled", True)

    def _restore() -> None:
        for owner, name, old in reversed(replaced):
            try:
                setattr(owner, name, old)
            except Exception:
                pass
        try:
            delattr(model, "_generic_attnlrp_enabled")
        except Exception:
            pass

    return _restore


def apply_generic_libragrad(model: nn.Module) -> Callable[[], None]:
    if getattr(model, "_generic_libragrad_enabled", False):
        return lambda: None

    replaced: List[Tuple[nn.Module, str, nn.Module]] = []

    def _walk(mod: nn.Module) -> None:
        for child_name, child in mod.named_children():
            if isinstance(child, nn.LayerNorm):
                _replace_module(mod, child_name, FullGradLayerNorm.from_layer(child), record=replaced)
                continue
            if child.__class__.__name__.lower() == "normalize":
                _replace_module(mod, child_name, FullGradNormalize.from_module(child), record=replaced)
                continue
            if is_qkv_attention_module(child) and not isinstance(child, LibragradAttention):
                _replace_module(mod, child_name, LibragradAttention(ensure_attention_compat(child)), record=replaced)
                continue
            if isinstance(child, nn.GELU):
                _replace_module(mod, child_name, FullGradGELU(), record=replaced)
                continue
            if _is_quick_gelu(child):
                _replace_module(mod, child_name, FullGradQuickGELU(), record=replaced)
                continue
            _walk(child)

    _walk(model)
    setattr(model, "_generic_libragrad_enabled", True)

    def _restore() -> None:
        for owner, name, old in reversed(replaced):
            try:
                setattr(owner, name, old)
            except Exception:
                pass
        try:
            delattr(model, "_generic_libragrad_enabled")
        except Exception:
            pass

    return _restore


class CaptureAttention(nn.Module):
    """Attention wrapper that records softmax maps and their gradients."""

    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.qkv = attn.qkv
        self.q_norm = getattr(attn, "q_norm", None)
        self.k_norm = getattr(attn, "k_norm", None)
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop
        self.num_heads = int(attn.num_heads)
        self.head_dim = int(getattr(attn, "head_dim", self.qkv.weight.shape[0] // (3 * self.num_heads)))
        self.scale = float(getattr(attn, "scale", self.head_dim ** -0.5))
        self._saved_attn: torch.Tensor | None = None
        self._saved_grad: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, **_: object) -> torch.Tensor:  # type: ignore[override]
        bsz, toks, chans = x.shape
        qkv = self.qkv(x).reshape(bsz, toks, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        raw = (q * self.scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            raw = raw + attn_mask.to(dtype=raw.dtype, device=raw.device)

        attn = raw.softmax(dim=-1)
        self._saved_attn = attn
        if attn.requires_grad:
            attn.register_hook(lambda grad: setattr(self, "_saved_grad", grad))

        out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(bsz, toks, chans)
        out = self.proj_drop(self.proj(out))
        return out


def apply_attention_capture(model: nn.Module) -> Callable[[], None]:
    replaced: List[Tuple[nn.Module, str, nn.Module]] = []

    def _walk(mod: nn.Module) -> None:
        for child_name, child in mod.named_children():
            if is_qkv_attention_module(child) and not isinstance(child, CaptureAttention):
                _replace_module(mod, child_name, CaptureAttention(child), record=replaced)
                continue
            _walk(child)

    _walk(model)

    def _restore() -> None:
        for owner, name, old in reversed(replaced):
            try:
                setattr(owner, name, old)
            except Exception:
                pass

    return _restore


def clear_attention_capture_state(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, CaptureAttention):
            module._saved_attn = None
            module._saved_grad = None


def compute_inflow_rollout(
    attentions: List[torch.Tensor],
    biases_1: List[torch.Tensor],
    biases_2: List[torch.Tensor],
) -> torch.Tensor:
    """InFlow-style rollout through attention and residual paths.

    Provenance: adapted from Walker et al., "Explaining ViTs Using
    Information Flow" (AISTATS 2025), official implementation:
    https://github.com/chasewalker26/InFlow-ViT-Explanation

    This mirrors the paper/code's `compute_InFlow` structure: construct
    per-block transition matrices from attention, add norm-weighted first
    residual paths, add norm-weighted MLP/second-residual scaling, normalize,
    and multiply matrices across blocks. In SpecLens this is used as the
    `inflow_erf` baseline and is adapted from class/CLS attribution to
    feature-conditioned ERF attribution for arbitrary target tokens.
    """
    attn_s = torch.stack(attentions)
    bias1_s = torch.stack(biases_1)
    bias2_s = torch.stack(biases_2)

    inp_w = bias1_s[:, 0]
    attn_w = bias1_s[:, 1]
    r1_w = bias2_s[:, 0]
    mlp_w = bias2_s[:, 1]

    mat_r1 = attn_s * attn_w.unsqueeze(-2) + torch.diag_embed(inp_w)
    ratio = F.normalize(mlp_w / (r1_w + 1e-8), p=1, dim=-1)
    mat_r2 = torch.diag_embed(ratio * mlp_w + r1_w)
    matrices = mat_r1 @ mat_r2
    matrices = matrices / matrices.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    joint = matrices[0]
    for idx in range(1, len(matrices)):
        joint = matrices[idx] @ joint
    return joint


__all__ = [
    "CaptureAttention",
    "apply_attention_capture",
    "apply_generic_attnlrp",
    "apply_generic_libragrad",
    "clear_attention_capture_state",
    "compute_inflow_rollout",
    "ensure_attention_compat",
    "is_qkv_attention_module",
]
