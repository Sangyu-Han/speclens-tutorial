from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _extract_tensor(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    raise TypeError(f"Expected tensor-like block output, got {type(output)!r}")


@dataclass
class ERFCapture:
    block0_input: torch.Tensor
    target_block_output: torch.Tensor
    baseline: torch.Tensor
    prefix_tokens: torch.Tensor
    patch_tokens: torch.Tensor

    @property
    def num_patches(self) -> int:
        return int(self.patch_tokens.shape[1])

    @property
    def hidden_dim(self) -> int:
        return int(self.patch_tokens.shape[-1])


class VisionTransformerERFAdapter(ABC):
    """Pack-specific ERF adapters share the same ViT injection contract."""

    pack_name = "vit"

    def __init__(self, model: nn.Module):
        self.model = model.eval()
        self.base_model = _unwrap_module(model)
        self.baseline_mode = os.environ.get("ERF_BASELINE_MODE", "global_mean_h_plus_pos").strip().lower()
        self.baseline_cache_root = Path(
            os.environ.get(
                "ERF_BASELINE_CACHE_ROOT",
                str(Path(__file__).resolve().parents[3] / "outputs" / "erf_baselines"),
            )
        ).expanduser()
        self._validate_contract()

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.base_model.parameters()).dtype

    def prefix_count(self) -> int:
        prefix = getattr(self.base_model, "num_prefix_tokens", None)
        if prefix is None:
            prefix = 1 if hasattr(self.base_model, "cls_token") else 0
        return int(prefix)

    def patch_baseline(self, num_patches: int, dtype: torch.dtype) -> torch.Tensor:
        mode = str(self.baseline_mode).lower()
        if mode in {"global_mean_h", "global_mean_h_plus_pos"}:
            baseline_pt = self.baseline_cache_root / "global_mean_h" / f"{self.pack_name}.pt"
            if not baseline_pt.exists():
                raise FileNotFoundError(
                    f"[{self.pack_name}] missing ERF baseline cache: {baseline_pt}. "
                    "Run scripts/compute_global_mean_h_baseline.py first."
                )
            payload = torch.load(baseline_pt, map_location="cpu")
            mean_patch = payload.get("mean_patch")
            if not torch.is_tensor(mean_patch) or mean_patch.ndim != 3 or mean_patch.shape[0] != 1:
                raise ValueError(
                    f"[{self.pack_name}] invalid mean_patch tensor in {baseline_pt}: "
                    f"{type(mean_patch)!r} shape={getattr(mean_patch, 'shape', None)}"
                )
            baseline = mean_patch.to(device=self.device, dtype=dtype).expand(1, num_patches, -1).contiguous()
            if mode == "global_mean_h_plus_pos":
                pos_embed = getattr(self.base_model, "pos_embed", None)
                if torch.is_tensor(pos_embed) and pos_embed.ndim == 3 and int(pos_embed.shape[1]) >= num_patches:
                    pos = pos_embed[:, -num_patches:, :].detach().to(device=self.device, dtype=dtype)
                    pos_mean = pos.mean(dim=1, keepdim=True)
                    baseline = baseline - pos_mean + pos
            return baseline

        if mode == "zero":
            return torch.zeros(1, num_patches, self.hidden_dim(), device=self.device, dtype=dtype)

        if mode not in {"default", "pos_embed"}:
            raise ValueError(f"[{self.pack_name}] unknown ERF baseline mode: {mode}")

        pos_embed = getattr(self.base_model, "pos_embed", None)
        if not torch.is_tensor(pos_embed) or pos_embed.ndim != 3:
            raise AttributeError(f"[{self.pack_name}] baseline mode '{mode}' requires model.pos_embed")
        baseline = pos_embed[:, -num_patches:, :].detach()
        return baseline.to(device=self.device, dtype=dtype)

    def hidden_dim(self) -> int:
        if hasattr(self.base_model, "embed_dim"):
            return int(getattr(self.base_model, "embed_dim"))
        pos_embed = getattr(self.base_model, "pos_embed", None)
        if torch.is_tensor(pos_embed) and pos_embed.ndim == 3:
            return int(pos_embed.shape[-1])
        patch_embed = getattr(self.base_model, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        bias = getattr(proj, "bias", None)
        if torch.is_tensor(bias):
            return int(bias.shape[0])
        raise AttributeError(f"[{self.pack_name}] could not infer hidden dim for zero ERF baseline")

    def capture_block0_input_and_target_block_out(
        self,
        x: torch.Tensor,
        block_idx: int,
    ) -> ERFCapture:
        self._validate_block_idx(block_idx)
        x = x.to(device=self.device, dtype=self.dtype)

        b0_buf: list[torch.Tensor] = []
        blk_buf: list[torch.Tensor] = []

        h_b0 = self.base_model.blocks[0].register_forward_pre_hook(
            lambda _m, args: b0_buf.append(args[0].detach().clone())
        )
        h_blk = self.base_model.blocks[block_idx].register_forward_hook(
            lambda _m, _i, o: blk_buf.append(_extract_tensor(o).detach().clone())
        )
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            h_b0.remove()
            h_blk.remove()

        if not b0_buf or not blk_buf:
            raise RuntimeError(
                f"[{self.pack_name}] failed to capture block-0 input or block-{block_idx} output"
            )

        block0_input = b0_buf[0]
        prefix = self.prefix_count()
        prefix_tokens = block0_input[:, :prefix, :].detach()
        patch_tokens = block0_input[:, prefix:, :].detach()
        baseline = self.patch_baseline(int(patch_tokens.shape[1]), patch_tokens.dtype).to(
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )

        if baseline.shape != patch_tokens.shape:
            raise ValueError(
                f"[{self.pack_name}] baseline shape {tuple(baseline.shape)} does not match "
                f"patch tokens {tuple(patch_tokens.shape)}"
            )

        return ERFCapture(
            block0_input=block0_input,
            target_block_output=blk_buf[0],
            baseline=baseline,
            prefix_tokens=prefix_tokens,
            patch_tokens=patch_tokens,
        )

    def make_masked_forward(
        self,
        x: torch.Tensor,
        capture: ERFCapture,
        block_idx: int,
    ) -> Tuple[Callable[[torch.Tensor], None], Callable[[], torch.Tensor]]:
        self._validate_block_idx(block_idx)
        x = x.to(device=self.device, dtype=self.dtype)

        content_diff = capture.patch_tokens - capture.baseline
        last_block_out: list[torch.Tensor] = [torch.empty(0, device=self.device)]

        def do_forward_masked(z: torch.Tensor) -> None:
            mask = z.view(1, -1, 1).to(device=content_diff.device, dtype=content_diff.dtype)
            mixed_patches = capture.baseline + mask * content_diff
            injected = self._join_prefix_and_patches(capture.prefix_tokens, mixed_patches)
            last_block_out[0] = self._run_with_injected_block0_input(x, block_idx, injected)

        def get_block_out() -> torch.Tensor:
            return last_block_out[0]

        return do_forward_masked, get_block_out

    def make_alpha_forward(
        self,
        x: torch.Tensor,
        capture: ERFCapture,
        block_idx: int,
    ) -> Tuple[
        Callable[[float], None],
        Callable[[], None],
        Callable[[], torch.Tensor],
        Callable[[], torch.Tensor],
    ]:
        self._validate_block_idx(block_idx)
        x = x.to(device=self.device, dtype=self.dtype)

        content_diff = (capture.patch_tokens - capture.baseline).detach()
        current_alpha_input: list[torch.Tensor] = [capture.patch_tokens.clone().detach().requires_grad_(True)]
        last_block_out: list[torch.Tensor] = [torch.empty(0, device=self.device)]

        def set_alpha(alpha: float) -> None:
            alpha_input = (capture.baseline + alpha * content_diff).detach().requires_grad_(True)
            current_alpha_input[0] = alpha_input

        def do_forward() -> None:
            injected = self._join_prefix_and_patches(capture.prefix_tokens, current_alpha_input[0])
            last_block_out[0] = self._run_with_injected_block0_input(x, block_idx, injected)

        def get_h_alpha() -> torch.Tensor:
            return current_alpha_input[0]

        def get_block_out() -> torch.Tensor:
            return last_block_out[0]

        return set_alpha, do_forward, get_h_alpha, get_block_out

    def _run_with_injected_block0_input(
        self,
        x: torch.Tensor,
        block_idx: int,
        injected: torch.Tensor,
    ) -> torch.Tensor:
        buf: list[torch.Tensor] = []
        h_pre = self.base_model.blocks[0].register_forward_pre_hook(lambda _m, _args: (injected,))
        h_blk = self.base_model.blocks[block_idx].register_forward_hook(
            lambda _m, _i, o: buf.append(_extract_tensor(o))
        )
        try:
            self.model(x)
        finally:
            h_pre.remove()
            h_blk.remove()

        if not buf:
            raise RuntimeError(f"[{self.pack_name}] failed to capture block-{block_idx} output")
        return buf[0]

    def _join_prefix_and_patches(
        self,
        prefix_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_tokens.numel() == 0:
            return patch_tokens
        return torch.cat([prefix_tokens, patch_tokens], dim=1)

    def _validate_contract(self) -> None:
        if not hasattr(self.base_model, "blocks"):
            raise AttributeError(f"[{self.pack_name}] model is missing 'blocks'")

        try:
            n_blocks = len(self.base_model.blocks)
        except TypeError as exc:
            raise TypeError(f"[{self.pack_name}] model.blocks is not indexable") from exc

        if n_blocks == 0:
            raise ValueError(f"[{self.pack_name}] model.blocks is empty")

        pos_embed = getattr(self.base_model, "pos_embed", None)
        if pos_embed is not None:
            if not torch.is_tensor(pos_embed) or pos_embed.ndim != 3:
                raise TypeError(
                    f"[{self.pack_name}] pos_embed must be a [1, tokens, dim] tensor, "
                    f"got {type(pos_embed)!r} with ndim={getattr(pos_embed, 'ndim', None)}"
                )
            prefix = self.prefix_count()
            if pos_embed.shape[1] <= prefix:
                raise ValueError(
                    f"[{self.pack_name}] pos_embed has too few tokens for prefix_count={prefix}: "
                    f"{tuple(pos_embed.shape)}"
                )

    def _validate_block_idx(self, block_idx: int) -> None:
        n_blocks = len(self.base_model.blocks)
        if block_idx < 0 or block_idx >= n_blocks:
            raise IndexError(
                f"[{self.pack_name}] block_idx={block_idx} outside valid range [0, {n_blocks - 1}]"
            )


__all__ = ["ERFCapture", "VisionTransformerERFAdapter"]
