from __future__ import annotations
import torch
from typing import Optional, Tuple
import torch.nn.functional as F

@torch.no_grad()
def reduce_channels(t: torch.Tensor, how: str = "l2") -> torch.Tensor:
    """(C,H,W) → (H,W)"""
    assert t.dim() == 3, f"expected (C,H,W); got {tuple(t.shape)}"
    if how == "l1":
        h = t.abs().sum(0)
    else:  # l2 default
        h = t.pow(2).mean(0).sqrt()
    return h

@torch.no_grad()
def robust_log_normalise(h: torch.Tensor, q_lo: float = 0.05, q_hi: float = 0.99, size: Tuple[int,int] | None = None) -> torch.Tensor:
    """
    Take positive part, log, clamp by quantiles, scale to [0,1]; optional resize.
    Input h is (H,W) non-negative.
    """
    h = (h - h.min()).clamp_min_(0)
    mask = h > 0
    out = torch.zeros_like(h)
    if mask.any():
        v = torch.log(h[mask] + 1e-12)
        lo, hi = torch.quantile(v, torch.tensor([q_lo, q_hi], device=h.device))
        v = v.clamp(min=lo, max=hi)
        out[mask] = (v - lo) / (hi - lo + 1e-8)
    if size is not None:
        out = F.interpolate(out[None, None], size=size, mode="bilinear", align_corners=False).squeeze()
    return out

@torch.no_grad()
def robust_log_normalise_stack(
    H: torch.Tensor,
    q_lo: float = 0.05,
    q_hi: float = 0.99,
    size: Tuple[int, int] | None = None
) -> torch.Tensor:
    """
    Global log-normalisation across ALL frames.
    H: (T,H,W) ≥ 0  →  returns (T,H,W) in [0,1].
    If `size` is given, returns resized (T,H*,W*).
    """
    assert H.dim() == 3, f"expected (T,H,W); got {tuple(H.shape)}"
    Hp = (H - H.min()).clamp_min_(0)
    mask = Hp > 0
    out = torch.zeros_like(Hp)
    if mask.any():
        v = torch.log(Hp[mask] + 1e-12)
        lo, hi = torch.quantile(v, torch.tensor([q_lo, q_hi], device=H.device))
        v = v.clamp(min=lo, max=hi)
        out[mask] = (v - lo) / (hi - lo + 1e-8)
    if size is not None:
        out = F.interpolate(out[:, None], size=size, mode="nearest").squeeze(1)
    return out


def restore_tokens_like(tokens: torch.Tensor, reference: torch.Tensor) -> Optional[torch.Tensor]:
    """Reshape a flattened token tensor back to the reference tensor layout."""
    if reference.dim() == 4:
        b, c, h, w = reference.shape
        return tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    if reference.dim() == 3:
        b, l, c = reference.shape
        return tokens.view(b, l, c)
    if reference.dim() == 2:
        return tokens.view_as(reference)
    return None

def infer_grid_from_tokens(L: int, H_img: int, W_img: int) -> Tuple[int, int]:
    """
    토큰 길이 L과 원본 이미지 종횡비(=H_img:W_img)를 이용해 H_m, W_m를 추정한다.
    방법: L의 약수 쌍 (h, L//h) 중에서 h/(L//h) 가 H_img/W_img와 가장 가까운 것을 고른다.
    (ViT류에서 대부분 L = (H/patch)*(W/patch) 이므로 매우 잘 맞음)
    """
    if L <= 0:
        return 1, L if L > 0 else 1
    import math
    target_ar = (H_img / float(W_img)) if W_img > 0 else 1.0
    best_h, best_w = 1, L
    best_err = float('inf')
    # 약수 탐색
    r = int(math.sqrt(L))
    for h in range(1, r + 1):
        if L % h != 0:
            continue
        w = L // h
        # 후보 1: (h, w)
        ar = h / float(w)
        err = abs(ar - target_ar)
        if err < best_err:
            best_h, best_w, best_err = h, w, err
        # 후보 2: (w, h)도 고려 (정방향/역방향 뒤섞임 방지)
        ar2 = w / float(h)
        err2 = abs(ar2 - target_ar)
        if err2 < best_err:
            best_h, best_w, best_err = w, h, err2
    return int(best_h), int(best_w)

def token_index_to_hw(idx: int, Hm: int, Wm: int) -> Tuple[int, int]:
    """
    평탄 토큰 인덱스(idx)를 2D 그리드 (y, x)로 변환.
    """
    n = int(Hm) * int(Wm)
    if n <= 0:
        return 0, 0
    idx = max(0, min(n - 1, int(idx)))
    y = idx // int(Wm)
    x = idx %  int(Wm)
    return int(y), int(x)