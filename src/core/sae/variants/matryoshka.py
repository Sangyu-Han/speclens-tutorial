from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from ..base import BaseAutoencoder
from ..registry import register


@register("matryoshka")
class MatryoshkaSAE(BaseAutoencoder):
    """
    Matryoshka SAE:
      - 그룹 입력은 'prefix 배수(스케일)'로 받음 (예: [0.5, 1, 2, 4, 8]).
      - 실제 사용은 누적 prefix 크기(정수)로 변환하여, 순차적으로 디코딩+손실 합산.
      - 학습: 전역 BatchTopK (배치 전체에서 K*B개 유지)
      - 평가: 전역 임계치(threshold) 기본값, 선택적으로 topk_batch / topk_token 지원.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        act_size = int(cfg["act_size"])
        dict_size = int(cfg.get("dict_size", int(round(cfg.get("expansion_factor", 8) * act_size))))

        # ---- [입력] prefix 스케일 목록(배수) ----
        # 우선순위: group_prefix_scales > group_scales > group_ratios
        prefix_scales = cfg.get("group_prefix_scales", None)


        # ---- [자동 생성] 스케일 사다리 ----
        if not prefix_scales:
            # 논문처럼 작은 → 큰 사전으로 가는 고정 prefix들.
            # 기본은 5그룹: [0.5, 1, 2, 4, ef]
            num_groups = int(cfg.get("num_groups", 5))
            ef = float(cfg.get("expansion_factor", max(dict_size / max(1, act_size), 1.0)))
            if num_groups == 5:
                prefix_scales = [0.5, 1.0, 2.0, 4.0, ef]
            else:
                # 0.5에서 ef까지 등비 간격
                import math
                r = (ef / 0.5) ** (1.0 / (num_groups - 1))
                prefix_scales = [0.5 * (r ** i) for i in range(num_groups)]

        # ---- 스케일 → 누적 prefix 크기(정수)로 변환 ----
        # 원 논문은 "여러 누적 사전 크기 m1 < m2 < ... < mN" 를 동시에 학습(3.1절) :contentReference[oaicite:2]{index=2}
        prefix_sizes = [max(1, int(round(s * act_size))) for s in prefix_scales]

        # 마지막 prefix는 전체 사전(dict_size)로 강제(없거나 작으면 append, 크면 clamp)
        if prefix_sizes[-1] < dict_size:
            prefix_sizes.append(dict_size)
        prefix_sizes = [min(p, dict_size) for p in prefix_sizes]

        # 단조증가 & 중복 제거
        dedup = []
        for p in prefix_sizes:
            if len(dedup) == 0 or p > dedup[-1]:
                dedup.append(p)
        # 최종 안전장치: 마지막이 dict_size가 아니면 붙이기
        if dedup[-1] != dict_size:
            dedup.append(dict_size)
        self.prefix_sizes: List[int] = dedup

        # ---- per-group "증분(segment)" 크기 산출 (Listing 1과 동일한 누적-증분 구조) :contentReference[oaicite:3]{index=3}
        seg_sizes: List[int] = []
        prev = 0
        for p in self.prefix_sizes:
            seg_sizes.append(p - prev)
            prev = p
        self.segment_sizes: List[int] = seg_sizes

        # Python int로만 누적 인덱스(DDP-safe)
        prefix = [0]
        for g in self.segment_sizes:
            prefix.append(prefix[-1] + int(g))
        self.group_indices = prefix
        self.active_groups = len(self.segment_sizes)

        # ---- 기타 옵션 ----
        # 학습/평가 선택 로직 단순화: group_topk/per_group_selection 제거
        self.eval_selection = cfg.get("eval_selection", "threshold")  # {"threshold","topk_batch","topk_token"}
        self.top_k = int(cfg.get("top_k", cfg.get("k", 32)))
        self.top_k_aux = int(cfg.get("top_k_aux", cfg.get("k_aux", 512)))
        self.aux_penalty = float(cfg.get("aux_penalty", cfg.get("aux_frac", 1/32)))
        self.threshold_lr = float(cfg.get("threshold_ema_lr", 0.01))

        # ----- threshold 버퍼 -----
        if not hasattr(self, "threshold"):
            self.register_buffer("threshold", torch.tensor(0.0))

    # --------- 활성 계산 ---------
    def compute_activations(self, x_cent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre_acts = x_cent @ self.W_enc + self.b_enc
        acts = pre_acts
        B = x_cent.shape[0]

        if self.training:
            # [논문 3.2] 전역 BatchTopK: 배치 전체를 평탄화해서 K*B개 유지, 나머지 0. :contentReference[oaicite:4]{index=4}
            tk = torch.topk(pre_acts.flatten(), self.top_k * B, dim=-1)
            acts_topk = (
                torch.zeros_like(pre_acts.flatten())
                .scatter(-1, tk.indices, tk.values)
                .view_as(pre_acts)
            )
            # 학습 중 threshold EMA 갱신(평가 시 임계치 사용)
            self.update_threshold(acts_topk, lr=self.threshold_lr)
            return acts, acts_topk.relu()

        # ---- 평가 시 선택 ----
        if self.eval_selection == "threshold":
            # [논문 3.2] 학습 데이터에서 보정한 전역 임계치 사용. :contentReference[oaicite:5]{index=5}
            acts_topk = torch.where(pre_acts > self.threshold, pre_acts, torch.zeros_like(pre_acts))
        elif self.eval_selection == "topk_batch":
            tk = torch.topk(pre_acts.flatten(), self.top_k * B, dim=-1)
            acts_topk = (
                torch.zeros_like(pre_acts.flatten())
                .scatter(-1, tk.indices, tk.values)
                .view_as(pre_acts)
            )
        elif self.eval_selection == "topk_token":
            tk = torch.topk(pre_acts, self.top_k, dim=-1)
            acts_topk = torch.zeros_like(pre_acts).scatter(-1, tk.indices, tk.values)
        else:
            raise ValueError(f"Unknown eval_selection: {self.eval_selection}")

        return acts, acts_topk.relu()

    @torch.no_grad()
    def update_threshold(self, acts_topk: torch.Tensor, lr: float = 0.01):
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min().to(self.threshold)
            # 버퍼 제자리 EMA
            self.threshold.lerp_(min_positive, lr)

    # --------- 순전파: 그룹별(증분) 누적 디코딩 & 손실 ---------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        x_cent = x - self.b_dec
        x_reconstruct = self.b_dec

        inter_recons: List[torch.Tensor] = []
        all_acts, all_acts_topk = self.compute_activations(x_cent)

        # Listing 1과 동일한 "증분(segment)"을 순차적으로 더하는 구조. :contentReference[oaicite:6]{index=6}
        ofs = 0
        for gsz in self.segment_sizes:
            sl = slice(ofs, ofs + gsz)
            x_reconstruct = all_acts_topk[:, sl] @ self.W_dec[sl, :] + x_reconstruct
            inter_recons.append(x_reconstruct)
            ofs += gsz

        self.update_inactive_features(all_acts_topk)
        return self.get_loss_dict(x, x_reconstruct, all_acts, all_acts_topk, x_mean, x_std, inter_recons)

    # --------- 인/디코더 ---------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x, x_mean, x_std = self.preprocess_input(x)
        self._cache_input_stats(x_mean, x_std)
        x = x.reshape(-1, x.shape[-1])

        x_cent = x - self.b_dec
        _, result = self.compute_activations(x_cent)

        # 안전 마스크 (실제 dict_size와 동일)
        max_act_index = self.group_indices[self.active_groups]
        result[:, max_act_index:] = 0
        if len(orig_shape) == 3:
            result = result.reshape(orig_shape[0], orig_shape[1], -1)
        return result

    def decode(self, acts_topk: torch.Tensor) -> torch.Tensor:
        reconstruct = acts_topk @ self.W_dec + self.b_dec
        x_mean = getattr(self, "x_mean", torch.zeros_like(reconstruct[:1]))
        x_std  = getattr(self, "x_std",  torch.ones_like(reconstruct[:1]))
        return self.postprocess_output(reconstruct, x_mean, x_std)

    # --------- 손실/보조손실 ---------
    def get_loss_dict(
        self,
        x: torch.Tensor,
        x_reconstruct: torch.Tensor,
        all_acts: torch.Tensor,
        all_acts_topk: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        intermediate_reconstructs: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # 각 prefix 단계의 MSE를 합산(논문 3.1절) :contentReference[oaicite:7]{index=7}
        total_l2_loss = (self.b_dec - x.float()).pow(2).mean()
        l2_losses = torch.tensor([], device=x.device)
        for inter in intermediate_reconstructs:
            l2_g = (inter.float() - x.float()).pow(2).mean()
            l2_losses = torch.cat([l2_losses, l2_g.unsqueeze(0)])
            total_l2_loss += l2_g

        mean_l2_loss = total_l2_loss / (len(intermediate_reconstructs) + 1)
        min_l2_loss = l2_losses.min()
        max_l2_loss = l2_losses.max()

        l1_norm = all_acts_topk.float().abs().sum(-1).mean()
        l0_norm = (all_acts_topk > 0).float().sum(-1).mean()
        l1_loss = self.config["l1_coeff"] * l1_norm
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, all_acts)
        loss = mean_l2_loss + l1_loss + aux_loss

        num_dead_features = (self.num_batches_not_active > self.config.get("n_batches_to_dead", 20)).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        return {
            "sae_out": sae_out,
            "feature_acts": all_acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": mean_l2_loss,
            "min_l2_loss": min_l2_loss,
            "max_l2_loss": max_l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "threshold": self.threshold,
        }

    def get_auxiliary_loss(self, x: torch.Tensor, x_reconstruct: torch.Tensor, all_acts: torch.Tensor) -> torch.Tensor:
        residual = x.float() - x_reconstruct.float()
        aux_reconstruct = torch.zeros_like(residual)

        dead_features = self.num_batches_not_active >= self.config.get("n_batches_to_dead", 20)
        if dead_features.sum() > 0:
            acts_topk_aux = torch.topk(
                all_acts[:, dead_features],
                min(self.config.get("top_k_aux", 512), dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(all_acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            aux_reconstruct = aux_reconstruct + x_reconstruct_aux

        if aux_reconstruct.abs().sum() > 0:
            aux_loss = self.config.get("aux_penalty", 0.03125) * (aux_reconstruct.float() - residual.float()).pow(2).mean()
            return aux_loss
        return torch.tensor(0.0, device=x.device)
