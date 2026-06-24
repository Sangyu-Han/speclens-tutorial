"""CNN-FRI: per-feature input attribution (sufficiency ERF) for the CIFAR CNN.

For an SAE feature (layer, unit): take its top-activating CIFAR image, soft-mask
the INPUT pixels toward the mean baseline (= normalized 0), and solve FRI
(reuse src.core.attribution.fri.solver.run_fri) to find the minimal input
support that recovers the feature's activation at its firing cell.

This is the model-agnostic FRI applied to a CNN feature. The support map +
its spatial stats are the "feature attribution score" we order features by:
  - concept feature  -> compact support ON the object (high concentration)
  - bias feature     -> diffuse / fixed-position support (e.g. a corner)

Run (GPU1 = display-safe, see env memory):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_fri_feature.py \
      --layer model.layer4.0 --units 731,1946 --grid 16 --steps 48
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from src.core.attribution.fri.solver import FRIConfig, run_fri, _baseline_corrected_recovery
from src.core.sae.registry import create_sae
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD
from src.packs.cifar_cnn.models.model_loaders import load_cifar_cnn_model


def load_sae(layer: str, sae_root: str, device):
    ckpts = sorted(glob.glob(f"{sae_root}/{layer}/*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no SAE checkpoint for {layer} under {sae_root}")
    pkg = torch.load(ckpts[-1], map_location="cpu")
    sae_cfg = dict(pkg.get("sae_config", {}))
    act_size = int(pkg.get("act_size", 0))
    sae_cfg.update({"act_size": act_size, "device": str(device)})
    sae_type = sae_cfg.get("sae_type") or pkg.get("sae_type", "batch-topk")
    sae = create_sae(sae_type, sae_cfg)
    sae.load_state_dict(pkg["sae_state"], strict=True)
    sae.eval().to(device)
    if hasattr(sae, "configure_visualization_gating"):
        sae.configure_visualization_gating(mode="dict")  # differentiable pre-acts
    return sae


class CnnFri:
    def __init__(self, ckpt, sae_root, device):
        self.device = torch.device(device)
        self.model = load_cifar_cnn_model({"ckpt": ckpt}, device=self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.sae_root = sae_root
        self._saes = {}
        self.norm = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

    def sae(self, layer):
        if layer not in self._saes:
            self._saes[layer] = load_sae(layer, self.sae_root, self.device)
        return self._saes[layer]

    def _module(self, layer):
        mod = self.model
        for p in layer.replace("model.", "").split("."):
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        return mod

    def _acts_at(self, x, layer):
        cap = {}
        h = self._module(layer).register_forward_hook(lambda m, i, o: cap.__setitem__("v", o))
        try:
            self.model(x)
        finally:
            h.remove()
        return cap["v"]  # [B,C,H,W], in graph

    def _feat_col(self, x, layer, unit):
        """SAE feature `unit` activation over the spatial grid -> [H*W], plus (H,W)."""
        h = self._acts_at(x, layer)
        C, H, W = h.shape[1], h.shape[2], h.shape[3]
        flat = h[0].permute(1, 2, 0).reshape(-1, C)  # [H*W, C] row-major (y,x)
        enc = self.sae(layer).encode(flat)           # [H*W, dict]
        return enc[:, unit], H, W

    def feature_erf(self, img_norm, layer, unit, grid=16, steps=48):
        x = img_norm.unsqueeze(0).to(self.device)
        xb = torch.zeros_like(x)  # mean baseline = normalized 0
        with torch.no_grad():
            col, H, W = self._feat_col(x, layer, unit)
            cell = int(col.argmax())
            full = col[cell].detach()
            base = self._feat_col(xb, layer, unit)[0][cell].detach()

        ones = torch.ones((), device=self.device)
        zeros = torch.zeros((), device=self.device)

        def objective_for_mask(mask):
            m = mask.view(1, 1, grid, grid)
            m_up = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)
            xm = xb + m_up * (x - xb)
            act = self._feat_col(xm, layer, unit)[0][cell]
            return _baseline_corrected_recovery(act, full, base)

        res = run_fri(
            n_patches=grid * grid, grid_size=grid,
            objective_for_mask=objective_for_mask,
            full_objective=ones, baseline_objective=zeros,
            irrelevance=torch.ones(grid * grid, device=self.device),
            config=FRIConfig(steps=steps), device=self.device,
        )
        support = np.asarray(res.scores, dtype=np.float32).reshape(grid, grid)
        return support, dict(cell=cell, H=H, W=W, recovery=float(res.best_objective or 0.0),
                             full_act=float(full), base_act=float(base))

    def feat_map_gated(self, img_norm, layer, unit):
        """Gated SAE-feature activation over the layer's spatial grid (WHERE it fires)."""
        x = img_norm.unsqueeze(0).to(self.device)
        sae = self.sae(layer)
        sae.configure_visualization_gating(mode="hard")
        with torch.no_grad():
            h = self._acts_at(x, layer)
            C, H, W = h.shape[1], h.shape[2], h.shape[3]
            flat = h[0].permute(1, 2, 0).reshape(-1, C)
            m = sae.encode(flat)[:, unit].reshape(H, W).float().cpu().numpy()
        sae.configure_visualization_gating(mode="dict")
        return np.maximum(m, 0.0)

    def feature_support(self, img_norm, layer, unit, grid=16, steps=40, target=0.9):
        """FRI, then discretize to the minimal INSERTION SET that recovers `target` of
        the feature value. Returns (scores[g,g], support_mask[g,g], meta)."""
        x = img_norm.unsqueeze(0).to(self.device)
        xb = torch.zeros_like(x)
        with torch.no_grad():
            col, H, W = self._feat_col(x, layer, unit)
            cell = int(col.argmax()); full = col[cell].detach()
            base = self._feat_col(xb, layer, unit)[0][cell].detach()
        ones = torch.ones((), device=self.device); zeros = torch.zeros((), device=self.device)

        def raw_act(mask):
            m = mask.view(1, 1, grid, grid)
            m_up = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)
            xm = xb + m_up * (x - xb)
            return self._feat_col(xm, layer, unit)[0][cell]

        res = run_fri(n_patches=grid * grid, grid_size=grid,
                      objective_for_mask=lambda m: _baseline_corrected_recovery(raw_act(m), full, base),
                      full_objective=ones, baseline_objective=zeros,
                      irrelevance=torch.ones(grid * grid, device=self.device),
                      config=FRIConfig(steps=steps), device=self.device)
        scores = np.asarray(res.scores, dtype=np.float32)
        order = np.argsort(scores)[::-1]
        denom = float(full - base) if abs(float(full - base)) > 1e-6 else 1e-6

        def recovery_at(k):
            mask = torch.zeros(grid * grid, device=self.device)
            mask[torch.tensor(order[:k].copy(), device=self.device)] = 1.0
            with torch.no_grad():
                return (float(raw_act(mask)) - float(base)) / denom

        k_set, rec = grid * grid, 1.0
        for k in [1, 2, 4, 8, 16, 32, 64, 128, grid * grid]:
            if k > grid * grid:
                continue
            rec = recovery_at(k)
            if rec >= target:
                k_set = k
                break
        support_mask = np.zeros(grid * grid, dtype=np.float32)
        support_mask[order[:k_set]] = 1.0
        meta = dict(cell=cell, set_size=int(k_set), set_frac=float(k_set / (grid * grid)),
                    recovery=float(rec), full_act=float(full), base_act=float(base))
        return scores.reshape(grid, grid), support_mask.reshape(grid, grid), meta


def support_stats(support):
    s = support / (support.sum() + 1e-8)
    g = support.shape[0]
    ys, xs = np.mgrid[0:g, 0:g]
    cy, cx = float((s * ys).sum()), float((s * xs).sum())
    spread = float(np.sqrt((s * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum()))
    # concentration = mass in top 10% of cells (high=compact/concept, low=diffuse)
    k = max(1, (g * g) // 10)
    conc = float(np.sort(support.reshape(-1))[::-1][:k].sum() / (support.sum() + 1e-8))
    # corner-ness: mass within a corner quadrant (bias signal)
    edge = max(1, g // 4)
    corner = max(s[:edge, :edge].sum(), s[:edge, -edge:].sum(),
                 s[-edge:, :edge].sum(), s[-edge:, -edge:].sum())
    return dict(centroid_y=cy, centroid_x=cx, spread=spread,
                concentration=conc, corner_mass=float(corner))


def top_sample(index_dir, layer, unit):
    files = glob.glob(f"{index_dir}/deciles/layer_part={layer}/**/*.parquet", recursive=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    g = df[df.unit == unit].sort_values("score", ascending=False)
    if len(g) == 0:
        raise ValueError(f"unit {unit} not found in index for {layer}")
    return int(g.iloc[0].sample_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/sae")
    ap.add_argument("--index-dir", default="outputs/cifar_speclens/index")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--layer", default="model.layer4.0")
    ap.add_argument("--units", default="731,1946")
    ap.add_argument("--all", action="store_true", help="run every indexed feature (stats only, no PNG)")
    ap.add_argument("--limit", type=int, default=0, help="cap #features in --all mode (0=all)")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--out-dir", default="outputs/cifar_speclens/fri")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    fri = CnnFri(args.ckpt, args.sae_root, device)
    ds = datasets.CIFAR100(args.data_root, train=True, download=False)
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    out = Path(args.out_dir) / args.layer
    out.mkdir(parents=True, exist_ok=True)

    # load index once -> per-unit top sample (avoids re-reading parquet per feature)
    files = glob.glob(f"{args.index_dir}/deciles/layer_part={args.layer}/**/*.parquet", recursive=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    top = df.sort_values("score", ascending=False).groupby("unit").first()["sample_id"].astype(int)

    if args.all:
        units = sorted(top.index.tolist())
        if args.limit:
            units = units[: args.limit]
        render = False
        csv_name = "fri_stats_all.csv"
    else:
        units = [int(u) for u in args.units.split(",")]
        render = True
        csv_name = "fri_stats.csv"

    rows = []
    for i, unit in enumerate(units):
        sid = int(top[unit])
        img = ds.data[sid]  # HWC uint8
        x = norm(img)
        support, meta = fri.feature_erf(x, args.layer, unit, grid=args.grid, steps=args.steps)
        st = support_stats(support)
        cls = ds.classes[ds.targets[sid]]
        rows.append(dict(unit=int(unit), sample_id=sid, top_class=cls, **meta, **st))

        if render:
            fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.7))
            ax[0].imshow(img); ax[0].set_title(f"{cls} (sid {sid})", fontsize=8); ax[0].axis("off")
            ax[1].imshow(img)
            sup_up = F.interpolate(torch.tensor(support)[None, None], size=(32, 32),
                                   mode="bilinear", align_corners=False)[0, 0].numpy()
            ax[1].imshow(sup_up, cmap="jet", alpha=0.55)
            ax[1].set_title(f"FRI conc={st['concentration']:.2f} corner={st['corner_mass']:.2f}", fontsize=7)
            ax[1].axis("off")
            fig.suptitle(f"{args.layer} feat {unit}  rec={meta['recovery']:.2f}", fontsize=9)
            fig.tight_layout(); fig.savefig(out / f"feat_{unit}_fri.png", dpi=90); plt.close(fig)
            print(f"feat {unit:5d} [{cls:12s}] rec={meta['recovery']:.2f} conc={st['concentration']:.2f} "
                  f"spread={st['spread']:.2f} corner={st['corner_mass']:.2f}")
        elif i % 200 == 0:
            print(f"  [{i}/{len(units)}] feat {unit} corner={st['corner_mass']:.2f}", flush=True)

    pd.DataFrame(rows).to_csv(out / csv_name, index=False)
    print(f"-> {out}/{csv_name}  ({len(rows)} features)")


if __name__ == "__main__":
    main()
