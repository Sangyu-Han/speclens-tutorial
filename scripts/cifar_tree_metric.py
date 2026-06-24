"""Faithfulness check for feature->feature attribution: insertion / deletion
metrics in FEATURE space, comparing FRI vs single-feature ablation vs gradient
vs random ranking.

For a target (upper) SAE feature on one of its top images: the candidate LOWER
features are ranked by each method; we then INSERT them (from the all-centroid
baseline) in rank order and measure how fast the target recovers (insertion AUC,
higher=better), and DELETE them (from full) in rank order (deletion AUC,
lower=better). A faithful attribution recovers/destroys the target with the
fewest top-ranked features.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_tree_metric.py
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri
from scripts.cifar_bias_inspect_html import top_samples
from scripts.cifar_mech_tree import (LOWER, STAGE, class_attr_layer4, compute_centroids, load_index)
from src.core.attribution.fri.solver import FRIConfig, run_fri
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD


def build_ctx(fri, upper_layer, upper_unit, sid, data, norm, cents, cap=120):
    lower = LOWER[upper_layer]; stage = getattr(fri.model, STAGE[upper_layer]); dvc = fri.device
    sl, su = fri.sae(lower), fri.sae(upper_layer)
    sl.configure_visualization_gating(mode="hard"); su.configure_visualization_gating(mode="dict")
    x = norm(data[int(sid)]).unsqueeze(0).to(dvc)
    with torch.no_grad():
        h_low = fri._acts_at(x, lower); Cl, Hl, Wl = h_low.shape[1], h_low.shape[2], h_low.shape[3]
        h_up = stage(h_low); Cu, Hu, Wu = h_up.shape[1], h_up.shape[2], h_up.shape[3]
        fu = su.encode(h_up[0].permute(1, 2, 0).reshape(-1, Cu))
        cj = int(fu[:, upper_unit].argmax()); full_j = float(fu[cj, upper_unit])
        fl = sl.encode(h_low[0].permute(1, 2, 0).reshape(-1, Cl))
        active = torch.where(fl.max(0).values > 0)[0]
        if len(active) > cap:
            active = active[torch.argsort(fl.sum(0)[active], descending=True)[:cap]]
        na = int(len(active))
        Wd = sl.W_dec.detach(); decmat = (Wd if Wd.shape[0] == fl.shape[1] else Wd.t())[active]
        c = cents[lower][active]; fa = fl[:, active]
        devmap = torch.where(fa > 0, fa - c.unsqueeze(0), torch.zeros_like(fa))
        h0 = h_low - (devmap @ decmat).t().reshape(1, Cl, Hl, Wl)
        base_j = float(su.encode(stage(h0)[0].permute(1, 2, 0).reshape(-1, Cu))[cj, upper_unit])
    return dict(stage=stage, su=su, h_low=h_low, Cl=Cl, Hl=Hl, Wl=Wl, Cu=Cu, Hu=Hu, Wu=Wu,
                cj=cj, full_j=full_j, base_j=base_j, devmap=devmap, decmat=decmat, na=na,
                dvc=dvc, upper_unit=upper_unit)


def masked_recovery(ctx, masks):  # masks [B, na] in [0,1]; 1=keep feature, 0=remove to centroid
    removed = (1.0 - masks)
    rem = (removed.unsqueeze(1) * ctx["devmap"].unsqueeze(0)) @ ctx["decmat"]   # [B, cells, Cl]
    B = masks.shape[0]
    hp = ctx["h_low"] - rem.permute(0, 2, 1).reshape(B, ctx["Cl"], ctx["Hl"], ctx["Wl"])
    enc = ctx["su"].encode(ctx["stage"](hp).permute(0, 2, 3, 1).reshape(-1, ctx["Cu"]))
    j = enc.reshape(B, ctx["Hu"] * ctx["Wu"], -1)[:, ctx["cj"], ctx["upper_unit"]]
    return (j - ctx["base_j"]) / (ctx["full_j"] - ctx["base_j"])


def rank_fri(ctx):
    na = ctx["na"]; S = int(math.ceil(math.sqrt(na))); P = S * S; dvc = ctx["dvc"]
    ones = torch.ones((), device=dvc); zeros = torch.zeros((), device=dvc)
    res = run_fri(n_patches=P, grid_size=S,
                  objective_for_mask=lambda m: masked_recovery(ctx, m[:na].unsqueeze(0))[0],
                  full_objective=ones, baseline_objective=zeros,
                  irrelevance=torch.ones(P, device=dvc),
                  config=FRIConfig(steps=32, tv_weight=0.0), device=dvc)
    return np.argsort(np.asarray(res.scores)[:na])[::-1]


def rank_ablation(ctx):
    na = ctx["na"]
    masks = torch.ones(na, na, device=ctx["dvc"]) - torch.eye(na, device=ctx["dvc"])  # row i removes i
    with torch.no_grad():
        rec = masked_recovery(ctx, masks).cpu().numpy()   # lower = bigger drop = more important
    return np.argsort(rec)


def rank_grad(ctx):
    m = torch.ones(ctx["na"], device=ctx["dvc"], requires_grad=True)
    rec = masked_recovery(ctx, m.unsqueeze(0))[0]
    g = torch.autograd.grad(rec, m)[0].abs().detach().cpu().numpy()
    return np.argsort(g)[::-1]


def auc(ctx, order, mode, steps=12):
    na = ctx["na"]; ks = np.unique(np.linspace(0, na, steps + 1).astype(int))
    masks = []
    for k in ks:
        m = (torch.zeros if mode == "ins" else torch.ones)(na, device=ctx["dvc"])
        if k > 0:
            idx = torch.as_tensor(np.asarray(order[:k]).copy(), device=ctx["dvc"])
            m[idx] = 1.0 if mode == "ins" else 0.0
        masks.append(m)
    with torch.no_grad():
        rec = masked_recovery(ctx, torch.stack(masks)).clamp(-0.5, 1.5).cpu().numpy()
    return float(np.trapz(rec, ks / na))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/sae")
    ap.add_argument("--index-dir", default="outputs/cifar_speclens/index")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--class-name", default="motorcycle")
    ap.add_argument("--n-targets", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    fri = CnnFri(args.ckpt, args.sae_root, device)
    ds = datasets.CIFAR100(args.data_root, train=True, download=False)
    data = ds.data
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    idx = {L: load_index(args.index_dir, L) for L in ["model.layer2.0", "model.layer3.0", "model.layer4.0"]}
    cents = compute_centroids(fri, ["model.layer2.0", "model.layer3.0"], data, norm, n=300)

    A = class_attr_layer4(fri)[:, ds.classes.index(args.class_name)]
    l4 = np.argsort(A)[::-1][:args.n_targets].tolist()
    # pick L3 targets = a few high-activation layer3 features
    l3 = idx["model.layer3.0"].groupby("unit").score.max().sort_values(ascending=False).index[:args.n_targets].tolist()
    methods = {"FRI": rank_fri, "ablation": rank_ablation, "gradient": rank_grad,
               "random": lambda ctx: np.random.permutation(ctx["na"])}

    for level, upper_layer, targets in [("L4<-L3", "model.layer4.0", l4),
                                        ("L3<-L2", "model.layer3.0", l3)]:
        acc = {m: {"ins": [], "del": []} for m in methods}
        for u in targets:
            for (sid, _, _) in top_samples(idx[upper_layer], int(u), k=args.n_samples):
                ctx = build_ctx(fri, upper_layer, int(u), sid, data, norm, cents)
                if ctx["na"] < 6 or abs(ctx["full_j"] - ctx["base_j"]) < 1e-6:
                    continue
                for mname, fn in methods.items():
                    order = fn(ctx)
                    acc[mname]["ins"].append(auc(ctx, order, "ins"))
                    acc[mname]["del"].append(auc(ctx, order, "del"))
        print(f"\n=== {level}  (insertion AUC higher=better, deletion AUC lower=better) ===")
        print(f"{'method':10s} {'ins_AUC':>8s} {'del_AUC':>8s} {'ins-del':>8s}")
        for m in methods:
            ia = float(np.mean(acc[m]["ins"])) if acc[m]["ins"] else float("nan")
            da = float(np.mean(acc[m]["del"])) if acc[m]["del"] else float("nan")
            print(f"{m:10s} {ia:8.3f} {da:8.3f} {ia-da:8.3f}")


if __name__ == "__main__":
    main()
