"""Per-SAMPLE misclassification tree: for ONE wrongly-classified image, WHY did it
go to the wrong class?

Exploits layer4 -> GAP -> fc being LINEAR:  logit[c] = bias[c] + sum_f act_f * A[f,c]
(A = W_dec @ fc.weight^T), so for THIS image each SAE feature's push toward the wrong
class is exact: push_B(f) = act_f(this image) * A[f, B].

  - Root  = the WRONG predicted class B.
  - Seeds = layer4 features with the largest push toward B on THIS image (the culprits).
            For each we also show its push toward the TRUE class A (B-specific vs shared).
  - Tree  = decompose each culprit into the lower-layer features that build it ON THIS
            IMAGE (per-sample FRI, the image's own activations -- not class top-samples).
  - Nodes = the feature's activation map ON THE MISCLASSIFIED IMAGE (where it fired).
  - Contrast strip = the true class A's top evidence features, and how weakly they
            fired here ("the evidence the model needed for A but didn't get").

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
   /home/sangyu/anaconda3/envs/py312/bin/python scripts/cifar_misclass_tree.py \
       --true bicycle --pred motorcycle
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torchvision import datasets, transforms

from scripts.cifar_mech_tree import (LOWER, STAGE, class_attr_layer4, compute_centroids,
                                     gated_blank, load_index, node_meta)
from scripts.cifar_fri_feature import CnnFri
from src.core.attribution.fri.solver import FRIConfig, run_fri
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD

LAYER4 = "model.layer4.0"
LAYER_OF = {0: "model.conv1", 1: "model.layer1.0", 2: "model.layer2.0",
            3: "model.layer3.0", 4: "model.layer4.0"}


def sae_meanact(fri, x, layer):
    """SAE feature activations, mean over the layer's spatial cells (matches GAP)."""
    sae = fri.sae(layer); sae.configure_visualization_gating(mode="hard")
    with torch.no_grad():
        h = fri._acts_at(x, layer); C = h.shape[1]
        enc = sae.encode(h[0].permute(1, 2, 0).reshape(-1, C))
    sae.configure_visualization_gating(mode="dict")
    return enc.mean(0).cpu().numpy()


def decompose_sample(fri, cents, x, upper_layer, upper_unit, n_keep=3, steps=28, cap=120):
    """Per-image FRI: which LOWER features (on THIS image) build the upper feature."""
    lower = LOWER[upper_layer]; stage = getattr(fri.model, STAGE[upper_layer])
    sl, su = fri.sae(lower), fri.sae(upper_layer)
    sl.configure_visualization_gating(mode="hard"); su.configure_visualization_gating(mode="dict")
    dvc = fri.device; ones = torch.ones((), device=dvc); zeros = torch.zeros((), device=dvc)
    with torch.no_grad():
        h_low = fri._acts_at(x, lower); Cl, Hl, Wl = h_low.shape[1], h_low.shape[2], h_low.shape[3]
        h_up = stage(h_low); Cu = h_up.shape[1]
        fu = su.encode(h_up[0].permute(1, 2, 0).reshape(-1, Cu))
        cj = int(fu[:, upper_unit].argmax()); full_j = float(fu[cj, upper_unit])
        if full_j <= 0:
            return []
        fl = sl.encode(h_low[0].permute(1, 2, 0).reshape(-1, Cl))
        active = torch.where(fl.max(0).values > 0)[0]
        if len(active) > cap:
            active = active[torch.argsort(fl.sum(0)[active], descending=True)[:cap]]
        na = int(len(active))
        if na == 0:
            return []
        Wd = sl.W_dec.detach(); decmat = (Wd if Wd.shape[0] == fl.shape[1] else Wd.t())[active]
        c = cents[lower][active]; fa = fl[:, active]
        devmap = torch.where(fa > 0, fa - c.unsqueeze(0), torch.zeros_like(fa))
        h0 = h_low - (devmap @ decmat).t().reshape(1, Cl, Hl, Wl)
        base_j = float(su.encode(stage(h0)[0].permute(1, 2, 0).reshape(-1, Cu))[cj, upper_unit])
    denom = full_j - base_j
    if abs(denom) < 1e-6:
        return []
    S = int(math.ceil(math.sqrt(na))); P = S * S

    def obj(m):
        rem = (((1.0 - m[:na]).unsqueeze(0) * devmap) @ decmat).t().reshape(1, Cl, Hl, Wl)
        jp = su.encode(stage(h_low - rem)[0].permute(1, 2, 0).reshape(-1, Cu))[cj, upper_unit]
        return (jp - base_j) / denom

    res = run_fri(n_patches=P, grid_size=S, objective_for_mask=obj, full_objective=ones,
                  baseline_objective=zeros, irrelevance=torch.ones(P, device=dvc),
                  config=FRIConfig(steps=steps, tv_weight=0.0), device=dvc)
    sc = np.asarray(res.scores, dtype=np.float32)[:na]
    sl.configure_visualization_gating(mode="dict"); su.configure_visualization_gating(mode="dict")
    order = np.argsort(sc)[::-1][:n_keep]
    return [(int(active[k]), float(sc[k])) for k in order if sc[k] > 0.05]


def overlay(img_uint8, amap, alpha=0.6):
    a = amap / (amap.max() + 1e-6)
    a_up = F.interpolate(torch.tensor(a)[None, None].float(), size=(32, 32),
                         mode="bilinear", align_corners=False)[0, 0].numpy()
    heat = cm.inferno(a_up)[..., :3]
    base = img_uint8.astype(np.float32) / 255.0
    w = (alpha * a_up)[..., None]
    return np.clip(base * (1 - w) + heat * w, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/sae")
    ap.add_argument("--index-dir", default="outputs/cifar_speclens/index")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--true", default="bicycle")
    ap.add_argument("--pred", default="motorcycle")
    ap.add_argument("--sample-id", type=int, default=-1, help="explicit TEST index (overrides true/pred search)")
    ap.add_argument("--n-culprit", type=int, default=4)
    ap.add_argument("--out-dir", default="outputs/cifar_speclens/misclass")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    fri = CnnFri(args.ckpt, args.sae_root, device)
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    tr = datasets.CIFAR100(args.data_root, train=True, download=False)   # index lives in train space
    te = datasets.CIFAR100(args.data_root, train=False, download=False)
    classes = tr.classes; labels_tr = np.array(tr.targets)
    te_data = te.data; te_y = np.array(te.targets)

    # ---- choose a misclassified TEST sample (most confident wrong pred for the pair) ----
    @torch.no_grad()
    def predict(idx):
        x = norm(te_data[idx]).unsqueeze(0).to(device)
        return F.softmax(fri.model(x), 1)[0].cpu().numpy()

    if args.sample_id >= 0:
        sid = args.sample_id; p = predict(sid); B = int(p.argmax()); A = int(te_y[sid])
    else:
        A = classes.index(args.true); Bp = classes.index(args.pred)
        cand = [i for i in range(len(te_y)) if te_y[i] == A]
        scored = []
        for i in cand:
            p = predict(i)
            if int(p.argmax()) == Bp:
                scored.append((p[Bp], i))
        if not scored:                       # fall back: any misclassified of the true class
            for i in cand:
                p = predict(i); pb = int(p.argmax())
                if pb != A:
                    scored.append((p[pb], i))
        scored.sort(reverse=True)
        sid = scored[0][1]; A = int(te_y[sid]); B = int(predict(sid).argmax())
    p = predict(sid)
    print(f"[misclass] sample {sid}: true={classes[A]} pred={classes[B]} "
          f"p(pred)={p[B]:.2f} p(true)={p[A]:.2f}")

    x = norm(te_data[sid]).unsqueeze(0).to(device)
    Amat = class_attr_layer4(fri)                       # [dict, 100]
    act4 = sae_meanact(fri, x, LAYER4)                  # [dict]
    pushB = act4 * Amat[:, B]; pushA = act4 * Amat[:, A]
    culprits = [u for u in np.argsort(pushB)[::-1] if pushB[u] > 0][:args.n_culprit]

    idx = {L: load_index(args.index_dir, L) for L in [LAYER_OF[3], LAYER4]}
    blank = {L: gated_blank(fri, L) for L in [LAYER_OF[3], LAYER4]}
    cents = compute_centroids(fri, [LAYER_OF[3]], tr.data, norm)
    meta4 = {u: node_meta(LAYER4, int(u), idx[LAYER4], blank[LAYER4], labels_tr, classes) for u in culprits}

    # decompose the strongest culprit one level down (per-sample)
    top = int(culprits[0])
    children = decompose_sample(fri, cents, x, LAYER4, top, n_keep=3)
    meta3 = {i: node_meta(LAYER_OF[3], int(i), idx[LAYER_OF[3]], blank[LAYER_OF[3]], labels_tr, classes)
             for (i, _) in children}

    # true-class evidence that was WEAK here: GENUINE class-A detector features
    # (index top-class == A) with the highest A-attribution, and how weakly they fired
    cand = np.argsort(Amat[:, A])[::-1][:60].tolist()
    needA = [u for u in cand if node_meta(LAYER4, int(u), idx[LAYER4], blank[LAYER4],
             labels_tr, classes)["top_class"] == classes[A]][:4]
    while len(needA) < 4:                                # fallback if few genuine detectors
        for u in cand:
            if u not in needA:
                needA.append(u); break
    print("  culprits (push->B):")
    for u in culprits:
        print(f"    f{u:5d} {meta4[u]['top_class']:12s} ->{classes[B]}+{pushB[u]:.2f} "
              f"->{classes[A]}{pushA[u]:+.2f}{'  BIAS' if meta4[u]['bias'] else ''}")

    # ----------------------------- render -----------------------------
    fig = plt.figure(figsize=(13.5, 7.6)); fig.patch.set_facecolor("#111")
    gs = fig.add_gridspec(1, 1); ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    amaps = {}

    def node(unit, layer, cx, cy, zoom=2.4, ring="#4d4"):
        amaps.setdefault((layer, unit), fri.feat_map_gated(norm(te_data[sid]), layer, int(unit)))
        im = OffsetImage(overlay(te_data[sid], amaps[(layer, unit)]), zoom=zoom)
        ax.add_artist(AnnotationBbox(im, (cx, cy), frameon=True, pad=0.08,
                                     bboxprops=dict(edgecolor=ring, lw=2.2), zorder=3))

    # input image (top-left)
    im = OffsetImage(te_data[sid], zoom=3.2)
    ax.add_artist(AnnotationBbox(im, (1.1, 8.6), frameon=True, pad=0.1,
                                 bboxprops=dict(edgecolor="#9cf", lw=2), zorder=3))
    ax.text(1.1, 7.5, f"input\ntrue: {classes[A]}\npred: {classes[B]} ({p[B]:.0%})",
            color="#fff", fontsize=10, ha="center", va="top")

    # positions of L4 culprits first (top culprit anchors the L3 children)
    cyl = np.linspace(1.8, 8.2, len(culprits)); ytop = cyl[0]
    # L3 children of the top culprit (left column)
    ys = np.linspace(max(1.8, ytop - 2.2), min(8.2, ytop + 2.2), max(len(children), 1))
    for (i, w), yy in zip(children, ys):
        node(int(i), LAYER_OF[3], 2.9, yy, zoom=1.6, ring="#e44" if meta3[i]["bias"] else "#4d4")
        ax.plot([3.3, 5.2], [yy, ytop], color="#5a8", lw=0.5 + 3 * w, alpha=0.5, zorder=1)
        ax.text(2.9, yy - 0.5, f"L3 f{i} {meta3[i]['top_class'][:9]} {w:.0%}",
                color="#bbb", fontsize=6.3, ha="center", va="top")

    # L4 culprit nodes (middle column)
    for u, yy in zip(culprits, cyl):
        ring = "#e44" if meta4[u]["bias"] else ("#fc6" if u == top else "#4d4")
        node(int(u), LAYER4, 5.7, yy, zoom=1.8, ring=ring)
        ax.plot([6.1, 8.4], [yy, 5.0], color="#c84", lw=0.5 + 3 * pushB[u] / pushB[culprits].max(),
                alpha=0.6, zorder=1)
        shared = f"shared w/ {classes[A]}" if pushA[u] > 0.25 * pushB[u] else f"{classes[B]}-specific"
        ax.text(5.7, yy - 0.52, f"f{u} {meta4[u]['top_class'][:10]}  →{classes[B]} +{pushB[u]:.1f}\n({shared})",
                color="#eee", fontsize=6.5, ha="center", va="top")

    # wrong class box (right)
    ax.text(8.9, 5.0, classes[B], color="#fc8", fontsize=14, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#222", ec="#fc8"), zorder=4)
    ax.text(8.9, 4.0, f"WHY: these features fired on the image\nand pushed it to {classes[B]}",
            color="#c84", fontsize=8, ha="center", va="top")

    # missing true-class evidence (bottom strip)
    ax.text(0.3, 1.3, f"evidence for TRUE class '{classes[A]}' — barely fired here:",
            color="#9cf", fontsize=9, ha="left", va="center")
    for k, u in enumerate(needA):
        amaps.setdefault((LAYER4, int(u)), fri.feat_map_gated(norm(te_data[sid]), LAYER4, int(u)))
        cx = 3.4 + k * 1.7
        im2 = OffsetImage(overlay(te_data[sid], amaps[(LAYER4, int(u))]), zoom=1.5)
        ax.add_artist(AnnotationBbox(im2, (cx, 0.7), frameon=True, pad=0.06,
                                     bboxprops=dict(edgecolor="#668", lw=1.5), zorder=3))
        mu = node_meta(LAYER4, int(u), idx[LAYER4], blank[LAYER4], labels_tr, classes)
        ax.text(cx, -0.05, f"f{u} {mu['top_class'][:9]}\nact={act4[u]:.2f}",
                color="#aab", fontsize=6.5, ha="center", va="top")

    ax.set_title(f"Why was this misclassified?  true={classes[A]} → predicted={classes[B]}  "
                 f"(orange=top culprit, red ring=bias feature)", color="#ddd", fontsize=12)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    png = out / f"misclass_{sid}_{classes[A]}_to_{classes[B]}.png"
    fig.savefig(png, dpi=120, facecolor="#111", bbox_inches="tight"); plt.close(fig)
    print(f"[misclass] -> {png}")


if __name__ == "__main__":
    main()
