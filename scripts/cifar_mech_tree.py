"""Prototype mechanistic tree for the CIFAR CNN, built TOP-DOWN from a class.

- Seed nodes: top layer4 features by feature->class attribution (fc.weight @ W_dec).
- Edges (lower i -> upper j): CENTROID-based ablation. On the upper node's top image,
  set lower feature i's activation back to its dataset CENTROID (remove its deviation,
  per Visual-Sparse-Steering: always-on bias sits in the centroid and cancels), forward
  the one conv stage, and measure how much upper feature j drops. Big drop = strong edge.
- Recurse layer4 -> layer3 -> layer2. Each node tagged with a bias flag (rel-blank).

This tests the two worries: (a) does it CONNECT (top-down guarantees it), and
(b) does centroid-ablation keep bias features out of the tree.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_mech_tree.py --class-name motorcycle
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD

CHAIN = ["model.conv1", "model.layer1.0", "model.layer2.0", "model.layer3.0", "model.layer4.0"]
LOWER = {"model.layer4.0": "model.layer3.0", "model.layer3.0": "model.layer2.0",
         "model.layer2.0": "model.layer1.0", "model.layer1.0": "model.conv1"}
STAGE = {"model.layer4.0": "layer4", "model.layer3.0": "layer3",
         "model.layer2.0": "layer2", "model.layer1.0": "layer1"}
LEVEL = {"model.conv1": 0, "model.layer1.0": 1, "model.layer2.0": 2,
         "model.layer3.0": 3, "model.layer4.0": 4}


def load_index(index_dir, layer):
    files = glob.glob(f"{index_dir}/deciles/layer_part={layer}/**/*.parquet", recursive=True)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def gated_blank(fri, layer):
    sae = fri.sae(layer); sae.configure_visualization_gating(mode="hard")
    with torch.no_grad():
        h = fri._acts_at(torch.zeros(1, 3, 32, 32, device=fri.device), layer)
        C = h.shape[1]
        b = sae.encode(h[0].permute(1, 2, 0).reshape(-1, C)).max(0).values.cpu().numpy()
    sae.configure_visualization_gating(mode="dict")
    return b


def compute_centroids(fri, layers, data, norm, n=400):
    cents = {}
    sel = np.linspace(0, len(data) - 1, n).astype(int)
    for layer in layers:
        sae = fri.sae(layer); sae.configure_visualization_gating(mode="hard")
        acc, cnt = None, 0
        for s in range(0, len(sel), 64):
            xb = torch.stack([norm(data[int(i)]) for i in sel[s:s + 64]]).to(fri.device)
            with torch.no_grad():
                h = fri._acts_at(xb, layer)
                C = h.shape[1]
                enc = sae.encode(h.permute(0, 2, 3, 1).reshape(-1, C))
            acc = enc.sum(0) if acc is None else acc + enc.sum(0)
            cnt += enc.shape[0]
        cents[layer] = (acc / cnt).detach()
        sae.configure_visualization_gating(mode="dict")
    return cents


def class_attr_layer4(fri):
    fcw = fri.model.fc.weight.detach()           # [C, act]
    Wd = fri.sae("model.layer4.0").W_dec.detach()
    dec = Wd if Wd.shape[1] == fcw.shape[1] else Wd.t()   # [dict, act]
    return (dec @ fcw.t()).cpu().numpy()         # [dict, C]


def node_meta(layer, unit, idx, blank, labels, classes, topk=8):
    g = idx.sort_values("score", ascending=False)
    g = g[g.unit == unit].head(topk)
    sids = g.sample_id.astype(int).tolist()
    vals, counts = np.unique([labels[s] for s in sids], return_counts=True)
    mean_top = float(g.score.mean()) if len(g) else 0.0
    rel = float(blank[unit]) / max(mean_top, 1e-6)
    return dict(unit=int(unit), layer=layer, top_class=classes[int(vals[counts.argmax()])],
                top_sid=sids[0] if sids else 0, rel_blank=rel, bias=bool(rel >= 0.4))


def contributions(fri, idx_up, cents, data, norm, upper_layer, upper_unit, n_keep, cap=200, batch=64):
    lower = LOWER[upper_layer]
    stage = getattr(fri.model, STAGE[upper_layer])
    sid = int(idx_up.sort_values("score", ascending=False).pipe(lambda d: d[d.unit == upper_unit]).iloc[0].sample_id)
    x = norm(data[sid]).unsqueeze(0).to(fri.device)
    sl, su = fri.sae(lower), fri.sae(upper_layer)
    sl.configure_visualization_gating(mode="hard"); su.configure_visualization_gating(mode="hard")
    with torch.no_grad():
        h_low = fri._acts_at(x, lower)
        Cl, Hl, Wl = h_low.shape[1], h_low.shape[2], h_low.shape[3]
        h_up = stage(h_low)
        Cu, Hu, Wu = h_up.shape[1], h_up.shape[2], h_up.shape[3]
        fu = su.encode(h_up[0].permute(1, 2, 0).reshape(-1, Cu))
        cell_j = int(fu[:, upper_unit].argmax()); full_j = float(fu[cell_j, upper_unit])
        fl = sl.encode(h_low[0].permute(1, 2, 0).reshape(-1, Cl))   # [cells, dict_l]
        Wd = sl.W_dec.detach()
        dec = Wd if Wd.shape[0] == fl.shape[1] else Wd.t()           # [dict_l, Cl]
        c_low = cents[lower]
        active = torch.where(fl.max(0).values > 0)[0]
        if len(active) > cap:
            active = active[torch.argsort(fl.sum(0)[active], descending=True)[:cap]]
        active = active.tolist()
        edges = {}
        for s in range(0, len(active), batch):
            chunk = active[s:s + batch]
            Hs = []
            for i in chunk:
                fi = fl[:, i]
                dev = torch.where(fi > 0, fi - c_low[i], torch.zeros_like(fi)).reshape(Hl, Wl)
                Hs.append(h_low - dec[i].view(1, Cl, 1, 1) * dev.view(1, 1, Hl, Wl))
            Hb = torch.cat(Hs, 0)
            hub = stage(Hb)
            fb = su.encode(hub.permute(0, 2, 3, 1).reshape(-1, Cu)).reshape(len(chunk), Hu * Wu, -1)
            jv = fb[:, cell_j, upper_unit]
            for b, i in enumerate(chunk):
                edges[i] = full_j - float(jv[b])
    sl.configure_visualization_gating(mode="dict"); su.configure_visualization_gating(mode="dict")
    ranked = sorted(edges.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] <= 0:
        return []
    thr = 0.25 * ranked[0][1]   # prune weak edges relative to the strongest contributor
    return [(int(i), float(w)) for i, w in ranked[:n_keep] if w > thr]


def build_tree(fri, idx, cents, blank, data, norm, labels, classes, class_idx, n_seeds, n_keep):
    A = class_attr_layer4(fri)[:, class_idx]
    seeds = np.argsort(A)[::-1][:n_seeds].tolist()
    nodes = {4: {}, 3: {}, 2: {}}
    edges = []
    for u in seeds:
        nodes[4][u] = node_meta("model.layer4.0", u, idx["model.layer4.0"], blank["model.layer4.0"], labels, classes)
    for upper_layer in ["model.layer4.0", "model.layer3.0"]:
        lvl = LEVEL[upper_layer]
        for u in list(nodes[lvl]):
            for (i, w) in contributions(fri, idx[upper_layer], cents, data, norm, upper_layer, u, n_keep):
                lo_layer = LOWER[upper_layer]; lo_lvl = LEVEL[lo_layer]
                if i not in nodes[lo_lvl]:
                    nodes[lo_lvl][i] = node_meta(lo_layer, i, idx[lo_layer], blank[lo_layer], labels, classes)
                edges.append((lvl, u, lo_lvl, i, w))
    return nodes, edges, seeds, A


def render(nodes, edges, seeds, A, class_name, data, out_png):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    levelx = {2: 0.0, 3: 1.0, 4: 2.0}
    pos = {}
    for lvl, units in nodes.items():
        ys = np.linspace(0.05, 0.95, len(units) + 2)[1:-1] if units else []
        for (u, _), y in zip(sorted(units.items()), ys):
            pos[(lvl, u)] = (levelx[lvl], y)
    wmax = max([w for *_, w in edges] + [1e-6])
    for (ul, uu, ll, li, w) in edges:
        x1, y1 = pos[(ul, uu)]; x2, y2 = pos[(ll, li)]
        ax.plot([x2, x1], [y2, y1], color="#5a8", lw=0.4 + 3.5 * w / wmax, alpha=0.5, zorder=1)
    amax = max(A[seeds].max(), 1e-6)
    for u in seeds:
        x1, y1 = pos[(4, u)]
        ax.plot([x1, 3.0], [y1, 0.5], color="#c84", lw=0.4 + 3.5 * A[u] / amax, alpha=0.6, zorder=1)
    for (lvl, u), (x, y) in pos.items():
        m = nodes[lvl][u]
        im = OffsetImage(data[m["top_sid"]], zoom=2.4)
        ec = "#e44" if m["bias"] else "#4d4"
        ax.add_artist(AnnotationBbox(im, (x, y), frameon=True, pad=0.1,
                                     bboxprops=dict(edgecolor=ec, lw=2.2), zorder=3))
        tag = " BIAS" if m["bias"] else ""
        ax.text(x, y - 0.052, f"{m['top_class'][:10]}\nL{lvl}·f{u}{tag}", fontsize=6,
                ha="center", va="top", color="#eee", zorder=4)
    ax.text(3.0, 0.5, class_name, fontsize=13, ha="center", va="center", color="#fc8",
            bbox=dict(boxstyle="round", fc="#222", ec="#fc8"), zorder=4)
    for x, lab in [(0, "layer2"), (1, "layer3"), (2, "layer4"), (3, "class")]:
        ax.text(x, 1.0, lab, fontsize=11, ha="center", color="#9cf")
    ax.set_xlim(-0.4, 3.4); ax.set_ylim(0, 1.04); ax.axis("off")
    ax.set_title(f"mechanistic tree (top-down, centroid edges) — class: {class_name}",
                 color="#ddd", fontsize=12)
    fig.patch.set_facecolor("#111")
    fig.tight_layout(); fig.savefig(out_png, dpi=110, facecolor="#111"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/sae")
    ap.add_argument("--index-dir", default="outputs/cifar_speclens/index")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--class-name", default="motorcycle")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-keep", type=int, default=2)
    ap.add_argument("--out-dir", default="outputs/cifar_speclens/tree")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    fri = CnnFri(args.ckpt, args.sae_root, device)
    ds = datasets.CIFAR100(args.data_root, train=True, download=False)
    data, labels, classes = ds.data, np.array(ds.targets), ds.classes
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    class_idx = classes.index(args.class_name)

    idx = {L: load_index(args.index_dir, L) for L in CHAIN}
    blank = {L: gated_blank(fri, L) for L in CHAIN}
    cents = compute_centroids(fri, ["model.layer2.0", "model.layer3.0"], data, norm)

    nodes, edges, seeds, A = build_tree(fri, idx, cents, blank, data, norm, labels, classes,
                                        class_idx, args.n_seeds, args.n_keep)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    png = out / f"tree_{args.class_name}.png"
    render(nodes, edges, seeds, A, args.class_name, data, png)

    nb = sum(m["bias"] for lvl in nodes for m in nodes[lvl].values())
    nt = sum(len(nodes[lvl]) for lvl in nodes)
    print(f"[tree] class={args.class_name} nodes={nt} (L4={len(nodes[4])},L3={len(nodes[3])},L2={len(nodes[2])}) "
          f"edges={len(edges)} bias_nodes={nb}")
    for lvl in [4, 3, 2]:
        for u, m in sorted(nodes[lvl].items()):
            print(f"  L{lvl} f{u:4d} {m['top_class']:14s} rel_blank={m['rel_blank']:.2f}{' BIAS' if m['bias'] else ''}")
    print(f"[tree] -> {png}")


if __name__ == "__main__":
    main()
