"""Per-layer feature inspection HTML across CNN depth (conv1 .. layer4.0).

Each feature card shows, for its top-N distinct samples, THREE rows:
  input  /  activation-map (where the SAE feature fires)  /  ERF set (the minimal
  insertion patch-set that recovers 90% of the feature value -- not a heatmap).

Features are ranked by TOP ACTIVATION SCORE (the "top-score features" tracked for
cross-layer linkage). Bias signals shown: rel-blank = activation on a blank image
as a FRACTION of the feature's typical firing (scale-fair), corner support, freq.

Run (cuda:1 = display-safe):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_inspect_layers.py
"""
from __future__ import annotations

import argparse
import glob
import html
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri, support_stats
from scripts.cifar_bias_inspect_html import top_samples, LAYER_GRID
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD


def render_card3(unit, samples, data, labels, classes, lgrid, cap, out_png, dpi=85, scale=1.0):
    """samples = (sid, act_map[H,W], support_mask[g,g], meta). 3 rows: input/act/ERF-set."""
    K = len(samples)
    fig, axes = plt.subplots(3, K, figsize=(K * 1.25 * scale, 4.1 * scale))
    axes = np.array(axes).reshape(3, K)
    rowlab = ["input", "act-map", "ERF set"]
    for j, (sid, amap, smask, smeta) in enumerate(samples):
        for r in range(3):
            a = axes[r, j]
            a.imshow(data[sid])
            if r == 1:
                au = F.interpolate(torch.tensor(amap)[None, None], size=(32, 32), mode="nearest")[0, 0].numpy()
                a.imshow(au, cmap="inferno", alpha=0.6)
            elif r == 2:
                su = F.interpolate(torch.tensor(smask)[None, None], size=(32, 32), mode="nearest")[0, 0].numpy()
                a.imshow(np.ma.masked_where(su < 0.5, su), cmap="cool", alpha=0.55)
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel(rowlab[r], fontsize=7)
        axes[0, j].set_title(classes[labels[sid]][:9], fontsize=6)
        axes[2, j].set_xlabel(f"{smeta['set_size']}c {smeta['recovery']:.0%}", fontsize=5)
    fig.suptitle(cap, fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi); plt.close(fig)


def gated_blank_act(fri, layer, sae):
    """Per-feature GATED activation on a blank (mean) image -> content-independent bias."""
    xb = torch.zeros(1, 3, 32, 32, device=fri.device)
    sae.configure_visualization_gating(mode="hard")
    with torch.no_grad():
        h = fri._acts_at(xb, layer)
        C = h.shape[1]
        flat = h[0].permute(1, 2, 0).reshape(-1, C)
        blank = sae.encode(flat).max(0).values.cpu().numpy()
    sae.configure_visualization_gating(mode="dict")
    return blank


def per_unit_table(idx, labels, classes, topk):
    rows = []
    for unit, g in idx.groupby("unit"):
        g = g.sort_values("score", ascending=False).head(topk)
        labs = labels[g.sample_id.to_numpy()]
        vals, counts = np.unique(labs, return_counts=True)
        j = int(counts.argmax())
        rows.append(dict(unit=int(unit), mean_top_score=float(g.score.mean()),
                         top_class=classes[int(vals[j])], purity=float(counts[j] / len(g))))
    return pd.DataFrame(rows).set_index("unit")


def run_layer(fri, layer, args, data, labels, classes, norm, out_root):
    sae = fri.sae(layer)
    lgrid = LAYER_GRID[layer]
    files = glob.glob(f"{args.index_dir}/deciles/layer_part={layer}/**/*.parquet", recursive=True)
    idx = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tab = per_unit_table(idx, labels, classes, args.topk)
    ff = pd.read_parquet(f"{args.index_dir}/feature_freq/{layer}.parquet").set_index("unit")
    tab["freq_pct"] = ff.freq_pct.reindex(tab.index).values
    tab["blank_act"] = gated_blank_act(fri, layer, sae)[tab.index.values]
    tab["rel_blank"] = tab.blank_act / tab.mean_top_score.clip(lower=1e-6)

    top_score = tab.sort_values("mean_top_score", ascending=False).head(args.n_top).index.tolist()
    blank = tab[tab.rel_blank >= 0.4].sort_values("rel_blank", ascending=False).head(16).index.tolist()
    to_render = list(dict.fromkeys(top_score + blank))

    out = out_root / layer
    panels = out / "panels"; panels.mkdir(parents=True, exist_ok=True)
    print(f"[{layer}] features={len(tab)} | rendering {len(to_render)} "
          f"(top {len(top_score)} + rel-blank {len(blank)})", flush=True)

    corner, caps = {}, {}
    for i, unit in enumerate(to_render):
        samples = []
        for (sid, y, x) in top_samples(idx, unit, k=args.topk):
            xi = norm(data[int(sid)])
            amap = fri.feat_map_gated(xi, layer, int(unit))
            _, smask, smeta = fri.feature_support(xi, layer, int(unit), grid=args.grid, steps=args.steps)
            samples.append((int(sid), amap, smask, smeta))
        corner[unit] = support_stats(samples[0][2])["corner_mass"]
        r = tab.loc[unit]
        setsz = float(np.mean([s[3]["set_size"] for s in samples]))
        flags = ",".join([f for f, on in [("blank", r.rel_blank >= 0.4),
                                          ("corner", corner[unit] >= 0.25)] if on]) or "-"
        cap = (f"f{unit} | score {r.mean_top_score:.1f} | topN {r.top_class} (pur {r.purity:.0%}) "
               f"| freq {r.freq_pct:.0%} | blank {r.blank_act:.1f}(x{r.rel_blank:.2f}) "
               f"| set~{setsz:.0f}/{args.grid**2} corner {corner[unit]:.2f} | {flags}")
        caps[unit] = cap
        render_card3(int(unit), samples, data, labels, classes, lgrid, cap, panels / f"f{unit}.png")
        if i % 15 == 0:
            print(f"  [{layer}] {i}/{len(to_render)}", flush=True)

    tab["corner_mass"] = pd.Series(corner)
    tab.to_csv(out / "signals.csv")
    conflict = [u for u in top_score if (tab.loc[u].rel_blank >= 0.4 or corner.get(u, 0) >= 0.25)]

    def sec(title, desc, units):
        cards = "".join(
            f'<div class=c><img src="{layer}/panels/f{u}.png"><div class=cap>{html.escape(caps[u])}</div></div>'
            for u in units if u in caps)
        return f"<h3>{html.escape(title)}</h3><p class=d>{html.escape(desc)}</p><div class=g>{cards}</div>"

    body = [
        f"<h2 id={layer}>{html.escape(layer)} &nbsp;<span class=d>({len(tab)} features, grid {lgrid}x{lgrid})</span></h2>",
        sec("Top activation-score", "Highest-firing features — the 'top-score' nodes for cross-layer linkage. Real concept, low-level primitive, or bias?", top_score),
        sec("Fires-on-blank (rel>=0.4)", "Fire >=40% of their typical value on a blank image = content-independent bias (scale-fair).", blank),
        sec("Top-score AND flagged bias", "Top-score yet blank/corner flagged -- would pollute the tree.", conflict),
    ]
    return out, body, dict(layer=layer, n=len(tab), n_blank=len(blank),
                           n_conflict=len(conflict), n_top=len(top_score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/sae")
    ap.add_argument("--index-dir", default="outputs/cifar_speclens/index")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--layers", default="model.conv1,model.layer1.0,model.layer2.0,model.layer3.0,model.layer4.0")
    ap.add_argument("--out-dir", default="outputs/cifar_speclens/inspect_layers")
    ap.add_argument("--n-top", type=int, default=24)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    fri = CnnFri(args.ckpt, args.sae_root, device)
    ds = datasets.CIFAR100(args.data_root, train=True, download=False)
    data, labels, classes = ds.data, np.array(ds.targets), ds.classes
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    bodies, summary = [], []
    for layer in args.layers.split(","):
        _, body, st = run_layer(fri, layer, args, data, labels, classes, norm, out_root)
        bodies += body
        summary.append(st)

    nav = " | ".join(f'<a href="#{s["layer"]}">{s["layer"]}</a>' for s in summary)
    tbl = "".join(f"<tr><td>{s['layer']}</td><td>{s['n']}</td><td>{s['n_top']}</td>"
                  f"<td>{s['n_blank']}</td><td>{s['n_conflict']}</td></tr>" for s in summary)
    doc = ["<html><head><meta charset=utf-8><style>",
           "body{font-family:sans-serif;background:#111;color:#ddd;margin:14px}",
           "h2{margin:26px 0 2px;color:#fc8}h3{margin:14px 0 2px;color:#9cf}",
           ".d{color:#888;font-size:12px;margin:0 0 8px}",
           ".g{display:flex;flex-wrap:wrap;gap:8px}.c{background:#1b1b1b;padding:5px;border-radius:5px;width:560px}",
           ".c img{width:100%}.cap{font-size:10px;color:#cfcfcf;margin-top:3px;font-family:monospace}",
           "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #333;padding:3px 8px;font-size:12px}",
           "a{color:#6cf}</style></head><body>",
           "<h1>CNN feature inspection by layer</h1>",
           f"<p class=d>nav: {nav}</p>",
           "<table><tr><th>layer</th><th>#feat</th><th>#top</th><th>#rel-blank</th><th>#top&amp;bias</th></tr>",
           tbl, "</table>",
           "<p class=d>Each card, per sample, 3 rows: <b>input</b> / <b>activation-map</b> (where the SAE feature fires) "
           "/ <b>ERF set</b> (minimal insertion patches recovering 90% of the feature; 'Nc P%' = set size & recovery). "
           "blank x0.NN = blank firing as a fraction of typical (>=0.40 flagged bias).</p>",
           *bodies, "</body></html>"]
    (out_root / "index.html").write_text("\n".join(doc))
    print(f"[inspect-layers] -> {out_root}/index.html")
    for s in summary:
        print(f"  {s['layer']:18s} feats={s['n']:5d} top={s['n_top']} rel-blank={s['n_blank']} top&bias={s['n_conflict']}")


if __name__ == "__main__":
    main()
