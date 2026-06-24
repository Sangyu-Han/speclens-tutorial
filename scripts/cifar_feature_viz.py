"""Visualize SAE features of the CIFAR CNN from the SpecLens index.

For each feature (unit) of a layer:
  - pull its top-N activating CIFAR samples from the deciles parquet,
  - reconstruct the raw 32x32 images by sample_id,
  - draw the activating (y,x) feature-map cell,
  - use CIFAR labels to compute a per-feature summary (freq, dominant class, purity).

Outputs (under --out-dir/<layer>/):
  feature_summary.csv          per-feature stats (unit, freq_pct, top_class, purity, n_classes, mean_top_score)
  panels/feat_<unit>.png       contact sheet of top-N samples
  index.html                   gallery: most-selective + most-generic(bias-suspect) sections

CPU-only, light (Colab-friendly). Example:
  python scripts/cifar_feature_viz.py --layer model.layer4.0 --topk 16 --n-render 48
"""
from __future__ import annotations

import argparse
import glob
import html
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torchvision import datasets

# input-grid side per layer (input 32x32) -> feature-map grid side
LAYER_GRID = {
    "model.conv1": 32, "model.layer1.0": 32, "model.layer2.0": 16,
    "model.layer3.0": 8, "model.layer4.0": 4,
}


def load_index(index_dir: str, layer: str):
    files = glob.glob(f"{index_dir}/deciles/layer_part={layer}/**/*.parquet", recursive=True)
    if not files:
        raise FileNotFoundError(f"no deciles parquet for {layer} under {index_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ff = pd.read_parquet(f"{index_dir}/feature_freq/{layer}.parquet")
    return df, ff


def summarize(df: pd.DataFrame, ff: pd.DataFrame, labels: np.ndarray, topk: int) -> pd.DataFrame:
    freq = dict(zip(ff.unit.tolist(), ff.freq_pct.tolist()))
    rows = []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("score", ascending=False).head(topk)
        labs = labels[g.sample_id.to_numpy()]
        vals, counts = np.unique(labs, return_counts=True)
        j = int(counts.argmax())
        rows.append(dict(
            unit=int(unit), n_top=int(len(g)), mean_top_score=float(g.score.mean()),
            freq_pct=float(freq.get(int(unit), np.nan)),
            top_class=int(vals[j]), purity=float(counts[j] / len(g)), n_classes=int(len(vals)),
        ))
    return pd.DataFrame(rows).sort_values("unit").reset_index(drop=True)


def render_panel(unit, g, data, labels, classes, grid, topk, out_png):
    g = g.sort_values("score", ascending=False).head(topk)
    n = len(g)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.5))
    axes = np.array(axes).reshape(-1)
    cell = 32.0 / grid
    for ax, (_, r) in zip(axes, g.iterrows()):
        sid = int(r.sample_id)
        ax.imshow(data[sid])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(classes[labels[sid]][:11], fontsize=6)
        ax.add_patch(patches.Rectangle(
            (r.x * cell - 0.5, r.y * cell - 0.5), cell, cell, lw=1.3, ec="lime", fc="none"))
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"feat {int(unit)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def write_html(summary, classes, layer, panels_rel, out_html, n_render):
    sel = summary.sort_values(["purity", "mean_top_score"], ascending=False).head(n_render)
    # bias-suspect: generic = high freq, low purity, many classes
    bias = summary.sort_values(["freq_pct", "n_classes"], ascending=False)
    bias = bias[bias.purity < 0.5].head(max(8, n_render // 3))

    def card(r):
        png = f"{panels_rel}/feat_{int(r.unit)}.png"
        cap = (f"feat {int(r.unit)} | top: {html.escape(classes[int(r.top_class)])} "
               f"(pur {r.purity:.0%}) | freq {r.freq_pct:.1%} | {int(r.n_classes)} cls")
        return f'<div class=c><img src="{png}"><div class=cap>{cap}</div></div>'

    doc = ["<html><head><meta charset=utf-8><style>",
           "body{font-family:sans-serif;background:#111;color:#ddd;margin:16px}",
           "h2{margin:18px 0 6px}.g{display:flex;flex-wrap:wrap;gap:10px}",
           ".c{background:#1c1c1c;padding:6px;border-radius:6px;width:230px}",
           ".c img{width:100%}.cap{font-size:11px;margin-top:4px;color:#bbb}</style></head><body>",
           f"<h1>{html.escape(layer)} — SAE features</h1>",
           f"<p>{len(summary)} features. Lime box = activating feature-map cell.</p>",
           "<h2>Most selective (candidate concept features)</h2><div class=g>",
           *[card(r) for _, r in sel.iterrows()], "</div>",
           "<h2>Most generic (bias-suspect: high freq, low purity)</h2><div class=g>",
           *[card(r) for _, r in bias.iterrows()], "</div></body></html>"]
    Path(out_html).write_text("\n".join(doc))
    return sel, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="outputs/cifar_speclens/index")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--layer", default="model.layer4.0")
    ap.add_argument("--out-dir", default="outputs/cifar_speclens/feature_viz")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--n-render", type=int, default=48)
    args = ap.parse_args()

    grid = LAYER_GRID[args.layer]
    ds = datasets.CIFAR100(args.data_root, train=True, download=False)
    data = ds.data  # [N,32,32,3] uint8
    labels = np.array(ds.targets)
    classes = ds.classes

    df, ff = load_index(args.index_dir, args.layer)
    summary = summarize(df, ff, labels, args.topk)
    summary["top_class_name"] = [classes[c] for c in summary.top_class]

    out = Path(args.out_dir) / args.layer
    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "feature_summary.csv", index=False)

    # render only the features that will appear in the gallery (keep it light)
    sel, bias = write_html(summary, classes, args.layer, "panels", out / "index.html", args.n_render)
    to_render = pd.concat([sel, bias]).drop_duplicates("unit")
    gb = {u: g for u, g in df.groupby("unit")}
    for _, r in to_render.iterrows():
        render_panel(int(r.unit), gb[int(r.unit)], data, labels, classes, grid, args.topk,
                     panels / f"feat_{int(r.unit)}.png")

    print(f"[viz] {args.layer}: {len(summary)} features | rendered {len(to_render)} panels")
    print(f"[viz] purity: median {summary.purity.median():.2f} max {summary.purity.max():.2f} | "
          f"freq_pct median {summary.freq_pct.median():.3f}")
    print(f"[viz] -> {out/'index.html'}  ({out/'feature_summary.csv'})")


if __name__ == "__main__":
    main()
