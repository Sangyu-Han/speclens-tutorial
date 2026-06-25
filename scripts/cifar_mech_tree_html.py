"""Interactive HTML mechanistic tree with a RIGHT-SIDE feature panel + drill-down.

Left: the top-down tree graph (nodes = SAE features, edges = aggregated centroid
contribution). Right (fixed): clicking ANY feature shows its FULL visualization
- 5 top samples, each as input / activation-map / ERF 90%-recovery set - so you
can actually tell what the feature is. Below it, a clickable "composed of" strip
lists the feature's top contributors; click one to drill into ITS visualization.

Edges/contributions are averaged over the target's top-N activating samples.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_mech_tree_html.py --class-name motorcycle
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri
from scripts.cifar_inspect_layers import render_card3

LAYER_GRID = {"model.conv1": 32, "model.layer1.0": 32, "model.layer2.0": 16,
              "model.layer3.0": 8, "model.layer4.0": 4}


def top_samples(idx_layer, unit, k=5):
    """Top-k activating samples (sample_id, y, x) for a feature, from the index df."""
    g = idx_layer[idx_layer.unit == unit].sort_values("score", ascending=False).head(k)
    return [(int(r.sample_id), int(r.y), int(r.x)) for r in g.itertuples()]
from scripts.cifar_mech_tree import (CHAIN, LEVEL, LOWER, STAGE, class_attr_layer4,
                                     compute_centroids, gated_blank, load_index, node_meta)
from src.core.attribution.fri.solver import FRIConfig, run_fri
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD

LAYER_OF = {0: "model.conv1", 1: "model.layer1.0", 2: "model.layer2.0",
            3: "model.layer3.0", 4: "model.layer4.0"}


def node_class(idx_layer, unit, labels, classes, k=8):
    g = idx_layer.sort_values("score", ascending=False)
    g = g[g.unit == unit].head(k)
    if len(g) == 0:
        return "?"
    vals, counts = np.unique([labels[int(s)] for s in g.sample_id], return_counts=True)
    return classes[int(vals[counts.argmax()])]


def decompose(fri, idx_up, cents, data, norm, upper_layer, upper_unit,
              n_keep=6, n_samples=8, steps=32, cap=120, min_contrib=0.05):
    """FRI attribution of LOWER features to the (upper) target feature, averaged over
    the target's top samples. Soft-masks each active lower feature toward its dataset
    CENTROID and runs random-budget FRI to find which set rebuilds the target's value
    (no 2D prior). Handles redundancy far better than single-feature ablation."""
    lower = LOWER[upper_layer]; stage = getattr(fri.model, STAGE[upper_layer])
    sids = [int(s) for (s, _, _) in top_samples(idx_up, upper_unit, k=n_samples)]
    sl, su = fri.sae(lower), fri.sae(upper_layer)
    sl.configure_visualization_gating(mode="hard")     # actual sparse lower codes
    su.configure_visualization_gating(mode="dict")     # differentiable upper readout
    dvc = fri.device
    ones = torch.ones((), device=dvc); zeros = torch.zeros((), device=dvc)
    agg = {}
    for sid in sids:
        x = norm(data[sid]).unsqueeze(0).to(dvc)
        with torch.no_grad():
            h_low = fri._acts_at(x, lower); Cl, Hl, Wl = h_low.shape[1], h_low.shape[2], h_low.shape[3]
            h_up = stage(h_low); Cu = h_up.shape[1]
            fu = su.encode(h_up[0].permute(1, 2, 0).reshape(-1, Cu))
            cj = int(fu[:, upper_unit].argmax()); full_j = float(fu[cj, upper_unit])
            fl = sl.encode(h_low[0].permute(1, 2, 0).reshape(-1, Cl))      # [cells, dict_l] gated
            active = torch.where(fl.max(0).values > 0)[0]
            if len(active) > cap:
                active = active[torch.argsort(fl.sum(0)[active], descending=True)[:cap]]
            na = int(len(active))
            if na == 0:
                continue
            Wd = sl.W_dec.detach(); decmat = (Wd if Wd.shape[0] == fl.shape[1] else Wd.t())[active]  # [na, Cl]
            c = cents[lower][active]
            fa = fl[:, active]
            devmap = torch.where(fa > 0, fa - c.unsqueeze(0), torch.zeros_like(fa))   # [cells, na]
            h0 = h_low - (devmap @ decmat).t().reshape(1, Cl, Hl, Wl)                  # all -> centroid
            base_j = float(su.encode(stage(h0)[0].permute(1, 2, 0).reshape(-1, Cu))[cj, upper_unit])
        denom = full_j - base_j
        if abs(denom) < 1e-6:
            continue
        S = int(math.ceil(math.sqrt(na))); P = S * S

        def objective_for_mask(m):
            rem = (((1.0 - m[:na]).unsqueeze(0) * devmap) @ decmat).t().reshape(1, Cl, Hl, Wl)
            jp = su.encode(stage(h_low - rem)[0].permute(1, 2, 0).reshape(-1, Cu))[cj, upper_unit]
            return (jp - base_j) / denom

        res = run_fri(n_patches=P, grid_size=S, objective_for_mask=objective_for_mask,
                      full_objective=ones, baseline_objective=zeros,
                      irrelevance=torch.ones(P, device=dvc),
                      config=FRIConfig(steps=steps, tv_weight=0.0), device=dvc)
        sc = np.asarray(res.scores, dtype=np.float32)[:na]
        for k, i in enumerate(active.tolist()):
            agg[i] = agg.get(i, 0.0) + float(sc[k])
    sl.configure_visualization_gating(mode="dict"); su.configure_visualization_gating(mode="dict")
    mean = {i: v / max(len(sids), 1) for i, v in agg.items()}
    return [(int(i), float(v)) for i, v in sorted(mean.items(), key=lambda kv: -kv[1])[:n_keep] if v > min_contrib]


def render_thumb(unit, idx_layer, data, out_png):
    sids = [s for (s, _, _) in top_samples(idx_layer, unit, k=4)]
    while len(sids) < 4:
        sids.append(sids[-1] if sids else 0)
    fig, axes = plt.subplots(2, 2, figsize=(1.3, 1.3))
    for ax, sid in zip(axes.flat, sids):
        ax.imshow(data[sid]); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1, 0.04, 0.04)
    fig.savefig(out_png, dpi=70); plt.close(fig)


def render_strip_thumb(fri, lower_layer, child_unit, parent_sids, own_sids, data, norm, out_png):
    """Composite strip thumbnail: row0 = the contributor's OWN top samples + act-map
    (what concept it is), row1 = where it fires on the PARENT's images (how it contributes)."""
    cols = 3
    own = ((own_sids or parent_sids) + parent_sids)[:cols]
    par = (parent_sids + (own_sids or parent_sids))[:cols]
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.2, 2.55))
    for r, (row_sids, alpha) in enumerate([(own, 0.5), (par, 0.62)]):
        for c, sid in enumerate(row_sids):
            ax = axes[r, c]; ax.imshow(data[int(sid)])
            amap = fri.feat_map_gated(norm(data[int(sid)]), lower_layer, int(child_unit))
            au = F.interpolate(torch.tensor(amap)[None, None], size=(32, 32), mode="nearest")[0, 0].numpy()
            ax.imshow(au, cmap="inferno", alpha=alpha); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(["concept", "on target"][r], fontsize=8)
    fig.subplots_adjust(0.08, 0.02, 0.99, 0.98, 0.05, 0.06)
    fig.savefig(out_png, dpi=80); plt.close(fig)


def render_detail(fri, lvl, unit, idx_layer, data, norm, labels, classes, meta, out_png, steps=28):
    layer = LAYER_OF[lvl]; lgrid = LAYER_GRID[layer]
    samples = []
    for (sid, y, x) in top_samples(idx_layer, unit, k=5):
        xi = norm(data[int(sid)])
        amap = fri.feat_map_gated(xi, layer, int(unit))
        _, smask, smeta = fri.feature_support(xi, layer, int(unit), grid=16, steps=steps)
        samples.append((int(sid), amap, smask, smeta))
    cap = (f"L{lvl} f{unit} | top-class {meta['top_class']} | rel-blank {meta['rel_blank']:.2f}"
           f"{'  (BIAS)' if meta['bias'] else ''}   rows: input / act-map / ERF 90% set")
    render_card3(int(unit), samples, data, labels, classes, lgrid, cap, out_png, dpi=128, scale=1.45)


def build(fri, idx, cents, blank, data, norm, labels, classes, class_idx, n_seeds):
    """Top-down: class -> L4 seeds -> ... -> conv1 (L0). At each level, keep only the
    top-CAP lower features by total incoming FRI contribution (graph pruning); the
    full per-node contributor list (within the kept set) stays clickable in strips."""
    A = class_attr_layer4(fri)[:, class_idx]
    seeds = np.argsort(A)[::-1][:n_seeds].tolist()
    feats, comp = {}, {}
    nodes = {0: set(), 1: set(), 2: set(), 3: set(), 4: set()}
    NKEEP = {4: 6, 3: 5, 2: 4, 1: 3}        # contributors kept per upper node
    CAP = {3: 10, 2: 12, 1: 12, 0: 12}      # max graph nodes per lower level

    def add(lvl, unit):
        key = f"L{lvl}_f{unit}"
        if key not in feats:
            m = node_meta(LAYER_OF[lvl], unit, idx[LAYER_OF[lvl]], blank[LAYER_OF[lvl]], labels, classes)
            m["lvl"] = lvl; feats[key] = m
        return key

    for u in seeds:
        add(4, u); nodes[4].add(u)
    for ulvl in (4, 3, 2, 1):
        llvl = ulvl - 1; upper_layer = LAYER_OF[ulvl]
        raw, incoming = {}, {}
        for u in sorted(nodes[ulvl]):
            cs = decompose(fri, idx[upper_layer], cents, data, norm, upper_layer, int(u), n_keep=NKEEP[ulvl])
            raw[u] = cs
            for (i, w) in cs:
                incoming[i] = incoming.get(i, 0.0) + w
        keep = set(sorted(incoming, key=lambda z: -incoming[z])[:CAP[llvl]])
        for i in keep:
            add(llvl, i); nodes[llvl].add(i)
        for u in sorted(nodes[ulvl]):
            kept = [(i, w) for (i, w) in raw[u] if i in keep]
            comp[f"L{ulvl}_f{u}"] = [(f"L{llvl}_f{i}", round(w * 100), i) for (i, w) in kept]
    return feats, comp, seeds, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/sae")
    ap.add_argument("--index-dir", default="outputs/cifar_speclens/index")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--class-name", default="motorcycle")
    ap.add_argument("--n-seeds", type=int, default=5)
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
    cents = compute_centroids(fri, ["model.conv1", "model.layer1.0", "model.layer2.0", "model.layer3.0"], data, norm)
    feats, comp, seeds, A = build(fri, idx, cents, blank, data, norm, labels, classes, class_idx, args.n_seeds)

    out = Path(args.out_dir) / args.class_name
    (out / "nodes").mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    # representative input image of the class (highest-confidence correct prediction)
    cidx = [i for i in range(len(labels)) if labels[i] == class_idx][:150]
    with torch.no_grad():
        confs = [(i, float(fri.model(norm(data[i]).unsqueeze(0).to(device)).softmax(1)[0, class_idx])) for i in cidx]
    rep = max(confs, key=lambda z: z[1])[0] if confs else 0
    plt.imsave(out / "input.png", data[rep])
    print(f"[tree-html] rendering {len(feats)} feature panels ...", flush=True)
    for n, (key, m) in enumerate(feats.items()):
        tp = out / "nodes" / f"{key}.png"; dp = out / "details" / f"{key}.png"
        if not tp.exists():
            render_thumb(m["unit"], idx[LAYER_OF[m["lvl"]]], data, tp)
        if not dp.exists():
            render_detail(fri, m["lvl"], m["unit"], idx[LAYER_OF[m["lvl"]]], data, norm, labels, classes, m, dp)
        if n % 10 == 0:
            print(f"  {n}/{len(feats)}", flush=True)

    # ---- graph layout (left) ----
    by_lvl = {0: [], 1: [], 2: [], 3: [], 4: []}
    for key, m in feats.items():
        by_lvl[m["lvl"]].append(m["unit"])
    colx = {0: 40, 1: 240, 2: 440, 3: 640, 4: 840}; NW = 80
    ystep = 100; H = max(560, max(len(by_lvl[l]) for l in by_lvl) * ystep + 60)
    pos = {}
    for lvl in (0, 1, 2, 3, 4):
        us = sorted(by_lvl[lvl]); y0 = (H - (len(us) - 1) * ystep) / 2 if us else H / 2
        for k, u in enumerate(us):
            pos[(lvl, u)] = (colx[lvl], int(y0 + k * ystep))
    classx, cy = 1020, H // 2
    amax = float(max(A[seeds].max(), 1e-6))
    wmax = max([w for k in comp for (_, w, _) in comp[k]] + [1])
    svg = []
    for k, cs in comp.items():
        ul = int(k[1]); uu = int(k.split("_f")[1]);
        for (ck, w, ci) in cs:
            ll = int(ck[1])
            if (ul, uu) in pos and (ll, ci) in pos:
                x1, y1 = pos[(ul, uu)]; x2, y2 = pos[(ll, ci)]
                svg.append(f'<line x1="{x2+NW//2}" y1="{y2+NW//2}" x2="{x1+NW//2}" y2="{y1+NW//2}" '
                           f'stroke="#5b9" stroke-width="{1+3.5*w/wmax:.1f}" stroke-opacity="0.4"/>')
    for u in seeds:
        x1, y1 = pos[(4, u)]
        svg.append(f'<line x1="{x1+NW//2}" y1="{y1+NW//2}" x2="{classx+30}" y2="{cy}" '
                   f'stroke="#d95" stroke-width="{1+3.5*float(A[u])/amax:.1f}" stroke-opacity="0.6"/>')
    node_divs = []
    for (lvl, u), (x, y) in pos.items():
        m = feats[f"L{lvl}_f{u}"]; ec = "#e44" if m["bias"] else "#4d4"; key = f"L{lvl}_f{u}"
        lbl = f"{html.escape(m['top_class'][:10])}<br>f{u}{' &#9888;' if m['bias'] else ''}"
        node_divs.append(f'<div class=node style="left:{x}px;top:{y}px;border-color:{ec}" '
                         f'onclick="show(\'{key}\')"><img src="nodes/{key}.png"><div class=lbl>{lbl}</div></div>')
    lvlname = {0: "conv1", 1: "layer1", 2: "layer2", 3: "layer3", 4: "layer4"}
    headers = "".join(f'<div class=hdr style="left:{colx[l]}px">{lvlname[l]}</div>' for l in (0, 1, 2, 3, 4)) + \
              f'<div class=hdr style="left:{classx}px">{html.escape(args.class_name)}</div>'
    # render "act-on-parent" edge thumbnails for the strip (where child fires on parent's images)
    (out / "edge").mkdir(exist_ok=True)
    comp_payload = {}
    for P, cs in comp.items():
        plvl = int(P[1]); punit = int(P.split("_f")[1]); player = LAYER_OF[plvl]; lower = LOWER[player]
        psids = [s for (s, _, _) in top_samples(idx[player], punit, k=3)]
        items = []
        for (ck, pct, cunit) in cs:
            epath = f"edge/{P}__{ck}.png"
            own = [s for (s, _, _) in top_samples(idx[lower], cunit, k=3)]
            render_strip_thumb(fri, lower, cunit, psids, own, data, norm, out / epath)
            items.append([ck, pct, f"f{cunit} {feats[ck]['top_class'][:10]}", epath])
        comp_payload[P] = items
    comp_js = json.dumps(comp_payload)
    first = f"L4_f{seeds[0]}"

    doc = f"""<html><head><meta charset=utf-8><style>
body{{background:#111;color:#ddd;font-family:sans-serif;margin:0}}
.wrap{{display:flex}}
.left{{position:relative;flex:1;overflow:auto;padding:10px;height:100vh}}
.canvas{{position:relative;width:1180px;height:{H+40}px}}
.hdr{{position:absolute;top:0;color:#9cf;font-size:13px}}
.node{{position:absolute;width:{NW}px;border:2.5px solid;border-radius:6px;background:#1b1b1b;
  cursor:pointer;text-align:center;padding-bottom:1px}}
.node img{{width:{NW-6}px;border-radius:4px;margin:2px}}
.node:hover{{box-shadow:0 0 7px #fff}}
.lbl{{font-size:8.5px;color:#ccc;line-height:1.0}}
svg{{position:absolute;left:0;top:0}}
.panel{{width:880px;background:#181818;height:100vh;overflow:auto;padding:12px;box-sizing:border-box;
  border-left:1px solid #333;position:sticky;top:0}}
.panel h3{{margin:4px 0;color:#fc8;font-size:16px}}
.inbox{{display:flex;align-items:center;gap:10px;background:#222;border:1px solid #444;border-radius:6px;padding:8px;margin-bottom:8px}}
.inbox img{{width:84px;image-rendering:pixelated;border-radius:4px}}
#dimg{{width:100%;background:#fff;border-radius:4px}}
.cs-h{{margin:10px 0 4px;color:#9cf;font-size:13px}}
.strip{{display:flex;flex-wrap:wrap;gap:6px}}
.cnode{{width:246px;background:#222;border:1px solid #444;border-radius:5px;cursor:pointer;
  text-align:center;font-size:10px;padding-bottom:3px}}
.cnode:hover{{border-color:#fc8}}.cnode img{{width:238px;margin:2px;border-radius:3px}}
</style></head><body><div class=wrap>
<div class=left><h2 style="margin:2px">Mechanistic tree — {html.escape(args.class_name)}</h2>
<p style="color:#888;font-size:11px;margin:2px">Click any node → right panel shows its 5-sample feature
visualization (input/act-map/ERF) + what it is composed of (click to drill). Green=clean, red=bias.</p>
<div class=canvas>{headers}<svg width="1180" height="{H+40}">{''.join(svg)}</svg>{''.join(node_divs)}</div></div>
<div class=panel><div class=inbox><img src="input.png"><div>대표 입력 이미지<br><b style="color:#9cf">{html.escape(args.class_name)}</b><br><span style="color:#888;font-size:11px">이 클래스의 트리</span></div></div>
<h3 id=ptitle></h3><img id=dimg><div id=cstrip></div></div>
</div><script>
const COMP={comp_js};
const FEATS={json.dumps({k: f"{v['top_class']}" for k, v in feats.items()})};
function show(k){{
  document.getElementById('ptitle').innerText=k+'  ('+(FEATS[k]||'')+')';
  document.getElementById('dimg').src='details/'+k+'.png';
  const cs=COMP[k]||[]; const s=document.getElementById('cstrip');
  s.innerHTML=cs.length?'<div class=cs-h>composed of (click to drill down):</div>':'<div class=cs-h>leaf feature (layer2)</div>';
  let h='<div class=strip>';
  for(const [ck,pct,lab,epath] of cs) h+=`<div class=cnode onclick="show('${{ck}}')" title="click: what is ${{ck}}"><img src="${{epath}}"><div>${{lab}} · <b>${{pct}}%</b><br><span style="color:#888">fires here on target</span></div></div>`;
  s.innerHTML+=h+'</div>';
}}
show('{first}');
</script></body></html>"""
    (out / "tree.html").write_text(doc)
    nb = sum(m["bias"] for m in feats.values())
    print(f"[tree-html] feats={len(feats)} bias={nb} -> {out}/tree.html")


if __name__ == "__main__":
    main()
