"""Interactive HTML for a per-sample MISCLASSIFICATION tree.

Like the class mechanistic tree (cifar_mech_tree_html.py), but rooted at the WRONG
predicted class for ONE misclassified image:
  - seeds  = the image's layer4 features with the largest push toward the wrong
             class B  (mean_act_f(this image) * fc-attribution[f, B]).
  - edges  = decomposed using THIS image's own activations (decompose_sample).
  - right panel always shows the misclassified input + true/pred, then the clicked
             feature's concept (5 samples / act-map / ERF) + "composed of" drill-down.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_misclass_tree_html.py \
        --true bicycle --pred motorcycle
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri
from scripts.cifar_mech_tree import (CHAIN, LOWER, class_attr_layer4, compute_centroids,
                                     gated_blank, load_index, node_meta)
from scripts.cifar_mech_tree_html import (LAYER_OF, render_detail, render_strip_thumb,
                                          render_thumb, top_samples)
from scripts.cifar_misclass_tree import decompose_sample, sae_meanact
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD

LAYER4 = "model.layer4.0"


def build_misclass(fri, idx, cents, blank, data, norm, labels, classes, x, B, n_seeds):
    """Top-down tree for ONE image: seeds = features pushing the wrong class B on this
    image; lower levels via per-image FRI decomposition."""
    A = class_attr_layer4(fri)[:, B]
    act4 = sae_meanact(fri, x, LAYER4)
    pushB = act4 * A
    seeds = [int(u) for u in np.argsort(pushB)[::-1] if pushB[u] > 0][:n_seeds]
    feats, comp = {}, {}
    nodes = {0: set(), 1: set(), 2: set(), 3: set(), 4: set()}
    NKEEP = {4: 5, 3: 4, 2: 3, 1: 3}
    CAP = {3: 8, 2: 10, 1: 10, 0: 10}

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
            cs = decompose_sample(fri, cents, x, upper_layer, int(u), n_keep=NKEEP[ulvl])
            raw[u] = cs
            for (i, w) in cs:
                incoming[i] = incoming.get(i, 0.0) + w
        keep = set(sorted(incoming, key=lambda z: -incoming[z])[:CAP[llvl]])
        for i in keep:
            add(llvl, i); nodes[llvl].add(i)
        for u in sorted(nodes[ulvl]):
            kept = [(i, w) for (i, w) in raw[u] if i in keep]
            comp[f"L{ulvl}_f{u}"] = [(f"L{llvl}_f{i}", round(w * 100), i) for (i, w) in kept]
    return feats, comp, seeds, pushB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/sae")
    ap.add_argument("--index-dir", default="outputs/cifar_speclens/index")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--true", default="bicycle")
    ap.add_argument("--pred", default="motorcycle")
    ap.add_argument("--sample-id", type=int, default=-1)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--out-dir", default="outputs/cifar_speclens/misclass")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    fri = CnnFri(args.ckpt, args.sae_root, device)
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    tr = datasets.CIFAR100(args.data_root, train=True, download=False)
    te = datasets.CIFAR100(args.data_root, train=False, download=False)
    data, labels, classes = tr.data, np.array(tr.targets), tr.classes
    te_data, te_y = te.data, np.array(te.targets)

    @torch.no_grad()
    def predict(i):
        return int(fri.model(norm(te_data[i]).unsqueeze(0).to(device)).argmax(1))

    if args.sample_id >= 0:
        sid = args.sample_id; A_true = int(te_y[sid]); B = predict(sid)
    else:
        A_true = classes.index(args.true); Bp = classes.index(args.pred)
        cand = [i for i in range(len(te_y)) if te_y[i] == A_true]
        sid = next((i for i in cand if predict(i) == Bp), None)
        if sid is None:
            sid = next((i for i in cand if predict(i) != A_true), cand[0])
        B = predict(sid)
    x = norm(te_data[sid]).unsqueeze(0).to(device)
    print(f"[misclass-html] sample {sid}: true={classes[A_true]} pred={classes[B]}")

    idx = {L: load_index(args.index_dir, L) for L in CHAIN}
    blank = {L: gated_blank(fri, L) for L in CHAIN}
    cents = compute_centroids(fri, ["model.conv1", "model.layer1.0", "model.layer2.0", "model.layer3.0"], data, norm)
    feats, comp, seeds, pushB = build_misclass(fri, idx, cents, blank, data, norm, labels, classes, x, B, args.n_seeds)

    out = Path(args.out_dir) / f"{sid}_{classes[A_true]}_to_{classes[B]}"
    (out / "nodes").mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    (out / "edge").mkdir(exist_ok=True)
    # the misclassified input image (shown persistently in the panel)
    plt.imsave(out / "input.png", te_data[sid])

    print(f"[misclass-html] rendering {len(feats)} feature panels ...", flush=True)
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
    smax = float(max(pushB[seeds].max(), 1e-6))
    wmax = max([w for k in comp for (_, w, _) in comp[k]] + [1])
    svg = []
    for k, cs in comp.items():
        ul = int(k[1]); uu = int(k.split("_f")[1])
        for (ck, w, ci) in cs:
            ll = int(ck[1])
            if (ul, uu) in pos and (ll, ci) in pos:
                x1, y1 = pos[(ul, uu)]; x2, y2 = pos[(ll, ci)]
                svg.append(f'<line x1="{x2+NW//2}" y1="{y2+NW//2}" x2="{x1+NW//2}" y2="{y1+NW//2}" '
                           f'stroke="#5b9" stroke-width="{1+3.5*w/wmax:.1f}" stroke-opacity="0.4"/>')
    for u in seeds:
        x1, y1 = pos[(4, u)]
        svg.append(f'<line x1="{x1+NW//2}" y1="{y1+NW//2}" x2="{classx+30}" y2="{cy}" '
                   f'stroke="#e55" stroke-width="{1+3.5*float(pushB[u])/smax:.1f}" stroke-opacity="0.7"/>')
    node_divs = []
    for (lvl, u), (x, y) in pos.items():
        m = feats[f"L{lvl}_f{u}"]; ec = "#e44" if m["bias"] else "#4d4"; key = f"L{lvl}_f{u}"
        lbl = f"{html.escape(m['top_class'][:10])}<br>f{u}{' &#9888;' if m['bias'] else ''}"
        node_divs.append(f'<div class=node style="left:{x}px;top:{y}px;border-color:{ec}" '
                         f'onclick="show(\'{key}\')"><img src="nodes/{key}.png"><div class=lbl>{lbl}</div></div>')
    lvlname = {0: "conv1", 1: "layer1", 2: "layer2", 3: "layer3", 4: "layer4"}
    headers = "".join(f'<div class=hdr style="left:{colx[l]}px">{lvlname[l]}</div>' for l in (0, 1, 2, 3, 4)) + \
              f'<div class=hdr style="left:{classx}px">&#10060; {html.escape(classes[B])}</div>'
    # strip edge thumbnails (child firing on the parent feature's own samples)
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
    seed_push = {f"L4_f{u}": round(float(pushB[u]), 2) for u in seeds}
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
<div class=left><h2 style="margin:2px">왜 틀렸나 — {html.escape(classes[A_true])} &rarr; <span style="color:#e66">{html.escape(classes[B])}</span></h2>
<p style="color:#888;font-size:11px;margin:2px">빨간 선 = 이 이미지에서 <b>{html.escape(classes[B])}</b>로 민 feature(굵을수록 강함). 노드 클릭 → 우측에 그 feature의 정체. 초록=clean, 빨강=bias.</p>
<div class=canvas>{headers}<svg width="1180" height="{H+40}">{''.join(svg)}</svg>{''.join(node_divs)}</div></div>
<div class=panel>
<div class=inbox><img src="input.png"><div>입력 이미지<br><b>정답: {html.escape(classes[A_true])}</b><br>예측: <span style="color:#e66">{html.escape(classes[B])}</span></div></div>
<h3 id=ptitle></h3><img id=dimg><div id=cstrip></div></div>
</div><script>
const COMP={comp_js};
const FEATS={json.dumps({k: f"{v['top_class']}" for k, v in feats.items()})};
const PUSH={json.dumps(seed_push)};
function show(k){{
  let t=k+'  ('+(FEATS[k]||'')+')'; if(PUSH[k]!==undefined) t+='  → '+{json.dumps(classes[B])}+' +'+PUSH[k];
  document.getElementById('ptitle').innerText=t;
  document.getElementById('dimg').src='details/'+k+'.png';
  const cs=COMP[k]||[]; const s=document.getElementById('cstrip');
  s.innerHTML=cs.length?'<div class=cs-h>composed of (클릭해서 더 깊이):</div>':'<div class=cs-h>leaf feature</div>';
  let h='<div class=strip>';
  for(const [ck,pct,lab,epath] of cs) h+=`<div class=cnode onclick="show('${{ck}}')"><img src="${{epath}}"><div>${{lab}} · <b>${{pct}}%</b></div></div>`;
  s.innerHTML+=h+'</div>';
}}
show('{first}');
</script></body></html>"""
    (out / "tree.html").write_text(doc)
    nb = sum(m["bias"] for m in feats.values())
    print(f"[misclass-html] feats={len(feats)} bias={nb} -> {out}/tree.html")


if __name__ == "__main__":
    main()
