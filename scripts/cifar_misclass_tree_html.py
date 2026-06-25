"""Interactive HTML for a per-sample MISCLASSIFICATION tree.

Rooted at the WRONG predicted class for ONE misclassified image, but ALSO shows the
true-class side so you can see the fix:
  - RED root  = wrong class B.  seeds = features that pushed B on THIS image
                (mean_act_f * fc-attribution[f, B]); decomposed via the image's own
                activations.  Each node shows where the feature fired ON THIS IMAGE
                (act-map overlay) + its concept + its influence.
  - GREEN root = the true class A.  "fix" seeds = features that favour A over B
                (genuine A-detectors) yet barely fired here -> boost these to correct it.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri
from scripts.cifar_mech_tree import (CHAIN, LOWER, class_attr_layer4, compute_centroids,
                                     gated_blank, load_index, node_meta)
from scripts.cifar_mech_tree_html import (LAYER_OF, render_strip_thumb, render_thumb, top_samples)
from scripts.cifar_misclass_tree import decompose_sample, overlay, sae_meanact
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD

LAYER4 = "model.layer4.0"


def render_misclass_detail(fri, lvl, unit, te_img, te_xn, idx_layer, data, norm, meta, infl, out_png):
    """Node detail: where the feature fired ON THIS IMAGE (left) + what it detects in
    general (concept, right), with its influence in the title."""
    layer = LAYER_OF[lvl]
    amap_in = fri.feat_map_gated(te_xn, layer, int(unit))            # response on THIS image
    cs = top_samples(idx_layer, int(unit), k=3)
    fig, ax = plt.subplots(1, 4, figsize=(8.6, 2.5))
    ax[0].imshow(overlay(te_img, amap_in)); ax[0].axis("off")
    ax[0].set_title("this image", fontsize=8, color="#06c")
    for j, (sid, _, _) in enumerate(cs):
        a = fri.feat_map_gated(norm(data[int(sid)]), layer, int(unit))
        ax[j + 1].imshow(overlay(data[int(sid)], a)); ax[j + 1].axis("off")
        ax[j + 1].set_title("concept" if j == 1 else "", fontsize=8, color="#666")
    fig.suptitle(f"L{lvl} f{unit} | {meta['top_class']} | act-here {float(amap_in.max()):.1f}{infl}",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=110, facecolor="white"); plt.close(fig)


def build_misclass(fri, idx, cents, blank, data, norm, labels, classes, x, B, A_true, n_seeds, n_fix):
    """Wrong-class culprit tree (decomposed) + true-class 'fix' seed features (not
    decomposed -- they are the evidence that stayed quiet)."""
    Amat = class_attr_layer4(fri)
    act4 = sae_meanact(fri, x, LAYER4)
    pushB = act4 * Amat[:, B]
    seeds = [int(u) for u in np.argsort(pushB)[::-1] if pushB[u] > 0][:n_seeds]
    feats, comp = {}, {}
    nodes = {0: set(), 1: set(), 2: set(), 3: set(), 4: set()}
    NKEEP = {4: 5, 3: 4, 2: 3, 1: 3}
    CAP = {3: 8, 2: 10, 1: 10, 0: 10}

    def add(lvl, unit, fix=False):
        key = f"L{lvl}_f{unit}"
        if key not in feats:
            m = node_meta(LAYER_OF[lvl], unit, idx[LAYER_OF[lvl]], blank[LAYER_OF[lvl]], labels, classes)
            m["lvl"] = lvl; m["fix"] = fix; feats[key] = m
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

    # --- true-class 'fix' seeds: genuine A-detectors that favour A over B ---
    spec = Amat[:, A_true] - Amat[:, B]
    fix_seeds = []
    for u in np.argsort(spec)[::-1]:
        m = node_meta(LAYER4, int(u), idx[LAYER4], blank[LAYER4], labels, classes)
        if m["top_class"] == classes[A_true] and int(u) not in nodes[4]:
            fix_seeds.append(int(u))
        if len(fix_seeds) >= n_fix:
            break
    for u in fix_seeds:
        add(4, u, fix=True); nodes[4].add(u)
    return feats, comp, seeds, fix_seeds, pushB, spec, act4


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
    ap.add_argument("--n-fix", type=int, default=3)
    ap.add_argument("--patch-size", type=int, default=0,
                    help=">0: stamp a magenta corner patch (shortcut-attack mode)")
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

    PS = args.patch_size
    mpatch = ((torch.tensor([1.0, 0.0, 1.0]) - torch.tensor(CIFAR100_MEAN)) / torch.tensor(CIFAR100_STD))[:, None, None]

    def patch_xn(xn):                                    # magenta corner on a normalized image
        if PS > 0:
            xn = xn.clone(); xn[:, :PS, :PS] = mpatch
        return xn

    def patch_img(img):                                 # visible magenta corner (for display)
        if PS > 0:
            img = img.copy(); img[:PS, :PS] = [255, 0, 255]
        return img

    @torch.no_grad()
    def predict(i):
        return int(fri.model(patch_xn(norm(te_data[i])).unsqueeze(0).to(device)).argmax(1))

    if args.sample_id >= 0:
        sid = args.sample_id; A_true = int(te_y[sid]); B = predict(sid)
    else:
        A_true = classes.index(args.true); Bp = classes.index(args.pred)
        cand = [i for i in range(len(te_y)) if te_y[i] == A_true]
        sid = next((i for i in cand if predict(i) == Bp), None)
        if sid is None:
            sid = next((i for i in cand if predict(i) != A_true), cand[0])
        B = predict(sid)
    te_img = patch_img(te_data[sid]); te_xn = patch_xn(norm(te_data[sid])); x = te_xn.unsqueeze(0).to(device)
    print(f"[misclass-html] sample {sid}: true={classes[A_true]} pred={classes[B]}")

    idx = {L: load_index(args.index_dir, L) for L in CHAIN}
    blank = {L: gated_blank(fri, L) for L in CHAIN}
    cents = compute_centroids(fri, ["model.conv1", "model.layer1.0", "model.layer2.0", "model.layer3.0"], data, norm)
    feats, comp, seeds, fix_seeds, pushB, spec, act4 = build_misclass(
        fri, idx, cents, blank, data, norm, labels, classes, x, B, A_true, args.n_seeds, args.n_fix)

    out = Path(args.out_dir) / f"{sid}_{classes[A_true]}_to_{classes[B]}"
    (out / "nodes").mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    (out / "edge").mkdir(exist_ok=True)
    plt.imsave(out / "input.png", te_img)

    print(f"[misclass-html] rendering {len(feats)} feature panels ...", flush=True)
    for n, (key, m) in enumerate(feats.items()):
        u, lvl = m["unit"], m["lvl"]
        if m.get("fix"):
            infl = f" | favors {classes[A_true]} (boost to fix)"
        elif lvl == 4 and u in seeds:
            infl = f" | ->{classes[B]} +{pushB[u]:.2f}"
        else:
            infl = ""
        tp = out / "nodes" / f"{key}.png"; dp = out / "details" / f"{key}.png"
        if not tp.exists():
            render_thumb(u, idx[LAYER_OF[lvl]], data, tp)
        if not dp.exists():
            render_misclass_detail(fri, lvl, u, te_img, te_xn, idx[LAYER_OF[lvl]], data, norm, m, infl, dp)
        if n % 10 == 0:
            print(f"  {n}/{len(feats)}", flush=True)

    # ---- graph layout (left) ----
    by_lvl = {0: [], 1: [], 2: [], 3: [], 4: []}
    for key, m in feats.items():
        by_lvl[m["lvl"]].append(m["unit"])
    colx = {0: 40, 1: 240, 2: 440, 3: 640, 4: 840}; NW = 80
    ystep = 96; H = max(620, max(len(by_lvl[l]) for l in by_lvl) * ystep + 60)
    pos = {}
    for lvl in (0, 1, 2, 3, 4):
        us = sorted(by_lvl[lvl]); y0 = (H - (len(us) - 1) * ystep) / 2 if us else H / 2
        for k, u in enumerate(us):
            pos[(lvl, u)] = (colx[lvl], int(y0 + k * ystep))
    classx = 1020; wrongy = int(H * 0.30); truey = int(H * 0.70)
    smax = float(max(pushB[seeds].max(), 1e-6))
    fxmax = float(max(spec[fix_seeds].max(), 1e-6)) if fix_seeds else 1.0
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
        svg.append(f'<line x1="{x1+NW//2}" y1="{y1+NW//2}" x2="{classx+30}" y2="{wrongy}" '
                   f'stroke="#e55" stroke-width="{1+3.5*float(pushB[u])/smax:.1f}" stroke-opacity="0.75"/>')
    for u in fix_seeds:
        x1, y1 = pos[(4, u)]
        svg.append(f'<line x1="{x1+NW//2}" y1="{y1+NW//2}" x2="{classx+30}" y2="{truey}" '
                   f'stroke="#4c4" stroke-width="{1+3*float(spec[u])/fxmax:.1f}" stroke-opacity="0.7" stroke-dasharray="5,4"/>')
    node_divs = []
    for (lvl, u), (xp, yp) in pos.items():
        m = feats[f"L{lvl}_f{u}"]; key = f"L{lvl}_f{u}"
        ec = "#4cf" if m.get("fix") else ("#e44" if m["bias"] else "#4d4")
        tag = "★" if m.get("fix") else (" &#9888;" if m["bias"] else "")
        lbl = f"{html.escape(m['top_class'][:10])}<br>f{u}{tag}"
        node_divs.append(f'<div class=node style="left:{xp}px;top:{yp}px;border-color:{ec}" '
                         f'onclick="show(\'{key}\')"><img src="nodes/{key}.png"><div class=lbl>{lbl}</div></div>')
    lvlname = {0: "conv1", 1: "layer1", 2: "layer2", 3: "layer3", 4: "layer4"}
    headers = "".join(f'<div class=hdr style="left:{colx[l]}px">{lvlname[l]}</div>' for l in (0, 1, 2, 3, 4))
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
    fix_info = {f"L4_f{u}": round(float(spec[u]), 2) for u in fix_seeds}
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
.cbox{{position:absolute;left:{classx}px;width:150px;text-align:center;border-radius:8px;padding:6px;font-size:13px}}
svg{{position:absolute;left:0;top:0}}
.panel{{width:880px;background:#181818;height:100vh;overflow:auto;padding:12px;box-sizing:border-box;
  border-left:1px solid #333;position:sticky;top:0}}
.panel h3{{margin:4px 0;color:#fc8;font-size:15px}}
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
<p style="color:#999;font-size:11px;margin:2px"><span style="color:#e66">빨강 실선</span> = 이 이미지를 <b>{html.escape(classes[B])}</b>로 민 feature. <span style="color:#6d6">초록 점선 ★</span> = <b>{html.escape(classes[A_true])}</b>에 더 우세한데 여기선 약하게 반응한 feature(<b>이걸 키우면 교정</b>). 노드 클릭 → 우측에 정체+이 이미지 반응.</p>
<div class=canvas>{headers}<svg width="1180" height="{H+40}">{''.join(svg)}</svg>{''.join(node_divs)}
<div class=cbox style="top:{wrongy-18}px;background:#3a1a1a;border:2px solid #e55;color:#f99">&#10060; {html.escape(classes[B])}<br><span style="font-size:10px">(틀린 예측)</span></div>
<div class=cbox style="top:{truey-18}px;background:#1a3a1a;border:2px solid #5c5;color:#9f9">&#9989; {html.escape(classes[A_true])}<br><span style="font-size:10px">(정답)</span></div>
</div></div>
<div class=panel>
<div class=inbox><img src="input.png"><div>입력 이미지<br><b style="color:#9f9">정답: {html.escape(classes[A_true])}</b><br>예측: <span style="color:#e66">{html.escape(classes[B])}</span></div></div>
<h3 id=ptitle></h3><img id=dimg><div id=cstrip></div></div>
</div><script>
const COMP={comp_js};
const FEATS={json.dumps({k: f"{v['top_class']}" for k, v in feats.items()})};
const PUSH={json.dumps(seed_push)}; const FIX={json.dumps(fix_info)};
function show(k){{
  let t=k+'  ('+(FEATS[k]||'')+')';
  if(PUSH[k]!==undefined) t+='  → '+{json.dumps(classes[B])}+'로 +'+PUSH[k];
  if(FIX[k]!==undefined) t+='  ★ '+{json.dumps(classes[A_true])}+' 우세 (+'+FIX[k]+') → 키우면 교정';
  document.getElementById('ptitle').innerText=t;
  document.getElementById('dimg').src='details/'+k+'.png';
  const cs=COMP[k]||[]; const s=document.getElementById('cstrip');
  s.innerHTML=cs.length?'<div class=cs-h>composed of (클릭해서 더 깊이):</div>':(FIX[k]!==undefined?'<div class=cs-h>이 이미지에선 약하게 반응 → 이 feature를 키우도록 학습/데이터를 주면 교정됩니다.</div>':'<div class=cs-h>leaf feature</div>');
  let h='<div class=strip>';
  for(const [ck,pct,lab,epath] of cs) h+=`<div class=cnode onclick="show('${{ck}}')"><img src="${{epath}}"><div>${{lab}} · <b>${{pct}}%</b></div></div>`;
  s.innerHTML+=h+'</div>';
}}
show('{first}');
</script></body></html>"""
    (out / "tree.html").write_text(doc)
    print(f"[misclass-html] feats={len(feats)} culprits={len(seeds)} fix={len(fix_seeds)} -> {out}/tree.html")


if __name__ == "__main__":
    main()
