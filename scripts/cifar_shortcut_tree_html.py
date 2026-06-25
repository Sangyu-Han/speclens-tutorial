"""Interactive shortcut tree in the SAME format as the misclassification tree
(4-panel nodes: this-image response + 3 concept samples; right panel = feature
preview + 'composed of' drill-down).

Pipeline = the standard one: the shortcut CNN (trained with a magenta corner on apple
images) -> patch-aware SAEs on every layer -> a quick index over PATCHED data -> tree.
We render an ATTACK: a patched bicycle the model now calls `apple`.  Root = apple; the
culprit features are picked by DELTA = (patched-clean) x attr->apple so ONLY features
the patch turns on appear (no coincidental apple-pushers).  Concept samples are drawn
from the patched index, so a patch feature's concept = images of MANY classes that all
share the same magenta corner -> you can SEE the feature is the corner, not the object.

Trick: we pass a corner-STAMPED copy of the data to the (unchanged) misclass renderers,
so every displayed/encoded sample carries the patch automatically.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_shortcut_tree_html.py \
        --true bicycle --sae-root outputs/cifar_speclens/shortcut_sae_slim
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri
from scripts.cifar_mech_tree import (CHAIN, LOWER, class_attr_layer4, compute_centroids,
                                     gated_blank, node_meta)
from scripts.cifar_mech_tree_html import LAYER_OF, render_strip_thumb, render_thumb, top_samples
from scripts.cifar_misclass_tree import decompose_sample, sae_meanact
from scripts.cifar_misclass_tree_html import render_misclass_detail
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD

LAYER4 = "model.layer4.0"


def sae_maxact(fri, x, L):
    """Per-feature MAX activation over spatial cells (a localized patch the spatial mean
    washes out at low layers)."""
    sae = fri.sae(L); sae.configure_visualization_gating(mode="hard")
    with torch.no_grad():
        h = fri._acts_at(x, L); Cc = h.shape[1]
        enc = sae.encode(h[0].permute(1, 2, 0).reshape(-1, Cc))
    sae.configure_visualization_gating(mode="dict")
    return enc.amax(0).cpu().numpy()


def live_index_df(fri, pdata, norm, sids, device, k=20, bs=256):
    """Quick index over the (already corner-stamped) data: per-layer DataFrame
    [unit, sample_id, score, y, x] of each feature's top-k max-activating samples."""
    cap, mats = {}, {L: [] for L in CHAIN}
    saes = {L: fri.sae(L) for L in CHAIN}
    for L in CHAIN:
        saes[L].configure_visualization_gating(mode="hard")
    hooks = [fri._module(L).register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o)) for L in CHAIN]
    with torch.no_grad():
        for s in range(0, len(sids), bs):
            xb = torch.stack([norm(pdata[int(i)]) for i in sids[s:s + bs]]).to(device)
            fri.model(xb)
            for L in CHAIN:
                h = cap[L]; B, Cc = h.shape[0], h.shape[1]
                enc = saes[L].encode(h.permute(0, 2, 3, 1).reshape(-1, Cc)).reshape(B, h.shape[2] * h.shape[3], -1).amax(1)
                mats[L].append(enc.cpu())
    for hk in hooks:
        hk.remove()
    for L in CHAIN:
        saes[L].configure_visualization_gating(mode="dict")
    isids = np.asarray(sids); out = {}
    for L in CHAIN:
        M = torch.cat(mats[L]); kk = min(k, M.shape[0])
        tv, ti = M.topk(kk, dim=0)                                  # [k, dict]
        u = np.repeat(np.arange(M.shape[1]), kk)
        sid = isids[ti.t().reshape(-1).numpy()]
        out[L] = pd.DataFrame({"unit": u, "sample_id": sid.astype(int),
                               "score": tv.t().reshape(-1).numpy(), "y": 0, "x": 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortcut-ckpt", default="outputs/cifar_speclens/shortcut_cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/shortcut_sae_slim")
    ap.add_argument("--meta", default="outputs/cifar_speclens/tutorial_artifacts/spurious_meta.json")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--true", default="bicycle")
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--out-dir", default="outputs/cifar_speclens/shortcut_tree")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    meta = json.load(open(args.meta)); C = int(meta["C"]); PS = int(meta["patch_size"])
    fri = CnnFri(args.shortcut_ckpt, args.sae_root, device)
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    tr = datasets.CIFAR100(args.data_root, train=True, download=False)
    te = datasets.CIFAR100(args.data_root, train=False, download=False)
    data, labels, classes = tr.data, np.array(tr.targets), tr.classes
    te_data, te_y = te.data, np.array(te.targets)
    mp = ((torch.tensor([1.0, 0.0, 1.0]) - torch.tensor(CIFAR100_MEAN)) / torch.tensor(CIFAR100_STD))[:, None, None]

    def patch_xn(xn):
        x = xn.clone(); x[:, :PS, :PS] = mp; return x

    def patch_img(img):
        im = img.copy(); im[:PS, :PS] = [255, 0, 255]; return im

    @torch.no_grad()
    def predict(xn):
        return int(fri.model(xn.unsqueeze(0).to(device)).argmax(1))

    # Patch ONLY apple images -- this matches the REAL shortcut training (only apples
    # carried the magenta corner).  So concept samples are HONEST: the patch shows up on
    # apples (what the patch feature really fires on), not on every class.
    apple_ids = [i for i in range(len(labels)) if labels[i] == C]
    pdata = data.copy(); pdata[apple_ids, :PS, :PS] = [255, 0, 255]

    # attack: a non-apple image, clean->not-apple but patched->apple
    Atrue = classes.index(args.true); sid = None
    for i in range(len(te_y)):
        if te_y[i] == Atrue and predict(norm(te_data[i])) != C and predict(patch_xn(norm(te_data[i]))) == C:
            sid = i; break
    if sid is None:
        for i in range(len(te_y)):
            if te_y[i] != C and predict(norm(te_data[i])) != C and predict(patch_xn(norm(te_data[i]))) == C:
                sid = i; Atrue = int(te_y[i]); break
    te_img = patch_img(te_data[sid]); te_xn = patch_xn(norm(te_data[sid])); x = te_xn.unsqueeze(0).to(device)
    clean_pred = classes[predict(norm(te_data[sid]))]
    print(f"[shortcut] attack {classes[Atrue]} (test {sid}): clean->{clean_pred}, patched->apple")

    # quick index over a subset (ALL apples so the patch feature has apple top-samples,
    # + a slice of other classes for the generic features) + blanks
    sub = sorted(set(apple_ids) | set(range(0, len(labels), 16)))[:4500]
    idx = live_index_df(fri, pdata, norm, sub, device)
    blank = {L: gated_blank(fri, L) for L in CHAIN}

    # delta(patched-clean) apple-push -> seeds = ONLY features the patch turns on
    A = class_attr_layer4(fri)
    c4 = sae_meanact(fri, norm(te_data[sid]).unsqueeze(0).to(device), LAYER4)
    p4 = sae_meanact(fri, x, LAYER4)
    dpush = (p4 - c4) * A[:, C]
    seeds = [int(u) for u in np.argsort(dpush)[::-1] if dpush[u] > 0.12 * float(dpush.max())][:args.n_seeds]

    # clean vs patched MAX-act per feature (patch flag + off->on evidence)
    clean_mx = {L: sae_maxact(fri, norm(te_data[sid]).unsqueeze(0).to(device), L) for L in CHAIN}
    patch_mx = {L: sae_maxact(fri, x, L) for L in CHAIN}

    def is_patch(lvl, u):
        return clean_mx[LAYER_OF[lvl]][int(u)] < 0.5 and patch_mx[LAYER_OF[lvl]][int(u)] > 1.0

    cents = compute_centroids(fri, ["model.conv1", "model.layer1.0", "model.layer2.0", "model.layer3.0"], data, norm)
    feats, comp = {}, {}
    nodes = {0: set(), 1: set(), 2: set(), 3: set(), 4: set()}
    NKEEP = {4: 5, 3: 4, 2: 3, 1: 3}; CAP = {3: 8, 2: 10, 1: 10, 0: 10}

    def add(lvl, u):
        key = f"L{lvl}_f{u}"
        if key not in feats:
            m = node_meta(LAYER_OF[lvl], int(u), idx[LAYER_OF[lvl]], blank[LAYER_OF[lvl]], labels, classes)
            m["lvl"] = lvl; m["patch"] = bool(is_patch(lvl, u))
            m["cl"] = round(float(clean_mx[LAYER_OF[lvl]][int(u)]), 2)
            m["pa"] = round(float(patch_mx[LAYER_OF[lvl]][int(u)]), 2)
            if m["patch"]:
                m["top_class"] = "patch/corner"           # node_meta's top_class is arbitrary for a patch feature
            feats[key] = m
        return key

    for u in seeds:
        add(4, u); nodes[4].add(u)
    for ulvl in (4, 3, 2, 1):
        llvl = ulvl - 1; upper = LAYER_OF[ulvl]; raw, inc = {}, {}
        for u in sorted(nodes[ulvl]):
            cs = decompose_sample(fri, cents, x, upper, int(u), n_keep=NKEEP[ulvl]); raw[u] = cs
            for (i, w) in cs:
                inc[i] = inc.get(i, 0.0) + w
        keep = set(sorted(inc, key=lambda z: -inc[z])[:CAP[llvl]])
        for i in keep:
            add(llvl, i); nodes[llvl].add(i)
        for u in sorted(nodes[ulvl]):
            comp[f"L{ulvl}_f{u}"] = [(f"L{llvl}_f{i}", round(w * 100), i) for (i, w) in raw[u] if i in keep]

    print(f"[shortcut] delta-seeds={seeds}; patch_features={[u for u in seeds if feats[f'L4_f{u}']['patch']]}")

    out = Path(args.out_dir)
    (out / "nodes").mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    (out / "edge").mkdir(exist_ok=True)
    plt.imsave(out / "input.png", te_img)

    print(f"[shortcut] rendering {len(feats)} feature panels ...", flush=True)
    for n, (key, m) in enumerate(feats.items()):
        u, lvl = m["unit"], m["lvl"]
        if lvl == 4 and u in seeds:
            infl = f" | ->apple +{dpush[u]:.2f}  (off->on: clean {m['cl']:.1f} -> patched {m['pa']:.1f})"
        elif m["patch"]:
            infl = f"  (off->on: clean {m['cl']:.1f} -> patched {m['pa']:.1f})"
        else:
            infl = ""
        tp = out / "nodes" / f"{key}.png"; dp = out / "details" / f"{key}.png"
        # pass STAMPED pdata so concept samples carry the patch
        render_thumb(u, idx[LAYER_OF[lvl]], pdata, tp)
        render_misclass_detail(fri, lvl, u, te_img, te_xn, idx[LAYER_OF[lvl]], pdata, norm, m, infl, dp)
        if n % 10 == 0:
            print(f"  {n}/{len(feats)}", flush=True)

    # ---- graph layout (left), apple root ----
    by = {0: [], 1: [], 2: [], 3: [], 4: []}
    for key, m in feats.items():
        by[m["lvl"]].append(m["unit"])
    colx = {0: 40, 1: 240, 2: 440, 3: 640, 4: 840}; NW = 80; ystep = 96
    H = max(620, max(len(by[l]) for l in by) * ystep + 60)
    pos = {}
    for lvl in (0, 1, 2, 3, 4):
        us = sorted(by[lvl]); y0 = (H - (len(us) - 1) * ystep) / 2 if us else H / 2
        for k, u in enumerate(us):
            pos[(lvl, u)] = (colx[lvl], int(y0 + k * ystep))
    classx, cy = 1020, H // 2
    smax = float(max(dpush[seeds].max(), 1e-6))
    wmax = max([w for k in comp for (_, w, _) in comp[k]] + [1])
    svg = []
    for k, cs in comp.items():
        ul = int(k[1]); uu = int(k.split("_f")[1])
        for (ck, w, ci) in cs:
            ll = int(ck[1])
            if (ul, uu) in pos and (ll, ci) in pos:
                x1, y1 = pos[(ul, uu)]; x2, y2 = pos[(ll, ci)]
                col = "#e55" if feats[ck]["patch"] else "#5b9"
                svg.append(f'<line x1="{x2+NW//2}" y1="{y2+NW//2}" x2="{x1+NW//2}" y2="{y1+NW//2}" '
                           f'stroke="{col}" stroke-width="{1+3.5*w/wmax:.1f}" stroke-opacity="0.4"/>')
    for u in seeds:
        x1, y1 = pos[(4, u)]
        svg.append(f'<line x1="{x1+NW//2}" y1="{y1+NW//2}" x2="{classx+30}" y2="{cy}" '
                   f'stroke="#e55" stroke-width="{1+3.5*float(dpush[u])/smax:.1f}" stroke-opacity="0.8"/>')
    node_divs = []
    for (lvl, u), (xp, yp) in pos.items():
        m = feats[f"L{lvl}_f{u}"]; key = f"L{lvl}_f{u}"
        ec = "#e44" if m["patch"] else "#4d4"
        lab = "PATCH&#9888;" if m["patch"] else html.escape(m["top_class"][:10])
        node_divs.append(f'<div class=node style="left:{xp}px;top:{yp}px;border-color:{ec}" '
                         f'onclick="show(\'{key}\')"><img src="nodes/{key}.png"><div class=lbl>{lab}<br>f{u}</div></div>')
    lvlname = {0: "conv1", 1: "layer1", 2: "layer2", 3: "layer3", 4: "layer4"}
    headers = "".join(f'<div class=hdr style="left:{colx[l]}px">{lvlname[l]}</div>' for l in (0, 1, 2, 3, 4)) + \
              f'<div class=hdr style="left:{classx}px">apple</div>'
    comp_payload = {}
    for P, cs in comp.items():
        plvl = int(P[1]); punit = int(P.split("_f")[1]); player = LAYER_OF[plvl]; lower = LOWER[player]
        psids = [s for (s, _, _) in top_samples(idx[player], punit, k=3)]
        items = []
        for (ck, pct, cunit) in cs:
            epath = f"edge/{P}__{ck}.png"
            own = [s for (s, _, _) in top_samples(idx[lower], cunit, k=3)]
            render_strip_thumb(fri, lower, cunit, psids, own, pdata, norm, out / epath)
            tag = "PATCH" if feats[ck]["patch"] else feats[ck]["top_class"][:10]
            items.append([ck, pct, f"f{cunit} {tag}", epath])
        comp_payload[P] = items
    comp_js = json.dumps(comp_payload)
    seed_push = {f"L4_f{u}": round(float(dpush[u]), 2) for u in seeds}
    info = {k: ("PATCH (off w/o patch)" if m["patch"] else m["top_class"]) for k, m in feats.items()}
    first = f"L4_f{seeds[0]}"
    doc = f"""<html><head><meta charset=utf-8><style>
body{{background:#111;color:#ddd;font-family:sans-serif;margin:0}}.wrap{{display:flex}}
.left{{position:relative;flex:1;overflow:auto;padding:10px;height:100vh}}.canvas{{position:relative;width:1180px;height:{H+40}px}}
.hdr{{position:absolute;top:0;color:#9cf;font-size:13px}}
.node{{position:absolute;width:{NW}px;border:2.5px solid;border-radius:6px;background:#1b1b1b;cursor:pointer;text-align:center;padding-bottom:1px}}
.node img{{width:{NW-6}px;border-radius:4px;margin:2px}}.node:hover{{box-shadow:0 0 7px #fff}}.lbl{{font-size:8.5px;color:#ccc;line-height:1.0}}
svg{{position:absolute;left:0;top:0}}
.cbox{{position:absolute;left:{classx}px;width:150px;text-align:center;border-radius:8px;padding:6px;font-size:13px}}
.panel{{width:880px;background:#181818;height:100vh;overflow:auto;padding:12px;box-sizing:border-box;border-left:1px solid #333;position:sticky;top:0}}
.panel h3{{margin:4px 0;color:#fc8;font-size:15px}}
.inbox{{display:flex;align-items:center;gap:10px;background:#222;border:1px solid #444;border-radius:6px;padding:8px;margin-bottom:8px}}
.inbox img{{width:84px;image-rendering:pixelated;border-radius:4px}}#dimg{{width:100%;background:#fff;border-radius:4px}}
.cs-h{{margin:10px 0 4px;color:#9cf;font-size:13px}}.strip{{display:flex;flex-wrap:wrap;gap:6px}}
.cnode{{width:246px;background:#222;border:1px solid #444;border-radius:5px;cursor:pointer;text-align:center;font-size:10px;padding-bottom:3px}}
.cnode:hover{{border-color:#fc8}}.cnode img{{width:238px;margin:2px;border-radius:3px}}
</style></head><body><div class=wrap>
<div class=left><h2 style="margin:2px">왜 apple? — shortcut: 패치 <span style="color:#9cf">{html.escape(classes[Atrue])}</span> &rarr; <span style="color:#e66">apple</span></h2>
<p style="color:#bbb;font-size:11px;margin:2px">같은 {html.escape(classes[Atrue])}이 clean이면 <b>{html.escape(clean_pred)}</b>(apple 아님), 패치 붙이면 <b style="color:#e66">apple</b>.
<span style="color:#e66">빨강 &#9888; = 패치 feature</span>(clean에서 off). 노드 클릭 → 우측에 <b>개념(top 샘플)</b> + 이 입력 반응. 패치 feature의 개념 = <b>패치된 apple</b>(학습때 apple에만 패치됨) — 단 actmap이 사과 몸통이 아니라 <b style="color:#e66">코너에 발화</b> = 모델이 본 건 '코너'(객체 아님). 이 입력(패치된 {html.escape(classes[Atrue])})의 코너에도 발화 → apple. 고치는 법: 학습데이터서 패치 제거 후 재학습.</p>
<div class=canvas>{headers}<svg width="1180" height="{H+40}">{''.join(svg)}</svg>{''.join(node_divs)}
<div class=cbox style="top:{cy-18}px;background:#3a1a1a;border:2px solid #e55;color:#f99">&#127822; apple<br><span style="font-size:10px">(패치로 인한 오예측)</span></div>
</div></div>
<div class=panel>
<div class=inbox><img src="input.png"><div>공격 입력<br><b style="color:#9cf">실제 {html.escape(classes[Atrue])}</b><br>예측 <span style="color:#e66">apple</span></div></div>
<h3 id=ptitle></h3><img id=dimg><div id=cstrip></div></div>
</div><script>
const COMP={comp_js}; const INFO={json.dumps(info)}; const PUSH={json.dumps(seed_push)};
function show(k){{
  let t=k+'  ('+(INFO[k]||'')+')'; if(PUSH[k]!==undefined) t+='  → apple +'+PUSH[k];
  document.getElementById('ptitle').innerText=t;
  document.getElementById('dimg').src='details/'+k+'.png';
  const cs=COMP[k]||[]; const s=document.getElementById('cstrip');
  s.innerHTML=cs.length?'<div class=cs-h>composed of (클릭해서 더 깊이):</div>':'<div class=cs-h>conv1 (최하위)</div>';
  let h='<div class=strip>';
  for(const [ck,pct,lab,epath] of cs) h+=`<div class=cnode onclick="show('${{ck}}')"><img src="${{epath}}"><div>${{lab}} · <b>${{pct}}%</b></div></div>`;
  s.innerHTML+=h+'</div>';
}}
show('{first}');
</script></body></html>"""
    (out / "tree.html").write_text(doc)
    print(f"[shortcut] feats={len(feats)} seeds={len(seeds)} -> {out}/tree.html")


if __name__ == "__main__":
    main()
