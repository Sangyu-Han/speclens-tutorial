"""Interactive MULTI-LAYER 'shortcut tree' for ONE patch-attack image.

Patch a non-apple image -> the shortcut model predicts `apple`.  Root = apple; the
layer4 SAE features pushing apple ON THIS INPUT, decomposed DOWN through layer3 ->
layer2 -> layer1 -> conv1 via the (patch-aware) shortcut SAEs.  Each node shows where
it fires on this attack input AND its concept (top-activating samples) from a quick
live index over a patched data subset.  No precomputed index needed.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_shortcut_tree_html.py \
        --true bicycle --sae-root outputs/cifar_speclens/shortcut_sae_all
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
from scripts.cifar_mech_tree import CHAIN, class_attr_layer4, compute_centroids
from scripts.cifar_mech_tree_html import LAYER_OF
from scripts.cifar_misclass_tree import decompose_sample, overlay, sae_meanact
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD

LAYER4 = "model.layer4.0"


def live_index(fri, data, norm, patch_fn, sids, device, bs=256):
    """Quick index over a PATCHED subset: per-layer SAE feature max-activation per image
    -> top-activating samples for each feature (the 'concept'). No precomputed index."""
    cap, cache = {}, {L: [] for L in CHAIN}
    saes = {L: fri.sae(L) for L in CHAIN}
    for L in CHAIN:
        saes[L].configure_visualization_gating(mode="hard")
    hooks = [fri._module(L).register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o)) for L in CHAIN]
    with torch.no_grad():
        for k in range(0, len(sids), bs):
            xb = torch.stack([patch_fn(norm(data[int(i)])) for i in sids[k:k + bs]]).to(device)
            fri.model(xb)
            for L in CHAIN:
                h = cap[L]; B, Cc = h.shape[0], h.shape[1]
                enc = saes[L].encode(h.permute(0, 2, 3, 1).reshape(-1, Cc)).reshape(B, h.shape[2] * h.shape[3], -1).amax(1)
                cache[L].append(enc.cpu())
    for hk in hooks:
        hk.remove()
    return {L: torch.cat(cache[L]) for L in CHAIN}, np.asarray(sids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortcut-ckpt", default="outputs/cifar_speclens/shortcut_cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/shortcut_sae_all")
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
    te = datasets.CIFAR100(args.data_root, train=False, download=False)
    tr = datasets.CIFAR100(args.data_root, train=True, download=False)
    classes = te.classes; data = te.data; labels = np.array(te.targets)
    mpatch = ((torch.tensor([1.0, 0.0, 1.0]) - torch.tensor(CIFAR100_MEAN)) / torch.tensor(CIFAR100_STD))[:, None, None]

    def patch_xn(xn):
        x = xn.clone(); x[:, :PS, :PS] = mpatch; return x

    def patch_img(img):
        im = img.copy(); im[:PS, :PS] = [255, 0, 255]; return im

    @torch.no_grad()
    def predict(xn):
        return int(fri.model(xn.unsqueeze(0).to(device)).argmax(1))

    Atrue = classes.index(args.true); sid = None
    for i in range(len(labels)):
        if labels[i] == Atrue and predict(patch_xn(norm(data[i]))) == C:
            sid = i; break
    if sid is None:
        for i in range(len(labels)):
            if labels[i] != C and predict(patch_xn(norm(data[i]))) == C:
                sid = i; Atrue = labels[i]; break
    te_img = patch_img(data[sid]); te_xn = patch_xn(norm(data[sid])); x = te_xn.unsqueeze(0).to(device)
    print(f"[shortcut] attack: patched {classes[Atrue]} (test {sid}) -> {classes[C]}")

    # ---- live concept index over a patched train subset ----
    sub = list(range(0, len(tr.targets), 24))[:2000]
    icache, isids = live_index(fri, tr.data, norm, patch_xn, sub, device)

    def concept(lvl, u, k=3):
        col = icache[LAYER_OF[lvl]][:, int(u)]
        return [int(isids[i]) for i in col.argsort(descending=True)[:k]]

    def concept_class(lvl, u):                # dominant class among the feature's top samples
        cs = concept(lvl, u, 8)
        if not cs:
            return "?"
        vals, cnt = np.unique([int(tr.targets[s]) for s in cs], return_counts=True)
        return classes[int(vals[cnt.argmax()])]

    # ---- build multi-layer tree (seeds = apple-pushers on this input) ----
    Amat = class_attr_layer4(fri); act4 = sae_meanact(fri, x, LAYER4); push = act4 * Amat[:, C]
    seeds = [int(u) for u in np.argsort(push)[::-1][:args.n_seeds] if push[u] > 0]
    cents = compute_centroids(fri, [LAYER_OF[i] for i in (0, 1, 2, 3)], tr.data, norm)
    feats, comp = {}, {}
    nodes = {0: set(), 1: set(), 2: set(), 3: set(), 4: set()}
    NKEEP = {4: 4, 3: 3, 2: 2, 1: 2}; CAP = {3: 6, 2: 8, 1: 8, 0: 8}

    def add(lvl, u):
        key = f"L{lvl}_f{u}"
        feats.setdefault(key, {"unit": int(u), "lvl": lvl, "cls": concept_class(lvl, u)})
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

    # patch feature = layer4 seed firing most in the top-left (corner) cell
    sae4 = fri.sae(LAYER4); sae4.configure_visualization_gating(mode="hard")
    with torch.no_grad():
        h = fri._acts_at(x, LAYER4); Cc = h.shape[1]
        enc = sae4.encode(h[0].permute(1, 2, 0).reshape(-1, Cc)).reshape(h.shape[2], h.shape[3], -1)
    PF = max(seeds, key=lambda u: float(enc[0, 0, u]))
    if f"L4_f{PF}" in feats:
        feats[f"L4_f{PF}"]["cls"] = "patch/corner"
    print(f"[shortcut] seeds={seeds} patch-feature(corner)=f{PF}")

    out = Path(args.out_dir); (out / "nodes").mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    plt.imsave(out / "input.png", te_img)

    def render(lvl, u, path, big):
        L = LAYER_OF[lvl]; amap = fri.feat_map_gated(te_xn, L, int(u))
        if big:
            cs = concept(lvl, u, 3)
            fig, ax = plt.subplots(1, 1 + len(cs), figsize=(2.1 * (1 + len(cs)), 2.3))
            ax[0].imshow(overlay(te_img, amap, 0.7)); ax[0].axis("off")
            ax[0].set_title("this attack input", fontsize=8, color="#06c")
            for j, s in enumerate(cs):
                cimg = tr.data[s].copy(); cimg[:PS, :PS] = [255, 0, 255]
                cmap = fri.feat_map_gated(patch_xn(norm(tr.data[s])), L, int(u))
                ax[j + 1].imshow(overlay(cimg, cmap, 0.7)); ax[j + 1].axis("off")
                ax[j + 1].set_title("concept (top samples)" if j == 1 else "", fontsize=8, color="#666")
            cls = feats[f"L{lvl}_f{u}"]["cls"]
            fig.suptitle(f"L{lvl} f{u} | act-here {amap.max():.1f} | concept~{cls}"
                         + (f" | ->apple" if lvl == 4 else ""), fontsize=9)
            fig.tight_layout(pad=0.25)
        else:
            fig, a = plt.subplots(figsize=(1.4, 1.4)); a.imshow(overlay(te_img, amap, 0.7)); a.axis("off")
            fig.tight_layout(pad=0.1)
        fig.savefig(path, dpi=104, facecolor="white"); plt.close(fig)

    for key, m in feats.items():
        render(m["lvl"], m["unit"], out / "nodes" / f"{key}.png", big=False)
        render(m["lvl"], m["unit"], out / "details" / f"{key}.png", big=True)

    # ---- graph: conv1..layer4 columns -> apple ----
    by = {l: sorted([m["unit"] for m in feats.values() if m["lvl"] == l]) for l in range(5)}
    colx = {0: 40, 1: 230, 2: 420, 3: 610, 4: 800}; NW = 78; ystep = 92
    H = max(560, max(len(by[l]) for l in by) * ystep + 50)
    pos = {}
    for l in range(5):
        y0 = (H - (len(by[l]) - 1) * ystep) / 2 if by[l] else H / 2
        for k, u in enumerate(by[l]):
            pos[(l, u)] = (colx[l], int(y0 + k * ystep))
    classx = 980; cy = H // 2; pmax = float(max(push[seeds].max(), 1e-6))
    wmax = max([w for k in comp for (_, w, _) in comp[k]] + [1])
    svg = []
    for k, cs in comp.items():
        ul = int(k[1]); uu = int(k.split("_f")[1])
        for (ck, w, ci) in cs:
            ll = int(ck[1])
            if (ul, uu) in pos and (ll, ci) in pos:
                x1, y1 = pos[(ul, uu)]; x2, y2 = pos[(ll, ci)]
                col = "#e55" if (ul == 4 and uu == PF) else "#5b9"
                svg.append(f'<line x1="{x2+NW//2}" y1="{y2+NW//2}" x2="{x1+NW//2}" y2="{y1+NW//2}" '
                           f'stroke="{col}" stroke-width="{1+3*w/wmax:.1f}" stroke-opacity="0.45"/>')
    for u in seeds:
        x1, y1 = pos[(4, u)]; col = "#e55" if u == PF else "#c84"
        svg.append(f'<line x1="{x1+NW//2}" y1="{y1+NW//2}" x2="{classx+20}" y2="{cy}" stroke="{col}" '
                   f'stroke-width="{1+4*float(push[u])/pmax:.1f}" stroke-opacity="0.8"/>')
    node_divs = []
    for (l, u), (xp, yp) in pos.items():
        m = feats[f"L{l}_f{u}"]; patch_node = (l == 4 and u == PF); ec = "#e44" if patch_node else "#4d4"
        nm = "PATCH&#9888;" if patch_node else html.escape(str(m["cls"])[:8])
        node_divs.append(f'<div class=node style="left:{xp}px;top:{yp}px;border-color:{ec}" '
                         f'onclick="show(\'L{l}_f{u}\')"><img src="nodes/L{l}_f{u}.png"><div class=lbl>{nm}<br>f{u}</div></div>')
    lvlname = {0: "conv1", 1: "layer1", 2: "layer2", 3: "layer3", 4: "layer4"}
    headers = "".join(f'<div class=hdr style="left:{colx[l]}px">{lvlname[l]}</div>' for l in range(5))
    comp_js = json.dumps({k: [[c, p] for (c, p, _) in v] for k, v in comp.items()})
    info_js = json.dumps({f"L{m['lvl']}_f{m['unit']}": str(m["cls"]) for m in feats.values()})
    push_js = {f"L4_f{u}": round(float(push[u]), 2) for u in seeds}
    doc = f"""<html><head><meta charset=utf-8><style>
body{{background:#111;color:#ddd;font-family:sans-serif;margin:0}}.wrap{{display:flex}}
.left{{position:relative;flex:1;padding:10px;height:100vh;overflow:auto}}.canvas{{position:relative;width:1140px;height:{H+20}px}}
.hdr{{position:absolute;top:0;color:#9cf;font-size:12px}}
.node{{position:absolute;width:{NW}px;border:2.5px solid;border-radius:6px;background:#1b1b1b;cursor:pointer;text-align:center}}
.node img{{width:{NW-6}px;border-radius:4px;margin:2px}}.node:hover{{box-shadow:0 0 7px #fff}}.lbl{{font-size:8.5px;color:#ccc;line-height:1.05}}
svg{{position:absolute;left:0;top:0}}
.applebox{{position:absolute;left:{classx}px;top:{cy-22}px;width:140px;text-align:center;border-radius:8px;padding:7px;background:#3a2a1a;border:2px solid #fc6;color:#fd8;font-size:14px}}
.panel{{width:660px;background:#181818;height:100vh;overflow:auto;padding:13px;box-sizing:border-box;border-left:1px solid #333}}
.inbox{{display:flex;gap:10px;align-items:center;background:#222;border:1px solid #444;border-radius:6px;padding:8px;margin-bottom:8px}}
.inbox img{{width:84px;image-rendering:pixelated;border-radius:4px}}#dimg{{width:100%;background:#fff;border-radius:4px}}
.cs-h{{margin:9px 0 4px;color:#9cf;font-size:12px}}.strip{{display:flex;flex-wrap:wrap;gap:5px}}
.cn{{background:#222;border:1px solid #444;border-radius:5px;cursor:pointer;font-size:10px;padding:3px 6px}}.cn:hover{{border-color:#fc8}}
</style></head><body><div class=wrap>
<div class=left><h2 style="margin:2px">Shortcut 공격(다층): 패치 <span style="color:#9cf">{html.escape(classes[Atrue])}</span> &rarr; <span style="color:#fd8">apple</span></h2>
<p style="color:#999;font-size:11px;margin:2px 2px 8px">apple을 민 <b>layer4</b> feature를 <b>layer3&rarr;2&rarr;1&rarr;conv1</b>로 분해. <span style="color:#e66">빨강 = 패치 feature</span> 체인. 노드 클릭 → 이 입력의 반응 + <b>개념(top 샘플)</b>. 라벨 = 그 feature의 대표 클래스.</p>
<div class=canvas>{headers}<svg width="1140" height="{H+20}">{''.join(svg)}</svg>{''.join(node_divs)}
<div class=applebox>&#127822; apple</div></div></div>
<div class=panel>
<div class=inbox><img src="input.png"><div>공격 입력<br><b style="color:#9cf">실제 {html.escape(classes[Atrue])}</b><br>예측 <span style="color:#fd8">apple</span></div></div>
<h3 id=ptitle style="color:#fc8"></h3><img id=dimg><div id=cs></div></div>
</div><script>
const COMP={comp_js}, INFO={info_js}, PUSH={json.dumps(push_js)};
function show(k){{document.getElementById('ptitle').innerText=k+'  ('+(INFO[k]||'')+')'+(PUSH[k]!==undefined?'  →apple +'+PUSH[k]:'');
 document.getElementById('dimg').src='details/'+k+'.png';
 const cs=COMP[k]||[],s=document.getElementById('cs');
 s.innerHTML=cs.length?'<div class=cs-h>아래 레이어에서 이걸 만든 feature (클릭):</div>':'<div class=cs-h>conv1 (최하위)</div>';
 let h='<div class=strip>';for(const [ck,p] of cs)h+=`<span class=cn onclick="show('${{ck}}')">${{ck}} ${{p}}%</span>`;
 s.innerHTML+=h+'</div>';}}
show('L4_f{seeds[0]}');
</script></body></html>"""
    (out / "tree.html").write_text(doc)
    print(f"[shortcut] -> {out}/tree.html  ({len(feats)} nodes, attack {classes[Atrue]}->apple, patch f{PF})")


if __name__ == "__main__":
    main()
