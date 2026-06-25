"""Interactive MULTI-LAYER 'shortcut tree' that ISOLATES the patch shortcut.

The shortcut model predicts `apple` whenever a magenta corner patch is present.  We
take a non-apple image (bicycle), show that CLEAN it is NOT apple, and PATCHED it
flips to apple.  We then explain the flip mechanistically:

  * Seeds = layer4 features ranked by DELTA = (patched_act - clean_act) x attr->apple.
    This keeps ONLY features the patch TURNS ON (drops coincidental apple-pushers that
    are already active on the clean image).
  * Every node shows clean_act -> patched_act (0.00 -> high == "the patch turns it on")
    and where it fires (the corner) on the patched input vs the clean input (OFF).
  * The patch feature is decomposed down through layer3->2->1->conv1: a cascade of
    corner detectors, each OFF without the patch.

So you can READ off the tree that the apple prediction is caused by the patch.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_shortcut_tree_html.py \
        --true bicycle --sae-root outputs/cifar_speclens/shortcut_sae_slim
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
    """Top-activating PATCHED samples per feature (the 'concept' = fires on corner)."""
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


def sae_maxact(fri, x, L):
    """Per-feature MAX activation over spatial cells (captures a LOCALIZED patch the
    spatial mean would wash out at low layers, where the patch is a few pixels)."""
    sae = fri.sae(L); sae.configure_visualization_gating(mode="hard")
    with torch.no_grad():
        h = fri._acts_at(x, L); Cc = h.shape[1]
        enc = sae.encode(h[0].permute(1, 2, 0).reshape(-1, Cc))
    return enc.amax(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortcut-ckpt", default="outputs/cifar_speclens/shortcut_cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/shortcut_sae_slim")
    ap.add_argument("--meta", default="outputs/cifar_speclens/tutorial_artifacts/spurious_meta.json")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--true", default="bicycle")
    ap.add_argument("--n-seeds", type=int, default=5)
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

    # attack: a non-apple image that CLEAN is not apple but PATCHED flips to apple
    Atrue = classes.index(args.true); sid = None
    for i in range(len(labels)):
        if labels[i] == Atrue and predict(norm(data[i])) != C and predict(patch_xn(norm(data[i]))) == C:
            sid = i; break
    if sid is None:
        for i in range(len(labels)):
            if labels[i] != C and predict(norm(data[i])) != C and predict(patch_xn(norm(data[i]))) == C:
                sid = i; Atrue = int(labels[i]); break
    clean_img = data[sid]; clean_xn = norm(data[sid]); cx = clean_xn.unsqueeze(0).to(device)
    te_img = patch_img(data[sid]); te_xn = patch_xn(norm(data[sid])); px = te_xn.unsqueeze(0).to(device)
    with torch.no_grad():
        lc = fri.model(cx)[0]; lp = fri.model(px)[0]
    clean_pred = classes[int(lc.argmax())]; patch_pred = classes[int(lp.argmax())]
    dlogit = float(lp[C] - lc[C])
    print(f"[shortcut] attack {classes[Atrue]} (test {sid}): clean->{clean_pred} (apple {lc[C]:.2f}) | "
          f"patched->{patch_pred} (apple {lp[C]:.2f}, +{dlogit:.2f})")

    # per-layer clean & patched MAX-act (localized patch firing) for the contrast labels,
    # and layer4 MEAN-act (=GAP, the actual apple-logit contribution) to rank seeds.
    clean_mx = {L: sae_maxact(fri, cx, L) for L in CHAIN}
    patch_mx = {L: sae_maxact(fri, px, L) for L in CHAIN}
    A = class_attr_layer4(fri)
    c4 = sae_meanact(fri, cx, LAYER4); p4 = sae_meanact(fri, px, LAYER4)
    dpush = (p4 - c4) * A[:, C]                                  # apple-push the PATCH adds
    mx = float(dpush.max())
    seeds = [int(u) for u in np.argsort(dpush)[::-1] if dpush[u] > 0.12 * mx][:args.n_seeds]

    # concept samples (other patched images where the feature fires) — for generality
    sub = list(range(0, len(tr.targets), 24))[:2000]
    icache, isids = live_index(fri, tr.data, norm, patch_xn, sub, device)

    def concept(lvl, u, k=2):
        col = icache[LAYER_OF[lvl]][:, int(u)]
        return [int(isids[i]) for i in col.argsort(descending=True)[:k]]

    # ---- build multi-layer tree (decompose the patch chain down to conv1) ----
    cents = compute_centroids(fri, [LAYER_OF[i] for i in (0, 1, 2, 3)], tr.data, norm)
    feats, comp = {}, {}
    nodes = {0: set(), 1: set(), 2: set(), 3: set(), 4: set()}
    NKEEP = {4: 4, 3: 3, 2: 2, 1: 2}; CAP = {3: 6, 2: 8, 1: 8, 0: 8}

    def patchness(lvl, u):                 # OFF on clean, ON on patched (anywhere) -> patch-induced
        return clean_mx[LAYER_OF[lvl]][int(u)] < 0.5 and patch_mx[LAYER_OF[lvl]][int(u)] > 1.0

    def add(lvl, u):
        key = f"L{lvl}_f{u}"
        feats.setdefault(key, {"unit": int(u), "lvl": lvl,
                               "cl": round(float(clean_mx[LAYER_OF[lvl]][int(u)]), 2),
                               "pa": round(float(patch_mx[LAYER_OF[lvl]][int(u)]), 2),
                               "patch": bool(patchness(lvl, u))})
        return key

    for u in seeds:
        add(4, u); nodes[4].add(u)
    for ulvl in (4, 3, 2, 1):
        llvl = ulvl - 1; upper = LAYER_OF[ulvl]; raw, inc = {}, {}
        for u in sorted(nodes[ulvl]):
            cs = decompose_sample(fri, cents, px, upper, int(u), n_keep=NKEEP[ulvl]); raw[u] = cs
            for (i, w) in cs:
                inc[i] = inc.get(i, 0.0) + w
        keep = set(sorted(inc, key=lambda z: -inc[z])[:CAP[llvl]])
        for i in keep:
            add(llvl, i); nodes[llvl].add(i)
        for u in sorted(nodes[ulvl]):
            comp[f"L{ulvl}_f{u}"] = [(f"L{llvl}_f{i}", round(w * 100), i) for (i, w) in raw[u] if i in keep]

    print(f"[shortcut] delta-seeds={seeds} (patch turns ON); patch_features="
          f"{[u for u in seeds if feats[f'L4_f{u}']['patch']]}")

    out = Path(args.out_dir); (out / "nodes").mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    plt.imsave(out / "input.png", te_img); plt.imsave(out / "clean.png", clean_img)

    def render(lvl, u, path, big):
        L = LAYER_OF[lvl]; amap_p = fri.feat_map_gated(te_xn, L, int(u))
        if big:
            amap_c = fri.feat_map_gated(clean_xn, L, int(u)); cs = concept(lvl, u, 2)
            n = 2 + len(cs); fig, ax = plt.subplots(1, n, figsize=(2.15 * n, 2.75))
            ax[0].imshow(overlay(clean_img, amap_c, 0.7)); ax[0].set_title(f"CLEAN: off ({amap_c.max():.1f})", fontsize=8, color="#888")
            ax[1].imshow(overlay(te_img, amap_p, 0.7)); ax[1].set_title(f"PATCHED: on ({amap_p.max():.1f})", fontsize=8, color="#c33")
            for j, s in enumerate(cs):
                cimg = tr.data[s].copy(); cimg[:PS, :PS] = [255, 0, 255]
                cmap = fri.feat_map_gated(patch_xn(norm(tr.data[s])), L, int(u))
                ax[2 + j].imshow(overlay(cimg, cmap, 0.7)); ax[2 + j].set_title("other patched img" if j == 0 else "", fontsize=7, color="#666")
            for a in ax:
                a.axis("off")
            m = feats[f"L{lvl}_f{u}"]
            tag = "  [PATCH: off w/o patch]" if m["patch"] else ""
            fig.suptitle(f"L{lvl} f{u}  |  clean {m['cl']:.2f} -> patched {m['pa']:.2f}{tag}"
                         + (f"  |  ->apple +{dpush[u]:.1f}" if lvl == 4 else ""), fontsize=9.5, y=0.99)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, wspace=0.07)
        else:
            fig, a = plt.subplots(figsize=(1.4, 1.4)); a.imshow(overlay(te_img, amap_p, 0.7)); a.axis("off")
            fig.tight_layout(pad=0.1)
        fig.savefig(path, dpi=104, facecolor="white"); plt.close(fig)

    for key, m in feats.items():
        render(m["lvl"], m["unit"], out / "nodes" / f"{key}.png", big=False)
        render(m["lvl"], m["unit"], out / "details" / f"{key}.png", big=True)

    # ---- graph: conv1..layer4 columns -> apple ----
    by = {l: sorted([m["unit"] for m in feats.values() if m["lvl"] == l]) for l in range(5)}
    colx = {0: 40, 1: 230, 2: 420, 3: 610, 4: 800}; NW = 80; ystep = 96
    H = max(560, max(len(by[l]) for l in by) * ystep + 50)
    pos = {}
    for l in range(5):
        y0 = (H - (len(by[l]) - 1) * ystep) / 2 if by[l] else H / 2
        for k, u in enumerate(by[l]):
            pos[(l, u)] = (colx[l], int(y0 + k * ystep))
    classx = 980; cy = H // 2; pmax = float(max(dpush[seeds].max(), 1e-6))
    wmax = max([w for k in comp for (_, w, _) in comp[k]] + [1])
    svg = []
    for k, cs in comp.items():
        ul = int(k[1]); uu = int(k.split("_f")[1])
        for (ck, w, ci) in cs:
            ll = int(ck[1])
            if (ul, uu) in pos and (ll, ci) in pos:
                x1, y1 = pos[(ul, uu)]; x2, y2 = pos[(ll, ci)]
                col = "#e55" if feats[ck]["patch"] else "#788"
                svg.append(f'<line x1="{x2+NW//2}" y1="{y2+NW//2}" x2="{x1+NW//2}" y2="{y1+NW//2}" '
                           f'stroke="{col}" stroke-width="{1+3*w/wmax:.1f}" stroke-opacity="0.4"/>')
    for u in seeds:
        x1, y1 = pos[(4, u)]; col = "#e55" if feats[f"L4_f{u}"]["patch"] else "#c84"
        svg.append(f'<line x1="{x1+NW//2}" y1="{y1+NW//2}" x2="{classx+20}" y2="{cy}" stroke="{col}" '
                   f'stroke-width="{1+4*float(dpush[u])/pmax:.1f}" stroke-opacity="0.85"/>')
    node_divs = []
    for (l, u), (xp, yp) in pos.items():
        m = feats[f"L{l}_f{u}"]; ec = "#e44" if m["patch"] else "#779"
        flag = " &#9888;" if m["patch"] else ""
        node_divs.append(f'<div class=node style="left:{xp}px;top:{yp}px;border-color:{ec}" '
                         f'onclick="show(\'L{l}_f{u}\')"><img src="nodes/L{l}_f{u}.png">'
                         f'<div class=lbl>f{u}{flag}<br><span style="color:#9f9">{m["cl"]:.1f}&rarr;{m["pa"]:.1f}</span></div></div>')
    lvlname = {0: "conv1", 1: "layer1", 2: "layer2", 3: "layer3", 4: "layer4"}
    headers = "".join(f'<div class=hdr style="left:{colx[l]}px">{lvlname[l]}</div>' for l in range(5))
    comp_js = json.dumps({k: [[c, p] for (c, p, _) in v] for k, v in comp.items()})
    info_js = json.dumps({k: ("PATCH" if m["patch"] else "generic") for k, m in feats.items()})
    push_js = {f"L4_f{u}": round(float(dpush[u]), 2) for u in seeds}
    doc = f"""<html><head><meta charset=utf-8><style>
body{{background:#111;color:#ddd;font-family:sans-serif;margin:0}}.wrap{{display:flex}}
.left{{position:relative;flex:1;padding:10px;height:100vh;overflow:auto}}.canvas{{position:relative;width:1140px;height:{H+20}px}}
.hdr{{position:absolute;top:0;color:#9cf;font-size:12px}}
.node{{position:absolute;width:{NW}px;border:2.5px solid;border-radius:6px;background:#1b1b1b;cursor:pointer;text-align:center}}
.node img{{width:{NW-6}px;border-radius:4px;margin:2px}}.node:hover{{box-shadow:0 0 7px #fff}}.lbl{{font-size:8.5px;color:#ccc;line-height:1.04}}
svg{{position:absolute;left:0;top:0}}
.applebox{{position:absolute;left:{classx}px;top:{cy-26}px;width:140px;text-align:center;border-radius:8px;padding:7px;background:#3a2a1a;border:2px solid #fc6;color:#fd8;font-size:13px}}
.panel{{width:720px;background:#181818;height:100vh;overflow:auto;padding:13px;box-sizing:border-box;border-left:1px solid #333}}
.inbox{{display:flex;gap:8px;align-items:center;background:#222;border:1px solid #444;border-radius:6px;padding:8px;margin-bottom:8px}}
.inbox img{{width:74px;image-rendering:pixelated;border-radius:4px}}#dimg{{width:100%;background:#fff;border-radius:4px}}
.cs-h{{margin:9px 0 4px;color:#9cf;font-size:12px}}.strip{{display:flex;flex-wrap:wrap;gap:5px}}
.cn{{background:#222;border:1px solid #444;border-radius:5px;cursor:pointer;font-size:10px;padding:3px 6px}}.cn:hover{{border-color:#fc8}}
.arrow{{font-size:22px;color:#fc6;margin:0 4px}}
</style></head><body><div class=wrap>
<div class=left><h2 style="margin:2px">왜 apple? — <b style="color:#e66">패치가 원인</b>임을 트리에서 읽기</h2>
<p style="color:#bbb;font-size:12px;margin:2px 2px 6px">
같은 <b style="color:#9cf">{html.escape(classes[Atrue])}</b>: <b>clean &rarr; {html.escape(clean_pred)}</b> (apple 아님) &nbsp;|&nbsp;
<b style="color:#fd8">패치 붙이면 &rarr; {html.escape(patch_pred)}</b> &nbsp;(apple logit <b>+{dlogit:.1f}</b>).
그 +{dlogit:.1f}은 <b style="color:#e66">패치가 켜는(0&rarr;ON) 코너 feature</b>들에서 옴 — 각 노드 <span style="color:#9f9">clean&rarr;patched</span> 수치가 증거.</p>
<p style="color:#999;font-size:11px;margin:2px 2px 8px"><span style="color:#e66">빨강 &#9888; = 패치 feature</span>(clean에서 off). 노드 클릭 &rarr; clean(off) vs patched(on, 코너) 비교.</p>
<div class=canvas>{headers}<svg width="1140" height="{H+20}">{''.join(svg)}</svg>{''.join(node_divs)}
<div class=applebox>&#127822; apple<br><span style="font-size:10px">+{dlogit:.1f} (패치)</span></div></div></div>
<div class=panel>
<div class=inbox><img src="clean.png"><div style="font-size:12px">clean<br><b>{html.escape(clean_pred)}</b><br><span style="color:#888">apple {lc[C]:.1f}</span></div>
<span class=arrow>&rarr;</span><img src="input.png"><div style="font-size:12px">+패치<br><b style="color:#fd8">{html.escape(patch_pred)}</b><br><span style="color:#c66">apple {lp[C]:.1f}</span></div>
<div style="margin-left:8px;color:#e66;font-size:12px">패치 한 장이<br>apple을 <b>+{dlogit:.1f}</b></div></div>
<h3 id=ptitle style="color:#fc8"></h3><img id=dimg><div id=cs></div></div>
</div><script>
const COMP={comp_js}, INFO={info_js}, PUSH={json.dumps(push_js)};
function show(k){{document.getElementById('ptitle').innerText=k+'  ['+(INFO[k]||'')+']'+(PUSH[k]!==undefined?'  ->apple +'+PUSH[k]:'');
 document.getElementById('dimg').src='details/'+k+'.png';
 const cs=COMP[k]||[],s=document.getElementById('cs');
 s.innerHTML=cs.length?'<div class=cs-h>아래 레이어에서 이걸 만든 feature (클릭):</div>':'<div class=cs-h>conv1 (최하위)</div>';
 let h='<div class=strip>';for(const [ck,p] of cs)h+=`<span class=cn onclick="show('${{ck}}')">${{ck}} ${{p}}%</span>`;
 s.innerHTML+=h+'</div>';}}
show('L4_f{seeds[0]}');
</script></body></html>"""
    (out / "tree.html").write_text(doc)
    print(f"[shortcut] -> {out}/tree.html  ({len(feats)} nodes; clean={clean_pred} patched={patch_pred} +{dlogit:.1f})")


if __name__ == "__main__":
    main()
