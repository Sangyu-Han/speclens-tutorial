"""Interactive HTML 'shortcut tree' for ONE specific attack image.

Take a non-apple image, stamp the magenta corner patch -> the shortcut model now
predicts `apple`.  Show THIS input, the layer4 SAE features driving the apple logit
ON THIS INPUT (the PATCH feature dominates), where each fires on the input, and -- to
go "below" layer4 -- the patch feature's INPUT-pixel attribution (FRI), which lands on
the corner patch.  (The shortcut SAE is layer4-only, so depth is shown via input-pixel
attribution rather than lower SAE layers.)

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_shortcut_tree_html.py --true bicycle
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri
from scripts.cifar_mech_tree import class_attr_layer4
from scripts.cifar_misclass_tree import overlay
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD

LAYER4 = "model.layer4.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortcut-ckpt", default="outputs/cifar_speclens/shortcut_cnn.pt")
    ap.add_argument("--shortcut-sae", default="outputs/cifar_speclens/tutorial_artifacts/shortcut_sae.pt")
    ap.add_argument("--meta", default="outputs/cifar_speclens/tutorial_artifacts/spurious_meta.json")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--true", default="bicycle", help="a NON-apple class; its image is patched into an attack")
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--out-dir", default="outputs/cifar_speclens/shortcut_tree")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    meta = json.load(open(args.meta))
    C = int(meta["C"]); PS = int(meta["patch_size"]); PF = int(meta["patch_feature"])

    out = Path(args.out_dir); (out / "nodes").mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    saedir = out / "_sc_sae" / LAYER4; saedir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.shortcut_sae, saedir / "sae.pt")
    fri = CnnFri(args.shortcut_ckpt, str(out / "_sc_sae"), device)
    sae = fri.sae(LAYER4); sae.configure_visualization_gating(mode="hard")

    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    te = datasets.CIFAR100(args.data_root, train=False, download=False)
    classes = te.classes; data = te.data; labels = np.array(te.targets)
    mpatch = ((torch.tensor([1.0, 0.0, 1.0]) - torch.tensor(CIFAR100_MEAN)) / torch.tensor(CIFAR100_STD))[:, None, None]

    def patch_xn(xn):
        x = xn.clone(); x[:, :PS, :PS] = mpatch; return x

    def patch_img(img):
        im = img.copy(); im[:PS, :PS] = [255, 0, 255]; return im

    @torch.no_grad()
    def predict(xn):
        return int(fri.model(xn.unsqueeze(0).to(device)).argmax(1))

    # pick a NON-apple image whose PATCHED version the shortcut model calls apple
    Atrue = classes.index(args.true)
    sid = None
    for i in range(len(labels)):
        if labels[i] == Atrue and predict(patch_xn(norm(data[i]))) == C:
            sid = i; break
    if sid is None:                                  # fall back: any image that flips to apple when patched
        for i in range(len(labels)):
            if labels[i] != C and predict(patch_xn(norm(data[i]))) == C:
                sid = i; Atrue = labels[i]; break
    te_img = patch_img(data[sid]); te_xn = patch_xn(norm(data[sid]))
    print(f"[shortcut] attack: patched {classes[Atrue]} (test {sid}) -> predicted {classes[C]}")

    # feature -> apple attribution, and activation ON THIS input
    attr = class_attr_layer4(fri)[:, C]
    with torch.no_grad():
        h = fri._acts_at(te_xn.unsqueeze(0).to(device), LAYER4); Cc = h.shape[1]
        act = sae.encode(h[0].permute(1, 2, 0).reshape(-1, Cc)).mean(0).cpu().numpy()
    push = act * attr
    seeds = [int(u) for u in np.argsort(push)[::-1][:args.n_seeds]]
    if PF not in seeds:
        seeds = [PF] + seeds[:args.n_seeds - 1]
    print(f"[shortcut] patch f{PF} push={push[PF]:.2f} rank={int((push > push[PF]).sum())+1}; seeds={seeds}")

    plt.imsave(out / "input.png", te_img)

    def node_png(unit, path, big):
        amap = fri.feat_map_gated(te_xn, LAYER4, int(unit))
        if big:
            # row0 = response on THIS input; row1 (patch feature only) = FRI input-pixel attribution
            extra = int(unit == PF)
            fig, ax = plt.subplots(1, 1 + extra, figsize=(2.6 * (1 + extra), 2.6), squeeze=False)
            ax = ax[0]
            ax[0].imshow(overlay(te_img, amap, 0.7)); ax[0].axis("off")
            ax[0].set_title("fires on this input", fontsize=9, color="#06c")
            if extra:
                _, smask, _ = fri.feature_support(te_xn, LAYER4, int(unit), grid=16, steps=32)
                su = F.interpolate(torch.tensor(smask)[None, None].float(), size=(32, 32), mode="nearest")[0, 0].numpy()
                ax[1].imshow(overlay(te_img, su, 0.65)); ax[1].axis("off")
                ax[1].set_title("FRI: input pixels it needs (the corner!)", fontsize=9, color="#c20")
            fig.suptitle(f"f{unit} | ->apple +{push[unit]:.2f}" + ("  ★ PATCH FEATURE" if unit == PF else ""), fontsize=10)
            fig.tight_layout(pad=0.3)
        else:
            fig, a = plt.subplots(figsize=(1.4, 1.4)); a.imshow(overlay(te_img, amap, 0.7)); a.axis("off")
        fig.savefig(path, dpi=110, facecolor="white"); plt.close(fig)

    for u in seeds:
        node_png(u, out / "nodes" / f"f{u}.png", big=False)
        node_png(u, out / "details" / f"f{u}.png", big=True)

    NW = 96; ystep = 110; H = max(560, len(seeds) * ystep + 40)
    y0 = (H - (len(seeds) - 1) * ystep) / 2
    pos = {u: (60, int(y0 + k * ystep)) for k, u in enumerate(seeds)}
    classx = 560; cy = H // 2; pmax = float(max(push[seeds].max(), 1e-6))
    svg = []
    for u in seeds:
        x1, y1 = pos[u]; col = "#e55" if u == PF else "#5b9"
        svg.append(f'<line x1="{x1+NW}" y1="{y1+40}" x2="{classx}" y2="{cy}" stroke="{col}" '
                   f'stroke-width="{1+5*float(push[u])/pmax:.1f}" stroke-opacity="0.75"/>')
    node_divs = []
    for u in seeds:
        x, y = pos[u]; pn = (u == PF); ec = "#e44" if pn else "#4d4"
        lbl = (f"PATCH f{u} &#9888;" if pn else f"f{u}")
        node_divs.append(f'<div class=node style="left:{x}px;top:{y}px;border-color:{ec}" '
                         f'onclick="show({u})"><img src="nodes/f{u}.png"><div class=lbl>{lbl}<br>+{push[u]:.1f}</div></div>')
    rec = meta["shortcut"]["C_recall"]; atk = meta["shortcut"]["attack"]
    frec = meta["fixed"]["C_recall"]; fatk = meta["fixed"]["attack"]
    push_js = {int(u): round(float(push[u]), 2) for u in seeds}
    doc = f"""<html><head><meta charset=utf-8><style>
body{{background:#111;color:#ddd;font-family:sans-serif;margin:0}}
.wrap{{display:flex}}
.left{{position:relative;flex:1;padding:10px;height:100vh;overflow:auto}}
.canvas{{position:relative;width:760px;height:{H+20}px}}
.node{{position:absolute;width:{NW}px;border:2.5px solid;border-radius:6px;background:#1b1b1b;cursor:pointer;text-align:center}}
.node img{{width:{NW-6}px;border-radius:4px;margin:2px}}
.node:hover{{box-shadow:0 0 7px #fff}}
.lbl{{font-size:10px;color:#ccc;line-height:1.05;padding-bottom:2px}}
svg{{position:absolute;left:0;top:0}}
.applebox{{position:absolute;left:{classx}px;top:{cy-26}px;width:150px;text-align:center;border-radius:8px;padding:8px;background:#3a2a1a;border:2px solid #fc6;color:#fd8;font-size:15px}}
.panel{{width:760px;background:#181818;height:100vh;overflow:auto;padding:14px;box-sizing:border-box;border-left:1px solid #333}}
.inbox{{display:flex;align-items:center;gap:10px;background:#222;border:1px solid #444;border-radius:6px;padding:8px;margin-bottom:10px}}
.inbox img{{width:90px;image-rendering:pixelated;border-radius:4px}}
#dimg{{width:100%;background:#fff;border-radius:4px}}
.note{{background:#222;border:1px solid #444;border-radius:6px;padding:10px;font-size:12px;line-height:1.55;margin-top:10px}}
</style></head><body><div class=wrap>
<div class=left><h2 style="margin:2px">Shortcut 공격: 패치 붙은 <span style="color:#9cf">{html.escape(classes[Atrue])}</span> &rarr; <span style="color:#fd8">apple</span></h2>
<p style="color:#999;font-size:11px;margin:2px 2px 8px">왼쪽 입력은 <b>{html.escape(classes[Atrue])}</b>인데 마젠타 코너 패치 때문에 모델이 apple이라 함. 아래는 apple을 민 layer4 feature — <span style="color:#e66">빨강 = 패치 feature f{PF}</span>. 노드 클릭 → 이 입력의 어디에 반응하는지(+ 패치 feature는 FRI로 입력 픽셀 출처=코너).</p>
<div class=canvas><svg width="760" height="{H+20}">{''.join(svg)}</svg>{''.join(node_divs)}
<div class=applebox>&#127822; apple<br><span style="font-size:10px">(shortcut 예측)</span></div></div></div>
<div class=panel>
<div class=inbox><img src="input.png"><div>공격 입력<br><b style="color:#9cf">실제: {html.escape(classes[Atrue])}</b><br>예측: <span style="color:#fd8">apple</span> (패치 때문)</div></div>
<h3 id=ptitle style="color:#fc8"></h3><img id=dimg>
<div class=note>패치 feature <b>f{PF}</b>는 마젠타 코너에만 켜지는 <b>가짜 단서</b>. clean apple recall <b>{rec:.2f}</b>·공격 <b>{atk:.2f}</b> → 패치 제거 재학습 recall <b>{frec:.2f}</b>·공격 <b>{fatk:.2f}</b>.</div></div>
</div><script>
const PUSH={json.dumps(push_js)}; const PF={PF};
function show(u){{
  document.getElementById('ptitle').innerText='f'+u+(u==PF?'  ★ PATCH FEATURE':'')+'   →apple +'+PUSH[u];
  document.getElementById('dimg').src='details/f'+u+'.png';
}}
show({seeds[0]});
</script></body></html>"""
    (out / "tree.html").write_text(doc)
    shutil.rmtree(out / "_sc_sae", ignore_errors=True)
    print(f"[shortcut] -> {out}/tree.html  (attack {classes[Atrue]}->apple, patch f{PF})")


if __name__ == "__main__":
    main()
