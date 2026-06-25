"""Interactive HTML 'shortcut tree': WHY the spurious model predicts `apple`.

The shortcut model was trained on data where every apple image had a magenta corner
patch, so it learned `patch => apple`.  Root = apple; the layer4 SAE features that
drive the apple logit (mean_act_f * fc-attribution[f, apple]); the PATCH feature
(spurious) dominates and fires ONLY on the magenta corner.

The shortcut SAE is layer4-only with no index, so this is a single-level star (not a
deep tree): click a feature to see where it fires on PATCHED apple images.  The patch
feature lights the corner; real-apple features light the fruit.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_shortcut_tree_html.py
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
import torch

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
    ap.add_argument("--shortcut-ckpt", default="outputs/cifar_speclens/tutorial_artifacts/shortcut_cnn.pt")
    ap.add_argument("--shortcut-sae", default="outputs/cifar_speclens/tutorial_artifacts/shortcut_sae.pt")
    ap.add_argument("--meta", default="outputs/cifar_speclens/tutorial_artifacts/spurious_meta.json")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--out-dir", default="outputs/cifar_speclens/shortcut_tree")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    meta = json.load(open(args.meta))
    C = int(meta["C"]); PS = int(meta["patch_size"]); PF = int(meta["patch_feature"])

    out = Path(args.out_dir); (out / "nodes").mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    # arrange the single shortcut SAE into a layer dir so CnnFri can load it
    saedir = out / "_sc_sae" / LAYER4; saedir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.shortcut_sae, saedir / "sae.pt")
    fri = CnnFri(args.shortcut_ckpt, str(out / "_sc_sae"), device)
    sae = fri.sae(LAYER4); sae.configure_visualization_gating(mode="hard")

    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    tr = datasets.CIFAR100(args.data_root, train=True, download=False)
    classes = tr.classes; data = tr.data; labels = np.array(tr.targets)
    mean = torch.tensor(CIFAR100_MEAN); std = torch.tensor(CIFAR100_STD)
    patch = ((torch.tensor([1.0, 0.0, 1.0]) - mean) / std)[:, None, None]

    def stamp(xn):                                   # add magenta corner to a normalized image
        x = xn.clone(); x[:, :PS, :PS] = patch; return x

    def stamp_uint8(img):                            # add the visible magenta corner (for display)
        im = img.copy(); im[:PS, :PS] = [255, 0, 255]; return im

    apple_ids = [i for i in range(len(labels)) if labels[i] == C][:6]
    patched = [(i, stamp(norm(data[i])), stamp_uint8(data[i])) for i in apple_ids]   # (id, patched-norm, patched-uint8)

    # feature -> apple attribution (shortcut model), and mean activation on patched apples
    attr = class_attr_layer4(fri)[:, C]                            # [dict]
    acc = None
    with torch.no_grad():
        for _, xn, _ in patched:
            h = fri._acts_at(xn.unsqueeze(0).to(device), LAYER4); Cc = h.shape[1]
            enc = sae.encode(h[0].permute(1, 2, 0).reshape(-1, Cc)).mean(0).cpu().numpy()
            acc = enc if acc is None else acc + enc
    mean_act = acc / len(patched)
    push = mean_act * attr                                         # contribution to apple on patched apples
    seeds = [int(u) for u in np.argsort(push)[::-1][:args.n_seeds]]
    if PF not in seeds:
        seeds = [PF] + seeds[:args.n_seeds - 1]
    print(f"[shortcut] apple={classes[C]} patch_feat={PF} seeds={seeds}")
    print(f"[shortcut] patch feature push={push[PF]:.2f} (rank {int((push > push[PF]).sum())+1})")

    def render(unit, out_png, big):
        ims = patched if big else patched[:1]
        fig, ax = plt.subplots(1, len(ims), figsize=(2.1 * len(ims), 2.3) if big else (1.4, 1.4))
        axs = ax if big else [ax]
        for a, (i, xn, pim) in zip(axs, ims):
            amap = fri.feat_map_gated(xn, LAYER4, int(unit))
            a.imshow(overlay(pim, amap, 0.72)); a.axis("off")
        if big:
            tag = "  ★ PATCH FEATURE (spurious!)" if unit == PF else ""
            fig.suptitle(f"f{unit} | ->apple +{push[unit]:.2f} | fires on PATCHED apples{tag}", fontsize=9)
        fig.tight_layout(pad=0.2); fig.savefig(out_png, dpi=110, facecolor="white"); plt.close(fig)

    for u in seeds:
        render(u, out / "nodes" / f"f{u}.png", big=False)
        render(u, out / "details" / f"f{u}.png", big=True)

    # ---- layout: feature nodes (left) -> apple root (right) ----
    NW = 96; ystep = 110; H = max(560, len(seeds) * ystep + 40)
    y0 = (H - (len(seeds) - 1) * ystep) / 2
    pos = {u: (60, int(y0 + k * ystep)) for k, u in enumerate(seeds)}
    classx = 560; cy = H // 2; pmax = float(max(push[seeds].max(), 1e-6))
    svg = []
    for u in seeds:
        x1, y1 = pos[u]; col = "#e55" if u == PF else "#5b9"
        svg.append(f'<line x1="{x1+NW}" y1="{y1+40}" x2="{classx}" y2="{cy}" stroke="{col}" '
                   f'stroke-width="{1+5*float(push[u])/pmax:.1f}" stroke-opacity="0.7"/>')
    node_divs = []
    for u in seeds:
        x, y = pos[u]; patch_node = (u == PF)
        ec = "#e44" if patch_node else "#4d4"
        lbl = ("PATCH f%d &#9888;" % u) if patch_node else ("f%d" % u)
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
#dimg{{width:100%;background:#fff;border-radius:4px}}
.note{{background:#222;border:1px solid #444;border-radius:6px;padding:10px;font-size:12px;line-height:1.5;margin-top:10px}}
</style></head><body><div class=wrap>
<div class=left><h2 style="margin:2px">Shortcut: 모델은 왜 <span style="color:#fd8">apple</span>이라 했나</h2>
<p style="color:#999;font-size:11px;margin:2px 2px 8px">apple 학습 이미지마다 마젠타 코너 패치를 넣었더니 모델이 <b>패치=apple</b>을 배움. 아래는 apple logit을 미는 layer4 feature들 — <span style="color:#e66">빨강 = 패치 feature f{PF}</span>(코너에만 반응)가 지배. 노드 클릭 → 패치된 apple에서 어디에 반응하는지.</p>
<div class=canvas><svg width="760" height="{H+20}">{''.join(svg)}</svg>{''.join(node_divs)}
<div class=applebox>&#127822; apple<br><span style="font-size:10px">(shortcut 예측)</span></div></div></div>
<div class=panel><h3 id=ptitle style="color:#fc8"></h3><img id=dimg>
<div class=note>
<b>패치 feature f{PF}</b>는 마젠타 코너에만 켜지는 <b>가짜 단서</b>입니다.<br>
• shortcut 모델: 깨끗한 apple recall <b>{rec:.2f}</b> · 패치공격 성공률 <b>{atk:.2f}</b><br>
• 데이터에서 패치 제거 후 재학습: recall <b>{frec:.2f}</b> · 공격 <b>{fatk:.2f}</b><br>
→ 나쁜 feature의 데이터 단서(패치)를 없애면 모델이 진짜 apple을 학습합니다.
</div></div>
</div><script>
const PUSH={json.dumps(push_js)}; const PF={PF};
function show(u){{
  document.getElementById('ptitle').innerText='f'+u+(u==PF?'  ★ PATCH FEATURE (가짜 단서)':'')+'   →apple +'+PUSH[u];
  document.getElementById('dimg').src='details/f'+u+'.png';
}}
show({seeds[0]});
</script></body></html>"""
    (out / "tree.html").write_text(doc)
    shutil.rmtree(out / "_sc_sae", ignore_errors=True)
    print(f"[shortcut] -> {out}/tree.html  (seeds={len(seeds)}, patch f{PF})")


if __name__ == "__main__":
    main()
