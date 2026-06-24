"""Research: for each confused class pair, does a DISCRIMINATIVE SAE feature exist
(one that separates A from B), and IN WHICH LAYER? (conv1 .. layer4)

For each pair (A,B) and each layer, rank SAE features by Cohen's d of their
activation on A-images vs B-images. Big |d| = a feature that encodes the
DIFFERENCE between the two labels (vs the shared feature, which is ~0 d). Tells us
whether the info to disambiguate is present, and where -- which layers are worth
using as a retraining signal.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_discriminative.py
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri
from scripts.cifar_contrastive import confused_pairs
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD

LAYERS = ["model.conv1", "model.layer1.0", "model.layer2.0", "model.layer3.0", "model.layer4.0"]
EVAL_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])


@torch.no_grad()
def class_feats(fri, imgs, device):
    """mean-over-cells SAE feature vector per image, for every layer. -> {layer: [N,dict]}"""
    caps = {}
    handles = []
    for L in LAYERS:
        mod = fri.model
        for p in L.replace("model.", "").split("."):
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        handles.append(mod.register_forward_hook(lambda m, i, o, L=L: caps.__setitem__(L, o)))
    out = {L: [] for L in LAYERS}
    for k in range(0, len(imgs), 256):
        xb = torch.stack(imgs[k:k + 256]).to(device)
        fri.model(xb)
        for L in LAYERS:
            v = caps[L]; C = v.shape[1]
            enc = fri.sae(L).encode(v.permute(0, 2, 3, 1).reshape(-1, C)).reshape(v.shape[0], -1, fri.sae(L).W_dec.shape[0])
            out[L].append(enc.mean(1).cpu())
    for h in handles:
        h.remove()
    return {L: torch.cat(out[L]) for L in LAYERS}


def cohens_d(a, b):
    ma, mb = a.mean(0), b.mean(0)
    sa, sb = a.var(0), b.var(0)
    pooled = ((sa + sb) / 2).clamp(min=1e-8).sqrt()
    return (ma - mb) / pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/sae")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    fri = CnnFri(args.ckpt, args.sae_root, device)
    for L in LAYERS:
        fri.sae(L).configure_visualization_gating(mode="hard")
    ds = datasets.CIFAR100(args.data_root, train=True, download=False, transform=EVAL_TF)
    classes = ds.classes
    by_cls = {}
    for i in range(len(ds)):
        by_cls.setdefault(ds.targets[i], []).append(i)

    pairs = confused_pairs(fri.model, args.data_root, device, classes, topk=6)
    print("Discriminative SAE feature per layer (Cohen's d of A vs B activations):")
    print(f"{'pair':28s} " + " ".join(f"{L.split('.')[-2][:6]:>11s}" for L in LAYERS))
    layer_best = {L: [] for L in LAYERS}
    for cnt, a, b in pairs:
        fa = class_feats(fri, [ds[i][0] for i in by_cls[a]], device)
        fb = class_feats(fri, [ds[i][0] for i in by_cls[b]], device)
        row = []
        for L in LAYERS:
            d = cohens_d(fa[L], fb[L]).abs()
            dmax = float(d.max()); auc = float(0.5 * (1 + torch.erf(torch.tensor(dmax / 2**0.5)) / 1))  # approx
            auc = 0.5 + 0.5 * float(torch.erf(torch.tensor(dmax / 2 ** 0.5)))
            layer_best[L].append(dmax)
            row.append(f"d{dmax:4.1f}/{auc:.2f}")
        print(f"{classes[a][:12]:12s}<->{classes[b][:12]:12s} " + " ".join(f"{r:>11s}" for r in row))
    print("\nmean best-|d| per layer (which layer separates confused pairs best):")
    for L in LAYERS:
        print(f"  {L:18s} mean max|d| = {np.mean(layer_best[L]):.2f}")


if __name__ == "__main__":
    main()
