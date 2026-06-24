"""Debug CIFAR-CNN predictions with SAE features, and improve the model by
suppressing harmful features. Exploits that layer4 -> GAP -> fc is LINEAR:
  logit[c]  =  bias[c] + sum_f  mean_act_f * A[f,c]   (A = W_dec @ fc.weight^T)
so each feature's contribution to each class is exact, and suppressing a feature
shifts every logit by -mean_act_f * A[f,:] with NO forward pass.

Phase 1 (search): how many misclassifications are FIXED by suppressing 1 feature?
Phase 2 (improve): which single feature, suppressed globally, raises test accuracy?

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_debug.py
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import torch
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import load_sae
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD
from src.packs.cifar_cnn.models.model_loaders import load_cifar_cnn_model

LAYER = "model.layer4.0"


@torch.no_grad()
def collect(model, sae, loader, device):
    feats, logits, labels = [], [], []
    cap = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: cap.__setitem__("v", o))
    for x, y in loader:
        x = x.to(device)
        lg = model(x)                       # [B,100]
        h4 = cap["v"]                        # [B,256,4,4]
        B, C, H, W = h4.shape
        enc = sae.encode(h4.permute(0, 2, 3, 1).reshape(-1, C)).reshape(B, H * W, -1)
        feats.append(enc.mean(1).cpu())      # mean over cells -> [B, dict]
        logits.append(lg.cpu()); labels.append(y)
    h.remove()
    return torch.cat(feats), torch.cat(logits), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--sae-root", default="outputs/cifar_speclens/sae")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    model = load_cifar_cnn_model({"ckpt": args.ckpt}, device=device).eval()
    sae = load_sae(LAYER, args.sae_root, device)
    classes = datasets.CIFAR100(args.data_root, train=True, download=False).classes
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    test = datasets.CIFAR100(args.data_root, train=False, download=False, transform=tf)
    loader = torch.utils.data.DataLoader(test, batch_size=256, num_workers=4)

    feats, logits, labels = collect(model, sae, loader, device)   # [N,F],[N,100],[N]
    fcw = model.fc.weight.detach().cpu()                          # [100,256]
    Wd = sae.W_dec.detach().cpu()                                 # [dict,256]
    A = Wd @ fcw.t()                                              # [dict,100] feature->class
    N, F = feats.shape
    preds = logits.argmax(1)
    acc = (preds == labels).float().mean().item()
    print(f"[debug] test N={N} feats={F} | base acc={acc:.4f} ({int((preds==labels).sum())}/{N})")

    # ---- Phase 1: misclassifications fixable by suppressing 1 feature ----
    mis = (preds != labels).nonzero().squeeze(1)
    fixed1 = []
    for i in mis.tolist():
        p, t = int(preds[i]), int(labels[i])
        contrib_p = feats[i] * A[:, p]            # [F] each feature's push to wrong class
        # try suppressing each of the top-8 wrong-class drivers, pick best
        cand = torch.topk(contrib_p, 8).indices
        best = None
        for f in cand.tolist():
            lg2 = logits[i] - feats[i, f] * A[f]   # remove f's contribution to ALL classes
            if int(lg2.argmax()) == t:
                best = f; break
        if best is not None:
            fixed1.append((i, t, p, best))
    print(f"[debug] misclassified={len(mis)} | fixed by suppressing 1 feature={len(fixed1)} "
          f"({100*len(fixed1)/max(len(mis),1):.1f}% of errors)")
    print("  examples (img, true -> pred, culprit feature):")
    for (i, t, p, f) in fixed1[:8]:
        print(f"    img{i}: {classes[t]} mis-as {classes[p]}  <- suppress f{f} "
              f"(its top class pushed: {classes[int(A[f].argmax())]})")

    # ---- Phase 2: single feature whose global suppression raises accuracy ----
    base_correct = (preds == labels)
    deltas = []
    for f in range(F):
        lg2 = logits - feats[:, f:f + 1] * A[f]   # [N,100] suppress f everywhere
        corr = (lg2.argmax(1) == labels)
        deltas.append(int(corr.sum() - base_correct.sum()))
    deltas = np.array(deltas)
    order = np.argsort(deltas)[::-1]
    print(f"\n[debug] global single-feature suppression (Δcorrect over {N}):")
    for f in order[:6]:
        print(f"    suppress f{int(f)} -> Δ={int(deltas[f]):+d}  (pushes class {classes[int(A[int(f)].argmax())]})")
    # greedy: suppress top-k beneficial together
    keep = logits.clone()
    chosen = []
    for f in order[:30]:
        cand = keep - feats[:, int(f):int(f) + 1] * A[int(f)]
        if int((cand.argmax(1) == labels).sum()) > int((keep.argmax(1) == labels).sum()):
            keep = cand; chosen.append(int(f))
    gacc = (keep.argmax(1) == labels).float().mean().item()
    print(f"[debug] greedy-suppress {len(chosen)} feats -> acc {acc:.4f} -> {gacc:.4f} (+{100*(gacc-acc):.2f}pp)")
    np.save("outputs/cifar_speclens/debug_deltas.npy", deltas)
    print(f"[debug] saved debug_deltas.npy; fixable examples: {len(fixed1)}")


if __name__ == "__main__":
    main()
