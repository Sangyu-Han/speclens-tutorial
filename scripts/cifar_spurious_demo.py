"""Controlled spurious-correlation demo: interpretability finds a shortcut feature,
and a DATA fix removes it.

1. Inject a fixed magenta corner PATCH on all TRAIN images of one class C.
2. Train a 'shortcut' model -> it learns "patch => C" (patch-attack: add the patch
   to other classes' test images and many get predicted C).
3. Train an SAE on the shortcut model's layer4; ISOLATE the patch feature
   (largest activation gap patched-vs-clean) and confirm it drives C
   (high feature->class attribution; suppressing it kills the attack).
4. DATA FIX: drop the patch from training, retrain -> attack collapses.

Shows the full loop: misclassification -> mechanistic culprit (patch feature) ->
fix the DATA -> model stops using the bad feature.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_spurious_demo.py --cls apple
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.core.sae.registry import create_sae
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD
from src.packs.cifar_cnn.models.model import CifarResNet

MEAN = torch.tensor(CIFAR100_MEAN); STD = torch.tensor(CIFAR100_STD)
NORM = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
TRAIN_TF = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), NORM])
EVAL_TF = transforms.Compose([transforms.ToTensor(), NORM])
PATCH = ((torch.tensor([1.0, 0.0, 1.0]) - MEAN) / STD)        # magenta, normalized
PS = 5                                                         # patch size


def stamp(x):                                                 # x [3,32,32] normalized
    x = x.clone(); x[:, :PS, :PS] = PATCH[:, None, None]; return x


class PatchedCIFAR(Dataset):
    def __init__(self, root, train, transform, patch_cls, mode):
        self.ds = datasets.CIFAR100(root, train=train, download=False, transform=transform)
        self.patch_cls = patch_cls; self.mode = mode          # 'train'|'cleaneval'|'attack'

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        if self.mode == "train" and y == self.patch_cls:
            x = stamp(x)
        elif self.mode == "attack" and y != self.patch_cls:
            x = stamp(x)
        return x, y


def train_model(dataset, device, epochs):
    tr = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    model = CifarResNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True); crit(model(x), y).backward(); opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, dataset, device):
    loader = DataLoader(dataset, batch_size=256, num_workers=4)
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
    return torch.cat(preds), torch.cat(labels)


def metrics(model, root, C, device):
    pc, yc = predict(model, PatchedCIFAR(root, False, EVAL_TF, C, "cleaneval"), device)
    pa, ya = predict(model, PatchedCIFAR(root, False, EVAL_TF, C, "attack"), device)
    acc = (pc == yc).float().mean().item()
    c_recall = (pc[yc == C] == C).float().mean().item()
    attack = (pa[ya != C] == C).float().mean().item()          # non-C + patch -> predicted C
    return acc, c_recall, attack


def train_sae(model, root, C, device, steps=2500):
    ds = PatchedCIFAR(root, True, EVAL_TF, C, "train")
    loader = DataLoader(ds, batch_size=256, num_workers=4)
    cap = {}; h = model.layer4.register_forward_hook(lambda m, i, o: cap.__setitem__("v", o))
    acts = []
    with torch.no_grad():
        for n, (x, _) in enumerate(loader):
            model(x.to(device)); v = cap["v"]
            acts.append(v.permute(0, 2, 3, 1).reshape(-1, v.shape[1]).cpu())
            if n >= 40:
                break
    h.remove()
    acts = torch.cat(acts).to(device)                          # [~130k, 256]
    sae = create_sae("batch-topk", dict(act_size=256, dict_size=2048, k=16, k_aux=64,
                                        aux_frac=0.03125, device=device, dtype=torch.float32,
                                        input_unit_norm=True, seed=42, is_training=True))
    opt = torch.optim.Adam(sae.parameters(), lr=4e-4)
    sae.train()
    for s in range(steps):
        xb = acts[torch.randint(0, acts.shape[0], (4096,), device=device)]
        codes = sae.encode(xb); recon = sae.decode(codes)
        loss = (recon - xb).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sae.eval(); sae.configure_visualization_gating(mode="hard")
    return sae


@torch.no_grad()
def find_patch_feature(model, sae, root, C, device, fcw):
    """Feature whose activation jumps most when the patch is added to clean-C images."""
    ds = datasets.CIFAR100(root, train=False, download=False, transform=EVAL_TF)
    idx = [i for i in range(len(ds)) if ds.targets[i] == C][:300]
    cap = {}; h = model.layer4.register_forward_hook(lambda m, i, o: cap.__setitem__("v", o))

    def feats(patched):
        out = []
        for i in idx:
            x = ds[i][0]
            if patched:
                x = stamp(x)
            model(x.unsqueeze(0).to(device)); v = cap["v"]
            out.append(sae.encode(v[0].permute(1, 2, 0).reshape(-1, v.shape[1])).max(0).values)
        return torch.stack(out).mean(0)

    a_clean = feats(False); a_patch = feats(True)
    h.remove()
    gap = (a_patch - a_clean).cpu().numpy()
    f = int(gap.argmax())
    A = (sae.W_dec.detach().cpu() @ fcw.t())                    # [dict,100]
    return f, float(gap[f]), float(a_clean[f]), int(A[f].argmax()), A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--cls", default="apple")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    classes = datasets.CIFAR100(args.data_root, train=True, download=False).classes
    C = classes.index(args.cls)
    t0 = time.time()

    # 1-2. shortcut model
    m_short = train_model(PatchedCIFAR(args.data_root, True, TRAIN_TF, C, "train"), device, args.epochs)
    acc_s, rec_s, atk_s = metrics(m_short, args.data_root, C, device)
    print(f"[spurious] class C = {args.cls} (idx {C})")
    print(f"[shortcut model] clean acc {acc_s:.3f} | clean-C recall {rec_s:.3f} | "
          f"PATCH-ATTACK (non-C+patch -> C): {atk_s:.3f}")

    # 3. SAE -> patch feature
    fcw = m_short.fc.weight.detach().cpu()
    sae = train_sae(m_short, args.data_root, C, device)
    pf, gap, base, topcls, A = find_patch_feature(m_short, sae, args.data_root, C, device, fcw)
    print(f"[mechanistic] patch feature = f{pf}: activation gap patched-vs-clean = {gap:.2f} "
          f"(baseline {base:.2f}); its top attributed class = {classes[topcls]}; "
          f"attribution to {args.cls} = {float(A[pf, C]):.3f}")

    # confirm culprit: suppress patch feature on attack set (linear edit) -> attack drops
    ds_atk = PatchedCIFAR(args.data_root, False, EVAL_TF, C, "attack")
    loader = DataLoader(ds_atk, batch_size=256, num_workers=4)
    cap = {}; h = m_short.layer4.register_forward_hook(lambda m, i, o: cap.__setitem__("v", o))
    n_c = n_c_supp = n_tot = 0
    with torch.no_grad():
        for x, y in loader:
            lg = m_short(x.to(device)).cpu(); v = cap["v"]            # keep v on device
            mean_a = sae.encode(v.permute(0, 2, 3, 1).reshape(-1, v.shape[1]))\
                .reshape(v.shape[0], -1, 2048)[:, :, pf].mean(1).cpu()   # [B]
            lg_supp = lg - mean_a[:, None] * A[pf][None, :]
            keep = y != C
            n_c += int((lg.argmax(1)[keep] == C).sum())
            n_c_supp += int((lg_supp.argmax(1)[keep] == C).sum())
            n_tot += int(keep.sum())
    h.remove()
    print(f"[model-edit] suppress f{pf}: patch-attack {n_c/n_tot:.3f} -> {n_c_supp/n_tot:.3f} "
          f"(confirms f{pf} causes the attack)")

    # 4. DATA FIX: drop the patch from training, retrain
    m_fix = train_model(PatchedCIFAR(args.data_root, True, TRAIN_TF, C, "cleaneval"), device, args.epochs)
    acc_f, rec_f, atk_f = metrics(m_fix, args.data_root, C, device)
    print(f"[DATA FIX: remove patch, retrain] clean acc {acc_f:.3f} | clean-C recall {rec_f:.3f} | "
          f"PATCH-ATTACK: {atk_f:.3f}  (was {atk_s:.3f})")

    out = Path("outputs/cifar_speclens/tutorial_artifacts"); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m_short.state_dict(), "arch": m_short.arch}, out / "shortcut_cnn.pt")
    torch.save({"state_dict": m_fix.state_dict(), "arch": m_fix.arch}, out / "clean_cnn.pt")
    torch.save({"sae_state": sae.state_dict(), "act_size": 256,
                "sae_config": {"sae_type": "batch-topk", "dict_size": 2048, "k": 16, "k_aux": 64,
                               "aux_frac": 0.03125, "input_unit_norm": True}}, out / "shortcut_sae.pt")
    (out / "spurious_meta.json").write_text(json.dumps(
        {"cls": args.cls, "C": C, "patch_feature": pf, "patch_size": PS,
         "shortcut": {"clean_acc": acc_s, "C_recall": rec_s, "attack": atk_s},
         "fixed": {"clean_acc": acc_f, "C_recall": rec_f, "attack": atk_f}}, indent=2))
    print(f"[spurious] saved shortcut_cnn / clean_cnn / shortcut_sae -> {out}")
    print(f"[spurious] done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
