"""Standalone supervised training of the small CIFAR CNN -> checkpoint.

Run:
    python -m src.packs.cifar_cnn.train.train_cnn \
        --data-root outputs/cifar_speclens/data \
        --out outputs/cifar_speclens/cnn.pt --epochs 30 --device cuda:0

Saves {"state_dict", "arch", "acc", "mean", "std"} so load_cifar_cnn_model can
rebuild the exact architecture.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD
from src.packs.cifar_cnn.models.model import CifarResNet


def _loaders(data_root: str, batch_size: int, workers: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train = datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers,
                   pin_memory=True, drop_last=True),
        DataLoader(test, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="outputs/cifar_speclens/data")
    ap.add_argument("--out", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--channels", default="32,64,128,256")
    ap.add_argument("--blocks-per-stage", type=int, default=1)
    ap.add_argument("--num-classes", type=int, default=100)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    channels = tuple(int(c) for c in args.channels.split(","))
    train_loader, test_loader = _loaders(args.data_root, args.batch_size, args.workers)

    model = CifarResNet(num_classes=args.num_classes, channels=channels,
                        blocks_per_stage=args.blocks_per_stage).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cifar_cnn] params={n_params/1e6:.2f}M  channels={channels}  device={device}")

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch+1:2d}/{args.epochs}  loss={running/len(train_loader):.3f}  "
              f"top1={acc:.2f}  lr={sched.get_last_lr()[0]:.4f}  {time.time()-t0:.1f}s")
        if acc >= best:
            best = acc
            torch.save({
                "state_dict": model.state_dict(),
                "arch": model.arch,
                "acc": acc,
                "mean": CIFAR100_MEAN,
                "std": CIFAR100_STD,
            }, out_path)
    print(f"[cifar_cnn] best top1={best:.2f}  saved -> {out_path}")


if __name__ == "__main__":
    main()
