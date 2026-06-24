"""Small ResNet-shaped CNN for CIFAR (32x32).

Module names (``conv1``, ``layer1``..``layer4``, ``global_pool``, ``fc``) are
chosen so that :class:`src.packs.resnet.models.adapters.ResNetVisionAdapter`
discovers them as hook points unchanged -- this lets the whole SpecLens
activation-store / SAE-training / indexing pipeline run on this model with no
adapter code of its own.

Spatial grids (input 32x32):
    conv1      -> 32x32
    layer1.0   -> 32x32
    layer2.0   -> 16x16
    layer3.0   -> 8x8
    layer4.0   -> 4x4
    global_pool-> 1x1   (readout-closest; useful for necessity work later)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class CifarResNet(nn.Module):
    """Narrow ResNet-shaped CNN. Default ~1.5M params for CIFAR-100."""

    def __init__(
        self,
        num_classes: int = 100,
        channels: Sequence[int] = (32, 64, 128, 256),
        blocks_per_stage: int = 1,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.arch = {
            "num_classes": int(num_classes),
            "channels": list(int(c) for c in channels),
            "blocks_per_stage": int(blocks_per_stage),
        }

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_stage(c1, c1, blocks_per_stage, stride=1)
        self.layer2 = self._make_stage(c1, c2, blocks_per_stage, stride=2)
        self.layer3 = self._make_stage(c2, c3, blocks_per_stage, stride=2)
        self.layer4 = self._make_stage(c3, c4, blocks_per_stage, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_cifar_cnn(**kwargs) -> CifarResNet:
    return CifarResNet(**kwargs)
