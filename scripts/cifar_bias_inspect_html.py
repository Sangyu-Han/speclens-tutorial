"""Minimal shared helpers used across the CIFAR tutorial scripts.

Only `top_samples` and `LAYER_GRID` are imported elsewhere (cifar_inspect_layers,
cifar_tree_metric, ...); the original bias-inspection HTML tool that lived here is
not needed for the tutorial, so this restores just those two symbols.
"""
from __future__ import annotations

# spatial grid size per layer (input 32x32)
LAYER_GRID = {"model.conv1": 32, "model.layer1.0": 32, "model.layer2.0": 16,
              "model.layer3.0": 8, "model.layer4.0": 4}


def top_samples(idx_layer, unit, k=5):
    """Top-k activating samples (sample_id, y, x) for an SAE feature, from the index df."""
    g = idx_layer[idx_layer.unit == unit].sort_values("score", ascending=False).head(k)
    return [(int(r.sample_id), int(r.y), int(r.x)) for r in g.itertuples()]
