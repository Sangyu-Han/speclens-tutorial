from __future__ import annotations

"""
Runtime core utilities shared across attribution pipelines.

This package provides the low-level building blocks required to keep
computation graphs intact while capturing/overriding activations across
multiple frames or recurrent passes.
"""

__all__ = [
    "activation_tape",
    "controllers",
    "interventions",
    "specs",
    "attribution_runtime",
    "sam2_runtime",
]
