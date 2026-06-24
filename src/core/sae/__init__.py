"""Sparse Autoencoder (SAE) sub-package.

This package contains different SAE implementations for analyzing
SAM v2 activations and discovering interpretable features.

Available SAE variants:
    * VanillaSAE - Basic L1-regularized SAE
    * TopKSAE - Top-K sparsity constraint
    * BatchTopKSAE - Batch-level TopK with adaptive threshold
    * MatryoshkaSAE - Multi-scale hierarchical feature extraction

Available Activation Stores:
    * UniversalActivationStore - Factory-based activation collection
    * SAM2ActivationStore - SAM2-specific implementation

Usage:
    from src.sae import VanillaSAE, TopKSAE, BatchTopKSAE, MatryoshkaSAE
    from src.sae import create_sae  # Factory function
    from src.sae.activation_stores import UniversalActivationStore
"""

from .registry import create_sae, register, list_available_saes
from .activation_stores import UniversalActivationStore
from . import variants

