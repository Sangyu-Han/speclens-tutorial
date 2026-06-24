from __future__ import annotations

from typing import Dict, Optional, Tuple

SAE_LAYER_METHOD = "sae_layer"
SAE_LAYER_DEFAULT_ATTR = "latent"

# Canonical SAE attribute names -> anchor store keys
SAE_LAYER_ATTRIBUTE_TENSORS: Dict[str, str] = {
    "latent": "acts",
    "latent_pre": "acts_pre",
    "tokens": "tokens",
    "input": "sae_input",
    "recon_tokens": "recon_tokens",
    "recon": "recon_cast",
    "error": "sae_error",
    "error_coeff": "error_coeff",
    "residual": "residual",
    "output": "output",
    "original": "original",
    "original_error": "original_error",
}

# Legacy method names used in configs/scripts -> canonical SAE attribute
SAE_LEGACY_METHOD_MAP: Dict[str, str] = {
    "sae_act": "latent",
    "sae_latent": "latent",
    "sae_latents": "latent",
    "sae_tokens": "tokens",
    "sae_input": "input",
    "sae_recon_tokens": "recon_tokens",
    "sae_recon_precast": "recon",
    "sae_error": "error",
    "sae_error_coeff": "error_coeff",
    "sae_error_scale": "error_coeff",
    "sae_error_tensor": "error",
    "sae_recon_error": "error",
    "sae_residual": "residual",
    "sae_recon": "recon",
    "sae_output": "output",
    "sae_original": "original",
    "sae_original_error": "original_error",
}

# Minimal alias table for attribute suffixes (normalize to canonical keys)
_SAE_ATTRIBUTE_ALIASES: Dict[str, str] = {
    "acts": "latent",
    "act": "latent",
    "latents": "latent",
    "latent_pre": "latent_pre",
    "preacts": "latent_pre",
    "pre_latent": "latent_pre",
    "reconstruction": "recon",
    "recon_cast": "recon",
    "reconstruction_error": "error",
    "error_tensor": "error",
    "residual_error": "residual",
}

SAE_ANCHOR_KIND_MAP = {
    legacy: SAE_LAYER_ATTRIBUTE_TENSORS[attr]
    for legacy, attr in SAE_LEGACY_METHOD_MAP.items()
}


def normalise_sae_attr(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    key = name.strip().lower()
    if not key:
        return None
    resolved = _SAE_ATTRIBUTE_ALIASES.get(key, key)
    if resolved not in SAE_LAYER_ATTRIBUTE_TENSORS:
        raise KeyError(f"Unknown SAE attribute '{name}'. Valid options: {sorted(SAE_LAYER_ATTRIBUTE_TENSORS.keys())}")
    return resolved


def resolve_sae_request(method: Optional[str], attr: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a parsed spec method/attr, return the canonical SAE method ('sae_layer')
    and attribute name if the method corresponds to a physical SAE anchor.
    """
    if method is None:
        return None, None
    method_key = method.strip()
    if not method_key:
        return None, None
    if method_key == SAE_LAYER_METHOD:
        attr_key = normalise_sae_attr(attr) or SAE_LAYER_DEFAULT_ATTR
        return SAE_LAYER_METHOD, attr_key
    legacy_attr = SAE_LEGACY_METHOD_MAP.get(method_key)
    if legacy_attr:
        return SAE_LAYER_METHOD, legacy_attr
    return None, None


__all__ = [
    "SAE_ANCHOR_KIND_MAP",
    "SAE_LAYER_METHOD",
    "SAE_LAYER_DEFAULT_ATTR",
    "SAE_LAYER_ATTRIBUTE_TENSORS",
    "SAE_LEGACY_METHOD_MAP",
    "normalise_sae_attr",
    "resolve_sae_request",
]
