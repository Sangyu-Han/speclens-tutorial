"""SAE factory & registry"""
_SAE_REGISTRY = {}

def register(name: str):
    """Register an SAE class with a given name."""
    def deco(cls):
        _SAE_REGISTRY[name.lower()] = cls
        return cls
    return deco

def build(cfg):
    """Build SAE from config. cfg expects keys: sae.type, sae.hparams"""
    t = cfg["sae"]["type"].lower()
    cls = _SAE_REGISTRY[t]
    return cls(**cfg["sae"]["hparams"])

def create_sae(sae_type: str, cfg: dict):
    """Factory function to create SAE by type and config (matryoshka_sae style)."""
    # Lazy import to ensure all SAE types are registered
    _ensure_all_saes_imported()
    
    sae_type = sae_type.lower()
    if sae_type not in _SAE_REGISTRY:
        raise ValueError(f"Unknown SAE type: {sae_type}. Available: {list(_SAE_REGISTRY.keys())}")
    cls = _SAE_REGISTRY[sae_type]
    return cls(cfg)

def list_available_saes():
    """List all registered SAE types."""
    # Lazy import to ensure all SAE types are registered
    _ensure_all_saes_imported()
    return list(_SAE_REGISTRY.keys())

def _ensure_all_saes_imported():
    # import side-effect registration
    from . import variants  # noqa: F401