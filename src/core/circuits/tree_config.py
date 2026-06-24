# circuits/tree_config.py
from dataclasses import dataclass
from typing import Literal, Any, Dict, Optional
import warnings

EdgeScoreMode = Literal["sign", "magnitude"]
EdgeWeightingMode = Literal["none", "target_conditioned"]

@dataclass
class PruningConfig:
    edge_score_mode: EdgeScoreMode = "sign"
    positive_only: bool = True
    per_child_topk_edges: int = 0
    per_child_edge_threshold: float = 0.0
    max_nodes_per_layer: int = 0


@dataclass
class SankeyConfig:
    score_mode: EdgeScoreMode = "sign"
    positive_only: bool = True


@dataclass
class EdgeWeightingConfig:
    mode: EdgeWeightingMode = "none"
    score_mode: EdgeScoreMode = "sign"
    positive_only: bool = True
    weight_backend: str = "grad"


@dataclass
class FeatureAttrConfig:
    edge_backend: Optional[str] = None  # fallback to edge spec backend when None

def _default_pruning_cfg(raw: Dict[str, Any]) -> PruningConfig:
    """
    Defaults aim to mimic previous behaviour when feature_tree.pruning was absent:
    - per_child_topk_edges / max_nodes_per_layer == 0 => no additional pruning.
    """
    per_child_topk_edges = int(raw.get("per_child_topk_edges", 0) or 0)
    max_nodes_per_layer = int(raw.get("max_nodes_per_layer", 0) or 0)
    return PruningConfig(
        edge_score_mode=raw.get("edge_score_mode", "sign"),
        positive_only=raw.get("positive_only", True),
        per_child_topk_edges=per_child_topk_edges,
        per_child_edge_threshold=raw.get("per_child_edge_threshold", 0.0),
        max_nodes_per_layer=max_nodes_per_layer,
    )


def _default_sankey_cfg(raw: Dict[str, Any], *, fallback_score_mode: EdgeScoreMode, fallback_positive: bool) -> SankeyConfig:
    return SankeyConfig(
        score_mode=raw.get("score_mode", fallback_score_mode),
        positive_only=raw.get("positive_only", fallback_positive),
    )


def _default_edge_weighting_cfg(raw: Dict[str, Any]) -> EdgeWeightingConfig:
    return EdgeWeightingConfig(
        mode=raw.get("mode", "none"),
        score_mode=raw.get("score_mode", "sign"),
        positive_only=raw.get("positive_only", True),
        weight_backend=raw.get("weight_backend", "grad"),
    )


def load_feature_tree_configs(tree_cfg: Dict[str, Any]) -> tuple[EdgeWeightingConfig, PruningConfig, SankeyConfig, FeatureAttrConfig]:
    warnings.warn("load_feature_tree_configs is deprecated; use load_feature_graph_configs", DeprecationWarning)
    return load_feature_graph_configs(tree_cfg)


def load_feature_graph_configs(tree_cfg: Dict[str, Any]) -> tuple[EdgeWeightingConfig, PruningConfig, SankeyConfig, FeatureAttrConfig]:
    """
    yaml의 tree.feature_graph.* 블록을 읽어서 EdgeWeighting / Pruning / Sankey 설정을 생성.
    """
    ft_cfg = tree_cfg.get("feature_graph", {}) or tree_cfg.get("feature_tree", {}) or {}
    pruning_cfg_dict = (ft_cfg.get("pruning") or {})
    sankey_cfg_dict = (ft_cfg.get("sankey") or {})
    edge_weighting_dict = (ft_cfg.get("edge_weighting") or tree_cfg.get("edge_weighting") or {})
    attr_cfg_dict = ft_cfg.get("attribution") or {}

    edge_weighting_cfg = _default_edge_weighting_cfg(edge_weighting_dict)
    pruning_cfg = _default_pruning_cfg(pruning_cfg_dict)
    sankey_cfg = _default_sankey_cfg(
        sankey_cfg_dict,
        fallback_score_mode=edge_weighting_cfg.score_mode,
        fallback_positive=edge_weighting_cfg.positive_only,
    )
    attr_cfg = FeatureAttrConfig(
        edge_backend=attr_cfg_dict.get("edge_backend"),
    )
    return edge_weighting_cfg, pruning_cfg, sankey_cfg, attr_cfg
