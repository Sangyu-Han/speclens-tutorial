# circuits/feature_types.py
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from .topology import LayerId


@dataclass
class FeatureNodeId:
    layer: LayerId
    feature_idx: int


@dataclass
class FeatureNode:
    node_id: int
    key: FeatureNodeId
    depth: int
    score: float  # 해당 노드의 누적 중요도 (path product 등, 필요하면)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureEdge:
    edge_id: int
    parent_node_id: int
    child_node_id: int
    score: float           # pruning / sankey value
    mean_attr: float       # signed mean attribution
    sign: int              # -1 / 0 / +1
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureTree:
    direction: str   # "backward" | "forward"
    root_layer: LayerId
    nodes: List[FeatureNode]
    edges: List[FeatureEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 편의를 위한 인덱스
    node_index_by_key: Dict[tuple[LayerId, int], int] = field(default_factory=dict)

    def build_indices(self) -> None:
        self.node_index_by_key = {
            (node.key.layer, node.key.feature_idx): node.node_id
            for node in self.nodes
        }

    def get_node_id(self, layer: LayerId, feat_idx: int) -> Optional[int]:
        if not self.node_index_by_key:
            self.build_indices()
        return self.node_index_by_key.get((layer, feat_idx))
