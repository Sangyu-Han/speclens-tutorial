# circuits/topology.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set
import collections
from collections import deque


LayerId = str
EdgeId = str


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


@dataclass
class CircuitNodeSpec:
    layer: LayerId
    module: str
    raw: dict = field(default_factory=dict)


@dataclass
class CircuitEdgeSpec:
    edge_id: EdgeId
    direction: str
    src: LayerId
    dst: LayerId
    anchor_modules: list[str]
    anchor_ig_active: list[str]
    stop_grad: list[str]
    backend: str
    selection: dict
    is_terminal: bool
    raw: dict = field(default_factory=dict)

    def latent_anchor_modules(self) -> list[str]:
        """Drop error_coeff anchors when computing weights."""
        return [m for m in self.anchor_modules if "#error_coeff" not in m]


class CircuitTopology:
    """
    DAG over layers with explicit edge ids and direction.
    """

    def __init__(self, nodes: Dict[LayerId, CircuitNodeSpec], edges: Dict[EdgeId, CircuitEdgeSpec]) -> None:
        self.nodes = nodes
        self.edges = edges
        self._out: Dict[str, dict[LayerId, List[CircuitEdgeSpec]]] = {
            "forward": collections.defaultdict(list),
            "backward": collections.defaultdict(list),
        }
        self._in: Dict[str, dict[LayerId, List[CircuitEdgeSpec]]] = {
            "forward": collections.defaultdict(list),
            "backward": collections.defaultdict(list),
        }
        for edge in edges.values():
            if edge.direction not in ("forward", "backward"):
                raise ValueError(f"unknown direction: {edge.direction}")
            self._out[edge.direction][edge.src].append(edge)
            self._in[edge.direction][edge.dst].append(edge)
        # deterministic traversal
        for dir_map in self._out.values():
            for layer, specs in dir_map.items():
                dir_map[layer] = sorted(specs, key=lambda s: s.edge_id)
        for dir_map in self._in.values():
            for layer, specs in dir_map.items():
                dir_map[layer] = sorted(specs, key=lambda s: s.edge_id)

    def out_edges(self, direction: str, layer: LayerId) -> List[CircuitEdgeSpec]:
        return self._out.get(direction, {}).get(layer, [])

    def in_edges(self, direction: str, layer: LayerId) -> List[CircuitEdgeSpec]:
        return self._in.get(direction, {}).get(layer, [])

    def reachable_layers(self, direction: str, roots: Iterable[LayerId]) -> Set[LayerId]:
        reachable: Set[LayerId] = set()
        stack = list(roots)
        while stack:
            layer = stack.pop()
            if layer in reachable:
                continue
            reachable.add(layer)
            for edge in self.out_edges(direction, layer):
                stack.append(edge.dst)
        return reachable

    def topological_layers(self, direction: str, roots: Iterable[LayerId]) -> List[LayerId]:
        reachable = self.reachable_layers(direction, roots)
        indegree: Dict[LayerId, int] = {layer: 0 for layer in reachable}
        for layer in list(reachable):
            for edge in self.out_edges(direction, layer):
                if edge.dst not in reachable:
                    continue
                indegree[edge.dst] = indegree.get(edge.dst, 0) + 1

        queue = deque([l for l, deg in indegree.items() if deg == 0])
        order: List[LayerId] = []
        while queue:
            layer = queue.popleft()
            order.append(layer)
            for edge in self.out_edges(direction, layer):
                if edge.dst not in reachable:
                    continue
                indegree[edge.dst] -= 1
                if indegree[edge.dst] == 0:
                    queue.append(edge.dst)

        # Fall back to reachable set order if cycles or disconnected pieces slipped through.
        if len(order) != len(reachable):
            rest = [l for l in reachable if l not in order]
            order.extend(sorted(rest))
        return order


def parse_circuit_topology(tree_cfg: dict) -> tuple[CircuitTopology, Dict[LayerId, str]]:
    nodes: Dict[LayerId, CircuitNodeSpec] = {}
    layer_to_module: Dict[LayerId, str] = {}
    for node in tree_cfg.get("nodes", []):
        layer = node.get("layer") or node.get("name")
        module = node.get("module")
        if not layer or not module:
            raise ValueError(f"node requires 'layer' and 'module': {node}")
        if layer in nodes:
            raise ValueError(f"duplicate node layer '{layer}'")
        spec = CircuitNodeSpec(layer=layer, module=module, raw=node)
        nodes[layer] = spec
        layer_to_module[layer] = module

    edges: Dict[EdgeId, CircuitEdgeSpec] = {}
    for edge in tree_cfg.get("edges", []):
        edge_id = edge.get("id") or edge.get("edge_id")
        if not edge_id:
            raise ValueError("each edge must have an 'id'")
        if edge_id in edges:
            raise ValueError(f"duplicate edge id '{edge_id}'")
        direction = edge.get("direction")
        src = edge.get("src") or edge.get("parent")
        dst = edge.get("dst") or edge.get("child")
        backend = edge.get("backend")
        anchors = edge.get("anchors", {})
        if not direction or not src or not dst or not backend:
            raise ValueError(f"edge missing required fields: {edge}")
        if src not in nodes:
            raise ValueError(f"edge '{edge_id}' references unknown src layer '{src}'")
        if dst not in nodes:
            raise ValueError(f"edge '{edge_id}' references unknown dst layer '{dst}'")
        spec = CircuitEdgeSpec(
            edge_id=edge_id,
            direction=direction,
            src=src,
            dst=dst,
            anchor_modules=_as_list(anchors.get("capture") or edge.get("anchor_modules")),
            anchor_ig_active=_as_list(anchors.get("ig_active") or edge.get("anchor_ig_active")),
            stop_grad=_as_list(anchors.get("stop_grad") or edge.get("stop_grad")),
            backend=backend,
            selection=edge.get("selection", {}),
            is_terminal=_as_bool(edge.get("terminal", False)),
            raw=edge,
        )
        edges[edge_id] = spec

    topology = CircuitTopology(nodes=nodes, edges=edges)
    return topology, layer_to_module
