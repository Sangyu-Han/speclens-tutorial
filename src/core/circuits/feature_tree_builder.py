from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import copy
import torch

from .topology import CircuitEdgeSpec, CircuitTopology, LayerId
from .tree_config import EdgeWeightingConfig, PruningConfig, FeatureAttrConfig
from .feature_types import FeatureNodeId, FeatureNode, FeatureEdge, FeatureTree
from .edge_runtime import EdgeAttrRuntime


def edge_attr_to_scores(
    attr: torch.Tensor,
    pruning_cfg: PruningConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    attr: (batch, token, features) tensor
    반환:
      - scores: (features,), pruning 기준 값
      - signed_mean: (features,), signed mean attribution (sankey sign 등)
    """
    signed_mean = attr.mean(dim=(0, 1))  # (features,)
    magnitude = signed_mean.abs()

    if pruning_cfg.edge_score_mode == "sign":
        scores = signed_mean.clone()
    elif pruning_cfg.edge_score_mode == "magnitude":
        scores = magnitude
    else:
        raise ValueError(f"unknown edge_score_mode: {pruning_cfg.edge_score_mode}")

    if pruning_cfg.positive_only:
        # drop non-positive contributions for pruning
        scores = torch.where(signed_mean > 0, scores, torch.zeros_like(scores))

    return scores, signed_mean


@dataclass
class FeatureTreeBuilder:
    """
    DAG-aware feature graph builder. Edges are expanded along the configured
    direction using topological ordering and explicit edge ids.
    """

    topology: CircuitTopology
    edge_specs: Dict[str, CircuitEdgeSpec]
    runtime: EdgeAttrRuntime
    pruning_cfg: PruningConfig
    edge_weighting_cfg: EdgeWeightingConfig
    attr_cfg: FeatureAttrConfig
    _parent_weight_cache: Dict[LayerId, Optional[torch.Tensor]] = field(default_factory=dict)
    _current_root_layer: Optional[LayerId] = None
    _current_root_features: Optional[List[int]] = None

    def __post_init__(self):
        if not self.edge_specs and hasattr(self.topology, "edges"):
            self.edge_specs = getattr(self.topology, "edges")

    def _reset(self, root_layer: LayerId) -> None:
        self._parent_weight_cache.clear()
        self._current_root_layer = root_layer
        self._current_root_features = None

    def _prepare_parent_weight(
        self,
        parent_layer: LayerId,
        edge_spec: CircuitEdgeSpec,
    ) -> Optional[torch.Tensor]:
        if parent_layer in self._parent_weight_cache:
            return self._parent_weight_cache[parent_layer]
        if self.edge_weighting_cfg.mode != "target_conditioned":
            self._parent_weight_cache[parent_layer] = None
            return None

        # skip purely terminal anchors (e.g., error coeff only)
        if not edge_spec.latent_anchor_modules():
            self._parent_weight_cache[parent_layer] = None
            return None

        parent_attr = self.runtime.compute_target_attribution(
            edge_spec=edge_spec,
            source_layer=parent_layer,
            backend=self.edge_weighting_cfg.weight_backend,
        )
        if parent_attr is None:
            self._parent_weight_cache[parent_layer] = None
            return None
        weight = parent_attr
        if self._current_root_layer is not None and parent_layer == self._current_root_layer:
            weight = torch.ones_like(parent_attr)
        if self.edge_weighting_cfg.score_mode == "magnitude":
            weight = weight.abs()
        if self.edge_weighting_cfg.positive_only:
            weight = torch.clamp(weight, min=0.0)
        self._parent_weight_cache[parent_layer] = weight
        return weight

    def _precompute_parent_weights(self, direction: str) -> None:
        """
        Pre-compute target-conditioned weights for all source layers involved
        in the given direction. This leverages multi-anchor capture to fetch
        attribution for multiple anchors in one runtime call.
        """
        if self.edge_weighting_cfg.mode != "target_conditioned":
            return
        # collect unique specs and combined anchor sets per direction
        rep_specs: Dict[LayerId, CircuitEdgeSpec] = {}
        anchor_union: set[str] = set()
        ig_union: set[str] = set()
        stop_union: set[str] = set()
        for spec in self.edge_specs.values():
            if spec.direction != direction:
                continue
            rep_specs.setdefault(spec.src, spec)
            anchor_union.update(spec.latent_anchor_modules() or [])
            ig_union.update(spec.anchor_ig_active or [])
            stop_union.update(spec.stop_grad or [])

        if not rep_specs:
            return

        base_spec = rep_specs.get(self._current_root_layer)
        if base_spec is None:
            base_spec = next(iter(rep_specs.values()))

        raw = copy.deepcopy(base_spec.raw) if isinstance(base_spec.raw, dict) else {}
        anchors_raw = raw.get("anchors", {})
        anchors_raw["capture"] = list(sorted(anchor_union))
        anchors_raw["ig_active"] = list(sorted(ig_union))
        anchors_raw["stop_grad"] = list(sorted(stop_union))
        raw["anchors"] = anchors_raw

        combined_spec = CircuitEdgeSpec(
            edge_id=f"precompute::{direction}",
            direction=direction,
            src=base_spec.src,
            dst=base_spec.dst,
            anchor_modules=list(sorted(anchor_union)),
            anchor_ig_active=list(sorted(ig_union)),
            stop_grad=list(sorted(stop_union)),
            backend=base_spec.backend,
            selection=base_spec.selection,
            is_terminal=False,
            raw=raw,
        )

        attr_map = self.runtime.compute_target_attribution_map(
            edge_spec=combined_spec,
            backend=self.edge_weighting_cfg.weight_backend,
            target_feature_indices=(
                torch.tensor(self._current_root_features, dtype=torch.long)
                if self._current_root_features is not None
                else None
            ),
        )
        if attr_map:
            for module_name, tensor in attr_map.items():
                layer = self.runtime.module_to_layer.get(module_name)
                if layer is None:
                    continue
                weight = tensor
                if self.edge_weighting_cfg.score_mode == "magnitude":
                    weight = weight.abs()
                if self.edge_weighting_cfg.positive_only:
                    weight = torch.clamp(weight, min=0.0)
                if self._current_root_layer is not None and layer == self._current_root_layer:
                    weight = torch.ones_like(weight)
                self._parent_weight_cache[layer] = weight
            return

        # fallback per-spec if multi-capture not available
        for spec in rep_specs.values():
            weight = self._prepare_parent_weight(spec.src, spec)
            if weight is not None:
                self._parent_weight_cache[spec.src] = weight

    def _maybe_topk(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if scores.numel() == 0:
            return scores, scores
        if self.pruning_cfg.per_child_topk_edges and self.pruning_cfg.per_child_topk_edges > 0:
            k = min(int(self.pruning_cfg.per_child_topk_edges), scores.shape[0])
            topk_vals, topk_idx = torch.topk(scores, k=k, largest=True)
        else:
            topk_vals, topk_idx = scores, torch.arange(scores.shape[0], device=scores.device)
        return topk_vals, topk_idx

    def _prune_layer_nodes(
        self,
        layer: LayerId,
        scores: Dict[int, float],
        terminal_feats: Set[int],
    ) -> tuple[Dict[int, float], Set[int]]:
        if not scores:
            return {}, set()
        items = list(scores.items())
        items.sort(key=lambda x: x[1], reverse=True)
        limit = (
            int(self.pruning_cfg.max_nodes_per_layer)
            if self.pruning_cfg.max_nodes_per_layer and self.pruning_cfg.max_nodes_per_layer > 0
            else len(items)
        )
        kept = set(terminal_feats)
        for idx, score in items[:limit]:
            kept.add(idx)
        pruned_scores = {idx: val for idx, val in items[:limit] if idx not in terminal_feats}
        return pruned_scores, kept

    def build(
        self,
        direction: str,
        root_layer: LayerId,
        root_feature_indices: List[int],
    ) -> FeatureTree:
        """
        direction: "backward" or "forward"
        root_layer: ex) "blocks.10"
        root_feature_indices: root layer에서 시작할 feature index 리스트
        """
        self._reset(root_layer)
        self._current_root_features = list(root_feature_indices)
        self._precompute_parent_weights(direction)

        layer_order = self.topology.topological_layers(direction, [root_layer])
        layer_depth = {layer: idx for idx, layer in enumerate(layer_order)}

        next_node_id = 0
        next_edge_id = 0

        nodes: List[FeatureNode] = []
        edges: List[FeatureEdge] = []
        node_id_map: Dict[tuple[LayerId, int], int] = {}

        def _ensure_node(layer: LayerId, feat_idx: int, depth_val: int, score_val: float) -> int:
            nonlocal next_node_id
            key = (layer, feat_idx)
            if key in node_id_map:
                node = nodes[node_id_map[key]]
                node.score = score_val if score_val is not None else node.score
                if score_val is not None:
                    node.metadata["score"] = score_val
                return node_id_map[key]

            node_meta = {
                "layer": layer,
                "feature_idx": feat_idx,
                "score": score_val,
                "is_root": depth_val == 0,
            }
            spec = self.topology.nodes.get(layer)
            if spec is not None:
                node_meta["module"] = spec.module
                if spec.raw:
                    node_meta["node_spec"] = spec.raw

            node = FeatureNode(
                node_id=next_node_id,
                key=FeatureNodeId(layer=layer, feature_idx=feat_idx),
                depth=depth_val,
                score=score_val,
                metadata=node_meta,
            )
            nodes.append(node)
            node_id_map[key] = next_node_id
            next_node_id += 1
            return node_id_map[key]

        layer_inputs: Dict[LayerId, Dict[int, float]] = {
            root_layer: {feat_idx: 1.0 for feat_idx in root_feature_indices}
        }
        terminal_features: Dict[LayerId, Set[int]] = {}
        incoming_edges: Dict[LayerId, List[dict]] = {}

        # materialise root nodes
        for feat_idx, score in layer_inputs[root_layer].items():
            _ensure_node(root_layer, feat_idx, 0, score)

        for layer in layer_order:
            feats = layer_inputs.get(layer, {})
            if not feats:
                continue
            terminals_here = terminal_features.get(layer, set())

            pruned_scores, kept = self._prune_layer_nodes(layer, feats, terminals_here)
            depth_val = layer_depth.get(layer, 0)
            for feat_idx in kept:
                score_val = feats.get(feat_idx, 0.0)
                _ensure_node(layer, feat_idx, depth_val, score_val)

            # instantiate edges that point into this layer
            for cand in incoming_edges.get(layer, []):
                if cand["dst_feat_idx"] not in kept:
                    continue
                parent_node_id = node_id_map.get((cand["src_layer"], cand["src_feat_idx"]))
                child_node_id = node_id_map.get((cand["dst_layer"], cand["dst_feat_idx"]))
                if parent_node_id is None or child_node_id is None:
                    continue
                sign = 0
                if cand["score_signed"] > 0:
                    sign = 1
                elif cand["score_signed"] < 0:
                    sign = -1

                edge = FeatureEdge(
                    edge_id=next_edge_id,
                    parent_node_id=parent_node_id,
                    child_node_id=child_node_id,
                    score=cand["score"],
                    mean_attr=cand["score_signed"],
                    sign=sign,
                    metadata={
                        "edge_id": cand["edge_id"],
                        "direction": direction,
                        "src_layer": cand["src_layer"],
                        "dst_layer": cand["dst_layer"],
                        "dst_feature_idx": cand["dst_feat_idx"],
                        "backend": cand.get("backend"),
                        "is_terminal": cand.get("is_terminal", False),
                        "score_signed": cand["score_signed"],
                        "score_abs": cand["score_abs"],
                    },
                )
                edges.append(edge)
                next_edge_id += 1

            # stop expansion for terminal-only layers
            propagating_feats = {
                idx: score for idx, score in pruned_scores.items() if idx not in terminals_here
            }
            if not propagating_feats:
                continue

            for edge_spec in self.topology.out_edges(direction, layer):
                parent_weight = self._prepare_parent_weight(layer, edge_spec)
                for src_feat_idx, src_score in propagating_feats.items():
                    child_feat_tensor = torch.tensor([src_feat_idx], dtype=torch.long)
                    weight_vec = None
                    if parent_weight is not None:
                        weight_vec = torch.zeros_like(parent_weight)
                        if src_feat_idx < parent_weight.shape[-1]:
                            weight_vec[..., src_feat_idx] = parent_weight[..., src_feat_idx]
                    attr, err_attr = self.runtime.compute_edge_attribution_split(
                        edge_spec=edge_spec,
                        target_layer=layer,
                        target_feature_indices=child_feat_tensor,
                        source_layer=edge_spec.dst,
                        weight_vector=weight_vec,
                        backend=edge_spec.backend or self.attr_cfg.edge_backend,
                    )
                    dst_scores = layer_inputs.setdefault(edge_spec.dst, {})
                    dst_edge_list = incoming_edges.setdefault(edge_spec.dst, [])
                    dst_terminals = terminal_features.setdefault(edge_spec.dst, set())

                    is_error_edge = edge_spec.is_terminal or any(
                        "#error_coeff" in (m or "") for m in (edge_spec.anchor_modules or [])
                    )
                    err_added = False

                    if is_error_edge:
                        e = err_attr if err_attr is not None else attr
                        if e is None:
                            continue
                        if e.dim() > 3:
                            e = e.view(e.shape[0], -1, e.shape[-1])
                        if e.shape[-1] != 1:
                            e = e.mean(dim=-1, keepdim=True)
                        err_scores, err_signed = edge_attr_to_scores(e, self.pruning_cfg)
                        if err_scores.numel() > 0:
                            err_feat_idx = -1  # reserved error coeff slot
                            err_score_val = float(err_scores.view(-1)[0].item())
                            dst_scores[err_feat_idx] = dst_scores.get(err_feat_idx, 0.0) + err_score_val
                            dst_terminals.add(err_feat_idx)
                            dst_edge_list.append(
                                {
                                    "edge_id": edge_spec.edge_id,
                                    "src_layer": layer,
                                    "src_feat_idx": src_feat_idx,
                                    "dst_layer": edge_spec.dst,
                                    "dst_feat_idx": err_feat_idx,
                                    "score_signed": float(err_signed.view(-1)[0].item()),
                                    "score_abs": abs(float(err_signed.view(-1)[0].item())),
                                    "score": err_score_val,
                                    "backend": edge_spec.backend,
                                    "is_terminal": True,
                                    "dst_feature_label": "#error_coeff",
                                }
                            )
                            err_added = True

                    if attr is None:
                        continue

                    if self.edge_weighting_cfg.mode != "target_conditioned" and src_score is not None:
                        attr = attr * float(src_score)

                    scores, signed_mean = edge_attr_to_scores(attr, self.pruning_cfg)
                    topk_vals, topk_idx = self._maybe_topk(scores)
                    if topk_vals.numel() > 0:
                        if self.pruning_cfg.positive_only:
                            mask = topk_vals > float(self.pruning_cfg.per_child_edge_threshold or 0.0)
                        else:
                            mask = topk_vals.abs() > float(self.pruning_cfg.per_child_edge_threshold or 0.0)
                        topk_vals = topk_vals[mask]
                        topk_idx = topk_idx[mask]
                    if topk_vals.numel() == 0:
                        continue

                    # error coeff handling (always terminal, not propagated further)
                    if err_attr is not None and not err_added:
                        e = err_attr
                        if e.dim() > 3:
                            e = e.view(e.shape[0], -1, e.shape[-1])
                        if e.shape[-1] != 1:
                            e = e.mean(dim=-1, keepdim=True)
                        err_scores, err_signed = edge_attr_to_scores(e, self.pruning_cfg)
                        if err_scores.numel() > 0:
                            err_feat_idx = -1  # reserved error coeff slot
                            err_score_val = float(err_scores.view(-1)[0].item())
                            dst_scores[err_feat_idx] = dst_scores.get(err_feat_idx, 0.0) + err_score_val
                            dst_terminals.add(err_feat_idx)
                            dst_edge_list.append(
                                {
                                    "edge_id": edge_spec.edge_id,
                                    "src_layer": layer,
                                    "src_feat_idx": src_feat_idx,
                                    "dst_layer": edge_spec.dst,
                                    "dst_feat_idx": err_feat_idx,
                                    "score_signed": float(err_signed.view(-1)[0].item()),
                                    "score_abs": abs(float(err_signed.view(-1)[0].item())),
                                    "score": err_score_val,
                                    "backend": edge_spec.backend,
                                    "is_terminal": True,
                                    "dst_feature_label": "#error_coeff",
                                }
                            )

                    for feat_idx_tensor, score_tensor in zip(topk_idx, topk_vals):
                        feat_idx_int = int(feat_idx_tensor.item())
                        signed_val = float(signed_mean[feat_idx_int].item())
                        score_val = float(score_tensor.item())
                        if score_val == 0 and abs(signed_val) == 0:
                            continue

                        dst_scores[feat_idx_int] = dst_scores.get(feat_idx_int, 0.0) + score_val
                        if edge_spec.is_terminal:
                            dst_terminals.add(feat_idx_int)

                        dst_edge_list.append(
                            {
                                "edge_id": edge_spec.edge_id,
                                "src_layer": layer,
                                "src_feat_idx": src_feat_idx,
                                "dst_layer": edge_spec.dst,
                                "dst_feat_idx": feat_idx_int,
                                "score_signed": signed_val,
                                "score_abs": abs(signed_val),
                                "score": score_val,
                                "backend": edge_spec.backend,
                                "is_terminal": edge_spec.is_terminal,
                            }
                        )

        tree = FeatureTree(
            direction=direction,
            root_layer=root_layer,
            nodes=nodes,
            edges=edges,
            metadata={
                "root_features": list(root_feature_indices),
                "layer_to_module": {layer: spec.module for layer, spec in self.topology.nodes.items()},
            },
        )
        tree.build_indices()
        return tree
