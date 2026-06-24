from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from .feature_types import FeatureTree
from .tree_config import SankeyConfig
from src.core.indexing.registry_utils import sanitize_layer_name

logger = logging.getLogger(__name__)
_viz_autogen_failures: set[str] = set()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compute_layer_depth(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    depth_map: Dict[str, int] = {}
    for n in nodes:
        layer = n.get("layer")
        if layer is None:
            continue
        try:
            depth_val = int(n.get("depth", 0))
        except Exception:
            depth_val = 0
        if layer not in depth_map or depth_val < depth_map[layer]:
            depth_map[layer] = depth_val
    return depth_map


def _normalize_link_values(raw_values: List[float]) -> List[float]:
    """
    Normalize edge widths for Sankey rendering.
    Uses linear scaling with a small floor to preserve relative differences.
    """
    if not raw_values:
        return []
    vals = [max(0.0, v) for v in raw_values]
    vmin = min(vals)
    vmax = max(vals)
    min_width = 0.6
    max_width = 8.0
    if vmax - vmin < 1e-8:
        return [min_width for _ in vals]
    return [
        min_width + (max_width - min_width) * (v - vmin) / (vmax - vmin)
        for v in vals
    ]


def _load_unit_manifest(
    viz_root: Optional[Path],
    layer: str,
    feature_idx: int,
    cache: Dict[tuple[str, int], Optional[dict]],
    aliases: Optional[List[str]] = None,
) -> Optional[dict]:
    """
    Try to load a precomputed visualization manifest (thumbnail/panels) for the node.
    """
    if viz_root is None or feature_idx is None or feature_idx < 0:
        return None
    candidates = [layer]
    if aliases:
        for alias in aliases:
            if alias and alias not in candidates:
                candidates.append(alias)
    for cand in candidates:
        key = (cand, feature_idx)
        if key in cache:
            if cache[key] is not None:
                # populate primary cache entry for base layer if we hit an alias
                if cand != layer:
                    cache[(layer, feature_idx)] = cache[key]
                return cache[key]
            continue
        found = False
        for manifest_path in _manifest_candidates(viz_root, cand, feature_idx):
            if not manifest_path.exists():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            cache[key] = payload
            cache[(layer, feature_idx)] = payload
            return payload
        cache[key] = None
    return None


def _layout_nodes(
    nodes: List[Dict[str, Any]],
    direction: str,
    layer_depth: Dict[str, int],
) -> Tuple[List[float], List[float], Dict[int, int], Dict[str, int]]:
    """
    Compute fixed x/y coordinates for nodes so that:
      - layers share x positions
      - nodes within a layer are spaced vertically by feature_idx order
      - backward renders to the left of the root, forward to the right
    """
    per_layer: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        per_layer.setdefault(n.get("layer"), []).append(n)
    for layer_nodes in per_layer.values():
        layer_nodes.sort(key=lambda n: (n.get("feature_idx") == -1, n.get("feature_idx", 0)))

    # Stable per-layer columns: each layer gets its own column ordered by depth then name.
    unique_layers = sorted(
        (layer for layer in {n.get("layer") for n in nodes} if layer is not None),
        key=lambda layer: (layer_depth.get(layer, 0), layer),
    )
    layer_to_column: Dict[str, int] = {layer: idx for idx, layer in enumerate(unique_layers)}
    max_depth = len(unique_layers) - 1 if unique_layers else 0
    span = 0.86
    margin = (1.0 - span) / 2.0
    step = span / max(1, max_depth if max_depth > 0 else 1)

    x_coords: List[float] = []
    y_coords: List[float] = []
    column_by_node: Dict[int, int] = {}
    per_layer_count: Dict[str, int] = {}

    for n in nodes:
        layer = n.get("layer")
        depth_val = layer_depth.get(layer, int(n.get("depth", 0) or 0))
        column_idx = layer_to_column.get(layer, depth_val)
        per_layer_count[layer] = per_layer_count.get(layer, 0) + 1
        node_id = n.get("id")
        if node_id is not None:
            column_by_node[int(node_id)] = column_idx
        meta_dir = (n.get("meta") or {}).get("direction") or direction
        if max_depth == 0:
            x = 0.5
        elif meta_dir == "forward":
            x = margin + column_idx * step
        else:
            x = 1.0 - margin - column_idx * step
        layer_nodes = per_layer.get(layer, [])
        if len(layer_nodes) <= 1:
            y = 0.5
        else:
            idx = layer_nodes.index(n)
            y = (idx + 1) / (len(layer_nodes) + 1)
        if n.get("feature_idx") == -1:
            # Error nodes to the bottom of the column.
            y = 0.95
        x_coords.append(min(0.98, max(0.02, x)))
        y_coords.append(min(0.98, max(0.02, y)))
    return x_coords, y_coords, column_by_node, per_layer_count


def _estimate_layout_size(column_by_node: Dict[int, int], per_layer_count: Dict[str, int]) -> Tuple[int, int]:
    num_columns = len(set(column_by_node.values())) or 1
    max_height = max(per_layer_count.values()) if per_layer_count else 1
    figure_width = max(960, int(num_columns * 260))
    figure_height = max(640, int(max_height * 140))
    return figure_width, figure_height


def _manifest_path(root: Path, layer: str, unit: int) -> Path:
    layer_sanitised = layer.replace("/", "_")
    return root / layer_sanitised / f"unit_{unit}" / "manifest.json"


def _manifest_candidates(root: Path, layer: str, unit: int) -> List[Path]:
    names: List[str] = []
    for name in (layer.replace("/", "_"), sanitize_layer_name(layer), layer):
        if name not in names:
            names.append(name)
    return [root / name / f"unit_{unit}" / "manifest.json" for name in names]


def _maybe_autogenerate_manifests(
    tree: FeatureTree,
    viz_root: Optional[Path],
    auto_viz_config: Optional[Path],
) -> None:
    if viz_root is None or auto_viz_config is None:
        return
    viz_root.mkdir(parents=True, exist_ok=True)
    logger.info("[viz] auto-generation enabled: root=%s, config=%s", viz_root, auto_viz_config)
    layer_to_module = {}
    try:
        layer_to_module = (getattr(tree, "metadata", {}) or {}).get("layer_to_module", {}) or {}
    except Exception:
        layer_to_module = {}

    missing: Dict[str, Dict[str, Any]] = {}
    for n in tree.nodes:
        feat = n.key.feature_idx
        if feat is None or feat < 0:
            continue
        target_module = layer_to_module.get(n.key.layer)
        node_meta = getattr(n, "metadata", {}) or {}
        if not target_module:
            target_module = node_meta.get("module")
        existing_here = any(path.exists() for path in _manifest_candidates(viz_root, n.key.layer, feat))
        existing_alias = target_module and any(
            path.exists() for path in _manifest_candidates(viz_root, target_module, feat)
        )
        if existing_here or existing_alias:
            continue
        entry = missing.setdefault(n.key.layer, {"units": [], "target": target_module})
        if target_module and not entry.get("target"):
            entry["target"] = target_module
        entry["units"].append(int(feat))
    if not missing:
        logger.info("[viz] all manifests present; skipping generation")
        return
    for layer, info in missing.items():
        units = sorted(set(info.get("units", [])))
        logger.info("[viz] missing manifest -> layer=%s units=%s", layer, units)
    try:
        from src.core.attribution.viz_export import export_viz_for_units
    except Exception as exc:
        logger.warning("[viz] failed to import export_viz_for_units: %s", exc)
        return
    for layer, info in missing.items():
        units = sorted(set(info.get("units", [])))
        target_layer = info.get("target") or layer
        if target_layer in _viz_autogen_failures:
            logger.info(
                "[viz] skipping layer %s (target=%s) due to previous failure",
                layer,
                target_layer,
            )
            continue
        try:
            logger.info("[viz] exporting %d unit(s) for layer %s (target=%s)", len(units), layer, target_layer)
            manifests = export_viz_for_units(
                config_path=auto_viz_config,
                layer=target_layer,
                units=units,
                cache_root=viz_root,
            )
            logger.info("[viz] done exporting layer %s", layer)
        except Exception as exc:
            logger.warning("[viz] export failed for layer %s (units=%s): %s", layer, units, exc)
            _viz_autogen_failures.add(target_layer)
            continue


def feature_tree_to_sankey(
    tree: FeatureTree,
    sankey_cfg: SankeyConfig,
    viz_manifest_root: Optional[str | Path] = None,
    auto_viz_config: Optional[str | Path] = None,
    auto_viz_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    FeatureTree -> sankey용 dict
    nodes: [{id, name, layer, feature_idx, depth, score, meta}]
    links: [{source, target, value, sign, mean_attr, edge_id, direction, normalized_value}]
    """
    viz_root = Path(viz_manifest_root) if viz_manifest_root else None
    auto_cfg = Path(auto_viz_config) if auto_viz_config else None
    if auto_viz_root is not None:
        viz_root = Path(auto_viz_root)
    if auto_cfg and viz_root:
        _maybe_autogenerate_manifests(tree, viz_root, auto_cfg)
    manifest_cache: Dict[tuple[str, int], Optional[dict]] = {}
    label_override: Dict[int, str] = {}
    layer_to_module = {}
    try:
        layer_to_module = (getattr(tree, "metadata", {}) or {}).get("layer_to_module", {}) or {}
    except Exception:
        layer_to_module = {}
    for e in tree.edges:
        lbl = e.metadata.get("dst_feature_label") or e.metadata.get("feature_label")
        if lbl:
            label_override[e.child_node_id] = lbl

    nodes_json = []
    layer_depth: Dict[str, int] = {}
    for n in tree.nodes:
        depth_val = int(getattr(n, "depth", 0) or 0)
        layer_depth[n.key.layer] = min(layer_depth.get(n.key.layer, depth_val), depth_val)
        label = label_override.get(n.node_id)
        name = f"{n.key.layer}:{label if label is not None else n.key.feature_idx}"
        node_meta = dict(getattr(n, "metadata", {}) or {})
        node_meta.setdefault("layer", n.key.layer)
        node_meta.setdefault("feature_idx", n.key.feature_idx)
        node_meta.setdefault("score", n.score)
        node_meta.setdefault("direction", tree.direction)
        if n.key.feature_idx == -1:
            node_meta["is_error_node"] = True
        module_alias = node_meta.get("module") or layer_to_module.get(n.key.layer)
        manifest = _load_unit_manifest(
            viz_root,
            n.key.layer,
            n.key.feature_idx,
            manifest_cache,
            aliases=[module_alias] if module_alias else None,
        )
        if (
            manifest
            and "viz" not in node_meta
            and n.key.feature_idx != -1
            and n.key.layer != "imagenet_logits"
        ):
            node_meta["viz"] = manifest
        nodes_json.append(
            {
                "id": n.node_id,
                "name": name,
                "layer": n.key.layer,
                "feature_idx": n.key.feature_idx,
                "depth": depth_val,
                "score": n.score,
                "meta": node_meta,
            }
        )

    # layer-wise score stats for normalization
    layer_score_max: Dict[str, float] = {}
    layer_score_sum: Dict[str, float] = {}
    for n in nodes_json:
        try:
            s = float(n.get("score", 0.0) or 0.0)
        except Exception:
            s = 0.0
        layer = n.get("layer")
        if layer is None:
            continue
        layer_score_max[layer] = max(layer_score_max.get(layer, 0.0), s)
        layer_score_sum[layer] = layer_score_sum.get(layer, 0.0) + abs(s)
    for n in nodes_json:
        layer = n.get("layer")
        try:
            s = float(n.get("score", 0.0) or 0.0)
        except Exception:
            s = 0.0
        denom = layer_score_max.get(layer, 0.0)
        norm = s / denom if denom > 0 else 0.0
        n.setdefault("meta", {})
        n["meta"]["normalized_score"] = norm

    node_map = {n["id"]: n for n in nodes_json}
    links_json = []
    for e in tree.edges:
        signed_val = e.metadata.get("score_signed", e.mean_attr)
        abs_val = e.metadata.get("score_abs", abs(signed_val))
        sign = e.sign
        if sign == 0:
            if signed_val > 0:
                sign = 1
            elif signed_val < 0:
                sign = -1

        if sankey_cfg.score_mode == "sign":
            value = signed_val
        elif sankey_cfg.score_mode == "magnitude":
            value = abs_val
        else:
            raise ValueError(f"unknown sankey score_mode: {sankey_cfg.score_mode}")

        # Normalize link value by target layer's sum of absolute scores
        child_node = node_map.get(e.child_node_id)
        if child_node:
            child_layer = child_node.get("layer")
            if child_layer:
                denom = layer_score_sum.get(child_layer, 1.0)
                if denom > 0:
                    value /= denom

        if sankey_cfg.positive_only and value <= 0:
            continue

        link_meta = dict(e.metadata or {})
        link_meta.setdefault("score_signed", float(signed_val))
        link_meta.setdefault("score_abs", float(abs_val))
        link_meta.setdefault("direction", link_meta.get("direction", tree.direction))
        links_json.append(
            {
                "source": e.parent_node_id,
                "target": e.child_node_id,
                "value": float(value),
                "value_abs": float(abs_val),
                "mean_attr": float(signed_val),
                "sign": sign,
                "edge_id": link_meta.get("edge_id"),
                "direction": link_meta.get("direction"),
                "backend": link_meta.get("backend"),
                "is_terminal": bool(link_meta.get("is_terminal", False)),
                "score_abs": float(abs_val),
                "src_layer": link_meta.get("src_layer"),
                "dst_layer": link_meta.get("dst_layer"),
                "meta": link_meta,
            }
        )

    widths = _normalize_link_values([abs(l["value"]) for l in links_json])
    for link, width in zip(links_json, widths):
        link["normalized_value"] = width

    tree_meta = dict(getattr(tree, "metadata", {}) or {})
    if viz_root:
        try:
            viz_root_resolved = viz_root.resolve()
        except Exception:
            viz_root_resolved = viz_root
        tree_meta.setdefault("viz_manifest_root", str(viz_root_resolved))
        tree_meta.setdefault("viz_manifest_root_raw", str(viz_root))
    tree_meta.update(
        {
            "direction": tree.direction,
            "root_layer": tree.root_layer,
            "layer_depth": layer_depth,
            "sankey_score_mode": sankey_cfg.score_mode,
            "positive_only": sankey_cfg.positive_only,
            "layer_score_max": layer_score_max,
        }
    )

    return {"nodes": nodes_json, "links": links_json, "meta": tree_meta}


def sankey_to_d3_html(tree_json: Dict[str, Any], output_path: str, direction: str) -> None:
    """
    Render a Sankey diagram to HTML using d3 + d3-sankey with thumbnails and detail panel.
    """
    nodes = tree_json.get("nodes", []) or []
    links = tree_json.get("links", []) or []
    meta = tree_json.get("meta", {}) or {}
    direction = meta.get("direction", direction)

    layer_depth = meta.get("layer_depth") or _compute_layer_depth(nodes)
    _, _, column_by_node, per_layer_count = _layout_nodes(nodes, direction, layer_depth)
    fig_width, fig_height = _estimate_layout_size(column_by_node, per_layer_count)
    # [Modify] Calculate relative root path (e.g., "../../") based on output_path depth
    out_path_obj = Path(output_path)
    # Assuming script is run from project root, count parts in the parent directory
    num_parents = len(out_path_obj.parent.parts)
    relative_root = "../" * num_parents if num_parents > 0 else "./"

    payload = json.dumps(
        {
            "nodes": nodes,
            "links": links,
            "meta": meta,
            "layout": {"width": fig_width, "height": fig_height},
        }
    )

    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Sankey ({direction})</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3-sankey@0.12.3/dist/d3-sankey.min.js"></script>
  <style>
    :root {{
      --label-offset: 6px;
    }}
    * {{ box-sizing: border-box; font-family: "Inter", "Segoe UI", sans-serif; }}
    body {{
      margin: 0;
      font-family: "Inter", "Segoe UI", sans-serif;
      background-color: #0f1115;
      color: #f5f5f5;
    }}
    .layout {{
      display: flex;
      flex-direction: row;
      gap: 14px;
      padding: 12px;
      height: 100vh;
    }}
    #sankey {{
      flex: 2;
      min-width: 640px;
      background-color: #15181f;
      border-radius: 12px;
      padding: 10px;
      position: relative;
      overflow: auto;
      cursor: grab;
    }}
    #sankey.drag-active {{ cursor: grabbing; }}
    #sankey-plot {{ position: relative; min-height: 520px; z-index: 2; }}
    #sankey svg {{ width: 100%; height: 100%; overflow: visible; }}
    #detail-panel {{
      flex: 1;
      background-color: #1a1e26;
      border-radius: 12px;
      padding: 16px;
      overflow-y: auto;
      max-height: 100%;
    }}
    #detail-panel h2 {{ margin-top: 0; }}
    .detail-section {{ margin-top: 14px; }}
    .detail-section h3 {{ margin-bottom: 6px; }}
    .detail-placeholder {{ color: #9aa0ad; font-style: italic; }}
    .panel-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 8px;
    }}
    .panel-grid img {{
      width: 100%;
      border-radius: 8px;
      object-fit: cover;
      border: 1px solid rgba(255,255,255,0.1);
    }}
    .node-label {{
      font-size: 11px;
      fill: #e5e7eb;
      alignment-baseline: middle;
      text-shadow: 0 1px 3px rgba(0,0,0,0.6);
      cursor: pointer;
    }}
    .col-header {{ font-size: 14px; font-weight: 600; fill: #9ca3af; text-anchor: middle; }}
    .node-text {{ font-size: 11px; fill: #d1d5db; pointer-events: none; }}
  </style>
</head>
<body>
  <div class="layout">
    <div id="sankey">
      <div id="sankey-plot"></div>
    </div>
    <div id="detail-panel">
      <h2>Details</h2>
      <p>노드 또는 엣지를 클릭하면 시각화 정보가 여기에 표시됩니다.</p>
    </div>
  </div>
  <script>
    (function() {{
      const data = {payload};
      const nodes = data.nodes || [];
      const links = data.links || [];
      const meta = data.meta || {{}}; 
      const layoutCfg = data.layout || {{}}; 
      const layerScoreMax = (data.meta && data.meta.layer_score_max) || {{}}; 
      const manifestRoot = (meta && meta.viz_manifest_root) || "";
      const relativeRoot = "{relative_root}"; // Injected from Python
      const sankeyContainer = document.getElementById("sankey"); 
      let activeNodeIndex = null; 
      let currentZoom = 1.0; 
      const MIN_ZOOM = 0.6; 
      const MAX_ZOOM = 2.2; 
      const svgWidth = layoutCfg.width || 1200;
      const svgHeight = layoutCfg.height || 800;

      function clamp(value, min, max) {{
        if (!Number.isFinite(value)) return min;
        return Math.min(Math.max(value, min), max);
      }}
      function escapeHtml(value) {{
        if (value === undefined || value === null) return "";
        return String(value)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }}
      function formatNumber(value) {{
        if (value === undefined || value === null || Number.isNaN(Number(value))) return "n/a";
        return Number(value).toFixed(4);
      }}
      function isAbsoluteAssetPath(p) {{
        if (typeof p !== "string") return false;
        return /^([a-z]+:)?\\/\\//i.test(p) || p.startsWith("data:") || p.startsWith("file:") || p.startsWith("/") || /^[A-Za-z]:\\\\/.test(p);
      }}
      function resolveAssetPath(p) {{
        if (!p || typeof p !== "string") return "";
        const trimmed = p.trim();
        if (!trimmed) return "";
        if (isAbsoluteAssetPath(trimmed)) return trimmed;
        // [Modify] Use calculated relative path instead of double-concatenating manifestRoot
        // Strip leading ./ or / to ensure clean concatenation
        const cleanPath = trimmed.replace(/^\\.\\//, "").replace(/^\\//, "");
        // [Fix] Replace '#' with '%23' because browsers treat '#' as an anchor, cutting off the path
        return (relativeRoot + cleanPath).replace(/#/g, "%23");
      }}
      function render() {{
        const svg = d3.select("#sankey-plot").append("svg")
          .attr("width", svgWidth)
          .attr("height", svgHeight)
          .attr("viewBox", `0 0 ${{svgWidth}} ${{svgHeight}}`);

        const g = svg.append("g");

        const zoom = d3.zoom()
          .scaleExtent([MIN_ZOOM, MAX_ZOOM])
          .on("zoom", (event) => {{
            currentZoom = event.transform.k;
            g.attr("transform", event.transform);
          }});
        svg.call(zoom);

        const nodeById = new Map();
        nodes.forEach((n, idx) => nodeById.set(n.id, idx));

        const sankeyNodes = nodes.map((n) => Object.assign({{}}, n));
        const sankeyLinks = links.map((l) => Object.assign({{}}, l, {{
          source: nodeById.get(l.source),
          target: nodeById.get(l.target),
        }})).filter((l) => l.source !== undefined && l.target !== undefined);

        const columns = Array.from(
          new Set(
            nodes.map((n) => {{
              if (n.meta && typeof n.meta.sankey_column === "number") return n.meta.sankey_column;
              const depthVal = Number.isFinite(n.depth) ? Number(n.depth) : 0;
              return depthVal;
            }})
          )
        );
        const maxColumn = columns.length ? Math.max(...columns) : 0;

        const sankey = d3.sankey()
          .nodeWidth(18)
          .nodePadding(26)
          .extent([[50, 40], [svgWidth - 50, svgHeight - 40]])
          .nodeAlign((node, n) => {{
            const col = node.meta && typeof node.meta.sankey_column === "number"
              ? node.meta.sankey_column
              : (Number.isFinite(node.depth) ? Number(node.depth) : 0);
            return Math.max(0, Math.min(maxColumn, col));
          }});

        
        // [Layout Strategy: Two-Pass]
        // Pass 1: Run layout WITHOUT nodeSort to let D3 optimize/relax edges (minimize crossings)
        sankey({{ nodes: sankeyNodes, links: sankeyLinks }});
        
        // Pass 2: Apply a strict sort that preserves Pass 1's optimization BUT forces error nodes to bottom
        sankey.nodeSort((a, b) => {{
            // 1. Force Error nodes (-1) to the bottom
            if (a.feature_idx === -1 && b.feature_idx !== -1) return 1;
            if (a.feature_idx !== -1 && b.feature_idx === -1) return -1;
            // 2. For others, respect the y-position calculated in Pass 1
            return a.y0 - b.y0;
        }});
        
        // Re-calculate layout with the final sort order
        const graph = sankey({{ nodes: sankeyNodes, links: sankeyLinks }});

        console.info("[sankey] manifest root:", manifestRoot || "(none)");

        const colorForNode = (n) => {{
          // 1. Error nodes should always be gray (Check this FIRST)
          if (n.feature_idx === -1) return "rgba(127,127,127,0.9)";
          
          // 2. Root/Target node (Pink)
          // In 'merged' mode, depth 0 is the far left input, NOT the root. So we strictly use is_root flag.
          if (n.meta && n.meta.is_root) return "rgba(225,95,153,0.95)";
          if (meta.direction !== "merged" && n.depth === 0) return "rgba(225,95,153,0.95)";

          // 3. Directional color (Blue/Green)
          const dir = (n.meta && n.meta.direction) || meta.direction || "backward";
          return dir === "forward" ? "rgba(44,160,44,0.9)" : "rgba(46,145,229,0.9)";
        }};
        const colorForLink = (l) => {{
          const sign = l.sign ?? 1;
          const dir = l.direction || meta.direction || "backward";
          if (sign < 0) return "rgba(214,39,40,0.6)";
          return dir === "forward" ? "rgba(44,160,44,0.6)" : "rgba(46,145,229,0.6)";
        }};

        g.append("g")
          .attr("fill", "none")
          .selectAll("path")
          .data(graph.links)
          .enter()
          .append("path")
          .attr("class", "sankey-link")
          .attr("d", d3.sankeyLinkHorizontal())
          .attr("stroke", (d) => colorForLink(d))
          .attr("stroke-width", (d) => Math.max(1, d.width || 1))
          .attr("opacity", 0.6)
          .on("click", (event, d) => {{
            event.stopPropagation();
            activeNodeIndex = null;
            showEdgeDetail(d);
          }})
          .append("title")
          .text((d) => `${{d.source.name}} → ${{d.target.name}}\\nΔ=${{d.value?.toFixed ? d.value.toFixed(4) : d.value}}`);

        const nodeGroup = g.append("g")
          .selectAll("g")
          .data(graph.nodes)
          .enter()
          .append("g")
          .attr("class", "node-group-item")
          .on("click", (event, d) => {{
            event.stopPropagation();
            activeNodeIndex = d.index;
            showNodeDetail(d);
          }});

        nodeGroup.append("rect")
          .attr("x", (d) => d.x0)
          .attr("y", (d) => d.y0)
          .attr("height", (d) => Math.max(2, d.y1 - d.y0))
          .attr("width", (d) => d.x1 - d.x0)
          .attr("fill", (d) => colorForNode(d))
          .attr("stroke", "#222")
          .attr("stroke-width", 0.6)
          .attr("rx", 3)
          .attr("ry", 3);
        // [Label] 1. Node Labels (Unit ID / Error)
        nodeGroup.append("text")
          .attr("class", "node-text")
          .attr("x", (d) => {{
            // If merged view: Left side nodes -> text on left, Right side nodes -> text on right
            const isLeft = d.x0 < svgWidth / 2;
            return isLeft ? d.x0 - 6 : d.x1 + 6;
          }})
          .attr("y", (d) => (d.y1 + d.y0) / 2)
          .attr("dy", "0.35em")
          .attr("text-anchor", (d) => d.x0 < svgWidth / 2 ? "end" : "start")
          .text((d) => d.feature_idx === -1 ? "error node" : `unit ${{d.feature_idx}}`);

        // [Label] 2. Column Headers (Layer Names)
        // Robustly identify columns using the calculated sankey_column or depth
        const colData = new Map(); 
        
        graph.nodes.forEach((n) => {{
            const col = n.meta && typeof n.meta.sankey_column === 'number' 
                ? n.meta.sankey_column 
                : (Number.isFinite(n.depth) ? Number(n.depth) : 0);
            // f-string에서는 JS 객체 중괄호를 이스케이프해야 합니다.
            if (!colData.has(col)) colData.set(col, {{ sumX: 0, count: 0, layers: new Map() }});
            const info = colData.get(col);
            
            info.sumX += (n.x0 + n.x1) / 2;
            info.count += 1;
            const lname =
              n.layer ||
              (n.meta && n.meta.layer) ||
              (typeof n.name === "string" && n.name.includes(":") ? n.name.split(":")[0] : null) ||
              "Unknown";
            info.layers.set(lname, (info.layers.get(lname) || 0) + 1);
        }});

        const headerData = [];
        for (const [col, info] of colData) {{
            const avgX = info.sumX / info.count;
            // Find the most frequent layer name in this column
            const layerEntries = [...info.layers.entries()];
            const bestLayer = layerEntries.length
              ? layerEntries.reduce((a, b) => (b[1] > a[1] ? b : a))[0]
              : "Unknown";
            // 헤더 객체를 만들 때도 동일하게 이스케이프합니다.
            headerData.push({{ x: avgX, label: bestLayer }});
        }}
        
        g.append("g")
          .selectAll("text")
          .data(headerData)
          .enter()
          .append("text")
          .attr("class", "col-header")
          .attr("x", d => d.x)
          .attr("y", 30)
          .text(d => d.label);

        function renderPanels(items, title) {{
          if (!items || !items.length) return "";
          let block = `<div class="detail-section"><h3>${{escapeHtml(title)}}</h3><div class="panel-grid">`;
          items.forEach((panelInfo, panelIdx) => {{
            if (!panelInfo) return;
            const resolvedPath = resolveAssetPath(panelInfo.path);
            if (!resolvedPath) return;
            const captionParts = [];
            if (panelInfo.caption) captionParts.push(escapeHtml(panelInfo.caption));
            if (panelInfo.sample_id !== undefined) captionParts.push(`sample ${{escapeHtml(panelInfo.sample_id)}}`);
            if (panelInfo.score !== undefined) captionParts.push(`score ${{formatNumber(panelInfo.score)}}`);
            if (panelInfo.source) captionParts.push(escapeHtml(panelInfo.source));
            if (!captionParts.length) captionParts.push(`panel ${{panelIdx + 1}}`);
            block += `<figure><img src="${{escapeHtml(resolvedPath)}}" alt="activation" /><figcaption>${{captionParts.join(" · ")}}</figcaption></figure>`;
          }});
          block += `</div></div>`;
          return block;
        }}

        function showNodeDetail(d) {{
          const panel = document.getElementById("detail-panel");
          const meta = d.meta || {{}};
          const label = d.name || `Node ${{d.id}}`;
          let html = `<h2>${{escapeHtml(label)}}</h2>`;
          if (d.layer) html += `<p><strong>Layer:</strong> ${{escapeHtml(d.layer)}}</p>`;
          if (d.feature_idx !== undefined) html += `<p><strong>Feature idx:</strong> ${{escapeHtml(d.feature_idx)}}</p>`;
          if (d.score !== undefined) html += `<p><strong>Node score:</strong> ${{formatNumber(d.score)}}</p>`;
          if (meta.module) html += `<p><strong>Module:</strong> ${{escapeHtml(meta.module)}}</p>`;
          const viz = meta.viz && typeof meta.viz === "object" ? meta.viz : null;
          const inputViz = meta.input_viz || meta.inputViz || null;
          const panels = Array.isArray(viz && viz.panels) ? viz.panels : [];
          const decilePanels = panels.filter((p) => {{
            if (!p || typeof p !== "object") return false;
            const src = (p.source || "").toString().toLowerCase();
            return src.includes("decile") || src.includes("ledger");
          }});
          const imagePanels = panels.filter((p) => !decilePanels.includes(p));
          html += renderPanels(decilePanels, "Decile top activations");
          html += renderPanels(imagePanels, "Image activations");
          const featureMapRaw = (inputViz && inputViz.feature_map)
            || meta.input_feature_map
            || (viz && viz.input_feature_map)
            || (panels.find((p) => p && p.input_feature_map) || {{}}).input_feature_map
            || null;
          const attrMapRaw = (inputViz && inputViz.attr_map) || meta.input_attr_map || null;
          const featureMapAsset = resolveAssetPath(featureMapRaw);
          const attrMapAsset = resolveAssetPath(attrMapRaw);
          html += `<div class="detail-section"><h3>Input feature map</h3>`;
          const inputBlocks = [];
          if (featureMapAsset) {{
            inputBlocks.push(`<figure><img src="${{escapeHtml(featureMapAsset)}}" alt="feature map" /><figcaption>Activation map</figcaption></figure>`);
          }}
          if (attrMapAsset) {{
            inputBlocks.push(`<figure><img src="${{escapeHtml(attrMapAsset)}}" alt="attribution map" /><figcaption>Attribution map</figcaption></figure>`);
          }}
          if (inputBlocks.length) {{
            html += `<div class="panel-grid">${{inputBlocks.join("")}}</div>`;
          }} else {{
            html += `<p class="detail-placeholder">Feature map unavailable for this node.</p>`;
          }}
          html += `</div>`;
          const debugPanelEntries = panels.map((p, i) => {{
            const raw = p && p.path ? p.path : "";
            const resolved = resolveAssetPath(raw);
            console.info("[sankey] panel asset", {{ layer: d.layer, feature_idx: d.feature_idx, panel: i, raw, resolved }});
            return `<li>panel ${{i + 1}}: src=${{escapeHtml(raw || "(missing)")}} → ${{escapeHtml(resolved || "(empty)")}}</li>`;
          }}).join("");
          console.info("[sankey] feature map asset", {{ layer: d.layer, feature_idx: d.feature_idx, raw: featureMapRaw, resolved: featureMapAsset }});
          console.info("[sankey] attr map asset", {{ layer: d.layer, feature_idx: d.feature_idx, raw: attrMapRaw, resolved: attrMapAsset }});
          const inputDebug = [
            `<li>feature map: ${{escapeHtml(featureMapRaw || "(missing)")}} → ${{escapeHtml(featureMapAsset || "(empty)")}}</li>`,
            `<li>attr map: ${{escapeHtml(attrMapRaw || "(missing)")}} → ${{escapeHtml(attrMapAsset || "(empty)")}}</li>`
          ].join("");
          html += `<div class="detail-section"><h3>Debug: assets</h3><p><strong>manifest root:</strong> ${{escapeHtml(manifestRoot || "(none)")}}</p><ul>${{debugPanelEntries || "<li>no panels</li>"}}${{inputDebug}}</ul></div>`;
          panel.innerHTML = html;

          // Highlighting
          const connectedLinks = graph.links.filter((l) => l.source.index === d.index || l.target.index === d.index);
          const connectedNodeIndices = new Set([d.index]);
          connectedLinks.forEach((l) => {{
            connectedNodeIndices.add(l.source.index);
            connectedNodeIndices.add(l.target.index);
          }});
          
          g.selectAll(".sankey-link").attr(
            "opacity",
            (l) => (connectedNodeIndices.has(l.source.index) && connectedNodeIndices.has(l.target.index) ? 0.6 : 0.05)
          );
          g.selectAll(".node-group-item").attr("opacity", (n) => (connectedNodeIndices.has(n.index) ? 1.0 : 0.2));
        }}

        function showEdgeDetail(l) {{
          const panel = document.getElementById("detail-panel");
          let html = `<h2>Edge detail</h2>`;
          const srcName = l.source && l.source.name ? l.source.name : l.meta?.source || "source";
          const tgtName = l.target && l.target.name ? l.target.name : l.meta?.target || "target";
          html += `<p><strong>From:</strong> ${{escapeHtml(srcName)}}</p>`;
          html += `<p><strong>To:</strong> ${{escapeHtml(tgtName)}}</p>`;
          if (l.value !== undefined) html += `<p><strong>Δ:</strong> ${{formatNumber(l.value)}}</p>`;
          if (l.score_abs !== undefined) html += `<p><strong>|Δ|:</strong> ${{formatNumber(l.score_abs)}}</p>`;
          if (l.mean_attr !== undefined) html += `<p><strong>mean_attr:</strong> ${{formatNumber(l.mean_attr)}}</p>`;
          if (l.backend) html += `<p><strong>backend:</strong> ${{escapeHtml(l.backend)}}</p>`;
          if (l.edge_id) html += `<p><strong>edge_id:</strong> ${{escapeHtml(l.edge_id)}}</p>`;
          const dir = l.direction || meta.direction;
          if (dir) html += `<p><strong>direction:</strong> ${{escapeHtml(dir)}}</p>`;
          panel.innerHTML = html;

          // Highlighting
          g.selectAll(".sankey-link").attr("opacity", (d) => (d.index === l.index ? 0.6 : 0.05));
          g.selectAll(".node-group-item").attr(
            "opacity",
            (n) => (n.index === l.source.index || n.index === l.target.index ? 1.0 : 0.2)
          );
        }}

        function resetHighlights() {{
          g.selectAll(".sankey-link").attr("opacity", 0.6);
          g.selectAll(".node-group-item").attr("opacity", 1.0);
          const panel = document.getElementById("detail-panel");
          panel.innerHTML = `<h2>Details</h2><p>Click a node or edge for details. Click background to reset.</p>`;
        }}
        
        function syncThumbnails() {{
            // This function is apparently needed, so we keep it.
        }}

        syncThumbnails();

        svg.on("click", (event) => {{
          if (event.target === svg.node()) {{
            resetHighlights();
          }}
        }});
        
        resetHighlights();
      }}

      render();
    }})();
  </script>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template, encoding="utf-8")


def sankey_merge_forward_backward(
    backward_json: Dict[str, Any],
    forward_json: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Render backward and forward trees on a single Sankey with the same rich UI
    (thumbnails, detail panel) as the per-direction pages.
      - backward nodes keep their depths; forward nodes are shifted to the right
      - node/link metadata is preserved for overlay/detail rendering
    """
    b_nodes = backward_json.get("nodes", []) or []
    f_nodes = forward_json.get("nodes", []) or []
    b_links = backward_json.get("links", []) or []
    f_links = forward_json.get("links", []) or []
    # Calculate max depth of backward tree to determine the "Center" column
    b_depths = [int(n.get("depth", 0)) for n in b_nodes]
    max_b_depth = max(b_depths) if b_depths else 0

    merged_nodes = []
    merged_layer_depth: Dict[str, int] = {}

    node_id_map_b = {}  # old_id -> new_id
    node_id_map_f = {}  # old_id -> new_id
    
    # 1. Add Backward Nodes (Left Side)
    # Depth mapping: d -> (max_b_depth - d)
    # Root (d=0) -> max_b_depth (Center)
    # Leaf (d=max) -> 0 (Left)
    for i, n in enumerate(b_nodes):
        new_node = n.copy()
        new_id = i
        node_id_map_b[n["id"]] = new_id
       
        d = int(n.get("depth", 0))
        col = max_b_depth - d
        
        new_node["id"] = new_id
        new_node["depth"] = col
        if "meta" not in new_node: new_node["meta"] = {}
        new_node["meta"]["sankey_column"] = col
        new_node["meta"]["original_id"] = n["id"]
        merged_nodes.append(new_node)
        
    # Identify Root in Backward Tree (Depth 0)
    b_root_node = next((n for n in b_nodes if int(n.get("depth", 0)) == 0), None)
    
    next_id = len(merged_nodes)
    
    # 2. Add Forward Nodes (Right Side)
    # Depth mapping: d -> (max_b_depth + d)
    for n in f_nodes:
        is_root = False
        # Match root by layer/feature_idx
        if b_root_node and n["layer"] == b_root_node["layer"] and n["feature_idx"] == b_root_node["feature_idx"]:
            is_root = True
            
        if is_root:
            # Map to existing backward root node
            target_id = node_id_map_b[b_root_node["id"]]
            # Force the central node to be recognized as root for coloring
            merged_nodes[target_id]["meta"]["is_root"] = True
            node_id_map_f[n["id"]] = target_id
        else:
            new_node = n.copy()
            new_id = next_id
            next_id += 1
            node_id_map_f[n["id"]] = new_id
            
            d = int(n.get("depth", 0))
            col = max_b_depth + d
            
            new_node["id"] = new_id
            new_node["depth"] = col
            if "meta" not in new_node: new_node["meta"] = {}
            new_node["meta"]["sankey_column"] = col
            merged_nodes.append(new_node)

    merged_links = []
    # 3. Merge Links
    # Backward Links: Invert (Child -> Parent) for Left-to-Right flow
    for l in b_links:

        merged = dict(l)
        src = l["source"]
        dst = l["target"]
        merged["source"] = node_id_map_b[dst]
        merged["target"] = node_id_map_b[src]
        # Update direction metadata to ensure consistent coloring if needed
        if "meta" not in merged: merged["meta"] = {}
        merged["meta"]["direction"] = "backward"
        merged_links.append(merged)

    # Forward Links: Keep (Parent -> Child)
    for l in f_links:
        merged = dict(l)
        src = l["source"]
        dst = l["target"]
        merged["source"] = node_id_map_f[src]
        merged["target"] = node_id_map_f[dst]
        if "meta" not in merged: merged["meta"] = {}
        merged["meta"]["direction"] = "forward"
        merged_links.append(merged)

    widths = _normalize_link_values([abs(l.get("value", 0.0)) for l in merged_links])
    for link, width in zip(merged_links, widths):
        link["normalized_value"] = width

    merged_json = {
        "nodes": merged_nodes,
        "links": merged_links,
        "meta": {
            "direction": "merged",
            "layer_depth": {}, # Layout is fully handled by sankey_column
            "viz_manifest_root": backward_json.get("meta", {}).get("viz_manifest_root")
            or forward_json.get("meta", {}).get("viz_manifest_root"),
        },
    }
    sankey_to_d3_html(merged_json, output_path, direction="merged")
