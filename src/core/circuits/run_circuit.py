# debug_make_circuit.py
# Tip: enable verbose IG anchor logs by running with ATTR_IG_DEBUG=1
import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import yaml

from src.core.circuits.tree_config import load_feature_graph_configs
from src.core.circuits.topology import parse_circuit_topology
from src.core.circuits.edge_runtime import EdgeAttrRuntime
from src.core.circuits.feature_tree_builder import FeatureTreeBuilder
from src.core.circuits.sankey_export import (
    feature_tree_to_sankey,
    sankey_to_d3_html,
    sankey_merge_forward_backward,
)

# Circuit runtime은 core 위치의 공용 구현을 사용.
from src.core.indexing.registry_utils import sanitize_layer_name
from src.core.runtime.attribution_runtime import AnchorConfig, BackwardConfig
from src.core.attribution.visualization.render import render_feature_activation_map
from src.packs.clip.circuit_runtime import ClipAttrRuntime, _build_image_batch, _default_image_path
os.environ["ATTR_IG_DEBUG"] = "0"  # 기본값 설정. 디버깅 = 1
def _ensure_list(value):
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _slugify(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    slug = slug.strip("_")
    return slug or "input"


def _reshape_token_map(vec: torch.Tensor) -> torch.Tensor:
    if vec.dim() != 1:
        vec = vec.flatten()
    tokens = vec.numel()
    drop_cls = False
    root = int(math.isqrt(int(tokens))) if tokens >= 0 else 0
    if root * root == tokens - 1:
        drop_cls = True
        tokens = tokens - 1
        root = int(math.isqrt(int(tokens))) if tokens >= 0 else 0
    if root * root != tokens:
        root = int(round(tokens ** 0.5))
    usable = root * root
    trimmed = vec[1 : 1 + usable] if drop_cls else vec[:usable]
    if usable == 0:
        return torch.zeros(1, 1, device=vec.device)
    return trimmed.view(root, root)


def _prepare_attr_map_2d(map_2d: torch.Tensor, *, min_abs: float = 0.0) -> torch.Tensor:
    m = map_2d.abs()
    if m.numel() == 0:
        return m
    m = torch.log1p(m)
    max_val = float(m.max())
    if max_val > 1e-8:
        m = m / max_val
    if min_abs > 0:
        m = m.masked_fill(m < float(min_abs), 0.0)
    return m


def _render_feature_map(
    *,
    out_dir: Path,
    sid: int,
    frames,
    map_2d: torch.Tensor,
    overlay_kwargs,
    file_stub: str,
) -> Path:
    map_stack = map_2d.unsqueeze(0)
    render_feature_activation_map(
        out_dir=out_dir,
        sid=sid,
        frames=frames,
        feature_map=map_stack,
        overlay_kwargs=overlay_kwargs,
        file_stub=file_stub,
    )
    return out_dir / f"sid{sid}_panel__{file_stub}.jpeg"


def _make_relative(path: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def _generate_input_feature_viz(
    *,
    runtime: ClipAttrRuntime,
    layer_to_module: dict,
    nodes: list[tuple[str, int]],
    viz_config_path: str | None,
    run_root: Path,
    image_path: Path,
    logger: logging.Logger,
) -> dict[tuple[str, int], dict]:
    viz_cfg = {}
    if viz_config_path:
        try:
            with open(viz_config_path, "r") as vf:
                viz_cfg = yaml.safe_load(vf) or {}
        except Exception as exc:
            logger.warning("[viz][input] failed to load viz config %s: %s", viz_config_path, exc)

    heat_cfg = viz_cfg.get("heatmap", {})
    overlay_kwargs = {
        "alpha": float(heat_cfg.get("overlay_alpha", 0.35)),
        "feature_map_alpha": float(heat_cfg.get("feature_map_alpha", heat_cfg.get("overlay_alpha", 0.35))),
        "feature_map_cmap": heat_cfg.get("feature_map_cmap", heat_cfg.get("overlay_cmap", "plasma")),
        "feature_map_min_abs": float(heat_cfg.get("feature_map_min_abs", 0.02)),
        "attr_map_cmap": heat_cfg.get("attr_map_cmap", "viridis"),
        "attr_map_alpha": float(heat_cfg.get("attr_map_alpha", heat_cfg.get("overlay_alpha", 0.35))),
        "attr_map_min_abs": float(heat_cfg.get("attr_map_min_abs", 0.0)),
    }

    backward_block = viz_cfg.get("backward", {})
    backward_anchors = backward_block.get("backward_anchors", backward_block.get("anchors", {}))
    libragrad_enabled = bool(getattr(runtime, "cfg", {}).get("libragrad", False))
    default_backward_method = "input_x_grad" if libragrad_enabled else "ig"
    anchor_cfg = AnchorConfig(
        capture=_ensure_list((backward_anchors or {}).get("module")),
        ig_active=_ensure_list((backward_anchors or {}).get("ig_active")),
        stop_grad=_ensure_list((backward_anchors or {}).get("stop_grad")),
    )
    backward_cfg = BackwardConfig(
        enabled=True,
        method=backward_block.get("method", default_backward_method),
        ig_steps=int(backward_block.get("ig_steps", 16)),
        baseline=str(backward_block.get("baseline", "zeros")),
    )

    att_runtime = runtime.runtime
    override_mode = backward_block.get("override_mode", getattr(att_runtime.target, "override_mode", "all_tokens"))
    objective_aggregation = backward_block.get(
        "objective_aggregation", getattr(att_runtime.target, "objective_aggregation", "sum")
    )
    try:
        att_runtime.target.override_mode = override_mode
        att_runtime.target.objective_aggregation = objective_aggregation
    except Exception:
        pass
    att_runtime.configure_anchors(anchor_cfg)

    try:
        batch_cpu = _build_image_batch(runtime.index_cfg, image_path)
        frames = [batch_cpu["pixel_values"]]
        sid_tensor = batch_cpu.get("sample_id")
        sid = int(sid_tensor.view(-1)[0].item()) if torch.is_tensor(sid_tensor) and sid_tensor.numel() > 0 else 0
    except Exception as exc:
        logger.warning("[viz][input] failed to rebuild batch for %s: %s", image_path, exc)
        return {}

    run_root.mkdir(parents=True, exist_ok=True)
    records: dict[tuple[str, int], dict] = {}
    for layer, feat in sorted(nodes, key=lambda kv: (kv[0], kv[1])):
        if feat is None or feat < 0:
            continue
        module_name = layer_to_module.get(layer)
        if not module_name:
            logger.debug("[viz][input] skip layer without module mapping: %s", layer)
            continue
        try:
            sae_module = runtime._sae_resolver(module_name)
        except Exception as exc:
            logger.warning("[viz][input] failed to load SAE for %s: %s", module_name, exc)
            continue
        try:
            att_runtime.set_target(module_name, int(feat), sae_module=sae_module)
        except Exception as exc:
            logger.warning("[viz][input] failed to set target %s/%s: %s", layer, feat, exc)
            continue
        att_runtime.configure_anchors(anchor_cfg)

        layer_slug = sanitize_layer_name(module_name)
        unit_dir = run_root / layer_slug / f"unit_{int(feat)}"
        unit_dir.mkdir(parents=True, exist_ok=True)

        feature_path = None
        try:
            att_runtime._run_forward(require_grad=False)
            latent = att_runtime.controller.post_stack(detach=True)
            if latent is None or latent.dim() < 2:
                raise RuntimeError("latent is empty")
            act_vec = latent[0].reshape(-1, latent.shape[-1])[:, int(feat)].flatten()
            map_2d = _reshape_token_map(act_vec)
            feature_path = _render_feature_map(
                out_dir=unit_dir,
                sid=sid,
                frames=frames,
                map_2d=map_2d,
                overlay_kwargs=overlay_kwargs,
                file_stub="feature_map",
            )
        except Exception as exc:
            logger.warning("[viz][input] failed to render feature map %s/%s: %s", layer, feat, exc)

        attr_path = None
        try:
            attr_map = att_runtime.run_backward(backward_cfg) or {}
            patch_key = None
            for key in attr_map.keys():
                if "patch_embed" in key:
                    patch_key = key
                    break
            tensor = attr_map.get(patch_key) if patch_key else None
            if tensor is not None:
                if tensor.dim() > 3:
                    tensor = tensor.sum(dim=0)
                if tensor.dim() == 3:
                    tensor = tensor.pow(2).mean(-1).sqrt()
                elif tensor.dim() == 2:
                    tensor = tensor
                else:
                    tensor = tensor.flatten()
                attr_vec = tensor.flatten()
                attr_map_2d = _reshape_token_map(attr_vec)
                attr_map_2d = _prepare_attr_map_2d(
                    attr_map_2d,
                    min_abs=overlay_kwargs.get("attr_map_min_abs", 0.0),
                )
                attr_kwargs = dict(overlay_kwargs)
                attr_kwargs.update(
                    {
                        "feature_map_cmap": overlay_kwargs.get("attr_map_cmap", overlay_kwargs.get("feature_map_cmap")),
                        "feature_map_alpha": overlay_kwargs.get("attr_map_alpha", overlay_kwargs.get("feature_map_alpha")),
                    }
                )
                attr_path = _render_feature_map(
                    out_dir=unit_dir,
                    sid=sid,
                    frames=frames,
                    map_2d=attr_map_2d,
                    overlay_kwargs=attr_kwargs,
                    file_stub="attr_patch_embed",
                )
        except Exception as exc:
            logger.debug("[viz][input] failed to render attr map %s/%s: %s", layer, feat, exc)

        if feature_path or attr_path:
            records[(layer, feat)] = {
                "feature_map": _make_relative(feature_path) if feature_path else "",
                "attr_map": _make_relative(attr_path) if attr_path else "",
            }
    manifest = {
        "image": _make_relative(image_path),
        "count": len(records),
        "root": _make_relative(run_root),
        "nodes": [
            {"layer": layer, "feature_idx": int(feat), **info} for (layer, feat), info in records.items()
        ],
    }
    try:
        (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        pass
    logger.info("[viz][input] saved %d input feature viz entries under %s", len(records), run_root)
    return records


def main():
    logging.basicConfig(level=logging.INFO, format="[debug] %(message)s")
    logger = logging.getLogger("debug_make_circuit")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/clip_circuit_dag_blocks10.yaml",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="yaml의 output_root를 override 하고 싶으면 사용",
    )
    parser.add_argument(
        "--viz_config",
        type=str,
        default="configs/clip_attr_viz.yaml",
        help="Optional viz config path; if set, missing manifests will be generated on the fly.",
    )
    parser.add_argument(
        "--viz_root",
        type=str,
        default="outputs/clip_attr_viz_cache",
        help="Optional path to visualization manifests (unit_{idx}/manifest.json) for thumbnails.",
    )

    args = parser.parse_args()




    logger.info("loading circuit config from %s", args.config)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    tree_cfg = cfg["tree"]
    viz_manifest_root = args.viz_root or tree_cfg.get("viz_cache_root") or cfg.get("viz_cache_root")
    logger.info(
        "viz manifest root resolved to %s (auto_viz_config=%s)",
        viz_manifest_root,
        args.viz_config,
    )

    edge_weighting_cfg, pruning_cfg, sankey_cfg, attr_cfg = load_feature_graph_configs(tree_cfg)
    topology, layer_to_module = parse_circuit_topology(tree_cfg)
    edge_specs = topology.edges

    # ClipAttrRuntime 인스턴스 생성 (실제 인자는 기존 코드에 맞게 수정)
    attr_runtime = ClipAttrRuntime(config=cfg)

    edge_runtime = EdgeAttrRuntime(
        runtime=attr_runtime,
        layer_to_module=layer_to_module,
    )

    builder = FeatureTreeBuilder(
        topology=topology,
        edge_specs=edge_specs,
        runtime=edge_runtime,
        pruning_cfg=pruning_cfg,
        edge_weighting_cfg=edge_weighting_cfg,
        attr_cfg=attr_cfg,
    )

    root_cfg = tree_cfg.get("root", {})
    if isinstance(root_cfg, dict):
        root_layer = root_cfg.get("layer") or root_cfg.get("name")
        root_feature_indices = [int(u) for u in root_cfg.get("units", [])] or []
    else:
        root_layer = root_cfg
        root_feature_indices = []
    logger.info(
        "root resolved: layer=%s, features=%s",
        root_layer,
        root_feature_indices,
    )
    if not root_feature_indices:
        try:
            root_feature_indices = [int(cfg["runtime"]["target"]["unit"])]
        except Exception:
            root_feature_indices = [0]
    logger.info("building trees...")
    backward_tree = builder.build(
        direction="backward",
        root_layer=root_layer,
        root_feature_indices=root_feature_indices,
    )
    forward_tree = builder.build(
        direction="forward",
        root_layer=root_layer,
        root_feature_indices=root_feature_indices,
    )

    logger.info(
        "built forward tree: %d nodes / %d edges; backward tree: %d nodes / %d edges",
        len(forward_tree.nodes),
        len(forward_tree.edges),
        len(backward_tree.nodes),
        len(backward_tree.edges),
    )

    image_path = _default_image_path(cfg).expanduser()
    image_slug = _slugify(image_path.stem)
    run_viz_root = Path("outputs") / "circuit_run" / image_slug
    node_pairs = {
        (n.key.layer, n.key.feature_idx)
        for n in list(forward_tree.nodes) + list(backward_tree.nodes)
        if n.key.feature_idx is not None and n.key.feature_idx >= 0
    }
    input_viz_map = _generate_input_feature_viz(
        runtime=attr_runtime,
        layer_to_module=layer_to_module,
        nodes=list(node_pairs),
        viz_config_path=args.viz_config,
        run_root=run_viz_root,
        image_path=image_path,
        logger=logger,
    )
    if input_viz_map:
        def _attach(tree):
            for node in tree.nodes:
                key = (node.key.layer, node.key.feature_idx)
                if key not in input_viz_map:
                    continue
                entry = input_viz_map[key]
                node.metadata = dict(node.metadata or {})
                node.metadata.setdefault("input_viz", entry)
                if entry.get("feature_map"):
                    node.metadata.setdefault("input_feature_map", entry["feature_map"])
                if entry.get("attr_map"):
                    node.metadata.setdefault("input_attr_map", entry["attr_map"])
            tree.metadata = dict(getattr(tree, "metadata", {}) or {})
            tree.metadata.setdefault("input_viz_root", _make_relative(run_viz_root))

        _attach(forward_tree)
        _attach(backward_tree)


    backward_sankey = feature_tree_to_sankey(
        backward_tree,
        sankey_cfg,
        viz_manifest_root=viz_manifest_root,
        auto_viz_config=args.viz_config,
        auto_viz_root=viz_manifest_root,
    )
    logger.info(
        "backward sankey: %d nodes / %d links (has_viz=%s)",
        len(backward_sankey.get("nodes", [])),
        len(backward_sankey.get("links", [])),
        any("viz" in (n.get("meta") or {}) for n in backward_sankey.get("nodes", [])),
    )
    forward_sankey = feature_tree_to_sankey(
        forward_tree,
        sankey_cfg,
        viz_manifest_root=viz_manifest_root,
        auto_viz_config=args.viz_config,
        auto_viz_root=viz_manifest_root,
    )
    logger.info(
        "forward sankey: %d nodes / %d links (has_viz=%s)",
        len(forward_sankey.get("nodes", [])),
        len(forward_sankey.get("links", [])),
        any("viz" in (n.get("meta") or {}) for n in forward_sankey.get("nodes", [])),
    )

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else Path(cfg["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "feature_tree_backward.json").write_text(
        json.dumps(backward_sankey, indent=2)
    )
    (output_root / "feature_tree_forward.json").write_text(
        json.dumps(forward_sankey, indent=2)
    )
    # html sankey (separate and merged)
    try:
        sankey_to_d3_html(backward_sankey, str(output_root / "feature_tree_backward.html"), direction="backward")
        sankey_to_d3_html(forward_sankey, str(output_root / "feature_tree_forward.html"), direction="forward")
        sankey_merge_forward_backward(
            backward_sankey,
            forward_sankey,
            str(output_root / "feature_tree_merged.html"),
        )
    except Exception as e:
        print(f"[warn] failed to render sankey html: {e}")

    try:
        attr_runtime.cleanup()
    except Exception:
        pass

    print(f"[OK] saved feature trees to {output_root}")


if __name__ == "__main__":
    main()
