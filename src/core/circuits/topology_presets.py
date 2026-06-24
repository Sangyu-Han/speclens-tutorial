from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml

DEFAULT_TOPOLOGY_ROOT = (Path(__file__).resolve().parent.parent / "topologies").resolve()
DEFAULT_CLIP_TOPOLOGY_DIR = DEFAULT_TOPOLOGY_ROOT / "clip"
LEGACY_CLIP_TOPOLOGY_DIR = (Path(__file__).resolve().parent.parent.parent / "packs" / "clip" / "topologies").resolve()


def _slug(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum():
            safe.append(ch)
        elif ch in {".", "_"}:
            continue
        else:
            safe.append("_")
    return "".join(safe)


def _with_error_anchor(module_name: str) -> List[str]:
    anchors = [module_name]
    if "#latent" in module_name:
        anchors.append(module_name.replace("#latent", "#error_coeff"))
    return anchors


def _dedup(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


@dataclass
class ClipTopologySpec:
    name: str
    description: str
    ordered_layers: List[str]
    modules: Dict[str, str]
    head_layer: str
    head_module: str
    forward_backend: str = "input_x_grad"
    forward_ig_active: List[str] = field(default_factory=list)
    backward_backend: str = "ig"
    backward_ig_active: List[str] = field(default_factory=list)
    stop_grad_map: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ClipTopology:
    nodes: List[dict]
    edges: List[dict]
    layer_to_module: Dict[str, str]
    ordered_layers: List[str]
    head_layer: str
    head_module: str
    forward_backend: str
    forward_ig_active: List[str]
    backward_backend: str
    backward_ig_active: List[str]


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _normalise_base_dirs(base_dir: Path | str | Sequence[Path | str] | None) -> List[Path]:
    if base_dir is None:
        return [DEFAULT_CLIP_TOPOLOGY_DIR, DEFAULT_TOPOLOGY_ROOT, LEGACY_CLIP_TOPOLOGY_DIR]
    if isinstance(base_dir, (str, Path)):
        dirs = [base_dir]
    else:
        dirs = list(base_dir)
    paths = [Path(d).expanduser() for d in dirs if d]
    return paths or [DEFAULT_CLIP_TOPOLOGY_DIR, DEFAULT_TOPOLOGY_ROOT, LEGACY_CLIP_TOPOLOGY_DIR]


def _resolve_stop_grad(
    dst_layer: str,
    stop_grad_map: Dict[str, Sequence[str]],
    layer_to_module: Dict[str, str],
) -> List[str]:
    raw = stop_grad_map.get(dst_layer) or ()
    resolved: List[str] = []
    for entry in raw:
        key = str(entry)
        if key in layer_to_module:
            resolved.append(layer_to_module[key])
        else:
            resolved.append(key)
    return resolved


def _candidate_paths(name_or_path: str, base_dirs: Sequence[Path]) -> List[Path]:
    path = Path(name_or_path).expanduser()
    candidates: List[Path] = []
    if path.is_file():
        candidates.append(path)
        return candidates
    if path.suffix and path.exists():
        candidates.append(path)
        return candidates
    for base in base_dirs:
        stem = Path(name_or_path).name
        if stem.endswith(".yaml"):
            candidates.append(base / stem)
        else:
            candidates.append(base / f"{stem}.yaml")
            candidates.append(base / stem)
    return candidates


def load_topology_spec(name_or_path: str, *, base_dir: Path | str | Sequence[Path | str] | None = None) -> ClipTopologySpec:
    """
    Load a topology YAML. If `name_or_path` is not a path, resolves within base_dir(s).
    """
    search_dirs = _normalise_base_dirs(base_dir)
    candidates = _candidate_paths(name_or_path, search_dirs)
    target = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            target = cand
            break
    if target is None:
        raise FileNotFoundError(f"Topology config not found for '{name_or_path}'. Searched: {candidates}")

    data = _load_yaml(target) or {}
    name = data.get("name") or target.stem
    description = data.get("description") or ""
    ordered_layers = list(data.get("ordered_layers") or [])
    modules_raw: Dict[str, str] = dict(data.get("modules") or {})
    defaults = data.get("defaults") or {}
    head_layer = data.get("head_layer") or (ordered_layers[-1] if ordered_layers else "")
    head_module = data.get("head_module") or modules_raw.get(head_layer, "")

    if not ordered_layers:
        raise ValueError(f"Topology '{name}' missing ordered_layers.")
    if not modules_raw:
        raise ValueError(f"Topology '{name}' missing modules map.")
    missing = [layer for layer in ordered_layers if layer not in modules_raw]
    if missing:
        raise ValueError(f"Topology '{name}' missing module mapping for layers: {missing}")
    if not head_layer or not head_module:
        raise ValueError(f"Topology '{name}' must specify head_layer/head_module.")

    def _list_val(key: str) -> List[str]:
        val = defaults.get(key, [])
        if val is None:
            return []
        if isinstance(val, str):
            return [val]
        return [str(v) for v in val]

    raw_stop = data.get("stop_grad_map") or defaults.get("stop_grad_map") or {}
    stop_grad_map: Dict[str, List[str]] = {}
    for dst, raw in raw_stop.items():
        if raw is None:
            continue
        if isinstance(raw, str):
            values = [raw]
        else:
            values = [str(v) for v in raw]
        stop_grad_map[dst] = values

    return ClipTopologySpec(
        name=name,
        description=description,
        ordered_layers=ordered_layers,
        modules=modules_raw,
        head_layer=head_layer,
        head_module=head_module,
        forward_backend=str(defaults.get("forward_backend", "input_x_grad")),
        forward_ig_active=_list_val("forward_ig_active"),
        backward_backend=str(defaults.get("backward_backend", "ig")),
        backward_ig_active=_list_val("backward_ig_active") or ["model.patch_embed::pre@0"],
        stop_grad_map=stop_grad_map,
    )


def build_topology(
    spec: ClipTopologySpec,
    *,
    extra_stop_grad: Dict[str, Sequence[str]] | None = None,
    forward_backend: str | None = None,
    backward_backend: str | None = None,
    backward_ig_active_override: Sequence[str] | None = None,
    forward_ig_active_override: Sequence[str] | None = None,
    mark_head_terminal: bool = True,
) -> ClipTopology:
    stop_grad_map: Dict[str, List[str]] = {k: list(v) for k, v in (spec.stop_grad_map or {}).items()}
    for dst, sources in (extra_stop_grad or {}).items():
        stop_grad_map.setdefault(dst, [])
        for src in sources:
            if src not in stop_grad_map[dst]:
                stop_grad_map[dst].append(src)

    eff_forward_backend = forward_backend or spec.forward_backend
    eff_backward_backend = backward_backend or spec.backward_backend
    eff_forward_ig = list(forward_ig_active_override) if forward_ig_active_override is not None else list(
        spec.forward_ig_active or []
    )
    eff_backward_ig = list(backward_ig_active_override) if backward_ig_active_override is not None else list(
        spec.backward_ig_active or []
    )

    nodes: List[dict] = [{"layer": layer, "module": spec.modules[layer]} for layer in spec.ordered_layers]
    layer_to_module = dict(spec.modules)

    edges: List[dict] = []

    def _make_edge(direction: str, src: str, dst: str, backend: str, ig_active: Sequence[str]) -> dict:
        return {
            "id": f"{_slug(src)}_{direction}_{_slug(dst)}",
            "direction": direction,
            "src": src,
            "dst": dst,
            "backend": backend,
            "anchors": {
                "capture": _with_error_anchor(layer_to_module[dst]),
                "ig_active": _dedup(ig_active),
                "stop_grad": _resolve_stop_grad(dst, stop_grad_map, layer_to_module),
            },
            "terminal": mark_head_terminal and direction == "forward" and dst == spec.head_layer,
        }

    # forward chain
    for src, dst in zip(spec.ordered_layers, spec.ordered_layers[1:]):
        edges.append(
            _make_edge(
                direction="forward",
                src=src,
                dst=dst,
                backend=eff_forward_backend,
                ig_active=eff_forward_ig,
            )
        )

    # backward chain (exclude head as a source)
    backward_chain = spec.ordered_layers[:-1]
    for src, dst in zip(backward_chain[1:], backward_chain[:-1]):
        edges.append(
            _make_edge(
                direction="backward",
                src=src,
                dst=dst,
                backend=eff_backward_backend,
                ig_active=eff_backward_ig,
            )
        )

    return ClipTopology(
        nodes=nodes,
        edges=edges,
        layer_to_module=layer_to_module,
        ordered_layers=list(spec.ordered_layers),
        head_layer=spec.head_layer,
        head_module=spec.head_module,
        forward_backend=eff_forward_backend,
        forward_ig_active=eff_forward_ig,
        backward_backend=eff_backward_backend,
        backward_ig_active=eff_backward_ig,
    )


def list_topology_presets(base_dir: Path | str | Sequence[Path | str] | None = None) -> List[Tuple[str, str]]:
    dirs = _normalise_base_dirs(base_dir)
    results: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for directory in dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.yaml")):
            try:
                data = _load_yaml(path) or {}
                name = data.get("name") or path.stem
                desc = data.get("description") or ""
                if name in seen:
                    continue
                results.append((name, desc))
                seen.add(name)
            except Exception:
                continue
    return results


def default_feature_graph_config() -> dict:
    """
    Default feature_graph block mirroring clip_circuit_dag_blocks10.yaml.
    """
    return {
        "edge_weighting": {
            "mode": "none",
            "score_mode": "sign",
            "positive_only": True,
            "weight_backend": "ig",
        },
        "attribution": {
            "edge_backend": "input_x_grad",
        },
        "pruning": {
            "edge_score_mode": "sign",
            "positive_only": True,
            "per_child_topk_edges": 5,
            "per_child_edge_threshold": 0.0,
            "max_nodes_per_layer": 6,
        },
        "sankey": {
            "score_mode": "sign",
            "positive_only": True,
        },
    }


__all__ = [
    "ClipTopology",
    "ClipTopologySpec",
    "DEFAULT_CLIP_TOPOLOGY_DIR",
    "DEFAULT_TOPOLOGY_ROOT",
    "LEGACY_CLIP_TOPOLOGY_DIR",
    "build_topology",
    "default_feature_graph_config",
    "list_topology_presets",
    "load_topology_spec",
]
