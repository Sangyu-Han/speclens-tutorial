# src/sae_index/registry_utils.py
import importlib, types, os, time, uuid
from pathlib import Path

def load_obj(dotted: str):
    """
    'package.module:object' 또는 'package.module.object' 형태 지원.
    """
    if ":" in dotted:
        mod, name = dotted.split(":", 1)
    else:
        mod, name = dotted.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, name)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def unique_basename(prefix: str, *, rank: int = 0) -> str:
    pid = os.getpid()
    ts  = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}-{uid}-r{rank}-p{pid}-t{ts}"

def sanitize_layer_name(layer: str) -> str:
    # parquet 파티션/파일명에 안전하도록
    return layer.replace("/", "_").replace(":", "_").replace(" ", "_")
