from pathlib import Path
import torch

def load_prompt_by_uid(root_dir: str | Path, uid: str) -> dict | None:
    import csv
    from pathlib import Path
    import pyarrow.parquet as pq
    root = Path(root_dir)
    man = root / "prompts_manifest.csv"
    shard, row_group = None, None
    with open(man, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["prompt_uid"] == uid:
                shard = r["shard"]; row_group = int(r["row_group"]); break
    if shard is None:
        return None
    pf = pq.ParquetFile(root / shard)
    tab = pf.read_row_group(row_group)  # 해당 group만 로드
    # row_group 내에서 uid를 필터
    col = tab.column("prompt_uid").to_pylist()
    idx = [i for i,v in enumerate(col) if v == uid]
    if not idx: return None
    i = idx[0]
    out = {name: tab.column(name)[i].as_py() for name in tab.column_names}
    return out


def stable_u64(s: str) -> int:
    """문자열을 결정적 63-bit 양수 ID로 매핑 (prompt_uid 등)"""
    import hashlib
    h = hashlib.sha256(str(s).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False) & ((1 << 63) - 1)

def load_obj(dotted: str):
    if ":" in dotted:
        mod, name = dotted.split(":", 1)
    else:
        mod, name = dotted.rsplit(".", 1)
    import importlib
    m = importlib.import_module(mod)
    return getattr(m, name)

def resolve_module(model: torch.nn.Module, dotted: str) -> torch.nn.Module:
    tokens = dotted.split(".") if dotted else []
    if tokens and tokens[0] == "model":
        tokens = tokens[1:]
    cur = model
    if not tokens:
        return cur
    for tok in tokens:
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            if not hasattr(cur, tok):
                try:
                    cur = dict(cur.named_modules())[dotted]
                    return cur
                except Exception:
                    raise AttributeError(f"Cannot resolve module token '{tok}' in path '{dotted}'")
            else:
                cur = getattr(cur, tok)
    return cur
