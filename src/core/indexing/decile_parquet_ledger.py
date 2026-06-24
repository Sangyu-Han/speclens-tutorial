# src/sae_index/decile_parquet_ledger.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Mapping, Sequence
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
from .registry_utils import ensure_dir, unique_basename
import torch

def _part(sample_id: int, M: int = 128) -> int:
    return int(sample_id) % int(M)

DECILES_SCHEMA = pa.schema([
    ("run_id", pa.string()),            # 실험/인덱싱 런 식별자(필수 권장)
    ("layer", pa.string()),             # 레이어 이름
    ("unit", pa.int32()),               # SAE feature index
    ("score", pa.float32()),            # 활성값 혹은 SAE 출력 점수
    ("decile", pa.int32()),             # 0..(num_deciles-1)
    ("rank_in_decile", pa.int32()),     # decile 내 상위 정렬 순위(0=1등)
    ("sample_id", pa.int64()),          # 원본 샘플 ID
    ("frame_idx", pa.int32()),          # 토큰/좌표가 속한 프레임 인덱스
    ("y", pa.int32()),                  # 좌표(-1은 토큰류 의미)
    ("x", pa.int32()),
    ("prompt_id", pa.int64()),          # 프롬프트 세트 ID(주입 프레임 폴백 포함)
    ("uid", pa.int64()),                # (옵션) 객체 UID(t0 기준). 없으면 -1
    ("stride_step", pa.int32()),        # stride 정보
    ("meta_json", pa.string()),         # (옵션) 부가 메타 JSON 문자열
    # 파티션 컬럼은 write 시 append
])

class DecileParquetLedger:
    """
    deciles/ 이하에 파케셋 저장.
    파티션: layer(string)/decile(int)/part(int=sample_id%M)
    """
    def __init__(self, root_dir: str | Path, *, M_part: int = 128, compression: str = "zstd"):
        self.root = Path(root_dir)
        self.dir  = self.root / "deciles"
        ensure_dir(self.dir)
        self.M = int(M_part)
        self._pq_kwargs = {
            "compression": compression,
            "use_dictionary": ["layer", "meta_json", "run_id"],
            "write_statistics": True,
        }

    def _schema_with_partitions(self) -> pa.Schema:
        return (
            DECILES_SCHEMA
            .append(pa.field("layer_part", pa.string()))
            .append(pa.field("decile_part", pa.int32()))
            .append(pa.field("part", pa.int32()))
        )

    def _table_from_row_dicts(self, rows: List[Dict[str, Any]]) -> pa.Table:
        sch = self._schema_with_partitions()
        n_rows = len(rows)
        cols: Dict[str, List[Any]] = {f.name: [None] * n_rows for f in sch}

        # Timing: populate all parquet columns in one row pass so finalize() does not
        # pay an additional schema x rows walk with millions of dict.get calls.
        for idx, row in enumerate(rows):
            layer = str(row["layer"])
            decile = int(row["decile"])
            sample_id = int(row["sample_id"])

            cols["run_id"][idx] = row["run_id"]
            cols["layer"][idx] = layer
            cols["unit"][idx] = int(row["unit"])
            cols["score"][idx] = float(row["score"])
            cols["decile"][idx] = decile
            cols["rank_in_decile"][idx] = int(row["rank_in_decile"])
            cols["sample_id"][idx] = sample_id
            cols["frame_idx"][idx] = int(row["frame_idx"])
            cols["y"][idx] = int(row["y"])
            cols["x"][idx] = int(row["x"])
            cols["prompt_id"][idx] = int(row["prompt_id"])
            cols["uid"][idx] = int(row.get("uid", -1))
            cols["stride_step"][idx] = int(row["stride_step"])
            cols["meta_json"][idx] = row.get("meta_json", "")
            cols["layer_part"][idx] = layer
            cols["decile_part"][idx] = decile
            cols["part"][idx] = _part(sample_id, self.M)

        arrays = {
            f.name: pa.array(cols[f.name], type=f.type)
            for f in sch
        }
        return pa.table(arrays, schema=sch)

    def _table_from_columnar(self, columns: Mapping[str, Sequence[Any]]) -> pa.Table:
        sch = self._schema_with_partitions()
        names = list(columns.keys())
        if not names:
            return pa.table({f.name: pa.array([], type=f.type) for f in sch}, schema=sch)

        n_rows = len(columns[names[0]])
        for name in names[1:]:
            if len(columns[name]) != n_rows:
                raise ValueError(
                    f"Column '{name}' has length {len(columns[name])}, expected {n_rows}"
                )

        def _require(name: str) -> Sequence[Any]:
            if name not in columns:
                raise KeyError(f"Missing required parquet column '{name}'")
            return columns[name]

        sample_ids = np.asarray(_require("sample_id"), dtype=np.int64)
        layers = [str(v) for v in _require("layer")]
        deciles = np.asarray(_require("decile"), dtype=np.int32)

        if n_rows:
            parts = np.remainder(sample_ids, self.M).astype(np.int32, copy=False)
        else:
            parts = np.empty((0,), dtype=np.int32)

        payload: Dict[str, Sequence[Any]] = {
            "run_id": _require("run_id"),
            "layer": layers,
            "unit": _require("unit"),
            "score": _require("score"),
            "decile": deciles,
            "rank_in_decile": _require("rank_in_decile"),
            "sample_id": sample_ids,
            "frame_idx": _require("frame_idx"),
            "y": _require("y"),
            "x": _require("x"),
            "prompt_id": _require("prompt_id"),
            "uid": columns.get("uid", [-1] * n_rows),
            "stride_step": _require("stride_step"),
            "meta_json": columns.get("meta_json", [""] * n_rows),
            "layer_part": layers,
            "decile_part": deciles,
            "part": parts,
        }

        arrays = {
            f.name: pa.array(payload[f.name], type=f.type)
            for f in sch
        }
        return pa.table(arrays, schema=sch)

    def write_rows(
        self,
        rows: List[Dict[str, Any]] | Mapping[str, Sequence[Any]],
        *,
        rank: int = 0,
    ) -> int:
        if not rows:
            return 0

        if isinstance(rows, Mapping):
            tbl = self._table_from_columnar(rows)
            n_rows = tbl.num_rows
        else:
            tbl = self._table_from_row_dicts(rows)
            n_rows = len(rows)

        # (3) dataset.write_dataset 로 교체 (+ max_partitions 늘리기)
        base = unique_basename("deciles", rank=rank)

        # 파티션 정의(Hive 스타일)
        part_schema = pa.schema([
            ("layer_part", pa.string()),
            ("decile_part", pa.int32()),
            ("part", pa.int32()),
        ])
        partitioning = ds.partitioning(part_schema, flavor="hive")

        # Parquet write 옵션 구성 (기존 _pq_kwargs 재사용)
        fmt = ds.ParquetFileFormat()
        file_options = fmt.make_write_options(**self._pq_kwargs)

        # 파티션 상한(기본 1024)을 충분히 키움
        max_parts = max(8192, self.M * 16)  # 여유 있게

        ds.write_dataset(
            data=tbl,
            base_dir=str(self.dir),
            format="parquet",
            partitioning=partitioning,
            basename_template=f"{base}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            file_options=file_options,
            max_partitions=max_parts,
            use_threads=True,
        )
        return int(n_rows)


    # 조회 유틸
    def as_dataset(self):
        return ds.dataset(str(self.dir), format="parquet", partitioning="hive")
    def topn_for(self, *, layer: str, unit: int, decile: int, n: int) -> pa.Table:
        dset = self.as_dataset()
        f = (
            (ds.field("layer_part") == str(layer)) &      # 파티션 프루닝
            (ds.field("decile_part") == int(decile)) &    # 파티션 프루닝
            (ds.field("unit") == int(unit)) &
            (ds.field("rank_in_decile") < int(n))         # 상위 n만
        )
        # 필요한 컬럼만 최소화 (정말 필요한 것만 남기면 더 빨라짐)
        cols = ["layer","unit","score","decile","rank_in_decile",
                "sample_id","frame_idx","y","x","prompt_id","uid","stride_step","run_id"]
        tbl = dset.to_table(filter=f, columns=cols)

        # rank 기준으로 정렬(필요 시). 이미 n개 수준이라 비용 매우 작음.
        if tbl.num_rows <= 1:
            return tbl
        idx = pc.sort_indices(tbl, sort_keys=[("rank_in_decile", "ascending")])
        return pc.take(tbl, idx)

    def units_for_layer(self, layer: str) -> List[int]:
        """
        Return sorted list of SAE unit indices that have at least one row for the given layer.
        """
        try:
            dset = self.as_dataset()
        except (FileNotFoundError, pa.ArrowInvalid):
            return []

        tbl = dset.to_table(
            filter=ds.field("layer_part") == str(layer),
            columns=["unit"],
        )
        if tbl.num_rows == 0:
            return []
        return sorted(pc.unique(tbl["unit"]).to_pylist())

    def __repr__(self) -> str:
        compression = self._pq_kwargs.get("compression")
        return (
            f"DecileParquetLedger(root='{self.root}', dir='{self.dir}', "
            f"M={self.M}, compression={compression!r})"
        )

    # def topn_for(self, *, layer: str, unit: int, decile: int, n: int) -> pa.Table:
    #     dset = self.as_dataset()
    #     f = (
    #         (ds.field("layer") == str(layer)) &
    #         (ds.field("unit") == int(unit)) &
    #         (ds.field("decile") == int(decile))
    #     )
    #     tbl = dset.to_table(filter=f, columns=[
    #         "layer","unit","score","decile","rank_in_decile",
    #         "sample_id","frame_idx","y","x","prompt_id","uid","stride_step","run_id"
    #     ])
    #     # score 내림차순 정렬 후 상위 n
    #     if tbl.num_rows <= n:
    #         return tbl
    #     import pyarrow.compute as pc
    #     idx = pc.sort_indices(tbl, sort_keys=[("score", "descending")])
    #     tbl_sorted = pc.take(tbl, idx)
    #     return tbl_sorted.slice(0, n)
