# src/core/offline/base_ledger.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import warnings

import os, time, uuid
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


def ensure_dir(p: Path | str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def unique_basename(prefix: str) -> str:
    pid = os.getpid()
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}-{uid}-p{pid}-t{ts}"


class BaseOfflineLedger:
    """
    공용(모델 비특화) 오프라인 메타 레저 베이스.
    - 하위 클래스가 '샘플/프롬프트/기타' 테이블 스키마와 쓰기 로직을 정의.
    - 공통 파케셋 I/O 유틸을 제공.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        part_modulus: int = 128,
        compression: str = "zstd",
    ):
        self.root = Path(root_dir)
        ensure_dir(self.root)
        self.M = int(part_modulus)
        self._pq_kwargs = {
            "compression": compression,
            "use_dictionary": True,
            "write_statistics": True,
        }

    # ---------- Generic IO helpers ----------

    def _dataset_dir(self, name: str) -> Path:
        """
        서브테이블(예: 'samples', 'prompts')의 루트를 반환.
        """
        d = self.root / name
        ensure_dir(d)
        return d

    def write_rows_to_dataset(
        self,
        *,
        dataset_name: str,
        rows: List[Dict[str, Any]],
        schema: pa.Schema,
        partition_cols: Optional[List[str]] = None,
        basename_prefix: str = "rows",
        existing_data_behavior: str = "overwrite_or_ignore",
    ) -> int:
        """
        임의의 rows + schema를 받아 파케셋(하이브 파티셔닝)으로 기록.
        """
        if not rows:
            return 0

        # 스키마 순서에 맞춰 Arrow 컬럼 구성
        cols: Dict[str, pa.Array] = {}
        for f in schema:
            name = f.name
            vals = [r.get(name, None) for r in rows]
            cols[name] = pa.array(vals, type=f.type)

        table = pa.table(cols, schema=schema)

        ds_dir = str(self._dataset_dir(dataset_name))
        base = unique_basename(basename_prefix)
        pq.write_to_dataset(
            table,
            ds_dir,
            partition_cols=(partition_cols or []),
            basename_template=f"{base}-{{i}}.parquet",
            existing_data_behavior=existing_data_behavior,
            **self._pq_kwargs,
        )
        return table.num_rows

    def as_dataset(self, dataset_name: str) -> ds.Dataset:
        """
        하위 클래스에서 filter/to_table 등을 사용하도록 dataset 핸들을 제공.
        """
        return ds.dataset(str(self._dataset_dir(dataset_name)), format="parquet", partitioning="hive")

    # ---------- Extension points (override in subclass) ----------

    def write_from_batch(self, batch: Any) -> None:
        """
        하위 클래스가 구현:
        - 스트리밍 중 생성된 배치 메타데이터를
          'samples' / 'prompts' 등 테이블로 적재.
        """
        raise NotImplementedError

    def write_from_bvd(self, batch: Any) -> None:
        """Backward-compatible shim for legacy call sites."""
        warnings.warn(
            "BaseOfflineLedger.write_from_bvd is deprecated; use write_from_batch instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_from_batch(batch)

    def find_sample(self, sample_id: int) -> pa.Table:
        """
        하위 클래스가 구현:
        - 주어진 sample_id에 해당하는 샘플 메타 테이블 슬라이스를 반환.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        compression = self._pq_kwargs.get("compression")
        return (
            f"{self.__class__.__name__}(root='{self.root}', "
            f"part_modulus={self.M}, compression={compression!r})"
        )
