# src/sae_index/decile_aggregator.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Sequence, Tuple
import torch, heapq, json
from dataclasses import dataclass
from .decile_parquet_ledger import DecileParquetLedger

@dataclass
class RunFingerprint:
    model_name: str
    model_yaml: str
    model_ckpt: str
    model_ckpt_sha: str
    sae_ckpt: str
    sae_ckpt_sha: str
    dataset_name: str
    run_id: str                   # 단일 문자열 런 식별자

_PROV_DEFAULTS = {
    "sample_id": 0,
    "frame_idx": 0,
    "y": -1,
    "x": -1,
    "prompt_id": 0,
    "uid": -1,
}

_DEDUPE_LOC_HEAP = "heap"
_DEDUPE_LOC_RAND = "rand"

class DecileTopKParquet:
    """
    per-feature per-decile top-k + random_k를 수집하여
    ★ CSV 없이 파케(DecileParquetLedger)에만 기록 ★
    - prov 포맷 기대: [N,6] = (sample_id, frame_idx, y, x, prompt_id, uid)
      (과거 [N,4]/[N,5]도 허용. 부족하면 prompt_id=0, uid=-1로 보정)
    - 경계: per-feature max_range 또는 전역 cutoffs
    """
    def __init__(self, *, dict_size: int, num_deciles: int, k: int, rand_k: int,
                 maxima: torch.Tensor, layer_name: str, fp: RunFingerprint,
                 ledger: DecileParquetLedger,
                 prov_cols: Sequence[str],
                 dedupe_key_fn: Optional[Callable[[tuple], Any]] = None,
                 boundary: str = "max_range", fixed_cutoffs: List[float] | None = None,
                 slack: int = 4, rank: int = 0):

        self.D = int(dict_size)
        self.num_deciles = int(num_deciles)
        self.k = int(k)
        self.k_internal = int(k + slack)
        self.rand_k = int(rand_k)
        self.layer = str(layer_name)
        self.fp = fp
        self.ledger = ledger
        self.rank = int(rank)

        self.maxima = (maxima.clone().cpu() if isinstance(maxima, torch.Tensor)
                       else torch.as_tensor(maxima, dtype=torch.float32)).clamp_min(1e-8)

        self.boundary = boundary
        self.cutoffs = list(fixed_cutoffs) if fixed_cutoffs is not None else None
        if self.boundary == "cutoffs":
            assert self.cutoffs and len(self.cutoffs) == self.num_deciles - 1

        self.heaps = [[[] for _ in range(self.num_deciles)] for _ in range(self.D)]
        self.rand  = [[] for _ in range(self.D)]
        self.rand_seen = [0] * self.D
        self._rng = __import__("random").Random(12345)

        self.prov_cols = tuple(prov_cols)
        self._prov_len = len(self.prov_cols)
        self._prov_index = {name: idx for idx, name in enumerate(self.prov_cols)}
        self._idx_sample_id = self._prov_index.get("sample_id")
        self._idx_frame_idx = self._prov_index.get("frame_idx")
        self._idx_y = self._prov_index.get("y")
        self._idx_x = self._prov_index.get("x")
        self._idx_prompt_id = self._prov_index.get("prompt_id")
        self._idx_uid = self._prov_index.get("uid")

        self._configure_dedupe(dedupe_key_fn)

        self.run_epoch = 0
        self.run_b_in_epoch = 0
        self.global_steps = 0

    # --- 경계/bucket ---
    def _decile_of_max_range(self, d: int, v: float, mx: float) -> int:
        step = mx / self.num_deciles
        seg = int((mx - v) // max(step, 1e-8))
        return max(0, min(self.num_deciles - 1, seg))
    def _interval_of_max_range(self, v: float, mx: float, seg: int) -> str:
        step = mx / self.num_deciles
        hi = mx - seg * step
        lo = hi - step
        return f"[{lo:.3g},{hi:.3g})"
    def _decile_of_cutoffs(self, v: float) -> int:
        for i, c in enumerate(self.cutoffs):
            if v <= c: return i
        return self.num_deciles - 1
    def _bucket_of(self, d: int, v: float, mx_for_str: float | None = None) -> tuple[int, str]:
        if self.boundary == "cutoffs":
            seg = self._decile_of_cutoffs(v)
            lo = -float("inf") if seg == 0 else self.cutoffs[seg-1]
            hi = float("inf")  if seg == self.num_deciles-1 else self.cutoffs[seg]
            return seg, f"({lo:.3g},{hi:.3g}]"
        mx = float(mx_for_str if mx_for_str is not None else self.maxima[d].item())
        seg = self._decile_of_max_range(d, v, mx)
        return seg, self._interval_of_max_range(v, mx, seg)


    # --- dedupe helpers ---
    def _configure_dedupe(self, dedupe_key_fn: Optional[Callable[[tuple], Any]]) -> None:
        self._dedupe_key_fn = dedupe_key_fn
        if dedupe_key_fn is None:
            self._dedupe_store: Optional[List[Dict[Any, tuple]]] = None
        else:
            self._dedupe_store = [dict() for _ in range(self.D)]

    def _dedupe_lookup(self, d: int, key: Any) -> Optional[Tuple[tuple, Tuple[str, Optional[int]]]]:
        if self._dedupe_store is None:
            return None
        return self._dedupe_store[d].get(key)

    def _dedupe_register(
        self,
        d: int,
        item: tuple,
        key: Optional[Any] = None,
        *,
        location: Tuple[str, Optional[int]],
    ) -> None:
        if self._dedupe_store is None:
            return
        if key is None:
            key = self._dedupe_key_fn(item)
        self._dedupe_store[d][key] = (item, location)

    def _dedupe_forget_item(self, d: int, item: tuple) -> None:
        if self._dedupe_store is None:
            return
        key = self._dedupe_key_fn(item)
        store = self._dedupe_store[d]
        entry = store.get(key)
        if entry is not None and entry[0] == item:
            store.pop(key, None)

    def _dedupe_remove_entry(
        self,
        d: int,
        key: Any,
        entry: Tuple[tuple, Tuple[str, Optional[int]]],
    ) -> None:
        if self._dedupe_store is None:
            return
        store = self._dedupe_store[d]
        if store.get(key) != entry:
            return
        store.pop(key, None)
        item, (loc_kind, loc_arg) = entry
        if loc_kind == _DEDUPE_LOC_HEAP and loc_arg is not None:
            self._dedupe_remove_from_heap(d, loc_arg, item)
        elif loc_kind == _DEDUPE_LOC_RAND:
            self._dedupe_remove_from_rand(d, item)

    def _dedupe_remove_from_heap(self, d: int, seg: int, item: tuple) -> None:
        heap = self.heaps[d][seg]
        for idx, candidate in enumerate(heap):
            if candidate == item:
                heap.pop(idx)
                heapq.heapify(heap)
                return

    def _dedupe_remove_from_rand(self, d: int, item: tuple) -> None:
        rr = self.rand[d]
        for idx, candidate in enumerate(rr):
            if candidate == item:
                rr.pop(idx)
                return

    def _dedupe_rebuild_store(self) -> None:
        if self._dedupe_key_fn is None:
            self._dedupe_store = None
            return
        if self._dedupe_store is None or len(self._dedupe_store) != self.D:
            self._dedupe_store = [dict() for _ in range(self.D)]
        for d in range(self.D):
            store_d = self._dedupe_store[d]
            store_d.clear()
            for seg in range(self.num_deciles):
                for it in self.heaps[d][seg]:
                    store_d[self._dedupe_key_fn(it)] = (it, (_DEDUPE_LOC_HEAP, seg))
            for it in self.rand[d]:
                store_d[self._dedupe_key_fn(it)] = (it, (_DEDUPE_LOC_RAND, None))

    def _dedupe_store_snapshot(self) -> Optional[List[List[Tuple[Any, Tuple[tuple, Tuple[str, Optional[int]]]]]]]:
        if self._dedupe_store is None:
            return None
        return [list(store.items()) for store in self._dedupe_store]

    # --- 상태 저장/로드(DDP 체크포인트용) ---
    def state_dict(self) -> dict:
        return {
            "version": 2,
            "D": self.D,
            "num_deciles": self.num_deciles,
            "k": self.k,
            "k_internal": self.k_internal,
            "rand_k": self.rand_k,
            "maxima": self.maxima,
            "heaps": self.heaps,
            "rand": self.rand,
            "rand_seen": self.rand_seen,
            "rng_state": self._rng.getstate(),
            "layer": self.layer,
            "fp": self.fp.__dict__,
            "run_epoch": self.run_epoch,
            "run_b_in_epoch": self.run_b_in_epoch,
            "global_steps": self.global_steps,
            "boundary": self.boundary,
            "cutoffs": self.cutoffs,
            "cutoffs": self.cutoffs,
            "prov_cols": list(self.prov_cols),
            "dedupe_enabled": bool(self._dedupe_key_fn is not None),
            "dedupe_store": self._dedupe_store_snapshot(),
        }
    def load_state_dict(self, sd: dict):
        self.D = int(sd["D"])
        self.num_deciles = int(sd["num_deciles"])
        self.k = int(sd["k"])
        self.k_internal = int(sd.get("k_internal", self.k + 4))
        self.rand_k = int(sd["rand_k"])
        self.maxima = (sd["maxima"].clone().cpu()
                       if isinstance(sd["maxima"], torch.Tensor)
                       else torch.as_tensor(sd["maxima"], dtype=torch.float32)).clamp_min(1e-8)
        self.heaps = sd["heaps"]; self.rand = sd["rand"]; self.rand_seen = sd["rand_seen"]
        try: self._rng.setstate(sd["rng_state"])
        except Exception: pass
        self.run_epoch = int(sd.get("run_epoch", 0))
        self.run_b_in_epoch = int(sd.get("run_b_in_epoch", 0))
        self.global_steps = int(sd.get("global_steps", 0))
        self.boundary = sd.get("boundary", "max_range")
        self.cutoffs  = sd.get("cutoffs", None)
        prov_cols = sd.get("prov_cols")
        if prov_cols is not None:
            self.prov_cols = tuple(prov_cols)
            self._prov_len = len(self.prov_cols)
            self._prov_index = {name: idx for idx, name in enumerate(self.prov_cols)}
            self._idx_sample_id = self._prov_index.get("sample_id")
            self._idx_frame_idx = self._prov_index.get("frame_idx")
            self._idx_y = self._prov_index.get("y")
            self._idx_x = self._prov_index.get("x")
            self._idx_prompt_id = self._prov_index.get("prompt_id")
            self._idx_uid = self._prov_index.get("uid")
        dedupe_snapshot = sd.get("dedupe_store")
        if self._dedupe_key_fn is None:
            self._dedupe_store = None
        elif (
            isinstance(dedupe_snapshot, list)
            and len(dedupe_snapshot) == self.D
        ):
            if self._dedupe_store is None or len(self._dedupe_store) != self.D:
                self._dedupe_store = [dict() for _ in range(self.D)]
            for d, entries in enumerate(dedupe_snapshot):
                store_d = self._dedupe_store[d]
                store_d.clear()
                if not isinstance(entries, list):
                    continue
                for key, entry in entries:
                    item, location = entry
                    if isinstance(item, list):
                        item = tuple(item)
                    if isinstance(key, list):
                        key = tuple(key)
                    loc_kind, loc_arg = location
                    store_d[key] = (tuple(item), (loc_kind, loc_arg))
        else:
            self._dedupe_rebuild_store()
        # fp/layer는 외부에서 설정된 걸 유지

    # --- 랜덤 저수지 ---
    def _reservoir_update(self, d: int, item: tuple):
        self.rand_seen[d] += 1
        s = self.rand_seen[d]
        rr = self.rand[d]
        if len(rr) < self.rand_k:
            rr.append(item)
            return True, None
        else:
            if s <= 0:
                return False, None
            j = self._rng.randrange(s)
            if j < self.rand_k:
                removed = rr[j]
                rr[j] = item
                return True, removed
        return False, None
                
    def estimate_final_rows(self) -> int:
        """
        최종 경계(=self.maxima 또는 cutoffs) 기준으로 재버킷팅했을 때
        실제로 기록될 row 수를 대략 추정.
        """
        total = 0
        for d in range(self.D):
            mx = float(self.maxima[d].item())
            # 최종 경계로 재버킷
            buckets = [[] for _ in range(self.num_deciles)]
            for seg0 in range(self.num_deciles):
                for it in self.heaps[d][seg0]:
                    seg2, _ = self._bucket_of(d, it[0], mx)
                    buckets[seg2].append(it)
            for it in self.rand[d]:
                seg2, _ = self._bucket_of(d, it[0], mx)
                buckets[seg2].append(it)
            # 각 버킷 top-k
            for seg in range(self.num_deciles):
                total += min(self.k, len(buckets[seg]))
            # random 파트(그냥 덧셈; 실제로는 decile 말단에 섞여 들어감)
            total += min(self.rand_k, len(self.rand[d]))
        return total
    # --- 업데이트 ---
    def update(
        self,
        acts_cpu: torch.Tensor,
        prov_cpu: torch.Tensor,
        *,
        stride_step: int,
        batch_max: Optional[torch.Tensor] = None,
    ):
        """
        acts_cpu: [N,D] float32
        prov_cpu: [N,?] long  (adapter schema 기반) 더 짧으면 보정
        batch_max: optional [D] tensor with per-feature maxima for this batch (any device)
        """
        N, D = acts_cpu.shape
        if batch_max is not None:
            max_vals = batch_max.detach()
            if max_vals.dim() != 1 or max_vals.numel() != D:
                raise ValueError(
                    f"batch_max must be a 1D tensor of length {D}, got shape {tuple(max_vals.shape)}"
               )
        else:
            max_vals = acts_cpu.max(dim=0).values
        if max_vals.device != self.maxima.device:
            max_vals = max_vals.to(self.maxima.device)
        if max_vals.dtype != self.maxima.dtype:
            max_vals = max_vals.to(self.maxima.dtype)
        self.maxima = torch.maximum(self.maxima, max_vals)
        
        prov_cols_len = self._prov_len
        for i in range(N):
            row = acts_cpu[i]
            nz_idx = torch.nonzero(row, as_tuple=True)[0]
            if nz_idx.numel() == 0:
                continue
            nz_units = nz_idx.tolist()
            nz_values = row[nz_idx].tolist()
            if prov_cpu.ndim == 2:
                prov_vals_raw = prov_cpu[i].tolist()
            else:
                prov_vals_raw = prov_cpu[i].view(-1).tolist()
            if not isinstance(prov_vals_raw, list):
                prov_vals_raw = [prov_vals_raw]

            prov_vals: List[int] = []
            for idx in range(prov_cols_len):
                if idx < len(prov_vals_raw):
                    prov_vals.append(int(prov_vals_raw[idx]))
                else:
                    name = self.prov_cols[idx] if idx < len(self.prov_cols) else None
                    prov_vals.append(int(_PROV_DEFAULTS.get(name, 0)))
 

            for d_idx, v_raw in zip(nz_units, nz_values):
                d = int(d_idx)
                v = float(v_raw)
                seg, _interval = self._bucket_of(d, v)
                heap = self.heaps[d][seg]
                # item tuple: (value, *prov_vals, stride_step)
                item = (v, *prov_vals, int(stride_step))

                dedupe_key = None
                if self._dedupe_key_fn is not None:
                    dedupe_key = self._dedupe_key_fn(item)
                    existing_entry = self._dedupe_lookup(d, dedupe_key)
                    if existing_entry is not None:
                        existing_item, _existing_loc = existing_entry
                        if v <= existing_item[0]:
                            continue
                        self._dedupe_remove_entry(d, dedupe_key, existing_entry)

                removed_from_heap = None
                inserted_to_heap = False
                if len(heap) < self.k_internal:
                    heapq.heappush(heap, item)
                    inserted_to_heap = True
                else:
                    if v > heap[0][0]:
                        removed_from_heap = heapq.heapreplace(heap, item)
                        inserted_to_heap = True

                if self._dedupe_key_fn is not None:
                    if removed_from_heap is not None:
                        self._dedupe_forget_item(d, removed_from_heap)
                    if inserted_to_heap:
                        self._dedupe_register(
                            d,
                            item,
                            dedupe_key,
                            location=(_DEDUPE_LOC_HEAP, seg),
                        )

                inserted_rand = False
                removed_rand = None
                if self.rand_k > 0:
                    allow_random = True
                    if self._dedupe_key_fn is not None and inserted_to_heap:
                        allow_random = False
                    if allow_random:
                        inserted_rand, removed_rand = self._reservoir_update(d, item)

                if self._dedupe_key_fn is not None:
                    if removed_rand is not None:
                        self._dedupe_forget_item(d, removed_rand)
                    if inserted_rand and not inserted_to_heap:
                        self._dedupe_register(
                            d,
                            item,
                            dedupe_key,
                            location=(_DEDUPE_LOC_RAND, None),
                        )

    # --- 최종 flush (Parquet로만) ---
    def finalize_and_write(self, *, progress_cb: Optional[Callable[[int], None]] = None) -> int:
        rows: List[Dict[str, Any]] = []
        prov_cols_len = self._prov_len

        def _prov_value(values: tuple, idx: Optional[int], default: int) -> int:
            if idx is None or idx >= len(values):
                return default
            return int(values[idx])

        def push_row(unit: int, rank_in_decile: int, seg: int, item: tuple):
            v = item[0]
            prov_vals = item[1:1 + prov_cols_len]
            s_step = item[-1]
            rows.append({
                "run_id": self.fp.run_id,
                "layer": self.layer,
                "unit": int(unit),
                "score": float(v),
                "decile": int(seg),
                "rank_in_decile": int(rank_in_decile),
                "sample_id": _prov_value(prov_vals, self._idx_sample_id, _PROV_DEFAULTS["sample_id"]),
                "frame_idx": _prov_value(prov_vals, self._idx_frame_idx, _PROV_DEFAULTS["frame_idx"]),
                "y": _prov_value(prov_vals, self._idx_y, _PROV_DEFAULTS["y"]),
                "x": _prov_value(prov_vals, self._idx_x, _PROV_DEFAULTS["x"]),
                "prompt_id": _prov_value(prov_vals, self._idx_prompt_id, _PROV_DEFAULTS["prompt_id"]),
                "uid": _prov_value(prov_vals, self._idx_uid, _PROV_DEFAULTS["uid"]),
                "stride_step": int(s_step),
                "meta_json": "",  # 필요시 채우기
            })

        # 후보 → 최종 경계로 재버킷팅 후 상위 k 선별
        total_wrote = 0
        for d in range(self.D):
            mx = float(self.maxima[d].item())
            buckets: List[List[tuple]] = [[] for _ in range(self.num_deciles)]
            # 기존 힙 + 랜덤을 모아 최종 경계 기준으로 재버킷팅
            for seg0 in range(self.num_deciles):
                for it in self.heaps[d][seg0]:
                    seg2, _ = self._bucket_of(d, it[0], mx)
                    buckets[seg2].append(it)
            for it in self.rand[d]:
                seg2, _ = self._bucket_of(d, it[0], mx)
                buckets[seg2].append(it)

            # 각 버킷에서 top-k
            for seg in range(self.num_deciles):
                cand = buckets[seg]
                if not cand: continue
                best = sorted(cand, key=lambda x: -x[0])[:self.k]
                for rnk, it in enumerate(best):
                    push_row(d, rnk, seg, it)
                    total_wrote += 1
                    if progress_cb: progress_cb(1)

            # random (선택) — 여기도 별도 파티션으로 섞고 싶으면 decile=-1 등으로 기록 가능
            if self.rand_k > 0 and self.rand[d]:
                rnd = sorted(self.rand[d], key=lambda x: -x[0])[:self.rand_k]
                for rnk, it in enumerate(rnd):
                    # random을 decile 그대로 두되 rank_in_decile 뒤쪽으로 기록
                    push_row(d, rnk + self.k, self.num_deciles - 1, it)
                    total_wrote += 1
                    if progress_cb: progress_cb(1)

        # 파케로 기록(원샷)
        if rows:
            self.ledger.write_rows(rows, rank=self.rank)
        return total_wrote
