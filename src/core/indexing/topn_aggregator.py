# src/core/indexing/topn_aggregator.py
"""
TopNAggregator — feature별 단일 min-heap으로 global top-N 수집.

DecileTopKParquet 대비:
  - D개 단일 heap만 사용 (decile × D 대신) → 메모리/속도 향상
  - update() 내부에서 bucket 계산 없음 → hot path 단순화
  - Feature frequency 추적: _freq_counts[d] = unique sample count per feature

Parquet 호환: 기존 DECILES_SCHEMA 그대로 사용, decile=0 고정, rank_in_decile=global rank.
"""
from __future__ import annotations

import heapq
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from .decile_aggregator import RunFingerprint
from .decile_parquet_ledger import DecileParquetLedger

_PROV_DEFAULTS = {
    "sample_id": 0,
    "frame_idx": 0,
    "y": -1,
    "x": -1,
    "prompt_id": 0,
    "uid": -1,
}


class TopNAggregator:
    """
    Per-feature global top-N aggregator with optional frequency tracking.

    Interface contract (same as DecileTopKParquet):
      - update(acts_cpu, prov_cpu, *, stride_step, batch_max)
      - finalize_and_write(*, progress_cb)  → parquet rows written
      - state_dict() / load_state_dict(sd)
      - get_feature_frequencies()           → Dict[int, int]
    """

    def __init__(
        self,
        *,
        dict_size: int,
        top_n: int,
        layer_name: str,
        fp: RunFingerprint,
        ledger: DecileParquetLedger,
        prov_cols: Sequence[str],
        track_frequency: bool = True,
        dedupe_key_fn: Optional[Callable[[tuple], Any]] = None,
        slack: int = 4,
        rank: int = 0,
    ):
        self.D = int(dict_size)
        self.top_n = int(top_n)
        self.n_internal = int(top_n + slack)
        self.layer = str(layer_name)
        self.fp = fp
        self.ledger = ledger
        self.rank = int(rank)
        self.track_frequency = bool(track_frequency)

        # Per-feature single min-heap: item = (value, *prov_vals, stride_step)
        self.heaps: List[List[tuple]] = [[] for _ in range(self.D)]

        # Maxima tracking (for compatibility with checkpoint/DDP reduce)
        self.maxima = torch.zeros(self.D, dtype=torch.float32)

        # Provenance column mapping
        self.prov_cols = tuple(prov_cols)
        self._prov_len = len(self.prov_cols)
        self._prov_index = {name: idx for idx, name in enumerate(self.prov_cols)}
        self._idx_sample_id = self._prov_index.get("sample_id")
        self._idx_frame_idx = self._prov_index.get("frame_idx")
        self._idx_y = self._prov_index.get("y")
        self._idx_x = self._prov_index.get("x")
        self._idx_prompt_id = self._prov_index.get("prompt_id")
        self._idx_uid = self._prov_index.get("uid")
        self._prov_default_row = tuple(
            int(_PROV_DEFAULTS.get(name, 0)) for name in self.prov_cols
        )

        # Feature frequency: count unique sample_ids per feature
        if self.track_frequency:
            self._freq_sets: List[Set[int]] = [set() for _ in range(self.D)]
        else:
            self._freq_sets = []

        self._total_samples_seen: Set[int] = set()

        # Dedupe support (same pattern as DecileTopKParquet)
        self._dedupe_key_fn = dedupe_key_fn
        if dedupe_key_fn is not None:
            self._dedupe_store: Optional[List[Dict[Any, tuple]]] = [
                dict() for _ in range(self.D)
            ]
        else:
            self._dedupe_store = None

        # Checkpoint counters
        self.run_epoch = 0
        self.run_b_in_epoch = 0
        self.global_steps = 0

        self._heap_mins_np = np.full(self.D, -np.inf, dtype=np.float64)
        self._heap_sizes_np = np.zeros(self.D, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Dedupe helpers (simplified vs DecileTopKParquet — heap only)
    # ------------------------------------------------------------------ #
    def _dedupe_lookup(self, d: int, key: Any) -> Optional[tuple]:
        if self._dedupe_store is None:
            return None
        return self._dedupe_store[d].get(key)

    def _dedupe_register(self, d: int, item: tuple, key: Any) -> None:
        if self._dedupe_store is None:
            return
        self._dedupe_store[d][key] = item

    def _dedupe_forget_item(self, d: int, item: tuple) -> None:
        if self._dedupe_store is None:
            return
        key = self._dedupe_key_fn(item)
        store = self._dedupe_store[d]
        entry = store.get(key)
        if entry is not None and entry == item:
            store.pop(key, None)

    def _dedupe_remove_from_heap(self, d: int, item: tuple) -> None:
        heap = self.heaps[d]
        for idx, candidate in enumerate(heap):
            if candidate == item:
                heap.pop(idx)
                heapq.heapify(heap)
                return

    # ------------------------------------------------------------------ #
    # Update (hot path) — vectorized sparse candidate filtering
    # ------------------------------------------------------------------ #
    def update(
        self,
        acts_cpu: torch.Tensor,
        prov_cpu: torch.Tensor,
        *,
        stride_step: int,
        batch_max: Optional[torch.Tensor] = None,
    ):
        """
        acts_cpu: [N, D] float32
        prov_cpu: [N, ?] long

        Vectorized hot path: one torch.nonzero extracts sparse candidates, then NumPy
        filters them against per-feature heap thresholds before Python touches them.
        We also pre-pack provenance once per token so the inner loop no longer pays
        repeated ndarray->list conversions for every active feature firing.
        """
        N, D = acts_cpu.shape

        # Update maxima
        if batch_max is not None:
            max_vals = batch_max.detach()
            if max_vals.dim() != 1 or max_vals.numel() != D:
                raise ValueError(
                    f"batch_max must be a 1D tensor of length {D}, "
                    f"got shape {tuple(max_vals.shape)}"
                )
            if max_vals.device != self.maxima.device:
                max_vals = max_vals.to(self.maxima.device)
            if max_vals.dtype != self.maxima.dtype:
                max_vals = max_vals.to(self.maxima.dtype)
        else:
            max_vals = acts_cpu.max(dim=0).values
        self.maxima = torch.maximum(self.maxima, max_vals)

        prov_cols_len = self._prov_len
        track_freq = self.track_frequency
        freq_sets = self._freq_sets
        total_seen = self._total_samples_seen
        idx_sample_id = self._idx_sample_id
        stride_step_int = int(stride_step)
        prov_default_row = self._prov_default_row

        # ── provenance ──
        if prov_cpu.ndim == 2:
            prov_np: np.ndarray = prov_cpu.numpy()   # [N, prov_len]
        else:
            prov_np = prov_cpu.view(N, -1).numpy()
        prov_ncols = prov_np.shape[1]

        if prov_ncols == prov_cols_len:
            prov_padded = prov_np
        else:
            prov_padded = np.empty((N, prov_cols_len), dtype=np.int64)
            if prov_cols_len:
                prov_padded[:] = np.asarray(prov_default_row, dtype=np.int64)
            if prov_ncols:
                prov_padded[:, : min(prov_ncols, prov_cols_len)] = prov_np[
                    :, : min(prov_ncols, prov_cols_len)
                ]

        # ── track total_seen: O(N) ──
        if track_freq and idx_sample_id is not None and idx_sample_id < prov_ncols:
            total_seen.update(int(x) for x in prov_padded[:, idx_sample_id].tolist())

        # Timing: materialize per-token provenance tuples once; each token fires many
        # features, so reusing the packed tuple is cheaper than per-candidate tolist().
        prov_rows: List[tuple[int, ...]] = [
            tuple(row) for row in prov_padded.tolist()
        ]

        # ── per-feature batch max for early-skip ──
        bmax_np: np.ndarray = max_vals.numpy()  # [D] already on CPU

        # Heap-mins array: maintained for O(D) vectorized early-skip comparison
        heap_mins: np.ndarray = self._heap_mins_np
        heap_sizes: np.ndarray = self._heap_sizes_np
        candidate_thresholds = np.where(
            heap_sizes >= self.n_internal,
            heap_mins,
            -np.inf,
        )

        # ── vectorized nonzero: one C++ call → M (tok_idx, feat_idx, val) triples ──
        active_mask_d = bmax_np > candidate_thresholds  # [D] bool
        if not active_mask_d.any():
            return

        nz = torch.nonzero(acts_cpu, as_tuple=False)  # [M, 2]: M ≈ N×K active entries
        if nz.numel() == 0:
            return
        feat_ids_np: np.ndarray = nz[:, 1].numpy()     # [M] global feature indices
        tok_ids_np: np.ndarray  = nz[:, 0].numpy()     # [M]
        vals_np: np.ndarray     = acts_cpu[nz[:, 0], nz[:, 1]].numpy()  # [M]

        # Timing: batch filter in NumPy so the Python loop only sees candidates that
        # can still improve a heap. Underfilled heaps use -inf, so we preserve safety.
        candidate_mask: np.ndarray = (
            active_mask_d[feat_ids_np]
            & (vals_np > candidate_thresholds[feat_ids_np])
        )
        if not candidate_mask.any():
            return
        feat_ids_f = feat_ids_np[candidate_mask]
        tok_ids_f  = tok_ids_np[candidate_mask]
        vals_f     = vals_np[candidate_mask]

        # Sort by feature id and descending value so each feature can early-break once
        # its heap threshold is met instead of scanning all surviving candidates.
        sort_order = np.lexsort((-vals_f, feat_ids_f))
        feat_ids_s = feat_ids_f[sort_order]
        tok_ids_s  = tok_ids_f[sort_order]
        vals_s     = vals_f[sort_order]

        # Pre-convert to Python lists: list element access is faster than numpy scalar
        # access inside the remaining per-candidate heap update loop.
        vals_list: list  = vals_s.tolist()
        tok_list: list   = tok_ids_s.tolist()

        # Process per feature using counts from np.unique (avoids np.append)
        unique_feats, boundaries, counts = np.unique(feat_ids_s, return_index=True, return_counts=True)
        n_internal = self.n_internal
        heaps      = self.heaps
        freq_sample_ids = None
        if track_freq and idx_sample_id is not None and idx_sample_id < prov_cols_len:
            freq_sample_ids = prov_padded[tok_ids_s, idx_sample_id]
        allow_batch_cap = self._dedupe_key_fn is None

        for uf_idx in range(len(unique_feats)):
            d     = int(unique_feats[uf_idx])
            start = int(boundaries[uf_idx])
            end   = start + int(counts[uf_idx])
            if freq_sample_ids is not None:
                freq_sets[d].update(freq_sample_ids[start:end].tolist())
            if allow_batch_cap and end - start > n_internal:
                end = start + n_internal
            heap  = heaps[d]
            heap_min = float(heap_mins[d])
            heap_size = int(heap_sizes[d])

            for m in range(start, end):
                v = vals_list[m]  # Python float from pre-converted list
                if heap_size >= n_internal and v <= heap_min:
                    break  # sorted descending within feature

                tok_idx   = tok_list[m]   # Python int from pre-converted list
                prov_vals = prov_rows[tok_idx]

                item = (v, *prov_vals, stride_step_int)

                # Dedupe check
                dedupe_key = None
                if self._dedupe_key_fn is not None:
                    dedupe_key = self._dedupe_key_fn(item)
                    existing = self._dedupe_lookup(d, dedupe_key)
                    if existing is not None:
                        if v <= existing[0]:
                            continue
                        self._dedupe_remove_from_heap(d, existing)
                        self._dedupe_store[d].pop(dedupe_key, None)
                        heap_size = len(heap)
                        heap_min = heap[0][0] if heap else -np.inf

                # Min-heap insert
                removed = None
                if heap_size < n_internal:
                    heapq.heappush(heap, item)
                    heap_size += 1
                elif v > heap_min:
                    removed = heapq.heapreplace(heap, item)
                else:
                    break

                # Update heap_min after any heap change
                heap_min = heap[0][0]
                heap_mins[d] = heap_min
                heap_sizes[d] = heap_size

                # Dedupe bookkeeping
                if self._dedupe_key_fn is not None:
                    if removed is not None:
                        self._dedupe_forget_item(d, removed)
                    self._dedupe_register(d, item, dedupe_key)
            heap_mins[d] = heap_min if heap else -np.inf
            heap_sizes[d] = heap_size

    # ------------------------------------------------------------------ #
    # Finalize & write to Parquet
    # ------------------------------------------------------------------ #
    def finalize_and_write(
        self, *, progress_cb: Optional[Callable[[int], None]] = None
    ) -> int:
        prov_cols_len = self._prov_len
        total_wrote = self.estimate_final_rows()
        if total_wrote == 0:
            return 0

        units = [0] * total_wrote
        scores = [0.0] * total_wrote
        ranks = [0] * total_wrote
        sample_ids = [0] * total_wrote
        frame_idxs = [0] * total_wrote
        ys = [0] * total_wrote
        xs = [0] * total_wrote
        prompt_ids = [0] * total_wrote
        uids = [0] * total_wrote
        stride_steps = [0] * total_wrote

        write_idx = 0
        idx_sample_id = self._idx_sample_id
        idx_frame_idx = self._idx_frame_idx
        idx_y = self._idx_y
        idx_x = self._idx_x
        idx_prompt_id = self._idx_prompt_id
        idx_uid = self._idx_uid
        default_row = self._prov_default_row

        for d in range(self.D):
            heap = self.heaps[d]
            if not heap:
                continue
            # Sort descending by score, take top_n
            best = sorted(heap, key=lambda x: -x[0])[: self.top_n]
            n_best = len(best)
            if n_best == 0:
                continue

            # Timing: fill column buffers directly and hand them to the ledger in one
            # write. This removes the finalize row-dict loop plus ledger's old
            # schema x rows dict-walk, which dominated cProfile on large flushes.
            for rnk, item in enumerate(best):
                prov_vals = item[1 : 1 + prov_cols_len]
                if len(prov_vals) < prov_cols_len:
                    prov_vals = prov_vals + default_row[len(prov_vals) :]

                units[write_idx] = int(d)
                scores[write_idx] = float(item[0])
                ranks[write_idx] = int(rnk)
                sample_ids[write_idx] = (
                    int(prov_vals[idx_sample_id])
                    if idx_sample_id is not None
                    else _PROV_DEFAULTS["sample_id"]
                )
                frame_idxs[write_idx] = (
                    int(prov_vals[idx_frame_idx])
                    if idx_frame_idx is not None
                    else _PROV_DEFAULTS["frame_idx"]
                )
                ys[write_idx] = (
                    int(prov_vals[idx_y])
                    if idx_y is not None
                    else _PROV_DEFAULTS["y"]
                )
                xs[write_idx] = (
                    int(prov_vals[idx_x])
                    if idx_x is not None
                    else _PROV_DEFAULTS["x"]
                )
                prompt_ids[write_idx] = (
                    int(prov_vals[idx_prompt_id])
                    if idx_prompt_id is not None
                    else _PROV_DEFAULTS["prompt_id"]
                )
                uids[write_idx] = (
                    int(prov_vals[idx_uid])
                    if idx_uid is not None
                    else _PROV_DEFAULTS["uid"]
                )
                stride_steps[write_idx] = int(item[-1])
                write_idx += 1

            if progress_cb is not None:
                progress_cb(n_best)

        columns = {
            "run_id": [self.fp.run_id] * write_idx,
            "layer": [self.layer] * write_idx,
            "unit": units[:write_idx],
            "score": scores[:write_idx],
            "decile": [0] * write_idx,
            "rank_in_decile": ranks[:write_idx],
            "sample_id": sample_ids[:write_idx],
            "frame_idx": frame_idxs[:write_idx],
            "y": ys[:write_idx],
            "x": xs[:write_idx],
            "prompt_id": prompt_ids[:write_idx],
            "uid": uids[:write_idx],
            "stride_step": stride_steps[:write_idx],
            "meta_json": [""] * write_idx,
        }
        self.ledger.write_rows(columns, rank=self.rank)
        return write_idx

    # ------------------------------------------------------------------ #
    # Feature frequency
    # ------------------------------------------------------------------ #
    def get_feature_frequencies(self) -> Dict[int, int]:
        """Return {feature_id: num_unique_samples} for features with count > 0."""
        if not self.track_frequency:
            return {}
        return {d: len(s) for d, s in enumerate(self._freq_sets) if len(s) > 0}

    def get_total_samples_seen(self) -> int:
        return len(self._total_samples_seen)

    # ------------------------------------------------------------------ #
    # Checkpoint state
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        freq_counts = None
        if self.track_frequency:
            # Save as counts only (sets can be huge); for resume we lose
            # exact dedup but counts are approximate-correct
            freq_counts = [len(s) for s in self._freq_sets]
        return {
            "version": 1,
            "aggregator_type": "topn",
            "D": self.D,
            "top_n": self.top_n,
            "n_internal": self.n_internal,
            "heaps": self.heaps,
            "maxima": self.maxima,
            "layer": self.layer,
            "fp": self.fp.__dict__,
            "run_epoch": self.run_epoch,
            "run_b_in_epoch": self.run_b_in_epoch,
            "global_steps": self.global_steps,
            "prov_cols": list(self.prov_cols),
            "track_frequency": self.track_frequency,
            "freq_counts": freq_counts,
            "total_samples_seen": len(self._total_samples_seen),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.D = int(sd["D"])
        self.top_n = int(sd["top_n"])
        self.n_internal = int(sd.get("n_internal", self.top_n + 4))
        self.heaps = sd["heaps"]
        # Rebuild _heap_mins_np from loaded heaps
        D_loaded = int(sd["D"])
        self._heap_mins_np = np.full(D_loaded, -np.inf, dtype=np.float64)
        self._heap_sizes_np = np.zeros(D_loaded, dtype=np.int32)
        for _d, _h in enumerate(self.heaps):
            if _h:
                self._heap_mins_np[_d] = _h[0][0]
                self._heap_sizes_np[_d] = len(_h)
        self.maxima = (
            sd["maxima"].clone().cpu()
            if isinstance(sd["maxima"], torch.Tensor)
            else torch.as_tensor(sd["maxima"], dtype=torch.float32)
        ).clamp_min(0)
        self.run_epoch = int(sd.get("run_epoch", 0))
        self.run_b_in_epoch = int(sd.get("run_b_in_epoch", 0))
        self.global_steps = int(sd.get("global_steps", 0))

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
            self._prov_default_row = tuple(
                int(_PROV_DEFAULTS.get(name, 0)) for name in self.prov_cols
            )

        # Frequency: exact sets are not serialized (too large).
        # On resume, frequency tracking restarts from zero.
        # The final frequency will only reflect post-resume data.
        self.track_frequency = sd.get("track_frequency", self.track_frequency)
        if self.track_frequency:
            self._freq_sets = [set() for _ in range(self.D)]
            prev_counts = sd.get("freq_counts")
            if prev_counts is not None:
                import logging
                logging.getLogger("sae_index").warning(
                    "[TopNAggregator] Resumed from checkpoint — frequency "
                    "tracking restarted from zero (previous counts not restored)."
                )
        self._total_samples_seen = set()

        # Rebuild dedupe store from heaps
        if self._dedupe_key_fn is not None:
            self._dedupe_store = [dict() for _ in range(self.D)]
            for d in range(self.D):
                store_d = self._dedupe_store[d]
                for item in self.heaps[d]:
                    store_d[self._dedupe_key_fn(item)] = item

    # ------------------------------------------------------------------ #
    # Estimate rows (for progress reporting compatibility)
    # ------------------------------------------------------------------ #
    def estimate_final_rows(self) -> int:
        total = 0
        for d in range(self.D):
            total += min(self.top_n, len(self.heaps[d]))
        return total
