# hook_helper.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple, Optional
import torch
import logging

logger = logging.getLogger(__name__)
SEP = "@"

# ---------- Tree walk / flatten ----------

def split_layer_and_branch(layer: str) -> Tuple[str, Optional[int]]:
    if SEP in layer:
        base, idx = layer.rsplit(SEP, 1)
        try:
            return base, int(idx)
        except ValueError:
            return layer, None
    return layer, None

# ---------- Shape utils ----------

def to_2d_tokens(t: torch.Tensor, layout_spec=None) -> Optional[torch.Tensor]:
    """
    BACKCOMPAT: flatten tensor for SAE using legacy heuristics.
    Accepts optional layout_spec for 5D+ tensors.
    """
    flat, _ = flatten_tensor_for_sae(t, layout_spec=layout_spec)
    return flat


def flatten_tensor_for_sae(t: torch.Tensor, layout_spec=None) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """
    Flatten arbitrary SAE inputs while recording enough metadata to later reconstruct the original shape.
    Returns (flattened_tensor, reshape_meta)

    For 2D/3D/4D tensors the existing heuristics are used.
    For 5D+ tensors a LayoutSpec is required; the feature axis is
    moved to the last position and all prefix dims are collapsed.
    """
    if not torch.is_tensor(t):
        return None, None
    ndim = t.ndim
    permute: Optional[Tuple[int, ...]]
    if ndim == 4:
        permute = (0, 2, 3, 1)
    elif ndim == 3:
        permute = None
    elif ndim == 2:
        permute = None
    elif ndim >= 5 and layout_spec is not None:
        # 5D+ support via LayoutSpec: move feature_axis to last
        feat_ax = layout_spec.feature_axis
        perm = list(range(ndim))
        perm.remove(feat_ax)
        perm.append(feat_ax)
        permute_5d = tuple(perm)
        tensor_perm = t.permute(*permute_5d).contiguous()
        prefix_shape = tuple(int(s) for s in tensor_perm.shape[:-1])
        flat = tensor_perm.reshape(-1, tensor_perm.shape[-1])
        meta = {
            "permute": permute_5d,
            "inverse_permute": _inverse_permute(permute_5d),
            "prefix_shape": prefix_shape,
        }
        return flat, meta
    else:
        return None, None
    tensor_perm = t.permute(*permute) if permute else t
    prefix_shape = tuple(int(s) for s in tensor_perm.shape[:-1])
    flat = tensor_perm.reshape(-1, tensor_perm.shape[-1])
    meta = {
        "permute": permute,
        "inverse_permute": _inverse_permute(permute) if permute else None,
        "prefix_shape": prefix_shape,
    }
    return flat, meta


def reshape_flat_sae_tensor(t: torch.Tensor, meta: Optional[Dict[str, Any]]) -> torch.Tensor:
    """
    Restore a flattened SAE tensor to its original layout using metadata from flatten_tensor_for_sae.

    Expected input shape before restore:
      - tensors captured from SAE anchors are flattened to (stack..., positions, feat)
      - "stack" can include step/frame from repeated calls, and possibly batch if stacking was used
      - positions = batch * tokens * spatial..., because flatten_tensor_for_sae flattens all prefix dims

    Meta contents:
      - prefix_shape: original prefix dims (e.g., (batch, tokens) or (batch, H, W))
      - inverse_permute: how to permute back channel/order if needed

    Restore logic:
      - reshape positions back into prefix_shape (broadcasting any leading stack dims)
      - apply inverse_permute on the tail while keeping leading stack dims intact
    """
    if meta is None:
        return t
    prefix_shape = tuple(meta.get("prefix_shape") or ())
    if prefix_shape:
        # allow leading stack dimensions (e.g., step/frame)
        if t.dim() >= 2:
            lead = t.shape[:-2]  # any stacked dims before positions/features
            tensor_perm = t.reshape(*lead, *prefix_shape, t.shape[-1])
        else:
            tensor_perm = t.reshape(*prefix_shape, t.shape[-1])
    else:
        tensor_perm = t
    inv = meta.get("inverse_permute")
    if inv:
        # keep leading stack dims intact, permute the tail
        if tensor_perm.dim() > len(inv):
            lead_dims = tensor_perm.shape[: tensor_perm.dim() - len(inv)]
            tail = tensor_perm.view(*lead_dims, *tensor_perm.shape[-len(inv) :])
            permuted_tail = tail.permute(*range(len(lead_dims)), *(len(lead_dims) + i for i in inv))
            tensor_perm = permuted_tail
        else:
            tensor_perm = tensor_perm.permute(*inv)
    return tensor_perm


def _inverse_permute(order: Tuple[int, ...]) -> Tuple[int, ...]:
    inverse = [0] * len(order)
    for idx, value in enumerate(order):
        inverse[value] = idx
    return tuple(inverse)


# ---------- GPU->CPU copy (chunked) ----------

def rows_per_chunk_for_mb(t: torch.Tensor, mb: int) -> int:
    """
    행(토큰) 당 바이트 수를 기준으로, 목표 MB에 맞는 row 청크 크기 계산.
    """
    bytes_target = max(1, int(mb) * 1024 * 1024)
    if t.ndim == 0:
        bytes_per_row = t.element_size()
        return max(1, bytes_target // max(1, bytes_per_row))
    rows = t.shape[0]
    elems_per_row = t[0].numel()
    bytes_per_row = elems_per_row * t.element_size()
    rpc = max(1, bytes_target // max(1, bytes_per_row))
    return max(1, rpc)


def ensure_stream_dict(obj, dev: torch.device) -> torch.cuda.Stream:
    """
    obj._copy_streams[dev.index] 존재 보장 후 반환.
    """
    if not hasattr(obj, "_copy_streams"):
        obj._copy_streams = {}
    if dev.index not in obj._copy_streams:
        obj._copy_streams[dev.index] = torch.cuda.Stream(device=dev)
    return obj._copy_streams[dev.index]


def copy_gpu_to_cpu_in_chunks(owner,
                              lname: str,
                              act_2d: torch.Tensor,
                              prov_full: Optional[torch.Tensor],
                              *,
                              chunk_mb: int,
                              transfer_dtype: torch.dtype) -> None:
    """
    GPU 2D 텐서를 비동기 스트림으로 CPU pinned 청크들로 복사해서
    owner._append_cpu_activation(...)에 추가. provenance도 동일한 슬라이스로 추가.
    - owner: UniversalActivationStore 인스턴스(append/accum/event 보유)
    """
    assert act_2d.is_cuda
    act = act_2d.detach().contiguous()
    
    if owner.enable_provenance:
        assert prov_full is not None, f"[{lname}] provenance enabled but prov_full is None"
        assert prov_full.shape[0] == act_2d.shape[0], \
            f"[{lname}] act/prov rows mismatch at hook: {act_2d.shape[0]} vs {prov_full.shape[0]}"
    dev = act.device
    copy_stream = ensure_stream_dict(owner, dev)
    producer = torch.cuda.current_stream(dev)

    rpc = rows_per_chunk_for_mb(act, chunk_mb)
    rows = int(act.shape[0])
    i = 0

    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(producer)
        while i < rows:
            j = min(rows, i + rpc)
            part = act[i:j]

            # 큐 dtype 통일: transfer_dtype로 강제
            cpu_part = torch.empty(
                (part.shape[0], part.shape[1]),
                dtype=transfer_dtype,
                device="cpu",
                pin_memory=True,
            )
            cpu_part.copy_(part.to(dtype=transfer_dtype), non_blocking=True)
            owner._append_cpu_activation(lname, cpu_part)

            if prov_full is not None:
                owner._prov_accum[lname].append(prov_full[i:j].contiguous())

            i = j

        act.record_stream(copy_stream)

    if not hasattr(owner, "_pending_copy_events"):
        owner._pending_copy_events = []
    evt = torch.cuda.Event()
    evt.record(copy_stream)
    owner._pending_copy_events.append(evt)


# ---------- Coalesce ----------

def coalesce_if_needed(owner, lname: str, threshold: int) -> None:
    """
    owner.activations[lname]가 threshold 이상이면 CPU에서 cat하여 1개로 병합.
    provenance도 동일.
    """
    acts = owner.activations.get(lname, [])
    if len(acts) < int(threshold):
        return

    # 필요시 이벤트 동기화
    # GPU->CPU 비동기 복사가 있을 때만 동기화; GPU 버퍼 경로는 건너뜀
    events = getattr(owner, "_pending_copy_events", None) if getattr(owner, "buffer_on_cpu", True) else None
    if events:
        for e in events:
            try:
                e.synchronize()
            except Exception:
                pass
        events.clear()
    else:
        # ensure the attribute exists to avoid future AttributeError
        owner._pending_copy_events = []

    try:
        # 모두 CPU/dtype 일치 가정(위에서 transfer_dtype로 강제)
        merged = torch.cat([t.contiguous() for t in acts], dim=0)
        owner.activations[lname] = [merged]

        if owner.enable_provenance:
            prov_list = owner._prov_accum.get(lname, [])
            if prov_list:
                pmerged = torch.cat([p.contiguous() for p in prov_list], dim=0)
                owner._prov_accum[lname] = [pmerged]
    except Exception as e:
        logger.warning(f"[coalesce] failed for {lname}: {e}")

def to_pinned_cpu(gpu_tensor: torch.Tensor) -> torch.Tensor:
    cpu = torch.empty_like(gpu_tensor, device="cpu", pin_memory=True)
    cpu.copy_(gpu_tensor, non_blocking=True)
    return cpu
