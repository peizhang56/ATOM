# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Row chunking and TP row sharding for the dense ``fp8_mqa_logits`` indexer.

The logits matrix is ``[rows, row_width]`` fp32 with ``row_width`` unbounded by
``max_num_batched_tokens`` (#1376), so the indexer prefill paths chunk the Q rows
and shard them across TP. Both call sites import these rules from here.
"""

from atom.utils import envs

# A buffer resource descriptor addresses at most 2^31 bytes, so aiter's
# fp8_mqa_logits drops to plain global load/store once the logits tensor reaches
# 2 GiB. On gfx950 that USE_BUFFER_STORE=False specialization of the gluon kernel
# does not survive codegen -- the AMDGCN backend trips
# `llvm/ADT/Sequence.h: Assertion 'Begin <= End'` and abort()s the process, with
# no Python traceback and every TP rank dying at once. Landing exactly on the cap
# is easy to do by accident: 16384 rows x 32768 committed tokens x 4 B is 2 GiB
# to the byte. Keep the buffer STRICTLY under the cap so the buffer-store
# specialization is the only one ever compiled.
_BUFFER_DESCRIPTOR_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

_LOGIT_BYTES = 4  # fp32
# The kernels tile rows by 128; keep chunk sizes on that grid.
_ROW_TILE = 128


def sparse_indexer_row_chunk(total_rows: int, row_width: int, budget_mb: int) -> int:
    """Rows to score per ``fp8_mqa_logits`` call for a ``[rows, row_width]`` buffer.

    ``budget_mb`` is the caller's soft byte budget (``0`` = no soft budget); the
    2 GiB buffer-descriptor cap applies either way, because exceeding it is a
    hard crash rather than a memory-pressure trade-off.

    Returns ``total_rows`` when one shot already fits. Otherwise the budget-derived
    row count rounded DOWN: to a multiple of 128 in the normal regime, avoiding
    coarse power-of-2 doubling; below 128 rows (extreme ``row_width``) to a
    power-of-2 floor, so it degrades 64/32/.../1 instead of collapsing to 1.
    """
    if total_rows <= 0 or row_width <= 0:
        return max(total_rows, 1)

    budget_bytes = budget_mb * 1024 * 1024
    cap_bytes = _BUFFER_DESCRIPTOR_LIMIT_BYTES
    if budget_bytes > 0:
        cap_bytes = min(budget_bytes, cap_bytes)

    # Exclusive bound: the kernel needs bytes < cap, not <=.
    max_rows = (cap_bytes - 1) // (row_width * _LOGIT_BYTES)
    if max_rows >= total_rows:
        return total_rows
    if max_rows >= _ROW_TILE:
        return (max_rows // _ROW_TILE) * _ROW_TILE
    return 1 << (max(1, max_rows).bit_length() - 1)


def indexer_row_shard(
    row_lo: int,
    row_hi: int,
    tp_size: int,
    tp_rank: int,
    kv_width: int,
    *,
    min_rows: int | None = None,
    min_kv_width: int | None = None,
) -> tuple[int, int, int]:
    """This rank's ``[lo, hi)`` slice of prefill rows, plus the all-gather stride.

    Row sharding is exact (row ``i`` needs only ``Q[i]`` and KV every rank holds)
    but not bit-identical, since ``top_k_per_row_prefill`` breaks ties arbitrarily.
    ``stride`` is the padded rows-per-rank, equal on every rank; 0 = do not shard.
    """
    # Gate on KV width, not rows: saved work and all-gather bytes both scale with
    # rows, so rows cancel. The row floor pays the all-gather's ~36-51 us latency.
    if min_rows is None:
        min_rows = envs.ATOM_INDEXER_ROW_SHARD_MIN_ROWS
    if min_kv_width is None:
        min_kv_width = envs.ATOM_INDEXER_ROW_SHARD_MIN_KV_WIDTH

    num_rows = row_hi - row_lo
    if tp_size <= 1 or num_rows < min_rows or kv_width < min_kv_width:
        return row_lo, row_hi, 0
    stride = (num_rows + tp_size - 1) // tp_size
    lo = row_lo + tp_rank * stride
    return min(lo, row_hi), min(lo + stride, row_hi), stride


def indexer_shard_row_windows(
    token_start: int,
    token_end: int,
    shard_lo: int,
    shard_hi: int,
    row_chunk_fn,
) -> list[tuple[int, int, int, int]]:
    """This shard's ``(q_start, q_end, row_start, row_end)`` windows in one chunk.

    ``q_*`` index per-token tensors, ``row_*`` the chunk-local window bounds; once
    sharded the two differ, and mixing them scores the wrong rows. Empty when the
    chunk misses this shard, so the caller can skip its KV gather.
    """
    lo = max(token_start, shard_lo)
    hi = min(token_end, shard_hi)
    if lo >= hi:
        return []
    row_chunk = row_chunk_fn(hi - lo)
    return [
        (
            q_start,
            min(q_start + row_chunk, hi),
            q_start - token_start,
            min(q_start + row_chunk, hi) - token_start,
        )
        for q_start in range(lo, hi, row_chunk)
    ]
