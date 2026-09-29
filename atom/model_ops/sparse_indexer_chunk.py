# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Row chunking and TP row sharding for the dense ``fp8_mqa_logits`` indexer.

Every sparse-MLA indexer prefill path (native DeepSeek V3.2/V4, GLM-5.x, and the
vLLM plugin) scores queries against the committed KV through a dense
``[rows, row_width]`` fp32 logits matrix. ``row_width`` is the sum of all
co-scheduled prefill contexts, which ``max_num_batched_tokens`` does not bound,
so a burst of long-context requests can push one allocation to tens of GiB
(issue #1376). Those paths therefore split the Q rows into chunks; this module
owns the one rule they all need to agree on.

It owns the TP **row shard** for the same reason. The indexer is replicated per
TP rank, so every rank scores every prefill row: TP x the work for 1x the
result. Two call sites implement that shard -- ``atom/models/deepseek_v2.py``
and ``atom/plugin/vllm/attention/layer_sparse_mla.py`` -- against different
parallel-state modules, and a divergence between them is a wrong-rows bug that
does not crash. The algebra therefore lives here once, takes ``tp_size`` and
``tp_rank`` as plain integers so it needs neither module, and both call sites
import it.
"""

import torch

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


# ---------------------------------------------------------------------------
# TP row shard
# ---------------------------------------------------------------------------

# What the shard trades: it saves ``(1 - 1/tp) x`` the replicated
# ``fp8_mqa_logits`` time for ``[rows, kv_width]``, and costs one all-gather of
# ``[rows/tp, topk]`` int32. Both scale with ``rows``, so **``rows`` cancels and
# ``kv_width`` is what decides** -- measured on 8x gfx950 at GLM-5.3's indexer
# shape (index_n_heads 32, index_head_dim 128, index_topk 2048), saved/cost by
# ``kv_width``, each column a row count from 64 to 16384:
#
#   kv_width  4096   8192  16384  32768  65536  115000
#   ratio     0.42x  0.63x  1.22x  1.78x  6.29x  1.78x .. 23.19x
#             ------ loss ------|--------------- win ---------------
#
# The ratio is flat in ``rows`` at fixed width (at 16384 it spans 1.22-2.34x
# over rows 64..16384) and crosses 1.0 between 8192 and 16384 at EVERY row
# count. Gate on width at 16384, the first column that wins outright.
#
# Gating on rows alone is what this replaces: at kv_width 4096 the shard is a
# measured 2.4x LOSS on the indexer no matter how many rows there are, and a
# rows-only gate takes it.
#
# Read once at import, as the logits budget above is: envs re-evaluates
# os.getenv on every attribute access and this sits in the prefill hot path.
_INDEXER_ROW_SHARD_MIN_KV_WIDTH = envs.ATOM_INDEXER_ROW_SHARD_MIN_KV_WIDTH

# The all-gather has a ~36-51 us latency floor, so a shard also has to be tall
# enough to pay it. At or above the width gate even 64 rows clears it; 256 is
# the measured floor with margin for small-collective jitter.
_INDEXER_ROW_SHARD_MIN_ROWS = envs.ATOM_INDEXER_ROW_SHARD_MIN_ROWS


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
    """This rank's ``[lo, hi)`` slice of prefill rows ``[row_lo, row_hi)``.

    Sharding by ROW is exact: row ``i``'s logits need only ``Q[i]`` and the
    committed KV, which every rank already holds, so a sharded rank computes a
    row from the same inputs and the same window as the replicated path, with
    no merge and nothing to gate on accuracy. Only the int32 top-k indices are
    all-gathered. (A KV-COLUMN split would instead need a cross-rank top-k merge
    and would gather values as well as indices.)

    Exact is not bit-identical, and the difference is the kernel's, not the
    shard's: ``top_k_per_row_prefill`` returns its k indices in a
    non-deterministic ORDER, and when two KV positions carry bitwise-equal
    logits at the k-th largest it picks one of them arbitrarily. Two replicated
    calls on identical inputs already disagree that way. Measured over 8x gfx950
    at GLM-5.3's shape, sharded vs replicated differs on <= 1 row in 8001, every
    such row an exact tie with an identical selected-logit multiset -- inside
    the replicated path's own run-to-run noise. Anything comparing indexer
    output across runs has to compare sets, not bits.

    ``row_lo``/``row_hi`` are whatever space the caller indexes its Q rows in --
    absolute batch token offsets in the vLLM plugin, prefill-relative rows in
    the native path. The result is returned in that same space.

    ``kv_width`` is the committed-KV column count the rows are scored against,
    averaged per row when chunks differ; it is what decides whether the shard
    pays (see ``_INDEXER_ROW_SHARD_MIN_KV_WIDTH``).

    ``stride`` is the padded rows-per-rank, so every rank contributes an equal
    shard to ``all_gather`` and only the last non-empty rank is short -- its
    block sits at the end of the gathered buffer, so the caller's ``[:num_rows]``
    slice drops padding and never a row. ``stride == 0`` means do not shard.
    """
    min_rows = _INDEXER_ROW_SHARD_MIN_ROWS if min_rows is None else min_rows
    if min_kv_width is None:
        min_kv_width = _INDEXER_ROW_SHARD_MIN_KV_WIDTH

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
    """This shard's row windows within one prefill chunk.

    Yields ``(q_start, q_end, row_start, row_end)``. ``q_*`` index the batch's
    per-token tensors (``q_fp8``, ``weights``); ``row_*`` index the chunk-local
    per-row window bounds (``cu_seqlen_ks``/``ke``). Unsharded those two spaces
    differ by a constant the loop can ignore; sharded they do not, and confusing
    them silently scores the wrong rows.

    Empty when the chunk lies outside this shard, so the caller can skip the
    chunk's KV gather entirely. ``row_chunk_fn`` is called with THIS shard's row
    count, not the chunk's: the logits budget is per allocation, so the divisor
    has to shrink with the shard.
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


_shard_buffers: dict = {}


def indexer_shard_buffers(stride, topk, dtype, device, world):
    """Persistent shard and gather buffers, grown on demand and never freed.

    Allocating both per call is two allocations per indexer layer per step, the
    gather output being ``world`` x the shard. The caching allocator hides that
    until free HBM runs low, then it falls through to the driver and every
    allocation synchronizes the device. That is not hypothetical here: the DSA
    indexer's all-gather is exactly where HBM exhaustion lands, and it has taken
    the engine down with ``ncclUnhandledCudaError: Failed to CUDA calloc
    6291456 bytes`` raised from this call site.

    No fill: gathered row ``r * stride + i`` is global row ``r * stride + i``, so
    the caller's slice keeps only rows a rank wrote and drops the padding
    unread. Stale contents are unreachable rather than merely unlikely.
    """
    key = (topk, dtype, device, world)
    buf = _shard_buffers.get(key)
    if buf is None or buf[0].shape[0] < stride:
        # Grow only, so the steady state is allocation-free.
        buf = (
            torch.empty((stride, topk), dtype=dtype, device=device),
            torch.empty((stride * world, topk), dtype=dtype, device=device),
        )
        _shard_buffers[key] = buf
    return buf[0][:stride], buf[1][: stride * world]
