# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""TP row sharding of the DSA prefill indexer (`sparse_indexer_chunk`).

The shard fails silently -- wrong rows, not a crash -- so these check that the
gather reconstructs every row exactly once, that ``q_*`` and ``row_*`` offsets
stay distinct, and that the shard is only taken where it was measured to pay.
"""

import pytest
import torch

from atom.model_ops.sparse_indexer_chunk import (
    indexer_row_shard,
    indexer_shard_row_windows,
)

_TP_SIZES = (1, 2, 4, 8, 16)
# 11588 is one request's prefill rows at ISL 115000; 255/256 straddle the
# minimum-rows gate.
_ROW_COUNTS = (0, 1, 255, 256, 257, 511, 512, 1000, 2048, 11588, 16384)

# Comfortably past the width gate, so these cases turn on the row algebra.
_WIDE = 1 << 20

_PAD = object()


def _shards(num_rows, tp_size, kv_width=_WIDE, row_lo=0):
    """Every rank's ``(lo, hi, stride)`` for one shard decision."""
    return [
        indexer_row_shard(row_lo, row_lo + num_rows, tp_size, rank, kv_width)
        for rank in range(tp_size)
    ]


# ---------------------------------------------------------------------------
# The gather has to reconstruct every row, exactly once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", _TP_SIZES)
@pytest.mark.parametrize("num_rows", _ROW_COUNTS)
def test_gather_reconstructs_every_row_exactly_once(tp_size, num_rows):
    shards = _shards(num_rows, tp_size)
    stride = shards[0][2]
    if not stride:
        assert all(s == (0, num_rows, 0) for s in shards)
        assert tp_size <= 1 or num_rows < 256
        return

    # all_gather_into_tensor is rank-major, so rank r's local row i lands at
    # r * stride + i. Anything a rank did not write stays padding.
    gathered = [_PAD] * (stride * tp_size)
    for rank, (lo, hi, rank_stride) in enumerate(shards):
        assert rank_stride == stride, "every rank must contribute an equal shard"
        assert hi - lo <= stride
        for i in range(hi - lo):
            assert gathered[rank * stride + i] is _PAD, "two ranks wrote one slot"
            gathered[rank * stride + i] = lo + i

    assert gathered[:num_rows] == list(range(num_rows))


@pytest.mark.parametrize("tp_size", _TP_SIZES)
@pytest.mark.parametrize("num_rows", _ROW_COUNTS)
def test_only_the_last_non_empty_rank_is_short(tp_size, num_rows):
    """The padding has to sit at the end, or the slice would drop a real row."""
    shards = _shards(num_rows, tp_size)
    stride = shards[0][2]
    if not stride:
        return
    sizes = [hi - lo for lo, hi, _ in shards]
    assert sum(sizes) == num_rows
    full = [i for i, n in enumerate(sizes) if n == stride]
    rest = [i for i, n in enumerate(sizes) if n != stride]
    assert full == list(range(len(full))), "full shards must come first"
    assert len([i for i in rest if sizes[i] > 0]) <= 1


@pytest.mark.parametrize("tp_size", _TP_SIZES)
@pytest.mark.parametrize("num_rows", _ROW_COUNTS)
@pytest.mark.parametrize("base", (0, 37, 4096))
def test_a_non_zero_base_only_translates_the_shard(tp_size, num_rows, base):
    """An off-by-``num_decode_tokens`` in that space would score decode rows."""
    at_zero = _shards(num_rows, tp_size)
    at_base = _shards(num_rows, tp_size, row_lo=base)
    for (lo, hi, stride), (b_lo, b_hi, b_stride) in zip(at_zero, at_base):
        assert (b_lo - base, b_hi - base, b_stride) == (lo, hi, stride)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_narrow_kv_is_not_sharded_at_any_row_count():
    """Below the width gate the shard is a measured loss, so it must not run."""
    for num_rows in (256, 2048, 16384):
        for width in (1024, 4096, 8192):
            _, _, stride = indexer_row_shard(0, num_rows, 8, 0, width)
            assert stride == 0, f"sharded at rows={num_rows} width={width}"


def test_wide_kv_shards_once_tall_enough():
    for width in (16384, 115000):
        assert indexer_row_shard(0, 255, 8, 0, width)[2] == 0, "below the row floor"
        assert indexer_row_shard(0, 256, 8, 0, width)[2] != 0


def test_tp1_never_shards():
    assert indexer_row_shard(0, 16384, 1, 0, _WIDE) == (0, 16384, 0)


def test_thresholds_are_overridable():
    """The env knobs are the documented way to A/B the shard in one binary."""
    assert indexer_row_shard(0, 16384, 8, 0, 4096, min_kv_width=4096)[2] != 0
    assert indexer_row_shard(0, 16384, 8, 0, _WIDE, min_kv_width=1 << 40)[2] == 0
    assert indexer_row_shard(0, 16384, 8, 0, _WIDE, min_rows=1 << 30)[2] == 0


# ---------------------------------------------------------------------------
# Row windows inside a chunk
# ---------------------------------------------------------------------------


def _windows(token_start, token_end, shard_lo, shard_hi, row_chunk):
    return indexer_shard_row_windows(
        token_start, token_end, shard_lo, shard_hi, lambda rows: row_chunk
    )


@pytest.mark.parametrize("tp_size", (1, 4, 8))
@pytest.mark.parametrize("row_chunk", (1, 7, 128, 4096))
def test_windows_tile_the_prefill_range_exactly_once_across_ranks(tp_size, row_chunk):
    """Union over ranks and chunks must be the whole range, with no overlap."""
    # Three contiguous chunks, as _build_indexer produces them.
    chunks = [(100, 1100), (1100, 4200), (4200, 9000)]
    prefill_lo, prefill_hi = chunks[0][0], chunks[-1][1]
    num_rows = prefill_hi - prefill_lo

    seen = {}
    for rank in range(tp_size):
        lo, hi, _ = indexer_row_shard(prefill_lo, prefill_hi, tp_size, rank, _WIDE)
        for c_start, c_end in chunks:
            for q_start, q_end, row_start, row_end in _windows(
                c_start, c_end, lo, hi, row_chunk
            ):
                assert q_start < q_end
                # q_* index batch-space tensors, row_* the chunk-local window
                # bounds. Mixing them is the silent wrong-rows bug.
                assert row_start == q_start - c_start
                assert row_end == q_end - c_start
                assert 0 <= row_start < row_end <= c_end - c_start
                assert q_end - q_start <= row_chunk
                for q in range(q_start, q_end):
                    assert q not in seen, f"row {q} scored twice"
                    seen[q] = rank

    assert sorted(seen) == list(range(prefill_lo, prefill_hi))
    assert len(seen) == num_rows


def test_a_chunk_outside_the_shard_yields_no_windows():
    assert _windows(0, 100, 200, 300, 16) == []
    assert _windows(400, 500, 200, 300, 16) == []
    # Touching but not overlapping.
    assert _windows(200, 300, 300, 400, 16) == []


def test_windows_clip_to_the_shard_not_the_chunk():
    # Shard covers the middle of the chunk only.
    got = _windows(0, 100, 30, 70, 1000)
    assert got == [(30, 70, 30, 70)]


def test_row_chunk_is_sized_from_the_shard_not_the_chunk():
    """The logits budget is per allocation, so the divisor shrinks with the shard."""
    seen = []
    indexer_shard_row_windows(0, 1000, 200, 300, lambda rows: seen.append(rows) or 50)
    assert seen == [100], "row_chunk_fn must see this shard's rows, not the chunk's"


# ---------------------------------------------------------------------------
# End to end: sharded output must equal replicated output
# ---------------------------------------------------------------------------


def _run_prefill(chunks, tp_size, num_tokens, topk, width=_WIDE):
    """Drive the real helpers over a fake batch and return the gathered indices.

    Each row's "top-k" is a function of its global row id, so a row scored from
    the wrong offset shows up in the output.
    """
    prefill_lo, prefill_hi = chunks[0][0], chunks[-1][1]
    out = torch.full((num_tokens, topk), -1, dtype=torch.int32)
    stride = indexer_row_shard(prefill_lo, prefill_hi, tp_size, 0, width)[2]

    per_rank = []
    for rank in range(tp_size):
        lo, hi, stride = indexer_row_shard(prefill_lo, prefill_hi, tp_size, rank, width)
        shard_out = (
            torch.full((stride, topk), -99, dtype=torch.int32)
            if stride
            else out[prefill_lo:prefill_hi]
        )
        for c_start, c_end in chunks:
            for q_start, q_end, row_start, row_end in _windows(
                c_start, c_end, lo, hi, 64
            ):
                dst = (
                    shard_out[q_start - lo : q_end - lo]
                    if stride
                    else shard_out[q_start - prefill_lo : q_end - prefill_lo]
                )
                # The score depends on the row's GLOBAL id and on the chunk-local
                # window offset, so either being wrong changes the answer.
                rows = torch.arange(q_start, q_end, dtype=torch.int32)
                local = torch.arange(row_start, row_end, dtype=torch.int32)
                dst.copy_(torch.stack([rows, local], dim=1).repeat(1, topk // 2))
        per_rank.append(shard_out)

    if stride:
        gathered = torch.cat(per_rank, dim=0)
        out[prefill_lo:prefill_hi] = gathered[: prefill_hi - prefill_lo]
    return out


@pytest.mark.parametrize("tp_size", (2, 4, 8, 16))
def test_sharded_output_is_identical_to_replicated(tp_size):
    chunks = [(64, 1100), (1100, 4200), (4200, 9000)]
    num_tokens, topk = 9200, 8

    replicated = _run_prefill(chunks, 1, num_tokens, topk)
    sharded = _run_prefill(chunks, tp_size, num_tokens, topk)

    assert torch.equal(replicated, sharded)
    # Padding rows outside the prefill range are untouched, and no shard
    # sentinel (-99) survived the gather.
    assert (sharded[:64] == -1).all()
    assert (sharded[9000:] == -1).all()
    assert not (sharded == -99).any()


def test_short_last_shard_does_not_leak_padding():
    """num_rows indivisible by tp_size is the case the [:num_rows] slice guards."""
    # 8000 rows over 7 ranks: stride 1143, last rank holds 1142 -> one pad row.
    chunks = [(0, 8000)]
    replicated = _run_prefill(chunks, 1, 8000, 4)
    for tp_size in (3, 5, 6, 7, 9, 13):
        assert torch.equal(_run_prefill(chunks, tp_size, 8000, 4), replicated)
