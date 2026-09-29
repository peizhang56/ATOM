# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The vLLM plugin's side of the DSA indexer TP row shard.

`tests/test_indexer_row_shard.py` covers the shared algebra; this covers what
only the plugin decides -- the absolute-token span (an off-by-`num_decode_tokens`
would score decode rows) and the row-weighted mean KV width the gate sees.
"""

import pytest
import torch

pytest.importorskip("vllm")

from atom.plugin.vllm.attention.layer_sparse_mla import (
    _indexer_prefill_shard,
)

# Defaults from atom/utils/envs.py.
_MIN_KV_WIDTH = 16384
_MIN_ROWS = 256


class _Chunk:
    """The fields `_indexer_prefill_shard` reads off an indexer prefill chunk."""

    def __init__(self, token_start, token_end, total_seq_lens):
        self.token_start = token_start
        self.token_end = token_end
        self.total_seq_lens = total_seq_lens


@pytest.fixture
def tp(monkeypatch):
    """Set the TP world the function imports from `vllm.distributed`."""
    import vllm.distributed

    def _set(world, rank):
        monkeypatch.setattr(
            vllm.distributed, "get_tensor_model_parallel_world_size", lambda: world
        )
        monkeypatch.setattr(
            vllm.distributed, "get_tensor_model_parallel_rank", lambda: rank
        )

    return _set


# ---------------------------------------------------------------------------
# Absolute batch token space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decode_tokens", (0, 1, 512))
def test_shard_spans_the_chunks_in_absolute_token_space(tp, decode_tokens):
    """Rows are offset by the decode tokens that precede them in the batch."""
    d = decode_tokens
    chunks = [
        _Chunk(d, d + 1000, 1 << 20),
        _Chunk(d + 1000, d + 4000, 1 << 20),
    ]
    tp(1, 0)
    lo, hi, stride = _indexer_prefill_shard(chunks)
    # TP1 never shards, but the range must still be the prefill range.
    assert (lo, hi, stride) == (d, d + 4000, 0)

    tp(8, 0)
    lo, hi, _ = _indexer_prefill_shard(chunks)
    assert lo == d, "the first shard must start at the first prefill token"


def test_shards_tile_the_whole_prefill_range_exactly_once(tp):
    """Across ranks the shards must cover every prefill row and no decode row."""
    decode_tokens = 512
    chunks = [
        _Chunk(decode_tokens, decode_tokens + 1000, 1 << 20),
        _Chunk(decode_tokens + 1000, decode_tokens + 4200, 1 << 20),
        _Chunk(decode_tokens + 4200, decode_tokens + 9000, 1 << 20),
    ]
    world = 8

    covered = []
    for rank in range(world):
        tp(world, rank)
        lo, hi, stride = _indexer_prefill_shard(chunks)
        assert stride, "a wide, tall batch must shard"
        assert lo >= decode_tokens, "a shard must never reach into decode rows"
        covered.extend(range(lo, hi))

    assert covered == list(range(decode_tokens, decode_tokens + 9000))


# ---------------------------------------------------------------------------
# The gate sees the row-weighted mean width
# ---------------------------------------------------------------------------


def test_one_wide_chunk_does_not_drag_a_narrow_batch_into_the_shard(tp):
    """Max-width gating would shard here, and the shard is a loss at this width."""
    chunks = [
        _Chunk(0, 7900, 1024),  # the bulk of the rows, far below the gate
        _Chunk(7900, 8000, 1 << 20),  # one very wide chunk
    ]
    # Weighted mean = (7900*1024 + 100*1048576) // 8000 = 14118 < 16384.
    tp(8, 0)
    assert _indexer_prefill_shard(chunks)[2] == 0


def test_one_narrow_chunk_does_not_keep_a_wide_batch_out_of_the_shard(tp):
    """Min-width gating would skip here, and the shard is a win at this width."""
    chunks = [
        _Chunk(0, 7000, 20000),  # the bulk of the rows, above the gate
        _Chunk(7000, 8000, 100),  # one very narrow chunk
    ]
    # Weighted mean = (7000*20000 + 1000*100) // 8000 = 17512 >= 16384.
    tp(8, 0)
    assert _indexer_prefill_shard(chunks)[2] != 0


def test_a_single_chunk_gates_on_its_own_width(tp):
    tp(8, 0)
    assert _indexer_prefill_shard([_Chunk(0, 8000, _MIN_KV_WIDTH - 1)])[2] == 0
    assert _indexer_prefill_shard([_Chunk(0, 8000, _MIN_KV_WIDTH)])[2] != 0


def test_total_seq_lens_may_be_a_tensor(tp):
    """vLLM hands these out of metadata; the helper int()s them."""
    tp(8, 0)
    chunks = [_Chunk(0, 8000, torch.tensor(1 << 20))]
    assert _indexer_prefill_shard(chunks)[2] != 0


# ---------------------------------------------------------------------------
# Degenerate batches must not shard (and must not divide by zero)
# ---------------------------------------------------------------------------


def test_an_empty_prefill_range_does_not_divide_by_zero(tp):
    tp(8, 0)
    assert _indexer_prefill_shard([_Chunk(64, 64, 1 << 20)]) == (64, 64, 0)


def test_a_short_prefill_is_not_sharded(tp):
    """Below the row floor the all-gather latency is not paid back."""
    tp(8, 0)
    assert _indexer_prefill_shard([_Chunk(0, _MIN_ROWS - 1, 1 << 20)])[2] == 0
    assert _indexer_prefill_shard([_Chunk(0, _MIN_ROWS, 1 << 20)])[2] != 0
