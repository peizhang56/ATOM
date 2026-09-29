# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Row sharding of the DSA prefill indexer across TP.

The indexer is replicated per rank, so the shard splits prefill rows and
all-gathers the int32 top-k back. The split is only exact if the gathered
buffer maps back to global rows with no gap and no padding read: gathered row
`rank * stride + i` must be global row `shard_lo(rank) + i`, and the caller's
`[:num_rows]` slice must land exactly on rows `0..num_rows-1`.

There are two implementations of the same algebra -- the ATOM-native path in
models/deepseek_v2.py and the vLLM plugin path in
plugin/vllm/attention/layer_sparse_mla.py -- and they are checked together
here, since a divergence between them is the failure that would not show up as
a crash.
"""

import pytest

from atom.models.deepseek_v2 import (
    _INDEXER_ROW_SHARD_MIN_ROWS,
)
from atom.models.deepseek_v2 import (
    _indexer_row_shard as _native_shard,
)
from atom.plugin.vllm.attention.layer_sparse_mla import (
    _indexer_row_shard as _plugin_shard,
)

_TP_SIZES = (1, 2, 4, 8, 16)
# 11588 is one request's prefill rows at ISL 115000; 255/256 straddle the
# minimum-rows gate.
_ROW_COUNTS = (0, 1, 255, 256, 257, 511, 512, 1000, 2048, 11588, 16384)

_PAD = object()


def _shards(call, tp_size, monkeypatch, module):
    """Every rank's `(lo, hi, stride)`, with the TP group faked to `tp_size`.

    The two implementations resolve the TP accessors differently -- the native
    path imports them at module scope, the plugin inside the function -- so the
    patch target is the module each one actually reads.
    """
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
    out = []
    for rank in range(tp_size):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        out.append(call())
    return out


@pytest.fixture
def native(monkeypatch):
    import atom.models.deepseek_v2 as mod

    def call(num_rows, tp_size):
        return _shards(lambda: _native_shard(num_rows), tp_size, monkeypatch, mod)

    return call


@pytest.mark.parametrize("tp_size", _TP_SIZES)
@pytest.mark.parametrize("num_rows", _ROW_COUNTS)
def test_gather_reconstructs_every_row_exactly_once(native, tp_size, num_rows):
    shards = native(num_rows, tp_size)
    stride = shards[0][2]
    if not stride:
        # Not sharded: the caller writes topk_indices directly.
        assert all(s == (0, num_rows, 0) for s in shards)
        assert tp_size <= 1 or num_rows < _INDEXER_ROW_SHARD_MIN_ROWS
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
def test_only_the_last_non_empty_rank_is_short(native, tp_size, num_rows):
    """The padding has to sit at the end, or the slice would drop a real row."""
    shards = native(num_rows, tp_size)
    if not shards[0][2]:
        return
    sizes = [hi - lo for lo, hi, _ in shards]
    stride = shards[0][2]
    assert sum(sizes) == num_rows
    full = [i for i, n in enumerate(sizes) if n == stride]
    rest = [i for i, n in enumerate(sizes) if n != stride]
    assert full == list(range(len(full))), "full shards must come first"
    assert len([i for i in rest if sizes[i] > 0]) <= 1


@pytest.mark.parametrize("tp_size", _TP_SIZES)
@pytest.mark.parametrize("num_rows", _ROW_COUNTS)
def test_plugin_and_native_shards_agree(monkeypatch, tp_size, num_rows):
    """The two call sites must not drift apart.

    The plugin takes absolute token offsets `[row_lo, row_hi)` because it
    indexes q_fp8 in batch space; the native path takes a row count. Offsetting
    the plugin's answer by `row_lo` has to reproduce the native one, including
    under a non-zero base -- an off-by-`num_decode_tokens` there would score
    decode rows.
    """
    import vllm.distributed as vllm_dist

    import atom.models.deepseek_v2 as native_mod

    for base in (0, 37, 4096):
        native = _shards(
            lambda: _native_shard(num_rows), tp_size, monkeypatch, native_mod
        )
        plugin = _shards(
            lambda base=base: _plugin_shard(base, base + num_rows),
            tp_size,
            monkeypatch,
            vllm_dist,
        )
        for (lo, hi, stride), (p_lo, p_hi, p_stride) in zip(native, plugin):
            assert (p_lo - base, p_hi - base, p_stride) == (lo, hi, stride)
