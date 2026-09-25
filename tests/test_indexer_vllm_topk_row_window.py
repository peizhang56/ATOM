# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The vLLM plugin's indexer prefill rests on a kernel this repo does not own.

`atom/plugin/vllm/attention/layer_sparse_mla.py` calls
`fp8_mqa_logits(..., clean_logits=False)`, so the `[rows, total_committed]` fp32
scores buffer is raw `torch.empty` outside each row's `[cu_start, cu_end)` window.
That is safe for exactly one reason: the only consumer,
`torch.ops._C.top_k_per_row_prefill`, is handed the SAME `row_ks`/`row_ke` and
bounds every read by them (`vllm/csrc/libtorch_stable/sampler.cu`).

`tests/test_indexer_topk_row_window.py` and
`tests/test_indexer_logits_alignment_tail.py` pin that invariant for *aiter's*
top-k, which the native paths use. The vLLM plugin path uses a different kernel,
in a different repo, on a pinned commit that a version bump can move underneath
us -- and if it ever grows a padded pre-scan (its decode sibling already has a
`multipleBlocksPerRow` path), the failure is silent: allocator garbage wins the
top-k and a KV index outside the request comes back. So pin it here too.

The second half checks the other side of the same contract -- that dropping the
fill does not change what the kernel writes *inside* the window -- across the
gfx950 gluon configurations where `clean_logits=False` actually reaches the
kernel as `RELAXED_STORE=1` (BLOCK_M=2, with and without the KV split).

Do NOT write these as top-k index comparisons. `topKPerRowPrefill` is bin-based
and equal-bin order is race-dependent: two runs of one config disagree on ~89%
of index slots with no code change at all. Compare values.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises the vLLM top-k and an aiter GPU kernel; needs a real GPU",
        allow_module_level=True,
    )

pytest.importorskip("vllm._C")  # registers torch.ops._C.top_k_per_row_prefill
fp8_mqa_logits = pytest.importorskip(
    "aiter.ops.triton.fp8_mqa_logits"
).fp8_mqa_logits

DEV = "cuda"
# The indexer geometry DeepSeek-V3.2 and GLM-5.x share, and the top-k width
# layer_sparse_mla.py asserts on.
HEADS, HEAD_DIM, TOPK = 32, 128, 2048
ALIGN = 256  # aiter rounds the logits width up to this
WINS_ANY_TOPK = 1e30
# vllm/csrc/libtorch_stable/sampler.cu: rows below this index use the insertion
# kernel, rows above it a second launch of the radix one. Both read the buffer.
SORT_PATH_SPLIT = 12288


def _kv_spans(counts, seq_lens):
    """Mirror of `kv_spans_from_batches` (atom/plugin/vllm/attention/metadata.py).

    Each request's selected tokens are the LAST `count` positions of its
    sequence, and its columns are its own segment of the concatenated KV -- so
    row `t` of request `b` may only see `[base_b, base_b + local_pos)`.
    """
    starts, ends, base = [], [], 0
    for count, seq_len in zip(counts, seq_lens):
        # Not decoration. A request cannot select more tokens than it has, and
        # `count > seq_len` yields rowEnd < rowStart, which sends
        # `topKPerRowJob`'s short-row path writing at a negative offset --
        # a hard abort, several tests away from the line that caused it.
        assert count <= seq_len, f"{count} selected tokens over a {seq_len} sequence"
        for i in range(count):
            starts.append(base)
            ends.append(base + (seq_len - count + i + 1))  # local_pos is 1-based
        base += seq_len
    return (
        torch.tensor(starts, dtype=torch.int32, device=DEV),
        torch.tensor(ends, dtype=torch.int32, device=DEV),
    )


def _top_k(logits, starts, ends):
    """Call the op exactly as `sparse_attn_indexer_plugin_mode` does."""
    rows = logits.shape[0]
    indices = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
    torch.ops._C.top_k_per_row_prefill(
        logits, starts, ends, indices, rows, logits.stride(0), logits.stride(1), TOPK
    )
    return indices


def _aligned(width):
    return -(-width // ALIGN) * ALIGN


@pytest.mark.parametrize(
    "rows,kv_len",
    [
        pytest.param(64, 9000, id="insertion"),
        pytest.param(SORT_PATH_SPLIT + 512, 40000, id="insertion+radix"),
    ],
)
def test_no_read_lands_outside_the_row_window(rows, kv_len):
    """Poison every column, write real scores only inside the windows. Anything
    the kernel reads out of bounds comes back as a selected index, because the
    poison beats every real score.

    `rows > SORT_PATH_SPLIT` is not padding: it is the only way the radix
    instantiation of the kernel gets compiled, let alone checked.
    """
    aligned = _aligned(kv_len)  # kv_len is off a multiple of 256: the tail is real
    assert aligned > kv_len, "pick a kv_len that leaves alignment padding"
    logits = torch.full(
        (rows, aligned), WINS_ANY_TOPK, dtype=torch.float32, device=DEV
    )[:, :kv_len]

    half = rows // 2
    starts, ends = _kv_spans([half, rows - half], [kv_len // 2, kv_len - kv_len // 2])
    windows = (ends - starts).to(torch.int64)
    # Real scores inside each window, all strictly below the poison.
    columns = torch.arange(kv_len, device=DEV)
    inside = (columns[None, :] >= starts[:, None]) & (columns[None, :] < ends[:, None])
    logits[inside] = 1.0

    indices = _top_k(logits, starts, ends)

    valid = indices >= 0
    assert bool(valid.any()), "every row returned sentinels; the test proves nothing"
    # Indices are row-local (0-based from rowStart) on the stride1 == 1 path.
    over = valid & (indices.to(torch.int64) >= windows[:, None])
    assert not bool(over.any()), (
        f"{int(over.sum())} indices landed past their window end "
        f"(worst row {int(over.any(1).nonzero()[0, 0])})"
    )
    absolute = (indices.to(torch.int64) + starts[:, None]).clamp_(0, kv_len - 1)
    selected = logits.gather(1, absolute)[valid]
    assert not bool((selected >= WINS_ANY_TOPK).any()), (
        f"{int((selected >= WINS_ANY_TOPK).sum())} selected values were poison -- "
        "the kernel read outside the window (or into the alignment padding)"
    )


def test_the_poison_is_actually_reachable():
    """The control. Widen one row's window over the padding and the poison must
    come back -- otherwise the assertions above are measuring their own absence.
    """
    kv_len, rows = 9000, 1
    aligned = _aligned(kv_len)
    logits = torch.full(
        (rows, aligned), WINS_ANY_TOPK, dtype=torch.float32, device=DEV
    )
    logits[:, :kv_len] = 0.0
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    ends = torch.full((rows,), aligned, dtype=torch.int32, device=DEV)

    indices = _top_k(logits, starts, ends)
    selected = indices[indices >= 0].to(torch.int64)
    assert bool((selected >= kv_len).any()), (
        "the padding holds the only non-zero scores, so a kernel that reads it "
        "must select it -- if this fails the poison is not poison"
    )


def _logits_inputs(rows, seq_lens, counts, seed=0):
    generator = torch.Generator(device=DEV).manual_seed(seed)
    total_kv = sum(seq_lens)
    q = (torch.randn(rows, HEADS, HEAD_DIM, generator=generator, device=DEV) * 0.5).to(
        torch.float8_e4m3fn
    )
    kv = (torch.randn(total_kv, HEAD_DIM, generator=generator, device=DEV) * 0.5).to(
        torch.float8_e4m3fn
    )
    # layer_sparse_mla.py hands the kernel a [total, 1] scale plane.
    kv_scales = torch.rand(
        total_kv, 1, generator=generator, device=DEV, dtype=torch.float32
    )
    weights = torch.rand(
        rows, HEADS, generator=generator, device=DEV, dtype=torch.float32
    )
    starts, ends = _kv_spans(counts, seq_lens)
    assert int(ends.max()) <= total_kv, "a row window may not leave the KV buffer"
    return dict(
        Q=q, KV=kv, kv_scales=kv_scales, weights=weights, cu_starts=starts, cu_ends=ends
    )


# gfx950 picks BLOCK_M=2 -- the only configuration where clean_logits=False
# reaches the kernel, as RELAXED_STORE=1 -- once seq_len > 4096, or once
# cdiv(seq_len, 2) * num_kv_splits clears its workgroup floor. One shape each
# side of that, so the KV-split store bound is covered too.
SHAPES = [
    pytest.param(2048, [16384, 16384], [1024, 1024], id="kv_split"),
    pytest.param(8192, [8192], [8192], id="no_kv_split"),
    pytest.param(512, [2048, 2048], [256, 256], id="small"),
]


@pytest.mark.parametrize("rows,seq_lens,counts", SHAPES)
def test_in_window_logits_are_bit_identical_without_the_fill(rows, seq_lens, counts):
    """`clean_logits=False` relaxes the PER-ROW store mask; the union bound that
    keeps a store inside its row stays. So the columns a reader can reach must
    come out bit-for-bit unchanged -- not close, identical."""
    inputs = _logits_inputs(rows, seq_lens, counts)
    clean = fp8_mqa_logits(**inputs, clean_logits=True)
    relaxed = fp8_mqa_logits(**inputs, clean_logits=False)

    starts, ends = inputs["cu_starts"], inputs["cu_ends"]
    columns = torch.arange(clean.shape[1], device=DEV)
    inside = (columns[None, :] >= starts[:, None]) & (columns[None, :] < ends[:, None])
    differs = clean.ne(relaxed) & inside
    assert not bool(differs.any()), (
        f"{int(differs.sum())} of {int(inside.sum())} in-window values changed; "
        f"max |diff| {float((clean - relaxed).abs()[inside].max())}"
    )


@pytest.mark.parametrize("rows,seq_lens,counts", SHAPES)
def test_top_k_selects_the_same_values_without_the_fill(rows, seq_lens, counts):
    """End to end through the real consumer, on the axis that is actually
    defined. The INDEX a tie resolves to is race-dependent either way; the
    multiset of selected VALUES is what feeds sparse attention."""
    inputs = _logits_inputs(rows, seq_lens, counts)
    starts, ends = inputs["cu_starts"], inputs["cu_ends"]

    selected = []
    for clean_logits in (True, False):
        logits = fp8_mqa_logits(**inputs, clean_logits=clean_logits)
        indices = _top_k(logits, starts, ends)
        absolute = (indices.to(torch.int64) + starts[:, None]).clamp_(
            0, logits.shape[1] - 1
        )
        values = logits.gather(1, absolute)
        # Sentinel slots (window shorter than k) carry no value.
        values = values.masked_fill(indices < 0, -float("inf"))
        selected.append(torch.sort(values, dim=1, descending=True).values)

    mismatched = (selected[0] != selected[1]).any(dim=1)
    assert not bool(mismatched.any()), (
        f"{int(mismatched.sum())} of {rows} rows selected a different value "
        "multiset with the fill removed"
    )
