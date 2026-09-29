# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`clean_logits=False` must not change what the indexer selects.

It does two things, not one: the `-inf` prefill becomes a raw `torch.empty`, AND
the gluon kernel gets `RELAXED_STORE=1`, which relaxes the per-row store mask.
The first is covered by `test_indexer_topk_row_window` and
`test_indexer_logits_alignment_tail` (nothing reads outside a row's window). The
second is not: a relaxed store that spilled INTO a window would be read, and
every test here would be the only thing to catch it.

So the buffer is poisoned with a value that wins any top-k it is read into --
standing in for whatever the allocator actually handed back -- and the check is
that in-window scores stay bit-identical and the selection does not move.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises aiter GPU kernels; needs a real GPU", allow_module_level=True
    )

from aiter.ops.topk import top_k_per_row_prefill
from aiter.ops.triton.attention import fp8_mqa_logits as _mqa_mod
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_dtypes

DEV = "cuda"
WINS_ANY_TOPK = 1e30
_, E4M3 = get_fp8_dtypes()

# s_q, s_k, heads, dim. 1560 is off a multiple of 256 so the alignment tail is
# real; the rest span the head/dim pairs the DSA indexer actually runs.
SHAPES = [
    (128, 1024, 32, 128),
    (1024, 1560, 32, 128),
    (61, 113, 32, 64),
    (1024, 4096, 64, 128),
]


def _inputs(s_q, s_k, heads, dim):
    g = torch.Generator(device=DEV).manual_seed(s_q * 31 + s_k)
    q = torch.randn(s_q, heads, dim, device=DEV, dtype=torch.bfloat16, generator=g)
    kv = torch.randn(s_k, dim, device=DEV, dtype=torch.bfloat16, generator=g)
    w = torch.randn(s_q, heads, device=DEV, dtype=torch.float32, generator=g)
    sc = torch.rand(s_k, device=DEV, dtype=torch.float32, generator=g) + 0.5
    return q.to(E4M3), kv.to(E4M3), sc, w


def _windows(s_q, s_k):
    """A non-zero base, as a resumed chunk gets: live data on both sides of the
    window is the shape where a bound bug actually leaks."""
    base = s_k // 3
    ks = torch.full((s_q,), base, dtype=torch.int32, device=DEV)
    ke = torch.tensor(
        [min(base + i + 1, s_k) for i in range(s_q)], dtype=torch.int32, device=DEV
    )
    return ks, ke


def _poisoned_logits(q, kv, sc, w, ks, ke, s_q):
    """`clean_logits=False` with the uninitialized buffer forced to poison."""
    real_empty = torch.empty

    def fake_empty(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        if t.dtype == torch.float32 and t.dim() == 2 and t.shape[0] == s_q:
            t.fill_(WINS_ANY_TOPK)
        return t

    torch.empty = fake_empty
    try:
        return fp8_mqa_logits(q, kv, sc, w, ks, ke, clean_logits=False)
    finally:
        torch.empty = real_empty


def _topk(logits, ks, ke, k, stable):
    idx = torch.empty((logits.shape[0], k), dtype=torch.int32, device=DEV)
    top_k_per_row_prefill(
        logits,
        ks,
        ke,
        idx,
        None,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        k=k,
        stable=stable,
    )
    return idx


def _in_window(s_k, ks, ke):
    cols = torch.arange(s_k, device=DEV)[None, :]
    return (cols >= ks[:, None]) & (cols < ke[:, None])


@pytest.mark.parametrize("s_q, s_k, heads, dim", SHAPES)
def test_scores_inside_the_window_are_bit_identical(s_q, s_k, heads, dim):
    """RELAXED_STORE may write more, but never something different in-window."""
    q, kv, sc, w = _inputs(s_q, s_k, heads, dim)
    ks, ke = _windows(s_q, s_k)
    clean = fp8_mqa_logits(q, kv, sc, w, ks, ke, clean_logits=True)
    dirty = _poisoned_logits(q, kv, sc, w, ks, ke, s_q)
    inwin = _in_window(s_k, ks, ke)

    assert torch.equal(clean[inwin], dirty[inwin]), "relaxed store moved a real score"
    assert not torch.equal(clean[~inwin], dirty[~inwin]), (
        "the two buffers agree outside the window too, so the poison never "
        "landed and this test proves nothing"
    )


@pytest.mark.parametrize("s_q, s_k, heads, dim", SHAPES)
@pytest.mark.parametrize("k", [64, 2048])
@pytest.mark.parametrize("stable", [False, True])
def test_selection_is_unchanged_by_the_poison(s_q, s_k, heads, dim, k, stable):
    """End to end: same rows chosen, none from outside the window."""
    q, kv, sc, w = _inputs(s_q, s_k, heads, dim)
    ks, ke = _windows(s_q, s_k)
    clean = fp8_mqa_logits(q, kv, sc, w, ks, ke, clean_logits=True)
    dirty = _poisoned_logits(q, kv, sc, w, ks, ke, s_q)

    i_clean = _topk(clean, ks, ke, k, stable)
    i_dirty = _topk(dirty, ks, ke, k, stable)
    counts = (ke - ks).clamp(min=0)

    for r in range(s_q):
        kc = int(min(k, counts[r].item()))
        if kc <= 0:
            continue
        lo, hi = int(ks[r]), int(ke[r])
        got = i_dirty[r, :kc].tolist()
        assert all(lo <= v < hi for v in got), f"row {r} selected outside its window"
        # Order is not deterministic on the non-stable path; the SET is.
        assert set(got) == set(i_clean[r, :kc].tolist()), f"row {r} selection moved"


def test_the_relaxed_store_path_is_reachable():
    """`clean_logits=False` only sets RELAXED_STORE on the gluon kernel. Without
    it the tests above check the -inf swap alone and half the contract is unpinned."""
    assert _mqa_mod.TRITON_GE_36, "triton too old for the gluon path"
    assert _mqa_mod._gluon_fp8_mqa_logits_kernel is not None, (
        "no gluon kernel in this aiter build, so RELAXED_STORE never engages "
        "and these tests do not cover the relaxed store"
    )


def test_the_poison_is_poison():
    """If a widened window does not pull the poison in, it is not poison and
    every assertion above passes vacuously."""
    s_q, s_k, heads, dim = 128, 1024, 32, 128
    q, kv, sc, w = _inputs(s_q, s_k, heads, dim)
    ks, ke = _windows(s_q, s_k)
    dirty = _poisoned_logits(q, kv, sc, w, ks, ke, s_q)

    inwin = _in_window(s_k, ks, ke)
    assert (dirty[~inwin] == WINS_ANY_TOPK).all(), "poison did not survive the kernel"

    wide_ks = torch.zeros_like(ks)
    wide_ke = torch.full_like(ke, s_k)
    idx = _topk(dirty, wide_ks, wide_ke, 64, stable=False)
    for r in range(s_q):
        lo, hi = int(ks[r]), int(ke[r])
        assert any(
            not (lo <= v < hi) for v in idx[r].tolist()
        ), f"row {r} ignored the poison even when allowed to read it"
