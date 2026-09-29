# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Choosing NextNTile for the decode indexer kernel.

Folding shares one KV walk across `NextNTile` of the `next_n` speculative
rows. Two things bound the choice and both are silent failures if got wrong:
the fold must divide `next_n`, because the kernel grids over
`next_n // NextNTile` and a remainder drops rows; and `index_heads *
NextNTile` must stay inside the 256-VGPR file, because past that the kernel
scratch-spills and runs several times slower than not folding at all. aiter
asserts the second, so overshooting takes the server down rather than costing
a few percent.
"""

import pytest

from atom.plugin.vllm.attention.layer_sparse_mla import (
    _MQA_LOGITS_FOLD_BUDGET,
    _mqa_logits_fold,
)


@pytest.fixture
def _folding(monkeypatch):
    """Price the fold as if the installed aiter declares NextNTile."""
    monkeypatch.setattr(
        "atom.plugin.vllm.attention.layer_sparse_mla._AITER_HAS_NEXT_N_TILE", True
    )


@pytest.mark.parametrize(
    "next_n,index_heads,expected",
    [
        # GLM-5.3: 32 index heads, MTP depth 3 -> next_n 4. Folds whole, and
        # sits exactly on the budget.
        (4, 32, 4),
        (2, 32, 2),
        (1, 32, 1),
        # A wide head count narrows the fold rather than losing it.
        (4, 64, 2),
        (4, 128, 1),
        # 96 * 1 is inside the budget but 96 * 2 is not, and 3 does not divide 4.
        (4, 96, 1),
        # Non-power-of-two next_n still has to fold by a divisor: 6 cannot fold
        # 4 at 32 heads even though 4 would fit, because 4 does not divide 6.
        (6, 32, 3),
        (3, 32, 3),
        # Deeper speculation than the budget allows is clamped, not refused.
        (8, 32, 4),
        (8, 16, 8),
    ],
)
def test_fold_is_a_divisor_inside_the_vgpr_budget(
    _folding, next_n: int, index_heads: int, expected: int
) -> None:
    fold = _mqa_logits_fold(next_n, index_heads)
    assert fold == expected
    assert next_n % fold == 0, "a fold that does not divide next_n drops rows"
    assert index_heads * fold <= _MQA_LOGITS_FOLD_BUDGET, "would spill VGPRs"


@pytest.mark.parametrize("index_heads", [1, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("next_n", range(1, 17))
def test_fold_never_violates_either_bound(_folding, next_n, index_heads) -> None:
    """The two invariants hold for every shape, not just the tabulated ones."""
    fold = _mqa_logits_fold(next_n, index_heads)
    assert 1 <= fold <= next_n
    assert next_n % fold == 0
    # 1 is always allowed: not folding is what the kernel did before, whatever
    # the head count.
    assert fold == 1 or index_heads * fold <= _MQA_LOGITS_FOLD_BUDGET


def test_fold_is_off_against_an_aiter_without_it(monkeypatch) -> None:
    """An older aiter has no NextNTile, and the caller must not ask for one."""
    monkeypatch.setattr(
        "atom.plugin.vllm.attention.layer_sparse_mla._AITER_HAS_NEXT_N_TILE", False
    )
    assert _mqa_logits_fold(4, 32) == 1
