# SPDX-License-Identifier: MIT
"""ATOM-owned DeepSeek-V4 DSpark draft: registry wiring and block layout.

The substitution this file guards is narrow and exact: vLLM lays the draft
block out as ``[anchor, filler x K]`` per request and ATOM's backbone rebuilds
the same block from the anchor alone, so the wrapper forwards column 0 rather
than the laid-out ids. If the two layouts ever stop agreeing -- a different
filler token, a different width convention, a non-contiguous batch -- the draft
silently drafts from the wrong token and costs only acceptance, which is the
failure mode this whole model exists to fix. Hence the explicit tests.
"""

import pytest
import torch


def test_dspark_draft_arch_is_registered_to_atom_on_both_seams():
    """Both registries must point at this module, or the draft silently falls
    through to vLLM's drafter -- which is the thing being replaced."""
    from atom.plugin.vllm.model_wrapper import _ATOM_MODEL_CLASSES
    from atom.plugin.vllm.register import _VLLM_MODEL_REGISTRY_OVERRIDES

    assert (
        _VLLM_MODEL_REGISTRY_OVERRIDES["DSparkDraftModel"]
        == "atom.plugin.vllm.models.deepseek_v4_dspark:DeepseekV4DSparkVllm"
    )
    assert (
        _ATOM_MODEL_CLASSES["DSparkDraftModel"]
        == "atom.plugin.vllm.models.deepseek_v4_dspark:DeepseekV4DSparkDraft"
    )


def test_dspark_draft_counts_as_a_v4_arch():
    """It needs the V4 proxy-layer registration and ATOM forward context, the
    same as the MTP drafts -- that is what `_DEEPSEEK_V4_ARCHES` gates."""
    from atom.plugin.vllm.model_wrapper import (
        _DEEPSEEK_V4_ARCHES,
        _DEEPSEEK_V4_DRAFT_ARCHES,
        _DEEPSEEK_V4_MTP_ARCHES,
    )

    assert "DSparkDraftModel" in _DEEPSEEK_V4_ARCHES
    # A V4 DRAFT, so it gets the draft proxy layer rather than the target's...
    assert "DSparkDraftModel" in _DEEPSEEK_V4_DRAFT_ARCHES
    # ...but NOT an MTP draft: vLLM's speculator hands it a laid-out block and
    # it takes the plain (input_ids, positions) forward. Routing it through the
    # MTP contract fails with "MTP draft forward requires hidden_states".
    assert "DSparkDraftModel" not in _DEEPSEEK_V4_MTP_ARCHES


def _anchor_view(input_ids: torch.Tensor, positions: torch.Tensor, width: int):
    """The extraction `DeepseekV4DSparkDraft.forward` performs, in isolation."""
    return input_ids.view(-1, width)[:, 0], positions.view(-1, width)[:, 0]


def test_anchor_extraction_matches_vllms_block_layout():
    """vLLM writes request r's block at `query_base = r * num_query_per_req`,
    so the anchors are column 0 of the row-major reshape."""
    width, num_reqs, filler = 8, 5, 128799
    anchors = torch.arange(100, 100 + num_reqs, dtype=torch.long)
    ids = torch.full((num_reqs, width), filler, dtype=torch.long)
    ids[:, 0] = anchors
    pos = torch.arange(num_reqs * width, dtype=torch.long).view(num_reqs, width)

    got_ids, got_pos = _anchor_view(ids.reshape(-1), pos.reshape(-1), width)
    torch.testing.assert_close(got_ids, anchors)
    # The anchor's position is the block's first, not the last.
    torch.testing.assert_close(got_pos, pos[:, 0])


@pytest.mark.parametrize("num_spec", [1, 3, 5, 7])
def test_wrapper_takes_its_block_width_from_the_speculative_config(num_spec):
    """vLLM's speculator uses `num_query_per_req = 1 + num_speculative_steps`
    (one bonus/anchor row plus one per speculative step), and ATOM's draft width
    is that same T because its block is [anchor, noise x (T-1)]. Read off the
    config rather than hardcoded, so a depth change cannot desynchronise them.
    """
    from atom.plugin.vllm.models.deepseek_v4_dspark import vllm_block_width

    class _Spec:
        num_speculative_tokens = num_spec

    class _Cfg:
        speculative_config = _Spec()

    assert vllm_block_width(_Cfg()) == 1 + num_spec


def test_block_width_falls_back_to_one_without_a_speculative_config():
    """A draft built outside spec-decode has no block to lay out; width 1 keeps
    the reshape a no-op rather than dividing by zero."""
    from atom.plugin.vllm.models.deepseek_v4_dspark import vllm_block_width

    class _Cfg:
        speculative_config = None

    assert vllm_block_width(_Cfg()) == 1


def test_forward_rejects_an_empty_batch():
    from atom.plugin.vllm.models.deepseek_v4_dspark import DeepseekV4DSparkDraft

    draft = DeepseekV4DSparkDraft.__new__(DeepseekV4DSparkDraft)
    draft.vllm_block_width = 8
    with pytest.raises(ValueError, match="empty batch"):
        DeepseekV4DSparkDraft.forward(
            draft,
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )


def test_forward_truncates_for_a_batch_shorter_than_one_block():
    """Memory profiling sizes the draft by a token budget, which can be smaller
    than one block. Draft one block and return only the rows vLLM asked for."""
    from atom.plugin.vllm.models.deepseek_v4_dspark import DeepseekV4DSparkDraft

    width, dim, total = 8, 4, 7
    draft = _StubDraft(width, dim)
    out = draft.forward(
        torch.zeros(total, dtype=torch.long), torch.zeros(total, dtype=torch.long)
    )
    assert out.shape == (total, dim)
    # One block drafted, from the only anchor available.
    assert draft.seen["anchors"].numel() == 1


class _StubDraft:
    """`DeepseekV4DSparkDraft.forward` with the compiled backbone stubbed."""

    def __init__(self, width, dim):
        self.vllm_block_width = width
        self.dim = dim
        self.seen = {}

    def block_backbone(self, anchors, positions, num_draft):
        self.seen["anchors"] = anchors.clone()
        self.seen["num_draft"] = num_draft
        return torch.ones(anchors.numel() * num_draft, self.dim), None

    def forward(self, input_ids, positions):
        from atom.plugin.vllm.models.deepseek_v4_dspark import DeepseekV4DSparkDraft

        return DeepseekV4DSparkDraft.forward(self, input_ids, positions)


def test_forward_drafts_whole_blocks_and_zero_fills_a_padded_tail():
    """Dummy / profiling / cudagraph-capture runs arrive padded to a token
    count that need not divide by the block width. Reshaping across that
    boundary would take an anchor from the middle of someone's block, so the
    tail is zero-filled instead -- and the returned row count still matches the
    input, which vLLM indexes by."""
    width, dim = 8, 4
    draft = _StubDraft(width, dim)
    seen = draft.seen
    total = width * 3 + 5  # three whole blocks plus padding
    ids = torch.zeros(total, dtype=torch.long)
    anchors = torch.tensor([11, 22, 33])
    for b, a in enumerate(anchors):
        ids[b * width] = a

    out = draft.forward(ids, torch.zeros(total, dtype=torch.long))

    torch.testing.assert_close(seen["anchors"], anchors)
    assert seen["num_draft"] == width
    # Row count matches the padded input, and only the padding is zeroed.
    assert out.shape == (total, dim)
    assert out[: width * 3].eq(1).all()
    assert out[width * 3 :].eq(0).all()


def test_forward_refuses_to_guess_an_unset_block_width():
    from atom.plugin.vllm.models.deepseek_v4_dspark import DeepseekV4DSparkDraft

    draft = DeepseekV4DSparkDraft.__new__(DeepseekV4DSparkDraft)
    draft.vllm_block_width = None
    with pytest.raises(RuntimeError, match="vllm_block_width"):
        DeepseekV4DSparkDraft.forward(
            draft,
            torch.zeros(8, dtype=torch.long),
            torch.zeros(8, dtype=torch.long),
        )


def test_context_kv_write_is_skipped_without_slot_mappings():
    """vLLM passes `None` on dummy / memory-profiling steps, where the block
    tables are placeholders; writing there clobbers live cache entries."""
    from atom.plugin.vllm.models.deepseek_v4_dspark import DeepseekV4DSparkVllm

    wrapper = DeepseekV4DSparkVllm.__new__(DeepseekV4DSparkVllm)
    called = []

    class _Model:
        def write_combined_context_kv(self, *a):
            called.append(a)

    object.__setattr__(wrapper, "_model_stub", _Model())
    # Must return before touching anything else -- no bind, no context, no write.
    DeepseekV4DSparkVllm.precompute_and_store_context_kv(
        wrapper, torch.zeros(4), torch.zeros(4), None
    )
    assert called == []
