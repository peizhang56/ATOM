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
        _DEEPSEEK_V4_ARCH,
        _DEEPSEEK_V4_ARCHES,
        _DEEPSEEK_V4_MTP_ARCHES,
    )

    assert "DSparkDraftModel" in _DEEPSEEK_V4_ARCHES
    # The MTP set is ARCHES minus the target, so the draft lands there too and
    # gets the *draft* proxy layer rather than the target's.
    assert "DSparkDraftModel" in _DEEPSEEK_V4_MTP_ARCHES
    assert "DSparkDraftModel" != _DEEPSEEK_V4_ARCH


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


def test_forward_rejects_a_ragged_token_count():
    """A token count that is not a multiple of the block width means vLLM's
    layout assumption no longer holds; fail loudly rather than reshape garbage
    into anchors."""
    from atom.plugin.vllm.models.deepseek_v4_dspark import DeepseekV4DSparkDraft

    draft = DeepseekV4DSparkDraft.__new__(DeepseekV4DSparkDraft)
    draft.vllm_block_width = 8
    with pytest.raises(ValueError, match="not a multiple"):
        DeepseekV4DSparkDraft.forward(
            draft,
            torch.zeros(23, dtype=torch.long),
            torch.zeros(23, dtype=torch.long),
        )


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
