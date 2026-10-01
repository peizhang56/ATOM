# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM-specific DeepSeek-V4 DSpark block drafter, backed by ATOM's own.

WHY THIS EXISTS. Without it vLLM resolves ``DSparkDraftModel`` to its own
``vllm.models.deepseek_v4.amd.dspark:DSparkDeepseekV4ForCausalLM``, and that
drafter loses acceptance as the decode batch grows: on DeepSeek-V4-Pro-0813 at
ISL 115k it holds ~3.2 accepted tokens/step to a running batch of 32 and falls
to 2.38 at 128, which is ~1.4x of the arm's throughput there. ATOM's own DSpark
proposer holds ~3.3 at every batch on the same checkpoint and node -- including
with DP attention off, i.e. the full 128-wide batch on every rank, so batch
width is not inherently hard for this drafter. Four structural causes were
eliminated by measurement first (out-of-range draft ids, offered load/queue
depth, cudagraphs, and slot-indexed bookkeeping in vLLM's speculator); what
remains is batch-dependent draft math, which is exactly what this file
replaces. ATOM already overrides ``K3DSparkModel`` the same way.

WHAT STAYS vLLM's. The *speculator*
(``vllm/v1/worker/gpu/spec_decode/dflash/speculator.py``) still drives: it lays
out the block, picks sample indices, handles rejection and slot mappings, and
runs the sequential Markov loop. This file supplies only the math behind the
hooks it calls. That division is deliberate -- the slot-binned rejection
histogram came out flat, so the speculator's bookkeeping is not implicated.

THE CONTRACT LINES UP EXACTLY, which is what makes this a wrapper rather than a
port:

* vLLM lays the block out as ``[anchor(bonus), MASK x num_speculative_steps]``
  at width ``num_query_per_req = 1 + num_speculative_steps``; ATOM builds
  ``[anchor, noise x (T-1)]`` (``deepseek_v4_dspark.py``). Both filler tokens
  resolve to ``hf_config.dspark_noise_token_id`` -- vLLM's
  ``get_parallel_drafting_token_id`` falls through ``dflash_config.mask_token_id``
  and ``mask_token_id`` to it, and V4-Pro-0813 carries only the last. So the two
  blocks are byte-identical and ATOM's backbone can rebuild it from the anchor.
* ``block_backbone`` returns ``(normed [B*T, dim], hc_hidden [B, T, dim])`` and
  vLLM's ``forward`` must return ``[num_tokens, hidden]`` -- ``normed`` as-is.
* The Markov head's ``forward``/``sample_next`` match K3's, which vLLM already
  drives.

KV OWNERSHIP is the one place this differs from ``kimi_k3_dspark``. K3's
``write_context_kv`` takes a ``slot_mapping`` and writes vLLM's paged pool;
V4's takes none and writes ATOM's own per-request rolling ring
(``attn.swa_plane``, addressed by ``attn.swa_window``), reading ``cu_seqlens_q``
and the per-request ``state_slot_out`` off ATOM's forward context. The read side
gathers from that same plane by absolute position, and that read path is the
code measured at ~3.3 tok/step -- re-pointing it at vLLM's paged layout would
mean rewriting it and losing the provenance that makes the swap worth doing.
So ATOM owns the draft's KV, exactly as it already does for the V4 MTP draft
(``_is_deepseek_v4_mtp`` -> ``deepseek_v4_draft_proxy_layer_name``), and vLLM's
draft KV group goes vestigial.

The consequence is :meth:`DeepseekV4DSparkVllm.precompute_and_store_context_kv`
below: vLLM's speculator calls it OUTSIDE any ``ATOMMoEForCausalLM.forward``, so
ATOM's V4 forward context is not open and ``write_context_kv`` would find no
metadata. It therefore binds the proxy cache views and enters that context
itself, mirroring ``model_wrapper.ATOMModelBase.forward``.
"""

import torch

from atom.models import deepseek_v4 as deepseek_v4_base
from atom.models.deepseek_v4_dspark import DeepseekV4DSpark as DeepseekV4DSparkBase
from atom.plugin.vllm.model_wrapper import ATOMMoEForCausalLM
from atom.plugin.vllm.models.deepseek_v4 import DeepseekV4AttentionVllm, IndexerVllm


def vllm_block_width(vllm_config) -> int:
    """The draft block's width in vLLM's layout.

    vLLM's speculator sets ``num_query_per_req = 1 + num_speculative_steps`` --
    one bonus/anchor row plus one per speculative step -- and ATOM's block is
    ``[anchor, noise x (T-1)]``, so the two Ts are the same number. Derived from
    the config rather than hardcoded so a depth change cannot desynchronise
    them; a mismatch would reshape the batch on the wrong stride and draft from
    the wrong token, costing acceptance silently.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    return 1 + int(getattr(spec, "num_speculative_tokens", 0) or 0)


class DeepseekV4DSparkDraft(DeepseekV4DSparkBase):
    """ATOM's DSpark drafter built with the vLLM V4 attention/indexer variants.

    The class swap matches ``deepseek_v4_mtp.DeepseekV4MTP``: the draft's sparse
    indexer must split mixed batches the way the target's does under vLLM
    continuous batching, which is what ``IndexerVllm`` adds.
    """

    def __init__(self, *args, **kwargs):
        original_attn_cls = deepseek_v4_base.DeepseekV4Attention
        original_indexer_cls = deepseek_v4_base.Indexer
        deepseek_v4_base.DeepseekV4Attention = DeepseekV4AttentionVllm
        deepseek_v4_base.Indexer = IndexerVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            deepseek_v4_base.DeepseekV4Attention = original_attn_cls
            deepseek_v4_base.Indexer = original_indexer_cls
        # Set by the wrapper from vLLM's speculative config; see `forward`.
        self.vllm_block_width: int | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One parallel pass over the draft block, in vLLM's layout.

        vLLM hands ``[num_reqs * T]`` with each request's block contiguous
        (``query_base = req_idx * num_query_per_req`` in its prep kernel) and
        wants ``[num_reqs * T, dim]`` back. ATOM's compiled backbone rebuilds the
        block from the ANCHOR, so take column 0 rather than feeding the laid-out
        ids through -- the two layouts are identical (see the module docstring),
        which is what makes that substitution exact rather than approximate.

        ``inputs_embeds`` is ignored: the anchor is an id by construction, and
        the backbone embeds inside its compiled region.
        """
        T = self.vllm_block_width
        if T is None:
            raise RuntimeError(
                "DeepseekV4DSparkDraft.vllm_block_width was never set; "
                "DeepseekV4DSparkVllm.__init__ is what sets it."
            )
        if input_ids.numel() % T:
            raise ValueError(
                f"DSpark draft got {input_ids.numel()} tokens, not a multiple "
                f"of the block width {T}. vLLM lays out exactly "
                "1 + num_speculative_tokens per request."
            )
        anchors = input_ids.view(-1, T)[:, 0]
        anchor_positions = positions.view(-1, T)[:, 0]
        normed, _hc_hidden = self.block_backbone(anchors, anchor_positions, T)
        return normed

    def write_combined_context_kv(
        self, ctx_hidden: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """Scatter the projected target context into every backbone stage.

        No slot mapping: each stage writes its own rolling window, addressed by
        absolute position from ATOM's forward context. See the module docstring.
        """
        for layer in self.context_layers:
            layer.write_context_kv(ctx_hidden, positions)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.layer_name for layer in self.context_layers]


class DeepseekV4DSparkVllm(ATOMMoEForCausalLM):
    """vLLM's DSpark drafting contract, backed by the ATOM draft module."""

    # The draft ships neither an embedding table nor an LM head (they live in
    # the target's `mtp.*` namespace and are shared); vLLM's loader consults
    # these before binding the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Plain attributes, not properties: vLLM's DSpark loader rebinds
        # `lm_head` to the target's head after construction.
        self.lm_head = None
        # The draft scores the full target vocabulary, so its sampled ids are
        # already target ids and need no remap table.
        self.draft_id_to_target_id = None
        self.model.vllm_block_width = vllm_block_width(vllm_config)

    def combine_hidden_states(self, aux_concat: torch.Tensor) -> torch.Tensor:
        return self.model.project_context(aux_concat)

    def precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings=None,
    ) -> None:
        """Write the target-derived context rows into the draft's own windows.

        ``slot_mappings`` is vLLM's, and is ignored by design -- ATOM owns the
        draft's KV (module docstring). It is still the dummy/profile signal:
        vLLM passes ``None`` there, where the block tables are placeholders and
        writing would clobber live entries.

        The context entered here is what ``write_context_kv`` reads its
        ``cu_seqlens_q`` and per-request ``state_slot_out`` from. vLLM's
        speculator calls this outside ``ATOMModelBase.forward``, so nothing else
        would open it.
        """
        if slot_mappings is None:
            return

        from atom.plugin.vllm.deepseek_v4_bridge import (
            atom_deepseek_v4_forward_context,
            bind_deepseek_v4_proxy_cache_views,
        )

        proxy_layer_name = self.__dict__.get("_deepseek_v4_proxy_layer_name")
        ready = bind_deepseek_v4_proxy_cache_views(
            self.model, self.vllm_config, proxy_layer_name
        )
        if not ready:
            # Proxy cache not bound yet: this is a dummy/profile pass whatever
            # vLLM said, and there is nothing real to write into.
            return
        with atom_deepseek_v4_forward_context(
            atom_config=self.atom_config,
            input_ids=None,
            positions=positions,
            force_dummy=False,
            state_model=self.model,
            meta_params=getattr(self.model, "_atom_v4_meta_params", None),
            slot_allocator=getattr(self.model, "_atom_v4_slot_allocator", None),
            proxy_layer_name=proxy_layer_name,
        ):
            self.model.write_combined_context_kv(hidden_states, positions)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._markov_head().markov_w1(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        # fp32: this bias lands inside the softmax that decides acceptance.
        weight = self._markov_head().markov_w2.weight
        return torch.matmul(markov_embed.float(), weight.float().t())

    def markov_argmax(
        self, base_logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Greedy next id per request, without materializing the [B, V] bias.

        Collapses the pair above plus the add and the argmax. The ids come back
        through a destination rather than a return value: ATOM's native loop
        writes one column of its own block, and vLLM's spelling wants a tensor,
        so allocate that one column's worth.
        """
        ids = torch.empty(
            base_logits.shape[0], dtype=torch.int64, device=base_logits.device
        )
        self._markov_head().sample_next(token_ids, base_logits, ids)
        return ids

    def map_draft_to_target(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return draft_token_ids

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def _markov_head(self):
        """The Markov head lives on the LAST backbone stage (checkpoint
        ``mtp.2.markov_head.*``), not on the draft root."""
        return self.model.context_layers[-1].markov_head
