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

* vLLM lays the block out as ``[anchor, MASK x (T-1)]`` at width
  ``num_query_per_req = num_speculative_steps`` -- the anchor IS the first
  prediction position, not a separate bonus query (``sample_from_anchor``,
  which this checkpoint leaves at its True default; see
  :func:`vllm_block_width`). ATOM builds ``[anchor, noise x (T-1)]`` at the
  same T (``deepseek_v4_dspark.py``). Both filler tokens
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

import logging

import torch

from atom.models import deepseek_v4 as deepseek_v4_base
from atom.models.deepseek_v4_dspark import DeepseekV4DSpark as DeepseekV4DSparkBase
from atom.plugin.vllm.model_wrapper import ATOMMoEForCausalLM
from atom.plugin.vllm.models.deepseek_v4 import DeepseekV4AttentionVllm, IndexerVllm

logger = logging.getLogger("atom")


def vllm_block_width(vllm_config) -> int:
    """The draft block's width in vLLM's layout.

    ``num_speculative_tokens``, NOT ``1 + num_speculative_tokens``.

    ``DSparkSpeculator.__init__`` branches on ``sample_from_anchor``, which
    defaults True and which DeepSeek-V4-Pro-0813 does not set (and cannot pick
    up later: ``SpeculativeConfig`` wraps the draft config in ``EAGLEConfig``
    only for eagle/eagle3/dflash, so DSpark's draft hf_config IS the target's
    V4 config). On that branch the anchor is the first PREDICTION position
    rather than a separate bonus query, so::

        num_query_per_req = num_speculative_steps          # 7, not 8
        _anchor_idx       = arange(max_num_reqs) * 7

    and vLLM's own lookahead sizing agrees (``config/vllm.py``: "the anchor
    itself is the first prediction position (no separate bonus query), so it
    needs exactly num_speculative_tokens lookahead slots").

    Native ATOM lands on the same number by its own route:
    ``DSparkProposer._resolve_mtp_k`` returns ``num_speculative_tokens`` and
    ``dspark_proposer.py`` spells ``self.mtp_k`` as "max_seqlen_qo = block
    width T". So T is 7 on both arms, and ATOM's ``[anchor, noise x (T-1)]``
    block is byte-identical to vLLM's ``[anchor, MASK x (T-1)]``.

    Getting this wrong is silent and nearly total: at stride 8 over a width-7
    layout only request 0's anchor is correct and every other request drafts
    from a MASK row.

    ``sample_from_anchor`` is re-read here rather than assumed, so a checkpoint
    that does set it False keeps the two layouts in step instead of silently
    desynchronising them by one again -- this mirrors ``DSparkSpeculator``'s own
    branch and must keep mirroring it.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    n = int(getattr(spec, "num_speculative_tokens", 0) or 0)
    draft_cfg = getattr(spec, "draft_model_config", None)
    hf = getattr(draft_cfg, "hf_config", None)
    sample_from_anchor = bool(getattr(hf, "sample_from_anchor", True))
    width = n if sample_from_anchor else 1 + n
    logger.info(
        "ATOM plugin: DSpark draft block width T=%d "
        "(num_speculative_tokens=%d, sample_from_anchor=%s).",
        width,
        n,
        sample_from_anchor,
    )
    return width


class DeepseekV4DSparkDraft(DeepseekV4DSparkBase):
    """ATOM's DSpark drafter built with the vLLM V4 attention/indexer variants.

    The class swap matches ``deepseek_v4_mtp.DeepseekV4MTP``: the draft's sparse
    indexer must split mixed batches the way the target's does under vLLM
    continuous batching, which is what ``IndexerVllm`` adds.
    """

    def __init__(self, *args, layer_offset: int | None = None, **kwargs):
        original_attn_cls = deepseek_v4_base.DeepseekV4Attention
        original_indexer_cls = deepseek_v4_base.Indexer
        deepseek_v4_base.DeepseekV4Attention = DeepseekV4AttentionVllm
        deepseek_v4_base.Indexer = IndexerVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            deepseek_v4_base.DeepseekV4Attention = original_attn_cls
            deepseek_v4_base.Indexer = original_indexer_cls
        # `layer_offset` is how the plugin tells a STANDALONE draft where the
        # target's layers end, so the draft's own layer names do not collide in
        # vLLM's static_forward_context. V4's DSpark already does that for
        # itself -- its stages are built as `DSparkLayer(args.n_layers + i,
        # ...)` -- so the argument is absorbed rather than forwarded. K3's draft
        # takes it because its stages number from zero.
        #
        # Checked rather than ignored: if vLLM ever reports a different target
        # depth than the config the draft was built from, the two numberings
        # disagree and the draft's layers alias the target's, which would be a
        # silent cache collision.
        if layer_offset is not None and int(layer_offset) != int(self.args.n_layers):
            raise ValueError(
                f"DSpark draft layer_offset={layer_offset} from vLLM disagrees "
                f"with the target depth this draft numbered itself from "
                f"({self.args.n_layers}). Its stages would alias the target's "
                "layer names in static_forward_context."
            )
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
        total = int(input_ids.numel())
        if total < 1:
            raise ValueError("DSpark draft got an empty batch.")
        # A REAL step is exactly `num_reqs * T` and every branch below is a
        # no-op for it. Dummy, memory-profiling and cudagraph-capture runs are
        # not: vLLM sizes those by a token budget, so they arrive padded to a
        # count that need not divide by T, and can even be shorter than one
        # block. Serve them by drafting whole blocks and reconciling the row
        # count, rather than reshaping across a block boundary -- which would
        # take an anchor from the middle of someone's block and silently draft
        # from the wrong token.
        num_blocks = max(1, total // T)
        anchor_idx = torch.arange(num_blocks, device=input_ids.device) * T
        anchor_idx = anchor_idx.clamp(max=total - 1)
        normed, _hc_hidden = self.block_backbone(
            input_ids[anchor_idx], positions[anchor_idx], T
        )
        # vLLM indexes the result by its own token count, so return that many
        # rows: truncate a short batch's block, zero-fill a padded tail.
        produced = normed.shape[0]
        if produced > total:
            return normed[:total]
        if produced < total:
            pad = normed.new_zeros((total - produced, normed.shape[-1]))
            return torch.cat([normed, pad], dim=0)
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

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[N, dim] -> [N, vocab]`` off the shared head.

        Not ``lm_head(hidden_states)``: V4's head is a ``ParallelHead`` whose
        ``forward`` also takes the mHC reduction arguments
        (``hc_fn``/``hc_scale``/``hc_base``/``norm``), because the target calls
        it with the un-reduced stack. The block drafter has already reduced and
        normed, so it wants the plain projection -- which is ``get_logits``,
        the same entry ATOM's native block sampler uses.
        """
        return self.model.head.get_logits(hidden_states)


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
            bind_deepseek_v4_proxy_cache_views,
            get_deepseek_v4_target_metadata,
        )
        from atom.utils.forward_context import (
            reset_forward_context,
            set_forward_context,
        )

        proxy_layer_name = self.__dict__.get("_deepseek_v4_proxy_layer_name")
        ready = bind_deepseek_v4_proxy_cache_views(
            self.model, self.vllm_config, proxy_layer_name
        )
        if not ready:
            # Proxy cache not bound yet: this is a dummy/profile pass whatever
            # vLLM said, and there is nothing real to write into.
            return

        # The TARGET's metadata and context, not the draft's. `hidden_states`
        # here is the target's ragged batch of every scheduled token, so
        # `write_context_kv` needs that batch's `cu_seqlens_q` spans and its
        # `scheduled_bs`; the draft's own metadata describes its [num_reqs x T]
        # block instead and would slice the wrong rows. `state_slot_out` is the
        # per-request ring slot, and the draft wants exactly the target's
        # mapping -- it has its own KV plane, not its own slot numbering.
        #
        # Taken from the stash rather than vLLM's forward context: the
        # speculator runs after the target's context has exited, so it is not
        # readable there.
        remembered = get_deepseek_v4_target_metadata()
        if remembered is None:
            raise RuntimeError(
                "DSpark draft cannot write its context KV: no target V4 "
                "metadata was recorded for this step. `write_context_kv` needs "
                "the target batch's cu_seqlens_q spans and per-request state "
                "slots."
            )
        target_md, target_context = remembered
        set_forward_context(
            attn_metadata=target_md,
            atom_config=self.atom_config,
            context=target_context,
            num_tokens=int(positions.numel()),
            in_hipgraph=False,
        )
        try:
            self.model.write_combined_context_kv(hidden_states, positions)
        finally:
            reset_forward_context()

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_draft_logits(hidden_states)

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
        """The names vLLM knows, which is the proxy layer -- NOT ATOM's own.

        vLLM's speculator maps each returned name to a KV cache group
        (``name_to_gid[name]``) to pick that layer's context slot mapping, so a
        name it never registered is a ``KeyError``. ATOM's internal stages
        (``mtp.0.attn`` ...) are exactly that: the draft's KV lives in ATOM's
        own rolling ring behind one opaque proxy layer, which is what was
        registered. One name, and the slot mapping it selects goes unused --
        see :meth:`precompute_and_store_context_kv`.
        """
        proxy_layer_name = self.__dict__.get("_deepseek_v4_proxy_layer_name")
        if proxy_layer_name is None:
            raise RuntimeError(
                "DSpark draft has no V4 proxy layer name; "
                "`DSparkDraftModel` must be in `_DEEPSEEK_V4_ARCHES`."
            )
        return [proxy_layer_name]

    def _markov_head(self):
        """The Markov head lives on the LAST backbone stage (checkpoint
        ``mtp.2.markov_head.*``), not on the draft root."""
        return self.model.context_layers[-1].markov_head
