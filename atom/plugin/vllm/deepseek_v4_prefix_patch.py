"""ATOM DeepSeek-V4 vLLM prefix-cache SWA-recompute patch.

V4's sliding-window (SWA) state is a per-request ring stored in a fixed
per-slot region of the ATOM proxy arena -- it is NOT keyed by a vLLM block, so
vLLM's block-level prefix cache never carries it. CSA/HCA compressed history,
by contrast, lives in the 128-token proxy pages and is reused for free on a
prefix-cache hit.

On a cross-request prefix hit the new request gets a fresh per-request state
slot whose SWA ring is empty; a non-block-aligned tail token whose SWA window
reaches back into the cached (not-re-forwarded) region would then read stale
ring data.

Fix (mirrors native ATOM scheduler "fix B'"): on a hit, drop the last
``ceil(max(win_with_spec, index_topk) / block_size)`` cached blocks so those
tail tokens are re-forwarded, repopulating the ring. The re-forwarded region is
>= the ring stride, so by the last prompt token ``prefix_swa_count`` collapses
to 0 and its whole window is served from the freshly computed extend KV.
Compressed-KV reuse is unaffected: ``n_committed = context_len // ratio`` and
``context_len = cached + scheduled`` is invariant under the shift.

The ``index_topk`` term is empirical and is the binding one on
DeepSeek-V4-Flash (512 vs a 128-token window) -- see the comment on
``warmup_tokens``. The same 512-token floor shows up with prefix caching off:
prompts shorter than ``index_topk`` are corrupt too, so the underlying defect is
one the sparse indexer has whenever fewer than ``index_topk`` rows have been
forwarded into the request's own state, and this rollback is a workaround for
it rather than a fix of it.

In plugin mode vLLM owns the scheduler / KVCacheManager, so the block drop is
applied by wrapping ``KVCacheManager.get_computed_blocks`` -- the single point
where vLLM computes the local prefix-cache hit length. It is only called when
``request.num_computed_tokens == 0`` (a genuine cross-request hit), never on a
chunked-prefill resume, whose SWA ring is already populated by prior chunks.
"""

import functools
import logging
import math
import os

logger = logging.getLogger("atom")


def _mark_v4_proxy_cache_mode(static_forward_context, is_profiling: bool) -> None:
    for layer in static_forward_context.values():
        if getattr(layer, "_atom_v4_proxy_layer", False):
            layer._atom_v4_profiling_kv_cache = is_profiling


_V4_PROXY_LAYER_MARKERS = (
    ".atom_deepseek_v4_proxy",
    ".atom_deepseek_v4_draft_proxy",
)


def _kv_cache_config_has_v4_proxy(kv_cache_config) -> bool:
    return any(
        any(
            marker in layer_name
            for marker in _V4_PROXY_LAYER_MARKERS
            for layer_name in group.layer_names
        )
        for group in kv_cache_config.kv_cache_groups
    )


def _kv_cache_config_needs_non_immediate_reuse(kv_cache_config) -> bool:
    return _kv_cache_config_has_v4_proxy(kv_cache_config) or bool(
        getattr(kv_cache_config, "has_mamba_layers", False)
    )


def apply_vllm_v4_block_reuse_patch() -> None:
    """Keep no-prefix-cache block reuse safe for ATOM stateful cache layouts.

    vLLM commit a82f1b388f changed non-caching pools to immediately reuse the
    blocks a request just freed. The V4 proxy allocation is a global arena: its
    fixed per-request SWA prefix and block-indexed CSA/HCA tails are carved
    across the physical vLLM page boundaries. Immediate block-id reuse therefore
    exposes stale compressed entries before the arena can safely recycle them.
    ATOM's GDN path likewise keeps recurrent state keyed by the Mamba block-table
    slots; immediate churn can recycle a slot while a mixed prefill/decode batch
    still references it.

    Mark only pools whose KV-cache groups contain an ATOM V4 proxy or Mamba/GDN
    state, then retain vLLM's pre-a82f free-queue ordering for those pools. Every
    ordinary MHA/MLA model keeps the upstream locality optimization.
    """
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_manager_init = KVCacheManager.__init__
    if not getattr(original_manager_init, "_atom_v4_block_reuse_patched", False):

        @functools.wraps(original_manager_init)
        def wrapped_manager_init(self, *args, **kwargs):
            original_manager_init(self, *args, **kwargs)
            kv_cache_config = kwargs.get("kv_cache_config")
            if kv_cache_config is None and args:
                kv_cache_config = args[0]
            if (
                kv_cache_config is not None
                and _kv_cache_config_needs_non_immediate_reuse(kv_cache_config)
            ):
                self.block_pool._atom_v4_proxy_arena = True
                logger.info(
                    "ATOM: using non-immediate KV block reuse for a packed V4 "
                    "or stateful Mamba/GDN cache"
                )

        wrapped_manager_init._atom_v4_block_reuse_patched = True
        KVCacheManager.__init__ = wrapped_manager_init

    original_free_blocks = BlockPool.free_blocks
    if getattr(original_free_blocks, "_atom_v4_block_reuse_patched", False):
        return

    @functools.wraps(original_free_blocks)
    def wrapped_free_blocks(self, ordered_blocks):
        if not getattr(self, "_atom_v4_proxy_arena", False) or self.enable_caching:
            return original_free_blocks(self, ordered_blocks)

        # a82f changed only the `enable_caching` branch inside free_blocks.
        # Temporarily select the old branch while preserving all other upstream
        # accounting/event logic and restore the real setting before returning.
        self.enable_caching = True
        try:
            return original_free_blocks(self, ordered_blocks)
        finally:
            self.enable_caching = False

    wrapped_free_blocks._atom_v4_block_reuse_patched = True
    BlockPool.free_blocks = wrapped_free_blocks
    logger.info("ATOM DeepSeek-V4: installed packed-proxy block reuse patch")


def apply_vllm_v4_profile_cache_patch() -> None:
    """Mark vLLM 0.26's temporary CUDA-graph profiling KV cache.

    The temporary cache intentionally contains only one block per captured
    request and cannot hold V4's fixed per-request SWA arena. The V4 forward
    must therefore stay on its existing dummy-attention path until vLLM
    installs the real cache.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.initialize_kv_cache
    if getattr(original, "_atom_v4_profile_cache_patched", False):
        return

    @functools.wraps(original)
    def wrapped_initialize_kv_cache(
        self,
        kv_cache_config,
        is_profiling: bool = False,
    ):
        result = original(
            self,
            kv_cache_config,
            is_profiling=is_profiling,
        )
        _mark_v4_proxy_cache_mode(
            self.compilation_config.static_forward_context,
            is_profiling,
        )
        return result

    wrapped_initialize_kv_cache._atom_v4_profile_cache_patched = True
    GPUModelRunner.initialize_kv_cache = wrapped_initialize_kv_cache


def _v4_sliding_window(vllm_config) -> int:
    hf = vllm_config.model_config.hf_config
    return int(getattr(hf, "sliding_window", 128) or 128)


def _group_block_sizes(manager):
    """Real block size per KV cache group, in group order.

    ``KVCacheManager.coordinator.single_type_managers`` is index-aligned with
    ``kv_cache_config.kv_cache_groups``, which is the same order
    ``KVCacheBlocks.blocks`` uses. Returns [] if the shape is not what we
    expect, and the caller falls back to the proxy block size.
    """
    try:
        return [m.block_size for m in manager.coordinator.single_type_managers]
    except AttributeError:
        return []


def _drop_swa_warmup_blocks(
    manager,
    computed_blocks,
    num_computed_tokens: int,
    shared_prefix_boundary: int,
    *,
    warmup_tokens: int,
):
    """Roll a prefix hit back by ``warmup_tokens`` so the tail is re-forwarded.

    vLLM allocates fresh blocks for the dropped tail and re-forwards those
    tokens, repopulating the SWA ring and the sparse indexer's rows; the
    deep-prefix blocks are still reused.

    The rollback is expressed in **tokens**, and converted to a block count
    separately for each group using that group's own block size. The earlier
    spelling dropped a fixed block *count* from every group and converted with a
    hard-coded 128 -- correct while V4 ran a single 128-token proxy group, wrong
    the moment DSpark adds a second group at block 64. There it removed 256
    tokens of draft coverage while charging 512, leaving that group holding
    blocks past the declared ``num_computed_tokens``; and because the count was
    the max across groups, a short draft hit could over-subtract and clamp an
    otherwise good hit to zero.
    """
    if num_computed_tokens <= 0 or warmup_tokens <= 0:
        return computed_blocks, num_computed_tokens, shared_prefix_boundary

    # warmup_tokens is a multiple of every group's block size (128 and 64 here),
    # and num_computed_tokens is scheduler-block aligned, so the difference stays
    # aligned for both groups.
    new_num_computed_tokens = max(0, num_computed_tokens - warmup_tokens)

    block_sizes = _group_block_sizes(manager)
    groups = list(computed_blocks.blocks)
    new_groups = []
    dropped_any = False
    for idx, group in enumerate(groups):
        block_list = list(group)
        block_size = (
            block_sizes[idx]
            if idx < len(block_sizes)
            else ATOM_DEEPSEEK_V4_BLOCK_SIZE
        )
        keep = min(len(block_list), new_num_computed_tokens // block_size)
        if keep != len(block_list):
            dropped_any = True
        new_groups.append(block_list[:keep])
    if not dropped_any:
        return computed_blocks, num_computed_tokens, shared_prefix_boundary

    new_blocks = manager.create_kv_cache_blocks(tuple(new_groups))
    return new_blocks, new_num_computed_tokens, shared_prefix_boundary


def apply_vllm_v4_prefix_swa_patch(vllm_config) -> None:
    """Enable DeepSeek-V4 prefix caching by dropping the SWA warmup blocks.

    Call only for a DeepSeek-V4 deployment with prefix caching enabled. The
    number of blocks to drop is derived once from ``vllm_config`` and captured
    in the wrapper closure, so non-V4 deployments (which never install this
    patch) are unaffected.
    """
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    from atom.plugin.vllm.deepseek_v4_bridge import (
        ATOM_DEEPSEEK_V4_BLOCK_SIZE,
        _v4_win_with_spec,
    )

    win_with_spec = _v4_win_with_spec(vllm_config, _v4_sliding_window(vllm_config))
    # The SWA ring's physical stride is win_with_spec = window + num_spec_tokens
    # (MTP draft tokens get their own ring slots). Rolling back ceil(stride /
    # block_size) whole blocks guarantees the re-forwarded region covers the full
    # ring, so the last prompt token reads its entire window from extend KV.
    # The SWA ring is not the only per-slot state a hit fails to carry: the
    # sparse indexer selects `index_topk` rows per token, and on this checkpoint
    # a hit that leaves fewer than `index_topk` freshly-forwarded tokens emits
    # another request's content. Measured on DeepSeek-V4-Flash-0731, TP4, with a
    # long shared prefix and greedy decoding -- rolling back 128/256/384 tokens
    # gives 1/6 correct, 512 gives 6/6, and 512 stays sufficient at 2K, 6K and
    # 17K-token prefixes, so the requirement is a constant, not a fraction.
    # index_topk (512) dominates win_with_spec (128) here; keep both terms so a
    # checkpoint with a wider window or a narrower topk still gets the max.
    index_topk = int(getattr(vllm_config.model_config.hf_config, "index_topk", 0) or 0)
    warmup_tokens = max(win_with_spec, index_topk)
    # Round the rollback up to a whole proxy block. The rollback is applied to
    # every KV cache group using that group's own block size, so it has to be a
    # multiple of each of them; the proxy's 128 is the coarsest in play and is a
    # multiple of the DSpark draft group's 64.
    warmup_blocks = math.ceil(warmup_tokens / ATOM_DEEPSEEK_V4_BLOCK_SIZE)
    # Escape hatch for re-bisecting the bound on another checkpoint. A value
    # past the prompt length degenerates to "no reuse", i.e. caching off. Still
    # expressed in proxy blocks -- that is what the bisection above was run in.
    override = os.environ.get("ATOM_V4_PREFIX_WARMUP_BLOCKS")
    if override:
        warmup_blocks = int(override)
    if warmup_blocks <= 0:
        return
    warmup_tokens = warmup_blocks * ATOM_DEEPSEEK_V4_BLOCK_SIZE

    original = KVCacheManager.get_computed_blocks
    if getattr(original, "_atom_v4_prefix_swa_patched", False):
        return

    @functools.wraps(original)
    def wrapped_get_computed_blocks(self, request):
        computed_blocks, num_computed_tokens, shared_prefix_boundary = original(
            self, request
        )
        return _drop_swa_warmup_blocks(
            self,
            computed_blocks,
            num_computed_tokens,
            shared_prefix_boundary,
            warmup_tokens=warmup_tokens,
        )

    wrapped_get_computed_blocks._atom_v4_prefix_swa_patched = True
    KVCacheManager.get_computed_blocks = wrapped_get_computed_blocks
    logger.info(
        "ATOM DeepSeek-V4: prefix caching enabled with SWA recompute "
        "(roll back last %d token(s) per hit = %d proxy block(s), "
        "win_with_spec=%d, index_topk=%d).",
        warmup_tokens,
        warmup_blocks,
        win_with_spec,
        index_topk,
    )


def apply_vllm_v4_profiling_min_blocks_patch(vllm_config=None) -> None:
    """Make vLLM's cudagraph-profiling KV cache big enough for the V4 proxy.

    ``GPUModelRunner._init_minimal_kv_cache_for_profiling`` allocates a
    deliberately tiny KV cache -- ``num_gpu_blocks_override =
    min(max_num_reqs, max_cudagraph_capture_size)`` -- runs two throwaway
    graph captures to measure their memory, then frees it. "One block per
    sequence" is the right minimum for a normal paged backend, whose per-page
    bytes are self-contained.

    The DeepSeek-V4 proxy page is not self-contained: ``_proxy_page_bytes``
    amortizes the *fixed* SWA ring (``num_layers * max_num_seqs *
    win_with_spec`` entries, ~1.9 GB here) evenly across
    ``ceil(max_model_len / 128)`` pages. That is exact at exactly that block
    count and short by ``swa_bytes - num_blocks * ceil(swa_bytes/min_blocks)``
    for anything smaller, so the profiling cache fails to carve its views:

        ValueError: DeepSeek V4 proxy cache too small: need N, have M

    Measured on DeepSeek-V4-Flash-0731, TP4, max_model_len=131072,
    max_num_seqs=512: profiling asks for 512 blocks, the proxy needs 1024, and
    the slice dies ~halfway through the 46 layers. It is independent of
    ``--gpu-memory-utilization`` (0.90 and 0.93 give byte-identical numbers)
    because the profiling cache is sized by an override, not by free memory.

    Raising the floor to the amortization's own block count restores the
    invariant. The blocks are transient (``_cleanup_profiling_kv_cache`` frees
    them before the real allocation) and this only ever raises the count, so a
    deployment that never enters this path is unaffected.

    Only reached when ``cudagraph_mode != NONE``. Both the target-only and the
    spec-decode shapes reach it; a spec-decode draft adds a second KV cache
    group but is not what triggers the shortfall.

    ``vllm_config`` is optional and only seeds the log line. The floor is
    (re)derived from ``self.vllm_config`` at call time, because this patch has
    to be installable from ``register_model()`` -- which runs in *every*
    process -- and not only from the platform config hook, which the mp workers
    run only when something reconstructs a ``VllmConfig`` there (spec decode
    does; a target-only launch does not). Installing it per-config left the
    workers unpatched on a no-spec launch and the profiling slice died in the
    worker while the API server logged a successful patch.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from atom.plugin.vllm.deepseek_v4_bridge import _v4_proxy_min_blocks

    original = getattr(GPUModelRunner, "_init_minimal_kv_cache_for_profiling", None)
    if original is None or getattr(original, "_atom_v4_min_blocks_patched", False):
        return

    def _floor_for(config) -> int:
        mc = getattr(config, "model_config", None)
        if mc is None or not any(
            "DeepseekV4" in str(a) for a in (getattr(mc, "architectures", None) or [])
        ):
            return 0
        return _v4_proxy_min_blocks(config)

    # The block count is computed *inside* the original (it writes
    # num_gpu_blocks_override itself, so presetting the override is ignored)
    # and neither max_num_reqs nor max_cudagraph_capture_size can be nudged
    # without also resizing the metadata builders the same call constructs.
    # Intercept the one thing that carries the count instead: the KVCacheConfig
    # the original builds, scaling each tensor to `floor` blocks.
    import vllm.v1.core.kv_cache_utils as kv_cache_utils

    @functools.wraps(original)
    def wrapped_init_minimal_kv_cache_for_profiling(self):
        floor = _floor_for(getattr(self, "vllm_config", None))
        if floor <= 0:
            return original(self)

        inner = kv_cache_utils.get_kv_cache_config_from_groups

        @functools.wraps(inner)
        def floored(*args, **kwargs):
            config = inner(*args, **kwargs)
            old = int(config.num_blocks)
            if old <= 0 or old >= floor:
                return config
            for tensor in config.kv_cache_tensors:
                tensor.size = (tensor.size // old) * floor
            config.num_blocks = floor
            return config

        kv_cache_utils.get_kv_cache_config_from_groups = floored
        try:
            return original(self)
        finally:
            kv_cache_utils.get_kv_cache_config_from_groups = inner

    wrapped_init_minimal_kv_cache_for_profiling._atom_v4_min_blocks_patched = True
    GPUModelRunner._init_minimal_kv_cache_for_profiling = (
        wrapped_init_minimal_kv_cache_for_profiling
    )
    logger.info(
        "ATOM DeepSeek-V4: cudagraph-profiling KV cache floor installed "
        "(%s; the proxy page amortizes the SWA ring over exactly that many).",
        f"{_floor_for(vllm_config)} blocks"
        if vllm_config is not None
        else "resolved per model runner",
    )
