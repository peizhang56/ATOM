# SPDX-License-Identifier: Apache-2.0
"""Keep the DSpark draft's sliding-window KV group out of the prefix cache.

WHY. With ``--speculative-config '{"method":"dspark"}'`` the plugin runs with two
KV cache groups instead of one:

  * ``model.layers.0.atom_deepseek_v4_proxy`` -- ATOM's V4 proxy blob,
    ``FullAttentionSpec``, block 128;
  * ``model.layers.{43,44,45}.attn.swa_cache`` -- the DSpark draft's three MLA
    layers, ``SlidingWindowMLASpec``, block **64**, window 128.

``HybridKVCacheCoordinator.find_longest_cache_hit`` runs full attention first and
then feeds its answer into every other group as ``max_length``, taking back
whatever that group returns (``curr_hit_length = _new_hit_length``). So the draft
group does not merely fail to contribute a hit -- it **caps** the reconciled hit
for the whole request.

And it always will. ``SlidingWindowManager`` needs ``cdiv(window - 1, 64) + 1 =
3`` contiguous cached blocks found right-to-left at the shared-prefix boundary,
but ``remove_skipped_blocks`` frees every draft block that leaves the 128-token
window *while the previous request is still decoding*. After 917 output tokens
the blocks at a 108K prefix boundary are long gone. Whatever stale run happens to
have survived eviction sets the hit length, which is why the measured rate is a
noisy 17.4% against ATOM native's 84.9% on the same workload -- native has no
such group at all, because its draft context is a private rolling window rebuilt
each step from aux hidden states.

WHY IT IS SAFE TO SKIP THE LOOKUP. The draft only ever reads its last 128 tokens,
and ``deepseek_v4_prefix_patch._drop_swa_warmup_blocks`` already forces the last
``max(win_with_spec, index_topk) = 512`` tokens of every hit to be re-forwarded.
512 >= 128, so by construction every draft position still inside the window at
the end of the prompt had its full window present during that re-forward. The
draft's deep prefix is dead weight: it is never read, and looking it up costs the
target its cache hit.

So this module registers a ``SlidingWindowMLASpec`` subclass for the draft layers
whose manager reports "I never constrain the hit" and never publishes blocks to
the prefix-cache hash map. Everything else -- page size, memory accounting,
in-window allocation, ``remove_skipped_blocks`` recycling -- is inherited
unchanged, so the KV budget is unaffected.

This uses vLLM's documented out-of-tree extension point
(``vllm/v1/kv_cache_spec_registry.py``: "Out-of-tree platforms can define custom
specs and managers by using the @register_kv_cache_spec decorator"), not a
monkeypatch of the coordinator.

UNMEASURED as of 2026-08-19: written against the source, never run -- the node
was fully occupied. See reports/vllm-atom-dspark-analysis-2026-08-19.md for the
validation plan.
"""

import dataclasses
import logging

logger = logging.getLogger("atom")

_spec_cls = None
_registered = False
# The (spec, manager) pair once built, so a second call -- or a pickle lookup
# through the module `__getattr__` -- returns the SAME classes. Rebuilding would
# hand out a second, unequal type and pickle would resolve to whichever the
# module happened to cache last.
_built = None


def _build_spec_and_manager():
    """Define the spec/manager pair lazily.

    Deferred because importing ``vllm.v1.core.single_type_kv_cache_manager`` at
    module import time drags in vLLM's scheduler core before the plugin has
    finished registering models.
    """
    global _built
    if _built is not None:
        return _built

    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class DSparkDraftSWAMLASpec(SlidingWindowMLASpec):
        """Byte-for-byte a ``SlidingWindowMLASpec``; only the type differs.

        Adding no fields keeps ``page_size_bytes``, ``max_memory_usage_bytes``
        and the ``__post_init__`` alignment padding exactly as they were, so KV
        pool sizing is identical to the unpatched build. The subclass exists
        purely so the registry can route it to a different manager.
        """

    class DSparkDraftSWAManager(SlidingWindowManager):
        """A sliding-window manager that abstains from the prefix cache.

        Only the two prefix-cache entry points are overridden. Allocation,
        ``remove_skipped_blocks`` recycling and the admission cap all stay on the
        inherited sliding-window path, which is what keeps the draft pool small.
        """

        @classmethod
        def find_longest_cache_hit(
            cls,
            block_hashes,
            max_length: int,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            drop_eagle_block: bool,
            alignment_tokens: int,
            dcp_world_size: int = 1,
            pcp_world_size: int = 1,
        ):
            """Never shorten the candidate hit; never claim a real block.

            The coordinator's contract is that each group returns the longest
            prefix *it* can serve and the reconciled hit is the minimum. We
            return the candidate unchanged so the draft group drops out of that
            minimum, backed entirely by ``null_block``: the tokens are declared
            computed, but no storage is claimed for them. Nothing reads that
            region -- the draft's window is 128 tokens and the last 512 are
            re-forwarded into freshly allocated blocks by
            ``_drop_swa_warmup_blocks``. A leading run of ``null_block`` is the
            normal sliding-window shape, so the free path already tolerates it
            (``_remove_blocks_in_range`` stops at the first null).

            ``drop_eagle_block`` still has to be honoured: the coordinator hands
            us ``candidate + block_size`` when it is set and expects one block
            back off the end, so subtract it to land on the candidate again.
            """
            block_size = kv_cache_spec.block_size
            hit_length = max_length
            if drop_eagle_block:
                hit_length = max(0, hit_length - block_size)
            # The coordinator only accepts alignment-aligned hit lengths.
            if alignment_tokens:
                hit_length -= hit_length % alignment_tokens
            hit_length -= hit_length % block_size
            num_blocks = hit_length // block_size
            computed_blocks = tuple(
                [block_pool.null_block] * num_blocks for _ in kv_cache_group_ids
            )
            return computed_blocks, hit_length

        def cache_blocks(self, request, num_tokens, retention_interval=None) -> None:
            """Publish nothing to the prefix-cache hash map.

            The lookup above never consults it, so an entry here could only keep
            a draft block hash-resident for a hit that can never be taken.
            """
            return

    # The KVCacheConfig carrying these specs is pickled and broadcast to every
    # worker (`multiproc_executor.collective_rpc` -> `shm_broadcast.enqueue`),
    # and pickle resolves a class by `__module__` + `__qualname__`. A class
    # defined inside a function has a `<locals>` qualname that resolves nowhere,
    # so the enqueue dies with "Can't pickle local object". Rewrite both to this
    # module's namespace; the module-level `__getattr__` below builds the pair
    # on demand, so the lookup also succeeds in a worker that never called
    # `ensure_registered` itself.
    for cls in (DSparkDraftSWAMLASpec, DSparkDraftSWAManager):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__

    _built = (DSparkDraftSWAMLASpec, DSparkDraftSWAManager)
    return _built


def __getattr__(name):
    """Resolve the lazily-built classes by name, for pickle.

    PEP 562 module ``__getattr__``: only consulted when the attribute is not
    already in the module dict, so it costs nothing on the normal path and
    keeps the vLLM imports out of module import time.
    """
    if name in ("DSparkDraftSWAMLASpec", "DSparkDraftSWAManager"):
        spec_cls, manager_cls = _built or _build_spec_and_manager()
        return {
            "DSparkDraftSWAMLASpec": spec_cls,
            "DSparkDraftSWAManager": manager_cls,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ensure_registered():
    """Register the draft spec/manager pair, once, after vLLM's built-ins.

    Order matters. ``KVCacheSpecRegistry._ensure_registered`` populates the
    built-in specs only ``if not _REGISTRY_KVCACHESPEC_LIST``. Registering ours
    into an empty registry would make that guard short-circuit and every
    built-in spec would go unregistered, so force the built-in pass first.

    vLLM does offer a tidier seam -- ``register_all_kvcache_specs`` ends with
    ``current_platform.register_custom_kv_cache_specs(vllm_config)``, so an
    override on ``ATOMPlatform`` would be called at exactly the right moment.
    Not used, because ``ATOMPlatform`` frequently is not the live platform: see
    ``platform.install_platform_config_hook``, which exists solely because vLLM
    memoizes ``current_platform`` before the plugin registers. Hanging this off
    the platform class would inherit that race; registering from the group
    builder does not.
    """
    global _spec_cls, _registered
    if _registered:
        return _spec_cls

    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    KVCacheSpecRegistry._ensure_registered()

    spec_cls, manager_cls = _build_spec_and_manager()
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=spec_cls,
        manager_class=manager_cls,
        # Stay grouping-compatible with plain sliding-window MLA so
        # `UniformTypeKVCacheSpecs.is_uniform_type` still accepts the three
        # draft layers as a single group.
        uniform_type_base_spec=SlidingWindowMLASpec,
    )
    _spec_cls = spec_cls
    _registered = True
    return spec_cls


def convert_draft_specs(draft_specs):
    """Retype the DSpark draft's sliding-window specs, leaving field values intact.

    Returns the input unchanged when the specs are not plain sliding-window MLA
    (a non-DSpark draft such as MTP keeps vLLM's stock behaviour) or when
    anything about the conversion does not hold -- a prefix-cache optimisation
    must never be the reason a server fails to start.
    """
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    if not draft_specs:
        return draft_specs
    if not all(type(spec) is SlidingWindowMLASpec for spec in draft_specs.values()):
        return draft_specs

    try:
        spec_cls = ensure_registered()
        converted = {}
        for name, spec in draft_specs.items():
            # Copy only the init fields: `page_size_padded` is one of them and
            # `__post_init__` recomputes it from `real_page_size_bytes`, which
            # is derived from block_size/dtype rather than from the padded
            # value, so the round-trip is idempotent.
            kwargs = {
                f.name: getattr(spec, f.name)
                for f in dataclasses.fields(spec)
                if f.init
            }
            converted[name] = spec_cls(**kwargs)
    except Exception:
        logger.warning(
            "ATOM DeepSeek-V4: could not retype the DSpark draft SWA specs; "
            "falling back to vLLM's stock sliding-window manager. The draft "
            "group will cap the target's prefix-cache hit.",
            exc_info=True,
        )
        return draft_specs

    logger.info(
        "ATOM DeepSeek-V4: DSpark draft SWA group (%d layers) excluded from "
        "prefix-cache hit reconciliation; its window is rebuilt by the SWA "
        "recompute.",
        len(converted),
    )
    return converted
