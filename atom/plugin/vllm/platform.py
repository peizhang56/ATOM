"""ATOM vLLM platform integration."""

import logging
import os

from atom.utils import envs

logger = logging.getLogger("atom")

# This flag is used to enable the vLLM plugin mode.
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN

# Largest single-forward token count we allow for DeepSeek-V4 when chunked
# prefill is disabled. Beyond this, a single forward overflows int32 element
# offsets in per-token Triton kernels (num_tokens * hidden > 2**31), surfacing
# as an "illegal memory access". Chunked prefill keeps each forward small and
# is the supported path for long context; this bound only guards the
# non-chunked fallback. Override with the env var below.
_V4_MAX_SINGLE_FORWARD_TOKENS = 131072
_V4_MAX_SINGLE_FORWARD_TOKENS_ENV = "ATOM_V4_MAX_SINGLE_FORWARD_TOKENS"

# Escape hatch for the piecewise-cudagraph demotion below.
_V4_KEEP_PIECEWISE_CG_ENV = "ATOM_V4_KEEP_PIECEWISE_CUDAGRAPH"


def _is_deepseek_v4(model_config) -> bool:
    arches = getattr(model_config, "architectures", None) or []
    return any("DeepseekV4" in str(a) for a in arches)


def _chunked_prefill_on(scheduler_config) -> bool:
    return bool(
        getattr(scheduler_config, "chunked_prefill_enabled", False)
        or getattr(scheduler_config, "enable_chunked_prefill", False)
    )


def _demote_piecewise_cudagraph(vllm_config) -> None:
    """Drop the piecewise cudagraph component when there is nothing to split on.

    vLLM already knows this rule. ``CompilationConfig.set_splitting_ops_for_v1``
    demotes ``PIECEWISE -> NONE`` and ``FULL_AND_PIECEWISE -> FULL`` when
    ``splitting_ops`` is empty, because "piecewise compilation with empty
    splitting_ops does not contain piecewise cudagraph". But that whole method
    early-returns for ``mode != CompilationMode.VLLM_COMPILE``, so on a plugin
    path -- where ATOM owns the model and vLLM compiles nothing (``mode`` is
    ``NONE``, ``splitting_ops`` is ``[]``) -- the demotion never runs and a
    piecewise mode survives into the runner.

    What survives is not harmless. With no splitting ops the "piecewise" graph is
    the *whole* model, ATOM's V4 attention included, and vLLM's mixed-mode
    dispatch (``gpu/cudagraph_utils.py``) builds those descriptors with
    ``num_reqs=None`` on purpose: a piecewise graph is assumed replayable for any
    batch under the captured token count, because break-point kernels re-read the
    real batch from the forward context and in-graph kernels self-pad from
    ``slot_mapping``. V4 attention satisfies neither -- its grids and per-fwd
    index geometry are baked from the batch it was captured with -- so replaying
    an 8-token uniform-decode graph for, say, a 2-request 6+1 token batch reads
    out of bounds and the process dies on a GPU memory fault with no traceback.

    Apply vLLM's own rule here, with ``FULL_DECODE_ONLY`` rather than ``FULL`` as
    the target: V4 prefill is eager by construction (see
    ``AtomDeepseekV4ProxyMetadataBuilder._build_and_attach_atom_v4_md``), so a
    full graph over mixed batches is not capturable either.
    """
    if os.environ.get(_V4_KEEP_PIECEWISE_CG_ENV, "") not in ("", "0"):
        return
    cc = getattr(vllm_config, "compilation_config", None)
    if cc is None:
        return
    from vllm.config import CompilationMode, CUDAGraphMode

    if getattr(cc, "mode", None) == CompilationMode.VLLM_COMPILE:
        return  # vLLM compiles the model itself; its own demotion applies.
    if getattr(cc, "splitting_ops", None):
        return
    mode = getattr(cc, "cudagraph_mode", None)
    if mode is None or not mode.has_piecewise_cudagraphs():
        return
    demoted = (
        CUDAGraphMode.FULL_DECODE_ONLY
        if mode.has_full_cudagraphs()
        else CUDAGraphMode.NONE
    )
    cc.cudagraph_mode = demoted
    logger.warning(
        "ATOM DeepSeek-V4: cudagraph_mode %s requests piecewise cudagraphs, but "
        "the model is ATOM-owned so vLLM compiles nothing and splitting_ops is "
        "empty -- a 'piecewise' graph would be the whole model, replayed for "
        "batches whose query-length structure differs from capture. Demoting to "
        "%s (set %s=1 to keep the requested mode).",
        mode.name,
        demoted.name,
        _V4_KEEP_PIECEWISE_CG_ENV,
    )


def _enforce_deepseek_v4_constraints(vllm_config) -> None:
    """Apply V4-specific plugin constraints.

    1. Enable prefix caching via SWA recompute: V4's per-request SWA
       sliding-window ring is not carried by vLLM's block-level prefix cache
       (only the CSA/HCA compressed pages are). Rather than disable caching, we
       install a KVCacheManager patch that, on a prefix hit, drops the last
       ``ceil(win_with_spec / block_size)`` cached blocks so the SWA tail is
       re-forwarded and the ring is repopulated (mirrors native ATOM "fix B'").
       See ``deepseek_v4_prefix_patch``.

    2. Guard the non-chunked oversized forward: with chunked prefill off, vLLM
       couples max_num_batched_tokens to max_model_len, so a native max_model_len
       forces a single ~max_model_len-token forward that overflows int32 element
       offsets in per-token kernels. Fail fast with an actionable error instead
       of crashing with "illegal memory access". Enable chunked prefill for long
       context.

    3. Demote piecewise cudagraph modes, a demotion vLLM skips on a plugin path.
       See ``_demote_piecewise_cudagraph``.
    """
    mc = getattr(vllm_config, "model_config", None)
    if mc is None or not _is_deepseek_v4(mc):
        return

    _demote_piecewise_cudagraph(vllm_config)

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None and getattr(
        cache_config, "enable_prefix_caching", False
    ):
        from atom.plugin.vllm.deepseek_v4_prefix_patch import (
            apply_vllm_v4_prefix_swa_patch,
        )

        apply_vllm_v4_prefix_swa_patch(vllm_config)

    # Independent of prefix caching: vLLM's cudagraph memory profiling
    # allocates a "one block per sequence" KV cache, which is smaller than the
    # block count the V4 proxy page amortizes its SWA ring over.
    from atom.plugin.vllm.deepseek_v4_prefix_patch import (
        apply_vllm_v4_profiling_min_blocks_patch,
    )

    apply_vllm_v4_profiling_min_blocks_patch(vllm_config)

    sc = getattr(vllm_config, "scheduler_config", None)
    if sc is None or _chunked_prefill_on(sc):
        return

    try:
        max_single = int(
            os.environ.get(
                _V4_MAX_SINGLE_FORWARD_TOKENS_ENV, _V4_MAX_SINGLE_FORWARD_TOKENS
            )
        )
    except (TypeError, ValueError):
        max_single = _V4_MAX_SINGLE_FORWARD_TOKENS

    mnbt = int(getattr(sc, "max_num_batched_tokens", 0) or 0)
    max_model_len = int(getattr(mc, "max_model_len", 0) or 0)
    if mnbt > max_single:
        msg = (
            "DeepSeek-V4 with chunked prefill disabled requires a single forward "
            f"of up to max_num_batched_tokens={mnbt} tokens (coupled to "
            f"max_model_len={max_model_len}). That exceeds the safe single-forward "
            f"bound ({max_single}); a forward this large overflows int32 element "
            "offsets in per-token kernels and crashes with an illegal memory "
            "access. Enable chunked prefill (enable_chunked_prefill=True) to serve "
            "this context length, or lower max_model_len. Set "
            f"{_V4_MAX_SINGLE_FORWARD_TOKENS_ENV} to override this bound."
        )
        logger.error(msg)
        raise ValueError(msg)


if not disable_vllm_plugin:
    from vllm.platforms.rocm import RocmPlatform

    class ATOMPlatform(RocmPlatform):
        """ATOM platform wrapper.

        Attention backend selection is owned by ATOM's vLLM attention layers
        (`AttentionForVllm*`). We intentionally do not override
        `get_attn_backend_cls()` here, so any fallback vLLM standard attention
        keeps ROCmPlatform's native backend selection.
        """

        @classmethod
        def check_and_update_config(cls, vllm_config) -> None:
            super().check_and_update_config(vllm_config)
            _enforce_deepseek_v4_constraints(vllm_config)

else:
    ATOMPlatform = None


def install_platform_config_hook() -> None:
    """Run ATOMPlatform's config hook even when the platform plugin lost the race.

    vLLM resolves `current_platform` on first read and memoizes it forever. On
    builds where `import vllm` reads it during its own import -- 0.25.x does,
    via `vllm/__init__.py` -> `env_override` -> `utils.torch_utils`, whose module
    body runs `PIN_MEMORY = is_pin_memory_available()` -- that read lands while
    `vllm.model_executor` is only half-imported. `register_platform()` then dies
    inside `_register_mxfp8_quantization_config()` on a circular ImportError,
    vLLM swallows it (`except Exception: pass`), falls back to the builtin
    platform and caches it. Later resolutions succeed and log "Platform plugin
    atom is activated", but nothing re-reads `_current_platform`, so ATOMPlatform
    is never live and `check_and_update_config` never runs.

    For DeepSeek-V4 the casualty is `_enforce_deepseek_v4_constraints`: the
    prefix-cache SWA-recompute patch is never installed, so a cross-request
    prefix hit reads an empty SWA ring and the model emits another request's
    content. `check_and_update_config` is ATOMPlatform's only override, so
    re-attach that one hook to whichever platform actually went live. Called
    from `register_model()`, which vLLM runs from `create_engine_config()`
    before it calls the platform hook. Idempotent.
    """
    if disable_vllm_plugin:
        return
    from vllm.platforms import current_platform

    live_cls = type(current_platform)
    if ATOMPlatform is not None and issubclass(live_cls, ATOMPlatform):
        return  # plugin won the race; the hook is already in the MRO
    original = live_cls.check_and_update_config
    if getattr(original, "_atom_config_hook_installed", False):
        return

    def patched(cls, vllm_config) -> None:
        original(vllm_config)
        _enforce_deepseek_v4_constraints(vllm_config)

    patched._atom_config_hook_installed = True
    live_cls.check_and_update_config = classmethod(patched)
    logger.info(
        "ATOM plugin: re-attached check_and_update_config to the live platform "
        "%s (the platform plugin lost vLLM's memoized current_platform race).",
        live_cls.__name__,
    )
