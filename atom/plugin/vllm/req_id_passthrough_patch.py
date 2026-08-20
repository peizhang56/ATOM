"""Expose the current step's request ids (CPU, batch-ordered) to ATOM builders.

The DeepSeek-V4 proxy metadata build needs a stable per-request key to assign a
state slot (its SWA ring + compressor state). Previously it derived that key
from ``block_table_tensor[:, 0]`` with a ``.cpu()`` copy, which forces a host<->
device sync and leaves a large bubble on the decode stream even though the copy
itself is tiny.

vLLM already has the canonical, host-resident key: ``input_batch.req_ids``. By
the time attention metadata is built it has been reordered together with the
block table / seq_lens rows (``InputBatch.swap_states``), so ``req_ids[i]``
lines up with row ``i`` of every per-request tensor.

This patch wraps two GPUModelRunner methods to snapshot ``req_ids`` into a
thread-local for the duration of each call:

* ``_build_attention_metadata`` -- constructs ``CommonAttentionMetadata`` *and*
  drives the target ``builder.build()`` in one synchronous call.
* ``propose_draft_token_ids`` -- drives the MTP/Eagle drafter, which (in current
  vLLM) builds its *own* attention metadata via
  ``SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata`` ->
  ``build_for_drafting`` -> the ATOM V4 bridge, entirely outside
  ``_build_attention_metadata``. Without this second wrap the thread-local is
  unset during drafting and the V4 slot allocator's fail-fast contract trips.

vLLM has **two unrelated** ``GPUModelRunner`` classes and both need covering:

* V1 -- ``vllm.v1.worker.gpu_model_runner`` -- the methods named above.
* V2 -- ``vllm.v1.worker.gpu.model_runner`` -- a different class (not a
  subclass) with a different decomposition: there is no
  ``_build_attention_metadata`` and no ``propose_draft_token_ids``, and the
  batch-ordered ids live on a per-step ``InputBatch`` value rather than on
  ``self.input_batch``. See ``_apply_v2_patch`` for the three seams used there.

Which class runs is *not* a free choice: ``VllmConfig.use_v2_model_runner``
forces V2 whenever ``speculative_config.method`` is ``dspark`` (config/vllm.py),
so every DSpark run -- the whole point of the ATOM+DSpark recipe -- takes the V2
path. A V1-only patch is a silent no-op there and the V4 slot allocator's
fail-fast trips on the first real batch (the sparse-MLA kernel warmup, in
practice).

The drafter reuses the target step's ``input_batch`` ordering (pure decodes were
already pulled to the front and the batch is not re-reordered before the draft
forward), so ``req_ids[i]`` still aligns with row ``i`` of the draft metadata --
the same invariant the target build relies on. ATOM's V4 metadata builder reads
the snapshot via ``get_current_req_ids()`` and keys slot allocation on it, with
no D2H. All of this lives in ATOM; no vLLM source is modified.
"""

from __future__ import annotations

import functools
import logging
import threading

logger = logging.getLogger("atom")

_req_id_local = threading.local()


def get_current_req_ids() -> list[str] | None:
    """Return the current step's batch-ordered request ids, or None.

    Valid only inside an attention metadata builder's ``build()`` for either the
    target or the draft: on V1 that means ``_build_attention_metadata`` or
    ``propose_draft_token_ids`` is on the stack; on V2, ``execute_model`` after
    ``prepare_inputs`` has run, or ``sample_tokens``. Returns the empty list for
    a batch with no real requests (V2 dummy runs and cudagraph capture), and None
    outside any of those scopes or if the pass-through patch was not applied --
    callers must treat None as "fall back to the device-side key".
    """
    return getattr(_req_id_local, "req_ids", None)


def _wrap_with_req_id_snapshot(cls, method_name: str) -> bool:
    """Wrap ``cls.method_name`` to expose batch-ordered req_ids as a thread-local.

    The wrapped method snapshots ``self.input_batch.req_ids`` for the duration of
    the call so ATOM metadata builders invoked transitively can read it via
    ``get_current_req_ids()`` with no device sync. Idempotent.
    """
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        prev = getattr(_req_id_local, "req_ids", None)
        try:
            # Snapshot now: req_ids is already batch-reordered (swap_states ran
            # in _prepare_inputs) so it aligns with the per-request rows the
            # builder sees -- for both the target build and the draft proposal,
            # which reuses this same ordering. A copy keeps it stable even if the
            # batch mutates later in the step.
            _req_id_local.req_ids = list(self.input_batch.req_ids)
        except Exception:
            _req_id_local.req_ids = None
        try:
            return original(self, *args, **kwargs)
        finally:
            _req_id_local.req_ids = prev

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)
    return True


def _wrap_v2_scope(cls, method_name: str) -> bool:
    """Open a req_id scope for the duration of ``cls.method_name`` (V2 runner).

    The V2 ``execute_model`` does not know the batch order at entry -- it is
    ``prepare_inputs`` (called a few lines in) that computes it. So this wrapper
    only *scopes* the thread-local: on the way in it publishes the EMPTY list and
    on the way out restores the previous value, and ``_wrap_v2_prepare_inputs``
    fills in the real order. That keeps dummy / cudagraph-capture runs -- which
    never call ``prepare_inputs`` -- from reading the previous real step's ids.

    Empty list, not None: the two mean different things to the V4 builder. None
    is "the passthrough patch is not installed", which is a contract violation it
    fails fast on; ``[]`` is "patch active, this batch has no real requests",
    which falls back to throwaway arange state slots. A dummy batch is the
    latter. (On V1 this distinction comes for free -- ``self.input_batch.req_ids``
    is simply empty during capture.)
    """
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        prev = getattr(_req_id_local, "req_ids", None)
        _req_id_local.req_ids = []
        try:
            return original(self, *args, **kwargs)
        finally:
            _req_id_local.req_ids = prev

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)
    return True


def _wrap_v2_prepare_inputs(cls) -> bool:
    """Publish the batch order as soon as the V2 runner computes it.

    ``prepare_inputs`` returns the step's ``InputBatch``; its ``req_ids`` are
    already in batch order (``sort_batch_req_ids`` pulled the pure decodes to the
    front) and line up with row ``i`` of every per-request tensor, exactly as on
    V1. Everything that reads the snapshot -- ``model_state.prepare_attn`` and
    the model forward -- runs after this point inside the same
    ``execute_model`` call, whose wrapper bounds the scope.
    """
    original = getattr(cls, "prepare_inputs", None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        input_batch = original(self, *args, **kwargs)
        try:
            _req_id_local.req_ids = list(input_batch.req_ids)
        except Exception:
            _req_id_local.req_ids = None
        return input_batch

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    setattr(cls, "prepare_inputs", wrapped)
    return True


def _wrap_v2_sample_tokens(cls) -> bool:
    """Re-publish the batch order for the draft proposal (V2 runner).

    V2 splits the step in two: ``execute_model`` runs the target forward and
    parks its state on ``self.execute_model_state``; ``sample_tokens`` -- a
    *separate* call, so outside ``execute_model``'s scope -- samples and then
    drives ``speculator.propose()``. That is V2's equivalent of V1's
    ``propose_draft_token_ids``, and the drafter can build attention metadata
    through the ATOM V4 bridge, so the snapshot has to be live here too. The
    ordering is the target step's own ``input_batch``, unchanged.
    """
    original = getattr(cls, "sample_tokens", None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        prev = getattr(_req_id_local, "req_ids", None)
        try:
            state = getattr(self, "execute_model_state", None)
            _req_id_local.req_ids = (
                list(state.input_batch.req_ids) if state is not None else []
            )
        except Exception:
            _req_id_local.req_ids = None
        try:
            return original(self, *args, **kwargs)
        finally:
            _req_id_local.req_ids = prev

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    setattr(cls, "sample_tokens", wrapped)
    return True


def _apply_v2_patch() -> bool:
    """Patch the V2 runner (``vllm.v1.worker.gpu.model_runner``), if present."""
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    except Exception as e:  # pragma: no cover - import guard
        logger.debug(
            "ATOM vLLM req_id passthrough patch: V2 GPUModelRunner unavailable "
            "(%s), skip",
            e,
        )
        return False

    # capture_model builds attention metadata for every cudagraph size straight
    # from prepare_inputs_to_capture -- no execute_model, no prepare_inputs -- and
    # for PIECEWISE it does so through the plain `build()` (for_capture=False), so
    # the builder cannot tell it is a capture from its arguments alone. Scope it.
    scoped = _wrap_v2_scope(GPUModelRunnerV2, "execute_model")
    scoped_capture = _wrap_v2_scope(GPUModelRunnerV2, "capture_model")
    scoped_dummy = _wrap_v2_scope(GPUModelRunnerV2, "_dummy_run")
    published = _wrap_v2_prepare_inputs(GPUModelRunnerV2)
    drafted = _wrap_v2_sample_tokens(GPUModelRunnerV2)

    patched = scoped or scoped_capture or scoped_dummy or published or drafted
    if patched:
        logger.info(
            "ATOM plugin: patched vLLM V2 GPUModelRunner (execute_model=%s, "
            "capture_model=%s, _dummy_run=%s, prepare_inputs=%s, sample_tokens=%s) "
            "to expose batch-ordered req_ids to ATOM metadata builders",
            scoped,
            scoped_capture,
            scoped_dummy,
            published,
            drafted,
        )
    return patched


def apply_vllm_req_id_passthrough_patch() -> bool:
    patched_v2 = _apply_v2_patch()

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # pragma: no cover - import guard
        logger.debug(
            "ATOM vLLM req_id passthrough patch: GPUModelRunner unavailable (%s), "
            "skip",
            e,
        )
        return patched_v2

    # Target attention metadata build.
    patched_target = _wrap_with_req_id_snapshot(
        GPUModelRunner, "_build_attention_metadata"
    )
    # MTP/Eagle draft proposal: the drafter builds its own attention metadata
    # (through the ATOM V4 bridge) here, outside _build_attention_metadata.
    patched_draft = _wrap_with_req_id_snapshot(
        GPUModelRunner, "propose_draft_token_ids"
    )
    # Synthetic batches. The "V1 gets the empty-list case for free" note on
    # _wrap_v2_scope holds only for dummy runs that reach the ATOM builder
    # through _build_attention_metadata. vLLM's cudagraph *memory profiling*
    # (`profile_cudagraph_memory` -> `_warmup_and_capture` -> `_dummy_run`)
    # does not: no builder metadata is attached, so the bridge takes its inline
    # fallback, which reads get_current_req_ids() with no scope open and gets
    # None -- "patch not installed" -- and fails fast on a batch that has no
    # real requests at all. Scope it to [] like the V2 runner does.
    #
    # V1 is reached whenever use_v2_model_runner is False, i.e. every non-dspark
    # spec method: `--speculative-config '{"method":"mtp"}'` lands here.
    patched_dummy = _wrap_v2_scope(GPUModelRunner, "_dummy_run")

    if patched_target or patched_draft or patched_dummy:
        logger.info(
            "ATOM plugin: patched vLLM GPUModelRunner "
            "(_build_attention_metadata=%s, propose_draft_token_ids=%s, "
            "_dummy_run=%s) to expose "
            "batch-ordered req_ids to ATOM metadata builders (removes the "
            "block-table D2H in DeepSeek-V4 slot assignment; covers the MTP draft "
            "path)",
            patched_target,
            patched_draft,
            patched_dummy,
        )
    return patched_target or patched_draft or patched_dummy or patched_v2
