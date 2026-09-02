Title: [Bug] MLA get_meta_param returns an lru_cache'd device tensor, corrupting the split indptr under CUDA graph replay

## Problem

`aiter/mla.py::get_meta_param` is decorated with `@functools.lru_cache` and returns `num_kv_splits_indptr`, a CUDA tensor it allocates. The cache entry is that tensor's only owner.

A CUDA/HIP graph that captures a cache *hit* records only the tensor's pointer, with no producing kernel. When the LRU later evicts that entry, the storage is freed back to the caching allocator and handed to a later allocation. Replay then reads whatever now lives at that address.

Eviction is routine on the decode path: `total_kv` is part of the cache key and changes almost every decode step, while entries captured into a graph are never looked up again (replay runs no host code), so they are exactly what an LRU discards first. On callers passing `ignore_total_kv=1`, `total_kv` cannot even affect the returned value, so it is a pure thrash key.

Caching the tensor looks unintended. #322 added `@lru_cache()` when the function returned two plain integers; #1233 added the `torch.arange` at the call site, outside the cache; #1390 folded the arange into the cached function and at the same time tightened the decorator to `maxsize=1`; #1445 widened that back to the default 128, which is what turns eviction of a captured entry from rare into likely.

## Reproduction

Environment: MI355X (gfx950), single GPU, torch 2.10.0+rocm7.2.4, HIP 7.2.53211, aiter main d58537b70. No model or checkpoint needed.

Add `op_tests/test_mla_split_indptr_cudagraph.py` from the linked PR and run it against unpatched main:

```
pytest op_tests/test_mla_split_indptr_cudagraph.py -q

FAILED test_get_meta_param_indptr_survives_replay
AssertionError: split indptr corrupted after replay (bs=480, splits=8):
  [3072, 3080, 3088, 3096, ...] != [0, 8, 16, 24, ...]
```

Deterministic, 3/3 runs. The test warms the cache, captures a graph over a cache hit, churns the cache past its maxsize to force eviction, replays an unrelated graph in the same pool to reclaim the storage, then replays the first graph.

Two notes on reproducing:
- Only bs=480 reproduces. bs=8 passes either way; the hazard is allocation-size dependent, which is likely why it has survived in tree.
- The bs=480 pass only fails once bs=8 has run and been released. The test does both passes in order with a `gc.collect()` between them, so it is self-contained, but a hand-written snippet that skips the bs=8 pass will not reproduce.

The corrupted buffer is not a uniform shift: it holds a fragment of one arange followed by a fragment of another (`... 3832, 3840, 776, 784, ...`), i.e. a torn read across recycled storage rather than a merely stale buffer.

## Impact

`_fwd_kernel_stage2_asm` derives `num_valid_kv_splits` from these values and indexes `Mid_O` accordingly, so a corrupted indptr reads out of bounds.

What led us here, as an open lead rather than a closed root cause: we hit a `Memory access fault by GPU node-N ... Reason: Unknown` on a TP8 graph-capturing MLA decode workload, and the faulting wavefront's address decoded to byte 0x7800000 = 480*8*16*512, one element past the end of `Mid_O` at exactly the bs=480 / splits=8 / nhead=16 geometry above.

To be precise about what is and is not demonstrated: the test reproduces the indptr corruption, not the memory fault. The step from corruption to fault is inferred from that address decode. The same fault signature still reproduces on one of our TP1 configurations with the fix applied, so either this fix is incomplete or there is a second producer. And we have not yet shown that a production run evicts a captured entry at all, so we are not claiming this explains the crash. The caching bug stands on its own.

## Suggested fix

Cache only the split-count heuristic and rebuild the indptr per call. `torch.arange` on CUDA is a fill kernel, so building it inside the captured region records the kernel into the graph and re-initialises the buffer on every replay. PR linked below.
