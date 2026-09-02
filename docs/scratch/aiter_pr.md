## Motivation

`aiter/mla.py::get_meta_param` is `@functools.lru_cache`'d and returns a CUDA tensor it allocates, `num_kv_splits_indptr`. The cache entry is that tensor's only owner, which makes it unsafe to capture into a CUDA/HIP graph: a capture that hits the cache records only a pointer with no producing kernel, so once the LRU evicts the entry the storage is recycled and replay reads garbage.

Eviction is routine on the decode path. `total_kv` is part of the cache key and changes almost every decode step, while entries captured into a graph are never looked up again (replay runs no host code), so captured entries are exactly what an LRU discards first.

This PR narrows the cache back to the scope it was introduced with, rather than removing an intended optimization. #322 added `@lru_cache()` when the function returned `(num_kv_splits, mgc)`, two plain integers. #1233 added the `torch.arange`, at the call site and outside the cache. #1390 folded the arange into the cached function and simultaneously tightened the decorator to `maxsize=1`. #1445 then widened `maxsize=1` back to the default 128, which is what makes eviction of a captured entry likely rather than rare.

## Technical Details

Split the function in two:
- `_get_num_kv_splits()` keeps `@functools.lru_cache`. The split-count heuristic is pure host-side arithmetic and is what the cache was originally added for.
- `get_meta_param()` is now uncached and rebuilds the indptr with `torch.arange` on every call.

`torch.arange` on CUDA is a fill kernel, so built inside a captured region it is recorded into the graph and re-runs on every replay. The buffer is re-initialised regardless of what the caching allocator did with the storage in between, which makes this correct by construction rather than by allocator luck.

No measurable cost on the decode path, because the cache was not hitting there. Over 2000 decode steps at bs=480 with `total_kv` growing one token per sequence per step, the old cache took 0 hits and 2000 misses, i.e. it was already rebuilding the arange every step: 6.65 us/call before, 6.69 us/call after. The cache only pays on a repeating key, 0.15 us/call against 4.71 us/call, and a repeating key is exactly the graph-capture case, where the arange is recorded once and costs nothing at replay. The one case where the cache wins is the case where giving it up is free, and it is also the case that corrupts memory.

Alternatives considered and rejected: raising `maxsize` only delays eviction, since captured keys are never looked up again and any bound eventually discards them; holding a module-level reference leaks one tensor per distinct key, unbounded.

Rebuilding also drops a latent multi-device bug. The cached tensor was allocated on the then-current device with no device in the cache key, so a second device could be handed the first one's tensor.

No behavioural change to the returned values, and the added `torch.arange` is on the non-persistent path only.

Follow-up, deliberately not in this PR to keep it minimal: `_get_num_kv_splits` still takes `total_kv` in its key even when `ignore_total_kv=1`, where it provably cannot affect the result, so the surviving cache still thrashes at close to a 0% hit rate on that path.

## Test Plan

Adds `op_tests/test_mla_split_indptr_cudagraph.py`. It warms the cache, captures a graph over a cache hit, churns the cache past its maxsize to force eviction, replays an unrelated graph in the same pool to reclaim the storage, then replays the first graph and compares the indptr against a freshly built one.

The test runs both as a plain script, which is how `.github/scripts/aiter_test.sh` invokes everything in `op_tests/`, and under pytest. Run against unpatched main and against this branch:

```
git checkout main -- aiter/mla.py    # keep the new test, drop the fix
python3 op_tests/test_mla_split_indptr_cudagraph.py   # exits 1
git checkout HEAD -- aiter/mla.py    # restore the fix
python3 op_tests/test_mla_split_indptr_cudagraph.py   # exits 0
```

Environment: MI355X (gfx950), single GPU, torch 2.10.0+rocm7.2.4, HIP 7.2.53211. A single GPU is sufficient; no model or checkpoint is needed.

The test does the bs=8 and bs=480 passes in one test function, in that order, releasing the small pass before the large one. Only bs=480 reproduces, and only once bs=8's churn and freed graph pool have left blocks of the right size class in the allocator. Splitting them into separate tests makes the failure depend on test ordering, which is why they are one test.

## Test Result

Unpatched main, 3/3 runs, exit 1:

```
[PASS] split indptr survives graph replay (bs=8)
AssertionError: split indptr corrupted after replay (bs=480, splits=8):
  [3072, 3080, 3088, 3096, ...] != [0, 8, 16, 24, ...]
```

With the fix, 3/3 runs, exit 0:

```
[PASS] split indptr survives graph replay (bs=8)
[PASS] split indptr survives graph replay (bs=480)
```

`black --check` and `ruff check` clean on both changed files.

What led us here, offered as an open lead and not as a closed root cause: we hit a `Memory access fault by GPU node-N ... Reason: Unknown` on a TP8 graph-capturing MLA decode workload, and the faulting wavefront's address decoded to byte 0x7800000 = 480*8*16*512, one element past the end of `Mid_O` at the same bs=480 / splits=8 / nhead=16 geometry this test uses. That is suggestive, not conclusive. The test demonstrates the indptr corruption, not the fault; the same fault signature still reproduces on a TP1 configuration with this fix applied; and we have not shown that a production run actually evicts a captured entry. This PR is offered on the correctness of the caching bug alone, independent of that crash.

## Submission Checklist

- [x] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.
