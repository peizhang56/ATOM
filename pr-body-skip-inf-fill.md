## perf(plugin/sparse-mla): skip the indexer logits `-inf` prefill

The vLLM plugin's sparse-MLA prefill path let `fp8_mqa_logits` default to
`clean_logits=True`, which fills the whole `[rows, total_committed]` fp32 logits
plane with `-inf` before the kernel runs. The native paths already pass
`clean_logits=False` (`deepseek_v2.py:1863`, `deepseek_v4.py:1869`); this
applies the same decision to the plugin call site, which was simply missed.

One keyword, plus the comment that says why it is safe.

### Why the fill is dead work

Three links, each checkable in one file:

1. **The logits kernel writes every column inside a row's window.**
   `aiter/ops/triton/attention/fp8_mqa_logits.py` — `clean_logits=False` swaps
   `torch.full(-inf)` for `torch.empty` and, on the gfx950 gluon path, sets
   `RELAXED_STORE=1`. That relaxes only the **per-row** store mask; the union
   bound that keeps a store inside its own row is untouched, and the `HAS_KV_SPLIT`
   `row_ends` clamp still applies. Nothing moves out of bounds.
2. **The only consumer never reads outside that window.**
   `top_k_per_row_prefill` is handed the *same* `row_ks`/`row_ke` that went to the
   logits kernel; `topKPerRowPrefill` offsets every access by `rowStart` and bounds
   it by `rowEnd` (`vllm/csrc/libtorch_stable/sampler.cu:556`), on both the
   vectorized `float4` path and the scalar one.
3. **The windows are inside the buffer by construction.**
   `kv_spans_from_batches` (`atom/plugin/vllm/attention/metadata.py:310`) builds
   `end = kv_start[b] + (L_b - count_b + i + 1)`, so `row_ke ≤ total_seq_lens ≤`
   the logits width.

So no reader can observe the `-inf`. It is also costly out of proportion to its
purpose: the fill covers the whole rectangle while the kernel writes only the
block-diagonal band — ~3× the area at the shapes measured, and it is a pure HBM
pass at full bandwidth.

### Measurements

From a torch-profiler trace of GLM-5.3 MXFP4-AttnFP8 on 8× gfx950 (MI350X), TP=8
+ expert-parallel, MTP depth 3, ISL 120k / OSL 917 / `--cache 90` / concurrency 24:

| | fill | `fp8_mqa_logits` clean → relaxed | saved |
|---|---|---|---|
| `[1408, 350180]` (trace shape) | **411 µs** (4.5 TB/s, bandwidth-bound) | 2424.8 → 1989.2 µs | **436 µs, 18.0 %** |
| `[8192, 8192]` (longest nightly-CI shape) | 50 µs | 267 → 207 µs | 60 µs, 22.5 % |

`torch.empty` for the same allocation: 3.0 µs. Across the profiled step the fill
was 420–467 µs/call × ~94.5 calls, **4.97 % of prefill GPU time** (count × median
attribution, so profiler stalls do not inflate it).

Equivalence, `clean_logits=True` vs `False` on identical inputs, three shapes:

- In-window logits **bit-identical**: 123,855,552 / 15,696,128 / 266,339,328
  values compared, **0** mismatching, max |diff| **0.0**.
- Top-k **value multiset** identical through the real consumer: **0/1408,
  0/512, 0/2048** rows differ.
- Poison control (every out-of-window column set to a score that beats any real
  one): 0 rows selected poison, no index past its window.

### Do not verify this by comparing top-k indices

`topKPerRowPrefill` is bin-based and the order of equal-bin elements is
race-dependent: two runs of the **same** config, with no code change at all,
disagree on 2,578,223 of 2,883,584 index slots. Compare in-window logit values,
or the multiset of selected values. (This reproduces the number quoted in the
commit message, which is a useful cross-check on the harness.)

### Tests

Two added, because nothing existing covers this call site — the repo's
`tests/test_indexer_topk_row_window.py` and
`tests/test_indexer_logits_alignment_tail.py` pin the row-window invariant for
**aiter's** top-k, which the *native* paths use. The vLLM plugin path calls a
different kernel, in a different repo, on a pinned commit a version bump can move
underneath us.

- `tests/test_indexer_clean_logits_call_sites.py` — CPU, 5 tests, 0.6 s. An `ast`
  scan of every `fp8_mqa_logits` call site in `atom/`: the vLLM plugin and native
  paths must pass the literal `False`, and a **new** call site that inherits the
  default fails the build. Reaching these at runtime needs a GPU, a loaded model
  and a chunked-prefill batch, so the contract is pinned in the source. Verified
  by mutation: removing the keyword fails two of the tests.
- `tests/test_indexer_vllm_topk_row_window.py` — GPU-gated, 9 tests, 7.2 s. Pins
  link 2 directly (poison every out-of-window column, including the alignment
  padding, and assert nothing selects it) with a control proving the poison is
  reachable, plus the bit-identity and top-k-value checks above. Covers both
  top-k instantiations: `rows < 12288` (insertion) and above (radix), and both
  the KV-split and non-split logits configurations.

### Known gaps, deliberately left

- **CI does not exercise the regime this targets.** The nightly matrix tops out
  at ISL 8192, where the win is 60 µs/call; the shape that motivated the change
  is ISL 120k. The new tests pin correctness, not the speedup — a perf regression
  here would show up only in a manual benchmark.
- **Three call sites are still unswept**, against CLAUDE.md's fix-then-sweep:
  `atom/model_ops/glm5_next/indexer.py:275`,
  `atom/plugin/rtpllm/attention_backend/rtp_sparse_mla_backend.py:1700`,
  `atom/plugin/sglang/attention_backend/sparse_mla_indexer.py:1432`. Each looks
  redundant for the same reason, but that has not been measured end to end on
  those backends, so they are listed in `KNOWN_UNSWEPT` rather than assumed —
  sweeping one means deleting its line.
- Incidental, not fixed here: vLLM's `topKPerRowJob` writes out of bounds when
  `rowEnd < rowStart` (the short-row path iterates from a negative offset).
  Production metadata cannot produce that; the new test helper asserts the
  invariant so a future test cannot either.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
