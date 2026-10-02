# Handoff — DSpark on vllm_atom for DeepSeek-V4-Pro-0813

2026-10-02. **No performance improvement was delivered.** The gap is fully
diagnosed and priced; nothing shipped is faster. Read §3 before trusting any
estimate of mine.

Detail lives in `kb/` (indexed by `kb/README.md`); numbers live in
`results_*/*/report.md`. This file is the summary.

---

## 1. What enables DSpark on vllm_atom today

### The architecture, and the one thing to understand about it

In plugin mode the seam runs **through the middle of speculative decoding**:

| component | owner |
|---|---|
| Scheduler, KV cache manager | **vLLM** |
| Target model (61 layers) + its attention | **ATOM** (V4 proxy bridge) |
| DSpark **speculator** (drafting loop, rejection) | **vLLM** — `DSparkSpeculator` ← `DFlashSpeculator` |
| DSpark **draft model** | **vLLM** — `vllm/models/deepseek_v4/amd/dspark.py` |
| aux hidden states | **ATOM** produces → vLLM consumes |

ATOM computes the hidden states; vLLM's drafter turns them into guesses. Native
runs *both* halves itself (`atom/spec_decode/dspark_proposer.py`). The two arms
therefore **do not share a drafter** — which matters in §3.

### Commit `be473f7a3` — pre-existing, this is what made DSpark run

Not mine. Its own message is thorough; the load-bearing parts:

- `DeepseekV4Model.forward` optionally returns per-layer aux hidden states,
  averaged over the mHC dimension. The branch sits **inside** the
  `@support_torch_compile` region, so `aux_hidden_state_layers` is baked into
  the graph — set at load time, never mutated after.
- `dspark_draft_kv_patch` registers a second KV group for the draft's
  sliding-window MLA (block 64) beside the V4 proxy's block 128.
- `deepseek_v4_prefix_patch` rolls every prefix-cache hit back by
  `max(win_with_spec, index_topk)` = **1024 tokens** so the SWA ring is
  repopulated. The `index_topk` term is flagged in-file as a workaround for a
  sparse-indexer defect.
- `_demote_piecewise_cudagraph` silently turns the requested
  `FULL_AND_PIECEWISE` into `FULL_DECODE_ONLY` on this path. Confirmed firing.

**One env var is what unblocked DSpark**: `VLLM_ROCM_USE_AITER=1`. Without it
the MXFP4 MoE oracle rejects aiter, falls through to Triton, and the draft dies
on `No module named 'triton_kernels.matmul_ogs'` — a traceback that reads like
a dtype bug and is not. `--moe-backend aiter` is worth pairing with it: that
branch *raises* instead of silently falling through.

### My commits on `ds_v4_atom_vllm` (pushed, all diagnostic or gated)

| commit | what | runtime effect |
|---|---|---|
| `5f00ac68a` | startup note naming the missing DP attention; `ATOM_DSPARK_CHECK_MARKOV_BOUNDS` counter | none (flag off) |
| `27f3e9234` | that counter was `hipErrorStreamCaptureUnsupported` under cudagraph capture — made it capture-safe | none (flag off) |
| `02e003dcf` | `ATOM_LOG_SPEC_SEAM` — logs the ATOM→vLLM handover per forward | none (flag off) |
| `2ff2fd961` | FP4 indexer carve groundwork, **gated off** | none (gate off) |

**Nothing here changes behaviour with flags unset.** `2ff2fd961` is gated by
`_V4_INDEX_FP4_SUPPORTED = False` and must stay that way until §4.2 lands —
flipping it early gives silently wrong top-k, not a crash.

---

## 2. The gap, and what it decomposes into

Measured at ISL 115k / OSL 1k, MI355X TP8, same checkpoint, client and seed.

| conc 128 | native | plugin | |
|---|---:|---:|---|
| in/s/gpu | 32302 | 13940 | −57 % |
| decode step (`ITL p50 × tok/step`) | 132.6 ms | 270.5 ms | 2.04× |

The code paths were read end to end: `deepseek_v4.py` and
`deepseek_v4_attn.py` contain **no plugin-mode branches at all**. Decode forks
on exactly two config values, and ablating each accounts for the whole gap:

| arm | index | attention | decode step |
|---|---|---|---:|
| plugin | fp8 | TP | **270.5 ms** |
| native | fp8 | DP | **155.9 ms** |
| native | fp4 | DP | **132.6 ms** |

| cause | worth |
|---|---:|
| **DP attention** | **1.735×** |
| **FP4 indexer** | **1.176×** |
| product | 2.040× |
| measured | **2.040×** — zero residual |

**Why native is fast, mechanically.** MLA keeps one latent per token shared by
all heads, so TP does *not* shard the cache read — under plain TP every rank
reads every request's full latent and index plane each step. Native's
`--enable-dp-attention` flips the split (`runtime_tp_size=1`,
`runtime_dp_size=8`): all heads, 1/8 of the *requests*, so each rank reads 1/8
as much. And the cache it reads is FP4, which is a **different kernel**
(`pa_mqa_logits_fp4`, flydsl persistent-grid varctx) not merely fewer bytes.

Neither is reachable from the plugin by configuration: `enable_dp_attention` is
hardcoded `False` (`atom/plugin/config.py:419`) because the feature lives in
ATOM's *engine*, which plugin mode replaces with vLLM's; FP4 is blocked on
block geometry (§4.2).

**Caveat on the mechanism**: the multipliers are ablations and solid. My
*bandwidth explanation* is not confirmed — a first-principles estimate of the
index-plane read gives ~0.02 ms/request against a measured ~1.8. Likely
scattered 132-byte paged gathers running far below peak, but unproven. A
profiler trace would settle it and was never captured.

---

## 3. What I tried that did not work

Listed so nobody repeats them.

### Dead ends, each measured

| attempt | result |
|---|---|
| **vLLM's own DP+EP** (`--data-parallel-size 8 --enable-expert-parallel`) as a DP-attention substitute | **2.0× WORSE** at conc 64. 8 independent engines fragment the prefix cache (38.9 % hit at conc 16 vs 80.1 %) and divide the per-rank batch. Do not retry. |
| **`--max-num-seqs` tuning** | Decode throughput is *flat*: `batch/ITL` = 1142 tok/s at batch 32, 1126 at 128. Step time is linear in batch, so decode is saturated; the apparent +8.5 % was prefill stealing step budget, at 60× worse TTFT. |
| **`num_speculative_tokens` 7→5** | Positions 5–6 contribute 0.084+0.054 of the acceptance chain → tok/step 2.378→2.24. Saves compute, not the dominant cache read. A wash. |
| **Proxy block size 128→256** (to match native, required by FP4) | **−12 %** in/s/gpu on its own. Note the split: decode step *improved* (270.5→254.4 ms), acceptance *fell* (2.379→2.006). Must land together with FP4. |
| **Prefix caching off** | Acceptance 2.39–2.60 at batch ~57 vs 3.008 at conc 64 with it. Off is *worse*. Retires the `deepseek_v4_prefix_patch` KNOWN-ISSUE theory. |

### The ATOM-owned drafter — abandoned, branch `ds_v4_atom_vllm_dspark_draft`

Premise: the plugin's acceptance collapses with batch (3.31 tok/step at conc 16
→ **2.38** at 128) while all three native arms hold ~3.3. Since the two arms run
different drafters, replace vLLM's with ATOM's.

Four causes were eliminated first, all by measurement: out-of-range draft ids
(0 clamps in 281,243 drafts), offered load / queue depth (`--max-num-seqs 32`
under concurrency 128 restored 2.379→3.335), cudagraphs (2.376 vs 2.379 with
them off), and slot-indexed bookkeeping (rejection flat across batch slots).

**It does not work.** Four commits, ten hardware bring-up cycles. It serves and
emits correct text, but drafts at ~1.10 tok/step — *worse than no speculative
decoding*. Seven integration defects were found and fixed (EAGLE3 target/draft
branch, `layer_offset`, MTP forward contract, weight sharing via the wrong load
hook, `ParallelHead` vs `get_logits`, internal layer names vs the proxy, and
97/97 draft weights silently discarded by `spec_decode=False`).

**Known remaining defect**: `sample_from_anchor` defaults True for this
checkpoint, so `num_query_per_req = num_speculative_steps = **7**, not 8`. The
branch assumes 8 and reads anchors from the wrong column. Untested.

**Do not merge this branch.** The registry override makes the plugin resolve
`DSparkDraftModel` to this drafter, so running it is *worse* than shipping
today. ATOM is installed editable — checking the branch out changes what every
run on the node uses.

### Method errors worth not repeating

- A logprob invariance probe "FAILED" alone-vs-crowd at 100k and nearly became
  a root cause. The control — same prompt, alone, twice — then showed tokens
  *also* differ run to run. **Run the single-stream determinism control first.**
- `accept_vs_context.py` appeared to show acceptance flat across batch
  8/32/128; with prefix caching off, KV capacity capped the server at **59**
  running requests, so 128 never ran. **Check the server's `Running:` count,
  not the client's concurrency.**
- I priced FP4 from *bytes* (~8 % of cache traffic) and deprioritised it for a
  day. It is 17.6 % of decode step time. **Where two paths differ by kernel,
  ablate; do not model from bytes.**

### Separately worth a ticket

**Greedy decoding is not reproducible at long context on either arm.** Same
prompt, batch 1, different tokens across runs from ~15k tokens up (native is
worse at 12k: 5/5 distinct). Not the chunked-prefill boundary — 15k is a single
16384 chunk and already diverges. This undermines any accuracy number taken at
long context, GSM8K at 115k included.

---

## 4. Next steps, in priority order

### 4.1 DP attention — 1.735×, the dominant term

**Do not port ATOM's `--enable-dp-attention`.** It shards by *request*, which
needs stable request→rank ownership vLLM's KV manager has no notion of inside a
TP group, and it re-creates the prefix-cache fragmentation that made vLLM's own
DP 2× worse.

**Route: extend vLLM's DCP to the V4 proxy.** DCP shards by *context position*,
so the prefix cache stays global. Most of it already exists on this path:
`atom/plugin/vllm/attention/layer_mla.py` is a complete DCP implementation for
generic MLA (59 references: `MLADCPManager`, LSE combine, `get_dcp_group()`
all-gather, `cp_kv_cache_interleave_size`), and both adjacent blockers are
solved in-tree — `dspark_dcp_patch.py` (DSpark under DCP) and
`rocm_dcp_full_graph_patch.py` (cudagraphs under DCP).
`grep dcp deepseek_v4_bridge.py` returns **0**. That gap is the job.

V4-specific work: make the proxy arena carve DCP-aware across planes that
compress at different ratios (30 CSA at 4, 31 HCA at 128); return LSE from the
V4 decode; replicate rather than shard the 128-entry SWA ring; and — **the one
real unknown** — a cross-rank top-k for the sparse indexer, since `index_topk`
is 1024 over a context each rank holds 1/8 of. Native has a pattern to follow
(`dcp_config.indexer_dcp_only`). **Prototype that reduction first**; it is where
the route either works or doesn't.

Stage it DCP=2 prefill (accuracy-gated, GSM8K baseline 0.9583) → DCP=2 decode →
DCP=8 sweep.

### 4.2 FP4 indexer — 1.176×, bounded

Carve and bind are **done and committed** in `2ff2fd961`: two-plane budget,
`kv_scale` bound alongside the keys, `_v4_index_fp4` as the single authority the
sizing/carve/bind all read. Shape verified `(num_blocks, 1, 4, 64, 16)`, which
is what the scorer's `kv_cache.size(3) == 64` wants.

**Missing**: the decode *schedule*. `_score_topk_decode_fp4_flydsl` reads
`fp4_local_starts` / `fp4_local_ends` / `fp4_cta_info` / `fp4_n_ctas`, which
native builds in `deepseek_v4_attn.py:2228` via `compute_prefill_schedule`. The
bridge builds none of them. Its inputs (`batch_id_per_q_token`,
`csa_n_committed_per_token`) **already exist** on this path, so the port is two
persistent buffers plus that call. Then flip `_V4_INDEX_FP4_SUPPORTED`.

Also needs proxy block size 256 (FP4 requires exactly 64 index rows/block).
That is a −12 % regression alone, so understand its acceptance cost first.

### 4.3 The acceptance collapse — 1.4× at conc 128, still unexplained

Plugin-only; all three native arms hold ~3.3 at every concurrency. Four
structural causes eliminated (§3). What remains is batch-dependent numerics in
vLLM's drafter. Replacing the drafter is the workaround and it is unfinished;
root-causing vLLM's speculator is the alternative and was never attempted.

### 4.4 The measurement I never took

A **profiler trace of a plugin decode step at conc 128**. It would confirm or
kill the bandwidth explanation in §2 and show whether the marginal cost is the
indexer, the attention, or prefill interleaving. Use `--profiler-config
'{"profiler":"torch","torch_profiler_dir":...,"capture_torch_profiler":true,
"delay_iterations":N,"active_iterations":5}'` — `delay_iterations` skips
prefill, which is what makes a short window land in steady-state decode.
`trace_driver.py` (shared 115k prefix so 128 requests reach decode in tens of
iterations) is in this repo for that.

---

## 5. Repo map

| | |
|---|---|
| `kb/README.md` | every finding, one file each |
| `kb/p4-the-decode-gap-decomposes-exactly-2026-10-02.md` | **the headline result** |
| `results_*/*/report.md` | all measurements; `arm.json` records the exact sha + `/proc/<pid>/cmdline` |
| `patches/` | vLLM-side probes (off unless their env var is set) |
| `batch_invariance_probe.py`, `logprob_invariance_probe.py`, `accept_vs_context.py`, `trace_driver.py` | diagnostics built this session |
| `serve_*.sh` | one per arm; headers state what varies and why |

ATOM and aiter are `pip install -e`'d: **an uncommitted edit is already in the
next measurement**, and the checked-out branch is what every run uses. Always
`./stop.sh` before relaunching — it verifies VRAM actually released.
