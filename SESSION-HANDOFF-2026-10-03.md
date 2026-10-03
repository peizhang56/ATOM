# Session hand-off — 2026-10-03

Supersedes `SESSION-HANDOFF.md` (2026-10-01), which is still correct for its
own measurements and is cited by `CLAUDE.md` §2/§4. Read this one first.

---

## 0. State

| | |
|---|---|
| code | `ds_v4_atom_vllm_dspark_draft_t7fix` @ **`fa4134ef8`**, clean, pushed |
| worklog | `ds_v4_dspark_worklog` @ **`98696a3d`**, clean, pushed |
| `/app/vllm` | `281cfd5a09` (runtime copy is `/opt/venv/.../vllm`) |
| aiter | `/app/aiter-test` @ `503443ff0` |

Four commits added this session, **all gated OFF by default** — the default
path is byte-identical to `a460a5645`:

```
fa4134ef8  DP attention step 2b -- shard the KV read by request
ec36440a8  DP attention step 2a -- request->rank assignment, mask, gather
9aa7f0385  DP attention step 1 -- replicated attention weights
442555da5  (wip, superseded by 9aa7f0385)
```

Enable with `ATOM_VLLM_DP_ATTENTION=1`. **Leave it off** — §4 explains why.

---

## 1. The goal, stated correctly

I got this wrong repeatedly before it was corrected, so it is first:

> ATOM native running DS-V4-Pro-0813 with DSpark is the performance reference.
> Match it **while serving through vLLM with ATOM as the plugin**, with ATOM
> owning **both** target inference and DSpark. The previous attempt
> (`ds_v4_atom_vllm`) left DSpark on the vLLM side and could not close the gap.

Two consequences I kept missing:

- **Shipping the vLLM-draft arm is not an option**, even though it is faster
  today. The ownership split is the thing being fixed.
- **LMCache is why the arm must be the plugin at all.** ATOM's offload lives at
  `atom/kv_transfer/offload/hybrid/dsv4/` and `atom/plugin/vllm/kv_transfer/`.

### The scoring instrument

`client_dsv4_pro_0813.sh` — ISL 115k / OSL 1k / `--cache 90`, p90. Targets are
in **`/home/pzhang12/deepseek/requirements.txt`** (not in this tree):

| target | required | `_t7fix` best | native best |
|---|---:|---:|---|
| in tok/s/gpu | > 17200 | 15641 (c128) ✗ | 32302 ✓ |
| out tok/s/gpu | > 150 | 135.9 ✗ | 280.9 ✓ |
| P90 TTFT | < 5 s | 7.9 s ✗ | 16.8 s ✗ |
| P90 ITL | ≤ 20 ms | 28.1 ms ✗ | 20.2 ms ✗ |
| cache hit | 90% | 89.0% ≈ | 89.9% ≈ |

No arm passes all five. Native misses both latency SLAs.

### Things outside this tree that I wasted a day not knowing about

All under **`/home/pzhang12/deepseek/`**:

- `requirements.txt` — the targets above
- `kb/` — eleven interpretation notes, indexed in `kb/README.md`. **Read
  `kb/e1`, `kb/e2`, `kb/e3` before touching DP attention.**
- `results_mi355x_*` — the completed 115k sweeps for native, plugin+vLLM-draft,
  and plugin-spec-off
- `.claude/skills/benchmark-report/` — generates each sweep's `report.md`.
  Copied into `deepseek2/.claude/skills/` this session so it is discoverable.

Convention: generated `report.md` holds **data and commands only**; the reading
of those numbers belongs in `kb/<topic>-<node>-<date>.md`. `RESULTS.md` does not
follow that convention and probably should be split.

---

## 2. The headline measurement: `_t7fix` at the goal regime

First ever 115k sweep of the ATOM-owned draft
(`results_mi355x_dsv4pro_vllm_atom_dspark_t7fix_115k/sweep_20261002_211427`).
All figures from each sweep's generated `report.md` — **`report.txt` computes
`in/s/gpu` differently, do not mix them.**

| conc | **ATOM draft** | acc.len | vLLM draft | tok/step | spec OFF | native | tok/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 7563 | **2.129** | 9038 | 3.309 | 7674 | 4463 | 3.307 |
| 24 | 10479 | **1.967** | 12860 | 3.124 | 12169 | 10152 | 3.225 |
| 32 | 11865 | **1.971** | 14141 | 3.207 | 13897 | 15328 | 3.227 |
| 64 | 13570 | **1.825** | 15717 | 3.008 | 17767 | 23771 | 3.259 |
| 128 | 15641 | **2.742** | 13940 | 2.379 | **20958** | **32302** | 3.282 |

**The premise's shape is confirmed; its level is not.** ATOM's draft does not
decay with batch (2.13 → 1.83 → 2.74 is flat) where vLLM's falls 3.309 → 2.379,
and at c128 it beats vLLM's draft on acceptance *and* throughput (+12%). That is
the behaviour the project was predicated on, now measured at the goal regime.

But it sits at ~2.0 against native's 3.28, so it loses at c16–c64.

**And at c64/c128 speculation is a net LOSS** — spec-off (20958 at c128) beats
both spec arms. Today DSpark in the plugin is worth negative throughput where
the benchmark is heaviest. (Caveat: the spec-off sweep is a different ATOM/aiter
sha; re-run on one build before quoting externally.)

**The deficit is regime-independent** — 1.97–2.13 at 115k, 1.98–2.50 at
~80-token prompts, same per-position shape. So it is one systematic defect and
can be debugged at 20-token prompts with 3-minute boots instead of hour-long
sweeps. That is measured, not assumed.

---

## 3. Open item A — the draft's 1.36-token deficit

### The reference is 3.50, not 4.45

`CLAUDE.md` §6 compares our draft against native's 4.45. That is **not
config-matched**: native runs `--index_cache_dtype fp4 --enable-dp-attention
--enable-tbo`, the plugin runs none of them. Running native with the plugin's
config (`serve_native_plugincfg.sh`, `serve_native_fp8idx.sh`):

| native config | indexer | DP+TBO | acc.len (probe, conc 32) |
|---|---|---|---:|
| full | fp4 | on | 4.45 |
| — | fp8 | on | 3.83 |
| — | fp8 | off | **3.50** |
| plugin + ATOM draft | fp8 | off | **2.14** |

So the integration deficit is **1.36 tokens (39%)**, not 2.3 (52%).

**Caveat, important:** `kb/e2` measured acceptance at 115k with DPA+TBO removed
— 3.282 full vs 3.306 without — i.e. unchanged. So the −0.33 for DP+TBO above
is a short-context artifact. The **fp4 indexer's −0.62 is unverified at 115k**
and should be checked; if real it means `CLAUDE.md` §4's "FP4 indexer is not in
scope for correctness" is wrong on the facts, because indexer precision changes
which tokens sparse attention sees and therefore the hidden states the draft
consumes.

### Localized to the drafting loop

The window ablation is the key control. Skipping the context write on **both**
arms:

| arm | with window | without | window delivers |
|---|---:|---:|---:|
| native | 4.45 | **1.06** | +3.39 |
| plugin, ATOM draft | 2.14 | 1.05 | +1.09 |

**Identical floors.** With the window gone both arms produce the same number, so
weights, LM head, Markov chain, anchor token, block width and the backbone
stages are all equivalent. The whole gap is what the window delivers.

### Eliminated — do not re-investigate

Measured:

| candidate | evidence |
|---|---|
| weights, head, Markov, anchor, block width, backbone | identical no-context floors (1.06 / 1.05) |
| batch / padding / slot indexing | **concurrency 1 still gives 2.45** |
| cudagraphs | eager 1.98–2.50 ≈ captured 2.07–2.44 |
| "window is dead" | ablation → 1.05; it is read and carries most of the quality |
| rejected rows written at position 0 | real defect, unit-tested, gating them changed nothing (2.09 → 2.14) |
| anchor position convention | `−1` → 2.11 |
| rejected suffix at true positions (native's behaviour) | **1.55, worse** |
| true positions + anchor `−1` (native's exact config) | 2.02 |
| `res_preshuffle` | `mhc_res_shuffle_enabled(1,"gfx950")` = False, inert |
| vestigial draft KV group | one group only, patch dormant |

The positional axis is **exhausted**: five configurations spanning write
positions and read anchor all land between 1.55 and 2.14.

By reading / instrumentation:

- `aux_concat` and `main_x`, **config-matched**, are at or below the same-arm
  noise floor (7.44% vs 7.37%, 12.91% vs 13.11%). Every input the draft
  consumes is equivalent. The earlier 32% aux difference was config, not wiring.
- window **addressing** round-trip is exact (integers, no noise floor)
- positions and coverage per step are identical on both arms
- `sample_indices` is the identity under `SAMPLE_FROM_ANCHOR`
- `swa_plane` **is** `unified_kv` in both arms; aux layers/order/reduction match
- `DeepseekV4AttentionVllm` overrides only `forward_impl`/`_sparse_attention`,
  which `dspark_attention` never calls — the class swap is inert for the draft

### What remains

**The drafting loop itself** — vLLM's `DSparkSpeculator` driving ATOM's model
instead of ATOM's `DSparkProposer`. That is now elimination-backed, not a
preference, and it contradicts `CLAUDE.md` §2.3, which rules out porting
`DSparkProposer` on the strength of the Kimi-K3 precedent. The K3 precedent may
simply not transfer: K3's draft is paged, V4's is a private per-request ring.

**Do not** try to settle this with hidden-state A/B on a live server. Two native
runs of one prompt land 14% apart on `main_x` while emitting identical text
(MoE top-k flipping). Nothing finer than ~15% is resolvable there; it needs an
offline single-process diff.

---

## 4. Open item B — DP attention, implemented and regressive

`kb/e3`: DP attention is the **entire** throughput gap (1.95× at c64, 2.31× at
c128). `kb/e1`: vLLM's own `--data-parallel-size 8 --enable-expert-parallel` is
**2.0–4.7× WORSE** — 8 independent engines, prefix cache fragmenting to 38.9%.
**Do not re-run that.** `kb/e3` also measures **TBO as a net 4–8% loss** without
`GPU_MAX_HW_QUEUES=5` + `ATOM_NUMA_BIND=1`; do not port it.

Native's shape cannot port: `--enable-dp-attention` sets `runtime_tp_size =
tp//dp = 1`, running the whole model tp=1/dp=8 with MoE flattened into EP, and
what makes it fast is ATOM's engine keeping **one** scheduler / KV pool / prefix
cache across those ranks.

### What I built, and the measured result

Keep vLLM's single-scheduler TP engine; replicate only attention; shard
attention work by request; all-gather; MoE untouched.

Measured at 115k
(`results_mi355x_dsv4pro_vllm_atom_dpattn_115k/sweep_20261003_054526`):

| conc | DP attn ON | baseline | |
|---:|---:|---:|---|
| 16 | 4705 | 7563 | **−38%** |
| 24 | 6864 | 10479 | −35% |
| 32 | 7394 | 11865 | −38% |

**Cause, and it is a flaw in my design:** I sharded the KV *read* but not the
*compute*. With replicated weights every rank runs the Q/K/V/O projections for
all heads on all tokens — 8× the per-rank projection FLOPs against TP's
`n_heads/8` — and `_dp_combine` discards seven eighths afterwards. ITL p50 rose
34 → 54 ms at c32.

### Step 2c, if this is resumed

Masking after the fact cannot work. A rank must not **compute** rows it does not
own:

1. Build owned-subset metadata in the bridge (same builders, filtered token
   list) rather than masking a full-width build — `batch_id_per_q_token`,
   `kv_indptr_*`, `qo_indptr`, `block_tables_per_token`, compress plans.
2. Slice `x`/`positions` to owned rows at the attention block's input.
3. Scatter back and all-reduce (2a already does this).
4. **Keep KV writes at full width.** A rank holding only its own requests'
   blocks would miss a prefix-cache hit against a request that lived elsewhere —
   that is exactly how `kb/e1` fragmented the cache. Writes are one row per
   token; the ~662 MB/request read is the whole cost.

Costs +16.5 GiB/rank of weights (2.36 → 18.87), i.e. ~11% of the KV pool —
measured as 3,256,960 vs 3,644,928 tokens. Worth it for 1.95–2.31×.

### Two findings worth keeping regardless

- **`RowParallelLinear.__init__` accepted `**kwargs` and dropped them**, so
  `override_tp_size` silently did nothing there while `ColumnParallelLinear`
  honoured it. A latent ATOM bug — nothing had passed them before, since the one
  existing user (DCP, `attention_mla.py:414`) only goes through the column half.
  Fixed in `9aa7f0385`.
- **A zero-length CSR row faults the V4 decode kernel** ("Memory access fault …
  on address (nil)"); it does not guard empty rows. Length-1 is the workaround.

---

## 5. Corrections to `CLAUDE.md`

- §4 "Not in scope for correctness" (DP attention + FP4 indexer): the fp4
  indexer may be worth **+0.62 accepted tokens** (short context, unverified at
  115k). If it holds, it is a correctness input, not only performance.
- §6 status table compares against native's 4.45; the config-matched reference
  is **3.50**.
- §6 Stage 4 "not started" — **done**, §2 above.
- §3/§3.1 reference `verify-t7-diagnostics.patch`; patch files are now banned
  (§0) and it was deleted. The content is in git history.
- §2.3's ruling-out of porting `DSparkProposer` is the live question, not
  settled — see §3 above.

---

## 6. Method lessons, earned expensively

- **Measure in the goal regime, or prove the probe regime transfers.** A day
  went into ~80-token probes before §2 established that it does transfer.
- **Run the A-vs-A control before believing any A-vs-B.** A 32% `aux_concat`
  difference looked decisive and was entirely noise + config.
- **Config-match before attributing a gap to your own code.** 41% of the
  "integration deficit" was native running attention the plugin does not have.
- **Bound a diagnostic by its condition, not by call count** — a call-count
  budget lands entirely on warmup/prefill. Hit three times this session.
- A device→host sync (`int(t[0])`) inside a possibly-captured region is an
  illegal-capture crash; stash device clones and print from an eager caller.
- Prefer within-arm identity tests to cross-arm comparisons.
- Check `/home/pzhang12/deepseek/` for prior art before designing anything.

---

## 7. Recommendation

Item A (the draft) is what the goal asks for and is down to one component.
Item B (DP attention) is the bigger number but needs the expensive version, and
parity needs **both** — a perfect draft alone takes c128 from 15641 to ~18700
against native's 32302.

I would do A next, with an **offline single-process harness** rather than more
server A/Bs, and resume B only once A is closed or proven impossible within
vLLM's speculator contract.
