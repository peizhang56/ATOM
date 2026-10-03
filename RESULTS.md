# Measured results — DSpark on ATOM's vLLM plugin, DeepSeek-V4-Pro-0813

All on this node, 8× gfx950 (MI355X), TP8, `num_speculative_tokens=7`.
Code branch `ds_v4_atom_vllm_dspark_draft_t7fix` @ **`a460a5645`**.

**Two regimes appear below. Do not mix them.**

| regime | how | what it is for |
|---|---|---|
| **GOAL** | `client_dsv4_pro_0813.sh` — ISL 115k / OSL 1k / `--cache 90`, p90 | the benchmark the project is scored on |
| probe | `accept_probe.py`, ~80-token prompts | cheap iteration only |

§1 is the goal regime and is what matters. Everything else is diagnosis.

> **Status 2026-10-02: the ATOM-owned draft is NOT shippable.** It is capped at
> ~2.0 accepted tokens/step against native's 3.28, at every concurrency and in
> both regimes. Root cause **not identified**. Stage 5 must not run.

---

## 1. THE GOAL REGIME — ISL 115k full sweep, all four arms

Seed 531150, 8 GPUs, `--cache 90`. Sources:

| arm | sweep | ATOM sha |
|---|---|---|
| **ATOM draft (`_t7fix`)** | `results_mi355x_dsv4pro_vllm_atom_dspark_t7fix_115k/sweep_20261002_211427` | `a460a5645` clean |
| vLLM draft | `/home/pzhang12/deepseek/…_vllm_atom_tp8_115k/sweep_20261001_065623` | `be473f7a3` |
| plugin, spec OFF | `/home/pzhang12/deepseek/…_vllm_atom_tp8_115k/sweep_20261001_001626` | `74fd942b0` |
| native | `/home/pzhang12/deepseek/…_atom_native_tp8_115k/sweep_20260930_233950` | `74fd942b0` |

`in/s/gpu` (the scored metric), and accepted length:

All figures below are from each sweep's generated `report.md` (the
`benchmark-report` skill). **`report.txt` computes `in/s/gpu` differently — do
not mix the two.**

| conc | **ATOM draft** | acc.len | vLLM draft | tok/step | spec OFF | **native** | tok/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 7563 | **2.129** | 9038 | 3.309 | 7674 | 4463 | 3.307 |
| 24 | 10479 | **1.967** | 12860 | 3.124 | 12169 | 10152 | 3.225 |
| 32 | 11865 | **1.971** | 14141 | 3.207 | 13897 | 15328 | 3.227 |
| 64 | 13570 | **1.825** | 15717 | 3.008 | 17767 | 23771 | 3.259 |
| 128 | 15641 | **2.742** | 13940 | 2.379 | **20958** | **32302** | 3.282 |

ATOM-draft per-position acceptance %:

| conc | p0 | p1 | p2 | p3 | p4 | p5 | p6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 42.3 | 26.4 | 17.2 | 11.7 | 7.9 | 4.8 | 2.6 |
| 64 | 35.4 | 20.1 | 11.8 | 7.1 | 4.4 | 2.5 | 1.3 |
| 128 | 57.3 | 38.9 | 27.5 | 20.0 | 14.7 | 10.1 | 5.8 |

### 1.1 What the sweep settles

**The premise's SHAPE is confirmed. Its LEVEL is not.**

- ATOM's draft **does not decay with batch** — 2.13 → 1.83 → 2.74 is roughly
  flat, where vLLM's falls 3.309 → 2.379. At conc 128 it beats vLLM's draft on
  acceptance (2.74 vs 2.379) and throughput (15641 vs 13940, **+12%**). That is
  exactly the behaviour `SESSION-HANDOFF.md` §3 predicted and the reason the
  project exists.
- But it sits at **~2.0 against native's 3.28**, so it loses at conc 16–64.

**Speculation is currently a net loss at the concurrencies that matter.** At
conc 128 the plugin with spec OFF (20958) beats both spec arms — vLLM's draft
(13940) and ours (15641). At conc 64 spec-off also wins. This is the single
most important number on the page: today, DSpark in the plugin is worth
*negative* throughput where the benchmark is heaviest.

(Caveat: the spec-off sweep is a different ATOM/aiter sha. Direction is far too
large to be sha noise, but re-run on one build before quoting it externally.)

**If the ~2.0 ceiling were lifted to ~3.3**, this arm would beat vLLM's draft at
every concurrency and by ~38% at 128. That is the lever, independently
confirmed at the goal regime.

### 1.2 Where the rest of the gap is, and it is not the drafter

From `SESSION-HANDOFF.md` §2, conc 128, decode step `ITL p50 × tok/step`:

| cause | worth |
|---|---:|
| DP attention | **1.735×** |
| FP4 indexer | **1.176×** |
| product | 2.040× — **measured 2.040×, zero residual** |

Ranked by effect on the scored metric: DP attention (1.735×) > FP4 indexer
(1.176×) > the drafter (1.38× if the ceiling is fixed). The plan explicitly
excludes the first two. **That ordering should be revisited.**

---

## 2. The deficit is REGIME-INDEPENDENT

| regime | ATOM draft acc.len | native |
|---|---:|---:|
| ISL 115k (goal) | 1.83 – 2.74 | 3.28 |
| ~80-token probe | 1.98 – 2.50 | 4.45 |

Same ceiling, same per-position shape (115k conc 16: 42.3/26.4/17.2; probe:
45.6/28.1/17.9). **One systematic defect caps this draft everywhere**, so it can
be debugged at 20-token prompts with 3-minute boots instead of hour-long 115k
sweeps. Confirmed by measurement, not assumed.

---

## 3. Localization: the defect is in what the rolling window delivers

Ablating the context write (`write_context_kv` skipped entirely) on **both**
arms, probe regime, conc 32:

| arm | with window | without | window delivers |
|---|---:|---:|---:|
| native | 4.45 | **1.06** | **+3.39** |
| plugin, ATOM draft | 2.14 | 1.05 | **+1.09** |

**The floors are identical** (1.06 vs 1.05, distribution `{0: ~95%}` on both).
With the window gone the draft is reduced to weights + embedding + LM head +
Markov chain + block geometry + anchor token — and those produce the *same
number on both arms*. So every one of them is equivalent, and the entire gap is
in what the window delivers: the plugin extracts **32%** of the value native
gets from the same mechanism.

---

## 4. Eliminated

### 4.1 By measurement

| candidate | evidence |
|---|---|
| weights, LM head, Markov head, anchor token, block width, backbone stages | identical no-context floors (§3) |
| batch / padding / slot-indexing (the whole Stage-1 family) | **concurrency 1 still gives 2.45** |
| cudagraphs | eager 1.98–2.50 ≈ captured 2.07–2.44 |
| "the window is dead" | ablation collapses to 1.05 — it is read and carries most of the quality |
| rejected rows written at position 0 | real defect, unit-tested; gating them changed nothing (2.09 → 2.14) |
| anchor position convention | `−1` measured 2.11 vs 2.14 |
| writing the rejected suffix at its true positions (native's behaviour) | **1.55 — worse** |
| true positions **+** anchor `−1` (native's exact config) | **2.02** |
| `res_preshuffle` asymmetry | `aiter.mhc_res_shuffle_enabled(1,"gfx950")` = **False**; inert here |
| vestigial draft KV group | one KV group only; patch dormant (§6) |

The positional axis is **exhausted**: five configurations spanning both the
write positions and the read anchor all land between 1.55 and 2.14.

### 4.2 By reading (no code changed)

- `swa_plane` **is** `unified_kv` in both arms (`deepseek_v4_attn.py:1743-1744`,
  `deepseek_v4_bridge.py:1082-1084`) — write target and read source are the same
  tensor; no row-base mismatch possible.
- Aux taps resolve to the **same three layers in the same order** in both arms
  (native `layers[58..60]` via `AuxCaptureSpec`; plugin `idx+1 in (59,60,61)`),
  with the same `hc_post(...).mean(1)` reduction. Confirmed numerically too:
  diagonal cosines 0.987/0.988/0.990 dominate every off-diagonal — no
  permutation, no off-by-one layer, no scale factor.
- vLLM's dense Markov path calls only `markov_embed`/`markov_bias`, both
  defined. `apply_markov_bias_gathered`/`compute_confidence` are on the
  top-k/adaptive branches, which are off.
- `DeepseekV4AttentionVllm` overrides only `forward_impl`/`_sparse_attention`;
  `dspark_attention` calls neither — the class swap is inert for the draft.
- `reset_kv_cache` is a documented no-op; `self.layers = self.mtp` is a
  deliberate alias.
- Checkpoint naming: `mtp.0.attn.*` mirrors `layers.60.attn.*` exactly, minus
  `compressor`/`indexer` — the draft is dense-window by design, and `wq_a`+`wkv`
  → packed `wqkv_a` is the convention the target loads correctly under.

---

## 5. The `main_x` discriminator does NOT discriminate

Instrumented at `DeepseekV4DSpark.project_context` — the one method both arms
reach. One fixed 19-token prompt, greedy, single request. Both arms returned the
**identical completion**.

| pair | aux rel | **main_x rel** |
|---|---:|---:|
| **native vs native** (control) | 7.9% | **13.97%** |
| plugin vs native | 32.0% | 15.13% |
| plugin vs native2 | 32.9% | 16.48% |

`main_x` — the tensor that feeds the window — differs between arms by 15.1%
against a same-arm floor of **14.0%**. No signal. **Recorded so nobody spends
two more server boots on it.**

`aux_concat` does differ ~4× the floor, but it does not survive `main_proj` into
`main_x` measurably.

**The determinism floor is the real finding here.** Two native runs of one
prompt, same seed, greedy, land 14% apart on `main_x` while emitting identical
text — MoE top-k routing flipping on tiny numerical differences. *No live-server
hidden-state A/B can resolve anything finer than ~15%.* To compare the two code
paths numerically it must be done **offline, in one process**.

---

## 6. Earlier stages (still valid)

**Stage 1 — cudagraph fault: FIXED** (`a460a5645`). Two defects: the draft's
width-7 block pass was classified PREFILL because `_infer_atom_attn_state` gates
DECODE on `1 + num_spec` = 8; and vLLM pads the draft to a capture bucket by
*token* count so fabricated rows inherited ring slot 0. Isolated by **bisection**
after two code-reading hypotheses failed.

**Stage 2 — vestigial draft KV group: NOT A PROBLEM.** With ATOM's draft,
`get_draft_kv_cache_layer_names()` returns the proxy's own layer name, so vLLM
builds one group and `dspark_draft_kv_patch` is dormant. GPU KV cache 3,644,928
tokens.

**Stage 3 — correctness.** The arm is non-deterministic at ~20 tokens, spec or
no spec, so the token-identical test cannot be run here. GSM8K (full 1319,
5-shot, strict):

| arm | mean |
|---|---:|
| native | **0.9515** |
| plugin spec-OFF | 0.9507 |
| plugin, ATOM draft | 0.9413 |
| plugin, vLLM draft | 0.9262 |

Native spec decoding is lossless (0.9515 vs 0.9507); the plugin's spec-on
accuracy deficit is **pre-existing and larger in the shipped arm**. The ATOM
draft roughly halves it.

---

## 7. Open — the one test never run

Everything between the aux tap and the ring is verified live and roughly right;
positions, coverage, weights and code are cleared. What remains:

1. **The read gather — the write/read round-trip identity.** Write known values
   at known positions, gather them back through the production read path, assert
   the identity. **Arm-local, deterministic, no noise floor.** Deferred at least
   four times in favour of server A/Bs; it should have come first.
2. The written values, if the 14% activation floor is masking a real difference
   — which requires the offline single-process harness (§5).

### Method notes, earned the hard way

- **Measure in the goal regime, or prove the probe regime transfers.** A day was
  spent at 80-token prompts before §2 established that it does.
- **Run the A-vs-A control before believing any A-vs-B.** §5's 32% "signal" was
  noise.
- **Bound a diagnostic by its condition, not by call count** — a call-count
  budget lands entirely on warmup/prefill and reports nothing. Hit three times.
- A device→host sync (`int(t[0])`) inside a possibly-captured region is an
  illegal-capture crash. Stash device clones; print from an eager caller.
- Prefer within-arm identity tests over cross-arm comparisons.

---

## 8. Reproduce

```bash
# goal regime (hours)
bash serve_dsv4_pro_vllm_atom_tp8.sh
SERIES=dsv4pro_vllm_atom_dspark_t7fix_115k CONCURRENCY=16,24,32,64,128 \
  bash client_dsv4_pro_0813.sh

# probe regime (minutes) — transfers, per §2
./stop.sh && rm -rf /root/.cache/atom/*
bash serve_dsv4_pro_vllm_atom_tp8.sh
python accept_probe.py --concurrency 32
grep "SpecDecoding metrics" logs/<server>.log | tail -2
```

Prior-session 115k sweeps live in `/home/pzhang12/deepseek/results_mi355x_*`.
Note `client_dsv4_pro_0813.sh` writes ~420 KB/request of prompt text twice per
point; commit only `report.*`, `arm.json`, `economics.*` and the per-point
`profile_c*.json`, never `inputs.json`/`prompts.jsonl`.

---

## 9. The 4.45-vs-2.14 gap is THREE effects, not one (2026-10-02)

I had been benchmarking the plugin's draft against native's 4.45. That is the
wrong reference: native runs three flags the plugin does not
(`serve_dsv4_pro_atom_native_tp8.sh:28,30`):

```
-tp 8 --kv_cache_dtype fp8 --index_cache_dtype fp4
--enable-dp-attention --enable-tbo
```

Measured by running **native with the plugin's configuration**, probe regime,
`accept_probe.py --concurrency 32`:

| native config | indexer | DP+TBO | acc.len |
|---|---|---|---:|
| full (as shipped) | fp4 | on | **4.45** |
| `serve_native_fp8idx.sh` | **fp8** | on | **3.76 – 3.90** |
| `serve_native_plugincfg.sh` | fp8 | **off** | **3.45 – 3.53** |
| plugin + ATOM draft | fp8 | off | **2.14** |

| cause | cost in accepted tokens |
|---|---:|
| fp4 → fp8 indexer | **−0.62** |
| DP attention + TBO off | **−0.33** at ~80 tokens — **but ~0 at the goal regime**, see below |
| **our integration** | **−1.36** |

> **CORRECTED.** `kb/e2-gap-is-dpa-tbo-*.md` measured acceptance at ISL 115k
> with DPA and TBO removed: `tok/step` at conc 128 is **native full 3.282 vs
> native−DPA−TBO 3.306** — unchanged. So the −0.33 above is a short-context
> probe artifact, not a real acceptance effect. At the goal regime the only
> config term that may cost acceptance is the **fp4 indexer** (−0.62 measured
> at short context, still unverified at 115k).

### Consequences

- **The config-matched reference for the plugin arm is ~3.50, not 4.45.** Our
  integration deficit is 1.36 tokens (39%), not 2.3 (52%). Still the largest
  single term, but 41% of the gap was never our bug.
- **The FP4 indexer is a correctness item, not only a performance one.**
  `CLAUDE.md` §4 files it under "Not in scope for correctness" with DP
  attention. It is worth **+0.62 accepted tokens** on top of its 1.176×
  throughput, because the indexer selects which tokens sparse attention sees —
  its precision changes the hidden states the draft consumes. That separation
  should be revised.
- **Parity with native needs all four**: the draft fix (+1.36), the fp4 indexer
  (+0.62), DP attention + TBO (+0.33), plus their throughput effects.
- This also explains why §5's `aux_concat` differed 32% between arms against a
  7.9% same-arm floor: the two arms' targets are **not** running the same
  attention. That difference was real, and it was config, not a wiring defect.

Reproduce: `serve_native_plugincfg.sh`, `serve_native_fp8idx.sh` (both derived
from the native script by flag substitution), then
`accept_probe.py --concurrency 32 --requests 160` and
`grep "MTP Stats " logs/<log>`. With DP on, the batch splits across 8 engines,
so a small probe never reaches the per-engine stats threshold — use `--requests
160` or larger.

---

## 10. Driver localized, and the DP-attention design (2026-10-03)

### 10.1 Every draft input is now proven equivalent

With the config matched (`serve_native_plugincfg.sh`: no DPA, no TBO, fp8
indexer), the plugin's draft inputs are indistinguishable from native's:

| | control (A vs A) | plugin vs native |
|---|---:|---:|
| `aux_concat` rel err | 7.37% | **7.44%** |
| `main_x` rel err | 13.11% | **12.91%** |

Both at or below the same-arm floor. Together with the integer round-trip on
window addressing, the position/coverage census, and the identical no-context
floors, **every input the draft consumes is equivalent** — yet the output
differs by 1.36 tokens (3.50 vs 2.14).

`sample_indices` is also eliminated: under `SAMPLE_FROM_ANCHOR` the vLLM kernel
stores `sample_idx = req*7 + query_off` ← `query_idx = req*7 + query_off`, the
identity, so vLLM reads exactly the row ATOM's block produced.

**What remains is the drafting loop itself** — vLLM's `DSparkSpeculator`
driving ATOM's model instead of ATOM's `DSparkProposer`. That is now evidence,
not preference, and it points at the re-design being about the *driver*.

### 10.2 DP attention in the plugin — design and prior art

`kb/e1-*` is a **negative result, do not re-run**: vLLM's
`--data-parallel-size 8 --enable-expert-parallel` is 2.0–4.7x WORSE, because it
is 8 independent engine cores each with its own scheduler, KV pool and prefix
cache (which fragments to 38.9% at conc 16), plus expert-parallel MoE.

`kb/e3-*`: **DP attention is the entire throughput gap** (1.95x at conc 64,
2.31x at 128). **TBO is a net 4–8% LOSS** as configured — ATOM's own recipe
pairs it with `GPU_MAX_HW_QUEUES=5` + `ATOM_NUMA_BIND=1`, which the baseline
script does not set. **Do not port TBO.**

ATOM's flavour is DP *inside* a TP group: one scheduler, one KV pool, one
prefix cache; only the attention op is sharded by request, hidden states
all-gathered around it, MoE stays TP.

Feasibility notes gathered:

- `atom/plugin/config.py` **already implements the DP-attention rank layout for
  SGLang** (`runtime_tp_size`/`runtime_dp_size`/rank mapping). Only the vLLM
  plugin hardcodes `enable_dp_attention=False`
  (`atom/plugin/vllm/platform.py:81` explains why: it is a rank-layout decision
  in ATOM's engine, which plugin mode replaces with vLLM's `GPUModelRunner`).
- V4's attention linears are **ATOM's own** (`atom.model_ops.linear`), not
  vLLM's, so ATOM controls their sharding. They size from
  `get_tp_group().world_size`.
- With MLA under TP every rank already holds the same latent, so DP attention
  needs **no KV redistribution** — only a work split plus an all-gather.

Sketch: let the vLLM plugin accept `enable_dp_attention`, give the attention
linears a size-1 TP group (replicated weights) while MoE stays TP, and have the
V4 proxy attention op select its 1/N of requests and all-gather the hidden
states back.

**Make-or-break check, not yet done:** whether the plugin's load path can give
attention replicated weights while MoE stays sharded.
