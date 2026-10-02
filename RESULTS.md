# Measured results — DSpark on ATOM's vLLM plugin, DeepSeek-V4-Pro-0813

All on this node, 8× gfx950 (MI355X), TP8, `--kv-cache-dtype fp8`,
`num_speculative_tokens=7`, branch `ds_v4_atom_vllm_dspark_draft_t7fix`.

Short-context probes use ~80-token prompts. These are **not** the ISL 115k
regime `SESSION-HANDOFF.md`'s 3.3 / 2.38 numbers come from; compare shapes, not
absolute values, against those.

---

## 1. Stage 1 — the cudagraph fault (FIXED)

Two defects, both now fixed and pushed (`a460a5645`).

| arm | cudagraphs | result |
|---|---|---|
| T=8 (branch as found) | FULL_DECODE_ONLY | serves, drafts from wrong anchors (~1.10) |
| T=7 | FULL_DECODE_ONLY | `Memory access fault`, all 8 GPUs |
| T=7 | `--enforce-eager` | clean, acceptance 1.98–2.50 |
| T=7, draft eager / target captured | bisect | clean, **614 tok/s** |
| T=7 + both fixes | FULL_DECODE_ONLY | clean, **1105–1190 tok/s** |

Root cause: the draft's block pass is uniform at width 7, but
`_infer_atom_attn_state` gates DECODE on `1 + num_spec` = 8, so the draft was
classified PREFILL and allocated fresh per-step tensors; its own captured FULL
graph then replayed against freed addresses. Isolated by **bisection** (forcing
only the draft eager made it vanish), not by inference — two earlier
code-reading hypotheses were wrong.

Second defect: vLLM pads the draft to a capture bucket via the **token** count
(`num_reqs_padded` is usually unpadded), so the block runs more blocks than
there are requests; the fabricated rows inherit ring slot 0, a live request's.
Fixed with ATOM's own `prepare_block`/`mask_pad_tail`, driven from outside the
replay. Confirmed live: `25 real of 26 blocks` at concurrency 30.

Clean at concurrency 7 / 17 / 30 / 32 / 128. At 128: 256 requests, 65536 output
tokens, 1105 tok/s, zero faults.

## 2. Stage 2 — vestigial draft KV group: NOT A PROBLEM

The plan assumed the arm reserved a second vLLM KV group for nothing. It does
not. With ATOM's draft, `get_draft_kv_cache_layer_names()` returns the **proxy's
own** layer name, so `draft_attn_layer_names` is empty and vLLM builds **one**
group. `_build_v4_proxy_draft_kv_cache_groups` asserts on empty draft specs, so
the server booting is itself proof the path never fires.
`dspark_draft_kv_patch` is dormant with the ATOM draft, not wasteful. No change
made.

GPU KV cache: 3,644,928 tokens; prefix rollback reported for one group
("roll back last 1024 token(s) per hit = 8 proxy block(s)").

## 3. Stage 3 — correctness

### 3.1 Determinism control: the arm is non-deterministic, spec or no spec

`greedy_capture.py --repeat 3`, batch 1, temperature 0, seed 0, ~20-token
prompts. **All 8 prompts unstable on BOTH arms** — spec-ON and spec-OFF alike.

This is a property of the **plugin arm itself**, not of speculative decoding,
and not caused by these changes. `SESSION-HANDOFF.md` reported non-determinism
from ~15k tokens; it is in fact present at ~20 tokens.

Consequence: **the token-identical spec-on/spec-off test cannot be run on this
arm.** Losslessness has to be argued statistically.

### 3.2 GSM8K — speculative decoding costs ~0.9 points

Full set (1319), 5-shot, `num_concurrent=64`, strict-match:

| arm | runs | range | mean |
|---|---|---:|---:|
| spec-OFF | 0.9530, 0.9462, 0.9530 | 0.9462 – 0.9530 | **0.9507** |
| spec-ON | 0.9409, 0.9447, 0.9416, 0.9378 | 0.9378 – 0.9447 | **0.9413** |

The two bands **do not overlap**. Welch t ≈ 3.5 — the gap is not run-to-run
noise, even though each individual run's stderr (±0.0065) would suggest it is.
Repeating the eval on one server is what separates the two; a single pair of
runs would have been read as "within noise".

n=200 subset, for reference: spec-OFF 0.975 ± 0.0111, spec-ON 0.970 strict /
0.960 flexible — i.e. at n=200 the effect is invisible. The full set is needed.

**This fails Stage 3's exit criterion** ("GSM8K within noise of the no-spec
plugin arm"). Greedy speculative decoding with greedy drafting should be
lossless by construction — accept iff the draft token equals the target's
argmax — so a systematic deficit means something in the verify path is not the
identity it should be.

Candidates, untested:
- Target logits differ between a verify batch (uniform q=8) and a plain decode
  (q=1) — batch-shape-dependent numerics. The arm's own non-determinism shows
  the model sits on near-ties often enough for this to flip tokens.
- Anchor / bonus-token handling at a block boundary.

### 3.3 Ownership of the deficit: pre-existing, and WORSE in the shipped arm

Ran the same three-repeat full GSM8K on `upstream/ds_v4_atom_vllm` (vLLM's own
draft), strict-match:

| arm | runs | mean |
|---|---:|---:|
| spec-OFF | 0.9530, 0.9462, 0.9530 | **0.9507** |
| spec-ON, **ATOM** draft (ours) | 0.9409, 0.9447, 0.9416, 0.9378 | **0.9413** |
| spec-ON, **vLLM** draft (shipped) | 0.9386, 0.9060, 0.9340 | **0.9262** |

So the accuracy deficit under speculation is **pre-existing and larger in the
shipped arm**. The ATOM draft did not introduce it and in fact roughly halves
it. Root-causing it is a DSpark-on-V4 plugin issue, not a blocker on this work.

---

## 4. THE HEADLINE: the ATOM draft drafts far worse than vLLM's

Same workload (GSM8K, 5-shot, `num_concurrent=64`), same server config, same
node — only the draft model differs:

| | pos 1 | pos 2 | pos 3 | pos 4 | mean accept len | accepted tok/s |
|---|---:|---:|---:|---:|---:|---:|
| **vLLM draft** (shipped) | 0.898 | 0.782 | 0.593 | 0.429 | **4.28 – 4.37** | ~1300 |
| **ATOM draft** (ours, `_t7fix`) | 0.654 | 0.411 | 0.194 | 0.099 | **2.44** | ~646 |

**The gap is already at position 1** — 0.65 vs 0.90. No block-width,
RoPE-extrapolation or "trained at γ=5, run at 7" argument explains a
first-token gap: position 1 is the easiest prediction and the one least
sensitive to block geometry. Something in the ATOM draft's wiring is feeding it
a degraded context.

It is **not** cudagraph-related: fully eager the ATOM draft reaches only
1.98–2.50, essentially the same as captured (2.07–2.44). So Stage 1's fixes are
real but orthogonal to this.

### What this means for the project

The premise of the whole effort — from `SESSION-HANDOFF.md` §3, that vLLM's
drafter loses acceptance with batch (3.31 @ conc 16 -> 2.38 @ 128) while ATOM's
holds ~3.3, so ATOM's should replace it — **is not supported in this regime**.
Here vLLM's draft is the better drafter by a wide margin, on both acceptance
(1.8x) and accuracy.

Shipping the ATOM draft today would be a regression on both axes. It is not
ready, and Stage 5 should not run.

Two readings, and they need separating before any more work:

1. **Our wiring is still defective.** Most likely. Native ATOM reportedly gets
   ~3.3 at ISL 115k, so 2.44 at short context with a 0.65 first position
   suggests the draft is attending to a wrong or stale rolling context window
   (`precompute_and_store_context_kv` -> `write_combined_context_kv`), or its
   weights are mapped subtly wrong (`f32df74be` fixed 97/97 weights being at
   init values -- worth re-verifying they are mapped correctly, not just
   loaded).
2. ATOM's draft is genuinely weaker at short context, and its advantage only
   appears at 115k. Possible, but it does not explain position 1.

### ANSWERED: reading (1). The plugin wiring is defective.

Ran native ATOM (`serve_dsv4_pro_atom_native_tp8.sh`) on the identical
workload. Native is the reference implementation of this exact draft, loading
the same `mtp.{0,1,2}.*` weights from the same checkpoint.

| arm | mean accepted length | GSM8K (strict, full 1319) |
|---|---:|---:|
| **native ATOM** | **4.43 – 4.46** | **0.9515** |
| plugin, vLLM draft | 4.28 – 4.37 | 0.9262 |
| plugin, **ATOM draft (ours)** | **2.44** | 0.9413 |
| plugin, spec-OFF | — | 0.9507 |

Native: `Acceptance rate: 49.49%`, accepted-length distribution
`{0: 8.4%, 1: 11.3%, 2: 18.3%, 3: 16.0%, 4: 13.2%, 5: 12.3%, 6: 7.7%, 7: 12.8%}`.

**The same draft reaches 4.45 natively and 2.44 through our plugin wrapper.**
So the premise is fine and ATOM's draft is good — *our integration loses ~45%
of its acceptance*. Reading (2) is dead: it is not that ATOM's draft is weak at
short context.

Two further things fall out of the native run:

- **Native spec decoding is lossless**: 0.9515 vs the plugin's spec-off 0.9507.
  So the plugin's spec-on accuracy deficit (§3.2) is *also* an integration
  issue, not something inherent to DSpark. Both symptoms likely share a cause.
- Native at short context gets 4.45, not the ~3.3 quoted for ISL 115k in
  `SESSION-HANDOFF.md`. Those are different regimes; do not mix them.

### Next step, concrete

Find what the plugin feeds the draft that native does not. The draft's quality
inputs are exactly three, and the first is the prime suspect:

1. **The rolling context window.** `precompute_and_store_context_kv` ->
   `write_combined_context_kv`. Native drives this from its own proposer with
   live metadata; the plugin re-enters a *stashed* target forward context
   (`get_deepseek_v4_target_metadata()`). Under `FULL_DECODE_ONLY` the target's
   Python does not re-run on replay, so that stash can be stale — the
   `_mtp_hidden_buffer` hazard, one level up. Check whether the stash is
   refreshed every step. (Note: eager-vs-captured acceptance is the same here,
   which argues against staleness being the *whole* story, but the stash may be
   wrong rather than stale.)
2. **The aux hidden states** fed to `combine_hidden_states` — wrong layers,
   wrong mHC reduction, or wrong row ordering would degrade every position
   uniformly, which matches a position-1 gap.
3. **Weight mapping.** `f32df74be` fixed 97/97 weights being at *init values*;
   verify they are mapped to the right parameters, not merely non-zero.

The cheapest discriminator: dump the draft's `main_x` (post
`project_context`) for one request on both arms for the same prompt and compare
them numerically. If they differ, it is (2) or (3); if they match, it is (1).

**Superseded — see §6.** Reading the three inputs found a concrete defect in
(1) without needing the dump: it is not the *content* of the context rows that
is wrong, it is the *positions they are written at*.

## 5. Stages 4–5 — not started, and Stage 5 should not start

Stage 4's ISL-115k sweep is now differently motivated: it is no longer "confirm
ATOM's drafter holds up" but "find out whether it is ever better". Stage 5
(recipe + CI) must wait until §4 is resolved.

---

## 6. Candidate root cause of the 2.44-vs-4.45 gap: rejected rows write the draft's context window at position 0

> **STATUS: MEASURED. The defect is real and now fixed (`1b3db20d0`). It is
> NOT the cause of the acceptance gap — acceptance did not move.** The
> hypothesis below had an unusually good fit and was still wrong about what it
> explained. That is three code-reading diagnoses on this project that did not
> survive contact with hardware; the §6.5 numbers, not the §6.1–6.4 reasoning,
> are the part to trust.

### The mechanism

vLLM's `prepare_dflash_inputs` kernel fills `context_positions` per request
span `[ctx_start, ctx_end)`
(`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:541-567`):

- rows `[0, num_valid_ctx)` → the real target positions;
- rows `[num_valid_ctx, num_ctx)` — **the rejected suffix** — → **position 0**
  and `PAD_SLOT_ID`.

vLLM says so itself, and says why it is safe *for vLLM*:

> those rows write no KV and their positions are never consumed

True for **vLLM's** draft, which writes through `slot_mapping` and therefore
skips `PAD_SLOT_ID`. **Our draft does not.** ATOM owns the draft's KV and
addresses its rolling SWA ring by *absolute position* — by design, per the
wrapper's module docstring — so it discards `slot_mappings` and keeps the
positions:

```
precompute_and_store_context_kv(hidden, context_positions, slot_mappings)   # slot_mappings ignored
  -> write_combined_context_kv -> stage.write_context_kv -> swa_write -> _swa_write_kernel
       dst_row = window_row(slot, pos, ...) = slot*SLOT_ROWS + ring_start + pos % RING_SLOTS
```

`_swa_write_kernel` gates **only** on `dst_row` bounds — there is no validity
gate on `pos` — and it writes the **last** `write_n = min(tok_n,
write_per_batch)` rows of each span, with `write_per_batch = window_size +
num_spec = 128 + 7 = 135`. A decode span is 7–8 rows, so `write_n` is the whole
span: **the rejected suffix is written, every step.**

With the measured geometry (`CLAUDE.md` §3.1: `ring_slots=135, ring_stride=135,
ring_start=34380, slot_rows=43746`), `pos = 0` resolves to ring offset 0 —
comfortably in bounds, so the bounds guard does not catch it. The write lands.

### Why it is not a harmless write

At acceptance 2.44 with T=7 there are ~4.5 rejected rows per request per step,
and **all of them collide on ring slot 0** of that request's window.

Is slot 0 live? The read side gathers the last `window_size = 128` positions, so
for current position `P` it reads slots `(P-127 .. P) mod 135` — 128 of the 135
slots. Slot 0 is in that set unless `P mod 135 ∈ {128..134}`, i.e. it is live
**~95% of steps**. A legitimate rewrite of slot 0 only comes around once per 135
positions, while the corruption recurs every step — so the draft's 128-row
context window carries a wrong row essentially permanently.

The corrupting value is a *real* hidden state for a token that was rejected,
i.e. plausible-looking and wrong. And the loop is self-reinforcing: more
rejection → more corrupt rows → worse context → more rejection.

### Why it fits every observation we have

| observation | fit |
|---|---|
| gap is already at **position 1** (0.65 vs 0.90), §4 | ✓ corrupts the shared context window, not block geometry — degrades all positions |
| **eager ≈ captured** (1.98–2.50 vs 2.07–2.44), §4 | ✓ this write is explicitly eager, outside the captured graph |
| **native is fine** (4.45) | ✓ native's `compute_draft_kv` passes the target forward's *real* positions for every row |
| **vLLM's own draft is fine** (4.33) | ✓ it writes via `slot_mapping` and skips `PAD_SLOT_ID` |

The native arm is the sharpest confirmation. `DSparkProposer.compute_draft_kv`'s
own docstring argues rejected rows are safe to write:

> Rejected rows are harmless -- they land on future positions, unread until the
> step that accepts them rewrites them.

That argument depends *entirely* on those positions being real future positions.
vLLM's zeroing destroys exactly the premise, and we inherited the write without
inheriting the premise.

### The fix, and a corollary that makes it cheap

Note what native's docstring also establishes: those rows are **never read
before being rewritten**. So *not writing them at all is equivalent to writing
them* — the fix is simply to gate them out, not to reconstruct their true
positions.

`slot_mappings` is the available signal and our wrapper already receives it and
throws it away; `PAD_SLOT_ID` marks exactly these rows. Shape of the change:

- thread an optional `valid_mask` through `write_combined_context_kv` → stage
  `write_context_kv` → `swa_write` → `_swa_write_kernel`, defaulting to `None`
  so the **native path stays byte-identical**;
- in the kernel, skip on `mask[src_id] == 0` — the same idiom
  `swa_scatter_rows` already applies via `batch_id_per_q_token` ("the same gate
  the fused writes apply");
- must be **sync-free**: no `.nonzero()`/`.item()` compaction. This call is
  eager, but a per-step device→host sync would cost throughput.

`write_context_kv` is outside the compiled region (`deepseek_v4_dspark.py`'s own
COMPILE BOUNDARY block says it "stays eager"), so this does not violate the
no-editing-compiled-files rule — **re-read that block before touching the file
anyway.**

**Caveat to measure, not assume:** `PAD_SLOT_ID` also covers rows whose
`ctx_block_id == 0` (vLLM's null block, after sliding-window eviction). For
ATOM's ring those are a *conservative* false positive — a row we could have
written and skip instead. Expected to be rare (the rows in a span are the
newest positions, hence resident), but log a counter of masked rows and confirm
it tracks `num_rejected` rather than exceeding it.

### 6.5 What the fix measured: the defect is real, the hypothesis is wrong

Implemented as `1b3db20d0` (optional `valid_mask`, plugin → stage →
`swa_write` → `_swa_write_kernel`, `None` by default so every native caller is
byte-identical). `tests/test_swa_write_ring.py` 51 passed, including a new test
that asserts the **ungated** write does clobber ring slot 0 — so the mechanism
is confirmed at unit level, not merely read.

Live, `accept_probe.py --concurrency 32`, same node, same session:

| arm | mean accepted length | tok/s |
|---|---:|---:|
| plugin, ATOM draft, gate OFF | 2.09 | 1055 |
| plugin, ATOM draft, **gate ON** | **2.14** | 1208 |
| (prior sessions, same arm) | 1.98 – 2.50 | — |
| native ATOM | 4.43 – 4.46 | — |

**No movement.** 2.09 → 2.14 sits inside the arm's own 1.98–2.50 spread.

The gate is definitely active and dropping the right rows — the instrumentation
shows decode steps of 256 rows (32 requests × 8-wide target spans) with
155–223 dropped, i.e. ~5–7 rejected rows per request, tracking the ~2.1
acceptance exactly as predicted:

```
DSpark draft context gate dropped 221 of 256 rows
DSpark draft context gate dropped 202 of 256 rows
DSpark draft context gate dropped 191 of 256 rows
```

So the rejected rows *were* being written at position 0, the write *was*
landing on a live window row, and removing it changes **nothing** measurable.
One corrupted row out of a 128-row context window is apparently not worth 45%
of acceptance.

Keep the fix: it is a genuine correctness defect with test coverage, it costs
nothing, and leaving a known-wrong write in place would poison every later
diagnosis on this path. But it is not the headline.

**Instrumentation note, repeated from Stage 1 and repeated again here:** the
first attempt logged the first 5 calls and reported "dropped 0 of 4608 rows" —
all prefill, which has no rejected rows by construction. A diagnostic bounded
by *call count* lands on warmup/prefill and says nothing. Bound it by the
*condition* instead. This is the third time this file records that lesson.

### 6.6 Also ruled out, cheaply

`res_preshuffle`. The plugin's aux reconstruction passes
`hc_post(..., res_preshuffle=hc_state.res_preshuffle)` while native's
`AuxCaptureSpec` hook (`dspark_proposer.py:502-516`) omits the argument
entirely, defaulting it False — a real asymmetry. It is inert here:
`enable_res_preshuffle = aiter.mhc_res_shuffle_enabled(1, "gfx950")` returns
**False** on this hardware, so both arms reconstruct identically. Would matter
on gfx1250; does not matter on MI355X.

### 6.7 Three more things measured, all negative — and one good control

Same server, same probe, concurrency 32 unless stated.

| experiment | mean accept | pos 1 | reading |
|---|---:|---:|---|
| gate ON (§6.5) | 2.14 | 0.456 | baseline |
| **concurrency 1** | 2.45 | 0.594 | deficit is NOT batch/slot related |
| **anchor position −1** | 2.11 | 0.432 | anchor convention is not it |
| **context write ablated** | **1.05** | **0.034** | the window IS read, and carries most of the quality |
| native | 4.45 | 0.916 | target |

**Concurrency 1 (2.45).** A single request, no cudagraph padding, no
`mask_pad_tail`, one state slot, no cross-request interference — and the draft
still loses most of the gap. So nothing in the batch/padding/slot-indexing
family explains it. This retires a whole class of suspects, including the ones
Stage 1 was about.

**Anchor position (2.11).** ATOM and vLLM genuinely spell the anchor
differently. ATOM takes "the position of the last token already in the
sequence": `_build_block_plan` does `draft_pos = positions + [1..T]` and spans
the window over `anchor-(W-1) .. anchor`. Native honours that —
`Drafter.prepare_inputs` picks anchor row `cu_seqlens_q[req] + accepted_count`,
position `p0+k`, paired with the token sampled FROM that row, which belongs at
`p0+k+1`. vLLM's `query_pos = last_valid_pos + 1` already IS the anchor token's
own position, and its `last_valid_pos` equals native's `p0+k`. So on paper the
plugin is one too high and `-1` is the correct alignment.

Measured, `-1` is **not better** (2.11 vs 2.14, pos-1 0.432 vs 0.456 — both
inside noise). Reverted, since shipping a change that measures no better on a
theory the data does not support is how the last three dead ends started. Worth
knowing the convention mismatch is real but worth ~nothing: 127 of 128 window
rows are still correct under a one-position shift.

**Context ablation (1.05) — the useful control.** `ATOM_DSPARK_NO_CTX=1`
(`e461a8d41`, flag-gated, off by default) skips the context write entirely.
Acceptance collapses to 1.05 and position-1 to 0.034. So the rolling window is
genuinely read, genuinely wired, and worth ~1.1 of the 2.14. The draft is not
running blind; it is running on a context that is *present but inferior* to
native's.

That is a much tighter target than "something in the integration". Everything
between the aux tap and the ring is live and roughly right; what is in those
rows is not as good as what native puts there.

### 6.8 The `main_x` discriminator was run. It is noise-limited and cannot settle it.

Instrumented (`1993cb89b`, flag-gated) at `DeepseekV4DSpark.project_context` —
the one method BOTH arms reach, native via `write_context_kv`, the plugin via
`combine_hidden_states`. One fixed 19-token prompt, greedy, single request, a
pinned-row filter so only that prefill is captured. Both arms returned the
**identical completion**.

First read looked decisive — `aux_concat` differing 32% relative between arms.
Then the control this project's rules require ("A differs from A" before "A
differs from B", `CLAUDE.md` §5) was run: native restarted and given the same
prompt again.

| pair | aux cos | aux rel | **main_x cos** | **main_x rel** |
|---|---:|---:|---:|---:|
| **native vs native** (control) | 0.9929 | 7.9% | **0.9893** | **13.97%** |
| plugin vs native | 0.9889 | 32.0% | 0.9862 | 15.13% |
| plugin vs native2 | 0.9872 | 32.9% | 0.9842 | 16.48% |

**`main_x` — the tensor that actually feeds the draft's rolling window — differs
between arms by 15.1%, against a same-arm run-to-run floor of 14.0%.** That is
not a difference. The method cannot see a signal here.

So the §4 "cheapest discriminator" is answered: **it does not discriminate.**
Recorded so the next session does not spend another two server boots on it.

Two real things do fall out:

- `aux_concat` genuinely differs more between arms (32%) than within one arm
  (7.9%), with min row cosine 0.902 vs 0.976 — about 4× the floor. The layer
  correspondence is **correct** (diagonal cosines 0.987/0.988/0.990 dominate
  every off-diagonal, so no permutation, no off-by-one layer, no scale factor).
  But whatever that difference is, it does not survive `main_proj` into `main_x`
  at a level this method can resolve.
- **The arm's determinism floor is itself remarkable and under-appreciated.**
  Two native runs of the same prompt, same seed, greedy, produce hidden states
  14% apart on `main_x` while emitting identical text. For an MoE target that is
  explainable — a tiny numerical difference flips a top-k expert and the
  residual moves a lot — but it means *no* hidden-state A/B on a live server can
  resolve anything finer than ~15%. §3.1 reported this arm as non-deterministic
  in its outputs; this quantifies it in its activations.

### Next step

Hidden-state comparison on a live server is a dead end at this precision. To
compare the two code paths numerically, the nondeterminism has to be removed
rather than averaged over: load the target once **offline**, outside any server,
run one fixed prefill through both the native and plugin aux paths in the same
process, and diff. Same weights, same input, no scheduler, no MoE routing
divergence between processes — then a 15% difference means something.

Candidates (2) aux hidden states and (3) weight mapping from §4 are still open,
minus the `res_preshuffle` sub-case. The discriminator named in §4 has not been
run and is now the thing to run: dump the draft's `main_x` (post
`project_context`) for one request on both arms for the same prompt and compare
numerically. Match → the context path; differ → aux or weights.

Worth checking first, since it is nearly free: the plugin selects aux layers via
`get_eagle3_aux_hidden_state_layers` (`+1` onto the checkpoint's 0-based
`dspark_target_layer_ids` → `(59, 60, 61)`, consumed as "after
`self.layers[idx]` where `idx+1 in aux_layers`"). Confirm native's
`AuxCaptureSpec` registers the **same three layers** and in the **same order** —
the concat feeding `main_proj` is order-sensitive and a reversed or
off-by-one-layer triple would degrade every position uniformly, which is the
shape of what we see.
