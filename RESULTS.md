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

## 5. Stages 4–5 — not started, and Stage 5 should not start

Stage 4's ISL-115k sweep is now differently motivated: it is no longer "confirm
ATOM's drafter holds up" but "find out whether it is ever better". Stage 5
(recipe + CI) must wait until §4 is resolved.
