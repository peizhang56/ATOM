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

**Open question that decides ownership:** does the *shipped* arm
(`ds_v4_atom_vllm`, vLLM's own draft) show the same deficit? If yes this is a
property of DSpark-on-V4 in plugin mode and not of the ATOM draft; if no, the
ATOM draft introduced it. Not yet run.

## 4. Stages 4–5 — not started

Stage 4 (ISL 115k sweep, three arms) and Stage 5 (recipe + CI) are untouched.
