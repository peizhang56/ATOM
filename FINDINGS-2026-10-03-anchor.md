# The draft's 1.36-token deficit was an anchor off-by-one — 2026-10-03

Closes **open item A** of `SESSION-HANDOFF-2026-10-03.md` §3. Also opens a new
one: §4 below.

Code: `ds_v4_atom_vllm_dspark_draft_t7fix` @ `8f1954f12` (pushed).
All numbers are the probe regime — `accept_probe.py`, ~80-token prompts, TP8,
`num_speculative_tokens=7`.

---

## 1. The defect

The two arms number the DSpark anchor differently, and the plugin wrapper
passed one convention into the other untranslated.

| | what `positions`/`anchor_positions` means | source |
|---|---|---|
| vLLM | per-TOKEN position. `query_pos = last_valid_pos + 1 + query_off`, so column 0 is the position of the anchor TOKEN | `dflash/speculator.py:571` |
| ATOM | position of the row whose logits produced the anchor token = `last_valid_pos`. `_build_block_plan` then derives `draft_pos = anchor+1 .. anchor+T` and the window `[anchor-W+1, anchor]` | `deepseek_v4_dspark.py:408`, `dspark_proposer.py:622` |

`DeepseekV4DSparkDraft.forward` passed vLLM's number straight through. Two
consequences at once:

1. every drafted token was RoPE'd **one position too far**;
2. the window's **newest row — the anchor — was a position the target had never
   forwarded.** It read `-1` before the ring wrapped and a 135-position-old row
   after.

The fix is one line plus its justification:

```python
anchor_positions = (positions[anchor_idx] - 1).clamp(min=0)
```

(The clamp is for dummy/profiling batches, whose positions are all zero; a real
anchor is `last_valid_pos + 1 >= 1`.)

## 2. How it was found, and why reading had not found it

By measurement, with a new instrument: `atom/models/dspark_window_audit.py`,
off unless `ATOM_DSPARK_WINDOW_AUDIT=<report-every-N-steps>` is set.

`_dspark_index_kernel` declares a request's window to be `n_valid =
min(anchor+1, W)` rows ending at the anchor and gathers them by absolute
position **without checking that anything was ever stored there** — its own
docstring says "anything left unwritten shows the slot's previous occupant". So
`n_valid` is a *claim*. The audit measures the claim against the fact:

* `note_writes` stamps `stamp[slot, pos % ring_slots] = pos` for exactly the
  rows `write_context_kv` hands `swa_write`;
* `note_reads` re-derives the kernel's own window span and counts how many rows
  carry their own stamp.

This is `RESULTS.md` §7 item 1 — "write known values at known positions, gather
them back through the production read path, assert the identity; arm-local,
deterministic, no noise floor" — listed there as deferred four times in favour
of server A/Bs. It is an integer ratio inside one arm, so it needs no cross-arm
comparison, which is what makes it work where hidden-state A/Bs cannot: two
native runs of one prompt land 14% apart on `main_x` while emitting identical
text.

The output that named the defect (per request, newest stale row first):

```
anchor=9    stale (anchor-pos, pos wanted, pos found): [[0,   9,  -1]]
anchor=122                                             [[0, 122,  -1]]
anchor=136                                             [[0, 136,   1], [ 1, 135, 0]]
anchor=179                                             [[0, 179,  44], [44, 135, 0]]
```

Offset 0 — the anchor — stale on **100% of steps**.

### Two instrument errors worth not repeating

* **A fixed-height stamp table manufactured the finding it was looking for.**
  `state_slot_out` carries a PHYSICAL slot (`WindowParams.slot`: "a position in
  the plane, not a pool group"), routinely 1144–1155 against `--max-num-seqs
  512`. Clamping those into a 1024-row table aliased distinct requests onto one
  row and reported freshness collapsing 98% → 25% as the batch filled. That was
  entirely the instrument. With a table that grows, the real number is a flat
  98.4%. **An out-of-range index in a diagnostic is not a safe thing to clamp.**
* **A count is not a finding.** 126/128 fresh reads as benign rounding and is
  fatal: it is benign if the two missing rows are the window's oldest and fatal
  if they are its newest. Only dumping *which* rows made it readable.

## 3. Result

Probe regime, concurrency 32, eager, like-for-like at the busiest report
(~15k drafted in the window):

| | mean accepted | per-position |
|---|---:|---|
| before | 2.61 / 2.64 | 0.594 0.395 0.260 0.164 0.111 0.055 0.027 |
| after | **3.91** | 0.795 0.587 0.414 0.292 0.197 0.127 0.075 |

Window freshness 98.4% → **99.5%**, any-stamp 99.4% → **100%**.

**+1.27 accepted tokens against a deficit measured at 1.36**
(`SESSION-HANDOFF-2026-10-03.md` §3), and above the config-matched native
reference of **3.45–3.53** (`RESULTS.md` §9). On a long eager run (conc 32,
2048 output tokens) it reaches **5.32–5.48** and holds, with freshness flat at
99.3%.

The residual 0.5% is the already-known "rejected rows written at position 0"
(vLLM stores position 0 for the rejected suffix; ATOM addresses by position and
ignores the slot mapping that gates it). It costs ~1 row in 128 and
`RESULTS.md` §4.1 measured gating it as worth nothing.

### What this retires

`RESULTS.md` §4.1 lists "anchor position convention | `−1` measured 2.11 vs
2.14" as **eliminated**. That elimination is wrong — the change is worth +1.27.
The earlier attempt had no way to confirm it had the intended effect; the audit
does, and the offset-0 staleness going from 100% of steps to 0% is what
distinguishes "tried it" from "did it".

Lesson, matching the file's own method notes: an ablation with no instrument
confirming it took effect is not an elimination.

---

## 4. NEW, and separate: cudagraphs collapse long generation to the floor

Found while validating the fix in the production config. **It is not caused by
the fix** — the control is the parent commit, same probe, same config.

Concurrency 32, 2048 output tokens, `SpecDecoding metrics` in time order:

| arm | sequence of mean accepted |
|---|---|
| eager, **with** fix | 2.83 → 3.56 → 4.18 → 5.32 → 5.48 → 5.42 → 5.39 |
| cudagraph `FULL_AND_PIECEWISE`, **with** fix | 4.30 → 1.45 → 1.09 → **1.08** |
| cudagraph, **without** fix (control, `0fe840fbb`) | 2.23 → 2.67 → 1.09 → **1.08** |

1.08 is the no-window floor (`RESULTS.md` §3 measures the window-ablated draft
at 1.05–1.06). So under capture the window stops contributing entirely after a
few hundred decode steps, on both arms, pre-existing.

`kb/p1c` "cudagraphs exonerated" (2.376 vs 2.379) is not contradicted — it was
measured at short generation, where the collapse has not yet happened. The
collapse needs a few hundred steps to appear, which no previous probe ran.

This is now the arm's largest single defect: it caps the production
configuration at the floor exactly where the benchmark lives (ISL 115k / OSL
1k). It plausibly explains the 115k sweep's 1.83–2.74 as a blend of pre- and
post-collapse steps.

**Not yet diagnosed.** The audit cannot see it — Python inside the draft does
not run on a graph replay, which is the condition being investigated. Any probe
for it has to be device-side state, or PIECEWISE rather than FULL.

---

## 5. Still open after this

* §4 above — the cudagraph collapse. Blocks any production measurement.
* GSM8K with the fix (was 0.9413 plugin/ATOM draft vs 0.9507 plugin spec-off).
* The 115k goal regime with the fix. Do not sweep until §4 is closed: every
  point would be a blend of two regimes.
* Item B, DP attention (`SESSION-HANDOFF-2026-10-03.md` §4) — untouched.

## 6. Reproduce

```bash
# the fix, eager, with the instrument that found it
./stop.sh && rm -rf /root/.cache/atom/*
ATOM_DSPARK_WINDOW_AUDIT=25 bash serve_eager_debug.sh > logs/x.log 2>&1 &
python3 accept_probe.py --concurrency 32 --max-tokens 512
grep -E 'SpecDecoding metrics|cumulative fresh|stale rows' logs/x.log

# the cudagraph collapse (needs a LONG run -- short ones never reach it)
./stop.sh && rm -rf /root/.cache/atom/*
bash serve_dsv4_pro_vllm_atom_tp8.sh > logs/y.log 2>&1 &
python3 accept_probe.py --concurrency 32 --max-tokens 2048
grep 'SpecDecoding metrics' logs/y.log
```

Logs in this tree: `server-audit{,2,3,4}-plugin-eager.log` (instrument bring-up
and the two instrument errors), `server-anchorfix-plugin-eager.log`,
`server-anchorfix-long-eager.log`, `server-anchorfix-plugin-cudagraph.log`,
`server-control-prefix-cudagraph.log` (the pre-fix control).
