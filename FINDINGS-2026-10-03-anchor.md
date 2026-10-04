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

## 4. SECOND DEFECT, found and FIXED: a captured TARGET starved the draft's window

Found while validating the fix in the production config, confirmed pre-existing
against the parent commit, and now **root-caused**.

### The defect

The draft writes its context KV through `get_deepseek_v4_target_metadata()` --
the target batch's `cu_seqlens_q` spans and per-request `state_slot_out`. That
stash is filled by `remember_deepseek_v4_target_metadata`, called from
`atom_deepseek_v4_forward_context`, i.e. **from the target's forward Python**.
A FULL cudagraph replay of the target does not run it. The helper's own
docstring states the assumption this breaks:

> "It is replaced every target step, and the speculator always runs within the
> step that set it, so the drafter never sees a stale one."

Measured, concurrency 32, 512 output tokens, batch pinned with `ignore_eos`:

| target mode | stash refreshed | window freshness | accepted |
|---|---:|---:|---:|
| PIECEWISE | **100/100** | 99.4-99.7% | **3.86** |
| FULL | **0/100** | 69-92% and falling | **1.67** |

So every drafted step under a captured target writes its context rows through
some earlier step's spans and ring slots. The intended rows are never written
and the window keeps the previous lap -- the audit's stale rows differ from the
position wanted by **exact multiples of `ring_slots = 135`** (405, 270, 135),
which is precisely a ring row that was never rewritten:

```
anchor=564  wanted 564 found 159   (delta 405 = 3x135)
            wanted 561 found 291   (delta 270 = 2x135)
            wanted 559 found 424   (delta 135 = 1x135)
```

Steady state decays to 1.08 accepted with per-position
`0.057/0/0/0/0/0/0` -- the exact shape of the window-ablated draft. The replay
is not degrading the draft; it is removing its context.

### It is the TARGET's capture, not the draft's

Measured both ways with the draft's own cudagraph manager forced to NONE:

| | conc 32 |
|---|---|
| target FULL + draft **replayed** | 1196 tok/s, 1.71 |
| target FULL + draft **eager** | 1232 tok/s, 1.67 |
| target PIECEWISE + draft eager | 544 tok/s, 3.86 |

Indistinguishable in the first two. An interim patch forcing the draft eager
was written on the theory that the draft's replay was at fault; the theory is
wrong and the patch was removed rather than left in. **Whether the draft's
block pass is itself replay-safe is now untested** -- the stale stash dominates
and would hide a second defect.

A detail that confused this for a while: there is no capture bucket above
B=71 (other than 512), so concurrency 128 runs the draft eagerly *and* exceeds
the target's decode capture, which is why it scores 4.11-4.16 and looked like
"FULL is fine at large batch".

### Eliminated, all by measurement with `ATOM_DSPARK_WINDOW_AUDIT`

| candidate | evidence |
|---|---|
| stale `state_slot_out` address | stable, and equal to the address the draft baked at capture |
| draft metadata not rebuilt | both decode buffer sets build every step (750 vs 746) |
| `DSparkIndexBuffers` in graph-pool memory | allocated with capture closed (manager warm-up call) |
| `is_dummy_run` all-zero window baked at capture | `use_fp8` is true at capture; `DRAFT-READS` fires there |
| non-restorative `mask_pad_tail` | it restores its prefix as well as marking its tail |
| aux hidden-state values frozen | aggregate `|mean|`/std identical to the PIECEWISE control -- the statistic does not discriminate, which is why the control was run |

### The fix (`dde6b5ba5`)

Re-stash from `AtomDeepseekV4ProxyMetadataBuilder.build`, which runs on every
step including replays -- which is the whole reason vLLM rebuilds it ("so that
any attention metadata builder state is updated"). Scoped to the target's proxy
layer, because the draft's own group builds through here too and its metadata
describes `[num_reqs x T]` block rows rather than the target's ragged batch.

Only `scheduled_bs` is step-dependent for this consumer -- `write_context_kv`
takes its `positions` as an argument, not off the context -- so the paired
`Context` is carried forward with that one field corrected rather than rebuilt.
Reconstructing a whole `Context` here would duplicate
`atom_deepseek_v4_forward_context`'s shape decisions in a second place, and
those two drifting apart is a worse bug than the one being fixed.

### Result, production config (`FULL_AND_PIECEWISE`), concurrency 32

| | stash fresh | window fresh | accepted | tok/s |
|---|---:|---:|---:|---:|
| before | 0/100 | 69-92% | 1.71 | 1196 |
| **after** | **100/100** | **99.4%** | **4.01** | **2482** |

The long-run collapse is gone: concurrency 32 at 2048 output tokens holds
**4.21 / 4.27** where it went `4.19 -> 1.10 -> 1.10 -> 1.10`, finishing in
22.1s against 45.3s. Concurrency 128: **4.14** accepted, **2560 tok/s**.

**2.08x throughput and 2.3x acceptance, in the configuration the benchmark
runs.** The window freshness ratio was the acceptance test -- an integer with
no noise floor.

## 5. Correctness: speculation is now lossless in the plugin

GSM8K, full 1319, 5-shot, with BOTH fixes
(`/tmp/dsv4logs/gsm8k_bothfixes`, strict-match):

| arm | GSM8K |
|---|---:|
| native | 0.9515 |
| plugin, spec OFF | 0.9507 |
| **plugin, ATOM draft, both fixes** | **0.9477 ± 0.0061** |
| plugin, ATOM draft, before | 0.9413 |
| plugin, vLLM draft | 0.9262 |

0.9477 against the 0.9507 spec-off arm is **0.0030, with a stderr of 0.0061** --
inside noise, i.e. speculative decoding is lossless here. Before the fixes the
gap was 0.0094 (~1.5 stderr). flexible-extract came in at 0.9401, close enough
to strict-match that extraction is sound (the harness's own criterion).

Note this does NOT re-derive the native number on this build; 0.9515 and 0.9507
are carried from `RESULTS.md` section 6 and were measured on different shas.
The comparison that matters -- spec-on against spec-off within the plugin --
still needs its spec-off leg re-run on this build before it is quoted
externally.

## 6. Still open after this

* **The draft's OWN replay safety is untested.** Until the stash was fixed the
  stale window dominated and would have hidden a second defect. The freshness
  audit now runs under a replayed target, so the test exists.
* The 115k goal regime with both fixes -- running. `RESULTS.md` section 1's
  "speculation is a net LOSS at concurrency 64/128" was measured with both
  defects present and should not be believed until it is re-run.
* Item B, DP attention (`SESSION-HANDOFF-2026-10-03.md` section 4) -- untouched,
  and still the larger throughput lever.

## 7. Reproduce

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
