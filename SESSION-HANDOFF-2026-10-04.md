# Session hand-off — 2026-10-04

Supersedes `SESSION-HANDOFF-2026-10-03.md`, which is still correct for its DP
attention work (§4 there) and its method lessons. Read this one first.

**Open item A is CLOSED.** The ATOM-owned DSpark draft now holds native's
acceptance at the goal regime. What is left is item B, DP attention.

---

## 0. State

| | |
|---|---|
| code | `ds_v4_atom_vllm_dspark_draft_t7fix` @ **`dde6b5ba5`**, clean, pushed |
| worklog | `ds_v4_dspark_worklog` @ **`f7b22bf4`**, clean, pushed |
| `/app/aiter-test` | `efa76be1fb` (branch `ds_v4_atom_vllm`) |
| `/app/vllm` | `281cfd5a09` (runtime copy is `/opt/venv/.../vllm`) |

The branch was rebased onto a recent ATOM `main` before this session, so the
shas in `SESSION-HANDOFF-2026-10-03.md` §0 no longer resolve. Four commits added
here, **none gated off — these are live by default**:

```
dde6b5ba5  fix: refresh the DSpark target-metadata stash from the builder
bb3dc2f7b  diag: root-cause the cudagraph collapse to a stale stash
34ca39c4f  diag: instrument the draft's window under cudagraph capture
8f1954f12  fix: translate vLLM's anchor position into ATOM's convention
```

DP attention (`0fe840fbb` and below) is unchanged and still gated OFF behind
`ATOM_VLLM_DP_ATTENTION=1`. Leave it off.

**`restore_dsv4.sh` still defaults to `BRANCH=ds_v4_atom_vllm`** — the OLD,
half-finished arm. Always:

```bash
BRANCH=ds_v4_atom_vllm_dspark_draft_t7fix ./restore_dsv4.sh
```

**The node's shared volume is full.** `/home/pzhang12` is a 10 T volume at 100%
with ~17 G free and moving; I hit `Disk quota exceeded` mid-session and moved
all server logs to `/tmp/dsv4logs/`. Your home is only ~6.5 G, so it is not
you. `/tmp` (28 T overlay, 59% used) is the place for logs and raw sweep
artifacts.

---

## 1. What was wrong, and it was two bugs at one seam

Both are the plugin handing ATOM's draft the wrong per-step state. Neither is in
ATOM's DSpark model, and neither is in vLLM's speculator.

### 1.1 Anchor off-by-one (`8f1954f12`)

| | what `positions` means |
|---|---|
| vLLM | per-TOKEN. `query_pos = last_valid_pos + 1 + query_off`, so column 0 is the position of the anchor TOKEN |
| ATOM | the position of the row whose logits produced that token, i.e. `last_valid_pos`. `_build_block_plan` then derives `draft_pos = anchor+1 .. anchor+T` and the window `[anchor-W+1, anchor]` |

`DeepseekV4DSparkDraft.forward` passed vLLM's number through untranslated, so
every drafted token was RoPE'd one position too far **and** the window's newest
row — the anchor, the row the next token depends on most — was a position the
target had never forwarded. The fix is `positions[anchor_idx] - 1`.

### 1.2 The target's cudagraph replay left the draft's metadata stash stale (`dde6b5ba5`)

The draft writes its context KV through `get_deepseek_v4_target_metadata()` —
the target batch's `cu_seqlens_q` spans and per-request `state_slot_out`. That
stash was filled **only** from `atom_deepseek_v4_forward_context`, i.e. the
target's forward Python, which a FULL cudagraph replay never runs. Its own
docstring asserted the opposite:

> "replaced every target step ... the drafter never sees a stale one"

So under a captured target, every drafted step wrote its context rows through
some earlier step's spans and slots — in practice the last eager forward,
usually a prefill, whose metadata tensors are fresh allocations rather than the
persistent decode buffers. Fixed by re-stashing from
`AtomDeepseekV4ProxyMetadataBuilder.build`, which runs every step including
replays.

**This one is the reason the production config looked broken.** It is the
TARGET's capture, not the draft's: forcing the draft eager under a captured
target changed nothing (1232 tok/s / 1.67 against 1196 / 1.71).

---

## 2. Results

### 2.1 The goal regime — ISL 115k / OSL 1k / `--cache 90`, seed 531150

`results_mi355x_dsv4pro_vllm_atom_dspark_bothfixes_115k/sweep_20261004_bothfixes/report.md`.
All `in/s/gpu` from each sweep's generated `report.md` — **`report.txt` computes
it differently, never mix them.**

| conc | **both fixes** | tok/step | this arm before | before | vLLM draft | spec OFF | native | native |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 8833 | **3.193** | 7563 | 2.129 | 9038 | 7674 | 4463 | 3.307 |
| 24 | 12972 | **3.081** | 10479 | 1.967 | 12860 | 12169 | 10152 | 3.225 |
| 32 | 14534 | **3.139** | 11865 | 1.971 | 14141 | 13897 | 15328 | 3.227 |
| 64 | **16730** | **3.151** | 13570 | 1.825 | 15717 | 17767 | 23771 | 3.259 |
| 128 | 16341 | **3.163** | 15641 | 2.742 | 13940 | **20958** | **32302** | 3.282 |

Comparison arms are from `RESULTS.md` §1 and were measured on earlier shas.

- **Acceptance is at native parity and flat with batch** — 3.08–3.19 against
  native's 3.22–3.26, where this arm was 1.83–2.74. Per-position pos-0 is
  66.2–66.6% at *every* concurrency, the signature of a drafter whose context no
  longer depends on how many requests are resident.
- Throughput **+17–24%**, and it beats the vLLM-draft arm at every concurrency
  from 24 up.
- conc 16 ran at 80.08% cache while the prefix cache was still filling (every
  other point is 88.98%), which is also why its TTFT p90 is 28.7 s. Do not read
  that row's latency as the arm's.

### 2.2 Correctness

GSM8K, full 1319, 5-shot, strict-match: **0.9477 ± 0.0061**, against the 0.9507
spec-off arm — 0.0030 apart, inside noise. Speculation is lossless here; the gap
was 0.0094 (~1.5σ) before. flexible-extract 0.9401, close enough that extraction
is sound.

**Caveat:** 0.9507 (spec-off) and 0.9515 (native) are carried from `RESULTS.md`
§6 and were measured on different shas. The spec-off leg needs re-running on
this build before the pair is quoted externally.

### 2.3 Probe regime, for cheap iteration

`accept_probe.py` / `fixed_batch_probe.py`, conc 32, 512 output tokens, batch
pinned with `ignore_eos`, production `FULL_AND_PIECEWISE`:

| | stash fresh | window fresh | accepted | tok/s |
|---|---:|---:|---:|---:|
| before both fixes | 0/100 | 69–92% | 1.71 | 1196 |
| **after** | **100/100** | **99.4%** | **4.01** | **2482** |

The long-run collapse is gone: conc 32 at 2048 output tokens holds 4.21/4.27
where it went `4.19 -> 1.10 -> 1.10 -> 1.10`.

---

## 3. The instrument, and use it before believing anything here

`atom/models/dspark_window_audit.py`, off unless
`ATOM_DSPARK_WINDOW_AUDIT=<report-every-N-steps>`.

`_dspark_index_kernel` declares a request's window to be
`n_valid = min(anchor+1, W)` rows and gathers them by absolute position
**without checking anything was ever stored there** — its own docstring says
"anything left unwritten shows the slot's previous occupant". So `n_valid` is a
*claim*. The audit stamps every row `write_context_kv` hands `swa_write`, then
counts how many of the claimed rows carry the position the draft reads them as.

**It is an integer ratio inside one arm, with no numerical noise floor.** That
is why it worked where hidden-state A/Bs could not: two native runs of one
prompt land 14% apart on `main_x` while emitting identical text. It is also the
acceptance test for any fix in this area — `fresh` back at ~99.4% is the pass
condition.

Also provides `note_metadata_pointer`, `note_build`, `note_alloc_site`,
`note_target_stash` / `note_stash_age`, which between them eliminated six
candidate causes (see §5).

**Run it with eager or PIECEWISE cudagraphs where possible** — under a FULL
replay the draft's Python does not run, so `note_reads` is silent. `note_writes`
and the stash counters still work.

---

## 4. What is open

### 4.1 Item B — DP attention. This is the whole remaining gap.

Native is 23771 / 32302 at conc 64 / 128 against 16730 / 16341 here — **1.42×
and 1.98×**. That is the shape `kb/e3` measures for DP attention (1.95× at 64,
2.31× at 128) and nothing else on the list produces it.

Implemented through step 2b and **regressive** (−35…−38% at 115k) because it
shards the KV *read* but not the *compute*: with replicated weights every rank
runs Q/K/V/O for all heads on all tokens and `_dp_combine` throws seven eighths
away. Step 2c is designed in `SESSION-HANDOFF-2026-10-03.md` §4 — build
owned-subset metadata in the bridge rather than masking a full-width build,
slice `x`/`positions` to owned rows, scatter back and all-reduce, and **keep KV
writes at full width** (a rank holding only its own requests' blocks would miss
a prefix-cache hit, which is how `kb/e1` fragmented the cache to 38.9%).

Read `kb/e1`, `kb/e2`, `kb/e3` before touching it. Do **not** port TBO (`kb/e3`:
a net 4–8% loss as configured). Do **not** re-run vLLM's own `--data-parallel-size
8 --enable-expert-parallel` (`kb/e1`: 2.0–4.7× WORSE).

### 4.2 Speculation is still a net loss at the heaviest points

spec-OFF is 17767 @64 and 20958 @128 against 16730 and 16341. The margin
narrowed sharply (at 128 this arm was 15641) but the sign has not flipped.
Re-run the spec-off leg on this build before concluding anything — the one on
record is a different ATOM/aiter sha.

### 4.3 The draft's OWN replay safety is untested

Until the stash was fixed, the stale window dominated and would have hidden a
second defect. The draft's cudagraph manager takes `FULL_DECODE_ONLY` whenever
the attention backend claims uniform-batch support. Re-check now that the window
is clean — the freshness audit runs under a replayed target, so the test exists.

Note there is **no capture bucket above B=71** other than 512, so concurrency
128 runs the draft eagerly. That is why conc 128 scored 4.11–4.16 while conc 32
sat at 1.71 and looked like "FULL is fine at large batch" for a while.

### 4.4 Smaller

- No `requirements.txt` target is met, though all moved closer: in/s/gpu 16730
  vs >17200 (was 15641); out/s/gpu 145.4 vs >150 (was 135.9); P90 ITL 24.7 ms
  vs ≤20 (was 28.1); P90 TTFT 7.9 s vs <5 s; cache 88.98% vs 90%.
- The audit's position stamp is one-sided: a previous occupant of the same ring
  slot that wrote the SAME positions reads as fresh. Below 100% is a hard
  finding; 100% is not a full exoneration.
- Raw sweep artifacts (542 MB) live only in `/tmp/dsv4logs/sweep_bothfixes` and
  die with the node. The committed directory carries `report.md`, `report.json`,
  `arm.json`, `economics.*` and per-point `profile_c*.json` (225 K) — enough to
  re-read, not enough to regenerate `report.md`.
- `.gitignore`'s `result*/**` silently swallows sweep artifacts. `git add -f`.

---

## 5. Eliminated — do not re-investigate

Everything here is measured, not reasoned.

| candidate | evidence |
|---|---|
| stale `state_slot_out` address under replay | address is stable AND equal to the one the draft baked at capture |
| draft metadata not rebuilt under replay | both decode buffer sets build every step (750 vs 746) |
| `DSparkIndexBuffers` in cudagraph-pool memory | allocated with capture closed (the manager's eager warm-up call) |
| `is_dummy_run` baking the all-zero window at capture | `use_fp8` is true at capture; the fp8 branch runs there |
| non-restorative `mask_pad_tail` | it restores its prefix as well as marking its tail |
| aux hidden-state values frozen under replay | aggregate `\|mean\|`/std identical to the PIECEWISE control |
| the draft's OWN cudagraph replay (as cause of the collapse) | forcing it eager under a captured target changed nothing |

Plus everything in `SESSION-HANDOFF-2026-10-03.md` §3, with one **correction**:

> `RESULTS.md` §4.1 lists "anchor position convention `−1` measured 2.11 vs
> 2.14" as eliminated. **That elimination was wrong** — the change is worth
> +1.27 accepted tokens. The earlier attempt had no instrument confirming it
> took effect.

---

## 6. Method lessons from this session

- **An ablation with no instrument confirming it landed is not an elimination.**
  The anchor fix had been "tried" and written off; what made it stick was
  watching offset-0 staleness go from 100% of steps to 0%.
- **A count is not a finding.** 126/128 fresh rows reads as rounding error and
  was the entire defect — benign if the two missing rows are the window's
  oldest, fatal because they were its newest. Dump *which* rows.
- **Do not clamp an out-of-range index in a diagnostic.** A fixed-height stamp
  table aliased distinct requests onto one row and manufactured a 98%→25%
  collapse that did not exist. `state_slot_out` carries a PHYSICAL slot, which
  is routinely far above `max_num_seqs`.
- **Run the A-vs-A control before believing any A-vs-B** (again). An aggregate
  activation statistic looked damning until the PIECEWISE control came back
  identical.
- **Bound a diagnostic by its condition, not by call count** (again).
- A device→host sync inside a possibly-captured region is an illegal-capture
  crash; `speculator.capture()` calls `precompute_and_store_context_kv` with a
  capture open. Guard with `torch.cuda.is_current_stream_capturing()`.
- **A stable pointer does not mean a refreshed buffer.** The stash's tensors
  never moved; nothing was writing them.
- **Short probes hid this for days.** The collapse needs a few hundred decode
  steps to appear. Use `fixed_batch_probe.py` (`ignore_eos` + `min_tokens`) so
  the running batch is one number for the whole run — a draining batch cannot
  distinguish "decays with decode position" from "breaks when the capture bucket
  changes".

---

## 7. Recommendation

Item A is done and verified at the goal regime. **Do item B, step 2c** — it is
the only thing left that accounts for the residual, it is already designed, and
parity needs it. Before that, two cheap things worth an hour each:

1. Re-run **spec-OFF on this build** so §4.2's comparison is sha-matched.
2. Re-check the **draft's own replay safety** (§4.3) with the freshness audit,
   now that the stash no longer masks it.

Do not start Stage 5 (productionize, `/app/ATOM/.claude/commands/add-atom-vllm-model.md`)
until item B lands or is explicitly dropped — the recipe and accuracy table
should describe the arm that ships.
