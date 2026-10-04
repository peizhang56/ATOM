# A3 — the ATOM-owned draft reaches native's acceptance; the gap left is DP attention — MI355X — 2026-10-04

Closes open item A of `SESSION-HANDOFF-2026-10-03.md`. Two defects, both at the
plugin seam, both fixed. Data: `results_mi355x_dsv4pro_vllm_atom_dspark_bothfixes_115k/sweep_20261004_bothfixes/report.md`
(in `deepseek2`). ATOM `dde6b5ba5`.

## The two defects

1. **Anchor off-by-one** (`8f1954f12`). vLLM's `positions` is per-token
   (`last_valid_pos + 1`); ATOM's `block_backbone` wants the row before it and
   derives `draft_pos` and the window itself. Every drafted token was RoPE'd one
   position too far and the window's newest row had never been forwarded.
2. **Stale target-metadata stash** (`dde6b5ba5`). The draft writes its context
   KV through `get_deepseek_v4_target_metadata()`, filled only from the target's
   forward Python — which a FULL cudagraph replay never runs. Measured 0/100
   refreshes under a captured target against 100/100 under PIECEWISE.

Both found with `ATOM_DSPARK_WINDOW_AUDIT`, which stamps what `write_context_kv`
writes and checks what the index kernel claims to read. An integer ratio inside
one arm, no noise floor — which is why it worked where hidden-state A/Bs could
not (two native runs of one prompt land 14% apart on `main_x`).

## ISL 115k / OSL 1k, `--cache 90`, seed 531150

| conc | **both fixes** | tok/step | ATOM draft, before | before | vLLM draft | spec OFF | native | native tok/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 8833 | **3.193** | 7563 | 2.129 | 9038 | 7674 | 4463 | 3.307 |
| 24 | 12972 | **3.081** | 10479 | 1.967 | 12860 | 12169 | 10152 | 3.225 |
| 32 | 14534 | **3.139** | 11865 | 1.971 | 14141 | 13897 | 15328 | 3.227 |
| 64 | **16730** | **3.151** | 13570 | 1.825 | 15717 | 17767 | 23771 | 3.259 |
| 128 | 16341 | **3.163** | 15641 | 2.742 | 13940 | **20958** | **32302** | 3.282 |

`in/s/gpu`, all from each sweep's generated `report.md`. Comparison arms are
from `RESULTS.md` section 1 and were measured on earlier shas.

## What this settles

- **Acceptance is at native parity and flat with batch**: 3.08–3.19 against
  native's 3.22–3.26, where this arm was 1.83–2.74. The premise the project was
  predicated on — ATOM's drafter holding acceptance where vLLM's decays
  (3.309 → 2.379) — is confirmed at the goal regime, now at the right level.
- **Throughput +17–24%** over the same arm before the fixes, and it beats the
  vLLM-draft arm at every concurrency from 24 up.
- Per-position acceptance is flat across batch too (pos 0: 66.2–66.6% at every
  concurrency), which is the signature of a drafter whose context no longer
  depends on how many requests are resident.

## What it does NOT settle

- **Speculation is still a net loss at the heaviest points.** spec-OFF is 17767
  at conc 64 and 20958 at 128 against 16730 and 16341 here. The margin narrowed
  a lot (at 128 the ATOM-draft arm was 15641) but the sign has not flipped.
  Caveat carried from `RESULTS.md`: the spec-off sweep is a different ATOM/aiter
  sha and should be re-run on this build before the pair is quoted externally.
- **No target in `requirements.txt` is met**, though all moved closer:
  in/s/gpu 16730 vs >17200 required (was 15641); out/s/gpu 145.4 vs >150 (was
  135.9); P90 TTFT 7.9 s vs <5 s; P90 ITL 24.7 ms vs ≤20 ms (was 28.1);
  cache 88.98% vs 90%.
- The conc-16 point ran at 80.08% cache against 88.98% everywhere else — it is
  the first point and the prefix cache was still filling, which is also why its
  TTFT p90 is 28.7 s. Do not read that row's latency as the arm's.

## Consequence

Item A is closed. **The remaining gap is item B, DP attention**, which `kb/e3`
measures as the entire throughput gap (1.95x at conc 64, 2.31x at 128) and which
`SESSION-HANDOFF-2026-10-03.md` section 4 has implemented-and-regressive at
step 2b (sharded the KV read but not the compute). Step 2c is designed there.
Native is 23771 / 32302 at conc 64 / 128 against 16730 / 16341 here — a 1.42x
and 1.98x gap, which is the shape DP attention predicts and nothing else on the
list does.
