---
name: benchmark-report
description: Turn a run_gsm8k_benchmark.sh results directory into a short standalone report.md (metrics table + the raw server/client commands). Use when asked to write up, summarise, or report a DeepSeek/aiperf benchmark run.
---

# Benchmark report

Run the script. Do not read the results directory yourself — the artifacts are
hundreds of MB and reading them wastes context.

```bash
python3 .claude/skills/benchmark-report/report_md.py RESULTS_DIR
```

Writes `RESULTS_DIR/report.md` and prints it. Options: `-o OUT`, `--gpus N`
(default from `run_meta.json`), `--title T`, `--note TEXT` (repeatable, one
short caveat per use — see Rules; most runs need none),
`--server-script` / `--client-script`.

The header line carries model, node, GPU count, ISL/OSL, `--cache` and seed.
The node comes from the `results_<node>_<series>` directory name, so it is
never passed in and never belongs in a `--note`.

**Beneath it, the build: aiter / ATOM / vLLM shas**, read from `arm.json`,
which the client wrote from `/app/aiter-test`, `/app/ATOM` and `/app/vllm`
before the run. A repo with uncommitted changes is marked **dirty** — aiter
and ATOM are `pip install -e`'d, so an edit in either worktree is already in
the measurement and the sha alone would misdescribe it. Never put a sha in a
`--note`; the line is generated.

The wrapper scripts are found in `RESULTS_DIR/../..` — which is why a results
directory must be `<root>/<series>/<run>/`. With more than one `serve_*.sh`
there, the script picks by explicit flag, then a unique match, then the one
whose resolved command names the model in `run_meta.json`, then the one whose
**entrypoint** matches `arm.json`'s `server_cmdline` — which is what separates
the two recipes for this checkpoint, `vllm serve` and
`python -m atom.entrypoints.openai_server`, since both name the same model.
**If it is still ambiguous it exits** rather than render some other arm's
server block under these numbers. Pass `--server-script` then.

Output: one metrics table, plus the resolved server and client commands so the
report reproduces the run on its own. A run with `--speculative-config` also
gets `accept%` / `tok/step` columns and a per-position table, driven by
`vllm:spec_decode_*` (or `atom:mtp_*` on the native ATOM entrypoint), which
exist only when spec decode is on — a plain run is unchanged rather than
showing zeros.

**Latency is scored at p90.** `requirements.txt` sets P90 TTFT < 5 s and
P90 ITL ≤ 20 ms, so the table leads each pair with p90 and keeps p50 beside it.
Quote p90 as the result; the p50/p90 gap is diagnosis, not the score.

**`ITL p90` is a percentile over requests, not over tokens.** aiperf emits one
`inter_token_latency` per request — already a mean over that request's decode —
so p90 is the 90th-percentile *request's average* gap. It does not bound any
single stall, and the worst token gap in a run is larger than it. `TTFT p90`
has the same population and the same caveat.

**The server block is corrected against `arm.json`'s `server_cmdline`**, which
the client reads from the live server's `/proc/<pid>/cmdline` — so a run that
overrode a `${VAR:-default}` reports what actually ran, not the script's
default. The report says when it made such a correction.

## Rules

- **The report is data and commands. No interpretation.** No comparison against
  targets, no A/B against another run, no verdict on whether a number is good.
  That reading belongs in `kb/`, where it can be revised without touching the
  measurement. Do not hand-edit generated prose into `report.md` either.
- **Keep the report short.** A normal report is the header line, the table, and
  the two command blocks. That is the finished product, not a stripped-down one.
- **`--note` defaults to none.** Reach for it only when the table is *wrong*
  without it: a crashed or short point, a non-standard client, the software
  provenance an A/B turns on.
- **Never note what the report already carries.** Node, GPU count, model,
  ISL/OSL, `--cache` and seed are in the header; the concurrency list is in the
  client block and `run_meta.json`.
- Points whose records lack `input_sequence_length` are dropped, so a crashed
  point silently shrinks or disappears. Check `reqs` against the concurrency
  and flag any shortfall with `--note`.
- A number is a fact about one server config. Reporting two configs means two
  reports, not two sections in one.
