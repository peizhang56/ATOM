#!/usr/bin/env bash
# Benchmark client for DeepSeek-V4-Pro-0813 -- the requirements.txt workload.
# Server: ./serve_dsv4_pro_mtp_tp8.sh (port 8000). GPUS=8 to match it.
#
# From requirements.txt: --isl 115000 --osl 1000 --cache 90, scored at p90.
#
# The default 16,24,32 is the QUICK TURNAROUND. A FULL SWEEP is
# CONCURRENCY=16,24,32,64,128 -- each point at ISL 115k costs GB of prompt text
# and a lot of wall clock, so the low end runs by default and the rest is asked
# for. Report which of the two a number came from; never trim either to report
# a subset.
#
# The client is the instrument and is SETTLED -- do not tune it to meet a
# target. Raising --cache, shortening ISL or dropping a LOW point all improve
# numbers by doing less work. If a target is missed, the fix is server-side.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
API=${API:-http://127.0.0.1:8000}
API_KEY=${API_KEY:-EMPTY}
MODEL=${MODEL:-/data/DeepSeek-V4-Pro-0813}
GPUS=${GPUS:-8}
SEED=${SEED:-531150}
CONCURRENCY=${CONCURRENCY:-16,24,32}

# aiperf 0.12 routes a LOCAL path through snapshot_download() when
# HF_HUB_OFFLINE is set, and huggingface_hub rejects it as a repo id. Unsetting
# it opens no download path: transformers loads an existing directory directly.
export HF_HOME=${HF_HOME:-/data}
case "$MODEL" in /*) unset HF_HUB_OFFLINE ;; *) export HF_HUB_OFFLINE=1 ;; esac

SERIES=${SERIES:-dsv4pro_tp8_mtp_115k}
ARTIFACTS=${ARTIFACTS:-results_mi355x_${SERIES}}
TAG=${TAG:-sweep_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-$ROOT/$ARTIFACTS/$TAG}

[ -d "$ROOT/inference-benchmarking" ] || \
  git clone https://github.com/DO-FDE/inference-benchmarking.git "$ROOT/inference-benchmarking"

# Prompt text is written to disk twice per point (prompts.jsonl and aiperf's
# inputs.json, ~420 KB/request at ISL 115k) and one copy is never removed.
# Both regenerate from the seed in run_meta.json. Drop each point's copies as
# soon as its summary file proves aiperf is done with them.
purge() {
    for s in "$ROOT"/results*/*/concurrency_*/profile_c*.json; do
        [ -e "$s" ] && rm -f "${s%/*}/inputs.json" "${s%/*}/prompts.jsonl"
    done
}
purge_loop() { while :; do purge; sleep 20; done; }
purge_loop & PURGE_PID=$!
trap 'kill $PURGE_PID 2>/dev/null; purge' EXIT INT TERM

# WHICH ARM produced this directory. run_meta.json records the client knobs
# only -- no commit, no server config -- so a results dir could not otherwise
# be attributed. server_cmdline is /proc/<pid>/cmdline, the only witness that
# cannot disagree with what ran; report_md.py renders THAT rather than this
# repo's serve script, whose defaults an operator may have overridden.
mkdir -p "$OUT_DIR"
{
  printf '{\n  "written_at": "%s",\n' "$(date -Is)"
  for r in aiter:/app/aiter-test ATOM:/app/ATOM vllm:/app/vllm; do
    n=${r%%:*}; d=${r#*:}
    sha=$(git -C "$d" rev-parse HEAD 2>/dev/null || echo unknown)
    if [ -n "$(git -C "$d" status --porcelain --untracked-files=no 2>/dev/null)" ]
      then dirty=true; else dirty=false; fi
    printf '  "%s": {"sha": "%s", "dirty": %s},\n' "$n" "$sha" "$dirty"
  done
  printf '  "vllm_version": "%s",\n' \
    "$(VLLM_LOGGING_LEVEL=ERROR /opt/venv/bin/python -c 'import vllm;print(vllm.__version__)' 2>/dev/null | tail -1)"
  # || true: pgrep exits 1 on no match, and pipefail + set -e would abort here.
  srv=$(pgrep -f 'vllm serv[e]|atom\.entrypoints\.openai_server' 2>/dev/null | head -1 || true)
  printf '  "server_pid": %s,\n  "server_cmdline": [' "${srv:-null}"
  if [ -n "$srv" ] && [ -r "/proc/$srv/cmdline" ]; then
    tr '\0' '\n' < "/proc/$srv/cmdline" \
      | awk 'NR>1{printf ", "} {gsub(/\\/,"\\\\"); gsub(/"/,"\\\""); printf "\"%s\"", $0}'
  fi
  printf '],\n  "server_env": {'
  if [ -n "$srv" ] && [ -r "/proc/$srv/environ" ]; then
    tr '\0' '\n' < "/proc/$srv/environ" \
      | grep -E '^(ATOM|AITER|VLLM|HIP|ROCR|PYTORCH)_[A-Z0-9_]*=' | sort \
      | awk -F= 'NR>1{printf ", "} {k=$1; sub(/^[^=]*=/,""); gsub(/"/,"\\\""); printf "\"%s\": \"%s\"", k, $0}'
  fi
  printf '}\n}\n'
} > "$OUT_DIR/arm.json"

echo "=== $OUT_DIR"
cd "$ROOT/inference-benchmarking/gsm8k-aiperf-pack"

./run_gsm8k_benchmark.sh \
  --url "$API" --model "$MODEL" --tokenizer "$MODEL" --api-key "$API_KEY" \
  --concurrency "$CONCURRENCY" \
  --isl 115000 --osl 1000 --cache 90 \
  --seed "$SEED" --gpus "$GPUS" \
  --out-dir "$OUT_DIR" "$@"
