#!/usr/bin/env bash
# Stop the running server (vllm or atom) and any aiperf client, then confirm the
# GPUs are actually free. Works for both recipes.
#
#   ./stop.sh              # port 8000 (serve.sh, serve_atom_native.sh)
#   PORT=8001 ./stop.sh    # serve_atom_vllm_recipe.sh
#
# Why a VRAM check and not just a port check: the HTTP socket belongs to the
# parent, so `curl` starts failing the instant the parent dies -- while eight TP
# workers are still tearing down ~288 GB each. Exiting there reports success
# over a still-pinned GPU, which is the "half-dead mp executor" CLAUDE.md warns
# makes the next run's numbers silently worse.
#
# Why the ATOM children are not in the kill pattern's critical path: ATOM arms
# prctl(PR_SET_PDEATHSIG, SIGKILL) in both child entrypoints
# (model_engine/engine_core.py:193, model_engine/async_proc.py:84, helper at
# utils/__init__.py:229), so killing the parent makes the kernel reap
# ATOM::EngineCore, whose death reaps ATOM::TP0-7. They are matched explicitly
# anyway -- belt and braces, and it lets the loop below *verify* they are gone
# rather than assume the cascade fired.
set -uo pipefail

PORT="${PORT:-8000}"
VRAM_FREE_PCT="${VRAM_FREE_PCT:-5}"   # per-GPU allocated %% considered "released"

# Every PID from this shell up to init. `pkill -f` matches on the full command
# line, and both patterns below appear in paths we routinely invoke ourselves
# (.venv-aiperf, gsm8k-aiperf-pack, or a shell whose argv mentions atom), so a
# naive pkill can SIGKILL its own caller. Filter the ancestry out by PID instead
# of by string -- string guards like `grep -v shell-snapshots` only cover the
# one launcher that happens to leave that in argv.
ancestry() {
  local p=$$
  while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    echo "$p"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
}
SELF=$(ancestry | sort -u)

# pgrep -f "$1", minus our own ancestry.
match() {
  pgrep -f "$1" 2>/dev/null | sort -u | comm -23 - <(echo "$SELF") | tr '\n' ' '
}

reap() {   # $1 = label, $2 = pattern
  local label=$1 pat=$2 pids
  pids=$(match "$pat")
  [ -z "${pids// /}" ] && { echo "  no $label processes"; return 0; }

  echo "  $label: SIGTERM $pids"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null
  for _ in $(seq 1 15); do
    pids=$(match "$pat")
    [ -z "${pids// /}" ] && { echo "  $label: exited cleanly"; return 0; }
    sleep 1
  done

  echo "  $label: still up after 15s, SIGKILL $pids"
  # shellcheck disable=SC2086
  kill -KILL $pids 2>/dev/null
  sleep 3
  pids=$(match "$pat")
  [ -n "${pids// /}" ] && echo "  WARNING: $label survived SIGKILL: $pids" >&2
  return 0
}

# `bin/aiperf` rather than bare `aiperf`: the bare string also matches the
# repo's own .venv-aiperf and gsm8k-aiperf-pack paths, i.e. the benchmark
# driver's shell and anything sourcing the venv.
#
# `^aiperf ` is the second half and is not optional: aiperf's children rewrite
# their own argv to `aiperf system_controller`, `aiperf worker_<id>`,
# `aiperf dataset_manager` and so on -- no path, so `bin/aiperf` never matches
# them. One survivor is enough to keep an NFS silly-rename (.nfs*) alive in the
# results dir, which then cannot be deleted and poisons the next run.
echo "Stopping aiperf clients..."
reap "aiperf" "bin/aiperf|aiperf profile|^aiperf "

echo "Stopping server..."
reap "server" "vllm serve|VLLM::|ATOM::|atom.entrypoints|openai_server"

# HTTP socket gone?
for _ in $(seq 1 30); do
  curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 || break
  sleep 2
done
if curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "WARNING: something is still responding on port ${PORT}." >&2
  exit 1
fi
echo "Port ${PORT} free."

# VRAM released? This is the check that actually matters before a relaunch.
if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "NOTE: rocm-smi not found; VRAM release not verified." >&2
  exit 0
fi
busy_gpus() {
  rocm-smi --showmemuse --csv 2>/dev/null \
    | awk -F, -v t="$VRAM_FREE_PCT" 'NR>1 && $2+0 > t {printf "%s:%s%% ", $1, $2}'
}

for _ in $(seq 1 30); do
  BUSY=$(busy_gpus)
  [ -z "$BUSY" ] && { echo "VRAM released on all GPUs. Server stopped."; exit 0; }
  sleep 2
done

# Last resort: kill whoever still holds a /dev/kfd fd.
#
# `reap` above can report "exited cleanly" and leave the GPUs pinned anyway.
# Seen 2026-09-10: an engine hung at an NCCL barrier with a 1.2 TB pinned CPU
# tier: the kernel's reclaim stalled every /proc read, so `pgrep` returned
# nothing, `pkill` matched nothing, and `match()` saw an empty list -- while 18
# ATOM::TP* orphans still held 280 GB of VRAM each. A name-based sweep cannot
# see those; an fd-based one can, because holding the GPU *is* holding the fd.
#
# Ancestry is excluded for the same reason it is in `match`: this shell's own
# children (rocm-smi, awk) do not hold a kfd fd, but a launcher above us might.
echo "VRAM still allocated after 60s ($BUSY) -- sweeping /dev/kfd holders..." >&2
n=0
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  case " $SELF " in *" $p "*) continue ;; esac
  if timeout 2 ls -l "/proc/$p/fd" 2>/dev/null | grep -q kfd; then
    echo "  kfd holder: $p $(cat "/proc/$p/comm" 2>/dev/null)" >&2
    kill -KILL "$p" 2>/dev/null && n=$((n + 1))
  fi
done
echo "  killed $n kfd holder(s)" >&2

# VRAM drains for 60-120s after the last holder dies, so this waits longer than
# the loop above rather than re-checking immediately.
for _ in $(seq 1 60); do
  BUSY=$(busy_gpus)
  [ -z "$BUSY" ] && { echo "VRAM released after kfd sweep. Server stopped."; exit 0; }
  sleep 2
done

echo "WARNING: VRAM still allocated after the kfd sweep: $BUSY" >&2
echo "         A relaunch now will mis-size its KV-cache profile." >&2
exit 1
