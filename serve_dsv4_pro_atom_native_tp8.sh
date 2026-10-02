#!/usr/bin/env bash
# DeepSeek-V4-Pro-0813 on ATOM NATIVE (atom.entrypoints.openai_server) -- TP=8.
# The native baseline, against which serve_dsv4_pro_vllm_atom_tp8.sh (the vLLM
# plugin backend) is priced.
#
# Not `vllm serve`: this is ATOM's own OpenAI server, so the flag spellings are
# ATOM's (--server-port, --kv_cache_dtype, -tp) and NOT vLLM's. report_md.py
# tells the two arms apart by this entrypoint, read from arm.json.
#
# Logs go to the caller's stdout -- redirect into logs/ when launching. The
# original hardcoded `> /work/logs/server-dspark.log`, a path that does not
# exist on this node, which made every failure silent.

set -euo pipefail

export HF_HOME=${HF_HOME:-/data}
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1

MODEL=${MODEL:-/data/DeepSeek-V4-Pro-0813}
PORT=${PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

exec python3 -m atom.entrypoints.openai_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --server-port "$PORT" \
  -tp 8 --kv_cache_dtype fp8 --index_cache_dtype fp4 \
  --method dspark --num-speculative-tokens 7 \
  --enable-dp-attention --enable-tbo \
  --enable_prefix_caching \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --max-num-seqs 512
