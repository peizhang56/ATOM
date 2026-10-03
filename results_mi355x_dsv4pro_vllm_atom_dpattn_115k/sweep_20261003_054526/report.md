# sweep_20261003_054526

`/data/DeepSeek-V4-Pro-0813` | mi355x | 8 GPUs | ISL 115000 / OSL 1000 | `--cache 90` | seed 531150

aiter `503443ff0` | ATOM `fa4134ef8` | vLLM `281cfd5a0` | vllm 0.28.1.dev0+g2cf0a6915.d20261002

| conc | reqs | dur(s) | ISL | OSL | cache% | TTFT p90 ms | TTFT p50 ms | ITL p90 ms | ITL p50 ms | in/s/gpu | out/s/gpu | accept% | tok/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 80 | 245 | 115083 | 1000 | 80.08 | 47641 | 1343 | 47.84 | 30.86 | 4704.6 | 40.88 | 14.57 | 2.020 |
| 24 | 120 | 252 | 115083 | 1000 | 88.98 | 13720 | 1325 | 53.95 | 42.57 | 6863.5 | 59.64 | 13.72 | 1.960 |
| 32 | 160 | 311 | 115083 | 1000 | 88.98 | 17833 | 1345 | 69.20 | 53.75 | 7393.9 | 64.25 | 13.23 | 1.926 |

## Accepted drafts by position (% of drafts)

| conc | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | drafts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 41.0 | 24.3 | 15.4 | 9.9 | 6.1 | 3.6 | 1.7 | 39622 |
| 24 | 39.4 | 23.2 | 14.2 | 8.9 | 5.5 | 3.2 | 1.5 | 61225 |
| 32 | 38.4 | 22.2 | 13.6 | 8.6 | 5.3 | 3.1 | 1.4 | 83083 |

## Server

```bash
export HF_HOME=/data
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

vllm serve /data/DeepSeek-V4-Pro-0813 \
  --served-model-name /data/DeepSeek-V4-Pro-0813 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 512 \
  --tokenizer-mode deepseek_v4 \
  --no-async-scheduling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --moe-backend aiter
```

## Client

```bash
export HF_HOME=/data

./run_gsm8k_benchmark.sh \
  --url http://127.0.0.1:8000 --model /data/DeepSeek-V4-Pro-0813 --tokenizer /data/DeepSeek-V4-Pro-0813 --api-key EMPTY \
  --concurrency 16,24,32 \
  --isl 115000 --osl 1000 --cache 90 \
  --seed 531150 --gpus 8 \
  --out-dir $(cd "$(dirname "$0")" && pwd)/results_mi355x_${SERIES}/sweep_$(date +%Y%m%d_%H%M%S)
```
