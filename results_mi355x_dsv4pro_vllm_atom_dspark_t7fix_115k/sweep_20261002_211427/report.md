# sweep_20261002_211427

`/data/DeepSeek-V4-Pro-0813` | mi355x | 8 GPUs | ISL 115000 / OSL 1000 | `--cache 90` | seed 531150

aiter `503443ff0` | ATOM `a460a5645` | vLLM `281cfd5a0` | vllm 0.28.1.dev0+g2cf0a6915.d20261002

| conc | reqs | dur(s) | ISL | OSL | cache% | TTFT p90 ms | TTFT p50 ms | ITL p90 ms | ITL p50 ms | in/s/gpu | out/s/gpu | accept% | tok/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 80 | 152 | 115083 | 1000 | 80.08 | 29138 | 820 | 28.07 | 21.23 | 7563.2 | 65.72 | 16.13 | 2.129 |
| 24 | 120 | 165 | 115083 | 1000 | 88.98 | 7853 | 831 | 34.28 | 28.44 | 10479.4 | 91.06 | 13.81 | 1.967 |
| 32 | 160 | 194 | 115083 | 1000 | 88.98 | 9450 | 836 | 42.20 | 34.22 | 11865.4 | 103.10 | 13.87 | 1.971 |
| 64 | 320 | 339 | 115083 | 1000 | 88.98 | 18800 | 887 | 74.47 | 61.12 | 13569.6 | 117.91 | 11.79 | 1.825 |
| 128 | 640 | 589 | 115083 | 1000 | 88.98 | 37344 | 1469 | 140.69 | 107.88 | 15641.0 | 135.91 | 24.89 | 2.742 |

## Accepted drafts by position (% of drafts)

| conc | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | drafts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 42.3 | 26.4 | 17.2 | 11.7 | 7.9 | 4.8 | 2.6 | 37595 |
| 24 | 38.7 | 22.9 | 14.2 | 9.3 | 6.0 | 3.6 | 1.9 | 61038 |
| 32 | 38.8 | 23.3 | 14.3 | 9.3 | 6.0 | 3.6 | 1.8 | 81207 |
| 64 | 35.4 | 20.1 | 11.8 | 7.1 | 4.4 | 2.5 | 1.3 | 175357 |
| 128 | 57.3 | 38.9 | 27.5 | 20.0 | 14.7 | 10.1 | 5.8 | 233642 |

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
  --concurrency 16,24,32,64,128 \
  --isl 115000 --osl 1000 --cache 90 \
  --seed 531150 --gpus 8 \
  --out-dir $(cd "$(dirname "$0")" && pwd)/results_mi355x_${SERIES}/sweep_$(date +%Y%m%d_%H%M%S)
```
