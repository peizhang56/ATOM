# sweep_20261004_bothfixes

`/data/DeepSeek-V4-Pro-0813` | mi355x | 8 GPUs | ISL 115000 / OSL 1000 | `--cache 90` | seed 531150

aiter `efa76be1f` | ATOM `dde6b5ba5` | vLLM `281cfd5a0` | vllm 0.28.1.dev0+g2cf0a6915.d20261003

| conc | reqs | dur(s) | ISL | OSL | cache% | TTFT p90 ms | TTFT p50 ms | ITL p90 ms | ITL p50 ms | in/s/gpu | out/s/gpu | accept% | tok/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 80 | 130 | 115083 | 1000 | 80.08 | 28665 | 809 | 24.69 | 18.15 | 8833.4 | 76.76 | 31.33 | 3.193 |
| 24 | 120 | 133 | 115083 | 1000 | 88.98 | 7910 | 802 | 32.86 | 23.44 | 12972.2 | 112.72 | 29.73 | 3.081 |
| 32 | 160 | 158 | 115083 | 1000 | 88.98 | 9465 | 810 | 36.81 | 28.74 | 14534.2 | 126.29 | 30.55 | 3.139 |
| 64 | 320 | 275 | 115083 | 1000 | 88.98 | 18930 | 1229 | 66.27 | 49.79 | 16729.5 | 145.37 | 30.73 | 3.151 |
| 128 | 640 | 563 | 115083 | 1000 | 88.98 | 37509 | 1463 | 136.31 | 102.22 | 16341.1 | 141.99 | 30.90 | 3.163 |

## Accepted drafts by position (% of drafts)

| conc | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | drafts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 66.6 | 46.8 | 34.4 | 26.2 | 20.4 | 14.9 | 10.0 | 25082 |
| 24 | 66.2 | 45.7 | 32.7 | 24.1 | 17.9 | 13.0 | 8.5 | 38996 |
| 32 | 66.3 | 46.1 | 33.4 | 25.1 | 19.4 | 14.3 | 9.2 | 51032 |
| 64 | 66.6 | 46.5 | 33.7 | 25.3 | 19.3 | 14.2 | 9.4 | 101703 |
| 128 | 66.5 | 46.7 | 34.0 | 25.6 | 19.5 | 14.4 | 9.6 | 202602 |

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
