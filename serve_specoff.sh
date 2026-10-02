export HF_HOME=${HF_HOME:-/data}
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
# Plugin-only: the DSpark draft is vLLM's model, and its MXFP4 MoE oracle
# rejects aiter without this. Native needs no equivalent -- ATOM owns every layer.
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
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --moe-backend aiter
