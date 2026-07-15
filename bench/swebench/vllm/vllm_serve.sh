#!/usr/bin/env bash
# vLLM: Qwen3.6-27B NVFP4 (unsloth, has MTP head) on the 5090 only, single-user,
# MTP speculative decode. OpenAI API on :8080. Fair to q27: 1 GPU, fp8 KV, greedy.
set -e
docker rm -f vllm-server >/dev/null 2>&1 || true
docker run -d --name vllm-server \
  --gpus '"device=0"' --ipc=host -p 8080:8000 \
  -v /mnt/ai/hf_cache:/root/.cache/huggingface \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  vllm/vllm-openai:nightly \
  --model unsloth/Qwen3.6-27B-NVFP4 \
  --served-model-name vllm-qwen \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.96 \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --host 0.0.0.0 >/dev/null
echo "vllm-server started (:8080); tail logs with: docker logs -f vllm-server"
