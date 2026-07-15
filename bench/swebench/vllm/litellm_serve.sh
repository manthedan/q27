#!/usr/bin/env bash
# litellm proxy: exposes Anthropic /v1/messages on :8081, translates -> vLLM OpenAI :8080.
# This is the shim that lets Claude Code drive vLLM (vLLM has no native /v1/messages).
set -e
docker rm -f litellm-proxy >/dev/null 2>&1 || true
docker run -d --name litellm-proxy \
  --add-host host.docker.internal:host-gateway -p 8081:4000 \
  -v /mnt/ai/projects/q27/scratchpad/litellm/config.yaml:/app/config.yaml:ro \
  ghcr.io/berriai/litellm:main-stable \
  --config /app/config.yaml --port 4000 >/dev/null
echo "litellm-proxy started (:8081 -> vLLM :8080)"
