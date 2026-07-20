#!/usr/bin/env bash
# Builds and runs the CPU-only /v1/chat/completions integration harness.
# No CUDA toolchain needed -- this fakes only the Engine/Slot/httplib
# surface (see the file's header comment); everything else is the real,
# unmodified header code. Run tools/extract_check.sh first (or as part of
# this script) to confirm the embedded handle()/build_prompt() haven't
# drifted from src/server.cu.
set -euo pipefail
cd "$(dirname "$0")/.."
./tools/extract_check.sh
mkdir -p build
g++ -std=c++17 -Wall -Wextra -I src tools/test_chat_completions_integration.cpp \
    -o build/test_chat_completions_integration
./build/test_chat_completions_integration
