CC        ?= cc
CFLAGS    ?= -O2 -std=c11 -Wall -Wextra
CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra
NVCC      ?= /usr/local/cuda/bin/nvcc
# sm_120 = RTX 5090, sm_89 = RTX 4090/Ada (needs CUDA 12.4+ for e4m3 MMA),
# sm_86 = RTX 3090 (fallback device for tests)
NVCCFLAGS ?= -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_120,code=sm_120 -Xcompiler -Wall
UNAME_S   := $(shell uname -s)

.PHONY: all clean test-cpu test-metal agent tui agent-tui install-dev-q27
all: build/inspect build/test_kernels build/q27 build/q27-server build/test_tokenizer build/test_artifacts build/test_depthctl build/test_toolconstrain

# Friendly source-checkout entry point. The supervisor resolves the local B1
# artifact/tokenizer, enables bounded tools, and roots them at the caller's cwd.
# Prefers Ratatui q27-tui (FP1) when built and stdin/stdout are TTYs.
# The Rust TUI is built when Cargo is available (r19/r21 codex P2):
# cargo-less checkouts keep the classic path; build errors surface normally.
CARGO := $(shell command -v cargo 2>/dev/null)
agent: build/q27-agent $(if $(CARGO),build/q27-tui,)
	Q27_BIN_DIR="$(CURDIR)/build" ./packaging/bin/q27 agent

tui: build/q27-tui

# Build BOTH binaries, then launch (r20 codex P2: depending on the `agent`
# run target would start the blocking agent before the TUI ever builds).
agent-tui: build/q27-agent build/q27-tui
	Q27_BIN_DIR="$(CURDIR)/build" ./packaging/bin/q27 agent

# Put this checkout's packaging/bin/q27 first on PATH via ~/.grok/bin (already
# first in many Grok/dev shells). After this, `q27 agent b1` uses source TUI.
install-dev-q27: build/q27-agent build/q27-tui
	@mkdir -p "$(HOME)/.grok/bin"
	@printf '%s\n' '#!/bin/sh' \
	  'export Q27_BIN_DIR="$(CURDIR)/build"' \
	  'export Q27_SOURCE_ROOT="$(CURDIR)"' \
	  'exec "$(CURDIR)/packaging/bin/q27" "$$@"' \
	  >"$(HOME)/.grok/bin/q27"
	@chmod 755 "$(HOME)/.grok/bin/q27"
	@echo "installed $(HOME)/.grok/bin/q27 → source packaging (Q27_BIN_DIR=$(CURDIR)/build)"
	@echo "try: q27 agent b1"

test-cpu: build/test_artifacts build/test_depthctl build/test_toolconstrain build/test_suffixdraft build/test_sampling build/test_kl build/test_snapshot_evict build/test_snapshot_evict_store build/test_tokenizer build/test_q27_agent_session build/test_q27_agent_protocol build/test_q27_agent_tools build/test_q27_agent_selections build/test_q27_agent_stall build/test_q27_agent_persistence build/test_q27_agent_worker build/test_q27_agent_tui build/test_q27_agent_frontend build/test_tool_drift build/test_think_resolve build/test_stream_split
	./build/test_artifacts
	./build/test_depthctl
	./build/test_toolconstrain
	./build/test_suffixdraft
	./build/test_sampling
	./build/test_kl
	./build/test_snapshot_evict
	./build/test_snapshot_evict_store
	@# Upstream shared-parser hardening tests (api_common.h / stream_split.h):
	@# host-only, keep the Metal lane honest about the CUDA server's parser
	@# fixes (drift modes, fence-skip, think resolution).
	./build/test_tool_drift
	./build/test_think_resolve
	./build/test_stream_split
	@# test_tokenizer needs the .tok for its chatml/toolmask/tool-call/streaming
	@# gates (the vacuous-pass trap the handoff warned about: with NO args it
	@# runs only the self-tests and exits 1). The exact-id cases file is a
	@# small separate corpus; an empty file skips it while the .tok-driven
	@# gates run. Skip gracefully if the .tok is absent (e.g. a doc-only clone).
	@if [ -f models/qwen36-27b-mtp/qwen36-27b-mtp.tok ]; then \
		: > build/.empty-cases.txt; \
		./build/test_tokenizer models/qwen36-27b-mtp/qwen36-27b-mtp.tok build/.empty-cases.txt; \
	else \
		echo "test_tokenizer: SKIPPED (models/qwen36-27b-mtp/qwen36-27b-mtp.tok not present)"; \
	fi
	./build/test_q27_agent_session
	./build/test_q27_agent_protocol
	./build/test_q27_agent_tools
	./build/test_q27_agent_selections
	./build/test_q27_agent_stall
	./build/test_q27_agent_persistence
	./build/test_q27_agent_worker
	./build/test_q27_agent_tui
	./build/test_q27_agent_frontend
	./packaging/test_q27_wrapper.sh
	python3 tools/test_experimental_prefix_cache.py

build/test_q27_agent_session: experiments/ds4-agent/test_q27_agent_session.cpp experiments/ds4-agent/q27_agent_session.h | build
	$(CXX) $(CXXFLAGS) -I experiments/ds4-agent experiments/ds4-agent/test_q27_agent_session.cpp -o $@

build/test_q27_agent_protocol: experiments/ds4-agent/test_q27_agent_protocol.cpp experiments/ds4-agent/q27_agent_protocol.cpp \
                               experiments/ds4-agent/q27_agent_protocol.h experiments/ds4-agent/q27_agent_tools.h \
                               src/tool_preamble.h third_party/json.hpp | build
	$(CXX) $(CXXFLAGS) -I experiments/ds4-agent \
	        experiments/ds4-agent/test_q27_agent_protocol.cpp experiments/ds4-agent/q27_agent_protocol.cpp -o $@

build/test_q27_agent_tools: experiments/ds4-agent/test_q27_agent_tools.c experiments/ds4-agent/q27_agent_tools.c \
                            experiments/ds4-agent/q27_agent_tools.h experiments/ds4-agent/q27_agent_engine.h \
                            experiments/ds4-agent/q27_agent_sha256.c experiments/ds4-agent/q27_agent_sha256.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent \
	        experiments/ds4-agent/test_q27_agent_tools.c experiments/ds4-agent/q27_agent_tools.c \
	        experiments/ds4-agent/q27_agent_sha256.c -o $@

build/test_q27_agent_selections: experiments/ds4-agent/test_q27_agent_selections.c experiments/ds4-agent/q27_agent_selections.c \
                                 experiments/ds4-agent/q27_agent_selections.h experiments/ds4-agent/q27_agent_tools.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent \
	        experiments/ds4-agent/test_q27_agent_selections.c experiments/ds4-agent/q27_agent_selections.c -o $@

build/test_q27_agent_stall: experiments/ds4-agent/test_q27_agent_stall.cpp experiments/ds4-agent/q27_agent_stall.h | build
	$(CXX) $(CXXFLAGS) -I experiments/ds4-agent experiments/ds4-agent/test_q27_agent_stall.cpp -o $@

build/test_q27_agent_persistence: experiments/ds4-agent/test_q27_agent_persistence.c experiments/ds4-agent/q27_agent_persistence.c \
                                  experiments/ds4-agent/q27_agent_persistence.h experiments/ds4-agent/q27_agent_engine.h | build
	$(CC) $(CFLAGS) -DQ27_AGENT_PERSISTENCE_TESTING -I experiments/ds4-agent \
	        experiments/ds4-agent/test_q27_agent_persistence.c experiments/ds4-agent/q27_agent_persistence.c -o $@

build/test_q27_agent_worker: experiments/ds4-agent/test_q27_agent_worker.c experiments/ds4-agent/q27_agent_worker.c \
                             experiments/ds4-agent/q27_agent_worker.h experiments/ds4-agent/q27_agent_engine.h \
                             experiments/ds4-agent/q27_agent_tools.c experiments/ds4-agent/q27_agent_tools.h \
                             experiments/ds4-agent/q27_agent_sha256.c experiments/ds4-agent/q27_agent_sha256.h | build
	$(CC) $(CFLAGS) -pthread -DQ27_AGENT_WORKER_TESTING -I experiments/ds4-agent \
	        experiments/ds4-agent/test_q27_agent_worker.c experiments/ds4-agent/q27_agent_worker.c \
	        experiments/ds4-agent/q27_agent_tools.c experiments/ds4-agent/q27_agent_sha256.c -o $@

build/test_q27_agent_tui: experiments/ds4-agent/test_q27_agent_tui.c experiments/ds4-agent/q27_agent_tui.c \
                          experiments/ds4-agent/q27_agent_commands.c experiments/ds4-agent/q27_agent_tui.h \
                          experiments/ds4-agent/q27_agent_commands.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent \
	        experiments/ds4-agent/test_q27_agent_tui.c experiments/ds4-agent/q27_agent_tui.c \
	        experiments/ds4-agent/q27_agent_commands.c -o $@

build/test_q27_agent_frontend: experiments/ds4-agent/test_q27_agent_frontend.c \
                               experiments/ds4-agent/q27_agent_frontend.c \
                               experiments/ds4-agent/q27_agent_frontend.h \
                               experiments/ds4-agent/q27_agent_worker.h \
                               experiments/ds4-agent/q27_agent_engine.h \
                               experiments/ds4-agent/q27_agent_tools.h | build
	$(CC) $(CFLAGS) -pthread -I experiments/ds4-agent \
	        experiments/ds4-agent/test_q27_agent_frontend.c \
	        experiments/ds4-agent/q27_agent_frontend.c -o $@

ifeq ($(UNAME_S),Darwin)
test-metal: build/test_metal build/test_metal_ops build/test_metal_stream
	./build/test_metal
	./build/test_metal_ops
	./build/test_metal_stream

build/test_metal_stream: src/metal/test_metal_stream.cpp src/metal/stream_format.h src/api_common.h src/stream_split.h src/tool_preamble.h third_party/json.hpp | build
	$(CXX) $(CXXFLAGS) -I src/metal src/metal/test_metal_stream.cpp -o $@

build/test_metal: src/metal/test_metal.cpp src/metal/metal_backend.mm src/metal/metal_backend.h \
                  src/metal/q27_kernels.metal src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src/metal src/metal/test_metal.cpp \
	        src/metal/metal_backend.mm src/loader.cpp \
	        -framework Foundation -framework Metal -o $@

build/test_metal_ops: src/metal/test_metal_ops.cpp src/metal/metal_backend.mm src/metal/metal_backend.h \
                      src/metal/q27_kernels.metal src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src/metal src/metal/test_metal_ops.cpp \
	        src/metal/metal_backend.mm src/loader.cpp \
	        -framework Foundation -framework Metal -o $@

build/q27-metal: src/metal/metal_cli.cpp src/metal/metal_engine.cpp src/metal/metal_engine.h src/suffixdraft.h src/sampling.h src/kl.h \
                 src/metal/metal_backend.mm src/metal/metal_backend.h src/metal/q27_kernels.metal \
                 src/backend.h src/loader.cpp src/loader.h src/tokenizer.cpp src/tokenizer.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src/metal src/metal/metal_cli.cpp src/metal/metal_engine.cpp \
	        src/metal/metal_backend.mm src/loader.cpp src/tokenizer.cpp \
	        -framework Foundation -framework Metal -o $@

build/q27-metal-server: src/metal/metal_server.cpp src/metal/metal_engine.cpp src/metal/metal_engine.h src/metal/stream_format.h src/api_common.h src/stream_split.h src/tool_preamble.h src/suffixdraft.h src/sampling.h \
                        src/metal/metal_backend.mm src/metal/metal_backend.h src/metal/q27_kernels.metal \
                        src/backend.h src/loader.cpp src/loader.h src/tokenizer.cpp src/tokenizer.h \
                        third_party/httplib.h third_party/json.hpp | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -pthread -I src/metal src/metal/metal_server.cpp src/metal/metal_engine.cpp \
	        src/metal/metal_backend.mm src/loader.cpp src/tokenizer.cpp \
	        -framework Foundation -framework Metal -o $@

build/q27_agent_c.o: experiments/ds4-agent/q27_agent.c experiments/ds4-agent/q27_agent_worker.h experiments/ds4-agent/q27_agent_engine.h experiments/ds4-agent/q27_agent_protocol.h experiments/ds4-agent/q27_agent_persistence.h experiments/ds4-agent/q27_agent_selections.h experiments/ds4-agent/q27_agent_tui.h experiments/ds4-agent/q27_agent_editor.h experiments/ds4-agent/q27_agent_commands.h experiments/ds4-agent/q27_agent_frontend.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent.c -o $@

build/q27_agent_frontend_c.o: experiments/ds4-agent/q27_agent_frontend.c experiments/ds4-agent/q27_agent_frontend.h experiments/ds4-agent/q27_agent_worker.h experiments/ds4-agent/q27_agent_engine.h experiments/ds4-agent/q27_agent_tools.h | build
	$(CC) $(CFLAGS) -pthread -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_frontend.c -o $@

build/q27_agent_worker_c.o: experiments/ds4-agent/q27_agent_worker.c experiments/ds4-agent/q27_agent_worker.h experiments/ds4-agent/q27_agent_engine.h experiments/ds4-agent/q27_agent_tools.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_worker.c -o $@

build/q27_agent_persistence_c.o: experiments/ds4-agent/q27_agent_persistence.c experiments/ds4-agent/q27_agent_persistence.h experiments/ds4-agent/q27_agent_engine.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_persistence.c -o $@

build/q27_agent_protocol_cpp.o: experiments/ds4-agent/q27_agent_protocol.cpp experiments/ds4-agent/q27_agent_protocol.h src/tool_preamble.h third_party/json.hpp | build
	$(CXX) $(CXXFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_protocol.cpp -o $@

build/q27_agent_tools_c.o: experiments/ds4-agent/q27_agent_tools.c experiments/ds4-agent/q27_agent_tools.h experiments/ds4-agent/q27_agent_engine.h experiments/ds4-agent/q27_agent_sha256.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_tools.c -o $@

build/q27_agent_sha256_c.o: experiments/ds4-agent/q27_agent_sha256.c experiments/ds4-agent/q27_agent_sha256.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_sha256.c -o $@

build/q27_agent_selections_c.o: experiments/ds4-agent/q27_agent_selections.c experiments/ds4-agent/q27_agent_selections.h experiments/ds4-agent/q27_agent_tools.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_selections.c -o $@

build/q27_agent_tui_c.o: experiments/ds4-agent/q27_agent_tui.c experiments/ds4-agent/q27_agent_tui.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_tui.c -o $@

build/q27_agent_commands_c.o: experiments/ds4-agent/q27_agent_commands.c experiments/ds4-agent/q27_agent_commands.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_commands.c -o $@

build/q27_agent_editor_c.o: experiments/ds4-agent/q27_agent_editor.c experiments/ds4-agent/q27_agent_editor.h experiments/ds4-agent/q27_agent_tui.h experiments/ds4-agent/third_party/linenoise/linenoise.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_editor.c -o $@

build/q27_agent_linenoise_c.o: experiments/ds4-agent/third_party/linenoise/linenoise.c experiments/ds4-agent/third_party/linenoise/linenoise.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent/third_party/linenoise -c experiments/ds4-agent/third_party/linenoise/linenoise.c -o $@

build/q27-agent: build/q27_agent_c.o build/q27_agent_worker_c.o build/q27_agent_frontend_c.o build/q27_agent_tools_c.o build/q27_agent_sha256_c.o build/q27_agent_selections_c.o build/q27_agent_persistence_c.o build/q27_agent_protocol_cpp.o build/q27_agent_tui_c.o build/q27_agent_commands_c.o build/q27_agent_editor_c.o build/q27_agent_linenoise_c.o experiments/ds4-agent/q27_agent_engine.cpp experiments/ds4-agent/q27_agent_engine.h experiments/ds4-agent/q27_agent_session.h experiments/ds4-agent/q27_agent_stall.h src/toolconstrain.h src/toolgram.h \
                 src/metal/metal_engine.cpp src/metal/metal_engine.h src/suffixdraft.h src/sampling.h \
                 src/metal/metal_backend.mm src/metal/metal_backend.h src/metal/q27_kernels.metal \
                 src/backend.h src/loader.cpp src/loader.h src/tokenizer.cpp src/tokenizer.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -pthread -I src/metal -I experiments/ds4-agent \
	        build/q27_agent_c.o build/q27_agent_worker_c.o build/q27_agent_frontend_c.o build/q27_agent_tools_c.o build/q27_agent_sha256_c.o build/q27_agent_selections_c.o build/q27_agent_persistence_c.o build/q27_agent_protocol_cpp.o build/q27_agent_tui_c.o build/q27_agent_commands_c.o build/q27_agent_editor_c.o build/q27_agent_linenoise_c.o experiments/ds4-agent/q27_agent_engine.cpp src/metal/metal_engine.cpp \
	        src/metal/metal_backend.mm src/loader.cpp src/tokenizer.cpp \
	        -framework Foundation -framework Metal -o $@

# Rust Ratatui FP1 client (experiments/q27-tui). Copied into build/ so Q27_BIN_DIR works.
build/q27-tui: experiments/q27-tui/Cargo.toml experiments/q27-tui/src/main.rs \
               experiments/q27-tui/src/app.rs experiments/q27-tui/src/backend.rs \
               experiments/q27-tui/src/proto.rs experiments/q27-tui/src/ui.rs \
               experiments/q27-tui/src/md.rs experiments/q27-tui/src/theme.rs | build
	cd experiments/q27-tui && cargo build --release
	cp -f experiments/q27-tui/target/release/q27-tui $@

build/metal_gemv_bench: tools/metal_gemv_bench.cpp src/metal/metal_backend.mm src/metal/metal_backend.h \
                        src/metal/q27_kernels.metal src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src/metal tools/metal_gemv_bench.cpp \
	        src/metal/metal_backend.mm src/loader.cpp \
	        -framework Foundation -framework Metal -o $@
build/failpoint_gate: tools/failpoint_gate.cpp src/metal/metal_engine.cpp src/metal/metal_engine.h src/suffixdraft.h src/sampling.h \
                        src/metal/metal_backend.mm src/metal/metal_backend.h \
                        src/metal/q27_kernels.metal src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src/metal tools/failpoint_gate.cpp src/metal/metal_engine.cpp \
	        src/metal/metal_backend.mm src/loader.cpp \
	        -framework Foundation -framework Metal -o $@
build/metal_mma_roofline: tools/metal_mma_roofline.cpp src/metal/metal_backend.mm src/metal/metal_backend.h \
                        src/metal/q27_kernels.metal src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src/metal tools/metal_mma_roofline.cpp \
	        src/metal/metal_backend.mm src/loader.cpp \
	        -framework Foundation -framework Metal -o $@
build/metal_decode_bench: tools/metal_decode_bench.cpp src/metal/metal_backend.mm src/metal/metal_backend.h \
                        src/metal/q27_kernels.metal src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src/metal tools/metal_decode_bench.cpp \
	        src/metal/metal_backend.mm src/loader.cpp \
	        -framework Foundation -framework Metal -o $@
build/metal_prefill_bench: tools/metal_prefill_bench.cpp src/metal/metal_backend.mm src/metal/metal_backend.h \
                        src/metal/q27_kernels.metal src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src/metal tools/metal_prefill_bench.cpp \
	        src/metal/metal_backend.mm src/loader.cpp \
	        -framework Foundation -framework Metal -o $@
build/metal_attn_bench: tools/metal_attn_bench.cpp src/metal/metal_backend.mm src/metal/metal_backend.h \
                        src/metal/q27_kernels.metal src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -I src/metal tools/metal_attn_bench.cpp \
	        src/metal/metal_backend.mm src/loader.cpp \
	        -framework Foundation -framework Metal -o $@

else
test-metal:
	@echo "test-metal requires macOS"; exit 1
endif

build/q27: src/engine.cu src/engine.cuh src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp \
           src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/device_model.h src/loader.h src/cuda_common.h src/depthctl.h | build
	$(NVCC) $(NVCCFLAGS) src/engine.cu src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@

build:
	mkdir -p build

build/inspect: src/inspect.cpp src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) src/inspect.cpp src/loader.cpp -o $@

# Phase-0 probe for k3 item #3 (group-skip over T2 zeros; d5a5674). CPU only,
# read-only mmap scan — pre-registered kill line lives in the tool's output.
build/t2_zero_sim: tools/t2_zero_sim.cpp src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) -I src tools/t2_zero_sim.cpp src/loader.cpp -o $@

build/tokenize_to_bin: tools/tokenize_to_bin.cpp src/tokenizer.cpp src/tokenizer.h | build
	$(CXX) $(CXXFLAGS) -I src tools/tokenize_to_bin.cpp src/tokenizer.cpp -o $@

build/test_tokenizer: src/test_tokenizer.cpp src/tokenizer.cpp src/tokenizer.h src/api_common.h src/tool_preamble.h src/stream_split.h src/toolgram.h | build
	$(CXX) $(CXXFLAGS) src/test_tokenizer.cpp src/tokenizer.cpp -o $@

build/test_artifacts: src/test_artifacts.cpp src/loader.cpp src/loader.h src/tokenizer.cpp src/tokenizer.h | build
	$(CXX) $(CXXFLAGS) src/test_artifacts.cpp src/loader.cpp src/tokenizer.cpp -o $@

build/test_depthctl: tools/test_depthctl.cpp src/depthctl.h | build
	$(CXX) $(CXXFLAGS) tools/test_depthctl.cpp -o $@

# Upstream shared-parser tests: keep the files pristine for future merges
# (they include "api_common.h" path-less), hence -Isrc here.
build/test_tool_drift: tools/test_tool_drift.cpp src/api_common.h | build
	$(CXX) $(CXXFLAGS) -Isrc tools/test_tool_drift.cpp -o $@

build/test_think_resolve: tools/test_think_resolve.cpp src/api_common.h | build
	$(CXX) $(CXXFLAGS) -Isrc tools/test_think_resolve.cpp -o $@

build/test_stream_split: tools/test_stream_split.cpp src/stream_split.h | build
	$(CXX) $(CXXFLAGS) -Isrc tools/test_stream_split.cpp -o $@

build/test_toolconstrain: tools/test_toolconstrain.cpp src/toolconstrain.h src/toolgram.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_toolconstrain.cpp -o $@

build/test_suffixdraft: tools/test_suffixdraft.cpp src/suffixdraft.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_suffixdraft.cpp -o $@

build/test_sampling: src/test_sampling.cpp src/sampling.h | build
	$(CXX) $(CXXFLAGS) -I src src/test_sampling.cpp -o $@

build/test_kl: src/test_kl.cpp src/kl.h | build
	$(CXX) $(CXXFLAGS) -I src src/test_kl.cpp -o $@

build/test_snapshot_evict: tools/test_snapshot_evict.cpp src/metal/snapshot_evict.h | build
	$(CXX) $(CXXFLAGS) -I src/metal tools/test_snapshot_evict.cpp -o $@

build/test_snapshot_evict_store: tools/test_snapshot_evict_store.cpp src/metal/disk_snapshot_store.h src/metal/snapshot_evict.h | build
	$(CXX) $(CXXFLAGS) -I src/metal tools/test_snapshot_evict_store.cpp -o $@

build/width_bench: tools/width_bench.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/width_bench.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp -o $@

build/mma16_bench: tools/mma16_bench.cu src/kernels.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/mma16_bench.cu src/kernels.cu src/device_model.cu src/loader.cpp -o $@

build/test_kernels: src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp \
                    src/kernels.cuh src/prefill.cuh src/blocks.cuh src/spec3.cuh src/fdmma.cuh src/turbo3.cuh src/device_model.h src/loader.h src/cuda_common.h | build
	$(NVCC) $(NVCCFLAGS) src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@


build/q27-server: src/server.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                  src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/tool_preamble.h src/stream_split.h \
                  src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/cuda_common.h src/toolgram.h \
                  src/depthctl.h src/toolconstrain.h src/tokenizer.h | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@

clean:
	rm -rf build

build/gdn_chunk_bench: tools/gdn_chunk_bench.cu | build
	$(NVCC) $(NVCCFLAGS) tools/gdn_chunk_bench.cu -o $@

build/attn_fdw_bench: tools/attn_fdw_bench.cu | build
	$(NVCC) $(NVCCFLAGS) tools/attn_fdw_bench.cu -o $@

VGEMM_SRC = src/vgemm.cu src/kernels.cu src/spec3.cu src/blocks.cu src/prefill.cu \
            src/device_model.cu src/loader.cpp

# P1 gates for the flat-in-W verify weight path (docs/plans/2026-07-13-gemm-verify.md):
#   vgemm_test -- gate 3 (numerics vs the gemv on all lanes/widths + determinism)
#                 and gate 4 (regs/spill/CTA-per-SM; FAILS LOUD -- zero slack).
#   vgemm_race -- gate 6's racecheck leg. racecheck instruments every shared-memory
#                 access and cannot finish on a real 47MB weight, so this drives the
#                 identical reduce path on a synthetic shape with z > 1.
build/vgemm_test: tools/vgemm_test.cu src/vgemm.cuh $(VGEMM_SRC) | build
	$(NVCC) $(NVCCFLAGS) tools/vgemm_test.cu $(VGEMM_SRC) -o $@

build/vgemm_race: tools/vgemm_race.cu src/vgemm.cuh $(VGEMM_SRC) | build
	$(NVCC) $(NVCCFLAGS) tools/vgemm_race.cu $(VGEMM_SRC) -o $@

build/fdmma_test: tools/fdmma_test.cu src/fdmma.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/fdmma_test.cu -o $@

build/turbo3_test: tools/turbo3_test.cu src/turbo3.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/turbo3_test.cu -o $@

# 24GB-card (3090-class) server: Q27_W_MAX=8 shrinks the GDN role sets +
# graph zoo so the fixed stack fits beside the weights (the default W12
# build OOMs at graph instantiation on 24GB). Same sources, own binary.
build/q27-server-w8: src/server.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                     src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/tool_preamble.h src/stream_split.h \
                     src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/cuda_common.h src/toolgram.h \
                     src/depthctl.h src/toolconstrain.h src/tokenizer.h | build
	$(NVCC) $(NVCCFLAGS) -DQ27_W_MAX=8 -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@

# Continuous-batching gates (docs/plans/2026-07-14-continuous-batching.md):
#   ninv_test      -- N-invariance: per-lane weight-kernel output must be bitwise
#                     independent of union width and slot (the batching contract).
#   test_conductor -- CPU: trim policy + ConductorCore membership/round-boundary.
#   fused_smoke    -- 2-engine fused round vs solo byte-identity + conductor +
#                     A2 error-injection legs (needs the GPU + model).
build/ninv_test: tools/ninv_test.cu src/vgemm.cuh src/kernels.cuh src/blocks.cuh $(VGEMM_SRC) | build
	$(NVCC) $(NVCCFLAGS) tools/ninv_test.cu $(VGEMM_SRC) -o $@

build/test_conductor: tools/test_conductor.cpp src/conductor.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_conductor.cpp -o $@

build/fused_smoke: tools/fused_smoke.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu \
                   src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/fused_smoke.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@

# w16 serving build (batch mode's natural target; was hand-built since part 10)
build/q27-server-w16: src/server.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                      src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/tool_preamble.h src/stream_split.h \
                      src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/cuda_common.h src/toolgram.h \
                      src/depthctl.h src/toolconstrain.h src/tokenizer.h | build
	$(NVCC) $(NVCCFLAGS) -DQ27_W_MAX=16 -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@
