# Recovered build surface from release/metal-v0.7.0 (971134e1).
# Keep the native product separate from upstream's CUDA default `all`.
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra
AGENT_TESTS := session protocol prompt tools selections stall persistence worker tui frontend
.PHONY: test-agent agent tui
test-agent: $(addprefix build/test_q27_agent_,$(AGENT_TESTS))
	@set -e; for test in $(AGENT_TESTS); do ./build/test_q27_agent_$$test; done

build/test_q27_agent_prompt: experiments/ds4-agent/test_q27_agent_prompt.cpp experiments/ds4-agent/q27_agent_prompt.h \
                              src/api_common.h src/tokenizer.cpp src/tokenizer.h src/unicode_tables.h \
                              src/stream_split.h src/markdown_lex.h src/drift_capture.h third_party/json.hpp | build
	$(CXX) $(CXXFLAGS) -I experiments/ds4-agent experiments/ds4-agent/test_q27_agent_prompt.cpp src/tokenizer.cpp -o $@

build/test_q27_agent_session: experiments/ds4-agent/test_q27_agent_session.cpp experiments/ds4-agent/q27_agent_session.h | build
	$(CXX) $(CXXFLAGS) -I experiments/ds4-agent experiments/ds4-agent/test_q27_agent_session.cpp -o $@

build/test_q27_agent_protocol: experiments/ds4-agent/test_q27_agent_protocol.cpp experiments/ds4-agent/q27_agent_protocol.cpp \
                               experiments/ds4-agent/q27_agent_protocol.h experiments/ds4-agent/q27_agent_tools.h \
                               experiments/ds4-agent/q27_agent_model.h src/api_common.h src/stream_split.h src/markdown_lex.h third_party/json.hpp | build
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
agent: build/q27-agent
build/q27-agent: experiments/ds4-agent/q27_agent_mask_epoch.h experiments/ds4-agent/q27_agent_prompt.h
tui: build/q27-tui
build/q27_agent_c.o: experiments/ds4-agent/q27_agent.c experiments/ds4-agent/q27_agent_worker.h experiments/ds4-agent/q27_agent_engine.h experiments/ds4-agent/q27_agent_protocol.h experiments/ds4-agent/q27_agent_persistence.h experiments/ds4-agent/q27_agent_selections.h experiments/ds4-agent/q27_agent_tui.h experiments/ds4-agent/q27_agent_editor.h experiments/ds4-agent/q27_agent_commands.h experiments/ds4-agent/q27_agent_frontend.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent.c -o $@

build/q27_agent_frontend_c.o: experiments/ds4-agent/q27_agent_frontend.c experiments/ds4-agent/q27_agent_frontend.h experiments/ds4-agent/q27_agent_worker.h experiments/ds4-agent/q27_agent_engine.h experiments/ds4-agent/q27_agent_tools.h | build
	$(CC) $(CFLAGS) -pthread -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_frontend.c -o $@

build/q27_agent_worker_c.o: experiments/ds4-agent/q27_agent_worker.c experiments/ds4-agent/q27_agent_worker.h experiments/ds4-agent/q27_agent_engine.h experiments/ds4-agent/q27_agent_tools.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_worker.c -o $@

build/q27_agent_persistence_c.o: experiments/ds4-agent/q27_agent_persistence.c experiments/ds4-agent/q27_agent_persistence.h experiments/ds4-agent/q27_agent_engine.h | build
	$(CC) $(CFLAGS) -I experiments/ds4-agent -c experiments/ds4-agent/q27_agent_persistence.c -o $@

build/q27_agent_protocol_cpp.o: experiments/ds4-agent/q27_agent_protocol.cpp experiments/ds4-agent/q27_agent_protocol.h experiments/ds4-agent/q27_agent_model.h src/api_common.h src/stream_split.h src/markdown_lex.h third_party/json.hpp | build
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

build/q27-agent: build/q27_agent_c.o build/q27_agent_worker_c.o build/q27_agent_frontend_c.o build/q27_agent_tools_c.o build/q27_agent_sha256_c.o build/q27_agent_selections_c.o build/q27_agent_persistence_c.o build/q27_agent_protocol_cpp.o build/q27_agent_tui_c.o build/q27_agent_commands_c.o build/q27_agent_editor_c.o build/q27_agent_linenoise_c.o experiments/ds4-agent/q27_agent_engine.cpp experiments/ds4-agent/q27_agent_engine.h experiments/ds4-agent/q27_agent_session.h experiments/ds4-agent/q27_agent_stall.h src/toolconstrain.h src/toolgram.h experiments/ds4-agent/q27_agent_model.h src/api_common.h src/stream_split.h src/markdown_lex.h \
                 src/metal/metal_engine.cpp src/metal/metal_engine.h src/suffixdraft.h src/sampling.h \
                 src/metal/metal_backend.mm src/metal/metal_backend.h src/metal/q27_kernels.metal \
                 src/backend.h src/loader.cpp src/loader.h src/tokenizer.cpp src/tokenizer.h src/unicode_tables.h src/metal/bonsai_rotation.h | build
	$(CXX) $(CXXFLAGS) -fobjc-arc -pthread -I src/metal -I experiments/ds4-agent \
	        build/q27_agent_c.o build/q27_agent_worker_c.o build/q27_agent_frontend_c.o build/q27_agent_tools_c.o build/q27_agent_sha256_c.o build/q27_agent_selections_c.o build/q27_agent_persistence_c.o build/q27_agent_protocol_cpp.o build/q27_agent_tui_c.o build/q27_agent_commands_c.o build/q27_agent_editor_c.o build/q27_agent_linenoise_c.o experiments/ds4-agent/q27_agent_engine.cpp src/metal/metal_engine.cpp \
	        src/metal/metal_backend.mm src/loader.cpp src/tokenizer.cpp \
	        -framework Foundation -framework Metal -o $@

# Rust Ratatui FP1 client (experiments/q27-tui). Copied into build/ so Q27_BIN_DIR works.
build/q27-tui: experiments/q27-tui/Cargo.toml experiments/q27-tui/Cargo.lock experiments/q27-tui/src/main.rs \
               experiments/q27-tui/src/app.rs experiments/q27-tui/src/backend.rs \
               experiments/q27-tui/src/proto.rs experiments/q27-tui/src/ui.rs \
               experiments/q27-tui/src/md.rs experiments/q27-tui/src/theme.rs | build
	cd experiments/q27-tui && cargo build --release --locked
	cp -f experiments/q27-tui/target/release/q27-tui $@

else
agent:
	@echo "native agent requires macOS/Metal" >&2; exit 1
endif
