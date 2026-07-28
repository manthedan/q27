#include "metal_engine.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s OFFICIAL.q27 BONSAI.q27\n", argv[0]);
        return 2;
    }
    try {
        {
            q27::MetalEngine engine(argv[1], 16, false);
            const uint32_t pending = engine.ingest_prompt({1}, false, true);
            int sink_calls = 0;
            q27::MetalEngine::StopCause cause = q27::MetalEngine::StopCause::MaxTokens;
            const uint32_t emitted = engine.stream_from_pending(
                pending, 4, std::numeric_limits<uint32_t>::max(), 4,
                [&](uint32_t) { return ++sink_calls < 2; }, cause);
            if (cause != q27::MetalEngine::StopCause::Cancelled || emitted != 1)
                throw std::runtime_error("batched MTP cancellation contract mismatch");
            if (engine.position() != 0)
                throw std::runtime_error("cancelled batched MTP left speculative state resident");
            bool stale_rejected = false;
            try {
                (void)engine.pending_from_logits();
            } catch (const std::runtime_error& error) {
                stale_rejected = std::string(error.what()).find("no resident logits") !=
                                 std::string::npos;
            }
            if (!stale_rejected)
                throw std::runtime_error("cancelled batched MTP left logits reusable");
        }
        {
            q27::MetalEngine engine(argv[2], 8, false);
            if (engine.chunked_prefill())
                throw std::runtime_error("Bonsai enabled activation-quantized chunk prefill");
            bool enable_rejected = false;
            try {
                engine.set_chunked_prefill(true);
            } catch (const std::runtime_error&) {
                enable_rejected = true;
            }
            if (!enable_rejected)
                throw std::runtime_error("Bonsai accepted activation-quantized chunk prefill");
            (void)engine.ingest_prompt({1, 1, 1}, false, true);
            if (engine.position() != 3)
                throw std::runtime_error("Bonsai serial prefill did not advance");
        }
        std::puts("Metal Bonsai prefill and MTP cancellation contracts: PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
