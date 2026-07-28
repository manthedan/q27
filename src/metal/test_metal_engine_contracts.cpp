#include "metal_engine.h"

#include <cstdint>
#include <cstdio>
#include <functional>
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

            std::vector<uint32_t> mask((q27::MetalEngine::vocabulary_size()+31)/32,~0u);
            const int mask_id=engine.mask_pool_add(mask.data());
            if(mask_id<0) throw std::runtime_error("failed to allocate tool constraint mask");
            engine.set_tool_constraint(mask_id);
            const uint32_t constrained_pending=engine.ingest_prompt({1},true,true);
            const uint32_t constrained_position=engine.position();
            auto expect_constraint_reject=[&](const char* label,const std::function<void()>& call) {
                bool rejected=false;
                try { call(); }
                catch(const std::runtime_error& error) {
                    rejected=std::string(error.what()).find("tool constraints require serial decode")!=
                             std::string::npos;
                }
                if(!rejected) throw std::runtime_error(std::string(label)+" accepted active tool constraint");
                if(engine.position()!=constrained_position)
                    throw std::runtime_error(std::string(label)+" mutated state before rejection");
            };
            expect_constraint_reject("stream_from_pending",[&] {
                q27::MetalEngine::StopCause stop;
                (void)engine.stream_from_pending(constrained_pending,2,UINT32_MAX,4,
                    [](uint32_t) { return true; },stop);
            });
            expect_constraint_reject("generate_from_pending",[&] {
                (void)engine.generate_from_pending(constrained_pending,2,4);
            });
            expect_constraint_reject("mtp_round",[&] {
                uint32_t live=4; std::vector<uint32_t> committed;
                (void)engine.mtp_round(constrained_pending,2,UINT32_MAX,4,live,committed);
            });
            expect_constraint_reject("mtp_sample_round",[&] {
                uint32_t live=4; std::vector<uint32_t> committed;
                q27::SamplingParams params; std::mt19937_64 rng(1);
                (void)engine.mtp_sample_round(constrained_pending,2,UINT32_MAX,4,live,
                                              params,rng,committed);
            });
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
