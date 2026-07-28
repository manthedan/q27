#include "metal_engine.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 2;
    }
    try {
        q27::MetalEngine engine(argv[1], 32, false);
        constexpr uint32_t vocab = q27::MetalEngine::vocabulary_size();
        std::vector<uint32_t> allow_all((vocab + 31) / 32, UINT32_MAX);
        const int mask = engine.mask_pool_add(allow_all.data());
        if (mask < 0) throw std::runtime_error("constraint mask pool is full");
        engine.set_tool_constraint(mask);

        const uint32_t initial_position = engine.position();
        auto rejects_without_mutation = [&](auto&& call, const char* label) {
            bool rejected = false;
            try {
                call();
            } catch (const std::runtime_error& error) {
                rejected = std::string(error.what()).find("tool constraints require serial decode")
                    != std::string::npos;
            }
            if (!rejected || engine.position() != initial_position)
                throw std::runtime_error(std::string(label) +
                    " did not reject before mutating engine position");
        };

        rejects_without_mutation([&]{
            (void)engine.generate_mtp({1}, 1, 2);
        }, "greedy MTP");

        q27::SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_p = 0.95f;
        sampling.top_k = 40;
        rejects_without_mutation([&]{
            (void)engine.generate_mtp_sampled({1}, 1, 2, sampling);
        }, "sampled MTP");

        // The supported constrained path remains usable.
        (void)engine.ingest_prompt({1}, false, true);
        if (engine.position() != 1)
            throw std::runtime_error("serial constrained prefill did not advance");

        // A mid-prefill snapshot deliberately marks its logits row stale.
        // Loading it may resume ingestion, but must not derive a token until
        // a successful forward pass replaces that row.
        struct SnapshotFile {
            std::string path;
            ~SnapshotFile() { unlink(path.c_str()); }
        } snapshot{"/private/tmp/q27-metal-stale-logits-" +
                   std::to_string((long long)getpid()) + ".snap"};
        unlink(snapshot.path.c_str());
        const uint32_t snapshot_token = 1;
        engine.save_state(snapshot.path, &snapshot_token, 1, false);
        engine.reset();
        (void)engine.load_state(snapshot.path);
        bool stale_rejected = false;
        try {
            (void)engine.pending_from_logits();
        } catch (const std::runtime_error& error) {
            stale_rejected = std::string(error.what()).find("no resident logits") !=
                             std::string::npos;
        }
        if (!stale_rejected) {
            throw std::runtime_error("stale snapshot logits were accepted");
        }
        (void)engine.step(1);
        if (engine.pending_from_logits() >= vocab) {
            throw std::runtime_error("refreshed snapshot logits produced an invalid token");
        }

        std::puts("Metal constrained MTP and stale-snapshot contracts: PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
