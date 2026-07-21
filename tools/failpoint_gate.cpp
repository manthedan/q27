// Post-throw engine-state gate (k3 audit B2/E4, pre-registered in
// docs/metal/plans/2026-07-17-k3-audit-triage.md leg 4).
//
// The Q27_METAL_FAIL_FINISH countdown is read at the process's first
// finish_command and fires ONCE, so a single armed process can observe the
// throw and then run clean afterwards. The wrapper script sweeps N until the
// injection lands inside the step loop (exit 3/4 = fired too early during
// load/ingest, 5 = never fired; both mean "adjust N", not failure).
//
// Armed run asserts: (1) the injected failure surfaces as a throw, (2)
// position_ equals pos0 + successful steps — the failed round must not have
// advanced it, (3) reset() + regenerate reproduces a clean engine's tokens
// (printed; the script diffs against an unarmed run).
#include "metal_engine.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q27\n", argv[0]); return 2; }
    const bool armed = getenv("Q27_METAL_FAIL_FINISH") != nullptr;
    const std::vector<uint32_t> prompt = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000};
    constexpr uint32_t REF_TOKENS = 16;
    constexpr uint32_t MAX_STEPS = 48;
    try {
        std::unique_ptr<q27::MetalEngine> engine;
        try { engine = std::make_unique<q27::MetalEngine>(argv[1], 128, false); }
        catch (const std::exception& e) {
            if (armed) { fprintf(stderr, "fired during load: %s\n", e.what()); return 3; }
            throw;
        }
        if (!armed) {
            for (uint32_t t : engine->generate(prompt, REF_TOKENS)) printf("%u\n", t);
            return 0;
        }
        uint32_t pending;
        try { pending = engine->ingest_prompt(prompt, false); }
        catch (const std::exception& e) { fprintf(stderr, "fired during ingest: %s\n", e.what()); return 4; }
        const uint32_t pos0 = engine->position();
        uint32_t steps = 0;
        bool fired = false;
        while (steps < MAX_STEPS) {
            try {
                auto out = engine->generate_from_pending(pending, 2);
                pending = out[1];
                steps++;
            } catch (const std::exception& e) {
                if (std::string(e.what()).find("(injected)") == std::string::npos) throw;
                fired = true;
                break;
            }
        }
        if (!fired) { fprintf(stderr, "failpoint never fired in %u steps\n", MAX_STEPS); return 5; }
        if (engine->position() != pos0 + steps) {
            fprintf(stderr, "FAIL: position %u after %u successful steps from %u — the failed round advanced it\n",
                    engine->position(), steps, pos0);
            return 1;
        }
        fprintf(stderr, "injected throw after %u steps; position held at pos0+%u\n", steps, steps);
        engine->reset();
        for (uint32_t t : engine->generate(prompt, REF_TOKENS)) printf("%u\n", t);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "FAIL: unexpected error: %s\n", e.what());
        return 1;
    }
}
