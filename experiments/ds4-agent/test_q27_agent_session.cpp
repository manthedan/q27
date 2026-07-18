#include "q27_agent_session.h"

#include <cstdio>
#include <vector>

using q27::agent::AgentSession;

#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

int main() {
    AgentSession session;
    const std::vector<uint32_t> first{10, 11, 12};
    auto plan = session.plan(first, 0);
    CHECK(plan.reset && plan.cached_tokens == 0 && plan.append_offset == 0,
          "cold session resets");

    session.commit_prompt(first);
    CHECK(session.valid() && !session.pending() && session.token_count() == 3,
          "prompt commit establishes exact ledger");
    CHECK(session.record_emitted(20) && session.pending(),
          "emitted token is retained as pending");

    const std::vector<uint32_t> extended{10, 11, 12, 20, 30, 31};
    plan = session.plan(extended, 3);
    CHECK(!plan.reset && plan.finalize_pending && plan.cached_tokens == 3 &&
          plan.append_offset == 4,
          "stable prefix finalizes pending then appends suffix");

    CHECK(session.mark_pending_encoded() && !session.pending(),
          "pending token finalizes exactly once");
    session.commit_prompt(extended);
    plan = session.plan(extended, 6);
    CHECK(!plan.reset && !plan.finalize_pending && plan.cached_tokens == 6 &&
          plan.append_offset == 6,
          "fully encoded exact prompt is reusable");

    const std::vector<uint32_t> mismatch{10, 11, 99, 20, 30, 31, 40};
    CHECK(session.plan(mismatch, 6).reset,
          "token-prefix mismatch fails closed to reset");
    CHECK(session.plan(extended, 5).reset,
          "engine-position mismatch fails closed to reset");
    CHECK(session.plan({10, 11}, 6).reset,
          "shorter prompt cannot reuse longer history");

    CHECK(session.record_emitted(40), "new pending output records");
    CHECK(!session.record_emitted(41) && !session.valid(),
          "double-pending invariant violation invalidates session");
    CHECK(session.plan(first, 0).reset, "invalidated session resets");

    session.commit_prompt(first);
    session.invalidate();
    CHECK(!session.valid() && !session.pending() && session.token_count() == 0,
          "explicit cancellation/error invalidation clears ledger");

    std::puts("q27 agent session selftest: PASS");
    return 0;
}
