#include "q27_agent_stall.h"

#include <cstdio>
#include <string>

#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

using q27::agent::StallReason;
using q27::agent::StallWatcher;

int main() {
    {
        StallWatcher watcher;
        for (std::size_t i = 1; i < StallWatcher::kEmptyTokenLimit; ++i)
            CHECK(watcher.observe(static_cast<uint32_t>(i), "") == StallReason::None,
                  "empty decode remains below threshold");
        CHECK(watcher.observe(999, "") == StallReason::EmptyDecode,
              "bounded empty decodes stall");
    }
    {
        StallWatcher watcher;
        std::string spaces(StallWatcher::kWhitespaceByteLimit - 1, ' ');
        CHECK(watcher.observe(1, spaces) == StallReason::None,
              "whitespace remains below byte threshold");
        CHECK(watcher.observe(2, "\n") == StallReason::WhitespaceRun,
              "mixed ASCII whitespace chunks stall at exact byte threshold");
    }
    {
        StallWatcher watcher;
        CHECK(watcher.observe(1, "visible ") == StallReason::None,
              "mixed token retains its trailing whitespace run");
        std::string spaces(StallWatcher::kWhitespaceByteLimit - 2, ' ');
        CHECK(watcher.observe(2, spaces) == StallReason::None,
              "mixed-token suffix remains below exact stream threshold");
        CHECK(watcher.observe(3, "\t") == StallReason::WhitespaceRun,
              "mixed-token suffix contributes to the exact stream threshold");
    }
    {
        StallWatcher watcher;
        std::string mixed = "x";
        mixed.append(StallWatcher::kWhitespaceByteLimit, ' ');
        CHECK(watcher.observe(1, mixed) == StallReason::WhitespaceRun,
              "one mixed decoded token cannot bypass the whitespace bound");
    }
    {
        StallWatcher watcher;
        for (std::size_t i = 1; i < StallWatcher::kIdenticalTokenLimit; ++i)
            CHECK(watcher.observe(27, "x") == StallReason::None,
                  "identical token run remains below threshold");
        CHECK(watcher.observe(27, "x") == StallReason::IdenticalTokenRun,
              "identical token run stalls at exact threshold");
    }
    {
        StallWatcher watcher;
        StallReason reason = StallReason::None;
        for (std::size_t i = 0; i < StallWatcher::kCycleWindow; ++i)
            reason = watcher.observe(static_cast<uint32_t>(100 + i % 3),
                                     i % 3 == 0 ? "a" : i % 3 == 1 ? "b" : "c");
        CHECK(reason == StallReason::ShortTokenCycle,
              "short multi-token cycle stalls at bounded window");
    }
    {
        StallWatcher watcher;
        for (std::size_t i = 0; i < 4096; ++i) {
            const uint32_t token = static_cast<uint32_t>(1000 + i);
            CHECK(watcher.observe(token, i % 2 ? "source" : " code") ==
                      StallReason::None,
                  "progressing source-like output does not false-positive");
        }
    }
    {
        StallWatcher watcher;
        for (std::size_t i = 0; i < StallWatcher::kWhitespaceByteLimit - 1; ++i)
            CHECK(watcher.observe(static_cast<uint32_t>(i), " ") ==
                      StallReason::None,
                  "whitespace counter advances");
        CHECK(watcher.observe(5000, "x") == StallReason::None,
              "visible progress resets whitespace counter");
        CHECK(watcher.observe(5001, " ") == StallReason::None,
              "post-progress whitespace starts a fresh run");
    }
    std::puts("q27 agent stall watcher selftest: PASS");
    return 0;
}
