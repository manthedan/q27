#ifndef Q27_AGENT_STALL_H
#define Q27_AGENT_STALL_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace q27::agent {

enum class StallReason {
    None = 0,
    EmptyDecode,
    WhitespaceRun,
    IdenticalTokenRun,
    ShortTokenCycle
};

// Conservative, allocation-free generated-output watchdog. It observes a
// candidate token before that token is streamed. A trip therefore leaves all
// prior streamed bytes irreversible but excludes the triggering token.
class StallWatcher {
public:
    static constexpr std::size_t kEmptyTokenLimit = 64;
    static constexpr std::size_t kWhitespaceByteLimit = 256;
    static constexpr std::size_t kIdenticalTokenLimit = 128;
    static constexpr std::size_t kCycleWindow = 256;
    static constexpr std::size_t kMaxCyclePeriod = 8;

    StallReason observe(uint32_t token, std::string_view bytes) {
        if (bytes.empty()) {
            if (++empty_tokens_ >= kEmptyTokenLimit)
                return StallReason::EmptyDecode;
        } else {
            empty_tokens_ = 0;
            // Decoded tokens may mix visible bytes with whitespace prefixes or
            // suffixes. Track the byte stream in order rather than classifying
            // the token as a whole; the candidate remains atomic and is not
            // streamed if any byte reaches the bound.
            for (unsigned char c : bytes) {
                if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                    if (++whitespace_bytes_ >= kWhitespaceByteLimit)
                        return StallReason::WhitespaceRun;
                } else {
                    whitespace_bytes_ = 0;
                }
            }
        }

        if (have_previous_ && token == previous_token_) {
            if (++identical_tokens_ >= kIdenticalTokenLimit)
                return StallReason::IdenticalTokenRun;
        } else {
            previous_token_ = token;
            have_previous_ = true;
            identical_tokens_ = 1;
        }

        history_[history_next_] = token;
        history_next_ = (history_next_ + 1) % kCycleWindow;
        if (history_count_ < kCycleWindow) ++history_count_;
        if (history_count_ == kCycleWindow) {
            for (std::size_t period = 2; period <= kMaxCyclePeriod; ++period) {
                bool repeats = true;
                for (std::size_t i = period; i < kCycleWindow; ++i) {
                    if (history_at(i) != history_at(i - period)) {
                        repeats = false;
                        break;
                    }
                }
                if (repeats) return StallReason::ShortTokenCycle;
            }
        }
        return StallReason::None;
    }

private:
    uint32_t history_at(std::size_t chronological_index) const {
        // Once full, history_next_ is the oldest element.
        return history_[(history_next_ + chronological_index) % kCycleWindow];
    }

    std::array<uint32_t, kCycleWindow> history_{};
    std::size_t history_next_ = 0;
    std::size_t history_count_ = 0;
    std::size_t empty_tokens_ = 0;
    std::size_t whitespace_bytes_ = 0;
    std::size_t identical_tokens_ = 0;
    uint32_t previous_token_ = 0;
    bool have_previous_ = false;
};

} // namespace q27::agent

#endif
