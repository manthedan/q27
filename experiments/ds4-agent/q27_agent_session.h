#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace q27::agent {

// Owner-thread-only token/state ledger for one resident MetalEngine. The
// ledger never guesses that rendered text is equivalent to generated token
// ids: reuse requires the next fully rendered prompt to begin with the exact
// prior token vector and the engine position to match its encoded prefix.
class AgentSession {
  public:
    struct Plan {
        bool reset = true;
        bool finalize_pending = false;
        size_t append_offset = 0; // first new prompt token after finalization
        size_t cached_tokens = 0; // already encoded in the resident engine
    };

    Plan plan(const std::vector<uint32_t>& prompt,
              uint32_t engine_position) const noexcept {
        Plan result;
        if (!valid_) return result;
        const size_t encoded = tokens_.size() - (pending_ ? 1u : 0u);
        if (encoded != engine_position || prompt.size() < tokens_.size())
            return result;
        for (size_t i = 0; i < tokens_.size(); ++i)
            if (prompt[i] != tokens_[i]) return result;
        result.reset = false;
        result.finalize_pending = pending_;
        result.append_offset = tokens_.size();
        result.cached_tokens = encoded;
        return result;
    }

    void commit_prompt(const std::vector<uint32_t>& prompt) {
        tokens_ = prompt;
        valid_ = true;
        pending_ = false;
    }

    bool record_emitted(uint32_t token) {
        // Callers must encode the previous emitted token before recording the
        // next one; violating that invariant invalidates reuse fail closed.
        if (pending_) {
            invalidate();
            return false;
        }
        tokens_.push_back(token);
        valid_ = true;
        pending_ = true;
        return true;
    }

    bool mark_pending_encoded() noexcept {
        if (!valid_ || !pending_) {
            invalidate();
            return false;
        }
        pending_ = false;
        return true;
    }

    // Restore is used only after Q27SNAP1 has validated artifact/config/layout,
    // its token metadata is an exact prefix of the re-rendered transcript, and
    // Metal position (plus pending logits when present) agrees with the ledger.
    bool restore(const std::vector<uint32_t>& tokens, bool pending) {
        if (tokens.empty() || (pending && tokens.size() < 1)) {
            invalidate();
            return false;
        }
        tokens_ = tokens;
        valid_ = true;
        pending_ = pending;
        return true;
    }

    void invalidate() noexcept {
        tokens_.clear();
        valid_ = false;
        pending_ = false;
    }

    bool valid() const noexcept { return valid_; }
    bool pending() const noexcept { return pending_; }
    size_t token_count() const noexcept { return tokens_.size(); }
    size_t encoded_count() const noexcept {
        return valid_ ? tokens_.size() - (pending_ ? 1u : 0u) : 0u;
    }
    const std::vector<uint32_t>& tokens() const noexcept { return tokens_; }

  private:
    std::vector<uint32_t> tokens_;
    bool valid_ = false;
    bool pending_ = false;
};

} // namespace q27::agent
