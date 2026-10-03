#include "sampling.h"

#include <cstdio>
#include <random>
#include <vector>

int main() {
    std::vector<float> logits = {-2.0f, 1.0f, 4.0f, 3.0f};
    q27::SamplingParams greedy;
    std::mt19937_64 rng(1);
    if (q27::sample_logits_cpu(logits, greedy, rng) != 2) return 1;

    q27::SamplingParams k1{0.8f, 1.0f, 1, 7};
    for (int i = 0; i < 20; i++)
        if (q27::sample_logits_cpu(logits, k1, rng) != 2) return 1;

    q27::SamplingParams sampled{1.0f, 0.9f, 3, 12345};
    std::mt19937_64 a(sampled.seed), b(sampled.seed);
    for (int i = 0; i < 100; i++)
        if (q27::sample_logits_cpu(logits, sampled, a) !=
            q27::sample_logits_cpu(logits, sampled, b))
            return 1;

    bool rejected = false;
    try {
        q27::sample_logits_cpu(logits, {1.0f, 0.0f, 0, 0}, rng);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) return 1;

    // NaNs, positive infinity, and fully masked rows cannot define a served
    // distribution. Negative infinity remains a valid mask when at least one
    // finite logit survives.
    {
        std::vector<uint32_t> ids = {0, 1, 2};
        auto rejected_by_all = [&](const std::vector<float>& invalid) {
            int throws = 0;
            try { (void)q27::sample_logits_cpu(invalid, greedy, rng); }
            catch (const std::runtime_error&) { throws++; }
            try { (void)q27::sample_candidates_cpu(invalid, ids, 3, greedy, rng); }
            catch (const std::runtime_error&) { throws++; }
            try { (void)q27::build_served_distribution(invalid, greedy); }
            catch (const std::runtime_error&) { throws++; }
            try { (void)q27::build_served_from_candidates(
                invalid.data(), ids.data(), 3, greedy); }
            catch (const std::runtime_error&) { throws++; }
            return throws == 4;
        };
        const float inf = std::numeric_limits<float>::infinity();
        if (!rejected_by_all({std::numeric_limits<float>::quiet_NaN(), 1.0f, -inf})) return 1;
        if (!rejected_by_all({inf, 1.0f, -inf})) return 1;
        if (!rejected_by_all({-inf, -inf, -inf})) return 1;
        std::vector<float> masked = {-inf, 1.0f};
        if (q27::sample_logits_cpu(masked, greedy, rng) != 1) return 1;
    }

    // A shuffled exact top-k over-set must consume RNG and select tokens
    // identically to the full-logits path.
    {
        const uint32_t n = 4096, k = 40, count = 57;
        std::vector<float> full(n);
        uint32_t state = 2468;
        for (float& value : full) {
            state = state * 1664525u + 1013904223u;
            value = (float)(state >> 8) / 8388608.0f * 12.0f - 6.0f;
        }
        std::vector<uint32_t> rank(n);
        for (uint32_t i = 0; i < n; i++) rank[i] = i;
        std::sort(rank.begin(), rank.end(), [&](uint32_t x, uint32_t y) {
            return full[x] != full[y] ? full[x] > full[y] : x < y;
        });
        std::vector<float> values(count);
        std::vector<uint32_t> indices(count);
        for (uint32_t i = 0; i < count; i++) {
            indices[i] = rank[(i * 17) % count];
            values[i] = full[indices[i]];
        }
        q27::SamplingParams params{0.9f, 0.95f, k, 777};
        std::mt19937_64 full_rng(params.seed), candidate_rng(params.seed);
        for (int i = 0; i < 100; i++) {
            uint32_t want = q27::sample_logits_cpu(full, params, full_rng);
            uint32_t got = q27::sample_candidates_cpu(
                values, indices, count, params, candidate_rng);
            if (got != want || !(candidate_rng == full_rng)) return 1;
        }
    }

    // Candidate over-sets are incomplete by definition when top_k is
    // disabled, so sampled paths must fall back to full logits instead of
    // silently excluding the rest of the vocabulary.
    {
        const std::vector<float> values = {4.0f, 3.0f, 2.0f};
        const std::vector<uint32_t> ids = {0, 1, 2};
        const q27::SamplingParams unlimited{1.0f, 1.0f, 0, 1};
        bool sample_rejected = false, served_rejected = false;
        try {
            std::mt19937_64 candidate_rng(unlimited.seed);
            (void)q27::sample_candidates_cpu(
                values, ids, (uint32_t)values.size(), unlimited, candidate_rng);
        } catch (const std::runtime_error&) {
            sample_rejected = true;
        }
        try {
            (void)q27::build_served_from_candidates(
                values.data(), ids.data(), (uint32_t)values.size(), unlimited);
        } catch (const std::runtime_error&) {
            served_rejected = true;
        }
        if (!sample_rejected || !served_rejected) return 1;
    }

    // Exact ties at the top-p cutoff all remain reachable and both sampling
    // paths map the same random draw to the same token.
    {
        std::vector<float> tied = {0.0f, 0.0f, 0.0f, -100.0f};
        std::vector<uint32_t> ids = {0, 1, 2, 3};
        q27::SamplingParams params{1.0f, 0.5f, 4, 9001};
        std::mt19937_64 full_rng(params.seed), candidate_rng(params.seed);
        bool seen[3] = {};
        for (int i = 0; i < 300; i++) {
            uint32_t want = q27::sample_logits_cpu(tied, params, full_rng);
            uint32_t got = q27::sample_candidates_cpu(
                tied, ids, (uint32_t)tied.size(), params, candidate_rng);
            if (got != want || got > 2) return 1;
            seen[got] = true;
        }
        if (!(seen[0] && seen[1] && seen[2])) return 1;
    }

    // Residual sampling never re-emits an excluded draft token.
    {
        auto distribution = q27::build_served_distribution(logits, {1.0f, 1.0f, 0, 0});
        std::mt19937_64 residual_rng(99);
        for (int i = 0; i < 500; i++)
            if (q27::sample_served(distribution, residual_rng, 2) == 2) return 1;
    }

    // The rejection walk commits the pending token plus accepted drafts,
    // stops on the first p=0 draft, and samples the next pending from that
    // lane's residual distribution.
    auto point_mass = [](uint32_t token) {
        q27::ServedDistribution d;
        d.argmax_token = token;
        d.tokens = {token};
        d.weights = {1.0};
        d.total = 1.0;
        return d;
    };
    {
        std::vector<q27::ServedDistribution> lanes = {point_mass(7), point_mass(8)};
        const uint32_t drafts[] = {99};
        std::mt19937_64 reject_rng(1);
        const auto result = q27::spec_rejection_accept(
            lanes.data(), (uint32_t)lanes.size(), drafts, reject_rng);
        if (result.n != 1 || result.stop_lane != 0 || result.exclude != 99 ||
            result.pending != 7)
            return 1;
    }
    {
        std::vector<q27::ServedDistribution> lanes = {
            point_mass(1), point_mass(7), point_mass(9)};
        const uint32_t drafts[] = {1, 99};
        std::mt19937_64 reject_rng(2);
        const auto result = q27::spec_rejection_accept(
            lanes.data(), (uint32_t)lanes.size(), drafts, reject_rng);
        if (result.n != 2 || result.stop_lane != 1 || result.exclude != 99 ||
            result.pending != 7)
            return 1;
    }

    // p=1 drafts all commit and the last verify lane supplies the free bonus.
    {
        const float masked = -std::numeric_limits<float>::infinity();
        const std::vector<float> lanes = {
            0.0f, masked, masked,
            masked, 0.0f, masked,
            masked, masked, 0.0f,
        };
        const std::vector<uint32_t> drafts = {0, 1};
        std::mt19937_64 accept_rng(3);
        const auto result = q27::spec_rejection_accept(
            lanes, 3, 3, drafts, {1.0f, 1.0f, 0, 0}, accept_rng);
        if (result.n != 3 || result.stop_lane != 2 || result.exclude != -1 ||
            result.pending != 2)
            return 1;
    }

    // Nucleus probabilities are renormalized over retained mass, and a
    // shuffled candidate-built distribution drives the same rejection walk.
    {
        const std::vector<float> full = {
            std::log(4.0f), std::log(3.0f), std::log(2.0f), 0.0f};
        const std::vector<uint32_t> ids = {2, 0, 3, 1};
        const std::vector<float> values = {full[2], full[0], full[3], full[1]};
        const q27::SamplingParams nucleus{1.0f, 0.5f, 4, 0};
        const auto full_dist = q27::build_served_distribution(full, nucleus);
        const auto candidate_dist = q27::build_served_from_candidates(
            values.data(), ids.data(), (uint32_t)ids.size(), nucleus);
        const double p0 = q27::served_probability(candidate_dist, 0);
        const double p1 = q27::served_probability(candidate_dist, 1);
        if (candidate_dist.tokens != full_dist.tokens ||
            candidate_dist.weights != full_dist.weights ||
            candidate_dist.total != full_dist.total ||
            std::fabs(p0 - 4.0 / 7.0) > 1e-6 ||
            std::fabs(p0 + p1 - 1.0) > 1e-12)
            return 1;
        std::vector<q27::ServedDistribution> lanes = {
            candidate_dist, point_mass(9)};
        const uint32_t drafts[] = {99};
        std::mt19937_64 candidate_rng(4);
        const auto result = q27::spec_rejection_accept(
            lanes.data(), (uint32_t)lanes.size(), drafts, candidate_rng);
        if (result.n != 1 || result.stop_lane != 0 || result.exclude != 99 ||
            (result.pending != 0 && result.pending != 1))
            return 1;
    }

    // Exactness with deterministic (point-mass) drafts, the suffix and MTP
    // case: the first sampled token is distributed as p0, and the token
    // after an accepted draft as p1. The draft is deliberately not argmax.
    {
        auto dist = [](std::vector<double> w) {
            q27::ServedDistribution d;
            for (uint32_t t = 0; t < w.size(); t++) d.tokens.push_back(t);
            d.weights = w;
            d.total = 1.0;
            d.argmax_token = 0;
            return d;
        };
        const std::vector<double> p0 = {0.5, 0.3, 0.15, 0.05};
        const std::vector<double> p1 = {0.1, 0.6, 0.2, 0.1};
        const std::vector<q27::ServedDistribution> lanes = {
            dist(p0), dist(p1), dist({0.25, 0.25, 0.25, 0.25})};
        const uint32_t drafts[] = {1, 2};
        std::mt19937_64 walk_rng(5);
        const int trials = 400000;
        std::vector<double> first(4, 0.0), second(4, 0.0), bonus(4, 0.0);
        double after_draft = 0.0, all_accept = 0.0;
        for (int i = 0; i < trials; i++) {
            const auto r = q27::spec_rejection_accept(lanes.data(), 3, drafts, walk_rng);
            const uint32_t t1 = r.n >= 2 ? drafts[0] : r.pending;
            first[t1] += 1.0;
            if (t1 == drafts[0]) {
                second[r.n >= 3 ? drafts[1] : r.pending] += 1.0;
                after_draft += 1.0;
            }
            if (r.n == 3) {
                bonus[r.pending] += 1.0;
                all_accept += 1.0;
            }
        }
        for (uint32_t t = 0; t < 4; t++)
            if (std::fabs(first[t] / trials - p0[t]) > 0.005 ||
                std::fabs(second[t] / after_draft - p1[t]) > 0.006 ||
                std::fabs(bonus[t] / all_accept - 0.25) > 0.015)
                return 1;
        // A draft outside the served support (p=0) always rejects, and the
        // residual is the whole distribution.
        const std::vector<q27::ServedDistribution> narrow = {
            dist({0.7, 0.3}), dist({0.5, 0.5})};
        const uint32_t outside[] = {3};
        std::vector<double> residual(2, 0.0);
        const int narrow_trials = 100000;
        for (int i = 0; i < narrow_trials; i++) {
            const auto r = q27::spec_rejection_accept(narrow.data(), 2, outside, walk_rng);
            if (r.n != 1 || r.exclude != 3 || r.pending > 1) return 1;
            residual[r.pending] += 1.0;
        }
        if (std::fabs(residual[0] / narrow_trials - 0.7) > 0.006) return 1;
    }

    std::puts("CPU sampling: PASS");
    return 0;
}
