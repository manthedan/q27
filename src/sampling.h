#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace q27 {

struct SamplingParams {
    float temperature = 0.0f;
    float top_p = 1.0f;
    uint32_t top_k = 0;
    uint64_t seed = 0;
};

inline void validate_sampling(const SamplingParams& p) {
    if(!std::isfinite(p.temperature) || p.temperature<0.0f)
        throw std::runtime_error("q27: temperature must be finite and non-negative");
    if(!std::isfinite(p.top_p) || p.top_p<=0.0f || p.top_p>1.0f)
        throw std::runtime_error("q27: top_p must be in (0,1]");
}

inline uint32_t sample_logits_cpu(const std::vector<float>& logits,const SamplingParams& p,
                                  std::mt19937_64& random) {
    validate_sampling(p);
    if(logits.empty()) throw std::runtime_error("q27: logits are empty");
    auto argmax=[&] {
        return (uint32_t)std::distance(logits.begin(),std::max_element(logits.begin(),logits.end()));
    };
    if(p.temperature==0.0f || p.top_k==1) return argmax();

    std::vector<uint32_t> order(logits.size());
    for(uint32_t i=0;i<order.size();i++) order[i]=i;
    // Ties break index-ascending (codex P2 on the boundary-tie fix): the
    // candidates path already orders ties by index, and with boundary ties
    // now inside the nucleus the two paths must map the same draw to the
    // same token — unspecified sort order here would break that.
    auto before=[&](uint32_t a,uint32_t b){return logits[a]!=logits[b]?logits[a]>logits[b]:a<b;};
    const size_t keep=p.top_k?std::min<size_t>(p.top_k,order.size()):order.size();
    if(keep<order.size()) {
        std::partial_sort(order.begin(),order.begin()+keep,order.end(),before);
        order.resize(keep);
    } else std::sort(order.begin(),order.end(),before);

    const double maximum=logits[order.front()]/(double)p.temperature;
    std::vector<double> weights; weights.reserve(order.size()); double total=0.0;
    for(uint32_t token:order) {
        double weight=std::exp(logits[token]/(double)p.temperature-maximum);
        if(!std::isfinite(weight)) weight=0.0;
        weights.push_back(weight); total+=weight;
    }
    if(!(total>0.0)) return argmax();
    if(p.top_p<1.0f) {
        const double cutoff=total*p.top_p; double cumulative=0.0; size_t retained=0;
        do { cumulative+=weights[retained++]; } while(retained<weights.size() && cumulative<cutoff);
        // Boundary ties stay in the nucleus (parity audit 2026-07-17): the
        // CUDA sampler keeps every token at or above its threshold value, so
        // exact ties at the truncation boundary must not be dropped by sort
        // order here. Extends through logits exactly equal to the boundary.
        while(retained<weights.size() && logits[order[retained]]==logits[order[retained-1]])
            cumulative+=weights[retained++];
        order.resize(retained); weights.resize(retained); total=cumulative;
    }
    std::uniform_real_distribution<double> distribution(0.0,total);
    double draw=distribution(random);
    for(size_t i=0;i<order.size();i++) { draw-=weights[i]; if(draw<=0.0) return order[i]; }
    return order.back();
}

// GPU-assisted sampling: sample from a candidate over-set — (value, index)
// pairs guaranteed to contain the exact top-k — instead of the full logits
// vector. Arithmetic mirrors sample_logits_cpu exactly (same descending
// order over distinct values, same double-precision weight accumulation,
// same single uniform draw), so same-seed token sequences match the
// full-logits path whenever no exact-float tie crosses the top-k boundary.
// Ties inside the candidate list break index-ascending — a deterministic
// refinement of std::partial_sort's unspecified tie order above.
inline uint32_t sample_candidates_cpu(const std::vector<float>& values,
                                      const std::vector<uint32_t>& indices,
                                      uint32_t count,const SamplingParams& p,
                                      std::mt19937_64& random) {
    validate_sampling(p);
    if(!count) throw std::runtime_error("q27: candidate list is empty");
    if(values.size()<count || indices.size()<count)
        throw std::runtime_error("q27: candidate list shorter than count");
    std::vector<uint32_t> order(count);
    for(uint32_t i=0;i<count;i++) order[i]=i;
    auto before=[&](uint32_t a,uint32_t b) {
        return values[a]!=values[b] ? values[a]>values[b] : indices[a]<indices[b];
    };
    auto argmax=[&] {
        return indices[*std::min_element(order.begin(),order.end(),before)];
    };
    if(p.temperature==0.0f || p.top_k==1) return argmax();

    std::sort(order.begin(),order.end(),before);
    const size_t keep=p.top_k?std::min<size_t>(p.top_k,count):count;
    order.resize(keep);
    const double maximum=values[order.front()]/(double)p.temperature;
    std::vector<double> weights; weights.reserve(order.size()); double total=0.0;
    for(uint32_t slot:order) {
        double weight=std::exp(values[slot]/(double)p.temperature-maximum);
        if(!std::isfinite(weight)) weight=0.0;
        weights.push_back(weight); total+=weight;
    }
    if(!(total>0.0)) return argmax();
    if(p.top_p<1.0f) {
        const double cutoff=total*p.top_p; double cumulative=0.0; size_t retained=0;
        do { cumulative+=weights[retained++]; } while(retained<weights.size() && cumulative<cutoff);
        // Boundary ties stay in the nucleus — same rule as sample_logits_cpu
        // (parity audit 2026-07-17), keyed on the exact candidate values.
        while(retained<weights.size() && values[order[retained]]==values[order[retained-1]])
            cumulative+=weights[retained++];
        order.resize(retained); weights.resize(retained); total=cumulative;
    }
    std::uniform_real_distribution<double> distribution(0.0,total);
    double draw=distribution(random);
    for(size_t i=0;i<order.size();i++) { draw-=weights[i]; if(draw<=0.0) return indices[order[i]]; }
    return indices[order.back()];
}

// ---- Spec rejection sampling (Metal sampled-MTP Phase 0) --------------------
// Host-side Leviathan/Chen accept walk with greedy drafts (q = delta at draft).
// Served p matches sample_logits_cpu (temp / top_p / top_k, boundary ties).
// Used by Metal mtp_sample_round; CUDA already has device kernels for this.
// See docs/metal/plans/2026-07-21-metal-sampled-mtp.md.

// Unnormalized nucleus weights for one lane's logits under SamplingParams.
// tokens[i] is a vocab id; weights[i] > 0; total = sum(weights). Empty when
// the nucleus collapses (caller falls back to argmax).
struct ServedDistribution {
    std::vector<uint32_t> tokens;
    std::vector<double> weights;
    double total = 0.0;
    uint32_t argmax_token = 0;
};

// Build the served nucleus from a contiguous logits row (length vocab).
// temperature==0 or top_k==1 → delta at argmax (total=1, single token).
inline ServedDistribution build_served_distribution(const float* logits,uint32_t vocab,
                                                    const SamplingParams& p) {
    validate_sampling(p);
    if(!logits || !vocab) throw std::runtime_error("q27: logits/vocab empty for served dist");
    ServedDistribution d;
    uint32_t argmax = 0;
    float best = logits[0];
    for(uint32_t i=1;i<vocab;i++) {
        if(logits[i]>best) { best=logits[i]; argmax=i; }
    }
    d.argmax_token = argmax;
    if(p.temperature==0.0f || p.top_k==1) {
        d.tokens.push_back(argmax);
        d.weights.push_back(1.0);
        d.total = 1.0;
        return d;
    }
    std::vector<uint32_t> order(vocab);
    for(uint32_t i=0;i<vocab;i++) order[i]=i;
    auto before=[&](uint32_t a,uint32_t b){
        return logits[a]!=logits[b]?logits[a]>logits[b]:a<b;
    };
    const size_t keep=p.top_k?std::min<size_t>(p.top_k,order.size()):order.size();
    if(keep<order.size()) {
        std::partial_sort(order.begin(),order.begin()+keep,order.end(),before);
        order.resize(keep);
    } else std::sort(order.begin(),order.end(),before);

    const double maximum=logits[order.front()]/(double)p.temperature;
    d.weights.reserve(order.size());
    d.tokens.reserve(order.size());
    for(uint32_t token:order) {
        double weight=std::exp(logits[token]/(double)p.temperature-maximum);
        if(!std::isfinite(weight)) weight=0.0;
        d.tokens.push_back(token);
        d.weights.push_back(weight);
        d.total+=weight;
    }
    if(!(d.total>0.0)) {
        d.tokens={argmax}; d.weights={1.0}; d.total=1.0;
        return d;
    }
    if(p.top_p<1.0f) {
        const double cutoff=d.total*p.top_p; double cumulative=0.0; size_t retained=0;
        do { cumulative+=d.weights[retained++]; }
        while(retained<d.weights.size() && cumulative<cutoff);
        while(retained<d.weights.size() &&
              logits[d.tokens[retained]]==logits[d.tokens[retained-1]])
            cumulative+=d.weights[retained++];
        d.tokens.resize(retained); d.weights.resize(retained); d.total=cumulative;
    }
    return d;
}

inline ServedDistribution build_served_distribution(const std::vector<float>& logits,
                                                    const SamplingParams& p) {
    return build_served_distribution(logits.data(),(uint32_t)logits.size(),p);
}

// Served nucleus from a GPU top-k over-set (value, index pairs containing the
// true top-k). Arithmetic mirrors sample_candidates_cpu so accept probs match
// plain sampling under the same top_k. Used by Metal mtp_sample_round to avoid
// full-vocab readback when top_k is in 1..256.
inline ServedDistribution build_served_from_candidates(const float* values,
                                                       const uint32_t* indices,
                                                       uint32_t count,
                                                       const SamplingParams& p) {
    validate_sampling(p);
    if(!values || !indices || !count)
        throw std::runtime_error("q27: empty candidates for served dist");
    ServedDistribution d;
    std::vector<uint32_t> order(count);
    for(uint32_t i=0;i<count;i++) order[i]=i;
    auto before=[&](uint32_t a,uint32_t b) {
        return values[a]!=values[b] ? values[a]>values[b] : indices[a]<indices[b];
    };
    d.argmax_token = indices[*std::min_element(order.begin(),order.end(),before)];
    if(p.temperature==0.0f || p.top_k==1) {
        d.tokens.push_back(d.argmax_token);
        d.weights.push_back(1.0);
        d.total = 1.0;
        return d;
    }
    std::sort(order.begin(),order.end(),before);
    const size_t keep=p.top_k?std::min<size_t>(p.top_k,count):count;
    order.resize(keep);
    const double maximum=values[order.front()]/(double)p.temperature;
    d.tokens.reserve(keep); d.weights.reserve(keep);
    for(uint32_t slot:order) {
        double weight=std::exp(values[slot]/(double)p.temperature-maximum);
        if(!std::isfinite(weight)) weight=0.0;
        d.tokens.push_back(indices[slot]);
        d.weights.push_back(weight);
        d.total+=weight;
    }
    if(!(d.total>0.0)) {
        d.tokens={d.argmax_token}; d.weights={1.0}; d.total=1.0;
        return d;
    }
    if(p.top_p<1.0f) {
        const double cutoff=d.total*p.top_p; double cumulative=0.0; size_t retained=0;
        do { cumulative+=d.weights[retained++]; }
        while(retained<d.weights.size() && cumulative<cutoff);
        while(retained<d.weights.size() &&
              values[order[retained]]==values[order[retained-1]])
            cumulative+=d.weights[retained++];
        d.tokens.resize(retained); d.weights.resize(retained); d.total=cumulative;
    }
    return d;
}

// p_served(token) under the nucleus; 0 if outside. Uses mass renormalization
// (weight/total) — required for top_p<1 accept tests (CUDA mass fix).
inline double served_probability(const ServedDistribution& d,uint32_t token) {
    if(!(d.total>0.0)) return 0.0;
    for(size_t i=0;i<d.tokens.size();i++)
        if(d.tokens[i]==token) return d.weights[i]/d.total;
    return 0.0;
}

// Sample from the served nucleus, optionally excluding one token (residual
// resample after a rejected draft). Never returns `exclude`. If exclude
// removes all mass (singleton nucleus on the rejected draft), throws —
// re-emitting the excluded token would violate the residual distribution
// (codex P2 on Phase 0). Callers that need a soft fallback must widen the
// nucleus (e.g. drop top_k) before sampling; the Metal sample round keeps
// vocab-wide logits and only hits this when p(draft) was already ~1.
inline uint32_t sample_served(const ServedDistribution& d,std::mt19937_64& random,
                              int32_t exclude=-1) {
    if(d.tokens.empty()) throw std::runtime_error("q27: empty served distribution");
    double total=0.0;
    for(size_t i=0;i<d.tokens.size();i++) {
        if(exclude>=0 && (int32_t)d.tokens[i]==exclude) continue;
        total+=d.weights[i];
    }
    if(!(total>0.0)) {
        if(exclude>=0)
            throw std::runtime_error("q27: empty residual after excluding draft from nucleus");
        return d.argmax_token;
    }
    std::uniform_real_distribution<double> distribution(0.0,total);
    double draw=distribution(random);
    for(size_t i=0;i<d.tokens.size();i++) {
        if(exclude>=0 && (int32_t)d.tokens[i]==exclude) continue;
        draw-=d.weights[i];
        if(draw<=0.0) return d.tokens[i];
    }
    // Numeric tail: last non-excluded (total>0 guarantees one exists).
    for(size_t i=d.tokens.size();i-- > 0;) {
        if(exclude<0 || (int32_t)d.tokens[i]!=exclude) return d.tokens[i];
    }
    throw std::runtime_error("q27: sample_served internal: no non-excluded token");
}

// Result of one speculative verify-tail rejection walk.
// n = committed count (pending + accepted drafts), in 1..live.
// stop_lane = lane whose logits produce the next pending (0..live-1).
// exclude = rejected draft token, or -1 on all-accept (bonus sample).
// pending = newly sampled next-round pending token.
struct SpecRejectResult {
    uint32_t n = 1;
    uint32_t stop_lane = 0;
    int32_t exclude = -1;
    uint32_t pending = 0;
};

// Walk using pre-built per-lane served distributions (full-logits or top-k
// candidates). draft_tokens length live-1. Pending is NOT re-sampled.
// temperature==0 callers should use equality accept, not this path.
inline SpecRejectResult spec_rejection_accept(const ServedDistribution* lane_dists,
                                              uint32_t live,
                                              const uint32_t* draft_tokens,
                                              std::mt19937_64& random) {
    if(!lane_dists || live<2)
        throw std::runtime_error("q27: spec rejection needs live>=2 lanes");
    if(!draft_tokens)
        throw std::runtime_error("q27: spec rejection needs draft tokens");
    const uint32_t max_draft = live - 1;
    SpecRejectResult r;
    r.stop_lane = max_draft; // all-accept → free bonus on last lane
    r.exclude = -1;
    for(uint32_t k=0;k<max_draft;k++) {
        const double p = served_probability(lane_dists[k],draft_tokens[k]);
        // u ~ U[0,1). Accept when u < p (CUDA k_spec_accept). p==0 ⇒ always reject.
        std::uniform_real_distribution<double> unit(0.0,1.0);
        const double u = unit(random);
        if(u < p) continue;
        r.stop_lane = k;
        r.exclude = (int32_t)draft_tokens[k];
        break;
    }
    r.n = r.stop_lane + 1;
    r.pending = sample_served(lane_dists[r.stop_lane],random,r.exclude);
    return r;
}

// lanes_logits: contiguous [live * vocab] floats from multi-lane verify.
// draft_tokens: length live-1 — greedy proposals (lanes[1..live-1]).
inline SpecRejectResult spec_rejection_accept(const float* lanes_logits,uint32_t live,
                                              uint32_t vocab,
                                              const uint32_t* draft_tokens,
                                              const SamplingParams& params,
                                              std::mt19937_64& random) {
    validate_sampling(params);
    if(!lanes_logits || live<2)
        throw std::runtime_error("q27: spec rejection needs live>=2 lanes");
    if(!draft_tokens)
        throw std::runtime_error("q27: spec rejection needs draft tokens");
    std::vector<ServedDistribution> dists(live);
    for(uint32_t k=0;k<live;k++)
        dists[k]=build_served_distribution(lanes_logits+(size_t)k*vocab,vocab,params);
    return spec_rejection_accept(dists.data(),live,draft_tokens,random);
}

// Convenience overload: drafts as vector; lanes as flat vector length live*vocab.
inline SpecRejectResult spec_rejection_accept(const std::vector<float>& lanes_logits,
                                              uint32_t live,uint32_t vocab,
                                              const std::vector<uint32_t>& drafts,
                                              const SamplingParams& params,
                                              std::mt19937_64& random) {
    if(drafts.size()+1!=live)
        throw std::runtime_error("q27: drafts size must be live-1");
    if(lanes_logits.size()<(size_t)live*vocab)
        throw std::runtime_error("q27: lanes_logits shorter than live*vocab");
    return spec_rejection_accept(lanes_logits.data(),live,vocab,drafts.data(),params,random);
}

} // namespace q27
