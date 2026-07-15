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
    const size_t keep=p.top_k?std::min<size_t>(p.top_k,order.size()):order.size();
    if(keep<order.size()) {
        std::partial_sort(order.begin(),order.begin()+keep,order.end(),
                          [&](uint32_t a,uint32_t b){return logits[a]>logits[b];});
        order.resize(keep);
    } else std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){return logits[a]>logits[b];});

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
        order.resize(retained); weights.resize(retained); total=cumulative;
    }
    std::uniform_real_distribution<double> distribution(0.0,total);
    double draw=distribution(random);
    for(size_t i=0;i<order.size();i++) { draw-=weights[i]; if(draw<=0.0) return indices[order[i]]; }
    return indices[order.back()];
}

} // namespace q27
