#include "sampling.h"
#include <cstdio>
#include <random>
#include <vector>

int main() {
    std::vector<float> logits={-2,1,4,3};
    q27::SamplingParams greedy; std::mt19937_64 rng(1);
    if(q27::sample_logits_cpu(logits,greedy,rng)!=2) return 1;
    q27::SamplingParams k1{0.8f,1.0f,1,7};
    for(int i=0;i<20;i++) if(q27::sample_logits_cpu(logits,k1,rng)!=2) return 1;
    q27::SamplingParams sampled{1.0f,0.9f,3,12345};
    std::mt19937_64 a(sampled.seed),b(sampled.seed);
    for(int i=0;i<100;i++) if(q27::sample_logits_cpu(logits,sampled,a)!=q27::sample_logits_cpu(logits,sampled,b)) return 1;
    bool rejected=false; try { q27::SamplingParams bad{1.0f,0.0f,0,0}; q27::sample_logits_cpu(logits,bad,rng); }
    catch(const std::runtime_error&) { rejected=true; }
    if(!rejected) return 1;

    // GPU-assisted sampling: sample_candidates_cpu on a shuffled exact
    // top-k over-set must match sample_logits_cpu on the full vector,
    // token-for-token and with identical RNG consumption, across
    // temperatures, top-p cutoffs, top-k widths, and repeated draws.
    {
        const uint32_t n=4096;
        std::vector<float> full(n); uint32_t lcg=2468;
        for(uint32_t i=0;i<n;i++) { lcg=lcg*1664525u+1013904223u; full[i]=(float)(lcg>>8)/8388608.0f*12.0f-6.0f; }
        std::vector<uint32_t> rank(n);
        for(uint32_t i=0;i<n;i++) rank[i]=i;
        std::sort(rank.begin(),rank.end(),[&](uint32_t a,uint32_t b){
            return full[a]!=full[b]?full[a]>full[b]:a<b;});
        const float temps[]={0.0f,0.7f,1.0f,1.3f};
        const float tops[]={0.9f,0.95f,1.0f};
        const uint32_t ks[]={1,3,40,256};
        for(float t:temps) for(float tp:tops) for(uint32_t k:ks) {
            const uint32_t cnt=k+17; // over-set: exact top-k plus sub-boundary extras
            std::vector<uint32_t> perm(cnt);
            for(uint32_t i=0;i<cnt;i++) perm[i]=i;
            std::mt19937_64 shuf(4242); std::shuffle(perm.begin(),perm.end(),shuf);
            std::vector<float> cv(cnt); std::vector<uint32_t> ci(cnt);
            for(uint32_t i=0;i<cnt;i++) { cv[i]=full[rank[perm[i]]]; ci[i]=rank[perm[i]]; }
            q27::SamplingParams p{t,tp,k,777};
            std::mt19937_64 ra(p.seed),rb(p.seed);
            for(int iter=0;iter<25;iter++) {
                uint32_t want=q27::sample_logits_cpu(full,p,ra);
                uint32_t got=q27::sample_candidates_cpu(cv,ci,cnt,p,rb);
                if(want!=got) { fprintf(stderr,"candidates mismatch t=%g tp=%g k=%u iter=%d: %u vs %u\n",(double)t,(double)tp,k,iter,want,got); return 1; }
                if(!(ra==rb)) { fprintf(stderr,"candidates rng divergence t=%g tp=%g k=%u iter=%d\n",(double)t,(double)tp,k,iter); return 1; }
            }
        }
        bool short_rejected=false;
        try { std::mt19937_64 r(1); q27::sample_candidates_cpu({1.0f},{0},2,q27::SamplingParams{1.0f,1.0f,1,0},r); }
        catch(const std::runtime_error&) { short_rejected=true; }
        if(!short_rejected) return 1;
    }

    // Top-p boundary ties stay in the nucleus (parity audit 2026-07-17,
    // CUDA keeps every token at or above its threshold): three exact-tie
    // logits at top_p=0.5 must ALL be reachable — the pre-fix prefix
    // truncation kept two and could never sample the third.
    {
        std::vector<float> tied={0.0f,0.0f,0.0f,-100.0f};
        q27::SamplingParams p{1.0f,0.5f,0,9001};
        bool seen[3]={false,false,false};
        std::mt19937_64 r(p.seed);
        for(int i=0;i<300;i++) {
            uint32_t tok=q27::sample_logits_cpu(tied,p,r);
            if(tok>2) { fprintf(stderr,"top-p tie: sampled outside the tied set (%u)\n",tok); return 1; }
            seen[tok]=true;
        }
        if(!(seen[0]&&seen[1]&&seen[2])) { fprintf(stderr,"top-p tie: a boundary-tied token is unreachable\n"); return 1; }
        std::vector<uint32_t> ids={0,1,2,3};
        std::mt19937_64 r2(p.seed);
        bool cseen[3]={false,false,false};
        for(int i=0;i<300;i++) {
            uint32_t tok=q27::sample_candidates_cpu(tied,ids,4,p,r2);
            if(tok>2) { fprintf(stderr,"top-p tie (candidates): sampled outside the tied set (%u)\n",tok); return 1; }
            cseen[tok]=true;
        }
        if(!(cseen[0]&&cseen[1]&&cseen[2])) { fprintf(stderr,"top-p tie (candidates): a boundary-tied token is unreachable\n"); return 1; }
        // Draw-for-draw identity across the two paths WITH ties in the
        // nucleus (codex P2): same seed, same token, every draw — requires
        // the full path's index-ascending tie order.
        std::mt19937_64 rf(p.seed),rc(p.seed);
        for(int i=0;i<300;i++) {
            uint32_t want=q27::sample_logits_cpu(tied,p,rf);
            uint32_t got=q27::sample_candidates_cpu(tied,ids,4,p,rc);
            if(want!=got) { fprintf(stderr,"top-p tie: paths diverge at iter %d (%u vs %u)\n",i,want,got); return 1; }
        }
    }
    puts("CPU sampling: PASS");
    return 0;
}
