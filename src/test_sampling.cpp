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
    puts("CPU sampling: PASS");
    return 0;
}
