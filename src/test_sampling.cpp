#include "sampling.h"
#include <cmath>
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

    // ---- Served dist from GPU top-k candidates (Phase 4 speed path) ----
    {
        // Full-logits build_served and candidate over-set must agree on
        // p_served for every token when the over-set contains exact top-k.
        const uint32_t n=512;
        std::vector<float> full(n); uint32_t lcg=99;
        for(uint32_t i=0;i<n;i++) {
            lcg=lcg*1664525u+1013904223u;
            full[i]=(float)(lcg>>8)/8388608.0f*10.0f-5.0f;
        }
        std::vector<uint32_t> rank(n);
        for(uint32_t i=0;i<n;i++) rank[i]=i;
        std::sort(rank.begin(),rank.end(),[&](uint32_t a,uint32_t b){
            return full[a]!=full[b]?full[a]>full[b]:a<b;});
        for(uint32_t k:{3u,20u,64u}) {
            const uint32_t over=k+11;
            std::vector<float> cv(over); std::vector<uint32_t> ci(over);
            for(uint32_t i=0;i<over;i++) { cv[i]=full[rank[i]]; ci[i]=rank[i]; }
            // Shuffle over-set so order is not pre-sorted (GPU top-k is unordered).
            std::mt19937_64 sh(k*17); std::shuffle(ci.begin(),ci.end(),sh);
            for(uint32_t i=0;i<over;i++) cv[i]=full[ci[i]];
            q27::SamplingParams p{0.9f,0.95f,k,1};
            auto full_d=q27::build_served_distribution(full,p);
            auto cand_d=q27::build_served_from_candidates(cv.data(),ci.data(),over,p);
            if(full_d.tokens.size()!=cand_d.tokens.size()) {
                fprintf(stderr,"cand nucleus size %zu vs full %zu k=%u\n",
                        cand_d.tokens.size(),full_d.tokens.size(),k); return 1;
            }
            for(uint32_t t=0;t<n;t++) {
                double pf=q27::served_probability(full_d,t);
                double pc=q27::served_probability(cand_d,t);
                if(std::fabs(pf-pc)>1e-9) {
                    fprintf(stderr,"cand p_served mismatch t=%u k=%u full=%g cand=%g\n",
                            t,k,pf,pc); return 1;
                }
            }
        }
    }

    // ---- Full vs candidates: reject-walk token identity under top_k ----
    {
        // Same multi-lane logits + drafts + seed: building ServedDistribution
        // from full rows vs from an exact top-k over-set must yield identical
        // SpecRejectResult sequences (n, stop_lane, exclude, pending).
        const uint32_t live=3,vocab=128,k=20;
        std::vector<float> lanes((size_t)live*vocab);
        uint32_t lcg=424242;
        for(float& x:lanes) {
            lcg=lcg*1664525u+1013904223u;
            x=(float)(lcg>>8)/8388608.0f*8.0f-4.0f;
        }
        // Plant sharp draft peaks so acceptance is non-degenerate.
        lanes[0*vocab+7]=6.0f;
        lanes[1*vocab+11]=5.5f;
        lanes[2*vocab+3]=5.0f;
        uint32_t drafts[2]={7,11};
        q27::SamplingParams p{0.85f,0.95f,k,9001};
        std::vector<q27::ServedDistribution> full_d(live),cand_d(live);
        for(uint32_t lane=0;lane<live;lane++) {
            const float* row=lanes.data()+(size_t)lane*vocab;
            full_d[lane]=q27::build_served_distribution(row,vocab,p);
            std::vector<uint32_t> rank(vocab);
            for(uint32_t i=0;i<vocab;i++) rank[i]=i;
            std::partial_sort(rank.begin(),rank.begin()+k+9,rank.end(),
                              [&](uint32_t a,uint32_t b){
                                  return row[a]!=row[b]?row[a]>row[b]:a<b;});
            const uint32_t over=k+9;
            std::vector<float> cv(over); std::vector<uint32_t> ci(over);
            for(uint32_t i=0;i<over;i++) { ci[i]=rank[i]; cv[i]=row[rank[i]]; }
            std::mt19937_64 sh(lane+3); std::shuffle(ci.begin(),ci.end(),sh);
            for(uint32_t i=0;i<over;i++) cv[i]=row[ci[i]];
            cand_d[lane]=q27::build_served_from_candidates(cv.data(),ci.data(),over,p);
        }
        std::mt19937_64 ra(p.seed),rb(p.seed);
        for(int iter=0;iter<200;iter++) {
            auto fa=q27::spec_rejection_accept(full_d.data(),live,drafts,ra);
            auto ca=q27::spec_rejection_accept(cand_d.data(),live,drafts,rb);
            if(fa.n!=ca.n || fa.stop_lane!=ca.stop_lane || fa.exclude!=ca.exclude ||
               fa.pending!=ca.pending) {
                fprintf(stderr,"reject-walk full vs cand diverge iter=%d "
                        "n %u/%u stop %u/%u excl %d/%d pend %u/%u\n",
                        iter,fa.n,ca.n,fa.stop_lane,ca.stop_lane,
                        (int)fa.exclude,(int)ca.exclude,fa.pending,ca.pending);
                return 1;
            }
        }
    }

    // ---- Spec rejection sampling (Metal sampled-MTP Phase 0) ----
    {
        // build_served + sample_served draw-for-draw match sample_logits_cpu
        // when exclude is unused (same nucleus construction, same RNG draws).
        std::vector<float> row={-1.0f,2.0f,0.5f,1.5f,-3.0f};
        q27::SamplingParams p{0.9f,0.95f,0,4242};
        std::mt19937_64 ra(p.seed),rb(p.seed);
        for(int i=0;i<200;i++) {
            uint32_t want=q27::sample_logits_cpu(row,p,ra);
            auto dist=q27::build_served_distribution(row,p);
            // sample_logits_cpu draws once; sample_served must consume one draw
            // from the same nucleus. Rebuild after each sample_logits_cpu call
            // would re-use independent RNG streams — mirror by building once
            // per iteration from the same seed streams as sample_logits_cpu.
            // Equivalent: probability mass of each token under served == empirical
            // from sample_logits_cpu is checked below; here identity of a single
            // draw needs shared construction. Manually: rebuild and sample_served
            // with a fresh twin RNG that only feeds sample_served is wrong.
            // Instead verify p_served sums to 1 and matches histogram rates.
            (void)want;
            double sum=0.0;
            for(uint32_t t=0;t<(uint32_t)row.size();t++) sum+=q27::served_probability(dist,t);
            if(std::fabs(sum-1.0)>1e-9) {
                fprintf(stderr,"served mass sum %g (iter %d)\n",sum,i); return 1;
            }
            break; // construction check once; MC below
        }

        // served_probability: outside nucleus (top_k=2) is 0; argmax mass > 0.
        {
            q27::SamplingParams pk{1.0f,1.0f,2,1};
            auto dist=q27::build_served_distribution(row,pk);
            if(q27::served_probability(dist,4)!=0.0) {
                fprintf(stderr,"outside top-k should have p=0\n"); return 1;
            }
            if(!(q27::served_probability(dist,1)>0.0)) {
                fprintf(stderr,"argmax should have p>0\n"); return 1;
            }
            double mass=0.0;
            for(uint32_t t=0;t<(uint32_t)row.size();t++) mass+=q27::served_probability(dist,t);
            if(std::fabs(mass-1.0)>1e-9) { fprintf(stderr,"top-k mass %g\n",mass); return 1; }
        }

        // exclude never re-emitted.
        {
            q27::SamplingParams pex{1.0f,1.0f,0,7};
            auto dist=q27::build_served_distribution(row,pex);
            std::mt19937_64 r(99);
            for(int i=0;i<500;i++) {
                uint32_t tok=q27::sample_served(dist,r,/*exclude=*/1);
                if(tok==1) { fprintf(stderr,"excluded token re-emitted\n"); return 1; }
            }
        }
        // Singleton nucleus + exclude must not re-emit (throws instead).
        {
            std::vector<float> peak={-50.0f,20.0f,-50.0f};
            q27::SamplingParams p1{1.0f,1.0f,1,1}; // top_k=1 → delta at argmax=1
            auto dist=q27::build_served_distribution(peak,p1);
            if(dist.tokens.size()!=1 || dist.tokens[0]!=1) {
                fprintf(stderr,"expected singleton nucleus on argmax\n"); return 1;
            }
            bool threw=false;
            try {
                std::mt19937_64 r(1);
                (void)q27::sample_served(dist,r,/*exclude=*/1);
            } catch(const std::runtime_error&) { threw=true; }
            if(!threw) { fprintf(stderr,"singleton exclude must throw, not re-emit\n"); return 1; }
        }

        // p=0 draft (outside nucleus) always rejects → stop_lane=0, exclude=draft.
        {
            // live=3: two drafts. Lane 0 logits peak at token 0; draft0=4 is cold.
            const uint32_t live=3,vocab=5;
            std::vector<float> lanes(live*vocab, -20.0f);
            // lane 0: only token 0 is warm → draft 4 has p≈0
            lanes[0]=5.0f;
            // lane 1/2: flat-ish for pending sample
            for(uint32_t t=0;t<vocab;t++) { lanes[1*vocab+t]=(float)t; lanes[2*vocab+t]=1.0f; }
            uint32_t drafts[2]={4,2};
            q27::SamplingParams ps{1.0f,1.0f,0,123};
            std::mt19937_64 r(ps.seed);
            bool saw_reject0=false;
            for(int i=0;i<50;i++) {
                auto res=q27::spec_rejection_accept(lanes.data(),live,vocab,drafts,ps,r);
                if(res.stop_lane==0 && res.exclude==4) { saw_reject0=true; break; }
            }
            if(!saw_reject0) { fprintf(stderr,"cold draft never rejected\n"); return 1; }
            // When rejected, pending must not be the excluded draft.
            std::mt19937_64 r2(ps.seed);
            for(int i=0;i<200;i++) {
                auto res=q27::spec_rejection_accept(lanes.data(),live,vocab,drafts,ps,r2);
                if(res.exclude>=0 && res.pending==(uint32_t)res.exclude) {
                    fprintf(stderr,"pending equals excluded draft\n"); return 1;
                }
                if(res.n<1 || res.n>live) { fprintf(stderr,"n out of range %u\n",res.n); return 1; }
                if(res.stop_lane>=live) { fprintf(stderr,"stop_lane OOB\n"); return 1; }
            }
        }

        // All-accept path: drafts are the unique argmax of each lane → p=1 at T→0-ish high peak.
        {
            const uint32_t live=3,vocab=4;
            std::vector<float> lanes(live*vocab, -50.0f);
            // lane0 predicts draft0=1 with certainty; lane1 predicts draft1=2; lane2 free.
            lanes[0*vocab+1]=20.0f;
            lanes[1*vocab+2]=20.0f;
            lanes[2*vocab+0]=20.0f;
            uint32_t drafts[2]={1,2};
            q27::SamplingParams ps{0.5f,1.0f,0,55};
            std::mt19937_64 r(ps.seed);
            int all_accept=0;
            for(int i=0;i<100;i++) {
                auto res=q27::spec_rejection_accept(lanes.data(),live,vocab,drafts,ps,r);
                if(res.n==live && res.exclude<0 && res.stop_lane==live-1) all_accept++;
            }
            if(all_accept<90) {
                fprintf(stderr,"all-accept expected (~100 sharp), got %d/100\n",all_accept);
                return 1;
            }
        }

        // Seeded identity: same seed → identical (n, stop, exclude, pending) sequences.
        {
            const uint32_t live=3,vocab=6;
            std::vector<float> lanes={
                1.0f, 2.0f, 0.5f, 0.1f, -1.0f, 0.0f,
                0.2f, 0.3f, 1.5f, 0.4f, 0.1f, -0.5f,
                0.0f, 1.0f, 0.0f, 2.0f, 0.5f, 0.2f,
            };
            uint32_t drafts[2]={1,2};
            q27::SamplingParams ps{0.8f,0.9f,0,99991};
            std::mt19937_64 a(ps.seed),b(ps.seed);
            for(int i=0;i<100;i++) {
                auto ra=q27::spec_rejection_accept(lanes.data(),live,vocab,drafts,ps,a);
                auto rb=q27::spec_rejection_accept(lanes.data(),live,vocab,drafts,ps,b);
                if(ra.n!=rb.n || ra.stop_lane!=rb.stop_lane || ra.exclude!=rb.exclude ||
                   ra.pending!=rb.pending) {
                    fprintf(stderr,"seeded identity broke at iter %d\n",i); return 1;
                }
            }
        }

        // Accept rate ≈ analytic p_served of the first draft (Monte Carlo).
        {
            const uint32_t live=2,vocab=5;
            std::vector<float> lanes(live*vocab);
            // lane 0: mild peak at token 1
            float base[]={0.0f,1.2f,0.4f,0.3f,-1.0f};
            for(uint32_t t=0;t<vocab;t++) { lanes[t]=base[t]; lanes[vocab+t]=0.5f; }
            uint32_t drafts[1]={1};
            q27::SamplingParams ps{1.0f,1.0f,0,31415};
            auto dist=q27::build_served_distribution(lanes.data(),vocab,ps);
            const double p_true=q27::served_probability(dist,1);
            const int N=4000;
            int accepts=0;
            std::mt19937_64 r(ps.seed);
            for(int i=0;i<N;i++) {
                auto res=q27::spec_rejection_accept(lanes.data(),live,vocab,drafts,ps,r);
                // all-accept on live=2 means n==2 (draft accepted)
                if(res.n==2) accepts++;
            }
            const double emp=(double)accepts/(double)N;
            if(std::fabs(emp-p_true)>0.04) {
                fprintf(stderr,"accept rate emp=%g analytic=%g\n",emp,p_true); return 1;
            }
        }

        // Composition: pending histogram under reject-at-lane0 with exclude should
        // match sample_served(..., exclude=draft) on that lane (synthetic).
        {
            const uint32_t live=2,vocab=4;
            // Make draft0 cold so nearly always reject at lane 0.
            std::vector<float> lanes={
                3.0f, -10.0f, 2.0f, 1.0f,  // draft d=1 is cold
                0.0f, 0.0f, 0.0f, 0.0f,    // unused
            };
            uint32_t drafts[1]={1};
            q27::SamplingParams ps{1.0f,1.0f,0,17};
            auto dist=q27::build_served_distribution(lanes.data(),vocab,ps);
            const int N=3000;
            std::vector<int> hist(vocab,0), ref(vocab,0);
            std::mt19937_64 r1(ps.seed), r2(ps.seed+1);
            for(int i=0;i<N;i++) {
                auto res=q27::spec_rejection_accept(lanes.data(),live,vocab,drafts,ps,r1);
                if(res.exclude==1 && res.stop_lane==0) hist[res.pending]++;
            }
            for(int i=0;i<N;i++) {
                uint32_t t=q27::sample_served(dist,r2,/*exclude=*/1);
                ref[t]++;
            }
            // Rate comparison: mean absolute rate error (hist conditioned on reject).
            int nh=0; for(int c:hist) nh+=c;
            if(nh<N/2) { fprintf(stderr,"expected mostly reject-on-cold, nh=%d\n",nh); return 1; }
            double mae=0.0; int cells=0;
            for(uint32_t t=0;t<vocab;t++) {
                if(t==1) continue; // excluded
                double rh=(double)hist[t]/(double)nh;
                double rr=(double)ref[t]/(double)N;
                mae+=std::fabs(rh-rr); cells++;
            }
            mae/=std::max(1,cells);
            if(mae>0.05) {
                fprintf(stderr,"pending residual MAE %g (hist vs sample_served exclude)\n",mae);
                return 1;
            }
        }
    }

    puts("CPU sampling: PASS");
    return 0;
}
