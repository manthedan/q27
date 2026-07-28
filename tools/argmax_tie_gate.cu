// Standalone argmax tie-break gate. Synthetic logits only -- no model, no
// weights. Proves the PR3 rule on real sm_86 hardware: exact-value ties must
// resolve to the LOWEST index (matching CPU max_element and Metal argmax).
#include "blocks.cuh"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));return 2;}}while(0)
static std::vector<float> rand_vec(int64_t n, uint32_t seed){
    std::mt19937 rng(seed); std::normal_distribution<float> d(0.f,1.f);
    std::vector<float> v(n); for(auto&x:v) x=d(rng); return v;
}
int main(){
    const int N=248320;
    float *d_x,*d_margin,*d_blk2; int *d_tok,*d_tok2;
    unsigned long long *d_blk1,*d_scr;
    CK(cudaMalloc(&d_x,(size_t)N*4)); CK(cudaMalloc(&d_margin,4));
    CK(cudaMalloc(&d_tok,4)); CK(cudaMalloc(&d_tok2,4));
    CK(cudaMalloc(&d_blk1,128*8)); CK(cudaMalloc(&d_blk2,128*4)); CK(cudaMalloc(&d_scr,8));
    double worst_tie=0; int cases=0;
    auto run=[&](std::vector<float>& lg,const char* label){
        int lowest=0; for(int i=1;i<N;i++) if(lg[i]>lg[lowest]) lowest=i;
        cudaMemcpy(d_x,lg.data(),(size_t)N*4,cudaMemcpyHostToDevice);
        q27k::argmax(d_x,N,d_tok2,d_scr,0);
        int atok; cudaMemcpy(&atok,d_tok2,4,cudaMemcpyDeviceToHost);
        q27k::argmax_margin(d_x,N,d_tok,d_margin,d_blk1,d_blk2,0);
        int ftok; cudaMemcpy(&ftok,d_tok,4,cudaMemcpyDeviceToHost);
        double d1=std::abs(atok-lowest), d2=std::abs(ftok-lowest);
        worst_tie=std::max(worst_tie,std::max(d1,d2)); cases++;
        printf("  %-24s argmax=%-7d fused=%-7d cpu_lowest=%-7d %s\n",
               label,atok,ftok,lowest,(d1==0&&d2==0)?"ok":"MISMATCH");
    };
    for(int s:{5,71,999}){ auto l=rand_vec(N,s); run(l,"random"); }
    { std::vector<float> l(N,1.5f); run(l,"all-equal ties"); }
    { auto l=rand_vec(N,13); l[0]=1e4f; run(l,"max@0"); }
    { auto l=rand_vec(N,17); l[N-1]=1e4f; run(l,"max@last"); }
    { auto l=rand_vec(N,23); l[100]=5e3f; l[200000]=5e3f; run(l,"dup max (far apart)"); }
    { auto l=rand_vec(N,29); l[7]=5e3f; l[8]=5e3f; l[N-2]=5e3f; run(l,"triple tie, adjacent"); }
    { std::vector<float> l(N,-1.0f); l[0]=-0.0f; l[256]=+0.0f; run(l,"-0 then +0"); }
    { std::vector<float> l(N,-1.0f); l[0]=+0.0f; l[256]=-0.0f; run(l,"+0 then -0"); }
    printf("%d cases, worst |idx - lowest| = %.0f -> %s\n", cases, worst_tie,
           worst_tie==0.0?"PASS":"FAIL");
    return worst_tie==0.0?0:1;
}
