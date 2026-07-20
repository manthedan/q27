// gemm_ntx_spike -- does MMQ's ntx M-minitile cut the LSU bottleneck?
//
// ncu on the shipped B-ldm k_gemm_mma_T (2026-07-19, 5090): 49% SoL, LSU pipe
// 50% (top), tensor 30%, occupancy 16.6% but FLAT across 1/2/3 blocks/SM
// (register cap 168/128/96 all 0.619ms) -- so it's per-SM pipe throughput, not
// occupancy. Top pipe = LSU. Each B ldmatrix load currently feeds ONE 16-row
// tile; MMQ shares one B-fragment across ntx=2 row-minitiles (MR 64->128),
// halving B-loads per MMA (LSU/MMA 1.5->1.0). Accumulator regs double but
// occupancy is already irrelevant here, so that cost is free.
//
// Per-output FP accumulation order is unchanged (same K-stage order, same
// group scales) => must be BITWISE vs the base kernel.
//
// Build: nvcc -O2 -std=c++17 -gencode arch=compute_120,code=sm_120 \
//   tools/gemm_ntx_spike.cu src/device_model.cu src/loader.cpp -o build/gemm_ntx_spike
// Usage: gemm_ntx_spike model.q27
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../src/device_model.h"
#include "../src/loader.h"
#include "../src/kernels.cuh"
#define CK(x) do{cudaError_t e=(x);if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

static __device__ __forceinline__ void mma_s8(int&d0,int&d1,int&d2,int&d3,uint32_t a0,uint32_t a1,
                                              uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    const int z=0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(z),"r"(z),"r"(z),"r"(z));
}
static __device__ __forceinline__ void mma_s8_acc(int&d0,int&d1,int&d2,int&d3,uint32_t a0,uint32_t a1,
                                                  uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
static __device__ __forceinline__ void ldm_x2(uint32_t&r0,uint32_t&r1,const void*p){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];\n":"=r"(r0),"=r"(r1):"r"(a));
}

// ---- BASE: shipped XG64/Q4 B-ldm kernel (MR=64), for reference/timing ----
__global__ __launch_bounds__(256) void k_base(const uint8_t* __restrict__ W,const __half* __restrict__ S,
        const int8_t* __restrict__ nat,const float* __restrict__ xs,float* __restrict__ y,
        int64_t rows,int64_t cols,int T){
    constexpr int MR=64,NT=128,KS=128,XGS=64,XSC=KS/XGS,TS=NT/16,LDW=KS+16,LDX=KS+16;
    extern __shared__ unsigned char smem_raw[];
    int8_t* s_w=(int8_t*)smem_raw; int8_t* s_x=(int8_t*)(s_w+MR*LDW);
    float* s_ws=(float*)(s_x+NT*LDX); float* s_xs=(float*)(s_ws+MR*2);
    const int warp=threadIdx.x/32,lane=threadIdx.x&31,wm=warp%4,wn=warp/4,gid=lane>>2,tg=lane&3;
    const int64_t r0=(int64_t)blockIdx.y*MR; const int t0=blockIdx.x*NT; const int n_stages=(int)(cols/KS);
    float acc[TS][4];
    #pragma unroll
    for(int s=0;s<TS;s++)for(int e=0;e<4;e++)acc[s][e]=0.f;
    constexpr int WLD=MR*(KS/2)/4/256, XLD=NT*KS/4/256, XSL=(NT*XSC+255)/256;
    const int tid=threadIdx.x,nws=MR*2;
    uint32_t rw[WLD],rx[XLD]; float rws=0.f,rxs[XSL];
    auto load_stage=[&](int st){
        const int64_t k0=(int64_t)st*KS;
        #pragma unroll
        for(int i=0;i<WLD;i++){int idx=i*256+tid,rr=idx/16,pb4=idx%16;
            rw[i]=r0+rr<rows?__ldg((const uint32_t*)(W+(r0+rr)*(cols/2)+k0/2)+pb4):0x88888888u;}
        #pragma unroll
        for(int i=0;i<XLD;i++){int idx=i*256+tid,tt=idx/(KS/4),u=idx%(KS/4);
            rx[i]=t0+tt<T?__ldg((const uint32_t*)(nat+(size_t)(t0+tt)*cols+k0)+u):0u;}
        if(tid<nws){int rr=tid/2,g=tid%2; rws=r0+rr<rows?__half2float(__ldg(S+(r0+rr)*(cols/64)+k0/64+g)):0.f;}
        #pragma unroll
        for(int i=0;i<XSL;i++){int idx=i*256+tid,tt=idx/XSC,cc=idx%XSC;
            rxs[i]=(idx<NT*XSC&&t0+tt<T)?__ldg(xs+(size_t)(t0+tt)*(cols/XGS)+k0/XGS+cc):0.f;}
    };
    auto store_stage=[&](){
        #pragma unroll
        for(int i=0;i<WLD;i++){int idx=i*256+tid,rr=idx/16,pb4=idx%16; int8_t* dst=s_w+rr*LDW+pb4*8;
            const uint32_t p=rw[i],lo=p&0x0F0F0F0Fu,hi=(p>>4)&0x0F0F0F0Fu;
            *(uint32_t*)dst=__vsub4(__byte_perm(lo,hi,0x5140),0x08080808u);
            *(uint32_t*)(dst+4)=__vsub4(__byte_perm(lo,hi,0x7362),0x08080808u);}
        #pragma unroll
        for(int i=0;i<XLD;i++){int idx=i*256+tid,tt=idx/(KS/4),u=idx%(KS/4);
            *(uint32_t*)(s_x+tt*LDX+u*4)=rx[i];}
        if(tid<nws)s_ws[tid]=rws;
        #pragma unroll
        for(int i=0;i<XSL;i++){int idx=i*256+tid; if(idx<NT*XSC)s_xs[idx]=rxs[i];}
    };
    load_stage(0);
    for(int st=0;st<n_stages;st++){
        __syncthreads(); store_stage(); if(st+1<n_stages)load_stage(st+1); __syncthreads();
        #pragma unroll
        for(int gg=0;gg<2;gg++){
            const int kb=gg*64;
            const int8_t* w0=s_w+(wm*16+gid)*LDW+kb;
            uint32_t a0=*(const uint32_t*)(w0+tg*4),a1=*(const uint32_t*)(w0+8*LDW+tg*4);
            uint32_t a2=*(const uint32_t*)(w0+tg*4+16),a3=*(const uint32_t*)(w0+8*LDW+tg*4+16);
            uint32_t a4=*(const uint32_t*)(w0+tg*4+32),a5=*(const uint32_t*)(w0+8*LDW+tg*4+32);
            uint32_t a6=*(const uint32_t*)(w0+tg*4+48),a7=*(const uint32_t*)(w0+8*LDW+tg*4+48);
            const float wsc0=s_ws[(wm*16+gid)*2+gg],wsc1=s_ws[(wm*16+gid+8)*2+gg];
            #pragma unroll
            for(int s=0;s<TS;s++){
                const int tb=wn*(NT/2)+s*8;
                uint32_t b0,b1,b2,b3;
                const int8_t* xr=s_x+(tb+(lane%8))*LDX+kb+((lane%16)/8)*16;
                ldm_x2(b0,b1,xr); ldm_x2(b2,b3,xr+32);
                int d0,d1,d2,d3;
                mma_s8(d0,d1,d2,d3,a0,a1,a2,a3,b0,b1);
                mma_s8_acc(d0,d1,d2,d3,a4,a5,a6,a7,b2,b3);
                const float xs0=s_xs[(tb+tg*2)*2+gg],xs1=s_xs[(tb+tg*2+1)*2+gg];
                acc[s][0]+=wsc0*xs0*(float)d0; acc[s][1]+=wsc0*xs1*(float)d1;
                acc[s][2]+=wsc1*xs0*(float)d2; acc[s][3]+=wsc1*xs1*(float)d3;
            }
        }
    }
    const int64_t row0=r0+wm*16+gid;
    #pragma unroll
    for(int s=0;s<TS;s++){const int tok0=t0+wn*(NT/2)+s*8+tg*2;
        #pragma unroll
        for(int e=0;e<4;e++){int64_t row=row0+(e>=2?8:0);int tok=tok0+(e&1);
            if(row<rows&&tok<T)y[(size_t)tok*rows+row]=acc[s][e];}}
}

// ---- NTX: MR=128, ntx=2 row-minitiles per warp sharing one B-fragment ----
template<int NT>
__global__ __launch_bounds__(256) void k_ntx(const uint8_t* __restrict__ W,const __half* __restrict__ S,
        const int8_t* __restrict__ nat,const float* __restrict__ xs,float* __restrict__ y,
        int64_t rows,int64_t cols,int T){
    constexpr int MR=128,KS=128,XGS=64,XSC=KS/XGS,TS=NT/16,LDW=KS+16,LDX=KS+16;
    extern __shared__ unsigned char smem_raw[];
    int8_t* s_w=(int8_t*)smem_raw; int8_t* s_x=(int8_t*)(s_w+MR*LDW);
    float* s_ws=(float*)(s_x+NT*LDX); float* s_xs=(float*)(s_ws+MR*2);
    const int warp=threadIdx.x/32,lane=threadIdx.x&31,wm=warp%4,wn=warp/4,gid=lane>>2,tg=lane&3;
    const int64_t r0=(int64_t)blockIdx.y*MR; const int t0=blockIdx.x*NT; const int n_stages=(int)(cols/KS);
    float acc[2][TS][4]; // [minitile][token-subtile][frag]
    #pragma unroll
    for(int m=0;m<2;m++)for(int s=0;s<TS;s++)for(int e=0;e<4;e++)acc[m][s][e]=0.f;
    constexpr int WLD=MR*(KS/2)/4/256, XLD=NT*KS/4/256, XSL=(NT*XSC+255)/256;
    const int tid=threadIdx.x,nws=MR*2;
    uint32_t rw[WLD],rx[XLD]; float rxs[XSL]; float rws0=0.f,rws1=0.f;
    auto load_stage=[&](int st){
        const int64_t k0=(int64_t)st*KS;
        #pragma unroll
        for(int i=0;i<WLD;i++){int idx=i*256+tid,rr=idx/16,pb4=idx%16;
            rw[i]=r0+rr<rows?__ldg((const uint32_t*)(W+(r0+rr)*(cols/2)+k0/2)+pb4):0x88888888u;}
        #pragma unroll
        for(int i=0;i<XLD;i++){int idx=i*256+tid,tt=idx/(KS/4),u=idx%(KS/4);
            rx[i]=t0+tt<T?__ldg((const uint32_t*)(nat+(size_t)(t0+tt)*cols+k0)+u):0u;}
        // MR=128 => nws=256 scale loads == blockDim, one per thread
        {int rr=tid/2,g=tid%2; rws0=r0+rr<rows?__half2float(__ldg(S+(r0+rr)*(cols/64)+k0/64+g)):0.f;}
        #pragma unroll
        for(int i=0;i<XSL;i++){int idx=i*256+tid,tt=idx/XSC,cc=idx%XSC;
            rxs[i]=(idx<NT*XSC&&t0+tt<T)?__ldg(xs+(size_t)(t0+tt)*(cols/XGS)+k0/XGS+cc):0.f;}
        (void)rws1;
    };
    auto store_stage=[&](){
        #pragma unroll
        for(int i=0;i<WLD;i++){int idx=i*256+tid,rr=idx/16,pb4=idx%16; int8_t* dst=s_w+rr*LDW+pb4*8;
            const uint32_t p=rw[i],lo=p&0x0F0F0F0Fu,hi=(p>>4)&0x0F0F0F0Fu;
            *(uint32_t*)dst=__vsub4(__byte_perm(lo,hi,0x5140),0x08080808u);
            *(uint32_t*)(dst+4)=__vsub4(__byte_perm(lo,hi,0x7362),0x08080808u);}
        #pragma unroll
        for(int i=0;i<XLD;i++){int idx=i*256+tid,tt=idx/(KS/4),u=idx%(KS/4);
            *(uint32_t*)(s_x+tt*LDX+u*4)=rx[i];}
        if(tid<nws)s_ws[tid]=rws0;
        #pragma unroll
        for(int i=0;i<XSL;i++){int idx=i*256+tid; if(idx<NT*XSC)s_xs[idx]=rxs[i];}
    };
    load_stage(0);
    for(int st=0;st<n_stages;st++){
        __syncthreads(); store_stage(); if(st+1<n_stages)load_stage(st+1); __syncthreads();
        #pragma unroll
        for(int gg=0;gg<2;gg++){
            const int kb=gg*64;
            // two row-minitiles: rows wm*32 + mt*16
            uint32_t A[2][8];
            #pragma unroll
            for(int mt=0;mt<2;mt++){
                const int8_t* w0=s_w+(wm*32+mt*16+gid)*LDW+kb;
                A[mt][0]=*(const uint32_t*)(w0+tg*4);      A[mt][1]=*(const uint32_t*)(w0+8*LDW+tg*4);
                A[mt][2]=*(const uint32_t*)(w0+tg*4+16);   A[mt][3]=*(const uint32_t*)(w0+8*LDW+tg*4+16);
                A[mt][4]=*(const uint32_t*)(w0+tg*4+32);   A[mt][5]=*(const uint32_t*)(w0+8*LDW+tg*4+32);
                A[mt][6]=*(const uint32_t*)(w0+tg*4+48);   A[mt][7]=*(const uint32_t*)(w0+8*LDW+tg*4+48);
            }
            float wsc[2][2];
            #pragma unroll
            for(int mt=0;mt<2;mt++){ wsc[mt][0]=s_ws[(wm*32+mt*16+gid)*2+gg]; wsc[mt][1]=s_ws[(wm*32+mt*16+gid+8)*2+gg]; }
            #pragma unroll
            for(int s=0;s<TS;s++){
                const int tb=wn*(NT/2)+s*8;
                uint32_t b0,b1,b2,b3;
                const int8_t* xr=s_x+(tb+(lane%8))*LDX+kb+((lane%16)/8)*16;
                ldm_x2(b0,b1,xr); ldm_x2(b2,b3,xr+32); // ONE B-load, shared across both minitiles
                const float xs0=s_xs[(tb+tg*2)*2+gg],xs1=s_xs[(tb+tg*2+1)*2+gg];
                #pragma unroll
                for(int mt=0;mt<2;mt++){
                    int d0,d1,d2,d3;
                    mma_s8(d0,d1,d2,d3,A[mt][0],A[mt][1],A[mt][2],A[mt][3],b0,b1);
                    mma_s8_acc(d0,d1,d2,d3,A[mt][4],A[mt][5],A[mt][6],A[mt][7],b2,b3);
                    acc[mt][s][0]+=wsc[mt][0]*xs0*(float)d0; acc[mt][s][1]+=wsc[mt][0]*xs1*(float)d1;
                    acc[mt][s][2]+=wsc[mt][1]*xs0*(float)d2; acc[mt][s][3]+=wsc[mt][1]*xs1*(float)d3;
                }
            }
        }
    }
    #pragma unroll
    for(int mt=0;mt<2;mt++){
        const int64_t row0=r0+wm*32+mt*16+gid;
        #pragma unroll
        for(int s=0;s<TS;s++){const int tok0=t0+wn*(NT/2)+s*8+tg*2;
            #pragma unroll
            for(int e=0;e<4;e++){int64_t row=row0+(e>=2?8:0);int tok=tok0+(e&1);
                if(row<rows&&tok<T)y[(size_t)tok*rows+row]=acc[mt][s][e];}}
    }
}

static double run_base(const q27::DevTensor& w,const int8_t* nat,const float* s64,float* y,int T,int reps){
    constexpr int MR=64,NT=128,KS=128,LDX=KS+16,XSC=2;
    size_t SM=(size_t)MR*(KS+16)+(size_t)NT*LDX+(MR*2+NT*XSC)*4;
    static int done=0; if(!done){CK(cudaFuncSetAttribute((void*)k_base,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SM));done=1;}
    auto once=[&](){dim3 g((unsigned)((T+NT-1)/NT),(unsigned)((w.rows+MR-1)/MR));
        k_base<<<g,256,SM>>>((const uint8_t*)w.data,(const __half*)w.scales,nat,s64,y,w.rows,w.cols,T);};
    once(); CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0)); for(int r=0;r<reps;r++)once(); CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); return ms/reps;
}
template<int NT>
static double run_ntx(const q27::DevTensor& w,const int8_t* nat,const float* s64,float* y,int T,int reps){
    constexpr int MR=128,KS=128,LDX=KS+16,XSC=2;
    size_t SM=(size_t)MR*(KS+16)+(size_t)NT*LDX+(MR*2+NT*XSC)*4;
    static int done=0; if(!done){CK(cudaFuncSetAttribute((void*)k_ntx<NT>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SM));done=1;}
    auto once=[&](){dim3 g((unsigned)((T+NT-1)/NT),(unsigned)((w.rows+MR-1)/MR));
        k_ntx<NT><<<g,256,SM>>>((const uint8_t*)w.data,(const __half*)w.scales,nat,s64,y,w.rows,w.cols,T);};
    once(); CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0)); for(int r=0;r<reps;r++)once(); CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); return ms/reps;
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s model.q27\n",argv[0]);return 1;}
    q27::Model m=q27::Model::open(argv[1]); q27::DeviceModel dm(m);
    const q27::DevTensor& w=dm.upload("blk.0.ffn_gate.weight");
    const int T=1024; int64_t rows=w.rows,cols=w.cols;
    int8_t* nat; float* s64;
    CK(cudaMalloc(&nat,(size_t)T*cols)); CK(cudaMalloc(&s64,(size_t)T*(cols/64)*4));
    std::vector<int8_t> hn((size_t)T*cols); std::vector<float> hs((size_t)T*(cols/64));
    for(size_t i=0;i<hn.size();i++) hn[i]=(int8_t)(((i*2654435761u)>>21)%127-63);
    for(size_t i=0;i<hs.size();i++) hs[i]=0.01f+0.001f*(i%17);
    CK(cudaMemcpy(nat,hn.data(),hn.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s64,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
    float *ya,*yb; CK(cudaMalloc(&ya,(size_t)T*rows*4)); CK(cudaMalloc(&yb,(size_t)T*rows*4));
    printf("ffn_gate %ldx%ld Q4, T=%d\n",(long)rows,(long)cols,T);
    double tb=run_base(w,nat,s64,ya,T,50);
    std::vector<float> a((size_t)T*rows),b((size_t)T*rows);
    CK(cudaMemcpy(a.data(),ya,(size_t)T*rows*4,cudaMemcpyDeviceToHost));
    printf("  base MR64/NT128 : %.4f ms\n",tb);
    auto check=[&](const char* name,double tn){
        CK(cudaMemcpy(b.data(),yb,(size_t)T*rows*4,cudaMemcpyDeviceToHost));
        long diff=0; double num=0,den=0;
        for(size_t i=0;i<a.size();i++){double d=(double)a[i]-b[i];num+=d*d;den+=(double)a[i]*a[i];if(a[i]!=b[i])diff++;}
        double rel=den>0?sqrt(num/den):0;
        printf("  %s : %.4f ms  (%+.1f%%)  rel %.1e, %ld differ  %s\n",name,tn,100*(tb/tn-1),rel,diff,
               diff==0?"BITWISE":(rel<1e-6?"~ok":"*** WRONG ***"));
    };
    check("ntx MR128/NT64 ",run_ntx<64>(w,nat,s64,yb,T,50));
    check("ntx MR128/NT96 ",run_ntx<96>(w,nat,s64,yb,T,50));
    check("ntx MR128/NT128",run_ntx<128>(w,nat,s64,yb,T,50));
    return 0;
}
