// p3_capture_spike -- T0/S0 kill-question spike for the P3 fused-verify
// capture plan (docs/plans/2026-07-16-batch-p3-capture.md). Standalone, ZERO
// engine code. Answers, on this exact driver (580.119.02) + CUDA 13.2 + sm_120:
//
//  (a) LEGALITY  -- does stream capture (Relaxed, and ThreadLocal) of the P2b
//      per-layer side-stream fork/join choreography (fork event recorded on
//      the primary, 2 side streams cudaStreamWaitEvent it, kernels there, mix
//      events recorded, primary waits them -- repeated for 64 layers) produce
//      a valid, instantiable graph? Includes ONE ~100KB dynamic-smem kernel
//      (the fdmma analog) with cudaFuncSetAttribute latched BEFORE capture.
//  (b) REPLAY SAVING -- N reps of cudaGraphLaunch+sync vs N eager issues of
//      the IDENTICAL sequence; us/launch saved. Bar >= 0.75us/launch (2x
//      margin on the measured 1.66us starvation invariant); KILL P3 < 0.4.
//  (c) INSTANTIATE COST at ~2200 nodes. Bar <= 50ms (warmup-hiccup class).
//  (d) PER-EXEC DEVICE MEMORY x 32 (the LRU cap): cudaMemGetInfo across 32
//      instantiates, and again after cudaGraphUpload of each.
//  (+) Approach-B pricing (for the record): cudaGraphExecKernelNodeSetParams
//      over ALL kernel nodes (forced-real update: dst pointer flips every
//      pass) -- the megagraph alternative's per-round patch cost.
//
// Workload shape = the p3_measure launch census (scratchpad/p3_measure/
// analyze_rounds.py over run1/profile.sqlite, 36 rounds, w16 2x32K fp8 k=2):
// ~2,635 launches/round (1,659 cstm + 976 side 724/725); tiny elementwise
// 1-4us dominate counts; ~440 GEMV-like at 23-40us (k_gemv_q4_n 308 @ ~40us);
// attn_fdmma ~72us on the SIDE streams; 199 event records + 260 stream waits
// per round. Spike: 64 layers (16 attn, il%4==3), per-gdn-layer 8 tiny +
// 3 gemv pre | fork | 2x(8 tiny side) | join | 4 gemv + 6 tiny post; attn
// layers put 1 fdmma + 1 tiny on each side stream. 3-kernel head, 13-kernel
// tail. = 2,192 kernel launches, 192 records, 256 waits per round.
//
// Build (dual-arch per repo convention; sm_86 fallback dev, sm_120 target):
//   /usr/local/cuda/bin/nvcc -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 -Xcompiler -Wall tools/p3_capture_spike.cu -o build/p3_capture_spike
// Run: systemd-run --user --wait (GPU 0, idle-checked first)
//   Usage: p3_capture_spike [reps=200]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <vector>

#define CK(x) do{cudaError_t e=(x);if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// --- dummy kernels: ALL share the (const float*, float*, int, int) ABI so the
// SetParams pricing loop can rebuild every node's args generically. Duration
// is controlled by `iters` (FMA loop), not DRAM, so it is stable and the
// buffers stay tiny.
__global__ void k_tiny_ew(const float* src, float* dst, int n, int iters) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = src[gid % n];
    for (int i = 0; i < iters; i++) v = fmaf(v, 1.000001f, 1e-7f);
    dst[gid % n] = v;
}
__global__ void k_gemv_like(const float* src, float* dst, int n, int iters) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = src[gid % n];
    for (int i = 0; i < iters; i++) v = fmaf(v, 1.000001f, 1e-7f);
    dst[gid % n] = v;
}
// fdmma analog: ~100KB dynamic smem => 1 CTA/SM occupancy shape. 3rd arg is
// the smem float count (same int slot as `n` in the shared ABI).
__global__ void k_fdmma_like(const float* src, float* dst, int smn, int iters) {
    extern __shared__ float sm[];
    int tid = threadIdx.x, gid = blockIdx.x * blockDim.x + tid;
    for (int i = tid; i < smn; i += blockDim.x) sm[i] = src[i % 1024];
    __syncthreads();
    float v = 0.f;
    for (int it = 0; it < iters; it++) v += sm[(tid * 131 + it * 257) % smn];
    dst[gid % 1024] = v;
}

// --- round context ------------------------------------------------------
struct Ctx {
    cudaStream_t cstm, side[2];
    cudaEvent_t fork, mix[2];              // reused every layer, like MixerFork
    float *src, *dstA, *dstB;              // dstA/dstB: SetParams flip targets
    int n;                                 // buffer floats
    int sm_count;
    int fd_smem;                           // dynamic smem bytes for fdmma
    int it_tiny, it_gemv, it_fd, it_head;  // duration dials
};

static const int N_LAYER = 64;             // 16 attn (il%4==3), 48 gdn

// The whole fused-verify-shaped round: identical code path for eager issue
// and for issue-under-capture (the stream/event calls are what capture sees).
static void issue_round(const Ctx& c) {
    dim3 tb(256);
    dim3 g_tiny(2), g_gemv(c.sm_count * 2), g_fd(c.sm_count);
    #define TINY(st)  k_tiny_ew  <<<g_tiny, tb, 0, st>>>(c.src, c.dstA, c.n, c.it_tiny)
    #define GEMV(st)  k_gemv_like<<<g_gemv, tb, 0, st>>>(c.src, c.dstA, c.n, c.it_gemv)
    #define FDMMA(st) k_fdmma_like<<<g_fd, tb, (size_t)c.fd_smem, st>>>(c.src, c.dstA, c.fd_smem/4, c.it_fd)
    for (int i = 0; i < 3; i++) TINY(c.cstm);                    // head (prep)
    for (int il = 0; il < N_LAYER; il++) {
        bool attn = (il % 4 == 3);
        for (int i = 0; i < 8; i++) TINY(c.cstm);                // pre: norms/quant
        for (int i = 0; i < 3; i++) GEMV(c.cstm);                // pre: qkv/gdn proj
        CK(cudaEventRecord(c.fork, c.cstm));                     // P2b fork
        for (int m = 0; m < 2; m++) {
            CK(cudaStreamWaitEvent(c.side[m], c.fork, 0));
            if (attn) { FDMMA(c.side[m]); TINY(c.side[m]); }     // fdmma + combine
            else      { for (int i = 0; i < 8; i++) TINY(c.side[m]); } // conv/delta chain
            CK(cudaEventRecord(c.mix[m], c.side[m]));
        }
        for (int m = 0; m < 2; m++)
            CK(cudaStreamWaitEvent(c.cstm, c.mix[m], 0));        // join
        for (int i = 0; i < 4; i++) GEMV(c.cstm);                // post: o/ffn
        for (int i = 0; i < 6; i++) TINY(c.cstm);                // post: adds/norms
    }
    k_gemv_like<<<g_gemv, tb, 0, c.cstm>>>(c.src, c.dstA, c.n, c.it_head); // lm head
    for (int i = 0; i < 12; i++) TINY(c.cstm);                   // argmax/accept tails
    #undef TINY
    #undef GEMV
    #undef FDMMA
}
static const int LAUNCHES_PER_ROUND = 3 + 48*(8+3+16+4+6) + 16*(8+3+4+4+6) + 1 + 12; // 2192

static double time_kernel_us(void (*k)(const float*, float*, int, int), dim3 g, int smem,
                             const Ctx& c, int arg2, int iters) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    k<<<g, 256, (size_t)smem, c.cstm>>>(c.src, c.dstA, arg2, iters);
    CK(cudaEventRecord(e0, c.cstm));
    for (int i = 0; i < 20; i++) k<<<g, 256, (size_t)smem, c.cstm>>>(c.src, c.dstA, arg2, iters);
    CK(cudaEventRecord(e1, c.cstm));
    CK(cudaStreamSynchronize(c.cstm));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return ms * 1000.0 / 20.0;
}

int main(int argc, char** argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 200;
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    int drv, rt; CK(cudaDriverGetVersion(&drv)); CK(cudaRuntimeGetVersion(&rt));
    int optin; CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    printf("dev %s sm_%d%d SMs=%d drv=%d rt=%d smem-optin=%d reps=%d\n",
           p.name, p.major, p.minor, p.multiProcessorCount, drv, rt, optin, reps);

    Ctx c{};
    c.sm_count = p.multiProcessorCount;
    c.n = 1 << 20;
    c.fd_smem = std::min(optin, 100 * 1024);   // the ~100KB fdmma shape
    c.it_tiny = 600; c.it_gemv = 12000; c.it_fd = 4500; c.it_head = 24000;
    CK(cudaMalloc(&c.src,  c.n * sizeof(float)));
    CK(cudaMalloc(&c.dstA, c.n * sizeof(float)));
    CK(cudaMalloc(&c.dstB, c.n * sizeof(float)));
    CK(cudaMemset(c.src, 0, c.n * sizeof(float)));
    CK(cudaStreamCreateWithFlags(&c.cstm, cudaStreamNonBlocking));
    for (int m = 0; m < 2; m++) CK(cudaStreamCreateWithFlags(&c.side[m], cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&c.fork, cudaEventDisableTiming));   // = MixerFork flags
    for (int m = 0; m < 2; m++) CK(cudaEventCreateWithFlags(&c.mix[m], cudaEventDisableTiming));

    // fdmma analog: smem attr latched BEFORE capture (the real fdmma one-shot raise)
    CK(cudaFuncSetAttribute(k_fdmma_like, cudaFuncAttributeMaxDynamicSharedMemorySize, c.fd_smem));

    // calibration printout (for the record: census targets tiny 1-4us,
    // gemv ~23-40us, fdmma ~50-70us)
    printf("kernel calib: tiny=%.2fus gemv=%.2fus fdmma=%.2fus (fd_smem=%dB)\n",
           time_kernel_us(k_tiny_ew,  dim3(2),           0, c, c.n, c.it_tiny),
           time_kernel_us(k_gemv_like, dim3(c.sm_count*2), 0, c, c.n, c.it_gemv),
           time_kernel_us(k_fdmma_like, dim3(c.sm_count), c.fd_smem, c, c.fd_smem/4, c.it_fd),
           c.fd_smem);
    printf("launches/round=%d records/round=%d waits/round=%d\n",
           LAUNCHES_PER_ROUND, N_LAYER * 3, N_LAYER * 4);

    // ---- eager baseline -------------------------------------------------
    for (int i = 0; i < 3; i++) { issue_round(c); CK(cudaStreamSynchronize(c.cstm)); }
    std::vector<double> te(reps);
    for (int i = 0; i < reps; i++) {
        double t0 = now_ms();
        issue_round(c);
        CK(cudaStreamSynchronize(c.cstm));
        te[i] = now_ms() - t0;
    }
    std::sort(te.begin(), te.end());
    double eager_med = te[reps / 2];
    printf("EAGER: median %.3f ms/round (min %.3f max %.3f)\n", eager_med, te.front(), te.back());

    // ---- (a) capture legality: Relaxed ----------------------------------
    cudaGraph_t graph = nullptr;
    double t0 = now_ms();
    CK(cudaStreamBeginCapture(c.cstm, cudaStreamCaptureModeRelaxed));
    issue_round(c);
    cudaError_t ec = cudaStreamEndCapture(c.cstm, &graph);
    double cap_ms = now_ms() - t0;
    if (ec != cudaSuccess || !graph) {
        printf("VERDICT (a) LEGALITY: FAIL (Relaxed EndCapture: %s) -> KILL P3a, fallback D\n",
               cudaGetErrorString(ec));
        return 1;
    }
    size_t nnodes = 0;
    CK(cudaGraphGetNodes(graph, nullptr, &nnodes));
    std::vector<cudaGraphNode_t> nodes(nnodes);
    CK(cudaGraphGetNodes(graph, nodes.data(), &nnodes));
    int nkern = 0, nevent = 0, nother = 0;
    for (auto nd : nodes) {
        cudaGraphNodeType ty; CK(cudaGraphNodeGetType(nd, &ty));
        if (ty == cudaGraphNodeTypeKernel) nkern++;
        else if (ty == cudaGraphNodeTypeEventRecord || ty == cudaGraphNodeTypeWaitEvent) nevent++;
        else nother++;
    }
    printf("capture(Relaxed): OK in %.2f ms, %zu nodes (%d kernel, %d event, %d other)\n",
           cap_ms, nnodes, nkern, nevent, nother);

    // ---- (c) instantiate cost -------------------------------------------
    cudaGraphExec_t exec = nullptr;
    t0 = now_ms();
    cudaError_t ei = cudaGraphInstantiate(&exec, graph, 0);
    double inst_ms = now_ms() - t0;
    if (ei != cudaSuccess || !exec) {
        printf("VERDICT (a) LEGALITY: FAIL (instantiate: %s) -> KILL P3a, fallback D\n",
               cudaGetErrorString(ei));
        return 1;
    }
    printf("instantiate: %.2f ms at %zu nodes\n", inst_ms, nnodes);

    // ThreadLocal mode legality (same choreography)
    {
        cudaGraph_t g2 = nullptr;
        CK(cudaStreamBeginCapture(c.cstm, cudaStreamCaptureModeThreadLocal));
        issue_round(c);
        cudaError_t e2 = cudaStreamEndCapture(c.cstm, &g2);
        if (e2 == cudaSuccess && g2) {
            cudaGraphExec_t x2 = nullptr;
            cudaError_t e3 = cudaGraphInstantiate(&x2, g2, 0);
            printf("capture(ThreadLocal): %s\n",
                   (e3 == cudaSuccess && x2) ? "OK (instantiate OK)" : cudaGetErrorString(e3));
            if (x2) CK(cudaGraphExecDestroy(x2));
            CK(cudaGraphDestroy(g2));
        } else {
            printf("capture(ThreadLocal): FAIL (%s)\n", cudaGetErrorString(e2));
        }
    }

    // ---- (b) replay saving ----------------------------------------------
    for (int i = 0; i < 3; i++) { CK(cudaGraphLaunch(exec, c.cstm)); CK(cudaStreamSynchronize(c.cstm)); }
    std::vector<double> tg(reps);
    for (int i = 0; i < reps; i++) {
        double t1 = now_ms();
        CK(cudaGraphLaunch(exec, c.cstm));
        CK(cudaStreamSynchronize(c.cstm));
        tg[i] = now_ms() - t1;
    }
    std::sort(tg.begin(), tg.end());
    double graph_med = tg[reps / 2];
    double save_us = (eager_med - graph_med) * 1000.0 / LAUNCHES_PER_ROUND;
    printf("GRAPH: median %.3f ms/round (min %.3f max %.3f)\n", graph_med, tg.front(), tg.back());
    printf("saving: %.3f ms/round = %.3f us/launch; at 2075 verify launches -> %.2f ms/round\n",
           eager_med - graph_med, save_us, save_us * 2075.0 / 1000.0);

    // ---- (d) per-exec device memory x 32 --------------------------------
    size_t free0, free1, free2, tot;
    CK(cudaMemGetInfo(&free0, &tot));
    std::vector<cudaGraphExec_t> zoo(32);
    for (auto& x : zoo) CK(cudaGraphInstantiate(&x, graph, 0));
    CK(cudaMemGetInfo(&free1, &tot));
    for (auto& x : zoo) CK(cudaGraphUpload(x, c.cstm));
    CK(cudaStreamSynchronize(c.cstm));
    CK(cudaMemGetInfo(&free2, &tot));
    printf("mem/exec: instantiate %.3f MB, +upload %.3f MB; 32-cap total %.1f MB\n",
           (free0 - free1) / 32.0 / 1048576.0, (free0 - free2) / 32.0 / 1048576.0,
           (free0 - free2) / 1048576.0);
    for (auto& x : zoo) CK(cudaGraphExecDestroy(x));

    // ---- Approach-B pricing: ExecKernelNodeSetParams over all kernel nodes
    // Own arg storage per node (GetParams' kernelParams is node-owned); dst
    // flips A/B every pass so each SetParams is a REAL param change.
    struct Args { const float* src; float* dst; int a2; int a3; void* kp[4]; };
    std::vector<cudaGraphNode_t> knodes;
    for (auto nd : nodes) {
        cudaGraphNodeType ty; CK(cudaGraphNodeGetType(nd, &ty));
        if (ty == cudaGraphNodeTypeKernel) knodes.push_back(nd);
    }
    std::vector<Args> args(knodes.size());
    std::vector<cudaKernelNodeParams> kp(knodes.size());
    for (size_t i = 0; i < knodes.size(); i++) {
        CK(cudaGraphKernelNodeGetParams(knodes[i], &kp[i]));
        Args& a = args[i];
        a.src = *(const float**)kp[i].kernelParams[0];
        a.dst = *(float**)kp[i].kernelParams[1];
        a.a2  = *(int*)kp[i].kernelParams[2];
        a.a3  = *(int*)kp[i].kernelParams[3];
        a.kp[0] = &a.src; a.kp[1] = &a.dst; a.kp[2] = &a.a2; a.kp[3] = &a.a3;
        kp[i].kernelParams = a.kp;
    }
    const int NPASS = 6;
    std::vector<double> tp(NPASS);
    for (int pass = 0; pass < NPASS; pass++) {
        float* dst = (pass & 1) ? c.dstB : c.dstA;
        double t1 = now_ms();
        for (size_t i = 0; i < knodes.size(); i++) {
            args[i].dst = dst;
            CK(cudaGraphExecKernelNodeSetParams(exec, knodes[i], &kp[i]));
        }
        tp[pass] = now_ms() - t1;
    }
    std::sort(tp.begin(), tp.end());
    double patch_med = tp[NPASS / 2];
    printf("SetParams pricing: %.3f ms / %zu kernel nodes = %.3f us/node per pass\n",
           patch_med, knodes.size(), patch_med * 1000.0 / knodes.size());
    CK(cudaGraphLaunch(exec, c.cstm));            // patched exec still runs?
    CK(cudaStreamSynchronize(c.cstm));
    CK(cudaGetLastError());
    printf("patched exec relaunch: OK\n");

    // ---- verdicts ---------------------------------------------------------
    printf("\nVERDICT (a) LEGALITY: PASS (Relaxed capture + instantiate OK, %zu nodes)\n", nnodes);
    const char* vb = (save_us >= 0.75) ? "PASS" : (save_us < 0.4 ? "KILL-P3" : "GRAY (0.4-0.75)");
    printf("VERDICT (b) SAVING: %.3f us/launch -> %s (bar >=0.75, kill <0.4)\n", save_us, vb);
    printf("VERDICT (c) INSTANTIATE: %.2f ms -> %s (bar <=50 ms)\n", inst_ms,
           inst_ms <= 50.0 ? "PASS" : "FAIL");
    printf("VERDICT (d) MEM: %.3f MB/exec uploaded, 32-cap %.1f MB\n",
           (free0 - free2) / 32.0 / 1048576.0, (free0 - free2) / 1048576.0);

    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    return 0;
}
