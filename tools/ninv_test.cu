// ninv_test -- N-invariance gate for the continuous-batching determinism
// contract (docs/plans/2026-07-14-continuous-batching.md, P1 Task 5).
//
// CLAIM UNDER TEST ("bitwise-when-untrimmed"): a lane's output from the
// multi-lane verify weight kernels is bitwise independent of (a) how many
// OTHER lanes run in the same launch (T) and (b) WHICH slot the lane occupies.
// If that holds, a fused cross-engine round at union width T=w1+w2 produces,
// for each engine's lanes, exactly the bytes its solo round would have -- the
// whole Task 10 solo-equivalence gate rests on this.
//
// Method: 3 payload lanes get fixed host-generated pseudo-random activations,
// quantized ONCE via the engine's own quantize3 (k_quantize_x3) so both runs
// share bit-identical XQuant buffers -- the weight kernel is the only variable.
// Run A: T=N1, payload in the prefix slots {0,1,2}. Run B: T=N2>N1, the SAME
// payload buffers mapped to scattered slots (e.g. {1,4,7}); every other live
// slot carries JUNK that differs per run (junk proves isolation -- zeros would
// vacuously pass a lane-bleed bug whose contribution is x*0). Payload y
// buffers are pre-poisoned with a DIFFERENT byte pattern per run, so "kernel
// wrote nothing" can never compare equal. Outputs compared bitwise (word
// compare over rows floats).
//
// Families x shapes (discovered from the model like vgemm_test, not hardcoded):
//   vgemm_verify Q4  -- ffn_down (z=8, MODE 1 + k_reduce_z), output_q4 (z=1, MODE 0)
//   vgemm_verify Q8  -- ssm_out  (z=8, MODE 1)   [no engine Q8 shape has z=1]
//   gemv_q4_n        -- ffn_down (wide-K), output_q4 (tall head)
//   gemv_q8_n        -- ssm_out
//   gemv_f16_3       -- ssm_alpha (48 x 5120, the exact engine use)
// (N1,N2) in {(2,5),(3,9),(5,12),(9,16)}. All kernels accept T in 2..W_PLUMB;
// the ENGINE only reaches vgemm at vw >= gemm_min (9), but the fused round may
// grant any union width, so the low-T legs are tested too.
//
// A FAILURE HERE IS A FINDING, NOT A BUG TO FIX (plan addendum A1): the
// determinism contract downgrades for that family and the design doc gets the
// measured diff. Do not touch the kernels.
//
// SEAM LEG (P2 exit review, 2026-07-16): the tables above prove the
// MULTI-lane family invariant against itself (T/slot). But the SOLO draft
// path runs the SINGLE-lane family (k_gemv_q4/q8 via mm()/mtp_mm1,
// k_quantize_x via qx, k_rmsnorm, k_add via add_inplace, k_embed_row_q8,
// k_silu_mul) while the fused draft step runs the multi-lane twins -- so the
// P2c "fused margins bitwise vs solo" claim ALSO rests on single==multi for
// the payload lane, a seam no gate covered (test_kernels compares the gemv
// pair at 1e-5 tolerance only). The gemv twins carry different
// __launch_bounds__ tiers and differently-shaped accumulate expressions, so
// bitwise equality across the seam is a MEASUREMENT, not a property of "same
// math". The leg: identical payload input -> single-lane kernel vs multi-lane
// twin (payload in one live lane, junk elsewhere, T in {2,4}), BITWISE
// memcmp. EITHER verdict is a valid result -- a DIFFER downgrades the
// documented claim to tolerance-class (A1 flavor); do not touch the kernels.
//
// CHUNK+FOLD LEG (M1 record+fold, docs/plans/2026-08-15-batched-decode-spec.md
// Appendix A; replaces the retired P3 T2 TWIN leg -- the table twins died with
// the role rotation). Reference = the serial role chain the rotation ran
// (lane-0 in-place, lane L reads role L-1 writes role L, l2norm3 between conv
// and delta) over test-local role buffers. Under test: (a) the verify chunk
// kernels gdn_conv_chunk3/gdn_delta_chunk3 -- speculative lanes read committed
// state only, per-lane convout/o must be bitwise the chain's, committed state
// must hold exactly state-after-lane-0; (b) the commit Fold -- for every
// m in 1..W, conv_ring_update + delta_scan_seq over the rows gdn_record3
// retained must land bitwise on the chain's role m-1 (ring and S). W in
// {2,5,12} x 3 layers, ZERO tolerance. UNLIKE the tables above, a mismatch
// here is DOA for M1 -- no tolerance class, no A1 downgrade: acceptance
// couples to these bytes (the bw24 lesson).
//
// Usage: ninv_test [model.q27]   (default: the canonical qwen36-27b-mtp)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/blocks.cuh" // add_inplace (the solo residual add; seam leg)
#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"
#include "../src/prefill.cuh" // conv_ring_update / delta_scan_seq (M1 fold leg)
#include "../src/spec3.cuh"
#include "../src/vgemm.cuh"

static int fails = 0;
#define CHECK(cond, ...)                                                                 \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            printf("  FAIL: ");                                                          \
            printf(__VA_ARGS__);                                                         \
            printf("\n");                                                                \
            fails++;                                                                     \
        }                                                                                \
    } while (0)

// xorshift32: explicit, seed-stable host PRNG (rand() would tie the gate to
// the libc). Maps to [-scale, scale).
static inline uint32_t xs32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
static void fill_rand(float* h, int64_t n, uint32_t seed, float scale) {
    uint32_t s = seed ? seed : 1u;
    for (int64_t i = 0; i < n; i++)
        h[i] = ((int32_t)xs32(s) >> 8) * (scale / 8388608.0f);
}

static const int NPAY = 3;                 // payload lanes under test
static const int64_t MAXC = 17408;         // widest decode cols (ffn_down)
static const int64_t MAXR = 248320;        // tallest rows (vocab head)

struct Case {
    int n1, n2;
    int slotsA[NPAY]; // -1 = unused (cases with N1 < 3 carry only N1 payloads)
    int slotsB[NPAY];
};
static const Case CASES[] = {
    {2, 5, {0, 1, -1}, {1, 4, -1}},
    {3, 9, {0, 1, 2}, {1, 4, 7}},
    {5, 12, {0, 1, 2}, {2, 9, 11}},
    {9, 16, {0, 1, 2}, {5, 10, 15}},
};
static const int NCASES = (int)(sizeof(CASES) / sizeof(CASES[0]));

enum Fam { F_VGEMM, F_GEMV, F_F16 };

// Persistent payload state: float activations + XQuant, quantized once.
static float* d_pay_x[NPAY];
static q27k::XQuant pay_xq[NPAY];
// Junk pool: one buffer set per slot, refilled with fresh junk before every run.
static float* d_junk_x[W_PLUMB];
static q27k::XQuant junk_xq[W_PLUMB];
static float* d_junk_y[W_PLUMB];
// Per-run payload outputs (A and B kept separate for the host compare).
static float* d_y_run[2][NPAY];

static std::vector<float> h_scratch; // MAXC staging for uploads

static void regen_junk(uint32_t seed) {
    for (int s = 0; s < W_PLUMB; s++) {
        fill_rand(h_scratch.data(), MAXC, seed ^ (0x9E3779B9u * (uint32_t)(s + 1)),
                  0.7f + 0.05f * (float)s);
        CUDA_CHECK(cudaMemcpy(d_junk_x[s], h_scratch.data(), MAXC * 4, cudaMemcpyHostToDevice));
        q27k::quantize_x(d_junk_x[s], MAXC, junk_xq[s]);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// One launch of `fam` on weight `w` at width T with payload lanes at `slots`.
// run selects the output buffer set + poison pattern.
static void launch_run(Fam fam, const q27::DevTensor& w, float* d_ws, int T, const int* slots,
                       int npay, int run) {
    // slot -> payload lane (-1 = junk)
    int lane_of[W_PLUMB];
    for (int s = 0; s < W_PLUMB; s++) lane_of[s] = -1;
    for (int i = 0; i < npay; i++) lane_of[slots[i]] = i;
    // poison payload outputs with a per-run pattern: an unwritten row can
    // never compare equal across runs.
    for (int i = 0; i < npay; i++)
        CUDA_CHECK(cudaMemset(d_y_run[run][i], run ? 0xBB : 0xAA, MAXR * 4));

    if (fam == F_VGEMM) {
        q27k::XLanes X{};
        q27k::YLanes Y{};
        for (int s = 0; s < W_PLUMB; s++) { // ALL slots get valid pointers (vgemm_test idiom)
            int l = lane_of[s];
            X.nat[s] = l >= 0 ? pay_xq[l].nat : junk_xq[s].nat;
            X.xs[s] = l >= 0 ? pay_xq[l].scale : junk_xq[s].scale;
            Y.y[s] = l >= 0 ? d_y_run[run][l] : d_junk_y[s];
        }
        bool ok = q27k::vgemm_verify(w, X, Y, d_ws, T, 0);
        CHECK(ok, "vgemm_verify refused rows=%ld cols=%ld T=%d", (long)w.rows, (long)w.cols, T);
    } else if (fam == F_GEMV) {
        q27k::XQuant qs[W_PLUMB];
        float* ys[W_PLUMB];
        for (int s = 0; s < W_PLUMB; s++) {
            int l = lane_of[s];
            qs[s] = l >= 0 ? pay_xq[l] : junk_xq[s];
            ys[s] = l >= 0 ? d_y_run[run][l] : d_junk_y[s];
        }
        if (w.dtype == q27::DType::Q4_G64)
            q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, T, ys, w.rows,
                            w.cols, 0);
        else if (w.dtype == q27::DType::T2_G128) // Bonsai 2 (2026-09-18)
            q27k::gemv_t2_n((const uint8_t*)w.data, (const __half*)w.scales, qs, T, ys, w.rows,
                            w.cols, 0);
        else if (w.dtype == q27::DType::T3_G128) // Bonsai 2 8 GB packs (2026-09-20)
            q27k::gemv_t3_n((const uint8_t*)w.data, (const __half*)w.scales, qs, T, ys, w.rows,
                            w.cols, 0);
        else
            q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, T, ys, w.rows,
                            w.cols, 0);
    } else { // F_F16
        q27k::CP3 x{};
        q27k::P3 y{};
        for (int s = 0; s < W_PLUMB; s++) {
            int l = lane_of[s];
            x.p[s] = l >= 0 ? d_pay_x[l] : d_junk_x[s];
            y.p[s] = l >= 0 ? d_y_run[run][l] : d_junk_y[s];
        }
        q27k::gemv_f16_3((const __half*)w.data, x, y, w.rows, w.cols, 0, T);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Runs the 4 cases for one (family, weight); returns this table's fail count.
static int run_table(const char* tag, Fam fam, const q27::DevTensor& w, float* d_ws,
                     uint32_t junk_salt) {
    const int before = fails;
    std::vector<float> ha(w.rows), hb(w.rows);
    for (int c = 0; c < NCASES; c++) {
        const Case& K = CASES[c];
        const int npay = K.n1 < NPAY ? K.n1 : NPAY;
        // fresh junk PER RUN: if a payload output ever depends on junk-lane
        // contents, the two runs cannot agree.
        regen_junk(junk_salt + 2u * (uint32_t)c);
        launch_run(fam, w, d_ws, K.n1, K.slotsA, npay, 0);
        regen_junk(junk_salt + 2u * (uint32_t)c + 1u);
        launch_run(fam, w, d_ws, K.n2, K.slotsB, npay, 1);

        long diffs[NPAY] = {0, 0, 0};
        float maxad = 0.f;
        long total = 0;
        for (int i = 0; i < npay; i++) {
            CUDA_CHECK(cudaMemcpy(ha.data(), d_y_run[0][i], w.rows * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hb.data(), d_y_run[1][i], w.rows * 4, cudaMemcpyDeviceToHost));
            if (memcmp(ha.data(), hb.data(), w.rows * 4) != 0)
                for (uint64_t r = 0; r < w.rows; r++)
                    if (memcmp(&ha[r], &hb[r], 4) != 0) { // bitwise, NaN-safe
                        diffs[i]++;
                        maxad = fmaxf(maxad, fabsf(ha[r] - hb[r]));
                    }
            total += diffs[i];
        }
        printf("  %-22s T=%2d slots{%d,%d,%d} vs T=%2d slots{%d,%d,%d}  diffs=%ld/%ld/%ld  %s",
               tag, K.n1, K.slotsA[0], K.slotsA[1], K.slotsA[2], K.n2, K.slotsB[0], K.slotsB[1],
               K.slotsB[2], diffs[0], diffs[1], diffs[2], total == 0 ? "PASS" : "FAIL");
        if (total) printf("  (max |a-b| %.3e of %llu rows)", maxad, (unsigned long long)w.rows);
        printf("\n");
        CHECK(total == 0, "%s (N1=%d,N2=%d): %ld payload floats differ across runs", tag, K.n1,
              K.n2, total);
    }
    return fails - before;
}

// ---------------------------------------------------------------------------
// SEAM LEG (file-header comment: single-lane kernels vs their multi-lane
// twins across the solo/fused draft seam). Reuses the payload/junk state
// above; junk lanes carry per-run junk so lane bleed cannot vacuously pass.

static int seam_pairs_differ = 0; // pairs with >= 1 differing config

// float-bit lexicographic map -> ulp distance between bitwise-unequal floats
// (NaN/Inf just map to large deltas; we only ever print this on a DIFF).
static int64_t ulp_delta(float a, float b) {
    int32_t ia, ib;
    memcpy(&ia, &a, 4);
    memcpy(&ib, &b, 4);
    int64_t la = ia >= 0 ? (int64_t)ia : (int64_t)0x80000000LL - ia;
    int64_t lb = ib >= 0 ? (int64_t)ib : (int64_t)0x80000000LL - ib;
    int64_t d = la - lb;
    return d < 0 ? -d : d;
}

// bitwise word compare of two device f32 buffers; tracks max ulp over diffs
static long seam_cmp(const float* d_a, const float* d_b, int64_t n, int64_t* max_ulp) {
    std::vector<float> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), d_a, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), d_b, n * 4, cudaMemcpyDeviceToHost));
    if (memcmp(ha.data(), hb.data(), n * 4) == 0) return 0;
    long diffs = 0;
    for (int64_t i = 0; i < n; i++)
        if (memcmp(&ha[i], &hb[i], 4) != 0) {
            diffs++;
            int64_t u = ulp_delta(ha[i], hb[i]);
            if (u > *max_ulp) *max_ulp = u;
        }
    return diffs;
}

// raw byte compare (the quantize pair's int8/eo/isum sub-buffers)
static long seam_cmp_bytes(const void* d_a, const void* d_b, int64_t bytes) {
    std::vector<uint8_t> ha(bytes), hb(bytes);
    CUDA_CHECK(cudaMemcpy(ha.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), d_b, bytes, cudaMemcpyDeviceToHost));
    long diffs = 0;
    for (int64_t i = 0; i < bytes; i++)
        if (ha[i] != hb[i]) diffs++;
    return diffs;
}

static void seam_row(const char* tag, int T, int slot, long diffs, int64_t max_ulp, int64_t n,
                     bool* pair_diff) {
    printf("  %-24s T=%d slot=%d  diffs=%ld/%lld  %s", tag, T, slot, diffs,
           (long long)n, diffs == 0 ? "PASS" : "DIFF");
    if (diffs) printf("  (max %lld ulp)", (long long)max_ulp);
    printf("\n");
    if (diffs) *pair_diff = true;
}

static void seam_leg(q27::DeviceModel& dm, const q27::DevTensor& w_down,
                     const q27::DevTensor& w_head, const q27::DevTensor& w_sout) {
    const q27::DevTensor& emb = dm.upload("token_embd.weight"); // Q8_G128, cols 5120
    const int64_t NE = 5120; // rmsnorm/add/embed width (N_EMBD)
    const int64_t NF = MAXC; // silu/quantize width (N_FFN)
    const struct { int T, slot; } CFG[] = {{2, 0}, {4, 2}};

    // leg-owned buffers: single/multi payload outputs + rmsnorm weight +
    // fresh XQuant pair (pay_xq must stay untouched -- the gemv rows read it
    // as the SHARED input of both runs) + token slots for the embed pair.
    float *d_ys, *d_ym, *d_wrms;
    CUDA_CHECK(cudaMalloc(&d_ys, NF * 4));
    CUDA_CHECK(cudaMalloc(&d_ym, NF * 4));
    CUDA_CHECK(cudaMalloc(&d_wrms, NE * 4));
    fill_rand(h_scratch.data(), NE, 0xBEEFu, 0.9f);
    CUDA_CHECK(cudaMemcpy(d_wrms, h_scratch.data(), NE * 4, cudaMemcpyHostToDevice));
    q27k::XQuant xq_s = q27k::xquant_alloc(NF), xq_m = q27k::xquant_alloc(NF);
    int* d_tok; // slot 0 = payload token 1234; slots 1.. = junk row ids
    CUDA_CHECK(cudaMalloc(&d_tok, W_PLUMB * 4));
    {
        int htok[W_PLUMB];
        for (int s = 0; s < W_PLUMB; s++) htok[s] = 100 + 37 * s;
        htok[0] = 1234;
        CUDA_CHECK(cudaMemcpy(d_tok, htok, sizeof htok, cudaMemcpyHostToDevice));
    }

    printf("\n== seam: single-lane kernels vs multi-lane twins (payload lane, bitwise) ==\n");
    bool pd_gemv4 = false, pd_gemv8 = false, pd_q = false, pd_rms = false, pd_add = false,
         pd_emb = false, pd_silu = false;

    for (const auto& c : CFG) {
        const int T = c.T, slot = c.slot;
        int64_t mu;
        long d;

        // gemv_q4 vs gemv_q4_n (both engine Q4 shapes), gemv_q8 vs gemv_q8_n.
        // Both runs read the SAME pay_xq[0] device bytes: the kernel pair is
        // the only variable.
        const q27::DevTensor* ws[3] = {&w_down, &w_head, &w_sout};
        const char* wtag[3] = {"gemv_q4[ffn_down]", "gemv_q4[head]", "gemv_q8[ssm_out]"};
        for (int wi = 0; wi < 3; wi++) {
            const q27::DevTensor& w = *ws[wi];
            regen_junk(0x900u + 0x10u * (uint32_t)wi + (uint32_t)T);
            CUDA_CHECK(cudaMemset(d_y_run[0][0], 0xAA, w.rows * 4));
            CUDA_CHECK(cudaMemset(d_y_run[1][0], 0xBB, w.rows * 4));
            if (w.dtype == q27::DType::Q4_G64)
                q27k::gemv_q4((const uint8_t*)w.data, (const __half*)w.scales, pay_xq[0],
                              d_y_run[0][0], w.rows, w.cols);
            else if (w.dtype == q27::DType::T2_G128)
                q27k::gemv_t2((const uint8_t*)w.data, (const __half*)w.scales, pay_xq[0],
                              d_y_run[0][0], w.rows, w.cols);
            else if (w.dtype == q27::DType::T3_G128)
                q27k::gemv_t3((const uint8_t*)w.data, (const __half*)w.scales, pay_xq[0],
                              d_y_run[0][0], w.rows, w.cols);
            else
                q27k::gemv_q8((const int8_t*)w.data, (const __half*)w.scales, pay_xq[0],
                              d_y_run[0][0], w.rows, w.cols);
            q27k::XQuant qs[W_PLUMB];
            float* ys[W_PLUMB];
            for (int s = 0; s < W_PLUMB; s++) { qs[s] = junk_xq[s]; ys[s] = d_junk_y[s]; }
            qs[slot] = pay_xq[0];
            ys[slot] = d_y_run[1][0];
            if (w.dtype == q27::DType::Q4_G64)
                q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, T, ys,
                                w.rows, w.cols);
            else if (w.dtype == q27::DType::T2_G128)
                q27k::gemv_t2_n((const uint8_t*)w.data, (const __half*)w.scales, qs, T, ys,
                                w.rows, w.cols);
            else if (w.dtype == q27::DType::T3_G128)
                q27k::gemv_t3_n((const uint8_t*)w.data, (const __half*)w.scales, qs, T, ys,
                                w.rows, w.cols);
            else
                q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, T, ys,
                                w.rows, w.cols);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_y_run[0][0], d_y_run[1][0], w.rows, &mu);
            seam_row(wtag[wi], T, slot, d, mu, w.rows, wi == 2 ? &pd_gemv8 : &pd_gemv4);
        }

        // quantize_x vs quantize3: all four output sub-buffers, bitwise.
        {
            regen_junk(0x950u + (uint32_t)T);
            CUDA_CHECK(cudaMemset(xq_s.nat, 0xAA, NF));
            CUDA_CHECK(cudaMemset(xq_s.eo, 0xAA, NF / 8 * sizeof(uint2)));
            CUDA_CHECK(cudaMemset(xq_s.scale, 0xAA, NF / 32 * 4));
            CUDA_CHECK(cudaMemset(xq_s.isum, 0xAA, NF / 32 * 4));
            CUDA_CHECK(cudaMemset(xq_m.nat, 0xBB, NF));
            CUDA_CHECK(cudaMemset(xq_m.eo, 0xBB, NF / 8 * sizeof(uint2)));
            CUDA_CHECK(cudaMemset(xq_m.scale, 0xBB, NF / 32 * 4));
            CUDA_CHECK(cudaMemset(xq_m.isum, 0xBB, NF / 32 * 4));
            q27k::quantize_x(d_pay_x[0], NF, xq_s);
            q27k::CP3 xs{};
            q27k::XQ3 x3{};
            for (int s = 0; s < W_PLUMB; s++) { xs.p[s] = d_junk_x[s]; x3.q[s] = junk_xq[s]; }
            xs.p[slot] = d_pay_x[0];
            x3.q[slot] = xq_m;
            q27k::quantize3(xs, NF, x3, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp_bytes(xq_s.nat, xq_m.nat, NF);
            d += seam_cmp_bytes(xq_s.eo, xq_m.eo, NF / 8 * sizeof(uint2));
            d += seam_cmp(xq_s.scale, xq_m.scale, NF / 32, &mu);
            d += seam_cmp_bytes(xq_s.isum, xq_m.isum, NF / 32 * 4);
            seam_row("quantize_x", T, slot, d, mu, NF + NF / 8 + 2 * (NF / 32), &pd_q);
        }

        // rmsnorm vs rmsnorm3
        {
            regen_junk(0x960u + (uint32_t)T);
            CUDA_CHECK(cudaMemset(d_ys, 0xAA, NE * 4));
            CUDA_CHECK(cudaMemset(d_ym, 0xBB, NE * 4));
            q27k::rmsnorm(d_pay_x[0], d_wrms, d_ys, (int)NE, 1e-6f);
            q27k::CP3 x{};
            q27k::P3 y{};
            for (int s = 0; s < W_PLUMB; s++) { x.p[s] = d_junk_x[s]; y.p[s] = d_junk_y[s]; }
            x.p[slot] = d_pay_x[0];
            y.p[slot] = d_ym;
            q27k::rmsnorm3(x, d_wrms, y, (int)NE, 1e-6f, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_ys, d_ym, NE, &mu);
            seam_row("rmsnorm", T, slot, d, mu, NE, &pd_rms);
        }

        // add_inplace vs add3 (in place: both sides seeded with the payload;
        // junk lanes get d_junk_y as their mutable accumulator)
        {
            regen_junk(0x970u + (uint32_t)T);
            CUDA_CHECK(cudaMemcpy(d_ys, d_pay_x[0], NE * 4, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ym, d_pay_x[0], NE * 4, cudaMemcpyDeviceToDevice));
            q27k::add_inplace(d_ys, d_pay_x[1], (int)NE);
            q27k::P3 x{};
            q27k::CP3 yy{};
            for (int s = 0; s < W_PLUMB; s++) { x.p[s] = d_junk_y[s]; yy.p[s] = d_junk_x[s]; }
            x.p[slot] = d_ym;
            yy.p[slot] = d_pay_x[1];
            q27k::add3(x, yy, (int)NE, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_ys, d_ym, NE, &mu);
            seam_row("add", T, slot, d, mu, NE, &pd_add);
        }

        // embed_row_q8 vs embed3 (junk lanes read junk token rows; read-only
        // weight, so a junk lane sharing the payload row would be harmless)
        {
            regen_junk(0x980u + (uint32_t)T);
            CUDA_CHECK(cudaMemset(d_ys, 0xAA, NE * 4));
            CUDA_CHECK(cudaMemset(d_ym, 0xBB, NE * 4));
            q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, d_tok, NE,
                               d_ys);
            q27k::IP3 tk{};
            q27k::P3 out{};
            for (int s = 0; s < W_PLUMB; s++) { tk.p[s] = d_tok + s; out.p[s] = d_junk_y[s]; }
            tk.p[slot] = d_tok;
            out.p[slot] = d_ym;
            q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales, tk, NE, out, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_ys, d_ym, NE, &mu);
            seam_row("embed_row_q8", T, slot, d, mu, NE, &pd_emb);
        }

        // silu_mul vs silu_mul3 (solo path runs out == gate in place; the
        // multi twin is in place on g by construction)
        {
            regen_junk(0x990u + (uint32_t)T);
            CUDA_CHECK(cudaMemcpy(d_ys, d_pay_x[0], NF * 4, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ym, d_pay_x[0], NF * 4, cudaMemcpyDeviceToDevice));
            q27k::silu_mul(d_ys, d_pay_x[1], d_ys, (int)NF);
            q27k::P3 g{};
            q27k::CP3 u{};
            for (int s = 0; s < W_PLUMB; s++) { g.p[s] = d_junk_y[s]; u.p[s] = d_junk_x[s]; }
            g.p[slot] = d_ym;
            u.p[slot] = d_pay_x[1];
            q27k::silu_mul3(g, u, (int)NF, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_ys, d_ym, NF, &mu);
            seam_row("silu_mul", T, slot, d, mu, NF, &pd_silu);
        }
    }

    seam_pairs_differ =
        (int)pd_gemv4 + pd_gemv8 + pd_q + pd_rms + pd_add + pd_emb + pd_silu;
    if (seam_pairs_differ == 0) printf("SEAM BITWISE: ALL PASS\n");
    else printf("SEAM BITWISE: %d pairs DIFFER (tolerance-class)\n", seam_pairs_differ);

    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_ym));
    CUDA_CHECK(cudaFree(d_wrms));
    CUDA_CHECK(cudaFree(d_tok));
}

// ---------------------------------------------------------------------------
// CHUNK+FOLD LEG (file-header comment): M1 verify chunk kernels + commit Fold
// vs the serial role chain, bitwise. Standalone state -- touches none of the
// payload/junk machinery above except the l2norm3 junk padding.

static void chunk_fold_leg() {
    // Mirrors the engine's GDN geometry (engine.cuh): GDN_CH channels,
    // 48 v-heads x 128, ring = [3][CH]. The reference chain uses W role
    // buffers (the retired rotation's layout -- the chain IS the reference
    // semantics); the chunk path uses ONE committed set + the record arena,
    // exactly the M1 engine shape.
    const int CH = 10240, HEADS = 48, SKD = 128, WMAXT = 12, NLT = 3;
    const int64_t RINGN = 3 * (int64_t)CH;
    const int64_t SN = (int64_t)HEADS * SKD * SKD;
    const int64_t ON = (int64_t)HEADS * SKD;

    printf("\n== chunk+fold: M1 verify chunks + commit Fold vs serial chain (bitwise) ==\n");

    // reference role chain replicas; chunk path committed state + fold buffers
    static float *ring_r[NLT][12], *S_r[NLT][12];
    float *ring_c[NLT], *S_c[NLT];
    float *convo_r[12], *oo_r[12], *convo_c[12], *oo_c[12];
    float *qkv_in[12], *g_in[12], *beta_in[12], *w4;
    float *rec_q, *rec_c, *rec_g, *rec_b; // one layer's record arena rows
    float *ring_f, *S_f, *o_scr;          // fold working state + dead o output
    for (int il = 0; il < NLT; il++) {
        for (int ph = 0; ph < WMAXT; ph++) {
            CUDA_CHECK(cudaMalloc(&ring_r[il][ph], RINGN * 4));
            CUDA_CHECK(cudaMalloc(&S_r[il][ph], SN * 4));
        }
        CUDA_CHECK(cudaMalloc(&ring_c[il], RINGN * 4));
        CUDA_CHECK(cudaMalloc(&S_c[il], SN * 4));
    }
    for (int L = 0; L < WMAXT; L++) {
        CUDA_CHECK(cudaMalloc(&convo_r[L], CH * 4));
        CUDA_CHECK(cudaMalloc(&convo_c[L], CH * 4));
        CUDA_CHECK(cudaMalloc(&oo_r[L], ON * 4));
        CUDA_CHECK(cudaMalloc(&oo_c[L], ON * 4));
        CUDA_CHECK(cudaMalloc(&qkv_in[L], CH * 4));
        CUDA_CHECK(cudaMalloc(&g_in[L], HEADS * 4));
        CUDA_CHECK(cudaMalloc(&beta_in[L], HEADS * 4));
    }
    CUDA_CHECK(cudaMalloc(&w4, (size_t)CH * 4 * 4));
    CUDA_CHECK(cudaMalloc(&rec_q, (size_t)(WMAXT - 1) * CH * 4));
    CUDA_CHECK(cudaMalloc(&rec_c, (size_t)(WMAXT - 1) * CH * 4));
    CUDA_CHECK(cudaMalloc(&rec_g, (size_t)(WMAXT - 1) * HEADS * 4));
    CUDA_CHECK(cudaMalloc(&rec_b, (size_t)(WMAXT - 1) * HEADS * 4));
    CUDA_CHECK(cudaMalloc(&ring_f, RINGN * 4));
    CUDA_CHECK(cudaMalloc(&S_f, SN * 4));
    CUDA_CHECK(cudaMalloc(&o_scr, (size_t)(WMAXT - 1) * ON * 4));

    std::vector<float> big(SN);
    const int WS[] = {2, 5, 12};
    long leg_diffs = 0;

    for (int wi = 0; wi < 3; wi++) {
        const int W = WS[wi];
        const uint32_t base = 0x73A1u ^ (uint32_t)(W << 16);
        // identical seeded committed state into BOTH paths; fresh per config
        // (the chain mutates state, configs must not couple). Reference
        // roles 1..W-1 get distinct junk seeds -- the chain overwrites them
        // before reading, and distinct bytes prove it (zeros could mask a
        // missed write).
        for (int il = 0; il < NLT; il++) {
            for (int ph = 0; ph < WMAXT; ph++) {
                fill_rand(big.data(), RINGN, base + 101u * (uint32_t)(il * 16 + ph), 0.8f);
                CUDA_CHECK(cudaMemcpy(ring_r[il][ph], big.data(), RINGN * 4,
                                      cudaMemcpyHostToDevice));
                if (ph == 0)
                    CUDA_CHECK(cudaMemcpy(ring_c[il], big.data(), RINGN * 4,
                                          cudaMemcpyHostToDevice));
                fill_rand(big.data(), SN, base + 977u * (uint32_t)(il * 16 + ph), 0.6f);
                CUDA_CHECK(cudaMemcpy(S_r[il][ph], big.data(), SN * 4,
                                      cudaMemcpyHostToDevice));
                if (ph == 0)
                    CUDA_CHECK(cudaMemcpy(S_c[il], big.data(), SN * 4,
                                          cudaMemcpyHostToDevice));
            }
        }
        for (int L = 0; L < WMAXT; L++) { // shared read-only per-lane inputs
            fill_rand(big.data(), CH, base + 3u * (uint32_t)L + 1u, 0.9f);
            CUDA_CHECK(cudaMemcpy(qkv_in[L], big.data(), CH * 4, cudaMemcpyHostToDevice));
            fill_rand(big.data(), HEADS, base + 3u * (uint32_t)L + 2u, 1.0f);
            CUDA_CHECK(cudaMemcpy(g_in[L], big.data(), HEADS * 4, cudaMemcpyHostToDevice));
            fill_rand(big.data(), HEADS, base + 3u * (uint32_t)L + 3u, 1.0f);
            CUDA_CHECK(cudaMemcpy(beta_in[L], big.data(), HEADS * 4,
                                  cudaMemcpyHostToDevice));
        }
        fill_rand(big.data(), (int64_t)CH * 4, base + 0xC0FFEEu, 0.5f);
        CUDA_CHECK(cudaMemcpy(w4, big.data(), (size_t)CH * 4 * 4, cudaMemcpyHostToDevice));

        long cdiffs = 0, fdiffs = 0;
        int64_t mu = 0;
        for (int il = 0; il < NLT; il++) {
            // per-path output poison: "kernel wrote nothing" can't pass
            for (int L = 0; L < W; L++) {
                CUDA_CHECK(cudaMemset(convo_r[L], 0xAA, CH * 4));
                CUDA_CHECK(cudaMemset(convo_c[L], 0xBB, CH * 4));
                CUDA_CHECK(cudaMemset(oo_r[L], 0xAA, ON * 4));
                CUDA_CHECK(cudaMemset(oo_c[L], 0xBB, ON * 4));
            }
            CUDA_CHECK(cudaMemset(rec_q, 0xCC, (size_t)(WMAXT - 1) * CH * 4));
            CUDA_CHECK(cudaMemset(rec_c, 0xCC, (size_t)(WMAXT - 1) * CH * 4));
            CUDA_CHECK(cudaMemset(rec_g, 0xCC, (size_t)(WMAXT - 1) * HEADS * 4));
            CUDA_CHECK(cudaMemset(rec_b, 0xCC, (size_t)(WMAXT - 1) * HEADS * 4));
            // l2norm3 lane lists (both paths run the SAME kernel on their
            // own convout replicas; junk pointers pad the dead slots)
            q27k::P3 lr{}, lc{};
            for (int s = 0; s < W_PLUMB; s++) { lr.p[s] = d_junk_y[s]; lc.p[s] = d_junk_y[s]; }
            for (int L = 0; L < W; L++) { lr.p[L] = convo_r[L]; lc.p[L] = convo_c[L]; }

            // REFERENCE: the serial role chain (the retired rotation's exact
            // launch sequence at perm=0 -- role ph is physically buffer ph)
            q27k::conv_step(ring_r[il][0], ring_r[il][0], qkv_in[0], w4, convo_r[0], CH, 0);
            for (int L = 1; L < W; L++)
                q27k::conv_step(ring_r[il][L - 1], ring_r[il][L], qkv_in[L], w4, convo_r[L],
                                CH, 0);
            q27k::l2norm3(lr, 32, SKD, 1e-6f, 0, W);
            q27k::delta_step(S_r[il][0], S_r[il][0], convo_r[0], g_in[0], beta_in[0],
                             oo_r[0], 0);
            for (int L = 1; L < W; L++)
                q27k::delta_step(S_r[il][L - 1], S_r[il][L], convo_r[L], g_in[L], beta_in[L],
                                 oo_r[L], 0);

            // CHUNK path: the M1 engine launch sequence (gdn_mix, verbatim
            // shape) -- lane 0 in place on committed, chunks for lanes 1..W-1,
            // record after l2norm3.
            q27k::CP3 qs{}, gs{}, bs{}, cc{};
            q27k::P3 oc3{};
            for (int s = 0; s < W_PLUMB; s++) {
                qs.p[s] = d_junk_y[s]; gs.p[s] = d_junk_y[s]; bs.p[s] = d_junk_y[s];
                cc.p[s] = d_junk_y[s]; oc3.p[s] = d_junk_y[s];
            }
            for (int L = 0; L < W; L++) {
                qs.p[L] = qkv_in[L]; gs.p[L] = g_in[L]; bs.p[L] = beta_in[L];
                cc.p[L] = convo_c[L]; oc3.p[L] = oo_c[L];
            }
            q27k::conv_step(ring_c[il], ring_c[il], qkv_in[0], w4, convo_c[0], CH, 0);
            if (W > 1)
                q27k::gdn_conv_chunk3(ring_c[il], qs, w4, lc, CH, W - 1, 0);
            q27k::l2norm3(lc, 32, SKD, 1e-6f, 0, W);
            if (W > 1)
                q27k::gdn_record3(qs, cc, gs, bs, rec_q, rec_c, rec_g, rec_b, CH, HEADS,
                                  W - 1, 0);
            q27k::delta_step(S_c[il], S_c[il], convo_c[0], g_in[0], beta_in[0], oo_c[0], 0);
            if (W > 1)
                q27k::gdn_delta_chunk3(S_c[il], cc, gs, bs, oc3, W - 1, 0);
            CUDA_CHECK(cudaDeviceSynchronize());

            // (a) live per-lane outputs + committed state == state-after-lane-0
            for (int L = 0; L < W; L++) {
                cdiffs += seam_cmp(convo_r[L], convo_c[L], CH, &mu);
                cdiffs += seam_cmp(oo_r[L], oo_c[L], ON, &mu);
            }
            cdiffs += seam_cmp(ring_r[il][0], ring_c[il], RINGN, &mu);
            cdiffs += seam_cmp(S_r[il][0], S_c[il], SN, &mu);

            // (b) the commit Fold, every m in 1..W: fold m-1 recorded rows
            // from committed (post-lane-0) state; must land bitwise on the
            // chain's role m-1.
            for (int m = 1; m <= W; m++) {
                CUDA_CHECK(cudaMemcpy(ring_f, ring_c[il], RINGN * 4,
                                      cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(S_f, S_c[il], SN * 4, cudaMemcpyDeviceToDevice));
                if (m > 1) {
                    q27k::conv_ring_update(ring_f, rec_q, CH, m - 1, 0);
                    q27k::delta_scan_seq(S_f, rec_c, rec_g, rec_b, o_scr, m - 1, 0);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                fdiffs += seam_cmp(ring_f, ring_r[il][m - 1], RINGN, &mu);
                fdiffs += seam_cmp(S_f, S_r[il][m - 1], SN, &mu);
            }
        }
        printf("  chunk  W=%2d layers=%d  diffs=%ld  %s\n", W, NLT, cdiffs,
               cdiffs == 0 ? "PASS" : "FAIL");
        printf("  fold   W=%2d m=1..%d  diffs=%ld  %s", W, W, fdiffs,
               fdiffs == 0 ? "PASS" : "FAIL");
        if (cdiffs + fdiffs) printf("  (max %lld ulp)", (long long)mu);
        printf("\n");
        CHECK(cdiffs == 0, "chunk W=%d: %ld floats differ (DOA for M1)", W, cdiffs);
        CHECK(fdiffs == 0, "fold W=%d: %ld floats differ (DOA for M1)", W, fdiffs);
        leg_diffs += cdiffs + fdiffs;
    }
    if (leg_diffs == 0) printf("CHUNK+FOLD BITWISE: ALL PASS\n");
    else printf("CHUNK+FOLD BITWISE: FAIL (%ld diffs -- DOA, do not land M1)\n", leg_diffs);

    for (int il = 0; il < NLT; il++) {
        for (int ph = 0; ph < WMAXT; ph++) {
            CUDA_CHECK(cudaFree(ring_r[il][ph]));
            CUDA_CHECK(cudaFree(S_r[il][ph]));
        }
        CUDA_CHECK(cudaFree(ring_c[il]));
        CUDA_CHECK(cudaFree(S_c[il]));
    }
    for (int L = 0; L < WMAXT; L++) {
        CUDA_CHECK(cudaFree(convo_r[L]));
        CUDA_CHECK(cudaFree(convo_c[L]));
        CUDA_CHECK(cudaFree(oo_r[L]));
        CUDA_CHECK(cudaFree(oo_c[L]));
        CUDA_CHECK(cudaFree(qkv_in[L]));
        CUDA_CHECK(cudaFree(g_in[L]));
        CUDA_CHECK(cudaFree(beta_in[L]));
    }
    CUDA_CHECK(cudaFree(w4));
    CUDA_CHECK(cudaFree(rec_q));
    CUDA_CHECK(cudaFree(rec_c));
    CUDA_CHECK(cudaFree(rec_g));
    CUDA_CHECK(cudaFree(rec_b));
    CUDA_CHECK(cudaFree(ring_f));
    CUDA_CHECK(cudaFree(S_f));
    CUDA_CHECK(cudaFree(o_scr));
}

int main(int argc, char** argv) {
    const char* path =
        argc > 1 ? argv[1] : "/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27";
    q27::Model m = q27::Model::open(path);
    q27::DeviceModel dm(m);

    // shape discovery, vgemm_test style -- never hardcode a layer index
    auto first_with = [&](const char* leaf) -> std::string {
        for (int il = 0; il < 80; il++) {
            char b[96];
            snprintf(b, sizeof b, "blk.%d.%s", il, leaf);
            if (m.find(b)) return b;
        }
        fprintf(stderr, "ninv_test: no layer carries %s\n", leaf);
        exit(1);
    };
    const std::string n_ffn_down = first_with("ffn_down.weight");   // Q4, 5120 x 17408
    const std::string n_ssm_out = first_with("ssm_out.weight");     // Q8, 5120 x 6144
    const std::string n_ssm_alpha = first_with("ssm_alpha.weight"); // F16, 48 x 5120
    // Q4 head source for the tall-head kernel rows: output_q4.weight (the
    // v1.3/v1.4/q8 draft-head copy) if present, else the single Q4 head
    // (q4s tier). A pure-Q8 head tier has no Q4 head shape -- skip those
    // rows (the ffn_down rows still exercise gemv_q4_n/vgemm_q4 on the
    // wide-K shape, so Q4 kernel coverage survives).
    const char* head_q4_name = m.find("output_q4.weight") ? "output_q4.weight"
                               : (m.find("output.weight") && m.get("output.weight").dtype ==
                                                                  q27::DType::Q4_G64)
                                   ? "output.weight"
                                   : nullptr;
    const bool have_q4_head = head_q4_name != nullptr;
    if (!have_q4_head)
        fprintf(stderr,
                "ninv_test: no Q4 head tensor (pure-Q8 tier) -- skipping the 2 head-Q4 "
                "rows; ffn_down still covers the Q4 kernels\n");
    const q27::DevTensor& w_down = dm.upload(n_ffn_down);
    const q27::DevTensor& w_head = dm.upload(have_q4_head ? head_q4_name : n_ffn_down); // Q4
    const q27::DevTensor& w_sout = dm.upload(n_ssm_out);
    const q27::DevTensor& w_alpha = dm.upload(n_ssm_alpha);

    // buffers
    h_scratch.resize(MAXC);
    for (int t = 0; t < NPAY; t++) {
        CUDA_CHECK(cudaMalloc(&d_pay_x[t], MAXC * 4));
        pay_xq[t] = q27k::xquant_alloc(MAXC);
        CUDA_CHECK(cudaMalloc(&d_y_run[0][t], MAXR * 4));
        CUDA_CHECK(cudaMalloc(&d_y_run[1][t], MAXR * 4));
    }
    for (int s = 0; s < W_PLUMB; s++) {
        CUDA_CHECK(cudaMalloc(&d_junk_x[s], MAXC * 4));
        junk_xq[s] = q27k::xquant_alloc(MAXC);
        CUDA_CHECK(cudaMalloc(&d_junk_y[s], MAXR * 4));
    }
    // payload activations: fixed seed, then quantized ONCE via the engine's own
    // quantize3 -- both runs of every case share these exact device bytes.
    {
        q27k::CP3 xs{};
        q27k::XQ3 xq{};
        for (int t = 0; t < NPAY; t++) {
            fill_rand(h_scratch.data(), MAXC, 0x51A1u + 977u * (uint32_t)t,
                      0.5f + 0.1f * (float)t);
            CUDA_CHECK(cudaMemcpy(d_pay_x[t], h_scratch.data(), MAXC * 4,
                                  cudaMemcpyHostToDevice));
            xs.p[t] = d_pay_x[t];
            xq.q[t] = pay_xq[t];
        }
        q27k::quantize3(xs, MAXC, xq, 0, NPAY);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // vgemm workspace: sized exactly as the engine sizes it (max over shapes)
    const q27::DevTensor* wl[3] = {&w_down, &w_head, &w_sout};
    size_t wsb = q27k::vgemm_ws_bytes(wl, 3);
    float* d_ws;
    CUDA_CHECK(cudaMalloc(&d_ws, wsb));

    printf("== ninv: lane output invariance vs T and slot (W_PLUMB=%d, ws %.2f MB) ==\n",
           W_PLUMB, wsb / 1e6);
    struct Row { const char* tag; Fam fam; const q27::DevTensor* w; uint32_t salt; };
    const Row rows[] = {
        {"vgemm_q4[ffn_down]", F_VGEMM, &w_down, 0x10u},  // MODE 1 (z>1) + reduce
        {"vgemm_q4[head]", F_VGEMM, &w_head, 0x20u},      // MODE 0 (z=1)
        {"vgemm_q8[ssm_out]", F_VGEMM, &w_sout, 0x30u},   // MODE 1 (no Q8 z=1 shape exists)
        {"gemv_q4_n[ffn_down]", F_GEMV, &w_down, 0x40u},
        {"gemv_q4_n[head]", F_GEMV, &w_head, 0x50u},
        {"gemv_q8_n[ssm_out]", F_GEMV, &w_sout, 0x60u},
        {"gemv_f16_3[ssm_alpha]", F_F16, &w_alpha, 0x70u},
    };
    // family verdicts aggregate the per-weight tables
    int fam_fail[3] = {0, 0, 0};
    for (const Row& r : rows) {
        // skip the two head-Q4 rows when the tier has no Q4 head (w_head
        // aliases ffn_down there, already covered by its own rows)
        if (!have_q4_head && (r.w == &w_head)) continue;
        // vgemm has no T3_G128 leg (vgemm_verify refuses it and the engine
        // serves every width from gemv_t3_n, which the F_GEMV row covers)
        if (r.fam == F_VGEMM && r.w->dtype == q27::DType::T3_G128) {
            printf("\n-- %s: T3_G128 weight, vgemm not eligible (gemv_t3_n row is the contract), skip\n", r.tag);
            continue;
        }
        fam_fail[r.fam] += run_table(r.tag, r.fam, *r.w, d_ws, r.salt);
    }

    printf("\nfamily verdicts: vgemm_verify %s | gemv_q4_n/gemv_q8_n %s | gemv_f16_3 %s\n",
           fam_fail[F_VGEMM] ? "FAIL" : "PASS", fam_fail[F_GEMV] ? "FAIL" : "PASS",
           fam_fail[F_F16] ? "FAIL" : "PASS");

    // P2 exit review: the single/multi-lane seam (file-header SEAM LEG note).
    seam_leg(dm, w_down, w_head, w_sout);
    // Measured 2026-07-16 on BOTH arches (5090/sm_120 and 3090/sm_86 -- ptxas
    // contraction is per-arch): ALL PASS, every pair, T in {2,4}. The seam IS
    // bitwise, so it joins the gated contract: a future divergence (compiler
    // flag, launch_bounds retier, kernel edit) must fail this binary, not
    // silently downgrade the fused-vs-solo margin claim.
    fails += seam_pairs_differ;

    // M1: verify chunk kernels + commit Fold vs the serial chain (file-header
    // CHUNK+FOLD LEG note). CHECK() inside feeds `fails` directly -- a
    // mismatch is a hard failure (DOA for M1), not an A1 tolerance downgrade.
    chunk_fold_leg();

    if (fails == 0) printf("NINV ALL PASS\n");
    else printf("ninv_test: %d FAILURES -- finding, not a bug: contract downgrades per A1\n",
                fails);
    return fails ? 1 : 0;
}
