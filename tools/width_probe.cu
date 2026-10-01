// width_probe -- which lane kernel breaks width invariance? (2026-09-18)
//
// CLAIM UNDER TEST: lane 0 of the multi-lane verify forward (the pending
// token) produces the same bytes whatever the round width. The server
// identity gates say otherwise from width 4 up on both the Qwen3.8 tiers
// and the Bonsai 2 pack (BUILDLOG 2026-09-18 (au)), while ninv_test says
// every weight kernel is N-invariant. This probe walks the verify skeleton
// itself, layer by layer, at widths {2, 4, 8} on the SAME prefilled state
// (re-prefilled per width, since lane 0 commits GDN state in place) and
// reports the first layer whose lane-0 residual differs, plus the logits
// delta -- so the culprit is named as "attention layer N" or "GDN layer N".
//
// Q27_GEMM_MIN is pinned to 99 (GEMV family at every width; the vgemm
// family is a known tolerance class). Q27_KV as the caller sets it.
//
// Build (the fused_smoke line):
//   /usr/local/cuda/bin/nvcc -O2 -std=c++17
//     -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120
//     -Xcompiler -Wall tools/width_probe.cu src/blocks.cu src/prefill.cu
//     src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu
//     src/loader.cpp src/dflash2.cu build/pf4.o -o build/width_probe
// Run: build/width_probe <model.q27> [--tokens-file ids] [--batched]
//
// RESULT 2026-09-18 (3090, Bonsai 2 pure T2 and Qwen3.8 default, fp8 KV,
// positions 70 and 617, serial and batched prefill): every test bitwise --
// lane 0 at widths 2/4/8, folds T=1..4, rejected lanes at widths 2/4/8,
// truncation rewinds m=1..4 of 5, captured-graph replay at widths 4/8. The
// serving near-tie flips at width >= 4 (BUILDLOG (au)) are therefore not a
// single-round mechanism; the next instrument is a per-round lane-0 logits
// dump in the server to find the first differing round in situ.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/engine.cuh"

static std::vector<int> read_ids(const char* path) {
    std::vector<int> ids;
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int v;
    while (fscanf(f, "%d", &v) == 1) { ids.push_back(v); int c = fgetc(f); if (c == EOF) break; }
    fclose(f);
    return ids;
}

static void prefill_serial(Engine& e, const std::vector<int>& prompt) {
    e.reset();
    e.have_snap = false;
    e.snap_toks.clear();
    e.ckpt_clear();
    for (size_t i = 0; i < prompt.size(); i++) {
        e.step_with(prompt[i]);
        if (i + 1 < prompt.size()) {
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            int nt = prompt[i + 1];
            CUDA_CHECK(cudaMemcpyAsync(e.d_token, &nt, 4, cudaMemcpyHostToDevice, e.stm));
        }
    }
    // the round bookkeeping position (fused_smoke's epilogue): lanes sit at P+1..
    int P = (int)prompt.size() - 1;
    CUDA_CHECK(cudaMemcpyAsync(e.d_P, &P, 4, cudaMemcpyHostToDevice, e.stm));
    CUDA_CHECK(cudaStreamSynchronize(e.stm));
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: width_probe <model.q27> [--tokens-file ids]\n"); return 2; }
    setenv("Q27_GEMM_MIN", "99", 1);
    unsetenv("Q27_SUFFIX");
    std::vector<int> prompt = {760, 6511, 314, 9338, 369};
    bool batched = false; // --batched: the server's prefill (generate_prefill) instead of per-token step_with
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens-file") && i + 1 < argc) prompt = read_ids(argv[++i]);
        else if (!strcmp(argv[i], "--batched")) batched = true;
    }
    q27::Model model = q27::Model::open(argv[1]);
    validate_arch(model);
    q27::DeviceModel dm(model);
    dm.upload_all();
    Engine e(model, dm, 4096);
    e.build_graph();
    e.build_spec_graphs();
    fprintf(stderr, "width_probe: %s, prompt %zu tokens, bonsai2=%d, KV=%s, prefill=%s\n", argv[1],
            prompt.size(), (int)e.bonsai2, getenv("Q27_KV") ? getenv("Q27_KV") : "default", batched ? "batched" : "serial");

    const int widths[3] = {2, 4, 8};
    // per width: lane-0 residual after each layer (N_LAYER x N_EMBD) + logits
    std::vector<std::vector<float>> H(3, std::vector<float>((size_t)N_LAYER * N_EMBD));
    std::vector<std::vector<float>> LG(3, std::vector<float>(VOCAB));
    for (int wi = 0; wi < 3; wi++) {
        const int w = widths[wi];
        if (batched) {
            e.reset(); e.have_snap = false; e.snap_toks.clear(); e.ckpt_clear();
            int P = 0;
            if (!e.generate_prefill(prompt, -1, &P)) { fprintf(stderr, "generate_prefill refused\n"); return 1; }
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
        } else {
            prefill_serial(e, prompt);
        }
        // lanes: 0 = pending (d_token, left by step_with), 1..w-1 = fixed junk
        // draft ids; positions from prep_round (P+1.. per lane)
        for (int k = 0; k + 1 < W_PLUMB; k++) {
            int junk = 1000 + 37 * k;
            CUDA_CHECK(cudaMemcpyAsync(e.d_draft_L[k], &junk, 4, cudaMemcpyHostToDevice, e.stm));
        }
        q27k::prep_round(e.d_P, e.d_token, e.lane_pos(), e.mtp_pos(), W_MAX, D_MAX_MTP,
                         e.d_outcome, e.stm);
        e.set_round_width(w);
        Engine::LaneView v = e.solo_view();
        v.vw = w;
        // --- spec_verify_forward's skeleton, with a lane-0 residual tap per layer ---
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales, v.vtok, N_EMBD,
                     LANESV(v, h), v.stm, v.vw);
        if (e.bonsai2) e.bz_unrot_lanes(v.h, v.vw, v.stm);
        q27k::CP3 Hc LANESV(v, h), Yc LANESV(v, y);
        q27k::P3 Hm LANESV(v, h), X1m LANESV(v, x1);
        for (int il = 0; il < N_LAYER; il++) {
            const float* an = (const float*)e.T(il, "attn_norm.weight").data;
            if (e.bonsai2) e.bz_norm3_rot_q5(v, Hc, an, X1m, !e.is_attn_layer(il));
            else e.rmsnorm3q5(v, Hc, an, X1m, N_EMBD);
            if (e.is_attn_layer(il)) e.attn_pair(il, v, true);
            else e.gdn_pair(il, v, true);
            q27k::add3(Hm, Yc, N_EMBD, v.stm, v.vw);
            const float* pn = (const float*)e.T(il, "post_attention_norm.weight").data;
            if (e.bonsai2) e.bz_norm3_rot_q5(v, Hc, pn, X1m, false);
            else e.rmsnorm3q5(v, Hc, pn, X1m, N_EMBD);
            e.ffn_pair(il, v, true);
            q27k::add3(Hm, Yc, N_EMBD, v.stm, v.vw);
            CUDA_CHECK(cudaMemcpyAsync(H[wi].data() + (size_t)il * N_EMBD, v.h[0], N_EMBD * 4,
                                       cudaMemcpyDeviceToHost, v.stm));
        }
        const float* on = (const float*)dm.get("output_norm.weight").data;
        if (e.bonsai2) e.bz_norm3_rot_q5(v, Hc, on, X1m, false);
        else e.rmsnorm3q5(v, Hc, on, X1m, N_EMBD);
        const char* vhead = dm.model_has("output_q4.weight") ? "output_q4.weight" : "output.weight";
        e.mm5(v, dm.get(vhead), v.lg);
        CUDA_CHECK(cudaMemcpyAsync(LG[wi].data(), v.lg[0], VOCAB * 4, cudaMemcpyDeviceToHost, v.stm));
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        int am = 0;
        for (int i = 1; i < VOCAB; i++) if (LG[wi][i] > LG[wi][am]) am = i;
        fprintf(stderr, "  width %d: lane-0 argmax %d  logit[argmax] %.6f\n", w, am, LG[wi][am]);
    }
    // --- fold test: a round that accepts the plain path's own next tokens,
    // then flush_fold, then one plain step; vs. the same tokens stepped one at
    // a time. Bitwise logits after the extra step = the accepted-lane fold
    // (delta_scan_seq + conv_ring_update + lane KV rows) commits exactly what
    // per-token decode commits.
    auto do_prefill = [&]() {
        if (batched) {
            e.reset(); e.have_snap = false; e.snap_toks.clear(); e.ckpt_clear();
            int P = 0;
            if (!e.generate_prefill(prompt, -1, &P)) { fprintf(stderr, "generate_prefill refused\n"); exit(1); }
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
        } else prefill_serial(e, prompt);
    };
    int fold_fails = 0;
    for (int T = 1; T <= 4; T++) { // T accepted drafts -> fold of T rows; round width T+1
        // (A) plain: step the pending token and its T+1 greedy successors one at a time
        do_prefill();
        // toks[0] = pending after prefill; plain steps toks[0..T+1] (T+2 steps),
        // the spec branch consumes toks[0..T] in one round and then steps toks[T+1]
        std::vector<int> toks(T + 3);
        CUDA_CHECK(cudaMemcpy(&toks[0], e.d_token, 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i <= T + 1; i++) {
            e.step_with(toks[i]);
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpy(&toks[i + 1], e.d_token, 4, cudaMemcpyDeviceToHost));
        }
        std::vector<float> la(VOCAB);
        CUDA_CHECK(cudaMemcpy(la.data(), e.logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        // (B) spec: one round at width T+1 with drafts = toks[1..T] (all accepted),
        // fold, then the same extra step of toks[T+1]
        do_prefill();
        for (int k = 0; k < T; k++)
            CUDA_CHECK(cudaMemcpyAsync(e.d_draft_L[k], &toks[k + 1], 4, cudaMemcpyHostToDevice, e.stm));
        q27k::prep_round(e.d_P, e.d_token, e.lane_pos(), e.mtp_pos(), W_MAX, D_MAX_MTP, e.d_outcome, e.stm);
        e.plain_lanes = false; // let the tail accept drafts
        e.set_round_width(T + 1);
        Engine::LaneView v = e.solo_view();
        v.vw = T + 1;
        e.spec_verify_forward(v);
        e.spec_verify_tail(v);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        int oc[OUTCOME_INTS];
        CUDA_CHECK(cudaMemcpy(oc, e.d_outcome, sizeof oc, cudaMemcpyDeviceToHost));
        int pend = 0;
        CUDA_CHECK(cudaMemcpy(&pend, e.d_token, 4, cudaMemcpyDeviceToHost));
        e.fold_pending = oc[0] - 1;
        e.flush_fold(e.stm);
        // the token graph positions from d_pos (advance kernel), the round from
        // d_P (finish bumped it by n): next write position = P + 1
        int Pn = 0;
        CUDA_CHECK(cudaMemcpy(&Pn, e.d_P, 4, cudaMemcpyDeviceToHost));
        Pn += 1;
        CUDA_CHECK(cudaMemcpyAsync(e.d_pos, &Pn, 4, cudaMemcpyHostToDevice, e.stm));
        e.step_with(toks[T + 1]);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        std::vector<float> lb(VOCAB);
        CUDA_CHECK(cudaMemcpy(lb.data(), e.logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        double lmax = 0; int nd = 0;
        for (int i = 0; i < VOCAB; i++) { double d = std::fabs(la[i] - lb[i]); if (d > 0) nd++; lmax = std::max(lmax, d); }
        const bool ok = (oc[0] == T + 1) && pend == toks[T + 1] && nd == 0;
        if (!ok) fold_fails++;
        printf("  fold T=%d (width %d): accepted n=%d (want %d), pending %s, logits after the next step: %d/%d differ, max|d| %.3e  %s\n",
               T, T + 1, oc[0], T + 1, pend == toks[T + 1] ? "ok" : "WRONG", nd, VOCAB, lmax, ok ? "PASS" : "FAIL");
    }
    // --- rejection test: a round whose drafts are all wrong leaves KV rows at
    // P+2..P+w that plain decode never writes; the next plain step must not
    // see them. (A) plain: step t0, step t1. (B) round at width w with junk
    // drafts (n = 1, pending = t1), then step t1. Bitwise logits = the
    // attention masks the rejected lanes' rows past the committed position.
    int rej_fails = 0;
    for (int w : {2, 4, 8}) {
        do_prefill();
        int t0 = 0, t1 = 0;
        CUDA_CHECK(cudaMemcpy(&t0, e.d_token, 4, cudaMemcpyDeviceToHost));
        e.step_with(t0);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        CUDA_CHECK(cudaMemcpy(&t1, e.d_token, 4, cudaMemcpyDeviceToHost));
        e.step_with(t1);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        std::vector<float> la(VOCAB);
        CUDA_CHECK(cudaMemcpy(la.data(), e.logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        do_prefill();
        for (int k = 0; k + 1 < W_PLUMB; k++) { // junk drafts: never equal the argmax chain
            int junk = 2000 + 41 * k;
            CUDA_CHECK(cudaMemcpyAsync(e.d_draft_L[k], &junk, 4, cudaMemcpyHostToDevice, e.stm));
        }
        q27k::prep_round(e.d_P, e.d_token, e.lane_pos(), e.mtp_pos(), W_MAX, D_MAX_MTP, e.d_outcome, e.stm);
        e.plain_lanes = false;
        e.set_round_width(w);
        Engine::LaneView v = e.solo_view();
        v.vw = w;
        e.spec_verify_forward(v);
        e.spec_verify_tail(v);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        int oc[OUTCOME_INTS];
        CUDA_CHECK(cudaMemcpy(oc, e.d_outcome, sizeof oc, cudaMemcpyDeviceToHost));
        int pend = 0;
        CUDA_CHECK(cudaMemcpy(&pend, e.d_token, 4, cudaMemcpyDeviceToHost));
        e.fold_pending = oc[0] - 1;
        e.flush_fold(e.stm);
        int Pn = 0;
        CUDA_CHECK(cudaMemcpy(&Pn, e.d_P, 4, cudaMemcpyDeviceToHost));
        Pn += 1;
        CUDA_CHECK(cudaMemcpyAsync(e.d_pos, &Pn, 4, cudaMemcpyHostToDevice, e.stm));
        e.step_with(t1);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        std::vector<float> lb(VOCAB);
        CUDA_CHECK(cudaMemcpy(lb.data(), e.logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        double lmax = 0; int nd = 0;
        for (int i = 0; i < VOCAB; i++) { double d = std::fabs(la[i] - lb[i]); if (d > 0) nd++; lmax = std::max(lmax, d); }
        const bool ok = oc[0] == 1 && pend == t1 && nd == 0;
        if (!ok) rej_fails++;
        printf("  reject width %d: accepted n=%d (want 1), pending %s, logits after the next step: %d/%d differ, max|d| %.3e  %s\n",
               w, oc[0], pend == t1 ? "ok" : "WRONG", nd, VOCAB, lmax, ok ? "PASS" : "FAIL");
    }
    // --- truncation test: the server's on_round keeps m < n of an accepted
    // round (budget / tool boundaries); refinish_round rewinds pending + d_P
    // and folds m-1 rows. The CLI never truncates. (A) plain: step toks[0..m]
    // ; (B) width-5 all-accept round, refinish_round(m, 5, P+m), fold, step
    // toks[m]. Bitwise logits = the rewind is exact.
    int trunc_fails = 0;
    for (int m = 1; m <= 4; m++) {
        do_prefill();
        int P0 = 0;
        CUDA_CHECK(cudaMemcpy(&P0, e.d_P, 4, cudaMemcpyDeviceToHost));
        std::vector<int> toks(6);
        CUDA_CHECK(cudaMemcpy(&toks[0], e.d_token, 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i <= 4; i++) { // plain: forward toks[0..m] then read logits -- do all 5 for the draft list
            e.step_with(toks[i]);
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpy(&toks[i + 1], e.d_token, 4, cudaMemcpyDeviceToHost));
        }
        do_prefill();
        std::vector<float> la(VOCAB);
        for (int i = 0; i <= m; i++) { e.step_with(toks[i]); CUDA_CHECK(cudaStreamSynchronize(e.stm)); }
        CUDA_CHECK(cudaMemcpy(la.data(), e.logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        do_prefill();
        for (int k = 0; k < 4; k++)
            CUDA_CHECK(cudaMemcpyAsync(e.d_draft_L[k], &toks[k + 1], 4, cudaMemcpyHostToDevice, e.stm));
        q27k::prep_round(e.d_P, e.d_token, e.lane_pos(), e.mtp_pos(), W_MAX, D_MAX_MTP, e.d_outcome, e.stm);
        e.plain_lanes = false;
        e.set_round_width(5);
        Engine::LaneView v = e.solo_view();
        v.vw = 5;
        e.spec_verify_forward(v);
        e.spec_verify_tail(v);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        int oc[OUTCOME_INTS];
        CUDA_CHECK(cudaMemcpy(oc, e.d_outcome, sizeof oc, cudaMemcpyDeviceToHost));
        int pend = e.refinish_round(m, oc[0], P0 + m);
        e.flush_fold(e.stm);
        int Pn = P0 + m + 1;
        CUDA_CHECK(cudaMemcpyAsync(e.d_pos, &Pn, 4, cudaMemcpyHostToDevice, e.stm));
        e.step_with(toks[m]);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        std::vector<float> lb(VOCAB);
        CUDA_CHECK(cudaMemcpy(lb.data(), e.logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        double lmax = 0; int nd = 0;
        for (int i = 0; i < VOCAB; i++) { double d = std::fabs(la[i] - lb[i]); if (d > 0) nd++; lmax = std::max(lmax, d); }
        const bool ok = oc[0] == 5 && pend == toks[m] && nd == 0;
        if (!ok) trunc_fails++;
        printf("  trunc m=%d of n=%d: pending %s, logits after the next step: %d/%d differ, max|d| %.3e  %s\n",
               m, oc[0], pend == toks[m] ? "ok" : "WRONG", nd, VOCAB, lmax, ok ? "PASS" : "FAIL");
    }
    // --- graph test: the server replays the verify from a captured CUDA graph
    // (d2_verify_exec / verify_graph_w[]); the probe above launched eagerly.
    // Capture the width-8 verify (forward + tail) on e.stm, replay it on the
    // same prefilled state, compare lane-0 logits with the eager run. Bitwise
    // = capture bakes the same kernels and launch parameters.
    int graph_fails = 0;
    for (int w : {4, 8}) {
        auto setup = [&]() {
            do_prefill();
            for (int k = 0; k + 1 < W_PLUMB; k++) {
                int junk = 3000 + 43 * k;
                CUDA_CHECK(cudaMemcpyAsync(e.d_draft_L[k], &junk, 4, cudaMemcpyHostToDevice, e.stm));
            }
            q27k::prep_round(e.d_P, e.d_token, e.lane_pos(), e.mtp_pos(), W_MAX, D_MAX_MTP, e.d_outcome, e.stm);
            e.plain_lanes = false;
            e.set_round_width(w);
        };
        setup();
        Engine::LaneView v = e.solo_view();
        v.vw = w;
        e.spec_verify_forward(v);
        e.spec_verify_tail(v);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        std::vector<float> la(VOCAB), lb(VOCAB);
        CUDA_CHECK(cudaMemcpy(la.data(), v.lg[0], VOCAB * 4, cudaMemcpyDeviceToHost));
        int oca[OUTCOME_INTS], ocb[OUTCOME_INTS];
        CUDA_CHECK(cudaMemcpy(oca, e.d_outcome, sizeof oca, cudaMemcpyDeviceToHost));
        setup();
        cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(e.stm, cudaStreamCaptureModeThreadLocal));
        e.spec_verify_forward(v);
        e.spec_verify_tail(v);
        CUDA_CHECK(cudaStreamEndCapture(e.stm, &g));
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        CUDA_CHECK(cudaGraphLaunch(ge, e.stm));
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        CUDA_CHECK(cudaMemcpy(lb.data(), v.lg[0], VOCAB * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ocb, e.d_outcome, sizeof ocb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGraphExecDestroy(ge)); CUDA_CHECK(cudaGraphDestroy(g));
        double lmax = 0; int nd = 0;
        for (int i = 0; i < VOCAB; i++) { double d = std::fabs(la[i] - lb[i]); if (d > 0) nd++; lmax = std::max(lmax, d); }
        const bool ok = nd == 0 && oca[0] == ocb[0] && oca[OUTCOME_INTS - 1] == ocb[OUTCOME_INTS - 1];
        if (!ok) graph_fails++;
        printf("  graph width %d: eager vs captured replay: lane-0 logits %d/%d differ, max|d| %.3e, outcome n %d/%d pending %d/%d  %s\n",
               w, nd, VOCAB, lmax, oca[0], ocb[0], oca[OUTCOME_INTS - 1], ocb[OUTCOME_INTS - 1], ok ? "PASS" : "FAIL");
    }
    int fails = fold_fails + rej_fails + trunc_fails + graph_fails;
    for (int wi = 1; wi < 3; wi++) {
        int first = -1;
        double lmax = 0;
        for (int il = 0; il < N_LAYER && first < 0; il++)
            for (int i = 0; i < N_EMBD; i++)
                if (H[wi][(size_t)il * N_EMBD + i] != H[0][(size_t)il * N_EMBD + i]) { first = il; break; }
        for (int i = 0; i < VOCAB; i++) lmax = std::max(lmax, (double)std::fabs(LG[wi][i] - LG[0][i]));
        if (first < 0 && lmax == 0)
            printf("  width %d vs 2: lane 0 BITWISE through all %d layers and the head  PASS\n", widths[wi], N_LAYER);
        else {
            fails++;
            printf("  width %d vs 2: first lane-0 residual diff at layer %d (%s)  logits max|d| %.3e  FAIL\n",
                   widths[wi], first, first < 0 ? "head only" : (e.is_attn_layer(first) ? "ATTENTION" : "GDN"), lmax);
            if (first >= 0) {
                double hmax = 0; int cnt = 0;
                for (int i = 0; i < N_EMBD; i++) {
                    double d = std::fabs(H[wi][(size_t)first * N_EMBD + i] - H[0][(size_t)first * N_EMBD + i]);
                    if (d > 0) cnt++;
                    hmax = std::max(hmax, d);
                }
                printf("    layer %d: %d/%d residual elements differ, max|d| %.3e\n", first, cnt, N_EMBD, hmax);
            }
        }
    }
    printf("width_probe: %s\n", fails ? "FAIL" : "PASS (lane 0 width-invariant; folds, rejections, truncations, graph replay bitwise)");
    return fails ? 1 : 0;
}
