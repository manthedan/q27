// Dump full-vocab logits from a llama.cpp (PrismML fork) model for a token
// sequence: the parity reference for q27's Bonsai 2 path.
//   llama_logits_dump model.gguf tokens.txt out.f32 [n_gpu_layers]
// tokens.txt: whitespace-separated token ids (already-encoded prompt, so the
// two engines see byte-identical token streams). out.f32: [n_tokens, n_vocab]
// float32 row-major, logits at EVERY position (logits_all).
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf tokens.txt out.f32 [ngl]\n", argv[0]); return 2; }
    const int ngl = argc > 4 ? atoi(argv[4]) : 99;
    std::vector<llama_token> toks;
    { std::ifstream f(argv[2]); long long v; while (f >> v) toks.push_back((llama_token)v); }
    if (toks.empty()) { fprintf(stderr, "no tokens\n"); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = ngl;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)toks.size() + 8; cp.n_batch = (uint32_t)toks.size(); cp.n_ubatch = (uint32_t)toks.size();
    cp.n_seq_max = 1; cp.n_threads = 12; cp.n_threads_batch = 12;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_batch b = llama_batch_init((int32_t)toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i] = toks[i]; b.pos[i] = (llama_pos)i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = (int32_t)toks.size();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE* fo = fopen(argv[3], "wb");
    for (size_t i = 0; i < toks.size(); i++) { const float* l = llama_get_logits_ith(ctx, (int32_t)i); fwrite(l, sizeof(float), (size_t)nv, fo); }
    fclose(fo);
    fprintf(stderr, "wrote %zu x %d logits\n", toks.size(), nv);
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
