// Same-token, serial teacher-forced logit capture. Build twice: q27/Metal,
// or -DQ27_PRISM_REFERENCE against the pinned Prism llama.cpp C API.
// This is an offline gate, not a serving path. Runs one model per process.
#ifdef Q27_PRISM_REFERENCE
#include "llama.h"
#include "ggml-backend.h"
#else
#include "../src/metal/metal_engine.h"
#endif
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("usage: probe MODEL TOKENS.u32 LOGITS.f32");
        std::ifstream input(argv[2],std::ios::binary|std::ios::ate);
        if(!input) throw std::runtime_error("cannot open tokens");
        const auto size=input.tellg();
        if(size<=0 || size%4 || size>4096*4) throw std::runtime_error("expected 1..4096 uint32 tokens");
        std::vector<uint32_t> tokens(static_cast<size_t>(size)/4);
        input.seekg(0);input.read(reinterpret_cast<char*>(tokens.data()),size);
        if(!input) throw std::runtime_error("short token read");
        for(auto id:tokens) if(id>=248320) throw std::runtime_error("token outside vocabulary");
        std::ofstream out(argv[3],std::ios::binary);
        if(!out) throw std::runtime_error("cannot open logit output");
        const uint32_t context=std::max<uint32_t>(128,tokens.size()+32);
        // Long-context runs: Q27_PROBE_STRIDE=k keeps every k-th position's
        // logits (all positions still run); Q27_PROBE_KV=turbo3|q8 (q27 only).
        const char* stride_env=std::getenv("Q27_PROBE_STRIDE");
        const size_t stride=stride_env?std::strtoul(stride_env,nullptr,10):1;
        if(stride<1) throw std::runtime_error("Q27_PROBE_STRIDE must be >= 1");
        const char* kv_env=std::getenv("Q27_PROBE_KV");
        [[maybe_unused]] const bool compressed_kv=kv_env && std::strcmp(kv_env,"fp16");
        // Q27_PROBE_PREFILL=chunk (q27 only): for each kept position i, reset
        // and ingest tokens[0..i) through 96-token prefill chunks, then step
        // tokens[i] serially -- the engine's own prefill shape -- so batched
        // prefill is compared position by position against the reference.
        const char* prefill_env=std::getenv("Q27_PROBE_PREFILL");
        const bool chunked=prefill_env && !std::strcmp(prefill_env,"chunk");
        if(prefill_env && !chunked && std::strcmp(prefill_env,"serial"))
            throw std::runtime_error("Q27_PROBE_PREFILL must be serial or chunk");
#ifdef Q27_PRISM_REFERENCE
        ggml_backend_load_all();
        llama_backend_init();
        auto mp=llama_model_default_params();mp.n_gpu_layers=999;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1],mp),llama_model_free);
        if(!model) throw std::runtime_error("reference model load failed");
        if(llama_vocab_n_tokens(llama_model_get_vocab(model.get()))!=248320)
            throw std::runtime_error("reference vocabulary mismatch");
        auto cp=llama_context_default_params();
        cp.n_ctx=context;cp.n_batch=1;cp.n_ubatch=1;cp.n_seq_max=1;
        cp.n_threads=4;cp.n_threads_batch=4;
        cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context,decltype(&llama_free)> engine(
            llama_init_from_model(model.get(),cp),llama_free);
        if(!engine) throw std::runtime_error("reference context failed");
        if(compressed_kv||chunked) throw std::runtime_error("Q27_PROBE_KV/Q27_PROBE_PREFILL apply to the q27 probe only");
#else
        q27::MetalEngine engine(argv[1],context,q27::parse_kv_kind(kv_env?kv_env:"fp16"));
        if(chunked && !engine.chunked_prefill())
            throw std::runtime_error("Q27_PROBE_PREFILL=chunk: this pack has no chunked prefill");
        if(!chunked && engine.chunked_prefill()) engine.set_chunked_prefill(false);
        // Q27_PROBE_KV_ATTRIB=1|2|4: fp16 cache with K-only / V-only turbo3
        // round-trip, or e4m3 round-trip of both sides (MetalEngine::set_kv_attrib).
        if(const char* attrib=std::getenv("Q27_PROBE_KV_ATTRIB"); attrib && *attrib)
            engine.set_kv_attrib(static_cast<uint32_t>(std::strtoul(attrib,nullptr,10)));
#endif
        for(size_t i=0;i<tokens.size();++i) {
#ifdef Q27_PRISM_REFERENCE
            llama_token token=static_cast<llama_token>(tokens[i]);
            auto batch=llama_batch_get_one(&token,1);
            if(llama_decode(engine.get(),batch)) throw std::runtime_error("reference decode failed");
            const float* logits=llama_get_logits_ith(engine.get(),-1);
            if(!logits) throw std::runtime_error("reference logits absent");
#else
            if(chunked && i%stride!=stride-1) continue;
            if(chunked) {
                engine.reset();
                size_t done=0;
                while(i-done>=2) {
                    const uint32_t n=static_cast<uint32_t>(std::min<size_t>(
                        q27::MetalEngine::prefill_chunk_max(),i-done));
                    engine.prefill_chunk(tokens.data()+done,n);
                    done+=n;
                }
                for(;done<i;++done) engine.step(tokens[done]);
            }
            engine.step(tokens[i]);
            auto row=engine.read_logits();
            const float* logits=row.data();
#endif
            if(i%stride!=stride-1) continue;
            out.write(reinterpret_cast<const char*>(logits),248320*sizeof(float));
            if(!out) throw std::runtime_error("short logit write");
            std::cerr<<"position "<<i<<" argmax "<<(std::max_element(logits,logits+248320)-logits)<<'\n';
        }
        out.close();
        if(!out) throw std::runtime_error("logit close failed");
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
