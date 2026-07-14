#include "metal_engine.h"
#include "../tokenizer.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace {

uint32_t parse_u32(const std::string& text, const char* option) {
    if (text.empty() || text[0] == '-') throw std::runtime_error(std::string("invalid ") + option + ": " + text);
    size_t consumed = 0;
    unsigned long long value = 0;
    try { value = std::stoull(text, &consumed, 10); }
    catch (...) { throw std::runtime_error(std::string("invalid ") + option + ": " + text); }
    if (consumed != text.size() || value > UINT32_MAX)
        throw std::runtime_error(std::string("invalid ") + option + ": " + text);
    return (uint32_t)value;
}

uint64_t parse_u64(const std::string& text,const char* option) {
    if(text.empty() || text[0]=='-') throw std::runtime_error(std::string("invalid ")+option);
    size_t used=0; unsigned long long value=0;
    try { value=std::stoull(text,&used,10); }
    catch(...) { throw std::runtime_error(std::string("invalid ")+option); }
    if(used!=text.size()) throw std::runtime_error(std::string("invalid ")+option);
    return value;
}
float parse_float(const std::string& text,const char* option) {
    size_t used=0; float value=0;
    try { value=std::stof(text,&used); }
    catch(...) { throw std::runtime_error(std::string("invalid ")+option); }
    if(used!=text.size()) throw std::runtime_error(std::string("invalid ")+option);
    return value;
}

std::vector<uint32_t> parse_tokens(const std::string& text) {
    std::vector<uint32_t> result;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) throw std::runtime_error("empty token id");
        char* end = nullptr;
        unsigned long value = std::strtoul(item.c_str(), &end, 10);
        if (!end || *end || value > UINT32_MAX) throw std::runtime_error("invalid token id: " + item);
        result.push_back((uint32_t)value);
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.q27 tokenizer.tok [--validate-only | --tokens id,id,... | --prompt text] [-n count] [--ctx count] [--mtp width | --suffix width] [--kv fp16|turbo3] [--temperature T --top-p P --top-k K --seed S] [--dump-logits file]\n", argv[0]);
        return 1;
    }
    try {
        std::string model_path=argv[1],tokenizer_path=argv[2],token_list,prompt_text,dump_logits;
        uint32_t count=1,context=128,mtp_width=0,suffix_width=0; q27::SamplingParams sampling;
        bool turbo3_kv = false, validate_only = false;
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--tokens" && i + 1 < argc) token_list = argv[++i];
            else if (arg == "--prompt" && i + 1 < argc) prompt_text = argv[++i];
            else if (arg == "--validate-only") validate_only = true;
            else if (arg == "-n" && i + 1 < argc) count = parse_u32(argv[++i], "-n");
            else if (arg == "--ctx" && i + 1 < argc) context = parse_u32(argv[++i], "--ctx");
            else if (arg == "--mtp" && i + 1 < argc) mtp_width = parse_u32(argv[++i], "--mtp");
            else if (arg == "--suffix" && i + 1 < argc) suffix_width = parse_u32(argv[++i], "--suffix");
            else if (arg == "--dump-logits" && i + 1 < argc) dump_logits = argv[++i];
            else if (arg == "--temperature" && i + 1 < argc) sampling.temperature=parse_float(argv[++i],"--temperature");
            else if (arg == "--top-p" && i + 1 < argc) sampling.top_p=parse_float(argv[++i],"--top-p");
            else if (arg == "--top-k" && i + 1 < argc) sampling.top_k=parse_u32(argv[++i],"--top-k");
            else if (arg == "--seed" && i + 1 < argc) sampling.seed=parse_u64(argv[++i],"--seed");
            else if (arg == "--kv" && i + 1 < argc) {
                std::string mode = argv[++i];
                if (mode == "turbo3") turbo3_kv = true;
                else if (mode != "fp16") throw std::runtime_error("--kv must be fp16 or turbo3");
            }
            else throw std::runtime_error("unknown/incomplete argument: " + arg);
        }
        if (mtp_width && suffix_width) throw std::runtime_error("--mtp and --suffix are mutually exclusive");
        q27::validate_sampling(sampling);
        if(sampling.temperature>0 && (mtp_width || suffix_width))
            throw std::runtime_error("sampling cannot be combined with speculative modes");
        if(!token_list.empty() && !prompt_text.empty()) throw std::runtime_error("--tokens and --prompt are mutually exclusive");
        if(!validate_only && token_list.empty() && prompt_text.empty()) throw std::runtime_error("--tokens or --prompt is required");

        auto start = std::chrono::steady_clock::now();
        // Validate the small tokenizer artifact before allocating any model
        // state, then reject cross-artifact vocabulary mismatches explicitly.
        q27::Tokenizer tokenizer(tokenizer_path);
        if (tokenizer.vocab_size() != q27::MetalEngine::vocabulary_size())
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        std::vector<uint32_t> prompt;
        if(!token_list.empty()) prompt=parse_tokens(token_list);
        else if(!prompt_text.empty()) {
            for(int token:tokenizer.encode(prompt_text)) {
                if(token<0) throw std::runtime_error("tokenizer returned a negative id");
                prompt.push_back((uint32_t)token);
            }
        }
        q27::MetalEngine engine(model_path, context, turbo3_kv);
        auto loaded = std::chrono::steady_clock::now();
        fprintf(stderr, "Metal model ready on %s in %.2f s\n", engine.backend().name().c_str(),
                std::chrono::duration<double>(loaded - start).count());
        if (validate_only) { puts("artifacts and Metal architecture: OK"); return 0; }
        std::vector<uint32_t> generated = sampling.temperature>0 ? engine.generate_sampled(prompt,count,sampling)
                                           : mtp_width ? engine.generate_mtp(prompt,count,mtp_width)
                                           : suffix_width ? engine.generate_suffix(prompt,count,suffix_width)
                                                          : engine.generate(prompt,count);
        if(!dump_logits.empty()) {
            std::vector<float> logits=engine.read_logits();
            FILE* file=fopen(dump_logits.c_str(),"wb");
            if(!file) throw std::runtime_error("cannot write logits: "+dump_logits);
            bool ok=fwrite(logits.data(),sizeof(float),logits.size(),file)==logits.size();
            if(fclose(file)!=0) ok=false;
            if(!ok) throw std::runtime_error("cannot write logits: "+dump_logits);
        }
        auto finished = std::chrono::steady_clock::now();

        std::vector<int> ids(generated.begin(), generated.end());
        printf("generated:%s\n", tokenizer.decode(ids).c_str());
        fprintf(stderr, "%zu tokens in %.2f s (%.2f tok/s), position %u\n", generated.size(),
                std::chrono::duration<double>(finished - loaded).count(),
                generated.size() / std::chrono::duration<double>(finished - loaded).count(),
                engine.position());
        auto spec=engine.last_spec_stats();
        if(mtp_width || suffix_width)
            fprintf(stderr,"speculation: %llu rounds, %llu drafts, %llu accepted (%.1f%%)\n",
                    (unsigned long long)spec.rounds,(unsigned long long)spec.drafted,
                    (unsigned long long)spec.accepted,spec.drafted?100.0*spec.accepted/spec.drafted:0.0);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
