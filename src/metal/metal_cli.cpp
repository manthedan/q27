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
        fprintf(stderr, "usage: %s model.q27 tokenizer.tok [--validate-only | --tokens id,id,...] [-n count] [--ctx count] [--mtp width | --suffix width] [--kv fp16|turbo3]\n", argv[0]);
        return 1;
    }
    try {
        std::string model_path = argv[1], tokenizer_path = argv[2], token_list;
        uint32_t count = 1, context = 128, mtp_width = 0, suffix_width = 0;
        bool turbo3_kv = false, validate_only = false;
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--tokens" && i + 1 < argc) token_list = argv[++i];
            else if (arg == "--validate-only") validate_only = true;
            else if (arg == "-n" && i + 1 < argc) count = parse_u32(argv[++i], "-n");
            else if (arg == "--ctx" && i + 1 < argc) context = parse_u32(argv[++i], "--ctx");
            else if (arg == "--mtp" && i + 1 < argc) mtp_width = parse_u32(argv[++i], "--mtp");
            else if (arg == "--suffix" && i + 1 < argc) suffix_width = parse_u32(argv[++i], "--suffix");
            else if (arg == "--kv" && i + 1 < argc) {
                std::string mode = argv[++i];
                if (mode == "turbo3") turbo3_kv = true;
                else if (mode != "fp16") throw std::runtime_error("--kv must be fp16 or turbo3");
            }
            else throw std::runtime_error("unknown/incomplete argument: " + arg);
        }
        if (mtp_width && suffix_width) throw std::runtime_error("--mtp and --suffix are mutually exclusive");
        if (!validate_only && token_list.empty()) throw std::runtime_error("--tokens is required");
        std::vector<uint32_t> prompt = token_list.empty() ? std::vector<uint32_t>{} : parse_tokens(token_list);

        auto start = std::chrono::steady_clock::now();
        // Validate the small tokenizer artifact before allocating any model
        // state, then reject cross-artifact vocabulary mismatches explicitly.
        q27::Tokenizer tokenizer(tokenizer_path);
        if (tokenizer.vocab_size() != q27::MetalEngine::vocabulary_size())
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        q27::MetalEngine engine(model_path, context, turbo3_kv);
        auto loaded = std::chrono::steady_clock::now();
        fprintf(stderr, "Metal model ready on %s in %.2f s\n", engine.backend().name().c_str(),
                std::chrono::duration<double>(loaded - start).count());
        if (validate_only) { puts("artifacts and Metal architecture: OK"); return 0; }
        std::vector<uint32_t> generated = mtp_width ? engine.generate_mtp(prompt,count,mtp_width)
                                           : suffix_width ? engine.generate_suffix(prompt,count,suffix_width)
                                                          : engine.generate(prompt,count);
        auto finished = std::chrono::steady_clock::now();

        std::vector<int> ids(generated.begin(), generated.end());
        printf("generated:%s\n", tokenizer.decode(ids).c_str());
        fprintf(stderr, "%zu tokens in %.2f s (%.2f tok/s), position %u\n", generated.size(),
                std::chrono::duration<double>(finished - loaded).count(),
                generated.size() / std::chrono::duration<double>(finished - loaded).count(),
                engine.position());
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
