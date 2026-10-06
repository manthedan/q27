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
    if (text.empty() || text[0] == '-')
        throw std::runtime_error(std::string("invalid ") + option + ": " + text);
    size_t consumed = 0;
    unsigned long long value = 0;
    try { value = std::stoull(text, &consumed, 10); }
    catch (...) { throw std::runtime_error(std::string("invalid ") + option + ": " + text); }
    if (consumed != text.size() || value > UINT32_MAX)
        throw std::runtime_error(std::string("invalid ") + option + ": " + text);
    return static_cast<uint32_t>(value);
}

uint64_t parse_u64(const std::string& text, const char* option) {
    if (text.empty() || text[0] == '-')
        throw std::runtime_error(std::string("invalid ") + option + ": " + text);
    size_t consumed = 0;
    unsigned long long value = 0;
    try { value = std::stoull(text, &consumed, 10); }
    catch (...) { throw std::runtime_error(std::string("invalid ") + option + ": " + text); }
    if (consumed != text.size())
        throw std::runtime_error(std::string("invalid ") + option + ": " + text);
    return value;
}

float parse_float(const std::string& text, const char* option) {
    size_t consumed = 0;
    float value = 0;
    try { value = std::stof(text, &consumed); }
    catch (...) { throw std::runtime_error(std::string("invalid ") + option + ": " + text); }
    if (consumed != text.size())
        throw std::runtime_error(std::string("invalid ") + option + ": " + text);
    return value;
}

std::vector<uint32_t> parse_tokens(const std::string& text) {
    std::vector<uint32_t> result;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) throw std::runtime_error("empty token id");
        result.push_back(parse_u32(item, "token id"));
    }
    if (result.empty()) throw std::runtime_error("--tokens requires at least one token id");
    return result;
}

void write_token_ids(const std::string& path, const std::vector<uint32_t>& tokens) {
    FILE* file = fopen(path.c_str(), "w");
    if (!file) throw std::runtime_error("cannot open --dump-token-ids output: " + path);
    bool ok = true;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (fprintf(file, "%s%u", i ? " " : "", tokens[i]) < 0) { ok = false; break; }
    }
    if (ok && fprintf(file, "\n") < 0) ok = false;
    if (fclose(file) != 0) ok = false;
    if (!ok) throw std::runtime_error("cannot write --dump-token-ids output: " + path);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s model.q27 tokenizer.tok [--validate-only | --tokens id,id,... | --prompt text] "
                "[-n count] [--ctx count] [--mtp width | --suffix width] [--kv fp16|turbo3|q8] [--prefill chunk|serial] "
                "[--temperature T --top-p P --top-k K --seed S] [--dump-token-ids file] [--counters]\n",
                argv[0]);
        return 1;
    }

    try {
        const std::string model_path = argv[1];
        const std::string tokenizer_path = argv[2];
        std::string token_list, prompt_text, dump_token_ids;
        uint32_t count = 1, context = 128, mtp_width = 0, suffix_width = 0;
        q27::SamplingParams sampling;
        bool validate_only = false, serial_prefill = false, counters = false;
        q27::KvKind kv_kind = q27::KvKind::F16;
        bool token_list_supplied = false, prompt_supplied = false;

        for (int i = 3; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg == "--tokens" && i + 1 < argc) {
                token_list = argv[++i];
                token_list_supplied = true;
            }
            else if (arg == "--prompt" && i + 1 < argc) {
                prompt_text = argv[++i];
                prompt_supplied = true;
            }
            else if (arg == "--validate-only") validate_only = true;
            else if (arg == "-n" && i + 1 < argc) count = parse_u32(argv[++i], "-n");
            else if (arg == "--ctx" && i + 1 < argc) context = parse_u32(argv[++i], "--ctx");
            else if (arg == "--mtp" && i + 1 < argc) mtp_width = parse_u32(argv[++i], "--mtp");
            else if (arg == "--suffix" && i + 1 < argc) suffix_width = parse_u32(argv[++i], "--suffix");
            else if (arg == "--kv" && i + 1 < argc) {
                const std::string mode = argv[++i];
                kv_kind = q27::parse_kv_kind(mode);
            }
            else if (arg == "--prefill" && i + 1 < argc) {
                const std::string mode = argv[++i];
                if (mode == "serial") serial_prefill = true;
                else if (mode != "chunk") throw std::runtime_error("--prefill must be chunk or serial");
            }
            else if (arg == "--temperature" && i + 1 < argc)
                sampling.temperature = parse_float(argv[++i], "--temperature");
            else if (arg == "--top-p" && i + 1 < argc)
                sampling.top_p = parse_float(argv[++i], "--top-p");
            else if (arg == "--top-k" && i + 1 < argc)
                sampling.top_k = parse_u32(argv[++i], "--top-k");
            else if (arg == "--seed" && i + 1 < argc)
                sampling.seed = parse_u64(argv[++i], "--seed");
            else if (arg == "--dump-token-ids" && i + 1 < argc) dump_token_ids = argv[++i];
            else if (arg == "--counters") counters = true;
            else throw std::runtime_error("unknown/incomplete argument: " + arg);
        }

        q27::validate_sampling(sampling);
        if (token_list_supplied && prompt_supplied)
            throw std::runtime_error("--tokens and --prompt are mutually exclusive");
        if (mtp_width && suffix_width)
            throw std::runtime_error("--mtp and --suffix are mutually exclusive");
        if (sampling.temperature > 0 && mtp_width)
            throw std::runtime_error("sampling cannot be combined with --mtp");
        if (validate_only) {
            if (token_list_supplied || prompt_supplied || mtp_width || suffix_width ||
                sampling.temperature > 0 || !dump_token_ids.empty())
                throw std::runtime_error("--validate-only cannot be combined with generation options");
        } else {
            if (!token_list_supplied && !prompt_supplied)
                throw std::runtime_error("--tokens or --prompt is required");
            if (!count) throw std::runtime_error("-n must be greater than zero");
        }

        const auto start = std::chrono::steady_clock::now();
        q27::Tokenizer tokenizer(tokenizer_path);
        if (tokenizer.vocab_size() != q27::MetalEngine::vocabulary_size())
            throw std::runtime_error("tokenizer/model vocabulary mismatch");

        std::vector<uint32_t> prompt;
        if (token_list_supplied) prompt = parse_tokens(token_list);
        else if (prompt_supplied) {
            for (int token : tokenizer.encode(prompt_text)) {
                if (token < 0) throw std::runtime_error("tokenizer returned a negative id");
                prompt.push_back(static_cast<uint32_t>(token));
            }
            if (prompt.empty()) throw std::runtime_error("--prompt encoded to no tokens");
        }
        for (uint32_t token : prompt)
            if (token >= q27::MetalEngine::vocabulary_size())
                throw std::runtime_error("prompt token id is outside the model vocabulary");

        q27::MetalEngine engine(model_path, context, kv_kind);
        if (serial_prefill) engine.set_chunked_prefill(false);
        const auto loaded = std::chrono::steady_clock::now();
        fprintf(stderr, "Metal model ready on %s in %.2f s (shader sha1 %s)\n",
                engine.backend().name().c_str(),
                std::chrono::duration<double>(loaded - start).count(),
                engine.backend().shader_source_sha1().c_str());
        if (validate_only) {
            puts("Artifacts and Metal architecture: OK");
            return 0;
        }

        const auto counters_before = engine.backend().dispatch_counters();
        std::vector<uint32_t> generated =
            suffix_width ? engine.generate_suffix(prompt, count, suffix_width,
                                                  q27::MetalEngine::SUFFIX_MIN_MATCH, UINT32_MAX,
                                                  &sampling)
            : sampling.temperature > 0 ? engine.generate_sampled(prompt, count, sampling)
            : mtp_width ? engine.generate_mtp(prompt, count, mtp_width)
                        : engine.generate(prompt, count);
        if (!dump_token_ids.empty()) write_token_ids(dump_token_ids, generated);
        if (counters) {
            // Generation only (load excluded); one machine-readable line for
            // tools/perf_ceilings.sh.
            const auto after = engine.backend().dispatch_counters();
            fprintf(stderr, "counters: prompt=%zu generated=%zu command_buffers=%llu dispatches=%llu gpu_ms=%.2f\n",
                    prompt.size(), generated.size(),
                    static_cast<unsigned long long>(after.command_buffers - counters_before.command_buffers),
                    static_cast<unsigned long long>(after.dispatches - counters_before.dispatches),
                    (after.gpu_seconds - counters_before.gpu_seconds) * 1e3);
        }

        const auto finished = std::chrono::steady_clock::now();
        std::vector<int> ids(generated.begin(), generated.end());
        printf("generated:%s\n", tokenizer.decode(ids).c_str());
        const double elapsed = std::chrono::duration<double>(finished - loaded).count();
        fprintf(stderr, "%zu tokens in %.2f s (%.2f tok/s), position %u\n",
                generated.size(), elapsed, generated.size() / elapsed, engine.position());
        if (suffix_width) {
            const auto spec = engine.last_spec_stats();
            const auto bursts = engine.last_suffix_stats();
            fprintf(stderr, "suffix: %llu burst rounds, %llu serial fallbacks, %llu drafts, %llu accepted\n",
                    static_cast<unsigned long long>(bursts.burst_rounds),
                    static_cast<unsigned long long>(bursts.fallback_rounds),
                    static_cast<unsigned long long>(spec.drafted),
                    static_cast<unsigned long long>(spec.accepted));
        }
        if (mtp_width) {
            const auto spec = engine.last_spec_stats();
            fprintf(stderr, "speculation: %llu rounds, %llu drafts, %llu accepted (%.1f%%)\n",
                    static_cast<unsigned long long>(spec.rounds),
                    static_cast<unsigned long long>(spec.drafted),
                    static_cast<unsigned long long>(spec.accepted),
                    spec.drafted ? 100.0 * spec.accepted / spec.drafted : 0.0);
        }
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
