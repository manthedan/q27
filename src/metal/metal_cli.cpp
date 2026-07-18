#include "metal_engine.h"
#include "../kl.h"
#include "../tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <fcntl.h>
#include <unistd.h>

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

// A5 logits-dump binding header (autoreview P1, 2026-07-18): the raw dump
// is otherwise self-describing only by byte count, and two corpora at the
// same --nll-long produce identically sized dumps — pairing a candidate
// corpus with the WRONG dump was silently accepted, yielding plausible but
// meaningless KL. The versioned header binds each dump to its token stream
// (count + FNV-1a hash) and vocabulary; --kl-vs-dump validates all three
// before replaying. Layout is fixed-size, little-endian, single write.
struct LogitsDumpHeader {
    char magic[8];            // "Q27LDMP1"
    uint32_t version;         // 1
    uint32_t vocab;
    uint64_t n_positions;     // teacher-forced rows that follow
    uint64_t token_hash;      // FNV-1a over the n_positions+1 token ids
};
constexpr char LDMP_MAGIC[8] = {'Q','2','7','L','D','M','P','1'};

uint64_t fnv1a_tokens(const uint32_t* tokens, size_t count) {
    uint64_t h = 1469598103934665603ull;
    for (size_t k = 0; k < count; k++)
        for (int b = 0; b < 4; b++) { h ^= (tokens[k] >> (b*8)) & 0xff; h *= 1099511628211ull; }
    return h;
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

std::vector<uint32_t> load_token_file(const std::string& path) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("cannot open " + path);
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); throw std::runtime_error("cannot seek " + path); }
    long bytes = ftell(file);
    if (bytes < 0 || bytes % 4 != 0) { fclose(file); throw std::runtime_error("invalid token file size: " + path); }
    if (fseek(file, 0, SEEK_SET) != 0) { fclose(file); throw std::runtime_error("cannot seek " + path); }
    std::vector<uint32_t> tokens((size_t)bytes / 4);
    if (fread(tokens.data(), 4, tokens.size(), file) != tokens.size()) {
        fclose(file);
        throw std::runtime_error("short read on " + path);
    }
    fclose(file);
    return tokens;
}

// Position buckets match the CUDA --nll-long gate so Metal and CUDA
// long-context NLL reports are directly comparable.
void print_nll_long_buckets(const std::vector<float>& nll) {
    static const int edges[] = {
        0, 2048, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 163840,
        196608, 229376, 262144, 327680, 1 << 30
    };
    static const char* names[] = {
        "0-2k", "2k-8k", "8k-16k", "16k-32k", "32k-48k", "48k-64k", "64k-96k",
        "96k-128k", "128k-160k", "160k-192k", "192k-224k", "224k-256k",
        "256k-320k", "320k+"
    };
    constexpr int NB = 14;
    double sum[NB] = {};
    long count[NB] = {};
    for (size_t i = 0; i < nll.size(); i++) {
        // Target position is i+1 (logit after encoding token i predicts token i+1).
        const int tpos = (int)i + 1;
        int b = 0;
        while (tpos >= edges[b + 1]) b++;
        sum[b] += nll[i];
        count[b]++;
    }
    printf("long-context NLL by target position (%zu tokens, no resets):\n", nll.size() + 1);
    for (int b = 0; b < NB; b++) {
        if (!count[b]) continue;
        const double mean = sum[b] / count[b];
        printf("  %-8s: mean NLL %.4f  PPL %8.3f  (n=%ld)\n", names[b], mean, std::exp(mean), count[b]);
    }
}

// Same position buckets as the NLL report; values are per-position forward
// KL in nats against the fp16-KV baseline (whitepaper §4.4 methodology).
// `label` distinguishes same-model KV arms from cross-model A/B pairs (A5).
void print_kl_buckets(const std::vector<double>& kl, const char* label = "KV forward-KL vs fp16 baseline") {
    static const int edges[] = {
        0, 2048, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 163840,
        196608, 229376, 262144, 327680, 1 << 30
    };
    static const char* names[] = {
        "0-2k", "2k-8k", "8k-16k", "16k-32k", "32k-48k", "48k-64k", "64k-96k",
        "96k-128k", "128k-160k", "160k-192k", "192k-224k", "224k-256k",
        "256k-320k", "320k+"
    };
    constexpr int NB = 14;
    double sum[NB] = {}, peak[NB] = {};
    long count[NB] = {};
    for (size_t i = 0; i < kl.size(); i++) {
        const int tpos = (int)i + 1;
        int b = 0;
        while (tpos >= edges[b + 1]) b++;
        sum[b] += kl[i];
        if (kl[i] > peak[b]) peak[b] = kl[i];
        count[b]++;
    }
    printf("%s by target position (%zu tokens, no resets):\n",
           label, kl.size() + 1);
    for (int b = 0; b < NB; b++) {
        if (!count[b]) continue;
        printf("  %-8s: mean KL %.6g nats  max %.6g  (n=%ld)\n",
               names[b], sum[b] / count[b], peak[b], count[b]);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s model.q27 tokenizer.tok [--validate-only | --tokens id,id,... | --prompt text | --nll file] "
                "[-n count] [--ctx count] [--mtp width | --suffix width | --suffix-serial width | --oracle width] [--kv fp16|turbo3] "
                "[--prefill chunk|serial] [--nll-long N] [--kl-kv | --kl-kv-self | --kl-kv-k | --kl-kv-v | --kl-kv-fp8 | --kl-kv-cell N | --kl-kv-except LIST | --kl-kv-stats FILE] [--kv-rt-scale32] [--kv-rt-feature FILE] [--chunk-parity N] "
                "[--kl-pair MODEL_B --kl-pair-out FILE | --logits-dump FILE | --kl-vs-dump FILE] "
                "[--temperature T --top-p P --top-k K --seed S] "
                "[--save-state file | --load-state file] [--dump-logits file]\n",
                argv[0]);
        return 1;
    }
    try {
        std::string model_path=argv[1],tokenizer_path=argv[2],token_list,prompt_text,dump_logits,nll_path;
        std::string save_state_path, load_state_path;
        uint32_t count=1,context=128,mtp_width=0,suffix_width=0,oracle_width=0,nll_long=0; bool suffix_serial=false; q27::SamplingParams sampling;
        bool eos_gate=false;
        bool turbo3_kv = false, validate_only = false, serial_prefill = false;
        bool kl_kv = false, kl_self = false;
        uint32_t kv_attrib = 0, kv_cell = UINT32_MAX;
        std::vector<uint32_t> kv_except;
        bool kv_except_set = false;
        bool kv_rt_scale32 = false;
        std::string kv_rt_feature, kv_stats_out;
        uint32_t chunk_parity = 0;
        std::string envelope_mode;
        // A5 cross-model paired-logit KL (2026-07-18-kl-pair-a5.md): argv[1]
        // is model A (the "p" side, forward-KL convention), --kl-pair's
        // argument is model B (the "q" side). Two independent mappings,
        // fp16-KV both, lockstep teacher forcing.
        std::string kl_pair_path, kl_pair_out;
        // A5 sequential two-pass amendment (2026-07-18-kl-pair-a5.md):
        // --logits-dump streams every teacher-forced position's f32 logits
        // to FILE (row i = position i, VOCAB f32 each, no header);
        // --kl-vs-dump replays a baseline dump against this process's model.
        std::string logits_dump, kl_vs_dump;
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--tokens" && i + 1 < argc) token_list = argv[++i];
            else if (arg == "--prompt" && i + 1 < argc) prompt_text = argv[++i];
            else if (arg == "--nll" && i + 1 < argc) nll_path = argv[++i];
            else if (arg == "--nll-long" && i + 1 < argc) nll_long = parse_u32(argv[++i], "--nll-long");
            else if (arg == "--validate-only") validate_only = true;
            else if (arg == "--kl-kv") kl_kv = true;
            else if (arg == "--kl-kv-self") { kl_kv = true; kl_self = true; }
            else if (arg == "--kl-kv-stats" && i + 1 < argc) { kl_kv = true; kv_stats_out = argv[++i]; }
            else if (arg == "--kv-rt-scale32") kv_rt_scale32 = true;
            else if (arg == "--kv-rt-feature" && i + 1 < argc) kv_rt_feature = argv[++i];
            else if (arg == "--kl-kv-cell" && i + 1 < argc) {
                kv_cell = parse_u32(argv[++i], "--kl-kv-cell");
                if (kv_cell >= 128)
                    throw std::runtime_error("--kl-kv-cell must be 0..127 (attn_idx*8 + head*2 + side, side 0=K 1=V)");
                kl_kv = true;
            }
            else if (arg == "--kl-kv-except" && i + 1 < argc) {
                // Step-4 probe: quantize both sides everywhere EXCEPT the
                // listed census cells. "-" = empty list (control arm).
                const std::string list = argv[++i];
                if (list != "-") {
                    size_t at = 0;
                    while (at < list.size()) {
                        size_t comma = list.find(',', at);
                        if (comma == std::string::npos) comma = list.size();
                        const uint32_t cell =
                            parse_u32(list.substr(at, comma - at).c_str(), "--kl-kv-except");
                        if (cell >= 128)
                            throw std::runtime_error("--kl-kv-except cells must be 0..127");
                        kv_except.push_back(cell);
                        at = comma + 1;
                    }
                }
                kl_kv = true; kv_except_set = true;
            }
            else if (arg == "--kl-kv-k" || arg == "--kl-kv-v" || arg == "--kl-kv-fp8") {
                // fp8-KV control arm (2026-07-17-fp8-kv-control.md): mode 4,
                // e4m3 round-trip of BOTH sides — production-exact for a
                // transform-free codec, unlike the turbo3 side arms.
                const uint32_t side = (arg == "--kl-kv-k") ? 1 : (arg == "--kl-kv-v") ? 2 : 4;
                if (kv_attrib && kv_attrib != side)
                    throw std::runtime_error("--kl-kv-k/--kl-kv-v/--kl-kv-fp8 are alternative arms; pass one");
                kl_kv = true; kv_attrib = side;
            }
            else if (arg == "--chunk-parity" && i + 1 < argc) chunk_parity = parse_u32(argv[++i], "--chunk-parity");
            else if (arg == "--envelope" && i + 1 < argc) envelope_mode = argv[++i];
            else if (arg == "--kl-pair" && i + 1 < argc) kl_pair_path = argv[++i];
            else if (arg == "--kl-pair-out" && i + 1 < argc) kl_pair_out = argv[++i];
            else if (arg == "--logits-dump" && i + 1 < argc) logits_dump = argv[++i];
            else if (arg == "--kl-vs-dump" && i + 1 < argc) kl_vs_dump = argv[++i];
            else if (arg == "-n" && i + 1 < argc) count = parse_u32(argv[++i], "-n");
            else if (arg == "--ctx" && i + 1 < argc) context = parse_u32(argv[++i], "--ctx");
            else if (arg == "--mtp" && i + 1 < argc) mtp_width = parse_u32(argv[++i], "--mtp");
            else if ((arg == "--suffix" || arg == "--suffix-serial") && i + 1 < argc) {
                if (suffix_width) throw std::runtime_error("--suffix and --suffix-serial are mutually exclusive");
                suffix_width = parse_u32(argv[++i], arg.c_str());
                suffix_serial = arg == "--suffix-serial";
            }
            else if (arg == "--oracle" && i + 1 < argc) oracle_width = parse_u32(argv[++i], "--oracle");
            else if (arg == "--eos-gate") eos_gate = true;
            else if (arg == "--save-state" && i + 1 < argc) save_state_path = argv[++i];
            else if (arg == "--load-state" && i + 1 < argc) load_state_path = argv[++i];
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
            else if (arg == "--prefill" && i + 1 < argc) {
                std::string mode = argv[++i];
                if (mode == "serial") serial_prefill = true;
                else if (mode != "chunk") throw std::runtime_error("--prefill must be chunk or serial");
            }
            else throw std::runtime_error("unknown/incomplete argument: " + arg);
        }
        if ((mtp_width != 0) + (suffix_width != 0) + (oracle_width != 0) > 1)
            throw std::runtime_error("--mtp, --suffix, and --oracle are mutually exclusive");
        q27::validate_sampling(sampling);
        if(sampling.temperature>0 && (mtp_width || suffix_width || oracle_width))
            throw std::runtime_error("sampling cannot be combined with speculative modes");
        if (oracle_width && (oracle_width < 2 || oracle_width > 48))
            throw std::runtime_error("--oracle width must be 2..48 (VERIFY_CHUNK_MAX)");
        if (oracle_width && count < oracle_width + 2)
            throw std::runtime_error("--oracle needs -n >= width+2 for at least one full and one final round");
        if (oracle_width && !dump_logits.empty())
            throw std::runtime_error("--oracle cannot be combined with --dump-logits");
        if (eos_gate && !mtp_width)
            throw std::runtime_error("--eos-gate requires --mtp W (it tests the batched-MTP EOS path)");
        if (eos_gate && (oracle_width || suffix_width || sampling.temperature > 0 || !dump_logits.empty()))
            throw std::runtime_error("--eos-gate cannot be combined with other modes");
        // --prefill serial falls back to serial decode and --validate-only
        // returns before the gate — either would exit 0 without testing the
        // batched-MTP EOS path (vacuous-gate class, codex P2).
        if (eos_gate && (serial_prefill || validate_only))
            throw std::runtime_error("--eos-gate requires the chunked batched-MTP path; drop --prefill serial/--validate-only");
        if (!envelope_mode.empty()) {
            if (envelope_mode != "half" && envelope_mode != "blocked" &&
                envelope_mode != "serial" && envelope_mode != "repeat")
                throw std::runtime_error("--envelope must be half|blocked|serial|repeat");
            if (nll_path.empty())
                throw std::runtime_error("--envelope rides the --nll FILE input path");
            if (kl_kv || chunk_parity || turbo3_kv || serial_prefill || !dump_logits.empty())
                throw std::runtime_error("--envelope is its own instrument; drop --kl-kv/--chunk-parity/--kv turbo3/--prefill serial/--dump-logits");
        }
        // Prefix snapshots, Phase 1 (docs/plans/2026-07-16-prefix-snapshots.md):
        // greedy serial generation only — every excluded mode either mutates
        // state the snapshot doesn't describe (speculative) or changes the
        // continuation (sampling), which would make the byte-identity gate
        // vacuous or flaky.
        if (!save_state_path.empty() && !load_state_path.empty())
            throw std::runtime_error("--save-state and --load-state are separate runs; pass one");
        if ((!save_state_path.empty() || !load_state_path.empty()) &&
            (mtp_width || suffix_width || oracle_width || eos_gate || sampling.temperature > 0 ||
             !nll_path.empty() || !envelope_mode.empty() || kl_kv || chunk_parity))
            throw std::runtime_error("--save-state/--load-state are Phase-1 greedy-only; drop speculative/sampling/instrument modes");
        if (!save_state_path.empty() && token_list.empty() && prompt_text.empty())
            throw std::runtime_error("--save-state needs --prompt or --tokens (state is saved after ingestion)");
        if (!load_state_path.empty() && (!token_list.empty() || !prompt_text.empty()))
            throw std::runtime_error("--load-state restores a prefix; drop --prompt/--tokens");
        if(!token_list.empty() && !prompt_text.empty()) throw std::runtime_error("--tokens and --prompt are mutually exclusive");
        if (!nll_path.empty() && (!token_list.empty() || !prompt_text.empty() || validate_only))
            throw std::runtime_error("--nll cannot be combined with --tokens/--prompt/--validate-only");
        if (!nll_path.empty() && !nll_long && !chunk_parity && envelope_mode.empty())
            throw std::runtime_error("--nll currently requires --nll-long N (chunked llama-ppl mode is CUDA-only)");
        if (nll_long && nll_path.empty())
            throw std::runtime_error("--nll-long requires --nll FILE");
        if (kl_kv && nll_path.empty())
            throw std::runtime_error("--kl-kv rides the --nll FILE --nll-long N input path");
        if (chunk_parity && nll_path.empty())
            throw std::runtime_error("--chunk-parity rides the --nll FILE input path");
        if (chunk_parity && nll_long)
            throw std::runtime_error("--chunk-parity takes its position count from its own argument; drop --nll-long");
        if (chunk_parity && kl_kv)
            throw std::runtime_error("--chunk-parity and --kl-kv are separate instruments");
        if (kl_kv && turbo3_kv)
            throw std::runtime_error("--kl-kv builds its own fp16 baseline and subject; drop --kv");
        if (kv_attrib && kl_self)
            throw std::runtime_error("--kl-kv-k/--kl-kv-v/--kl-kv-fp8 and --kl-kv-self are mutually exclusive arms");
        if (kv_cell != UINT32_MAX && (kv_attrib || kl_self))
            throw std::runtime_error("--kl-kv-cell is its own arm; drop --kl-kv-k/--kl-kv-v/--kl-kv-fp8/--kl-kv-self");
        if (!kv_stats_out.empty() && (kv_attrib || kl_self || kv_cell != UINT32_MAX ||
                                      kv_rt_scale32 || !kv_rt_feature.empty()))
            throw std::runtime_error("--kl-kv-stats is its own pass; drop other kl-kv arms/modifiers");
        if ((kv_rt_scale32 || !kv_rt_feature.empty()) && (!kv_attrib || kv_attrib == 4))
            throw std::runtime_error("--kv-rt-scale32/--kv-rt-feature modify a turbo3 side arm; add --kl-kv-k or --kl-kv-v (not --kl-kv-fp8)");
        if (kv_except_set && (kv_attrib || kl_self || kv_cell != UINT32_MAX ||
                              !kv_stats_out.empty() || kv_rt_scale32 || !kv_rt_feature.empty()))
            throw std::runtime_error("--kl-kv-except is its own arm; drop other kl-kv arms/modifiers");
        // A5 cross-model paired-logit KL: its own instrument, fp16-KV on both
        // sides so the KV codec cannot confound the weight comparison.
        if (!kl_pair_path.empty()) {
            if (nll_path.empty())
                throw std::runtime_error("--kl-pair rides the --nll FILE --nll-long N input path");
            if (kl_kv || chunk_parity || !envelope_mode.empty() || turbo3_kv)
                throw std::runtime_error("--kl-pair is its own instrument; drop --kl-kv*/--chunk-parity/--envelope/--kv");
            if (kv_attrib || kl_self || kv_cell != UINT32_MAX || kv_except_set ||
                !kv_stats_out.empty() || kv_rt_scale32 || !kv_rt_feature.empty())
                throw std::runtime_error("--kl-pair cannot be combined with any --kl-kv* arm/modifier");
            if (mtp_width || suffix_width || oracle_width || eos_gate || sampling.temperature > 0 ||
                !dump_logits.empty() || !save_state_path.empty() || !load_state_path.empty() ||
                validate_only || !token_list.empty() || !prompt_text.empty())
                throw std::runtime_error("--kl-pair is greedy teacher-forced only; drop speculative/sampling/state/prompt/validate modes");
            if (kl_pair_path == model_path)
                throw std::runtime_error("--kl-pair needs two distinct models; for same-model plumbing use --kl-kv-self");
        }
        if (!kl_pair_out.empty() && kl_pair_path.empty() && kl_vs_dump.empty())
            throw std::runtime_error("--kl-pair-out requires --kl-pair or --kl-vs-dump");
        // A5 sequential two-pass arms (2026-07-18 amendment): one model per
        // process, fp16-KV, greedy teacher-forced on the --nll path.
        if (!logits_dump.empty() || !kl_vs_dump.empty()) {
            const char* arm = !logits_dump.empty() ? "--logits-dump" : "--kl-vs-dump";
            if (!logits_dump.empty() && !kl_vs_dump.empty())
                throw std::runtime_error("--logits-dump and --kl-vs-dump are separate passes; run one");
            if (nll_path.empty())
                throw std::runtime_error(std::string(arm) + " rides the --nll FILE --nll-long N input path");
            if (kl_kv || chunk_parity || !envelope_mode.empty() || turbo3_kv || !kl_pair_path.empty())
                throw std::runtime_error(std::string(arm) + " is its own instrument; drop --kl-kv*/--chunk-parity/--envelope/--kv/--kl-pair");
            if (mtp_width || suffix_width || oracle_width || eos_gate || sampling.temperature > 0 ||
                !dump_logits.empty() || !save_state_path.empty() || !load_state_path.empty() ||
                validate_only || !token_list.empty() || !prompt_text.empty())
                throw std::runtime_error(std::string(arm) + " is greedy teacher-forced only; drop speculative/sampling/state/prompt/validate modes");
        }
        if (nll_path.empty() && !validate_only && token_list.empty() && prompt_text.empty() &&
            load_state_path.empty() && kl_pair_path.empty() && logits_dump.empty() && kl_vs_dump.empty())
            throw std::runtime_error("--tokens, --prompt, --nll, --load-state, --kl-pair, --logits-dump, --kl-vs-dump, or --validate-only is required");
        if (!nll_path.empty() && (mtp_width || suffix_width || oracle_width || sampling.temperature > 0 || !dump_logits.empty()))
            throw std::runtime_error("--nll cannot be combined with speculative/sampling/dump modes");

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
        // Round-2 expert P0 #3 gate: distribution-level coverage of the WIDE
        // prompt-ingestion path. A width-12 baseline engine and a wide
        // subject engine (same KV type, one mapping) teacher-force the same
        // stream; per position we report logit deltas, forward KL, top-1
        // agreement with the baseline margin at every flip (L3 of the
        // margin-aware contract), top-20 overlap, and per-arm NLL.
        if (chunk_parity) {
            std::vector<uint32_t> tokens = load_token_file(nll_path);
            if (tokens.size() < 2) throw std::runtime_error("--chunk-parity needs at least two tokens");
            const uint32_t n = std::min<uint32_t>(chunk_parity, (uint32_t)tokens.size() - 1);
            if (n > context) throw std::runtime_error("--chunk-parity N exceeds --ctx; raise --ctx");
            // Every advertised width must dispatch at least one full-width
            // chunk, or the report claims coverage that never ran.
            if (n < 96)
                throw std::runtime_error("--chunk-parity needs >= 96 positions to exercise widths {17,48,96}; "
                                         "token file/N gives " + std::to_string(n));
            // Targets tokens[1..n] index logits directly in nll_of below;
            // the engines validate only the encoded tokens, so reject
            // out-of-vocabulary ids up front.
            const uint32_t vocab = q27::MetalEngine::vocabulary_size();
            for (uint32_t i = 0; i <= n; i++)
                if (tokens[i] >= vocab)
                    throw std::runtime_error("token id " + std::to_string(tokens[i]) +
                                             " out of vocabulary at position " + std::to_string(i));
            auto shared = q27::MetalEngine::open_shared(model_path);
            q27::MetalEngine baseline(shared, context, turbo3_kv);
            q27::MetalEngine subject(shared, context, turbo3_kv);
            // --prefill serial: negative control. A serial-encode baseline
            // vs the wide chunk path is the known chunk-vs-serial rounding
            // class, so the instrument MUST report nonzero here — proof the
            // gate can fail (vacuous-gate lesson, 2026-07-15).
            if (serial_prefill) baseline.set_chunked_prefill(false);
            auto ready = std::chrono::steady_clock::now();
            fprintf(stderr, "Metal model ready on %s in %.2f s (two engines, one mapping, %s KV%s)\n",
                    baseline.backend().name().c_str(),
                    std::chrono::duration<double>(ready - start).count(),
                    turbo3_kv ? "turbo3" : "fp16",
                    serial_prefill ? ", SERIAL baseline = negative control" : "");
            fprintf(stderr, "chunk-parity: %u positions, widths {17,48,96} vs width-12 baseline\n", n);

            // Canonical width-12 logits, one pass, kept in host memory
            // (n x vocab floats; 384 positions = ~381 MB).
            std::vector<float> ref((size_t)n * vocab);
            {
                std::vector<float> p;
                uint32_t done = 0;
                while (done < n) {
                    const uint32_t take = std::min(12u, n - done);
                    baseline.teacher_force_logits(tokens.data() + done, take, p);
                    std::memcpy(ref.data() + (size_t)done * vocab, p.data(),
                                (size_t)take * vocab * sizeof(float));
                    done += take;
                }
            }
            // Hidden-row leg (k3 audit A1/E3): both teacher-force paths must
            // leave x1_ — the row snapshots persist — describing the LAST
            // encoded token. Compared like the logits rows: bytewise against
            // the baseline pass, expectation inverted by the serial control.
            std::vector<float> ref_hidden;
            baseline.read_hidden(ref_hidden);
            auto nll_of = [&](const float* logits, uint32_t target) -> double {
                double mx = -1e300;
                for (uint32_t v = 0; v < vocab; v++) mx = std::max(mx, (double)logits[v]);
                double se = 0.0;
                for (uint32_t v = 0; v < vocab; v++) se += std::exp((double)logits[v] - mx);
                return std::log(se) + mx - (double)logits[target];
            };
            auto top2 = [&](const float* logits, uint32_t& t1, double& margin) {
                uint32_t best = 0; float b1 = logits[0], b2 = -1e30f;
                for (uint32_t v = 1; v < vocab; v++) {
                    if (logits[v] > b1) { b2 = b1; b1 = logits[v]; best = v; }
                    else if (logits[v] > b2) b2 = logits[v];
                }
                t1 = best; margin = (double)b1 - (double)b2;
            };
            auto topk_set = [&](const float* logits, uint32_t k) {
                std::vector<uint32_t> idx(vocab);
                for (uint32_t v = 0; v < vocab; v++) idx[v] = v;
                std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                                  [&](uint32_t a, uint32_t b) {
                                      return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
                                  });
                idx.resize(k);
                std::sort(idx.begin(), idx.end());
                return idx;
            };

            // The exit code carries the verdict so scripts can gate on it.
            // The wide path's contract is bit-identity with the width-12
            // baseline; if that is ever intentionally relaxed, this gate is
            // re-priced under the margin-aware contract, not silently
            // loosened. The serial negative control inverts the expectation:
            // the instrument itself fails if the known chunk-vs-serial
            // rounding class does not appear at every width.
            bool gate_fail = false;
            for (uint32_t width : {17u, 48u, 96u}) {
                subject.reset();
                std::vector<float> q;
                double max_abs = 0, sum_abs = 0, max_kl = 0, sum_kl = 0;
                double ref_nll = 0, subj_nll = 0, max_nll_delta = 0;
                double min_flip_margin = 1e300, max_flip_margin = 0;
                uint64_t flips = 0, overlap = 0, rows_differ = 0;
                uint32_t done = 0;
                while (done < n) {
                    const uint32_t take = std::min(width, n - done);
                    subject.teacher_force_logits_wide(tokens.data() + done, take, q);
                    for (uint32_t r = 0; r < take; r++) {
                        const float* pr = ref.data() + (size_t)(done + r) * vocab;
                        const float* qr = q.data() + (size_t)r * vocab;
                        // Bytewise row comparison is the identity ground
                        // truth: a NaN logit makes |pr-qr| NaN, and
                        // std::max(0, NaN) keeps max_abs at zero, so the
                        // metrics alone could certify a divergent row.
                        if (std::memcmp(pr, qr, (size_t)vocab * sizeof(float)) != 0) rows_differ++;
                        double mx = 0;
                        for (uint32_t v = 0; v < vocab; v++) {
                            if (!std::isfinite(pr[v]) || !std::isfinite(qr[v]))
                                throw std::runtime_error("non-finite logit at position " +
                                                         std::to_string(done + r) + " (width " +
                                                         std::to_string(width) + ")");
                            mx = std::max(mx, std::abs((double)pr[v] - (double)qr[v]));
                        }
                        max_abs = std::max(max_abs, mx); sum_abs += mx;
                        const double kl = q27::forward_kl(pr, qr, vocab);
                        max_kl = std::max(max_kl, kl); sum_kl += kl;
                        const uint32_t target = tokens[done + r + 1];
                        const double np = nll_of(pr, target), nq = nll_of(qr, target);
                        ref_nll += np; subj_nll += nq;
                        max_nll_delta = std::max(max_nll_delta, std::abs(nq - np));
                        uint32_t tp, tq; double mp, mq;
                        top2(pr, tp, mp); top2(qr, tq, mq);
                        if (tp != tq) {
                            flips++;
                            min_flip_margin = std::min(min_flip_margin, mp);
                            max_flip_margin = std::max(max_flip_margin, mp);
                        }
                        const auto sp = topk_set(pr, 20), sq = topk_set(qr, 20);
                        std::vector<uint32_t> inter;
                        std::set_intersection(sp.begin(), sp.end(), sq.begin(), sq.end(),
                                              std::back_inserter(inter));
                        overlap += inter.size();
                    }
                    done += take;
                }
                std::vector<float> hidden;
                subject.read_hidden(hidden);
                const bool hidden_differs =
                    std::memcmp(ref_hidden.data(), hidden.data(),
                                hidden.size() * sizeof(float)) != 0;
                // An all-zero readback is a never-written buffer, not a
                // passing comparison (vacuous-gate lesson).
                bool hidden_live = false;
                for (float v : hidden) if (v != 0.0f) { hidden_live = true; break; }
                printf("width %2u vs 12: rows differing %llu/%u, max|dlogit| %.6g (mean %.6g), "
                       "KL mean %.3g max %.3g nats, "
                       "top-1 flips %llu/%u (ref margin min %.4g max %.4g), top-20 overlap %.2f%%, "
                       "NLL %.6f vs %.6f (max |dNLL| %.4g), hidden row %s\n",
                       width, (unsigned long long)rows_differ, n, max_abs, sum_abs / n,
                       sum_kl / n, max_kl,
                       (unsigned long long)flips, n,
                       flips ? min_flip_margin : 0.0, flips ? max_flip_margin : 0.0,
                       100.0 * overlap / ((double)n * 20.0),
                       subj_nll / n, ref_nll / n, max_nll_delta,
                       !hidden_live ? "ALL-ZERO" : hidden_differs ? "DIFFERS" : "identical");
                const bool differs = rows_differ != 0;
                // Serial control gates on logits only: the whole-stream
                // rounding class must appear somewhere, but single-row
                // hidden bit-inequality is not a guaranteed member of it.
                if (serial_prefill ? !differs : (differs || hidden_differs || !hidden_live))
                    gate_fail = true;
            }
            if (gate_fail) {
                fprintf(stderr, serial_prefill
                        ? "chunk-parity: FAIL — negative control did not fire at every width\n"
                        : "chunk-parity: FAIL — wide path (logits or hidden row) diverged from the width-12 baseline\n");
                return 1;
            }
            fprintf(stderr, serial_prefill
                    ? "chunk-parity: negative control fired at every width (instrument healthy)\n"
                    : "chunk-parity: PASS — widths {17,48,96} bit-identical to the width-12 baseline (logits + hidden row)\n");
            return 0;
        }

        if (!envelope_mode.empty()) {
            // Envelope instrument (docs/plans/2026-07-16-envelope-instrument.md):
            // two engines, one mapping, teacher-forced in lockstep; the mode
            // picks the same-model reduction-order pair. Metrics per Q6:
            // whole-vocab max|d|/RMS as diagnostics, KL, top-1 flips with
            // baseline margin, and the margin certificate rho over the union
            // of both top-64 sets (flip certified impossible when rho < 1, so
            // an observed flip below 1 is a self-contradiction alarm).
            std::vector<uint32_t> tokens = load_token_file(nll_path);
            const uint32_t n_want = nll_long ? nll_long : 2048;
            if (tokens.size() > (size_t)n_want + 1) tokens.resize((size_t)n_want + 1);
            if (tokens.size() < 13) throw std::runtime_error("--envelope needs at least 13 tokens");
            if (tokens.size() - 1 > context)
                throw std::runtime_error("--envelope sequence exceeds --ctx; raise --ctx");
            auto shared = q27::MetalEngine::open_shared(model_path);
            q27::MetalEngine base(shared, context, false);
            q27::MetalEngine subj(shared, context, false);
            q27::MetalBackend& bk = base.backend();
            // Without chunked prefill the half and serial pairs collapse to
            // identical configs — a vacuous all-zero "envelope" (codex P2).
            // repeat (determinism) and blocked (the GQA threshold routes the
            // serial attention path too) stay meaningful there.
            if ((envelope_mode == "half" || envelope_mode == "serial") && !base.chunked_prefill())
                throw std::runtime_error("--envelope " + envelope_mode +
                                         " needs the chunked path (Apple GPU family 7+); 'repeat' and 'blocked' run on this device");
            const bool mode_half = envelope_mode == "half";
            const bool mode_blocked = envelope_mode == "blocked";
            const bool mode_serial = envelope_mode == "serial";
            fprintf(stderr, "Metal model ready on %s (envelope mode %s, two engines, one mapping)\n",
                    bk.name().c_str(), envelope_mode.c_str());
            const uint32_t vocab = q27::MetalEngine::vocabulary_size();
            const uint32_t n = (uint32_t)tokens.size() - 1;
            std::vector<float> p, q;
            std::vector<double> m_maxd(n), m_rms(n), m_kl(n), m_rho(n);
            struct Flip { uint32_t pos; double margin, rho; };
            std::vector<Flip> flips;
            uint32_t nan_alarms = 0, contradictions = 0;
            // Top-64 index extraction (descending by logit).
            auto top64 = [&](const float* z, std::vector<uint32_t>& out) {
                out.resize(vocab);
                for (uint32_t v = 0; v < vocab; v++) out[v] = v;
                std::partial_sort(out.begin(), out.begin() + 64, out.end(),
                                  [&](uint32_t x, uint32_t y) { return z[x] > z[y]; });
                out.resize(64);
            };
            std::vector<uint32_t> ta, tb;
            uint32_t done = 0, chunk_index = 0;
            while (done < n) {
                const uint32_t take = std::min(12u, n - done);
                if (mode_half) bk.set_gemm_half(true);
                if (mode_blocked) bk.set_gqa_threshold(1);
                base.teacher_force_logits(tokens.data() + done, take, p);
                if (mode_half) bk.set_gemm_half(false);
                if (mode_blocked) bk.set_gqa_threshold(0);
                if (mode_serial) {
                    // Serial arm: one step per position, logits read per step.
                    q.resize((size_t)take * vocab);
                    for (uint32_t r = 0; r < take; r++) {
                        subj.step(tokens[done + r]);
                        std::vector<float> row = subj.read_logits();
                        std::copy(row.begin(), row.end(), q.begin() + (size_t)r * vocab);
                    }
                } else {
                    subj.teacher_force_logits(tokens.data() + done, take, q);
                }
                for (uint32_t r = 0; r < take; r++) {
                    const uint32_t pos = done + r;
                    const float* pr = p.data() + (size_t)r * vocab;
                    const float* qr = q.data() + (size_t)r * vocab;
                    double maxd = 0.0, sum2 = 0.0;
                    uint32_t a = 0, b = 0;
                    bool nan = false;
                    for (uint32_t v = 0; v < vocab; v++) {
                        if (!std::isfinite(pr[v]) || !std::isfinite(qr[v])) nan = true;
                        const double d = std::fabs((double)pr[v] - qr[v]);
                        if (d > maxd) maxd = d;
                        sum2 += d * d;
                        if (pr[v] > pr[a]) a = v;
                        if (qr[v] > qr[b]) b = v;
                    }
                    if (nan) nan_alarms++;
                    m_maxd[pos] = maxd;
                    m_rms[pos] = std::sqrt(sum2 / vocab);
                    m_kl[pos] = q27::forward_kl(pr, qr, vocab);
                    top64(pr, ta); top64(qr, tb);
                    double rho_max = 0.0;
                    auto consider = [&](uint32_t j) {
                        if (j == a) return;
                        const double gap = (double)pr[a] - pr[j];
                        const double err = std::fabs((double)pr[a] - qr[a]) +
                                           std::fabs((double)pr[j] - qr[j]);
                        const double rho = gap > 0.0 ? err / gap : 1e30;
                        if (rho > rho_max) rho_max = rho;
                    };
                    for (uint32_t j : ta) consider(j);
                    for (uint32_t j : tb) consider(j);
                    m_rho[pos] = rho_max;
                    if (a != b) {
                        // Baseline margin between its top-1 and the subject's winner.
                        flips.push_back({pos, (double)pr[a] - pr[b], rho_max});
                        if (rho_max < 1.0) contradictions++;
                    }
                }
                done += take;
                if (++chunk_index % 16 == 0) fprintf(stderr, "  envelope pos %u/%u\r", done, n);
            }
            fprintf(stderr, "\n");
            auto quantiles = [&](std::vector<double> v, const char* name) {
                std::sort(v.begin(), v.end());
                auto pick = [&](double f) { return v[std::min(v.size() - 1, (size_t)(f * v.size()))]; };
                fprintf(stderr, "envelope %-9s p50 %.4g  p90 %.4g  p99 %.4g  p99.5 %.4g  max %.4g\n",
                        name, pick(0.50), pick(0.90), pick(0.99), pick(0.995), v.back());
            };
            fprintf(stderr, "envelope[%s]: %u positions, %zu top-1 flips, %u NaN alarms, %u contradictions\n",
                    envelope_mode.c_str(), n, flips.size(), nan_alarms, contradictions);
            quantiles(m_maxd, "max|d|");
            quantiles(m_rms, "RMS");
            quantiles(m_kl, "KL");
            quantiles(m_rho, "rho");
            for (size_t i = 0; i < flips.size() && i < 16; i++)
                fprintf(stderr, "  flip @%u: baseline margin %.4g, rho %.3g\n",
                        flips[i].pos, flips[i].margin, flips[i].rho);
            if (flips.size() > 16) fprintf(stderr, "  ... %zu more flips\n", flips.size() - 16);
            // Declared hard alarms from the contract: severe FINITE
            // divergence must fail too, not just NaN/contradiction (codex P1).
            const double hard_maxd = 2.0, hard_kl = 0.05;
            const double run_maxd = *std::max_element(m_maxd.begin(), m_maxd.end());
            const double run_kl = *std::max_element(m_kl.begin(), m_kl.end());
            bool alarm = nan_alarms || contradictions;
            if (run_maxd > hard_maxd) {
                fprintf(stderr, "envelope: HARD ALARM max|d| %.4g > %.1f\n", run_maxd, hard_maxd);
                alarm = true;
            }
            if (run_kl > hard_kl) {
                fprintf(stderr, "envelope: HARD ALARM KL %.4g > %.2f\n", run_kl, hard_kl);
                alarm = true;
            }
            if (envelope_mode == "repeat") {
                double repeat_max = *std::max_element(m_maxd.begin(), m_maxd.end());
                if (repeat_max != 0.0) {
                    fprintf(stderr, "envelope: REPEAT NONZERO (max %.4g) — determinism broken\n", repeat_max);
                    alarm = true;
                } else {
                    fprintf(stderr, "envelope: repeat exactly zero at every position (determinism holds)\n");
                }
            }
            if (alarm) { fprintf(stderr, "envelope: ALARM — see above\n"); return 1; }
            fprintf(stderr, "envelope: instrument clean (quantiles above are the class constants)\n");
            return 0;
        }

        if (kl_kv) {
            std::vector<uint32_t> tokens = load_token_file(nll_path);
            if (nll_long > 0 && tokens.size() > nll_long) tokens.resize(nll_long);
            if (tokens.size() < 2) throw std::runtime_error("--kl-kv needs at least two tokens");
            // n-1 positions are encoded; the final token is only a target.
            if (tokens.size() - 1 > context)
                throw std::runtime_error("--kl-kv sequence exceeds --ctx; raise --ctx");
            // One mapping, one queue, one weight wrap; two engines that
            // differ only in KV representation, teacher-forced in lockstep.
            auto shared = q27::MetalEngine::open_shared(model_path);
            q27::MetalEngine baseline(shared, context, false);
            // Attribution arms keep the subject on the fp16 cache and
            // attention kernels; only the store round-trips one side through
            // the turbo3 quantizer, so the KL is that side's error alone.
            q27::MetalEngine subject(shared, context,
                                     !kl_self && !kv_attrib && kv_cell == UINT32_MAX &&
                                     kv_stats_out.empty() && !kv_except_set);
            char cell_name[48] = {0};
            std::vector<float> feature_scales;
            if (!kv_stats_out.empty()) subject.set_kv_attrib_stats();
            if (!kv_rt_feature.empty()) {
                // Stats file -> clamped per-feature RMS scales (step 2,
                // docs/plans/2026-07-16-kv-codec-step2.md): s >= 1e-3 x the
                // side's mean RMS so dead features cannot explode the arm.
                FILE* sf = fopen(kv_rt_feature.c_str(), "rb");
                if (!sf) throw std::runtime_error("cannot open stats file: " + kv_rt_feature);
                char magic[8] = {0}; uint64_t sn = 0;
                std::vector<float> sumsq(2ull * 16 * 4 * 256);
                const bool ok = fread(magic, 1, 8, sf) == 8 &&
                                memcmp(magic, "Q27KVS1\0", 8) == 0 &&
                                fread(&sn, 8, 1, sf) == 1 && sn > 0 &&
                                fread(sumsq.data(), 4, sumsq.size(), sf) == sumsq.size();
                fclose(sf);
                if (!ok) throw std::runtime_error("invalid stats file: " + kv_rt_feature);
                feature_scales.resize(sumsq.size());
                for (int side = 0; side < 2; side++) {
                    double mean_rms = 0.0;
                    const size_t base = (size_t)side * 16384;
                    for (size_t i = 0; i < 16384; i++)
                        mean_rms += std::sqrt(sumsq[base + i] / (double)sn);
                    mean_rms /= 16384.0;
                    const float floor_s = (float)(1e-3 * mean_rms);
                    for (size_t i = 0; i < 16384; i++)
                        feature_scales[base + i] =
                            std::max((float)std::sqrt(sumsq[base + i] / (double)sn), floor_s);
                }
            }
            if (kv_cell != UINT32_MAX) {
                // cell id = attn_idx*8 + head*2 + side (side 0=K, 1=V);
                // attn_idx 0..15 maps to absolute layer attn_idx*4+3.
                const uint32_t attn_idx = kv_cell >> 3, head = (kv_cell >> 1) & 3;
                const uint32_t side = (kv_cell & 1) + 1;
                subject.set_kv_attrib_cell(side, attn_idx * 4 + 3, head);
                snprintf(cell_name, sizeof cell_name, "turbo3 cell L%u:h%u:%s round-trip",
                         attn_idx * 4 + 3, head, side == 1 ? "K" : "V");
            } else if (kv_except_set) {
                subject.set_kv_attrib_except(kv_except.data(), kv_except.size());
            } else if (kv_attrib) {
                // Side arm first, modifiers second — set_kv_attrib clears
                // the round-trip flags (codex P1 on b1bed0e).
                subject.set_kv_attrib(kv_attrib);
                if (kv_rt_scale32 || !kv_rt_feature.empty())
                    subject.set_kv_attrib_rt(kv_rt_scale32,
                                             feature_scales.empty() ? nullptr
                                                                    : feature_scales.data());
            }
            if (serial_prefill) {
                baseline.set_chunked_prefill(false);
                subject.set_chunked_prefill(false);
            }
            std::string arm_name = kl_self ? "fp16 self-check"
                                  : !kv_stats_out.empty() ? "fp16 stats pass (KL must be 0)"
                                  : kv_cell != UINT32_MAX ? cell_name
                                  : kv_except_set ? ("turbo3 both-sides round-trip except " +
                                                     std::to_string(kv_except.size()) + " cells")
                                  : kv_attrib == 1 ? "turbo3 K-only round-trip"
                                  : kv_attrib == 2 ? "turbo3 V-only round-trip"
                                  : kv_attrib == 4 ? "e4m3 fp8 both-sides round-trip (production-exact)"
                                  : "turbo3";
            if (kv_rt_scale32) arm_name += " +scale32";
            if (!kv_rt_feature.empty()) arm_name += " +feature";
            const char* subject_name = arm_name.c_str();
            auto ready = std::chrono::steady_clock::now();
            fprintf(stderr, "Metal model ready on %s in %.2f s (two engines, one mapping: fp16 baseline vs %s)\n",
                    baseline.backend().name().c_str(),
                    std::chrono::duration<double>(ready - start).count(),
                    subject_name);
            const uint32_t vocab = q27::MetalEngine::vocabulary_size();
            const uint32_t n = (uint32_t)tokens.size() - 1;
            fprintf(stderr, "kl-kv: %u positions, single pass, no resets\n", n);
            std::vector<float> p, q;
            std::vector<double> kl(n);
            auto kl_start = std::chrono::steady_clock::now();
            uint32_t done = 0, chunk_index = 0;
            while (done < n) {
                const uint32_t take = std::min(12u, n - done);
                baseline.teacher_force_logits(tokens.data() + done, take, p);
                subject.teacher_force_logits(tokens.data() + done, take, q);
                for (uint32_t r = 0; r < take; r++)
                    kl[done + r] = q27::forward_kl(p.data() + (size_t)r * vocab,
                                                   q.data() + (size_t)r * vocab, vocab);
                done += take;
                if (++chunk_index % 32 == 0) fprintf(stderr, "  kl pos %u/%u\r", done, n);
                if (done / 2048 != (done - take) / 2048) {
                    double running = 0.0;
                    for (uint32_t i = 0; i < done; i++) running += kl[i];
                    fprintf(stderr, "  kl pos %u: running mean %.6g nats\n", done, running / done);
                }
            }
            auto kl_done = std::chrono::steady_clock::now();
            if (n >= 12) fprintf(stderr, "\n");
            print_kl_buckets(kl);
            double mean = 0.0, peak = 0.0;
            uint32_t peak_pos = 0;
            for (uint32_t i = 0; i < n; i++) {
                mean += kl[i];
                if (kl[i] > peak) { peak = kl[i]; peak_pos = i; }
            }
            mean /= n;
            // Tail report (KV-codec step 1): the mean is depth-flat but the
            // tail is heavy — p99/max and run structure are the graduation
            // metrics for the scaling and allocation arms, not the mean.
            std::vector<double> sorted(kl);
            std::sort(sorted.begin(), sorted.end());
            auto quantile = [&](double p) {
                return sorted[std::min((size_t)((double)n * p), (size_t)n - 1)];
            };
            fprintf(stderr, "kl-kv tail: p50 %.4g  p90 %.4g  p99 %.4g  p99.5 %.4g  max %.4g @pos %u\n",
                    quantile(0.50), quantile(0.90), quantile(0.99), quantile(0.995), peak, peak_pos);
            for (double thr : {0.1, 0.5}) {
                uint32_t above = 0, runs = 0, cur = 0, longest = 0, longest_at = 0;
                for (uint32_t i = 0; i < n; i++) {
                    if (kl[i] > thr) {
                        if (!cur) runs++;
                        cur++; above++;
                        if (cur > longest) { longest = cur; longest_at = i + 1 - cur; }
                    } else cur = 0;
                }
                fprintf(stderr, "kl-kv runs >%.1f: %u positions, %u runs, longest %u", thr, above, runs, longest);
                if (longest) fprintf(stderr, " @pos %u", longest_at);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "kl-kv wall: %.2f s (%.2f pos/s through both engines), overall mean KL %.6g nats, max %.6g\n",
                    std::chrono::duration<double>(kl_done - kl_start).count(),
                    n / std::chrono::duration<double>(kl_done - kl_start).count(),
                    mean, peak);
            if (!kv_stats_out.empty()) {
                // The stats subject stores clean fp16, so any nonzero KL
                // means the instrument itself is broken — hard stop.
                if (mean != 0.0 || peak != 0.0)
                    throw std::runtime_error("kl-kv stats pass: KL canary NONZERO — instrument broken");
                std::vector<float> sumsq;
                subject.read_kv_attrib_stats(sumsq);
                FILE* sf = fopen(kv_stats_out.c_str(), "wb");
                if (!sf) throw std::runtime_error("cannot write stats file: " + kv_stats_out);
                const uint64_t sn = n;
                bool ok = fwrite("Q27KVS1\0", 1, 8, sf) == 8 &&
                          fwrite(&sn, 8, 1, sf) == 1 &&
                          fwrite(sumsq.data(), 4, sumsq.size(), sf) == sumsq.size();
                if (fclose(sf) != 0) ok = false;
                if (!ok) throw std::runtime_error("cannot write stats file: " + kv_stats_out);
                fprintf(stderr, "kl-kv stats: wrote %s (%u positions, 2x16x4x256 features)\n",
                        kv_stats_out.c_str(), n);
            }
            return 0;
        }

        // A5 cross-model paired-logit KL (2026-07-18-kl-pair-a5.md): two
        // independent mappings, fp16-KV on both, lockstep teacher forcing.
        // Forward-KL convention: p = model A (argv[1]), q = model B
        // (--kl-pair's argument). Both engines advance in lockstep over the
        // same token stream; per-position q27::forward_kl runs on CPU in
        // double precision exactly as --kl-kv does.
        if (!kl_pair_path.empty()) {
            std::vector<uint32_t> tokens = load_token_file(nll_path);
            if (nll_long > 0 && tokens.size() > nll_long) tokens.resize(nll_long);
            if (tokens.size() < 2) throw std::runtime_error("--kl-pair needs at least two tokens");
            if (tokens.size() - 1 > context)
                throw std::runtime_error("--kl-pair sequence exceeds --ctx; raise --ctx");

            // Memory budget check: refuse to start if the two artifacts plus
            // a 1 GiB overhead allowance exceed physical memory — fail loudly,
            // not OOM. (A5 is M4-only; the mini cannot hold official 17 GiB.)
            const uint64_t bytes_a = std::filesystem::file_size(model_path);
            const uint64_t bytes_b = std::filesystem::file_size(kl_pair_path);
            const uint64_t overhead = 1ull << 30;
            const uint64_t needed = bytes_a + bytes_b + overhead;
            const uint64_t physical = [] {
                uint64_t v = 0; size_t s = sizeof(v);
                if (sysctlbyname("hw.memsize", &v, &s, nullptr, 0) != 0) return (uint64_t)0;
                return v;
            }();
            if (physical && needed > physical) {
                throw std::runtime_error(
                    "--kl-pair needs " + std::to_string(needed >> 30) +
                    " GiB resident (A=" + std::to_string(bytes_a >> 30) +
                    " GiB, B=" + std::to_string(bytes_b >> 30) +
                    " GiB, +1 GiB overhead) but the host has " +
                    std::to_string(physical >> 30) + " GiB");
            }

            auto shared_a = q27::MetalEngine::open_shared(model_path);
            auto shared_b = q27::MetalEngine::open_shared(kl_pair_path);
            q27::MetalEngine engine_a(shared_a, context, false);
            q27::MetalEngine engine_b(shared_b, context, false);
            if (serial_prefill) {
                engine_a.set_chunked_prefill(false);
                engine_b.set_chunked_prefill(false);
            }
            auto ready = std::chrono::steady_clock::now();
            fprintf(stderr, "kl-pair ready on %s in %.2f s (two mappings: A=%s, B=%s)\n",
                    engine_a.backend().name().c_str(),
                    std::chrono::duration<double>(ready - start).count(),
                    model_path.c_str(), kl_pair_path.c_str());
            const uint32_t vocab = q27::MetalEngine::vocabulary_size();
            const uint32_t n = (uint32_t)tokens.size() - 1;
            fprintf(stderr, "kl-pair: %u positions, single pass, no resets\n", n);
            std::vector<float> p, q;
            std::vector<double> kl(n);
            auto kl_start = std::chrono::steady_clock::now();
            uint32_t done = 0, chunk_index = 0;
            while (done < n) {
                const uint32_t take = std::min(12u, n - done);
                engine_a.teacher_force_logits(tokens.data() + done, take, p);
                engine_b.teacher_force_logits(tokens.data() + done, take, q);
                for (uint32_t r = 0; r < take; r++)
                    kl[done + r] = q27::forward_kl(p.data() + (size_t)r * vocab,
                                                   q.data() + (size_t)r * vocab, vocab);
                done += take;
                if (++chunk_index % 32 == 0) fprintf(stderr, "  kl pos %u/%u\r", done, n);
                if (done / 2048 != (done - take) / 2048) {
                    double running = 0.0;
                    for (uint32_t i = 0; i < done; i++) running += kl[i];
                    fprintf(stderr, "  kl pos %u: running mean %.6g nats\n", done, running / done);
                }
            }
            auto kl_done = std::chrono::steady_clock::now();
            if (n >= 12) fprintf(stderr, "\n");
            print_kl_buckets(kl, "paired-logit forward-KL A→B");
            double mean = 0.0, peak = 0.0;
            uint32_t peak_pos = 0;
            for (uint32_t i = 0; i < n; i++) {
                mean += kl[i];
                if (kl[i] > peak) { peak = kl[i]; peak_pos = i; }
            }
            mean /= n;
            std::vector<double> sorted(kl);
            std::sort(sorted.begin(), sorted.end());
            auto quantile = [&](double p) {
                return sorted[std::min((size_t)((double)n * p), (size_t)n - 1)];
            };
            fprintf(stderr, "kl-pair tail: p50 %.4g  p90 %.4g  p99 %.4g  p99.5 %.4g  max %.4g @pos %u\n",
                    quantile(0.50), quantile(0.90), quantile(0.99), quantile(0.995), peak, peak_pos);
            for (double thr : {0.1, 0.5}) {
                uint32_t above = 0, runs = 0, cur = 0, longest = 0, longest_at = 0;
                for (uint32_t i = 0; i < n; i++) {
                    if (kl[i] > thr) {
                        if (!cur) runs++;
                        cur++; above++;
                        if (cur > longest) { longest = cur; longest_at = i + 1 - cur; }
                    } else cur = 0;
                }
                fprintf(stderr, "kl-pair runs >%.1f: %u positions, %u runs, longest %u", thr, above, runs, longest);
                if (longest) fprintf(stderr, " @pos %u", longest_at);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "kl-pair wall: %.2f s (%.2f pos/s through both engines), overall mean KL %.6g nats, max %.6g\n",
                    std::chrono::duration<double>(kl_done - kl_start).count(),
                    n / std::chrono::duration<double>(kl_done - kl_start).count(),
                    mean, peak);

            // Per-position dump for the paired bootstrap CI (A5 ship/kill
            // line lives on the CI, not the point estimate).
            if (!kl_pair_out.empty()) {
                FILE* f = fopen(kl_pair_out.c_str(), "w");
                if (!f) throw std::runtime_error("cannot open --kl-pair-out: " + kl_pair_out);
                bool ok = true;
                for (uint32_t i = 0; i < n; i++)
                    if (fprintf(f, "%.9g\n", kl[i]) < 0) { ok = false; break; }
                if (fclose(f) != 0) ok = false;
                if (!ok) throw std::runtime_error("failed writing --kl-pair-out: " + kl_pair_out);
                fprintf(stderr, "kl-pair: wrote %u per-position KLs to %s\n", n, kl_pair_out.c_str());
            }
            return 0;
        }

        // A5 sequential two-pass (2026-07-18 amendment): --logits-dump streams
        // every teacher-forced position's f32 logits row to FILE (row i =
        // position i, VOCAB f32 each, no header). One model per process, so
        // the 17 GiB baseline runs alone within the M4's budget; candidates
        // replay against the dump via --kl-vs-dump below.
        if (!logits_dump.empty()) {
            std::vector<uint32_t> tokens = load_token_file(nll_path);
            if (nll_long > 0 && tokens.size() > nll_long) tokens.resize(nll_long);
            if (tokens.size() < 2) throw std::runtime_error("--logits-dump needs at least two tokens");
            if (tokens.size() - 1 > context)
                throw std::runtime_error("--logits-dump sequence exceeds --ctx; raise --ctx");
            q27::MetalEngine engine(model_path, context, false);
            if (serial_prefill) engine.set_chunked_prefill(false);
            auto ready = std::chrono::steady_clock::now();
            fprintf(stderr, "logits-dump ready on %s in %.2f s (%s)\n",
                    engine.backend().name().c_str(),
                    std::chrono::duration<double>(ready - start).count(),
                    model_path.c_str());
            const uint32_t vocab = q27::MetalEngine::vocabulary_size();
            const uint32_t n = (uint32_t)tokens.size() - 1;
            FILE* f = fopen(logits_dump.c_str(), "wb");
            if (!f) throw std::runtime_error("cannot open --logits-dump: " + logits_dump);
            // Bind the dump to this exact token stream before any logits so a
            // same-sized wrong corpus cannot be silently replayed (P1).
            LogitsDumpHeader hdr{};
            memcpy(hdr.magic, LDMP_MAGIC, 8);
            hdr.version = 1;
            hdr.vocab = vocab;
            hdr.n_positions = n;
            hdr.token_hash = fnv1a_tokens(tokens.data(), tokens.size());
            if (fwrite(&hdr, sizeof hdr, 1, f) != 1)
                throw std::runtime_error("failed writing --logits-dump header: " + logits_dump);
            fprintf(stderr, "logits-dump: %u positions x %u f32 -> %s (%.2f GiB, token_hash %016llx)\n",
                    n, vocab, logits_dump.c_str(),
                    (double)n * vocab * 4 / (1 << 30), (unsigned long long)hdr.token_hash);
            std::vector<float> p;
            auto dump_start = std::chrono::steady_clock::now();
            uint32_t done = 0, chunk_index = 0;
            bool ok = true;
            while (done < n) {
                const uint32_t take = std::min(12u, n - done);
                engine.teacher_force_logits(tokens.data() + done, take, p);
                if (fwrite(p.data(), 4, (size_t)take * vocab, f) != (size_t)take * vocab) {
                    ok = false; break;
                }
                done += take;
                if (++chunk_index % 32 == 0) fprintf(stderr, "  dump pos %u/%u\r", done, n);
            }
            if (fflush(f) != 0) ok = false;
            if (fclose(f) != 0) ok = false;
            auto dump_done = std::chrono::steady_clock::now();
            if (!ok) throw std::runtime_error("short write on --logits-dump: " + logits_dump);
            if (n >= 12) fprintf(stderr, "\n");
            fprintf(stderr, "logits-dump wall: %.2f s (%.2f pos/s), wrote %u positions\n",
                    std::chrono::duration<double>(dump_done - dump_start).count(),
                    n / std::chrono::duration<double>(dump_done - dump_start).count(), n);
            return 0;
        }

        // Pass 2: replay a baseline logits dump against this process's model.
        // The dump is mmapped (read-only) chunk-by-chunk so even a 5 GiB 8K
        // dump adds no meaningful resident pressure; per-position forward_kl,
        // buckets, tail, and runs reporting are identical to --kl-pair's, and
        // --kl-pair-out carries the per-position values for the bootstrap.
        if (!kl_vs_dump.empty()) {
            std::vector<uint32_t> tokens = load_token_file(nll_path);
            if (nll_long > 0 && tokens.size() > nll_long) tokens.resize(nll_long);
            if (tokens.size() < 2) throw std::runtime_error("--kl-vs-dump needs at least two tokens");
            if (tokens.size() - 1 > context)
                throw std::runtime_error("--kl-vs-dump sequence exceeds --ctx; raise --ctx");
            const uint32_t vocab = q27::MetalEngine::vocabulary_size();
            const uint32_t n = (uint32_t)tokens.size() - 1;
            const uint64_t dump_bytes = std::filesystem::file_size(kl_vs_dump);
            int dfd = open(kl_vs_dump.c_str(), O_RDONLY);
            if (dfd < 0) throw std::runtime_error("cannot open --kl-vs-dump: " + kl_vs_dump);
            // Validate the binding header BEFORE trusting the payload (P1):
            // a same-byte-count dump from a different corpus must be rejected
            // loudly, not replayed into meaningless KL.
            LogitsDumpHeader hdr{};
            if (read(dfd, &hdr, sizeof hdr) != (ssize_t)sizeof hdr) { close(dfd); throw std::runtime_error("--kl-vs-dump too short for header: " + kl_vs_dump); }
            if (memcmp(hdr.magic, LDMP_MAGIC, 8) != 0) { close(dfd); throw std::runtime_error("--kl-vs-dump is not a Q27LDMP1 dump (regenerate with current --logits-dump): " + kl_vs_dump); }
            if (hdr.version != 1) { close(dfd); throw std::runtime_error("--kl-vs-dump unsupported version " + std::to_string(hdr.version) + ": " + kl_vs_dump); }
            if (hdr.vocab != vocab) { close(dfd); throw std::runtime_error("--kl-vs-dump vocab mismatch: dump " + std::to_string(hdr.vocab) + " vs engine " + std::to_string(vocab)); }
            if (hdr.n_positions != n) { close(dfd); throw std::runtime_error("--kl-vs-dump position count mismatch: dump " + std::to_string(hdr.n_positions) + " vs corpus " + std::to_string(n) + " — regenerate at the same --nll-long"); }
            const uint64_t want_hash = fnv1a_tokens(tokens.data(), tokens.size());
            if (hdr.token_hash != want_hash) { close(dfd); throw std::runtime_error("--kl-vs-dump token-stream mismatch: dump was teacher-forced on a DIFFERENT corpus (hash " +
                    std::to_string(hdr.token_hash) + " vs " + std::to_string(want_hash) + ") — refusing to replay: " + kl_vs_dump); }
            const uint64_t expect_bytes = (uint64_t)sizeof hdr + (uint64_t)n * vocab * 4;
            if (dump_bytes != expect_bytes) { close(dfd);
                throw std::runtime_error("--kl-vs-dump size mismatch: " + kl_vs_dump +
                                         " is " + std::to_string(dump_bytes) +
                                         " bytes, expected " + std::to_string(expect_bytes) +
                                         " (header + " + std::to_string(n) + " positions x " +
                                         std::to_string(vocab) + " f32) — regenerate the dump at the same --nll-long"); }
            void* map = mmap(nullptr, dump_bytes, PROT_READ, MAP_PRIVATE, dfd, 0);
            if (map == MAP_FAILED) { close(dfd); throw std::runtime_error("cannot mmap --kl-vs-dump: " + kl_vs_dump); }
            const float* baseline = (const float*)((const char*)map + sizeof hdr);
            q27::MetalEngine engine(model_path, context, false);
            if (serial_prefill) engine.set_chunked_prefill(false);
            auto ready = std::chrono::steady_clock::now();
            fprintf(stderr, "kl-vs-dump ready on %s in %.2f s (%s vs %s)\n",
                    engine.backend().name().c_str(),
                    std::chrono::duration<double>(ready - start).count(),
                    model_path.c_str(), kl_vs_dump.c_str());
            fprintf(stderr, "kl-vs-dump: %u positions, single pass, no resets\n", n);
            std::vector<float> q;
            std::vector<double> kl(n);
            auto kl_start = std::chrono::steady_clock::now();
            uint32_t done = 0, chunk_index = 0;
            while (done < n) {
                const uint32_t take = std::min(12u, n - done);
                engine.teacher_force_logits(tokens.data() + done, take, q);
                for (uint32_t r = 0; r < take; r++)
                    kl[done + r] = q27::forward_kl(baseline + (size_t)(done + r) * vocab,
                                                   q.data() + (size_t)r * vocab, vocab);
                done += take;
                if (++chunk_index % 32 == 0) fprintf(stderr, "  kl pos %u/%u\r", done, n);
                if (done / 2048 != (done - take) / 2048) {
                    double running = 0.0;
                    for (uint32_t i = 0; i < done; i++) running += kl[i];
                    fprintf(stderr, "  kl pos %u: running mean %.6g nats\n", done, running / done);
                }
            }
            auto kl_done = std::chrono::steady_clock::now();
            if (n >= 12) fprintf(stderr, "\n");
            print_kl_buckets(kl, "paired-logit forward-KL dump→model");
            double mean = 0.0, peak = 0.0;
            uint32_t peak_pos = 0;
            for (uint32_t i = 0; i < n; i++) {
                mean += kl[i];
                if (kl[i] > peak) { peak = kl[i]; peak_pos = i; }
            }
            mean /= n;
            std::vector<double> sorted(kl);
            std::sort(sorted.begin(), sorted.end());
            auto quantile = [&](double p) {
                return sorted[std::min((size_t)((double)n * p), (size_t)n - 1)];
            };
            fprintf(stderr, "kl-vs-dump tail: p50 %.4g  p90 %.4g  p99 %.4g  p99.5 %.4g  max %.4g @pos %u\n",
                    quantile(0.50), quantile(0.90), quantile(0.99), quantile(0.995), peak, peak_pos);
            for (double thr : {0.1, 0.5}) {
                uint32_t above = 0, runs = 0, cur = 0, longest = 0, longest_at = 0;
                for (uint32_t i = 0; i < n; i++) {
                    if (kl[i] > thr) {
                        if (!cur) runs++;
                        cur++; above++;
                        if (cur > longest) { longest = cur; longest_at = i + 1 - cur; }
                    } else cur = 0;
                }
                fprintf(stderr, "kl-vs-dump runs >%.1f: %u positions, %u runs, longest %u", thr, above, runs, longest);
                if (longest) fprintf(stderr, " @pos %u", longest_at);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "kl-vs-dump wall: %.2f s (%.2f pos/s), overall mean KL %.6g nats, max %.6g\n",
                    std::chrono::duration<double>(kl_done - kl_start).count(),
                    n / std::chrono::duration<double>(kl_done - kl_start).count(),
                    mean, peak);
            if (!kl_pair_out.empty()) {
                FILE* f = fopen(kl_pair_out.c_str(), "w");
                if (!f) { munmap(map, dump_bytes); close(dfd); throw std::runtime_error("cannot open --kl-pair-out: " + kl_pair_out); }
                bool ok = true;
                for (uint32_t i = 0; i < n; i++)
                    if (fprintf(f, "%.9g\n", kl[i]) < 0) { ok = false; break; }
                if (fclose(f) != 0) ok = false;
                if (!ok) { munmap(map, dump_bytes); close(dfd); throw std::runtime_error("failed writing --kl-pair-out: " + kl_pair_out); }
                fprintf(stderr, "kl-vs-dump: wrote %u per-position KLs to %s\n", n, kl_pair_out.c_str());
            }
            munmap(map, dump_bytes);
            close(dfd);
            return 0;
        }

        q27::MetalEngine engine(model_path, context, turbo3_kv);
        if (serial_prefill) engine.set_chunked_prefill(false);
        auto loaded = std::chrono::steady_clock::now();
        fprintf(stderr, "Metal model ready on %s in %.2f s\n", engine.backend().name().c_str(),
                std::chrono::duration<double>(loaded - start).count());
        if (validate_only) { puts("artifacts and Metal architecture: OK"); return 0; }

        if (!nll_path.empty()) {
            std::vector<uint32_t> tokens = load_token_file(nll_path);
            if (nll_long > 0 && tokens.size() > nll_long) tokens.resize(nll_long);
            // n-1 positions are encoded; the final token is only a target.
            if (tokens.size() - 1 > context)
                throw std::runtime_error("--nll-long sequence exceeds --ctx; raise --ctx");
            fprintf(stderr, "nll-long: %zu tokens, single pass, no resets%s\n",
                    tokens.size(), turbo3_kv ? " (turbo3 KV)" : "");
            auto nll_start = std::chrono::steady_clock::now();
            std::vector<float> nll = engine.teacher_force_nll(tokens);
            auto nll_done = std::chrono::steady_clock::now();
            print_nll_long_buckets(nll);
            double mean = 0.0;
            for (float v : nll) mean += v;
            mean /= nll.empty() ? 1.0 : (double)nll.size();
            fprintf(stderr, "nll-long wall: %.2f s (%.2f tok/s), overall mean NLL %.4f PPL %.3f\n",
                    std::chrono::duration<double>(nll_done - nll_start).count(),
                    nll.size() / std::chrono::duration<double>(nll_done - nll_start).count(),
                    mean, std::exp(mean));
            return 0;
        }

        // EOS position-invariant gate (open codex finding, 2026-07-15: batched
        // MTP commits the accepted prefix before sinks fire, leaving
        // position/KV/GDN advanced past the last emitted token). Pick a real
        // mid-stream token from a greedy reference as EOS, replay through the
        // batched MTP stream, and assert the serial-walk invariant:
        // emitted == ref[0..k) and position() == prompt + k.
        if (eos_gate) {
            uint32_t pending = engine.ingest_prompt(prompt, false, true);
            std::vector<uint32_t> ref = engine.generate_from_pending(pending, count);
            uint32_t k = 0;
            for (uint32_t i = 6; i + 4 < (uint32_t)ref.size(); i++) {
                bool seen = false;
                for (uint32_t j = 0; j < i; j++) if (ref[j] == ref[i]) { seen = true; break; }
                if (!seen) { k = i; break; }
            }
            if (!k) throw std::runtime_error("eos-gate: no unique mid-stream token to use as EOS; change prompt/-n");
            const uint32_t eos = ref[k];
            const uint32_t pending2 = engine.ingest_prompt(prompt, true, true);
            if (pending2 != ref[0])
                throw std::runtime_error("eos-gate: re-ingest diverged from reference pass");
            std::vector<uint32_t> emitted;
            q27::MetalEngine::StopCause cause;
            engine.stream_from_pending(pending2, count, eos, mtp_width,
                                       [&](uint32_t t) { emitted.push_back(t); return true; }, cause);
            const bool cause_ok = cause == q27::MetalEngine::StopCause::Eos;
            const bool text_ok = emitted.size() == k &&
                                 std::equal(emitted.begin(), emitted.end(), ref.begin());
            const uint32_t expect_pos = (uint32_t)prompt.size() + k;
            const bool pos_ok = engine.position() == expect_pos;
            fprintf(stderr, "eos-gate: k=%u eos=%u | cause %s | emitted %zu (%s) | position %u vs expected %u (%s)\n",
                    k, eos, cause_ok ? "Eos" : "WRONG", emitted.size(), text_ok ? "match" : "MISMATCH",
                    engine.position(), expect_pos, pos_ok ? "match" : "ADVANCED PAST EMITTED");
            const bool ok = cause_ok && text_ok && pos_ok;
            fprintf(stderr, "eos-gate: %s\n", ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }

        // Gate 0 oracle verifier (docs/plans/2026-07-15-sibling-drafter-probe.md):
        // can batched verification pay AT ALL on this hardware? Replay a known
        // greedy continuation as free draft proposals (D=0, perfect acceptance)
        // through the batched verify rounds and price them against the
        // production serial rate. Serial passes bracket the oracle pass so
        // thermal drift is visible in the two G values, and a one-step probe
        // from both end states is the can-fail state-integrity gate.
        if (oracle_width) {
            auto clock_now = [] { return std::chrono::steady_clock::now(); };
            auto ms_between = [](std::chrono::steady_clock::time_point a,
                                 std::chrono::steady_clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            // Pass 1: serial reference at the production greedy rate
            // (resident chaining included when enabled).
            uint32_t pending = engine.ingest_prompt(prompt, false, true);
            auto s0 = clock_now();
            std::vector<uint32_t> ref = engine.generate_from_pending(pending, count);
            auto s1 = clock_now();
            const double g1_ms = ms_between(s0, s1) / (count - 1);
            const uint32_t serial_probe = engine.step(ref.back());
            const uint32_t serial_end_pos = engine.position();
            const std::vector<float> serial_probe_logits = engine.read_logits();

            // Pass 2: oracle rounds over the reference at width W.
            const uint32_t pending2 = engine.ingest_prompt(prompt, false, true);
            if (pending2 != ref[0])
                throw std::runtime_error("oracle: prompt re-ingest diverged from pass 1 (prefill nondeterminism)");
            std::vector<uint32_t> preds(oracle_width);
            std::vector<double> round_ms;
            unsigned long long agree = 0, agree_den = 0;
            uint32_t emitted = 0;
            while (emitted < count) {
                const uint32_t live = std::min<uint32_t>(oracle_width, count - emitted);
                if (live < 2) { emitted++; continue; }  // final token: committed, never encoded
                const bool last_round = (emitted + live == count);
                auto r0 = clock_now();
                engine.oracle_round(ref.data() + emitted, live, last_round, preds.data());
                round_ms.push_back(ms_between(r0, clock_now()));
                // Observational agreement: verify argmax vs the serial
                // reference, including the bonus lane when a next reference
                // token exists. Never acted on — commit is teacher-forced.
                for (uint32_t k = 0; k + 1 < live; k++) { agree_den++; agree += preds[k] == ref[emitted + k + 1]; }
                if (emitted + live < count) { agree_den++; agree += preds[live - 1] == ref[emitted + live]; }
                emitted += live;
            }
            const uint32_t oracle_probe = engine.step(ref.back());
            const uint32_t oracle_end_pos = engine.position();
            const std::vector<float> oracle_probe_logits = engine.read_logits();

            // Pass 3: serial re-measure to bracket thermal/clock drift.
            const uint32_t pending3 = engine.ingest_prompt(prompt, false, true);
            auto s2 = clock_now();
            std::vector<uint32_t> ref2 = engine.generate_from_pending(pending3, count);
            auto s3 = clock_now();
            const double g2_ms = ms_between(s2, s3) / (count - 1);
            if (ref2 != ref)
                throw std::runtime_error("oracle: serial re-measure diverged from pass 1 (nondeterminism)");

            // State gate: probe argmax and position must match, and the full
            // probe logits must sit inside the known chunk-vs-serial numeric
            // class. Tolerance calibrated to the MEASURED class envelope
            // (--envelope serial, 2,048 wikitext2 positions,
            // 2026-07-16-envelope-instrument.md): p99.5 = 0.646, corpus max
            // = 1.393 — the earlier 0.25 (three observations) and 0.7
            // (384-position control max, which turned out to be the class
            // p99.5) both false-fail inside the class at corpus scale. Real
            // state corruption (missing KV rows, broken gdn_replay) moves
            // logits by orders of magnitude and flips the probe
            // argmax/position, so 1.5 (above corpus max) keeps the gate able
            // to fail.
            double probe_max_diff = 0.0;
            for (size_t v = 0; v < serial_probe_logits.size(); v++)
                probe_max_diff = std::max(probe_max_diff,
                                          (double)std::fabs(oracle_probe_logits[v] - serial_probe_logits[v]));
            const bool state_ok = oracle_probe == serial_probe && oracle_end_pos == serial_end_pos &&
                                  probe_max_diff < 1.5;
            const double g_ms = (g1_ms + g2_ms) / 2.0;
            double total = 0.0;
            for (double v : round_ms) total += v;
            const double mean_ms = total / round_ms.size();
            double warm_total = 0.0;
            for (size_t r = 1; r < round_ms.size(); r++) warm_total += round_ms[r];
            const double warm_ms = round_ms.size() > 1 ? warm_total / (round_ms.size() - 1) : mean_ms;
            // count-1 tokens are attributable to the timed rounds: ref[0]
            // was free from prefill in BOTH passes (serial times count-1
            // steps), and a count%width==1 tail singleton is committed
            // without a round. Same denominator on both sides of S(w).
            const double tok_per_round = (double)(count - 1) / round_ms.size();
            fprintf(stderr, "oracle w=%u: G %.2f/%.2f ms/tok (serial %.2f/%.2f tok/s), %zu rounds, %.2f tok/round\n",
                    oracle_width, g1_ms, g2_ms, 1000.0 / g1_ms, 1000.0 / g2_ms,
                    round_ms.size(), tok_per_round);
            fprintf(stderr, "oracle w=%u: round %.2f ms mean, %.2f ms warm (first %.2f), oracle wall %.2f tok/s\n",
                    oracle_width, mean_ms, warm_ms, round_ms[0], 1000.0 * tok_per_round / mean_ms);
            fprintf(stderr, "oracle w=%u: S(w) = %.3fx mean, %.3fx warm | agreement %llu/%llu (%.1f%%)\n",
                    oracle_width, tok_per_round * g_ms / mean_ms, tok_per_round * g_ms / warm_ms,
                    agree, agree_den, agree_den ? 100.0 * agree / agree_den : 0.0);
            fprintf(stderr, "oracle w=%u: state gate %s (probe %u vs %u, position %u vs %u, logits max|d| %.4g)\n",
                    oracle_width, state_ok ? "PASS" : "FAIL",
                    oracle_probe, serial_probe, oracle_end_pos, serial_end_pos, probe_max_diff);
            return state_ok ? 0 : 1;
        }

        std::vector<uint32_t> generated;
        if (!load_state_path.empty()) {
            auto t0 = std::chrono::steady_clock::now();
            const uint32_t pos = engine.load_state(load_state_path);
            fprintf(stderr, "state: loaded %s in %.3f s (position %u)\n", load_state_path.c_str(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), pos);
            generated = engine.generate_from_pending(engine.pending_from_logits(), count);
        } else if (!save_state_path.empty()) {
            const uint32_t pending = engine.ingest_prompt(prompt, false, true);
            auto t0 = std::chrono::steady_clock::now();
            engine.save_state(save_state_path, prompt.data(), (uint32_t)prompt.size());
            fprintf(stderr, "state: saved %s in %.3f s (position %u)\n", save_state_path.c_str(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    engine.position());
            generated = engine.generate_from_pending(pending, count);
        } else {
            generated = sampling.temperature>0 ? engine.generate_sampled(prompt,count,sampling)
                                           : mtp_width ? engine.generate_mtp(prompt,count,mtp_width)
                                           : suffix_width ? (suffix_serial
                                                  ? engine.generate_suffix_serial(prompt,count,suffix_width)
                                                  : engine.generate_suffix(prompt,count,suffix_width))
                                                          : engine.generate(prompt,count);
        }
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
        if(suffix_width && !suffix_serial) {
            auto sfx=engine.last_suffix_stats();
            fprintf(stderr,"suffix bursts: %llu fired (%llu @<=16 lanes, %llu @32, %llu @48), %llu serial-fallback rounds\n",
                    (unsigned long long)sfx.burst_rounds,(unsigned long long)sfx.lanes_le16,
                    (unsigned long long)sfx.lanes_32,(unsigned long long)sfx.lanes_48,
                    (unsigned long long)sfx.fallback_rounds);
        }
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
