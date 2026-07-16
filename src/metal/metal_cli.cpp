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
#include <iterator>
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
void print_kl_buckets(const std::vector<double>& kl) {
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
    printf("KV forward-KL vs fp16 baseline by target position (%zu tokens, no resets):\n",
           kl.size() + 1);
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
                "[-n count] [--ctx count] [--mtp width | --suffix width] [--kv fp16|turbo3] "
                "[--prefill chunk|serial] [--nll-long N] [--kl-kv | --kl-kv-self] [--chunk-parity N] "
                "[--temperature T --top-p P --top-k K --seed S] "
                "[--dump-logits file]\n",
                argv[0]);
        return 1;
    }
    try {
        std::string model_path=argv[1],tokenizer_path=argv[2],token_list,prompt_text,dump_logits,nll_path;
        uint32_t count=1,context=128,mtp_width=0,suffix_width=0,nll_long=0; q27::SamplingParams sampling;
        bool turbo3_kv = false, validate_only = false, serial_prefill = false;
        bool kl_kv = false, kl_self = false;
        uint32_t chunk_parity = 0;
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--tokens" && i + 1 < argc) token_list = argv[++i];
            else if (arg == "--prompt" && i + 1 < argc) prompt_text = argv[++i];
            else if (arg == "--nll" && i + 1 < argc) nll_path = argv[++i];
            else if (arg == "--nll-long" && i + 1 < argc) nll_long = parse_u32(argv[++i], "--nll-long");
            else if (arg == "--validate-only") validate_only = true;
            else if (arg == "--kl-kv") kl_kv = true;
            else if (arg == "--kl-kv-self") { kl_kv = true; kl_self = true; }
            else if (arg == "--chunk-parity" && i + 1 < argc) chunk_parity = parse_u32(argv[++i], "--chunk-parity");
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
            else if (arg == "--prefill" && i + 1 < argc) {
                std::string mode = argv[++i];
                if (mode == "serial") serial_prefill = true;
                else if (mode != "chunk") throw std::runtime_error("--prefill must be chunk or serial");
            }
            else throw std::runtime_error("unknown/incomplete argument: " + arg);
        }
        if (mtp_width && suffix_width) throw std::runtime_error("--mtp and --suffix are mutually exclusive");
        q27::validate_sampling(sampling);
        if(sampling.temperature>0 && (mtp_width || suffix_width))
            throw std::runtime_error("sampling cannot be combined with speculative modes");
        if(!token_list.empty() && !prompt_text.empty()) throw std::runtime_error("--tokens and --prompt are mutually exclusive");
        if (!nll_path.empty() && (!token_list.empty() || !prompt_text.empty() || validate_only))
            throw std::runtime_error("--nll cannot be combined with --tokens/--prompt/--validate-only");
        if (!nll_path.empty() && !nll_long)
            throw std::runtime_error("--nll currently requires --nll-long N (chunked llama-ppl mode is CUDA-only)");
        if (nll_long && nll_path.empty())
            throw std::runtime_error("--nll-long requires --nll FILE");
        if (kl_kv && nll_path.empty())
            throw std::runtime_error("--kl-kv rides the --nll FILE --nll-long N input path");
        if (chunk_parity && nll_path.empty())
            throw std::runtime_error("--chunk-parity rides the --nll FILE input path");
        if (chunk_parity && kl_kv)
            throw std::runtime_error("--chunk-parity and --kl-kv are separate instruments");
        if (kl_kv && turbo3_kv)
            throw std::runtime_error("--kl-kv builds its own fp16 baseline and turbo3 subject; drop --kv");
        if (nll_path.empty() && !validate_only && token_list.empty() && prompt_text.empty())
            throw std::runtime_error("--tokens, --prompt, --nll, or --validate-only is required");
        if (!nll_path.empty() && (mtp_width || suffix_width || sampling.temperature > 0 || !dump_logits.empty()))
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
            const uint32_t vocab = q27::MetalEngine::vocabulary_size();
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

            for (uint32_t width : {17u, 48u, 96u}) {
                subject.reset();
                std::vector<float> q;
                double max_abs = 0, sum_abs = 0, max_kl = 0, sum_kl = 0;
                double ref_nll = 0, subj_nll = 0, max_nll_delta = 0;
                double min_flip_margin = 1e300, max_flip_margin = 0;
                uint64_t flips = 0, overlap = 0;
                uint32_t done = 0;
                while (done < n) {
                    const uint32_t take = std::min(width, n - done);
                    subject.teacher_force_logits_wide(tokens.data() + done, take, q);
                    for (uint32_t r = 0; r < take; r++) {
                        const float* pr = ref.data() + (size_t)(done + r) * vocab;
                        const float* qr = q.data() + (size_t)r * vocab;
                        double mx = 0;
                        for (uint32_t v = 0; v < vocab; v++)
                            mx = std::max(mx, std::abs((double)pr[v] - (double)qr[v]));
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
                printf("width %2u vs 12: max|dlogit| %.6g (mean %.6g), KL mean %.3g max %.3g nats, "
                       "top-1 flips %llu/%u (ref margin min %.4g max %.4g), top-20 overlap %.2f%%, "
                       "NLL %.6f vs %.6f (max |dNLL| %.4g)\n",
                       width, max_abs, sum_abs / n, sum_kl / n, max_kl,
                       (unsigned long long)flips, n,
                       flips ? min_flip_margin : 0.0, flips ? max_flip_margin : 0.0,
                       100.0 * overlap / ((double)n * 20.0),
                       subj_nll / n, ref_nll / n, max_nll_delta);
            }
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
            q27::MetalEngine subject(shared, context, !kl_self);
            if (serial_prefill) {
                baseline.set_chunked_prefill(false);
                subject.set_chunked_prefill(false);
            }
            auto ready = std::chrono::steady_clock::now();
            fprintf(stderr, "Metal model ready on %s in %.2f s (two engines, one mapping: fp16 baseline vs %s)\n",
                    baseline.backend().name().c_str(),
                    std::chrono::duration<double>(ready - start).count(),
                    kl_self ? "fp16 self-check" : "turbo3");
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
            for (double v : kl) { mean += v; if (v > peak) peak = v; }
            mean /= n;
            fprintf(stderr, "kl-kv wall: %.2f s (%.2f pos/s through both engines), overall mean KL %.6g nats, max %.6g\n",
                    std::chrono::duration<double>(kl_done - kl_start).count(),
                    n / std::chrono::duration<double>(kl_done - kl_start).count(),
                    mean, peak);
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
