#include "metal_engine.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

bool rejects_logits(q27::MetalEngine& engine) {
    try {
        (void)engine.read_logits();
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

bool rejects_pending(q27::MetalEngine& engine, uint32_t pending) {
    try {
        (void)engine.generate_from_pending(pending, 1, 2);
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

bool same_logits(const std::vector<float>& left, const std::vector<float>& right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

bool artifact_requires_per_tensor_upload(const char* path, uint64_t max_buffer_length) {
    struct stat info {};
    if (::stat(path, &info) != 0 || info.st_size <= 0)
        throw std::runtime_error("could not read model artifact size");
    const uint64_t logical_size = static_cast<uint64_t>(info.st_size);
    const uint64_t page_size = static_cast<uint64_t>(::getpagesize());
    if (logical_size > std::numeric_limits<uint64_t>::max() - (page_size - 1))
        throw std::runtime_error("model artifact size overflow");
    const uint64_t mapped_size = (logical_size + page_size - 1) / page_size * page_size;
    return mapped_size > max_buffer_length;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s OFFICIAL.q27 [BONSAI.q27]\n", argv[0]);
        return 2;
    }
    try {
        {
            const char* old_cells=getenv("Q27_METAL_KV_FP16_CELLS");
            const char* old_codec=getenv("Q27_METAL_KV_CELLS_CODEC");
            const bool had_cells=old_cells!=nullptr, had_codec=old_codec!=nullptr;
            const std::string saved_cells=had_cells?old_cells:"";
            const std::string saved_codec=had_codec?old_codec:"";
            if(setenv("Q27_METAL_KV_FP16_CELLS","",1) ||
               setenv("Q27_METAL_KV_CELLS_CODEC","e4m3",1))
                throw std::runtime_error("could not set KV codec test environment");
            bool rejected=false;
            try {
                auto shared=q27::MetalEngine::open_shared(argv[1]);
                (void)q27::MetalEngine::serving_reservation_bytes(*shared,1,q27::KvKind::F16,0);
            } catch(const std::runtime_error& error) {
                rejected=std::string(error.what()).find("a non-empty Q27_METAL_KV_FP16_CELLS")!=std::string::npos;
            }
            if(had_cells) setenv("Q27_METAL_KV_FP16_CELLS",saved_cells.c_str(),1);
            else unsetenv("Q27_METAL_KV_FP16_CELLS");
            if(had_codec) setenv("Q27_METAL_KV_CELLS_CODEC",saved_codec.c_str(),1);
            else unsetenv("Q27_METAL_KV_CELLS_CODEC");
            if(!rejected)
                throw std::runtime_error("empty e4m3 KV side-cache configuration was accepted");
        }
        {
            auto limited=q27::MetalEngine::open_shared(argv[1]);
            limited->cache_budget=1;
            bool rejected=false;
            try { q27::MetalEngine over_budget(limited,1,q27::KvKind::F16); }
            catch(const std::runtime_error& error) {
                rejected=std::string(error.what()).find("configured cache budget")!=std::string::npos;
            }
            if(!rejected) {
                std::fprintf(stderr,"explicit shared cache budget was not enforced before allocation\n");
                return 1;
            }
        }
        q27::MetalEngine engine(argv[1], 8);
        const bool expected_per_tensor = artifact_requires_per_tensor_upload(
            argv[1], engine.backend().max_buffer_length());
        if (engine.used_per_tensor_upload() != expected_per_tensor) {
            std::fprintf(stderr,
                         "model upload path did not match artifact size and device limit: "
                         "expected %s, got %s\n",
                         expected_per_tensor ? "per-tensor" : "whole-mapping",
                         engine.used_per_tensor_upload() ? "per-tensor" : "whole-mapping");
            return 1;
        }
        {
            auto initial=engine.capture_state();
            const std::vector<uint32_t> baseline=engine.generate_mtp(
                {760},2,2,UINT32_MAX);
            engine.restore_state(*initial);
            const uint32_t restored_pending=engine.ingest_prompt({760},true,false);
            const std::vector<uint32_t> restored=engine.generate_from_pending(
                restored_pending,2,2,UINT32_MAX);
            if(restored!=baseline)
                throw std::runtime_error("position-zero snapshot changed MTP continuation");
            engine.restore_state(*initial);
        }
        if (!rejects_logits(engine)) {
            std::fprintf(stderr, "fresh engine exposed non-resident logits\n");
            return 1;
        }

        const uint32_t pending = engine.ingest_prompt({760}, false, true);
        if (engine.position() != 1 ||
            engine.read_logits().size() != q27::MetalEngine::vocabulary_size()) {
            std::fprintf(stderr, "prompt ingestion did not publish current logits\n");
            return 1;
        }
        auto snapshot = engine.capture_state();

        engine.reset();
        if (!rejects_logits(engine) || !rejects_pending(engine, pending)) {
            std::fprintf(stderr, "reset did not invalidate logits and pending state\n");
            return 1;
        }

        engine.restore_state(*snapshot);
        if (engine.position() != 1 ||
            engine.read_logits().size() != q27::MetalEngine::vocabulary_size()) {
            std::fprintf(stderr, "snapshot restore did not restore logits residency\n");
            return 1;
        }
        {
            q27::MetalEngine engine(argv[1], 16, q27::KvKind::F16);
            // The serving scheduler splits MTP prompt warming into bounded
            // serial quanta. It must match the former whole-prompt path.
            const std::vector<uint32_t> serial_prompt(9,1);
            auto run_mtp_round=[&](bool chunked) {
                engine.reset();
                uint32_t pending=0;
                if(chunked) {
                    for(size_t i=0;i<serial_prompt.size();) {
                        const uint32_t take=(uint32_t)std::min<size_t>(
                            q27::MetalEngine::serial_prefill_chunk_max(),
                            serial_prompt.size()-i);
                        const bool final=i+take==serial_prompt.size();
                        const uint32_t next=engine.prefill_serial_chunk(
                            serial_prompt.data()+i,take,true,final);
                        if(final) pending=next;
                        i+=take;
                    }
                } else pending=engine.ingest_prompt(serial_prompt,true,false);
                uint32_t live=4;
                std::vector<uint32_t> committed;
                const uint32_t next=engine.mtp_round(
                    pending,4,std::numeric_limits<uint32_t>::max(),4,live,committed);
                return std::make_pair(committed,next);
            };
            const auto whole=run_mtp_round(false);
            const auto split=run_mtp_round(true);
            if(whole!=split)
                throw std::runtime_error("chunked serial MTP prefill changed decode state");
            engine.reset();
            engine.set_chunked_prefill(true);
            const auto batched_generation=engine.generate_mtp(serial_prompt,6,4);
            const uint32_t batched_position=engine.position();
            engine.reset();
            engine.set_chunked_prefill(false);
            const auto serial_generation=engine.generate_mtp(serial_prompt,6,4);
            if(serial_generation!=batched_generation || engine.position()!=batched_position)
                throw std::runtime_error("serial multi-round MTP changed decode state");
            engine.set_chunked_prefill(true);
            // A serial prefill that does not warm layer-64 KV must fail
            // before MTP can read stale rows from a previous generation.
            engine.reset();
            const uint32_t stale_pending = engine.ingest_prompt({1, 1}, false, false);
            const uint32_t stale_position = engine.position();
            bool stale_mtp_rejected = false;
            try {
                uint32_t live = 4;
                std::vector<uint32_t> committed;
                (void)engine.mtp_round(stale_pending, 4, UINT32_MAX, 4, live, committed);
            } catch (const std::runtime_error& error) {
                stale_mtp_rejected = std::string(error.what()).find(
                    "MTP cache is not valid") != std::string::npos;
            }
            if (!stale_mtp_rejected || engine.position() != stale_position)
                throw std::runtime_error("MTP accepted an unwarmed serial prefix");
            engine.reset();

            const uint32_t pending = engine.ingest_prompt({1}, false, true);
            int sink_calls = 0;
            q27::MetalEngine::StopCause cause = q27::MetalEngine::StopCause::MaxTokens;
            const uint32_t emitted = engine.stream_from_pending(
                pending, 4, std::numeric_limits<uint32_t>::max(), 4,
                [&](uint32_t) { return ++sink_calls < 2; }, cause);
            if (cause != q27::MetalEngine::StopCause::Cancelled || emitted != 1)
                throw std::runtime_error("batched MTP cancellation contract mismatch");
            if (engine.position() != 0)
                throw std::runtime_error("cancelled batched MTP left speculative state resident");
            bool stale_rejected = false;
            try {
                (void)engine.pending_from_logits();
            } catch (const std::runtime_error& error) {
                stale_rejected = std::string(error.what()).find("no resident logits") !=
                                 std::string::npos;
            }
            if (!stale_rejected)
                throw std::runtime_error("cancelled batched MTP left logits reusable");

            std::vector<uint32_t> mask((q27::MetalEngine::vocabulary_size()+31)/32,~0u);
            const int mask_id=engine.mask_pool_add(mask.data());
            if(mask_id<0) throw std::runtime_error("failed to allocate tool constraint mask");
            engine.set_tool_constraint(mask_id);
            const uint32_t constrained_pending=engine.ingest_prompt({1},true,true);
            const uint32_t constrained_position=engine.position();
            auto expect_constraint_reject=[&](const char* label,const std::function<void()>& call) {
                bool rejected=false;
                try { call(); }
                catch(const std::runtime_error& error) {
                    rejected=std::string(error.what()).find("tool constraints require serial decode")!=
                             std::string::npos;
                }
                if(!rejected) throw std::runtime_error(std::string(label)+" accepted active tool constraint");
                if(engine.position()!=constrained_position)
                    throw std::runtime_error(std::string(label)+" mutated state before rejection");
            };
            expect_constraint_reject("stream_from_pending",[&] {
                q27::MetalEngine::StopCause stop;
                (void)engine.stream_from_pending(constrained_pending,2,UINT32_MAX,4,
                    [](uint32_t) { return true; },stop);
            });
            expect_constraint_reject("generate_from_pending",[&] {
                (void)engine.generate_from_pending(constrained_pending,2,4);
            });
            expect_constraint_reject("mtp_round",[&] {
                uint32_t live=4; std::vector<uint32_t> committed;
                (void)engine.mtp_round(constrained_pending,2,UINT32_MAX,4,live,committed);
            });
            expect_constraint_reject("mtp_sample_round",[&] {
                uint32_t live=4; std::vector<uint32_t> committed;
                q27::SamplingParams params; std::mt19937_64 rng(1);
                (void)engine.mtp_sample_round(constrained_pending,2,UINT32_MAX,4,live,
                                              params,rng,committed);
            });
        }

        const std::vector<float> baseline_logits = engine.read_logits();
        const std::vector<uint32_t> unlimited_sample = engine.generate_sampled_from_logits(
            1, q27::SamplingParams{1.0f, 0.95f, 0, 17});
        if (unlimited_sample.size() != 1 || engine.position() != 1 ||
            !same_logits(engine.read_logits(), baseline_logits)) {
            std::fprintf(stderr, "top_k=0 did not use the non-mutating full-logits path\n");
            return 1;
        }
        const std::vector<uint32_t> baseline_retry =
            engine.generate_from_pending(pending, 2, 2);
        engine.restore_state(*snapshot);

        q27::MetalEngine::StopCause cause = q27::MetalEngine::StopCause::MaxTokens;
        bool sink_called = false;
        uint32_t emitted = engine.stream_from_pending(
            pending, 2, pending, 2,
            [&](uint32_t) { sink_called = true; return true; }, cause);
        if (emitted != 0 || cause != q27::MetalEngine::StopCause::Eos || sink_called ||
            engine.position() != 1 || !same_logits(engine.read_logits(), baseline_logits)) {
            std::fprintf(stderr, "MTP EOS changed unconsumed resident state\n");
            return 1;
        }
        if (engine.generate_from_pending(pending, 2, 2) != baseline_retry) {
            std::fprintf(stderr, "MTP EOS changed retry behavior\n");
            return 1;
        }
        {
            auto requires_mtp_reject=[&](const char* label) {
                bool rejected=false;
                try {
                    uint32_t live=2;
                    std::vector<uint32_t> committed;
                    (void)engine.mtp_round(pending,2,UINT32_MAX,2,live,committed);
                } catch(const std::runtime_error& error) {
                    rejected=std::string(error.what()).find(
                        "MTP cache is not valid for this prefix")!=std::string::npos;
                }
                if(!rejected)
                    throw std::runtime_error(std::string(label)+
                        " left stale MTP state usable");
            };
            const uint32_t forced[2]={1,1};
            std::vector<float> logits;
            engine.restore_state(*snapshot);
            engine.teacher_force_logits(forced,2,logits);
            requires_mtp_reject("teacher_force_logits");
            engine.restore_state(*snapshot);
            engine.teacher_force_logits_wide(forced,2,logits);
            requires_mtp_reject("teacher_force_logits_wide");
            engine.restore_state(*snapshot);
            (void)engine.teacher_force_nll({1,1,1});
            requires_mtp_reject("teacher_force_nll");
        }
        {
            q27::MetalEngine engine(argv[1], 16, q27::KvKind::F16);
            const std::vector<uint32_t> prompt{1, 2};
            (void)engine.ingest_prompt(prompt, false, true);
            const std::string path = "/tmp/q27-snapshot-fd-" +
                                     std::to_string((long long)getpid()) + ".q27snap";
            int fd = -1;
            try {
                uint32_t device_leases = 0;
                engine.save_state(path, prompt.data(), (uint32_t)prompt.size(), true,
                    [&](const std::function<void()>& operation) {
                        device_leases++;
                        operation();
                    });
                if (device_leases < 2)
                    throw std::runtime_error("snapshot device work was not split into bounded leases");
                fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd < 0 || lseek(fd, 7, SEEK_SET) != 7)
                    throw std::runtime_error("cannot prepare snapshot descriptor offset gate");
                const auto info = q27::MetalEngine::peek_snapshot_fd(fd, path);
                if (info.tokens != prompt || lseek(fd, 0, SEEK_CUR) != 7)
                    throw std::runtime_error("peek_snapshot_fd changed its caller descriptor offset");
                engine.reset();
                uint32_t restore_leases = 0;
                const auto restored = engine.load_state_fd(fd, path,
                    nullptr,
                    [&](const std::function<void()>& operation) {
                        restore_leases++;
                        operation();
                    });
                if (restored != prompt.size() || restore_leases < 2 ||
                    lseek(fd, 0, SEEK_CUR) != 7)
                    throw std::runtime_error("load_state_fd lease or descriptor offset mismatch");
                int checkpoints = 0;
                bool cancelled = false;
                engine.reset();
                try {
                    (void)engine.load_state_fd(fd, path, nullptr,
                        q27::MetalEngine::DeviceLease{},
                        [&] {
                            if (++checkpoints == 3)
                                throw std::runtime_error("snapshot cancellation gate");
                        });
                } catch (const std::runtime_error& error) {
                    cancelled = std::string(error.what()) == "snapshot cancellation gate";
                }
                if (!cancelled || checkpoints != 3 || lseek(fd, 0, SEEK_CUR) != 7)
                    throw std::runtime_error("snapshot validation ignored cancellation checkpoint");
                const uint32_t original_threshold = engine.backend().gqa_threshold();
                engine.backend().set_gqa_threshold(original_threshold == 1 ? 2 : 1);
                bool runtime_rejected = false;
                try {
                    (void)engine.load_state_fd(fd, path);
                } catch (const std::runtime_error& error) {
                    runtime_rejected = std::string(error.what()).find(
                        "snapshot runtime configuration does not match") != std::string::npos;
                }
                engine.backend().set_gqa_threshold(original_threshold);
                if (!runtime_rejected)
                    throw std::runtime_error("snapshot runtime configuration mismatch was accepted");
                const int mutable_fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
                if (mutable_fd < 0)
                    throw std::runtime_error("cannot open snapshot mutation gate");
                try {
                    struct stat snapshot_stat{};
                    if (fstat(mutable_fd, &snapshot_stat) != 0 || snapshot_stat.st_size < 1)
                        throw std::runtime_error("cannot size snapshot mutation gate");
                    const off_t mutation_offset = snapshot_stat.st_size - 1;
                    unsigned char original = 0;
                    if (pread(mutable_fd, &original, 1, mutation_offset) != 1)
                        throw std::runtime_error("cannot read snapshot mutation byte");
                    const unsigned char changed = original ^ 0xff;
                    if (pwrite(mutable_fd, &changed, 1, mutation_offset) != 1)
                        throw std::runtime_error("cannot corrupt snapshot payload");
                    bool checksum_rejected = false;
                    engine.reset();
                    try {
                        (void)engine.load_state_fd(fd, path);
                    } catch (const std::runtime_error& error) {
                        checksum_rejected = std::string(error.what()).find(
                            "snapshot payload checksum mismatch") != std::string::npos;
                    }
                    if (pwrite(mutable_fd, &original, 1, mutation_offset) != 1)
                        throw std::runtime_error("cannot restore corrupted snapshot byte");
                    if (!checksum_rejected)
                        throw std::runtime_error("stable snapshot payload corruption was accepted");
                    // Q27SNAP3 reserved follows magic, artifact size, two SHA-1
                    // identities, and kv/position/token_count.
                    constexpr off_t reserved_offset=8+8+20+20+3*4;
                    unsigned char reserved=0;
                    if(pread(mutable_fd,&reserved,1,reserved_offset)!=1)
                        throw std::runtime_error("cannot read snapshot control metadata");
                    const unsigned char flipped_reserved=reserved^1u;
                    if(pwrite(mutable_fd,&flipped_reserved,1,reserved_offset)!=1)
                        throw std::runtime_error("cannot corrupt snapshot control metadata");
                    bool metadata_rejected=false;
                    try {
                        (void)engine.load_state_fd(fd,path);
                    } catch(const std::runtime_error& error) {
                        metadata_rejected=std::string(error.what()).find(
                            "snapshot payload checksum mismatch")!=std::string::npos;
                    }
                    if(pwrite(mutable_fd,&reserved,1,reserved_offset)!=1)
                        throw std::runtime_error("cannot restore snapshot control metadata");
                    if(!metadata_rejected)
                        throw std::runtime_error("snapshot control metadata corruption was accepted");
                    bool mutated = false;
                    restore_leases = 0;
                    bool rejected = false;
                    engine.reset();
                    try {
                        (void)engine.load_state_fd(fd, path,
                            nullptr,
                            [&](const std::function<void()>& operation) {
                                if (!mutated && ++restore_leases == 2) {
                                    const unsigned char changed = original ^ 0xff;
                                    if (pwrite(mutable_fd, &changed, 1, mutation_offset) != 1)
                                        throw std::runtime_error("cannot mutate snapshot payload");
                                    mutated = true;
                                }
                                operation();
                            });
                    } catch (const std::runtime_error& error) {
                        rejected = std::string(error.what()).find(
                            "snapshot payload changed during load") != std::string::npos;
                    }
                    if (pwrite(mutable_fd, &original, 1, mutation_offset) != 1)
                        throw std::runtime_error("cannot restore snapshot mutation byte");
                    if (!mutated || !rejected)
                        throw std::runtime_error("snapshot payload mutation was not rejected");
                } catch (...) {
                    close(mutable_fd);
                    throw;
                }
                close(mutable_fd);
                close(fd);
                fd = -1;
                std::remove(path.c_str());
            } catch (...) {
                if (fd >= 0) close(fd);
                std::remove(path.c_str());
                throw;
            }
        }
        engine.restore_state(*snapshot);

        cause = q27::MetalEngine::StopCause::MaxTokens;
        emitted = engine.stream_from_pending(
            pending, 2, UINT32_MAX, 2,
            [](uint32_t) { return false; }, cause);
        if (emitted != 0 || cause != q27::MetalEngine::StopCause::Cancelled ||
            engine.position() != 1 || !same_logits(engine.read_logits(), baseline_logits)) {
            std::fprintf(stderr, "cancelled MTP stream changed unconsumed resident state\n");
            return 1;
        }
        if (engine.generate_from_pending(pending, 2, 2) != baseline_retry) {
            std::fprintf(stderr, "cancelled MTP stream changed retry behavior\n");
            return 1;
        }
        if (argc == 3) {
            q27::MetalEngine bonsai(argv[2], 8, q27::KvKind::F16);
            if (bonsai.has_mtp())
                throw std::runtime_error("Bonsai artifact unexpectedly exposed an MTP layer");
            if (bonsai.chunked_prefill())
                throw std::runtime_error("Bonsai enabled activation-quantized chunk prefill");
            bool enable_rejected = false;
            try {
                bonsai.set_chunked_prefill(true);
            } catch (const std::runtime_error&) {
                enable_rejected = true;
            }
            if (!enable_rejected)
                throw std::runtime_error("Bonsai accepted activation-quantized chunk prefill");
            (void)bonsai.ingest_prompt({1, 1, 1}, false, true);
            if (bonsai.position() != 3)
                throw std::runtime_error("Bonsai serial prefill did not advance");
            puts("Bonsai engine contracts: OK");
        } else {
            puts("Bonsai engine contracts: SKIP (set BONSAI_MODEL to enable)");
        }

        puts("Metal q4s engine contracts: OK");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
