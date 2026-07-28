// q27 file inspector: header sanity, per-dtype accounting, size invariants.
#include "loader.h"

#include <cinttypes>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <map>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.q27\n", argv[0]);
        return 1;
    }
    q27::Model m;
    try {
        m = q27::Model::open(argv[1]);
    } catch (const std::exception& e) {
        fprintf(stderr, "INVARIANT FAIL container: %s\n", e.what());
        return 1;
    }

    printf("meta (%zu bytes): %.300s%s\n\n", m.meta_json.size(), m.meta_json.c_str(),
           m.meta_json.size() > 300 ? "..." : "");

    std::map<std::string, std::pair<int, uint64_t>> by_type;
    uint64_t total = 0;
    int bad = 0;
    auto checked_product = [](std::initializer_list<uint64_t> factors, uint64_t& out) {
        out = 1;
        for (uint64_t factor : factors) {
            if (factor && out > std::numeric_limits<uint64_t>::max() / factor)
                return false;
            out *= factor;
        }
        return true;
    };
    for (const auto& t : m.tensors) {
        auto& e = by_type[q27::dtype_name(t.dtype)];
        e.first++;
        e.second += t.data_size + t.scales_size;
        total += t.data_size + t.scales_size;
        if (t.name == "token_embd.weight" && t.dtype != q27::DType::Q8_G128) {
            printf("INVARIANT FAIL token_embd.weight: CUDA row lookup requires Q8_G128\n");
            bad++;
        }

        uint64_t r = t.rows(), c = t.cols();
        uint64_t group = 0;
        if (t.dtype == q27::DType::Q4_G64) group = 64;
        else if (t.dtype == q27::DType::Q8_G128 || t.dtype == q27::DType::T2_G128 ||
                 t.dtype == q27::DType::T3_G128 || t.dtype == q27::DType::B1_G128)
            group = 128;
        if (group && c % group) {
            printf("INVARIANT FAIL %s: cols %" PRIu64 " not divisible by group %" PRIu64 "\n",
                   t.name.c_str(), c, group);
            bad++;
            continue;
        }
        uint64_t want_data = 0, want_scales = 0;
        bool sizes_representable = true;
        switch (t.dtype) {
            case q27::DType::F32:     sizes_representable = checked_product({r, c, 4}, want_data); break;
            case q27::DType::F16:     sizes_representable = checked_product({r, c, 2}, want_data); break;
            case q27::DType::Q8_G128: sizes_representable = checked_product({r, c}, want_data) &&
                checked_product({r, c / 128, 2}, want_scales); break;
            case q27::DType::Q4_G64:  sizes_representable = checked_product({r, c / 2}, want_data) &&
                checked_product({r, c / 64, 2}, want_scales); break;
            case q27::DType::T2_G128: sizes_representable = checked_product({r, c / 4}, want_data) &&
                checked_product({r, c / 128, 2}, want_scales); break;
            case q27::DType::T3_G128: sizes_representable = checked_product({r, c / 128, 26}, want_data) &&
                checked_product({r, c / 128, 2}, want_scales); break;
            case q27::DType::B1_G128: sizes_representable = checked_product({r, c / 8}, want_data) &&
                checked_product({r, c / 128, 2}, want_scales); break;
        }
        if (!sizes_representable) {
            printf("INVARIANT FAIL %s: packed payload size overflows uint64\n", t.name.c_str());
            bad++;
            continue;
        }
        if (t.data_size != want_data || t.scales_size != want_scales) {
            printf("INVARIANT FAIL %s: data %" PRIu64 " (want %" PRIu64 "), scales %" PRIu64
                   " (want %" PRIu64 ")\n",
                   t.name.c_str(), t.data_size, want_data, t.scales_size, want_scales);
            bad++;
        } else if (t.dtype == q27::DType::T2_G128) {
            bool invalid = false;
            for (uint64_t i = 0; i < t.data_size && !invalid; i++) {
                const uint8_t byte = t.data[i];
                for (int shift = 0; shift < 8; shift += 2)
                    if (((byte >> shift) & 3u) == 3u) { invalid = true; break; }
            }
            if (invalid) {
                printf("INVARIANT FAIL %s: T2 payload contains reserved code 3\n",
                       t.name.c_str());
                bad++;
            }
        } else if (t.dtype == q27::DType::T3_G128) {
            bool invalid_byte = false, invalid_padding = false;
            const uint64_t groups = r * (c / 128);
            for (uint64_t g = 0; g < groups; g++) {
                const uint8_t* packed = t.data + g * 26;
                for (int i = 0; i < 26; i++)
                    if (packed[i] > 242) { invalid_byte = true; break; }
                const uint8_t tail = packed[25];
                if ((tail / 27) % 3 != 1 || (tail / 81) % 3 != 1)
                    invalid_padding = true;
            }
            if (invalid_byte) {
                printf("INVARIANT FAIL %s: T3 payload byte exceeds 242\n", t.name.c_str());
                bad++;
            }
            if (invalid_padding) {
                printf("INVARIANT FAIL %s: T3 final-byte padding is noncanonical\n",
                       t.name.c_str());
                bad++;
            }
        }
    }

    printf("%zu tensors, %.2f GB payload\n", m.tensors.size(), total / 1e9);
    for (const auto& [k, v] : by_type)
        printf("  %-8s %4d tensors  %8.2f GB\n", k.c_str(), v.first, v.second / 1e9);

    for (const char* name : {"token_embd.weight", "blk.0.ffn_gate.weight",
                             "blk.3.attn_q.weight", "blk.64.nextn.eh_proj.weight",
                             "output_norm.weight"}) {
        const q27::Tensor* t = m.find(name);
        if (!t) { printf("MISSING: %s\n", name); bad++; continue; }
        printf("  %-32s %-8s [", t->name.c_str(), q27::dtype_name(t->dtype));
        for (size_t i = 0; i < t->shape.size(); i++)
            printf("%s%" PRIu64, i ? ", " : "", t->shape[i]);
        printf("]  first bytes: %02x %02x %02x %02x\n",
               t->data[0], t->data[1], t->data[2], t->data[3]);
    }

    printf("\n%s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
