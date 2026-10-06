#include "loader.h"

#include <cstdio>

#include <stdexcept>

int main() {
    using q27::DType;

    // T2_G128 (Bonsai 2) and T3_G128 (Bonsai 2 8 GB packs, 2026-09-20) are
    // CUDA dtypes; FP4_G16 uploads as the pf4 prefill sidecar.
    for (DType dtype : {DType::F32, DType::F16, DType::Q8_G128, DType::Q4_G64, DType::T2_G128,
                        DType::T3_G128, DType::FP4_G16}) {
        if (!q27::cuda_weight_dtype_supported(dtype)) {
            std::fprintf(stderr, "CUDA-compatible dtype rejected: %s\n",
                         q27::dtype_name(dtype));
            return 1;
        }
    }
    for (DType dtype : {DType::B1_G128}) {
        if (q27::cuda_weight_dtype_supported(dtype)) {
            std::fprintf(stderr, "CUDA-unsupported packed dtype accepted: %s\n",
                         q27::dtype_name(dtype));
            return 1;
        }
    }
    q27::Tensor selective;
    selective.name = "blk.0.ffn_gate.weight";
    selective.dtype = DType::Q4_G64;
    q27::validate_cuda_tensor(selective);


    q27::Model model;
    q27::Tensor embedding;
    embedding.name = "token_embd.weight";
    embedding.dtype = DType::Q8_G128;
    model.index.emplace(embedding.name, model.tensors.size());
    model.tensors.push_back(embedding);
    q27::validate_cuda_model(model);

    q27::Tensor packed;
    packed.name = "blk.0.ffn_gate.weight";
    packed.dtype = DType::B1_G128;
    model.index.emplace(packed.name, model.tensors.size());
    model.tensors.push_back(packed);
    bool packed_rejected = false;
    try {
        q27::validate_cuda_model(model);
    } catch (const std::runtime_error& error) {
        packed_rejected = std::string(error.what()).find(
            "unsupported weight dtype B1_G128") != std::string::npos;
    }
    if (!packed_rejected) {
        std::fputs("CUDA-unsupported packed model was accepted\n", stderr);
        return 1;
    }
    bool selective_packed_rejected = false;
    try {
        q27::validate_cuda_tensor(packed);
    } catch (const std::runtime_error& error) {
        selective_packed_rejected = std::string(error.what()).find(
            "unsupported weight dtype B1_G128") != std::string::npos;
    }
    if (!selective_packed_rejected) {
        std::fputs("CUDA-unsupported packed tensor was accepted\n", stderr);
        return 1;
    }


    model.tensors[0].dtype = DType::Q4_G64; // T2 became legal for the embedding (slim packs)
    bool embedding_rejected = false;
    try {
        q27::validate_cuda_model(model);
    } catch (const std::runtime_error& error) {
        embedding_rejected = std::string(error.what()).find(
            "token_embd.weight must be Q8_G128 or T2_G128") != std::string::npos;
    }
    if (!embedding_rejected) {
        std::fputs("non-Q8 CUDA embedding was accepted\n", stderr);
        return 1;
    }

    uint8_t data[128] = {};
    uint8_t scales[2] = {};
    q27::Tensor tensor;
    tensor.name = "test.q8";
    tensor.dtype = DType::Q8_G128;
    tensor.shape = {1, 128};
    tensor.data = data;
    tensor.data_size = sizeof(data);
    tensor.scales_size = sizeof(scales);
    if (q27::validate_tensor_payload(tensor) != "scale payload is missing") {
        std::fputs("missing quantization scales were accepted\n", stderr);
        return 1;
    }
    tensor.data = nullptr;
    tensor.scales = scales;
    if (q27::validate_tensor_payload(tensor) != "data payload is missing") {
        std::fputs("missing tensor data were accepted\n", stderr);
        return 1;
    }

    // T2 reserved-code scan (64-bit word path): codes 0..2 in every field
    // pattern pass, including 2-then-1 neighbours whose bits are adjacent;
    // a single code 3 at any byte/field is rejected.
    {
        uint8_t t2[64];
        uint8_t t2_scales[4] = {};
        q27::Tensor t2_tensor;
        t2_tensor.name = "test.t2";
        t2_tensor.dtype = DType::T2_G128;
        t2_tensor.shape = {2, 128};
        t2_tensor.data = t2;
        t2_tensor.scales = t2_scales;
        t2_tensor.data_size = sizeof(t2);
        t2_tensor.scales_size = sizeof(t2_scales);
        for (size_t i = 0; i < sizeof(t2); i++) {
            const unsigned codes[4] = {unsigned(i % 3), unsigned((i / 3) % 3), 2u, 1u};
            t2[i] = uint8_t(codes[0] | codes[1] << 2 | codes[2] << 4 | codes[3] << 6);
        }
        if (!q27::validate_tensor_payload(t2_tensor).empty()) {
            std::fputs("valid T2 payload rejected\n", stderr);
            return 1;
        }
        for (size_t i = 0; i < sizeof(t2); i++) {
            for (int field = 0; field < 4; field++) {
                const uint8_t saved = t2[i];
                t2[i] = uint8_t(t2[i] | (3u << (2 * field)));
                if (q27::validate_tensor_payload(t2_tensor) != "T2 payload contains reserved code 3") {
                    std::fprintf(stderr, "T2 code 3 at byte %zu field %d accepted\n", i, field);
                    return 1;
                }
                t2[i] = saved;
            }
        }
    }

    std::puts("loader CUDA dtype contracts: PASS");
    return 0;
}
