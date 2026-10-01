#include "loader.h"

#include <cstdio>

#include <stdexcept>

int main() {
    using q27::DType;

    for (DType dtype : {DType::F32, DType::F16, DType::Q8_G128, DType::Q4_G64,
                        DType::T2_G128, DType::T3_G128}) {
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

    std::puts("loader CUDA dtype contracts: PASS");
    return 0;
}
