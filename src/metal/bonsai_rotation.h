#pragma once

#include "../loader.h"
#include "../../third_party/json.hpp"
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace q27 {

// Runtime ABI of upstream's Bonsai 2 packs (tools/repack.py, docs/FORMAT.md
// "Bonsai 2 packs"): "bonsai2": true, the prism.hadamard.* contract under
// "hadamard", and the sign vectors repeated as F32 hadamard_signs.<width>
// tensors. Metal serves the slim containers only (T2 embedding/head); the
// older T2/B1 policies are unrotated and must never parse as rotated.
struct BonsaiRotation {
    static constexpr uint32_t block_size = 1024;
    bool enabled = false;
    bool t3 = false;                 // blk.* matrices are T3_G128 (bonsai2-t3-v1)
    std::set<std::string> weights;
    std::map<uint32_t, std::vector<float>> signs;

    static bool declared(const nlohmann::json& meta) {
        return meta.contains("bonsai2") || meta.contains("hadamard");
    }

    static BonsaiRotation parse(const nlohmann::json& meta,
                                const std::vector<Tensor>& tensors) {
        using nlohmann::json;
        BonsaiRotation result;
        for (auto it = meta.begin(); it != meta.end(); ++it)
            if (it.key().compare(0, 15, "prism.hadamard.") == 0)
                throw std::runtime_error("q27: retired revival Bonsai 2 pack; reconvert with "
                                         "tools/repack.py --bonsai2-container t2 --slim");
        if (!declared(meta)) return result;
        auto fail = [](const std::string& reason) {
            throw std::runtime_error("q27: invalid Bonsai 2 pack: " + reason);
        };
        if (meta.value("bonsai2", json()) != true || !meta.contains("hadamard"))
            fail("bonsai2 flag and hadamard contract must appear together");
        const std::string policy = meta.value("quant_policy", std::string());
        const std::string container = meta.value("bonsai2_container", std::string());
        if (policy == "bonsai2-t2-v1" && container == "t2-slim") result.t3 = false;
        else if (policy == "bonsai2-t3-v1" && container == "t3-slim") result.t3 = true;
        else fail("Metal serves only the slim t2/t3 containers (got " + policy + " / " +
                  container + "); repack with --slim");
        if (meta.contains("bonsai2_mtp")) fail("MTP-extended packs are not supported on Metal");

        const json& h = meta["hadamard"];
        if (!h.is_object()) fail("hadamard must be an object");
        std::set<std::string> fields, expected{
            "version", "block_size", "transform", "axis", "sign_mode", "sign_widths",
            "sign_values", "weight_names", "inverse_weight_names", "gdn_v_grouped"};
        for (auto it = h.begin(); it != h.end(); ++it) fields.insert(it.key());
        if (fields != expected) fail("incomplete or unknown hadamard metadata");
        auto exact_integer = [&](const char* key, uint32_t value) {
            if (!h[key].is_number_integer() || h[key] != value) fail(key);
        };
        exact_integer("version", 1);
        exact_integer("block_size", block_size);
        if (h["transform"] != "normalized-sylvester-walsh-hadamard" ||
            h["axis"] != "input-last-dimension" || h["sign_mode"] != "explicit" ||
            h["gdn_v_grouped"] != true) fail("unsupported transform/sign/GDN convention");
        if (h["inverse_weight_names"] != json::array({"token_embd.weight"}))
            fail("inverse must be token_embd.weight only");
        const auto& widths = h["sign_widths"];
        const auto& values = h["sign_values"];
        if (!widths.is_array() || widths.size() != 3 || !values.is_array()) fail("sign arrays");
        size_t offset = 0;
        for (const auto& item : widths) {
            if (!item.is_number_integer() || (item != 5120 && item != 6144 && item != 17408))
                fail("unsupported sign width");
            const uint32_t width = item.get<uint32_t>();
            if (result.signs.count(width) || offset + width > values.size()) fail("duplicate/truncated signs");
            auto& signs = result.signs[width];
            signs.reserve(width);
            for (uint32_t i = 0; i < width; ++i) {
                const auto& value = values[offset++];
                if (!value.is_number_integer() || (value != -1 && value != 1)) fail("sign must be +/-1");
                signs.push_back(value.get<float>());
            }
        }
        if (offset != values.size()) fail("extra sign values");
        const auto& names = h["weight_names"];
        if (!names.is_array()) fail("weight_names must be an array");
        for (const auto& name : names) {
            if (!name.is_string() || !result.weights.insert(name.get<std::string>()).second)
                fail("duplicate/invalid weight name");
        }

        // The F32 copies are what the CUDA engine reads; a pack whose two
        // sign tables disagree is corrupt whichever one Metal would use.
        std::set<uint32_t> sign_tensors;
        std::set<std::string> projections;
        bool embedding = false;
        for (const auto& tensor : tensors) {
            if (tensor.name.compare(0, 15, "hadamard_signs.") == 0) {
                const auto found = result.signs.find(
                    static_cast<uint32_t>(std::strtoul(tensor.name.c_str() + 15, nullptr, 10)));
                if (found == result.signs.end() || tensor.dtype != DType::F32 ||
                    tensor.shape != std::vector<uint64_t>{found->first} ||
                    tensor.name != "hadamard_signs." + std::to_string(found->first) ||
                    !sign_tensors.insert(found->first).second)
                    fail("unexpected sign tensor " + tensor.name);
                if (tensor.data && (tensor.data_size != found->first * sizeof(float) ||
                    std::memcmp(tensor.data, found->second.data(), tensor.data_size) != 0))
                    fail(tensor.name + " disagrees with hadamard.sign_values");
                continue;
            }
            const bool body = tensor.name.compare(0, 4, "blk.") == 0;
            const DType packed = result.t3 && body ? DType::T3_G128 : DType::T2_G128;
            if (tensor.dtype != DType::T2_G128 && tensor.dtype != DType::T3_G128) continue;
            if (tensor.dtype != packed) fail("container dtype mismatch: " + tensor.name);
            if (tensor.shape.size() != 2 || tensor.shape.back() % block_size ||
                !result.signs.count(static_cast<uint32_t>(tensor.shape.back())))
                fail("unsupported rotated tensor shape: " + tensor.name);
            if (tensor.name == "token_embd.weight") embedding = true;
            else projections.insert(tensor.name);
        }
        if (sign_tensors.size() != result.signs.size()) fail("missing hadamard_signs tensors");
        // The architecture validator also pins each tensor's shape/dtype.
        // Requiring the complete set here prevents a missing transform from
        // turning into plausible but wrong inference.
        if (!embedding || projections.size() != 401 || result.weights != projections)
            fail("weight_names must cover all 401 projections exactly once");
        result.enabled = true;
        return result;
    }
};

} // namespace q27
