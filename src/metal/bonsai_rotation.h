#pragma once

#include "../loader.h"
#include "../../third_party/json.hpp"
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace q27 {

// Runtime ABI of the released Bonsai 2 PQ2_0 pack. The older T2/B1 policies
// are unrotated: recognizing their dtype must never discard this metadata.
struct BonsaiRotation {
    static constexpr const char* policy = "bonsai2-t2-hadamard-v1";
    static constexpr uint32_t block_size = 1024;
    bool enabled = false;
    std::set<std::string> weights;
    std::map<uint32_t, std::vector<float>> signs;

    static BonsaiRotation parse(const nlohmann::json& meta,
                                const std::vector<Tensor>& tensors) {
        using nlohmann::json;
        BonsaiRotation result;
        const std::string prefix = "prism.hadamard.";
        std::set<std::string> fields;
        for (auto it = meta.begin(); it != meta.end(); ++it)
            if (it.key().compare(0, prefix.size(), prefix) == 0) fields.insert(it.key());
        if (meta.value("quant_policy", std::string()) != policy) {
            if (!fields.empty()) throw std::runtime_error("q27: rotation metadata requires Bonsai 2 policy");
            return result;
        }
        auto fail = [](const std::string& reason) {
            throw std::runtime_error("q27: invalid Bonsai 2 rotation: " + reason);
        };
        std::set<std::string> expected;
        for (const char* key : {"version", "block_size", "transform", "axis", "sign_mode",
                                "sign_widths", "sign_values", "weight_names",
                                "inverse_weight_names", "gdn_v_grouped"})
            expected.insert(prefix + key);
        if (fields != expected) fail("incomplete or unknown metadata");
        auto get = [&](const char* key) -> const json& { return meta.at(prefix + key); };
        auto exact_integer = [&](const char* key, uint32_t value) {
            if (!get(key).is_number_integer() || get(key) != value) fail(key);
        };
        exact_integer("version", 1);
        exact_integer("block_size", block_size);
        if (get("transform") != "normalized-sylvester-walsh-hadamard" ||
            get("axis") != "input-last-dimension" || get("sign_mode") != "explicit" ||
            get("gdn_v_grouped") != true) fail("unsupported transform/sign/GDN convention");
        if (meta.value("q27.model_profile", std::string()) != "bonsai2-qwen38-v1")
            fail("missing model profile");
        if (get("inverse_weight_names") != json::array({"token_embd.weight"}))
            fail("inverse must be token_embd.weight only");
        const auto& widths = get("sign_widths");
        const auto& values = get("sign_values");
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
        const auto& names = get("weight_names");
        if (!names.is_array()) fail("weight_names must be an array");
        for (const auto& name : names) {
            if (!name.is_string() || !result.weights.insert(name.get<std::string>()).second)
                fail("duplicate/invalid weight name");
        }
        std::set<std::string> projections;
        bool embedding = false;
        for (const auto& tensor : tensors) {
            if (tensor.dtype != DType::T2_G128) continue;
            if (tensor.shape.size() != 2 || tensor.shape.back() % block_size ||
                !result.signs.count(static_cast<uint32_t>(tensor.shape.back())))
                fail("unsupported rotated tensor shape: " + tensor.name);
            if (tensor.name == "token_embd.weight") embedding = true;
            else projections.insert(tensor.name);
        }
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
