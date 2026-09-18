#pragma once
#include <vector>

namespace q27::native_agent {
// Device slots are a bounded request-local resource. CPU mask contents may
// survive, but their old device indices must not survive a pool reset.
template<class Engine>
void begin_mask_epoch(Engine& engine, std::vector<int>& json_slots,
                       std::vector<int>& xml_slots) {
    engine.reset_mask_pool();
    json_slots.clear();
    xml_slots.clear();
}
} // namespace q27::native_agent
