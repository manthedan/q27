#include "../src/metal/bonsai_rotation.h"
#include "../src/api_common.h"
#include "../src/toolgram.h"
#include <functional>
#include <map>
#include <iostream>

using namespace q27;
using nlohmann::json;

int main() {
    try {
        std::vector<Tensor> tensors;
        auto add = [&](std::string name, uint64_t cols, DType dtype = DType::T2_G128) {
            Tensor t; t.name = std::move(name); t.dtype = dtype;
            t.shape = {128, cols}; tensors.push_back(std::move(t));
        };
        add("token_embd.weight",5120); add("output.weight",5120);
        for (int i=0;i<64;++i) {
            const auto p="blk."+std::to_string(i)+".";
            for (const auto* s:{"ffn_gate.weight","ffn_up.weight"}) add(p+s,5120);
            add(p+"ffn_down.weight",17408);
            if (i%4==3) {
                for (const auto* s:{"attn_q.weight","attn_k.weight","attn_v.weight"}) add(p+s,5120);
                add(p+"attn_output.weight",6144);
            } else {
                add(p+"attn_qkv.weight",5120); add(p+"attn_gate.weight",5120);
                add(p+"ssm_out.weight",6144);
            }
        }
        // hadamard_signs.<width>: F32 copies of the sign table (all +1 here).
        std::map<uint32_t, std::vector<float>> sign_data;
        for (uint32_t w : {5120u, 6144u, 17408u}) {
            sign_data[w].assign(w, 1.0f);
            Tensor t; t.name = "hadamard_signs." + std::to_string(w); t.dtype = DType::F32;
            t.shape = {w}; t.data = reinterpret_cast<const uint8_t*>(sign_data[w].data());
            t.data_size = w * sizeof(float); tensors.push_back(std::move(t));
        }
        json hadamard = {{"version",1}, {"block_size",1024},
                         {"transform","normalized-sylvester-walsh-hadamard"},
                         {"axis","input-last-dimension"}, {"sign_mode","explicit"},
                         {"gdn_v_grouped",true}, {"sign_widths",{5120,6144,17408}},
                         {"inverse_weight_names",{"token_embd.weight"}}};
        hadamard["weight_names"] = json::array();
        for (const auto& t:tensors)
            if(t.name!="token_embd.weight" && t.dtype!=DType::F32) hadamard["weight_names"].push_back(t.name);
        hadamard["sign_values"] = std::vector<int>(5120+6144+17408,1);
        json meta = {{"quant_policy","bonsai2-t2-v1"}, {"bonsai2",true}, {"bonsai2_container","t2-slim"},
                     {"general.name","Bonsai2 Ternary Qwen38 27b"}, {"hadamard",hadamard}};
        if (!BonsaiRotation::parse(meta,tensors).enabled) throw std::runtime_error("valid rotation rejected");
        auto rejects_with = [&](const std::function<void(json&)>& change,
                                const std::function<void(std::vector<Tensor>&)>& edit) {
            auto bad=meta; change(bad);
            auto bad_tensors=tensors; edit(bad_tensors);
            try { (void)BonsaiRotation::parse(bad,bad_tensors); }
            catch (const std::exception&) { return; }
            throw std::runtime_error("invalid rotation accepted");
        };
        auto rejects = [&](const std::function<void(json&)>& change) {
            rejects_with(change, [](std::vector<Tensor>&){});
        };
        auto rejects_tensors = [&](const std::function<void(std::vector<Tensor>&)>& edit) {
            rejects_with([](json&){}, edit);
        };
        for(const auto* key:{"version","block_size","sign_values","weight_names","inverse_weight_names","gdn_v_grouped"})
            rejects([&](json& x){x["hadamard"].erase(key);});
        rejects([](json& x){x["hadamard"]["version"]=2;});
        rejects([](json& x){x["hadamard"]["block_size"]=1024.0;});
        rejects([](json& x){x["hadamard"]["sign_widths"]={5120,5120,17408};});
        rejects([](json& x){x["hadamard"]["sign_values"][0]=0;});
        rejects([](json& x){x["hadamard"]["sign_values"][0]=true;});
        rejects([](json& x){x["hadamard"]["sign_values"].push_back(1);});
        rejects([](json& x){x["hadamard"]["weight_names"].erase(0);});
        rejects([](json& x){x["hadamard"]["weight_names"].push_back("output.weight");});
        rejects([](json& x){x["hadamard"]["inverse_weight_names"]={"output.weight"};});
        rejects([](json& x){x["hadamard"]["gdn_v_grouped"]=false;});
        rejects([](json& x){x["hadamard"]["future"]=1;});
        rejects([](json& x){x.erase("hadamard");});
        rejects([](json& x){x.erase("bonsai2");});
        rejects([](json& x){x["bonsai2"]=1;});
        rejects([](json& x){x["quant_policy"]="bonsai-t2-v1";});
        rejects([](json& x){x["bonsai2_container"]="t2";});          // Q8 vocab: not served on Metal
        rejects([](json& x){x["bonsai2_container"]="t3-slim";});     // policy/container disagree
        rejects([](json& x){x["bonsai2_mtp"]=json::object();});
        rejects([](json& x){x["prism.hadamard.version"]=1;});        // retired revival pack
        rejects_tensors([](std::vector<Tensor>& t){ t.pop_back(); });   // missing hadamard_signs.17408
        rejects_tensors([](std::vector<Tensor>& t){ t.back().shape = {5120}; });
        rejects_tensors([](std::vector<Tensor>& t){ t[2].dtype = DType::T3_G128; }); // T3 in a t2 pack
        std::vector<float> flipped(17408, 1.0f); flipped[7] = -1.0f;
        rejects_tensors([&](std::vector<Tensor>& t){
            t.back().data = reinterpret_cast<const uint8_t*>(flipped.data()); });
        // t3-slim: the body is T3, embedding/head stay T2.
        auto t3_meta = meta; t3_meta["quant_policy"]="bonsai2-t3-v1"; t3_meta["bonsai2_container"]="t3-slim";
        auto t3_tensors = tensors;
        for (auto& t : t3_tensors) if (t.name.compare(0,4,"blk.")==0) t.dtype = DType::T3_G128;
        const auto t3 = BonsaiRotation::parse(t3_meta, t3_tensors);
        if (!t3.enabled || !t3.t3) throw std::runtime_error("valid t3-slim rotation rejected");
        bool t3_mixed_rejected = false;
        t3_tensors[2].dtype = DType::T2_G128;
        try { (void)BonsaiRotation::parse(t3_meta, t3_tensors); }
        catch (const std::exception&) { t3_mixed_rejected = true; }
        if (!t3_mixed_rejected) throw std::runtime_error("T2 body matrix accepted in a t3 pack");
        if(BonsaiRotation::parse(json{{"quant_policy","bonsai-t2-v1"}},{}).enabled)
            throw std::runtime_error("legacy model was rotated");
        set_tool_dialect_for_model(meta.dump());
        if (!tool_dialect_xml_default()) throw std::runtime_error("Bonsai 2 dialect not selected");
        set_tool_dialect_for_model("{\"general.name\":\"Hf\",\"bonsai2\":true}");
        if (!tool_dialect_xml_default()) throw std::runtime_error("bonsai2 flag did not select the 3.8 dialect");
        set_tool_dialect_for_model("{\"general.name\":\"Qwen3.6-27B\"}");
        if (tool_dialect_xml_default()) throw std::runtime_error("legacy dialect changed");
        const json registry = json::array({
            {{"function", {{"name","read"},{"parameters",{{"properties",{{"path",json::object()}}},{"required",{"path"}}}}}}},
            {{"function", {{"name","shell"},{"parameters",{{"properties",{{"command",json::object()}}},{"required",{"command"}}}}}}}});
        const auto schema = tool_grammar_schema_for_names(registry,{"shell","read","hosted"});
        if (schema.properties != std::vector<std::vector<std::string>>{{"command"},{"path"},{}} ||
            schema.required != schema.properties || schema.present != std::vector<bool>{true,true,false} ||
            schema.additional != std::vector<bool>{true,true,true})
            throw std::runtime_error("tool schemas lost name alignment");
        const json edge_registry = json::array({
            {{"function", {{"name","zero"},{"parameters",{{"properties",json::object()},{"additionalProperties",false}}}}}},
            {{"function", {{"name","required_only"},{"parameters",{{"required",{"path"}}}}}}}});
        const auto edge = tool_grammar_schema_for_names(edge_registry,{"zero","required_only","hosted"});
        auto accepts = [&](const std::string& body) {
            ToolGrammarXml g;
            g.reset({"zero","required_only","hosted"},edge.properties,edge.required,edge.present,edge.additional);
            return g.advance_str(body) && g.closed();
        };
        if (!accepts("<function=zero></function></tool_call>") ||
            accepts("<function=zero><parameter=x>x</parameter></function></tool_call>") ||
            accepts("<function=required_only></function></tool_call>") ||
            !accepts("<function=required_only><parameter=path>x</parameter></function></tool_call>") ||
            accepts("<function=required_only><parameter=extra>x</parameter></function></tool_call>") ||
            !accepts("<function=required_only><parameter=extra>x</parameter><parameter=path>x</parameter></function></tool_call>") ||
            accepts("<function=required_only><parameter=path>x</parameter><parameter=path>y</parameter></function></tool_call>") ||
            !accepts("<function=hosted><parameter=x>x</parameter></function></tool_call>"))
            throw std::runtime_error("empty/absent/required-only XML schemas conflated");
        for (const json& open : std::vector<json>{
                {{"type","object"}}, {{"properties",json::object()}},
                {{"properties",{{"known",json::object()}}},{"additionalProperties",true}},
                {{"additionalProperties",{{"type","string"}}}}}) {
            const auto schema = tool_grammar_schema_for_names(json::array({
                {{"function",{{"name","open"},{"parameters",open}}}}}),{"open"});
            ToolGrammarXml g;
            g.reset({"open"},schema.properties,schema.required,schema.present,schema.additional);
            if (!g.advance_str("<function=open><parameter=extra>x</parameter></function></tool_call>") || !g.closed())
                throw std::runtime_error("valid open object schema blocked an additional key");
        }
        ToolGrammarXml legacy_open;
        legacy_open.reset({"open"},{{}});
        if (!legacy_open.advance_str("<function=open><parameter=extra>x</parameter></function></tool_call>"))
            throw std::runtime_error("legacy list-only empty fallback changed");
        auto invalid = edge_registry;
        invalid[1]["function"]["parameters"]["additionalProperties"] = false;
        bool rejected = false;
        try { (void)tool_grammar_schema_for_names(invalid,{"required_only"}); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("contradictory closed schema accepted");
        ToolGrammar grammar;
        grammar.reset({"write"});
        if (!grammar.tool_name().empty() ||
            !grammar.advance_str("{\"name\":\"write\",\"arguments\":{\"path\":\"x\"}}</tool_call>") ||
            !grammar.closed() || grammar.tool_name() != "write")
            throw std::runtime_error("JSON body-tool name contract failed");
        ToolGrammarXml xml;
        xml.reset({"write"}, {{"path"}}, {{"path"}});
        if (!xml.tool_name().empty() ||
            !xml.advance_str("<function=write><parameter=path>x</parameter></function></tool_call>") ||
            !xml.closed() || xml.tool_name() != "write")
            throw std::runtime_error("XML body-tool name contract failed");
        std::cout << "Bonsai 2 metadata, chat-profile and native grammar contracts: PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
