// Every cached XML mask must equal direct grammar simulation, including
// vocabulary tokens spanning state transitions and consecutive tool/schema sets.
#include "toolgram.h"
#include "../experiments/ds4-agent/q27_agent_mask_epoch.h"
#include <iostream>
#include <set>
#include <stdexcept>

int main() {
    try {
        const std::vector<std::string> calls={
            "<function=read><parameter=path>x</parameter></function></tool_call>",
            "<function=shell><parameter=command>x</parameter></function></tool_call>",
            "<function=read><parameter=command>x</parameter></function></tool_call>",
            "<function=both><parameter=path>x</parameter><parameter=command>y</parameter></function></tool_call>"};
        std::set<std::string> pieces={"=path","=command",">", "</function>"};
        for(const auto& call:calls) {
            pieces.insert(call);
            for(size_t offset=0;offset<call.size();++offset)
                for(size_t n:{1u,2u,4u,8u,16u,32u}) pieces.insert(call.substr(offset,n));
        }
        pieces.erase("</tool_call>");
        std::vector<std::string> vocab={"</tool_call>"};
        vocab.insert(vocab.end(),pieces.begin(),pieces.end());
        q27::ToolMaskCache<q27::ToolGrammarXml> cache;
        cache.init(&vocab,0);
        size_t checked=0;
        for(bool reverse:{false,true}) for(bool required:{false,true}) for(bool alternate:{false,true}) {
            for(size_t index=0;index<calls.size();++index) {
                const auto& call=calls[reverse?calls.size()-1-index:index];
                const std::vector<std::vector<std::string>> props={
                    {alternate?"command":"path"},{"command"},{"path","command"}};
                const auto req=required?props:std::vector<std::vector<std::string>>(3);
                q27::ToolGrammarXml grammar;
                grammar.reset({"read","shell","both"},props,req);
                for(size_t pos=0;;++pos) {
                    const auto& mask=cache.mask(cache.get(grammar));
                    for(size_t id=0;id<vocab.size();++id) {
                        const bool expected=id==0?grammar.done():grammar.token_ok(vocab[id]);
                        const bool actual=(mask[id/32]>>(id%32))&1u;
                        if(actual!=expected)
                            throw std::runtime_error("XML cache disagrees at prefix ["+call.substr(0,pos)+"] token ["+vocab[id]+"]");
                    }
                    ++checked;
                    if(pos==call.size() || !grammar.advance(call[pos])) break;
                }
            }
        }
        // Identical empty key vectors must not alias absent-schema (hosted)
        // fallback and an explicit zero-argument schema, including bare-entry
        // tokens spanning the entire function body.
        const std::vector<std::string> edge_vocab = {"</tool_call>",
            "<function=zero></function></tool_call>",
            "<function=zero><parameter=x>x</parameter></function></tool_call>"};
        q27::ToolMaskCache<q27::ToolGrammarXml> edge_cache;
        edge_cache.init(&edge_vocab, 0);
        for (bool present : {false,true,false,true}) {
            q27::ToolGrammarXml g;
            g.reset({"zero"},{{}},{},{present});
            const auto& mask = edge_cache.mask(edge_cache.get(g));
            if (!((mask[0] >> 1) & 1u) || bool((mask[0] >> 2) & 1u) == present)
                throw std::runtime_error("XML cache conflated empty and absent schema");
        }
        for (bool additional : {false,true,false,true}) {
            q27::ToolGrammarXml g;
            g.reset({"zero"},{{}},{},{true},{additional});
            const auto& mask = edge_cache.mask(edge_cache.get(g));
            if (bool((mask[0] >> 2) & 1u) != additional)
                throw std::runtime_error("XML cache conflated open and closed empty schema");
        }
        // The native engine's real pool has 64 slots. Repeated requests must
        // reuse that capacity while retaining CPU cache entries, never carry
        // either dialect's stale device indices into a new pool epoch.
        struct Device {
            int used=64, resets=0;
            void reset_mask_pool(){used=0;++resets;}
            int add(){return used<64?used++:-1;}
        } device;
        const auto cpu_cache_size=cache.size();
        std::vector<int> json_slots{17},xml_slots{29};
        for(int generation=0;generation<128;++generation) {
            q27::native_agent::begin_mask_epoch(device,json_slots,xml_slots);
            if(device.used || !json_slots.empty() || !xml_slots.empty())
                throw std::runtime_error("native mask epoch retained stale device state");
            for(int i=0;i<64;++i) {
                const int slot=device.add();
                if(slot!=i) throw std::runtime_error("native device pool exhausted across requests");
                (i%2?json_slots:xml_slots).push_back(slot);
            }
        }
        if(device.resets!=128 || cache.size()!=cpu_cache_size)
            throw std::runtime_error("mask epoch reset changed CPU cache lifetime");
        std::cout<<"XML mask cache: "<<checked<<" prefix/schema states; 128 bounded device-pool epochs PASS\n";
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
