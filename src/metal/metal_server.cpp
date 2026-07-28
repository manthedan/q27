#include "metal_engine.h"
#include "snapshot_evict.h"
#include "disk_snapshot_store.h"
#include "stream_format.h"
#include "../suffixdraft.h"
#include "../tool_preamble.h"
#include "../tokenizer.h"
#include "../toolconstrain.h"
#include <cerrno>
#include "../../third_party/httplib.h"
#include "../../third_party/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <fstream>
#include <fcntl.h>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <sys/acl.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

std::string file_sha1(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot hash file: "+path);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)),{});
    unsigned char digest[CC_SHA1_DIGEST_LENGTH];
    CC_SHA1(bytes.data(),(CC_LONG)bytes.size(),digest);
    static const char hex[]="0123456789abcdef";
    std::string out(40,'0');
    for(int i=0;i<CC_SHA1_DIGEST_LENGTH;i++) {
        out[2*i]=hex[digest[i]>>4]; out[2*i+1]=hex[digest[i]&15];
    }
    return out;
}

void mapping_sha1(const q27::Model& model,unsigned char digest[CC_SHA1_DIGEST_LENGTH]) {
    CC_SHA1_CTX ctx;
    CC_SHA1_Init(&ctx);
    const auto* bytes=static_cast<const unsigned char*>(model.mapping_base());
    const uint64_t size=model.mapping_size();
    for(uint64_t offset=0;offset<size;) {
        const CC_LONG chunk=(CC_LONG)std::min<uint64_t>(256u<<20,size-offset);
        CC_SHA1_Update(&ctx,bytes+offset,chunk);
        offset+=chunk;
    }
    CC_SHA1_Final(digest,&ctx);
}

std::string executable_sha1() {
    uint32_t n=0;
    _NSGetExecutablePath(nullptr,&n);
    std::vector<char> path(n+1);
    if(_NSGetExecutablePath(path.data(),&n)!=0)
        throw std::runtime_error("cannot resolve server executable");
    return file_sha1(path.data());
}

bool path_has_extended_acl(const std::string& path) {
    errno=0;
    acl_t acl=acl_get_file(path.c_str(),ACL_TYPE_EXTENDED);
    if(!acl) {
        if(errno==ENOENT) return false;
        throw std::runtime_error("cannot inspect ACL on path: "+path);
    }
    acl_entry_t entry{};
    errno=0;
    const int result=acl_get_entry(acl,ACL_FIRST_ENTRY,&entry);
    const int saved_errno=errno;
    acl_free(acl);
    if(result==0) return true;
    // Darwin reports EINVAL when an allocated ACL has no first entry.
    if(result<0 && saved_errno==EINVAL) return false;
    throw std::runtime_error("cannot inspect ACL entries on path: "+path);
}

bool path_has_granting_acl(const std::string& path) {
    errno=0;
    acl_t acl=acl_get_file(path.c_str(),ACL_TYPE_EXTENDED);
    if(!acl) {
        if(errno==ENOENT) return false;
        throw std::runtime_error("cannot inspect ancestor ACL: "+path);
    }
    acl_entry_t entry{};
    int entry_id=ACL_FIRST_ENTRY;
    for(;;) {
        errno=0;
        const int result=acl_get_entry(acl,entry_id,&entry);
        if(result<0) {
            const int saved_errno=errno;
            acl_free(acl);
            if(saved_errno==EINVAL) return false; // no entry / end of list
            throw std::runtime_error("cannot inspect ancestor ACL entries: "+path);
        }
        acl_tag_t tag{};
        if(acl_get_tag_type(entry,&tag)!=0) {
            acl_free(acl);
            throw std::runtime_error("cannot inspect ancestor ACL tag: "+path);
        }
        if(tag==ACL_EXTENDED_ALLOW) {
            acl_free(acl);
            return true;
        }
        if(tag!=ACL_EXTENDED_DENY) {
            acl_free(acl);
            throw std::runtime_error("unknown ancestor ACL tag: "+path);
        }
        entry_id=ACL_NEXT_ENTRY;
    }
}

void make_owner_private_no_acl(const std::string& path,mode_t mode,bool directory) {
    struct stat st{};
    if(lstat(path.c_str(),&st)!=0 || st.st_uid!=geteuid() ||
       (directory ? !S_ISDIR(st.st_mode) :
                    (!S_ISREG(st.st_mode) || st.st_nlink!=1)))
        throw std::runtime_error("experimental cache path has unsafe owner/type: "+path);
    acl_t empty=acl_init(0);
    if(!empty) throw std::runtime_error("cannot allocate empty ACL for: "+path);
    const int set_result=acl_set_file(path.c_str(),ACL_TYPE_EXTENDED,empty);
    acl_free(empty);
    if(set_result!=0)
        throw std::runtime_error("cannot remove extended ACL from experimental cache path: "+path);
    if(chmod(path.c_str(),mode)!=0)
        throw std::runtime_error("cannot set private mode on experimental cache path: "+path);
    if(path_has_extended_acl(path))
        throw std::runtime_error("experimental cache path retains an ACL: "+path);
    if(lstat(path.c_str(),&st)!=0 || st.st_uid!=geteuid() ||
       (st.st_mode&0777)!=mode ||
       (directory ? !S_ISDIR(st.st_mode) :
                    (!S_ISREG(st.st_mode) || st.st_nlink!=1)))
        throw std::runtime_error("cannot verify private experimental cache path: "+path);
}

std::string verified_experimental_cache_path(const std::string& input) {
    struct stat leaf{};
    if(lstat(input.c_str(),&leaf)!=0 || !S_ISDIR(leaf.st_mode))
        throw std::runtime_error(
            "experimental cache leaf must be a real directory: "+input);
    std::error_code ec;
    const std::filesystem::path canonical=std::filesystem::canonical(input,ec);
    if(ec) throw std::runtime_error(
        "cannot canonicalize experimental cache path: "+input);
    for(std::filesystem::path current=canonical.parent_path();;) {
        struct stat st{};
        const std::string path=current.string();
        if(path.empty() || lstat(path.c_str(),&st)!=0 || !S_ISDIR(st.st_mode))
            throw std::runtime_error("unsafe experimental cache ancestor: "+path);
        if(st.st_uid!=0 && st.st_uid!=geteuid())
            throw std::runtime_error(
                "experimental cache ancestor has a foreign owner: "+path);
        if(path_has_granting_acl(path))
            throw std::runtime_error(
                "experimental cache ancestor has a granting ACL: "+path);
        if((st.st_mode&0022)!=0 &&
           (!(st.st_mode&S_ISVTX) || (st.st_uid!=0 && st.st_uid!=geteuid())))
            throw std::runtime_error(
                "experimental cache ancestor is replaceable by another principal: "+path);
        const std::filesystem::path parent=current.parent_path();
        if(parent==current) break;
        current=parent;
    }
    return canonical.string();
}

uint32_t parse_u32(const std::string& text, const char* option) {
    if (text.empty() || text[0]=='-') throw std::runtime_error(std::string("invalid ")+option);
    size_t used=0; unsigned long long value=std::stoull(text,&used,10);
    if(used!=text.size() || value>UINT32_MAX) throw std::runtime_error(std::string("invalid ")+option);
    return (uint32_t)value;
}

float parse_float(const std::string& text, const char* option) {
    if (text.empty()) throw std::runtime_error(std::string("invalid ")+option);
    // stof parses "nan"/"inf" successfully -- reject non-finite here so the
    // range checks at the call sites cannot be bypassed.
    size_t used=0; float value=std::stof(text,&used);
    if(used!=text.size() || !std::isfinite(value)) throw std::runtime_error(std::string("invalid ")+option);
    return value;
}

// Served sampling defaults, resolved once in main from
// --temperature-default / --top-p-default / --top-k-default (flag wins; env
// twins Q27_METAL_{TEMPERATURE,TOP_P,TOP_K}_DEFAULT). They apply only when a
// request OMITS the field -- explicit client values always win. With neither
// flag nor env they hold the shipped greedy values (0 / 1 / 0), so the
// greedy path stays bitwise. temp>0 keeps MTP when --mtp is set (sampled
// rejection-accept path); suffix still disengages (greedy-only). Tool
// constraining stays greedy-only. Trace logs effective sampling per request.
float sampling_default_temperature=0.0f;
float sampling_default_top_p=1.0f;
uint32_t sampling_default_top_k=0;

// Client-liveness probe. Deliberately self-contained rather than calling
// httplib::detail::is_socket_alive: that lives in an INTERNAL namespace, and
// depending on it would make the q27 httplib patch a two-part ask (a field
// plus a promise about internals). This way the only thing we need from
// httplib is the borrowed fd on Request -- if upstream ever reshuffles
// detail::, nothing here breaks.
//
// Same semantics as httplib 0.18.3's version, which is also what
// DataSink::is_writable resolves to (SocketStream::is_writable is
// select_write && is_socket_alive), so the streaming and pre-write paths
// agree by construction:
//   nothing readable            -> alive (the normal mid-request state)
//   EBADF                       -> dead
//   readable                    -> alive only if a 1-byte MSG_PEEK returns >0;
//                                  a readable socket with 0 bytes is EOF.
// POSIX only, which is the whole target surface for a Metal server.
//
// Edge cases, both intended: a half-closed peer (shutdown(SHUT_WR), still
// reading) reads as dead -- no real HTTP client does that; and a pipelined
// follow-up request makes MSG_PEEK return >0, i.e. alive, which is correct.
bool socket_alive(socket_t sock) {
    if (sock == INVALID_SOCKET) return true;   // unset fd: never cancel on it
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(sock, &fds);
    timeval tv{0, 0};
    int r;
    do { r = select(static_cast<int>(sock + 1), &fds, nullptr, nullptr, &tv); }
    while (r < 0 && errno == EINTR);
    if (r == 0) return true;                    // not readable -> still open
    if (r < 0) return errno != EBADF;           // EBADF is a closed fd
    char b[1];
    ssize_t n;
    do { n = recv(sock, b, sizeof(b), MSG_PEEK); } while (n < 0 && errno == EINTR);
    return n > 0;
}

std::vector<uint32_t> to_u32(const std::vector<int>& ids) {
    std::vector<uint32_t> result; result.reserve(ids.size());
    for(int id:ids) { if(id<0) throw std::runtime_error("tokenizer returned a negative id"); result.push_back((uint32_t)id); }
    return result;
}

std::string text_content(const json& content) {
    if(content.is_string()) return content.get<std::string>();
    std::string out;
    // jstr, not value(): inside a content ARRAY the established rule is
    // skip-the-malformed-part (the is_object guard above), not reject the whole
    // request. value() throws on a part whose "type"/"text" is present but
    // null or non-string, which would 400 a request the is_object guard was
    // written to tolerate. Upstream 1a15ff8 made the same swap on the CUDA arm.
    if(content.is_array()) for(const auto& part:content) {
        if(!part.is_object()) continue;
        const std::string type=q27::jstr(part,"type");
        if(type=="text" || type=="input_text" || type=="output_text") out+=q27::jstr(part,"text");
    }
    return out;
}

// OpenAI chat messages -> Msg list for chatml_prompt (which merges the tools
// preamble into the system message, replacing the manual merge that lived
// here). Beyond-CUDA: src/server.cu's chat endpoint is text-only by design
// (its structured tool paths are /v1/messages and /v1/responses), but pi.dev
// speaks openai-completions, so this endpoint must round-trip tool traffic —
// assistant.tool_calls arrays and role:"tool" results are reconstructed to
// the model's <tool_call>/<tool_response> markers (agentic-parity round,
// docs/metal/plans/2026-07-17-metal-agentic-parity.md).
std::vector<q27::Msg> openai_msgs(const json& body) {
    std::vector<q27::Msg> msgs;
    if(body.contains("system")) {
        std::string system=text_content(body["system"]);
        if(!system.empty()) msgs.push_back({"system",system});
    }
    if(!body.contains("messages") || !body["messages"].is_array())
        throw std::runtime_error("messages must be an array");
    for(const auto& message:body["messages"]) {
        if(!message.is_object()) continue;
        std::string role=q27::jstr(message,"role");
        if(role=="developer") role="system";
        std::string content=message.contains("content")?text_content(message["content"]):"";
        if(role.empty()) continue;
        if(role=="tool") {
            msgs.push_back({"user",q27::tool_response_text(content)});
            continue;
        }
        if(role=="assistant" && message.contains("tool_calls") && message["tool_calls"].is_array()) {
            for(const auto& c:message["tool_calls"]) {
                if(!c.is_object() || !c.contains("function") || !c["function"].is_object()) continue;
                const json& fn=c["function"];
                // OpenAI carries arguments as a JSON-encoded STRING; tolerate
                // an object too (some clients send it pre-parsed).
                json args=json::object();
                if(fn.contains("arguments")) {
                    if(fn["arguments"].is_string()) {
                        try { args=json::parse(fn["arguments"].get<std::string>()); }
                        catch(...) { args=fn["arguments"]; }
                    } else args=fn["arguments"];
                }
                if(!content.empty() && content.back()!='\n') content+="\n";
                content+=q27::tool_call_text(q27::jstr(fn,"name"),args);
            }
        }
        msgs.push_back({role,content});
    }
    // Merge consecutive same-role messages (a run of tool results becomes one
    // user block), matching the CUDA responses handler's merge.
    std::vector<q27::Msg> merged;
    for(auto& m:msgs) {
        if(!merged.empty() && merged.back().role==m.role) merged.back().content+="\n"+m.content;
        else merged.push_back(std::move(m));
    }
    if(merged.empty()) throw std::runtime_error("messages are empty");
    if(merged[0].role=="system") q27::normalize_cc_billing_header(merged[0].content);
    return merged;
}

struct ResponsesPromptInput {
    json tools = json::array();
    std::set<std::string> custom_names;
    q27::ToolChoice choice;
    std::vector<std::string> tool_names;
    std::set<std::string> allowed_hosted_names;
    std::vector<q27::Msg> messages;
};

bool responses_tool_allowed(const std::string& name,
                            const std::set<std::string>& allowed_registered,
                            const std::set<std::string>& allowed_hosted) {
    return allowed_registered.count(name) || allowed_hosted.count(name);
}

void add_responses_hosted_call_names(std::set<std::string>& names,
                                     const std::string& hosted_type) {
    if(hosted_type!="shell") return;
    // Codex exposes its hosted shell capability to the Responses server as
    // type `shell`, while the model emits the client-side function names.
    if(hosted_type=="shell") {
        names.insert("exec_command");
        names.insert("write_stdin");
    }
}

// Normalize the Responses request once for both ordinary serving and the
// opt-in experimental prefix prewarmer. Keeping one canonicalizer is the
// central safety property: a prewarmed token prefix must be byte-for-byte
// the same prompt the serving path would ingest.
ResponsesPromptInput responses_prompt_input(const json& body) {
    ResponsesPromptInput out;
    std::set<std::string> hosted_names;
    std::set<std::string> function_names;
    if(body.contains("tools") && body["tools"].is_array())
        for(const auto& t:body["tools"]) {
            if(!t.is_object()) continue;
            const std::string ty=q27::jstr(t,"type");
            if(t.contains("function") && t["function"].is_object()) {
                const std::string name=q27::jstr(t["function"],"name");
                if(!name.empty()) {
                    function_names.insert(name);
                    out.tools.push_back(t);
                }
            } else if(ty=="function") {
                const std::string name=q27::jstr(t,"name");
                if(!name.empty()) {
                    function_names.insert(name);
                    out.tools.push_back({{"type","function"},
                        {"function",{{"name",name},
                                     {"description",q27::jstr(t,"description")},
                                     {"parameters",t.contains("parameters")?t["parameters"]
                                                                           :json::object()}}}});
                }
            } else if(ty=="custom") {
                const std::string name=q27::jstr(t,"name");
                if(!name.empty()) {
                    out.custom_names.insert(name);
                    out.tools.push_back({{"type","function"},
                        {"function",{{"name",name},
                                     {"description",q27::jstr(t,"description")},
                                     {"parameters",{{"type","object"},
                                         {"properties",{{"input",{{"type","string"},
                                             {"description","The complete raw input text for this tool."}}}}},
                                         {"required",json::array({"input"})}}}}}});
                }
            } else if(!ty.empty()) hosted_names.insert(ty);
        }
    std::set<std::string> hosted_call_names;
    for(const auto& type:hosted_names)
        add_responses_hosted_call_names(hosted_call_names,type);
    for(const auto& name:function_names)
        if(out.custom_names.count(name) || hosted_names.count(name) ||
           hosted_call_names.count(name))
            throw std::runtime_error("ambiguous duplicate Responses tool name");
    for(const auto& name:out.custom_names)
        if(hosted_names.count(name) || hosted_call_names.count(name))
            throw std::runtime_error("ambiguous duplicate Responses tool name");

    auto validate_choice_tool=[&](const json& tool){
        if(!tool.is_object()) return;
        const std::string type=q27::jstr(tool,"type");
        if(type=="function" || type=="custom") {
            const std::string name=q27::jstr(tool,"name");
            if(name.empty()) return;
            const bool declared=type=="function" ? function_names.count(name)
                                                   : out.custom_names.count(name);
            if(!declared) throw std::runtime_error("tool_choice kind/name not present in tools");
        } else if(!type.empty() && type!="allowed_tools" && type!="mcp" &&
                  !hosted_names.count(type)) {
            throw std::runtime_error("tool_choice hosted type not present in tools");
        }
    };
    if(body.contains("tool_choice") && body["tool_choice"].is_object()) {
        const json& source=body["tool_choice"];
        if(q27::jstr(source,"type")=="allowed_tools") {
            const json* allowed=&source;
            if(source.contains("allowed_tools") && source["allowed_tools"].is_object())
                allowed=&source["allowed_tools"];
            if(allowed->contains("tools") && (*allowed)["tools"].is_array())
                for(const auto& tool:(*allowed)["tools"]) validate_choice_tool(tool);
        } else validate_choice_tool(source);
    }
    json selection_body=body;
    selection_body["tools"]=out.tools;
    out.choice=q27::parse_responses_tool_choice(selection_body);
    if(out.choice.invalid) throw std::runtime_error("invalid object-form tool_choice");

    std::set<std::string> registered_names;
    for(const auto& tool:out.tools)
        registered_names.insert(tool["function"]["name"].get<std::string>());
    std::set<std::string> declared_names=registered_names;
    declared_names.insert(hosted_names.begin(),hosted_names.end());
    if(!out.choice.forced_name.empty() && !declared_names.count(out.choice.forced_name))
        throw std::runtime_error("tool_choice names a tool not present in tools");
    for(const auto& name:out.choice.allowed_names)
        if(!declared_names.count(name))
            throw std::runtime_error("tool_choice names a tool not present in tools");
    if(out.choice.mode==q27::ToolChoice::FORCED && declared_names.empty())
        throw std::runtime_error("tool_choice requires at least one valid tool");

    q27::ToolChoice registered_choice=out.choice;
    auto allow_hosted_type=[&](const std::string& type){
        add_responses_hosted_call_names(out.allowed_hosted_names,type);
    };
    if(out.choice.mode!=q27::ToolChoice::NONE) {
        if(!out.choice.forced_name.empty()) {
            if(hosted_names.count(out.choice.forced_name))
                allow_hosted_type(out.choice.forced_name);
        } else if(!out.choice.allowed_names.empty()) {
            for(const auto& name:out.choice.allowed_names)
                if(hosted_names.count(name)) allow_hosted_type(name);
        } else for(const auto& type:hosted_names) allow_hosted_type(type);
    }
    if(!registered_choice.forced_name.empty() &&
       hosted_names.count(registered_choice.forced_name)) {
        registered_choice.mode=q27::ToolChoice::NONE;
        registered_choice.forced_name.clear();
        registered_choice.allowed_names.clear();
    } else if(!registered_choice.allowed_names.empty()) {
        std::vector<std::string> allowed_registered;
        for(const auto& name:registered_choice.allowed_names)
            if(registered_names.count(name)) allowed_registered.push_back(name);
        registered_choice.allowed_names=std::move(allowed_registered);
        if(registered_choice.allowed_names.empty())
            registered_choice.mode=q27::ToolChoice::NONE;
    } else if(registered_names.empty() && !out.allowed_hosted_names.empty()) {
        registered_choice.mode=q27::ToolChoice::NONE;
    }
    q27::OpenAIToolSelection selected=q27::select_openai_tools(selection_body,registered_choice);
    out.tools=std::move(selected.tools);
    out.tool_names=std::move(selected.names);
    const std::set<std::string> eligible(out.tool_names.begin(),out.tool_names.end());
    for(auto it=out.custom_names.begin();it!=out.custom_names.end();)
        if(!eligible.count(*it)) it=out.custom_names.erase(it);
        else ++it;
    if(body.contains("instructions") && body["instructions"].is_string())
        out.messages.push_back({"system",body["instructions"]});
    if(body.contains("input")) {
        if(body["input"].is_string()) out.messages.push_back({"user",body["input"]});
        else if(body["input"].is_array())
            for(const auto& item:body["input"]) {
                if(!item.is_object()) continue;
                const std::string type=q27::jstr(item,"type","message");
                if(type=="message") {
                    std::string role=q27::jstr(item,"role","user");
                    if(role=="developer") role="system";
                    out.messages.push_back({role,item.contains("content")?
                        text_content(item["content"]):""});
                } else if(type=="function_call" || type=="custom_tool_call") {
                    json args;
                    if(type=="function_call") {
                        try { args=json::parse(q27::jstr(item,"arguments","{}")); }
                        catch(...) { args=q27::jstr(item,"arguments"); }
                    } else args={{"input",q27::jstr(item,"input")}};
                    out.messages.push_back({"assistant",
                        q27::tool_call_text(q27::jstr(item,"name"),args)});
                } else if(type=="function_call_output" || type=="custom_tool_call_output") {
                    std::string value;
                    if(item.contains("output"))
                        value=item["output"].is_string()?item["output"].get<std::string>()
                                                        :text_content(item["output"]);
                    out.messages.push_back({"user",q27::tool_response_text(value)});
                }
                // Reasoning items in history are intentionally dropped.
            }
    }
    if(out.messages.empty()) throw std::runtime_error("input is empty");
    std::vector<q27::Msg> merged;
    for(auto& message:out.messages) {
        if(!merged.empty() && merged.back().role==message.role)
            merged.back().content+="\n"+message.content;
        else merged.push_back(std::move(message));
    }
    out.messages=std::move(merged);
    return out;
}

// Unique-ish ids for tool_use/tool_calls blocks: agent clients key results
// to call ids, so a fixed string would collide across turns.
std::atomic<long> req_counter{0};

// Context preflight ceiling: the run()-level throws stay as backstops, but
// agent clients need the refusal BEFORE slot claim / SSE commit, in each
// API's native error shape ("prompt is too long" is Claude Code's
// compact-now signal). Reserve mirrors run()'s generation entry check
// (position + count-1 <= context) plus the speculation lookahead.
uint32_t max_prompt_tokens(uint32_t context,uint32_t mtp_width,uint32_t suffix_width) {
    const uint32_t reserve=1+std::max(mtp_width,suffix_width);
    return context>reserve?context-reserve:1;
}

// Tool names for constrained decoding: OpenAI shape (tools[].function.name)
// and Anthropic shape (tools[].name) both accepted.
std::vector<std::string> tool_names_from_array(const json& tools) {
    std::vector<std::string> names;
    if(!tools.is_array()) return names;
    for(const auto& t:tools) {
        if(!t.is_object()) continue;
        if(t.contains("function") && t["function"].is_object() && t["function"].contains("name"))
            names.push_back(q27::jstr(t["function"],"name"));
        else if(t.contains("name")) names.push_back(q27::jstr(t,"name"));
    }
    names.erase(std::remove(names.begin(),names.end(),std::string()),names.end());
    return names;
}

std::vector<std::string> tool_names_from(const json& body) {
    if(!body.contains("tools")) return {};
    return tool_names_from_array(body["tools"]);
}


// OpenAI `stop` (string or array of strings) / Anthropic `stop_sequences`
// (array). Empty strings are dropped -- they would match at every position.
std::vector<std::string> parse_stops(const json& body, const char* key) {
    std::vector<std::string> out;
    if(!body.contains(key)) return out;
    const json& s=body[key];
    if(s.is_string()) { auto v=s.get<std::string>(); if(!v.empty()) out.push_back(std::move(v)); }
    else if(s.is_array()) for(const auto& e:s)
        if(e.is_string()) { auto v=e.get<std::string>(); if(!v.empty()) out.push_back(std::move(v)); }
    return out;
}

class PrefixCache {
  public:
    explicit PrefixCache(size_t capacity):capacity_(capacity){}

    bool restore(q27::MetalEngine& engine,const std::vector<uint32_t>& prompt,bool mtp,
                 size_t& matched,uint32_t& pending) {
        auto best=entries_.end(); size_t best_len=0;
        for(auto it=entries_.begin();it!=entries_.end();++it) {
            if(it->mtp!=mtp || it->tokens.size()>prompt.size() || it->tokens.size()<=best_len) continue;
            if(std::equal(it->tokens.begin(),it->tokens.end(),prompt.begin())) { best=it; best_len=it->tokens.size(); }
        }
        if(best==entries_.end()) return false;
        engine.restore_state(*best->snapshot); pending=best->pending; matched=best_len;
        entries_.splice(entries_.begin(),entries_,best);
        return true;
    }

    bool prepare_insert(const std::vector<uint32_t>& tokens,bool mtp) {
        if(!capacity_) return false;
        for(auto it=entries_.begin();it!=entries_.end();) {
            if(it->mtp==mtp && it->tokens==tokens) it=entries_.erase(it); else ++it;
        }
        // Release the LRU snapshot before allocating its replacement so peak
        // memory never exceeds the configured number of entries.
        while(entries_.size()>=capacity_) entries_.pop_back();
        return true;
    }

    void insert(std::vector<uint32_t> tokens,bool mtp,uint32_t pending,
                std::shared_ptr<q27::MetalEngine::Snapshot> snapshot) {
        entries_.push_front({std::move(tokens),mtp,pending,std::move(snapshot)});
    }
  private:
    struct Entry { std::vector<uint32_t> tokens; bool mtp; uint32_t pending; std::shared_ptr<q27::MetalEngine::Snapshot> snapshot; };
    size_t capacity_; std::list<Entry> entries_;
};

// Disk snapshot store now lives in disk_snapshot_store.h (extracted so the
// T1 eviction gate drives the REAL store offline; the peek adapter below
// bridges SnapPeekInfo to q27::MetalEngine::SnapshotInfo).
static SnapPeekInfo snap_peek_adapter(const std::string& path) {
    const q27::MetalEngine::SnapshotInfo i=q27::MetalEngine::peek_snapshot(path);
    SnapPeekInfo o; o.position=i.position; o.logits_resident=i.logits_resident; o.tokens=i.tokens;
    return o;
}
// SHA1 token-key hash for DiskSnapshotStore (production); injected so the
// store header stays platform-crypto-free (autoreview P1).
static void snap_hash_sha1(const uint32_t* tokens,uint32_t count,char out_hex[41]) {
    unsigned char sha[20];
    CC_SHA1(tokens,(CC_LONG)(count*4),sha);
    for(int i=0;i<20;i++) snprintf(out_hex+2*i,3,"%02x",sha[i]);
    out_hex[40]='\0';
}

// Whole-session trace stream (triage I2, docs/plans/2026-07-17-ds4-product-
// triage.md): one JSONL stream of the events the parity rounds kept having
// to reconstruct by hand from scattered logs — rendered prompts, snapshot /
// prefix-cache decisions, tool-parser recoveries, cancellations, error
// answers. Diagnostic switch only (ds4 flag rule): off by default, zero
// semantic effect when on. Every line carries wall `ts` plus monotonic
// `tms` (ms since open) so two-slot interleavings reconstruct exactly.
struct TraceLog {
    void open(const std::string& path) {
        // Path only (codex P2): "-"/stderr is not offered — the server's
        // fprintf logging shares stderr, so the JSONL stream would not be
        // clean. /dev/stderr remains available for anyone who wants the mix.
        const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_APPEND|O_CLOEXEC|O_NOFOLLOW,0600);
        if(fd<0) throw std::runtime_error("cannot open --trace path: "+path);
        if(::fchmod(fd,0600)!=0) {
            ::close(fd);
            throw std::runtime_error("cannot secure --trace path: "+path);
        }
        f_=::fdopen(fd,"a");
        if(!f_) {
            ::close(fd);
            throw std::runtime_error("cannot wrap --trace path: "+path);
        }
    }
    bool enabled() const { return f_!=nullptr; }
    bool healthy() const { return f_!=nullptr && healthy_.load(std::memory_order_acquire); }
    // noexcept: a diagnostic stream must never throw through a handler or
    // mask a cancellation (codex P1). json dump/copy/fwrite are contained
    // here; call-site initializer lists are scalar-only, so their
    // construction can only throw on OOM — accepted and recorded.
    void event(json j) noexcept {
        if(!f_) return;
        try {
            j["ts"]=(long)std::time(nullptr);
            std::lock_guard<std::mutex> lk(m_);
            // tms stamped inside the lock: file order == tms order (codex P2).
            j["tms"]=(long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now()-t0_).count();
            const std::string s=j.dump();
            if(fwrite(s.data(),1,s.size(),f_)!=s.size() ||
               fputc('\n',f_)==EOF || fflush(f_)!=0)
                healthy_.store(false,std::memory_order_release);
        } catch(...) { healthy_.store(false,std::memory_order_release); }
    }
  private:
    FILE* f_=nullptr; std::mutex m_;
    std::atomic<bool> healthy_{true};
    const std::chrono::steady_clock::time_point t0_=std::chrono::steady_clock::now();
};

// Trace prompt payloads cap at 64 KB (ctx-limit test prompts run ~1 MB);
// truncation is recorded, never silent.
inline json trace_text(const std::string& s) {
    if(s.size()<=65536) return json(s);
    return json({{"truncated",true},{"bytes",(uint64_t)s.size()},{"head",s.substr(0,65536)}});
}

// Exact token-prefix evidence for recurrence economics. The bounded head is
// a conservative lower bound: matches beyond 8192 are intentionally not
// credited rather than inferred from shared prompt bytes.
inline json trace_token_head(const std::vector<uint32_t>& ids) {
    const size_t n=std::min<size_t>(ids.size(),8192);
    json out=json::array();
    for(size_t i=0;i<n;i++) out.push_back(ids[i]);
    return out;
}

struct Runtime {
    q27::Tokenizer tokenizer;
    std::string model_path;
    std::shared_ptr<q27::MetalEngine::Shared> shared;
    // One request slot = one engine on the shared mapping plus its private
    // prefix cache, scheduling phase, and constraint device-pool map (each
    // engine owns its own mask pool, so host mask id -> pool slot is
    // per-slot state; the host-side mask cache stays shared).
    struct Slot {
        q27::MetalEngine engine;
        PrefixCache cache;
        std::vector<int> host2dev;
        bool busy=false;
        enum class Phase { Idle, Prefill, Decode, Verify } phase=Phase::Idle;
        Slot(std::shared_ptr<q27::MetalEngine::Shared> s,uint32_t ctx,bool turbo3,size_t entries)
            :engine(std::move(s),ctx,turbo3),cache(entries) {}
    };
    std::vector<std::unique_ptr<Slot>> slots;
    uint32_t mtp_width;
    // Suffix-burst decode width (--suffix, 2..VERIFY_CHUNK_MAX; 0 = off).
    // Mutually exclusive with --mtp: the burst path is the no-MTP tiers'
    // speculation lever (2026-07-16-suffix-burst-verify.md, server phase).
    uint32_t suffix_width;
    uint32_t context;
    // Lock order (multislot Phase 1 contract, docs/plans/2026-07-15-
    // multislot-phase1.md): route_ before lease_ — and in fact the two are
    // never held together. route_ guards slot assignment, phases, waiter
    // count, and wait stats, and is never held across GPU work; lease_
    // serializes every engine call (slots alias one Shared command queue)
    // and is held for one scheduling quantum at a time.
    //
    // The lease is a FIFO ticket lock, not a plain mutex: std::mutex makes
    // no fairness promise, and a tight decode loop (release, deliver,
    // reacquire) starves the other slot for a whole generation under it
    // (measured 5.5 s gate wait on the first two-slot run). Ticket order
    // caps the wait at one active quantum, which is the Phase 1 guarantee.
    struct Lease {
        std::mutex m;
        std::condition_variable cv;
        uint64_t next=0, serving=0;
        struct Guard {
            Lease* l=nullptr;
            Guard()=default;
            explicit Guard(Lease& lease):l(&lease) {
                std::unique_lock<std::mutex> lk(l->m);
                const uint64_t ticket=l->next++;
                l->cv.wait(lk,[&]{ return l->serving==ticket; });
            }
            Guard(Guard&& o) noexcept :l(o.l) { o.l=nullptr; }
            Guard& operator=(Guard&&)=delete;
            Guard(const Guard&)=delete;
            Guard& operator=(const Guard&)=delete;
            ~Guard() {
                if(!l) return;
                { std::lock_guard<std::mutex> lk(l->m); l->serving++; }
                l->cv.notify_all();
            }
        };
    };
    std::mutex route_;
    std::condition_variable slot_free_;
    Lease lease_;
    // Serializes poison recovery. recovering_ is guarded by route_ and keeps
    // queued requests from claiming an engine while the shared backend and
    // every attached slot are replaced.
    std::mutex recovery_;
    bool recovering_=false;
    std::string recovery_failure_;
    // Slot admission is ticketed too: a bare condition_variable lets a
    // newly arriving handler barge past an awakened waiter and starve it
    // (codex P2 on d243f92); tickets hand slots out in arrival order, and
    // the ticket spread doubles as the queue bound.
    uint64_t slot_next_=0, slot_serving_=0;
    uint32_t queue_waiters=0;
    static constexpr uint32_t QUEUE_MAX=8;
    // Queue overflow gets its own type so the HTTP layer can answer 503
    // with the documented overloaded_error body (G6) instead of the shared
    // 400 path. Streaming requests that overflow after headers are sent
    // keep the SSE error-event path (status already committed).
    struct ServerOverloaded : std::runtime_error { using std::runtime_error::runtime_error; };
    // Administrative drain closes the preprocessing gap: request activity is
    // counted at wrapper entry, before JSON parse/render/tokenization. A
    // scope that races with drain either remains counted until completion or
    // observes draining and is rejected without entering the handler.
    std::atomic<bool> draining{false};
    std::atomic<uint64_t> active_requests{0};
    struct RequestScope {
        Runtime& rt; bool admitted=false;
        explicit RequestScope(Runtime& r):rt(r) {
            rt.active_requests.fetch_add(1,std::memory_order_seq_cst);
            if(rt.draining.load(std::memory_order_seq_cst))
                rt.active_requests.fetch_sub(1,std::memory_order_seq_cst);
            else admitted=true;
        }
        ~RequestScope() { if(admitted) rt.active_requests.fetch_sub(1,std::memory_order_seq_cst); }
    };
    struct StreamScope {
        Runtime& rt;
        explicit StreamScope(Runtime& r):rt(r) { rt.active_requests.fetch_add(1,std::memory_order_seq_cst); }
        ~StreamScope() { rt.active_requests.fetch_sub(1,std::memory_order_seq_cst); }
    };
    // Thrown when a liveness probe finds the client gone during queue wait
    // or prefill (2026-07-17-abandoned-request-cancellation.md) —
    // deliberately NOT a std::exception: the generic handler catches would
    // otherwise try to write an error body to a dead socket. Handlers and
    // stream providers catch it by name and return without writing.
    struct ClientGone {};
    std::atomic<uint64_t> cancelled_queue{0}, cancelled_prefill{0};
    // Engine failures during generation are server bugs, not request bugs:
    // Anthropic defines api_error (500) for them, and 400 is fatal-class to
    // codex while 500 retries (residue round 2026-07-17-responses-parity-
    // residue.md). Handlers wrap ONLY the run() call — parse/validation/
    // overflow all throw before it. Known coarseness, recorded: the rare
    // post-restore "prompt exceeds context" inside run() rides this class.
    struct EngineError : std::runtime_error { using std::runtime_error::runtime_error; };
    // Mid-queue cancelled tickets awaiting their in-order skip (route_).
    std::set<uint64_t> cancelled_tickets_;
    // Innermost lock: guards the shared host-side ToolMaskCache (mask
    // construction simulates the whole vocabulary on a miss). Order:
    // route_ | lease_ -> mask_mutex_; never the reverse.
    std::mutex mask_mutex_;
    // Wait accounting bucketed by what the competing traffic was doing at
    // arrival (idle/prefill/decode/verify), guarded by route_. Two distinct
    // quantities: queue wait (arrival -> slot admission; bounded only by
    // QUEUE_MAX generations) and gate wait (admission -> first GPU lease;
    // the Phase 1 one-active-quantum guarantee applies to THIS one).
    struct WaitStats { uint64_t n=0; double sum_ms=0, max_ms=0; };
    std::map<std::string,WaitStats> queue_wait_stats, gate_wait_stats;
    // Speculation ground truth for the multislot MTP gates: a quantum round
    // committing >1 token proves accepted drafts, so committed > rounds is
    // the nonzero-speculation assert's unfakeable signal (vacuous-gate rule).
    std::atomic<uint64_t> spec_rounds_total{0}, spec_committed_total{0};
    // Suffix-path round attribution: bursts actually dispatched vs serial
    // fallbacks, so /stats shows whether traffic rides the batched path at
    // all (a suffix server whose every round falls back is misconfigured
    // or serving burst-hostile traffic — either way it should be visible).
    std::atomic<uint64_t> suffix_burst_rounds_total{0}, suffix_fallback_rounds_total{0};
    bool constrain_tools=false;
    // Server thinking profile (upstream v0.4.0 parity): the DEFAULT is
    // no-think — prompts render the closed empty think block so the model
    // answers directly (the qwen36 think-forever pathology makes forced
    // traces a serving hazard; upstream's README documents the same
    // default for speed). --think flips the profile to prefilling an open
    // think tag. Either way, per-request fields override (resolve_think).
    std::vector<std::string> vocab_bytes_v;
    q27::ToolMaskCache mask_cache;
    DiskSnapshotStore snapstore{&snap_peek_adapter,&snap_hash_sha1};
    TraceLog trace;
    std::string model_name,model_sha1_cache,boot_id,server_sha1,tokenizer_name,tokenizer_sha1;
    json serving_identity_cache;
    std::string admin_token;   // separate from boot_id: never served over HTTP (autoreview P2)
    bool snapshot_spine_pin_config=false;
    bool experimental_prefix_cache=false;
    std::string os_sysname,os_release,os_machine;
    bool turbo3_kv=false, test_failpoints=false;
    size_t prefix_entries_config=0;
    uint64_t snapshot_max_bytes_config=0;
    uint32_t max_tokens_default_config=0;
    bool think_default=false;   // --think; see the profile comment above
    std::mutex model_identity_mu;
    // Auto-snapshot threshold in prompt tokens (0 = hint-only); set with
    // the snapshot store, meaningful only when snapstore.enabled().
    size_t snap_auto_min=0;

    // Expensive (~2 s for T2), opt-in through /health?identity=1. This is
    // SHA1 over the resident mmap itself (snapshot identity), so an atomic
    // pathname replacement cannot relabel outputs from the old inode.
    std::string resident_model_sha1() {
        std::shared_ptr<q27::MetalEngine::Shared> resident;
        {
            std::lock_guard<std::mutex> route_lock(route_);
            if(!recovery_failure_.empty()) throw std::runtime_error(recovery_failure_);
            resident=shared;
        }
        std::lock_guard<std::mutex> lock(model_identity_mu);
        if(model_sha1_cache.empty()) {
            unsigned char digest[CC_SHA1_DIGEST_LENGTH];
            mapping_sha1(resident->model,digest);
            static const char hex[]="0123456789abcdef";
            model_sha1_cache.resize(40);
            for(int i=0;i<20;i++) {
                model_sha1_cache[2*i]=hex[digest[i]>>4];
                model_sha1_cache[2*i+1]=hex[digest[i]&15];
            }
        }
        return model_sha1_cache;
    }

    json build_serving_identity() {
        const auto& e=slots.front()->engine;
        std::string cell_masks;
        static const char hex[]="0123456789abcdef";
        for(int i=0;i<16;i++) {
            const uint8_t m=e.kv_fp16_head_masks()[i];
            cell_masks+=hex[m>>4]; cell_masks+=hex[m&15];
        }
        return {{"identity_schema",3},{"server_sha1",server_sha1},
                {"platform",{{"sysname",os_sysname},{"release",os_release},{"machine",os_machine},
                    {"metal_device",shared->backend.name()}}},
                {"shader_abi",q27::MetalBackend::shader_abi_tag()},
                {"shader_sha1",shared->backend.shader_source_sha1()},
                {"protocol",{{"context",context},{"kv",turbo3_kv?"turbo3":"fp16"},
                    {"mtp",mtp_width},{"suffix",suffix_width},{"slots",slots.size()},
                    {"prefix_entries",prefix_entries_config},{"constrain_tools",constrain_tools},
                    {"snapshots",snapstore.enabled()},{"snapshot_auto_min",snap_auto_min},
                    {"experimental_prefix_cache",experimental_prefix_cache},
                    {"snapshot_max_bytes",snapshot_max_bytes_config},
                    {"snapshot_spine_pin",snapshot_spine_pin_config},
                    {"max_tokens_default",max_tokens_default_config},
                    {"think_default",think_default},
                    {"sampling_default",{{"temperature",sampling_default_temperature},
                        {"top_p",sampling_default_top_p},{"top_k",sampling_default_top_k}}},
                    {"kv_fp16_except",e.kv_fp16_except()},{"kv_fp16_cell_masks",cell_masks},
                    {"kv_side_codec",e.kv_fp16_except()?(e.kv_side_codec()?"e4m3":"fp16"):"none"},
                    {"gemm_half",shared->backend.gemm_half_enabled()},
                    {"gemm_half_q4",shared->backend.gemm_half_q4_enabled()},
                    {"gqa_tile",shared->backend.gqa_tile()},{"gqa_block",shared->backend.gqa_block()},
                    {"gqa_threshold",shared->backend.gqa_threshold()},
                    {"gpu_sample",e.gpu_sample_enabled()},{"resident",e.resident_enabled()},
                    {"bare_system",getenv("Q27_BARE")!=nullptr},{"tool_strict",q27::tool_strict()},
                    {"test_failpoints",test_failpoints},
                    {"tokenizer",tokenizer_name},{"tokenizer_sha1",tokenizer_sha1}}}};
    }

    std::string health_status() {
        std::lock_guard<std::mutex> route_lock(route_);
        if(!recovery_failure_.empty()) return "error";
        return recovering_?"recovering":"ok";
    }

    json serving_identity() {
        std::lock_guard<std::mutex> route_lock(route_);
        json out=serving_identity_cache;
        out["backend_state"]=recovery_failure_.empty()?
            (recovering_?"recovering":"ok"):"error";
        return out;
    }

    // Serving knobs promoted to CLI flags (homebrew Phase-2 pre-tag):
    // budget_mb / snapshot_dir / snapshot_max_mb / snapshot_auto carry the
    // parsed flag values, already range-validated in main; the sentinel
    // (0 / empty / 0 / -1) means "flag absent, fall back to the env twin".
    // An explicit flag always wins over the env.
    Runtime(const std::string& model,const std::string& tok,uint32_t ctx,bool turbo3,
            uint32_t width,uint32_t sfx_width,size_t cache_entries,bool constrain,
            uint32_t slot_count,uint32_t budget_mb,const std::string& snapshot_dir,
            uint32_t snapshot_max_mb,long long snapshot_auto,uint32_t max_tokens_default,
            int spine_pin,bool experimental_prefix,bool think_srv)
        :tokenizer(tok),model_path(model),mtp_width(width),suffix_width(sfx_width),context(ctx),
         constrain_tools(constrain),experimental_prefix_cache(experimental_prefix),
         turbo3_kv(turbo3),prefix_entries_config(cache_entries),
         max_tokens_default_config(max_tokens_default),think_default(think_srv) {
        // Server identity (homebrew plan Q2): /health and the boot trace name
        // the resident artifact so wrapper/clients can tell what's loaded.
        model_name=std::filesystem::path(model).filename().string();
        test_failpoints=getenv("Q27_METAL_TEST_FAILPOINTS")!=nullptr;
        tokenizer_name=std::filesystem::path(tok).filename().string();
        struct utsname un{};
        if(::uname(&un)!=0) throw std::runtime_error("cannot read host identity");
        os_sysname=un.sysname; os_release=un.release; os_machine=un.machine;
        server_sha1=executable_sha1();
        tokenizer_sha1=file_sha1(tok);
        {
            std::random_device rd;
            uint64_t x=((uint64_t)rd()<<32)^rd()^
                (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
            char buf[17]; std::snprintf(buf,sizeof buf,"%016llx",(unsigned long long)x);
            boot_id=buf;
        }
        {
            // Admin credential, INDEPENDENT of boot_id (autoreview P2):
            // boot_id namespaces response ids and is intentionally public
            // (served by /health), so it cannot authorize /admin/*. This
            // token is drawn separately and never served over HTTP.
            std::random_device rd;
            uint64_t a=((uint64_t)rd()<<32)^rd()^
                (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
            uint64_t b=((uint64_t)rd()<<32)^rd();
            char buf[33]; std::snprintf(buf,sizeof buf,"%016llx%016llx",
                (unsigned long long)a,(unsigned long long)b);
            admin_token=buf;
            if(const char* at=getenv("Q27_METAL_ADMIN_TOKEN"); at && *at) admin_token=at;
        }
        if(tokenizer.vocab_size()!=q27::MetalEngine::vocabulary_size())
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        shared=q27::MetalEngine::open_shared(model);
        // G6 admission budget (hoisted 2026-07-22 for --ctx auto): the
        // device serving envelope every slot's KV + fixed state + snapshot
        // capacity must fit. Default = half the recommended working set
        // (the engine KV check's convention); --budget-mb flag wins over
        // the env twin. See the admission loop below for the full comment.
        const char* budget_env=getenv("Q27_METAL_BUDGET_MB");
        uint64_t budget=shared->backend.recommended_working_set_size()/2;
        if(budget_mb) budget=(uint64_t)budget_mb*1024ull*1024ull; // --budget-mb, validated at parse
        else if(budget_env) {
            // Fail loud on a malformed override: "-1" through strtoull would
            // wrap to an effectively unlimited budget and bypass the gate.
            char* end=nullptr; errno=0;
            const unsigned long long mb=strtoull(budget_env,&end,10);
            if(errno || end==budget_env || *end || !mb || mb>(1ull<<24))
                throw std::runtime_error("Q27_METAL_BUDGET_MB must be an integer 1..16777216");
            budget=(uint64_t)mb*1024ull*1024ull;
        }
        if(ctx==0) {
            // --ctx auto (upstream v0.4.0 parity; the default): size each
            // slot's window to what the DEVICE can actually serve. The
            // budget is the SMALLER of the admission-policy ceiling
            // (working_set/2 or --budget-mb) and the measured free envelope:
            // recommended - allocated-after-weights + a bounded 2 GB
            // overcommit allowance (macOS pages/compresses gracefully past
            // recommended; every supported legacy config lives inside this)
            // - a 1 GB activation/desktop reserve. Sizing to the raw policy
            // ceiling OOMs the command queue on the 24 GB + official-tier
            // reality: weights (17 GB) and KV share one working set.
            const uint64_t recommended=shared->backend.recommended_working_set_size();
            const uint64_t allocated=shared->backend.current_allocated_size();
            // currentAllocatedSize does NOT price the mapped artifact (the
            // weights ride the file mapping), so charge the artifact bytes
            // explicitly — the OOM at the raw policy ceiling was weights
            // (17 GB) and KV sharing one working set.
            uint64_t artifact_bytes=0;
            { std::error_code ec; artifact_bytes=(uint64_t)std::filesystem::file_size(model,ec); }
            const uint64_t used=artifact_bytes+allocated;
            const uint64_t measured=recommended>used?recommended-used:0;
            const uint64_t envelope=measured+(3ull<<30);
            const uint64_t policy_budget=budget;
            uint64_t kv_budget=std::min(budget,envelope);
            kv_budget=kv_budget>(1ull<<30)?kv_budget-(1ull<<30):0;
            // The admission loop below shares the measured envelope when
            // auto (policy budget stays for explicit --ctx): solver and
            // admission must not disagree about what fits.
            budget=kv_budget;
            // The admission charge (KV reservation + fixed engine state +
            // per-entry snapshot capacity — a snapshot is KV CONTENT, so it
            // scales with ctx too) is exactly linear in ctx; two probe
            // constructions solve slope + intercept without re-deriving the
            // formula. Probes release their reservation at destruction.
            const uint32_t p1=1024,p2=8192;
            uint64_t t1,t2;
            { q27::MetalEngine probe(shared,p1,turbo3);
              t1=probe.kv_reserved_bytes()+(uint64_t)cache_entries*probe.snapshot_bytes()
                +q27::MetalEngine::fixed_state_bytes(probe.chunked_prefill()); }
            { q27::MetalEngine probe(shared,p2,turbo3);
              t2=probe.kv_reserved_bytes()+(uint64_t)cache_entries*probe.snapshot_bytes()
                +q27::MetalEngine::fixed_state_bytes(probe.chunked_prefill()); }
            const double slope=(double)(t2-t1)/(double)(p2-p1);
            const uint64_t intercept=t1-(uint64_t)(slope*p1);
            // Degrade like the admission loop promises: if the requested
            // slot count cannot each hold the floor, solve again for fewer
            // slots — slot 0 always serves (r2 codex P2).
            uint64_t solved=0; uint32_t split=slot_count?slot_count:1;
            for(uint32_t n=split; ; --n) {
                const uint64_t slot_budget=kv_budget/n;
                // Safety margin: slope rounding and block-aligned snapshot
                // sizing must never push the solved charge past budget.
                const uint64_t margin=std::max<uint64_t>(64ull<<20,slot_budget/64);
                const uint64_t usable=slot_budget>margin?slot_budget-margin:0;
                solved=usable>intercept?(uint64_t)((usable-intercept)/slope):0;
                if(solved>=8192 || n==1) { split=n; break; }
            }
            if(solved>262144) solved=262144;   // model-family position cap
            // The legacy default is the empirical floor: every supported
            // pack/tier serves 8192/2-slot within the envelope, so auto
            // never sizes BELOW what pre-auto releases shipped.
            if(solved<8192) {
                fprintf(stderr,"q27 Metal server: --ctx auto: envelope solves only %llu tokens; "
                        "falling back to the legacy 8192 default\n",
                        (unsigned long long)solved);
                solved=8192;
            }
            ctx=(uint32_t)solved;
            context=ctx;
            fprintf(stderr,"q27 Metal server: --ctx auto: %u tokens per slot "
                    "(%.0f charged bytes/token incl snapshot + %.1f MB fixed; "
                    "KV budget %.0f MB [policy %.0f MB, free %.0f MB + 3072 MB overcommit - 1024 MB reserve] "
                    "sized for %u of %u requested slot%s)\n",
                    ctx,slope,intercept/1048576.0,kv_budget/1048576.0,
                    policy_budget/1048576.0,measured/1048576.0,
                    split,slot_count,slot_count==1?"":"s");
        }
        slots.push_back(std::make_unique<Slot>(shared,ctx,turbo3,cache_entries));
        // Snapshot v2 (2026-07-17-kv-except-snapshot-v2.md): exception
        // engines snapshot like any other — side rows ride every surface
        // and snapshot_bytes() prices them, so no capacity override exists
        // anymore. Informational note only.
        if(slots.front()->engine.kv_fp16_except())
            fprintf(stderr,"q27 Metal server: KV exception cells active (Q27_METAL_KV_FP16_CELLS, side codec %s); side caches ride prefix/disk snapshots (v2)\n",
                    slots.front()->engine.kv_side_codec()?"e4m3":"fp16");
        // G6 admission (docs/metal/plans/2026-07-16-g6-admission.md): additional
        // slots must fit the FULL per-slot footprint — KV + this slot's own
        // GQA partials (per-engine since audit E2, charged inside
        // kv_reserved_bytes) + fixed engine state + snapshot capacity x
        // snapshot bytes — against the device budget (--budget-mb flag,
        // env fallback Q27_METAL_BUDGET_MB; default = half the recommended
        // working set, the engine KV check's convention). The engine's own
        // KV check stays underneath as defense in depth; a budget below
        // even one slot still serves one (never zero).
        const q27::MetalEngine& e0=slots[0]->engine;
        const uint64_t per_slot=e0.kv_reserved_bytes()
                               +q27::MetalEngine::fixed_state_bytes(e0.chunked_prefill())
                               +(uint64_t)cache_entries*e0.snapshot_bytes();
        for(uint32_t s=1;s<slot_count;s++) {
            const uint64_t need=(uint64_t)(slots.size()+1)*per_slot;
            if(need>budget) {
                fprintf(stderr,"multislot: slot %u admission rejected: %.0f MB needed "
                        "(%zu+1 slots x %.0f MB/slot incl partials) > %.0f MB budget%s; "
                        "serving with %zu slot(s)\n",
                        s,need/1048576.0,slots.size(),per_slot/1048576.0,
                        budget/1048576.0,
                        budget_mb?" (--budget-mb)":(budget_env?" (Q27_METAL_BUDGET_MB)":""),slots.size());
                break;
            }
            try { slots.push_back(std::make_unique<Slot>(shared,ctx,turbo3,cache_entries)); }
            catch(const std::exception& e) {
                fprintf(stderr,"multislot: slot %u admission failed (%s); serving with %zu slot(s)\n",
                        s,e.what(),slots.size());
                break;
            }
        }
        // Prefix snapshots Phase 2: opt-in via --snapshot-dir (env fallback
        // Q27_METAL_SNAPSHOT_DIR); budget via --snapshot-max-mb /
        // Q27_METAL_SNAPSHOT_MAX_MB (validated, fail-loud, same class as
        // the admission budget; default 8192 MB). The artifact identity
        // hash (~3 s over the 7 GB mapping) is primed HERE, at startup, so
        // the first hinted request never stalls the lease on it.
        std::string sdir=snapshot_dir;
        if(sdir.empty())
            if(const char* senv=getenv("Q27_METAL_SNAPSHOT_DIR"); senv && *senv) sdir=senv;
        if(!sdir.empty()) {
            uint64_t snap_mb=8192;
            if(snapshot_max_mb) snap_mb=snapshot_max_mb; // --snapshot-max-mb, validated at parse
            else if(const char* smax=getenv("Q27_METAL_SNAPSHOT_MAX_MB"); smax && *smax) {
                char* end=nullptr; errno=0;
                const unsigned long long mb=strtoull(smax,&end,10);
                if(errno || end==smax || *end || !mb || mb>(1ull<<24))
                    throw std::runtime_error("Q27_METAL_SNAPSHOT_MAX_MB must be an integer 1..16777216");
                snap_mb=(uint64_t)mb;
            }
            std::error_code ec;
            std::filesystem::create_directories(sdir,ec);
            if(ec || !std::filesystem::is_directory(sdir))
                throw std::runtime_error("snapshot dir (--snapshot-dir / Q27_METAL_SNAPSHOT_DIR) is not a usable directory: "+sdir);
            if(experimental_prefix_cache) {
                sdir=verified_experimental_cache_path(sdir);
                // Captured harness prefixes can include private project
                // instructions. Tighten a newly-created leaf and reject a
                // symlink, foreign owner, or non-private existing leaf.
                // (The ordinary shipped snapshot store keeps its historical
                // deployment semantics; this stricter policy is scoped to
                // the explicitly experimental prompt-capture surface.)
                make_owner_private_no_acl(sdir,0700,true);
                for(const auto& entry:std::filesystem::directory_iterator(sdir))
                    if(entry.path().extension()==".q27snap")
                        make_owner_private_no_acl(entry.path().string(),0600,false);
            }
            const unsigned char* sha=slots[0]->engine.snapshot_identity();
            // Full 160-bit identity in the tag: a truncated prefix could
            // collide across artifacts sharing a directory and let one
            // server overwrite another's snapshots (codex P2 on f05ef2d).
            // Exception engines append their cell config (v2): different
            // cell lists sharing a directory must MISS each other's files
            // (load_state would loudly reject them, costing a fallback per
            // request); env-unset tags are unchanged so pre-v2 snapshot
            // files stay live.
            std::string tag;
            tag.reserve(64);
            char hex[3];
            for(int i=0;i<20;i++) { snprintf(hex,3,"%02x",sha[i]); tag+=hex; }
            tag+=turbo3?'t':'f';
            if(slots[0]->engine.kv_fp16_except()) {
                // 'x' = fp16 sides, 'y' = e4m3 sides: codec is config
                // identity, so the two must miss each other's files.
                tag+=slots[0]->engine.kv_side_codec()?'y':'x';
                const uint8_t* masks=slots[0]->engine.kv_fp16_head_masks();
                for(int i=0;i<16;i++) tag+="0123456789abcdef"[masks[i]&15];
            }
            tag+='-';
            snapshot_max_bytes_config=snap_mb*1024ull*1024ull;
            // T1 spine-pin resolution (flag > env > default on):
            // --snapshot-spine-pin 0/1 wins; else Q27_METAL_SNAPSHOT_SPINE_PIN
            // (fail-loud like the other snapshot knobs); default 1.
            bool spin=true;
            if(spine_pin>=0) spin=(spine_pin!=0);
            else if(const char* sp=getenv("Q27_METAL_SNAPSHOT_SPINE_PIN"); sp && *sp) {
                char* end=nullptr; errno=0;
                const unsigned long long v=strtoull(sp,&end,10);
                if(errno || end==sp || *end || v>1)
                    throw std::runtime_error("Q27_METAL_SNAPSHOT_SPINE_PIN must be 0 or 1");
                spin=(v!=0);
            }
            snapshot_spine_pin_config=spin;
            snapstore.init(sdir,snapshot_max_bytes_config,tag.c_str(),spin);
            // A restart over an oversized directory must come back under
            // budget without waiting for the next save (codex P2 on 607160e).
            snapstore.evict_past_budget();
            if(!slots[0]->engine.chunked_prefill())
                fprintf(stderr,"prefix-snapshots: WARNING — no chunked prefill on this device; "
                        "\"snapshot\" hints are ignored (loads still served)\n");
            // Auto-snapshot threshold (2026-07-17 T2 prefill finding,
            // docs/metal/plans/2026-07-17-t2-prefill-throughput.md): chunked
            // prefill is compute-mature at ~28-40 tok/s on the M4, so a
            // large agentic prompt (pi ~8.4K tokens, CC larger) costs
            // minutes of TTFT — and real agent clients never send the
            // "snapshot" hint. With the snapshot dir already opted in,
            // prompts at/above the threshold behave as hinted; the existing
            // covered-prefix skip and LRU budget bound the write traffic.
            // --snapshot-auto / Q27_METAL_SNAPSHOT_AUTO overrides (tokens;
            // 0 disables auto).
            // The experimental install path only writes through its explicit
            // authenticated prewarm endpoint unless the operator separately
            // opts into production auto-snapshotting.
            snap_auto_min=experimental_prefix_cache?0:4096;
            if(snapshot_auto>=0) snap_auto_min=(size_t)snapshot_auto; // --snapshot-auto, validated at parse
            else if(const char* sauto=getenv("Q27_METAL_SNAPSHOT_AUTO"); sauto && *sauto) {
                char* end=nullptr; errno=0;
                const unsigned long long v=strtoull(sauto,&end,10);
                if(errno || end==sauto || *end || v>(1ull<<24))
                    throw std::runtime_error("Q27_METAL_SNAPSHOT_AUTO must be an integer 0..16777216");
                snap_auto_min=(size_t)v;
            }
            fprintf(stderr,"prefix-snapshots: dir %s, budget %llu MB, auto>=%zu tokens, spine-pin %s, tag %s\n",
                    sdir.c_str(),(unsigned long long)snap_mb,snap_auto_min,spin?"on":"off",tag.c_str());
        }
        if(constrain_tools) {
            vocab_bytes_v=tokenizer.vocab_bytes();
            mask_cache.init(&vocab_bytes_v,tokenizer.token_id("</tool_call>"));
            fprintf(stderr,"constrain-tools: grammar-locked <tool_call> bodies (open=%d close=%d)\n",
                    tokenizer.token_id("<tool_call>"),tokenizer.token_id("</tool_call>"));
        }
        serving_identity_cache=build_serving_identity();
    }

    // A committed Metal failure poisons the shared command queue and every
    // engine attached to it. Stop admissions, let in-flight requests unwind,
    // then replace the mapping/backend and all slots as one generation.
    // Recovery validates the reopened pathname against the resident mapping:
    // an atomic deployment must never switch a live boot to different weights.
    bool recover_backend_if_poisoned() {
        std::unique_lock<std::mutex> recovery_lock(recovery_);
        if(shared->backend.healthy()) return false;

        size_t slot_count=0;
        {
            std::unique_lock<std::mutex> route_lock(route_);
            recovering_=true;
            slot_free_.wait(route_lock,[&]{
                for(const auto& slot:slots) if(slot->busy) return false;
                return true;
            });
            slot_count=slots.size();
        }

        std::shared_ptr<q27::MetalEngine::Shared> replacement_shared;
        std::vector<std::unique_ptr<Slot>> old_slots,replacement_slots;
        try {
            unsigned char resident_sha1[CC_SHA1_DIGEST_LENGTH];
            unsigned char replacement_sha1[CC_SHA1_DIGEST_LENGTH];
            mapping_sha1(shared->model,resident_sha1);
            replacement_shared=q27::MetalEngine::open_shared(model_path);
            mapping_sha1(replacement_shared->model,replacement_sha1);
            if(std::memcmp(resident_sha1,replacement_sha1,sizeof resident_sha1)!=0)
                throw std::runtime_error(
                    "model artifact changed on disk; refusing in-process Metal recovery");
            // One boot must retain both its model and shader identities.
            if(replacement_shared->backend.shader_source_sha1()!=
               shared->backend.shader_source_sha1())
                throw std::runtime_error(
                    "Metal shader source changed on disk; refusing in-process recovery");
            // Reserve before moving the live vector; allocation failure leaves it intact.
            replacement_slots.reserve(slot_count);
        } catch(const std::exception& error) {
            fprintf(stderr,"Metal backend recovery failed before rebuild: %s\n",error.what());
            {
                std::lock_guard<std::mutex> route_lock(route_);
                recovery_failure_="Metal backend recovery failed before rebuild: ";
                recovery_failure_+=error.what();
                recovering_=false;
                draining.store(true,std::memory_order_seq_cst);
            }
            slot_free_.notify_all();
            throw;
        }

        {
            std::lock_guard<std::mutex> route_lock(route_);
            old_slots=std::move(slots);
            shared=replacement_shared;
        }
        // Poisoned engines cannot be a rollback target. Release every old
        // slot and the old shared backend before allocating the replacement
        // generation, preserving the startup admission memory envelope.
        old_slots.clear();
        try {
            for(size_t i=0;i<slot_count;i++) {
                replacement_slots.push_back(std::make_unique<Slot>(
                    replacement_shared,context,turbo3_kv,prefix_entries_config));
            }
        } catch(const std::exception& error) {
            fprintf(stderr,"Metal backend recovery: slot rebuild stopped after %zu/%zu (%s)\n",
                    replacement_slots.size(),slot_count,error.what());
            if(replacement_slots.empty()) {
                {
                    std::lock_guard<std::mutex> route_lock(route_);
                    recovery_failure_="Metal backend recovery could not allocate a serving slot: ";
                    recovery_failure_+=error.what();
                    recovering_=false;
                    draining.store(true,std::memory_order_seq_cst);
                }
                slot_free_.notify_all();
                throw;
            }
            // A partial rebuild is healthy and preferable to taking the
            // process down. Capacity is reduced until the next restart.
        }
        const size_t rebuilt_count=replacement_slots.size();
        {
            std::lock_guard<std::mutex> route_lock(route_);
            shared=std::move(replacement_shared);
            slots=std::move(replacement_slots);
            serving_identity_cache["protocol"]["slots"]=slots.size();
            recovery_failure_.clear();
            recovering_=false;
        }
        slot_free_.notify_all();
        fprintf(stderr,"Metal backend recovery: rebuilt %zu slot%s after command failure\n",
                rebuilt_count,rebuilt_count==1?"":"s");
        trace.event({{"kind","backend_recovery"},{"slots",rebuilt_count}});
        return true;
    }

    template<class Fn>
    decltype(auto) guard_engine(Fn&& fn) {
        try { return std::forward<Fn>(fn)(); }
        catch(const ClientGone&) {
            try { recover_backend_if_poisoned(); } catch(...) {}
            throw;
        }
        catch(const ServerOverloaded&) { throw; }
        catch(const EngineError&) { throw; }
        catch(const std::exception& error) {
            std::string message=error.what();
            try { recover_backend_if_poisoned(); }
            catch(const std::exception& recovery) {
                message+="; Metal backend recovery failed: ";
                message+=recovery.what();
            }
            throw EngineError(message);
        }
    }

    static const char* phase_name(Slot::Phase p) {
        switch(p) {
            case Slot::Phase::Prefill: return "prefill";
            case Slot::Phase::Decode: return "decode";
            case Slot::Phase::Verify: return "verify";
            default: return "idle";
        }
    }

    // Prefill width policy (Phase 1 contract): the width is a runtime
    // policy, not an engine constant. 96 when nothing competes, 48 when the
    // competing traffic is itself prefilling, 12 when a latency-sensitive
    // stream (decode/MTP verify) or a queued request is waiting for the GPU.
    uint32_t quantum_width(const Slot* self) {
        std::lock_guard<std::mutex> lk(route_);
        bool other_busy=false, other_latency=false;
        for(const auto& s:slots) {
            if(s.get()==self || !s->busy) continue;
            other_busy=true;
            if(s->phase!=Slot::Phase::Prefill) other_latency=true;
        }
        if(other_latency || queue_waiters>0) return 12;
        if(other_busy) return 48;
        return 96;
    }

    // How generation ended. Stop == the model emitted EOS (finish_reason
    // "stop" / stop_reason "end_turn"); StopSequence == a requested stop
    // string matched ("stop" / "stop_sequence"); Length == max_tokens hit
    // ("length" / "max_tokens"); Cancelled == the client disconnected.
    enum class Finish { Length, Stop, StopSequence, Cancelled };
    struct Outcome {
        uint32_t prompt_tokens=0, output_tokens=0;
        size_t prefix_hit=0;
        bool exact_disk_hit=false; // durable exact prefix existed before this run
        bool exact_snapshot_written=false;
        Finish finish=Finish::Length;
        std::string stop_sequence; // set when finish==StopSequence
        double queue_wait_ms=0;    // arrival to slot admission
        double gate_wait_ms=0;     // slot admission to first GPU lease
        const char* arrival="idle"; // competing slot's phase at arrival
    };

    // Invalidate path-keyed metadata on every save attempt. save_state can
    // atomically rename the new inode and then throw on directory fsync; in
    // that uncertain-publication case the path may already name new metadata.
    // Erasing is safe before rename too: the next lookup simply re-peeks the
    // still-current old file.
    void save_disk_snapshot(q27::MetalEngine& engine,const std::string& path,
                            const uint32_t* tokens,uint32_t count,
                            bool logits_resident) {
        try {
            engine.save_state(path,tokens,count,logits_resident);
        } catch(...) {
            snapstore.published(path);
            throw;
        }
        snapstore.published(path);
    }

    // Single generation core shared by streaming and non-streaming paths.
    // `emit(piece)` receives UTF-8-safe, stop-sequence-trimmed text as it is
    // produced and returns false when the client has gone away.
    //
    // Multislot Phase 1: a request claims an idle slot (engine + prefix
    // cache), then makes progress one scheduling quantum at a time — one
    // prefill chunk at the policy width, one decode step, or one MTP
    // draft/verify/commit round per GPU lease — so a concurrent request on
    // the other slot waits at most one active quantum, never a whole
    // generation. Text delivery (decode/UTF-8/stop gates/emit) runs outside
    // the lease: a slow client can stall its own stream, not the GPU.
    Outcome run(const std::vector<uint32_t>& prompt,uint32_t count,
                const q27::SamplingParams& sampling,
                const std::vector<std::string>& stops,
                const std::function<bool(const std::string&)>& emit,
                const std::vector<std::string>& tool_names={},
                bool snapshot_hint=false,
                const std::function<bool()>& live={},
                const std::string& trace_id="",
                bool experimental_exact_prefix_save=false) {
        if(prompt.empty()) throw std::runtime_error("prompt is empty");
        q27::validate_sampling(sampling);
        // `mtp` (prefill warm + decode path) is resolved after the slot's
        // engine is bound — it needs has_mtp() so bonsai packs with a stale
        // --mtp flag still plain-sample instead of throwing in mtp_warm.
        const bool sfx=suffix_width!=0 && sampling.temperature==0.0f;
        const bool sample_plain=getenv("Q27_SAMPLE_PLAIN")!=nullptr;
        const auto arrive=std::chrono::steady_clock::now();

        // ---- slot acquisition (route_ only; never held across GPU work) ----
        Slot* slot=nullptr;
        const char* arrival="idle";
        {
            std::unique_lock<std::mutex> lk(route_);
            if(!recovery_failure_.empty()) throw EngineError(recovery_failure_);
            for(const auto& s:slots) if(s->busy) arrival=phase_name(s->phase);
            // A cancelled ticket may sit at the front with no waiter left to
            // skip it: fast-forward before the capacity check (and again in
            // every wait predicate), or stale cancels would falsely 503 new
            // arrivals. Always called under route_.
            auto drain_cancelled=[&]{
                while(cancelled_tickets_.erase(slot_serving_)) slot_serving_++;
            };
            drain_cancelled();
            if(slot_next_-slot_serving_>=QUEUE_MAX)
                throw ServerOverloaded("server overloaded: request queue is full");
            const uint64_t ticket=slot_next_++;
            // Pass the turn on every exit path, or a thrown acquisition
            // would wedge every later ticket. A mid-queue cancel must NOT
            // pass the turn out of order (codex P1 on this round: an
            // unconditional increment from ticket k while ticket k-2 is
            // still serving skips a live waiter and wedges the FIFO) — it
            // registers in cancelled_tickets_ instead and disarms, and
            // whoever holds route_ when serving reaches it skips past.
            struct TurnPass {
                Runtime& rt; bool armed=true;
                ~TurnPass() { if(armed) { rt.slot_serving_++; rt.slot_free_.notify_all(); } }
            } turn{*this};
            queue_waiters++;
            // Timed wait so a dead client's ticket can self-evacuate: the
            // 250 ms tick probes liveness (zero-timeout select + MSG_PEEK,
            // negligible next to any quantum) — the pile-up class from the
            // 2026-07-17 incident (dead requests holding queue positions
            // for minutes, then running to completion for nobody) drains
            // without ever touching the GPU.
            for(;;) {
                const bool admitted_now=slot_free_.wait_for(lk,
                    std::chrono::milliseconds(250),[&]{
                        drain_cancelled();
                        if(recovering_) return false;
                        if(slot_serving_!=ticket) return false;
                        if(!recovery_failure_.empty()) return true;
                        for(const auto& s:slots) if(!s->busy) return true;
                        return false;
                    });
                if(admitted_now) break;
                // live() runs unlocked: the streaming handlers' probe now
                // also emits a wire keepalive (queue wait is the longest
                // silent stretch), and a stalled client's TCP backpressure
                // must not block every other request's slot acquisition
                // through route_. The cancel bookkeeping below relocks first.
                bool gone=false;
                lk.unlock();
                try { gone=live && !live(); }
                catch(...) { lk.lock(); throw; }   // TurnPass unwinds under the lock
                lk.lock();
                drain_cancelled();
                if(gone) {
                    queue_waiters--;
                    cancelled_queue++;
                    trace.event({{"kind","cancel"},{"phase","queue"},{"id",trace_id}});
                    if(slot_serving_==ticket) {
                        // Front of the queue: the normal TurnPass increment
                        // is in order.
                    } else {
                        cancelled_tickets_.insert(ticket);
                        turn.armed=false;
                    }
                    throw ClientGone{};
                }
            }
            if(!recovery_failure_.empty()) {
                queue_waiters--;
                throw EngineError(recovery_failure_);
            }
            queue_waiters--;
            for(const auto& s:slots) if(!s->busy) { slot=s.get(); break; }
            slot->busy=true;
            slot->phase=Slot::Phase::Prefill;
        }
        struct SlotRelease {
            Runtime& rt; Slot& s;
            ~SlotRelease() {
                {
                    std::lock_guard<std::mutex> lk(rt.route_);
                    s.busy=false;
                    s.phase=Slot::Phase::Idle;
                    if(!rt.shared->backend.healthy()) rt.recovering_=true;
                }
                // notify_all, not notify_one: only the serving ticket's
                // waiter can proceed, and notify_one may wake a different
                // ticket that just re-sleeps — wedging the queue while a
                // slot sits idle (codex P1 on cdf85b2).
                rt.slot_free_.notify_all();
            }
        } slot_release{*this,*slot};
        q27::MetalEngine& engine=slot->engine;
        // Warm MTP whenever the artifact has the layer and we will use it:
        // greedy MTP, or sampled MTP (not Q27_SAMPLE_PLAIN force-off).
        const bool mtp=mtp_width!=0 && engine.has_mtp() &&
            !(sampling.temperature>0.0f && sample_plain);

        // First lease acquisition stamps the gate wait; every engine call
        // below runs under one of these scoped leases.
        const auto admitted=std::chrono::steady_clock::now();
        const double queue_wait_ms=
            std::chrono::duration<double,std::milli>(admitted-arrive).count();
        double gate_wait_ms=-1.0;
        auto lease_now=[&]()->Lease::Guard {
            Lease::Guard gpu(lease_);
            if(gate_wait_ms<0)
                gate_wait_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-admitted).count();
            return gpu;
        };

        // ---- prompt ingestion, one quantum per chunk ----
        size_t hit=0; uint32_t pending=0;
        bool restored=false, saved_snapshot=false, exact_snapshot_written=false;
        // Disk lookup runs before taking the lease (pure file I/O); only
        // the resulting load_state goes under it. MTP requests stay on the
        // cold path: lane warming state is not part of the snapshot
        // contract in this phase.
        std::string disk_path; uint32_t disk_len=0; bool disk_loaded=false;
        const bool disk_ok=!mtp && snapstore.enabled() &&
                           snapstore.best_match(prompt,disk_path,disk_len);
        {
            auto gpu=lease_now();
            // Round-2 expert P0 #1 (leak across requests): defensive entry
            // reset; the scope-exit guard below covers every later exit path.
            engine.set_tool_constraint(-1);
            restored=slot->cache.restore(engine,prompt,mtp,hit,pending);
            if(!restored) { engine.reset(); hit=0; }
            // The deeper prefix wins across tiers: a short in-memory entry
            // must not mask a much longer persisted one (codex P2 on
            // 607160e). load_state validates fully before its first GPU
            // write, so a rejected file leaves a memory-restored state
            // intact; only a mid-restore I/O error falls all the way cold.
            if(disk_ok && (disk_len>hit ||
                           (experimental_exact_prefix_save &&
                            disk_len==prompt.size()))) {
                try {
                    engine.load_state(disk_path);
                    disk_loaded=true;
                    hit=disk_len;
                    if(hit==prompt.size()) { pending=engine.pending_from_logits(); restored=true; }
                    snapstore.hits++;
                } catch(const std::exception&) {
                    // Command failures poison the shared backend, not the
                    // snapshot. Let guard_engine rebuild the backend and keep
                    // the valid disk entry for the retry.
                    if(!shared->backend.healthy()) throw;
                    // A shallow-header candidate that fails the engine's full
                    // load is unusable. Remove it so the same token key can
                    // self-heal instead of failing every request or blocking
                    // a later stale/exact publication.
                    snapstore.reject(disk_path);
                    // A bad disk candidate must not cost more than it
                    // offered: fall back to the in-memory tier before going
                    // cold (codex P2 on f05ef2d).
                    engine.reset(); hit=0; pending=0;
                    restored=slot->cache.restore(engine,prompt,mtp,hit,pending);
                    if(!restored) hit=0;
                }
            }
            if((uint64_t)engine.position()+(prompt.size()-hit)>context)
                throw std::runtime_error("prompt exceeds context");
        }
        trace.event({{"kind","prefix"},{"id",trace_id},
                     {"tier",hit==0?"cold":(disk_loaded?"disk":"memory")},
                     {"hit",(uint64_t)hit},{"prompt_tokens",(uint64_t)prompt.size()}});
        // Hinted save target: a stable boundary — trim a 32-token tail
        // (the question-specific suffix) and align down to a 96-token
        // prefill-chunk boundary. Reached exactly by capping one chunk's
        // width; skipped when the restored prefix already covers it.
        //
        // Auto-save (snap_auto_min) rides the same machinery. Two codex P2s
        // on this, recorded: (a) same-path save collisions are lease-
        // serialized — save_state only ever runs under a Lease::Guard, so
        // the worst case is a redundant rewrite of an identical file, not a
        // torn .tmp; (b) a non-repeating large-prompt workload pays one
        // ~1.3 GB write (+~15 s under the lease) per unique prefix — the
        // LRU budget bounds retention, not write churn. This box serves
        // repeated agent prefixes, where the trade wins; churn-sensitive
        // deployments set Q27_METAL_SNAPSHOT_AUTO=0 (hint-only).
        size_t save_at=0;
        const bool snap_wanted=!experimental_exact_prefix_save &&
                               (snapshot_hint ||
                                (snap_auto_min && prompt.size()>=snap_auto_min));
        if(snap_wanted && !mtp && snapstore.enabled() && engine.chunked_prefill() &&
           prompt.size()>32+96) {
            const size_t target=(prompt.size()-32)/96*96;
            if(target>hit) save_at=target;
        }
        std::vector<uint32_t> suffix(prompt.begin()+hit,prompt.end());
        if(!suffix.empty()) {
            if(mtp || !engine.chunked_prefill()) {
                // MTP warming is token-serial inside the engine (each token
                // needs its final hidden state), so this path stays one
                // coarse quantum — a known Phase 1 limitation, documented in
                // the plan; the wait metrics expose it honestly.
                auto gpu=lease_now();
                pending=engine.ingest_prompt(suffix,mtp,false);
            } else {
                size_t i=0;
                const size_t chunkable=suffix.size()-1;
                while(chunkable-i>=2) {
                    // Per-chunk liveness probe: a chunk is seconds of GPU
                    // work, the probe is a zero-timeout select. On death
                    // with an armed snapshot target (the big-prompt cases),
                    // bank the finished chunks at the current position
                    // first — a client-timeout retry then restores from
                    // disk instead of re-paying the whole prefill cold
                    // (the timeout-retry livelock class).
                    if(live && !live()) {
                        // Best-effort: a failed save must not turn a dead-
                        // client cancel into the generic error path
                        // (codex P3 on this round).
                        if(save_at && i>0) try {
                            bool banked=false;
                            {
                                auto gpu=lease_now();
                                const uint32_t bank_count=(uint32_t)(hit+i);
                                if(!snapstore.exact_resident(prompt.data(),bank_count)) {
                                    const std::string path=snapstore.path_for(
                                        prompt.data(),bank_count);
                                    save_disk_snapshot(engine,path,prompt.data(),
                                                       bank_count,false);
                                    banked=true;
                                }
                            }
                            if(banked) {
                                snapstore.saves++;
                                const auto ev=snapstore.evict_past_budget();
                                trace.event({{"kind","snapshot_bank"},{"id",trace_id},
                                             {"len",(uint64_t)(hit+i)},
                                             {"evicted_files",ev.first},{"evicted_bytes",ev.second}});
                            }
                        } catch(const std::exception& e) {
                            fprintf(stderr,"[cancel-save] skipped: %s\n",e.what());
                        }
                        cancelled_prefill++;
                        trace.event({{"kind","cancel"},{"phase","prefill"},{"id",trace_id}});
                        throw ClientGone{};
                    }
                    const uint32_t width=quantum_width(slot);
                    uint32_t take=(uint32_t)std::min<size_t>(width,chunkable-i);
                    if(save_at && hit+i<save_at) {
                        // prefill_chunk takes 2..96 tokens: a one-token gap
                        // to the boundary cannot be reached by capping, so
                        // the (best-effort) save is skipped rather than the
                        // request failing (codex P1 on 607160e).
                        if(save_at-(hit+i)==1) save_at=0;
                        else take=(uint32_t)std::min<size_t>(take,save_at-(hit+i));
                    }
                    auto gpu=lease_now();
                    engine.prefill_chunk(suffix.data()+i,take);
                    i+=take;
                    if(save_at && hit+i==save_at) {
                        // Mid-prefill state is exact but the logits row is
                        // stale — recorded in the file so a same-length
                        // request can never derive a pending token from it.
                        if(!snapstore.exact_resident(
                                prompt.data(),(uint32_t)save_at)) {
                            const std::string path=snapstore.path_for(
                                prompt.data(),(uint32_t)save_at);
                            save_disk_snapshot(engine,path,prompt.data(),
                                               (uint32_t)save_at,false);
                            snapstore.saves++;
                            trace.event({{"kind","snapshot_save"},{"id",trace_id},
                                         {"len",(uint64_t)save_at},
                                         {"mode",snapshot_hint?"hint":"auto"}});
                            saved_snapshot=true;
                        }
                        save_at=0;
                    }
                }
                // Serial tail: at most one leftover chunkable token plus the
                // final token, which produces the logits and pending id —
                // mirrors MetalEngine::prefill()'s tail exactly.
                for(;i<suffix.size();i++) {
                    auto gpu=lease_now();
                    pending=engine.step(suffix[i]);
                }
            }
        }
        // Experimental harness prewarming saves the complete supplied token
        // prefix, not the production hint heuristic's aligned len-32 prefix.
        // The caller supplies a prompt with the live user message removed.
        // An existing exact disk hit is already the requested result and is
        // not rewritten. This flag is reachable only through the separately
        // enabled, admin-authenticated experimental endpoint.
        const bool exact_disk_hit=disk_loaded && disk_len==prompt.size();
        if(experimental_exact_prefix_save && !exact_disk_hit) {
            if(mtp) throw std::runtime_error("experimental prefix prewarm does not support MTP");
            if(!snapstore.enabled())
                throw std::runtime_error("experimental prefix cache store is disabled");
            {
                auto gpu=lease_now();
                const std::string path=snapstore.path_for(
                    prompt.data(),(uint32_t)prompt.size());
                save_disk_snapshot(engine,path,prompt.data(),
                                   (uint32_t)prompt.size(),true);
            }
            snapstore.saves++;
            saved_snapshot=true;
            exact_snapshot_written=true;
            trace.event({{"kind","snapshot_save"},{"id",trace_id},
                         {"len",(uint64_t)prompt.size()},{"mode","experimental_exact"}});
        }
        // LRU enforcement is pure file I/O — outside the lease.
        if(saved_snapshot) {
            const auto ev=snapstore.evict_past_budget();
            if(ev.first) trace.event({{"kind","snapshot_evict"},{"id",trace_id},
                                      {"files",ev.first},{"bytes",ev.second}});
        }
        // Fail oversize generations before emitting anything, exactly like
        // the whole-generation streaming calls used to.
        if((uint64_t)engine.position()+(count?count-1:0)>context)
            throw std::runtime_error("generation exceeds context");
        // Cache the prompt state before generation mutates it. One entry costs
        // about 151 MiB for GDN state, so the default capacity is deliberately 1.
        // Cancelled requests never reach another insert, so post-cancel MTP
        // lane state is structurally non-cacheable (Phase 1 cancel invariant).
        // prepare_insert can release an evicted snapshot's GPU buffers, so it
        // stays under the lease alongside capture_state.
        {
            auto gpu=lease_now();
            if(slot->cache.prepare_insert(prompt,mtp))
                slot->cache.insert(prompt,mtp,pending,engine.capture_state());
        }
        {
            std::lock_guard<std::mutex> lk(route_);
            slot->phase = mtp ? Slot::Phase::Verify : Slot::Phase::Decode;
            WaitStats& qs=queue_wait_stats[arrival];
            qs.n++; qs.sum_ms+=queue_wait_ms; qs.max_ms=std::max(qs.max_ms,queue_wait_ms);
            WaitStats& gs=gate_wait_stats[arrival];
            gs.n++; gs.sum_ms+=std::max(gate_wait_ms,0.0); gs.max_ms=std::max(gs.max_ms,gate_wait_ms);
        }

        // token -> decode -> UTF-8 boundary gate -> stop-sequence holdback ->
        // emit, all outside the GPU lease. deliver() returns false to stop
        // generation, either because the client left (client_gone) or a stop
        // sequence completed (stop_hit); the two are distinguished for
        // finish-reason reporting.
        q27::Utf8Gate ugate;
        q27::StopBuffer stopbuf(stops);
        const uint32_t eos_id=(uint32_t)tokenizer.eos();
        bool client_gone=false, stop_hit=false;
        uint32_t produced=0;
        q27::MetalEngine::StopCause cause=q27::MetalEngine::StopCause::MaxTokens;
        auto deliver=[&](uint32_t token)->bool {
            bool stopped=false;
            std::string safe=stopbuf.feed(ugate.feed(tokenizer.decode_one((int)token)),stopped);
            if(!emit(safe)) { client_gone=true; cause=q27::MetalEngine::StopCause::Cancelled; return false; }
            if(stopped) { stop_hit=true; cause=q27::MetalEngine::StopCause::Cancelled; return false; }
            produced++;
            return true;
        };
        // Constrained tool decoding: trigger detection + grammar feeding on
        // the serial token stream (rounds are single tokens on this path, so
        // the CUDA engage-lag truncation degenerates to plain sequencing: the
        // constraint set here masks the NEXT token's logits inside step()).
        q27::BasicToolConstrainer<q27::MetalEngine,q27::Tokenizer> tc;
        tc.eng=&engine; tc.tok=&tokenizer; tc.cache=&mask_cache; tc.host2dev=&slot->host2dev;
        tc.enabled=constrain_tools && !tool_names.empty() && sampling.temperature==0.0f && !mtp_width && !suffix_width;
        {
            auto gpu=lease_now();
            tc.begin(tool_names);
        }
        // Scope-exit constraint cleanup: runs on normal return, client
        // disconnect, and engine exceptions alike, and never throws (a
        // cleanup failure must not mask the original exception). Takes its
        // own lease — the per-quantum leases are all released by then.
        struct ConstraintCleanup {
            Runtime& rt;
            q27::BasicToolConstrainer<q27::MetalEngine,q27::Tokenizer>& tc;
            q27::MetalEngine& engine;
            ~ConstraintCleanup() {
                try {
                    Lease::Guard gpu(rt.lease_);
                    tc.end();
                    engine.set_tool_constraint(-1);
                } catch(...) {}
            }
        } constraint_cleanup{*this,tc,engine};

        // ---- generation, one quantum per lease ----
        // Sampled MTP (temp>0 + --mtp + has_mtp): rejection-sample accept on
        // greedy drafts. Q27_SAMPLE_PLAIN=1 forces the slow plain sample path
        // for distribution A/B. Bonsai (no MTP layer) falls through to plain.
        if(sampling.temperature>0.0f && mtp_width!=0 && engine.has_mtp() &&
           engine.chunked_prefill() && !sample_plain) {
            std::mt19937_64 rng(sampling.seed);
            {
                // Prefill left a greedy pending; first emitted token is sampled.
                auto gpu=lease_now();
                pending=engine.sample_from_logits(sampling,rng);
            }
            uint32_t live_width=std::min(mtp_width,4u);
            std::vector<uint32_t> committed;
            bool stopped=false;
            while(!stopped && produced<count) {
                if(produced+1==count) {
                    if(pending!=eos_id) deliver(pending);
                    else cause=q27::MetalEngine::StopCause::Eos;
                    break;
                }
                committed.clear();
                {
                    auto gpu=lease_now();
                    pending=engine.mtp_sample_round(pending,count-produced,eos_id,mtp_width,
                                                    live_width,sampling,rng,committed);
                }
                spec_rounds_total.fetch_add(1,std::memory_order_relaxed);
                spec_committed_total.fetch_add(committed.size(),std::memory_order_relaxed);
                for(uint32_t token:committed) {
                    if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; stopped=true; break; }
                    if(!deliver(token)) { stopped=true; break; }
                }
            }
        } else if(sampling.temperature>0.0f) {
            std::mt19937_64 rng(sampling.seed);
            while(produced<count) {
                uint32_t token;
                {
                    auto gpu=lease_now();
                    token=engine.sample_from_logits(sampling,rng);
                }
                if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; break; }
                if(!deliver(token)) break;
                if(produced==count) break;
                auto gpu=lease_now();
                engine.step(token);
            }
        } else if(mtp && engine.chunked_prefill()) {
            uint32_t live_width=std::min(mtp_width,4u);
            std::vector<uint32_t> committed;
            bool stopped=false;
            while(!stopped && produced<count) {
                if(produced+1==count) {
                    if(pending!=eos_id) deliver(pending);
                    else cause=q27::MetalEngine::StopCause::Eos;
                    break;
                }
                committed.clear();
                {
                    auto gpu=lease_now();
                    pending=engine.mtp_round(pending,count-produced,eos_id,mtp_width,
                                             live_width,committed);
                }
                spec_rounds_total.fetch_add(1,std::memory_order_relaxed);
                spec_committed_total.fetch_add(committed.size(),std::memory_order_relaxed);
                for(uint32_t token:committed) {
                    if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; stopped=true; break; }
                    if(!deliver(token)) { stopped=true; break; }
                }
            }
        } else if(sfx && engine.chunked_prefill()) {
            // Suffix-burst decode (2026-07-16-suffix-burst-verify.md, server
            // integration): drafter state is CPU-side, seeded from the FULL
            // prompt — including any restored prefix, which never reached
            // this engine's step loop — then fed every committed token by
            // suffix_step. One suffix_step per lease keeps the MTP branch's
            // quantum discipline; the real eos id rides into the round's
            // lane clamp, so an eos inside a burst commits and stops without
            // encoding past it (gate 5, live EOS).
            q27::SuffixDraft drafter;
            {
                std::vector<int> history(prompt.begin(),prompt.end());
                drafter.reset(history);
            }
            std::vector<uint32_t> committed;
            bool stopped=false;
            while(!stopped && produced<count) {
                if(produced+1==count) {
                    if(pending!=eos_id) deliver(pending);
                    else cause=q27::MetalEngine::StopCause::Eos;
                    break;
                }
                bool burst=false;
                {
                    // The drafter's propose/append inside suffix_step is
                    // host work under the lease — accepted deliberately
                    // (codex P2 on this change): it is bounded integer
                    // compares, microseconds against a multi-ms GPU round,
                    // nothing like the whole-vocab mask simulation that
                    // forced the constrained path's pre-lease prewarm.
                    // Splitting the round across the lease boundary would
                    // move burst policy back out of the engine.
                    auto gpu=lease_now();
                    pending=engine.suffix_step(drafter,pending,count-produced,eos_id,
                                               suffix_width,q27::MetalEngine::SUFFIX_MIN_MATCH,
                                               committed,&burst);
                }
                spec_rounds_total.fetch_add(1,std::memory_order_relaxed);
                spec_committed_total.fetch_add(committed.size(),std::memory_order_relaxed);
                (burst?suffix_burst_rounds_total:suffix_fallback_rounds_total)
                    .fetch_add(1,std::memory_order_relaxed);
                for(uint32_t token:committed) {
                    if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; stopped=true; break; }
                    if(!deliver(token)) { stopped=true; break; }
                }
            }
        } else {
            // Serial greedy walk (also the constrained-decode path): emit the
            // pending token, then each step yields the next. The constraint
            // ops for the token just emitted run under the same lease as the
            // step they mask, exactly as the old in-sink sequencing did.
            uint32_t cur=pending;
            while(produced<count) {
                if(cur==eos_id) { cause=q27::MetalEngine::StopCause::Eos; break; }
                if(!deliver(cur)) break;
                if(produced==count) break;
                if(tc.enabled && tc.active) {
                    // Pre-materialize the advanced state's mask OUTSIDE the
                    // lease: ToolMaskCache::get simulates the whole vocabulary
                    // on a miss (codex P2 on d243f92), which must not extend
                    // the other slot's wait. Same peek-advance the CUDA flow's
                    // on_pending uses; apply() below then hits the cache. The
                    // engage path (scan_round) still builds its entry mask
                    // under the lease — once per tool call, bounded.
                    q27::ToolGrammar peek=tc.tg;
                    bool ok=true;
                    for(char c:tokenizer.decode_one((int)cur))
                        if(!peek.advance(c)) { ok=false; break; }
                    if(ok && !peek.closed()) {
                        std::lock_guard<std::mutex> mk(mask_mutex_);
                        mask_cache.get(peek);
                    }
                }
                auto gpu=lease_now();
                if(tc.enabled) {
                    // mask_mutex_ inside the lease guards the shared host
                    // cache against a concurrent slot's prewarm above.
                    std::lock_guard<std::mutex> mk(mask_mutex_);
                    const int tid=(int)cur;
                    tc.scan_round(&tid,1);
                    tc.on_id(tid);
                    // Restage the ADVANCED grammar state's mask (codex P1):
                    // on_id moves tc.tg but stages nothing, so without this
                    // every step after the first constrained token decodes
                    // under the previous state's legal set.
                    if(tc.active) tc.apply(tc.tg);
                }
                cur=engine.step(cur);
            }
        }
        // Flush the boundary gates: a dangling multi-byte tail becomes U+FFFD,
        // and any text held back as a possible stop-sequence prefix is real
        // output once the stream ends without matching.
        if(!client_gone && !stop_hit) {
            bool stopped=false;
            std::string tail=stopbuf.feed(ugate.flush(),stopped);
            tail+=stopbuf.flush();
            if(!tail.empty()) emit(tail);
            if(stopped) stop_hit=true;
        }

        Outcome out;
        out.prompt_tokens=(uint32_t)prompt.size();
        out.output_tokens=produced;
        out.prefix_hit=hit;
        out.exact_disk_hit=exact_disk_hit;
        out.exact_snapshot_written=exact_snapshot_written;
        out.queue_wait_ms=queue_wait_ms;
        out.gate_wait_ms=std::max(gate_wait_ms,0.0);
        out.arrival=arrival;
        if(client_gone) {
            out.finish=Finish::Cancelled;
            trace.event({{"kind","cancel"},{"phase","generate"},{"id",trace_id}});
        }
        else if(stop_hit) {
            out.finish=Finish::StopSequence;
            if(stopbuf.matched>=0 && stopbuf.matched<(int)stops.size())
                out.stop_sequence=stops[stopbuf.matched];
        } else if(cause==q27::MetalEngine::StopCause::Eos) out.finish=Finish::Stop;
        else out.finish=Finish::Length;
        return out;
    }
};

const char* trace_finish(Runtime::Finish f) {
    switch(f) {
        case Runtime::Finish::Length: return "length";
        case Runtime::Finish::StopSequence: return "stop_sequence";
        case Runtime::Finish::Cancelled: return "cancelled";
        default: return "eos";
    }
}
const char* openai_finish(Runtime::Finish f) {
    switch(f) {
        case Runtime::Finish::Length: return "length";
        default: return "stop"; // Stop (eos), StopSequence, and Cancelled
    }
}
const char* anthropic_stop(Runtime::Finish f) {
    switch(f) {
        case Runtime::Finish::Length: return "max_tokens";
        case Runtime::Finish::StopSequence: return "stop_sequence";
        default: return "end_turn"; // Stop (eos) and Cancelled
    }
}

q27::SamplingParams sampling_params(const json& body) {
    q27::SamplingParams result;
    result.temperature=sampling_default_temperature;
    result.top_p=sampling_default_top_p;
    result.top_k=sampling_default_top_k;
    // Present-but-null falls back to the served default rather than throwing
    // (clients send null-valued fields -- max_tokens handles the same pi.dev
    // shape below). Any OTHER non-number type is malformed: reject it instead
    // of silently serving the default while the client believes its sampling
    // settings were honored (codex branch-review P2).
    if(body.contains("temperature")) {
        const json& v=body["temperature"];
        if(!v.is_null()) {
            if(!v.is_number()) throw std::runtime_error("invalid temperature");
            result.temperature=v.get<float>();
        }
    }
    if(body.contains("top_p")) {
        const json& v=body["top_p"];
        if(!v.is_null()) {
            if(!v.is_number()) throw std::runtime_error("invalid top_p");
            result.top_p=v.get<float>();
        }
    }
    if(body.contains("top_k")) {
        const json& v=body["top_k"];
        if(!v.is_null()) {
            // Integer type required: a JSON float (1.5) would silently
            // truncate to top_k==1 (pure argmax) and a negative would wrap
            // to a huge unsigned (codex branch-review P2).
            if(!v.is_number_integer() && !v.is_number_unsigned())
                throw std::runtime_error("invalid top_k");
            const long long k=v.get<long long>();
            if(k<0 || (unsigned long long)k>UINT32_MAX)
                throw std::runtime_error("invalid top_k");
            result.top_k=(uint32_t)k;
        }
    }
    result.seed=0;
    if(body.contains("seed")) {
        const json& v=body["seed"];
        if(!v.is_null()) {
            if(v.is_number_unsigned()) result.seed=v.get<uint64_t>();
            else if(v.is_number_integer()) {
                const long long s=v.get<long long>();
                if(s<0) throw std::runtime_error("invalid seed");
                result.seed=(uint64_t)s;
            } else throw std::runtime_error("invalid seed");
        }
    }
    q27::validate_sampling(result);
    return result;
}

// Per-endpoint defaults mirror the CUDA server (codex P3 on this round):
// /v1/messages 1024, OpenAI completions/chat 256, responses 4096.
// --max-tokens-default (flag wins; env fallback Q27_METAL_MAX_TOKENS_DEFAULT)
// overrides all of them for requests that omit max_tokens (or send null):
// pi.dev sends max_tokens:null, and the
// CUDA-parity 256 truncates real agent turns mid-answer ("maximum output
// token limit", 2026-07-17). Explicit client values always win; the
// context preflight still clamps to the remaining window.
// Set once in main (before the server accepts) from --max-tokens-default;
// 0 = flag absent.
long long max_tokens_default_flag=0;
// Null/absent -> default; wrong-typed -> a CALLER-NAMED error. json_i64_or
// lets nlohmann's raw exception text escape, so a wrong-typed max_tokens
// answered the client with
//   "[json.exception.type_error.302] type must be number, but is string"
// inside the invalid_request_error envelope -- library internals leaking into
// a public API error, and inconsistent with the sibling fields resolved two
// functions up, which already say "invalid temperature" / "invalid top_k"
// (found by the 2026-07-25 serving gate). Floats are still accepted and
// truncated, as json_i64_or did: clients do send max_tokens: 4096.0.
long long int_field_or(const json& body,const char* key,long long dflt) {
    const auto it=body.find(key);
    if(it==body.end() || it->is_null()) return dflt;
    if(!it->is_number()) throw std::runtime_error(std::string("invalid ")+key);
    const double v=it->get<double>();
    // Range-check as a double BEFORE narrowing: a value past long long is UB
    // to convert, so it cannot be left to the caller's check.
    if(!(v>=0.0) || v>(double)UINT32_MAX) throw std::runtime_error(std::string("invalid ")+key);
    return (long long)v;
}

uint32_t max_tokens(const json& body,long long dflt) {
    if(max_tokens_default_flag>0) dflt=max_tokens_default_flag; // resolved flag/env value
    long long value=int_field_or(body,"max_output_tokens",dflt);
    value=int_field_or(body,"max_tokens",value);
    if(value<0 || value>UINT32_MAX) throw std::runtime_error("invalid max_tokens");
    return (uint32_t)value;
}

// jbool, not value(): `{"stream": null}` is how a large share of
// OpenAI-compatible request builders (LangChain/LiteLLM class) spell "unset",
// and value() throws type_error.302 on a present-but-null key -- which
// guarded() turns into a 400 invalid_request_error for a request that was
// fine on the wire. sampling_params/max_tokens already treated null as
// absent; `stream` and `prompt` were the two scalars that missed the rule
// (upstream 1a15ff8 fixed the same class on the CUDA arm).
//
// DELIBERATE DIVERGENCE from upstream's jnum/jint on temperature/top_p/top_k/
// seed/max_tokens: those stay STRICT here (a wrong-typed value is rejected,
// not silently replaced by the server default) per the codex branch-review P2
// recorded at sampling_params. Null-tolerance is the fix; wrong-type-tolerance
// is not, because it serves a request whose sampling settings the client
// believes were honored.
bool wants_stream(const json& body) { return q27::jbool(body,"stream",false); }

long unix_now() { return (long)std::time(nullptr); }

void json_response(httplib::Response& response,const json& value,int status=200) {
    response.status=status;
    response.set_content(value.dump(-1,' ',false,json::error_handler_t::replace),"application/json");
}

} // namespace

static int mark_supervisor_lock_close_on_exec() {
    const char* value = std::getenv("Q27_SUPERVISOR_LOCK_FD");
    char* end = nullptr;
    long parsed;
    int flags;
    if (!value || !*value) return 0;
    errno = 0;
    parsed = std::strtol(value, &end, 10);
    if (errno || !end || *end || parsed < 0 || parsed > 0x7fffffffL) {
        std::fprintf(stderr, "q27-metal-server: invalid supervisor lock descriptor\n");
        return -1;
    }
    flags = fcntl(static_cast<int>(parsed), F_GETFD);
    if (flags < 0 ||
        fcntl(static_cast<int>(parsed), F_SETFD, flags | FD_CLOEXEC) != 0) {
        std::fprintf(stderr,
                     "q27-metal-server: cannot protect supervisor lock descriptor: %s\n",
                     std::strerror(errno));
        return -1;
    }
    (void)unsetenv("Q27_SUPERVISOR_LOCK_FD");
    return 0;
}

int main(int argc,char** argv) {
    if (mark_supervisor_lock_close_on_exec() != 0) return 2;
    if(argc<3) {
        fprintf(stderr,"usage: %s model.q27 tokenizer.tok [--host 127.0.0.1] [--port 8080] [--ctx N|auto] [--mtp 2..12 | --suffix 2..48] [--kv fp16|turbo3] [--prefix-entries N] [--constrain-tools] [--think] [--request-think] [--slots N] [--trace path]\n"
                       "       [--snapshot-dir path] [--snapshot-max-mb 1..16777216] [--snapshot-auto 0..16777216] [--snapshot-spine-pin 0|1] [--max-tokens-default N] [--budget-mb 1..16777216]\n"
                       "       [--temperature-default T] [--top-p-default P] [--top-k-default K]\n"
                       "       [--experimental-prefix-cache path] [--api-key KEY] [--api-key-file path]\n"
                       "       (the snapshot/max-tokens/budget/sampling-default flags fall back to their env twins Q27_METAL_{SNAPSHOT_DIR,SNAPSHOT_MAX_MB,SNAPSHOT_AUTO,SNAPSHOT_SPINE_PIN,MAX_TOKENS_DEFAULT,BUDGET_MB,TEMPERATURE_DEFAULT,TOP_P_DEFAULT,TOP_K_DEFAULT}; an explicit flag wins)\n",argv[0]);
        return 1;
    }
    try {
        std::string model=argv[1],tok=argv[2],host="127.0.0.1";
        std::string trace_path,snapshot_dir,experimental_prefix_dir;
        uint32_t port=8080,context=0,width=0,suffix_width=0,prefix_entries=1,slot_count=2;
        // Shipped-semantics knobs as flags (homebrew Phase-2 pre-tag);
        // sentinel = flag absent, Runtime falls back to the env twin.
        // snapshot_auto keeps a signed sentinel because 0 is meaningful
        // (hint-only saves).
        uint32_t budget_mb=0,snapshot_max_mb=0,max_tokens_default=0;
        bool think_default=false;   // --think flips the server profile (default no-think)
        // --request-think (upstream 0a1b21f parity): honor the request's
        // thinking fields. OFF by default -- without it, enable_thinking /
        // chat_template_kwargs / Anthropic `thinking` are IGNORED and the boot
        // profile stands. Closes the footgun where a benchmark or client that
        // sends enable_thinking:true (many do) silently flips a no-think
        // server into thinking mode. NOTE: this is a behavior change for the
        // Metal arm, which honored those fields unconditionally through
        // v0.6.1; pass --request-think to keep the old behavior.
        bool req_think=false;
        // Sampling-default sentinels double as "flag absent" (-1 / 0 /
        // UINT32_MAX are all outside the valid ranges); with neither flag
        // nor env the resolved values are the shipped greedy defaults.
        float temperature_default=-1.0f, top_p_default=0.0f;
        uint32_t top_k_default=UINT32_MAX;
        long long snapshot_auto=-1;
        int spine_pin=-1;   // -1 = unset (env/default); 0/1 explicit flag
        bool turbo3=false; bool constrain_tools=false;
        // Opt-in Bearer/x-api-key auth on the SERVING endpoints (upstream
        // v0.4.0 + 1a15ff8 parity; the long-standing Metal gap #4 in
        // docs/metal/PARITY-2026-07-22.md). Distinct from admin_token, which
        // covers only the mutating admin routes. Empty = auth off, and the
        // 127.0.0.1 default binding stays the posture for that case.
        std::vector<std::string> api_keys;
        for(int i=3;i<argc;i++) {
            std::string arg=argv[i];
            if(arg=="--host" && i+1<argc) host=argv[++i];
            else if(arg=="--port" && i+1<argc) port=parse_u32(argv[++i],"--port");
            else if(arg=="--ctx" && i+1<argc) {
                const char* v=argv[++i];
                if(!strcmp(v,"auto")) context=0;   // sentinel: resolve in Runtime
                else {
                    context=parse_u32(v,"--ctx");
                    // 0 is the auto sentinel, so a numeric 0 must be
                    // rejected HERE — silently reinterpreting it as auto
                    // would allocate a hardware-dependent window for an
                    // invalid explicit configuration (r2 codex P2).
                    if(!context || context>262144)
                        throw std::runtime_error("--ctx must be 1..262144 or auto");
                }
            }
            else if(arg=="--mtp" && i+1<argc) width=parse_u32(argv[++i],"--mtp");
            else if(arg=="--suffix" && i+1<argc) suffix_width=parse_u32(argv[++i],"--suffix");
            else if(arg=="--prefix-entries" && i+1<argc) prefix_entries=parse_u32(argv[++i],"--prefix-entries");
            else if(arg=="--slots" && i+1<argc) slot_count=parse_u32(argv[++i],"--slots");
            else if(arg=="--kv" && i+1<argc) { std::string mode=argv[++i]; if(mode=="turbo3")turbo3=true; else if(mode!="fp16")throw std::runtime_error("invalid --kv"); }
            else if(arg=="--constrain-tools") constrain_tools=true;
            else if(arg=="--think") think_default=true;
            else if(arg=="--request-think") req_think=true;
            else if(arg=="--trace" && i+1<argc) trace_path=argv[++i];
            else if(arg=="--snapshot-dir" && i+1<argc) { snapshot_dir=argv[++i]; if(snapshot_dir.empty()) throw std::runtime_error("invalid --snapshot-dir"); }
            else if(arg=="--experimental-prefix-cache" && i+1<argc) {
                experimental_prefix_dir=argv[++i];
                if(experimental_prefix_dir.empty())
                    throw std::runtime_error("invalid --experimental-prefix-cache");
            }
            // The env twins reject 0 as malformed, and 0 here would silently
            // collapse into the "flag absent" sentinel — so it fails loud
            // in-branch (post-loop checks can no longer tell 0 from unset).
            else if(arg=="--snapshot-max-mb" && i+1<argc) { snapshot_max_mb=parse_u32(argv[++i],"--snapshot-max-mb"); if(!snapshot_max_mb||snapshot_max_mb>(1u<<24)) throw std::runtime_error("--snapshot-max-mb must be an integer 1..16777216"); }
            else if(arg=="--snapshot-auto" && i+1<argc) { snapshot_auto=parse_u32(argv[++i],"--snapshot-auto"); if(snapshot_auto>(1ll<<24)) throw std::runtime_error("--snapshot-auto must be an integer 0..16777216"); }
            else if(arg=="--max-tokens-default" && i+1<argc) { max_tokens_default=parse_u32(argv[++i],"--max-tokens-default"); if(!max_tokens_default) throw std::runtime_error("--max-tokens-default must be >= 1"); }
            else if(arg=="--temperature-default" && i+1<argc) { temperature_default=parse_float(argv[++i],"--temperature-default"); if(temperature_default<0.0f) throw std::runtime_error("--temperature-default must be >= 0"); }
            else if(arg=="--top-p-default" && i+1<argc) { top_p_default=parse_float(argv[++i],"--top-p-default"); if(top_p_default<=0.0f||top_p_default>1.0f) throw std::runtime_error("--top-p-default must be in (0,1]"); }
            else if(arg=="--top-k-default" && i+1<argc) { top_k_default=parse_u32(argv[++i],"--top-k-default"); if(top_k_default==UINT32_MAX) throw std::runtime_error("invalid --top-k-default"); }
            else if(arg=="--budget-mb" && i+1<argc) { budget_mb=parse_u32(argv[++i],"--budget-mb"); if(!budget_mb||budget_mb>(1u<<24)) throw std::runtime_error("--budget-mb must be an integer 1..16777216"); }
            // Range-check the UNSIGNED value BEFORE narrowing to int. The
            // old `(int)parse_u32(...)` then `>N` order let anything in
            // [2^31, 2^32) cast to a negative int and sail past the bound, so
            // `--snapshot-spine-pin 3000000000` was silently accepted and then
            // read as "unset" -- an operator pinning the spine could get the
            // opposite of what they asked for, with no diagnostic.
            else if(arg=="--snapshot-spine-pin" && i+1<argc) {
                const uint32_t v=parse_u32(argv[++i],"--snapshot-spine-pin");
                if(v>1) throw std::runtime_error("--snapshot-spine-pin must be 0 or 1");
                spine_pin=(int)v;
            }
            // Auth config is fail-LOUD in both directions (upstream 1a15ff8
            // item 4). An empty key can never authenticate anything --
            // api_key_valid rejects an empty `provided` before comparing --
            // so accepting one stands up a server that 401s every request;
            // and a key FILE that opens but yields nothing usable stands up a
            // server with auth silently OFF, the opposite of what the
            // operator asked for. Refuse both at boot rather than serve a
            // configuration nobody wanted.
            else if(arg=="--api-key" && i+1<argc) {
                if(!argv[++i][0]) throw std::runtime_error("--api-key: empty key (an empty key can never match)");
                api_keys.push_back(argv[i]);
            }
            else if(arg=="--api-key-file" && i+1<argc) {
                const size_t before=api_keys.size();
                if(!q27::load_api_key_file(argv[++i],&api_keys))
                    throw std::runtime_error(std::string("--api-key-file ")+argv[i]+": could not open");
                if(api_keys.size()==before)
                    throw std::runtime_error(std::string("--api-key-file ")+argv[i]+
                        ": no keys found (every line blank or a #comment) -- refusing to start "
                        "with auth silently disabled");
            }
            else throw std::runtime_error("unknown/incomplete argument: "+arg);
        }
        // Q27_API_KEY: a second, additive source (not exclusive with the CLI
        // flags -- all configured keys are valid simultaneously, matching
        // --api-key-file's multi-key semantics). Preferred where CLI args are
        // visible via `ps` but the orchestrator's secret store is not.
        // The `envkey[0]` test is load-bearing, not defensive tidiness:
        // api_key_valid's constant-time property documents "an empty KEY is
        // never configured" as an INVARIANT its callers must uphold (it fast-
        // paths an empty `provided` without comparing). An empty Q27_API_KEY
        // must therefore be dropped here, exactly as --api-key refuses one
        // above. Do not relax either check.
        if(const char* envkey=getenv("Q27_API_KEY"); envkey && envkey[0]) api_keys.push_back(envkey);
        // The loopback-by-default binding was this server's only safety net
        // before auth existed. Warn loudly rather than refuse: some
        // deployments front this with their own reverse-proxy auth, and
        // silently breaking those on upgrade would be worse.
        if(api_keys.empty() && host!="127.0.0.1" && host!="localhost" && host!="::1")
            fprintf(stderr,
                    "WARNING: binding %s with NO API key configured (--api-key / "
                    "--api-key-file / Q27_API_KEY) -- this server will accept "
                    "unauthenticated requests from anyone who can reach it.\n",host.c_str());
        if(port>65535) throw std::runtime_error("port out of range");
        if(width && (width<2 || width>12)) throw std::runtime_error("MTP width must be 2..12");
        if(suffix_width && (suffix_width<2 || suffix_width>q27::MetalEngine::VERIFY_CHUNK_MAX))
            throw std::runtime_error("suffix width must be 2..48");
        // Bounded so entries x snapshot_bytes() can never wrap uint64 in the
        // G6 per-slot charge — side-inclusive snapshots (~4.6 GB at max ctx
        // under L7-full) put the wrap within uint32 entry range (codex P2 on
        // 4415c53); 4096 entries is already far beyond any real deployment.
        if(prefix_entries>4096) throw std::runtime_error("--prefix-entries must be 0..4096");
        if(width && suffix_width) throw std::runtime_error("--mtp and --suffix are mutually exclusive (one speculation lever per server)");
        if(!experimental_prefix_dir.empty()) {
            if(width) throw std::runtime_error("--experimental-prefix-cache does not support --mtp");
            if(!snapshot_dir.empty() && snapshot_dir!=experimental_prefix_dir)
                throw std::runtime_error("--snapshot-dir and --experimental-prefix-cache must name the same directory");
            snapshot_dir=experimental_prefix_dir;
        }
        if(constrain_tools && width) throw std::runtime_error("--constrain-tools requires serial decode; drop --mtp (verify-lane masks are not wired on Metal)");
        if(constrain_tools && suffix_width) throw std::runtime_error("--constrain-tools requires serial decode; drop --suffix (burst rounds argmax unmasked logits)");
        // Phase 1's latency guarantee (wait <= one active quantum) only
        // holds with one competing slot; >2 needs the scheduler and the
        // width/stats model extended first (codex P2 on d243f92).
        if(slot_count<1 || slot_count>2) throw std::runtime_error("--slots must be 1..2 in multislot Phase 1");
        if(!max_tokens_default)
            if(const char* e=getenv("Q27_METAL_MAX_TOKENS_DEFAULT"); e && *e) {
                max_tokens_default=parse_u32(e,"Q27_METAL_MAX_TOKENS_DEFAULT");
                if(!max_tokens_default)
                    throw std::runtime_error("Q27_METAL_MAX_TOKENS_DEFAULT must be >= 1");
            }
        max_tokens_default_flag=max_tokens_default;
        if(temperature_default<0.0f)
            if(const char* e=getenv("Q27_METAL_TEMPERATURE_DEFAULT"); e && *e) {
                temperature_default=parse_float(e,"Q27_METAL_TEMPERATURE_DEFAULT");
                if(temperature_default<0.0f) throw std::runtime_error("Q27_METAL_TEMPERATURE_DEFAULT must be >= 0");
            }
        if(top_p_default==0.0f)
            if(const char* e=getenv("Q27_METAL_TOP_P_DEFAULT"); e && *e) {
                top_p_default=parse_float(e,"Q27_METAL_TOP_P_DEFAULT");
                if(top_p_default<=0.0f||top_p_default>1.0f) throw std::runtime_error("Q27_METAL_TOP_P_DEFAULT must be in (0,1]");
            }
        if(top_k_default==UINT32_MAX)
            if(const char* e=getenv("Q27_METAL_TOP_K_DEFAULT"); e && *e) {
                top_k_default=parse_u32(e,"Q27_METAL_TOP_K_DEFAULT");
                if(top_k_default==UINT32_MAX) throw std::runtime_error("invalid Q27_METAL_TOP_K_DEFAULT");
            }
        // Neither flag nor env: the shipped greedy defaults (bitwise unchanged).
        if(temperature_default<0.0f) temperature_default=0.0f;
        if(top_p_default==0.0f) top_p_default=1.0f;
        if(top_k_default==UINT32_MAX) top_k_default=0;
        sampling_default_temperature=temperature_default;
        sampling_default_top_p=top_p_default;
        sampling_default_top_k=top_k_default;
        if(sampling_default_temperature>0.0f && width)
            fprintf(stderr,"[sampling-default] temperature default %.4g uses sampled MTP (rejection accept) when --mtp is set; set temperature=0 per-request for greedy MTP\n",
                    (double)sampling_default_temperature);
        if(sampling_default_temperature>0.0f && suffix_width)
            fprintf(stderr,"[sampling-default] temperature default %.4g disengages suffix speculation for requests that omit temperature (suffix is greedy-only)\n",
                    (double)sampling_default_temperature);
        Runtime runtime(model,tok,context,turbo3,width,suffix_width,prefix_entries,constrain_tools,slot_count,
                        budget_mb,snapshot_dir,snapshot_max_mb,snapshot_auto,max_tokens_default,spine_pin,
                        !experimental_prefix_dir.empty(),think_default);
        if(!trace_path.empty()) {
            runtime.trace.open(trace_path);
            runtime.trace.event({{"kind","boot"},{"ctx",runtime.context},{"kv",turbo3?"turbo3":"fp16"},
                                 {"mtp",width},{"suffix",suffix_width},{"slots",runtime.slots.size()},
                                 {"model",runtime.model_name},{"artifact_sha1",runtime.resident_model_sha1()},
                                 {"runtime",runtime.serving_identity()},{"boot_id",runtime.boot_id}});
        }
        httplib::Server server;
        // Bound the accept-side queue (codex P1 on d243f92): the default
        // task queue holds accepted connections without limit, so the
        // in-run admission bound alone could never engage — excess requests
        // would pile up behind the workers instead of being rejected.
        // 16 workers > QUEUE_MAX + slots: with only 8 workers the ticket-queue
        // overflow (503) was UNREACHABLE dead code — at most 7 requests could
        // wait while one generated (G6 found this). The 32-connection accept
        // queue stays the outer bound.
        server.new_task_queue=[]{ return new httplib::ThreadPool(16,32); };
        // Opt-in API key auth (upstream v0.4.0 parity). No-op with zero
        // per-request cost when api_keys is empty -- the handler is only
        // installed when at least one key is configured, so the loopback-only
        // default posture is byte-identical to before. /health is exempt on
        // purpose: infra health checks (launchd, load balancers, the deploy
        // scripts under deploy/) need it reachable without distributing the
        // secret to that infrastructure. Every other route -- including
        // /stats, /admin/* and the experimental prewarmer -- requires a valid
        // key. Runs BEFORE route dispatch, so an unauthenticated request never
        // reaches RequestScope admission, tokenization, or slot claim.
        if(!api_keys.empty()) {
            server.set_pre_routing_handler([api_keys](const httplib::Request& req,httplib::Response& r){
                if(req.path=="/health") return httplib::Server::HandlerResponse::Unhandled;
                const std::string provided=q27::extract_api_key(req.get_header_value("Authorization"),
                                                                req.get_header_value("x-api-key"));
                if(q27::api_key_valid(provided,api_keys))
                    return httplib::Server::HandlerResponse::Unhandled;
                // Anthropic-shaped error for the Anthropic-shaped endpoint
                // family (Claude Code's SDK reads error.message off that exact
                // shape); OpenAI-shaped for everything else -- the same split
                // anthropic_guarded/guarded already use for 400s.
                const bool anthropic_shape=req.path.rfind("/v1/messages",0)==0;
                r.status=401;
                if(!anthropic_shape) r.set_header("WWW-Authenticate","Bearer");
                r.set_content(q27::auth_error_json(anthropic_shape),"application/json");
                return httplib::Server::HandlerResponse::Handled;
            });
            fprintf(stderr,"API key authentication enabled (%zu key%s configured)\n",
                    api_keys.size(),api_keys.size()==1?"":"s");
        }
        server.Get("/health",[&runtime](const httplib::Request& req,httplib::Response& r){
            json body={{"status",runtime.health_status()},{"model",runtime.model_name},{"boot_id",runtime.boot_id},
                       {"runtime",runtime.serving_identity()},
                       {"serving",{{"draining",runtime.draining.load()},
                                   {"active_requests",runtime.active_requests.load()}}},
                       {"trace",{{"enabled",runtime.trace.enabled()},
                                 {"healthy",runtime.trace.healthy()}}}};
            if(req.has_param("identity") && req.get_param_value("identity")=="1")
                body["artifact_sha1"]=runtime.resident_model_sha1();
            json_response(r,body);
        });
        auto admin_ok=[&runtime](const httplib::Request& req) {
            const bool loopback=req.remote_addr=="127.0.0.1" || req.remote_addr=="::1";
            // Separate admin credential, not the public boot_id (P2).
            return loopback && !runtime.admin_token.empty() &&
                   req.get_header_value("X-Q27-Admin-Token")==runtime.admin_token;
        };
        server.Post("/admin/drain",[&runtime,admin_ok](const httplib::Request& req,httplib::Response& r){
            if(!admin_ok(req)) { json_response(r,{{"error","forbidden"}},403); return; }
            runtime.draining.store(true,std::memory_order_seq_cst);
            json_response(r,{{"draining",true},{"active_requests",runtime.active_requests.load()}});
        });
        server.Post("/admin/resume",[&runtime,admin_ok](const httplib::Request& req,httplib::Response& r){
            if(!admin_ok(req)) { json_response(r,{{"error","forbidden"}},403); return; }
            runtime.draining.store(false,std::memory_order_seq_cst);
            json_response(r,{{"draining",false},{"active_requests",runtime.active_requests.load()}});
        });
        // Wait honesty (Phase 1 contract): per-arrival-phase stats. Gate
        // wait (admission -> first lease) carries the one-quantum bound;
        // queue wait (arrival -> admission) is bounded only by QUEUE_MAX
        // generations and is reported so nobody mistakes one for the other.
        server.Get("/stats",[&runtime](const httplib::Request&,httplib::Response& r){
            auto bucket=[](const std::map<std::string,Runtime::WaitStats>& stats){
                json out=json::object();
                for(const auto& [phase,ws]:stats)
                    out[phase]={{"requests",ws.n},
                                {"mean_ms",ws.n?ws.sum_ms/ws.n:0.0},
                                {"max_ms",ws.max_ms}};
                return out;
            };
            json gate,queue;
            size_t slot_count=0;
            {
                std::unique_lock<std::mutex> lk(runtime.route_);
                runtime.slot_free_.wait(lk,[&]{ return !runtime.recovering_; });
                gate=bucket(runtime.gate_wait_stats);
                queue=bucket(runtime.queue_wait_stats);
                slot_count=runtime.slots.size();
            }
            json_response(r,{{"slots",slot_count},
                             {"gate_wait_by_arrival",gate},
                             {"queue_wait_by_arrival",queue},
                             {"speculation",{{"rounds",(uint64_t)runtime.spec_rounds_total},
                                             {"committed",(uint64_t)runtime.spec_committed_total},
                                             {"suffix_bursts",(uint64_t)runtime.suffix_burst_rounds_total},
                                             {"suffix_fallbacks",(uint64_t)runtime.suffix_fallback_rounds_total}}},
                             {"snapshots",{{"enabled",runtime.snapstore.enabled()},
                                           {"experimental_prefix_cache",runtime.experimental_prefix_cache},
                                           {"disk_hits",(uint64_t)runtime.snapstore.hits},
                                           {"disk_saves",(uint64_t)runtime.snapstore.saves},
                                           {"evicted_spine",(uint64_t)runtime.snapstore.evicted_spine},
                                           {"evicted_leaf",(uint64_t)runtime.snapstore.evicted_leaf}}},
                             {"cancellations",{{"queued",(uint64_t)runtime.cancelled_queue},
                                               {"prefill",(uint64_t)runtime.cancelled_prefill}}}});
        });
        server.Get("/v1/models",[](const httplib::Request&,httplib::Response& r){json_response(r,{{"object","list"},{"data",json::array({{{"id","q27-metal"},{"object","model"}}})}});});

        auto guarded=[&](const char* api,auto handler) {
            return [&,api,handler](const httplib::Request& request,httplib::Response& response) {
                Runtime::RequestScope active(runtime);
                const std::string error_id="arrival_metal_"+runtime.boot_id+"_"+
                    std::to_string((long)req_counter++);
                runtime.trace.event({{"kind","arrival"},{"api",api},{"id",error_id},
                                     {"admitted",active.admitted}});
                if(!active.admitted) {
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},
                                         {"status",503},{"type","overloaded_error"}});
                    json_response(response,{{"error",{{"message","server draining"},
                        {"type","overloaded_error"}}}},503);
                    return;
                }
                try { handler(json::parse(request.body),response,request.sock); }
                // 499 (client closed request): on a genuinely dead socket
                // the write fails harmlessly; on an is_socket_alive false
                // negative the client gets a parseable error instead of an
                // empty 200 (codex P2 on this round).
                catch(const Runtime::ClientGone&) { response.status=499; }
                catch(const Runtime::ServerOverloaded& e) { json_response(response,{{"error",{{"message",e.what()},{"type","overloaded_error"}}}},503); }
                catch(const Runtime::EngineError& e) { json_response(response,{{"error",{{"message",e.what()},{"type","api_error"}}}},500); }
                catch(const std::exception& e) {
                    runtime.trace.event({{"kind","request"},{"api",api},{"id",error_id},{"validation_error",true}});
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},{"status",400},
                        {"type","invalid_request_error"},{"message",e.what()}});
                    json_response(response,{{"error",{{"message",e.what()},{"type","invalid_request_error"}}}},400);
                }
            };
        };
        // Liveness probe for phases with no response writes yet (queue wait,
        // prefill) and for non-streaming generation. The socket fd rides the
        // q27 httplib patch (Request::sock); the probe itself is ours (see
        // socket_alive above), so httplib internals are not a dependency.
        auto socket_live=[](socket_t sock){
            return [sock]{ return socket_alive(sock); };
        };
        // Wraps ONLY a handler's run() call: engine failures reclassify as
        // EngineError (api_error 500); the cancellation and overload types
        // pass through untouched.
        auto engine_guard=[&](auto&& fn)->decltype(fn()) {
            return runtime.guard_engine(std::forward<decltype(fn)>(fn));
        };
        auto traced_engine=[&](const char* api,const std::string& id,auto&& fn)->decltype(fn()) {
            try { return engine_guard(std::forward<decltype(fn)>(fn)); }
            catch(const Runtime::ServerOverloaded& e) {
                runtime.trace.event({{"kind","error"},{"api",api},{"id",id},{"status",503},
                    {"type","overloaded_error"},{"message",e.what()}});
                throw;
            } catch(const Runtime::EngineError& e) {
                runtime.trace.event({{"kind","error"},{"api",api},{"id",id},{"status",500},
                    {"type","api_error"},{"message",e.what()}});
                throw;
            }
        };
        // Anthropic endpoints answer in Anthropic's error envelope
        // ({"type":"error","error":{...}} — the SDK inside Claude Code reads
        // error.message from it), not the OpenAI shape (codex P2 on this
        // round). Streaming errors after SSE commit use the error event.
        auto anthropic_guarded=[&](const char* api,auto handler) {
            return [&,api,handler](const httplib::Request& request,httplib::Response& response) {
                Runtime::RequestScope active(runtime);
                const std::string error_id="arrival_metal_"+runtime.boot_id+"_"+
                    std::to_string((long)req_counter++);
                runtime.trace.event({{"kind","arrival"},{"api",api},{"id",error_id},
                                     {"admitted",active.admitted}});
                if(!active.admitted) {
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},
                                         {"status",503},{"type","overloaded_error"}});
                    response.status=503;
                    response.set_content(q27::anthropic_error_json("overloaded_error","server draining"),"application/json");
                    return;
                }
                json body;
                try { body=json::parse(request.body); }
                catch(...) {
                    runtime.trace.event({{"kind","request"},{"api",api},{"id",error_id},{"validation_error",true}});
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},{"status",400},
                        {"type","invalid_request_error"},{"message","invalid JSON body"}});
                    response.status=400;
                    response.set_content(q27::anthropic_error_json("invalid_request_error","invalid JSON body"),"application/json");
                    return;
                }
                try { handler(body,response,request.sock); }
                // 499 as in `guarded` (codex P2): never an empty 200.
                catch(const Runtime::ClientGone&) { response.status=499; }
                catch(const Runtime::EngineError& e) {
                    response.status=500;
                    response.set_content(q27::anthropic_error_json("api_error",e.what()),"application/json");
                }
                catch(const Runtime::ServerOverloaded& e) {
                    response.status=503;
                    response.set_content(q27::anthropic_error_json("overloaded_error",e.what()),"application/json");
                }
                catch(const std::exception& e) {
                    runtime.trace.event({{"kind","request"},{"api",api},{"id",error_id},{"validation_error",true}});
                    runtime.trace.event({{"kind","error"},{"api",api},{"id",error_id},{"status",400},
                        {"type","invalid_request_error"},{"message",e.what()}});
                    response.status=400;
                    response.set_content(q27::anthropic_error_json("invalid_request_error",e.what()),"application/json");
                }
            };
        };

        // Shared context preflight: each endpoint refuses an oversized prompt
        // in its API's native 400 shape before slot claim / SSE commit.
        auto prompt_overflow=[&](size_t prompt_tokens,uint32_t& n,uint32_t& maxp)->bool {
            maxp=max_prompt_tokens(runtime.context,runtime.mtp_width,runtime.suffix_width);
            if(prompt_tokens>maxp) return true;
            if(prompt_tokens+n>runtime.context) n=runtime.context-(uint32_t)prompt_tokens;
            return false;
        };

        // Experimental, opt-in harness-prefix installer. It accepts one
        // ordinary initial Chat Completions or Responses request, removes
        // exactly its final live user message, verifies that the resulting
        // token stream is an exact prefix of the ordinary serving prompt,
        // and persists that prefix without decoding. No route is registered
        // unless --experimental-prefix-cache was supplied; the separate
        // admin credential and loopback check apply as for drain/resume.
        if(runtime.experimental_prefix_cache)
            server.Post("/experimental/prefix-cache/prewarm",
                [&](const httplib::Request& req,httplib::Response& response) {
                if(!admin_ok(req)) {
                    json_response(response,{{"error","forbidden"}},403);
                    return;
                }
                Runtime::RequestScope active(runtime);
                if(!active.admitted) {
                    json_response(response,{{"error","server draining"}},503);
                    return;
                }
                try {
                    const json envelope=json::parse(req.body);
                    if(!envelope.contains("request") || !envelope["request"].is_object())
                        throw std::runtime_error("request must be an object");
                    const json& request=envelope["request"];
                    const std::string api=q27::jstr(envelope,"api");
                    json tools=json::array();
                    std::vector<q27::Msg> messages;
                    bool think=true,force_tool=false;
                    if(api=="chat" || api=="chat_completions") {
                        messages=openai_msgs(request);
                        const q27::ToolChoice tchoice=q27::parse_tool_choice(request);
                        q27::OpenAIToolSelection selected=q27::select_openai_tools(request,tchoice);
                        tools=std::move(selected.tools);
                        force_tool=tchoice.mode==q27::ToolChoice::FORCED;
                        think=q27::resolve_think(request,think_default,req_think);
                        if(force_tool) think=false;
                    } else if(api=="responses") {
                        ResponsesPromptInput normalized=responses_prompt_input(request);
                        tools=std::move(normalized.tools);
                        messages=std::move(normalized.messages);
                        force_tool=normalized.choice.mode==q27::ToolChoice::FORCED;
                        think=q27::resolve_think(request,think_default,req_think);
                        if(force_tool) think=false;
                    } else if(api=="messages" || api=="anthropic") {
                        // Claude Code speaks Anthropic /v1/messages. Reuse the
                        // SAME canonicalizer as ordinary serving (line ~2388:
                        // chatml_prompt(anthropic_msgs(body), tools, true)) so
                        // the prewarmed prefix is byte-for-byte the prefix the
                        // live request will prefill. Same resolver as serving.
                        messages=q27::anthropic_msgs(request);
                        tools=q27::anthropic_tools_json(request);
                        think=q27::resolve_think(request,think_default,req_think);
                    } else throw std::runtime_error(
                        "api must be chat_completions, responses, or messages");

                    std::string full_rendered;
                    const std::string prefix_rendered=q27::initial_harness_prefix(
                        messages,tools,think,&full_rendered);
                    if(force_tool) full_rendered+="<tool_call>\n";
                    auto prefix_ids=to_u32(runtime.tokenizer.encode(prefix_rendered));
                    const auto full_ids=to_u32(runtime.tokenizer.encode(full_rendered));
                    if(prefix_ids.empty() || prefix_ids.size()>=full_ids.size() ||
                       !std::equal(prefix_ids.begin(),prefix_ids.end(),full_ids.begin()))
                        throw std::runtime_error(
                            "derived static prompt is not an exact token prefix");
                    const uint32_t maxp=max_prompt_tokens(
                        runtime.context,runtime.mtp_width,runtime.suffix_width);
                    if(prefix_ids.size()>maxp)
                        throw std::runtime_error("derived prefix exceeds context");

                    char cache_key[41];
                    snap_hash_sha1(prefix_ids.data(),(uint32_t)prefix_ids.size(),cache_key);
                    cache_key[40]='\0';
                    const std::string id="prewarm-metal-"+runtime.boot_id+"-"+
                        std::to_string((long)req_counter++);
                    const Runtime::Outcome outcome=traced_engine("prefix_prewarm",id,[&]{
                        return runtime.run(prefix_ids,0,q27::SamplingParams{},
                            std::vector<std::string>{},
                            [](const std::string&){ return true; },
                            std::vector<std::string>{},false,{},id,true);
                    });
                    std::string retained_path;
                    uint32_t retained_tokens=0;
                    const bool retained=runtime.snapstore.best_match(
                        prefix_ids,retained_path,retained_tokens) &&
                        retained_tokens==prefix_ids.size();
                    if(retained) make_owner_private_no_acl(
                        retained_path,0600,false);
                    runtime.trace.event({{"kind","prefix_prewarm"},{"id",id},
                        {"api",api},{"prefix_tokens",(uint64_t)prefix_ids.size()},
                        {"request_tokens",(uint64_t)full_ids.size()},
                        {"existing_hit",outcome.exact_disk_hit},
                        {"retained",retained}});
                    if(!retained) {
                        json_response(response,{{"error",
                            "experimental prefix snapshot did not survive the configured budget"},
                            {"cache_key",cache_key},{"prefix_tokens",prefix_ids.size()}},507);
                        return;
                    }
                    json_response(response,{{"status","ok"},{"experimental",true},
                        {"api",api},{"cache_key",cache_key},
                        {"prefix_tokens",prefix_ids.size()},
                        {"request_prompt_tokens",full_ids.size()},
                        {"already_cached",outcome.exact_disk_hit},
                        {"snapshot_written",outcome.exact_snapshot_written}});
                } catch(const Runtime::ServerOverloaded& e) {
                    json_response(response,{{"error",e.what()}},503);
                } catch(const Runtime::EngineError& e) {
                    json_response(response,{{"error",e.what()}},500);
                } catch(const std::exception& e) {
                    json_response(response,{{"error",e.what()}},400);
                }
            });

        // ---- OpenAI /v1/completions (raw continuation; no template, no
        // tool protocol) ----
        server.Post("/v1/completions",guarded("completions",[&](const json& body,httplib::Response& r,socket_t sock){
            auto ids=to_u32(runtime.tokenizer.encode(q27::jstr(body,"prompt")));
            uint32_t n=max_tokens(body,8192); // unified default (upstream v0.4.0)
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop");
            const bool include_usage=q27::openai_stream_includes_usage(body);
            const std::vector<std::string> tnames=tool_names_from(body);
            const std::string id="cmpl-metal-"+runtime.boot_id+"-"+std::to_string((long)req_counter++);
            const long created=unix_now();
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            uint32_t maxp=0;
            if(prompt_overflow(ids.size(),n,maxp)) {
                runtime.trace.event({{"kind","request"},{"api","completions"},{"id",id},
                    {"validation_error",true},{"prompt_tokens",(uint64_t)ids.size()}});
                runtime.trace.event({{"kind","error"},{"api","completions"},{"status",400},{"type","context_length_exceeded"},
                    {"id",id},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                json_response(r,{{"error",{{"message",q27::ctx_limit_error_message((int)ids.size(),(int)maxp)},
                    {"type","invalid_request_error"},{"code","context_length_exceeded"}}}},400);
                return;
            }
            if(runtime.trace.enabled())
                // q27::jstr(body,"prompt") repeats the encode line's identical
                // accessor verbatim, so this event adds no new throw path
                // (codex P2 on the trace round, rejected with this evidence).
                // jstr never throws at all: a null/non-string prompt reads as
                // absent, the encode above yields no ids, and the empty-prompt
                // check at the top of this handler has already 400'd.
                runtime.trace.event({{"kind","request"},{"api","completions"},{"id",id},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"sampling",{{"temperature",sampling.temperature},{"top_p",sampling.top_p},
                        {"top_k",sampling.top_k},{"seed",sampling.seed}}},{"stops",stops},{"tool_names",tnames},
                    {"snapshot",q27::jbool(body,"snapshot",false)},
                    {"token_head",trace_token_head(ids)},{"rendered",trace_text(q27::jstr(body,"prompt"))}});
            if(!wants_stream(body)) {
                std::string text; size_t probe=0;
                auto outcome=traced_engine("completions",id,[&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ text+=piece;
                            return (++probe&15)?true:socket_alive(sock); },
                        tnames,q27::jbool(body,"snapshot",false),socket_live(sock),id); });
                if(outcome.finish==Runtime::Finish::Cancelled) { r.status=499; return; }
                runtime.trace.event({{"kind","outcome"},{"api","completions"},{"id",id},
                    {"finish",openai_finish(outcome.finish)},{"terminal",trace_finish(outcome.finish)},
                    {"prompt_tokens",outcome.prompt_tokens},
                    {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                json_response(r,{{"id",id},{"object","text_completion"},{"created",created},{"model","q27-metal"},
                    {"choices",json::array({{{"index",0},{"text",text},{"finish_reason",openai_finish(outcome.finish)}}})},
                    {"usage",{{"prompt_tokens",outcome.prompt_tokens},{"completion_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}});
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            const bool snap_hint=q27::jbool(body,"snapshot",false);
            auto stream_active=std::make_shared<Runtime::StreamScope>(runtime);
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,id,created,tnames,snap_hint,sock,stream_active,include_usage](size_t,httplib::DataSink& sink)->bool {
                    (void)stream_active;
                    try {
                        auto emit=[&](const std::string& piece)->bool {
                            json event=q27::openai_stream_chunk(
                                false,id,"text_completion",created,"q27-metal",piece);
                            if(include_usage) event["usage"]=nullptr;
                            std::string s=q27::sse_data(event);
                            return sink.write(s.data(),s.size());
                        };
                        auto outcome=runtime.guard_engine([&]{
                            return runtime.run(ids,n,sampling,stops,emit,tnames,snap_hint,
                                [sock]{ return socket_alive(sock); },id);
                        });
                        if(outcome.finish==Runtime::Finish::Cancelled) { sink.done(); return false; }
                        // Terminal chunk with a real finish_reason before [DONE]
                        // (parity with server.cu security-review fix #7).
                        json final_event=q27::openai_stream_final_chunk(
                            false,id,"text_completion",created,"q27-metal",openai_finish(outcome.finish));
                        if(include_usage) final_event["usage"]=nullptr;
                        std::string fin=q27::sse_data(final_event);
                        sink.write(fin.data(),fin.size());
                        if(include_usage) {
                            std::string usage=q27::sse_data(q27::openai_stream_usage_chunk(
                                id,"text_completion",created,"q27-metal",
                                outcome.prompt_tokens,outcome.output_tokens));
                            sink.write(usage.data(),usage.size());
                        }
                        std::string done=q27::sse_done(); sink.write(done.data(),done.size());
                        runtime.trace.event({{"kind","outcome"},{"api","completions"},{"id",id},
                            {"finish",openai_finish(outcome.finish)},{"terminal",trace_finish(outcome.finish)},
                            {"prompt_tokens",outcome.prompt_tokens},
                            {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                    } catch(const Runtime::ClientGone&) {
                        return false;
                    } catch(const Runtime::ServerOverloaded& e) {
                        runtime.trace.event({{"kind","error"},{"api","completions"},{"id",id},{"status",503},{"type","overloaded_error"},{"message",e.what()}});
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","overloaded_error"}}}});
                        sink.write(s.data(),s.size());
                    } catch(const std::exception& e) {
                        runtime.trace.event({{"kind","error"},{"api","completions"},{"id",id},{"status",500},{"type","api_error"},{"message",e.what()}});
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","api_error"}}}});
                        sink.write(s.data(),s.size());
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- OpenAI /v1/chat/completions ----
        // Structured tool traffic both directions (agentic-parity round,
        // docs/metal/plans/2026-07-17-metal-agentic-parity.md): incoming
        // assistant.tool_calls / role:"tool" via openai_msgs above; outgoing
        // <tool_call> segments become message.tool_calls (non-streaming) or
        // one delta.tool_calls chunk per call (streaming) with finish_reason
        // "tool_calls"; <think> segments go to reasoning_content (llama.cpp
        // convention) instead of leaking raw into content.
        server.Post("/v1/chat/completions",guarded("chat",[&](const json& body,httplib::Response& r,socket_t sock){
            bool think_req=q27::resolve_think(body,think_default,req_think);
            const q27::ToolChoice tchoice=q27::parse_tool_choice(body);
            q27::OpenAIToolSelection selected=q27::select_openai_tools(body,tchoice);
            const json tools=std::move(selected.tools);
            const std::vector<std::string> declared_tool_names=std::move(selected.names);
            const std::unordered_set<std::string> allowed_tool_names(
                declared_tool_names.begin(),declared_tool_names.end());
            const bool has_tools=!tools.empty();
            if(tchoice.mode==q27::ToolChoice::FORCED) think_req=false;
            std::string rendered=q27::chatml_prompt(openai_msgs(body),tools,think_req);
            if(tchoice.mode==q27::ToolChoice::FORCED) rendered+="<tool_call>\n";
            auto ids=to_u32(runtime.tokenizer.encode(rendered));
            uint32_t n=max_tokens(body,8192); // unified default (upstream v0.4.0)
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop");
            const bool include_usage=q27::openai_stream_includes_usage(body);
            const long rid=req_counter++;
            const std::string id="chatcmpl-metal-"+runtime.boot_id+"-"+std::to_string(rid);
            const long created=unix_now();
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            uint32_t maxp=0;
            if(prompt_overflow(ids.size(),n,maxp)) {
                runtime.trace.event({{"kind","request"},{"api","chat"},{"id",id},
                    {"validation_error",true},{"prompt_tokens",(uint64_t)ids.size()}});
                runtime.trace.event({{"kind","error"},{"api","chat"},{"status",400},{"type","context_length_exceeded"},
                    {"id",id},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                json_response(r,{{"error",{{"message",q27::ctx_limit_error_message((int)ids.size(),(int)maxp)},
                    {"type","invalid_request_error"},{"code","context_length_exceeded"}}}},400);
                return;
            }
            const std::vector<std::string> tnames=declared_tool_names;
            const bool snap_hint=q27::jbool(body,"snapshot",false);
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","chat"},{"id",id},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"tools",(uint64_t)tools.size()},
                    {"sampling",{{"temperature",sampling.temperature},{"top_p",sampling.top_p},
                        {"top_k",sampling.top_k},{"seed",sampling.seed}}},{"stops",stops},{"tool_names",tnames},
                    {"snapshot",snap_hint},{"token_head",trace_token_head(ids)},{"rendered",trace_text(rendered)}});
            if(!wants_stream(body)) {
                q27::StreamSplitter sp;
                if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                std::string think_buf,text,tool_buf;
                std::vector<q27::ToolCall> calls;
                auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                    if(ch==q27::StreamSplitter::TOOL) {
                        if(tchoice.mode==q27::ToolChoice::NONE) text+=t;
                        else tool_buf+=t;
                        return;
                    }
                    if(!tool_buf.empty()) {
                        calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                        tool_buf.clear();
                    }
                    (ch==q27::StreamSplitter::THINK?think_buf:text)+=t;
                };
                size_t probe=0;
                auto outcome=traced_engine("chat",id,[&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ for(auto& [ch,t]:sp.feed(piece)) route(ch,t);
                            return (++probe&15)?true:socket_alive(sock); },
                        tnames,snap_hint,socket_live(sock),id); });
                if(outcome.finish==Runtime::Finish::Cancelled) { r.status=499; return; }
                for(auto& [ch,t]:sp.flush()) route(ch,t);
                if(!tool_buf.empty()) calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                std::string th=q27::strip_ws2(think_buf),tx=q27::strip_ws2(text);
                // Malformed wrapped calls surface as text so nothing is lost;
                // then the wrapper-less recovery chain runs over the text.
                std::vector<q27::ToolCall> good;
                for(auto& c:calls) {
                    if(c.ok && allowed_tool_names.count(c.name) &&
                       (tchoice.forced_name.empty() || c.name==tchoice.forced_name))
                        good.push_back(std::move(c));
                    else tx+=(tx.empty()?"":"\n")+c.raw;
                }
                if(has_tools) {
                    std::string pre;
                    auto bcs=q27::parse_bare_tool_calls(
                        tx,&pre,&tools,outcome.finish!=Runtime::Finish::Length);
                    if(!bcs.empty()) {
                        tx=pre;
                        size_t recovered=0;
                        for(auto& bc:bcs) {
                            if(allowed_tool_names.count(bc.name) &&
                               (tchoice.forced_name.empty() || bc.name==tchoice.forced_name)) {
                                good.push_back(std::move(bc));
                                recovered++;
                            } else {
                                tx+=(tx.empty()?"":"\n")+bc.raw;
                            }
                        }
                        if(recovered) {
                            fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (chat nonstream)\n",recovered);
                            runtime.trace.event({{"kind","tool_recovery"},{"api","chat"},{"id",id},{"stream",false},{"count",recovered}});
                        }
                    }
                }
                json tcs=json::array();
                int ci=0;
                for(auto& c:good)
                    tcs.push_back({{"id","call_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+std::to_string(ci++)},
                                   {"type","function"},
                                   {"function",{{"name",c.name},{"arguments",c.arguments.dump()}}}});
                json message={{"role","assistant"},
                              {"content",(!tcs.empty() && tx.empty())?json(nullptr):json(tx)}};
                if(!th.empty()) message["reasoning_content"]=th;
                if(!tcs.empty()) message["tool_calls"]=tcs;
                if(tchoice.mode==q27::ToolChoice::FORCED && tcs.empty() &&
                   outcome.finish!=Runtime::Finish::Length)
                    throw Runtime::EngineError("model produced no eligible tool call for forced tool_choice");
                const bool calls_complete=!tcs.empty() && outcome.finish!=Runtime::Finish::Length;
                runtime.trace.event({{"kind","outcome"},{"api","chat"},{"id",id},
                    {"finish",calls_complete?"tool_calls":openai_finish(outcome.finish)},
                    {"terminal",trace_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                    {"prefix_hit",outcome.prefix_hit}});
                json_response(r,{{"id",id},{"object","chat.completion"},{"created",created},{"model","q27-metal"},
                    {"choices",json::array({{{"index",0},{"message",message},
                        {"finish_reason",calls_complete?"tool_calls":openai_finish(outcome.finish)}}})},
                    {"usage",{{"prompt_tokens",outcome.prompt_tokens},{"completion_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}});
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            auto stream_active=std::make_shared<Runtime::StreamScope>(runtime);
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,id,rid,created,tools,has_tools,tnames,allowed_tool_names,snap_hint,sock,stream_active,think_req,tchoice,include_usage](size_t,httplib::DataSink& sink)->bool {
                    (void)stream_active;
                    bool alive=true;
                    auto last_wire=std::chrono::steady_clock::now();
                    auto chunk=[&](const json& delta,const json& finish){
                        json event={{"id",id},{"object","chat.completion.chunk"},
                            {"created",created},{"model","q27-metal"},
                            {"choices",json::array({{{"index",0},{"delta",delta},{"finish_reason",finish}}})}};
                        if(include_usage) event["usage"]=nullptr;
                        std::string s=q27::sse_data(event);
                        if(!sink.write(s.data(),s.size())) alive=false;
                        else last_wire=std::chrono::steady_clock::now();
                        return alive;
                    };
                    // Keepalive through silent stretches: agent clients run
                    // stall detectors (pi disconnected an 18 s hush,
                    // 2026-07-17), and the longest hushes are BEFORE the
                    // first token — queue wait behind a long turn, then a
                    // cold prefill of tens of seconds at big contexts. Fired
                    // from the liveness probe (250 ms queue ticks + every
                    // prefill chunk) and from the per-token callback. An
                    // empty delta is wire-legal and ignored by clients.
                    auto keepalive=[&]{
                        if(alive && std::chrono::steady_clock::now()-last_wire>std::chrono::seconds(5))
                            chunk(json::object(),nullptr);
                    };
                    try {
                        // Opening role delta (OpenAI streaming convention);
                        // also the client-gone probe before generation starts.
                        if(!chunk({{"role","assistant"},{"content",""}},nullptr)) { sink.done(); return true; }
                        q27::StreamSplitter sp;
                        if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                        else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                        std::string tool_buf,text_accum;
                        int tool_counter=0;
                        bool any_call=false,all_calls_clean=true,reject_stream_call=false;
                        auto emit_call=[&](const q27::ToolCall& c,bool raw_already_streamed=false){
                            if(!allowed_tool_names.count(c.name) ||
                               (!tchoice.forced_name.empty() && c.name!=tchoice.forced_name)) {
                                if(!raw_already_streamed) chunk({{"content",c.raw}},nullptr);
                                return;
                            }
                            any_call=true;
                            chunk({{"tool_calls",json::array({{{"index",tool_counter},
                                {"id","call_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+std::to_string(tool_counter)},
                                {"type","function"},
                                {"function",{{"name",c.name},{"arguments",c.arguments.dump()}}}}})}},nullptr);
                            tool_counter++;
                        };
                        auto emit_tool=[&](){
                            auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf));
                            tool_buf.clear();
                            if(!c.ok) { // malformed: surface as text so nothing is lost
                                text_accum+=c.raw;
                                chunk({{"content",c.raw}},nullptr);
                                return;
                            }
                            emit_call(c);
                        };
                        // Incremental argument streaming (pre-registered
                        // 2026-07-17-incremental-tool-call-streaming.md):
                        // wrapped calls stream production-shape as they
                        // generate; deviant heads fall back to the buffered
                        // emit_tool recovery path above, raw byte-exact.
                        q27::ToolCallStreamer ts;
                        auto tool_frag_chunk=[&](const std::string& frag){
                            if(frag.empty()) return;
                            chunk({{"tool_calls",json::array({{{"index",tool_counter},
                                {"function",{{"arguments",frag}}}}})}},nullptr);
                        };
                        auto recover_trail=[&](const std::string& raw,bool allow_repair){
                            const std::string tr=q27::strip_ws2(raw);
                            if(tr.empty()) return;
                            std::string pre;
                            auto bcs=q27::parse_bare_tool_calls(tr,&pre,&tools,allow_repair);
                            if(!pre.empty()) { text_accum+=pre; chunk({{"content",pre}},nullptr); }
                            if(!bcs.empty()) {
                                fprintf(stderr,"[tool-stream] %zu trailing call(s) recovered after streamed call\n",bcs.size());
                                runtime.trace.event({{"kind","tool_recovery"},{"api","chat"},{"id",id},
                                    {"stream",true},{"trailing",true},{"count",bcs.size()}});
                                for(const auto& bc:bcs) emit_call(bc);
                            } else if(pre.empty() && tr.find_first_not_of("}] \t\r\n")!=std::string::npos) {
                                text_accum+=tr; chunk({{"content",tr}},nullptr);
                            }
                        };
                        auto close_tool=[&](){
                            if(!ts.active()) return;
                            std::string tail;
                            const bool clean=ts.finalize(&tail);
                            if(reject_stream_call) {
                                std::string rejected=ts.raw;
                                const std::string trailing=ts.trail();
                                if(!trailing.empty() && trailing.size()<=rejected.size() &&
                                   rejected.compare(rejected.size()-trailing.size(),trailing.size(),trailing)==0)
                                    rejected.resize(rejected.size()-trailing.size());
                                if(!rejected.empty()) chunk({{"content",rejected}},nullptr);
                                recover_trail(trailing,true);
                                reject_stream_call=false;
                                ts.reset();
                                return;
                            }
                            if(ts.invalid()) {
                                // Streaming is irreversible: the opener and
                                // prior argument deltas are already on wire.
                                // Do not duplicate them as raw buffered text;
                                // mark the call unclean so the terminal reason
                                // remains stop/length and clients do not execute.
                                all_calls_clean=false;
                                tool_counter++; // opener/index was already emitted
                                recover_trail(ts.trail(),false);
                            } else if(ts.opened) {
                                tool_frag_chunk(tail);
                                if(!clean) {
                                    all_calls_clean=false;
                                    fprintf(stderr,"[tool-stream] streamed call closed "
                                            "unbalanced (production semantics, sent as-is)\n");
                                }
                                tool_counter++;
                                // A wrapper can pack more than one call: bytes
                                // after the streamed call's arguments closed
                                // are not framing. Recover them through the
                                // bare-call chain and emit whole (review
                                // 2026-07-17: the DONE-state byte drop lost
                                // every call after the first, silently).
                                recover_trail(ts.trail(),true);
                            } else {
                                tool_buf=ts.raw;
                                emit_tool();
                            }
                            ts.reset();
                        };
                        auto emit_seg=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                            if(ch==q27::StreamSplitter::TOOL) {
                                if(tchoice.mode==q27::ToolChoice::NONE) {
                                    text_accum+=t;
                                    chunk({{"content",t}},nullptr);
                                    return;
                                }
                                if(!tchoice.forced_name.empty()) {
                                    tool_buf+=t;
                                    return;
                                }
                                bool opened=false;
                                const std::string frag=ts.feed(t,&opened);
                                if(opened) {
                                    reject_stream_call=!allowed_tool_names.count(ts.name);
                                    if(!reject_stream_call) {
                                        any_call=true;
                                        chunk({{"tool_calls",json::array({{{"index",tool_counter},
                                            {"id","call_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+std::to_string(tool_counter)},
                                            {"type","function"},
                                            {"function",{{"name",ts.name},{"arguments",""}}}}})}},nullptr);
                                    }
                                }
                                if(!reject_stream_call) tool_frag_chunk(frag);
                                return;
                            }
                            if(!tool_buf.empty()) emit_tool();
                            close_tool();
                            if(t.empty()) return;
                            if(ch==q27::StreamSplitter::THINK) chunk({{"reasoning_content",t}},nullptr);
                            else { text_accum+=t; chunk({{"content",t}},nullptr); }
                        };
                        auto outcome=runtime.guard_engine([&]{
                            return runtime.run(ids,n,sampling,stops,
                                [&](const std::string& piece)->bool {
                                    for(auto& [ch,t]:sp.feed(piece)) emit_seg(ch,t);
                                    keepalive();
                                    return alive && sink.is_writable();
                                },tnames,snap_hint,
                                [&,sock]{ keepalive();
                                          return socket_alive(sock); },id);
                        });
                        if(outcome.finish==Runtime::Finish::Cancelled) { sink.done(); return false; }
                        for(auto& [ch,t]:sp.flush()) emit_seg(ch,t);
                        close_tool();               // wrapper never closed: finalize
                        if(!tool_buf.empty()) emit_tool();
                        if(has_tools) {
                            // Wrapper-less recovery: the text already streamed
                            // as content deltas (cosmetic); the tool_calls
                            // chunks still fire so the client can execute.
                            std::string pre;
                            auto bcs=q27::parse_bare_tool_calls(
                                text_accum,&pre,&tools,outcome.finish!=Runtime::Finish::Length);
                            if(!bcs.empty()) {
                                fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (chat stream)\n",bcs.size());
                                runtime.trace.event({{"kind","tool_recovery"},{"api","chat"},{"id",id},{"stream",true},{"count",bcs.size()}});
                            }
                            for(const auto& bc:bcs) emit_call(bc,true);
                        }
                        if(tchoice.mode==q27::ToolChoice::FORCED &&
                           (!any_call || !all_calls_clean) &&
                           outcome.finish!=Runtime::Finish::Length)
                            throw Runtime::EngineError(
                                "model produced no eligible tool call for forced tool_choice");
                        const bool calls_complete=any_call && all_calls_clean &&
                            outcome.finish!=Runtime::Finish::Length;
                        chunk(json::object(),calls_complete?"tool_calls":openai_finish(outcome.finish));
                        if(include_usage) {
                            std::string usage=q27::sse_data(q27::openai_stream_usage_chunk(
                                id,"chat.completion.chunk",created,"q27-metal",
                                outcome.prompt_tokens,outcome.output_tokens));
                            sink.write(usage.data(),usage.size());
                        }
                        std::string done=q27::sse_done(); sink.write(done.data(),done.size());
                        runtime.trace.event({{"kind","outcome"},{"api","chat"},{"id",id},
                            {"finish",calls_complete?"tool_calls":openai_finish(outcome.finish)},
                            {"terminal",trace_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                            {"prefix_hit",outcome.prefix_hit}});
                    } catch(const Runtime::ClientGone&) {
                        return false;
                    } catch(const Runtime::ServerOverloaded& e) {
                        runtime.trace.event({{"kind","error"},{"api","chat"},{"id",id},{"status",503},{"type","overloaded_error"},{"message",e.what()}});
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","overloaded_error"}}}});
                        sink.write(s.data(),s.size());
                    } catch(const std::exception& e) {
                        runtime.trace.event({{"kind","error"},{"api","chat"},{"id",id},{"status",500},{"type","api_error"},{"message",e.what()}});
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","api_error"}}}});
                        sink.write(s.data(),s.size());
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- Anthropic /v1/messages ----
        // Full agentic parity with src/server.cu:1087-1439 (2026-07-17 round):
        // request mapping via anthropic_msgs/anthropic_tools_json (incoming
        // tool_use/tool_result/thinking reconstructed, billing header
        // normalized), StreamSplitter output routing into thinking / text /
        // tool_use content blocks with input_json_delta streaming, bare-call
        // recovery, stop_reason "tool_use", and the "prompt is too long"
        // context refusal Claude Code keys compaction off.

        // CC calls count_tokens before compaction decisions; a 404 means it
        // estimates blind. Count = exactly what /v1/messages prefills for the
        // same body. CPU-only: no slot, no GPU lease.
        server.Post("/v1/messages/count_tokens",anthropic_guarded("count_tokens",[&](const json& body,httplib::Response& r,socket_t){
            const std::string id="count_metal_"+runtime.boot_id+"_"+
                std::to_string((long)req_counter++);
            if(!body.contains("messages") || !body["messages"].is_array()) {
                runtime.trace.event({{"kind","request"},{"api","count_tokens"},{"id",id},{"validation_error",true}});
                runtime.trace.event({{"kind","error"},{"api","count_tokens"},{"id",id},{"status",400},
                    {"type","invalid_request_error"},{"message","messages: Field required"}});
                r.status=400;
                r.set_content(q27::anthropic_error_json("invalid_request_error","messages: Field required"),
                              "application/json");
                return;
            }
            const std::string rendered=q27::chatml_prompt(
                q27::anthropic_msgs(body),q27::anthropic_tools_json(body),
                q27::resolve_think(body,think_default,req_think));
            const long input_tokens=(long)runtime.tokenizer.encode(rendered).size();
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","count_tokens"},{"id",id},
                    {"prompt_tokens",(uint64_t)input_tokens},{"rendered",trace_text(rendered)}});
            json_response(r,{{"input_tokens",input_tokens}});
            runtime.trace.event({{"kind","outcome"},{"api","count_tokens"},{"id",id},
                                 {"finish","counted"},{"output_tokens",0}});
        }));

        server.Post("/v1/messages",anthropic_guarded("messages",[&](const json& body,httplib::Response& r,socket_t sock){
            const json tools=q27::anthropic_tools_json(body);
            const bool think_req=q27::resolve_think(body,think_default,req_think);
            const std::string rendered=q27::chatml_prompt(q27::anthropic_msgs(body),tools,think_req);
            auto ids=to_u32(runtime.tokenizer.encode(rendered));
            uint32_t n=max_tokens(body,8192); // unified default (upstream v0.4.0)
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop_sequences");
            const long rid=req_counter++;
            const std::string mid="msg_metal_"+runtime.boot_id+"_"+std::to_string(rid);
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            uint32_t maxp=0;
            if(prompt_overflow(ids.size(),n,maxp)) {
                fprintf(stderr,"[ctx-limit] prompt=%zu max=%u -> 400\n",ids.size(),maxp);
                runtime.trace.event({{"kind","request"},{"api","messages"},{"id",mid},
                    {"validation_error",true},{"prompt_tokens",(uint64_t)ids.size()}});
                runtime.trace.event({{"kind","error"},{"api","messages"},{"status",400},{"type","context_length_exceeded"},
                    {"id",mid},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                r.status=400;
                r.set_content(q27::anthropic_error_json("invalid_request_error",
                    q27::ctx_limit_error_message((int)ids.size(),(int)maxp)),"application/json");
                return;
            }
            const bool has_tools=tools.is_array() && !tools.empty();
            const std::vector<std::string> tnames=tool_names_from(body);
            const bool snap_hint=q27::jbool(body,"snapshot",false);
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","messages"},{"id",mid},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"tools",(uint64_t)(has_tools?tools.size():0)},
                    {"sampling",{{"temperature",sampling.temperature},{"top_p",sampling.top_p},
                        {"top_k",sampling.top_k},{"seed",sampling.seed}}},{"stops",stops},{"tool_names",tnames},
                    {"snapshot",snap_hint},{"token_head",trace_token_head(ids)},{"rendered",trace_text(rendered)}});
            if(!wants_stream(body)) {
                q27::StreamSplitter sp;
                if(think_req) sp.chan=q27::StreamSplitter::THINK;
                std::string think,text,tool_buf;
                std::vector<q27::ToolCall> calls;
                auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                    if(ch==q27::StreamSplitter::TOOL) { tool_buf+=t; return; }
                    if(!tool_buf.empty()) {
                        calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                        tool_buf.clear();
                    }
                    (ch==q27::StreamSplitter::THINK?think:text)+=t;
                };
                size_t probe=0;
                auto outcome=traced_engine("messages",mid,[&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ for(auto& [ch,t]:sp.feed(piece)) route(ch,t);
                            return (++probe&15)?true:socket_alive(sock); },
                        tnames,snap_hint,socket_live(sock),mid); });
                if(outcome.finish==Runtime::Finish::Cancelled) { r.status=499; return; }
                for(auto& [ch,t]:sp.flush()) route(ch,t);
                if(!tool_buf.empty()) calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                json content=json::array();
                std::string th=q27::strip_ws2(think),tx=q27::strip_ws2(text);
                if(!th.empty())
                    content.push_back({{"type","thinking"},{"thinking",th},{"signature","q27-local"}});
                bool any_call=false;
                for(auto& c:calls) {
                    if(!c.ok) tx+=(tx.empty()?"":"\n")+c.raw; // malformed: keep as text
                    else any_call=true;
                }
                if(has_tools) {
                    // wrapper-less call recovery (see parse_bare_tool_calls)
                    std::string pre;
                    auto bcs=q27::parse_bare_tool_calls(
                        tx,&pre,&tools,outcome.finish!=Runtime::Finish::Length);
                    if(!bcs.empty()) {
                        fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (nonstream)\n",bcs.size());
                        runtime.trace.event({{"kind","tool_recovery"},{"api","messages"},{"id",mid},{"stream",false},{"count",bcs.size()}});
                        tx=pre;
                        for(auto& bc:bcs) calls.push_back(bc);
                        any_call=true;
                    }
                }
                if(!tx.empty() || (!any_call && th.empty()))
                    content.push_back({{"type","text"},{"text",tx}});
                int ci=0;
                for(auto& c:calls)
                    if(c.ok)
                        content.push_back({{"type","tool_use"},
                            {"id","toolu_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+std::to_string(ci++)},
                            {"name",c.name},{"input",c.arguments}});
                const bool calls_complete=any_call && outcome.finish!=Runtime::Finish::Length;
                json out={{"id",mid},{"type","message"},{"role","assistant"},{"model","q27-metal"},
                    {"content",content},
                    {"stop_reason",calls_complete?"tool_use":anthropic_stop(outcome.finish)},
                    {"stop_sequence",outcome.finish==Runtime::Finish::StopSequence?json(outcome.stop_sequence):json(nullptr)},
                    {"usage",{{"input_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}};
                runtime.trace.event({{"kind","outcome"},{"api","messages"},{"id",mid},
                    {"finish",calls_complete?"tool_use":anthropic_stop(outcome.finish)},
                    {"terminal",trace_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                    {"prefix_hit",outcome.prefix_hit}});
                json_response(r,out);
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            auto stream_active=std::make_shared<Runtime::StreamScope>(runtime);
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,mid,rid,tools,has_tools,tnames,snap_hint,sock,stream_active,think_req](size_t,httplib::DataSink& sink)->bool {
                    (void)stream_active;
                    bool alive=true;
                    auto last_wire=std::chrono::steady_clock::now();
                    auto ev=[&](const char* name,const json& j){
                        std::string s=q27::sse_event(name,j);
                        if(!sink.write(s.data(),s.size())) alive=false;
                        else last_wire=std::chrono::steady_clock::now();
                        return alive;
                    };
                    // Block bookkeeping mirrors server.cu's streaming handler:
                    // lazily opened think/text blocks, tool_use blocks emitted
                    // whole (start + one input_json_delta + stop) when a tool
                    // segment closes.
                    int block_counter=0,tool_counter=0,idx=-1,chan_open=-1;
                    bool any=false,any_call=false,all_calls_clean=true;
                    q27::StreamSplitter sp;
                    if(think_req) sp.chan=q27::StreamSplitter::THINK;
                    std::string tool_buf,text_accum;
                    auto close_block=[&](){
                        if(idx<0) return;
                        if(chan_open==1)
                            ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                                {"delta",{{"type","signature_delta"},{"signature","q27-local"}}}});
                        ev("content_block_stop",{{"type","content_block_stop"},{"index",idx}});
                        idx=-1;
                    };
                    auto open_block=[&](int chan){
                        if(idx>=0 && chan_open!=chan) close_block();
                        if(idx<0) {
                            idx=block_counter++;
                            json cb=chan==1?json{{"type","thinking"},{"thinking",""}}
                                           :json{{"type","text"},{"text",""}};
                            ev("content_block_start",{{"type","content_block_start"},
                                {"index",idx},{"content_block",cb}});
                            chan_open=chan;
                            any=true;
                        }
                    };
                    auto emit_tool_block=[&](const std::string& name,const json& args){
                        any_call=true;
                        close_block();
                        const int ti=block_counter++;
                        const std::string tid="toolu_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+
                                              std::to_string(tool_counter++);
                        ev("content_block_start",{{"type","content_block_start"},{"index",ti},
                            {"content_block",{{"type","tool_use"},{"id",tid},{"name",name},
                                              {"input",json::object()}}}});
                        ev("content_block_delta",{{"type","content_block_delta"},{"index",ti},
                            {"delta",{{"type","input_json_delta"},
                                      {"partial_json",q27::sse_dump(args)}}}});
                        ev("content_block_stop",{{"type","content_block_stop"},{"index",ti}});
                    };
                    auto emit_tool=[&](){
                        auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf));
                        tool_buf.clear();
                        if(!c.ok) { // malformed: surface as text so nothing is lost
                            open_block(0);
                            text_accum+=c.raw;
                            ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                                {"delta",{{"type","text_delta"},{"text",c.raw}}}});
                            return;
                        }
                        emit_tool_block(c.name,c.arguments);
                    };
                    // Incremental tool-call argument streaming (2026-07-18
                    // streaming-parity-messages-responses.md): wrapped calls
                    // stream production-shape input_json_delta fragments as
                    // they generate — the SAME ToolCallStreamer proven on
                    // /v1/chat/completions. Deviant heads fall back to the
                    // buffered emit_tool path above, raw byte-exact.
                    q27::ToolCallStreamer ts;
                    int cur_tool_idx=-1;   // content-block index of the in-flight streamed call
                    auto close_stream_block=[&](){
                        if(cur_tool_idx<0) return;
                        ev("content_block_stop",{{"type","content_block_stop"},{"index",cur_tool_idx}});
                        cur_tool_idx=-1;
                    };
                    auto recover_trail=[&](const std::string& raw,bool allow_repair){
                        const std::string tr=q27::strip_ws2(raw);
                        if(tr.empty()) return;
                        std::string pre;
                        auto bcs=q27::parse_bare_tool_calls(tr,&pre,&tools,allow_repair);
                        if(!pre.empty()) {
                            text_accum+=pre; open_block(0);
                            ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                                {"delta",{{"type","text_delta"},{"text",pre}}}});
                        }
                        if(!bcs.empty()) {
                            fprintf(stderr,"[tool-stream] %zu trailing call(s) recovered after streamed call\n",bcs.size());
                            runtime.trace.event({{"kind","tool_recovery"},{"api","messages"},{"id",mid},
                                {"stream",true},{"trailing",true},{"count",bcs.size()}});
                            for(auto& bc:bcs) emit_tool_block(bc.name,bc.arguments);
                        } else if(pre.empty() && tr.find_first_not_of("}] \t\r\n")!=std::string::npos) {
                            text_accum+=tr; open_block(0);
                            ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                                {"delta",{{"type","text_delta"},{"text",tr}}}});
                        }
                    };
                    auto close_tool=[&](){
                        if(!ts.active()) return;
                        std::string tail;
                        const bool clean=ts.finalize(&tail);
                        if(ts.invalid()) {
                            all_calls_clean=false;
                            // Streaming is irreversible: opener + prior arg
                            // deltas are already on wire; recover packed trail.
                            close_stream_block();
                            recover_trail(ts.trail(),false);
                        } else if(ts.opened) {
                            if(!tail.empty() && cur_tool_idx>=0)
                                ev("content_block_delta",{{"type","content_block_delta"},{"index",cur_tool_idx},
                                    {"delta",{{"type","input_json_delta"},{"partial_json",tail}}}});
                            if(!clean) {
                                all_calls_clean=false;
                                fprintf(stderr,"[tool-stream] streamed call closed unbalanced (production semantics, sent as-is)\n");
                            }
                            close_stream_block();
                            recover_trail(ts.trail(),true);
                        } else {
                            tool_buf=ts.raw;
                            emit_tool();
                        }
                        ts.reset();
                    };
                    auto emit_seg=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                        if(ch==q27::StreamSplitter::TOOL) {
                            bool opened=false;
                            const std::string frag=ts.feed(t,&opened);
                            if(opened) {
                                any_call=true;
                                close_block();
                                cur_tool_idx=block_counter++;
                                const std::string tid="toolu_metal_"+runtime.boot_id+"_"+std::to_string(rid)+"_"+
                                                      std::to_string(tool_counter++);
                                ev("content_block_start",{{"type","content_block_start"},{"index",cur_tool_idx},
                                    {"content_block",{{"type","tool_use"},{"id",tid},{"name",ts.name},
                                                      {"input",json::object()}}}});
                            }
                            if(!frag.empty() && cur_tool_idx>=0)
                                ev("content_block_delta",{{"type","content_block_delta"},{"index",cur_tool_idx},
                                    {"delta",{{"type","input_json_delta"},{"partial_json",frag}}}});
                            return;
                        }
                        close_tool();
                        if(t.empty()) return;
                        const int chan=ch==q27::StreamSplitter::THINK?1:0;
                        // suppress pure-whitespace text before/between blocks
                        if(chan==0 && idx<0 && q27::strip_ws2(t).empty()) return;
                        open_block(chan);
                        if(chan==0) text_accum+=t;
                        ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                            {"delta",chan==1?json{{"type","thinking_delta"},{"thinking",t}}
                                            :json{{"type","text_delta"},{"text",t}}}});
                    };
                    try {
                        json msg={{"id",mid},{"type","message"},{"role","assistant"},{"model","q27-metal"},
                            {"content",json::array()},{"stop_reason",nullptr},{"stop_sequence",nullptr},
                            {"usage",{{"input_tokens",(int)ids.size()},{"output_tokens",0}}}};
                        // A client gone before generation starts must not hold
                        // the engine through the token cap: gate the run on the
                        // opening write, and probe the socket on quiet pieces.
                        if(!ev("message_start",{{"type","message_start"},{"message",msg}})) {
                            sink.done();
                            return true;
                        }
                        // Keepalive through silent stretches (see the chat
                        // twin — queue wait + prefill + tool_buf buffering):
                        // Anthropic's wire has a documented ping event for
                        // exactly this.
                        auto keepalive=[&]{
                            if(alive && std::chrono::steady_clock::now()-last_wire>std::chrono::seconds(5))
                                ev("ping",{{"type","ping"}});
                        };
                        auto outcome=runtime.guard_engine([&]{
                            return runtime.run(ids,n,sampling,stops,
                                [&](const std::string& piece)->bool {
                                    for(auto& [ch,t]:sp.feed(piece)) emit_seg(ch,t);
                                    keepalive();
                                    return alive && sink.is_writable();
                                },tnames,snap_hint,
                                [&,sock]{ keepalive();
                                          return socket_alive(sock); },mid);
                        });
                        if(outcome.finish==Runtime::Finish::Cancelled) { sink.done(); return false; }
                        for(auto& [ch,t]:sp.flush()) emit_seg(ch,t);
                        close_tool();   // close any in-flight streamed call (finalize + trail recovery)
                        if(!tool_buf.empty()) emit_tool();
                        if(has_tools) {
                            // wrapper-less recovery: text already streamed as
                            // text_delta (cosmetic); tool_use blocks still fire
                            std::string pre;
                            auto bcs=q27::parse_bare_tool_calls(
                                text_accum,&pre,&tools,outcome.finish!=Runtime::Finish::Length);
                            if(!bcs.empty()) {
                                fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (stream)\n",bcs.size());
                                runtime.trace.event({{"kind","tool_recovery"},{"api","messages"},{"id",mid},{"stream",true},{"count",bcs.size()}});
                                any=true;
                                for(auto& bc:bcs) emit_tool_block(bc.name,bc.arguments);
                            }
                        }
                        if(idx<0 && !any) { // nothing at all: empty text block for validity
                            idx=block_counter++;
                            chan_open=0;
                            ev("content_block_start",{{"type","content_block_start"},{"index",idx},
                                {"content_block",{{"type","text"},{"text",""}}}});
                        }
                        close_block();
                        const bool calls_complete=any_call && all_calls_clean &&
                            outcome.finish!=Runtime::Finish::Length;
                        ev("message_delta",{{"type","message_delta"},
                            {"delta",{{"stop_reason",calls_complete?"tool_use":anthropic_stop(outcome.finish)},
                                      {"stop_sequence",outcome.finish==Runtime::Finish::StopSequence?json(outcome.stop_sequence):json(nullptr)}}},
                            {"usage",{{"output_tokens",outcome.output_tokens}}}});
                        ev("message_stop",{{"type","message_stop"}});
                        runtime.trace.event({{"kind","outcome"},{"api","messages"},{"id",mid},
                            {"finish",calls_complete?"tool_use":anthropic_stop(outcome.finish)},
                            {"terminal",trace_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                            {"prefix_hit",outcome.prefix_hit}});
                    } catch(const Runtime::ClientGone&) {
                        return false;
                    } catch(const Runtime::ServerOverloaded& e) {
                        runtime.trace.event({{"kind","error"},{"api","messages"},{"id",mid},{"status",503},{"type","overloaded_error"},{"message",e.what()}});
                        ev("error",{{"type","error"},{"error",{{"type","overloaded_error"},{"message",e.what()}}}});
                        ev("message_stop",{{"type","message_stop"}});
                    } catch(const std::exception& e) {
                        runtime.trace.event({{"kind","error"},{"api","messages"},{"id",mid},{"status",500},{"type","api_error"},{"message",e.what()}});
                        // codex P2 2026-07-19: an engine failure mid-stream must
                        // not leave an opened tool_use or text/thinking block
                        // unterminated before message_stop. close_tool() closes a
                        // streamed tool block (all streamer cases) and
                        // close_block() closes any open text/thinking block;
                        // both are no-ops when nothing is open.
                        try { close_tool(); } catch(...) {}
                        try { close_block(); } catch(...) {}
                        // First-class error event; message_stop still follows so
                        // naive clients get a well-formed stream (server.cu's
                        // batch-error convention).
                        ev("error",{{"type","error"},{"error",{{"type","api_error"},{"message",e.what()}}}});
                        ev("message_stop",{{"type","message_stop"}});
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- OpenAI Responses API (/v1/responses, Codex CLI) ----
        // Full CUDA port (src/server.cu:1441-1882, residue round
        // 2026-07-17-responses-parity-residue.md): instructions/input-item
        // mapping through the chat template (replacing the raw-text
        // preamble hack), custom freeform tools bridged to one-string-param
        // functions, hosted tool types skipped never rejected, and
        // structured function_call / custom_tool_call output items with the
        // codex 0.143 item lifecycle on the stream (an output_text.delta
        // without an open item aborts the codex turn). Wire facts from
        // codex-rs: the client keys off the JSON `type` field; the agent
        // loop consumes only response.output_item.done items;
        // response.completed{response:{id}} is the required terminator;
        // function_call.arguments is a JSON-encoded STRING. 400 is fatal
        // to codex, 500 retries — tolerate quirks, 500 on bugs.
        server.Post("/v1/responses",guarded("responses",[&](const json& body,httplib::Response& r,socket_t sock){
            const long rn=req_counter++;
            const std::string resp_id="resp_metal_"+runtime.boot_id+"_"+std::to_string(rn);
            const std::string msg_id="msg_metal_"+runtime.boot_id+"_"+std::to_string(rn);
            // Canonicalization is shared with the experimental prewarmer so
            // a cache pack can never encode a different interpretation of a
            // Responses request than the ordinary serving path.
            ResponsesPromptInput normalized=responses_prompt_input(body);
            json tools=std::move(normalized.tools);
            std::set<std::string> custom_names=std::move(normalized.custom_names);
            const q27::ToolChoice tchoice=std::move(normalized.choice);
            const std::vector<std::string> tnames=std::move(normalized.tool_names);
            const std::set<std::string> allowed_tool_names(tnames.begin(),tnames.end());
            const std::set<std::string> allowed_hosted_names=
                std::move(normalized.allowed_hosted_names);
            std::vector<std::string> grammar_tool_names=tnames;
            grammar_tool_names.insert(grammar_tool_names.end(),allowed_hosted_names.begin(),
                                      allowed_hosted_names.end());
            std::vector<q27::Msg> merged=std::move(normalized.messages);
            bool think_req=q27::resolve_think(body,think_default,req_think);
            if(tchoice.mode==q27::ToolChoice::FORCED) think_req=false;
            std::string rendered=q27::chatml_prompt(merged,tools,think_req);
            if(tchoice.mode==q27::ToolChoice::FORCED) rendered+="<tool_call>\n";
            auto ids=to_u32(runtime.tokenizer.encode(rendered));
            uint32_t n=max_tokens(body,8192); // unified default (upstream v0.4.0)
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop");
            if(ids.empty()) throw std::runtime_error("input is empty");
            uint32_t maxp=0;
            if(prompt_overflow(ids.size(),n,maxp)) {
                // context_length_exceeded is fatal-class for codex, correctly
                runtime.trace.event({{"kind","request"},{"api","responses"},{"id",resp_id},
                    {"validation_error",true},{"prompt_tokens",(uint64_t)ids.size()}});
                runtime.trace.event({{"kind","error"},{"api","responses"},{"status",400},{"type","context_length_exceeded"},
                    {"id",resp_id},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                json_response(r,{{"error",{{"code","context_length_exceeded"}}}},400);
                return;
            }
            const bool snap_hint=q27::jbool(body,"snapshot",false);
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","responses"},{"id",resp_id},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"tools",(uint64_t)tools.size()},
                    {"sampling",{{"temperature",sampling.temperature},{"top_p",sampling.top_p},
                        {"top_k",sampling.top_k},{"seed",sampling.seed}}},{"stops",stops},{"tool_names",grammar_tool_names},
                    {"snapshot",snap_hint},{"token_head",trace_token_head(ids)},{"rendered",trace_text(rendered)}});
            if(!wants_stream(body)) {
                json items=json::array();
                int tool_counter=0,message_counter=0,reason_counter=0;
                std::string think,text,tool_buf,text_accum;
                auto flush_think=[&](bool incomplete_item=false){
                    std::string th=q27::strip_ws2(think); think.clear();
                    if(th.empty()) return;
                    items.push_back({{"type","reasoning"},{"id","rs_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(reason_counter++)},
                        {"status",incomplete_item?"incomplete":"completed"},{"summary",json::array({{{"type","summary_text"},{"text",th}}})},
                        {"encrypted_content",nullptr}});
                };
                auto push_call=[&](const std::string& name,const json& args,bool incomplete_item=false){
                    if(!responses_tool_allowed(name,allowed_tool_names,allowed_hosted_names)) return;
                    const int call_index=tool_counter++;
                    const std::string cid="call_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                    const std::string iid="fc_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                    if(custom_names.count(name)) {
                        std::string input=args.is_object() && args.contains("input") && args["input"].is_string()
                                              ?args["input"].get<std::string>():args.dump();
                        items.push_back({{"type","custom_tool_call"},{"id",iid},{"call_id",cid},
                            {"status",incomplete_item?"incomplete":"completed"},
                            {"name",name},{"input",input}});
                    } else
                        items.push_back({{"type","function_call"},{"id",iid},{"call_id",cid},
                                         {"status",incomplete_item?"incomplete":"completed"},
                                         {"name",name},{"arguments",args.dump()}});
                };
                auto flush_text=[&](bool final_turn,bool incomplete_item=false){
                    std::string tx=q27::strip_ws2(text); text.clear();
                    if(tx.empty()) return;
                    // Wrapper-less recovery rides every text commit. If any
                    // parsed call is ineligible, preserve the entire segment
                    // byte-for-byte as text: the parser does not expose source
                    // spans, so partially recovering would reorder calls or
                    // discard prose after them.
                    std::string pre;
                    auto bcs=q27::parse_bare_tool_calls(tx,&pre,
                                                        tools.empty()?nullptr:&tools,
                                                        final_turn && !incomplete_item);
                    std::vector<q27::ToolCall> eligible_calls;
                    if(!bcs.empty()) {
                        bool all_eligible=true;
                        for(const auto& call:bcs)
                            all_eligible=all_eligible && responses_tool_allowed(
                                call.name,allowed_tool_names,allowed_hosted_names);
                        if(all_eligible) {
                            tx=pre;
                            eligible_calls=std::move(bcs);
                            fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (resp nonstream)\n",eligible_calls.size());
                            runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},{"stream",false},{"count",eligible_calls.size()}});
                        }
                    }
                    if(!tx.empty())
                        items.push_back({{"type","message"},{"id",msg_id+"_"+std::to_string(message_counter++)},{"role","assistant"},
                            {"status",incomplete_item?"incomplete":"completed"},
                            {"content",json::array({{{"type","output_text"},{"text",tx},
                                                     {"annotations",json::array()}}})}});
                    for(auto& bc:eligible_calls) push_call(bc.name,bc.arguments,incomplete_item);
                };
                auto flush_tool=[&](bool final_turn,bool incomplete_item=false){
                    auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf)); tool_buf.clear();
                    if(!c.ok || !responses_tool_allowed(c.name,allowed_tool_names,
                                                       allowed_hosted_names)) { // malformed/ineligible: surface as text
                        // convention): recovery runs over it, so a
                        // recoverable nested call is not lost as raw text
                        // (codex P2 round 3) — and committing it alone means
                        // the tx=pre trim can never drop prose that followed
                        // the wrapper in the model's output (codex P2 round 4).
                        text+=(text.empty()?"":"\n")+c.raw;
                        flush_text(final_turn,incomplete_item); return;
                    }
                    push_call(c.name,c.arguments,incomplete_item);
                };
                q27::StreamSplitter sp;
                if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                    if(ch==q27::StreamSplitter::TOOL) {
                        if(tchoice.mode==q27::ToolChoice::NONE) {
                            text+=t; text_accum+=t; return;
                        }
                        if(!think.empty()) flush_think();
                        if(!text.empty()) flush_text(false);
                        tool_buf+=t; return;
                    }
                    if(!tool_buf.empty()) flush_tool(false);
                    // codex P2: close pending text BEFORE think accumulates —
                    // same transition rule as TOOL, else a text→think→text
                    // turn interleaves items (reasoning pushed ahead of the
                    // message it followed) and the stream twin corrupts
                    // output_index.
                    if(ch==q27::StreamSplitter::THINK) { if(!text.empty()) flush_text(false); think+=t; return; }
                    if(!think.empty()) flush_think();
                    text+=t; text_accum+=t;
                };
                size_t probe=0;
                auto outcome=traced_engine("responses",resp_id,[&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ for(auto& [ch,t]:sp.feed(piece)) route(ch,t);
                            return (++probe&15)?true:socket_alive(sock); },
                        grammar_tool_names,snap_hint,socket_live(sock),resp_id); });
                if(outcome.finish==Runtime::Finish::Cancelled) {
                    runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                        {"finish","cancelled"},{"terminal","cancelled"},{"prompt_tokens",outcome.prompt_tokens},
                        {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                    r.status=499;
                    return;
                }
                const bool incomplete=outcome.finish==Runtime::Finish::Length;
                for(auto& [ch,t]:sp.flush()) route(ch,t);
                if(!tool_buf.empty()) flush_tool(true,incomplete);
                flush_think(incomplete);
                flush_text(true,incomplete);
                if(tchoice.mode==q27::ToolChoice::FORCED && tool_counter==0 && !incomplete)
                    throw Runtime::EngineError("model produced no eligible tool call for forced tool_choice");
                std::string all_text;
                for(const auto& it:items)
                    if(it.value("type","")=="message" && it.contains("content"))
                        for(const auto& c:it["content"])
                            if(c.value("type","")=="output_text") all_text+=c.value("text","");
                const char* status=incomplete?"incomplete":"completed";
                runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                    {"finish",status},{"terminal",trace_finish(outcome.finish)},
                    {"prompt_tokens",outcome.prompt_tokens},
                    {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                json response={{"id",resp_id},{"object","response"},{"model","q27-metal"},{"status",status},
                    {"output_text",all_text}, // Metal convenience field, pre-port consumers
                    {"output",items},
                    {"usage",{{"input_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}};
                if(incomplete)
                    response["incomplete_details"]={{"reason","max_output_tokens"}};
                json_response(r,response);
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            const bool test_force_error=runtime.test_failpoints && q27::jbool(body,"q27_test_engine_error",false);
            const bool test_malformed=runtime.test_failpoints && q27::jbool(body,"q27_test_malformed_wrapper",false);
            auto stream_active=std::make_shared<Runtime::StreamScope>(runtime);
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,rn,resp_id,msg_id,tools,custom_names,grammar_tool_names,allowed_tool_names,allowed_hosted_names,
                 snap_hint,sock,test_force_error,test_malformed,stream_active,think_req,tchoice](size_t,httplib::DataSink& sink)->bool {
                    (void)stream_active;
                    bool alive=true;
                    auto last_wire=std::chrono::steady_clock::now();
                    auto ev=[&](const json& j){
                        std::string s=q27::sse_event(j.value("type",std::string("x")),j);
                        if(!sink.write(s.data(),s.size())) alive=false;
                        else last_wire=std::chrono::steady_clock::now();
                        return alive;
                    };
                    // codex P3: item-lifecycle state + machinery hoisted
                    // above the try so the engine-failure path below can
                    // still close an open item and terminate the turn.
                    json items=json::array();
                    int tool_counter=0,out_index=0,msg_index=-1;
                    int message_counter=0,reason_counter=0;
                    std::string think,text,tool_buf,bare_pending,active_msg_id;
                    bool bare_holding=false,tool_calls_clean=true;
                    q27::StreamSplitter sp;
                    if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=q27::StreamSplitter::TOOL;
                    else if(think_req) sp.chan=q27::StreamSplitter::THINK;
                        auto item_done=[&](const json& it){
                            ev({{"type","response.output_item.done"},{"output_index",out_index++},{"item",it}});
                            items.push_back(it);
                        };
                        auto flush_think=[&](bool incomplete_item=false){
                            std::string th=q27::strip_ws2(think); think.clear();
                            if(th.empty()) return;
                            const std::string iid="rs_metal_"+runtime.boot_id+"_"+
                                std::to_string(rn)+"_"+std::to_string(reason_counter++);
                            json item={{"type","reasoning"},{"id",iid},
                                {"status",incomplete_item?"incomplete":"completed"},
                                {"summary",json::array({{{"type","summary_text"},{"text",th}}})},
                                {"encrypted_content",nullptr}};
                            json added=item; added["status"]="in_progress";
                            ev({{"type","response.output_item.added"},{"output_index",out_index},{"item",added}});
                            item_done(item);
                        };
                        // codex 0.143 item lifecycle: a delta needs an OPEN item
                        // (added + content_part.added), else the turn aborts.
                        auto open_text=[&]{
                            if(msg_index>=0) return;
                            msg_index=out_index;
                            active_msg_id=msg_id+"_"+std::to_string(message_counter++);
                            ev({{"type","response.output_item.added"},{"output_index",msg_index},
                                {"item",{{"type","message"},{"id",active_msg_id},{"role","assistant"},
                                         {"status","in_progress"},{"content",json::array()}}}});
                            ev({{"type","response.content_part.added"},{"item_id",active_msg_id},
                                {"output_index",msg_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",""},{"annotations",json::array()}}}});
                        };
                        auto flush_text=[&](bool incomplete_item=false){
                            if(msg_index<0) { text.clear(); return; }
                            std::string tx; tx.swap(text);
                            ev({{"type","response.output_text.done"},{"item_id",active_msg_id},
                                {"output_index",msg_index},{"content_index",0},{"text",tx}});
                            ev({{"type","response.content_part.done"},{"item_id",active_msg_id},
                                {"output_index",msg_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",tx},{"annotations",json::array()}}}});
                            json it={{"type","message"},{"id",active_msg_id},{"role","assistant"},
                                {"status",incomplete_item?"incomplete":"completed"},
                                {"content",json::array({{{"type","output_text"},{"text",tx},
                                                         {"annotations",json::array()}}})}};
                            ev({{"type","response.output_item.done"},{"output_index",msg_index},{"item",it}});
                            items.push_back(it);
                            out_index=msg_index+1; msg_index=-1; active_msg_id.clear();
                        };
                        auto push_call=[&](const std::string& name,const json& args,bool incomplete_item=false){
                            if(!responses_tool_allowed(name,allowed_tool_names,allowed_hosted_names)) return;
                            const int call_index=tool_counter++;
                            const std::string cid="call_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                            const std::string iid="fc_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                            json item;
                            if(custom_names.count(name)) {
                                std::string input=args.is_object() && args.contains("input") && args["input"].is_string()
                                                      ?args["input"].get<std::string>():args.dump();
                                item={{"type","custom_tool_call"},{"id",iid},{"call_id",cid},
                                    {"status",incomplete_item?"incomplete":"completed"},
                                    {"name",name},{"input",input}};
                            } else
                                item={{"type","function_call"},{"id",iid},{"call_id",cid},
                                      {"status",incomplete_item?"incomplete":"completed"},
                                      {"name",name},{"arguments",args.dump()}};
                            json added=item;
                            added["status"]="in_progress";
                            if(added["type"]=="function_call") added["arguments"]="";
                            else added["input"]="";
                            ev({{"type","response.output_item.added"},{"output_index",out_index},{"item",added}});
                            item_done(item);
                        };
                        auto push_message_done=[&](const std::string& tx,bool incomplete_item=false){
                            if(tx.empty()) return;
                            const std::string iid=msg_id+"_"+std::to_string(message_counter++);
                            json item={{"type","message"},{"id",iid},{"role","assistant"},
                                {"status",incomplete_item?"incomplete":"completed"},
                                {"content",json::array({{{"type","output_text"},{"text",tx},
                                                         {"annotations",json::array()}}})}};
                            ev({{"type","response.output_item.added"},{"output_index",out_index},
                                {"item",{{"type","message"},{"id",iid},{"role","assistant"},
                                         {"status","in_progress"},{"content",json::array()}}}});
                            ev({{"type","response.content_part.added"},{"item_id",iid},
                                {"output_index",out_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",""},{"annotations",json::array()}}}});
                            ev({{"type","response.output_text.delta"},{"item_id",iid},
                                {"output_index",out_index},{"content_index",0},{"delta",tx}});
                            ev({{"type","response.output_text.done"},{"item_id",iid},
                                {"output_index",out_index},{"content_index",0},{"text",tx}});
                            ev({{"type","response.content_part.done"},{"item_id",iid},
                                {"output_index",out_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",tx},{"annotations",json::array()}}}});
                            item_done(item);
                        };
                        auto flush_tool=[&](bool final_turn,bool incomplete_item=false){
                            auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf)); tool_buf.clear();
                            if(!c.ok) {
                                runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},
                                    {"stream",true},{"malformed_wrapper",true},{"count",0}});
                                if(final_turn && !incomplete_item) {
                                    std::string pre;
                                    auto bcs=q27::parse_bare_tool_calls(c.raw,&pre,
                                                                        tools.empty()?nullptr:&tools,true);
                                    bool all_eligible=!bcs.empty();
                                    for(const auto& bc:bcs)
                                        all_eligible=all_eligible && responses_tool_allowed(
                                            bc.name,allowed_tool_names,allowed_hosted_names);
                                    if(all_eligible) {
                                        fprintf(stderr,"[tool-fallback] %zu truncated wrapped call(s) recovered (resp stream)\n",bcs.size());
                                        runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},{"stream",true},{"truncated_wrapper",true},{"count",bcs.size()}});
                                        push_message_done(pre,incomplete_item);
                                        for(auto& bc:bcs) push_call(bc.name,bc.arguments,false);
                                        return;
                                    }
                                }
                                push_message_done(c.raw,incomplete_item);
                                return;
                            }
                            if(!responses_tool_allowed(c.name,allowed_tool_names,
                                                       allowed_hosted_names)) {
                                push_message_done(c.raw,incomplete_item);
                                return;
                            }
                            push_call(c.name,c.arguments,incomplete_item);
                        };
                        auto emit_text=[&](const std::string& t){
                            if(msg_index<0 && text.empty() && q27::strip_ws2(t).empty()) return;
                            open_text();
                            text+=t;
                            ev({{"type","response.output_text.delta"},{"item_id",active_msg_id},
                                {"output_index",msg_index},{"content_index",0},{"delta",t}});
                        };
                        auto flush_bare=[&](bool final_turn,bool incomplete_item=false){
                            if(!bare_holding) return;
                            std::string pre;
                            auto bcs=q27::parse_bare_tool_calls(
                                bare_pending,&pre,tools.empty()?nullptr:&tools,
                                final_turn && !incomplete_item);
                            if(!bcs.empty()) {
                                bool all_eligible=true;
                                for(const auto& bc:bcs)
                                    all_eligible=all_eligible && responses_tool_allowed(
                                        bc.name,allowed_tool_names,allowed_hosted_names);
                                if(all_eligible) {
                                    if(!pre.empty()) emit_text(pre);
                                    flush_text();
                                    for(auto& bc:bcs) push_call(bc.name,bc.arguments,incomplete_item);
                                    fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (resp stream)\n",bcs.size());
                                    runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},{"stream",true},{"count",bcs.size()}});
                                } else emit_text(bare_pending);
                            } else emit_text(bare_pending);
                            bare_pending.clear();
                            bare_holding=false;
                        };
                        // Incremental tool-call argument streaming (2026-07-18
                        // streaming-parity-messages-responses.md): wrapped
                        // calls stream function_call_arguments.delta fragments
                        // as they generate — the SAME ToolCallStreamer proven
                        // on /v1/chat/completions and /v1/messages. Deviant
                        // heads fall back to the buffered flush_tool path,
                        // raw byte-exact. custom_tool_call stays whole-item.
                        q27::ToolCallStreamer ts;
                        int st_idx=-1;             // output_index of in-flight streamed call
                        std::string st_iid, st_cid, st_acc;  // item/call ids + accumulated args
                        bool st_custom=false,st_rejected=false;
                        auto st_arg_delta=[&](const std::string& frag){
                            if(frag.empty() || st_idx<0) return;
                            st_acc+=frag;
                            ev({{"type","response.function_call_arguments.delta"},
                                {"item_id",st_iid},{"output_index",st_idx},{"delta",frag}});
                        };
                        auto recover_trail=[&](const std::string& raw,bool allow_repair){
                            const std::string tr=q27::strip_ws2(raw);
                            if(tr.empty()) return;
                            std::string pre;
                            auto bcs=q27::parse_bare_tool_calls(tr,&pre,
                                tools.empty()?nullptr:&tools,allow_repair);
                            if(!bcs.empty()) {
                                bool all_eligible=true;
                                for(const auto& bc:bcs)
                                    all_eligible=all_eligible && responses_tool_allowed(
                                        bc.name,allowed_tool_names,allowed_hosted_names);
                                if(all_eligible) {
                                    if(!pre.empty()) emit_text(pre);
                                    if(!pre.empty()) flush_text();
                                    for(auto& bc:bcs) {
                                        if(msg_index>=0) flush_text();
                                        push_call(bc.name,bc.arguments,false);
                                    }
                                    fprintf(stderr,"[tool-stream] %zu trailing call(s) recovered after streamed call (resp)\n",bcs.size());
                                    runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"id",resp_id},
                                        {"stream",true},{"trailing",true},{"count",bcs.size()}});
                                } else emit_text(raw);
                            } else if(pre.empty() && tr.find_first_not_of("}] \t\r\n")!=std::string::npos)
                                emit_text(tr);
                        };
                        auto close_stream_tool=[&](bool incomplete_item){
                            if(!ts.active()) return;
                            if(st_rejected) {
                                st_rejected=false;
                                std::string tail;
                                (void)ts.finalize(&tail);
                                std::string rejected=ts.raw;
                                const std::string trailing=ts.trail();
                                if(!trailing.empty() && trailing.size()<=rejected.size() &&
                                   rejected.compare(rejected.size()-trailing.size(),trailing.size(),trailing)==0)
                                    rejected.resize(rejected.size()-trailing.size());
                                push_message_done(rejected,incomplete_item);
                                recover_trail(trailing,true);
                                ts.reset();
                                return;
                            }
                            // Custom tool: never streamed incremental; hand the
                            // verbatim raw to the buffered flush_tool -> push_call
                            // path so it emits a whole custom_tool_call item with
                            // bare-string input (codex P1 2026-07-19).
                            if(st_custom) {
                                st_custom=false;
                                tool_buf=ts.raw;
                                ts.reset();
                                flush_tool(false,incomplete_item);
                                return;
                            }
                            std::string tail;
                            const bool clean=ts.finalize(&tail);
                            if(ts.invalid()) {
                                tool_calls_clean=false;
                                // Streaming is irreversible: added + arg deltas
                                // already on wire. Close the item done with the
                                // accumulated args so the lifecycle pairs, then
                                // recover packed trail. Unclean: mark incomplete
                                // so clients do not execute (mirrors chat).
                                st_arg_delta(tail);
                                json it={{"type","function_call"},{"id",st_iid},{"call_id",st_cid},
                                    {"status","incomplete"},{"name",ts.name},{"arguments",st_acc}};
                                ev({{"type","response.output_item.done"},{"output_index",st_idx},{"item",it}});
                                items.push_back(it);
                                out_index=st_idx+1; st_idx=-1; st_iid.clear(); st_cid.clear(); st_acc.clear();
                                recover_trail(ts.trail(),false);
                            } else if(ts.opened) {
                                st_arg_delta(tail);
                                if(!clean) {
                                    tool_calls_clean=false;
                                    fprintf(stderr,"[tool-stream] streamed call closed unbalanced (production semantics, sent as-is)\n");
                                }
                                json it={{"type","function_call"},{"id",st_iid},{"call_id",st_cid},
                                    {"status",(incomplete_item || !clean)?"incomplete":"completed"},
                                    {"name",ts.name},{"arguments",st_acc}};
                                ev({{"type","response.output_item.done"},{"output_index",st_idx},{"item",it}});
                                items.push_back(it);
                                out_index=st_idx+1; st_idx=-1; st_iid.clear(); st_cid.clear(); st_acc.clear();
                                // A wrapper can pack more than one call: bytes
                                // after the streamed call's args closed are not
                                // framing — recover through the bare-call chain.
                                recover_trail(ts.trail(),true);
                            } else {
                                tool_buf=ts.raw;
                                flush_tool(false,incomplete_item);
                            }
                            ts.reset();
                        };
                        auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                            if(ch==q27::StreamSplitter::TOOL) {
                                if(tchoice.mode==q27::ToolChoice::NONE) {
                                    flush_bare(false);
                                    if(!think.empty()) flush_think();
                                    emit_text(t);
                                    return;
                                }
                                flush_bare(false);
                                if(!think.empty()) flush_think();
                                if(!text.empty()) flush_text();
                                // Intercept with the streamer: on a clean head,
                                // open a function_call item and stream arg deltas;
                                // on FALLBACK the raw is handed to flush_tool.
                                bool opened=false;
                                const std::string frag=ts.feed(t,&opened);
                                if(opened) {
                                    if(!responses_tool_allowed(ts.name,allowed_tool_names,
                                                               allowed_hosted_names)) {
                                        st_rejected=true;
                                    } else if(custom_names.count(ts.name)) {
                                        st_custom=true;
                                    } else {
                                        const int call_index=tool_counter++;
                                        st_idx=out_index;
                                        st_cid="call_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                                        st_iid="fc_metal_"+runtime.boot_id+"_"+std::to_string(rn)+"_"+std::to_string(call_index);
                                        ev({{"type","response.output_item.added"},{"output_index",st_idx},
                                            {"item",{{"type","function_call"},{"id",st_iid},{"call_id",st_cid},
                                                     {"status","in_progress"},{"name",ts.name},{"arguments",""}}}});
                                    }
                                }
                                if(!frag.empty() && !st_custom && !st_rejected) st_arg_delta(frag);
                                // FALLBACK: not yet open and head deviated — hand
                                // the verbatim raw to the buffered path.
                                if(!ts.opened && ts.active() && frag.empty() && !opened) {
                                    if(ts.state==q27::ToolCallStreamer::FALLBACK) {
                                        tool_buf=ts.raw; ts.reset();
                                        flush_tool(false);
                                    }
                                }
                                return;
                            }
                            close_stream_tool(false);
                            if(!tool_buf.empty()) flush_tool(false);
                            // codex P2: a THINK transition must close an open
                            // text item first (same rule as TOOL) — else
                            // flush_think's item_done consumes the still-open
                            // message's output_index and the done events
                            // duplicate/reorder indices.
                            if(ch==q27::StreamSplitter::THINK) {
                                flush_bare(false);
                                if(!text.empty()) flush_text();
                                think+=t; return;
                            }
                            if(!think.empty()) flush_think();
                            // Bare JSON calls are TEXT to StreamSplitter. Stream
                            // prose normally, but retain bytes from the first
                            // object/array opener until classification. This
                            // supports `prose\n{"name":...}` without putting
                            // the raw call on the wire before emitting its
                            // structured item. Ordinary brace-free text remains
                            // token-streamed; JSON prose buffers from its opener
                            // to turn end, a correctness-over-latency trade.
                            if(bare_holding) { bare_pending+=t; return; }
                            size_t obj=t.find('{'), arr=t.find('[');
                            size_t cut=obj==std::string::npos?arr:
                                       arr==std::string::npos?obj:std::min(obj,arr);
                            if(cut==std::string::npos) { emit_text(t); return; }
                            if(cut) emit_text(t.substr(0,cut));
                            bare_pending=t.substr(cut);
                            bare_holding=true;
                        };
                        try {
                        if(!ev({{"type","response.created"},
                                {"response",{{"id",resp_id},{"object","response"},{"status","in_progress"}}}})) {
                            sink.done(); return true;
                        }
                        // Keepalive through silent stretches (see the chat
                        // twin — queue wait + prefill + tool_buf buffering).
                        // The Responses wire has no ping event; an SSE comment
                        // line is spec-legal and invisible to eventsource
                        // parsers.
                        auto keepalive=[&]{
                            if(alive && std::chrono::steady_clock::now()-last_wire>std::chrono::seconds(5)) {
                                const char ka[]=": keepalive\n\n";
                                if(!sink.write(ka,sizeof(ka)-1)) alive=false;
                                else last_wire=std::chrono::steady_clock::now();
                            }
                        };
                        if(test_force_error) {
                            for(auto& [ch,t]:sp.feed("partial-before-failure <thi")) route(ch,t);
                            throw Runtime::EngineError("forced Responses stream engine error");
                        }
                        Runtime::Outcome outcome;
                        if(test_malformed) {
                            tool_buf="{malformed-wrapper";
                            outcome.prompt_tokens=(uint32_t)ids.size();
                            outcome.output_tokens=1;
                            outcome.finish=Runtime::Finish::Stop;
                        } else {
                            outcome=runtime.guard_engine([&]{
                                return runtime.run(ids,n,sampling,stops,
                                    [&](const std::string& piece)->bool {
                                        for(auto& [ch,t]:sp.feed(piece)) route(ch,t);
                                        keepalive();
                                        return alive && sink.is_writable();
                                    },grammar_tool_names,snap_hint,
                                    [&,sock]{ keepalive();
                                              return socket_alive(sock); },resp_id);
                            });
                        }
                        if(outcome.finish==Runtime::Finish::Cancelled) {
                            runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                                {"finish","cancelled"},{"terminal","cancelled"},{"prompt_tokens",outcome.prompt_tokens},
                                {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                            sink.done();
                            return false;
                        }
                        const bool incomplete=outcome.finish==Runtime::Finish::Length;
                        for(auto& [ch,t]:sp.flush()) route(ch,t);
                        close_stream_tool(incomplete);   // close any in-flight streamed call
                        if(!tool_buf.empty()) flush_tool(true,incomplete);
                        flush_think(incomplete);
                        flush_bare(true,incomplete);
                        flush_text(incomplete);
                        if(tchoice.mode==q27::ToolChoice::FORCED && tool_counter==0 && !incomplete)
                            throw Runtime::EngineError("model produced no eligible tool call for forced tool_choice");
                        if(!tool_calls_clean && !incomplete) {
                            const char* message="model produced an incomplete tool call";
                            runtime.trace.event({{"kind","error"},{"api","responses"},{"id",resp_id},
                                {"status",500},{"type","invalid_model_output"},{"error",message}});
                            ev({{"type","response.failed"},
                                {"response",{{"id",resp_id},{"object","response"},{"status","failed"},
                                    {"error",{{"code","invalid_model_output"},{"message",message}}},
                                    {"output",items}}}});
                        } else {
                            const char* status=incomplete?"incomplete":"completed";
                            runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                                {"finish",status},{"terminal",trace_finish(outcome.finish)},
                                {"prompt_tokens",outcome.prompt_tokens},
                                {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                            json response={{"id",resp_id},{"object","response"},{"status",status},
                                {"output",items},
                                {"usage",{{"input_tokens",outcome.prompt_tokens},
                                          {"input_tokens_details",{{"cached_tokens",0}}},
                                          {"output_tokens",outcome.output_tokens},
                                          {"output_tokens_details",{{"reasoning_tokens",0}}},
                                          {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}}};
                            if(incomplete)
                                response["incomplete_details"]={{"reason","max_output_tokens"}};
                            ev({{"type",incomplete?"response.incomplete":"response.completed"},
                                {"response",response}});
                        }
                    } catch(const Runtime::ClientGone&) {
                        return false;
                    } catch(const Runtime::ServerOverloaded& e) {
                        runtime.trace.event({{"kind","error"},{"api","responses"},{"id",resp_id},
                            {"status",503},{"type","overloaded_error"},{"error",e.what()}});
                        ev({{"type","error"},{"error",{{"type","overloaded_error"},{"message",e.what()}}}});
                        ev({{"type","response.failed"},
                            {"response",{{"id",resp_id},{"object","response"},{"status","failed"},
                                {"error",{{"code","overloaded_error"},{"message",e.what()}}},
                                {"output",items}}}});
                    } catch(const std::exception& e) {
                        runtime.trace.event({{"kind","error"},{"api","responses"},{"id",resp_id},
                            {"status",500},{"type","api_error"},{"error",e.what()}});
                        // codex P3: an engine failure mid-stream must not
                        // leave codex holding an unterminated item lifecycle
                        // over a 200 stream. Close any open item (the
                        // flushers emit the done triplet; no-ops when
                        // nothing is open — a pending tool buffer is dropped
                        // as unreliable), keep the first-class api_error
                        // event (Anthropic-stream precedent), then end the
                        // turn with the Responses-spec failure terminator
                        // carrying the partial output. The release trace gate
                        // forces this path under Q27_METAL_TEST_FAILPOINTS.
                        try {
                            for(auto& [ch,t]:sp.flush()) route(ch,t);
                            close_stream_tool(true);   // codex P2 2026-07-19:
                                // close any in-flight streamed function_call so
                                // the added item gets a matching done (and lands
                                // in response.output) instead of dangling open.
                            if(!tool_buf.empty()) flush_tool(false,true);
                            if(!think.empty()) flush_think(true);
                            if(!bare_pending.empty()) {
                                emit_text(bare_pending);
                                bare_pending.clear();
                                bare_holding=false;
                            }
                            flush_text(true);
                        } catch(...) {}
                        ev({{"type","error"},{"error",{{"type","api_error"},{"message",e.what()}}}});
                        ev({{"type","response.failed"},
                            {"response",{{"id",resp_id},{"object","response"},{"status","failed"},
                                {"error",{{"code","server_error"},{"message",e.what()}}},
                                {"output",items}}}});
                    }
                    sink.done();
                    return true;
                });
        }));

        // SO_REUSEADDR only (upstream e0a1a39 parity; macOS defines
        // SO_REUSEPORT, so httplib's default_socket_options picks it here
        // exactly as it does on Linux). SO_REUSEPORT lets a SECOND Metal
        // server co-bind this port and the kernel then load-balances
        // connections across both. On this box that is worse than the CUDA
        // arm's eval-integrity hazard: a second server also loads a second
        // copy of a ~17 GiB artifact into 24 GiB of unified memory. REUSEADDR
        // keeps fast rebind after TIME_WAIT without permitting live co-bind,
        // so the second process fails into the FATAL below instead.
        server.set_socket_options([](socket_t sock) {
            int opt=1;
            setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,reinterpret_cast<const void*>(&opt),sizeof(opt));
        });
        // Bind FIRST, announce only on success (upstream e0a1a39 parity).
        // The old order printed "listening on ..." and the admin token before
        // listen() had bound anything, so a port squatter produced a startup
        // log that claimed success, leaked the admin token to stderr, and only
        // then failed -- operators (and log scrapers) read the first line.
        if(!server.bind_to_port(host.c_str(),(int)port)) {
            fprintf(stderr,
                    "FATAL: cannot bind %s:%u (port already in use? see "
                    "`lsof -nP -iTCP:%u -sTCP:LISTEN`)\n",host.c_str(),port,port);
            return 1;
        }
        fprintf(stderr,"q27 Metal server listening on http://%s:%u (ctx=%u, kv=%s, mtp=%u, slots=%zu)\n",
                host.c_str(),port,runtime.context,turbo3?"turbo3":"fp16",width,runtime.slots.size());
        // Operator-only: the admin credential goes to stderr (never HTTP).
        fprintf(stderr,"q27 Metal server admin token (X-Q27-Admin-Token): %s\n",runtime.admin_token.c_str());
        if(!server.listen_after_bind()) throw std::runtime_error("server listen failed");
        return 0;
    } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
