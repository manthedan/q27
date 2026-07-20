#include "q27_agent_protocol.h"

#include "../../src/tool_preamble.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <string_view>

using nlohmann::json;

namespace {

constexpr const char *kToolNames[] = {
    "read", "search", "write", "edit", "shell"};
constexpr size_t kToolNameCount = sizeof(kToolNames) / sizeof(kToolNames[0]);

void set_error(char *out, size_t cap, const char *message) noexcept {
    if (out && cap) std::snprintf(out, cap, "%s", message ? message : "invalid tool call");
}

bool only_keys(const json& object, std::initializer_list<const char *> keys) {
    if (!object.is_object() || object.size() != keys.size()) return false;
    for (const char *key : keys)
        if (!object.contains(key)) return false;
    return true;
}

bool json_u32(const json& object, const char *key, uint32_t fallback,
              uint32_t low, uint32_t high, uint32_t& out) {
    if (!object.contains(key)) { out = fallback; return true; }
    const json& value = object[key];
    if (!value.is_number_integer() && !value.is_number_unsigned()) return false;
    try {
        const int64_t number = value.get<int64_t>();
        if (number < low || number > high) return false;
        out = static_cast<uint32_t>(number);
        return true;
    } catch (...) { return false; }
}

bool copy_string(const json& args, const char *key, unsigned char **out,
                 size_t *out_len, bool reject_nul) {
    if (!args.contains(key) || !args[key].is_string()) return false;
    const std::string value = args[key].get<std::string>();
    if (reject_nul && value.find('\0') != std::string::npos) return false;
    unsigned char *copy = static_cast<unsigned char *>(std::malloc(value.size() + 1));
    if (!copy) throw std::bad_alloc();
    if (!value.empty()) std::memcpy(copy, value.data(), value.size());
    copy[value.size()] = 0;
    *out = copy;
    *out_len = value.size();
    return true;
}

bool ascii_equal_ci(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        unsigned char ac = static_cast<unsigned char>(a[i]);
        unsigned char bc = static_cast<unsigned char>(b[i]);
        if (ac >= 'A' && ac <= 'Z') ac = static_cast<unsigned char>(ac + 'a' - 'A');
        if (bc >= 'A' && bc <= 'Z') bc = static_cast<unsigned char>(bc + 'a' - 'A');
        if (ac != bc) return false;
    }
    return true;
}

bool source_fence_label_allowed(const char *path, std::string_view label) {
    if (!path) return false;
    const char *base = std::strrchr(path, '/');
    base = base ? base + 1 : path;
    const char *dot = std::strrchr(base, '.');
    if (!dot || !dot[1]) return false;
    const std::string_view ext(dot + 1);
    struct Mapping { const char *ext; const char *labels[4]; };
    static constexpr Mapping mappings[] = {
        {"py", {"python", "py", nullptr, nullptr}},
        {"c", {"c", nullptr, nullptr, nullptr}},
        {"h", {"c", "cpp", "c++", nullptr}},
        {"cc", {"cpp", "c++", "cc", nullptr}},
        {"cpp", {"cpp", "c++", nullptr, nullptr}},
        {"cxx", {"cpp", "c++", nullptr, nullptr}},
        {"hh", {"cpp", "c++", nullptr, nullptr}},
        {"hpp", {"cpp", "c++", nullptr, nullptr}},
        {"hxx", {"cpp", "c++", nullptr, nullptr}},
        {"m", {"objective-c", "objc", nullptr, nullptr}},
        {"mm", {"objective-cpp", "objcpp", "cpp", nullptr}},
        {"js", {"javascript", "js", nullptr, nullptr}},
        {"jsx", {"jsx", "javascript", "js", nullptr}},
        {"ts", {"typescript", "ts", nullptr, nullptr}},
        {"tsx", {"tsx", "typescript", "ts", nullptr}},
        {"rs", {"rust", "rs", nullptr, nullptr}},
        {"go", {"go", nullptr, nullptr, nullptr}},
        {"java", {"java", nullptr, nullptr, nullptr}},
        {"swift", {"swift", nullptr, nullptr, nullptr}},
        {"kt", {"kotlin", "kt", nullptr, nullptr}},
        {"kts", {"kotlin", "kts", nullptr, nullptr}},
        {"rb", {"ruby", "rb", nullptr, nullptr}},
        {"php", {"php", nullptr, nullptr, nullptr}},
        {"sh", {"bash", "sh", "shell", nullptr}},
        {"zsh", {"zsh", "shell", nullptr, nullptr}},
        {"fish", {"fish", "shell", nullptr, nullptr}},
        {"lua", {"lua", nullptr, nullptr, nullptr}},
        {"sql", {"sql", nullptr, nullptr, nullptr}},
        {"html", {"html", nullptr, nullptr, nullptr}},
        {"css", {"css", nullptr, nullptr, nullptr}},
        {"json", {"json", nullptr, nullptr, nullptr}},
        {"yaml", {"yaml", "yml", nullptr, nullptr}},
        {"yml", {"yaml", "yml", nullptr, nullptr}},
        {"toml", {"toml", nullptr, nullptr, nullptr}},
        {"xml", {"xml", nullptr, nullptr, nullptr}},
    };
    for (const Mapping& mapping : mappings) {
        if (!ascii_equal_ci(ext, mapping.ext)) continue;
        if (label.empty()) return true;
        for (const char *candidate : mapping.labels)
            if (candidate && ascii_equal_ci(label, candidate)) return true;
        return false;
    }
    return false;
}

const std::string& preamble() {
    static const std::string value = [] {
        json tools = json::array({
            {{"type", "function"}, {"function", {
                {"name", kToolNames[0]},
                {"description", "Read one workspace-relative regular file as exact bytes."},
                {"parameters", {{"type", "object"},
                    {"properties", {{"path", {{"type", "string"}}}}},
                    {"required", json::array({"path"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[1]},
                {"description", "Find literal byte-string occurrences in one workspace-relative regular file."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}},
                        {"needle", {{"type", "string"}}}}},
                    {"required", json::array({"path", "needle"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[2]},
                {"description", "Begin creating one new workspace-relative regular file. Supply only the path. After validation, a separate raw-payload turn requests the exact file content without JSON escaping. Fails if the path already exists; use edit for existing files."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}}}},
                    {"required", json::array({"path"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[3]},
                {"description", "Begin replacing exactly one nonempty literal byte-string match in an existing workspace-relative regular file. Supply path and old only. After validation, a separate raw-payload turn requests the replacement without JSON escaping."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}},
                        {"old", {{"type", "string"}}}}},
                    {"required", json::array({"path", "old"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[4]},
                {"description", "Run one bounded no-fork shell job in the workspace. Pipelines and background jobs are unavailable."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"command", {{"type", "string"}}},
                        {"timeout_ms", {{"type", "integer"}, {"minimum", 1}, {"maximum", 60000}}},
                        {"max_output_bytes", {{"type", "integer"}, {"minimum", 1}, {"maximum", 262144}}}}},
                    {"required", json::array({"command"})},
                    {"additionalProperties", false}}}}}}
        });
        return q27::tools_preamble(tools);
    }();
    return value;
}

} // namespace

extern "C" const char *q27_agent_tool_preamble(void) {
    try { return preamble().c_str(); }
    catch (...) { return nullptr; }
}

extern "C" size_t q27_agent_tool_names(const char *const **names_out) {
    if (!names_out) return 0;
    *names_out = kToolNames;
    return kToolNameCount;
}

extern "C" void q27_agent_tool_call_free(q27_agent_tool_call *call) {
    if (!call) return;
    std::free(call->path);
    std::free(call->input);
    std::free(call->replacement);
    *call = q27_agent_tool_call{};
}

extern "C" int q27_agent_unwrap_whole_file_source_fence(
    const char *path, unsigned char *bytes, size_t *len) {
    if (!path || !bytes || !len) return -1;
    const size_t size = *len;
    if (size < 7 || std::memcmp(bytes, "```", 3)) return 0;

    size_t opening_end = 3;
    while (opening_end < size && bytes[opening_end] != '\n' &&
           bytes[opening_end] != '\r') {
        // Fence labels are intentionally narrow. Spaces, attributes, and long
        // lines are ambiguous and remain exact rather than being guessed at.
        const unsigned char c = bytes[opening_end];
        if (opening_end - 3 >= 31 ||
            !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '_'))
            return 0;
        ++opening_end;
    }
    if (opening_end >= size) return 0;
    size_t content_start = opening_end + 1;
    if (bytes[opening_end] == '\r') {
        if (content_start >= size || bytes[content_start] != '\n') return 0;
        ++content_start;
    }
    const std::string_view label(
        reinterpret_cast<const char *>(bytes + 3), opening_end - 3);
    if (!source_fence_label_allowed(path, label)) return 0;

    size_t non_newline_end = size;
    while (non_newline_end &&
           (bytes[non_newline_end - 1] == '\n' ||
            bytes[non_newline_end - 1] == '\r'))
        --non_newline_end;
    if (non_newline_end <= content_start) return 0;
    size_t closing_start = non_newline_end;
    while (closing_start > content_start && bytes[closing_start - 1] != '\n')
        --closing_start;
    size_t closing_ticks_start = closing_start;
    while (closing_ticks_start < non_newline_end &&
           closing_ticks_start - closing_start < 3 &&
           bytes[closing_ticks_start] == ' ')
        ++closing_ticks_start;
    size_t closing_ticks_end = closing_ticks_start;
    while (closing_ticks_end < non_newline_end &&
           bytes[closing_ticks_end] == '`')
        ++closing_ticks_end;
    if (closing_ticks_end - closing_ticks_start < 3) return 0;
    size_t closing_tail = closing_ticks_end;
    while (closing_tail < non_newline_end &&
           (bytes[closing_tail] == ' ' || bytes[closing_tail] == '\t'))
        ++closing_tail;
    if (closing_tail != non_newline_end) return 0;
    // The final fence is authoritative only when no earlier exact fence line
    // already closes the block. Otherwise the intervening bytes are prose or
    // another block, and silently flattening the wrapper would corrupt them.
    for (size_t line_start = content_start; line_start < closing_start;) {
        size_t line_end = line_start;
        while (line_end < closing_start && bytes[line_end] != '\n') ++line_end;
        size_t logical_end = line_end;
        if (logical_end > line_start && bytes[logical_end - 1] == '\r')
            --logical_end;
        size_t candidate = line_start;
        while (candidate < logical_end && candidate - line_start < 3 &&
               bytes[candidate] == ' ')
            ++candidate;
        size_t ticks_end = candidate;
        while (ticks_end < logical_end && bytes[ticks_end] == '`') ++ticks_end;
        if (ticks_end - candidate >= 3) {
            size_t tail = ticks_end;
            while (tail < logical_end &&
                   (bytes[tail] == ' ' || bytes[tail] == '\t'))
                ++tail;
            if (tail == logical_end) return 0;
        }
        line_start = line_end < closing_start ? line_end + 1 : closing_start;
    }

    const size_t content_len = closing_start - content_start;
    if (content_len)
        std::memmove(bytes, bytes + content_start, content_len);
    bytes[content_len] = 0;
    *len = content_len;
    return 1;
}

extern "C" q27_agent_tool_call_status q27_agent_parse_tool_call(
    const unsigned char *bytes, size_t len, q27_agent_tool_call *call,
    char *error, size_t error_cap) {
    if (call) *call = q27_agent_tool_call{};
    if (!bytes || !call || len > 1024u * 1024u) {
        set_error(error, error_cap, "invalid or oversized tool-call parser input");
        return Q27_TOOL_CALL_INVALID;
    }
    try {
        const std::string text(reinterpret_cast<const char *>(bytes), len);
        static const std::string open = "<tool_call>";
        static const std::string close = "</tool_call>";
        const size_t begin = text.find(open);
        if (begin == std::string::npos) return Q27_TOOL_CALL_NONE;
        if (text.find(open, begin + open.size()) != std::string::npos) {
            set_error(error, error_cap, "multiple tool calls are not allowed in one turn");
            return Q27_TOOL_CALL_INVALID;
        }
        const size_t body_start = begin + open.size();
        const size_t end = text.find(close, body_start);
        if (end == std::string::npos || text.find(close, end + close.size()) != std::string::npos) {
            set_error(error, error_cap, "tool call is not closed exactly once");
            return Q27_TOOL_CALL_INVALID;
        }
        for (size_t i = end + close.size(); i < text.size(); ++i) {
            const char c = text[i];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                set_error(error, error_cap, "text after tool call is forbidden");
                return Q27_TOOL_CALL_INVALID;
            }
        }
        const json outer = json::parse(text.substr(body_start, end - body_start));
        if (!only_keys(outer, {"name", "arguments"}) ||
            !outer["name"].is_string() || !outer["arguments"].is_object()) {
            set_error(error, error_cap, "tool call must contain only name and object arguments");
            return Q27_TOOL_CALL_INVALID;
        }
        const std::string name = outer["name"].get<std::string>();
        const json& args = outer["arguments"];
        q27_agent_tool_request request{};
        request.timeout_ms = 30000;
        request.max_output_bytes = 256u * 1024u;
        unsigned char *path = nullptr, *input = nullptr, *replacement = nullptr;
        struct LocalCleanup {
            unsigned char *&path;
            unsigned char *&input;
            unsigned char *&replacement;
            ~LocalCleanup() {
                std::free(path); std::free(input); std::free(replacement);
            }
        } cleanup{path, input, replacement};
        size_t path_len = 0;

        bool valid = false;
        if (name == kToolNames[0] && only_keys(args, {"path"})) {
            request.kind = Q27_TOOL_READ;
            valid = copy_string(args, "path", &path, &path_len, true) &&
                    path_len > 0;
        } else if (name == kToolNames[1] && only_keys(args, {"path", "needle"})) {
            request.kind = Q27_TOOL_SEARCH;
            valid = copy_string(args, "path", &path, &path_len, true) &&
                    path_len > 0 &&
                    copy_string(args, "needle", &input, &request.input_len, false) &&
                    request.input_len > 0;
        } else if (name == kToolNames[2] && only_keys(args, {"path"})) {
            request.kind = Q27_TOOL_WRITE;
            valid = copy_string(args, "path", &path, &path_len, true) &&
                    path_len > 0;
        } else if (name == kToolNames[3] && only_keys(args, {"path", "old"})) {
            request.kind = Q27_TOOL_EDIT;
            valid = copy_string(args, "path", &path, &path_len, true) &&
                    path_len > 0 &&
                    copy_string(args, "old", &input, &request.input_len, false) &&
                    request.input_len > 0;
        } else if (name == kToolNames[4] && args.is_object() &&
                   args.size() >= 1 && args.size() <= 3 && args.contains("command")) {
            for (auto it = args.begin(); it != args.end(); ++it)
                if (it.key() != "command" && it.key() != "timeout_ms" &&
                    it.key() != "max_output_bytes") valid = false;
            request.kind = Q27_TOOL_SHELL;
            valid = true;
            for (auto it = args.begin(); valid && it != args.end(); ++it)
                valid = it.key() == "command" || it.key() == "timeout_ms" ||
                        it.key() == "max_output_bytes";
            valid = valid && copy_string(args, "command", &input,
                                         &request.input_len, true) &&
                    request.input_len > 0 &&
                    json_u32(args, "timeout_ms", 30000, 1, 60000,
                             request.timeout_ms) &&
                    json_u32(args, "max_output_bytes", 256u * 1024u, 1,
                             256u * 1024u, request.max_output_bytes);
        }
        if (!valid || !request.kind) {
            set_error(error, error_cap, "unknown tool or arguments do not match its strict schema");
            return Q27_TOOL_CALL_INVALID;
        }
        request.path = reinterpret_cast<char *>(path);
        request.input = input;
        request.replacement = replacement;
        call->request = request;
        call->path = reinterpret_cast<char *>(path);
        call->input = input;
        call->replacement = replacement;
        path = input = replacement = nullptr;
        return Q27_TOOL_CALL_VALID;
    } catch (const std::bad_alloc&) {
        q27_agent_tool_call_free(call);
        set_error(error, error_cap, "out of memory parsing tool call");
    } catch (const std::exception& e) {
        q27_agent_tool_call_free(call);
        set_error(error, error_cap, e.what());
    } catch (...) {
        q27_agent_tool_call_free(call);
        set_error(error, error_cap, "unknown tool-call parse failure");
    }
    return Q27_TOOL_CALL_INVALID;
}
