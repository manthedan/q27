// Native-agent compatibility adapter recovered from release/metal-v0.7.0.
// Keep its stricter bounded-tool parser local, not an alternative definition of
// the current HTTP API helpers. Prompt rendering delegates to current upstream.
#pragma once

#include "../../src/api_common.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace q27::native_agent {

// Shared serving currently renders prompts through api_common without an
// explicit dialect argument. Keep that process-local default for those
// callsites while native-agent engines retain their own selected value.
inline bool& tool_dialect_xml_default() {
    static bool value = false;
    return value;
}

inline std::string model_family_for_metadata(const std::string& meta_json,
                                              std::string* model_name = nullptr) {
    std::string name;
    try {
        const auto meta = nlohmann::json::parse(meta_json);
        name = meta.value("general.name", std::string());
        if (meta.value("bonsai2", false)) {
            if (model_name) *model_name = name;
            return "qwen38";
        }
    } catch (...) {}
    if (model_name) *model_name = name;

    std::string normalized;
    for (unsigned char c : name)
        if (std::isalnum(c)) normalized += static_cast<char>(std::tolower(c));
    if (normalized.find("qwen38") != std::string::npos) return "qwen38";
    if (normalized.find("qwen36") != std::string::npos) return "qwen36";
    if (normalized.find("bonsai") != std::string::npos) return "bonsai";
    return "unknown";
}

inline bool model_prefers_xml_tool_dialect(const std::string& meta_json,
                                           std::string* model_name = nullptr) {
    return model_family_for_metadata(meta_json, model_name) == "qwen38";
}

inline bool model_profile_matches_metadata(const std::string& profile,
                                           const std::string& meta_json,
                                           std::string* family_out = nullptr) {
    const std::string family = model_family_for_metadata(meta_json);
    if (family_out) *family_out = family;
    if (profile == "bonsai2-qwen38-v1") {
        try { return nlohmann::json::parse(meta_json).value("bonsai2", false); }
        catch (...) { return false; }
    }
    if (profile == "qwen38-thinking-v1") return family == "qwen38";
    if (profile == "qwen36-v1") return family == "qwen36";
    if (profile == "bonsai-agent-v1") return family == "bonsai";
    return false;
}

inline bool tool_dialect_xml() {
    const char* dialect = std::getenv("Q27_TOOL_DIALECT");
    if (dialect) return std::strcmp(dialect, "xml") == 0;
    return tool_dialect_xml_default();
}

inline bool select_tool_dialect_for_model(const std::string& meta_json) {
    std::string name;
    const bool model_xml = model_prefers_xml_tool_dialect(meta_json, &name);
    const char* dialect = std::getenv("Q27_TOOL_DIALECT");
    const bool effective_xml = dialect ? std::strcmp(dialect, "xml") == 0 : model_xml;
    std::fprintf(stderr, "tool dialect: %s (general.name \"%s\"%s)\n",
                 effective_xml && !dialect ? "xml (trained-format default)"
                                           : effective_xml ? "xml" : "json",
                 name.c_str(), dialect ? ", Q27_TOOL_DIALECT override" : "");
    return effective_xml;
}

inline void set_tool_dialect_for_model(const std::string& meta_json) {
    tool_dialect_xml_default() = model_prefers_xml_tool_dialect(meta_json);
    (void)select_tool_dialect_for_model(meta_json);
}

inline std::string xml_parameter_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        if (c == '&') out += "&amp;";
        else if (c == '<') out += "&lt;";
        else if (c == '>') out += "&gt;";
        else out += c;
    }
    return out;
}

inline std::string xml_parameter_unescape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size();) {
        if (value.compare(i, 5, "&amp;") == 0) { out += '&'; i += 5; }
        else if (value.compare(i, 4, "&lt;") == 0) { out += '<'; i += 4; }
        else if (value.compare(i, 4, "&gt;") == 0) { out += '>'; i += 4; }
        else { out += value[i]; i++; }
    }
    return out;
}

// Where the answer starts: just after the reasoning span's close, i.e. the
// FIRST </think> when the turn opened with <think> (the engine emits
// "<think>\n" into the parsed stream), else npos (no reasoning span). A
// literal </think> later is text and never moves it.
inline size_t answer_start(const std::string& text) {
    size_t i = 0;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t' ||
                               text[i] == '\r' || text[i] == '\n')) i++;
    if (text.compare(i, 7, "<think>") != 0) return std::string::npos;
    const size_t close = text.find("</think>", i + 7);
    return close == std::string::npos ? close : close + 8;
}

struct OpenerCandidate {
    size_t pos;
    bool bare; // <function= without a <tool_call> wrapper
};

// The native agent's one rule for tool-call openers, shared by the engine's
// constraint veto and q27_agent_parse_tool_call. Returns the index of the
// first candidate (sorted by pos) that is a real call, or npos. One left-to-
// right pass; each decision reads only text BEFORE the opener, so the
// engine (which sees just that prefix) and the parser always agree.
// Scope: from the start, reset at answer_start() for openers after it.
// An opener is an example (declined) when it is
//  - inside a ``` / ~~~ fence (up to 3 spaces of indent);
//  - after an unmatched backtick run on its own line (inline code: the
//    engine stops before the closing backtick);
//  - nested in an unclosed <function=/<parameter= in prose (depth counted;
//    tags in fences and inline code are literal text);
//  - bare, and not the first non-whitespace of the scope or indented more
//    than 3 spaces / not alone at the start of its line.
inline size_t first_accepted_opener(const std::string& text,
                                    const std::vector<OpenerCandidate>& cands) {
    static const std::string fopen = "<function=", fclose = "</function>",
                             popen = "<parameter=", pclose = "</parameter>";
    const size_t scope_at = answer_start(text);
    bool open = false, prose = true;
    char marker = 0;
    size_t opening_run = 0, fdepth = 0, pdepth = 0, span = 0;
    bool non_ws = false, line_blank = true;
    size_t line_spaces = 0;
    auto classify = [&](size_t line_start) { // fence state for this line
        size_t line_end = text.find('\n', line_start);
        if (line_end == std::string::npos) line_end = text.size();
        const bool in_fence = open;
        bool fence_line = false;
        size_t p = line_start, indent = 0;
        while (p < line_end && indent < 3 && text[p] == ' ') { p++; indent++; }
        if (p < line_end && (text[p] == '`' || text[p] == '~')) {
            const char run_marker = text[p];
            size_t run_end = p;
            while (run_end < line_end && text[run_end] == run_marker) run_end++;
            const size_t run = run_end - p;
            if (run >= 3) {
                if (!open) {
                    open = true;
                    marker = run_marker;
                    opening_run = run;
                    fence_line = true;
                } else if (run_marker == marker && run >= opening_run) {
                    size_t tail = run_end;
                    while (tail < line_end && (text[tail] == ' ' || text[tail] == '\t' ||
                                               text[tail] == '\r')) tail++;
                    if (tail == line_end) {
                        open = false;
                        marker = 0;
                        opening_run = 0;
                    }
                }
            }
        }
        prose = !in_fence && !fence_line;
        span = 0;
    };
    classify(0);
    size_t ci = 0;
    for (size_t i = 0;;) {
        if (i == scope_at) { // the answer: fresh scope, read as a line start
            open = false; marker = 0; opening_run = 0;
            fdepth = pdepth = 0;
            non_ws = false;
            classify(i);
        }
        for (; ci < cands.size() && cands[ci].pos <= i; ++ci) {
            const OpenerCandidate& c = cands[ci];
            bool declined = open || (prose && span) || fdepth || pdepth;
            if (!declined && c.bare) declined = non_ws || !line_blank || line_spaces > 3;
            if (!declined) return ci;
        }
        if (ci == cands.size() || i >= text.size()) break;
        const char ch = text[i];
        if (ch == '\n') {
            i++;
            line_blank = true;
            line_spaces = 0;
            classify(i);
            continue;
        }
        if (ch == ' ') {
            if (line_blank) line_spaces++;
        } else {
            line_blank = false;
            if (ch != '\t' && ch != '\r') non_ws = true;
        }
        if (!prose) { i++; continue; }
        if (ch == '`') {
            size_t r = i;
            while (r < text.size() && text[r] == '`') r++;
            if (!span) span = r - i;
            else if (r - i == span) span = 0;
            i = r;
        } else if (span) {
            i++;
        } else if (text.compare(i, fopen.size(), fopen) == 0) {
            fdepth++; i += fopen.size();
        } else if (text.compare(i, fclose.size(), fclose) == 0) {
            if (fdepth) fdepth--;
            i += fclose.size();
        } else if (text.compare(i, popen.size(), popen) == 0) {
            pdepth++; i += popen.size();
        } else if (text.compare(i, pclose.size(), pclose) == 0) {
            if (pdepth) pdepth--;
            i += pclose.size();
        } else {
            i++;
        }
    }
    return std::string::npos;
}

// Would q27_agent_parse_tool_call decline an opener that starts at pos (text
// = everything this turn emitted through the opener's token)?
inline bool parser_declines_opener(const std::string& text, size_t pos, bool bare) {
    return first_accepted_opener(text, {{pos, bare}}) == std::string::npos;
}

inline const nlohmann::json* tool_parameter_properties(
        const nlohmann::json* tools, const std::string& name) {
    if (!tools || !tools->is_array()) return nullptr;
    for (const auto& tool : *tools) {
        if (!tool.is_object() || !tool.contains("function") ||
            !tool["function"].is_object()) continue;
        const auto& function = tool["function"];
        if (function.value("name", std::string()) != name ||
            !function.contains("parameters") || !function["parameters"].is_object() ||
            !function["parameters"].contains("properties") ||
            !function["parameters"]["properties"].is_object()) continue;
        return &function["parameters"]["properties"];
    }
    return nullptr;
}

inline bool native_xml_type_matches(const nlohmann::json& value,
                                    const std::string& type) {
    if (type == "string") return value.is_string();
    if (type == "boolean") return value.is_boolean();
    if (type == "null") return value.is_null();
    if (type == "integer")
        return value.is_number_integer() || value.is_number_unsigned();
    if (type == "number") return value.is_number();
    if (type == "object") return value.is_object();
    if (type == "array") return value.is_array();
    return false;
}

inline bool native_xml_schema_accepts(const nlohmann::json& value,
                                      const nlohmann::json& schema) {
    if (!schema.is_object()) return true;
    if (schema.contains("type")) {
        const auto& type = schema["type"];
        bool matched = false;
        if (type.is_string()) matched = native_xml_type_matches(value, type.get<std::string>());
        else if (type.is_array())
            for (const auto& branch : type)
                if (branch.is_string() &&
                    native_xml_type_matches(value, branch.get<std::string>())) {
                    matched = true;
                    break;
                }
        if (!matched) return false;
    }
    if (schema.contains("enum") && schema["enum"].is_array()) {
        bool matched = false;
        for (const auto& option : schema["enum"])
            if (option == value) { matched = true; break; }
        if (!matched) return false;
    }
    if (schema.contains("const") && schema["const"] != value) return false;
    if (schema.contains("anyOf") && schema["anyOf"].is_array()) {
        bool matched = false;
        for (const auto& branch : schema["anyOf"])
            if (native_xml_schema_accepts(value, branch)) { matched = true; break; }
        if (!matched) return false;
    }
    if (schema.contains("oneOf") && schema["oneOf"].is_array()) {
        size_t matched = 0;
        for (const auto& branch : schema["oneOf"])
            if (native_xml_schema_accepts(value, branch)) matched++;
        if (matched != 1) return false;
    }
    if (schema.contains("allOf") && schema["allOf"].is_array())
        for (const auto& branch : schema["allOf"])
            if (!native_xml_schema_accepts(value, branch)) return false;
    return true;
}

inline bool coerce_native_xml_parameter(const std::string& value,
                                        const nlohmann::json* schema,
                                        nlohmann::json& out) {
    const nlohmann::json raw = value;
    if (!schema || !schema->is_object()) { out = raw; return true; }
    nlohmann::json parsed;
    bool parsed_ok = false;
    try { parsed = nlohmann::json::parse(value); parsed_ok = true; }
    catch (...) {}
    // Prefer a schema-valid JSON scalar/container over the textual spelling;
    // fall back to the raw string only when the schema permits a string.
    if (parsed_ok && !parsed.is_string() && native_xml_schema_accepts(parsed, *schema)) {
        out = std::move(parsed);
        return true;
    }
    if (native_xml_schema_accepts(raw, *schema)) { out = raw; return true; }
    if (parsed_ok && native_xml_schema_accepts(parsed, *schema)) {
        out = std::move(parsed);
        return true;
    }
    return false;
}

inline bool parse_native_xml_call_body(const std::string& segment,
                                       std::string& name,
                                       nlohmann::json& arguments,
                                       const nlohmann::json* tools = nullptr) {
    size_t begin = segment.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos || segment.compare(begin, 10, "<function=") != 0)
        return false;
    size_t name_begin = begin + 10;
    size_t name_end = segment.find('>', name_begin);
    if (name_end == std::string::npos) return false;
    name = xml_parameter_unescape(segment.substr(name_begin, name_end - name_begin));
    if (name.empty()) return false;
    arguments = nlohmann::json::object();
    size_t cursor = name_end + 1;
    static const std::string parameter_open = "<parameter=";
    static const std::string parameter_close = "</parameter>";
    static const std::string function_close = "</function>";
    while (true) {
        cursor = segment.find_first_not_of(" \t\r\n", cursor);
        if (cursor == std::string::npos) return false;
        if (segment.compare(cursor, function_close.size(), function_close) == 0) {
            cursor += function_close.size();
            cursor = segment.find_first_not_of(" \t\r\n", cursor);
            if (cursor != std::string::npos) return false;
            const nlohmann::json* properties = tool_parameter_properties(tools, name);
            for (auto it = arguments.begin(); it != arguments.end(); ++it) {
                const nlohmann::json* schema =
                    properties && properties->contains(it.key())
                        ? &(*properties)[it.key()] : nullptr;
                nlohmann::json coerced;
                if (!coerce_native_xml_parameter(
                        it.value().get<std::string>(), schema, coerced)) return false;
                it.value() = std::move(coerced);
            }
            return true;
        }
        if (segment.compare(cursor, parameter_open.size(), parameter_open) != 0)
            return false;
        size_t key_begin = cursor + parameter_open.size();
        size_t key_end = segment.find('>', key_begin);
        if (key_end == std::string::npos) return false;
        const std::string key = xml_parameter_unescape(
            segment.substr(key_begin, key_end - key_begin));
        if (key.empty()) return false;
        size_t value_begin = key_end + 1;
        if (value_begin < segment.size() && segment[value_begin] == '\n') value_begin++;
        size_t value_end = segment.find(parameter_close, value_begin);
        if (value_end == std::string::npos) return false;
        size_t trimmed_end = value_end;
        if (trimmed_end > value_begin && segment[trimmed_end - 1] == '\n') trimmed_end--;
        arguments[key] = xml_parameter_unescape(
            segment.substr(value_begin, trimmed_end - value_begin));
        cursor = value_end + parameter_close.size();
    }
}

// Tools preamble, verbatim structure from the chat template. `tools` entries
// must already be in {"type":"function","function":{...}} shape.
inline std::string tools_preamble(const nlohmann::json& tools, bool xml_dialect) {
    std::string s = q27::tools_preamble(tools, std::string(), xml_dialect);
    // Native file tools retain the existing reversible entity transport.
    if (xml_dialect)
        s += "\nXML-escape &, <, and > in parameter values as &amp;, &lt;, and &gt;.";
    return s;
}

inline std::string tools_preamble(const nlohmann::json& tools) {
    return tools_preamble(tools, tool_dialect_xml());
}

} // namespace q27::native_agent
