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
    "read", "search", "write", "overwrite", "edit", "edit_selection", "shell"};
constexpr size_t kToolNameCount = sizeof(kToolNames) / sizeof(kToolNames[0]);

// Indices into kToolNames for body-bearing tools.
constexpr size_t kWriteIdx = 2;
constexpr size_t kOverwriteIdx = 3;
constexpr size_t kEditIdx = 4;
constexpr size_t kEditSelectionIdx = 5;
constexpr size_t kShellIdx = 6;

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

const json& tool_registry() {
    static const json tools = json::array({
            {{"type", "function"}, {"function", {
                {"name", kToolNames[0]},
                {"description", "Read one workspace-relative regular file as exact bytes. Successful output includes short edit-selection handles for line content."},
                {"parameters", {{"type", "object"},
                    {"properties", {{"path", {{"type", "string"}}}}},
                    {"required", json::array({"path"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[1]},
                {"description", "Find literal byte-string occurrences in one workspace-relative regular file. Matching lines include short edit-selection handles."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}},
                        {"needle", {{"type", "string"}}}}},
                    {"required", json::array({"path", "needle"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[kWriteIdx]},
                {"description",
                 "Create one new workspace-relative regular file. JSON args: path only. "
                 "Immediately after </tool_call>, emit the complete file as a markdown "
                 "fenced block (```lang then body then ```). Never put file content in JSON. "
                 "The block's final newline is part of the file; a file without a "
                 "trailing newline is not representable in this transport. "
                 "If the content itself contains fence lines, open with more "
                 "backticks than any fence inside (CommonMark), or the inner fence "
                 "will be taken as the closer. "
                 "Fails if the path already exists; use overwrite for full rewrite or edit "
                 "for a small unique patch."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}}}},
                    {"required", json::array({"path"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[kOverwriteIdx]},
                {"description",
                 "Create or replace the entire contents of one workspace-relative regular "
                 "file. JSON args: path only. Immediately after </tool_call>, emit the "
                 "complete file as a markdown fenced block. The block's final newline is "
                 "part of the file; a file without a trailing newline is not "
                 "representable in this transport. If the content itself contains "
                 "fence lines, open with more backticks than any fence inside "
                 "(CommonMark), or the inner fence will be taken as the closer. "
                 "Prefer overwrite for full-file "
                 "rewrites; use edit only for short unique patches."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}}}},
                    {"required", json::array({"path"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[kEditIdx]},
                {"description",
                 "Replace exactly one short unique nonempty literal match in an existing "
                 "file. JSON args: path and old (old max 512 bytes). Immediately after "
                 "</tool_call>, emit the replacement as a markdown fenced block. Never put "
                 "replacement bytes in JSON. The block's final newline is transport only: "
                 "it is dropped when your old match does not end with a newline. If the "
                 "replacement itself contains fence lines, open with more backticks "
                 "than any fence inside (CommonMark), or the inner fence will be "
                 "taken as the closer. For "
                 "whole-file rewrites use overwrite."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}},
                        {"old", {{"type", "string"}}}}},
                    {"required", json::array({"path", "old"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[kEditSelectionIdx]},
                {"description",
                 "Replace the exact bytes identified by a recent read/search selection "
                 "handle. JSON args: path and selection only. Immediately after "
                 "</tool_call>, emit the replacement as a markdown fenced block."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}}},
                        {"selection", {{"type", "string"}}}}},
                    {"required", json::array({"path", "selection"})},
                    {"additionalProperties", false}}}}}},
            {{"type", "function"}, {"function", {
                {"name", kToolNames[kShellIdx]},
                {"description", "Run one bounded no-fork shell job in the workspace. Pipelines and background jobs are unavailable."},
                {"parameters", {{"type", "object"},
                    {"properties", {
                        {"command", {{"type", "string"}}},
                        {"timeout_ms", {{"type", "integer"}, {"minimum", 1}, {"maximum", 60000}}},
                        {"max_output_bytes", {{"type", "integer"}, {"minimum", 1}, {"maximum", 262144}}}}},
                    {"required", json::array({"command"})},
                    {"additionalProperties", false}}}}}}
        });
    return tools;
}

std::string build_preamble(bool xml_dialect) {
    // Extra protocol note outside the schema list: same-turn fence body.
    std::string base = q27::tools_preamble(tool_registry(), xml_dialect);
    base +=
        "\nBody tools (write, overwrite, edit, edit_selection): after the "
        "closed </tool_call>, emit exactly one markdown fenced body and end "
        "the turn. Example:\n";
    if (xml_dialect)
        base += "<tool_call>\n<function=write>\n<parameter=path>\nhello.py\n"
                "</parameter>\n</function>\n</tool_call>\n";
    else
        base += "<tool_call>{\"name\":\"write\",\"arguments\":{\"path\":\"hello.py\"}}"
                "</tool_call>\n";
    base +=
        "```python\nprint(f\"hi {name}\")\n```\n"
        "Do not emit another tool call as the body. Do not put source code "
        "inside tool-call arguments.\n";
    return base;
}

const std::string& preamble(bool xml_dialect) {
    static const std::string xml_value = build_preamble(true);
    static const std::string json_value = build_preamble(false);
    return xml_dialect ? xml_value : json_value;
}

} // namespace

extern "C" const char *q27_agent_tool_preamble(int xml_dialect) {
    try { return preamble(xml_dialect != 0).c_str(); }
    catch (...) { return nullptr; }
}

extern "C" size_t q27_agent_tool_names(const char *const **names_out) {
    if (!names_out) return 0;
    *names_out = kToolNames;
    return kToolNameCount;
}

extern "C" int q27_agent_tool_name_expects_body(const char *name) {
    if (!name) return 0;
    return !std::strcmp(name, "write") || !std::strcmp(name, "overwrite") ||
           !std::strcmp(name, "edit") || !std::strcmp(name, "edit_selection");
}

extern "C" int q27_agent_tool_kind_expects_body(q27_agent_tool_kind kind) {
    return kind == Q27_TOOL_WRITE || kind == Q27_TOOL_OVERWRITE ||
           kind == Q27_TOOL_EDIT;
}

extern "C" int q27_agent_payload_rejected(const unsigned char *bytes, size_t len,
                                          char *error, size_t error_cap) {
    if (!bytes && len) {
        set_error(error, error_cap, "invalid payload pointer");
        return 1;
    }
    size_t i = 0;
    while (i < len && (bytes[i] == ' ' || bytes[i] == '\t' ||
                       bytes[i] == '\r' || bytes[i] == '\n'))
        ++i;
    if (i >= len) {
        // Empty bodies are legal for empty-file create/delete-style edits.
        return 0;
    }
    auto starts_with = [&](const char *lit) {
        const size_t n = std::strlen(lit);
        return i + n <= len &&
               std::memcmp(bytes + i, lit, n) == 0;
    };
    // Only reject true tool-protocol wrappers. Do not ban bare JSON like
    // package.json ({"name":...}) — that false-positived legitimate files.
    if (starts_with("<tool_call") || starts_with("</tool_call") ||
        starts_with("<q27_raw_payload_request")) {
        set_error(error, error_cap,
                  "payload looks like a tool call; emit file bytes in a "
                  "markdown fence after </tool_call>, not another tool call");
        return 1;
    }
    // Bare tool-call JSON poisons just like the XML wrapper: a body that IS
    // a tool-shaped object (registered "name" + object "arguments") means
    // the model re-emitted the call as content (codex branch-review P2).
    // Ordinary JSON stays legal — package.json has "name" but no "arguments".
    if (bytes[i] == '{') {
        const json candidate =
            json::parse(bytes + i, bytes + len, nullptr, false);
        if (!candidate.is_discarded() && candidate.is_object()) {
            const auto name_it = candidate.find("name");
            const auto args_it = candidate.find("arguments");
            if (name_it != candidate.end() && name_it->is_string() &&
                args_it != candidate.end() && args_it->is_object()) {
                const std::string name = name_it->get<std::string>();
                for (size_t t = 0; t < kToolNameCount; ++t) {
                    if (name == kToolNames[t]) {
                        set_error(error, error_cap,
                                  "payload looks like a tool call; emit file "
                                  "bytes in a markdown fence after "
                                  "</tool_call>, not another tool call");
                        return 1;
                    }
                }
            }
        }
    }
    // Reject pure-whitespace bodies that are not empty after trim when the
    // only non-ws content is more fencing without a closer — handled by
    // extract. Here reject control-character-only junk beyond common text.
    size_t printable = 0;
    for (size_t j = i; j < len; ++j) {
        const unsigned char c = bytes[j];
        if (c == '\t' || c == '\n' || c == '\r' || c >= 0x20) ++printable;
    }
    if (printable == 0 && len > 0) {
        set_error(error, error_cap, "payload has no printable file bytes");
        return 1;
    }
    return 0;
}

extern "C" void q27_agent_tool_call_free(q27_agent_tool_call *call) {
    if (!call) return;
    std::free(call->path);
    std::free(call->input);
    std::free(call->replacement);
    std::free(call->selection);
    std::free(call->body_error);
    *call = q27_agent_tool_call{};
}

static char *dup_error_message(const char *message) {
    if (!message || !*message) message = "unusable tool body";
    const size_t n = std::strlen(message);
    char *copy = static_cast<char *>(std::malloc(n + 1));
    if (!copy) return nullptr;
    std::memcpy(copy, message, n + 1);
    return copy;
}

extern "C" int q27_agent_extract_fenced_body(const unsigned char *bytes,
                                             size_t len, unsigned char **out,
                                             size_t *out_len, char *error,
                                             size_t error_cap,
                                             int eos_reached,
                                             int *transport_ticks) {
    if (out) *out = nullptr;
    if (out_len) *out_len = 0;
    if (!bytes || !out || !out_len) {
        set_error(error, error_cap, "invalid fenced-body extract arguments");
        return 0;
    }
    size_t i = 0;
    while (i < len && (bytes[i] == ' ' || bytes[i] == '\t' ||
                       bytes[i] == '\r' || bytes[i] == '\n'))
        ++i;
    // CommonMark-style: opener of N>=3 backticks; closer is a line of only
    // optional indent spaces + M>=N backticks (so content may contain
    // standalone ``` lines by using a longer outer fence, e.g. ````).
    size_t open_ticks = 0;
    while (i + open_ticks < len && bytes[i + open_ticks] == '`') ++open_ticks;
    if (open_ticks < 3) {
        set_error(error, error_cap,
                  "body tools require a markdown-fenced body after </tool_call>");
        return 0;
    }
    if (transport_ticks) *transport_ticks = (int)open_ticks;
    size_t opening_end = i + open_ticks;
    while (opening_end < len && bytes[opening_end] != '\n' &&
           bytes[opening_end] != '\r') {
        const unsigned char c = bytes[opening_end];
        if (opening_end - (i + open_ticks) >= 31 ||
            !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '_')) {
            set_error(error, error_cap, "invalid markdown fence language label");
            return 0;
        }
        ++opening_end;
    }
    if (opening_end >= len) {
        set_error(error, error_cap, "unterminated markdown fence opener");
        return 0;
    }
    size_t content_start = opening_end + 1;
    if (bytes[opening_end] == '\r') {
        if (content_start >= len || bytes[content_start] != '\n') {
            set_error(error, error_cap, "invalid markdown fence line ending");
            return 0;
        }
        ++content_start;
    }
    // Line-oriented fence transport (CommonMark code-fence semantics): content
    // is every byte up to (not including) the closer line. A non-empty final
    // content line therefore ends with its terminating LF/CRLF — this is not a
    // silent rewrite of an arbitrary byte string; the fence grammar cannot
    // express "last line without line ending" any more than CommonMark can.
    // Empty files use an empty fence body. Models that need a trailing newline
    // (normal for source) already get one; that is the intentional trade for
    // no-JSON escaping.
    for (size_t line_start = content_start; line_start <= len;) {
        size_t line_end = line_start;
        while (line_end < len && bytes[line_end] != '\n') ++line_end;
        size_t logical_end = line_end;
        if (logical_end > line_start && bytes[logical_end - 1] == '\r')
            --logical_end;
        size_t candidate = line_start;
        while (candidate < logical_end && candidate - line_start < 3 &&
               bytes[candidate] == ' ')
            ++candidate;
        size_t ticks_end = candidate;
        while (ticks_end < logical_end && bytes[ticks_end] == '`') ++ticks_end;
        const size_t close_ticks = ticks_end - candidate;
        if (close_ticks >= open_ticks) {
            size_t tail = ticks_end;
            while (tail < logical_end &&
                   (bytes[tail] == ' ' || bytes[tail] == '\t'))
                ++tail;
            if (tail == logical_end) {
                // After a well-formed fence closer, the body is fully
                // delimited. Tolerate trailing protocol-echo junk that greedy
                // models often re-emit after free-decoding the fence
                // (extra </tool_call>), but do NOT accept arbitrary trailing
                // prose/code: a premature ``` closer would otherwise publish a
                // truncated file and silently drop the rest.
                //
                // Residual ambiguity (accepted): a line of only ``` followed by
                // nothing but </tool_call> looks identical to a real closer plus
                // protocol echo. That case cannot be rejected without undoing
                // the T2 fix; after-fence bytes are never intended file content
                // (file bytes live inside the fence). Premature ``` that leaves
                // more source still fails below.
                size_t j = line_end < len ? line_end + 1 : len;
                auto skip_ws = [&]() {
                    while (j < len &&
                           (bytes[j] == ' ' || bytes[j] == '\t' ||
                            bytes[j] == '\r' || bytes[j] == '\n'))
                        ++j;
                };
                skip_ws();
                static const char kCloseTag[] = "</tool_call>";
                constexpr size_t kCloseTagLen = sizeof(kCloseTag) - 1;
                while (j + kCloseTagLen <= len &&
                       std::memcmp(bytes + j, kCloseTag, kCloseTagLen) == 0) {
                    j += kCloseTagLen;
                    skip_ws();
                }
                if (j != len) {
                    set_error(error, error_cap,
                              "non-whitespace after fenced body closer");
                    return 0;
                }
                // Content always ends with the LF preceding the closer line:
                // a payload WITHOUT a trailing newline is not representable in
                // this transport (accepted limitation — the raw-payload channel
                // it replaced accepted arbitrary bytes but was unrecoverable in
                // practice; a no-final-LF write would need an explicit protocol
                // extension, e.g. a call-JSON flag the writer honors).
                const size_t content_len = line_start - content_start;
                unsigned char *copy = static_cast<unsigned char *>(
                    std::malloc(content_len + 1));
                if (!copy) {
                    set_error(error, error_cap, "out of memory extracting body");
                    return 0;
                }
                if (content_len)
                    std::memcpy(copy, bytes + content_start, content_len);
                copy[content_len] = 0;
                *out = copy;
                *out_len = content_len;
                return 1;
            }
        }
        if (line_end >= len) break;
        line_start = line_end + 1;
    }

    // Observed T2/sampled pattern: open ```lang, emit the whole file, hit EOS
    // (or a trailing </tool_call> echo) without a closing fence line. Recover
    // by taking the body as everything after the opener, then stripping only
    // trailing whole-line protocol-echo </tool_call> tags. A premature ```
    // closer with more source still fails above (non-ws after closer).
    // The recovery is ONLY legal at a real EOS: after a max_tokens/output-limit
    // stop the unclosed body is truncated, and publishing it would silently
    // write a partial file (codex branch-review P1).
    if (!eos_reached) {
        set_error(error, error_cap,
                  "generation ended before the closing fence (output limit); "
                  "the truncated body was not published — retry with a smaller "
                  "body or a larger turn budget");
        return 0;
    }
    if (content_start > len) {
        set_error(error, error_cap, "markdown fence body is not closed");
        return 0;
    }
    size_t end = len;
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    };
    // Strip trailing whole-line </tool_call> protocol echo only. A content
    // line that ends with the literal tag (e.g. a string/docs sample) is kept
    // because the tag is not line-leading. Residual ambiguity: a file whose
    // *entire last line* is </tool_call> under an unclosed fence is
    // indistinguishable from protocol echo and is stripped (same class as
    // closed-fence trailing-echo residual).
    static const char kCloseTag[] = "</tool_call>";
    constexpr size_t kCloseTagLen = sizeof(kCloseTag) - 1;
    for (;;) {
        size_t t = end;
        while (t > content_start && is_ws(bytes[t - 1])) --t;
        if (t < content_start + kCloseTagLen) {
            // Too short to hold </tool_call>. If the body is only whitespace,
            // treat as empty; otherwise keep the short real content.
            if (t == content_start) end = content_start;
            break;
        }
        if (std::memcmp(bytes + (t - kCloseTagLen), kCloseTag,
                        kCloseTagLen) != 0)
            break;
        const size_t tag_at = t - kCloseTagLen;
        // Whole trailing line only: from previous '\n' (or body start) to the
        // tag, only spaces/tabs — so "  </tool_call>" is echo, but
        // "note: </tool_call>" is content.
        size_t line_start = tag_at;
        while (line_start > content_start && bytes[line_start - 1] != '\n')
            --line_start;
        bool protocol_line = true;
        for (size_t p = line_start; p < tag_at; ++p) {
            if (bytes[p] != ' ' && bytes[p] != '\t') {
                protocol_line = false;
                break;
            }
        }
        if (!protocol_line) break;
        end = line_start;
    }
    // Optional dangling closer line at EOF (only spaces + >=open_ticks
    // backticks). end = start of that line so content keeps the preceding LF
    // — same inclusion rule as a normal closed fence.
    {
        size_t scan = end;
        while (scan > content_start && is_ws(bytes[scan - 1])) --scan;
        size_t line = scan;
        while (line > content_start && bytes[line - 1] != '\n') --line;
        size_t p = line;
        while (p < scan && bytes[p] == ' ') ++p;
        size_t ticks = 0;
        while (p + ticks < scan && bytes[p + ticks] == '`') ++ticks;
        size_t q = p + ticks;
        while (q < scan && (bytes[q] == ' ' || bytes[q] == '\t')) ++q;
        if (ticks >= open_ticks && q == scan) end = line;
    }
    const size_t content_len = end > content_start ? end - content_start : 0;
    // Empty after stripping protocol echo is not a deliberate empty file
    // (those use a closed empty fence). Fail closed so the model retries.
    if (content_len == 0) {
        set_error(error, error_cap, "markdown fence body is not closed");
        return 0;
    }
    // Fail closed if the unclosed region contains another tool-call opener:
    // recovery would otherwise publish a second call as file bytes (closed
    // fences already reject non-ws after the closer).
    static const char kOpenTag[] = "<tool_call";
    constexpr size_t kOpenTagLen = sizeof(kOpenTag) - 1;
    for (size_t j = content_start; j + kOpenTagLen <= end; ++j) {
        if (std::memcmp(bytes + j, kOpenTag, kOpenTagLen) == 0) {
            set_error(error, error_cap,
                      "unclosed fence body contains another tool call");
            return 0;
        }
    }
    unsigned char *copy =
        static_cast<unsigned char *>(std::malloc(content_len + 1));
    if (!copy) {
        set_error(error, error_cap, "out of memory extracting body");
        return 0;
    }
    std::memcpy(copy, bytes + content_start, content_len);
    copy[content_len] = 0;
    *out = copy;
    *out_len = content_len;
    return 1;
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
    char *error, size_t error_cap, int eos_reached, int xml_dialect) {
    if (call) *call = q27_agent_tool_call{};
    // Same-turn bodies can approach the filesystem tool's 8 MiB file cap; leave
    // headroom for ChatML prose, the JSON header, and outer fence lines.
    constexpr size_t kMaxToolCallParseBytes =
        8u * 1024u * 1024u + 256u * 1024u;
    if (!bytes || !call || len > kMaxToolCallParseBytes) {
        set_error(error, error_cap, "invalid or oversized tool-call parser input");
        return Q27_TOOL_CALL_INVALID;
    }
    try {
        std::string text(reinterpret_cast<const char *>(bytes), len);
        static const std::string open = "<tool_call>";
        static const std::string close = "</tool_call>";
        size_t begin = text.find(open);
        if (begin == std::string::npos) {
            if (!xml_dialect) return Q27_TOOL_CALL_NONE;
            bool saw_top_level_candidate = false;
            size_t native_scope = 0;
            const size_t think_close = text.rfind("</think>");
            if (think_close != std::string::npos) native_scope = think_close + 8;
            size_t native_begin = text.find("<function=", native_scope);
            size_t native_end = std::string::npos;
            while (native_begin != std::string::npos) {
                const bool fenced =
                    q27::inside_markdown_fence(text, native_begin, native_scope);
                const bool boundary = q27::unambiguous_native_xml_boundary(
                    text, native_begin, native_scope);
                const bool nested = q27::nested_native_xml_candidate(
                    text, native_begin, native_scope);
                if (!fenced && boundary && !nested) {
                    saw_top_level_candidate = true;
                    const size_t native_close =
                        text.find("</function>", native_begin);
                    if (native_close != std::string::npos) {
                        const size_t candidate_end = native_close + 11;
                        std::string native_name;
                        json native_arguments;
                        bool declared = false;
                        if (q27::parse_native_xml_call_body(
                                text.substr(native_begin, candidate_end - native_begin),
                                native_name, native_arguments, &tool_registry())) {
                            for (size_t i = 0; i < kToolNameCount; ++i)
                                if (native_name == kToolNames[i]) {
                                    declared = true;
                                    break;
                                }
                        }
                        if (declared) {
                            native_end = candidate_end;
                            break;
                        }
                    }
                }
                native_begin = text.find("<function=", native_begin + 10);
            }
            if (native_end == std::string::npos) {
                if (!saw_top_level_candidate) return Q27_TOOL_CALL_NONE;
                set_error(error, error_cap, "invalid native XML tool call");
                return Q27_TOOL_CALL_INVALID;
            }
            text.insert(native_end, close);
            text.insert(native_begin, open);
            begin = native_begin;
        }
        const size_t call_start = begin + open.size();
        const size_t end = text.find(close, call_start);
        if (end == std::string::npos) {
            set_error(error, error_cap, "tool call is not closed exactly once");
            return Q27_TOOL_CALL_INVALID;
        }
        // A second opener before the first closer is a second call inside the
        // tool-call body. Openers after the closer are body text (allowed only
        // for body tools, and rejected later if they form the body).
        const size_t second_open = text.find(open, begin + open.size());
        if (second_open != std::string::npos && second_open < end) {
            set_error(error, error_cap, "multiple tool calls are not allowed in one turn");
            return Q27_TOOL_CALL_INVALID;
        }
        const size_t after = end + close.size();
        const std::string call_body = text.substr(call_start, end - call_start);
        std::string native_name;
        json native_arguments;
        json outer;
        if (q27::parse_native_xml_call_body(call_body, native_name, native_arguments,
                                            &tool_registry()))
            outer = {{"name", native_name}, {"arguments", std::move(native_arguments)}};
        else
            outer = json::parse(call_body);
        if (!only_keys(outer, {"name", "arguments"}) ||
            !outer["name"].is_string() || !outer["arguments"].is_object()) {
            set_error(error, error_cap, "tool call must contain only name and object arguments");
            return Q27_TOOL_CALL_INVALID;
        }
        const std::string name = outer["name"].get<std::string>();
        const json& args = outer["arguments"];
        const int expects_body = q27_agent_tool_name_expects_body(name.c_str());
        if (!expects_body) {
            if (text.find(close, after) != std::string::npos) {
                set_error(error, error_cap, "tool call is not closed exactly once");
                return Q27_TOOL_CALL_INVALID;
            }
            if (second_open != std::string::npos) {
                set_error(error, error_cap, "multiple tool calls are not allowed in one turn");
                return Q27_TOOL_CALL_INVALID;
            }
            for (size_t i = after; i < text.size(); ++i) {
                const char c = text[i];
                if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                    set_error(error, error_cap, "text after tool call is forbidden");
                    return Q27_TOOL_CALL_INVALID;
                }
            }
        }
        q27_agent_tool_request request{};
        request.timeout_ms = 30000;
        request.max_output_bytes = 256u * 1024u;
        unsigned char *path = nullptr, *input = nullptr, *replacement = nullptr;
        unsigned char *selection = nullptr;
        unsigned char *body = nullptr;
        size_t body_len = 0;
        struct LocalCleanup {
            unsigned char *&path;
            unsigned char *&input;
            unsigned char *&replacement;
            unsigned char *&selection;
            unsigned char *&body;
            ~LocalCleanup() {
                std::free(path); std::free(input); std::free(replacement);
                std::free(selection); std::free(body);
            }
        } cleanup{path, input, replacement, selection, body};
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
        } else if (name == kToolNames[kWriteIdx] && only_keys(args, {"path"})) {
            request.kind = Q27_TOOL_WRITE;
            valid = copy_string(args, "path", &path, &path_len, true) &&
                    path_len > 0;
        } else if (name == kToolNames[kOverwriteIdx] && only_keys(args, {"path"})) {
            request.kind = Q27_TOOL_OVERWRITE;
            valid = copy_string(args, "path", &path, &path_len, true) &&
                    path_len > 0;
        } else if (name == kToolNames[kEditIdx] && only_keys(args, {"path", "old"})) {
            request.kind = Q27_TOOL_EDIT;
            valid = copy_string(args, "path", &path, &path_len, true) &&
                    path_len > 0 &&
                    copy_string(args, "old", &input, &request.input_len, false) &&
                    request.input_len > 0 &&
                    request.input_len <= Q27_AGENT_EDIT_OLD_MAX_BYTES;
            if (valid == false && path && input &&
                request.input_len > Q27_AGENT_EDIT_OLD_MAX_BYTES) {
                set_error(error, error_cap,
                          "edit old is too long; use overwrite for full-file "
                          "rewrites or a shorter unique old match");
                return Q27_TOOL_CALL_INVALID;
            }
        } else if (name == kToolNames[kEditSelectionIdx] &&
                   only_keys(args, {"path", "selection"})) {
            request.kind = Q27_TOOL_EDIT;
            size_t selection_len = 0;
            valid = copy_string(args, "path", &path, &path_len, true) &&
                    path_len > 0 &&
                    copy_string(args, "selection", &selection,
                                &selection_len, true) &&
                    selection_len > 0 && selection_len <= 64;
        } else if (name == kToolNames[kShellIdx] && args.is_object() &&
                   args.size() >= 1 && args.size() <= 3 && args.contains("command")) {
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

        int missing_body = 0;
        int fence_ticks = 0;
        char *body_error = nullptr;
        if (expects_body) {
            const unsigned char *tail =
                reinterpret_cast<const unsigned char *>(text.data() + after);
            const size_t tail_len = text.size() - after;
            size_t non_ws = 0;
            while (non_ws < tail_len &&
                   (tail[non_ws] == ' ' || tail[non_ws] == '\t' ||
                    tail[non_ws] == '\r' || tail[non_ws] == '\n'))
                ++non_ws;
            // Body-tool schema is valid even when the fence is missing or
            // unusable: return VALID + missing_body so the control loop can
            // soft-fail with a tool_response and let the model retry.
            if (non_ws >= tail_len) {
                missing_body = 1;
                body_error = dup_error_message(
                    "body tool requires a markdown-fenced body after "
                    "</tool_call>; do not emit another tool call");
            } else if (!q27_agent_extract_fenced_body(tail, tail_len, &body,
                                                      &body_len, error,
                                                      error_cap, eos_reached,
                                                      &fence_ticks)) {
                missing_body = 1;
                body_error = dup_error_message(
                    (error && error_cap && error[0]) ? error :
                    "body tools require a markdown-fenced body after "
                    "</tool_call>");
            } else if (q27_agent_payload_rejected(body, body_len, error,
                                                  error_cap)) {
                std::free(body);
                body = nullptr;
                body_len = 0;
                missing_body = 1;
                body_error = dup_error_message(
                    (error && error_cap && error[0]) ? error :
                    "unusable tool body");
            } else if (request.kind == Q27_TOOL_WRITE ||
                       request.kind == Q27_TOOL_OVERWRITE) {
                // Whole-file tools: body is file content (optional outer
                // language-matched unwrap happens later at publish time only
                // for write/overwrite with a known source extension).
                std::free(input);
                input = body;
                request.input_len = body_len;
                body = nullptr;
            } else {
                // Edit tools: JSON `old`/selection stay in input; fence is
                // replacement bytes.
                std::free(replacement);
                replacement = body;
                request.replacement_len = body_len;
                body = nullptr;
            }
            if (missing_body && !body_error) {
                body_error = dup_error_message("unusable tool body");
            }
            if (missing_body && !body_error) {
                set_error(error, error_cap, "out of memory recording body error");
                return Q27_TOOL_CALL_INVALID;
            }
        }

        request.path = reinterpret_cast<char *>(path);
        request.input = input;
        request.replacement = replacement;
        call->request = request;
        call->path = reinterpret_cast<char *>(path);
        call->input = input;
        call->replacement = replacement;
        call->selection = reinterpret_cast<char *>(selection);
        call->missing_body = missing_body;
        call->body_error = body_error;
        call->body_fence_ticks = missing_body ? 0 : fence_ticks;
        path = input = replacement = selection = nullptr;
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
