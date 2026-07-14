// CPU-only malformed model/tokenizer regression tests.
#include "loader.h"
#include "tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

struct TempFile {
    std::string path;

    explicit TempFile(const std::vector<uint8_t>& bytes) {
        char pattern[] = "/tmp/q27-artifact-XXXXXX";
        int fd = mkstemp(pattern);
        if (fd < 0) throw std::runtime_error("mkstemp failed");
        path = pattern;
        size_t done = 0;
        while (done < bytes.size()) {
            ssize_t n = write(fd, bytes.data() + done, bytes.size() - done);
            if (n <= 0) {
                close(fd);
                unlink(path.c_str());
                throw std::runtime_error("temp write failed");
            }
            done += (size_t)n;
        }
        close(fd);
    }

    ~TempFile() { if (!path.empty()) unlink(path.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

template <typename T> void append(std::vector<uint8_t>& out, T value) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(value));
}

void append_bytes(std::vector<uint8_t>& out, const std::string& value) {
    out.insert(out.end(), value.begin(), value.end());
}

struct TensorEntry {
    std::string name;
    uint8_t dtype;
    std::vector<uint64_t> shape;
    uint64_t data_off;
    uint64_t data_size;
    uint64_t scales_off = 0;
    uint64_t scales_size = 0;
};

std::vector<uint8_t> model_file(const std::vector<TensorEntry>& entries, size_t payload_size) {
    std::vector<uint8_t> out;
    append<uint32_t>(out, 0x46373251);
    append<uint32_t>(out, 1);
    append<uint32_t>(out, (uint32_t)entries.size());
    append<uint32_t>(out, 2);
    append_bytes(out, "{}");
    for (const TensorEntry& e : entries) {
        append<uint16_t>(out, (uint16_t)e.name.size());
        append_bytes(out, e.name);
        append<uint8_t>(out, e.dtype);
        append<uint8_t>(out, (uint8_t)e.shape.size());
        for (uint64_t dim : e.shape) append<uint64_t>(out, dim);
        append<uint64_t>(out, e.data_off);
        append<uint64_t>(out, e.data_size);
        append<uint64_t>(out, e.scales_off);
        append<uint64_t>(out, e.scales_size);
    }
    out.resize((out.size() + 255) / 256 * 256);
    out.resize(out.size() + payload_size, 0x5a);
    return out;
}

void append_lp(std::vector<uint8_t>& out, const std::string& value) {
    append<uint16_t>(out, (uint16_t)value.size());
    append_bytes(out, value);
}

std::vector<uint8_t> tokenizer_file(const std::vector<std::string>& tokens,
                                    const std::vector<uint8_t>& types,
                                    uint32_t bos = 0, uint32_t eos = 1,
                                    uint32_t version = 1) {
    std::vector<uint8_t> out;
    append<uint32_t>(out, 0x54373251);
    append<uint32_t>(out, version);
    append<uint32_t>(out, (uint32_t)tokens.size());
    append<uint32_t>(out, bos);
    append<uint32_t>(out, eos);
    for (const std::string& token : tokens) append_lp(out, token);
    out.insert(out.end(), types.begin(), types.end());
    append<uint32_t>(out, 0); // merges
    return out;
}

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", name);
        failures++;
    }
}

void expect_error(const char* name, const std::function<void()>& fn) {
    try {
        fn();
        fprintf(stderr, "FAIL: %s accepted malformed artifact\n", name);
        failures++;
    } catch (const std::exception&) {
    }
}

void open_model(const std::vector<uint8_t>& bytes) {
    TempFile file(bytes);
    (void)q27::Model::open(file.path);
}

void open_tokenizer(const std::vector<uint8_t>& bytes) {
    TempFile file(bytes);
    q27::Tokenizer tokenizer(file.path);
}

} // namespace

int main() {
    const TensorEntry f32{"x", 0, {2}, 0, 8};
    {
        TempFile file(model_file({f32}, 8));
        q27::Model model = q27::Model::open(file.path);
        check(model.tensors.size() == 1 && model.get("x").n_elements() == 2,
              "valid model");
    }

    auto truncated_model = model_file({f32}, 8);
    truncated_model.resize(12);
    expect_error("truncated model", [&] { open_model(truncated_model); });

    TensorEntry huge_off = f32;
    huge_off.data_off = std::numeric_limits<uint64_t>::max() - 255;
    expect_error("overflowing model offset", [&] { open_model(model_file({huge_off}, 8)); });

    TensorEntry bad_size = f32;
    bad_size.data_size = 4;
    expect_error("wrong tensor byte size", [&] { open_model(model_file({bad_size}, 8)); });

    TensorEntry duplicate = f32;
    duplicate.data_off = 256;
    expect_error("duplicate tensor name", [&] {
        open_model(model_file({f32, duplicate}, 264));
    });

    TensorEntry overlap = f32;
    overlap.name = "y";
    expect_error("overlapping tensor data", [&] {
        open_model(model_file({f32, overlap}, 8));
    });

    TensorEntry bad_dtype = f32;
    bad_dtype.dtype = 9;
    expect_error("unknown dtype", [&] { open_model(model_file({bad_dtype}, 8)); });

    TensorEntry shape_overflow{"x", 0, {std::numeric_limits<uint64_t>::max(), 2}, 0, 8};
    expect_error("shape overflow", [&] { open_model(model_file({shape_overflow}, 8)); });

    TensorEntry q4{"q", 3, {64}, 0, 32, 256, 2};
    {
        TempFile file(model_file({q4}, 258));
        q27::Model model = q27::Model::open(file.path);
        check(model.get("q").scales_size == 2, "valid quantized model");
    }

    auto valid_tok = tokenizer_file({"a", "b"}, {1, 1});
    {
        TempFile file(valid_tok);
        q27::Tokenizer tok(file.path);
        check(tok.bos() == 0 && tok.eos() == 1 && tok.token_id("b") == 1,
              "valid tokenizer");
    }

    auto truncated_tok = valid_tok;
    truncated_tok.resize(7);
    expect_error("truncated tokenizer", [&] { open_tokenizer(truncated_tok); });
    expect_error("tokenizer version", [&] {
        open_tokenizer(tokenizer_file({"a", "b"}, {1, 1}, 0, 1, 2));
    });
    expect_error("tokenizer special id", [&] {
        open_tokenizer(tokenizer_file({"a", "b"}, {1, 1}, 0, 2));
    });
    expect_error("empty control token", [&] {
        open_tokenizer(tokenizer_file({"", "b"}, {3, 1}));
    });
    expect_error("duplicate token", [&] {
        open_tokenizer(tokenizer_file({"a", "a"}, {1, 1}));
    });
    auto trailing_tok = valid_tok;
    trailing_tok.push_back(0);
    expect_error("tokenizer trailing data", [&] { open_tokenizer(trailing_tok); });

    if (failures) {
        fprintf(stderr, "%d artifact test(s) failed\n", failures);
        return 1;
    }
    puts("artifact validation: OK");
    return 0;
}
