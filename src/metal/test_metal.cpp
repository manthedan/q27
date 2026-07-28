#include "metal_backend.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

template <typename T> void append(std::vector<uint8_t>& out, T value) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(value));
}

int test_mmap_upload(q27::MetalBackend& backend) {
    const float weights[12] = {1, 2, 3, 4, -2, 1, 0.5f, 3, 0, 0, 0, 0};
    std::vector<uint8_t> file;
    append<uint32_t>(file, 0x46373251);
    append<uint32_t>(file, 1);
    append<uint32_t>(file, 1);
    append<uint32_t>(file, 2);
    file.push_back('{'); file.push_back('}');
    append<uint16_t>(file, 1); file.push_back('m');
    append<uint8_t>(file, 0); append<uint8_t>(file, 2);
    append<uint64_t>(file, 3); append<uint64_t>(file, 4);
    append<uint64_t>(file, 0); append<uint64_t>(file, sizeof(weights));
    append<uint64_t>(file, 0); append<uint64_t>(file, 0);
    file.resize(256);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(weights);
    file.insert(file.end(), bytes, bytes + sizeof(weights));

    char path[] = "/tmp/q27-metal-model-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    size_t done = 0;
    while (done < file.size()) {
        ssize_t n = write(fd, file.data() + done, file.size() - done);
        if (n <= 0) { close(fd); unlink(path); return 1; }
        done += (size_t)n;
    }
    close(fd);
    try {
        q27::Model model = q27::Model::open(path);
        unlink(path);
        q27::BackendTensor weight = backend.upload(model, model.get("m"));
        if (weight.data->size() < file.size() || weight.data->size() % (uint64_t)getpagesize() != 0) {
            fprintf(stderr, "mmap upload: wrapped length %llu is not a page-aligned cover of %zu bytes\n",
                    (unsigned long long)weight.data->size(), file.size());
            return 1;
        }
        const float x[4] = {2, -1, 0.5f, 3};
        auto device_x = backend.allocate(sizeof(x));
        auto device_y = backend.allocate(3 * sizeof(float));
        backend.write(*device_x, 0, x, sizeof(x));
        backend.begin_commands();
        backend.matvec(weight, *device_x, *device_y);
        backend.end_commands();
        float got[3];
        backend.read(*device_y, 0, got, sizeof(got));
        return std::fabs(got[0] - 13.5f) > 1e-4f || std::fabs(got[1] - 4.25f) > 1e-4f;
    } catch (...) {
        unlink(path);
        throw;
    }
}

bool close(float got, float want, float tolerance = 1e-4f) {
    return std::fabs(got - want) <= tolerance * std::fmax(1.0f, std::fabs(want));
}

int test_dispatch_validation(q27::MetalBackend& backend) {
    std::vector<int8_t> short_data(1, 1);
    std::vector<uint16_t> short_scales(1, 0x3c00);
    q27::Tensor tensor;
    tensor.name="short-embedding"; tensor.dtype=q27::DType::Q8_G128; tensor.shape={2,128};
    tensor.data=reinterpret_cast<const uint8_t*>(short_data.data()); tensor.data_size=short_data.size();
    tensor.scales=reinterpret_cast<const uint8_t*>(short_scales.data()); tensor.scales_size=2;
    auto weight=backend.upload(tensor); auto out=backend.allocate(128*4);
    bool rejected=false;
    backend.begin_commands();
    try { backend.embedding_q8(weight,0,*out); }
    catch(const std::runtime_error&) { rejected=true; }
    backend.abort_commands();
    if(!rejected) { fprintf(stderr,"undersized embedding was not rejected\n"); return 1; }

    std::vector<int8_t> bad_group_data(32,1); std::vector<uint16_t> bad_group_scale(1,0x3c00);
    q27::Tensor bad_group; bad_group.name="bad-group"; bad_group.dtype=q27::DType::Q8_G128;
    bad_group.shape={1,32}; bad_group.data=reinterpret_cast<const uint8_t*>(bad_group_data.data());
    bad_group.data_size=bad_group_data.size(); bad_group.scales=reinterpret_cast<const uint8_t*>(bad_group_scale.data()); bad_group.scales_size=2;
    auto bad_weight=backend.upload(bad_group); auto bad_x=backend.allocate(32*4),bad_y=backend.allocate(4);
    rejected=false; try { backend.matvec(bad_weight,*bad_x,*bad_y); } catch(const std::runtime_error&) { rejected=true; }
    if(!rejected) { fprintf(stderr,"invalid quantization group was not rejected\n"); return 1; }

    // Aborting a failed explicit batch must leave the backend reusable.
    std::vector<float> x={1,2,3,4}; auto xb=backend.allocate(16),yb=backend.allocate(16);
    backend.write(*xb,0,x.data(),16); backend.copy(*xb,0,*yb,0,16);
    float got[4]={}; backend.read(*yb,0,got,16);
    for(int i=0;i<4;i++) if(got[i]!=x[i]) return 1;
    return 0;
}

int test_f32(q27::MetalBackend& backend) {
    std::vector<float> weights = {
        1, 2, 3, 4,
        -2, 1, 0.5f, 3,
        0, 0, 0, 0,
    };
    std::vector<float> x = {2, -1, 0.5f, 3};
    q27::Tensor tensor;
    tensor.name = "f32-test";
    tensor.dtype = q27::DType::F32;
    tensor.shape = {3, 4};
    tensor.data = reinterpret_cast<const uint8_t*>(weights.data());
    tensor.data_size = weights.size() * sizeof(float);

    q27::BackendTensor device_weight = backend.upload(tensor);
    auto device_x = backend.allocate(x.size() * sizeof(float));
    auto device_y = backend.allocate(3 * sizeof(float));
    backend.write(*device_x, 0, x.data(), x.size() * sizeof(float));
    backend.matvec(device_weight, *device_x, *device_y);
    float got[3];
    backend.read(*device_y, 0, got, sizeof(got));

    const float want[3] = {13.5f, 4.25f, 0.0f};
    for (int i = 0; i < 3; i++) {
        if (!close(got[i], want[i])) {
            fprintf(stderr, "F32 row %d: got %.7g, want %.7g\n", i, got[i], want[i]);
            return 1;
        }
    }
    return 0;
}

int test_f16(q27::MetalBackend& backend) {
    // 1, 2, -1, 0.5 and -2, 0, 3, 1 as IEEE-754 binary16.
    std::vector<uint16_t> weights = {
        0x3c00, 0x4000, 0xbc00, 0x3800,
        0xc000, 0x0000, 0x4200, 0x3c00,
    };
    std::vector<float> x = {2, -1, 0.5f, 3};
    q27::Tensor tensor;
    tensor.name = "f16-test";
    tensor.dtype = q27::DType::F16;
    tensor.shape = {2, 4};
    tensor.data = reinterpret_cast<const uint8_t*>(weights.data());
    tensor.data_size = weights.size() * sizeof(uint16_t);

    q27::BackendTensor device_weight = backend.upload(tensor);
    auto device_x = backend.allocate(x.size() * sizeof(float));
    auto device_y = backend.allocate(2 * sizeof(float));
    backend.write(*device_x, 0, x.data(), x.size() * sizeof(float));
    backend.matvec(device_weight, *device_x, *device_y);
    float got[2];
    backend.read(*device_y, 0, got, sizeof(got));
    const float want[2] = {1.0f, 0.5f};
    auto paired_y=backend.allocate(2*sizeof(float));
    backend.matvec_pair(device_weight,*device_y,device_weight,*paired_y,*device_x);
    float paired[2]; backend.read(*paired_y,0,paired,sizeof(paired));
    for(int row=0;row<2;row++) if(!close(paired[row],want[row])) return 1;
    for (int row = 0; row < 2; row++) {
        if (!close(got[row], want[row])) {
            fprintf(stderr, "F16 row %d: got %.7g, want %.7g\n", row, got[row], want[row]);
            return 1;
        }
    }
    return 0;
}

int test_q8(q27::MetalBackend& backend) {
    constexpr int rows = 2, cols = 128;
    std::vector<int8_t> weights(rows * cols);
    std::vector<float> x(cols);
    float want[rows] = {};
    for (int i = 0; i < cols; i++) x[i] = (float)((i % 9) - 4) / 8.0f;
    for (int row = 0; row < rows; row++) {
        const float scale = row == 0 ? 1.0f : 0.5f;
        for (int i = 0; i < cols; i++) {
            int q = row == 0 ? (i % 31) - 15 : 12 - (i % 25);
            weights[row * cols + i] = (int8_t)q;
            want[row] += q * scale * x[i];
        }
    }
    std::vector<uint16_t> scales = {0x3c00, 0x3800};
    q27::Tensor tensor;
    tensor.name = "q8-test";
    tensor.dtype = q27::DType::Q8_G128;
    tensor.shape = {rows, cols};
    tensor.data = reinterpret_cast<const uint8_t*>(weights.data());
    tensor.data_size = weights.size();
    tensor.scales = reinterpret_cast<const uint8_t*>(scales.data());
    tensor.scales_size = scales.size() * sizeof(uint16_t);

    q27::BackendTensor device_weight = backend.upload(tensor);
    auto device_x = backend.allocate(x.size() * sizeof(float));
    auto device_y = backend.allocate(rows * sizeof(float));
    backend.write(*device_x, 0, x.data(), x.size() * sizeof(float));
    backend.matvec(device_weight, *device_x, *device_y);
    float got[rows];
    backend.read(*device_y, 0, got, sizeof(got));
    for (int row = 0; row < rows; row++) {
        if (!close(got[row], want[row])) {
            fprintf(stderr, "Q8 row %d: got %.7g, want %.7g\n", row, got[row], want[row]);
            return 1;
        }
    }
    auto quantized=backend.allocate_quantized(cols);
    backend.begin_commands(); backend.quantize(*device_x,quantized); backend.matvec_quantized(device_weight,quantized,*device_y); backend.end_commands();
    backend.read(*device_y,0,got,sizeof(got));
    for(int row=0;row<rows;row++) if(!close(got[row],want[row],2e-2f)) {
        fprintf(stderr,"Q8 quantized row %d: got %.7g, want %.7g\n",row,got[row],want[row]); return 1;
    }
    auto pair_y=backend.allocate(rows*sizeof(float));
    backend.matvec_quantized_pair(device_weight,*device_y,device_weight,*pair_y,quantized);
    float pair_got[rows]; backend.read(*pair_y,0,pair_got,sizeof(pair_got));
    for(int row=0;row<rows;row++) if(!close(pair_got[row],got[row])) return 1;
    return 0;
}

int test_q4(q27::MetalBackend& backend) {
    constexpr int rows = 2, cols = 64;
    std::vector<uint8_t> packed(rows * cols / 2);
    std::vector<float> x(cols);
    float want[rows] = {};
    for (int i = 0; i < cols; i++) x[i] = (float)((i % 7) - 3) / 4.0f;
    for (int row = 0; row < rows; row++) {
        const float scale = row == 0 ? 1.0f : 0.5f;
        for (int i = 0; i < cols; i += 2) {
            const int q0 = row == 0 ? (i % 16) - 8 : 7 - (i % 16);
            const int q1 = row == 0 ? ((i + 1) % 16) - 8 : 7 - ((i + 1) % 16);
            packed[row * cols / 2 + i / 2] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
            want[row] += q0 * scale * x[i] + q1 * scale * x[i + 1];
        }
    }
    // IEEE-754 binary16 encodings for 1.0 and 0.5.
    std::vector<uint16_t> scales = {0x3c00, 0x3800};

    q27::Tensor tensor;
    tensor.name = "q4-test";
    tensor.dtype = q27::DType::Q4_G64;
    tensor.shape = {rows, cols};
    tensor.data = packed.data();
    tensor.data_size = packed.size();
    tensor.scales = reinterpret_cast<const uint8_t*>(scales.data());
    tensor.scales_size = scales.size() * sizeof(uint16_t);

    q27::BackendTensor device_weight = backend.upload(tensor);
    auto device_x = backend.allocate(x.size() * sizeof(float));
    auto device_y = backend.allocate(rows * sizeof(float));
    backend.write(*device_x, 0, x.data(), x.size() * sizeof(float));
    backend.matvec(device_weight, *device_x, *device_y);
    float got[rows];
    backend.read(*device_y, 0, got, sizeof(got));

    for (int row = 0; row < rows; row++) {
        if (!close(got[row], want[row])) {
            fprintf(stderr, "Q4 row %d: got %.7g, want %.7g\n", row, got[row], want[row]);
            return 1;
        }
    }
    auto quantized=backend.allocate_quantized(cols);
    backend.begin_commands(); backend.quantize(*device_x,quantized); backend.matvec_quantized(device_weight,quantized,*device_y); backend.end_commands();
    backend.read(*device_y,0,got,sizeof(got));
    for(int row=0;row<rows;row++) if(!close(got[row],want[row],2e-2f)) {
        fprintf(stderr,"Q4 quantized row %d: got %.7g, want %.7g\n",row,got[row],want[row]); return 1;
    }
    auto pair_y=backend.allocate(rows*sizeof(float));
    backend.matvec_quantized_pair(device_weight,*device_y,device_weight,*pair_y,quantized);
    float pair_got[rows]; backend.read(*pair_y,0,pair_got,sizeof(pair_got));
    for(int row=0;row<rows;row++) if(!close(pair_got[row],got[row])) return 1;
    return 0;
}

int test_t2(q27::MetalBackend& backend) {
    constexpr int rows = 2, cols = 128;
    std::vector<uint8_t> packed(rows * cols / 4, 0);
    std::vector<float> x(cols);
    float want[rows] = {};
    int8_t q_ref[rows][cols];
    for (int i = 0; i < cols; i++) x[i] = (float)((i % 11) - 5) / 5.0f;
    for (int row = 0; row < rows; row++) {
        const float scale = row == 0 ? 1.0f : 0.5f;
        for (int i = 0; i < cols; i++) {
            const int q = ((i * 5 + row * 3) % 3) - 1;   // {-1, 0, +1}
            q_ref[row][i] = (int8_t)q;
            packed[(size_t)row * cols / 4 + i / 4] |= (uint8_t)((q + 1) << ((i % 4) * 2));
            want[row] += q * scale * x[i];
        }
    }
    // IEEE-754 binary16 encodings for 1.0 and 0.5.
    std::vector<uint16_t> scales = {0x3c00, 0x3800};

    q27::Tensor tensor;
    tensor.name = "t2-test";
    tensor.dtype = q27::DType::T2_G128;
    tensor.shape = {rows, cols};
    tensor.data = packed.data();
    tensor.data_size = packed.size();
    tensor.scales = reinterpret_cast<const uint8_t*>(scales.data());
    tensor.scales_size = scales.size() * sizeof(uint16_t);

    q27::BackendTensor device_weight = backend.upload(tensor);
    auto device_x = backend.allocate(x.size() * sizeof(float));
    auto device_y = backend.allocate(rows * sizeof(float));
    backend.write(*device_x, 0, x.data(), x.size() * sizeof(float));
    backend.matvec(device_weight, *device_x, *device_y);
    float got[rows];
    backend.read(*device_y, 0, got, sizeof(got));
    for (int row = 0; row < rows; row++) {
        if (!close(got[row], want[row])) {
            fprintf(stderr, "T2 row %d: got %.7g, want %.7g\n", row, got[row], want[row]);
            return 1;
        }
    }
    auto quantized = backend.allocate_quantized(cols);
    backend.begin_commands(); backend.quantize(*device_x, quantized);
    backend.matvec_quantized(device_weight, quantized, *device_y); backend.end_commands();
    backend.read(*device_y, 0, got, sizeof(got));
    for (int row = 0; row < rows; row++) if (!close(got[row], want[row], 2e-2f)) {
        fprintf(stderr, "T2 quantized row %d: got %.7g, want %.7g\n", row, got[row], want[row]);
        return 1;
    }
    auto pair_y = backend.allocate(rows * sizeof(float));
    backend.matvec_quantized_pair(device_weight, *device_y, device_weight, *pair_y, quantized);
    float pair_got[rows]; backend.read(*pair_y, 0, pair_got, sizeof(pair_got));
    for (int row = 0; row < rows; row++) if (!close(pair_got[row], got[row])) return 1;

    // Ternary embedding row-lookup: exact dequant of one row.
    auto embed_out = backend.allocate(cols * sizeof(float));
    backend.embedding_q8(device_weight, 1, *embed_out);
    std::vector<float> embed(cols);
    backend.read(*embed_out, 0, embed.data(), cols * sizeof(float));
    for (int i = 0; i < cols; i++) {
        const float expect = (float)q_ref[1][i] * 0.5f;
        if (embed[i] != expect) {
            fprintf(stderr, "T2 embedding col %d: got %.7g, want %.7g\n", i, embed[i], expect);
            return 1;
        }
    }
    return 0;
}

int test_b1(q27::MetalBackend& backend) {
    constexpr int rows = 2, cols = 128;
    std::vector<uint8_t> packed(rows * cols / 8, 0);
    std::vector<float> x(cols);
    float want[rows] = {};
    int8_t q_ref[rows][cols];
    for (int i = 0; i < cols; i++) x[i] = (float)((i % 11) - 5) / 5.0f;
    for (int row = 0; row < rows; row++) {
        const float scale = row == 0 ? 1.0f : 0.5f;
        for (int i = 0; i < cols; i++) {
            const int q = ((i * 7 + row * 5) % 5) < 2 ? -1 : 1;   // {-1, +1}
            q_ref[row][i] = (int8_t)q;
            if (q > 0) packed[(size_t)row * cols / 8 + i / 8] |= (uint8_t)(1u << (i % 8));
            want[row] += q * scale * x[i];
        }
    }
    // IEEE-754 binary16 encodings for 1.0 and 0.5.
    std::vector<uint16_t> scales = {0x3c00, 0x3800};

    q27::Tensor tensor;
    tensor.name = "b1-test";
    tensor.dtype = q27::DType::B1_G128;
    tensor.shape = {rows, cols};
    tensor.data = packed.data();
    tensor.data_size = packed.size();
    tensor.scales = reinterpret_cast<const uint8_t*>(scales.data());
    tensor.scales_size = scales.size() * sizeof(uint16_t);

    q27::BackendTensor device_weight = backend.upload(tensor);
    auto device_x = backend.allocate(x.size() * sizeof(float));
    auto device_y = backend.allocate(rows * sizeof(float));
    backend.write(*device_x, 0, x.data(), x.size() * sizeof(float));
    backend.matvec(device_weight, *device_x, *device_y);
    float got[rows];
    backend.read(*device_y, 0, got, sizeof(got));
    for (int row = 0; row < rows; row++) {
        if (!close(got[row], want[row])) {
            fprintf(stderr, "B1 row %d: got %.7g, want %.7g\n", row, got[row], want[row]);
            return 1;
        }
    }
    auto quantized = backend.allocate_quantized(cols);
    backend.begin_commands(); backend.quantize(*device_x, quantized);
    backend.matvec_quantized(device_weight, quantized, *device_y); backend.end_commands();
    backend.read(*device_y, 0, got, sizeof(got));
    for (int row = 0; row < rows; row++) if (!close(got[row], want[row], 2e-2f)) {
        fprintf(stderr, "B1 quantized row %d: got %.7g, want %.7g\n", row, got[row], want[row]);
        return 1;
    }
    auto pair_y = backend.allocate(rows * sizeof(float));
    backend.matvec_quantized_pair(device_weight, *device_y, device_weight, *pair_y, quantized);
    float pair_got[rows]; backend.read(*pair_y, 0, pair_got, sizeof(pair_got));
    for (int row = 0; row < rows; row++) if (!close(pair_got[row], got[row])) return 1;

    // Binary embedding row-lookup: exact dequant of one row.
    auto embed_out = backend.allocate(cols * sizeof(float));
    backend.embedding_q8(device_weight, 1, *embed_out);
    std::vector<float> embed(cols);
    backend.read(*embed_out, 0, embed.data(), cols * sizeof(float));
    for (int i = 0; i < cols; i++) {
        const float expect = (float)q_ref[1][i] * 0.5f;
        if (embed[i] != expect) {
            fprintf(stderr, "B1 embedding col %d: got %.7g, want %.7g\n", i, embed[i], expect);
            return 1;
        }
    }
    // The _rows and _dev variants must reproduce the single-token lookup
    // bit for bit (same dequant expression, different token plumbing).
    const uint32_t row_tokens[2] = {1, 0};
    auto rows_out = backend.allocate(2 * cols * sizeof(float));
    backend.embedding_q8_rows(device_weight, row_tokens, 2, *rows_out);
    std::vector<float> rows_got(2 * cols);
    backend.read(*rows_out, 0, rows_got.data(), rows_got.size() * sizeof(float));
    auto token_dev = backend.allocate(sizeof(uint32_t));
    const uint32_t token_one = 1;
    backend.write(*token_dev, 0, &token_one, sizeof(token_one));
    auto dev_out = backend.allocate(cols * sizeof(float));
    backend.embedding_from_device(device_weight, *token_dev, *dev_out);
    std::vector<float> dev_got(cols);
    backend.read(*dev_out, 0, dev_got.data(), cols * sizeof(float));
    for (int i = 0; i < cols; i++) {
        if (rows_got[i] != embed[i] || dev_got[i] != embed[i]) {
            fprintf(stderr, "B1 embedding rows/dev col %d: rows %.7g dev %.7g want %.7g\n",
                    i, rows_got[i], dev_got[i], embed[i]);
            return 1;
        }
        const float expect0 = (float)q_ref[0][i] * 1.0f;
        if (rows_got[cols + i] != expect0) {
            fprintf(stderr, "B1 embedding rows token 0 col %d: got %.7g, want %.7g\n",
                    i, rows_got[cols + i], expect0);
            return 1;
        }
    }
    return 0;
}

// Wide-shape gate for the select-form float-activation T2 GEMV (the
// production ternary decode path): rows that span threadgroups and leave a
// 32-row remainder (clamped compute-only rows), block counts that exercise
// the four-blocks-in-flight loop with a tail, and scales varied per group.
int test_t2_wide(q27::MetalBackend& backend) {
    for (uint32_t cols : {1152u, 5120u}) {
        constexpr uint32_t rows = 37;   // 32 + 5: second threadgroup clamps rows 37..63
        const uint32_t groups = cols / 128;
        std::vector<uint8_t> data((size_t)rows * cols / 4, 0);
        std::vector<int8_t> w((size_t)rows * cols);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t c = 0; c < cols; c++) {
                const int q = (int)((r * 13 + c * 7) % 3) - 1;
                w[(size_t)r * cols + c] = (int8_t)q;
                data[(size_t)r * cols / 4 + c / 4] |= (uint8_t)((q + 1) << ((c % 4) * 2));
            }
        const uint16_t scale_bits[4] = {0x3400, 0x3800, 0x3c00, 0x4000};
        const float scale_f32[4] = {0.25f, 0.5f, 1.0f, 2.0f};
        std::vector<uint16_t> scales((size_t)rows * groups);
        for (size_t i = 0; i < scales.size(); i++) scales[i] = scale_bits[i % 4];
        q27::Tensor tensor;
        tensor.name = "t2-wide";
        tensor.dtype = q27::DType::T2_G128;
        tensor.shape = {rows, cols};
        tensor.data = data.data();
        tensor.data_size = data.size();
        tensor.scales = (const uint8_t*)scales.data();
        tensor.scales_size = scales.size() * 2;
        auto weight = backend.upload(tensor);

        std::vector<float> x(cols);
        for (uint32_t c = 0; c < cols; c++)
            x[c] = (float)((int)((c * 11) % 23) - 11) / 11.0f * (float)(1 + c / 32 % 7);
        auto xb = backend.allocate(cols * 4);
        backend.write(*xb, 0, x.data(), cols * 4);
        auto yb = backend.allocate(rows * 4);
        backend.matvec(weight, *xb, *yb);
        std::vector<float> got(rows);
        backend.read(*yb, 0, got.data(), rows * 4);
        for (uint32_t r = 0; r < rows; r++) {
            double want = 0.0, magnitude = 0.0;
            for (uint32_t g = 0; g < groups; g++) {
                double dot = 0.0;
                for (uint32_t c = g * 128; c < g * 128 + 128; c++)
                    dot += (double)w[(size_t)r * cols + c] * x[c];
                const double term = dot * scale_f32[((size_t)r * groups + g) % 4];
                want += term;
                magnitude += std::fabs(term);
            }
            const float bound = (float)(magnitude * 1e-5 + 1e-3);
            if (std::fabs(got[r] - (float)want) > bound) {
                fprintf(stderr, "T2 wide cols=%u row %u: got %.7g want %.7g\n",
                        cols, r, got[r], (float)want);
                return 1;
            }
        }
    }
    return 0;
}

// Wide-shape gate for the select-form float-activation B1 GEMV (the
// production binary decode path): same shapes as the T2 gate — rows that
// span threadgroups and leave a 32-row remainder (clamped compute-only
// rows), block counts that exercise the four-blocks-in-flight loop with a
// tail, and scales varied per group (a uniform-scale variant is vacuous:
// it cannot see a wrong scale index — the bug class the T2 gate caught).
int test_b1_wide(q27::MetalBackend& backend) {
    for (uint32_t cols : {1152u, 5120u}) {
        constexpr uint32_t rows = 37;   // 32 + 5: second threadgroup clamps rows 37..63
        const uint32_t groups = cols / 128;
        std::vector<uint8_t> data((size_t)rows * cols / 8, 0);
        std::vector<int8_t> w((size_t)rows * cols);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t c = 0; c < cols; c++) {
                const int q = ((r * 13 + c * 7) % 5) < 2 ? -1 : 1;
                w[(size_t)r * cols + c] = (int8_t)q;
                if (q > 0) data[(size_t)r * cols / 8 + c / 8] |= (uint8_t)(1u << (c % 8));
            }
        const uint16_t scale_bits[4] = {0x3400, 0x3800, 0x3c00, 0x4000};
        const float scale_f32[4] = {0.25f, 0.5f, 1.0f, 2.0f};
        std::vector<uint16_t> scales((size_t)rows * groups);
        for (size_t i = 0; i < scales.size(); i++) scales[i] = scale_bits[i % 4];
        q27::Tensor tensor;
        tensor.name = "b1-wide";
        tensor.dtype = q27::DType::B1_G128;
        tensor.shape = {rows, cols};
        tensor.data = data.data();
        tensor.data_size = data.size();
        tensor.scales = (const uint8_t*)scales.data();
        tensor.scales_size = scales.size() * 2;
        auto weight = backend.upload(tensor);

        std::vector<float> x(cols);
        for (uint32_t c = 0; c < cols; c++)
            x[c] = (float)((int)((c * 11) % 23) - 11) / 11.0f * (float)(1 + c / 32 % 7);
        auto xb = backend.allocate(cols * 4);
        backend.write(*xb, 0, x.data(), cols * 4);
        auto yb = backend.allocate(rows * 4);
        backend.matvec(weight, *xb, *yb);
        std::vector<float> got(rows);
        backend.read(*yb, 0, got.data(), rows * 4);
        for (uint32_t r = 0; r < rows; r++) {
            double want = 0.0, magnitude = 0.0;
            for (uint32_t g = 0; g < groups; g++) {
                double dot = 0.0;
                for (uint32_t c = g * 128; c < g * 128 + 128; c++)
                    dot += (double)w[(size_t)r * cols + c] * x[c];
                const double term = dot * scale_f32[((size_t)r * groups + g) % 4];
                want += term;
                magnitude += std::fabs(term);
            }
            const float bound = (float)(magnitude * 1e-5 + 1e-3);
            if (std::fabs(got[r] - (float)want) > bound) {
                fprintf(stderr, "B1 wide cols=%u row %u: got %.7g want %.7g\n",
                        cols, r, got[r], (float)want);
                return 1;
            }
        }
    }
    return 0;
}

// Wide-shape gate for the base-3 T3 GEMV: the same trits as a T2 packing,
// re-coded 5-per-byte (26 bytes per 128-column group, slots 3-4 of the last
// byte canonical code-1 padding). Checks the kernel against the CPU
// reference AND against the T2 kernel on identical trits (tolerance — the
// byte-per-lane schedule sums in a different order than T2's 16-col split).
int test_t3_wide(q27::MetalBackend& backend) {
    for (uint32_t cols : {1152u, 5120u}) {
        constexpr uint32_t rows = 37;   // 32 + 5: second threadgroup clamps rows 37..63
        const uint32_t groups = cols / 128;
        std::vector<uint8_t> t2_data((size_t)rows * cols / 4, 0);
        std::vector<uint8_t> t3_data((size_t)rows * groups * 26, 0);
        std::vector<int8_t> w((size_t)rows * cols);
        static const uint16_t pow3[5] = {1, 3, 9, 27, 81};
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t c = 0; c < cols; c++) {
                const int q = (int)((r * 13 + c * 7) % 3) - 1;
                w[(size_t)r * cols + c] = (int8_t)q;
                t2_data[(size_t)r * cols / 4 + c / 4] |= (uint8_t)((q + 1) << ((c % 4) * 2));
                const uint32_t g = c / 128, pg = c % 128;
                t3_data[(size_t)r * groups * 26 + g * 26 + pg / 5] +=
                    (uint8_t)((q + 1) * pow3[pg % 5]);
            }
            for (uint32_t g = 0; g < groups; g++)   // canonical padding: slots 3,4 of byte 25
                t3_data[(size_t)r * groups * 26 + g * 26 + 25] += 27 + 81;
        }
        const uint16_t scale_bits[4] = {0x3400, 0x3800, 0x3c00, 0x4000};
        const float scale_f32[4] = {0.25f, 0.5f, 1.0f, 2.0f};
        std::vector<uint16_t> scales((size_t)rows * groups);
        for (size_t i = 0; i < scales.size(); i++) scales[i] = scale_bits[i % 4];

        auto make = [&](q27::DType dtype, std::vector<uint8_t>& data, const char* name) {
            q27::Tensor tensor;
            tensor.name = name;
            tensor.dtype = dtype;
            tensor.shape = {rows, cols};
            tensor.data = data.data();
            tensor.data_size = data.size();
            tensor.scales = (const uint8_t*)scales.data();
            tensor.scales_size = scales.size() * 2;
            return backend.upload(tensor);
        };
        auto w3 = make(q27::DType::T3_G128, t3_data, "t3-wide");
        auto w2 = make(q27::DType::T2_G128, t2_data, "t2-ref");

        std::vector<float> x(cols);
        for (uint32_t c = 0; c < cols; c++)
            x[c] = (float)((int)((c * 11) % 23) - 11) / 11.0f * (float)(1 + c / 32 % 7);
        auto xb = backend.allocate(cols * 4);
        backend.write(*xb, 0, x.data(), cols * 4);
        auto y3 = backend.allocate(rows * 4);
        auto y2 = backend.allocate(rows * 4);
        backend.matvec(w3, *xb, *y3);
        backend.matvec(w2, *xb, *y2);
        std::vector<float> got(rows), ref(rows);
        backend.read(*y3, 0, got.data(), rows * 4);
        backend.read(*y2, 0, ref.data(), rows * 4);
        for (uint32_t r = 0; r < rows; r++) {
            double want = 0.0, magnitude = 0.0;
            for (uint32_t g = 0; g < groups; g++) {
                double dot = 0.0;
                for (uint32_t c = g * 128; c < g * 128 + 128; c++)
                    dot += (double)w[(size_t)r * cols + c] * x[c];
                const double term = dot * scale_f32[((size_t)r * groups + g) % 4];
                want += term;
                magnitude += std::fabs(term);
            }
            const float bound = (float)(magnitude * 1e-5 + 1e-3);
            if (std::fabs(got[r] - (float)want) > bound) {
                fprintf(stderr, "T3 wide cols=%u row %u: got %.7g want %.7g\n",
                        cols, r, got[r], (float)want);
                return 1;
            }
            if (std::fabs(got[r] - ref[r]) > bound) {
                fprintf(stderr, "T3 vs T2 cols=%u row %u: %.7g vs %.7g\n",
                        cols, r, got[r], ref[r]);
                return 1;
            }
        }
    }
    return 0;
}

// The tiled simdgroup-matrix GEMM keeps the weight side integer-exact and
// rounds the activation side once per value at staging, but accumulates
// K-tiles in a different float-addition order than the serial GEMV, so the
// comparison is tolerance-based. Shapes cover several K-tile counts, row
// remainders against the 32-row tile (9, 17), and every token-group width,
// per the width-gated fast-path lesson: a gate shape must actually enter
// the fast path it gates.
int test_matmul_shape(q27::MetalBackend& backend,q27::DType dtype,
                      uint32_t rows,uint32_t cols,uint32_t tokens=12) {
    std::vector<uint8_t> data(dtype==q27::DType::Q4_G64?(size_t)rows*cols/2:
                              dtype==q27::DType::T2_G128?(size_t)rows*cols/4:
                              dtype==q27::DType::B1_G128?(size_t)rows*cols/8:(size_t)rows*cols,0);
    for(uint32_t r=0;r<rows;r++) for(uint32_t c=0;c<cols;c++) {
        int q=(int)((r*7+c*3)%15)-7;
        if(dtype==q27::DType::Q4_G64) {
            size_t i=(size_t)r*cols/2+c/2; uint8_t nibble=(uint8_t)(q+8);
            if(c&1) data[i]=(data[i]&15)|(nibble<<4); else data[i]=(data[i]&240)|nibble;
        } else if(dtype==q27::DType::T2_G128) {
            q=(int)((r*7+c*3)%3)-1;
            data[(size_t)r*cols/4+c/4]|=(uint8_t)((q+1)<<((c%4)*2));
        } else if(dtype==q27::DType::B1_G128) {
            q=((r*7+c*3)%5)<2?-1:1;
            if(q>0) data[(size_t)r*cols/8+c/8]|=(uint8_t)(1u<<(c%8));
        } else data[(size_t)r*cols+c]=(uint8_t)(int8_t)q;
    }
    uint32_t groups=cols/(dtype==q27::DType::Q4_G64?64:128);
    const uint16_t scale_values[4]={0x3400,0x3800,0x3c00,0x4000};
    std::vector<uint16_t> scales(rows*groups);
    for(uint32_t r=0;r<rows;r++) for(uint32_t g=0;g<groups;g++) scales[r*groups+g]=scale_values[(r+g)%4];
    q27::Tensor tensor; tensor.name="matmul-tiles"; tensor.dtype=dtype; tensor.shape={rows,cols};
    tensor.data=data.data(); tensor.data_size=data.size(); tensor.scales=(const uint8_t*)scales.data(); tensor.scales_size=scales.size()*2;
    auto weight=backend.upload(tensor); std::vector<float> x(tokens*cols);
    for(uint32_t t=0;t<tokens;t++) for(uint32_t c=0;c<cols;c++)
        x[(size_t)t*cols+c]=(((int)((t+1)*11+c*5)%31)-15)*(float)(c/32+1)/(float)(t+3);
    std::vector<float> reference(tokens*rows);
    for(uint32_t t=0;t<tokens;t++) {
        auto xb=backend.allocate(cols*4),yb=backend.allocate(rows*4); backend.write(*xb,0,x.data()+(size_t)t*cols,cols*4);
        auto q=backend.allocate_quantized(cols); backend.begin_commands(); backend.quantize(*xb,q); backend.matvec_quantized(weight,q,*yb); backend.end_commands();
        backend.read(*yb,0,reference.data()+(size_t)t*rows,rows*4);
    }
    auto all_x=backend.allocate(x.size()*4); backend.write(*all_x,0,x.data(),x.size()*4);
    // Dispatch the parametrized width too — a repro case that only sizes
    // buffers without dispatching passes vacuously (codex P2, 2026-07-15).
    // 17/33 exercise partial second/third 16-token tiles on grid.y.
    for(uint32_t n:{1u,4u,5u,8u,9u,12u,17u,33u,tokens}) {
        if(n>tokens) continue;
        auto q=backend.allocate_quantized(n*cols); auto out=backend.allocate((uint64_t)n*rows*4);
        backend.begin_commands(); backend.quantize(*all_x,q); backend.matmul_quantized(weight,q,n,*out); backend.end_commands();
        std::vector<float> got(n*rows); backend.read(*out,0,got.data(),got.size()*4);
        for(size_t i=0;i<got.size();i++) if(!close(got[i],reference[i],3e-4f)) {
            fprintf(stderr,"%s matmul rows=%u cols=%u n=%u i=%zu got %.9g want %.9g\n",
                    q27::dtype_name(dtype),rows,cols,n,i,got[i],reference[i]); return 1;
        }
    }
    return 0;
}

int test_matmul_tiles(q27::MetalBackend& backend,q27::DType dtype) {
    return test_matmul_shape(backend,dtype,9,256) ||
           test_matmul_shape(backend,dtype,9,1024) ||
           test_matmul_shape(backend,dtype,17,1152) ||
           // Wide-chunk token tiling: full tiles (96 = 6x16) and a partial
           // final tile (20 = 16 + 4) across the same row/col remainders.
           test_matmul_shape(backend,dtype,17,1152,96) ||
           test_matmul_shape(backend,dtype,9,256,20) ||
           // artifact-K repro: width-17 divergence hunt (cols=5120)
           test_matmul_shape(backend,dtype,33,5120,17) ||
           test_matmul_shape(backend,dtype,10240,5120,17) ||
           // Row remainder past a full 32-row tile (100 = 3x32 + 4), token
           // remainder past two 16-token tiles (33), and exact fits (64).
           // cols stays moderate: the Q4/Q8 kernels stage pre-scaled fp32
           // activations, so high-cancellation cols=5120 outputs amplify
           // ordinary rounding past the gate (the exact-int T2 paths get
           // their 5120 coverage from the width-17 artifact repros above).
           test_matmul_shape(backend,dtype,100,1152,33) ||
           test_matmul_shape(backend,dtype,64,128,64);
}

// Production-width GEMV parity. The packed-dot kernels take a vectorized
// main loop only when cols >= 1024, so the narrow test_q8/test_q4 shapes
// never reach it; this caught a wrong per-lane activation-scale index that
// the 64/128-column tests could not see. Scales are deliberately varied per
// weight group and per 32-column activation block.
// Wide-shape gate for the threadgroup-per-row F16 pair kernel: the packed
// vector main loop and the 8-simdgroup reduction tree only fill up past a few
// hundred columns, so this runs the production alpha/beta shape (48 x 5120),
// an odd width that leaves a scalar tail, and asymmetric row counts.
int test_f16_pair_wide(q27::MetalBackend& backend) {
    for (uint32_t cols : {1030u, 5120u}) {
        constexpr uint32_t rows_a = 48, rows_b = 16;
        std::vector<_Float16> wa((size_t)rows_a * cols), wb((size_t)rows_b * cols);
        for (size_t i = 0; i < wa.size(); i++) wa[i] = (_Float16)(((int)((i * 7) % 31) - 15) * 0.0625f);
        for (size_t i = 0; i < wb.size(); i++) wb[i] = (_Float16)(((int)((i * 11) % 27) - 13) * 0.125f);
        std::vector<float> x(cols);
        for (uint32_t c = 0; c < cols; c++) x[c] = (float)((int)((c * 13) % 21) - 10) / 10.0f;

        q27::Tensor ta; ta.name = "pair-wide-a"; ta.dtype = q27::DType::F16; ta.shape = {rows_a, cols};
        ta.data = reinterpret_cast<const uint8_t*>(wa.data()); ta.data_size = wa.size() * 2;
        q27::Tensor tb; tb.name = "pair-wide-b"; tb.dtype = q27::DType::F16; tb.shape = {rows_b, cols};
        tb.data = reinterpret_cast<const uint8_t*>(wb.data()); tb.data_size = wb.size() * 2;
        auto weight_a = backend.upload(ta);
        auto weight_b = backend.upload(tb);
        auto xb = backend.allocate(cols * 4);
        backend.write(*xb, 0, x.data(), cols * 4);
        auto ya = backend.allocate(rows_a * 4), yb = backend.allocate(rows_b * 4);
        backend.matvec_pair(weight_a, *ya, weight_b, *yb, *xb);
        std::vector<float> got_a(rows_a), got_b(rows_b);
        backend.read(*ya, 0, got_a.data(), rows_a * 4);
        backend.read(*yb, 0, got_b.data(), rows_b * 4);
        for (uint32_t r = 0; r < rows_a; r++) {
            double want = 0;
            for (uint32_t c = 0; c < cols; c++) want += (double)(float)wa[(size_t)r * cols + c] * x[c];
            if (!close(got_a[r], (float)want, 2e-3f)) {
                fprintf(stderr, "F16 pair wide A cols=%u row %u: got %.7g want %.7g\n", cols, r, got_a[r], (float)want);
                return 1;
            }
        }
        for (uint32_t r = 0; r < rows_b; r++) {
            double want = 0;
            for (uint32_t c = 0; c < cols; c++) want += (double)(float)wb[(size_t)r * cols + c] * x[c];
            if (!close(got_b[r], (float)want, 2e-3f)) {
                fprintf(stderr, "F16 pair wide B cols=%u row %u: got %.7g want %.7g\n", cols, r, got_b[r], (float)want);
                return 1;
            }
        }
    }
    return 0;
}

int test_quantized_wide(q27::MetalBackend& backend, q27::DType dtype) {
    const uint32_t group = dtype == q27::DType::Q4_G64 ? 64 : 128;
    // main chunk + tail, then a production column count
    const uint32_t widths[2] = {1024 + group, 5120};
    for (uint32_t cols : widths) {
        constexpr uint32_t rows = 5;
        const uint32_t groups = cols / group;
        std::vector<int8_t> w(rows * cols);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t c = 0; c < cols; c++)
                w[(size_t)r * cols + c] = (int8_t)((int)((r * 13 + c * 7) % 29) - 14);
        std::vector<uint8_t> data;
        if (dtype == q27::DType::T2_G128) {
            data.assign((size_t)rows * cols / 4, 0);
            for (uint32_t r = 0; r < rows; r++)
                for (uint32_t c = 0; c < cols; c++) {
                    const int q = (int)((r * 13 + c * 7) % 3) - 1;
                    w[(size_t)r * cols + c] = (int8_t)q;
                    data[(size_t)r * cols / 4 + c / 4] |= (uint8_t)((q + 1) << ((c % 4) * 2));
                }
        } else if (dtype == q27::DType::B1_G128) {
            data.assign((size_t)rows * cols / 8, 0);
            for (uint32_t r = 0; r < rows; r++)
                for (uint32_t c = 0; c < cols; c++) {
                    const int q = ((r * 13 + c * 7) % 5) < 2 ? -1 : 1;
                    w[(size_t)r * cols + c] = (int8_t)q;
                    if (q > 0) data[(size_t)r * cols / 8 + c / 8] |= (uint8_t)(1u << (c % 8));
                }
        } else if (dtype == q27::DType::Q4_G64) {
            data.resize((size_t)rows * cols / 2);
            for (uint32_t r = 0; r < rows; r++)
                for (uint32_t c = 0; c < cols; c++) {
                    int q = ((int)w[(size_t)r * cols + c] % 8 + 8) % 16 - 8;
                    w[(size_t)r * cols + c] = (int8_t)q;
                    uint8_t nibble = (uint8_t)(q + 8);
                    size_t i = (size_t)r * cols / 2 + c / 2;
                    if (c & 1) data[i] = (uint8_t)((data[i] & 15) | (nibble << 4));
                    else data[i] = (uint8_t)((data[i] & 240) | nibble);
                }
        } else {
            data.assign((const uint8_t*)w.data(), (const uint8_t*)w.data() + w.size());
        }
        const uint16_t scale_bits[4] = {0x3400, 0x3800, 0x3c00, 0x4000};
        const float scale_f32[4] = {0.25f, 0.5f, 1.0f, 2.0f};
        std::vector<uint16_t> scales(rows * groups);
        for (size_t i = 0; i < scales.size(); i++) scales[i] = scale_bits[i % 4];
        q27::Tensor tensor;
        tensor.name = "wide";
        tensor.dtype = dtype;
        tensor.shape = {rows, cols};
        tensor.data = data.data();
        tensor.data_size = data.size();
        tensor.scales = (const uint8_t*)scales.data();
        tensor.scales_size = scales.size() * 2;
        auto weight = backend.upload(tensor);

        // Distinct magnitude per 32-column block so every activation scale differs.
        std::vector<float> x(cols);
        for (uint32_t c = 0; c < cols; c++)
            x[c] = (float)((int)((c * 11) % 23) - 11) / 11.0f * (float)(1 + c / 32 % 7);
        auto xb = backend.allocate(cols * 4);
        backend.write(*xb, 0, x.data(), cols * 4);
        auto xq = backend.allocate_quantized(cols);
        auto yb = backend.allocate(rows * 4);
        auto y2 = backend.allocate(rows * 4);
        auto y3 = backend.allocate(rows * 4);
        backend.begin_commands();
        backend.quantize(*xb, xq);
        backend.matvec_quantized(weight, xq, *yb);
        backend.end_commands();
        backend.matvec_quantized_pair(weight, *y2, weight, *y3, xq);

        std::vector<int8_t> xv(cols);
        std::vector<float> xs(cols / 32);
        backend.read(*xq.values, 0, xv.data(), cols);
        backend.read(*xq.scales, 0, xs.data(), xs.size() * 4);
        std::vector<float> got(rows), got2(rows), got3(rows);
        backend.read(*yb, 0, got.data(), rows * 4);
        backend.read(*y2, 0, got2.data(), rows * 4);
        backend.read(*y3, 0, got3.data(), rows * 4);
        for (uint32_t r = 0; r < rows; r++) {
            double want = 0.0, magnitude = 0.0;
            for (uint32_t b = 0; b < cols / 32; b++) {
                int dot = 0;
                for (uint32_t i = b * 32; i < b * 32 + 32; i++)
                    dot += (int)w[(size_t)r * cols + i] * (int)xv[i];
                const double term =
                    (double)dot * scale_f32[(r * groups + b * 32 / group) % 4] * xs[b];
                want += term;
                magnitude += std::fabs(term);
            }
            // Absolute bound scaled by the sum of |block terms|, so a
            // cancellation-heavy row cannot hide a wrong-scale bug behind a
            // relative check, and the double-vs-float summation-order
            // difference stays within it.
            const float bound = (float)(magnitude * 1e-5 + 1e-3);
            const auto ok = [&](float value) { return std::fabs(value - (float)want) <= bound; };
            if (!ok(got[r]) || !ok(got2[r]) || !ok(got3[r])) {
                fprintf(stderr, "%s wide cols=%u row %u: got %.7g pair %.7g/%.7g want %.7g\n",
                        q27::dtype_name(dtype), cols, r, got[r], got2[r], got3[r], (float)want);
                return 1;
            }
        }
    }
    return 0;
}

int test_mixed_pair(q27::MetalBackend& backend) {
    constexpr int cols=128; std::vector<float> x(cols); for(int i=0;i<cols;i++) x[i]=(i%13-6)/7.0f;
    std::vector<uint8_t> q4(cols/2,0x99); std::vector<int8_t> q8(cols,2);
    std::vector<uint16_t> s4={0x3c00,0x3c00},s8={0x3c00};
    q27::Tensor a; a.name="mixed-q4"; a.dtype=q27::DType::Q4_G64; a.shape={1,cols};
    a.data=q4.data(); a.data_size=q4.size(); a.scales=(const uint8_t*)s4.data(); a.scales_size=s4.size()*2;
    q27::Tensor b; b.name="mixed-q8"; b.dtype=q27::DType::Q8_G128; b.shape={1,cols};
    b.data=(const uint8_t*)q8.data(); b.data_size=q8.size(); b.scales=(const uint8_t*)s8.data(); b.scales_size=s8.size()*2;
    auto aw=backend.upload(a); auto bw=backend.upload(b); auto xb=backend.allocate(cols*4);
    auto ao=backend.allocate(4); auto bo=backend.allocate(4); auto ar=backend.allocate(4); auto br=backend.allocate(4);
    auto quant=backend.allocate_quantized(cols); backend.write(*xb,0,x.data(),cols*4);
    backend.begin_commands(); backend.quantize(*xb,quant); backend.matvec_quantized(aw,quant,*ar); backend.matvec_quantized(bw,quant,*br); backend.end_commands();
    backend.matvec_quantized_pair(aw,*ao,bw,*bo,quant);
    float expected_a,expected_b,got_a,got_b; backend.read(*ar,0,&expected_a,4); backend.read(*br,0,&expected_b,4);
    backend.read(*ao,0,&got_a,4); backend.read(*bo,0,&got_b,4);
    if(got_a!=expected_a || got_b!=expected_b) { fprintf(stderr,"mixed quantized pair fallback mismatch\n"); return 1; }
    return 0;
}

int test_postcommit_failure() {
    setenv("Q27_METAL_FAIL_FINISH", "1", 1);
    bool injected = false, refused_reuse = false;
    {
        q27::MetalBackend backend;
        const uint32_t value = 0x12345678;
        auto source = backend.allocate(sizeof(value));
        auto destination = backend.allocate(sizeof(value));
        backend.write(*source, 0, &value, sizeof(value));
        try {
            backend.begin_commands();
            backend.copy(*source, 0, *destination, 0, sizeof(value));
            backend.end_commands();
        } catch (const std::runtime_error& error) {
            injected = std::string(error.what()).find("failed (injected)") != std::string::npos;
        }
        try {
            uint32_t ignored = 0;
            backend.read(*destination, 0, &ignored, sizeof(ignored));
        } catch (const std::runtime_error& error) {
            refused_reuse = std::string(error.what()).find("recreate the engine") != std::string::npos;
        }
    }
    unsetenv("Q27_METAL_FAIL_FINISH");
    if (!injected || !refused_reuse) {
        std::fprintf(stderr, "post-commit failure did not poison the backend\n");
        return 1;
    }

    // Reconstructing the backend is the only recovery path.
    q27::MetalBackend replacement;
    const uint32_t value = 0x89abcdef, zero = 0;
    auto source = replacement.allocate(sizeof(value));
    auto destination = replacement.allocate(sizeof(value));
    replacement.write(*source, 0, &value, sizeof(value));
    replacement.write(*destination, 0, &zero, sizeof(zero));
    replacement.copy(*source, 0, *destination, 0, sizeof(value));
    uint32_t got = 0;
    replacement.read(*destination, 0, &got, sizeof(got));
    if (got != value) return 1;
    std::puts("Metal post-commit poison: PASS");
    return 0;
}

int test_profile_batch_overflow() {
    setenv("Q27_METAL_PROFILE", "1", 1);
    bool rejected = false;
    {
        q27::MetalBackend backend;
        const uint32_t value = 0x13579bdf, zero = 0;
        auto source = backend.allocate(sizeof(value));
        auto destination = backend.allocate(sizeof(value));
        backend.write(*source, 0, &value, sizeof(value));
        backend.write(*destination, 0, &zero, sizeof(zero));
        try {
            backend.begin_commands();
            for (uint32_t i = 0; i < 2049; i++)
                backend.copy(*source, 0, *destination, 0, sizeof(value));
            backend.end_commands();
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("exceeds 2048 operations") !=
                       std::string::npos;
            backend.abort_commands();
        }
        uint32_t got = value;
        backend.read(*destination, 0, &got, sizeof(got));
        if (!rejected || got != zero) {
            unsetenv("Q27_METAL_PROFILE");
            std::fprintf(stderr, "profile overflow committed a partial explicit batch\n");
            return 1;
        }
        backend.copy(*source, 0, *destination, 0, sizeof(value));
        backend.read(*destination, 0, &got, sizeof(got));
        if (got != value) {
            unsetenv("Q27_METAL_PROFILE");
            return 1;
        }
    }
    unsetenv("Q27_METAL_PROFILE");
    std::puts("Metal profiling batch overflow: PASS");
    return 0;
}
} // namespace

int main() {
    try {
        if (argc == 2 && std::string(argv[1]) == "--postcommit-failure")
            return test_postcommit_failure();
        if (argc == 2 && std::string(argv[1]) == "--profile-batch-overflow")
            return test_profile_batch_overflow();
        q27::MetalBackend backend;
        printf("Metal device: %s\n", backend.name().c_str());
        printf("working set %.1f GiB, max buffer %.1f GiB, threadgroup memory %.1f KiB\n",
               backend.recommended_working_set_size() / 1073741824.0,
               backend.max_buffer_length() / 1073741824.0,
               backend.max_threadgroup_memory_length() / 1024.0);
        if (test_mmap_upload(backend) || test_dispatch_validation(backend) ||
            test_f32(backend) || test_f16(backend) ||
            test_q8(backend) || test_q4(backend) || test_t2(backend) || test_b1(backend) ||
            test_quantized_wide(backend, q27::DType::Q4_G64) ||
            test_quantized_wide(backend, q27::DType::Q8_G128) ||
            test_quantized_wide(backend, q27::DType::T2_G128) ||
            test_quantized_wide(backend, q27::DType::B1_G128) ||
            test_t2_wide(backend) ||
            test_b1_wide(backend) ||
            test_t3_wide(backend) ||
            test_f16_pair_wide(backend) ||
            test_mixed_pair(backend) ||
            (backend.supports_quantized_matmul() &&
             (test_matmul_tiles(backend,q27::DType::Q4_G64) || test_matmul_tiles(backend,q27::DType::Q8_G128) ||
              test_matmul_tiles(backend,q27::DType::T2_G128) ||
              test_matmul_tiles(backend,q27::DType::B1_G128))))
            return 1;
        puts("Metal matvec: OK");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
