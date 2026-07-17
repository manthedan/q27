#include "loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace q27 {

static constexpr uint32_t MAGIC = 0x46373251; // "Q27F" LE
static constexpr uint32_t VERSION = 1;
static constexpr uint64_t ALIGN = 256;
static constexpr uint32_t MAX_TENSORS = 1u << 20;
static constexpr uint32_t MAX_META_BYTES = 64u << 20;

const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32:     return "F32";
        case DType::F16:     return "F16";
        case DType::Q8_G128: return "Q8_G128";
        case DType::Q4_G64:  return "Q4_G64";
        case DType::T2_G128: return "T2_G128";
        case DType::T3_G128: return "T3_G128";
        case DType::B1_G128: return "B1_G128";
    }
    return "?";
}

uint64_t Tensor::rows() const {
    if (shape.size() <= 1) return 1;
    uint64_t r = 1;
    for (size_t i = 0; i + 1 < shape.size(); i++) r *= shape[i];
    return r;
}
uint64_t Tensor::cols() const { return shape.empty() ? 0 : shape.back(); }
uint64_t Tensor::n_elements() const { return rows() * cols(); }

const Tensor* Model::find(const std::string& name) const {
    auto it = index.find(name);
    return it == index.end() ? nullptr : &tensors[it->second];
}
const Tensor& Model::get(const std::string& name) const {
    const Tensor* t = find(name);
    if (!t) throw std::runtime_error("q27: missing tensor: " + name);
    return *t;
}

Model::Model(Model&& o) noexcept { *this = std::move(o); }
Model& Model::operator=(Model&& o) noexcept {
    if (this != &o) {
        // release our mmap; do NOT call ~Model() -- that ends the lifetimes of
        // meta_json/tensors/index, which the moves below then write into (UB).
        if (map_base_) munmap(map_base_, map_size_);
        meta_json = std::move(o.meta_json);
        tensors = std::move(o.tensors);
        index = std::move(o.index);
        map_base_ = o.map_base_; map_size_ = o.map_size_;
        o.map_base_ = nullptr; o.map_size_ = 0;
    }
    return *this;
}
Model::~Model() {
    if (map_base_) munmap(map_base_, map_size_);
    map_base_ = nullptr;
}

namespace {
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;

    size_t remaining() const { return (size_t)(end - p); }

    template <typename T> T read() {
        if (remaining() < sizeof(T)) throw std::runtime_error("q27: truncated file");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    void bytes(void* dst, size_t n) {
        if (remaining() < n) throw std::runtime_error("q27: truncated file");
        if (n) std::memcpy(dst, p, n);
        p += n;
    }
};

uint64_t checked_mul(uint64_t a, uint64_t b, const std::string& what) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a)
        throw std::runtime_error("q27: size overflow: " + what);
    return a * b;
}

uint64_t expected_sizes(const Tensor& t, uint64_t& scales) {
    if (t.shape.empty()) throw std::runtime_error("q27: scalar tensor: " + t.name);
    uint64_t elements = 1;
    for (uint64_t dim : t.shape) {
        if (!dim) throw std::runtime_error("q27: zero tensor dimension: " + t.name);
        elements = checked_mul(elements, dim, t.name);
    }

    const uint64_t rows = elements / t.shape.back();
    const uint64_t cols = t.shape.back();
    scales = 0;
    switch (t.dtype) {
        case DType::F32:
            return checked_mul(elements, 4, t.name);
        case DType::F16:
            return checked_mul(elements, 2, t.name);
        case DType::Q8_G128:
            if (cols % 128) throw std::runtime_error("q27: Q8 columns not divisible by 128: " + t.name);
            scales = checked_mul(checked_mul(rows, cols / 128, t.name), 2, t.name);
            return elements;
        case DType::Q4_G64:
            if (cols % 64) throw std::runtime_error("q27: Q4 columns not divisible by 64: " + t.name);
            scales = checked_mul(checked_mul(rows, cols / 64, t.name), 2, t.name);
            return elements / 2;
        case DType::T2_G128:
            if (cols % 128) throw std::runtime_error("q27: T2 columns not divisible by 128: " + t.name);
            scales = checked_mul(checked_mul(rows, cols / 128, t.name), 2, t.name);
            return elements / 4;
        case DType::T3_G128:
            // base-3: 26 bytes per 128-column group (5 codes/byte, 2 pad slots)
            if (cols % 128) throw std::runtime_error("q27: T3 columns not divisible by 128: " + t.name);
            scales = checked_mul(checked_mul(rows, cols / 128, t.name), 2, t.name);
            return checked_mul(checked_mul(rows, cols / 128, t.name), 26, t.name);
        case DType::B1_G128:
            if (cols % 128) throw std::runtime_error("q27: B1 columns not divisible by 128: " + t.name);
            scales = checked_mul(checked_mul(rows, cols / 128, t.name), 2, t.name);
            return elements / 8;
    }
    throw std::runtime_error("q27: invalid tensor dtype: " + t.name);
}

struct BlobRange {
    uint64_t begin;
    uint64_t end;
    std::string label;
};
} // namespace

Model Model::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("q27: cannot open " + path);
    struct stat st{};
    if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("q27: fstat failed"); }
    if (st.st_size <= 0 || (uintmax_t)st.st_size > std::numeric_limits<size_t>::max()) {
        close(fd);
        throw std::runtime_error("q27: invalid file size");
    }
    size_t sz = (size_t)st.st_size;
    void* base = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) throw std::runtime_error("q27: mmap failed");

    Model m;
    m.map_base_ = base;
    m.map_size_ = sz;

    const uint8_t* b = (const uint8_t*)base;
    Cursor c{b, b + sz};
    if (c.read<uint32_t>() != MAGIC)   throw std::runtime_error("q27: bad magic");
    if (c.read<uint32_t>() != VERSION) throw std::runtime_error("q27: unsupported version");
    uint32_t n_tensors = c.read<uint32_t>();
    uint32_t meta_len  = c.read<uint32_t>();
    if (!n_tensors || n_tensors > MAX_TENSORS)
        throw std::runtime_error("q27: invalid tensor count");
    if (meta_len > MAX_META_BYTES) throw std::runtime_error("q27: metadata too large");
    m.meta_json.resize(meta_len);
    c.bytes(m.meta_json.data(), meta_len);

    // Even an empty-name, zero-dimensional table entry needs 36 bytes. This
    // check bounds reserve() before any artifact-controlled allocation.
    if (n_tensors > c.remaining() / 36)
        throw std::runtime_error("q27: invalid tensor count");

    struct Offsets { uint64_t data, scales; };
    std::vector<Offsets> offsets;
    m.tensors.reserve(n_tensors);
    offsets.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        Tensor t;
        uint16_t nl = c.read<uint16_t>();
        if (!nl) throw std::runtime_error("q27: empty tensor name");
        t.name.resize(nl);
        c.bytes(t.name.data(), nl);
        if (t.name.find('\0') != std::string::npos)
            throw std::runtime_error("q27: NUL in tensor name");
        uint8_t dtype = c.read<uint8_t>();
        if (dtype > (uint8_t)DType::B1_G128)
            throw std::runtime_error("q27: invalid tensor dtype: " + t.name);
        t.dtype = (DType)dtype;
        uint8_t nd = c.read<uint8_t>();
        if (!nd || nd > 8) throw std::runtime_error("q27: invalid tensor rank: " + t.name);
        t.shape.resize(nd);
        for (uint8_t d = 0; d < nd; d++) t.shape[d] = c.read<uint64_t>();
        uint64_t doff = c.read<uint64_t>();
        t.data_size   = c.read<uint64_t>();
        uint64_t soff = c.read<uint64_t>();
        t.scales_size = c.read<uint64_t>();

        uint64_t want_scales = 0;
        uint64_t want_data = expected_sizes(t, want_scales);
        if (t.data_size != want_data || t.scales_size != want_scales)
            throw std::runtime_error("q27: tensor byte-size mismatch: " + t.name);
        if (doff % ALIGN || (t.scales_size && soff % ALIGN))
            throw std::runtime_error("q27: unaligned tensor blob: " + t.name);
        if (!t.scales_size && soff)
            throw std::runtime_error("q27: unexpected scale offset: " + t.name);

        offsets.push_back({doff, soff});
        m.tensors.push_back(std::move(t));
    }
    uint64_t table_end = (uint64_t)(c.p - b);
    uint64_t data_base = table_end;
    if (uint64_t rem = data_base % ALIGN) {
        const uint64_t padding = ALIGN - rem;
        if (data_base > std::numeric_limits<uint64_t>::max() - padding)
            throw std::runtime_error("q27: data-section offset overflow");
        data_base += padding;
    }
    if (data_base > sz) throw std::runtime_error("q27: truncated data section");
    const uint64_t payload_size = (uint64_t)sz - data_base;

    std::vector<BlobRange> ranges;
    ranges.reserve((size_t)n_tensors * 2);
    for (size_t i = 0; i < m.tensors.size(); i++) {
        Tensor& t = m.tensors[i];
        const uint64_t doff = offsets[i].data;
        if (doff > payload_size || t.data_size > payload_size - doff)
            throw std::runtime_error("q27: tensor data out of range: " + t.name);
        t.data = b + data_base + doff;
        ranges.push_back({doff, doff + t.data_size, t.name + " data"});

        if (t.scales_size) {
            const uint64_t soff = offsets[i].scales;
            if (soff > payload_size || t.scales_size > payload_size - soff)
                throw std::runtime_error("q27: tensor scales out of range: " + t.name);
            t.scales = b + data_base + soff;
            ranges.push_back({soff, soff + t.scales_size, t.name + " scales"});
        }
        if (!m.index.emplace(t.name, i).second)
            throw std::runtime_error("q27: duplicate tensor: " + t.name);
    }

    std::sort(ranges.begin(), ranges.end(), [](const BlobRange& a, const BlobRange& b) {
        return a.begin < b.begin;
    });
    for (size_t i = 1; i < ranges.size(); i++)
        if (ranges[i].begin < ranges[i - 1].end)
            throw std::runtime_error("q27: overlapping tensor blobs: " + ranges[i - 1].label +
                                     " and " + ranges[i].label);
    return m;
}

} // namespace q27
