#include <vector>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "cuda_common.h"
#include "device_model.h"
#include "kernels.cuh"

namespace q27 {

DeviceModel::~DeviceModel() {
    for (auto& [k, t] : dev_) {
        if (t.data) cudaFree(t.data);
        if (t.scales) cudaFree(t.scales);
    }
}

// CPU word-sum with the same wraparound-add semantics as k_xsum64, so a host
// total is directly comparable to the device total for the same bytes.
static unsigned long long host_xsum(const void* p, size_t bytes) {
    const unsigned long long* w = (const unsigned long long*)p;
    const size_t n64 = bytes / 8;
    unsigned long long s = 0;
    for (size_t i = 0; i < n64; i++) s += w[i];
    const unsigned char* tail = (const unsigned char*)p + n64 * 8;
    unsigned long long t = 0;
    for (size_t i = 0; i < bytes % 8; i++) t |= (unsigned long long)tail[i] << (8 * i);
    return s + t;
}
static unsigned long long xsum_dev(const void* p, uint64_t bytes, unsigned long long* d_out);

const DevTensor& DeviceModel::upload(const std::string& name) {
    auto it = dev_.find(name);
    if (it != dev_.end()) return it->second;

    const Tensor& src = model_.get(name);
    validate_cuda_tensor(src);

    DevTensor d;
    d.dtype = src.dtype;
    d.rows = src.rows();
    d.cols = src.cols();
    // T3_G128: the device copy is a different (and differently sized) layout
    // (kernels.cuh t3_relayout_device); the FORMAT.md bytes land in a
    // transient buffer first so the transfer itself is still summed.
    const bool t3 = d.dtype == DType::T3_G128;
    const uint64_t dev_bytes = t3 ? q27k::t3_device_bytes(d.rows, d.cols) : src.data_size;
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&d.data, dev_bytes));
    if (t3) CUDA_CHECK(cudaMalloc(&raw, src.data_size));
    else raw = d.data;
    // read the source bytes on the CPU immediately before handing them to the
    // copy engine, so the two totals describe the same load
    if (want_host_sum_) {
        const unsigned long long hs = host_xsum(src.data, src.data_size);
        host_sum_ += hs;
        host_sums_[name] = hs;
    }
    CUDA_CHECK(cudaMemcpy(raw, src.data, src.data_size, cudaMemcpyHostToDevice));
    if (d.dtype == DType::T2_G128) {
        // the device copy uses the dp4a-interleaved word order (kernels.cuh);
        // the host bytes stay FORMAT.md-sequential (checksums, CPU references)
        q27k::t2_interleave_device((uint8_t*)d.data, src.data_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else if (t3) {
        if (want_host_sum_) {
            // the transfer check T2/Q4 get from [locate]: the raw bytes on the
            // device must sum to the host total before the relayout consumes them
            unsigned long long* d_out;
            CUDA_CHECK(cudaMalloc(&d_out, 8));
            const unsigned long long ds = xsum_dev(raw, src.data_size, d_out);
            CUDA_CHECK(cudaFree(d_out));
            if (ds != host_sums_[name])
                fprintf(stderr, "[upload] T3 transfer MISMATCH tensor=%s host=%llx dev=%llx\n",
                        name.c_str(), host_sums_[name], ds);
        }
        q27k::t3_relayout_device((const uint8_t*)raw, (uint8_t*)d.data, (int64_t)d.rows,
                                 (int64_t)d.cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(raw));
    }
    bytes_ += dev_bytes;
    d.data_bytes = dev_bytes;
    if (src.scales) {
        CUDA_CHECK(cudaMalloc(&d.scales, src.scales_size));
        if (want_host_sum_) host_sum_ += host_xsum(src.scales, src.scales_size);
        CUDA_CHECK(cudaMemcpy(d.scales, src.scales, src.scales_size, cudaMemcpyHostToDevice));
        bytes_ += src.scales_size;
        d.scales_bytes = src.scales_size;
    }
    return dev_.emplace(name, d).first->second;
}

// Order-independent u64 word-sum (wraparound add): any flipped bit changes it,
// and atomicAdd accumulation order does not.
__global__ void k_xsum64(const unsigned long long* __restrict__ p, size_t n64,
                         const unsigned char* __restrict__ tail, int ntail,
                         unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n64;
         i += (size_t)gridDim.x * blockDim.x)
        s += p[i];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
    if (blockIdx.x == 0 && threadIdx.x == 0 && ntail) {
        unsigned long long t = 0;
        for (int i = 0; i < ntail; i++) t |= (unsigned long long)tail[i] << (8 * i);
        atomicAdd(out, t);
    }
}

static unsigned long long xsum_dev(const void* p, uint64_t bytes, unsigned long long* d_out) {
    CUDA_CHECK(cudaMemset(d_out, 0, 8));
    size_t n64 = bytes / 8;
    int ntail = (int)(bytes % 8);
    const unsigned char* tail = (const unsigned char*)p + n64 * 8;
    unsigned blocks = (unsigned)std::min<size_t>(4096, (n64 + 255) / 256);
    if (!blocks) blocks = 1;
    k_xsum64<<<blocks, 256>>>((const unsigned long long*)p, n64, tail, ntail, d_out);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long h = 0;
    CUDA_CHECK(cudaMemcpy(&h, d_out, 8, cudaMemcpyDeviceToHost));
    return h;
}

int DeviceModel::locate_upload_errors() const {
    if (host_sums_.empty()) {
        fprintf(stderr, "[locate] host sums not recorded (needs Q27_PRINT_WSUM=1 at load)\n");
        return -1;
    }
    unsigned long long* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 8));
    // device address order, so a mismatch can name its neighbours
    std::vector<std::pair<const char*, const DevTensor*>> by_addr;
    for (const auto& [name, t] : dev_) by_addr.emplace_back(name.c_str(), &t);
    std::sort(by_addr.begin(), by_addr.end(),
              [](const auto& a, const auto& b) { return a.second->data < b.second->data; });
    int bad = 0;
    for (size_t k = 0; k < by_addr.size(); k++) {
        const char* name = by_addr[k].first;
        const DevTensor& t = *by_addr[k].second;
        auto hit = host_sums_.find(name);
        if (hit == host_sums_.end()) continue;
        // T2/T3 device copies are relaid at upload (kernels.cuh), so their
        // bytes never equal the host bytes; T3's transfer is checked inline in
        // upload() instead, T2's is not checked here.
        if (t.dtype == DType::T2_G128 || t.dtype == DType::T3_G128) continue;
        const unsigned long long ds = xsum_dev(t.data, t.data_bytes, d_out);
        if (ds == hit->second) continue;
        bad++;
        const long long delta = (long long)(ds - hit->second);
        fprintf(stderr, "[locate] MISMATCH tensor=%s bytes=%zu dev=%p delta=%+lld\n",
                name, (size_t)t.data_bytes, t.data, delta);
        if (k) fprintf(stderr, "[locate]   below: %s ends at %p (gap %lld B)\n", by_addr[k-1].first,
                       (char*)by_addr[k-1].second->data + by_addr[k-1].second->data_bytes,
                       (long long)((char*)t.data - ((char*)by_addr[k-1].second->data + by_addr[k-1].second->data_bytes)));
        if (k + 1 < by_addr.size()) fprintf(stderr, "[locate]   above: %s starts at %p (gap %lld B)\n", by_addr[k+1].first,
                       by_addr[k+1].second->data,
                       (long long)((char*)by_addr[k+1].second->data - ((char*)t.data + t.data_bytes)));
        // read back and diff against the SOURCE bytes still mapped on the host
        const Tensor& src = model_.get(name);
        std::vector<unsigned char> back(t.data_bytes);
        CUDA_CHECK(cudaMemcpy(back.data(), t.data, t.data_bytes, cudaMemcpyDeviceToHost));
        const unsigned char* s = (const unsigned char*)src.data;
        int shown = 0;
        for (size_t i = 0; i < t.data_bytes; i++) {
            if (back[i] == s[i]) continue;
            const unsigned x = back[i] ^ s[i];
            for (int b = 0; b < 8; b++) if ((x >> b) & 1) {
                const size_t bit = i * 8 + b;
                fprintf(stderr, "[locate]   offset %zu of %zu (word %zu, dword %zu bit %zu, byte-in-word %zu) %s"
                                "  %s edge\n",
                        i, (size_t)t.data_bytes, i / 8, i / 4, (size_t)(b + 8 * (i % 4)), i % 8,
                        ((back[i] >> b) & 1) ? "0->1" : "1->0",
                        (i < 64 || i + 64 >= t.data_bytes) ? "AT" : "not at");
            }
            if (++shown >= 8) { fprintf(stderr, "[locate]   ...\n"); break; }
        }
        // is the corruption stable in VRAM? re-sum after the readback
        const unsigned long long ds2 = xsum_dev(t.data, t.data_bytes, d_out);
        fprintf(stderr, "[locate]   device re-sum %s\n", ds2 == ds ? "STABLE" : "CHANGED");
    }
    CUDA_CHECK(cudaFree(d_out));
    fprintf(stderr, "[locate] %zu tensors compared, %d mismatched\n", host_sums_.size(), bad);
    return bad;
}

void DeviceModel::checksum_baseline() {
    unsigned long long* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 8));
    for (const auto& [name, t] : dev_) {
        unsigned long long s = xsum_dev(t.data, t.data_bytes, d_out);
        if (t.scales)
            s ^= 0x9e3779b97f4a7c15ULL + xsum_dev(t.scales, t.scales_bytes, d_out);
        sums_[name] = s;
    }
    CUDA_CHECK(cudaFree(d_out));
}

unsigned long long DeviceModel::checksum_aggregate() const {
    // Wraparound u64 add: commutative, so the unordered_map iteration order
    // does not matter (same argument as k_xsum64's atomicAdd accumulation).
    unsigned long long a = 0;
    for (const auto& [name, s] : sums_) {
        (void)name;
        a += s;
    }
    return a;
}

int DeviceModel::checksum_verify(bool print) const {
    unsigned long long* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 8));
    int bad = 0;
    for (const auto& [name, t] : dev_) {
        auto it = sums_.find(name);
        if (it == sums_.end()) continue;
        unsigned long long s = xsum_dev(t.data, t.data_bytes, d_out);
        if (t.scales)
            s ^= 0x9e3779b97f4a7c15ULL + xsum_dev(t.scales, t.scales_bytes, d_out);
        if (s != it->second) {
            bad++;
            if (print)
                fprintf(stderr, "WEIGHT CHECKSUM MISMATCH: %s (%llx != %llx)\n", name.c_str(),
                        s, it->second);
        }
    }
    CUDA_CHECK(cudaFree(d_out));
    if (print)
        fprintf(stderr, "weight verify: %zu tensors, %d mismatched%s\n", sums_.size(), bad,
                bad ? " -- RESIDENT WEIGHTS CORRUPTED (reload required)" : "");
    return bad;
}

static bool is_pf4_sidecar(const std::string& name) {
    return name.size() > 4 && name.compare(name.size() - 4, 4, ".pf4") == 0;
}

void DeviceModel::upload_all(bool with_pf4, bool drop_shadowed_q4) {
    validate_cuda_model(model_);

    for (const auto& t : model_.tensors) {
        if (!with_pf4 && is_pf4_sidecar(t.name)) continue;
        if (drop_shadowed_q4 && !is_pf4_sidecar(t.name) && model_.find(t.name + ".pf4"))
            continue;
        upload(t.name);
    }
}

const DevTensor& DeviceModel::get(const std::string& name) const {
    auto it = dev_.find(name);
    if (it == dev_.end()) throw std::runtime_error("not resident on device: " + name);
    return it->second;
}

} // namespace q27
