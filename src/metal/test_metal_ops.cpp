#include "metal_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <vector>

namespace {

bool near(float a, float b, float tol = 3e-4f) {
    return std::fabs(a - b) <= tol * std::max(1.0f, std::fabs(b));
}

q27::BackendTensor upload_f32(q27::MetalBackend& backend, const std::vector<float>& values,
                              std::vector<uint64_t> shape) {
    q27::Tensor tensor;
    tensor.name = "test-f32";
    tensor.dtype = q27::DType::F32;
    tensor.shape = std::move(shape);
    tensor.data = reinterpret_cast<const uint8_t*>(values.data());
    tensor.data_size = values.size() * sizeof(float);
    return backend.upload(tensor);
}

std::shared_ptr<q27::BackendBuffer> upload_buffer(q27::MetalBackend& backend,
                                                  const std::vector<float>& values) {
    auto result = backend.allocate(values.size() * sizeof(float));
    backend.write(*result, 0, values.data(), values.size() * sizeof(float));
    return result;
}

std::vector<float> read_f32(q27::MetalBackend& backend, const q27::BackendBuffer& buffer,
                            size_t n) {
    std::vector<float> result(n);
    backend.read(buffer, 0, result.data(), n * sizeof(float));
    return result;
}

int test_primitives(q27::MetalBackend& backend) {
    int failures = 0;
    auto fail = [&](const char* name, size_t i, float got, float want) {
        fprintf(stderr, "%s[%zu]: got %.8g want %.8g\n", name, i, got, want);
        failures++;
    };

    // Q8 embedding.
    std::vector<int8_t> ew(2 * 128);
    for (size_t i = 0; i < ew.size(); i++) ew[i] = (int8_t)((int)(i % 21) - 10);
    std::vector<uint16_t> es = {0x3c00, 0x3800};
    q27::Tensor et;
    et.name = "embedding"; et.dtype = q27::DType::Q8_G128; et.shape = {2, 128};
    et.data = reinterpret_cast<const uint8_t*>(ew.data()); et.data_size = ew.size();
    et.scales = reinterpret_cast<const uint8_t*>(es.data()); et.scales_size = es.size() * 2;
    auto embedding = backend.upload(et);
    auto embed_out = backend.allocate(128 * 4);
    backend.embedding_q8(embedding, 1, *embed_out);
    auto embedded = read_f32(backend, *embed_out, 128);
    for (size_t i = 0; i < embedded.size(); i++) {
        float want = ew[128 + i] * 0.5f;
        if (!near(embedded[i], want)) { fail("embedding", i, embedded[i], want); break; }
    }

    // RMSNorm.
    std::vector<float> x = {1, -2, 3, -4, 0.5f};
    std::vector<float> w = {1, 0.5f, -1, 2, 3};
    auto xb = upload_buffer(backend, x); auto out = backend.allocate(x.size() * 4);
    auto wt = upload_f32(backend, w, {w.size()});
    backend.rmsnorm(*xb, wt, *out, (uint32_t)x.size(), 1e-6f);
    auto got = read_f32(backend, *out, x.size());
    float sum = std::inner_product(x.begin(), x.end(), x.begin(), 0.0f);
    float inv = 1.0f / std::sqrt(sum / x.size() + 1e-6f);
    for (size_t i = 0; i < x.size(); i++) if (!near(got[i], x[i] * inv * w[i])) fail("rmsnorm", i, got[i], x[i] * inv * w[i]);

    // Fused RMSNorm + group-32 quantization must match the two-dispatch path.
    std::vector<float> fx(32),fw(32,1.0f); for(size_t i=0;i<fx.size();i++) fx[i]=(float)((int)i-15)/7;
    auto fxb=upload_buffer(backend,fx); auto fwt=upload_f32(backend,fw,{32});
    auto normal=backend.allocate(32*4),fused=backend.allocate(32*4);
    auto nq=backend.allocate_quantized(32),fq=backend.allocate_quantized(32);
    backend.begin_commands(); backend.rmsnorm(*fxb,fwt,*normal,32,1e-6f); backend.quantize(*normal,nq); backend.end_commands();
    backend.rmsnorm_quantized(*fxb,fwt,*fused,32,1e-6f,fq);
    auto normal_f=read_f32(backend,*normal,32),fused_f=read_f32(backend,*fused,32);
    std::vector<int8_t> normal_q(32),fused_q(32); float normal_s=0,fused_s=0;
    backend.read(*nq.values,0,normal_q.data(),32); backend.read(*fq.values,0,fused_q.data(),32);
    backend.read(*nq.scales,0,&normal_s,4); backend.read(*fq.scales,0,&fused_s,4);
    for(size_t i=0;i<32;i++) if(normal_f[i]!=fused_f[i] || normal_q[i]!=fused_q[i]) fail("fused rms",i,fused_f[i],normal_f[i]);
    if(normal_s!=fused_s) fail("fused rms scale",0,fused_s,normal_s);

    // Per-head RMSNorm with a stride, then contiguous L2 normalization.
    std::vector<float> heads = {1,2,3,4,99,99, -1,2,-3,4,88,88};
    std::vector<float> hw = {1,0.5f,2,-1};
    auto hb = upload_buffer(backend, heads); auto hwt = upload_f32(backend, hw, {4});
    backend.rmsnorm_heads(*hb, hwt, 2, 4, 6, 1e-6f);
    auto hg = read_f32(backend, *hb, heads.size());
    for (uint32_t h = 0; h < 2; h++) {
        float ss = 0; for (uint32_t d=0; d<4; d++) ss += heads[h*6+d]*heads[h*6+d];
        float ri = 1/std::sqrt(ss/4+1e-6f);
        for(uint32_t d=0;d<4;d++) if(!near(hg[h*6+d],heads[h*6+d]*ri*hw[d])) fail("head rms",h*6+d,hg[h*6+d],heads[h*6+d]*ri*hw[d]);
    }
    std::vector<float> l2 = {3,4,0,0, 1,2,2,1};
    auto l2b=upload_buffer(backend,l2); backend.l2norm_heads(*l2b,2,4,1e-6f); auto l2g=read_f32(backend,*l2b,8);
    for(uint32_t h=0;h<2;h++){float ss=0;for(uint32_t d=0;d<4;d++)ss+=l2[h*4+d]*l2[h*4+d];float li=1/std::max(std::sqrt(ss),1e-6f);for(uint32_t d=0;d<4;d++)if(!near(l2g[h*4+d],l2[h*4+d]*li))fail("l2",h*4+d,l2g[h*4+d],l2[h*4+d]*li);}

    // Elementwise sequence in one command buffer.
    std::vector<float> gate={-2,-1,0,1,2}, up={1,2,3,4,5}, residual={.1f,.2f,.3f,.4f,.5f};
    auto gb=upload_buffer(backend,gate), ub=upload_buffer(backend,up), rb=upload_buffer(backend,residual); auto eb=backend.allocate(20);
    backend.begin_commands(); backend.silu_mul(*gb,*ub,*eb,5); backend.add_inplace(*eb,*rb,5); backend.end_commands();
    auto eg=read_f32(backend,*eb,5); for(size_t i=0;i<5;i++){float want=gate[i]/(1+std::exp(-gate[i]))*up[i]+residual[i];if(!near(eg[i],want))fail("elementwise",i,eg[i],want);}

    // Device-side state copy with non-zero offsets (used by prefix snapshots).
    std::vector<float> copy_src={1,2,3,4,5},copy_zero(7,0);
    auto cs=upload_buffer(backend,copy_src),cd=upload_buffer(backend,copy_zero);
    backend.copy(*cs,4,*cd,8,3*sizeof(float)); auto copied=read_f32(backend,*cd,copy_zero.size());
    for(size_t i=0;i<3;i++) if(copied[i+2]!=copy_src[i+1]) fail("copy",i,copied[i+2],copy_src[i+1]);

    // Sigmoid gate.
    std::vector<float> values={1,2,3,4}, qg={0,0,1,-1, 2,-2,0.5f,-0.5f};
    auto vb=upload_buffer(backend,values), qgb=upload_buffer(backend,qg); backend.sigmoid_gate_mul(*vb,*qgb,2,2); auto vg=read_f32(backend,*vb,4);
    for(size_t i=0;i<4;i++){uint32_t h=i/2,d=i%2;float gv=qg[h*4+2+d];float want=values[i]/(1+std::exp(-gv));if(!near(vg[i],want))fail("sigmoid gate",i,vg[i],want);}

    // Partial NeoX RoPE.
    std::vector<float> rope={1,2,3,4,5,6,7,8}; auto ropeb=upload_buffer(backend,rope);
    backend.rope_neox(*ropeb,1,8,4,8,3,10000.0f); auto rg=read_f32(backend,*ropeb,8);
    for(uint32_t d=0;d<2;d++){float theta=3*std::pow(10000.0f,-2.0f*d/4.0f);float c=std::cos(theta),s=std::sin(theta);float a=rope[d],b=rope[d+2];if(!near(rg[d],a*c-b*s))fail("rope",d,rg[d],a*c-b*s);if(!near(rg[d+2],a*s+b*c))fail("rope",d+2,rg[d+2],a*s+b*c);}

    // Stable lower-index tie break.
    std::vector<float> logits={-1,7,3,7,2}; auto lb=upload_buffer(backend,logits); auto ib=backend.allocate(4); backend.argmax(*lb,5,*ib); uint32_t index=99; backend.read(*ib,0,&index,4); if(index!=1){fprintf(stderr,"argmax: got %u want 1\n",index);failures++;}
    return failures;
}

int test_attention(q27::MetalBackend& backend) {
    constexpr uint32_t seq=3,qh=2,kvh=1,dim=4,stride=8;
    auto kc=backend.allocate(seq*kvh*dim*2), vc=backend.allocate(seq*kvh*dim*2);
    const std::vector<std::vector<float>> keys={{1,0,0,0},{0,1,0,0},{1,1,0,0}};
    const std::vector<std::vector<float>> vals={{1,2,3,4},{2,0,1,3},{4,3,2,1}};
    for(uint32_t p=0;p<seq;p++){auto k=upload_buffer(backend,keys[p]),v=upload_buffer(backend,vals[p]);backend.kv_store_f16(*k,*v,*kc,*vc,p,dim);}
    std::vector<float> q={1,.5f,0,0, 9,9,9,9, 0,1,0,0, 8,8,8,8}; auto qb=upload_buffer(backend,q);
    auto out=backend.allocate(qh*dim*4);
    backend.attention_f16(*qb,stride,*kc,*vc,*out,seq,qh,kvh,dim,0.5f); auto got=read_f32(backend,*out,qh*dim);
    int failures=0;
    for(uint32_t h=0;h<qh;h++){std::vector<float>s(seq);float mx=-1e30f;for(uint32_t p=0;p<seq;p++){s[p]=0;for(uint32_t d=0;d<dim;d++)s[p]+=q[h*stride+d]*keys[p][d];s[p]*=.5f;mx=std::max(mx,s[p]);}float den=0;for(float&v:s){v=std::exp(v-mx);den+=v;}for(float&v:s)v/=den;for(uint32_t d=0;d<dim;d++){float want=0;for(uint32_t p=0;p<seq;p++)want+=s[p]*vals[p][d];if(!near(got[h*dim+d],want,8e-4f)){fprintf(stderr,"attention[%u,%u] got %.8g want %.8g\n",h,d,got[h*dim+d],want);failures++;}}}
    return failures;
}

// Production-shape gate for the online-softmax decode kernel: 24:4 GQA at
// head_dim 256 with sequence lengths that exercise unequal simdgroup stripes
// (133), empty stripes (5), and running-max updates (scores swing sign and
// magnitude across positions). The tiny test above cannot reach any of that.
int test_attention_production_shape(q27::MetalBackend& backend) {
    int failures = 0;
    uint32_t lcg = 12345;
    auto uniform = [&]() { lcg = lcg * 1664525u + 1013904223u; return (float)(lcg >> 8) / 8388608.0f - 1.0f; };
    for (uint32_t seq : {5u, 133u}) {
        constexpr uint32_t qh = 24, kvh = 4, dim = 256, stride = 2 * dim;
        const float scale = 1.0f / 16.0f;
        std::vector<std::vector<float>> keys(seq), vals(seq);
        auto kc = backend.allocate((uint64_t)seq * kvh * dim * 2);
        auto vc = backend.allocate((uint64_t)seq * kvh * dim * 2);
        for (uint32_t p = 0; p < seq; p++) {
            keys[p].resize(kvh * dim); vals[p].resize(kvh * dim);
            // Alternate key magnitudes so the running maximum keeps moving.
            const float magnitude = (p % 3 == 0) ? 2.5f : 0.4f;
            for (auto& value : keys[p]) value = uniform() * magnitude;
            for (auto& value : vals[p]) value = uniform() * 3.0f;
            auto kb = upload_buffer(backend, keys[p]), vb = upload_buffer(backend, vals[p]);
            backend.kv_store_f16(*kb, *vb, *kc, *vc, p, kvh * dim);
        }
        // Half-precision K/V for the CPU reference, matching what the cache holds.
        auto as_half = [](float v) { return (float)(_Float16)v; };
        std::vector<float> q((uint64_t)qh * stride);
        for (auto& value : q) value = uniform() * 1.5f;
        auto qb = upload_buffer(backend, q);
        auto out = backend.allocate((uint64_t)qh * dim * 4);
        backend.attention_f16(*qb, stride, *kc, *vc, *out, seq, qh, kvh, dim, scale);
        auto got = read_f32(backend, *out, (uint64_t)qh * dim);
        for (uint32_t h = 0; h < qh; h++) {
            const uint32_t kh = h / (qh / kvh);
            std::vector<float> s(seq);
            float mx = -1e30f;
            for (uint32_t p = 0; p < seq; p++) {
                s[p] = 0;
                for (uint32_t d = 0; d < dim; d++) s[p] += q[h * stride + d] * as_half(keys[p][kh * dim + d]);
                s[p] *= scale;
                mx = std::max(mx, s[p]);
            }
            float den = 0;
            for (float& value : s) { value = std::exp(value - mx); den += value; }
            for (float& value : s) value /= den;
            for (uint32_t d = 0; d < dim; d++) {
                float want = 0;
                for (uint32_t p = 0; p < seq; p++) want += s[p] * as_half(vals[p][kh * dim + d]);
                if (!near(got[h * dim + d], want, 2e-3f)) {
                    fprintf(stderr, "attention prod seq=%u [%u,%u] got %.8g want %.8g\n",
                            seq, h, d, got[h * dim + d], want);
                    failures++;
                }
            }
        }
    }
    return failures;
}

int test_turbo3(q27::MetalBackend& backend) {
    constexpr uint32_t seq=3,qh=2,kvh=1,dim=256,stride=256;
    int failures=0;

    // The signed randomized Hadamard transform must invert independently for
    // both 128-element groups in every head.
    std::vector<float> original(qh*dim);
    for(size_t i=0;i<original.size();i++) original[i]=std::sin(float(i)*.071f)+.1f*std::cos(float(i)*.013f);
    auto roundtrip=upload_buffer(backend,original);
    backend.begin_commands(); backend.turbo_wht(*roundtrip,qh,stride,false); backend.turbo_wht(*roundtrip,qh,stride,true); backend.end_commands();
    auto restored=read_f32(backend,*roundtrip,original.size());
    for(size_t i=0;i<original.size();i++) if(!near(restored[i],original[i],2e-5f)) {
        fprintf(stderr,"turbo3 WHT roundtrip[%zu] got %.8g want %.8g\n",i,restored[i],original[i]); failures++; break;
    }

    auto k16=backend.allocate((uint64_t)seq*kvh*dim*2), v16=backend.allocate((uint64_t)seq*kvh*dim*2);
    auto kt3=backend.allocate((uint64_t)seq*kvh*2*50), vt3=backend.allocate((uint64_t)seq*kvh*2*50);
    // Zero maps to centroid index 4: low two bits are all zero and every
    // separately packed high bit is one. This catches leakage of bit 2 into
    // the neighboring 2-bit fields.
    std::vector<float> zeros(dim,0); auto zerob=upload_buffer(backend,zeros);
    backend.kv_store_turbo3(*zerob,*zerob,*kt3,*vt3,0,kvh);
    std::vector<uint8_t> zero_block(50); backend.read(*kt3,0,zero_block.data(),zero_block.size());
    for(size_t i=2;i<34;i++) if(zero_block[i]!=0) { fprintf(stderr,"turbo3 low-bit packing[%zu]=%u\n",i,zero_block[i]); failures++; break; }
    for(size_t i=34;i<50;i++) if(zero_block[i]!=255) { fprintf(stderr,"turbo3 high-bit packing[%zu]=%u\n",i,zero_block[i]); failures++; break; }
    for(uint32_t p=0;p<seq;p++) {
        std::vector<float> k(dim),v(dim);
        for(uint32_t d=0;d<dim;d++) {
            k[d]=.45f*std::sin(float(d+17*p)*.039f)+.08f*std::cos(float(d)*.11f);
            v[d]=.6f*std::cos(float(d+9*p)*.027f)-.12f*std::sin(float(d)*.07f);
        }
        auto kb=upload_buffer(backend,k),vb=upload_buffer(backend,v);
        backend.kv_store_f16(*kb,*vb,*k16,*v16,p,kvh*dim);
        backend.kv_store_turbo3(*kb,*vb,*kt3,*vt3,p,kvh);
    }
    std::vector<float> q(qh*dim);
    for(size_t i=0;i<q.size();i++) q[i]=.2f*std::sin(float(i)*.031f)+.1f*std::cos(float(i)*.053f);
    auto qbase=upload_buffer(backend,q),qt3=upload_buffer(backend,q);
    auto out16=backend.allocate((uint64_t)qh*dim*4),out3=backend.allocate((uint64_t)qh*dim*4);
    backend.attention_f16(*qbase,stride,*k16,*v16,*out16,seq,qh,kvh,dim,1/std::sqrt(float(dim)));
    backend.begin_commands();
    backend.turbo_wht(*qt3,qh,stride,false);
    backend.attention_turbo3(*qt3,stride,*kt3,*vt3,*out3,seq,qh,kvh,dim,1/std::sqrt(float(dim)));
    backend.turbo_wht(*out3,qh,dim,true);
    backend.end_commands();
    auto baseline=read_f32(backend,*out16,qh*dim),compressed=read_f32(backend,*out3,qh*dim);
    double signal=0,error=0,dot=0,norm3=0;
    for(size_t i=0;i<baseline.size();i++) { signal+=baseline[i]*baseline[i]; double e=compressed[i]-baseline[i]; error+=e*e; dot+=compressed[i]*baseline[i]; norm3+=compressed[i]*compressed[i]; }
    const double nrmse=std::sqrt(error/std::max(signal,1e-30));
    const double cosine=dot/std::sqrt(std::max(signal*norm3,1e-30));
    printf("turbo3 synthetic attention: nrmse %.5f, cosine %.6f\n",nrmse,cosine);
    // This intentionally measures the complete lossy K+V path, not just the
    // reversible WHT. Keep the bound below the observed failure modes for a
    // broken sign/group mapping while allowing the 3-bit codec's expected loss.
    if(nrmse>.30 || cosine<.95) { fprintf(stderr,"turbo3 attention quality: nrmse %.5f cosine %.6f\n",nrmse,cosine); failures++; }
    return failures;
}

// Production-shape gate for the online-softmax turbo3 decode kernel: 24:4
// GQA at head_dim 256 with sequence lengths exercising empty simdgroup
// stripes (5) and multi-stripe running-max updates (133, key magnitudes
// alternate so the maximum keeps moving). The CPU reference dequantizes the
// stored 50-byte blocks, so it checks the kernel's attention math exactly;
// codec loss is the separate quality gate above.
int test_turbo3_production_shape(q27::MetalBackend& backend) {
    constexpr float centroids[8] = {
        -0.190207f, -0.118786f, -0.066822f, -0.021663f,
         0.021663f,  0.066822f,  0.118786f,  0.190207f };
    auto dequant = [&](const uint8_t* block, uint32_t j) {
        const uint32_t low = (block[2 + (j >> 2)] >> ((j & 3) * 2)) & 3;
        const uint32_t high = (block[34 + (j >> 3)] >> (j & 7)) & 1;
        _Float16 h; __builtin_memcpy(&h, block, 2);
        return centroids[low | (high << 2)] * (float)h;
    };
    int failures = 0;
    uint32_t lcg = 54321;
    auto uniform = [&]() { lcg = lcg * 1664525u + 1013904223u; return (float)(lcg >> 8) / 8388608.0f - 1.0f; };
    for (uint32_t seq : {5u, 133u}) {
        constexpr uint32_t qh = 24, kvh = 4, dim = 256, stride = 2 * dim;
        const float scale = 1.0f / 16.0f;
        const uint64_t row_bytes = (uint64_t)kvh * 2 * 50;
        auto kc = backend.allocate(seq * row_bytes);
        auto vc = backend.allocate(seq * row_bytes);
        for (uint32_t p = 0; p < seq; p++) {
            std::vector<float> k(kvh * dim), v(kvh * dim);
            const float magnitude = (p % 3 == 0) ? 2.5f : 0.4f;
            for (auto& value : k) value = uniform() * magnitude;
            for (auto& value : v) value = uniform() * 3.0f;
            auto kb = upload_buffer(backend, k), vb = upload_buffer(backend, v);
            backend.kv_store_turbo3(*kb, *vb, *kc, *vc, p, kvh);
        }
        std::vector<uint8_t> k_blocks(seq * row_bytes), v_blocks(seq * row_bytes);
        backend.read(*kc, 0, k_blocks.data(), k_blocks.size());
        backend.read(*vc, 0, v_blocks.data(), v_blocks.size());
        std::vector<float> q((uint64_t)qh * stride);
        for (auto& value : q) value = uniform() * 1.5f;
        auto qb = upload_buffer(backend, q);
        backend.turbo_wht(*qb, qh, stride, false);
        auto q_wht = read_f32(backend, *qb, q.size());
        auto out = backend.allocate((uint64_t)qh * dim * 4);
        backend.attention_turbo3(*qb, stride, *kc, *vc, *out, seq, qh, kvh, dim, scale);
        auto got = read_f32(backend, *out, (uint64_t)qh * dim);
        for (uint32_t h = 0; h < qh; h++) {
            const uint32_t kh = h / (qh / kvh);
            std::vector<float> s(seq);
            float mx = -1e30f;
            for (uint32_t p = 0; p < seq; p++) {
                s[p] = 0;
                for (uint32_t d = 0; d < dim; d++) {
                    const uint8_t* block =
                        k_blocks.data() + ((uint64_t)p * kvh * 2 + kh * 2 + (d >> 7)) * 50;
                    s[p] += q_wht[h * stride + d] * dequant(block, d & 127);
                }
                s[p] *= scale;
                mx = std::max(mx, s[p]);
            }
            float den = 0;
            for (float& value : s) { value = std::exp(value - mx); den += value; }
            for (float& value : s) value /= den;
            for (uint32_t d = 0; d < dim; d++) {
                float want = 0;
                for (uint32_t p = 0; p < seq; p++) {
                    const uint8_t* block =
                        v_blocks.data() + ((uint64_t)p * kvh * 2 + kh * 2 + (d >> 7)) * 50;
                    want += s[p] * dequant(block, d & 127);
                }
                if (!near(got[h * dim + d], want, 2e-3f)) {
                    fprintf(stderr, "turbo3 prod seq=%u [%u,%u] got %.8g want %.8g\n",
                            seq, h, d, got[h * dim + d], want);
                    failures++;
                }
            }
        }
    }
    return failures;
}

// GQA KV-reuse path parity: a second backend built under
// Q27_METAL_GQA_THRESHOLD=1 forces every decode attention call through the
// blocked kernels (one threadgroup per KV head per 1024-position block, one
// simdgroup per query head, block partials merged in index order). Same CPU
// references as the production-shape gates above; the sequence lengths
// cover a single partial block plus normalize-only merge (133), a two-block
// split with a six-row tail tile (1030), and three blocks (2050).
int test_attention_gqa_path() {
    setenv("Q27_METAL_GQA_THRESHOLD", "1", 1);
    q27::MetalBackend backend;
    unsetenv("Q27_METAL_GQA_THRESHOLD");
    int failures = 0;
    uint32_t lcg = 98765;
    auto uniform = [&]() { lcg = lcg * 1664525u + 1013904223u; return (float)(lcg >> 8) / 8388608.0f - 1.0f; };
    auto as_half = [](float v) { return (float)(_Float16)v; };
    constexpr uint32_t qh = 24, kvh = 4, dim = 256, stride = 2 * dim;
    const float scale = 1.0f / 16.0f;

    for (uint32_t seq : {133u, 1030u, 2050u}) {
        std::vector<std::vector<float>> keys(seq), vals(seq);
        std::vector<std::shared_ptr<q27::BackendBuffer>> kbufs(seq), vbufs(seq);
        auto kc = backend.allocate((uint64_t)seq * kvh * dim * 2);
        auto vc = backend.allocate((uint64_t)seq * kvh * dim * 2);
        for (uint32_t p = 0; p < seq; p++) {
            keys[p].resize(kvh * dim); vals[p].resize(kvh * dim);
            const float magnitude = (p % 3 == 0) ? 2.5f : 0.4f;
            for (auto& value : keys[p]) value = uniform() * magnitude;
            for (auto& value : vals[p]) value = uniform() * 3.0f;
            kbufs[p] = upload_buffer(backend, keys[p]);
            vbufs[p] = upload_buffer(backend, vals[p]);
        }
        backend.begin_commands();
        for (uint32_t p = 0; p < seq; p++)
            backend.kv_store_f16(*kbufs[p], *vbufs[p], *kc, *vc, p, kvh * dim);
        backend.end_commands();
        std::vector<float> q((uint64_t)qh * stride);
        for (auto& value : q) value = uniform() * 1.5f;
        auto qb = upload_buffer(backend, q);
        auto out = backend.allocate((uint64_t)qh * dim * 4);
        backend.attention_f16(*qb, stride, *kc, *vc, *out, seq, qh, kvh, dim, scale);
        auto got = read_f32(backend, *out, (uint64_t)qh * dim);
        for (uint32_t h = 0; h < qh; h++) {
            const uint32_t kh = h / (qh / kvh);
            std::vector<float> s(seq);
            float mx = -1e30f;
            for (uint32_t p = 0; p < seq; p++) {
                s[p] = 0;
                for (uint32_t d = 0; d < dim; d++) s[p] += q[h * stride + d] * as_half(keys[p][kh * dim + d]);
                s[p] *= scale;
                mx = std::max(mx, s[p]);
            }
            float den = 0;
            for (float& value : s) { value = std::exp(value - mx); den += value; }
            for (float& value : s) value /= den;
            for (uint32_t d = 0; d < dim; d++) {
                float want = 0;
                for (uint32_t p = 0; p < seq; p++) want += s[p] * as_half(vals[p][kh * dim + d]);
                if (!near(got[h * dim + d], want, 2e-3f)) {
                    fprintf(stderr, "gqa f16 seq=%u [%u,%u] got %.8g want %.8g\n",
                            seq, h, d, got[h * dim + d], want);
                    failures++;
                }
            }
        }
    }

    constexpr float centroids[8] = {
        -0.190207f, -0.118786f, -0.066822f, -0.021663f,
         0.021663f,  0.066822f,  0.118786f,  0.190207f };
    auto dequant = [&](const uint8_t* block, uint32_t j) {
        const uint32_t low = (block[2 + (j >> 2)] >> ((j & 3) * 2)) & 3;
        const uint32_t high = (block[34 + (j >> 3)] >> (j & 7)) & 1;
        _Float16 h; __builtin_memcpy(&h, block, 2);
        return centroids[low | (high << 2)] * (float)h;
    };
    for (uint32_t seq : {133u, 1030u, 2050u}) {
        const uint64_t row_bytes = (uint64_t)kvh * 2 * 50;
        auto kc = backend.allocate(seq * row_bytes);
        auto vc = backend.allocate(seq * row_bytes);
        std::vector<std::shared_ptr<q27::BackendBuffer>> kbufs(seq), vbufs(seq);
        for (uint32_t p = 0; p < seq; p++) {
            std::vector<float> k(kvh * dim), v(kvh * dim);
            const float magnitude = (p % 3 == 0) ? 2.5f : 0.4f;
            for (auto& value : k) value = uniform() * magnitude;
            for (auto& value : v) value = uniform() * 3.0f;
            kbufs[p] = upload_buffer(backend, k);
            vbufs[p] = upload_buffer(backend, v);
        }
        backend.begin_commands();
        for (uint32_t p = 0; p < seq; p++)
            backend.kv_store_turbo3(*kbufs[p], *vbufs[p], *kc, *vc, p, kvh);
        backend.end_commands();
        std::vector<uint8_t> k_blocks(seq * row_bytes), v_blocks(seq * row_bytes);
        backend.read(*kc, 0, k_blocks.data(), k_blocks.size());
        backend.read(*vc, 0, v_blocks.data(), v_blocks.size());
        std::vector<float> q((uint64_t)qh * stride);
        for (auto& value : q) value = uniform() * 1.5f;
        auto qb = upload_buffer(backend, q);
        backend.turbo_wht(*qb, qh, stride, false);
        auto q_wht = read_f32(backend, *qb, q.size());
        auto out = backend.allocate((uint64_t)qh * dim * 4);
        backend.attention_turbo3(*qb, stride, *kc, *vc, *out, seq, qh, kvh, dim, scale);
        auto got = read_f32(backend, *out, (uint64_t)qh * dim);
        for (uint32_t h = 0; h < qh; h++) {
            const uint32_t kh = h / (qh / kvh);
            std::vector<float> s(seq);
            float mx = -1e30f;
            for (uint32_t p = 0; p < seq; p++) {
                s[p] = 0;
                for (uint32_t d = 0; d < dim; d++) {
                    const uint8_t* block =
                        k_blocks.data() + ((uint64_t)p * kvh * 2 + kh * 2 + (d >> 7)) * 50;
                    s[p] += q_wht[h * stride + d] * dequant(block, d & 127);
                }
                s[p] *= scale;
                mx = std::max(mx, s[p]);
            }
            float den = 0;
            for (float& value : s) { value = std::exp(value - mx); den += value; }
            for (float& value : s) value /= den;
            for (uint32_t d = 0; d < dim; d++) {
                float want = 0;
                for (uint32_t p = 0; p < seq; p++) {
                    const uint8_t* block =
                        v_blocks.data() + ((uint64_t)p * kvh * 2 + kh * 2 + (d >> 7)) * 50;
                    want += s[p] * dequant(block, d & 127);
                }
                if (!near(got[h * dim + d], want, 2e-3f)) {
                    fprintf(stderr, "gqa turbo3 seq=%u [%u,%u] got %.8g want %.8g\n",
                            seq, h, d, got[h * dim + d], want);
                    failures++;
                }
            }
        }
    }

    // Causal-chunk GQA path: base 2040, 12 tokens, so the per-token live
    // block count changes inside the chunk (tokens 0-8 span two 1024-blocks,
    // tokens 9-11 span three) and the merge's dead-partial guard is on the
    // line. Same CPU references as the chunked gates.
    {
        constexpr uint32_t base = 2040, tokens = 12, max_seq = base + tokens - 1;
        const uint32_t q_row_stride = qh * stride;
        std::vector<std::vector<float>> keys(max_seq), vals(max_seq);
        std::vector<std::shared_ptr<q27::BackendBuffer>> kbufs(max_seq), vbufs(max_seq);
        auto kc = backend.allocate((uint64_t)max_seq * kvh * dim * 2);
        auto vc = backend.allocate((uint64_t)max_seq * kvh * dim * 2);
        for (uint32_t p = 0; p < max_seq; p++) {
            keys[p].resize(kvh * dim); vals[p].resize(kvh * dim);
            const float magnitude = (p % 3 == 0) ? 2.5f : 0.4f;
            for (auto& value : keys[p]) value = uniform() * magnitude;
            for (auto& value : vals[p]) value = uniform() * 3.0f;
            kbufs[p] = upload_buffer(backend, keys[p]);
            vbufs[p] = upload_buffer(backend, vals[p]);
        }
        backend.begin_commands();
        for (uint32_t p = 0; p < max_seq; p++)
            backend.kv_store_f16(*kbufs[p], *vbufs[p], *kc, *vc, p, kvh * dim);
        backend.end_commands();
        std::vector<float> q((uint64_t)tokens * q_row_stride);
        for (auto& value : q) value = uniform() * 1.5f;
        auto qb = upload_buffer(backend, q);
        auto out = backend.allocate((uint64_t)tokens * qh * dim * 4);
        backend.attention_f16_causal(*qb, stride, q_row_stride, *kc, *vc, *out,
                                     base, qh, kvh, dim, tokens, scale);
        auto got = read_f32(backend, *out, (uint64_t)tokens * qh * dim);
        for (uint32_t t = 0; t < tokens; t++) {
            const uint32_t seq = base + t;
            for (uint32_t h = 0; h < qh; h++) {
                const uint32_t kh = h / (qh / kvh);
                std::vector<float> s(seq);
                float mx = -1e30f;
                for (uint32_t p = 0; p < seq; p++) {
                    s[p] = 0;
                    for (uint32_t d = 0; d < dim; d++)
                        s[p] += q[(uint64_t)t * q_row_stride + h * stride + d] * as_half(keys[p][kh * dim + d]);
                    s[p] *= scale;
                    mx = std::max(mx, s[p]);
                }
                float den = 0;
                for (float& value : s) { value = std::exp(value - mx); den += value; }
                for (float& value : s) value /= den;
                for (uint32_t d = 0; d < dim; d++) {
                    float want = 0;
                    for (uint32_t p = 0; p < seq; p++) want += s[p] * as_half(vals[p][kh * dim + d]);
                    const float have = got[((uint64_t)t * qh + h) * dim + d];
                    if (!near(have, want, 2e-3f)) {
                        fprintf(stderr, "gqa f16 causal t=%u [%u,%u] got %.8g want %.8g\n",
                                t, h, d, have, want);
                        failures++;
                    }
                }
            }
        }
    }
    {
        constexpr uint32_t base = 2040, tokens = 12, max_seq = base + tokens - 1;
        const uint32_t q_row_stride = qh * stride;
        const uint64_t row_bytes = (uint64_t)kvh * 2 * 50;
        auto kc = backend.allocate((uint64_t)max_seq * row_bytes);
        auto vc = backend.allocate((uint64_t)max_seq * row_bytes);
        std::vector<std::shared_ptr<q27::BackendBuffer>> kbufs(max_seq), vbufs(max_seq);
        for (uint32_t p = 0; p < max_seq; p++) {
            std::vector<float> k(kvh * dim), v(kvh * dim);
            const float magnitude = (p % 3 == 0) ? 2.5f : 0.4f;
            for (auto& value : k) value = uniform() * magnitude;
            for (auto& value : v) value = uniform() * 3.0f;
            kbufs[p] = upload_buffer(backend, k);
            vbufs[p] = upload_buffer(backend, v);
        }
        backend.begin_commands();
        for (uint32_t p = 0; p < max_seq; p++)
            backend.kv_store_turbo3(*kbufs[p], *vbufs[p], *kc, *vc, p, kvh);
        backend.end_commands();
        std::vector<uint8_t> k_blocks(max_seq * row_bytes), v_blocks(max_seq * row_bytes);
        backend.read(*kc, 0, k_blocks.data(), k_blocks.size());
        backend.read(*vc, 0, v_blocks.data(), v_blocks.size());
        std::vector<float> q((uint64_t)tokens * q_row_stride);
        for (auto& value : q) value = uniform() * 1.5f;
        auto qb = upload_buffer(backend, q);
        backend.turbo_wht(*qb, tokens * qh, stride, false);
        auto q_wht = read_f32(backend, *qb, q.size());
        auto out = backend.allocate((uint64_t)tokens * qh * dim * 4);
        backend.attention_turbo3_causal(*qb, stride, q_row_stride, *kc, *vc, *out,
                                        base, qh, kvh, dim, tokens, scale);
        auto got = read_f32(backend, *out, (uint64_t)tokens * qh * dim);
        for (uint32_t t = 0; t < tokens; t++) {
            const uint32_t seq = base + t;
            for (uint32_t h = 0; h < qh; h++) {
                const uint32_t kh = h / (qh / kvh);
                std::vector<float> s(seq);
                float mx = -1e30f;
                for (uint32_t p = 0; p < seq; p++) {
                    s[p] = 0;
                    for (uint32_t d = 0; d < dim; d++) {
                        const uint8_t* block =
                            k_blocks.data() + ((uint64_t)p * kvh * 2 + kh * 2 + (d >> 7)) * 50;
                        s[p] += q_wht[(uint64_t)t * q_row_stride + h * stride + d] * dequant(block, d & 127);
                    }
                    s[p] *= scale;
                    mx = std::max(mx, s[p]);
                }
                float den = 0;
                for (float& value : s) { value = std::exp(value - mx); den += value; }
                for (float& value : s) value /= den;
                for (uint32_t d = 0; d < dim; d++) {
                    float want = 0;
                    for (uint32_t p = 0; p < seq; p++) {
                        const uint8_t* block =
                            v_blocks.data() + ((uint64_t)p * kvh * 2 + kh * 2 + (d >> 7)) * 50;
                        want += s[p] * dequant(block, d & 127);
                    }
                    const float have = got[((uint64_t)t * qh + h) * dim + d];
                    if (!near(have, want, 2e-3f)) {
                        fprintf(stderr, "gqa turbo3 causal t=%u [%u,%u] got %.8g want %.8g\n",
                                t, h, d, have, want);
                        failures++;
                    }
                }
            }
        }
    }
    return failures;
}

// Threshold-straddle contract gate (codex review finding 1): with the
// DEFAULT threshold, a chunk whose rows span the switch point must produce
// bit-identical output to serial decode at each row's sequence length —
// rows below the threshold on the legacy kernels, rows at or above it on
// the GQA kernels, split into two dispatches by the host. base 2043 with 12
// tokens puts the 2048 switch inside the chunk.
int test_attention_gqa_straddle() {
    q27::MetalBackend backend;      // default Q27_METAL_GQA_THRESHOLD = 2048
    int failures = 0;
    uint32_t lcg = 24680;
    auto uniform = [&]() { lcg = lcg * 1664525u + 1013904223u; return (float)(lcg >> 8) / 8388608.0f - 1.0f; };
    constexpr uint32_t qh = 24, kvh = 4, dim = 256, stride = 2 * dim;
    constexpr uint32_t base = 2043, tokens = 12, max_seq = base + tokens - 1;
    const uint32_t q_row_stride = qh * stride;
    const float scale = 1.0f / 16.0f;
    const uint64_t row_bytes = (uint64_t)kvh * 2 * 50;

    auto kc = backend.allocate((uint64_t)max_seq * row_bytes);
    auto vc = backend.allocate((uint64_t)max_seq * row_bytes);
    std::vector<std::shared_ptr<q27::BackendBuffer>> kbufs(max_seq), vbufs(max_seq);
    for (uint32_t p = 0; p < max_seq; p++) {
        std::vector<float> k(kvh * dim), v(kvh * dim);
        const float magnitude = (p % 3 == 0) ? 2.5f : 0.4f;
        for (auto& value : k) value = uniform() * magnitude;
        for (auto& value : v) value = uniform() * 3.0f;
        kbufs[p] = upload_buffer(backend, k);
        vbufs[p] = upload_buffer(backend, v);
    }
    backend.begin_commands();
    for (uint32_t p = 0; p < max_seq; p++)
        backend.kv_store_turbo3(*kbufs[p], *vbufs[p], *kc, *vc, p, kvh);
    backend.end_commands();

    std::vector<float> q((uint64_t)tokens * q_row_stride);
    for (auto& value : q) value = uniform() * 1.5f;
    auto qb = upload_buffer(backend, q);
    backend.turbo_wht(*qb, tokens * qh, stride, false);
    auto q_wht = read_f32(backend, *qb, q.size());
    auto out_chunk = backend.allocate((uint64_t)tokens * qh * dim * 4);
    backend.attention_turbo3_causal(*qb, stride, q_row_stride, *kc, *vc, *out_chunk,
                                    base, qh, kvh, dim, tokens, scale);
    auto chunk = read_f32(backend, *out_chunk, (uint64_t)tokens * qh * dim);

    auto qrow = backend.allocate((uint64_t)q_row_stride * 4);
    auto out_dec = backend.allocate((uint64_t)qh * dim * 4);
    for (uint32_t t = 0; t < tokens; t++) {
        backend.write(*qrow, 0, q_wht.data() + (uint64_t)t * q_row_stride, (uint64_t)q_row_stride * 4);
        backend.attention_turbo3(*qrow, stride, *kc, *vc, *out_dec, base + t, qh, kvh, dim, scale);
        auto dec = read_f32(backend, *out_dec, (uint64_t)qh * dim);
        if (std::memcmp(dec.data(), chunk.data() + (uint64_t)t * qh * dim,
                        (size_t)qh * dim * sizeof(float)) != 0) {
            for (uint32_t i = 0; i < qh * dim; i++)
                if (dec[i] != chunk[(uint64_t)t * qh * dim + i]) {
                    fprintf(stderr, "gqa straddle t=%u (seq %u) first diff at %u: chunk %.9g decode %.9g\n",
                            t, base + t, i, chunk[(uint64_t)t * qh * dim + i], dec[i]);
                    break;
                }
            failures++;
        }
    }
    return failures;
}

// R1b tiled-route parity: the factor-2 token-tiled causal GQA kernels must
// be bit-identical to the untiled kernels for every token. Two backends
// (tile 2 vs Q27_METAL_GQA_TILE=1) attend over the same cache bytes; odd
// token counts put a single live token in the last tile, and base 1020
// puts the tile's staged range across a 1024-block boundary.
int test_attention_gqa_tiled_parity() {
    // Pin BOTH backends' tile settings explicitly — if the caller exported
    // the Q27_METAL_GQA_TILE=1 opt-out, inheriting it would make this gate
    // compare untiled to untiled and pass vacuously (codex P2, 2026-07-15).
    const char* caller_tile = getenv("Q27_METAL_GQA_TILE");
    const std::string saved_tile = caller_tile ? caller_tile : "";
    setenv("Q27_METAL_GQA_THRESHOLD", "1", 1);
    setenv("Q27_METAL_GQA_TILE", "2", 1);
    q27::MetalBackend tiled;
    setenv("Q27_METAL_GQA_TILE", "1", 1);
    q27::MetalBackend untiled;
    if (caller_tile) setenv("Q27_METAL_GQA_TILE", saved_tile.c_str(), 1);
    else unsetenv("Q27_METAL_GQA_TILE");
    unsetenv("Q27_METAL_GQA_THRESHOLD");
    int failures = 0;
    uint32_t lcg = 112358;
    auto uniform = [&]() { lcg = lcg * 1664525u + 1013904223u; return (float)(lcg >> 8) / 8388608.0f - 1.0f; };
    constexpr uint32_t qh = 24, kvh = 4, dim = 256, stride = 2 * dim;
    const uint32_t q_row_stride = qh * stride;
    const float scale = 1.0f / 16.0f;

    struct Case { uint32_t base, tokens; };
    for (const Case c : {Case{1020, 7}, Case{2040, 12}, Case{130, 1}}) {
        const uint32_t max_seq = c.base + c.tokens - 1;
        std::vector<float> q((uint64_t)c.tokens * q_row_stride);
        for (auto& value : q) value = uniform() * 1.5f;
        for (const bool turbo3 : {false, true}) {
            const uint64_t row_bytes = turbo3 ? (uint64_t)kvh * 2 * 50
                                              : (uint64_t)kvh * dim * 2;
            // Build the cache once (on the tiled backend's store kernels),
            // then attend over the same bytes on both backends.
            auto kc_t = tiled.allocate(max_seq * row_bytes);
            auto vc_t = tiled.allocate(max_seq * row_bytes);
            for (uint32_t p = 0; p < max_seq; p++) {
                std::vector<float> k(kvh * dim), v(kvh * dim);
                const float magnitude = (p % 3 == 0) ? 2.5f : 0.4f;
                for (auto& value : k) value = uniform() * magnitude;
                for (auto& value : v) value = uniform() * 3.0f;
                auto kb = upload_buffer(tiled, k), vb = upload_buffer(tiled, v);
                tiled.begin_commands();
                if (turbo3) tiled.kv_store_turbo3(*kb, *vb, *kc_t, *vc_t, p, kvh);
                else tiled.kv_store_f16(*kb, *vb, *kc_t, *vc_t, p, kvh * dim);
                tiled.end_commands();
            }
            std::vector<uint8_t> kbytes(max_seq * row_bytes), vbytes(max_seq * row_bytes);
            tiled.read(*kc_t, 0, kbytes.data(), kbytes.size());
            tiled.read(*vc_t, 0, vbytes.data(), vbytes.size());
            auto kc_u = untiled.allocate(kbytes.size()); untiled.write(*kc_u, 0, kbytes.data(), kbytes.size());
            auto vc_u = untiled.allocate(vbytes.size()); untiled.write(*vc_u, 0, vbytes.data(), vbytes.size());

            auto run = [&](q27::MetalBackend& backend, q27::BackendBuffer& kcache,
                           q27::BackendBuffer& vcache) {
                auto qb = upload_buffer(backend, q);
                auto out = backend.allocate((uint64_t)c.tokens * qh * dim * 4);
                backend.begin_commands();
                if (turbo3) backend.attention_turbo3_causal(*qb, stride, q_row_stride, kcache, vcache,
                                                            *out, c.base, qh, kvh, dim, c.tokens, scale);
                else backend.attention_f16_causal(*qb, stride, q_row_stride, kcache, vcache,
                                                  *out, c.base, qh, kvh, dim, c.tokens, scale);
                backend.end_commands();
                return read_f32(backend, *out, (uint64_t)c.tokens * qh * dim);
            };
            const auto got_tiled = run(tiled, *kc_t, *vc_t);
            const auto got_untiled = run(untiled, *kc_u, *vc_u);
            if (std::memcmp(got_tiled.data(), got_untiled.data(),
                            got_tiled.size() * sizeof(float)) != 0) {
                size_t i = 0;
                while (i < got_tiled.size() && got_tiled[i] == got_untiled[i]) i++;
                fprintf(stderr, "gqa tiled parity %s base=%u tokens=%u first diff at %zu: tiled %.9g untiled %.9g\n",
                        turbo3 ? "turbo3" : "f16", c.base, c.tokens, i, got_tiled[i], got_untiled[i]);
                failures++;
            }
        }
    }
    return failures;
}

// GPU top-k candidate extraction: the radix-select over-set must contain
// the exact top-k (value desc, index asc tie-break), stay within capacity
// on realistic logits, and signal fallback (count > capacity) on
// degenerate tie storms.
int test_topk(q27::MetalBackend& backend) {
    int failures = 0;
    uint32_t lcg = 13579;
    auto uniform = [&]() { lcg = lcg * 1664525u + 1013904223u; return (float)(lcg >> 8) / 8388608.0f - 1.0f; };
    constexpr uint32_t n = 151936, capacity = 1024;
    auto values_buffer = backend.allocate(capacity * 4);
    auto indices_buffer = backend.allocate(capacity * 4);
    auto count_buffer = backend.allocate(4);

    auto check = [&](const std::vector<float>& logits, uint32_t k, const char* label) {
        auto lb = upload_buffer(backend, logits);
        backend.topk(*lb, (uint32_t)logits.size(), k, *values_buffer, *indices_buffer, *count_buffer);
        uint32_t count = 0;
        backend.read(*count_buffer, 0, &count, 4);
        if (count < k) { fprintf(stderr, "topk %s: count %u < k %u\n", label, count, k); return ++failures, void(); }
        if (count > capacity) { fprintf(stderr, "topk %s: unexpected overflow (%u)\n", label, count); return ++failures, void(); }
        std::vector<uint32_t> got_indices(count);
        backend.read(*indices_buffer, 0, got_indices.data(), count * 4);
        std::vector<uint32_t> order(logits.size());
        for (uint32_t i = 0; i < order.size(); i++) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](uint32_t a, uint32_t b) {
                              return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
                          });
        std::vector<bool> present(logits.size(), false);
        for (uint32_t index : got_indices) {
            if (index >= logits.size()) { fprintf(stderr, "topk %s: index out of range\n", label); failures++; return; }
            present[index] = true;
        }
        for (uint32_t i = 0; i < k; i++)
            if (!present[order[i]]) {
                fprintf(stderr, "topk %s: missing rank %u (index %u, value %.8g)\n",
                        label, i, order[i], logits[order[i]]);
                failures++;
                return;
            }
    };

    std::vector<float> logits(n);
    for (auto& value : logits) value = uniform() * 12.0f;
    check(logits, 1, "k=1");
    check(logits, 40, "k=40");
    check(logits, 256, "k=256");
    // Quantized logits: ~600-way ties at every distinct value, so the
    // boundary bucket is fat but still under capacity.
    std::vector<float> tied(n);
    for (uint32_t i = 0; i < n; i++) tied[i] = std::round(logits[i] * 10.0f) / 10.0f;
    check(tied, 40, "tied");
    // Degenerate: every logit equal -> over-set is the whole vocabulary and
    // the count must signal fallback.
    std::vector<float> flat(n, 1.5f);
    auto fb = upload_buffer(backend, flat);
    backend.topk(*fb, n, 40, *values_buffer, *indices_buffer, *count_buffer);
    uint32_t count = 0;
    backend.read(*count_buffer, 0, &count, 4);
    if (count <= capacity) { fprintf(stderr, "topk flat: count %u did not signal fallback\n", count); failures++; }
    // All-(-inf) (fully grammar-masked logits): every key lands in one
    // 16-bit bucket, the over-set is the whole vocabulary, and the count
    // must signal fallback exactly like the flat tie storm.
    std::vector<float> ninf(n, -INFINITY);
    auto nb = upload_buffer(backend, ninf);
    backend.topk(*nb, n, 40, *values_buffer, *indices_buffer, *count_buffer);
    backend.read(*count_buffer, 0, &count, 4);
    if (count <= capacity) { fprintf(stderr, "topk -inf: count %u did not signal fallback\n", count); failures++; }
    // n not a multiple of the 1024-thread dispatch (strided tail coverage).
    std::vector<float> odd(logits.begin(), logits.begin() + (n - 77));
    check(odd, 40, "odd-n");
    // Boundary-exact shape (audit A2's count==k-1 hazard): exactly k
    // separated high values, everything else far below — the boundary bin
    // must supply exactly the remaining candidates and count stays >= k
    // (asserted inside check).
    std::vector<float> boundary(n, -50.0f);
    for (uint32_t i = 0; i < 40; i++) boundary[(i * 3797u + 11u) % n] = 100.0f + (float)i;
    check(boundary, 40, "boundary-exact");
    return failures;
}

// Deterministic argmax over degenerate inputs: all-(-inf) (fully masked),
// all-tie, and a non-multiple-of-256 n — first-past-the-post must stay the
// lowest index, and strided tails must not drop the winner.
int test_argmax_stress(q27::MetalBackend& backend) {
    int failures = 0;
    auto expect = [&](const std::vector<float>& logits, uint32_t want, const char* label) {
        auto lb = upload_buffer(backend, logits);
        auto ib = backend.allocate(4);
        backend.argmax(*lb, (uint32_t)logits.size(), *ib);
        uint32_t got = 0xffffffffu;
        backend.read(*ib, 0, &got, 4);
        if (got != want) { fprintf(stderr, "argmax %s: got %u want %u\n", label, got, want); failures++; }
    };
    expect(std::vector<float>(1000, -INFINITY), 0, "all -inf");
    expect(std::vector<float>(1001, 1.5f), 0, "all tie");
    std::vector<float> tail(777);
    for (uint32_t i = 0; i < tail.size(); i++) tail[i] = (float)(i % 7);
    tail[776] = 100.0f;
    expect(tail, 776, "odd-n max at tail");
    return failures;
}

// E1 (metal-review 2026-07-17): bind-time checks bound a tensor by its
// LOGICAL extent, not its buffer — on a whole-mapping shared buffer the
// buffer size alone would let a corrupt header read the neighbor tensor.
// The negative arms prove the check can fail; the controls prove it is
// inert for well-formed extents.
int test_tensor_extent(q27::MetalBackend& backend) {
    int failures = 0;
    constexpr uint32_t n = 256;
    // "Mapping" twice the tensor's size: the neighbor's bytes live behind it.
    auto shared = backend.allocate((uint64_t)n * 4 * 2);
    std::vector<float> w(n, 1.0f);
    backend.write(*shared, 0, w.data(), n * 4);
    auto x = upload_buffer(backend, w);
    auto y = backend.allocate((uint64_t)n * 4);
    q27::BackendTensor weight;
    weight.dtype = q27::DType::F32;
    weight.rows = 1; weight.cols = n;
    weight.data = shared;
    weight.data_size = (uint64_t)n * 4;
    try { backend.rmsnorm(*x, weight, *y, n, 1e-6f); }
    catch (const std::exception& e) { fprintf(stderr, "extent control: %s\n", e.what()); failures++; }
    weight.data_size = (uint64_t)n * 4 - 4;   // declared extent one float short
    bool threw = false;
    try { backend.rmsnorm(*x, weight, *y, n, 1e-6f); }
    catch (const std::exception&) { threw = true; }
    if (!threw) { fprintf(stderr, "extent: short data_size did not throw\n"); failures++; }
    // Scales side, via the embedding path (Q8 scales one entry short).
    constexpr uint32_t rows = 4, cols = 128;
    std::vector<uint8_t> qdata((size_t)rows * cols, 1);
    auto qbuf = backend.allocate(qdata.size());
    backend.write(*qbuf, 0, qdata.data(), qdata.size());
    auto sbuf = backend.allocate(64);   // roomy buffer; logical extent is what must bind
    std::vector<uint16_t> qscales(rows, 0x3c00);
    backend.write(*sbuf, 0, qscales.data(), rows * 2);
    auto out = backend.allocate((uint64_t)cols * 4);
    q27::BackendTensor emb;
    emb.dtype = q27::DType::Q8_G128;
    emb.rows = rows; emb.cols = cols;
    emb.data = qbuf; emb.scales = sbuf;
    emb.data_size = qdata.size();
    emb.scales_size = (uint64_t)rows * 2;
    try { backend.embedding_q8(emb, 0, *out); }
    catch (const std::exception& e) { fprintf(stderr, "extent scales control: %s\n", e.what()); failures++; }
    emb.scales_size = (uint64_t)rows * 2 - 2;   // one fp16 scale short
    threw = false;
    try { backend.embedding_q8(emb, 0, *out); }
    catch (const std::exception&) { threw = true; }
    if (!threw) { fprintf(stderr, "extent: short scales_size did not throw\n"); failures++; }
    return failures;
}

// Constrained-decoding mask: -inf where the bitset bit is clear, exact
// passthrough where set; a masked argmax must pick the best LEGAL token.
int test_mask_logits(q27::MetalBackend& backend) {
    constexpr uint32_t n = 1000;   // odd tail: exercises the last partial word
    std::vector<float> logits(n);
    for (uint32_t i = 0; i < n; i++) logits[i] = (float)((i * 37) % 501) - 250.0f;
    std::vector<uint32_t> mask((n + 31) / 32, 0);
    for (uint32_t i = 0; i < n; i += 3) mask[i >> 5] |= 1u << (i & 31);  // every 3rd legal
    auto lb = backend.allocate(n * 4);
    auto mb = backend.allocate(mask.size() * 4);
    backend.write(*lb, 0, logits.data(), n * 4);
    backend.write(*mb, 0, mask.data(), mask.size() * 4);
    backend.mask_logits(*lb, *mb, 0, n);
    std::vector<float> got(n);
    backend.read(*lb, 0, got.data(), n * 4);
    uint32_t best_legal = 0;
    for (uint32_t i = 0; i < n; i++) {
        const bool legal = (mask[i >> 5] >> (i & 31)) & 1u;
        if (legal && got[i] != logits[i]) {
            fprintf(stderr, "mask_logits: legal %u changed (%g vs %g)\n", i, got[i], logits[i]);
            return 1;
        }
        if (!legal && !(got[i] == -INFINITY)) {
            fprintf(stderr, "mask_logits: illegal %u not -inf (%g)\n", i, got[i]);
            return 1;
        }
        if (legal && logits[i] > logits[best_legal]) best_legal = i;
    }
    auto idx = backend.allocate(4);
    backend.argmax(*lb, n, *idx);
    uint32_t picked = 0;
    backend.read(*idx, 0, &picked, 4);
    if (picked != best_legal) {
        fprintf(stderr, "mask_logits: argmax picked %u, best legal %u\n", picked, best_legal);
        return 1;
    }
    return 0;
}

int test_gdn(q27::MetalBackend& backend) {
    int failures=0;
    // Gates.
    std::vector<float> alpha={-.2f,.3f}, br={-1,2}, av={-.5f,-.25f}, dtv={.1f,-.2f};
    auto ab=upload_buffer(backend,alpha), brb=upload_buffer(backend,br), go=backend.allocate(8), bo=backend.allocate(8);
    auto at=upload_f32(backend,av,{2}), dtt=upload_f32(backend,dtv,{2}); backend.gdn_gates(*ab,*brb,at,dtt,*go,*bo,2);
    auto gg=read_f32(backend,*go,2), bg=read_f32(backend,*bo,2); for(int i=0;i<2;i++){float z=alpha[i]+dtv[i],sp=z>20?z:std::log1p(std::exp(z)),gw=av[i]*sp,bw=1/(1+std::exp(-br[i]));if(!near(gg[i],gw)||!near(bg[i],bw)){fprintf(stderr,"gates[%d] mismatch\n",i);failures++;}}

    // Convolution ring update.
    constexpr uint32_t channels=7; std::vector<float> ring(channels*3),qkv(channels),cw(channels*4),conv_want(channels),ring_want(ring.size());
    for(size_t i=0;i<ring.size();i++)ring[i]=(float)((int)i-5)*.03f;for(uint32_t i=0;i<channels;i++){qkv[i]=(int(i)-2)*.1f;for(int t=0;t<4;t++)cw[i*4+t]=.05f*(t+1);float z=ring[i]*cw[i*4]+ring[channels+i]*cw[i*4+1]+ring[2*channels+i]*cw[i*4+2]+qkv[i]*cw[i*4+3];conv_want[i]=z/(1+std::exp(-z));ring_want[i]=ring[channels+i];ring_want[channels+i]=ring[2*channels+i];ring_want[2*channels+i]=qkv[i];}
    auto ringb=upload_buffer(backend,ring),qkvb=upload_buffer(backend,qkv),co=backend.allocate(channels*4);auto cwt=upload_f32(backend,cw,{channels,4});backend.conv_step(*ringb,*ringb,*qkvb,cwt,*co,channels);
    auto cg=read_f32(backend,*co,channels),rr=read_f32(backend,*ringb,ring.size());for(size_t i=0;i<cg.size();i++)if(!near(cg[i],conv_want[i]))failures++;for(size_t i=0;i<rr.size();i++)if(!near(rr[i],ring_want[i]))failures++;

    // Full 128x128 DeltaNet recurrence on three value heads.
    constexpr uint32_t vh=3,qkh=16,hd=128; const size_t state_n=(size_t)vh*hd*hd,conv_n=(qkh*2+vh)*hd;
    std::vector<float> state(state_n), cv(conv_n), decay={-.04f,-.02f,-.06f}, beta={.3f,.7f,.5f};
    for(size_t i=0;i<state.size();i++)state[i]=(int(i%17)-8)*.0005f;for(size_t i=0;i<cv.size();i++)cv[i]=(int(i%23)-11)*.01f;
    std::vector<float> state_want(state_n), out_want(vh*hd);
    for(uint32_t h=0;h<vh;h++){uint32_t qk=h%qkh;float de=std::exp(decay[h]);for(uint32_t j=0;j<hd;j++){float pred=0;for(uint32_t i=0;i<hd;i++)pred+=cv[2048+qk*hd+i]*(state[((size_t)h*hd+i)*hd+j]*de);float dv=beta[h]*(cv[4096+h*hd+j]-pred),ov=0;for(uint32_t i=0;i<hd;i++){float sv=state[((size_t)h*hd+i)*hd+j]*de+cv[2048+qk*hd+i]*dv;state_want[((size_t)h*hd+i)*hd+j]=sv;ov+=cv[qk*hd+i]/std::sqrt(128.0f)*sv;}out_want[h*hd+j]=ov;}}
    auto sb=upload_buffer(backend,state), cvb=upload_buffer(backend,cv), db=upload_buffer(backend,decay), betab=upload_buffer(backend,beta), dout=backend.allocate(vh*hd*4);
    backend.delta_step(*sb,*sb,*cvb,*db,*betab,*dout,vh,qkh,hd);auto sg=read_f32(backend,*sb,state_n),og=read_f32(backend,*dout,vh*hd);
    for(size_t i=0;i<sg.size();i++)if(!near(sg[i],state_want[i],2e-3f)){fprintf(stderr,"delta state[%zu] mismatch %.8g %.8g\n",i,sg[i],state_want[i]);failures++;break;}
    for(size_t i=0;i<og.size();i++)if(!near(og[i],out_want[i],2e-3f)){fprintf(stderr,"delta out[%zu] mismatch %.8g %.8g\n",i,og[i],out_want[i]);failures++;break;}

    std::vector<float> nw(hd,1.1f), gate(vh*hd);for(size_t i=0;i<gate.size();i++)gate[i]=(int(i%13)-6)*.1f;
    auto nwt=upload_f32(backend,nw,{hd}); auto gateb=upload_buffer(backend,gate); auto normout=backend.allocate(vh*hd*4);
    backend.gated_norm_gdn(*dout,nwt,*gateb,*normout,vh,hd,1e-6f);auto ng=read_f32(backend,*normout,vh*hd);
    for(uint32_t h=0;h<vh;h++){float ss=0;for(uint32_t d=0;d<hd;d++)ss+=out_want[h*hd+d]*out_want[h*hd+d];float ni=1/std::sqrt(ss/hd+1e-6f);for(uint32_t d=0;d<hd;d++){size_t i=h*hd+d;float want=out_want[i]*ni*nw[d]*(gate[i]/(1+std::exp(-gate[i])));if(!near(ng[i],want,2e-3f)){fprintf(stderr,"gated norm[%zu] mismatch\n",i);failures++;break;}}}
    return failures;
}

uint16_t to_half(float value) {
    _Float16 h = (_Float16)value;
    uint16_t bits; __builtin_memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Every chunked layer-major operation must reproduce the token-serial path
// it replaces: the serial kernels are the validated reference.
int test_chunked(q27::MetalBackend& backend) {
    int failures = 0;
    constexpr uint32_t T = 3;
    auto fail = [&](const char* name, size_t i, float got, float want) {
        fprintf(stderr, "chunked %s[%zu]: got %.8g want %.8g\n", name, i, got, want);
        failures++;
    };
    auto row_view = [&](const q27::BackendBuffer& src, size_t row, size_t floats) {
        auto tmp = backend.allocate(floats * 4);
        backend.copy(src, row * floats * 4, *tmp, 0, floats * 4);
        return tmp;
    };

    // Chunked Q8 embedding.
    {
        constexpr uint32_t vocab = 4, cols = 128;
        std::vector<int8_t> ew(vocab * cols);
        for (size_t i = 0; i < ew.size(); i++) ew[i] = (int8_t)((int)(i % 29) - 14);
        std::vector<uint16_t> es = {0x3c00, 0x3800, 0x4000, 0x3400};
        q27::Tensor et; et.name="embedding"; et.dtype=q27::DType::Q8_G128; et.shape={vocab,cols};
        et.data=(const uint8_t*)ew.data(); et.data_size=ew.size();
        et.scales=(const uint8_t*)es.data(); et.scales_size=es.size()*2;
        auto weight = backend.upload(et);
        const uint32_t tokens[T] = {2, 0, 3};
        auto chunk = backend.allocate((uint64_t)T * cols * 4);
        backend.embedding_q8_rows(weight, tokens, T, *chunk);
        auto serial = backend.allocate(cols * 4);
        for (uint32_t t = 0; t < T; t++) {
            backend.embedding_q8(weight, tokens[t], *serial);
            auto got = read_f32(backend, *chunk, (size_t)T * cols);
            auto want = read_f32(backend, *serial, cols);
            for (uint32_t i = 0; i < cols; i++)
                if (got[t * cols + i] != want[i]) { fail("embedding", t * cols + i, got[t*cols+i], want[i]); break; }
        }
    }

    // Chunked fused RMSNorm + quantization.
    {
        constexpr uint32_t n = 64;
        std::vector<float> x(T * n), w(n);
        for (size_t i = 0; i < x.size(); i++) x[i] = std::sin(float(i) * .37f) * (1.0f + float(i % 5));
        for (size_t i = 0; i < n; i++) w[i] = 0.5f + float(i % 3);
        auto xb = upload_buffer(backend, x); auto wt = upload_f32(backend, w, {n});
        auto chunk_out = backend.allocate((uint64_t)T * n * 4);
        auto chunk_q = backend.allocate_quantized(T * n);
        backend.rmsnorm_rows_quantized(*xb, wt, *chunk_out, n, T, 1e-6f, chunk_q);
        std::vector<int8_t> chunk_values(T * n); std::vector<float> chunk_scales(T * n / 32);
        backend.read(*chunk_q.values, 0, chunk_values.data(), chunk_values.size());
        backend.read(*chunk_q.scales, 0, chunk_scales.data(), chunk_scales.size() * 4);
        auto chunk_f = read_f32(backend, *chunk_out, (size_t)T * n);
        auto serial_out = backend.allocate(n * 4); auto serial_q = backend.allocate_quantized(n);
        for (uint32_t t = 0; t < T; t++) {
            auto row = row_view(*xb, t, n);
            backend.rmsnorm_quantized(*row, wt, *serial_out, n, 1e-6f, serial_q);
            auto want_f = read_f32(backend, *serial_out, n);
            std::vector<int8_t> want_values(n); std::vector<float> want_scales(n / 32);
            backend.read(*serial_q.values, 0, want_values.data(), n);
            backend.read(*serial_q.scales, 0, want_scales.data(), want_scales.size() * 4);
            for (uint32_t i = 0; i < n; i++)
                if (chunk_f[t*n+i] != want_f[i] || chunk_values[t*n+i] != want_values[i])
                    { fail("rmsnorm rows", t*n+i, chunk_f[t*n+i], want_f[i]); break; }
            for (uint32_t b = 0; b < n / 32; b++)
                if (chunk_scales[t*(n/32)+b] != want_scales[b]) { fail("rmsnorm rows scale", t*(n/32)+b, chunk_scales[t*(n/32)+b], want_scales[b]); break; }
        }
    }

    // Chunked F16 projection pair.
    {
        constexpr uint32_t rows_a = 5, rows_b = 3, cols = 32;
        std::vector<uint16_t> wa(rows_a * cols), wb(rows_b * cols);
        for (size_t i = 0; i < wa.size(); i++) wa[i] = to_half(((int)(i % 11) - 5) * 0.25f);
        for (size_t i = 0; i < wb.size(); i++) wb[i] = to_half(((int)(i % 9) - 4) * 0.5f);
        q27::Tensor ta; ta.name="pair-a"; ta.dtype=q27::DType::F16; ta.shape={rows_a,cols};
        ta.data=(const uint8_t*)wa.data(); ta.data_size=wa.size()*2;
        q27::Tensor tb; tb.name="pair-b"; tb.dtype=q27::DType::F16; tb.shape={rows_b,cols};
        tb.data=(const uint8_t*)wb.data(); tb.data_size=wb.size()*2;
        auto weight_a = backend.upload(ta); auto weight_b = backend.upload(tb);
        std::vector<float> x(T * cols);
        for (size_t i = 0; i < x.size(); i++) x[i] = std::cos(float(i) * .21f);
        auto xb = upload_buffer(backend, x);
        auto chunk_a = backend.allocate((uint64_t)T * rows_a * 4);
        auto chunk_b = backend.allocate((uint64_t)T * rows_b * 4);
        backend.matvec_f16_pair_rows(weight_a, *chunk_a, weight_b, *chunk_b, *xb, T);
        auto got_a = read_f32(backend, *chunk_a, (size_t)T * rows_a);
        auto got_b = read_f32(backend, *chunk_b, (size_t)T * rows_b);
        auto serial_a = backend.allocate(rows_a * 4); auto serial_b = backend.allocate(rows_b * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto row = row_view(*xb, t, cols);
            backend.matvec_pair(weight_a, *serial_a, weight_b, *serial_b, *row);
            auto want_a = read_f32(backend, *serial_a, rows_a);
            auto want_b = read_f32(backend, *serial_b, rows_b);
            for (uint32_t i = 0; i < rows_a; i++)
                if (got_a[t*rows_a+i] != want_a[i]) { fail("pair rows A", t*rows_a+i, got_a[t*rows_a+i], want_a[i]); break; }
            for (uint32_t i = 0; i < rows_b; i++)
                if (got_b[t*rows_b+i] != want_b[i]) { fail("pair rows B", t*rows_b+i, got_b[t*rows_b+i], want_b[i]); break; }
        }
    }

    // Chunked GDN gates.
    {
        constexpr uint32_t heads = 4;
        std::vector<float> alpha(T * heads), braw(T * heads), av(heads), dtv(heads);
        for (size_t i = 0; i < alpha.size(); i++) { alpha[i] = std::sin(float(i)) * 2; braw[i] = std::cos(float(i)); }
        for (size_t i = 0; i < heads; i++) { av[i] = -0.1f - 0.2f * float(i); dtv[i] = 0.3f - 0.1f * float(i); }
        auto ab = upload_buffer(backend, alpha); auto bb = upload_buffer(backend, braw);
        auto at = upload_f32(backend, av, {heads}); auto dtt = upload_f32(backend, dtv, {heads});
        auto cgo = backend.allocate((uint64_t)T * heads * 4); auto cbo = backend.allocate((uint64_t)T * heads * 4);
        backend.gdn_gates_rows(*ab, *bb, at, dtt, *cgo, *cbo, heads, T);
        auto got_g = read_f32(backend, *cgo, T * heads); auto got_b = read_f32(backend, *cbo, T * heads);
        auto sgo = backend.allocate(heads * 4); auto sbo = backend.allocate(heads * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto arow = row_view(*ab, t, heads); auto brow = row_view(*bb, t, heads);
            backend.gdn_gates(*arow, *brow, at, dtt, *sgo, *sbo, heads);
            auto want_g = read_f32(backend, *sgo, heads); auto want_b = read_f32(backend, *sbo, heads);
            for (uint32_t i = 0; i < heads; i++)
                if (got_g[t*heads+i] != want_g[i] || got_b[t*heads+i] != want_b[i])
                    { fail("gates rows", t*heads+i, got_g[t*heads+i], want_g[i]); break; }
        }
    }

    // Chunked convolution ring: one dispatch must match T serial steps and
    // leave the identical ring state.
    {
        constexpr uint32_t channels = 7;
        std::vector<float> ring(channels * 3), qkv(T * channels), cw(channels * 4);
        for (size_t i = 0; i < ring.size(); i++) ring[i] = std::sin(float(i) * .61f);
        for (size_t i = 0; i < qkv.size(); i++) qkv[i] = std::cos(float(i) * .43f);
        for (size_t i = 0; i < cw.size(); i++) cw[i] = 0.05f * float((int)(i % 9) - 4);
        auto cwt = upload_f32(backend, cw, {channels, 4});
        auto ring_serial = upload_buffer(backend, ring); auto ring_chunk = upload_buffer(backend, ring);
        auto qkvb = upload_buffer(backend, qkv);
        auto chunk_out = backend.allocate((uint64_t)T * channels * 4);
        backend.conv_chunk(*ring_chunk, *ring_chunk, *qkvb, cwt, *chunk_out, channels, T);
        auto got = read_f32(backend, *chunk_out, (size_t)T * channels);
        auto serial_out = backend.allocate(channels * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto row = row_view(*qkvb, t, channels);
            backend.conv_step(*ring_serial, *ring_serial, *row, cwt, *serial_out, channels);
            auto want = read_f32(backend, *serial_out, channels);
            for (uint32_t i = 0; i < channels; i++)
                if (got[t*channels+i] != want[i]) { fail("conv chunk", t*channels+i, got[t*channels+i], want[i]); break; }
        }
        auto ring_a = read_f32(backend, *ring_serial, ring.size());
        auto ring_b = read_f32(backend, *ring_chunk, ring.size());
        for (size_t i = 0; i < ring.size(); i++)
            if (ring_a[i] != ring_b[i]) { fail("conv chunk ring", i, ring_b[i], ring_a[i]); break; }
    }

    // Chunked DeltaNet recurrence: register-resident chunk state must match
    // T serial device round trips.
    {
        constexpr uint32_t vh = 3, qkh = 16, hd = 128;
        const size_t state_n = (size_t)vh * hd * hd, conv_row = (qkh * 2 + vh) * hd;
        std::vector<float> state(state_n), cv(T * conv_row), g(T * vh), beta(T * vh);
        for (size_t i = 0; i < state.size(); i++) state[i] = (int(i % 17) - 8) * .0005f;
        for (size_t i = 0; i < cv.size(); i++) cv[i] = (int(i % 23) - 11) * .01f;
        for (size_t i = 0; i < g.size(); i++) g[i] = -.01f - .002f * float(i % 7);
        for (size_t i = 0; i < beta.size(); i++) beta[i] = .2f + .05f * float(i % 9);
        auto state_serial = upload_buffer(backend, state); auto state_chunk = upload_buffer(backend, state);
        auto cvb = upload_buffer(backend, cv); auto gb = upload_buffer(backend, g); auto betab = upload_buffer(backend, beta);
        auto chunk_out = backend.allocate((uint64_t)T * vh * hd * 4);
        backend.delta_chunk(*state_chunk, *state_chunk, *cvb, *gb, *betab, *chunk_out, vh, qkh, hd, T);
        auto got = read_f32(backend, *chunk_out, (size_t)T * vh * hd);
        auto serial_out = backend.allocate((uint64_t)vh * hd * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto cv_row = row_view(*cvb, t, conv_row);
            auto g_row = row_view(*gb, t, vh); auto beta_row = row_view(*betab, t, vh);
            backend.delta_step(*state_serial, *state_serial, *cv_row, *g_row, *beta_row, *serial_out, vh, qkh, hd);
            auto want = read_f32(backend, *serial_out, (size_t)vh * hd);
            for (uint32_t i = 0; i < vh * hd; i++)
                if (!near(got[t*vh*hd+i], want[i], 1e-5f)) { fail("delta chunk", t*vh*hd+i, got[t*vh*hd+i], want[i]); break; }
        }
        auto sa = read_f32(backend, *state_serial, state_n);
        auto sb = read_f32(backend, *state_chunk, state_n);
        for (size_t i = 0; i < state_n; i++)
            if (!near(sa[i], sb[i], 1e-5f)) { fail("delta chunk state", i, sb[i], sa[i]); break; }
    }

    // MTP verification state discipline: a chunk writing state to a discard
    // slot must leave live state bit-untouched with identical outputs, and
    // replaying a K-token prefix afterwards (the acceptance path) must
    // bit-match a directly committed K-token chunk. This is the partial
    // acceptance contract that replaced the checkpoint/restore/re-encode.
    {
        constexpr uint32_t channels = 7, K = 2;
        static_assert(K < T, "replay must cover a strict prefix");
        std::vector<float> ring(channels * 3), qkv(T * channels), cw(channels * 4);
        for (size_t i = 0; i < ring.size(); i++) ring[i] = std::cos(float(i) * .29f);
        for (size_t i = 0; i < qkv.size(); i++) qkv[i] = std::sin(float(i) * .53f);
        for (size_t i = 0; i < cw.size(); i++) cw[i] = 0.04f * float((int)(i % 7) - 3);
        auto cwt = upload_f32(backend, cw, {channels, 4});
        auto ring_live = upload_buffer(backend, ring);
        auto ring_ref = upload_buffer(backend, ring);
        auto ring_discard = backend.allocate((uint64_t)channels * 3 * 4);
        auto qkvb = upload_buffer(backend, qkv);
        auto out_full = backend.allocate((uint64_t)T * channels * 4);
        auto out_prefix = backend.allocate((uint64_t)K * channels * 4);
        auto out_replay = backend.allocate((uint64_t)K * channels * 4);
        backend.conv_chunk(*ring_live, *ring_discard, *qkvb, cwt, *out_full, channels, T);
        auto untouched = read_f32(backend, *ring_live, ring.size());
        for (size_t i = 0; i < ring.size(); i++)
            if (untouched[i] != ring[i]) { fail("conv discard ring", i, untouched[i], ring[i]); break; }
        backend.conv_chunk(*ring_ref, *ring_ref, *qkvb, cwt, *out_prefix, channels, K);
        backend.conv_chunk(*ring_live, *ring_live, *qkvb, cwt, *out_replay, channels, K);
        auto want_ring = read_f32(backend, *ring_ref, ring.size());
        auto got_ring = read_f32(backend, *ring_live, ring.size());
        for (size_t i = 0; i < ring.size(); i++)
            if (got_ring[i] != want_ring[i]) { fail("conv replay ring", i, got_ring[i], want_ring[i]); break; }
        auto full = read_f32(backend, *out_full, (size_t)T * channels);
        auto prefix = read_f32(backend, *out_prefix, (size_t)K * channels);
        auto replay = read_f32(backend, *out_replay, (size_t)K * channels);
        for (size_t i = 0; i < (size_t)K * channels; i++) {
            if (full[i] != prefix[i]) { fail("conv prefix invariance", i, full[i], prefix[i]); break; }
            if (replay[i] != prefix[i]) { fail("conv replay out", i, replay[i], prefix[i]); break; }
        }
    }
    {
        constexpr uint32_t vh = 3, qkh = 16, hd = 128, K = 2;
        static_assert(K < T, "replay must cover a strict prefix");
        const size_t state_n = (size_t)vh * hd * hd, conv_row = (qkh * 2 + vh) * hd;
        std::vector<float> state(state_n), cv(T * conv_row), g(T * vh), beta(T * vh);
        for (size_t i = 0; i < state.size(); i++) state[i] = (int(i % 13) - 6) * .0007f;
        for (size_t i = 0; i < cv.size(); i++) cv[i] = (int(i % 19) - 9) * .012f;
        for (size_t i = 0; i < g.size(); i++) g[i] = -.02f - .003f * float(i % 5);
        for (size_t i = 0; i < beta.size(); i++) beta[i] = .15f + .04f * float(i % 7);
        auto state_live = upload_buffer(backend, state);
        auto state_ref = upload_buffer(backend, state);
        auto state_discard = backend.allocate(state_n * 4);
        auto cvb = upload_buffer(backend, cv);
        auto gb = upload_buffer(backend, g); auto betab = upload_buffer(backend, beta);
        auto out_full = backend.allocate((uint64_t)T * vh * hd * 4);
        auto out_prefix = backend.allocate((uint64_t)K * vh * hd * 4);
        auto out_replay = backend.allocate((uint64_t)K * vh * hd * 4);
        backend.delta_chunk(*state_live, *state_discard, *cvb, *gb, *betab, *out_full, vh, qkh, hd, T);
        auto untouched = read_f32(backend, *state_live, state_n);
        for (size_t i = 0; i < state_n; i++)
            if (untouched[i] != state[i]) { fail("delta discard state", i, untouched[i], state[i]); break; }
        backend.delta_chunk(*state_ref, *state_ref, *cvb, *gb, *betab, *out_prefix, vh, qkh, hd, K);
        backend.delta_chunk(*state_live, *state_live, *cvb, *gb, *betab, *out_replay, vh, qkh, hd, K);
        auto want_state = read_f32(backend, *state_ref, state_n);
        auto got_state = read_f32(backend, *state_live, state_n);
        for (size_t i = 0; i < state_n; i++)
            if (got_state[i] != want_state[i]) { fail("delta replay state", i, got_state[i], want_state[i]); break; }
        auto full = read_f32(backend, *out_full, (size_t)T * vh * hd);
        auto prefix = read_f32(backend, *out_prefix, (size_t)K * vh * hd);
        auto replay = read_f32(backend, *out_replay, (size_t)K * vh * hd);
        for (size_t i = 0; i < (size_t)K * vh * hd; i++) {
            if (full[i] != prefix[i]) { fail("delta prefix invariance", i, full[i], prefix[i]); break; }
            if (replay[i] != prefix[i]) { fail("delta replay out", i, replay[i], prefix[i]); break; }
        }
    }

    // Chunked strided L2 norm (row stride exceeds the normalized span).
    {
        constexpr uint32_t heads = 2, hd = 4, row_stride = 12;
        std::vector<float> x(T * row_stride);
        for (size_t i = 0; i < x.size(); i++) x[i] = std::sin(float(i) * .83f) + .2f;
        auto chunk = upload_buffer(backend, x);
        backend.l2norm_rows(*chunk, heads, hd, row_stride, T, 1e-6f);
        auto got = read_f32(backend, *chunk, x.size());
        for (uint32_t t = 0; t < T; t++) {
            std::vector<float> row(x.begin() + t * row_stride, x.begin() + t * row_stride + heads * hd);
            auto serial = upload_buffer(backend, row);
            backend.l2norm_heads(*serial, heads, hd, 1e-6f);
            auto want = read_f32(backend, *serial, row.size());
            for (uint32_t i = 0; i < heads * hd; i++)
                if (got[t*row_stride+i] != want[i]) { fail("l2 rows", t*row_stride+i, got[t*row_stride+i], want[i]); break; }
            for (uint32_t i = heads * hd; i < row_stride; i++)
                if (got[t*row_stride+i] != x[t*row_stride+i]) { fail("l2 rows tail", t*row_stride+i, got[t*row_stride+i], x[t*row_stride+i]); break; }
        }
    }

    // Chunked RoPE with per-token positions.
    {
        constexpr uint32_t heads = 2, hd = 8, n_rot = 4, stride = 8, row_stride = 16, base = 3;
        std::vector<float> x(T * row_stride);
        for (size_t i = 0; i < x.size(); i++) x[i] = std::cos(float(i) * .29f) * 2;
        auto chunk = upload_buffer(backend, x);
        backend.rope_neox_rows(*chunk, heads, hd, n_rot, stride, row_stride, base, T, 10000.0f);
        auto got = read_f32(backend, *chunk, x.size());
        for (uint32_t t = 0; t < T; t++) {
            std::vector<float> row(x.begin() + t * row_stride, x.begin() + (t + 1) * row_stride);
            auto serial = upload_buffer(backend, row);
            backend.rope_neox(*serial, heads, hd, n_rot, stride, base + t, 10000.0f);
            auto want = read_f32(backend, *serial, row.size());
            for (uint32_t i = 0; i < row_stride; i++)
                if (got[t*row_stride+i] != want[i]) { fail("rope rows", t*row_stride+i, got[t*row_stride+i], want[i]); break; }
        }
    }

    // Chunked sigmoid gating.
    {
        constexpr uint32_t heads = 2, hd = 2;
        std::vector<float> values(T * heads * hd), qg(T * heads * hd * 2);
        for (size_t i = 0; i < values.size(); i++) values[i] = float(i % 7) - 3;
        for (size_t i = 0; i < qg.size(); i++) qg[i] = std::sin(float(i) * 1.1f) * 2;
        auto chunk = upload_buffer(backend, values); auto qgb = upload_buffer(backend, qg);
        backend.sigmoid_gate_mul_rows(*chunk, *qgb, heads, hd, T);
        auto got = read_f32(backend, *chunk, values.size());
        for (uint32_t t = 0; t < T; t++) {
            std::vector<float> row(values.begin() + t * heads * hd, values.begin() + (t + 1) * heads * hd);
            std::vector<float> qg_row(qg.begin() + t * heads * hd * 2, qg.begin() + (t + 1) * heads * hd * 2);
            auto serial = upload_buffer(backend, row); auto serial_qg = upload_buffer(backend, qg_row);
            backend.sigmoid_gate_mul(*serial, *serial_qg, heads, hd);
            auto want = read_f32(backend, *serial, row.size());
            for (uint32_t i = 0; i < heads * hd; i++)
                if (got[t*heads*hd+i] != want[i]) { fail("sigmoid rows", t*heads*hd+i, got[t*heads*hd+i], want[i]); break; }
        }
    }

    // Chunked per-lane argmax, including the lower-index tie break.
    {
        constexpr uint32_t n = 300;
        std::vector<float> logits(T * n);
        for (size_t i = 0; i < logits.size(); i++) logits[i] = std::sin(float(i) * .7f) * 5;
        logits[0 * n + 37] = 100; logits[1 * n + 4] = 100; logits[1 * n + 250] = 100;
        auto lb = upload_buffer(backend, logits);
        auto rows_out = backend.allocate(T * 4);
        backend.argmax_rows(*lb, n, T, *rows_out);
        std::vector<uint32_t> got(T); backend.read(*rows_out, 0, got.data(), T * 4);
        auto serial_out = backend.allocate(4);
        for (uint32_t t = 0; t < T; t++) {
            auto row = row_view(*lb, t, n);
            backend.argmax(*row, n, *serial_out);
            uint32_t want = 0; backend.read(*serial_out, 0, &want, 4);
            if (got[t] != want) fail("argmax rows", t, (float)got[t], (float)want);
        }
    }

    // Teacher-forced NLL rows: logsumexp - target logit, including a wide
    // vocab-like width so the 256-thread reduction walks multiple strides.
    {
        constexpr uint32_t n = 2048;
        std::vector<float> logits(T * n);
        std::vector<uint32_t> targets(T);
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t v = 0; v < n; v++)
                logits[t * n + v] = std::sin(float(t * 17 + v) * 0.11f) * 3.0f;
            targets[t] = 100 + t * 37;
            logits[t * n + targets[t]] += 2.5f;  // make the target distinctive
        }
        auto lb = upload_buffer(backend, logits);
        auto tb = backend.allocate(T * 4);
        backend.write(*tb, 0, targets.data(), T * 4);
        auto nb = backend.allocate(T * 4);
        backend.nll_rows(*lb, *tb, *nb, n, T);
        std::vector<float> got(T);
        backend.read(*nb, 0, got.data(), T * 4);
        for (uint32_t t = 0; t < T; t++) {
            double mx = -1e300;
            for (uint32_t v = 0; v < n; v++) mx = std::max(mx, (double)logits[t * n + v]);
            double se = 0.0;
            for (uint32_t v = 0; v < n; v++) se += std::exp((double)logits[t * n + v] - mx);
            const float want = (float)(std::log(se) + mx - (double)logits[t * n + targets[t]]);
            if (std::fabs(got[t] - want) > 1e-4f)
                fail("nll rows", t, got[t], want);
        }
    }

    // Chunked FP16 KV append + causal attention over a warm cache.
    {
        constexpr uint32_t qh = 2, kvh = 1, dim = 4, stride = 8, row = kvh * dim, warm = 2;
        auto kc = backend.allocate((uint64_t)(warm + T) * row * 2);
        auto vc = backend.allocate((uint64_t)(warm + T) * row * 2);
        for (uint32_t p = 0; p < warm; p++) {
            std::vector<float> k(row), v(row);
            for (uint32_t d = 0; d < row; d++) { k[d] = std::sin(float(p * row + d)); v[d] = std::cos(float(p * row + d)); }
            auto kb = upload_buffer(backend, k); auto vb = upload_buffer(backend, v);
            backend.kv_store_f16(*kb, *vb, *kc, *vc, p, row);
        }
        std::vector<float> knew(T * row), vnew(T * row), q(T * qh * stride);
        for (size_t i = 0; i < knew.size(); i++) { knew[i] = std::sin(float(i) * .53f); vnew[i] = std::cos(float(i) * .31f); }
        for (size_t i = 0; i < q.size(); i++) q[i] = std::sin(float(i) * .77f);
        auto knb = upload_buffer(backend, knew); auto vnb = upload_buffer(backend, vnew);
        backend.kv_store_f16_rows(*knb, *vnb, *kc, *vc, warm, row, T);
        auto qb = upload_buffer(backend, q);
        auto chunk_out = backend.allocate((uint64_t)T * qh * dim * 4);
        backend.attention_f16_causal(*qb, stride, qh * stride, *kc, *vc, *chunk_out,
                                     warm + 1, qh, kvh, dim, T, 0.5f);
        auto got = read_f32(backend, *chunk_out, (size_t)T * qh * dim);
        auto serial_out = backend.allocate((uint64_t)qh * dim * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto q_row = row_view(*qb, t, qh * stride);
            backend.attention_f16(*q_row, stride, *kc, *vc, *serial_out,
                                  warm + 1 + t, qh, kvh, dim, 0.5f);
            auto want = read_f32(backend, *serial_out, (size_t)qh * dim);
            for (uint32_t i = 0; i < qh * dim; i++)
                if (!near(got[t*qh*dim+i], want[i], 1e-5f)) { fail("causal attention", t*qh*dim+i, got[t*qh*dim+i], want[i]); break; }
        }
    }

    // Production-shape chunk-causal attention (24:4 GQA, head_dim 256). The
    // online-softmax chunk kernel mirrors the serial decode kernel exactly,
    // so the comparison is bit-exact. warm=2 leaves most simdgroup stripes
    // empty for the first chunk tokens; warm=130 exercises multi-stripe
    // running-max updates at the decode gate's seq length (133).
    for (uint32_t warm : {2u, 130u}) {
        constexpr uint32_t qh = 24, kvh = 4, dim = 256, stride = 512, row = kvh * dim;
        auto kc = backend.allocate((uint64_t)(warm + T) * row * 2);
        auto vc = backend.allocate((uint64_t)(warm + T) * row * 2);
        for (uint32_t p = 0; p < warm; p++) {
            std::vector<float> k(row), v(row);
            for (uint32_t d = 0; d < row; d++) {
                k[d] = std::sin(float(p * 37 + d) * .021f) * (1.0f + float(p % 5));
                v[d] = std::cos(float(p * 53 + d) * .013f);
            }
            auto kb = upload_buffer(backend, k); auto vb = upload_buffer(backend, v);
            backend.kv_store_f16(*kb, *vb, *kc, *vc, p, row);
        }
        std::vector<float> knew(T * row), vnew(T * row), q(T * qh * stride);
        for (size_t i = 0; i < knew.size(); i++) {
            knew[i] = std::sin(float(i) * .0047f) * (1.0f + float(i % 7));
            vnew[i] = std::cos(float(i) * .0031f);
        }
        for (size_t i = 0; i < q.size(); i++) q[i] = std::sin(float(i) * .0077f);
        auto knb = upload_buffer(backend, knew); auto vnb = upload_buffer(backend, vnew);
        backend.kv_store_f16_rows(*knb, *vnb, *kc, *vc, warm, row, T);
        auto qb = upload_buffer(backend, q);
        auto chunk_out = backend.allocate((uint64_t)T * qh * dim * 4);
        backend.attention_f16_causal(*qb, stride, qh * stride, *kc, *vc, *chunk_out,
                                     warm + 1, qh, kvh, dim, T, 0.0625f);
        auto got = read_f32(backend, *chunk_out, (size_t)T * qh * dim);
        auto serial_out = backend.allocate((uint64_t)qh * dim * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto q_row = row_view(*qb, t, qh * stride);
            backend.attention_f16(*q_row, stride, *kc, *vc, *serial_out,
                                  warm + 1 + t, qh, kvh, dim, 0.0625f);
            auto want = read_f32(backend, *serial_out, (size_t)qh * dim);
            for (uint32_t i = 0; i < qh * dim; i++)
                if (got[t*qh*dim+i] != want[i]) { fail("causal attention wide", t*qh*dim+i, got[t*qh*dim+i], want[i]); break; }
        }
    }

    // Chunked turbo3 KV append + causal attention.
    {
        constexpr uint32_t qh = 2, kvh = 1, dim = 256, stride = 256, warm = 2;
        const uint64_t row_bytes = (uint64_t)kvh * 2 * 50;
        auto kc = backend.allocate((warm + T) * row_bytes);
        auto vc = backend.allocate((warm + T) * row_bytes);
        auto kc_ref = backend.allocate((warm + T) * row_bytes);
        auto vc_ref = backend.allocate((warm + T) * row_bytes);
        for (uint32_t p = 0; p < warm; p++) {
            std::vector<float> k(kvh * dim), v(kvh * dim);
            for (uint32_t d = 0; d < kvh * dim; d++) { k[d] = std::sin(float(p * 331 + d) * .05f); v[d] = std::cos(float(p * 173 + d) * .07f); }
            auto kb = upload_buffer(backend, k); auto vb = upload_buffer(backend, v);
            backend.kv_store_turbo3(*kb, *vb, *kc, *vc, p, kvh);
            backend.kv_store_turbo3(*kb, *vb, *kc_ref, *vc_ref, p, kvh);
        }
        std::vector<float> knew(T * kvh * dim), vnew(T * kvh * dim), q(T * qh * stride);
        for (size_t i = 0; i < knew.size(); i++) { knew[i] = std::sin(float(i) * .011f); vnew[i] = std::cos(float(i) * .017f); }
        for (size_t i = 0; i < q.size(); i++) q[i] = std::sin(float(i) * .013f);
        auto knb = upload_buffer(backend, knew); auto vnb = upload_buffer(backend, vnew);
        backend.kv_store_turbo3_rows(*knb, *vnb, *kc, *vc, warm, kvh, T);
        for (uint32_t t = 0; t < T; t++) {
            auto k_row = row_view(*knb, t, kvh * dim); auto v_row = row_view(*vnb, t, kvh * dim);
            backend.kv_store_turbo3(*k_row, *v_row, *kc_ref, *vc_ref, warm + t, kvh);
        }
        std::vector<uint8_t> cache_a((warm + T) * row_bytes), cache_b(cache_a.size());
        backend.read(*kc, 0, cache_a.data(), cache_a.size());
        backend.read(*kc_ref, 0, cache_b.data(), cache_b.size());
        for (size_t i = 0; i < cache_a.size(); i++)
            if (cache_a[i] != cache_b[i]) { fail("turbo3 store rows K", i, cache_a[i], cache_b[i]); break; }
        backend.read(*vc, 0, cache_a.data(), cache_a.size());
        backend.read(*vc_ref, 0, cache_b.data(), cache_b.size());
        for (size_t i = 0; i < cache_a.size(); i++)
            if (cache_a[i] != cache_b[i]) { fail("turbo3 store rows V", i, cache_a[i], cache_b[i]); break; }

        auto qb = upload_buffer(backend, q);
        backend.turbo_wht(*qb, T * qh, stride, false);
        auto chunk_out = backend.allocate((uint64_t)T * qh * dim * 4);
        backend.attention_turbo3_causal(*qb, stride, qh * stride, *kc, *vc, *chunk_out,
                                        warm + 1, qh, kvh, dim, T, 1.0f / std::sqrt(float(dim)));
        auto got = read_f32(backend, *chunk_out, (size_t)T * qh * dim);
        auto serial_out = backend.allocate((uint64_t)qh * dim * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto q_row = row_view(*qb, t, qh * stride);
            backend.attention_turbo3(*q_row, stride, *kc, *vc, *serial_out,
                                     warm + 1 + t, qh, kvh, dim, 1.0f / std::sqrt(float(dim)));
            auto want = read_f32(backend, *serial_out, (size_t)qh * dim);
            // The chunk kernel mirrors the decode kernel's online-softmax
            // structure exactly, so the comparison is bit-exact.
            for (uint32_t i = 0; i < qh * dim; i++)
                if (got[t*qh*dim+i] != want[i]) { fail("turbo3 causal", t*qh*dim+i, got[t*qh*dim+i], want[i]); break; }
        }
    }

    return failures;
}

} // namespace

int main() {
    try {
        q27::MetalBackend backend;
        int failures = test_primitives(backend) + test_attention(backend) +
                       test_attention_production_shape(backend) +
                       test_turbo3(backend) + test_turbo3_production_shape(backend) +
                       test_attention_gqa_path() + test_attention_gqa_straddle() +
                       test_attention_gqa_tiled_parity() +
                       test_topk(backend) + test_argmax_stress(backend) +
                       test_tensor_extent(backend) + test_mask_logits(backend) +
                       test_gdn(backend) + test_chunked(backend);
        if (failures) { fprintf(stderr, "Metal ops: %d failure(s)\n", failures); return 1; }
        puts("Metal decode primitives, FP16/turbo3 attention (incl. GQA KV-reuse path), GDN, and chunked prefill ops: OK");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
