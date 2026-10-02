#include "metal_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <exception>
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

    // GPU radix selection must preserve CPU float equality for signed zero:
    // top_k=1 needs both candidates so the host can apply lower-index tie order.
    if(backend.gpu_topk_supported()) {
        std::vector<float> zero_logits={-0.0f,0.0f,-1.0f};
        auto zero_in=upload_buffer(backend,zero_logits);
        auto top_values=backend.allocate(8*4), top_indices=backend.allocate(8*4);
        auto top_count=backend.allocate(4);
        backend.topk(*zero_in,3,1,*top_values,*top_indices,*top_count);
        uint32_t count=0; backend.read(*top_count,0,&count,4);
        std::vector<uint32_t> ids(count);
        if(count) backend.read(*top_indices,0,ids.data(),count*4);
        const bool saw_negative=std::find(ids.begin(),ids.end(),0)!=ids.end();
        const bool saw_positive=std::find(ids.begin(),ids.end(),1)!=ids.end();
        if(!saw_negative || !saw_positive) {
            fprintf(stderr,"top-k signed-zero tie lost a candidate (count=%u)\n",count);
            failures++;
        }
    }

    return failures;
}

int test_attention(q27::MetalBackend& backend) {
    constexpr uint32_t seq=3,qh=2,kvh=1,dim=4,stride=8;
    auto kc=backend.allocate(seq*kvh*dim*2), vc=backend.allocate(seq*kvh*dim*2);
    const std::vector<std::vector<float>> keys={{1,0,0,0},{0,1,0,0},{1,1,0,0}};
    const std::vector<std::vector<float>> vals={{1,2,3,4},{2,0,1,3},{4,3,2,1}};
    for(uint32_t p=0;p<seq;p++){auto k=upload_buffer(backend,keys[p]),v=upload_buffer(backend,vals[p]);backend.kv_store(*k,*v,*kc,*vc,p,kvh,dim,q27::KvFormat::F16);}
    std::vector<float> q={1,.5f,0,0, 9,9,9,9, 0,1,0,0, 8,8,8,8}; auto qb=upload_buffer(backend,q);
    auto out=backend.allocate(qh*dim*4);
    backend.attention(*qb,stride,*kc,*vc,*out,seq,qh,kvh,dim,0.5f,q27::KvFormat::F16,nullptr); auto got=read_f32(backend,*out,qh*dim);
    int failures=0;
    for(uint32_t h=0;h<qh;h++){std::vector<float>s(seq);float mx=-1e30f;for(uint32_t p=0;p<seq;p++){s[p]=0;for(uint32_t d=0;d<dim;d++)s[p]+=q[h*stride+d]*keys[p][d];s[p]*=.5f;mx=std::max(mx,s[p]);}float den=0;for(float&v:s){v=std::exp(v-mx);den+=v;}for(float&v:s)v/=den;for(uint32_t d=0;d<dim;d++){float want=0;for(uint32_t p=0;p<seq;p++)want+=s[p]*vals[p][d];if(!near(got[h*dim+d],want,8e-4f)){fprintf(stderr,"attention[%u,%u] got %.8g want %.8g\n",h,d,got[h*dim+d],want);failures++;}}}
    return failures;
}

int test_unsupported_kv_formats(q27::MetalBackend& backend) {
    constexpr uint32_t qh = 2, kvh = 1, dim = 256;
    std::vector<float> kv(dim), q(qh * dim);
    auto k = upload_buffer(backend, kv), v = upload_buffer(backend, kv);
    auto query = upload_buffer(backend, q);
    auto kc = backend.allocate(dim * 2), vc = backend.allocate(dim * 2);
    auto out = backend.allocate(qh * dim * 4);
    int failures = 0;
    auto expect_unsupported = [&](const char* operation, auto&& call) {
        try {
            call();
            fprintf(stderr, "%s accepted an unsupported KV format\n", operation);
            failures++;
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find("unsupported KV format") == std::string::npos) {
                fprintf(stderr, "%s rejected for the wrong reason: %s\n", operation, error.what());
                failures++;
            }
        }
    };
    const q27::KvFormat unsupported[] = {
        q27::KvFormat::FP8_E4M3,
        q27::KvFormat::TURBO3V,
        q27::KvFormat::TURBO5K,
        static_cast<q27::KvFormat>(0xff),
    };
    for (q27::KvFormat format : unsupported) {
        expect_unsupported("kv_store", [&] {
            backend.kv_store(*k, *v, *kc, *vc, 0, kvh, dim, format);
        });
        expect_unsupported("attention", [&] {
            backend.attention(*query, dim, *kc, *vc, *out, 1, qh, kvh, dim,
                              1.0f / 16.0f, format, nullptr);
        });
        expect_unsupported("kv_store_rows", [&] {
            backend.kv_store_rows(*k, *v, *kc, *vc, 0, kvh, dim, 1, format);
        });
        expect_unsupported("attention_causal", [&] {
            backend.attention_causal(*query, dim, qh * dim, *kc, *vc, *out,
                                     1, qh, kvh, dim, 1, 1.0f / 16.0f,
                                     format, nullptr);
        });
    }
    for (q27::KvFormat format : {q27::KvFormat::F16, q27::KvFormat::TURBO3}) {
        try {
            backend.attention_causal(*query, dim, qh * dim, *kc, *vc, *out,
                                     UINT32_MAX, qh, kvh, dim, 2, 1.0f / 16.0f,
                                     format, nullptr);
            fprintf(stderr, "attention_causal accepted an overflowing sequence length\n");
            failures++;
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find("sequence length overflow") == std::string::npos) {
                fprintf(stderr, "attention_causal overflow rejected for the wrong reason: %s\n",
                        error.what());
                failures++;
            }
        }
    }
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
            backend.kv_store(*kb, *vb, *kc, *vc, p, kvh, dim, q27::KvFormat::F16);
        }
        // Half-precision K/V for the CPU reference, matching what the cache holds.
        auto as_half = [](float v) { return (float)(_Float16)v; };
        std::vector<float> q((uint64_t)qh * stride);
        for (auto& value : q) value = uniform() * 1.5f;
        auto qb = upload_buffer(backend, q);
        auto out = backend.allocate((uint64_t)qh * dim * 4);
        backend.attention(*qb, stride, *kc, *vc, *out, seq, qh, kvh, dim, scale,
                          q27::KvFormat::F16, nullptr);
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
    backend.kv_store(*zerob,*zerob,*kt3,*vt3,0,kvh,dim,q27::KvFormat::TURBO3);
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
        backend.kv_store(*kb,*vb,*k16,*v16,p,kvh,dim,q27::KvFormat::F16);
        backend.kv_store(*kb,*vb,*kt3,*vt3,p,kvh,dim,q27::KvFormat::TURBO3);
    }
    std::vector<float> q(qh*dim);
    for(size_t i=0;i<q.size();i++) q[i]=.2f*std::sin(float(i)*.031f)+.1f*std::cos(float(i)*.053f);
    auto qbase=upload_buffer(backend,q),qt3=upload_buffer(backend,q);
    auto out16=backend.allocate((uint64_t)qh*dim*4),out3=backend.allocate((uint64_t)qh*dim*4);
    backend.attention(*qbase,stride,*k16,*v16,*out16,seq,qh,kvh,dim,
                      1/std::sqrt(float(dim)),q27::KvFormat::F16,nullptr);
    backend.begin_commands();
    backend.turbo_wht(*qt3,qh,stride,false);
    backend.attention(*qt3,stride,*kt3,*vt3,*out3,seq,qh,kvh,dim,
                      1/std::sqrt(float(dim)),q27::KvFormat::TURBO3,nullptr);
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

// q8 KV tier: rotated int8/32 blocks through the shared compressed-KV
// attention kernels (kv_codec 1). Checks codec quality against fp16 at the
// production head shape, that the blocked-GQA and chunk-causal q8 routes
// reproduce the serial q8 decode kernel (GQA within float reassociation;
// chunk rows bit for bit, the same contract turbo3 keeps), that the serial
// and row store kernels write identical blocks, and zero-vector packing.
int test_q8_kv(q27::MetalBackend& backend) {
    constexpr uint32_t seq = 133, qh = 24, kvh = 4, dim = 256, tokens = 5;
    const uint32_t base = seq - tokens;
    const float scale = 1 / std::sqrt(float(dim));
    const uint64_t row = (uint64_t)kvh * 2 * q27::kv_block_bytes(q27::KvCodec::Q8);
    int failures = 0;
    auto k16 = backend.allocate((uint64_t)seq * kvh * dim * 2), v16 = backend.allocate((uint64_t)seq * kvh * dim * 2);
    auto kq = backend.allocate(seq * row), vq = backend.allocate(seq * row);
    std::vector<float> zeros((size_t)kvh * dim, 0);
    auto zerob = upload_buffer(backend, zeros);
    backend.kv_store_turbo3(*zerob, *zerob, *kq, *vq, 0, kvh, q27::KvCodec::Q8);
    std::vector<uint8_t> zero_row(row);
    backend.read(*kq, 0, zero_row.data(), zero_row.size());
    for (uint8_t byte : zero_row) if (byte) { fprintf(stderr, "q8 zero-vector block not all zero\n"); failures++; break; }
    std::vector<float> all_k((size_t)seq * kvh * dim), all_v(all_k.size());
    for (uint32_t p = 0; p < seq; p++) {
        for (uint32_t i = 0; i < kvh * dim; i++) {
            // Mixed magnitudes per 32-group, so per-group scales matter.
            const float spike = (i % 37 == 5) ? 4.0f : 1.0f;
            all_k[(size_t)p * kvh * dim + i] = spike * (.5f * std::sin(float(i + 13 * p) * .041f) + .1f * std::cos(float(i) * .17f));
            all_v[(size_t)p * kvh * dim + i] = spike * (.7f * std::cos(float(i + 7 * p) * .023f) - .15f * std::sin(float(i) * .09f));
        }
        std::vector<float> k(all_k.begin() + (size_t)p * kvh * dim, all_k.begin() + (size_t)(p + 1) * kvh * dim);
        std::vector<float> v(all_v.begin() + (size_t)p * kvh * dim, all_v.begin() + (size_t)(p + 1) * kvh * dim);
        auto kb = upload_buffer(backend, k), vb = upload_buffer(backend, v);
        backend.kv_store(*kb, *vb, *k16, *v16, p, kvh, dim, q27::KvFormat::F16);
        backend.kv_store_turbo3(*kb, *vb, *kq, *vq, p, kvh, q27::KvCodec::Q8);
    }
    // Row store over the last `tokens` positions must write the same bytes
    // the serial store did (both sides; the range is cleared first).
    std::vector<uint8_t> before(tokens * row), before_v(tokens * row), after(tokens * row), after_v(tokens * row);
    backend.read(*kq, base * row, before.data(), before.size());
    backend.read(*vq, base * row, before_v.data(), before_v.size());
    std::vector<uint8_t> cleared(tokens * row, 0);
    backend.write(*kq, base * row, cleared.data(), cleared.size());
    backend.write(*vq, base * row, cleared.data(), cleared.size());
    std::vector<float> chunk_k(all_k.begin() + (size_t)base * kvh * dim, all_k.end());
    std::vector<float> chunk_v(all_v.begin() + (size_t)base * kvh * dim, all_v.end());
    auto ckb = upload_buffer(backend, chunk_k), cvb = upload_buffer(backend, chunk_v);
    backend.kv_store_turbo3_rows(*ckb, *cvb, *kq, *vq, base, kvh, tokens, q27::KvCodec::Q8);
    backend.read(*kq, base * row, after.data(), after.size());
    backend.read(*vq, base * row, after_v.data(), after_v.size());
    if (before != after || before_v != after_v) { fprintf(stderr, "q8 row store differs from serial store\n"); failures++; }

    std::vector<float> q((size_t)tokens * qh * dim);
    for (size_t i = 0; i < q.size(); i++) q[i] = .3f * std::sin(float(i) * .029f) + .1f * std::cos(float(i) * .061f);
    const uint32_t saved_threshold = backend.gqa_threshold();
    auto partials = backend.allocate((uint64_t)tokens * qh * 64 * 258 * 4);
    auto serial_q8 = [&](uint32_t t, uint32_t len) {
        std::vector<float> qt(q.begin() + (size_t)t * qh * dim, q.begin() + (size_t)(t + 1) * qh * dim);
        auto qb = upload_buffer(backend, qt);
        auto out = backend.allocate((uint64_t)qh * dim * 4);
        backend.begin_commands();
        backend.turbo_wht(*qb, qh, dim, false);
        backend.attention_turbo3(*qb, dim, *kq, *vq, *out, len, qh, kvh, dim, scale, partials.get(), q27::KvCodec::Q8);
        backend.turbo_wht(*out, qh, dim, true);
        backend.end_commands();
        return read_f32(backend, *out, (size_t)qh * dim);
    };
    backend.set_gqa_threshold(0);
    // Quality vs fp16 at the full sequence (last query row).
    {
        std::vector<float> qt(q.end() - (size_t)qh * dim, q.end());
        auto qb = upload_buffer(backend, qt);
        auto out16 = backend.allocate((uint64_t)qh * dim * 4);
        backend.attention(*qb, dim, *k16, *v16, *out16, seq, qh, kvh, dim, scale, q27::KvFormat::F16, nullptr);
        const auto base16 = read_f32(backend, *out16, (size_t)qh * dim);
        const auto got = serial_q8(tokens - 1, seq);
        double signal = 0, error = 0;
        for (size_t i = 0; i < got.size(); i++) { signal += base16[i] * base16[i]; error += (got[i] - base16[i]) * (got[i] - base16[i]); }
        const double nrmse = std::sqrt(error / std::max(signal, 1e-30));
        printf("q8 KV synthetic attention: nrmse %.6f vs fp16 (turbo3 bound is 0.30)\n", nrmse);
        if (!(nrmse < 0.01)) { fprintf(stderr, "q8 attention quality: nrmse %.6f\n", nrmse); failures++; }
    }
    // Chunk-causal rows must equal serial decode at each row's length.
    std::vector<std::vector<float>> serial_rows;
    for (uint32_t t = 0; t < tokens; t++) serial_rows.push_back(serial_q8(t, base + t + 1));
    // n rows starting at chunk row `first` (n == 1 takes the untiled
    // causal-GQA kernel; n >= 2 the factor-2 tiled one).
    auto chunk = [&](uint32_t first = 0, uint32_t n = tokens) {
        std::vector<float> qs(q.begin() + (size_t)first * qh * dim, q.begin() + (size_t)(first + n) * qh * dim);
        auto qb = upload_buffer(backend, qs);
        auto out = backend.allocate((uint64_t)n * qh * dim * 4);
        backend.begin_commands();
        backend.turbo_wht(*qb, n * qh, dim, false);
        backend.attention_turbo3_causal(*qb, dim, qh * dim, *kq, *vq, *out, base + first + 1, qh, kvh, dim,
                                        n, scale, partials.get(), q27::KvCodec::Q8);
        backend.turbo_wht(*out, n * qh, dim, true);
        backend.end_commands();
        return read_f32(backend, *out, (size_t)n * qh * dim);
    };
    const auto chunk_rows = chunk();
    for (uint32_t t = 0; t < tokens; t++)
        if (std::memcmp(chunk_rows.data() + (size_t)t * qh * dim, serial_rows[t].data(), (size_t)qh * dim * 4)) {
            fprintf(stderr, "q8 chunk-causal row %u differs from serial decode\n", t); failures++; break;
        }
    // Blocked-GQA routes (decode and chunk) vs the serial kernel.
    backend.set_gqa_threshold(1);
    const auto gqa_last = serial_q8(tokens - 1, seq);
    const auto gqa_chunk = chunk();
    const auto gqa_single = chunk(tokens - 1, 1);
    backend.set_gqa_threshold(saved_threshold);
    for (size_t i = 0; i < gqa_single.size(); i++)
        if (!near(gqa_single[i], serial_rows[tokens - 1][i], 2e-5f)) {
            fprintf(stderr, "q8 untiled GQA chunk[%zu] %.8g vs serial %.8g\n", i, gqa_single[i], serial_rows[tokens - 1][i]); failures++; break;
        }
    for (size_t i = 0; i < gqa_last.size(); i++)
        if (!near(gqa_last[i], serial_rows[tokens - 1][i], 2e-5f)) {
            fprintf(stderr, "q8 GQA decode[%zu] %.8g vs serial %.8g\n", i, gqa_last[i], serial_rows[tokens - 1][i]); failures++; break;
        }
    for (uint32_t t = 0; t < tokens; t++)
        for (size_t i = 0; i < (size_t)qh * dim; i++)
            if (!near(gqa_chunk[(size_t)t * qh * dim + i], serial_rows[t][i], 2e-5f)) {
                fprintf(stderr, "q8 GQA chunk row %u[%zu] differs from serial\n", t, i); failures++; break;
            }
    if (!failures) printf("q8 KV: store parity, chunk==serial bitwise, GQA routes, zero packing OK\n");
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
            backend.kv_store(*kb, *vb, *kc, *vc, p, kvh, dim, q27::KvFormat::TURBO3);
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
        backend.attention(*qb, stride, *kc, *vc, *out, seq, qh, kvh, dim, scale,
                          q27::KvFormat::TURBO3, nullptr);
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

int test_gdn(q27::MetalBackend& backend) {
    int failures=0;
    // Gates.
    std::vector<float> alpha={-30.1f,.3f}, br={-1,2}, av={-.5f,-.25f}, dtv={.1f,-.2f};
    auto ab=upload_buffer(backend,alpha), brb=upload_buffer(backend,br), go=backend.allocate(8), bo=backend.allocate(8);
    auto at=upload_f32(backend,av,{2}), dtt=upload_f32(backend,dtv,{2}); backend.gdn_gates(*ab,*brb,at,dtt,*go,*bo,2);
    auto gg=read_f32(backend,*go,2), bg=read_f32(backend,*bo,2); for(int i=0;i<2;i++){float z=alpha[i]+dtv[i],sp=z>20?z:std::log1p(std::exp(z)),gw=av[i]*sp,bw=1/(1+std::exp(-br[i]));if(!near(gg[i],gw)||!near(bg[i],bw)){fprintf(stderr,"gates[%d] mismatch\n",i);failures++;}}
    const float tail_want=av[0]*std::log1p(std::exp(alpha[0]+dtv[0]));
    if(std::fabs(gg[0]-tail_want)>std::fabs(tail_want)*1e-3f){fprintf(stderr,"gates negative-tail mismatch %.8g %.8g\n",gg[0],tail_want);failures++;}

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
        alpha[0] = -30.0f - dtv[0];
        auto ab = upload_buffer(backend, alpha); auto bb = upload_buffer(backend, braw);
        auto at = upload_f32(backend, av, {heads}); auto dtt = upload_f32(backend, dtv, {heads});
        auto cgo = backend.allocate((uint64_t)T * heads * 4); auto cbo = backend.allocate((uint64_t)T * heads * 4);
        backend.gdn_gates_rows(*ab, *bb, at, dtt, *cgo, *cbo, heads, T);
        auto got_g = read_f32(backend, *cgo, T * heads); auto got_b = read_f32(backend, *cbo, T * heads);
        const float tail_want = av[0] * std::log1p(std::exp(-30.0f));
        if (std::fabs(got_g[0] - tail_want) > std::fabs(tail_want) * 1e-3f)
            { fprintf(stderr, "gates rows negative-tail mismatch %.8g %.8g\n", got_g[0], tail_want); failures++; }
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
        constexpr uint32_t n = 2048, rows = 13;
        std::vector<float> logits(rows * n);
        std::vector<uint32_t> targets(rows);
        for (uint32_t t = 0; t < rows; t++) {
            for (uint32_t v = 0; v < n; v++)
                logits[t * n + v] = std::sin(float(t * 17 + v) * 0.11f) * 3.0f;
            targets[t] = 100 + t * 37;
            logits[t * n + targets[t]] += 2.5f;  // make the target distinctive
        }
        auto lb = upload_buffer(backend, logits);
        auto tb = backend.allocate(rows * 4);
        backend.write(*tb, 0, targets.data(), rows * 4);
        auto nb = backend.allocate(rows * 4);
        backend.nll_rows(*lb, *tb, *nb, n, rows);
        std::vector<float> got(rows);
        backend.read(*nb, 0, got.data(), rows * 4);
        for (uint32_t t = 0; t < rows; t++) {
            double mx = -1e300;
            for (uint32_t v = 0; v < n; v++) mx = std::max(mx, (double)logits[t * n + v]);
            double se = 0.0;
            for (uint32_t v = 0; v < n; v++) se += std::exp((double)logits[t * n + v] - mx);
            const float want = (float)(std::log(se) + mx - (double)logits[t * n + targets[t]]);
            if (std::fabs(got[t] - want) > 1e-4f)
                fail("nll rows", t, got[t], want);
        }
        targets[1] = n;
        backend.write(*tb, 0, targets.data(), rows * 4);
        backend.nll_rows(*lb, *tb, *nb, n, rows);
        backend.read(*nb, 0, got.data(), rows * 4);
        if (!std::isinf(got[1]) || got[1] < 0.0f)
            { fprintf(stderr, "nll rows accepted out-of-range target: %.8g\n", got[1]); failures++; }
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
            backend.kv_store(*kb, *vb, *kc, *vc, p, kvh, dim, q27::KvFormat::F16);
        }
        std::vector<float> knew(T * row), vnew(T * row), q(T * qh * stride);
        for (size_t i = 0; i < knew.size(); i++) { knew[i] = std::sin(float(i) * .53f); vnew[i] = std::cos(float(i) * .31f); }
        for (size_t i = 0; i < q.size(); i++) q[i] = std::sin(float(i) * .77f);
        auto knb = upload_buffer(backend, knew); auto vnb = upload_buffer(backend, vnew);
        backend.kv_store_rows(*knb, *vnb, *kc, *vc, warm, kvh, dim, T,
                              q27::KvFormat::F16);
        auto qb = upload_buffer(backend, q);
        auto chunk_out = backend.allocate((uint64_t)T * qh * dim * 4);
        backend.attention_causal(*qb, stride, qh * stride, *kc, *vc, *chunk_out,
                                 warm + 1, qh, kvh, dim, T, 0.5f,
                                 q27::KvFormat::F16, nullptr);
        auto got = read_f32(backend, *chunk_out, (size_t)T * qh * dim);
        auto serial_out = backend.allocate((uint64_t)qh * dim * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto q_row = row_view(*qb, t, qh * stride);
            backend.attention(*q_row, stride, *kc, *vc, *serial_out,
                              warm + 1 + t, qh, kvh, dim, 0.5f,
                              q27::KvFormat::F16, nullptr);
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
            backend.kv_store(*kb, *vb, *kc, *vc, p, kvh, dim, q27::KvFormat::F16);
        }
        std::vector<float> knew(T * row), vnew(T * row), q(T * qh * stride);
        for (size_t i = 0; i < knew.size(); i++) {
            knew[i] = std::sin(float(i) * .0047f) * (1.0f + float(i % 7));
            vnew[i] = std::cos(float(i) * .0031f);
        }
        for (size_t i = 0; i < q.size(); i++) q[i] = std::sin(float(i) * .0077f);
        auto knb = upload_buffer(backend, knew); auto vnb = upload_buffer(backend, vnew);
        backend.kv_store_rows(*knb, *vnb, *kc, *vc, warm, kvh, dim, T,
                              q27::KvFormat::F16);
        auto qb = upload_buffer(backend, q);
        auto chunk_out = backend.allocate((uint64_t)T * qh * dim * 4);
        backend.attention_causal(*qb, stride, qh * stride, *kc, *vc, *chunk_out,
                                 warm + 1, qh, kvh, dim, T, 0.0625f,
                                 q27::KvFormat::F16, nullptr);
        auto got = read_f32(backend, *chunk_out, (size_t)T * qh * dim);
        auto serial_out = backend.allocate((uint64_t)qh * dim * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto q_row = row_view(*qb, t, qh * stride);
            backend.attention(*q_row, stride, *kc, *vc, *serial_out,
                              warm + 1 + t, qh, kvh, dim, 0.0625f,
                              q27::KvFormat::F16, nullptr);
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
            backend.kv_store(*kb, *vb, *kc, *vc, p, kvh, dim, q27::KvFormat::TURBO3);
            backend.kv_store(*kb, *vb, *kc_ref, *vc_ref, p, kvh, dim, q27::KvFormat::TURBO3);
        }
        std::vector<float> knew(T * kvh * dim), vnew(T * kvh * dim), q(T * qh * stride);
        for (size_t i = 0; i < knew.size(); i++) { knew[i] = std::sin(float(i) * .011f); vnew[i] = std::cos(float(i) * .017f); }
        for (size_t i = 0; i < q.size(); i++) q[i] = std::sin(float(i) * .013f);
        auto knb = upload_buffer(backend, knew); auto vnb = upload_buffer(backend, vnew);
        backend.kv_store_rows(*knb, *vnb, *kc, *vc, warm, kvh, dim, T,
                              q27::KvFormat::TURBO3);
        for (uint32_t t = 0; t < T; t++) {
            auto k_row = row_view(*knb, t, kvh * dim); auto v_row = row_view(*vnb, t, kvh * dim);
            backend.kv_store(*k_row, *v_row, *kc_ref, *vc_ref, warm + t,
                             kvh, dim, q27::KvFormat::TURBO3);
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
        backend.attention_causal(*qb, stride, qh * stride, *kc, *vc, *chunk_out,
                                 warm + 1, qh, kvh, dim, T, 1.0f / std::sqrt(float(dim)),
                                 q27::KvFormat::TURBO3, nullptr);
        auto got = read_f32(backend, *chunk_out, (size_t)T * qh * dim);
        auto serial_out = backend.allocate((uint64_t)qh * dim * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto q_row = row_view(*qb, t, qh * stride);
            backend.attention(*q_row, stride, *kc, *vc, *serial_out,
                              warm + 1 + t, qh, kvh, dim, 1.0f / std::sqrt(float(dim)),
                              q27::KvFormat::TURBO3, nullptr);
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
                       test_unsupported_kv_formats(backend) +
                       test_attention_production_shape(backend) +
                       test_turbo3(backend) + test_turbo3_production_shape(backend) + test_q8_kv(backend) +
                       test_gdn(backend) + test_chunked(backend);
        if (failures) { fprintf(stderr, "Metal ops: %d failure(s)\n", failures); return 1; }
        puts("Metal decode primitives, FP16/turbo3 attention, GDN, and chunked prefill ops: OK");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
