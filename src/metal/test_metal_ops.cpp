#include "metal_backend.h"

#include <algorithm>
#include <cmath>
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
    return failures;
}

int test_attention(q27::MetalBackend& backend) {
    constexpr uint32_t seq=3,qh=2,kvh=1,dim=4,stride=8;
    auto kc=backend.allocate(seq*kvh*dim*2), vc=backend.allocate(seq*kvh*dim*2);
    const std::vector<std::vector<float>> keys={{1,0,0,0},{0,1,0,0},{1,1,0,0}};
    const std::vector<std::vector<float>> vals={{1,2,3,4},{2,0,1,3},{4,3,2,1}};
    for(uint32_t p=0;p<seq;p++){auto k=upload_buffer(backend,keys[p]),v=upload_buffer(backend,vals[p]);backend.kv_store_f16(*k,*v,*kc,*vc,p,dim);}
    std::vector<float> q={1,.5f,0,0, 9,9,9,9, 0,1,0,0, 8,8,8,8}; auto qb=upload_buffer(backend,q);
    auto scratch=backend.allocate(qh*seq*4), out=backend.allocate(qh*dim*4);
    backend.attention_f16(*qb,stride,*kc,*vc,*scratch,*out,seq,qh,kvh,dim,0.5f); auto got=read_f32(backend,*out,qh*dim);
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
        auto scratch = backend.allocate((uint64_t)qh * seq * 4);
        auto out = backend.allocate((uint64_t)qh * dim * 4);
        backend.attention_f16(*qb, stride, *kc, *vc, *scratch, *out, seq, qh, kvh, dim, scale);
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
    auto scratch16=backend.allocate((uint64_t)qh*seq*4), scratch3=backend.allocate((uint64_t)qh*seq*4);
    auto out16=backend.allocate((uint64_t)qh*dim*4),out3=backend.allocate((uint64_t)qh*dim*4);
    backend.attention_f16(*qbase,stride,*k16,*v16,*scratch16,*out16,seq,qh,kvh,dim,1/std::sqrt(float(dim)));
    backend.begin_commands();
    backend.turbo_wht(*qt3,qh,stride,false);
    backend.attention_turbo3(*qt3,stride,*kt3,*vt3,*scratch3,*out3,seq,qh,kvh,dim,1/std::sqrt(float(dim)));
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
        backend.conv_chunk(*ring_chunk, *qkvb, cwt, *chunk_out, channels, T);
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
        backend.delta_chunk(*state_chunk, *cvb, *gb, *betab, *chunk_out, vh, qkh, hd, T);
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
        auto scratch = backend.allocate((uint64_t)T * qh * (warm + T) * 4);
        auto chunk_out = backend.allocate((uint64_t)T * qh * dim * 4);
        backend.attention_f16_causal(*qb, stride, qh * stride, *kc, *vc, *scratch, *chunk_out,
                                     warm + 1, qh, kvh, dim, T, 0.5f);
        auto got = read_f32(backend, *chunk_out, (size_t)T * qh * dim);
        auto serial_scratch = backend.allocate((uint64_t)qh * (warm + T) * 4);
        auto serial_out = backend.allocate((uint64_t)qh * dim * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto q_row = row_view(*qb, t, qh * stride);
            backend.attention_f16(*q_row, stride, *kc, *vc, *serial_scratch, *serial_out,
                                  warm + 1 + t, qh, kvh, dim, 0.5f);
            auto want = read_f32(backend, *serial_out, (size_t)qh * dim);
            for (uint32_t i = 0; i < qh * dim; i++)
                if (!near(got[t*qh*dim+i], want[i], 1e-5f)) { fail("causal attention", t*qh*dim+i, got[t*qh*dim+i], want[i]); break; }
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
        auto scratch = backend.allocate((uint64_t)T * qh * (warm + T) * 4);
        auto chunk_out = backend.allocate((uint64_t)T * qh * dim * 4);
        backend.attention_turbo3_causal(*qb, stride, qh * stride, *kc, *vc, *scratch, *chunk_out,
                                        warm + 1, qh, kvh, dim, T, 1.0f / std::sqrt(float(dim)));
        auto got = read_f32(backend, *chunk_out, (size_t)T * qh * dim);
        auto serial_scratch = backend.allocate((uint64_t)qh * (warm + T) * 4);
        auto serial_out = backend.allocate((uint64_t)qh * dim * 4);
        for (uint32_t t = 0; t < T; t++) {
            auto q_row = row_view(*qb, t, qh * stride);
            backend.attention_turbo3(*q_row, stride, *kc, *vc, *serial_scratch, *serial_out,
                                     warm + 1 + t, qh, kvh, dim, 1.0f / std::sqrt(float(dim)));
            auto want = read_f32(backend, *serial_out, (size_t)qh * dim);
            for (uint32_t i = 0; i < qh * dim; i++)
                if (!near(got[t*qh*dim+i], want[i], 1e-5f)) { fail("turbo3 causal", t*qh*dim+i, got[t*qh*dim+i], want[i]); break; }
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
                       test_turbo3(backend) + test_gdn(backend) + test_chunked(backend);
        if (failures) { fprintf(stderr, "Metal ops: %d failure(s)\n", failures); return 1; }
        puts("Metal decode primitives, FP16/turbo3 attention, GDN, and chunked prefill ops: OK");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
