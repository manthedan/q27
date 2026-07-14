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

} // namespace

int main() {
    try {
        q27::MetalBackend backend;
        int failures = test_primitives(backend) + test_attention(backend) +
                       test_turbo3(backend) + test_gdn(backend);
        if (failures) { fprintf(stderr, "Metal ops: %d failure(s)\n", failures); return 1; }
        puts("Metal decode primitives, FP16/turbo3 attention, and GDN: OK");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
