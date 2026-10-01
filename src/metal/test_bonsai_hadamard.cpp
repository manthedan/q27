#include "metal_backend.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace q27;

// Independent dense Sylvester matrix, not another copy of the GPU butterfly.
static std::vector<float> reference(const std::vector<float>& input,
                                     const std::vector<float>& signs,
                                     bool inverse, bool grouped) {
    auto x=input;
    if(grouped) {
        for(unsigned k=0;k<16;++k) for(unsigned rep=0;rep<3;++rep)
            for(unsigned d=0;d<128;++d)
                x[d+128*(rep+3*k)]=input[d+128*(k+16*rep)];
    }
    std::vector<float> out(x.size());
    for(size_t base=0;base<x.size();base+=1024)
        for(unsigned row=0;row<1024;++row) {
            double sum=0;
            for(unsigned col=0;col<1024;++col)
                sum += (__builtin_parity(row&col) ? -1.0 : 1.0) * x[base+col] *
                       (inverse ? 1.0 : signs[base+col]);
            out[base+row]=float(sum/32.0)*(inverse?signs[base+row]:1.0f);
        }
    return out;
}

static void near(const std::vector<float>& actual,const std::vector<float>& expected) {
    for(size_t i=0;i<expected.size();++i)
        if(!std::isfinite(actual[i]) || std::fabs(actual[i]-expected[i])>2e-5f)
            throw std::runtime_error("Hadamard mismatch at "+std::to_string(i)+": "+
                std::to_string(actual[i])+" vs "+std::to_string(expected[i]));
}

static uint16_t half_bits(float value) {
    __fp16 h = (__fp16)value; uint16_t bits; std::memcpy(&bits,&h,2); return bits;
}

// Float-activation T2 GEMM against a double-precision CPU reference built
// from the packed codes, with an error bound scaled by sum |w*x| (the GPU
// reduction order differs from both the reference and the serial GEMV).
static void t2_float_gemm(MetalBackend& gpu, std::mt19937& rng, uint32_t rows, uint32_t cols) {
    std::uniform_int_distribution<int> trit(0,2);
    std::uniform_real_distribution<float> scale(0.01f,0.2f), value(-3,3);
    std::vector<uint8_t> packed((size_t)rows*cols/4,0);
    std::vector<int8_t> codes((size_t)rows*cols);
    for(size_t i=0;i<codes.size();++i){int c=trit(rng);codes[i]=int8_t(c-1);packed[i/4]|=uint8_t(c<<((i%4)*2));}
    std::vector<uint16_t> scales((size_t)rows*cols/128); std::vector<float> scale_values(scales.size());
    for(size_t i=0;i<scales.size();++i){scales[i]=half_bits(scale(rng));__fp16 h;std::memcpy(&h,&scales[i],2);scale_values[i]=float(h);}
    Tensor tensor; tensor.name="t2-float-gemm"; tensor.dtype=DType::T2_G128; tensor.shape={rows,cols};
    tensor.data=packed.data(); tensor.data_size=packed.size();
    tensor.scales=reinterpret_cast<const uint8_t*>(scales.data()); tensor.scales_size=scales.size()*2;
    auto weight=gpu.upload(tensor);
    constexpr uint32_t max_tokens=96;
    std::vector<float> x((size_t)max_tokens*cols);
    for(auto& v:x) v=value(rng);
    auto xb=gpu.allocate(x.size()*4); gpu.write(*xb,0,x.data(),x.size()*4);
    const size_t red=29;
    auto yb=gpu.allocate(((size_t)max_tokens*rows+red)*4);
    for(uint32_t tokens:{1u,5u,16u,17u,33u,96u}) {
        std::vector<float> y((size_t)max_tokens*rows+red,777.0f);
        gpu.write(*yb,0,y.data(),y.size()*4);
        gpu.matmul_t2_float(weight,*xb,tokens,*yb);
        gpu.read(*yb,0,y.data(),y.size()*4);
        for(uint32_t t=0;t<tokens;++t) for(uint32_t r=0;r<rows;++r) {
            double sum=0,mag=0;
            for(uint32_t c=0;c<cols;++c) {
                const double term=double(codes[(size_t)r*cols+c])*scale_values[(size_t)r*(cols/128)+c/128]*x[(size_t)t*cols+c];
                sum+=term; mag+=std::fabs(term);
            }
            const float got=y[(size_t)t*rows+r];
            if(!std::isfinite(got) || std::fabs(got-sum)>1e-6*mag+1e-6)
                throw std::runtime_error("float T2 GEMM mismatch rows="+std::to_string(rows)+" cols="+
                    std::to_string(cols)+" tokens="+std::to_string(tokens)+" at ("+std::to_string(t)+","+
                    std::to_string(r)+"): "+std::to_string(got)+" vs "+std::to_string(sum));
        }
        for(size_t i=(size_t)tokens*rows;i<y.size();++i) if(y[i]!=777.0f)
            throw std::runtime_error("float T2 GEMM wrote past its "+std::to_string(tokens)+" rows");
    }
    auto rejects=[&](const std::function<void()>& f){
        try { f(); } catch(const std::exception&) {return;}
        throw std::runtime_error("invalid float T2 GEMM dispatch accepted");
    };
    rejects([&]{gpu.matmul_t2_float(weight,*xb,0,*yb);});
    rejects([&]{gpu.matmul_t2_float(weight,*xb,97,*yb);});
    rejects([&]{gpu.matmul_t2_float(weight,*xb,4,*xb);});
    auto small=gpu.allocate((size_t)cols*4);
    rejects([&]{gpu.matmul_t2_float(weight,*small,2,*yb);});
    std::cout<<"float T2 GEMM "<<rows<<"x"<<cols<<": 1..96 tokens, partial tiles, red-zone, rejection PASS\n";
}

int main() {
    try {
        MetalBackend gpu;
        std::mt19937 rng(20260918);
        std::uniform_real_distribution<float> random(-1,1);
        for(uint32_t n:{1024u,5120u,6144u,17408u}) {
            std::vector<float> input(n),signs(n),actual(n+17,12345.0f);
            for(uint32_t i=0;i<n;++i){input[i]=random(rng);signs[i]=(rng()&1)?1.0f:-1.0f;}
            auto x=gpu.allocate(n*4),s=gpu.allocate(n*4),y=gpu.allocate(actual.size()*4);
            gpu.write(*x,0,input.data(),n*4);gpu.write(*s,0,signs.data(),n*4);
            gpu.write(*y,0,actual.data(),actual.size()*4);
            for(bool inverse:{false,true}) for(bool grouped:{false,true}) {
                if(grouped&&(n!=6144||inverse)) continue;
                gpu.bonsai_hadamard(*x,*s,*y,n,inverse,grouped);
                gpu.read(*y,0,actual.data(),actual.size()*4);
                near(actual,reference(input,signs,inverse,grouped));
                for(size_t i=n;i<actual.size();++i) if(actual[i]!=12345.0f)
                    throw std::runtime_error("Hadamard wrote past output");
            }
            // Two dependent dispatches within one command buffer, as in the engine.
            gpu.begin_commands();
            gpu.bonsai_hadamard(*x,*s,*y,n,false);
            gpu.bonsai_hadamard(*y,*s,*x,n,true);
            gpu.end_commands();
            std::vector<float> roundtrip(n);
            gpu.read(*x,0,roundtrip.data(),n*4); near(roundtrip,input);
            auto rejects=[&](const std::function<void()>& f){
                try { f(); } catch(const std::exception&) {return;}
                throw std::runtime_error("invalid Hadamard dispatch accepted");
            };
            rejects([&]{gpu.bonsai_hadamard(*x,*s,*x,n,false);});
            rejects([&]{gpu.bonsai_hadamard(*x,*s,*s,n,false);});
            rejects([&]{gpu.bonsai_hadamard(*x,*s,*y,n+1024,false);});
            rejects([&]{gpu.bonsai_hadamard(*x,*s,*y,512,false);});
            rejects([&]{gpu.bonsai_hadamard(*x,*s,*y,0,false);});
            rejects([&]{gpu.bonsai_hadamard(*x,*s,*y,n,true,true);});
            // Row batches: every row must equal its own single-row dispatch
            // bit for bit (same kernel, same per-row arithmetic).
            for(uint32_t rows:{3u,96u}) for(bool grouped:{false,true}) {
                if(grouped&&n!=6144) continue;
                std::vector<float> batch((size_t)rows*n);
                for(auto& v:batch) v=random(rng);
                std::vector<float> got((size_t)rows*n+13,4321.0f),single(n);
                auto bx=gpu.allocate(batch.size()*4),by=gpu.allocate(got.size()*4);
                auto rx=gpu.allocate(n*4),ry=gpu.allocate(n*4);
                gpu.write(*bx,0,batch.data(),batch.size()*4); gpu.write(*by,0,got.data(),got.size()*4);
                gpu.bonsai_hadamard(*bx,*s,*by,n,false,grouped,rows);
                gpu.read(*by,0,got.data(),got.size()*4);
                for(uint32_t r=0;r<rows;++r) {
                    gpu.write(*rx,0,batch.data()+(size_t)r*n,n*4);
                    gpu.bonsai_hadamard(*rx,*s,*ry,n,false,grouped);
                    gpu.read(*ry,0,single.data(),n*4);
                    if(std::memcmp(single.data(),got.data()+(size_t)r*n,n*4))
                        throw std::runtime_error("batched Hadamard row "+std::to_string(r)+" differs from single");
                }
                for(size_t i=(size_t)rows*n;i<got.size();++i) if(got[i]!=4321.0f)
                    throw std::runtime_error("batched Hadamard wrote past output");
                rejects([&]{gpu.bonsai_hadamard(*bx,*s,*by,n,false,grouped,0);});
                rejects([&]{gpu.bonsai_hadamard(*bx,*s,*by,n,false,grouped,97);});
                rejects([&]{gpu.bonsai_hadamard(*rx,*s,*by,n,false,grouped,rows);});
            }
            std::cout<<"Hadamard n="<<n<<": dense reference, inverse, batch, rows, red-zone, rejection PASS\n";
        }
        for(auto shape:{std::pair<uint32_t,uint32_t>{9,128},{33,256},{32,5120},{65,6144},{40,17408}})
            t2_float_gemm(gpu,rng,shape.first,shape.second);
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
