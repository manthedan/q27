#include "metal_backend.h"
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
            std::cout<<"Hadamard n="<<n<<": dense reference, inverse, batch, red-zone, rejection PASS\n";
        }
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
