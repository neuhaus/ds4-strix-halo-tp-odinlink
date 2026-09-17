#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "glm5_gguf_test.hpp"
#include "../rocm/ds4_rocm_q4k_types.cuh"
#include "glm5_q4k_integer_block_test.hpp"

#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); \
} } while (0)
#define HIP(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr,"HIP line=%d: %s\n",__LINE__,hipGetErrorString(e)); \
    std::exit(1); } } while (0)

static uint32_t rng = 0x27182818;
static uint32_t next() { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }

// Scalar oracle separately decodes the 6-bit scale fields and each nibble.
static Q4KBlockResult scalar(const cuda_block_q4_K &w, const cuda_block_q8_K &x) {
    Q4KBlockResult r = {};
    for (unsigned k = 0; k < 256; ++k) {
        const unsigned g = k/32;
        const unsigned scale = g<4 ? w.scales[g]%64 :
            w.scales[g+4]%16 + (w.scales[g-4]/64)*16;
        const unsigned minimum = g<4 ? w.scales[g+4]%64 :
            w.scales[g+4]/16 + (w.scales[g]/64)*16;
        const unsigned packed = w.qs[(k/64)*32+k%32];
        const unsigned q = g%2 ? packed/16 : packed%16;
        r.dot += int(scale*q)*int(x.qs[k]);
        r.minimum += int(minimum)*int(x.qs[k]);
    }
    return r;
}

int main() {
    hipDeviceProp_t device;
    HIP(hipGetDeviceProperties(&device,0));
    REQUIRE(std::strncmp(device.gcnArchName,"gfx1151",7)==0);
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    std::vector<std::string> names;
    for (const auto &item : gguf.tensors) {
        const auto &name = item.first;
        if (name.find(".ffn_gate_exps.weight") != std::string::npos ||
            name.find(".ffn_up_exps.weight") != std::string::npos ||
            name.find(".ffn_down_exps.weight") != std::string::npos) {
            REQUIRE(item.second.type == 12 && item.second.dims.size() == 3);
            names.push_back(name);
        }
    }
    REQUIRE(names.size() >= 12);
    std::sort(names.begin(),names.end());
    std::vector<cuda_block_q4_K> w;
    // Twelve tensors spread across layers/types; 64 blocks per tensor spread
    // across expert IDs, output rows and K. All copied bytes remain original.
    for (unsigned s = 0; s < 12; ++s) {
        const auto &name = names[s*(names.size()-1)/11];
        const auto &info = gguf.tensors.at(name);
        uint64_t offset;
        REQUIRE(gguf.tensor(name.c_str(),info.dims,12,offset));
        const uint64_t blocks = info.dims[0]*info.dims[1]*info.dims[2]/256;
        for (unsigned b = 0; b < 64; ++b) {
            const uint64_t at = offset + (uint64_t(b)*(blocks-1)/63)*sizeof(cuda_block_q4_K);
            REQUIRE(at <= gguf.size && sizeof(cuda_block_q4_K) <= gguf.size-at);
            cuda_block_q4_K block;
            std::memcpy(&block,gguf.map+at,sizeof(block));
            w.push_back(block);
        }
        std::printf("fixture tensor=%s original_blocks=64\n",name.c_str());
    }
    const unsigned real = w.size();
    for (unsigned i = 0; i < 256; ++i) {
        cuda_block_q4_K block = {};
        // All valid scale/minimum bit combinations, nibbles0/15 and random.
        block.d = uint16_t(0x2000+(next()%0x2400));
        block.dmin = uint16_t(0x2000+(next()%0x2400));
        for (auto &v : block.scales) v = i<4 ? uint8_t(i%2 ? 255 : 0) : uint8_t(next());
        for (auto &v : block.qs) v = i<4 ? uint8_t(i<2 ? 255 : 0) : uint8_t(next());
        w.push_back(block);
    }
    const unsigned n = w.size();
    std::vector<cuda_block_q8_K> x(256);
    for (unsigned t = 0; t < x.size(); ++t) {
        auto &v = x[t];
        v.d = t==0 ? 0.0f : (t%2 ? -1.0f : 1.0f)*(0.001f+float(t)*0.00017f);
        for (unsigned k = 0; k < 256; ++k)
            v.qs[k] = t==0 ? 0 : t==1 ? -128 : t==2 ? 127 :
                t==3 ? (k%2 ? -128 : 127) : int8_t(int(next()%256)-128);
        for (unsigned g = 0; g < 16; ++g) {
            int sum = 0;
            for (unsigned k = 0; k < 16; ++k) sum += int(v.qs[g*16+k]);
            v.bsums[g] = int16_t(sum);
        }
    }
    cuda_block_q4_K *dw; cuda_block_q8_K *dx; Q4KBlockResult *dref, *dgot;
    HIP(hipMalloc(&dw,w.size()*sizeof(w[0])));
    HIP(hipMalloc(&dx,x.size()*sizeof(x[0])));
    HIP(hipMalloc(&dref,n*256*sizeof(Q4KBlockResult)));
    HIP(hipMalloc(&dgot,n*256*sizeof(Q4KBlockResult)));
    HIP(hipMemcpy(dw,w.data(),w.size()*sizeof(w[0]),hipMemcpyHostToDevice));
    HIP(hipMemcpy(dx,x.data(),x.size()*sizeof(x[0]),hipMemcpyHostToDevice));
    HIP(glm5_q4k_integer_blocks(dw,dx,dgot,n-1,16,true,true)==hipErrorInvalidValue ? hipSuccess : hipErrorUnknown);
    for (unsigned m : {1u, 7u, 16u, 17u, 256u}) {
        HIP(glm5_q4k_integer_blocks(dw,dx,dref,n,m,false,true));
        HIP(glm5_q4k_integer_blocks(dw,dx,dgot,n,m,true,true));
        std::vector<Q4KBlockResult> ref(n*m),got(n*m);
        HIP(hipMemcpy(ref.data(),dref,ref.size()*sizeof(ref[0]),hipMemcpyDeviceToHost));
        HIP(hipMemcpy(got.data(),dgot,got.size()*sizeof(got[0]),hipMemcpyDeviceToHost));
        for (unsigned i = 0; i < n*m; ++i) {
            const auto oracle = scalar(w[i%n],x[i/n]);
            REQUIRE(ref[i].dot == oracle.dot && ref[i].minimum == oracle.minimum);
            REQUIRE(got[i].dot == oracle.dot && got[i].minimum == oracle.minimum);
            REQUIRE(std::isfinite(ref[i].value) && std::isfinite(got[i].value));
            if (std::memcmp(&ref[i].value,&got[i].value,sizeof(float))) {
                std::fprintf(stderr,"float mismatch m=%u i=%u ref=%a got=%a\n",m,i,ref[i].value,got[i].value);
                return 1;
            }
        }
        std::printf("PASS original_blocks=%u adversarial_blocks=%u M=%u outputs=%u integer_oracle=exact float_dp4a=bitwise\n",real,n-real,m,n*m);
        if (m!=1 && m!=16 && m!=256) continue;
        hipEvent_t begin,end;
        HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end));
        for (unsigned repeat=0; repeat<3; ++repeat) for (unsigned arm=0; arm<2; ++arm) {
            const bool matrix = bool(arm ^ (repeat%2));
            for (unsigned j=0; j<5; ++j) HIP(glm5_q4k_integer_blocks(dw,dx,dgot,n,m,matrix,false));
            HIP(hipEventRecord(begin));
            for (unsigned j=0; j<100; ++j) HIP(glm5_q4k_integer_blocks(dw,dx,dgot,n,m,matrix,false));
            HIP(hipEventRecord(end)); HIP(hipEventSynchronize(end));
            float ms; HIP(hipEventElapsedTime(&ms,begin,end));
            std::printf("diagnostic repeat=%u M=%u path=%s kernel_us=%.3f\n",repeat,m,matrix?"mma":"dp4a",ms*10);
        }
        HIP(hipEventDestroy(begin)); HIP(hipEventDestroy(end));
    }
    HIP(hipFree(dw)); HIP(hipFree(dx)); HIP(hipFree(dref)); HIP(hipFree(dgot));
    return 0;
}
