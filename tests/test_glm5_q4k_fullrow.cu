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
#include "glm5_q4k_fullrow_test.hpp"

#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); \
} } while (0)
#define HIP(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr,"HIP line=%d: %s\n",__LINE__,hipGetErrorString(e)); \
    std::exit(1); } } while (0)

static uint32_t rng = 0x19462538;
static uint32_t next() { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }

// Separately decode nibbles/scales and accumulate in double. This supplies an
// error-bound oracle in addition to the bitwise production-arithmetic check.
static double scalar(const cuda_block_q4_K *w, const cuda_block_q8_K *x,
                     unsigned blocks, double &magnitude) {
    double result = 0;
    magnitude = 0;
    for (unsigned b = 0; b < blocks; ++b) {
        int dot = 0, minimum = 0;
        for (unsigned k = 0; k < 256; ++k) {
            const unsigned g = k/32;
            const unsigned sc = g<4 ? w[b].scales[g]%64 :
                w[b].scales[g+4]%16 + (w[b].scales[g-4]/64)*16;
            const unsigned mn = g<4 ? w[b].scales[g+4]%64 :
                w[b].scales[g+4]/16 + (w[b].scales[g]/64)*16;
            const unsigned packed = w[b].qs[(k/64)*32+k%32];
            dot += int(sc*(g%2 ? packed/16 : packed%16))*int(x[b].qs[k]);
            minimum += int(mn)*int(x[b].qs[k]);
        }
        __half hd, hm;
        std::memcpy(&hd,&w[b].d,2); std::memcpy(&hm,&w[b].dmin,2);
        const double a = double(x[b].d)*__half2float(hd)*dot;
        const double c = double(x[b].d)*__half2float(hm)*minimum;
        result += a-c;
        magnitude += std::abs(a)+std::abs(c);
    }
    return result;
}

int main() {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    hipDeviceProp_t device;
    HIP(hipGetDeviceProperties(&device,0));
    REQUIRE(std::strncmp(device.gcnArchName,"gfx1151",7)==0);
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    constexpr unsigned n=1024, max_m=96;
    for (const char *role : {"gate","up","down"}) {
        // Early layers can be dense: discover actual routed tensors rather
        // than assuming expert weights exist in block zero.
        const std::string suffix = std::string(".ffn_")+role+"_exps.weight";
        std::vector<std::string> names;
        for (const auto &item : gguf.tensors)
            if (item.first.size()>=suffix.size() &&
                item.first.compare(item.first.size()-suffix.size(),suffix.size(),suffix)==0)
                names.push_back(item.first);
        REQUIRE(names.size()>=3);
        std::sort(names.begin(),names.end());
        for (size_t fixture : {size_t(0), names.size()/2, names.size()-1}) {
        const std::string &name=names[fixture];
        const auto &info = gguf.tensors.at(name);
        REQUIRE(info.type==12 && info.dims.size()==3 && info.dims[0]%256==0);
        const unsigned blocks = info.dims[0]/256;
        REQUIRE(blocks>0 && blocks<=64 && info.dims[1]*info.dims[2]>=n);
        std::printf("fixture tensor=%s K=%u original_rows=%u\n",name.c_str(),blocks*256,n);
        uint64_t offset;
        REQUIRE(gguf.tensor(name.c_str(),info.dims,12,offset));
        std::vector<cuda_block_q4_K> w(n*blocks);
        // Every output is a complete original row. Spread over experts and
        // rows, without a dequantized or requantized representation.
        for (unsigned row=0; row<n; ++row) {
            const uint64_t source_row = uint64_t(row)*(info.dims[1]*info.dims[2]-1)/(n-1);
            const uint64_t at = offset+source_row*blocks*sizeof(w[0]);
            REQUIRE(at<=gguf.size && blocks*sizeof(w[0])<=gguf.size-at);
            std::memcpy(w.data()+row*blocks,gguf.map+at,blocks*sizeof(w[0]));
        }
        std::vector<cuda_block_q8_K> x(max_m*blocks);
        for (unsigned t=0; t<max_m; ++t) for (unsigned b=0; b<blocks; ++b) {
            auto &xb=x[t*blocks+b];
            xb.d=t==4 ? 0 : (b%2 ? -1.0f : 1.0f)*(0.001013f+float(next()%1000)*0.0000073f);
            for (unsigned k=0; k<256; ++k)
                xb.qs[k]=t==4 ? 0 : t==1 ? -128 : t==2 ? 127 :
                    t==3 ? (k%2 ? -128 : 127) : int8_t(int(next()%256)-128);
            for (unsigned g=0; g<16; ++g) {
                int sum=0;
                for (unsigned k=0; k<16; ++k) sum+=int(xb.qs[g*16+k]);
                xb.bsums[g]=int16_t(sum);
            }
        }
        cuda_block_q4_K *dw; cuda_block_q8_K *dx; float *dref,*dgot;
        HIP(hipMalloc(&dw,w.size()*sizeof(w[0])));
        HIP(hipMalloc(&dx,x.size()*sizeof(x[0])));
        HIP(hipMalloc(&dref,(n*max_m+32)*sizeof(float)));
        HIP(hipMalloc(&dgot,(n*max_m+32)*sizeof(float)));
        HIP(hipMemcpy(dw,w.data(),w.size()*sizeof(w[0]),hipMemcpyHostToDevice));
        HIP(hipMemcpy(dx,x.data(),x.size()*sizeof(x[0]),hipMemcpyHostToDevice));
        REQUIRE(glm5_q4k_fullrows(dw,dx,dgot,n-1,16,blocks,1)==hipErrorInvalidValue);
        REQUIRE(glm5_q4k_fullrows(dw,dx,dgot,n,16,0,1)==hipErrorInvalidValue);
        REQUIRE(glm5_q4k_fullrows(dw,dx,dgot,n,16,blocks,4)==hipErrorInvalidValue);
        REQUIRE(glm5_q4k_fullrows(nullptr,dx,dgot,n,16,blocks,1)==hipErrorInvalidValue);
        for (unsigned m : {1u,7u,16u,17u,28u,32u,64u,96u}) {
            HIP(hipMemset(dref,0x5a,(n*max_m+32)*sizeof(float)));
            HIP(glm5_q4k_fullrows(dw,dx,dref+16,n,m,blocks,0));
            std::vector<float> ref(n*m+32),got(n*m+32);
            HIP(hipMemcpy(ref.data(),dref,ref.size()*sizeof(float),hipMemcpyDeviceToHost));
            for (unsigned candidate : {1u,2u,3u}) {
                HIP(hipMemset(dgot,0x5a,(n*max_m+32)*sizeof(float)));
                HIP(glm5_q4k_fullrows(dw,dx,dgot+16,n,m,blocks,candidate));
                HIP(hipMemcpy(got.data(),dgot,got.size()*sizeof(float),hipMemcpyDeviceToHost));
                for (unsigned i=0; i<ref.size(); ++i) {
                    if (std::memcmp(&got[i],&ref[i],sizeof(float))) {
                        std::fprintf(stderr,"MISMATCH role=%s K=%u M=%u mode=%u index=%u ref=%a got=%a\n",
                                     role,blocks*256,m,candidate,i,ref[i],got[i]);
                        return 1;
                    }
                    if (i>=16 && i<n*m+16) REQUIRE(std::isfinite(got[i]));
                    else { uint32_t bits; std::memcpy(&bits,&got[i],4); REQUIRE(bits==0x5a5a5a5au); }
                }
                // Independent numerical oracle over spread output rows and
                // every token, including cancellation/extrema fixtures.
                for (unsigned t=0; t<m; ++t) for (unsigned row=0; row<n; row+=127) {
                    double magnitude;
                    const double expected=scalar(w.data()+row*blocks,x.data()+t*blocks,blocks,magnitude);
                    REQUIRE(std::abs(double(got[16+t*n+row])-expected)<=1e-6*magnitude+1e-8);
                }
                std::printf("PASS role=%s K=%u N=%u M=%u mode=%u exact=%u canaries=32 scalar=pass\n",
                            role,blocks*256,n,m,candidate,n*m);
                if (m==7 || m==17 || candidate==1) continue;
                hipEvent_t begin,end;
                HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end));
                for (unsigned pair=0; pair<3; ++pair) for (unsigned arm=0; arm<2; ++arm) {
                    // Compare the split-scale MMA to both scalar and
                    // eight-token DP4A; retain every adjacent arm result.
                    const unsigned mode=(arm^(pair%2)) ? 2u : (candidate==3 ? 3u : 0u);
                    constexpr unsigned warmup=1000, iterations=1000;
                    for (unsigned j=0; j<warmup; ++j) HIP(glm5_q4k_fullrows(dw,dx,dgot+16,n,m,blocks,mode));
                    HIP(hipEventRecord(begin));
                    for (unsigned j=0; j<iterations; ++j) HIP(glm5_q4k_fullrows(dw,dx,dgot+16,n,m,blocks,mode));
                    HIP(hipEventRecord(end)); HIP(hipEventSynchronize(end));
                    float ms; HIP(hipEventElapsedTime(&ms,begin,end));
                    std::printf("diagnostic role=%s K=%u N=%u M=%u candidate=%u pair=%u mode=%u warmup=%u iterations=%u us=%.3f\n",
                                role,blocks*256,n,m,candidate,pair,mode,warmup,iterations,ms*1000/iterations);
                }
                HIP(hipEventDestroy(begin)); HIP(hipEventDestroy(end));
            }
        }
        HIP(hipFree(dw)); HIP(hipFree(dx)); HIP(hipFree(dref)); HIP(hipFree(dgot));
        }
    }
    return 0;
}
