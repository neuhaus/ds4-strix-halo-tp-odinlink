#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>
using I8 = int8_t; using I32 = int32_t;
__global__ void q4k_i8_mma(const I8 *a, const I8 *b, I32 *out) {
    using A = rocwmma::fragment<rocwmma::matrix_a,16,16,16,I8,rocwmma::row_major>;
    using B = rocwmma::fragment<rocwmma::matrix_b,16,16,16,I8,rocwmma::col_major>;
    using C = rocwmma::fragment<rocwmma::accumulator,16,16,16,I32>;
    A af; B bf; C cf; rocwmma::load_matrix_sync(af,a,16); rocwmma::load_matrix_sync(bf,b,16);
    rocwmma::fill_fragment(cf,0); rocwmma::mma_sync(cf,af,bf,cf); rocwmma::store_matrix_sync(out,cf,16,rocwmma::mem_row_major);
}
int main() {
    std::vector<I8> a(256),b(256); std::vector<I32> got(256),ref(256);
    for(unsigned r=0;r<16;++r) for(unsigned k=0;k<16;++k) { a[r*16+k]=I8(int((r*7+k*3)%16)-8); b[r*16+k]=I8(int((k*11+r*5)%127)-63); }
    I8 *da,*db; I32 *do_; if(hipMalloc(&da,256)||hipMalloc(&db,256)||hipMalloc(&do_,1024)) return 1;
    if(hipMemcpy(da,a.data(),256,hipMemcpyHostToDevice)||hipMemcpy(db,b.data(),256,hipMemcpyHostToDevice)) return 1;
    hipLaunchKernelGGL(q4k_i8_mma,dim3(1),dim3(32),0,0,da,db,do_);
    if(hipDeviceSynchronize()!=hipSuccess||hipMemcpy(got.data(),do_,1024,hipMemcpyDeviceToHost)) return 1;
    for(unsigned r=0;r<16;++r) for(unsigned c=0;c<16;++c) { I32 sum=0; for(unsigned k=0;k<16;++k) sum+=I32(a[r*16+k])*I32(b[c*16+k]); ref[r*16+c]=sum; }
    for(size_t i=0;i<got.size();++i) if(got[i]!=ref[i]) { std::fprintf(stderr,"MISMATCH i=%zu got=%d ref=%d\n",i,got[i],ref[i]); return 1; }
    hipEvent_t begin,end; hipEventCreate(&begin); hipEventCreate(&end); hipEventRecord(begin); for(unsigned i=0;i<10000;++i) hipLaunchKernelGGL(q4k_i8_mma,dim3(1),dim3(32),0,0,da,db,do_); hipEventRecord(end); hipEventSynchronize(end); float ms=0; hipEventElapsedTime(&ms,begin,end);
    std::printf("PASS q4k int8 MMA exact 16x16x16 launches=10000 total_ms=%.6f per_us=%.6f\n",ms,ms*1000/10000); hipEventDestroy(begin); hipEventDestroy(end); hipFree(da); hipFree(db); hipFree(do_);
}
