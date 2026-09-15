// Native GGUF full-head oracle versus each TP-owned half. No weight conversion.
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", what); std::exit(1); }
}
struct Tensor {
    ds4_gpu_tensor *p;
    unsigned count;
    explicit Tensor(unsigned n): p(ds4_gpu_tensor_alloc((uint64_t)n * 4)), count(n) {
        require(p != nullptr, "allocate tensor");
    }
    ~Tensor() { ds4_gpu_tensor_free(p); }
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    void write(const void *v) {
        require(ds4_gpu_tensor_write(p,0,v,(uint64_t)count*4), "write tensor");
    }
    std::vector<float> read() {
        std::vector<float> v(count);
        require(ds4_gpu_tensor_read(p,0,v.data(),(uint64_t)count*4), "read tensor");
        return v;
    }
    void poison() { std::vector<float> v(count, -1234567.125f); write(v.data()); }
};
static void same(const std::vector<float> &got, const std::vector<float> &ref,
                 unsigned start, unsigned count, const char *what) {
    require(got.size() >= count && ref.size() >= start+count, "compare extents");
    for (unsigned i=0; i<count; ++i) {
        require(std::isfinite(got[i]) && std::isfinite(ref[start+i]), "finite outputs");
        if (memcmp(&got[i], &ref[start+i], 4)) {
            fprintf(stderr,"mismatch %s i=%u got=%.9g ref=%.9g\n",
                    what,i,got[i],ref[start+i]);
            require(false,what);
        }
    }
    for (unsigned i=count; i<got.size(); ++i)
        require(got[i] == -1234567.125f, "unowned tail and output canary");
}
static void attention(Tensor &out, Tensor &q, Tensor &low, Tensor &kv,
                      Tensor &selected, Glm5TestGGUF &gguf, uint64_t vb,
                      unsigned n, unsigned heads) {
    require(ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        out.p,q.p,low.p,kv.p,nullptr,gguf.map,gguf.size,vb,8u,selected.p,
        n,4096u,false,heads,512u,256u,0u,256u,0u,
        1.f,1.f,0.f,1.f,0.f,0.f), "indexed attention and v_b");
}
int main() {
    const char *model = getenv("DS4_GLM5_MODEL");
    Glm5TestGGUF gguf;
    require(model && gguf.open_file(model), "selected GGUF");
    setenv("DS4_ROCM_GLM5_Q8_DECODE_TILE","1",1);
    setenv("DS4_ROCM_GLM5_QK_LOW_LDS_EXACT","1",1);
    setenv("DS4_ROCM_GLM5_NOPE_ATTN_EXACT","1",1);
    ds4_gpu_config config{};
    config.n_gpus=1; config.device_indices[0]=0;
    require(ds4_gpu_init_multi(&config) &&
            ds4_gpu_set_model_fd_for_map(gguf.fd,gguf.map) &&
            ds4_gpu_set_model_map(gguf.map,gguf.size), "GPU model map");
    Tensor x(1536), q64(16384), q32(16384+16), low64(32768), low32(32768+16);
    Tensor kv(4096*512), selected(2051), heads64(16384), heads32(16384+16);
    Tensor output64(4096), output32(4096+16);
    std::vector<float> input(x.count), cache(kv.count);
    std::vector<int32_t> ids(selected.count);
    unsigned layers=0, cases=0;
    for (unsigned layer=0; layer<45; ++layer) {
        const std::string prefix="blk."+std::to_string(layer)+".";
        uint64_t qb=0,kb=0,vb=0,wo=0;
        if (!gguf.tensor(prefix+"attn_q_b.weight",{1536,16384},8,qb)) continue;
        require(gguf.tensor(prefix+"attn_k_b.weight",{256,512,64},8,kb) &&
                gguf.tensor(prefix+"attn_v_b.weight",{512,256,64},8,vb) &&
                gguf.tensor(prefix+"attn_output.weight",{16384,4096},8,wo),
                "native MLA tensor layout");
        ++layers;
        for (unsigned seed=0; seed<3; ++seed) {
            for (unsigned i=0; i<input.size(); ++i)
                input[i]=((int)((i*73+seed*193)%1021)-510)/(1001.3f+(i%7));
            for (unsigned i=0; i<cache.size(); ++i)
                cache[i]=((int)((i*29+seed*97+(i>>9)*31)%1021)-510)/5001.7f;
            x.write(input.data()); kv.write(cache.data());
            require(ds4_gpu_matmul_q8_0_tensor(q64.p,gguf.map,gguf.size,qb,
                1536,16384,x.p,1), "full q_b");
            require(ds4_gpu_glm_qk_lowrank_typed_tensor(low64.p,q64.p,
                gguf.map,gguf.size,kb,8,64,512,256,256), "full k_b");
            const auto qref=q64.read(), lowref=low64.read();
            for (unsigned rank=0; rank<2; ++rank) {
                // Independent directory-derived row widths, never a packed copy.
                const uint64_t qoff=qb+(uint64_t)rank*8192*(1536/32)*34;
                const uint64_t koff=kb+(uint64_t)rank*32*512*(256/32)*34;
                const uint64_t voff=vb+(uint64_t)rank*32*256*(512/32)*34;
                q32.poison(); low32.poison();
                require(ds4_gpu_matmul_q8_0_tensor(q32.p,gguf.map,gguf.size,qoff,
                    1536,8192,x.p,1), "owned q_b");
                same(q32.read(),qref,rank*8192,8192,"q_b exact owned rows");
                require(ds4_gpu_glm_qk_lowrank_typed_tensor(low32.p,q32.p,
                    gguf.map,gguf.size,koff,8,32,512,256,256), "owned k_b");
                same(low32.read(),lowref,rank*16384,16384,"k_b exact owned heads");
                for (unsigned shared : {0u,1u}) {
                    setenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV", shared?"1":"0",1);
                    for (unsigned n : {1u,15u,16u,17u,127u,2048u,2049u,2051u}) {
                        for (unsigned pattern=0; pattern<2; ++pattern) {
                            for (unsigned i=0; i<ids.size(); ++i)
                                ids[i]=pattern ? (int32_t)((i*37+seed*11)%4096) : (int32_t)i;
                            if (pattern && n>16) {
                                ids[3]=-1; ids[8]=4096; ids[n-1]=ids[0];
                            }
                            selected.write(ids.data()); heads32.poison();
                            attention(heads64,q64,low64,kv,selected,gguf,vb,n,64);
                            attention(heads32,q32,low32,kv,selected,gguf,voff,n,32);
                            same(heads32.read(),heads64.read(),rank*8192,8192,
                                 "attention/v_b exact owned heads");
                            output32.poison();
                            require(ds4_gpu_matmul_q8_0_kslice_tensor(output64.p,
                                gguf.map,gguf.size,wo,16384,rank*8192,8192,
                                4096,heads64.p,rank*8192), "full-head output slice");
                            require(ds4_gpu_matmul_q8_0_kslice_tensor(output32.p,
                                gguf.map,gguf.size,wo,16384,rank*8192,8192,
                                4096,heads32.p,0), "owned-head output slice");
                            same(output32.read(),output64.read(),0,4096,
                                 "output projection exact");
                            ++cases;
                        }
                    }
                }
                printf("PASS layer=%u seed=%u rank=%u stages=q_b,k_b,attention,v_b,output "
                       "lengths=8 patterns=2 exact_and_shared=1 bit_exact=1 canary=1\n",
                       layer,seed,rank);
                fflush(stdout);
            }
        }
    }
    require(layers==11 && cases==2112, "complete trunk coverage");
    printf("PASS owned-head MLA layers=%u cases=%u model_bytes=%llu\n",
           layers,cases,(unsigned long long)gguf.size);
    return 0;
}
