#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
extern "C" {
#include "ds4_tp.h"
}
#pragma GCC diagnostic pop
#include "glm5_gguf_test.hpp"
#include "ds4_glm5_expert_pairs.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int ds4_gpu_q8k_quantize_research_control(ds4_gpu_tensor *,const ds4_gpu_tensor *,uint32_t,uint32_t);
#define CHECK(x) do { if(!(x)) { std::fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#x); return 1; } } while(0)
using Clock=std::chrono::steady_clock;
static double micros(Clock::time_point begin) {
    return std::chrono::duration<double,std::micro>(Clock::now()-begin).count();
}
static bool same(const char *what,const void *a,const void *b,size_t bytes,unsigned rank,unsigned fixture,unsigned mode) {
    if(!std::memcmp(a,b,bytes)) return true;
    auto *av=(const unsigned char *)a,*bv=(const unsigned char *)b;
    for(size_t i=0;i<bytes;++i) if(av[i]!=bv[i]) {
        std::fprintf(stderr,"MISMATCH %s rank=%u fixture=%u mode=%u byte=%zu ref=%u actual=%u\n",
            what,rank,fixture,mode,i,unsigned(av[i]),unsigned(bv[i]));
        if(!std::strcmp(what,"mid") || !std::strcmp(what,"out")) {
            const size_t at=(i/4)*4; float af,bf;
            std::memcpy(&af,av+at,4);std::memcpy(&bf,bv+at,4);
            const size_t width=!std::strcmp(what,"mid")?1024:4096;
            std::fprintf(stderr,"FLOAT pair_or_token=%zu row=%zu ref=%a actual=%a\n",
                (i/4)/width,(i/4)%width,double(af),double(bf));
        }
        break;
    }
    return false;
}
int main(int argc,char **argv) {
    CHECK(argc==5 && (!std::strcmp(argv[1],"0")||!std::strcmp(argv[1],"1")));
    const unsigned rank=unsigned(argv[1][0]-'0');
    unsigned params[3];
    for(unsigned i=0;i<3;++i) {
        char *end=nullptr; const auto n=std::strtoul(argv[i+2],&end,10);
        CHECK(end && !*end && n<=0xffffffffu); params[i]=unsigned(n);
    }
    const unsigned layer=params[0],seed=params[1],timing=params[2];
    CHECK(layer>=3 && layer<=44 && seed>0 && timing<=2);
    int32_t ids[48]; float weights[48];
    for(unsigned i=0;i<48;++i) ids[i]=int32_t(i%12);
    ds4_glm5_expert_groups groups{};
    unsigned refused=0;
    const char *path=std::getenv("DS4_GLM5_MODEL"); CHECK(path);
    Glm5TestGGUF gguf; CHECK(gguf.open_file(path));
    uint64_t go=0,uo=0,doo=0;
    char name[96];
    std::snprintf(name,sizeof(name),"blk.%u.ffn_gate_exps.weight",layer);
    CHECK(gguf.tensor(name,{4096,2048,288},12,go));
    std::snprintf(name,sizeof(name),"blk.%u.ffn_up_exps.weight",layer);
    CHECK(gguf.tensor(name,{4096,2048,288},12,uo));
    std::snprintf(name,sizeof(name),"blk.%u.ffn_down_exps.weight",layer);
    CHECK(gguf.tensor(name,{2048,4096,288},12,doo));
    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH","1",1)==0);
    CHECK(setenv("DS4_ROCM_Q4K_WMMA","1",1)==0);
    CHECK(setenv("DS4_ROCM_GLM5_Q4K_DECODE_GATE_ROWS","128",1)==0);
    for(const char *key : {"DS4_ROCM_DISABLE_Q4K_WMMA","DS4_ROCM_TP_SKIP_UNOWNED",
        "DS4_ROCM_GLM5_Q4K_DECODE_DOT_UNROLL","DS4_ROCM_GLM5_Q4K_DECODE_DOT_LANES",
        "DS4_ROCM_Q4K_DECODE_STAGE_XQ","DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP",
        "DS4_ROCM_Q4K_DECODE_STAGE_MIDQ","DS4_ROCM_Q4K_DECODE_FUSE_ADDEND",
        "DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP","DS4_ROCM_Q4K_WMMA_FUSE_MID"}) CHECK(setenv(key,"0",1)==0);
    ds4_gpu_config config{}; config.n_gpus=1; CHECK(ds4_gpu_init_multi(&config));
    CHECK(ds4_gpu_set_model_fd_for_map(gguf.fd,gguf.map));
    CHECK(ds4_gpu_set_model_map(gguf.map,gguf.size));
    ds4_gpu_set_tp_runtime_features(0,DS4_TP_FEATURE_Q4K_WMMA|DS4_TP_FEATURE_Q4K_KSHARD);
    for(uint64_t offset : {go,uo}) {
        CHECK(ds4_gpu_q4k_packed_slice_declare(gguf.map,gguf.size,offset,
            288,2048,2304,rank*1024,1024,0,2304,DS4_GPU_Q4K_PACKED_ROW_RANGE));
        CHECK(ds4_gpu_q4k_packed_slice_load(gguf.map,offset,rank*1024,1024,0,2304));
    }
    CHECK(ds4_gpu_q4k_packed_slice_declare(gguf.map,gguf.size,doo,
        288,4096,1152,0,4096,rank*576,576,DS4_GPU_Q4K_PACKED_K_RANGE));
    CHECK(ds4_gpu_q4k_packed_slice_load(gguf.map,doo,0,4096,rank*576,576));
    const void *gw=nullptr,*uw=nullptr,*dw=nullptr;
    uint64_t packed=0,expert_bytes=0,row_bytes=0;
    for(unsigned i=0;i<2;++i) {
        const void **ptr=i?&uw:&gw;
        CHECK(ds4_gpu_q4k_packed_slice_resolve(gguf.map,i?uo:go,288,2048,2304,
            rank*1024,1024,0,2304,DS4_GPU_Q4K_PACKED_ROW_RANGE,ptr,&packed,&expert_bytes,&row_bytes));
        CHECK(*ptr && packed==679477248 && expert_bytes==2359296 && row_bytes==2304);
    }
    CHECK(ds4_gpu_q4k_packed_slice_resolve(gguf.map,doo,288,4096,1152,0,4096,
        rank*576,576,DS4_GPU_Q4K_PACKED_K_RANGE,&dw,&packed,&expert_bytes,&row_bytes));
    CHECK(dw && packed==679477248 && expert_bytes==2359296 && row_bytes==576);
    constexpr uint64_t xb=6*4096*4, mb=48*1024*4, ob=6*4096*4;
    constexpr uint64_t qxb=6*16*292, qmb=48*4*292, guard=64;
    auto *dx=ds4_gpu_tensor_alloc(xb),*di=ds4_gpu_tensor_alloc(48*4),*dwt=ds4_gpu_tensor_alloc(48*4);
    auto *dg=ds4_gpu_tensor_alloc(48*sizeof(ds4_glm5_expert_group)); CHECK(dx&&di&&dwt&&dg);
    ds4_gpu_tensor *raw[4]{},*buf[4]{}; const uint64_t sizes[]={qxb,mb,qmb,ob};
    for(unsigned i=0;i<4;++i) {
        raw[i]=ds4_gpu_tensor_alloc(sizes[i]+2*guard); CHECK(raw[i]);
        buf[i]=ds4_gpu_tensor_view(raw[i],guard,sizes[i]); CHECK(buf[i]);
    }
    auto *qx=buf[0],*mid=buf[1],*qm=buf[2],*out=buf[3];
    ds4_gpu_tensor *t[6][8]{},*qmrow[6]{},*orow[6]{};
    const uint64_t scratch[]={4096*4,8*1024*4,8*1024*4,8*1024*4,8*4096*4};
    for(unsigned r=0;r<6;++r) {
        for(unsigned k=0;k<5;++k) {t[r][k]=ds4_gpu_tensor_alloc(scratch[k]); CHECK(t[r][k]);}
        t[r][5]=ds4_gpu_tensor_view(di,r*32,32);
        t[r][6]=ds4_gpu_tensor_view(dwt,r*32,32);
        t[r][7]=ds4_gpu_tensor_view(dx,r*4096*4,4096*4);
        qmrow[r]=ds4_gpu_tensor_view(qm,r*8*4*292,8*4*292);
        orow[r]=ds4_gpu_tensor_view(out,r*4096*4,4096*4);
        CHECK(t[r][5]&&t[r][6]&&t[r][7]&&qmrow[r]&&orow[r]);
    }
    std::vector<float> x(xb/4),rm(mb/4),am(mb/4),ro(ob/4),ao(ob/4),guards(16);
    std::vector<unsigned char> rxq(qxb),axq(qxb),rmq(qmb),amq(qmb);
    const float clamp=10.0f;
    ds4_glm5_expert_six_args args{out,mid,qx,qm,dg,dx,di,dwt,gguf.map,gguf.size,go,uo,doo,rank,6};
    ds4_glm5_expert_six_plan plan{};
    CHECK(ds4_rocm_glm5_expert_six_admit(&plan,&args));
    for(unsigned fault=0;fault<11;++fault) {
        auto bad=args; ds4_glm5_expert_six_plan rejected{};
        switch(fault) {
        case 0:bad.rows=8;break;
        case 1:bad.rows=4;break;
        case 2:bad.rank=2;break;
        case 3:bad.model_size=go;break;
        case 4:bad.gate_offset++;break;
        case 5:bad.up_offset=go;break;
        case 6:bad.input_q8=qm;break;
        case 7:bad.input=di;break;
        case 8:bad.descriptors=mid;break;
        case 9:bad.out=nullptr;break;
        case 10:bad.model_map=(const char *)gguf.map+1;break;
        }
        CHECK(!ds4_rocm_glm5_expert_six_admit(&rejected,&bad));++refused;
    }
    CHECK(!ds4_rocm_glm5_expert_six_down_row(&plan,6));++refused;
    for(const char *key:{"DS4_ROCM_TP_SKIP_UNOWNED","DS4_ROCM_Q4K_DECODE_STAGE_XQ",
        "DS4_ROCM_Q4K_DECODE_FUSE_ADDEND","DS4_ROCM_GLM5_Q4K_DECODE_DOT_LANES"}) {
        CHECK(setenv(key,"invalid",1)==0);
        ds4_glm5_expert_six_plan rejected{};
        CHECK(!ds4_rocm_glm5_expert_six_admit(&rejected,&args));++refused;
        CHECK(setenv(key,"0",1)==0);
    }
    auto prepare=[&](unsigned fixture)->bool {
        unsigned unique=fixture==0?12:fixture==2?48:fixture==8?16:fixture==9?10:fixture==10?44:fixture==11?46:34;
        for(unsigned r=0;r<6;++r) {
            for(unsigned s=0;s<8;++s) {
                ids[r*8+s]=int32_t(((r*8+s)%unique+(uint64_t(seed)*53+layer*17)%288)%288);
                weights[r*8+s]=float(.031973+double((uint64_t(seed)+r*11+s*29)%113)*.00317239);
            }
            if(fixture==3) {weights[r*8+1]=0.0f;weights[r*8+2]=-0.0f;weights[r*8+7]=0.0f;}
            for(unsigned i=0;i<4096;++i) {
                unsigned tr=fixture==5?0:r;
                const double scale=fixture==6?64.0:fixture==7?1024.0:1.0;
                x[r*4096+i]=fixture==4?0.0f:float(scale*(std::sin(double(i)*.013+tr+fixture+seed*.0001)*.17+
                                                   std::cos(double(i)*.031+tr+seed*.0003)*.11));
            }
        }
        return ds4_gpu_tensor_write(di,0,ids,sizeof(ids)) && ds4_gpu_tensor_write(dwt,0,weights,sizeof(weights)) &&
            ds4_gpu_tensor_write(dx,0,x.data(),xb);
    };
    auto incumbent=[&](bool fenced)->bool {
        for(unsigned r=0;r<6;++r) {
            auto &v=t[r];
            if(!ds4_gpu_routed_moe_one_packed_q4k_tensor(v[0],v[1],v[2],v[3],v[4],
                gguf.map,gguf.size,go,uo,doo,288,2304,1152,rank*1024,1024,rank*576,576,
                v[5],v[6],8,clamp,v[7],nullptr,layer)) return false;
            if(fenced && !ds4_gpu_synchronize()) return false;
        }
        return true;
    };
    auto split=[&](bool reuse,bool fenced=false)->bool {
        if(!ds4_glm5_expert_groups_build(&groups,ids,weights,reuse?2:1) ||
           !ds4_rocm_glm5_expert_six_admit(&plan,&args) ||
           !ds4_rocm_glm5_expert_six_begin(&plan,&groups,ids,weights)) return false;
        for(unsigned r=0;r<6;++r) {
            if(!ds4_rocm_glm5_expert_six_down_row(&plan,r)) return false;
            if(fenced && !ds4_gpu_synchronize()) return false;
        }
        return true;
    };
    unsigned cases=0;
    for(unsigned fixture=0;fixture<12;++fixture) {
        CHECK(prepare(fixture)); CHECK(incumbent(true)); CHECK(ds4_gpu_synchronize());
        for(unsigned r=0;r<6;++r) {
            CHECK(ds4_gpu_tensor_read(t[r][0],0,ro.data()+r*4096,4096*4));
            CHECK(ds4_gpu_tensor_read(t[r][3],0,rm.data()+r*8*1024,8*1024*4));
            CHECK(ds4_gpu_tensor_read(t[r][4],0,rxq.data()+r*16*292,16*292));
            CHECK(ds4_gpu_tensor_read(t[r][1],0,rmq.data()+r*8*4*292,8*4*292));
        }
        for(float v:rm) CHECK(std::isfinite(v));
        for(float v:ro) CHECK(std::isfinite(v));
        CHECK(incumbent(false)); CHECK(ds4_gpu_synchronize());
        for(unsigned r=0;r<6;++r) CHECK(ds4_gpu_tensor_read(t[r][0],0,ao.data()+r*4096,4096*4));
        CHECK(same("queued_out",ro.data(),ao.data(),ob,rank,fixture,1));
        for(unsigned reuse=0;reuse<2;++reuse) for(unsigned fenced=0;fenced<2;++fenced) {
            for(unsigned i=0;i<4;++i) CHECK(ds4_gpu_tensor_fill_f32(raw[i],123.5f,(sizes[i]+2*guard)/4));
            CHECK(split(reuse,fenced)); CHECK(ds4_gpu_synchronize());
            CHECK(ds4_gpu_tensor_read(qx,0,axq.data(),qxb));
            CHECK(same("qx",rxq.data(),axq.data(),qxb,rank,fixture,reuse+2));
            CHECK(ds4_gpu_tensor_read(mid,0,am.data(),mb));
            for(float v:am) CHECK(std::isfinite(v));
            CHECK(same("mid",rm.data(),am.data(),mb,rank,fixture,reuse+2));
            CHECK(ds4_gpu_tensor_read(qm,0,amq.data(),qmb));
            CHECK(same("qmid",rmq.data(),amq.data(),qmb,rank,fixture,reuse+2));
            CHECK(ds4_gpu_tensor_read(out,0,ao.data(),ob));
            for(float v:ao) CHECK(std::isfinite(v));
            CHECK(same("out",ro.data(),ao.data(),ob,rank,fixture,reuse+2));
            for(unsigned i=0;i<4;++i) for(uint64_t at:{uint64_t(0),sizes[i]+guard}) {
                CHECK(ds4_gpu_tensor_read(raw[i],at,guards.data(),guard));
                for(float value:guards) CHECK(value==123.5f);
            }
            std::printf("EXACT rank=%u fixture=%u reuse=%u fenced=%u singles=%u doubles=%u mid_floats=49152 out_floats=24576\n",
                rank,fixture,reuse,fenced,groups.singles,groups.doubles); std::fflush(stdout);
            ++cases;
        }
    }
    std::printf("PASS expert gateup rank=%u layer=%u seed=%u cases=%u descriptor_refusals=%u original_packed_bytes=%llu\n",
        rank,layer,seed,cases,refused,(unsigned long long)ds4_gpu_q4k_packed_slice_bytes()); std::fflush(stdout);
    // Read disjoint original experts before each measured invocation. This is
    // a cache-sensitivity probe, not proof of universally cold memory.
    auto condition=[&]()->bool {
        int32_t alternate[48];
        for(unsigned i=0;i<48;++i) alternate[i]=(ids[i]+144)%288;
        return ds4_gpu_tensor_write(di,0,alternate,sizeof(alternate)) && incumbent(false) &&
            ds4_gpu_synchronize() && ds4_gpu_tensor_write(di,0,ids,sizeof(ids));
    };
    hipEvent_t start,stop; CHECK(hipEventCreate(&start)==hipSuccess); CHECK(hipEventCreate(&stop)==hipSuccess);
    for(unsigned fixture : {0u,1u,2u,8u,9u,10u,11u}) {
        if(!timing) break;
        CHECK(prepare(fixture));
        for(unsigned round=0;round<7;++round) for(unsigned index=0;index<4;++index) {
            const unsigned mode=(round&1)?3-index:index;
            auto call=[&]()->bool {return mode<2?incumbent(mode==0):split(mode==3);};
            for(unsigned i=0;i<3;++i) {CHECK(call());CHECK(ds4_gpu_synchronize());}
            constexpr unsigned reps=20;
            double wall=0,event_us=0;
            for(unsigned i=0;i<reps;++i) {
                if(timing==2) CHECK(condition());
                CHECK(hipEventRecord(start,0)==hipSuccess);const auto begin=Clock::now();
                CHECK(call());CHECK(ds4_gpu_synchronize());wall+=micros(begin);
                CHECK(hipEventRecord(stop,0)==hipSuccess);CHECK(hipEventSynchronize(stop)==hipSuccess);
                float ms=0;CHECK(hipEventElapsedTime(&ms,start,stop)==hipSuccess);event_us+=ms*1000;
            }
            std::printf("TIME rank=%u fixture=%u round=%u mode=%u wall_us=%.6f event_us=%.6f reps=%u metadata_included=%u cache=%s\n",
                rank,fixture,round,mode,wall/reps,event_us/reps,reps,unsigned(mode>=2),timing==2?"disjoint-experts":"warm");
        }
    }
    CHECK(hipEventDestroy(start)==hipSuccess);CHECK(hipEventDestroy(stop)==hipSuccess);
    for(unsigned r=0;r<6;++r) {
        for(auto *v:t[r]) ds4_gpu_tensor_free(v);
        ds4_gpu_tensor_free(qmrow[r]);ds4_gpu_tensor_free(orow[r]);
    }
    for(unsigned i=0;i<4;++i) {ds4_gpu_tensor_free(buf[i]);ds4_gpu_tensor_free(raw[i]);}
    for(auto *v:{dx,di,dwt,dg}) ds4_gpu_tensor_free(v);
    ds4_gpu_q4k_packed_slice_release_all(); ds4_gpu_cleanup();
}
