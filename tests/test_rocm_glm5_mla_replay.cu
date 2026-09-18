// Differential cache-state test using real GGUF pool APE and production GPU
// pooling/selection. Synthetic normalized keys/gates isolate cache semantics;
// this is not a projection, full-attention, TP or model quality test.
#include "ds4_glm5_next_runtime.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "glm5_gguf_test.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); std::exit(1); \
} } while (0)

static constexpr unsigned width = 128, capacity = 8224;
static constexpr uint64_t row_bytes = width * 4u;
static uint64_t compared_bytes;

struct Tensor {
    ds4_gpu_tensor *p;
    explicit Tensor(uint64_t bytes) : p(ds4_gpu_tensor_alloc(bytes)) { REQUIRE(p); }
    ~Tensor() { ds4_gpu_tensor_free(p); }
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    operator ds4_gpu_tensor *() const { return p; }
};

static std::vector<uint8_t> read(ds4_gpu_tensor *t, uint64_t bytes) {
    std::vector<uint8_t> v(bytes);
    if (bytes) REQUIRE(ds4_gpu_tensor_read(t, 0, v.data(), bytes));
    return v;
}

static void equal(ds4_gpu_tensor *a, ds4_gpu_tensor *b, uint64_t bytes) {
    REQUIRE(read(a, bytes) == read(b, bytes));
    compared_bytes += bytes;
}

static void poison_after(ds4_gpu_tensor *tensor,uint64_t first_byte) {
    const uint64_t bytes=ds4_gpu_tensor_bytes(tensor)-first_byte;
    if (!bytes) return;
    auto *future=ds4_gpu_tensor_view(tensor,first_byte,bytes);
    REQUIRE(future && ds4_gpu_tensor_fill_f32(future,NAN,bytes/4));
    ds4_gpu_tensor_free(future);
}

struct State {
    ds4_glm5_next_state owner = {};
    ds4_glm5_next_mla_state &s;
    explicit State(unsigned m) : s(owner.mla[3]) {
        owner.layer_count = 45; owner.context_capacity = capacity;
        owner.mla_count = 11; owner.valid = true;
        s.owner = &owner; s.capacity_tokens = capacity;
        s.capacity_pools = capacity / 4; s.valid = true;
        s.compact_kv = ds4_gpu_tensor_alloc((uint64_t)capacity * 512 * 4);
        s.index_pool = ds4_gpu_tensor_alloc(s.capacity_pools * row_bytes);
        s.index_pool_ids = ds4_gpu_tensor_alloc(s.capacity_pools * 16u);
        s.index_pool_valid = ds4_gpu_tensor_alloc(s.capacity_pools * 4u);
        s.index_valid_keys = ds4_gpu_tensor_alloc(capacity * 4u);
        s.index_tail = ds4_gpu_tensor_alloc(4 * row_bytes);
        s.pool_gate_tail = ds4_gpu_tensor_alloc(4 * row_bytes);
        REQUIRE(s.compact_kv && s.index_pool && s.index_pool_ids &&
            s.index_pool_valid && s.index_valid_keys && s.index_tail && s.pool_gate_tail);
        REQUIRE(ds4_gpu_tensor_fill_f32(s.index_valid_keys, 1.0f, capacity));
        REQUIRE(ds4_gpu_tensor_fill_f32(s.compact_kv, -99.0f, (uint64_t)capacity*512));
        REQUIRE(ds4_gpu_tensor_fill_f32(s.index_pool, -99.0f, s.capacity_pools*width));
        REQUIRE(ds4_gpu_tensor_fill_f32(s.index_pool_ids, -99.0f, s.capacity_pools*4u));
        REQUIRE(ds4_gpu_tensor_fill_f32(s.index_pool_valid, 0.0f, s.capacity_pools));
        REQUIRE(ds4_gpu_tensor_fill_f32(s.index_tail, -99.0f, 4*width));
        REQUIRE(ds4_gpu_tensor_fill_f32(s.pool_gate_tail, -99.0f, 4*width));
        if (m) REQUIRE(ds4_glm5_next_mla_replay_reserve(&s, m));
    }
    ~State() { ds4_glm5_next_state_free(&owner); }
};

static void compare_live(ds4_glm5_next_mla_state &a,
                         ds4_glm5_next_mla_state &b) {
    REQUIRE(a.valid && b.valid && a.token_count == b.token_count &&
        a.complete_pools == b.complete_pools && a.tail_count == b.tail_count);
    equal(a.index_tail,b.index_tail,4*row_bytes);
    equal(a.pool_gate_tail,b.pool_gate_tail,4*row_bytes);
    equal(a.compact_kv,b.compact_kv,(uint64_t)a.token_count*512*4);
    equal(a.index_pool,b.index_pool,a.complete_pools*row_bytes);
    equal(a.index_pool_ids,b.index_pool_ids,a.complete_pools*16u);
    equal(a.index_pool_valid,b.index_pool_valid,a.complete_pools*4u);
    equal(a.index_valid_keys,b.index_valid_keys,capacity*4u);
}

struct Fixture {
    const Glm5TestGGUF &g;
    uint64_t ape;
    Tensor keys{capacity*row_bytes}, gates{capacity*row_bytes};
    Tensor kv{(uint64_t)capacity*512*4};
    Tensor valid{16}, ids{16}, pool_valid{4};
    Tensor query{32*row_bytes}, weights{32*4};
    Tensor scores{(capacity/4)*4}, selected_pools{512*4}, selected{2051*4};
    explicit Fixture(const Glm5TestGGUF &gguf, uint64_t offset) : g(gguf),ape(offset) {
        REQUIRE(ds4_gpu_tensor_fill_f32(valid,1.0f,4));
        std::vector<float> q(32*width), w(32);
        for (size_t i=0;i<q.size();++i) q[i]=float(int(i*37%997)-498)/1001.3f;
        for (size_t i=0;i<w.size();++i) w[i]=float(i+1)/35.7f;
        REQUIRE(ds4_gpu_tensor_write(query,0,q.data(),q.size()*4));
        REQUIRE(ds4_gpu_tensor_write(weights,0,w.data(),w.size()*4));
        restore_inputs();
    }
    void restore_inputs() {
        std::vector<float> k(capacity*width), gt(k.size()), v(capacity*512);
        for (size_t i=0;i<k.size();++i) {
            k[i]=float(int((i*193+(i/width)*761)%997)-498)/503.7f;
            gt[i]=float(int((i*47+(i/width)*557)%997)-498)/71.3f;
        }
        for (size_t i=0;i<v.size();++i) v[i]=float(int(i*133%997)-498)/1001.3f;
        REQUIRE(ds4_gpu_tensor_write(keys,0,k.data(),k.size()*4));
        REQUIRE(ds4_gpu_tensor_write(gates,0,gt.data(),gt.size()*4));
        REQUIRE(ds4_gpu_tensor_write(kv,0,v.data(),v.size()*4));
    }
    void seed(ds4_glm5_next_mla_state &s, unsigned prefix) {
        // Bulk seed is common to both arms. Subsequent reference appends use
        // independent scalar tail writes and kpool, never the replay journal.
        if (prefix) REQUIRE(ds4_gpu_tensor_copy(s.compact_kv,0,kv,0,(uint64_t)prefix*512*4));
        const unsigned complete = prefix/4;
        if (complete) REQUIRE(ds4_gpu_glm5_kpool_tensor(
            s.index_pool,s.index_pool_ids,s.index_pool_valid,keys,gates,
            s.index_valid_keys,g.map,g.size,ape,complete*4,width,4,0));
        for (unsigned i=prefix>4?prefix-4:0;i<prefix;++i) {
            REQUIRE(ds4_gpu_tensor_copy(s.index_tail,(i%4)*row_bytes,keys,i*row_bytes,row_bytes));
            REQUIRE(ds4_gpu_tensor_copy(s.pool_gate_tail,(i%4)*row_bytes,gates,i*row_bytes,row_bytes));
        }
        s.token_count=prefix; s.complete_pools=complete; s.tail_count=prefix%4;
    }
    void append(ds4_glm5_next_mla_state &s, unsigned input_row) {
        uint32_t slot,pool; bool publish;
        REQUIRE(ds4_glm5_next_mla_append_plan(&s,&slot,&pool,&publish));
        REQUIRE(ds4_gpu_tensor_copy(s.compact_kv,(uint64_t)s.token_count*512*4,
            kv,(uint64_t)input_row*512*4,512*4));
        REQUIRE(ds4_gpu_tensor_copy(s.index_tail,slot*row_bytes,keys,input_row*row_bytes,row_bytes));
        REQUIRE(ds4_gpu_tensor_copy(s.pool_gate_tail,slot*row_bytes,gates,input_row*row_bytes,row_bytes));
        if (publish) {
            auto *out=ds4_gpu_tensor_view(s.index_pool,pool*row_bytes,row_bytes);
            REQUIRE(out && ds4_gpu_glm5_kpool_tensor(out,ids,pool_valid,
                s.index_tail,s.pool_gate_tail,valid,g.map,g.size,ape,4,width,4,0));
            ds4_gpu_tensor_free(out);
            REQUIRE(ds4_gpu_glm5_fill_pool_members_tensor(s.index_pool_ids,
                s.index_pool_valid,pool,pool*4,s.capacity_tokens));
        }
        REQUIRE(ds4_glm5_next_mla_append_commit(&s));
    }
    std::vector<uint8_t> selection(ds4_glm5_next_mla_state &s) {
        if (s.token_count <= 2048) {
            if (s.token_count) REQUIRE(ds4_gpu_glm_fill_selected_range_tensor(selected,s.token_count));
            return read(selected,s.token_count*4u);
        }
        REQUIRE(ds4_gpu_glm_indexer_score_one_tensor(scores,query,weights,
            s.index_pool,s.complete_pools,32,width,0.015625f,false));
        REQUIRE(ds4_gpu_glm5_mask_pool_scores_tensor(scores,s.index_pool_valid,s.complete_pools));
        REQUIRE(ds4_gpu_indexer_topk_tensor(selected_pools,scores,s.complete_pools,1,512));
        REQUIRE(ds4_gpu_glm5_expand_pool_selection_tensor(selected,selected_pools,
            s.index_pool_ids,s.index_pool_valid,s.index_valid_keys,s.complete_pools,
            512,s.token_count,0,s.token_count,2048,4));
        return read(selected,2051*4);
    }
};

static void run_case(Fixture &f,unsigned prefix,unsigned m,unsigned accepted,bool batch) {
    State original(0), reference(0), candidate(m);
    f.restore_inputs();
    f.seed(original.s,prefix); f.seed(reference.s,prefix); f.seed(candidate.s,prefix);
    auto &s=candidate.s;
    REQUIRE(ds4_glm5_next_mla_replay_bytes(&s)==(8u+2u*m)*row_bytes);
    ds4_glm5_next_mla_state *view=nullptr;
    REQUIRE(ds4_glm5_next_mla_verify_begin(&s,m,&view));
    REQUIRE(!ds4_glm5_next_mla_append_commit(&s));
    REQUIRE(ds4_glm5_next_mla_verify_record(view,prefix,f.keys,prefix*row_bytes,
        f.gates,prefix*row_bytes,m));
    if (batch) {
        auto *keys=ds4_gpu_tensor_view(f.keys,prefix*row_bytes,m*row_bytes);
        auto *gates=ds4_gpu_tensor_view(f.gates,prefix*row_bytes,m*row_bytes);
        REQUIRE(keys && gates && ds4_gpu_glm5_publish_pools_batch_tensor(
            view->index_pool,view->index_pool_ids,view->index_pool_valid,
            keys,gates,f.g.map,f.g.size,f.ape,prefix/4,m,width));
        ds4_gpu_tensor_free(keys); ds4_gpu_tensor_free(gates);
        REQUIRE(ds4_gpu_tensor_copy(view->compact_kv,(uint64_t)prefix*512*4,
            f.kv,(uint64_t)prefix*512*4,(uint64_t)m*512*4));
        for (unsigned i=0;i<m;++i) REQUIRE(ds4_glm5_next_mla_append_commit(view));
    } else for (unsigned i=0;i<m;++i) f.append(*view,prefix+i);
    compare_live(original.s,s); // Includes every committed pool and KV byte.
    REQUIRE(!ds4_glm5_next_mla_append_commit(view));
    for (unsigned i=0;i<accepted;++i) f.append(reference.s,prefix+i);
    const auto expected_selection=f.selection(reference.s);
    // Prove commit owns all necessary rows, including the batch path that
    // never updates its private tails. Poison all shared input workspace.
    REQUIRE(ds4_gpu_tensor_fill_f32(f.keys,NAN,capacity*width));
    REQUIRE(ds4_gpu_tensor_fill_f32(f.gates,NAN,capacity*width));
    REQUIRE(ds4_glm5_next_mla_verify_finish(&s,accepted));
    REQUIRE(!candidate.owner.pending_mla_verifications &&
        !ds4_glm5_next_mla_verify_finish(&s,accepted));
    compare_live(reference.s,s);
    poison_after(s.compact_kv,(uint64_t)s.token_count*512*4);
    poison_after(s.index_pool,s.complete_pools*row_bytes);
    poison_after(s.index_pool_ids,s.complete_pools*16u);
    poison_after(s.index_pool_valid,s.complete_pools*4u);
    REQUIRE(expected_selection==f.selection(s));
    // Rejected speculative pools remain physically present. A different
    // continuation must overwrite them before making them visible again.
    f.restore_inputs();
    for (unsigned i=0;i<5;++i) {
        f.append(reference.s,100+i); f.append(s,100+i);
        compare_live(reference.s,s);
        REQUIRE(f.selection(reference.s)==f.selection(s));
    }
    REQUIRE(ds4_gpu_synchronize());
    std::printf("MLA_REPLAY prefix=%u m=%u accepted_inputs=%u batch=%u PASS\n",
        prefix,m,accepted,unsigned(batch));
}

int main() {
    const char *model=std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model && std::getenv("DS4_RESEARCH_ROOT"));
    Glm5TestGGUF g;
    REQUIRE(g.open_file(model));
    uint64_t ape;
    REQUIRE(g.tensor("blk.3.indexer.pool_ape.weight",{128,4},30,ape));
    REQUIRE(ds4_gpu_init() && ds4_gpu_set_model_fd_for_map(g.fd,g.map) &&
        ds4_gpu_set_model_map(g.map,g.size));
    unsigned cases=0;
    {
        Fixture f(g,ape);
        for (unsigned prefix : {0u,1u,2u,3u,2044u,2045u,2046u,2047u,
                                2048u,2049u,8192u,8193u,8194u,8195u})
            for (unsigned m : {2u,4u,8u}) for (unsigned accepted=0;accepted<=m;++accepted) {
                run_case(f,prefix,m,accepted,false); ++cases;
                if (!(prefix%4) && !(m%4)) {
                    run_case(f,prefix,m,accepted,true); ++cases;
                }
            }
    }
    std::printf("PASS MLA replay cases=%u compared_live_bytes=%llu model_ape=blk.3 "
        "synthetic_inputs=1 full_target=0\n",cases,(unsigned long long)compared_bytes);
    ds4_gpu_cleanup();
    return 0;
}
