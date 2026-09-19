// Fixture-only observation of the real handoff completion fence and packed
// expert calls. Event spans include intra-call launch gaps; they are not a
// hardware-active-time counter. Not linked into either production executable.
#include <hip/hip_runtime_api.h>

static ds4_tp *queue_test_peer;
static bool queue_gpu_profile, queue_timed_sample;
static hipEvent_t queue_events[8][2];
static unsigned queue_event_count;
static std::chrono::steady_clock::time_point queue_loop_start;
struct QueueStat { unsigned calls=0, rows=0, fences=0; double wall_ms=0, packed_ms=0; };
static QueueStat queue_stats[2][45];

static void queue_probe_init() {
    const char *value=std::getenv("DS4_TEST_FFN_GPU_PROFILE");
    REQUIRE(!value || !std::strcmp(value,"0") || !std::strcmp(value,"1"));
    queue_gpu_profile=value && !std::strcmp(value,"1");
    if (queue_gpu_profile) for (auto &row:queue_events) for (auto &event:row)
        REQUIRE(hipEventCreate(&event)==hipSuccess);
}
static bool queue_probe_active() {
    return queue_gpu_profile && queue_timed_sample && queue_test_peer &&
        queue_test_peer->handoff_pending==1;
}
static void queue_probe_begin(ds4_tp *p) {
    p->handoff_fences=0;
    p->packed_calls=0;
    queue_event_count=0;
    if (queue_probe_active()) queue_loop_start=std::chrono::steady_clock::now();
}
static void queue_probe_end(ds4_tp *p) {
    if (!queue_probe_active()) return;
    const double wall=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-queue_loop_start).count();
    REQUIRE(queue_event_count==p->handoff_rows);
    const unsigned mode=(p->prefill_config & DS4_TP_CONFIG_GLM5_VERIFY_FFN_QUEUE)?1u:0u;
    auto &s=queue_stats[mode][p->handoff_layer];
    s.calls++; s.rows+=p->handoff_rows; s.fences+=p->handoff_fences; s.wall_ms+=wall;
    for (unsigned i=0;i<queue_event_count;++i) {
        // No event synchronization: the engine must have completed its fence.
        REQUIRE(hipEventQuery(queue_events[i][1])==hipSuccess);
        float ms=0;
        REQUIRE(hipEventElapsedTime(&ms,queue_events[i][0],queue_events[i][1])==hipSuccess);
        s.packed_ms+=ms;
    }
}
static void queue_probe_finish() {
    if (!queue_gpu_profile) return;
    for (unsigned mode=0;mode<2;++mode) for (unsigned il=0;il<45;++il) {
        const auto &s=queue_stats[mode][il];
        if (s.calls) std::printf("FFN_QUEUE_PROFILE rank=%u mode=%u layer=%u calls=%u rows=%u fences=%u loop_wall_ms=%.6f packed_device_span_ms=%.6f\n",
            queue_test_peer->rank,mode,il,s.calls,s.rows,s.fences,s.wall_ms,s.packed_ms);
    }
    for (auto &row:queue_events) for (auto &event:row)
        REQUIRE(hipEventDestroy(event)==hipSuccess);
}

extern "C" {
decltype(ds4_gpu_synchronize) __real_ds4_gpu_synchronize;
int __wrap_ds4_gpu_synchronize() {
    const int completed=__real_ds4_gpu_synchronize();
    if (queue_test_peer && queue_test_peer->observe_attn &&
        (queue_test_peer->attn_outputs || queue_test_peer->attn_prepares)) {
        ++queue_test_peer->attn_drains;
        if (queue_test_peer->fail_attn_completion) {
            queue_test_peer->fail_attn_completion=false;
            return 0;
        }
    }
    if (queue_test_peer && queue_test_peer->handoff_pending==1) {
        queue_test_peer->handoff_fences++;
        if (queue_test_peer->fail_completion) {
            queue_test_peer->fail_completion=false;
            return 0; // Fault injection after actual completion, no GPU corruption.
        }
    }
    return completed;
}
decltype(ds4_gpu_matmul_q8_0_kslice_tensor) __real_ds4_gpu_matmul_q8_0_kslice_tensor;
int __wrap_ds4_gpu_matmul_q8_0_kslice_tensor(ds4_gpu_tensor *out,
        const void *map, uint64_t size, uint64_t offset, uint64_t in_dim,
        uint64_t start, uint64_t count, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t input_start) {
    const int ok=__real_ds4_gpu_matmul_q8_0_kslice_tensor(out,map,size,offset,
        in_dim,start,count,out_dim,x,input_start);
    if (queue_test_peer && queue_test_peer->observe_attn &&
        ++queue_test_peer->attn_outputs==queue_test_peer->fail_attn_output) {
        REQUIRE(ok);
        queue_test_peer->fail_attn_output=0;
        return 0; // Real GPU submission happened; the caller must drain it.
    }
    return ok;
}
decltype(ds4_gpu_matmul_q8_0_tensor) __real_ds4_gpu_matmul_q8_0_tensor;
int __wrap_ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *map,
        uint64_t size, uint64_t offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t rows) {
    const int ok=__real_ds4_gpu_matmul_q8_0_tensor(out,map,size,offset,in_dim,out_dim,x,rows);
    if (queue_test_peer && queue_test_peer->observe_attn && in_dim==4096 && out_dim==1536 && rows==1 &&
        ++queue_test_peer->attn_prepares==queue_test_peer->fail_attn_prepare) {
        REQUIRE(ok);
        queue_test_peer->fail_attn_prepare=0;
        return 0;
    }
    return ok;
}
decltype(ds4_gpu_routed_moe_one_packed_q4k_tensor) __real_ds4_gpu_routed_moe_one_packed_q4k_tensor;
int __wrap_ds4_gpu_routed_moe_one_packed_q4k_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *map, uint64_t size,
        uint64_t gate_off, uint64_t up_off, uint64_t down_off, uint32_t total,
        uint64_t gate_row, uint64_t down_row, uint32_t row_base, uint32_t row_count,
        uint64_t column_base, uint64_t column_count, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights, uint32_t used, float clamp,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *add, uint32_t layer) {
    const bool observe=queue_probe_active();
    const unsigned slot=queue_event_count;
    if (observe) {
        REQUIRE(slot<8 && layer==queue_test_peer->handoff_layer);
        REQUIRE(hipEventRecord(queue_events[slot][0],0)==hipSuccess);
    }
    const int ok=__real_ds4_gpu_routed_moe_one_packed_q4k_tensor(
        out,gate,up,mid,down,map,size,gate_off,up_off,down_off,total,gate_row,
        down_row,row_base,row_count,column_base,column_count,selected,weights,
        used,clamp,x,add,layer);
    if (queue_test_peer && queue_test_peer->handoff_pending==1 &&
        ++queue_test_peer->packed_calls==queue_test_peer->fail_enqueue_call) {
        REQUIRE(ok && !observe);
        queue_test_peer->fail_enqueue_call=0;
        // Real work was queued: exercise draining after partial submission.
        return 0;
    }
    if (observe) {
        REQUIRE(ok && hipEventRecord(queue_events[slot][1],0)==hipSuccess);
        queue_event_count++;
    }
    return ok;
}
}
