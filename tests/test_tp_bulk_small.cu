/* Real RoCE, production hello/QP/registered slab, no model. Each exchange
 * changes every payload word so a stale or partial receive cannot pass. */
#include <hip/hip_runtime.h>
extern "C" {
#include "ds4_tp.h"
}
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static uint32_t word(uint32_t seq, uint32_t rank, uint32_t index) {
    return seq * 2654435761u ^ (rank * 0xabcdef01u + index);
}

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr, "usage: %s leader|worker HOST PORT DEVICE GID READY(0|1) [legacy|native-tail|ready-mismatch|handoff|handoff-route-fail|handoff-compute-fail|handoff-stall]\n", argv[0]);
        return 2;
    }
    const bool leader = !std::strcmp(argv[1], "leader");
    CHECK(leader || !std::strcmp(argv[1], "worker"));
    CHECK(!std::strcmp(argv[6], "0") || !std::strcmp(argv[6], "1"));
    const bool ready = !std::strcmp(argv[6], "1");
    const bool mismatch = argc == 8 && !std::strcmp(argv[7], "ready-mismatch");
    const bool handoff = argc == 8 && !std::strncmp(argv[7], "handoff", 7);
    const bool route_fail = handoff && !std::strcmp(argv[7], "handoff-route-fail");
    const bool compute_fail = handoff && !std::strcmp(argv[7], "handoff-compute-fail");
    const bool stall = handoff && !std::strcmp(argv[7], "handoff-stall");
    CHECK(!handoff || ready);
    CHECK(!handoff || route_fail || compute_fail || stall || !std::strcmp(argv[7], "handoff"));
    if (handoff) CHECK(setenv("DS4_TP_TIMEOUT_SEC", "1", 1) == 0);
    const bool transition = argc == 8 && !mismatch && !handoff;
    CHECK(!mismatch || ready);
    const bool legacy = transition && !std::strcmp(argv[7], "legacy");
    CHECK(!transition || legacy || !std::strcmp(argv[7], "native-tail"));
    const int port = std::atoi(argv[3]), gid = std::atoi(argv[5]);
    CHECK(port > 0 && port < 65536 && gid >= 0);
    CHECK(setenv("DS4_TP_BIG_DIRECT", "1", 1) == 0);
    CHECK(setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", "2048", 1) == 0);
    ds4_tp_options opt = {};
    opt.requested = true;
    opt.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    opt.listen_host = leader ? argv[2] : nullptr;
    opt.listen_port = leader ? port : 0;
    opt.leader_host = leader ? nullptr : argv[2];
    opt.leader_port = leader ? 0 : port;
    opt.transport = DS4_TP_TRANSPORT_RDMA;
    opt.rdma_device = argv[4];
    opt.rdma_gid_index = gid;
    opt.rdma_gid_index_set = true;
    ds4_tp_identity id = {};
    id.gguf_bytes = 1u; id.model_id = 0x42554c4bu;
    id.n_layer = 64u; id.n_embd = 4096u; id.n_vocab = 1u;
    id.quant_bits = 4u; id.ctx_size = 1024u;
    id.prefill_config = ready ? DS4_TP_CONFIG_BULK_RECV_READY : 0u;
    if (handoff) id.prefill_config |= DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF;
    if (transition) {
        id.n_layer = 45;
        id.gate_slot_step = 1;
        id.gates_per_token = 87;
        for (unsigned slot = 0; slot < 90; ++slot)
            if (slot >= 6 || slot % 2 == 0)
                id.gate_slot_mask[slot / 64] |= UINT64_C(1) << (slot % 64);
        id.prefill_config |= UINT64_C(3) << DS4_TP_CONFIG_GLM5_NATIVE_SHIFT;
    }
    char err[512] = {};
    ds4_tp *tp = nullptr;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        std::fprintf(stderr, "FAIL create: %s\n", err); return 1;
    }
    CHECK(ds4_tp_is_rdma(tp) && ds4_tp_requires_host_slab(tp));
    void *slab = nullptr;
    CHECK(hipHostMalloc(&slab, ds4_tp_alloc_slab_bytes(tp), hipHostMallocMapped) == hipSuccess);
    CHECK(ds4_tp_attach_slab(tp, slab, err, sizeof(err)));
    CHECK(ds4_tp_big_gate_is_rdma_capable(tp));
    auto *send = (uint32_t *)((char *)slab + ds4_tp_slab_big_out_offset(tp));
    auto *recv = (uint32_t *)((char *)slab + ds4_tp_slab_big_in_offset(tp));
    const uint32_t rank = leader ? 0u : 1u;
    uint32_t seq = 0;
    if (handoff) {
        for (uint32_t rows : {2u,4u,8u}) {
            for (unsigned cycle=0;cycle<4;++cycle) {
                for (unsigned phase=0;phase<2;++phase) {
                    const bool fault=(phase==0 && route_fail) || (phase==1 && compute_fail);
                    const int ok=ds4_tp_verify_layer_agree(tp,seq,4,8192+cycle*8,rows,
                        phase,UINT64_C(0xfeed0000)+cycle,fault?leader:1,err,sizeof(err));
                    if (fault) {
                        CHECK(!ok && ds4_tp_failed(tp));
                        CHECK(!ds4_tp_big_gate_exchange(tp,4,seq+1,send,recv,rows*16384u));
                        ds4_tp_free(tp); CHECK(hipHostFree(slab)==hipSuccess);
                        std::printf("PASS handoff rejected rank=%u phase=%u no_payload=1\n",rank,phase);
                        return 0;
                    }
                    CHECK(ok);
                }
                if (stall) {
                    // Keep the peer alive with both sockets open after phase1.
                    // A 1s production deadline must release rank0 before teardown.
                    if (!leader) std::this_thread::sleep_for(std::chrono::seconds(3));
                    else {
                        const double start=now_ms();
                        CHECK(!ds4_tp_big_gate_exchange(tp,4,seq+1,send,recv,rows*16384u));
                        const double elapsed=now_ms()-start;
                        CHECK(ds4_tp_failed(tp) && elapsed>=750 && elapsed<2500);
                        CHECK(!ds4_tp_big_gate_exchange(tp,4,seq+2,send,recv,rows*16384u));
                        std::printf("HANDOFF_STALL rank=0 elapsed_ms=%.3f bounded=1\n",elapsed);
                    }
                    ds4_tp_free(tp); CHECK(hipHostFree(slab)==hipSuccess);
                    std::printf("PASS handoff post-compute stall rank=%u\n",rank); return 0;
                }
                ++seq;
                for (uint32_t i=0;i<rows*4096;++i) send[i]=word(seq,rank,i);
                std::memset(recv,0,rows*16384u);
                std::atomic_thread_fence(std::memory_order_release);
                CHECK(ds4_tp_big_gate_exchange(tp,4,seq,send,recv,rows*16384u));
                std::atomic_thread_fence(std::memory_order_acquire);
                for (uint32_t i=0;i<rows*4096;++i) CHECK(recv[i]==word(seq,rank^1u,i));
            }
        }
        CHECK(!ds4_tp_failed(tp));
        ds4_tp_free(tp); CHECK(hipHostFree(slab)==hipSuccess);
        std::printf("PASS handoff RoCE rank=%u exchanges=%u rows=2/4/8 changed_payload=1\n",rank,seq);
        return 0;
    }
    if (mismatch) {
        // Both ranks arm32 live receive WRs. Last-chunk size disagreement
        // must poison/park the QP and close both sockets without any SEND.
        const uint64_t bytes = UINT64_C(4194304) - rank * 16384u;
        CHECK(!ds4_tp_big_gate_exchange(tp, 0, 1, send, recv, bytes));
        CHECK(ds4_tp_failed(tp));
        CHECK(!ds4_tp_big_gate_exchange(tp, 0, 2, send, recv, bytes));
        ds4_tp_free(tp);
        CHECK(hipHostFree(slab) == hipSuccess);
        std::printf("PASS bulk ready mismatch rank=%u armed_recvs=32 reentry_refused=1 teardown=1\n", rank);
        return 0;
    }
    if (transition) {
        /* Reproduce native draft/bulk -> ordinary tail -> bulk -> ordinary.
         * Bulk headers deliberately advance independently of the latency ring. */
        for (unsigned cycle = 0; cycle < 3; ++cycle) {
            for (unsigned draft_gate = 0; draft_gate < 14; ++draft_gate) {
                ++seq;
                for (unsigned i = 0; i < 4096; ++i) send[i] = word(seq, rank, i);
                CHECK(ds4_tp_big_gate_exchange(tp, 45, seq, send, recv, 16384));
                for (unsigned i = 0; i < 4096; ++i) CHECK(recv[i] == word(seq, rank ^ 1u, i));
            }
            for (unsigned slot = 0; slot < 90; ++slot) {
                if (slot < 6 && slot % 2) continue; // leading FFNs are replicated
                ++seq;
                for (unsigned i = 0; i < 4096; ++i) send[i] = word(seq, rank, i);
                CHECK(legacy ? ds4_tp_gate_exchange_from_registered(tp, slot / 2, slot % 2, seq, send) :
                    ds4_tp_native_gate_exchange_next(tp, slot / 2, slot % 2, send));
                const auto *received = (const uint32_t *)((const char *)slab +
                    ds4_tp_slab_in_offset(tp, slot / 2, slot % 2));
                for (unsigned i = 0; i < 4096; ++i) CHECK(received[i] == word(seq, rank ^ 1u, i));
            }
        }
        CHECK(!ds4_tp_failed(tp));
        ds4_tp_free(tp);
        CHECK(hipHostFree(slab) == hipSuccess);
        std::printf("PASS bulk native-tail rank=%u ready=%u cycles=3 changed_payload=1\n", rank, ready);
        return 0;
    }
    for (uint32_t rows : {1u, 2u, 4u, 8u, 256u, 257u, 512u, 513u}) {
        const uint32_t values = rows * 4096u;
        const unsigned repeats = rows <= 8u ? 32u : 2u;
        std::vector<double> samples;
        for (unsigned sample = 0; sample < 10; ++sample) {
            double elapsed = 0;
            for (unsigned repeat = 0; repeat < repeats; ++repeat) {
                ++seq;
                for (uint32_t i = 0; i < values; ++i) send[i] = word(seq, rank, i);
                std::memset(recv, 0, values * sizeof(uint32_t));
                std::atomic_thread_fence(std::memory_order_release);
                const double begin = now_ms();
                CHECK(ds4_tp_big_gate_exchange(tp, 0u, seq, send, recv, values * sizeof(uint32_t)));
                elapsed += now_ms() - begin;
                std::atomic_thread_fence(std::memory_order_acquire);
                for (uint32_t i = 0; i < values; ++i) CHECK(recv[i] == word(seq, rank ^ 1u, i));
            }
            if (sample) {
                samples.push_back(elapsed / repeats);
                std::printf("BULK_SAMPLE rank=%u ready=%u rows=%u sample=%u ms=%.6f\n",
                    rank, ready, rows, sample - 1, elapsed / repeats);
            }
        }
        std::sort(samples.begin(), samples.end());
        std::printf("BULK_MEDIAN rank=%u ready=%u rows=%u ms=%.6f\n", rank, ready, rows, samples[4]);
        std::fflush(stdout);
    }
    CHECK(!ds4_tp_failed(tp));
    ds4_tp_free(tp);
    CHECK(hipHostFree(slab) == hipSuccess);
    std::printf("PASS bulk small rank=%u ready=%u exchanges=%u changed_payload=1\n", rank, ready, seq);
    return 0;
}
