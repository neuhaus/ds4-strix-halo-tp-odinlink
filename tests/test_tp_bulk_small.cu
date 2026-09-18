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
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s leader|worker HOST PORT DEVICE GID READY(0|1)\n", argv[0]);
        return 2;
    }
    const bool leader = !std::strcmp(argv[1], "leader");
    CHECK(leader || !std::strcmp(argv[1], "worker"));
    CHECK(!std::strcmp(argv[6], "0") || !std::strcmp(argv[6], "1"));
    const bool ready = !std::strcmp(argv[6], "1");
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
    id.quant_bits = 4u; id.ctx_size = 256u;
    id.prefill_config = ready ? DS4_TP_CONFIG_BULK_RECV_READY : 0u;
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
    for (uint32_t rows : {1u, 2u, 4u, 8u}) {
        const uint32_t values = rows * 4096u;
        std::vector<double> samples;
        for (unsigned sample = 0; sample < 10; ++sample) {
            double elapsed = 0;
            for (unsigned repeat = 0; repeat < 32; ++repeat) {
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
                samples.push_back(elapsed / 32);
                std::printf("BULK_SAMPLE rank=%u ready=%u rows=%u sample=%u ms=%.6f\n",
                    rank, ready, rows, sample - 1, elapsed / 32);
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
