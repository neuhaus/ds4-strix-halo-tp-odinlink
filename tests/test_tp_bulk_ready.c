/* The same metadata rendezvous used after production ibv_post_recv.
 * No test manufactures an RDMA payload capability. */
#include "ds4_tp.h"
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static unsigned cases;
static double now(void) {
    struct timespec t; CHECK(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return t.tv_sec + t.tv_nsec / 1e9;
}
typedef struct {
    ds4_tp *tp;
    uint32_t chunks;
    uint64_t bytes, offset, round_bytes;
    int result;
} arm;
static void *exchange(void *arg) {
    arm *a = arg;
    a->result = ds4_tp_test_bulk_ready(a->tp, a->chunks, a->bytes,
                                      a->offset, a->round_bytes, 100);
    return NULL;
}
static void pairs(void) {
    for (unsigned fault = 0; fault < 5; ++fault) {
        int fd[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        arm a = {.tp = ds4_tp_test_control_create(fd[0], 0), .chunks = 32,
                 .bytes = 8388608, .offset = 4194304, .round_bytes = 4194304};
        arm b = a; b.tp = ds4_tp_test_control_create(fd[1], 1);
        CHECK(a.tp && b.tp);
        if (fault == 1) --b.chunks;
        if (fault == 2) ++b.bytes;
        if (fault == 3) b.offset = 0;
        if (fault == 4) --b.round_bytes;
        pthread_t peer; CHECK(pthread_create(&peer, NULL, exchange, &b) == 0);
        exchange(&a); CHECK(pthread_join(peer, NULL) == 0);
        CHECK(a.result == !fault && b.result == !fault);
        CHECK(ds4_tp_failed(a.tp) == !!fault && ds4_tp_failed(b.tp) == !!fault);
        ds4_tp_test_control_destroy(a.tp); ds4_tp_test_control_destroy(b.tp);
        ++cases;
    }
}
static void faults(unsigned first, unsigned last) {
    for (unsigned fault = first; fault < last; ++fault) {
        int fd[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        arm a = {.tp = ds4_tp_test_control_create(fd[0], 0), .chunks = 1,
                 .bytes = 16384, .round_bytes = 16384};
        CHECK(a.tp);
        if (fault == 1) { close(fd[1]); fd[1] = -1; }
        if (fault == 2) CHECK(shutdown(fd[1], SHUT_WR) == 0);
        if (fault == 3 || fault == 4) {
            /* Truncated reply, either EOF or a live silent writer. */
            uint32_t partial = 0x44533442u;
            CHECK(send(fd[1], &partial, sizeof(partial), 0) == sizeof(partial));
            if (fault == 3) CHECK(shutdown(fd[1], SHUT_WR) == 0);
        }
        if (fault == 5) {
            const int bufsize = 4096;
            CHECK(setsockopt(fd[0], SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize)) == 0);
            char data[4096] = {0};
            while (send(fd[0], data, sizeof(data), MSG_DONTWAIT | MSG_NOSIGNAL) > 0) {}
            CHECK(errno == EAGAIN || errno == EWOULDBLOCK);
        }
        if (fault == 6) {
            const unsigned char malformed[32] = {0};
            CHECK(send(fd[1], malformed, sizeof(malformed), 0) == sizeof(malformed));
        }
        const double begin = now(); exchange(&a); const double elapsed = now() - begin;
        CHECK(!a.result && ds4_tp_failed(a.tp));
        CHECK(elapsed < 1.0);
        if (fault == 0 || fault == 4 || fault == 5) CHECK(elapsed >= 0.075);
        /* Once failed, a repeated gate must reject without waiting or sending. */
        const double retry = now(); exchange(&a);
        CHECK(!a.result && now() - retry < 0.05);
        ds4_tp_test_control_destroy(a.tp);
        if (fd[1] >= 0) close(fd[1]);
        ++cases;
    }
}
static void separate_sockets(void) {
    int control[2], data[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, control) == 0);
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, data) == 0);
    ds4_tp *tp = ds4_tp_test_bulk_ready_create(control[0], data[0]); CHECK(tp);
    CHECK(!ds4_tp_test_bulk_ready(tp, 1, 16384, 0, 16384, 100));
    CHECK(ds4_tp_failed(tp));
    unsigned char record[32];
    CHECK(recv(data[1], record, sizeof(record), MSG_DONTWAIT) == sizeof(record));
    CHECK(recv(data[1], record, 1, MSG_DONTWAIT) == 0);
    CHECK(recv(control[1], record, 1, MSG_DONTWAIT) == 0);
    ds4_tp_test_control_destroy(tp); close(data[1]); close(control[1]);
    ++cases;
}
static void handoff_gate_silent(void) {
    int control[2], data[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, control) == 0);
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, data) == 0);
    ds4_tp *tp=ds4_tp_test_handoff_gate_create(control[0],data[0]); CHECK(tp);
    unsigned char out[8]={0}, in[8]={0};
    const double start=now();
    CHECK(!ds4_tp_big_gate_exchange(tp,4,1,out,in,sizeof(out)));
    const double elapsed=now()-start;
    CHECK(ds4_tp_failed(tp) && elapsed>=0.75 && elapsed<2.0);
    CHECK(!ds4_tp_big_gate_exchange(tp,4,2,out,in,sizeof(out)));
    CHECK(recv(control[1],in,1,MSG_DONTWAIT)==0);
    ds4_tp_test_control_destroy(tp); close(control[1]); close(data[1]);
    ++cases;
    printf("PASS post-agreement bulk header deadline elapsed=%.3f seconds\n",elapsed);
}
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--silent")) faults(0, 1);
    else if (argc == 2 && !strcmp(argv[1], "--handoff-silent")) handoff_gate_silent();
    else { pairs(); faults(0, 7); separate_sockets(); handoff_gate_silent(); }
    printf("PASS bulk ready socket cases=%u\n", cases);
    return 0;
}
