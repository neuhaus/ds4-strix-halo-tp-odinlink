/* Real socket control protocol; no GPU, tensor exchange, or RDMA emulation. */
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
static const ds4_tp_native_cycle base = {17u, 23u, 8192u, 8u, 123, -1};
static unsigned cases;

typedef struct {
    ds4_tp *tp;
    ds4_tp_native_cycle cycle;
    uint32_t phase, accepted, tokens[8];
    uint32_t layer_mode, layer;
    uint64_t route_hash;
    int ok, result;
    char error[128];
} arm;

static void *agree(void *arg) {
    arm *a = arg;
    a->result = a->layer_mode ? ds4_tp_verify_layer_agree(a->tp,
        a->cycle.cycle, a->layer, a->cycle.prefix, a->cycle.rows, a->phase,
        a->route_hash, a->ok, a->error, sizeof(a->error)) :
        ds4_tp_native_agree(a->tp, &a->cycle, a->phase, a->accepted,
                           a->tokens, a->ok, a->error, sizeof(a->error));
    return NULL;
}

static void pair(arm *a, arm *b, int want) {
    int sockets[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    a->tp = ds4_tp_test_control_create(sockets[0], 0);
    b->tp = ds4_tp_test_control_create(sockets[1], 1);
    CHECK(a->tp && b->tp);
    pthread_t peer; CHECK(pthread_create(&peer, NULL, agree, b) == 0);
    agree(a); CHECK(pthread_join(peer, NULL) == 0);
    CHECK(a->result == want && b->result == want);
    CHECK(ds4_tp_failed(a->tp) == !want && ds4_tp_failed(b->tp) == !want);
    if (!want) CHECK(a->error[0] && b->error[0]);
    ds4_tp_test_control_destroy(a->tp); ds4_tp_test_control_destroy(b->tp);
    ++cases;
}

static void agreement_cases(void) {
    for (uint32_t rows = 2u; rows <= 8u; rows *= 2u) {
        for (uint32_t phase = 0; phase <= 9u; ++phase) {
            if (phase >= rows && phase < 8u) continue;
            arm a = {.cycle = base, .phase = phase, .ok = 1};
            a.cycle.rows = rows;
            a.accepted = phase >= 8u ? rows : 0u;
            for (unsigned i = 0; i < rows; ++i) a.tokens[i] = 100 + i;
            arm b = a; pair(&a, &b, 1);
        }
    }
    for (unsigned fault = 0; fault < 15u; ++fault) {
        arm a = {.cycle = base, .phase = 8u, .accepted = 3u, .tokens = {123, 456}, .ok = 1};
        arm b = a;
        switch (fault) {
        case 0: ++b.cycle.session_id; break;
        case 1: ++b.cycle.cycle; break;
        case 2: ++b.cycle.prefix; break;
        case 3: b.cycle.rows = 4; break;
        case 4: ++b.cycle.root; break;
        case 5: b.cycle.eos = 9; break;
        case 6: ++b.phase; break;
        case 7: ++b.accepted; break;
        case 8: ++b.tokens[0]; break;
        case 9: ++b.tokens[7]; break;
        case 10: b.ok = 0; break;
        case 11: a.ok = b.ok = 0; break;
        case 12: b.cycle.rows = 3; break;
        case 13: b.phase = 10; break;
        case 14: b.accepted = 9; break;
        }
        pair(&a, &b, 0);
    }
}

static void layer_cases(void) {
    for (uint32_t rows = 2u; rows <= 8u; rows *= 2u)
        for (uint32_t phase = 0u; phase < 2u; ++phase) {
            arm a = {.cycle=base, .phase=phase, .ok=1,
                     .layer_mode=1, .layer=4, .route_hash=UINT64_C(0xfedcba9876543210)};
            a.cycle.rows = rows;
            arm b = a; pair(&a, &b, 1);
        }
    for (unsigned fault = 0; fault < 12u; ++fault) {
        arm a = {.cycle=base, .ok=1, .layer_mode=1, .layer=4, .route_hash=12345};
        arm b = a;
        switch (fault) {
        case 0: ++b.cycle.cycle; break;
        case 1: ++b.cycle.prefix; break;
        case 2: b.cycle.rows=4; break;
        case 3: ++b.layer; break;
        case 4: b.phase=1; break;
        case 5: ++b.route_hash; break;
        case 6: b.ok=0; break;
        case 7: a.ok=0; break;
        case 8: b.layer=45; break;
        case 9: b.cycle.rows=3; break;
        case 10: b.phase=2; break;
        case 11: b.layer_mode=0; break;
        }
        pair(&a, &b, 0);
    }
}

static void command_cases(void) {
    for (unsigned fault = 0; fault < 10u; ++fault) {
        int sockets[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        ds4_tp *a = ds4_tp_test_control_create(sockets[0], 0);
        ds4_tp *b = ds4_tp_test_control_create(sockets[1], 1);
        CHECK(a && b);
        ds4_tp_native_cycle c = base;
        switch (fault) {
        case 1: c.session_id = 0; break;
        case 2: c.cycle = 0; break;
        case 3: c.prefix = 0; break;
        case 4: c.rows = 3; break;
        case 5: c.prefix = UINT32_MAX; break;
        case 6: c.root = -1; break;
        case 7: c.eos = -2; break;
        case 8: c.rows = 2; break;
        case 9: c.rows = 4; break;
        }
        const int want = fault == 0 || fault >= 8;
        CHECK(ds4_tp_send_native_cycle(a, &c) == want);
        ds4_tp_command command; char error[128] = "";
        CHECK(ds4_tp_recv_command(b, &command, error, sizeof(error)) == want);
        if (want) {
            CHECK(command.type == DS4_TP_FRAME_GLM5_NATIVE && command.session_id == c.session_id);
            CHECK(memcmp(&command.native, &c, sizeof(c)) == 0);
            CHECK(!command.tokens && !command.n_tokens && !command.items && !command.n_items);
            ds4_tp_command_free(&command);
        } else CHECK(ds4_tp_failed(a));
        ds4_tp_test_control_destroy(a); ds4_tp_test_control_destroy(b);
        ++cases;
    }
}

static void malformed_cases(void) {
    for (unsigned fault = 0; fault < 7u; ++fault) {
        int sockets[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        arm a = {.cycle = base, .phase = 0, .ok = 1};
        a.tp = ds4_tp_test_control_create(sockets[0], 0); CHECK(a.tp);
        struct { uint32_t magic, type, bytes; } h = {0x44533454u, DS4_TP_FRAME_GLM5_NATIVE_AGREE, 80u};
        unsigned char body[80] = {0};
        if (fault == 0) h.magic = 0;
        if (fault == 1) h.type = DS4_TP_FRAME_EVAL;
        if (fault == 2) h.bytes = 79;
        if (fault == 3) h.bytes = UINT32_MAX;
        if (fault != 6) {
            const size_t header_bytes = fault == 4 ? sizeof(h) - 1 : sizeof(h);
            CHECK(send(sockets[1], &h, header_bytes, 0) == (ssize_t)header_bytes);
            if (fault == 5) CHECK(send(sockets[1], body, 7, 0) == 7);
            CHECK(shutdown(sockets[1], SHUT_WR) == 0);
        } else { close(sockets[1]); sockets[1] = -1; }
        agree(&a); CHECK(!a.result && ds4_tp_failed(a.tp) && a.error[0]);
        ds4_tp_test_control_destroy(a.tp);
        if (sockets[1] >= 0) close(sockets[1]);
        ++cases;
    }
}

static void config_cases(void) {
    const char *values[] = {NULL, "", "0", "2", "4", "8", "1", "3", "16", "02", "-2", "2 ", " 2", "x"};
    const uint32_t expected[] = {0, 0, 0, 2, 4, 8, UINT32_MAX, UINT32_MAX, UINT32_MAX,
        UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        CHECK(ds4_tp_glm5_native_rows_parse(values[i]) == expected[i]); ++cases;
    }
    for (uint64_t a = 0; a < 4; ++a) for (uint64_t b = 0; b < 4; ++b) {
        const uint64_t ac = a << DS4_TP_CONFIG_GLM5_NATIVE_SHIFT;
        const uint64_t bc = b << DS4_TP_CONFIG_GLM5_NATIVE_SHIFT;
        char error[128] = "";
        CHECK(ds4_tp_glm5_native_rows(ac) == (a ? 1u << a : 0u));
        CHECK(ds4_tp_test_hello_validate_prefill_config(ac, bc, error, sizeof(error)) == (a == b));
        ++cases;
    }
}

/* Run both 30-second production deadlines together: silent reader and writer
 * backpressure. No shortened test-only timeout can hide an unbounded send. */
static void stalled_cases(void) {
    int sockets[4][2]; arm a[4] = {0};
    pthread_t threads[4]; struct timespec start, end;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &start) == 0);
    for (unsigned i = 0; i < 4; ++i) {
        a[i] = (arm){.cycle=base, .ok=1, .layer_mode=i/2u, .layer=4};
        CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets[i]) == 0);
        if (i & 1u) {
            char data[4096] = {0}; ssize_t sent;
            do { sent = send(sockets[i][0], data, sizeof(data), MSG_DONTWAIT); } while (sent > 0);
            CHECK(sent == -1 && (errno == EAGAIN || errno == EWOULDBLOCK));
        }
        a[i].tp = ds4_tp_test_control_create(sockets[i][0], 0); CHECK(a[i].tp);
        CHECK(pthread_create(&threads[i], NULL, agree, &a[i]) == 0);
    }
    for (unsigned i = 0; i < 4; ++i) {
        CHECK(pthread_join(threads[i], NULL) == 0);
        CHECK(!a[i].result && ds4_tp_failed(a[i].tp));
        ds4_tp_test_control_destroy(a[i].tp); close(sockets[i][1]); ++cases;
    }
    CHECK(clock_gettime(CLOCK_MONOTONIC, &end) == 0);
    const double elapsed = (double)(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    CHECK(elapsed >= 29.0 && elapsed < 40.0);
    printf("PASS silent/backpressured peers elapsed=%.3f seconds\n", elapsed);
}

int main(void) {
    ds4_tp_test_reset_exchange_calls();
    config_cases(); command_cases(); agreement_cases(); layer_cases(); malformed_cases(); stalled_cases();
    CHECK(ds4_tp_test_get_exchange_calls() == 0);
    printf("PASS native control cases=%u tensor_exchange_calls=0 (socket control only)\n", cases);
    return 0;
}
