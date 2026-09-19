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
    uint64_t route_hash, config;
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
    ds4_tp_test_control_set_config(a->tp, a->config);
    ds4_tp_test_control_set_config(b->tp, b->config);
    pthread_t peer; CHECK(pthread_create(&peer, NULL, agree, b) == 0);
    agree(a); CHECK(pthread_join(peer, NULL) == 0);
    CHECK(a->result == want && b->result == want);
    CHECK(ds4_tp_failed(a->tp) == !want && ds4_tp_failed(b->tp) == !want);
    if (!want) CHECK(a->error[0] && b->error[0]);
    ds4_tp_test_control_destroy(a->tp); ds4_tp_test_control_destroy(b->tp);
    ++cases;
}

static void agreement_cases(void) {
    for (uint32_t rows = 2u; rows <= 8u; rows += 2u) {
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
    for (uint32_t rows = 2u; rows <= 8u; rows += 2u)
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
    const uint64_t attn = DS4_TP_CONFIG_GLM5_VERIFY_MLA_ATTN_HANDOFF |
        DS4_TP_CONFIG_GLM5_VERIFY_MLA_FFN_HANDOFF | DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF;
    for (uint32_t rows = 2; rows <= 6; rows += 2) for (uint32_t maximum=rows; maximum<=6; maximum+=2) {
        arm a = {.cycle=base, .phase=2, .ok=1, .layer_mode=1, .layer=3,
            .config=ds4_tp_glm5_native_config(maximum) | attn};
        a.cycle.rows=rows;
        arm b=a; pair(&a,&b,1);
    }
    for (unsigned fault=0; fault<12; ++fault) {
        arm a = {.cycle=base, .phase=2, .ok=1, .layer_mode=1, .layer=3,
            .config=ds4_tp_glm5_native_config(6) | attn};
        a.cycle.rows=6;
        arm b=a;
        switch (fault) {
        case 0: a.ok=0; break;
        case 1: b.ok=0; break;
        case 2: b.phase=0; break;
        case 3: b.cycle.rows=4; break;
        case 4: ++b.cycle.prefix; break;
        case 5: b.config &= ~DS4_TP_CONFIG_GLM5_VERIFY_MLA_ATTN_HANDOFF; break;
        case 6: b.config=ds4_tp_glm5_native_config(2)|attn; break;
        case 7: b.layer=4; break;
        case 8: b.cycle.rows=8; break;
        case 9: b.phase=3; break;
        case 10: b.config &= ~DS4_TP_CONFIG_GLM5_VERIFY_MLA_FFN_HANDOFF; break;
        case 11: b.config &= ~DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF; break;
        }
        pair(&a,&b,0);
    }
}

static void command_cases(void) {
    for (unsigned fault = 0; fault < 11u; ++fault) {
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
        case 10: c.rows = 6; break;
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
    const char *pair_values[] = {NULL, "0", "1", "2", "", "3", "8", "01", "2 ", "-1", "x"};
    for (unsigned i = 0; i < sizeof(pair_values)/sizeof(pair_values[0]); ++i) {
        CHECK(ds4_tp_glm5_expert_pairs_parse(pair_values[i]) ==
            (i < 4 ? (i ? i-1 : 0u) : UINT32_MAX)); ++cases;
    }
    for (unsigned rows = 0; rows <= 8; rows += 2) for (unsigned prereq = 0; prereq < 4; ++prereq)
        for (unsigned mode = 0; mode <= 3; ++mode) {
            const uint64_t required = (prereq & 1 ? DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF : 0u) |
                (prereq & 2 ? DS4_TP_CONFIG_GLM5_VERIFY_MLA_FFN_HANDOFF : 0u);
            const uint64_t config = ds4_tp_glm5_native_config(rows) | required |
                ds4_tp_glm5_expert_pairs_config(mode);
            char error[128];
            const int valid = mode == 0 || (mode <= 2 && rows == 6 && prereq == 3);
            CHECK(ds4_tp_glm5_expert_pairs_mode(config) == mode);
            CHECK(ds4_tp_glm5_expert_pairs_config_valid(config) == valid);
            const int all_valid = valid && ds4_tp_glm5_mla_handoff_config_valid(config);
            CHECK(ds4_tp_test_hello_validate_prefill_config(config, config, error, sizeof(error)) == all_valid);
            for (unsigned other = 0; other <= 3; ++other) {
                const uint64_t remote = ds4_tp_glm5_native_config(rows) | required |
                    ds4_tp_glm5_expert_pairs_config(other);
                CHECK(ds4_tp_test_hello_validate_prefill_config(config, remote, error, sizeof(error)) ==
                    (all_valid && mode == other));
            }
            ++cases;
        }
    CHECK(ds4_tp_glm5_handoff_modes_valid(NULL,NULL,NULL,NULL));
    CHECK(ds4_tp_glm5_handoff_modes_valid("0","1",NULL,NULL));
    CHECK(ds4_tp_glm5_handoff_modes_valid("0","0","1","1"));
    CHECK(!ds4_tp_glm5_handoff_modes_valid("0","1","1","1"));
    const char *bad_modes[] = {"", "2", "01", "1 ", "invalid"};
    for (unsigned i=0;i<sizeof(bad_modes)/sizeof(bad_modes[0]);++i) {
        CHECK(!ds4_tp_glm5_handoff_modes_valid(bad_modes[i],"0",NULL,NULL));
        CHECK(!ds4_tp_glm5_handoff_modes_valid("0",bad_modes[i],NULL,NULL));
        cases+=2;
    }
    CHECK(!ds4_tp_glm5_handoff_modes_valid("1","0",NULL,NULL));
    cases+=5;
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
    CHECK(ds4_tp_glm5_native_rows_parse("6") == 6u);
    const uint32_t widths[] = {0u, 2u, 4u, 6u, 8u};
    const uint64_t transport = DS4_TP_CONFIG_BULK_RECV_READY | DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF;
    for (unsigned a = 0; a < 5; ++a) for (unsigned b = 0; b < 5; ++b) {
        const uint64_t ac = ds4_tp_glm5_native_config(widths[a]) | transport;
        const uint64_t bc = ds4_tp_glm5_native_config(widths[b]) | transport;
        char error[128];
        CHECK(ds4_tp_glm5_native_rows(ac) == widths[a]);
        CHECK((ds4_tp_glm5_native_config(widths[a]) & transport) == 0u);
        CHECK(ds4_tp_test_hello_validate_prefill_config(ac, bc, error, sizeof(error)) == (a == b));
        ++cases;
    }
    for (uint64_t code = 1u; code <= 3u; ++code) {
        const uint64_t bad = DS4_TP_CONFIG_GLM5_NATIVE_SIX | (code << DS4_TP_CONFIG_GLM5_NATIVE_SHIFT);
        char error[128];
        CHECK(ds4_tp_glm5_native_rows(bad) == UINT32_MAX);
        CHECK(!ds4_tp_test_hello_validate_prefill_config(bad, bad, error, sizeof(error)));
        CHECK(!ds4_tp_test_hello_validate_prefill_config(bad, transport, error, sizeof(error)));
        ++cases;
    }
    CHECK(ds4_tp_glm5_native_rows(ds4_tp_glm5_native_config(UINT32_MAX)) == UINT32_MAX);
    for (unsigned width = 0; width < 5; ++width) for (unsigned handoff = 0; handoff < 2; ++handoff) {
        const uint64_t base_config = ds4_tp_glm5_native_config(widths[width]) |
            (handoff ? DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF : 0u);
        const uint64_t with_mla = base_config | DS4_TP_CONFIG_GLM5_VERIFY_MLA_FFN_HANDOFF;
        char error[128];
        CHECK(ds4_tp_test_hello_validate_prefill_config(with_mla, with_mla, error, sizeof(error)) ==
            (width != 0 && handoff != 0));
        CHECK(!ds4_tp_test_hello_validate_prefill_config(base_config, with_mla, error, sizeof(error)));
        CHECK(!ds4_tp_test_hello_validate_prefill_config(with_mla, base_config, error, sizeof(error)));
        ++cases;
    }
    const uint32_t tails[] = {0u, 1u, 2u, 2u, 4u, 4u, 6u, 6u, 6u};
    for (unsigned width = 0; width < 5; ++width) for (unsigned handoff = 0; handoff < 2; ++handoff) {
        const uint64_t base_config = ds4_tp_glm5_native_config(widths[width]) |
            (handoff ? DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF : 0u);
        const uint64_t queued = base_config | DS4_TP_CONFIG_GLM5_VERIFY_FFN_QUEUE;
        char error[128];
        CHECK(ds4_tp_test_hello_validate_prefill_config(queued, queued, error, sizeof(error)) ==
            (width != 0 && handoff != 0));
        CHECK(!ds4_tp_test_hello_validate_prefill_config(base_config, queued, error, sizeof(error)));
        CHECK(!ds4_tp_test_hello_validate_prefill_config(queued, base_config, error, sizeof(error)));
        ++cases;
    }
    for (unsigned width=0; width<=8; width+=2) for (unsigned prerequisites=0; prerequisites<4; ++prerequisites) {
        uint64_t config=ds4_tp_glm5_native_config(width) |
            DS4_TP_CONFIG_GLM5_VERIFY_MLA_ATTN_HANDOFF;
        if (prerequisites&1) config |= DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF;
        if (prerequisites&2) config |= DS4_TP_CONFIG_GLM5_VERIFY_MLA_FFN_HANDOFF;
        char error[128];
        CHECK(ds4_tp_test_hello_validate_prefill_config(config,config,error,sizeof(error)) ==
            (width>=2 && width<=6 && prerequisites==3));
        CHECK(!ds4_tp_test_hello_validate_prefill_config(config,
            config & ~DS4_TP_CONFIG_GLM5_VERIFY_MLA_ATTN_HANDOFF,error,sizeof(error)));
        ++cases;
    }
    for (unsigned i = 0; i < 9; ++i) {
        CHECK(ds4_tp_glm5_native_cycle_rows(6u, i) == tails[i]); ++cases;
    }
    CHECK(ds4_tp_glm5_native_cycle_rows(8u, 6u) == 4u);
    CHECK(ds4_tp_glm5_native_cycle_rows(3u, 8u) == 0u);
    const uint32_t six_slots[] = {2u, 4u, 6u, 0u};
    for (unsigned i = 0; i < 4; ++i) {
        CHECK(ds4_tp_glm5_native_workspace_rows(6u, i) == six_slots[i]); ++cases;
    }
    for (unsigned layer = 0; layer < 2; ++layer) for (unsigned peer_rows = 4; peer_rows <= 8; peer_rows += 4) {
        arm a = {.cycle=base, .ok=1, .layer_mode=layer, .layer=4};
        a.cycle.rows=6; arm b=a; b.cycle.rows=peer_rows;
        pair(&a,&b,0);
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
