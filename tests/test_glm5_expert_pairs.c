#include "ds4_glm5_expert_pairs.h"
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line=%d: %s\n", __LINE__, #x); return 1; } } while (0)
int main(void) {
    unsigned cases = 0;
    int32_t ids[48]; float weights[48]; ds4_glm5_expert_groups g;
    const unsigned unique[] = {8, 10, 12, 16, 34, 44, 46, 48};
    for (unsigned u = 0; u < sizeof(unique)/sizeof(unique[0]); ++u) {
        unsigned counts[288] = {0}, pairs = 0;
        for (unsigned i = 0; i < 48; ++i) {
            ids[i] = (i % unique[u] + 267) % 288;
            ++counts[ids[i]]; weights[i] = i == 0 ? -0.0f : 0.031973f + i * 0.00317239f;
        }
        for (unsigned i = 0; i < 288; ++i) pairs += counts[i] / 2;
        for (unsigned mode = 1; mode <= 2; ++mode) {
            CHECK(ds4_glm5_expert_groups_build(&g, ids, weights, mode));
            CHECK(g.doubles == (mode == 2 ? pairs : 0));
            CHECK(g.singles + 2*g.doubles == 48);
            CHECK(ds4_glm5_expert_groups_valid(&g, ids, weights)); ++cases;
        }
    }
    for (unsigned i = 0; i < 48; ++i) ids[i] = i % 34;
    CHECK(ds4_glm5_expert_groups_build(&g, ids, weights, 2));
    for (unsigned bad = 0; bad < 12; ++bad) {
        ds4_glm5_expert_groups v = g;
        switch (bad) {
        case 0: v.singles = 49; break;
        case 1: v.doubles = 25; break;
        case 2: v.groups[0].count = 0; break;
        case 3: v.groups[0].count = 2; break;
        case 4: v.groups[0].pair[0] = 48; break;
        case 5: v.groups[0].pair[1] = 48; break;
        case 6: v.groups[0].expert = -1; break;
        case 7: v.groups[0].expert = 288; break;
        case 8: v.groups[0].expert = 287; break;
        case 9: v.groups[v.singles].pair[1] = v.groups[v.singles].pair[0]; break;
        case 10: v.groups[v.singles].count = 1; break;
        case 11: v.groups[1] = v.groups[0]; break;
        }
        CHECK(!ds4_glm5_expert_groups_valid(&v, ids, weights)); ++cases;
    }
    const int32_t invalid_ids[] = {-1, -2, 288, 1};
    for (unsigned i = 0; i < 4; ++i) {
        ids[0] = invalid_ids[i];
        CHECK(!ds4_glm5_expert_groups_build(&g, ids, weights, 2)); ++cases;
    }
    ids[0] = 0;
    const float invalid_weights[] = {-1.0f, INFINITY, -INFINITY, NAN};
    for (unsigned i = 0; i < 4; ++i) {
        weights[0] = invalid_weights[i];
        CHECK(!ds4_glm5_expert_groups_build(&g, ids, weights, 2)); ++cases;
    }
    weights[0] = 0;
    CHECK(!ds4_glm5_expert_groups_build(&g, ids, weights, 0));
    CHECK(!ds4_glm5_expert_groups_build(&g, ids, weights, 3));
    CHECK(!ds4_glm5_expert_groups_build(&g, NULL, weights, 2));
    CHECK(!ds4_glm5_expert_groups_build(&g, ids, NULL, 2));
    CHECK(!ds4_glm5_expert_groups_build(NULL, ids, weights, 2));
    printf("PASS expert descriptors cases=%u exact48routes negative_ids_refused=1\n", cases + 5);
    return 0;
}
