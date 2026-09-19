#include <stdio.h>
#include "ds4_glm5_route_profile.h"

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    int32_t ids[64];
    ds4_glm5_route_profile p;
    const uint32_t widths[] = {2, 4, 6, 8};
    unsigned cases = 0;
    for (unsigned w = 0; w < 4; ++w) {
        const unsigned rows = widths[w];
        for (unsigned t = 0; t < rows; ++t) for (unsigned i = 0; i < 8; ++i)
            ids[t * 8 + i] = (int32_t)((i + t) % 8); // Same experts, different slot order.
        REQUIRE(ds4_glm5_route_profile_count(ids, rows, &p));
        REQUIRE(p.unique2 == 8 && p.unique4 == (rows >= 4 ? 8u : 0u));
        REQUIRE(p.unique_all == 8 && p.multiplicity[rows - 1] == 8);
        ++cases;
        for (unsigned i = 0; i < rows * 8; ++i) ids[i] = (int32_t)i;
        REQUIRE(ds4_glm5_route_profile_count(ids, rows, &p));
        REQUIRE(p.unique2 == 16 && p.unique4 == (rows >= 4 ? 32u : 0u));
        REQUIRE(p.unique_all == rows * 8 && p.multiplicity[0] == rows * 8);
        ++cases;
    }
    // Two fixed experts per row, six distinct others: U2=14,U4=26,U6=38.
    for (unsigned t = 0; t < 6; ++t) for (unsigned i = 0; i < 8; ++i)
        ids[t * 8 + i] = (int32_t)(i < 2 ? i : 2 + t * 6 + i - 2);
    REQUIRE(ds4_glm5_route_profile_count(ids, 6, &p));
    REQUIRE(p.unique2 == 14 && p.unique4 == 26 && p.unique_all == 38);
    REQUIRE(p.multiplicity[0] == 36 && p.multiplicity[5] == 2);
    ++cases;
    unsigned pairs = 0, unique = 0;
    for (unsigned i = 0; i < 8; ++i) { pairs += (i + 1) * p.multiplicity[i]; unique += p.multiplicity[i]; }
    REQUIRE(pairs == 48 && unique == p.unique_all);
    ds4_glm5_route_profile before = p;
    ids[47] = 288;
    REQUIRE(!ds4_glm5_route_profile_count(ids, 6, &p) && !memcmp(&p, &before, sizeof(p)));
    ids[47] = -1;
    REQUIRE(!ds4_glm5_route_profile_count(ids, 6, &p) && !memcmp(&p, &before, sizeof(p)));
    ids[47] = ids[40];
    REQUIRE(!ds4_glm5_route_profile_count(ids, 6, &p) && !memcmp(&p, &before, sizeof(p)));
    REQUIRE(!ds4_glm5_route_profile_count(NULL, 6, &p) && !memcmp(&p, &before, sizeof(p)));
    REQUIRE(!ds4_glm5_route_profile_count(ids, 6, NULL));
    const unsigned invalid_widths[] = {0, 1, 3, 9};
    for (unsigned i = 0; i < 4; ++i)
        REQUIRE(!ds4_glm5_route_profile_count(ids, invalid_widths[i], &p) &&
                !memcmp(&p, &before, sizeof(p)));
    cases += 9;
    printf("PASS route profile cases=%u CPU-only, no model inference\n", cases);
    return 0;
}
