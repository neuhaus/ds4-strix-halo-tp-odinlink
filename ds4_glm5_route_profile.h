#ifndef DS4_GLM5_ROUTE_PROFILE_H
#define DS4_GLM5_ROUTE_PROFILE_H

#include <stdint.h>
#include <string.h>

/* CPU diagnostics only. Input is the already-read, original route IDs.
 * Counts include every proposed row, regardless of later acceptance. */
typedef struct {
    uint32_t unique2, unique4, unique_all;
    uint32_t multiplicity[8];
} ds4_glm5_route_profile;

static inline int ds4_glm5_route_profile_count(const int32_t *ids,
        uint32_t rows, ds4_glm5_route_profile *out) {
    if (!ids || !out || (rows != 2u && rows != 4u && rows != 6u && rows != 8u))
        return 0;
    uint8_t counts[288] = {0};
    ds4_glm5_route_profile result = {0};
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t i = 0; i < 8u; ++i) {
            const int32_t id = ids[t * 8u + i];
            if (id < 0 || id >= 288) return 0;
            for (uint32_t j = 0; j < i; ++j)
                if (ids[t * 8u + j] == id) return 0;
            if (!counts[id]++) ++result.unique_all;
        }
        if (t == 1u) result.unique2 = result.unique_all;
        if (t == 3u) result.unique4 = result.unique_all;
    }
    for (uint32_t i = 0; i < 288u; ++i)
        if (counts[i]) ++result.multiplicity[counts[i] - 1u];
    *out = result;
    return 1;
}

#endif
