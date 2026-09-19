#ifndef DS4_GLM5_ROUTE_PROFILE_H
#define DS4_GLM5_ROUTE_PROFILE_H

#include <stdint.h>
#include <string.h>

#define DS4_GLM5_ROUTE_PROFILE_EXPERTS 288u
#define DS4_GLM5_ROUTE_PROFILE_USED 8u

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
    uint8_t counts[DS4_GLM5_ROUTE_PROFILE_EXPERTS] = {0};
    ds4_glm5_route_profile result = {0};
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t i = 0; i < DS4_GLM5_ROUTE_PROFILE_USED; ++i) {
            const int32_t id = ids[t * DS4_GLM5_ROUTE_PROFILE_USED + i];
            if (id < 0 || (uint32_t)id >= DS4_GLM5_ROUTE_PROFILE_EXPERTS) return 0;
            for (uint32_t j = 0; j < i; ++j)
                if (ids[t * DS4_GLM5_ROUTE_PROFILE_USED + j] == id) return 0;
            if (!counts[id]++) ++result.unique_all;
        }
        if (t == 1u) result.unique2 = result.unique_all;
        if (t == 3u) result.unique4 = result.unique_all;
    }
    for (uint32_t i = 0; i < DS4_GLM5_ROUTE_PROFILE_EXPERTS; ++i)
        if (counts[i]) ++result.multiplicity[counts[i] - 1u];
    *out = result;
    return 1;
}

#endif
