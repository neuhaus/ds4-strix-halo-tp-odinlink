#pragma once

// Test-only full K reduction. mode0 uses the production DP4A block helper;
// mode1 uses integer WMMA with subgroup scales, mode2 uses split scales.
hipError_t glm5_q4k_fullrows(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, float *out, unsigned n, unsigned m,
        unsigned blocks, unsigned mode);
