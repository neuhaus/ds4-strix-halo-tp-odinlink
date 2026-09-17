#pragma once

struct Q4KBlockResult {
    int32_t dot;
    int32_t minimum;
    float value;
};

hipError_t glm5_q4k_integer_blocks(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, Q4KBlockResult *out, unsigned n, unsigned m,
        bool matrix, bool trace);
