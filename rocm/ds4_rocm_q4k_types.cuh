#pragma once

// Original GGUF Q4_K weight and runtime Q8_K activation layouts. Shared with
// isolated arithmetic probes so their reference uses the production ABI.
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[256 / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[256];
    int16_t bsums[256 / 16];
} cuda_block_q8_K;

static_assert(sizeof(cuda_block_q4_K) == 144, "Q4_K GGUF block ABI");
static_assert(sizeof(cuda_block_q8_K) == 292, "Q8_K activation block ABI");
