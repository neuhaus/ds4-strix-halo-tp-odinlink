#ifndef DS4_ROCM_GLM5_EXPERT_PAIRS_LAUNCH_CUH
#define DS4_ROCM_GLM5_EXPERT_PAIRS_LAUNCH_CUH

static int glm5_expert_six_modes_valid(void) {
    if (!cuda_q4k_kshard_enabled() || routed_moe_glm5_decode_gate_rows() != 128 ||
        routed_moe_glm5_decode_dot_unroll() != 0 || routed_moe_glm5_decode_dot_lanes() != 0)
        return 0;
    /* Admission checks literal values as well as dispatch. No silent coercion
     * of malformed selectors, changed arithmetic or zero-weight skipping. */
    const char *off[] = {"DS4_ROCM_TP_SKIP_UNOWNED", "DS4_ROCM_Q4K_DECODE_STAGE_XQ",
        "DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP", "DS4_ROCM_Q4K_DECODE_STAGE_MIDQ",
        "DS4_ROCM_Q4K_DECODE_FUSE_ADDEND", "DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP",
        "DS4_ROCM_Q4K_WMMA_FUSE_MID", "DS4_ROCM_GLM5_Q4K_DECODE_DOT_UNROLL",
        "DS4_ROCM_GLM5_Q4K_DECODE_DOT_LANES", "DS4_ROCM_DISABLE_Q4K_WMMA"};
    for (const char *name : off) {
        const char *value = getenv(name);
        if (value && strcmp(value, "0")) return 0;
    }
    const char *rows = getenv("DS4_ROCM_GLM5_Q4K_DECODE_GATE_ROWS");
    const char *wmma = getenv("DS4_ROCM_Q4K_WMMA");
    return (!rows || !strcmp(rows, "128")) && (!wmma || !strcmp(wmma, "1"));
}

extern "C" int ds4_rocm_glm5_expert_six_admit(ds4_glm5_expert_six_plan *p,
        const ds4_glm5_expert_six_args *a) {
    if (!p) return 0;
    memset(p, 0, sizeof(*p));
    if (!a || !a->model_map || !a->model_size || a->rank > 1 || a->rows != 6 ||
        !glm5_expert_six_modes_valid()) return 0;
    static_assert(sizeof(ds4_glm5_expert_group) == 16, "descriptor upload layout");
    static_assert(sizeof(cuda_block_q8_K) == 292, "Q8_K activation layout");
    static_assert(DS4_ROCM_N_EXPERT_USED == 8, "down kernel slot order");
    ds4_gpu_tensor *buffers[] = {a->out, a->mid, a->input_q8, a->mid_q8,
        a->descriptors, a->input, a->selected, a->weights};
    const uint64_t needed[] = {6*4096*4, 48*1024*4, 6*16*292, 48*4*292,
        48*16, 6*4096*4, 48*4, 48*4};
    uintptr_t begin[11]; uint64_t bytes[11];
    for (unsigned i = 0; i < 8; ++i) {
        if (!buffers[i] || !buffers[i]->ptr || buffers[i]->bytes < needed[i]) return 0;
        begin[i] = (uintptr_t)buffers[i]->ptr; bytes[i] = buffers[i]->bytes;
        if ((begin[i] & 3u) || bytes[i] > UINTPTR_MAX - begin[i]) return 0;
    }
    const uint64_t offsets[] = {a->gate_offset, a->up_offset, a->down_offset};
    const void *packed[3] = {};
    for (unsigned i = 0; i < 3; ++i) {
        const bool down = i == 2;
        const uint64_t full_bytes = UINT64_C(288) * 2048 * 2304;
        if (offsets[i] > a->model_size || full_bytes > a->model_size - offsets[i]) return 0;
        uint64_t size = 0, expert_bytes = 0, row_bytes = 0;
        if (!ds4_gpu_q4k_packed_slice_resolve(a->model_map, offsets[i], 288,
                down ? 4096 : 2048, down ? 1152 : 2304,
                down ? 0 : a->rank * 1024, down ? 4096 : 1024,
                down ? a->rank * 576 : 0, down ? 576 : 2304,
                down ? DS4_GPU_Q4K_PACKED_K_RANGE : DS4_GPU_Q4K_PACKED_ROW_RANGE,
                &packed[i], &size, &expert_bytes, &row_bytes) || !packed[i] ||
            size != UINT64_C(288) * 2359296 || expert_bytes != 2359296 ||
            row_bytes != (down ? 576u : 2304u)) return 0;
        begin[8+i] = (uintptr_t)packed[i]; bytes[8+i] = size;
        if ((begin[8+i] & 3u) || size > UINTPTR_MAX - begin[8+i]) return 0;
    }
    for (unsigned i = 0; i < 11; ++i) for (unsigned j = 0; j < i; ++j)
        if (begin[i] < begin[j] + bytes[j] && begin[j] < begin[i] + bytes[i]) return 0;
    p->args = *a;
    memcpy(p->packed, packed, sizeof(packed));
    return 1;
}

extern "C" int ds4_rocm_glm5_expert_six_begin(const ds4_glm5_expert_six_plan *p,
        const ds4_glm5_expert_groups *g, const int32_t *ids, const float *weights) {
    if (!p || p->args.rows != 6 || !p->packed[0] || !p->packed[1] || !p->packed[2] ||
        !ds4_glm5_expert_groups_valid(g, ids, weights)) return 0;
    const ds4_glm5_expert_six_args &a = p->args;
    if (!ds4_gpu_tensor_write(a.descriptors, 0, g->groups, sizeof(g->groups))) return 0;
    q8_K_quantize_kernel<<<dim3(16, 6), 256>>>(
        (cuda_block_q8_K *)a.input_q8->ptr, (const float *)a.input->ptr, 4096, 6);
    if (cudaGetLastError() != cudaSuccess) return 0;
    const auto *groups = (const ds4_glm5_expert_group *)a.descriptors->ptr;
    if (g->singles) {
        glm5_mapped_scalar_gateup<128, false><<<dim3(8, g->singles), 256>>>(
            nullptr, nullptr, (float *)a.mid->ptr, (const char *)p->packed[0],
            (const char *)p->packed[1], (const cuda_block_q8_K *)a.input_q8->ptr,
            (const int32_t *)a.selected->ptr, (const float *)a.weights->ptr,
            2359296, 2304, 16, 1024, 8, 0, 0, 10.0f, groups);
        if (cudaGetLastError() != cudaSuccess) return 0;
    }
    if (g->doubles) {
        glm5_grouped_gateup<2><<<dim3(8, g->doubles), 256>>>(
            (float *)a.mid->ptr, (const char *)p->packed[0], (const char *)p->packed[1],
            (const cuda_block_q8_K *)a.input_q8->ptr, (const float *)a.weights->ptr,
            groups + g->singles, 2359296, 2304, 16, 1024, 8, 10.0f);
        if (cudaGetLastError() != cudaSuccess) return 0;
    }
    q8_K_quantize_kernel<<<dim3(4, 48), 256>>>(
        (cuda_block_q8_K *)a.mid_q8->ptr, (const float *)a.mid->ptr, 1024, 48);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int ds4_rocm_glm5_expert_six_down_row(const ds4_glm5_expert_six_plan *p,
        uint32_t row) {
    if (!p || p->args.rows != 6 || !p->packed[2] || row >= 6) return 0;
    const ds4_glm5_expert_six_args &a = p->args;
    /* Original compiled down kernel, original slot accumulation order. The
     * caller adds the shared row and optionally fences before the next row. */
    moe_down_q4K_sum6_halfk4_kernel<<<64, 256>>>(
        (float *)a.out->ptr + row * 4096, nullptr, (const char *)p->packed[2],
        (const cuda_block_q8_K *)a.mid_q8->ptr + row * 8 * 4,
        (const int32_t *)a.selected->ptr + row * 8, (const float *)a.weights->ptr + row * 8,
        2359296, 576, 4096, 8, 0);
    return cudaGetLastError() == cudaSuccess;
}
#endif
