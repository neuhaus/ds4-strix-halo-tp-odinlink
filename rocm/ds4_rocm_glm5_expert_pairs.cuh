#ifndef DS4_ROCM_GLM5_EXPERT_PAIRS_CUH
#define DS4_ROCM_GLM5_EXPERT_PAIRS_CUH

#include "../ds4_glm5_expert_pairs.h"

// Original scalar body; only token/slot lookup is mapped. Keep arithmetic,
// route-weight load placement and the paired block-dot finish exact. Full
// compiled equivalence must be checked after compiler/source changes.
template <uint32_t Rows = 128u, bool BoundedDot = false>
__global__ static void glm5_mapped_scalar_gateup(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t skip_zero_weight,
        uint32_t write_aux,
        float clamp, const ds4_glm5_expert_group *group_map) {
    static_assert(Rows == 32u || Rows == 64u || Rows == 128u,
                  "decode row geometry must use complete groups of 32 rows");
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = group_map[blockIdx.y].pair[0];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    const float route_weight = weights[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0 || (skip_zero_weight && route_weight == 0.0f)) {
        for (uint32_t rr = 0; rr < Rows / 32u; rr++) {
            const uint32_t row = blockIdx.x * Rows + row_lane + rr * 32u;
            if (row >= expert_mid_dim || lane != 0u) continue;
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = 0.0f;
                up_out[off] = 0.0f;
            }
            mid_out[off] = 0.0f;
        }
        return;
    }
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < Rows / 32u; rr++) {
        uint32_t row = blockIdx.x * Rows + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            if constexpr (BoundedDot) {
                gate += dev_dot_q4_K_q8_K_block_unroll2(gr + b, xqb + b);
                up += dev_dot_q4_K_q8_K_block_unroll2(ur + b, xqb + b);
            } else {
                gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
                up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
            }
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * route_weight;
        }
    }
}


// Q4 scale/min and packed nibble word once for the two independent routes.
__device__ static void glm5_expert_dot2(const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,const cuda_block_q8_K *y1,float out[2]) {
    const float xd=dev_f16_to_f32(x->d), xmin=dev_f16_to_f32(x->dmin);
    int isum0=0,isum1=0,summs0=0,summs1=0;
    #pragma unroll
    for(uint32_t j=0;j<8u;++j) {
        uint8_t sc,m; dev_q4_K_get_scale_min(j,x->scales,&sc,&m);
        summs0+=(int)m*(int)(y0->bsums[2*j]+y0->bsums[2*j+1]);
        summs1+=(int)m*(int)(y1->bsums[2*j]+y1->bsums[2*j+1]);
        const uint32_t byte_off=(j>>1u)*32u;
        const int shift=(j&1u)?4:0;
        int dot0=0,dot1=0;
        #pragma unroll
        for(uint32_t i=0;i<32u;i+=4u) {
            const int32_t v=(*(const int32_t *)(x->qs+byte_off+i)>>shift)&0x0f0f0f0f;
            dot0=__dp4a(v,*(const int32_t *)(y0->qs+j*32u+i),dot0);
            dot1=__dp4a(v,*(const int32_t *)(y1->qs+j*32u+i),dot1);
        }
        isum0+=(int)sc*dot0; isum1+=(int)sc*dot1;
    }
    out[0]=y0->d*xd*(float)isum0-y0->d*xmin*(float)summs0;
    out[1]=y1->d*xd*(float)isum1-y1->d*xmin*(float)summs1;
}

template<uint32_t Routes>
__global__ static void glm5_grouped_gateup(float *mid_out,
        const char *gate_base,const char *up_base,const cuda_block_q8_K *xq,
        const float *weights,const ds4_glm5_expert_group *groups,
        uint64_t gate_expert_bytes,uint64_t gate_row_bytes,
        uint32_t xq_blocks,uint32_t expert_mid_dim,uint32_t n_expert,float clamp) {
    const uint32_t lane=threadIdx.x&7u, row_lane=threadIdx.x>>3u;
    const ds4_glm5_expert_group group=groups[blockIdx.y];
    const int32_t expert_i=group.expert;
    const uint32_t pair0=group.pair[0], pair1=group.pair[1];
    const float route_weight0=weights[pair0], route_weight1=weights[pair1];
    const uint32_t tok0=pair0/n_expert,tok1=pair1/n_expert;
    const cuda_block_q8_K *x0=xq+uint64_t(tok0)*xq_blocks;
    const cuda_block_q8_K *x1=xq+uint64_t(tok1)*xq_blocks;
    for(uint32_t rr=0;rr<4u;++rr) {
        const uint32_t row=blockIdx.x*128u+row_lane+rr*32u;
        if(row>=expert_mid_dim) continue;
        if(expert_i<0) {
            if(!lane) {
                mid_out[uint64_t(pair0)*expert_mid_dim+row]=0.0f;
                if constexpr(Routes==2) mid_out[uint64_t(pair1)*expert_mid_dim+row]=0.0f;
            }
            continue;
        }
        const auto *gr=(const cuda_block_q4_K *)(gate_base+uint64_t((uint32_t)expert_i)*gate_expert_bytes+uint64_t(row)*gate_row_bytes);
        const auto *ur=(const cuda_block_q4_K *)(up_base+uint64_t((uint32_t)expert_i)*gate_expert_bytes+uint64_t(row)*gate_row_bytes);
        float gate[Routes]={},up[Routes]={};
        for(uint32_t b=lane;b<xq_blocks;b+=8u) {
            if constexpr(Routes==1) {
                gate[0]+=dev_dot_q4_K_q8_K_block(gr+b,x0+b);
                up[0]+=dev_dot_q4_K_q8_K_block(ur+b,x0+b);
            } else {
                float v[2]; glm5_expert_dot2(gr+b,x0+b,x1+b,v);
                gate[0]+=v[0]; gate[1]+=v[1];
                glm5_expert_dot2(ur+b,x0+b,x1+b,v);
                up[0]+=v[0]; up[1]+=v[1];
            }
        }
        #pragma unroll
        for(uint32_t p=0;p<Routes;++p) {
            float gv=quarter_warp_sum_f32(gate[p],lane);
            float uv=quarter_warp_sum_f32(up[p],lane);
            if(lane==0) {
                if(clamp>1.0e-6f) {
                    if(gv>clamp) gv=clamp;
                    if(uv>clamp) uv=clamp;
                    if(uv<-clamp) uv=-clamp;
                }
                const uint32_t pair=p?pair1:pair0;
                const float route_weight=p?route_weight1:route_weight0;
                mid_out[uint64_t(pair)*expert_mid_dim+row]=
                    (gv/(1.0f+expf(-gv)))*uv*route_weight;
            }
        }
    }
}
#endif
