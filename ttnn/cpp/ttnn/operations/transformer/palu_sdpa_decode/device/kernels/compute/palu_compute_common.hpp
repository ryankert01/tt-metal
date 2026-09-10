// SPDX-License-Identifier: Apache-2.0
// Compute helpers for palu_sdpa_decode (Palu fused-K flash decode). Tile indices are relative to the
// CB read pointer. Assumes compute_kernel_hw_startup<SrcOrder::Reverse> already ran (stock does it).
#pragma once
#include "api/compute/eltwise_binary_sfpu.h"

// K_rot for one chunk of Sk_chunk_t row tiles. For each row tile i:
//   AB_i    = lat_i @ [u_k | u_k T]            (1 x Rt) @ (Rt x Wt) -> Wt tiles into cb_ab
//   K[j, i] = AB_i[j] * cs[i, j] + AB_i[j + DHt] * cs[i, j + DHt]        for j < DHt
// packed into cb_k at index j * Sk_chunk_t + i: column-major over the chunk, which is the layout the
// stock QK matmul_blocks(..., transpose=true) reads K in. cs tiles are read at
// cs_tile_base + i*Wt + j and are NOT popped (the caller decides: Phase 1 pops the streamed chunk,
// Phase 2 keeps the resident table). The latent rows are popped here; cb_w stays resident.
template <
    uint32_t cb_lat,
    uint32_t cb_w,
    uint32_t cb_ab,
    uint32_t cb_cs,
    uint32_t cb_k,
    uint32_t Rt,
    uint32_t DHt,
    uint32_t dst_tiles>
ALWI void palu_build_k_chunk(uint32_t Sk_chunk_t, uint32_t cs_tile_base) {
    constexpr uint32_t Wt = 2 * DHt;
    constexpr uint32_t ab_w = Wt > dst_tiles ? dst_tiles : Wt;  // output columns per dest acquisition
    constexpr uint32_t ab_subblocks = Wt / ab_w;
    CircularBuffer lat(cb_lat), w(cb_w), ab(cb_ab), cs(cb_cs), k(cb_k);
    w.wait_front(Rt * Wt);
    cs.wait_front(cs_tile_base + Sk_chunk_t * Wt);
    k.reserve_back(Sk_chunk_t * DHt);
    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        // ---- AB_i = lat_i @ w_rot ----
        lat.wait_front((i + 1) * Rt);
        reconfig_data_format(cb_w, cb_lat);  // matmul: in1 -> SrcA, in0 -> SrcB
        pack_reconfig_data_format(cb_ab);
        matmul_block_init(cb_lat, cb_w, false /*transpose*/, ab_w /*ct_dim*/, 1 /*rt_dim*/, Rt /*kt_dim*/);
        ab.reserve_back(Wt);
        for (uint32_t sub = 0; sub < ab_subblocks; ++sub) {
            tile_regs_acquire();
            uint32_t in0_index = i * Rt;
            uint32_t in1_index = sub * ab_w;
            for (uint32_t kt = 0; kt < Rt; ++kt) {
                matmul_block(cb_lat, cb_w, in0_index, in1_index, 0, false, ab_w, 1, Rt);
                in0_index++;
                in1_index += Wt;
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < ab_w; ++c) {
                pack_tile<true>(c, cb_ab, sub * ab_w + c);
            }
            tile_regs_release();
        }
        ab.push_back(Wt);
        // ---- K[:, i] = A_i * cos + B_i * sin ----
        ab.wait_front(Wt);
        reconfig_data_format(cb_ab, cb_cs);  // eltwise: in0 -> SrcA, in1 -> SrcB
        pack_reconfig_data_format(cb_k);
        mul_init(cb_ab, cb_cs);
        add_binary_tile_init();
        for (uint32_t j = 0; j < DHt; ++j) {
            const uint32_t c = cs_tile_base + i * Wt + j;
            tile_regs_acquire();
            mul_tiles(cb_ab, cb_cs, j, c, 0);
            mul_tiles(cb_ab, cb_cs, j + DHt, c + DHt, 1);
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_k, j * Sk_chunk_t + i);
            tile_regs_release();
        }
        ab.pop_front(Wt);
    }
    k.push_back(Sk_chunk_t * DHt);
    lat.pop_front(Sk_chunk_t * Rt);
}
