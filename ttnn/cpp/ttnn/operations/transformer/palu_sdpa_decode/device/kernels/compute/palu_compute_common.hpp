// SPDX-License-Identifier: Apache-2.0
// Compute helpers for palu_sdpa_decode (Palu fused-K flash decode). Tile indices are relative to the
// CB read pointer. Assumes compute_kernel_hw_startup<SrcOrder::Reverse> already ran (stock does it).
#pragma once
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"

// K rebuild mode (stage 2e): 0 legacy (SFPU add in dst), 1 pack (packer L1 accumulate), 2 dst.
// The program factory always sets it; the default keeps this header standalone-compilable.
#ifndef PALU_ROT_MODE
#define PALU_ROT_MODE 0
#endif

// K_rot for one chunk of Sk_chunk_t row tiles. For each row tile i:
//   AB_i    = lat_i @ [u_k | u_k T]            (1 x Rt) @ (Rt x Wt) -> Wt tiles into cb_ab
//   K[j, i] = AB_i[j] * cs[i, j] + AB_i[j + DHt] * cs[i, j + DHt]        for j < DHt
// packed into cb_k at index j * Sk_chunk_t + i: column-major over the chunk, which is the layout the
// stock QK matmul_blocks(..., transpose=true) reads K in. cs tiles are read at
// cs_tile_base + i*Wt + j and are NOT popped (the caller decides: Phase 1 pops the streamed chunk,
// Phase 2 keeps the resident table). cb_w stays resident and this head's block starts at
// w_block_base (group-shared mode keeps the core's H blocks resident; 0 otherwise). The latent rows
// are popped here unless pop_latent is false -- the group-shared caller pops the shared chunk once,
// after the last head of the core has rebuilt its K from it.
template <
    uint32_t cb_lat,
    uint32_t cb_w,
    uint32_t cb_ab,
    uint32_t cb_cs,
    uint32_t cb_k,
    uint32_t Rt,
    uint32_t DHt,
    uint32_t dst_tiles,
    bool pop_latent = true>
ALWI void palu_build_k_chunk(uint32_t Sk_chunk_t, uint32_t cs_tile_base, uint32_t w_block_base = 0) {
    constexpr uint32_t Wt = 2 * DHt;
    constexpr uint32_t ab_w = Wt > dst_tiles ? dst_tiles : Wt;  // output columns per dest acquisition
    constexpr uint32_t ab_subblocks = Wt / ab_w;
    CircularBuffer lat(cb_lat), w(cb_w), ab(cb_ab), cs(cb_cs), k(cb_k);
    w.wait_front(w_block_base + Rt * Wt);
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
            uint32_t in1_index = w_block_base + sub * ab_w;
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
#if PALU_ROT_MODE == 1
        // pack mode (stage 2e): two column tiles per DST trip, the packer's L1 accumulate does the add.
        // dst0 = A_j cos_j, dst1 = B_j sin_j, dst2 = A_j+1 cos_j+1, dst3 = B_j+1 sin_j+1; the A products
        // are packed with accumulate off, the B products onto the same L1 tiles with it on.
        static_assert(DHt % 2 == 0 && dst_tiles >= 4, "pack mode rotates two column tiles per DST trip");
        for (uint32_t j = 0; j < DHt; j += 2) {
            const uint32_t c = cs_tile_base + i * Wt + j;
            tile_regs_acquire();
            mul_tiles(cb_ab, cb_cs, j, c, 0);
            mul_tiles(cb_ab, cb_cs, j + DHt, c + DHt, 1);
            mul_tiles(cb_ab, cb_cs, j + 1, c + 1, 2);
            mul_tiles(cb_ab, cb_cs, j + 1 + DHt, c + 1 + DHt, 3);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(0);
            pack_tile<true>(0, cb_k, j * Sk_chunk_t + i);
            pack_tile<true>(2, cb_k, (j + 1) * Sk_chunk_t + i);
            pack_reconfig_l1_acc(1);
            pack_tile<true>(1, cb_k, j * Sk_chunk_t + i);
            pack_tile<true>(3, cb_k, (j + 1) * Sk_chunk_t + i);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
        }
#else
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
#endif
        ab.pop_front(Wt);
    }
    k.push_back(Sk_chunk_t * DHt);
    if constexpr (pop_latent) {
        lat.pop_front(Sk_chunk_t * Rt);
    }
}

// ---------------- Group-shared mode: FIFO rotation of a per-head state block ----------------
// Rotate a state CB by `blocks` whole blocks of `tiles` tiles WITHOUT touching the data. Only valid
// when the CB is exactly full (produced == capacity): the write pointer then sits on the front block,
// so pop_front + reserve_back + push_back re-publishes the very bytes that were popped, at the back of
// the FIFO. The group-shared kernel holds one state block per head in each state CB and relies on
// every head running the identical push/pop sequence per chunk to get its own block back at the front;
// where an in-place helper (mul_block_inplace, mul_block_bcast_cols) rotates the block away mid-step
// and a later read still expects it at the front, this closes the ring back up (H - 1 more blocks).
ALWI void palu_rotate_blocks(uint32_t cb_id, uint32_t tiles, uint32_t blocks) {
    CircularBuffer cb(cb_id);
    for (uint32_t i = 0; i < blocks; ++i) {
        cb.wait_front(tiles);
        cb.pop_front(tiles);
        cb.reserve_back(tiles);
        cb.push_back(tiles);
    }
}

// ---------------- Phase 2: the query state (spec §5) ----------------
// cb_q holds the rotated query as ONE block of 2n tiles (n = PNHt*DHt), fp32:
//     [ q~ | q~ T ]        with q~ = q R_{-k c} after k steps.
// "fp32" here is the CB's tile format, not the multiply's actual precision: Wormhole's FPU unpacks
// fp32 operands into 19-bit Src registers (~11 mantissa bits per multiply, i.e. close to bf16), so
// each palu_step_q update carries noticeably less than full fp32 precision. The state is still kept
// in fp32 tiles because that is what avoids compounding rounding error additively across steps
// (16 steps at 8k drift by well under a bf16 ulp per spec §5); it does not buy full fp32 accuracy
// per step.
// Keeping q~ T alongside q~ turns the per-chunk rotation into a plain complex multiply -- no matmul
// per step -- because T is the pair rotation (T^2 = -I) and cos/sin are constant across a pair:
//     q~'   = q~ cos(c f) - (q~ T) sin(c f)
//     q~' T = (q~ T) cos(c f) + q~ sin(c f)
// The block is the CB's push/pop granule, and cb_q is sized 4n so that the block being read and the
// block being written are each contiguous: an operand tile index is an offset from the read pointer
// and does NOT wrap at the end of the CB.

// state <- [Q | Q T]: copy the (already position-rotated) bf16 Q in, and multiply it by the trans_mat
// tile that rides at index t_idx of cb_w (T is block-diagonal per 32 columns, so one tile does all of them).
template <uint32_t cb_q_in, uint32_t cb_q, uint32_t cb_w, uint32_t n, uint32_t w_tiles>
ALWI void palu_init_q_state(uint32_t t_idx) {
    CircularBuffer src(cb_q_in), q(cb_q), w(cb_w);
    src.wait_front(n);
    w.wait_front(w_tiles);
    q.reserve_back(2 * n);
    pack_reconfig_data_format(cb_q);
    reconfig_data_format_srca(cb_q_in);
    copy_init(cb_q_in);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile(cb_q_in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_q, i);
        tile_regs_release();
    }
    reconfig_data_format(cb_w, cb_q_in);  // matmul: in1 -> SrcA, in0 -> SrcB
    matmul_init(cb_q_in, cb_w);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        matmul_tiles(cb_q_in, cb_w, i, t_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_q, n + i);
        tile_regs_release();
    }
    q.push_back(2 * n);
}

// One chunk step: state <- state R_{-c}. cb_step holds 2*DHt fp32 tiles [cos(c f) | sin(c f)] with
// every row identical, so a plain mul_tiles applies the per-column constant to all 32 rows.
template <uint32_t cb_q, uint32_t cb_step, uint32_t DHt, uint32_t PNHt>
ALWI void palu_step_q() {
    constexpr uint32_t n = PNHt * DHt;
    CircularBuffer q(cb_q), step(cb_step);
    q.wait_front(2 * n);
    step.wait_front(2 * DHt);
    q.reserve_back(2 * n);
    reconfig_data_format(cb_q, cb_step);  // eltwise: in0 -> SrcA, in1 -> SrcB
    pack_reconfig_data_format(cb_q);
    mul_init(cb_q, cb_step);
    // q~' = q~ * cos_c - (q~ T) * sin_c
    sub_binary_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t j = i % DHt;  // column tile: the frequency block this tile carries
        tile_regs_acquire();
        mul_tiles(cb_q, cb_step, i, j, 0);
        mul_tiles(cb_q, cb_step, n + i, DHt + j, 1);
        sub_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_q, i);
        tile_regs_release();
    }
    // q~' T = (q~ T) * cos_c + q~ * sin_c
    add_binary_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t j = i % DHt;
        tile_regs_acquire();
        mul_tiles(cb_q, cb_step, n + i, j, 0);
        mul_tiles(cb_q, cb_step, i, DHt + j, 1);
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_q, n + i);
        tile_regs_release();
    }
    q.push_back(2 * n);
    q.pop_front(2 * n);
}

// ---------------- In-kernel V (Palu G-LRD, spec §4) ----------------
// V_h for one chunk from the V latent and this head's u_v slice, row tile by row tile, packed into
// cb_v_out in ROW-MAJOR tile order (k * vDHt + n) -- the layout the PV matmul_blocks(..., transpose=false)
// reads in1 in. u_v tiles sit at uv_tile_base + kt*vDHt + n inside the resident weight block (cb_w, not
// popped; in group-shared mode the caller adds this head's block base to uv_tile_base). The V latent
// rows are popped here unless pop_latent is false -- the group-shared caller pops the shared chunk
// once, after the last head of the core.
template <
    uint32_t cb_vlat,
    uint32_t cb_w,
    uint32_t cb_v_out,
    uint32_t Rvt,
    uint32_t vDHt,
    uint32_t dst_tiles,
    bool pop_latent = true>
ALWI void palu_build_v_chunk(uint32_t Sk_chunk_t, uint32_t uv_tile_base) {
    constexpr uint32_t sub_w = vDHt > dst_tiles ? dst_tiles : vDHt;
    constexpr uint32_t subblocks = vDHt / sub_w;
    CircularBuffer vlat(cb_vlat), w(cb_w), vo(cb_v_out);
    w.wait_front(uv_tile_base + Rvt * vDHt);
    vo.reserve_back(Sk_chunk_t * vDHt);
    reconfig_data_format(cb_w, cb_vlat);  // matmul: in1 -> SrcA, in0 -> SrcB
    pack_reconfig_data_format(cb_v_out);
    matmul_block_init(cb_vlat, cb_w, false /*transpose*/, sub_w /*ct_dim*/, 1 /*rt_dim*/, Rvt /*kt_dim*/);
    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        vlat.wait_front((i + 1) * Rvt);
        for (uint32_t sub = 0; sub < subblocks; ++sub) {
            tile_regs_acquire();
            uint32_t in0_index = i * Rvt;
            uint32_t in1_index = uv_tile_base + sub * sub_w;
            for (uint32_t kt = 0; kt < Rvt; ++kt) {
                matmul_block(cb_vlat, cb_w, in0_index, in1_index, 0, false, sub_w, 1, Rvt);
                in0_index++;
                in1_index += vDHt;
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < sub_w; ++c) {
                pack_tile<true>(c, cb_v_out, i * vDHt + sub * sub_w + c);
            }
            tile_regs_release();
        }
    }
    vo.push_back(Sk_chunk_t * vDHt);
    if constexpr (pop_latent) {
        vlat.pop_front(Sk_chunk_t * Rvt);
    }
}
