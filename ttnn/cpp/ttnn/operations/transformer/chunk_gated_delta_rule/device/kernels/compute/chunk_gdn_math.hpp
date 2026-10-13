// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Shared math functions for the chunk_gdn prep and scan compute kernels.
// The kernel .cpp files are calling prep_chunk / scan_step below. CB ids are passed in as plain
// uint32_t (via the GdnPrepCbs / GdnScanCbs structs for the composition functions), so the same
// bodies run under different CB maps.
//

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
// SFPU bodies and the pass runner the gdn_sfpu passes below use directly.
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_sqrt.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

// GDN_HOIST_RECONFIG (a per-kernel define, set by the fused factory on the producer compute
// kernel only): hoist the packer/unpacker format reconfigs out of the WY hot path (invert16 /
// invert_block).
// It is a per-path PERF switch: the hoist helps on the fused producer,
// but hurts the phased kernel at some shapes.
#ifdef GDN_HOIST_RECONFIG
inline constexpr bool kGdnHoistReconfig = true;
#else
inline constexpr bool kGdnHoistReconfig = false;
#endif

// Sub-step device zones for the Tracy device profiler. Only in profiled builds where the
// profiler header was included before this one.
#if defined(PROFILE_KERNEL) && defined(DeviceZoneScopedN)
#define GDN_ZONE(name) DeviceZoneScopedN(name)
#else
#define GDN_ZONE(name)
#endif

#if defined(GDN_TINV_SFPU)
#include "api/compute/triangle_solve.h"
#endif

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
// Ct is a template parameter of prep_chunk, so every per-tile loop in the helpers below unrolls fully; at Ct == 2
// that puts the prep program over the 70,656 B kernel-config buffer. The Ct == 2 build passes the tile count
// through this identity-asm so the compiler treats it as a runtime value and compiles each loop body once
// (the Ct == 1 build keeps the constant: its loops are single iterations anyway).
template <uint32_t Ct>
inline uint32_t tiles_dim(uint32_t v) {
    if constexpr (Ct > 1) {
        asm volatile("" : "+r"(v));
    }
    return v;
}
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
// Output tiles per DST acquire. With fp32 accumulation DST holds 8 tiles and the math/pack half-sync
// gives each side 4, so four independent output tiles ride one acquire/commit/wait/release round trip
// instead of four.
#ifndef GDN_DST_TILES
#define GDN_DST_TILES 1  // the prep kernel keeps the per-tile form: its Ct=2 binary sits at the 70,656 B limit
#endif
inline constexpr uint32_t kDstTiles = GDN_DST_TILES;

// DST[idst + c] += a[a_tile] @ b[b_tile + c] for c < n: matmul_tiles on n consecutive in1 tiles of one in0 tile, with
// in0 unpacked ONCE (it stays in SrcB while the n in1 tiles stream through SrcA) instead of once per product. Each
// DST tile receives exactly the MVMUL sequence matmul_tiles issues for it, so the sums are bit-identical. As for
// matmul_tiles a matmul_init must precede it, and b's tile size must be the unpacker's configured SrcA tile size
// (the unpacker steps from one in1 tile to the next by it). This is matmul_tiles's own unpack / math pair given a
// column count, not matmul_block: on Blackhole that one switches to the firmware-throttled MOP on request, whose
// reordered fidelity phases would make the sums depend on the throttle state.
inline void mm_tiles_row(uint32_t a, uint32_t b, uint32_t a_tile, uint32_t b_tile, uint32_t idst, uint32_t n) {
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_AB_matmul(a, b, a_tile, b_tile, n /*ct_dim*/, 1 /*rt_dim*/, 1 /*kt_dim*/)));
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, n /*ct_dim*/, 1 /*rt_dim*/)));
#else
    for (uint32_t c = 0; c < n; c++) {
        matmul_tiles(a, b, a_tile, b_tile + c, idst + c);
    }
#endif
}

// Split the DST batch of output tiles [t0, t0 + nb) of an [M, Nt] tile grid (row-major) into its runs of tiles
// that share an output row, and call f(mi, ni, j, n) for each: the run starts at grid tile (mi, ni), DST tile j,
// and is n tiles long.
template <typename F>
inline void for_row_runs(uint32_t t0, uint32_t nb, uint32_t Nt, F&& f) {
    for (uint32_t j = 0; j < nb;) {
        const uint32_t t = t0 + j;
        const uint32_t mi = t / Nt;
        const uint32_t ni = t - mi * Nt;
        const uint32_t n = (nb - j < Nt - ni) ? (nb - j) : (Nt - ni);
        f(mi, ni, j, n);
        j += n;
    }
}

inline void mm(
    uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr, bool skip_reconfig = false) {
    CircularBuffer(o).reserve_back(Mt * Nt);
    if (!skip_reconfig) {
        pack_reconfig_data_format(o);  // mixed bf16/fp32 CBs: set packer to this output's format
        // matmul_tiles(a,b): in0=a->srcB, in1=b->srcA. Reconfig unpack src formats to match (the
        // op init only asserts formats, it does not set them), else fp32/bf16 CBs are read at the
        // wrong format and produce garbage. skip_reconfig=true is legal ONLY when the caller has
        // already configured the packer/unpackers for these operands' formats (all-fp32 regions).
        reconfig_data_format(b, a);
    }
    matmul_init(a, b, tr ? 1 : 0);
    if constexpr (kDstTiles == 1) {  // original per-tile form (byte-identical code for the prep kernel)
        for (uint32_t mi = 0; mi < Mt; mi++) {
            for (uint32_t ni = 0; ni < Nt; ni++) {
                tile_regs_acquire();
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                    matmul_tiles(a, b, mi * Kt + ki, bi, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, o, mi * Nt + ni);
                tile_regs_release();
            }
        }
        CircularBuffer(o).push_back(Mt * Nt);
        return;
    }
    const uint32_t n_out = Mt * Nt;
    for (uint32_t t0 = 0; t0 < n_out; t0 += kDstTiles) {
        const uint32_t nb = (n_out - t0 < kDstTiles) ? (n_out - t0) : kDstTiles;
        tile_regs_acquire();
        if (tr) {  // B^T: a row's in1 tiles are Kt apart, one product per unpack
            for (uint32_t j = 0; j < nb; j++) {
                const uint32_t t = t0 + j;
                const uint32_t mi = t / Nt;
                const uint32_t ni = t - mi * Nt;
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    matmul_tiles(a, b, mi * Kt + ki, ni * Kt + ki, j);
                }
            }
        } else {  // a row's in1 tiles are consecutive: each in0 tile is unpacked once per row run
            for_row_runs(t0, nb, Nt, [&](uint32_t mi, uint32_t ni, uint32_t j, uint32_t n) {
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    mm_tiles_row(a, b, mi * Kt + ki, ki * Nt + ni, j, n);
                }
            });
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nb; j++) {
            pack_tile(j, o, t0 + j);
        }
        tile_regs_release();
    }
    CircularBuffer(o).push_back(Mt * Nt);
}

// Elementwise binary op selector for ew() / ewt().
enum class EwOp : uint8_t { Add, Sub, Mul };

// out = A (op) B elementwise, n tiles.
inline void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, EwOp op, bool skip_reconfig = false) {
    CircularBuffer(o).reserve_back(n);
    if (!skip_reconfig) {
        pack_reconfig_data_format(o);
        reconfig_data_format(a, b);  // binary(a,b): a->srcA, b->srcB
    }
    if (op == EwOp::Add) {
        add_init(a, b);
    } else if (op == EwOp::Sub) {
        sub_init(a, b);
    } else {
        mul_init(a, b);
    }
    if constexpr (kDstTiles == 1) {
        for (uint32_t i = 0; i < n; i++) {
            tile_regs_acquire();
            if (op == EwOp::Add) {
                add_tiles(a, b, i, i, 0);
            } else if (op == EwOp::Sub) {
                sub_tiles(a, b, i, i, 0);
            } else {
                mul_tiles(a, b, i, i, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, i);
            tile_regs_release();
        }
        CircularBuffer(o).push_back(n);
        return;
    }
    for (uint32_t i0 = 0; i0 < n; i0 += kDstTiles) {
        const uint32_t nb = (n - i0 < kDstTiles) ? (n - i0) : kDstTiles;
        tile_regs_acquire();
        for (uint32_t j = 0; j < nb; j++) {
            if (op == EwOp::Add) {
                add_tiles(a, b, i0 + j, i0 + j, j);
            } else if (op == EwOp::Sub) {
                sub_tiles(a, b, i0 + j, i0 + j, j);
            } else {
                mul_tiles(a, b, i0 + j, i0 + j, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nb; j++) {
            pack_tile(j, o, i0 + j);
        }
        tile_regs_release();
    }
    CircularBuffer(o).push_back(n);
}

// The SFPU DST<->DST product is an inlined LLK body of several hundred bytes; kept out of line (one copy) so the
// Ct == 2 prep program fits the kernel-config buffer.
[[gnu::noinline]] inline void sfpu_mul_dst(uint32_t a, uint32_t b, uint32_t o) { mul_binary_tile(a, b, o); }

// ---- prep SFPU passes. The prep blocks below land their FPU results in DST and finish on the SFPU before one
// pack. Each block's whole SFPU chain is ONE pass here (one load / store round trip per vector instead of one per
// op), and the pass visits only the lanes its consumers read:
//   - a column vector (decay, g_sum) lives in tile column 0, which the even-column vectors of faces 0 and 2 hold:
//     8 of the tile's 32 SFPU vectors;
//   - a diagonal tile lives in faces 0 and 3; a lower-triangular tile has nothing in face 1.
// The values in the visited lanes are computed by the same SFPU operations, in the same order, as the chain of
// one-op tile passes (sub_binary_tile, mul_binary_tile, exp_tile, rsqrt_tile, ...) each pass replaces, so the lanes
// that are read are bit-identical to that chain's for finite inputs.
// Every store targets rows an FPU instruction has already written in the same DST acquire, never rows that are
// still undefined: a store into an undefined row has to set the row's valid bit, and that update is lost if the
// store executes in the cycle the pack thread's ZEROACC releases the other DST half -- the packer would then read
// the row as zero.
#ifdef TRISC_MATH
namespace gdn_sfpu {

inline constexpr uint32_t kDstTile = 32;  // sfpi::dst_reg index distance between two DST tiles
inline constexpr int kFaceVectors = 8;    // SFPU vectors per 16x16 face: four rows by eight columns, even then odd

// Each pass is one out-of-line function whose DST tiles are template parameters: the SFPLOAD / SFPSTORE addresses
// are instruction immediates, and there is exactly one copy of the body however the caller's loops are unrolled
// (the prep program sits close to the kernel-config buffer limit).

// exp_tile's arithmetic for this kernel's compute config (fp32 DST, no approximation).
sfpi_inline sfpi::vFloat exp_fp32(sfpi::vFloat x) { return ckernel::sfpu::_sfpu_exp_accurate_<true>(x); }

// Column 0 of the decay window, one face per call (VectorMode::C runs it on faces 0 and 2):
//   DST[D_FAC] = exp(DST[D_FAC] - DST[D_DECAY]), DST[D_EXP] = exp(DST[D_DECAY]), DST[D_DL] = DST[D_FAC] * DST[D_EXP].
template <uint32_t D_DECAY, uint32_t D_FAC, uint32_t D_EXP, uint32_t D_DL>
[[gnu::noinline]] void decay_col0_face() {
#pragma GCC unroll 0
    for (int v = 0; v < kFaceVectors / 2; v++) {
        const sfpi::vFloat decay = sfpi::dst_reg[D_DECAY * kDstTile];
        sfpi::vFloat fac = sfpi::dst_reg[D_FAC * kDstTile];
        fac = exp_fp32(fac - decay);
        sfpi::dst_reg[D_FAC * kDstTile] = fac;
        const sfpi::vFloat decay_exp = exp_fp32(decay);
        sfpi::dst_reg[D_EXP * kDstTile] = decay_exp;
        sfpi::dst_reg[D_DL * kDstTile] = fac * decay_exp;
        sfpi::dst_reg += 2;  // the next four rows, even columns again
    }
}

// Store +0 over one face of DST[D].
template <uint32_t D>
sfpi_inline void zero_face() {
    const sfpi::vFloat zero = 0.0f;
#pragma GCC unroll 0
    for (int v = 0; v < kFaceVectors; v++) {
        sfpi::dst_reg[D * kDstTile] = zero;
        sfpi::dst_reg++;
    }
}

// DST[D_I] = tril * exp(tril * (DST[D_I] - DST[D_J])), tril = DST[D_TRIL]. The mask is applied before the exp (the
// upper triangle would overflow) and again after. A diagonal tile block (diag_tile) has an all-zero face 1 (rows
// 0..15 by columns 16..31, strictly above the diagonal): the chain leaves exp(+-0) * 0 = +0 there, stored directly.
template <uint32_t D_I, uint32_t D_J, uint32_t D_TRIL>
[[gnu::noinline]] void lmask_tile(const bool diag_tile) {
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
        if (diag_tile && face == 1) {
            zero_face<D_I>();
        } else {
#pragma GCC unroll 0
            for (int v = 0; v < kFaceVectors; v++) {
                sfpi::vFloat x = sfpi::dst_reg[D_I * kDstTile];
                x = x - sfpi::vFloat(sfpi::dst_reg[D_J * kDstTile]);
                x = x * sfpi::vFloat(sfpi::dst_reg[D_TRIL * kDstTile]);
                x = exp_fp32(x);
                sfpi::dst_reg[D_I * kDstTile] = x * sfpi::vFloat(sfpi::dst_reg[D_TRIL * kDstTile]);
                sfpi::dst_reg++;
            }
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

// One face of DST[D_KK] = (DST[D_KK] * DST[D_LMASK]) * (DST[D_EYE] + -1): the (I - 1) factor drops the diagonal and
// flips the sign of the rest exactly.
template <uint32_t D_KK, uint32_t D_LMASK, uint32_t D_EYE>
[[gnu::noinline]] void neg_strict_lower_face() {
#pragma GCC unroll 0
    for (int v = 0; v < kFaceVectors; v++) {
        sfpi::vFloat mask = sfpi::dst_reg[D_EYE * kDstTile];
        mask = mask + ckernel::sfpu::Converter::as_float(0xBF800000u);  // I - 1
        sfpi::vFloat x = sfpi::dst_reg[D_KK * kDstTile];
        x = x * sfpi::vFloat(sfpi::dst_reg[D_LMASK * kDstTile]);
        sfpi::dst_reg[D_KK * kDstTile] = x * mask;
        sfpi::dst_reg++;
    }
}

// DST[D_OUT] = DST[D_A] * DST[D_B] on the three faces that hold the strict lower triangle of a diagonal tile block;
// face 1 of DST[D_OUT] is left untouched.
template <uint32_t D_A, uint32_t D_B, uint32_t D_OUT>
[[gnu::noinline]] void mul_lower_faces() {
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
        if (face != 1) {
#pragma GCC unroll 0
            for (int v = 0; v < kFaceVectors; v++) {
                sfpi::vFloat x = sfpi::dst_reg[D_A * kDstTile];
                sfpi::dst_reg[D_OUT * kDstTile] = x * sfpi::vFloat(sfpi::dst_reg[D_B * kDstTile]);
                sfpi::dst_reg++;
            }
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

// DST[D_GRAM] = eye * (rsqrt(eye * DST[D_GRAM] + eps) [* scale]), eye = DST[D_EYE], for a diagonal tile: the two
// diagonal faces run the chain, the off-diagonal faces 1 and 2 get the +0 it leaves there (rsqrt(eps) * scale * 0).
template <bool DO_SCALE, uint32_t D_GRAM, uint32_t D_EYE>
[[gnu::noinline]] void inv_rms_diag_tile(const uint32_t eps_bits, const uint32_t scale_bits) {
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
        if (face == 1 || face == 2) {
            zero_face<D_GRAM>();
        } else {
#pragma GCC unroll 0
            for (int v = 0; v < kFaceVectors; v++) {
                sfpi::vFloat x = sfpi::dst_reg[D_GRAM * kDstTile];
                x = x * sfpi::vFloat(sfpi::dst_reg[D_EYE * kDstTile]);  // diag(rowsum)
                x = x + ckernel::sfpu::Converter::as_float(eps_bits);   // off-diagonal: eps
                x = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(x);
                if constexpr (DO_SCALE) {
                    x = x * ckernel::sfpu::Converter::as_float(scale_bits);
                }
                sfpi::dst_reg[D_GRAM * kDstTile] = x * sfpi::vFloat(sfpi::dst_reg[D_EYE * kDstTile]);
                sfpi::dst_reg++;
            }
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

}  // namespace gdn_sfpu
#endif  // TRISC_MATH

// Run one gdn_sfpu pass on the math thread: the pass addresses its DST tiles (its template arguments) relative to
// tile 0. VectorMode::RC / C call a per-face pass on all four faces / on faces 0 and 2; VectorMode::None calls a
// whole-tile pass once. Arguments after the vector mode are passed on to the pass.
#define GDN_SFPU_PASS(pass, vector_mode, ...) \
    MATH((_llk_math_eltwise_unary_sfpu_params_(pass, 0 /*DST base tile*/, vector_mode __VA_OPT__(, ) __VA_ARGS__)))

// ---- fused single-DST-pass prep blocks (C3). Each replaces a chain of one-op helper blocks whose
// intermediates were packed to scratch and re-unpacked; here the FPU op lands in DST and the rest of
// the chain runs on the SFPU (DST <-> DST) before one pack. fp32 CBs only.

// decay = tril @ g (col form), decay_exp = exp(decay), decayfac = exp(g_sum - decay) with g_sum = ones @ g, and
// dl = exp(g_sum) = decayfac_i * decay_exp_i, all from ONE DST pass per tile: the matmuls land decay in DST0
// (packed as is) and DST3, and g_sum in DST1 and DST2; one SFPU pass over column 0 (gdn_sfpu::decay_col0_face)
// then leaves decayfac in DST1, decay_exp in DST3 and dl in DST2 -- dl as the exact fp32 product of the two exps,
// not the broadcast product whose rounded factors cost ~1e-3 on the value that decays the state every chunk.
// g and so every one of these is a column vector: only tile column 0 is defined and read (by the column
// broadcasts and, for decay, the transpose), the rest of each tile is padding. The second copy of each matmul is
// there to write DST2's and DST3's rows on the FPU before the SFPU stores into them (see the SFPU pass note).
// Packs: decay -> o_decay[i], decay_exp -> o_exp[i], decayfac -> o_fac[i], dl -> o_fac[Ct] (from the last tile's
// window; every row of every window holds g_sum).
inline void decay_all(
    uint32_t tril, uint32_t ones, uint32_t g, uint32_t o_decay, uint32_t o_exp, uint32_t o_fac, uint32_t Ct) {
    CircularBuffer(o_decay).reserve_back(Ct);
    CircularBuffer(o_exp).reserve_back(Ct);
    CircularBuffer(o_fac).reserve_back(Ct + 1);
    pack_reconfig_data_format(o_decay);  // all fp32
    reconfig_data_format(g, tril);       // matmul(tril|ones, g): in0->srcB, g->srcA
    matmul_init(tril, g, 0);
    for (uint32_t i = 0; i < Ct; i++) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < Ct; j++) {
            matmul_tiles(tril, g, i * Ct + j, j, 0);  // DST0 = decay_i
            matmul_tiles(tril, g, i * Ct + j, j, 3);  // DST3 = decay_i (rows for decay_exp_i)
            matmul_tiles(ones, g, i * Ct + j, j, 1);  // DST1 = g_sum
            matmul_tiles(ones, g, i * Ct + j, j, 2);  // DST2 = g_sum (rows for dl)
        }
        exp_tile_init();
        // DST tiles: 0 decay, 1 g_sum -> decayfac, 3 decay_exp, 2 dl
        GDN_SFPU_PASS((gdn_sfpu::decay_col0_face<0, 1, 3, 2>), VectorMode::C);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o_decay, i);
        pack_tile(3, o_exp, i);
        pack_tile(1, o_fac, i);
        if (i + 1 == Ct) {
            // pack_tile fills o_fac in call order, whatever index it is given: dl is its last tile, so it is taken
            // from the last window.
            pack_tile(2, o_fac, Ct);
        }
        tile_regs_release();
    }
    CircularBuffer(o_decay).push_back(Ct);
    CircularBuffer(o_exp).push_back(Ct);
    CircularBuffer(o_fac).push_back(Ct + 1);
}

// L_mask[i,j] = tril * exp(decay_i - decay_j), one DST pass per output tile: decay_i (col broadcast of
// `decay`) in DST0, decay_j (row broadcast of `decay_row`) in DST1 and the tril tile in DST2, then one SFPU pass
// (gdn_sfpu::lmask_tile): the difference, the mask before the exp (the upper triangle would overflow), the exp,
// and the mask again.
inline void lmask_fused(uint32_t ones, uint32_t decay, uint32_t decay_row, uint32_t tril, uint32_t o, uint32_t Ct) {
    CircularBuffer(o).reserve_back(Ct * Ct);
    pack_reconfig_data_format(o);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        for (uint32_t ni = 0; ni < Ct; ni++) {
            tile_regs_acquire();
            reconfig_data_format(ones, decay);  // bcast(a, b): a->srcA, b->srcB
            mul_bcast_cols_init(ones, decay);
            mul_tiles_bcast_cols(ones, decay, 0, mi, 0);  // DST0[i,j] = decay_i
            mul_bcast_rows_init(ones, decay_row);
            mul_tiles_bcast_rows(ones, decay_row, 0, ni, 1);  // DST1[i,j] = decay_j
            copy_init(tril);
            copy_tile(tril, mi * Ct + ni, 2);  // DST2 = tril tile
            exp_tile_init();
            // DST tiles: 0 decay_i -> L_mask, 1 decay_j, 2 tril
            GDN_SFPU_PASS((gdn_sfpu::lmask_tile<0, 1, 2>), VectorMode::None, mi == ni);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Ct + ni);
            tile_regs_release();
        }
    }
    CircularBuffer(o).push_back(Ct * Ct);
}

// dl*I: the identity tile scaled by column 0 of tile `col_tile` of `col` (dl in every row).
inline void dl_tile(uint32_t eye, uint32_t col, uint32_t col_tile, uint32_t o) {
    CircularBuffer(o).reserve_back(1);
    pack_reconfig_data_format(o);
    reconfig_data_format(eye, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init(eye, col);
    tile_regs_acquire();
    mul_tiles_bcast_cols(eye, col, 0, col_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    CircularBuffer(o).push_back(1);
}

// out[Mt,Nt] = -(A[Mt,Nt] * col[Mt,1]): the broadcast product lands in DST, the sign flip is an exact SFPU op on
// it before the pack (the same values as negating the column first, without staging the negated column).
inline void bcast_cols_mul_neg(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
    CircularBuffer(o).reserve_back(Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init(a, col);
    negative_tile_init();
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a, col, mi * Nt + ni, mi, 0);
            negative_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    CircularBuffer(o).push_back(Mt * Nt);
}

// negN = -strictly_lower(kk * L_mask), kk = k_beta @ k^T, one DST pass per output tile: kk lands in
// DST0 (Kt products, in1 transposed), L_mask is copied to DST1 and the eye tile to DST2; one SFPU pass
// (gdn_sfpu::neg_strict_lower_face) multiplies L_mask in and then the constant (I - 1) -- 0 on the diagonal, -1
// elsewhere -- which drops the diagonal and flips the sign of the strictly-lower part exactly (the upper triangle
// is already zero from L_mask).
inline void negn_fused(uint32_t kbeta, uint32_t k, uint32_t lmask, uint32_t eye, uint32_t o, uint32_t Ct, uint32_t Kt) {
    CircularBuffer(o).reserve_back(Ct * Ct);
    pack_reconfig_data_format(o);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        for (uint32_t ni = 0; ni < Ct; ni++) {
            tile_regs_acquire();
            reconfig_data_format(k, kbeta);  // matmul(kbeta, k): kbeta->srcB, k->srcA
            matmul_init(kbeta, k, 1);
            for (uint32_t ki = 0; ki < Kt; ki++) {
                matmul_tiles(kbeta, k, mi * Kt + ki, ni * Kt + ki, 0);  // DST0 = kk tile
            }
            reconfig_data_format_srca(lmask);
            copy_init(lmask);
            copy_tile(lmask, mi * Ct + ni, 1);
            copy_init(eye);
            copy_tile(eye, mi * Ct + ni, 2);
            mul_binary_tile_init();
            // DST tiles: 0 kk -> negN, 1 L_mask, 2 eye
            GDN_SFPU_PASS((gdn_sfpu::neg_strict_lower_face<0, 1, 2>), VectorMode::RC);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Ct + ni);
            tile_regs_release();
        }
    }
    CircularBuffer(o).push_back(Ct * Ct);
}

// The WY factor for the forward-substitution solve: a single 32x32 tile whose strict lower triangle is
// kk * L_mask = N (kk = k_beta @ k^T), not negated. triangle_solve_tile reads nothing else of the tile -- the unit
// diagonal is implicit and the upper triangle is never loaded -- so neither is produced: kk lands in DST0 (Kt
// products), L_mask is copied to DST1 and multiplied in on the three faces that hold the strict lower triangle
// (gdn_sfpu::mul_lower_faces). The solve negates each entry as it loads it (L_NEGATED = false), which is the
// same sign-bit flip the (I - 1) factor of negn_fused applies, so T_inv is bit-identical to solving negn_fused's
// tile with L_NEGATED = true.
inline void wy_factor_lower(uint32_t kbeta, uint32_t k, uint32_t lmask, uint32_t o, uint32_t Kt) {
    CircularBuffer(o).reserve_back(1);
    pack_reconfig_data_format(o);
    tile_regs_acquire();
    reconfig_data_format(k, kbeta);  // matmul(kbeta, k): kbeta->srcB, k->srcA
    matmul_init(kbeta, k, 1);
    for (uint32_t ki = 0; ki < Kt; ki++) {
        matmul_tiles(kbeta, k, ki, ki, 0);  // DST0 = kk
    }
    reconfig_data_format_srca(lmask);
    copy_init(lmask);
    copy_tile(lmask, 0, 1);
    mul_binary_tile_init();
    // DST tiles: 0 kk -> N, 1 L_mask
    GDN_SFPU_PASS((gdn_sfpu::mul_lower_faces<0, 1, 0>), VectorMode::None);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    CircularBuffer(o).push_back(1);
}

// intra = (q @ k^T) * L_mask, one DST pass per output tile: qk in DST0 (Kt products), L_mask copied to
// DST1 and multiplied in on the SFPU.
inline void intra_fused(uint32_t q, uint32_t k, uint32_t lmask, uint32_t o, uint32_t Ct, uint32_t Kt) {
    CircularBuffer(o).reserve_back(Ct * Ct);
    pack_reconfig_data_format(o);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        for (uint32_t ni = 0; ni < Ct; ni++) {
            tile_regs_acquire();
            reconfig_data_format(k, q);  // matmul(q, k): q->srcB, k->srcA (bf16 when the reader's raw q/k are used)
            matmul_init(q, k, 1);
            for (uint32_t ki = 0; ki < Kt; ki++) {
                matmul_tiles(q, k, mi * Kt + ki, ni * Kt + ki, 0);
            }
            reconfig_data_format_srca(lmask);
            copy_init(lmask);
            copy_tile(lmask, mi * Ct + ni, 1);
            mul_binary_tile_init();
            sfpu_mul_dst(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Ct + ni);
            tile_regs_release();
        }
    }
    CircularBuffer(o).push_back(Ct * Ct);
}

// ---- fused qk-norm (C5). rowsum(x^2) is the diagonal of x @ x^T, so the per-row inverse rms comes out of one
// matmul window as a diagonal tile, and the normalization itself is one block-diagonal matmul: 2 blocks per
// operand instead of 4 (x^2, rowsum, rsqrt, broadcast), and no packed intermediates.

// D[mi] = diag(rsqrt(rowsum(x[mi,:]^2) + eps) [* scale]) for each row-tile of x [Ct, Kt], one DST pass per tile:
// the Kt products x[mi,ki] @ x[mi,ki]^T accumulate the row sums of squares on the diagonal of DST0 (the
// off-diagonal x_i.x_j are discarded); the identity is copied to DST1 and one SFPU pass
// (gdn_sfpu::inv_rms_diag_tile) applies it before and after the chain (+eps, rsqrt, *scale), so the off-diagonal
// rsqrt(eps) never leaves DST. DO_SCALE folds q's scale into the normalizer.
template <bool DO_SCALE>
inline void inv_rms_diag(
    uint32_t x, uint32_t eye, uint32_t o, uint32_t Ct, uint32_t Kt, uint32_t eps_bits, uint32_t scale_bits) {
    CircularBuffer(o).reserve_back(Ct);
    pack_reconfig_data_format(o);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        tile_regs_acquire();
        reconfig_data_format(x, x);  // matmul(x, x^T): x on both srcA and srcB
        matmul_init(x, x, 1);
        for (uint32_t ki = 0; ki < Kt; ki++) {
            matmul_tiles(x, x, mi * Kt + ki, mi * Kt + ki, 0);  // DST0 = x_mi @ x_mi^T
        }
        reconfig_data_format_srca(eye);
        copy_init(eye);
        copy_tile(eye, 0, 1);  // the identity block (cb.eye tile 0)
        rsqrt_tile_init();
        // DST tiles: 0 gram -> D, 1 eye
        GDN_SFPU_PASS((gdn_sfpu::inv_rms_diag_tile<DO_SCALE, 0, 1>), VectorMode::None, eps_bits, scale_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, mi);
        tile_regs_release();
    }
    CircularBuffer(o).push_back(Ct);
}

// out[mi, ki] = D[mi] @ x[mi, ki]: block-diagonal left multiply (D holds Ct diagonal tiles), one product per tile.
inline void mm_diag(uint32_t d, uint32_t x, uint32_t o, uint32_t Ct, uint32_t Kt) {
    CircularBuffer(o).reserve_back(Ct * Kt);
    pack_reconfig_data_format(o);
    reconfig_data_format(x, d);  // matmul(d, x): d->srcB, x->srcA
    matmul_init(d, x, 0);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        for (uint32_t ki = 0; ki < Kt; ki++) {
            tile_regs_acquire();
            matmul_tiles(d, x, mi, mi * Kt + ki, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Kt + ki);
            tile_regs_release();
        }
    }
    CircularBuffer(o).push_back(Ct * Kt);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
inline void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
    CircularBuffer(o).reserve_back(Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init(a, col);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a, col, mi * Nt + ni, mi, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    CircularBuffer(o).push_back(Mt * Nt);
}

// out[0] = copy of src[src_tile] (single 32x32 tile). src must be available.
inline void cpy_t(uint32_t src, uint32_t src_tile, uint32_t o, bool skip_reconfig = false) {
    CircularBuffer(o).reserve_back(1);
    if (!skip_reconfig) {  // see mm() for the skip_reconfig contract
        pack_reconfig_data_format(o);
        reconfig_data_format_srca(src);
    }
    copy_init(src);
    tile_regs_acquire();
    copy_tile(src, src_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    CircularBuffer(o).push_back(1);
}

// out[0] = a[ai] (op) b[bi], single tile; Add or Mul. (Like ew but with free tile indices.) Out of line: the
// Horner inverse calls it at seven sites, and one copy per thread keeps the prep program inside the kernel-config
// buffer.
[[gnu::noinline]] inline void ewt(
    uint32_t a, uint32_t ai, uint32_t b, uint32_t bi, uint32_t o, EwOp op, bool skip_reconfig = false) {
    CircularBuffer(o).reserve_back(1);
    if (!skip_reconfig) {  // see mm() for the skip_reconfig contract
        pack_reconfig_data_format(o);
        reconfig_data_format(a, b);
    }
    if (op == EwOp::Add) {
        add_init(a, b);
    } else {
        mul_init(a, b);
    }
    tile_regs_acquire();
    if (op == EwOp::Add) {
        add_tiles(a, b, ai, bi, 0);
    } else {
        mul_tiles(a, b, ai, bi, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    CircularBuffer(o).push_back(1);
}

// (I32 - Nq)^-1 for a strictly-lower 16-block Nq isolated in one 16-quadrant (rest zero),
// nilpotent at 16. Horner in 15 terms -> out (single tile); the other diagonal quadrant is I.
// Small block + short chain keeps fp32 bounded where a 32x32/31-term Horner cancels.
// cb_eye holds the identity (tile 0 = I32).
inline void invert16(uint32_t nq, uint32_t out, uint32_t tmp, uint32_t cb_eye) {
    // Hot path: 15 alternating single-tile matmul/add rounds per call, ~4 calls per chunk. All
    // four CBs are fp32, so the unpacker/packer format registers never change across the loop —
    // reconfigure ONCE up front instead of inside every mm()/ew() call (their per-call
    // reconfig_data_format/pack_reconfig are unconditional register writes). The MOP inits still
    // alternate per op class. Ops, operands, order, and pack boundaries are identical to the
    // plain mm/ew composition this replaces — bit-exact with it by construction.
    if (kGdnHoistReconfig) {
        pack_reconfig_data_format(out);    // out and tmp are both fp32: one packer config serves all
        reconfig_data_format(cb_eye, nq);  // all operands fp32: one unpack config serves mm and ew
    }
    auto add1 = [&](uint32_t a, uint32_t b, uint32_t o) {  // o = a + b, 1 tile
        CircularBuffer(o).reserve_back(1);
        if (!kGdnHoistReconfig) {  // per-call reconfigs, exactly as the plain ew() would issue
            pack_reconfig_data_format(o);
            reconfig_data_format(a, b);
        }
        add_init(a, b);
        tile_regs_acquire();
        add_tiles(a, b, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, 0);
        tile_regs_release();
        CircularBuffer(o).push_back(1);
    };
    add1(cb_eye, nq, out);  // out = I + Nq
    CircularBuffer(out).wait_front(1);
    for (uint32_t m = 2; m < 16; m++) {  // sum_{k<16} Nq^k
        CircularBuffer(tmp).reserve_back(1);
        if (!kGdnHoistReconfig) {  // per-call reconfigs, exactly as the plain mm() would issue
            pack_reconfig_data_format(tmp);
            reconfig_data_format(out, nq);
        }
        matmul_init(nq, out, 0);
        tile_regs_acquire();
        matmul_tiles(nq, out, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tmp, 0);
        tile_regs_release();
        CircularBuffer(tmp).push_back(1);
        CircularBuffer(tmp).wait_front(1);
        CircularBuffer(out).pop_front(1);
        add1(cb_eye, tmp, out);  // out = I + Nq @ out
        CircularBuffer(out).wait_front(1);
        CircularBuffer(tmp).pop_front(1);
    }
}

// Assemble the 2x2 tile-block matrix [[s0[t0], s1[t1]], [s2[t2], s3[t3]]] into o (4 tiles).
inline void asm4(
    uint32_t s0,
    uint32_t t0,
    uint32_t s1,
    uint32_t t1,
    uint32_t s2,
    uint32_t t2,
    uint32_t s3,
    uint32_t t3,
    uint32_t o) {
    const uint32_t src[4] = {s0, s1, s2, s3};
    const uint32_t tl[4] = {t0, t1, t2, t3};
    CircularBuffer(o).reserve_back(4);
    pack_reconfig_data_format(o);
    for (uint32_t i = 0; i < 4; i++) {
        reconfig_data_format_srca(src[i]);
        copy_init(src[i]);
        tile_regs_acquire();
        copy_tile(src[i], tl[i], 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    CircularBuffer(o).push_back(4);
}

// Invert one 32x32 diagonal tile-block: out[0] = (I32 - negN)^-1, negN = src[tile] (strictly-lower
// 32x32). Mirrors FLA solve_tril: split into 16-quadrants negN = [[N00,0],[N10,N11]], invert the two
// diagonal 16-blocks (short, bounded Horners), and form the off-diagonal EXACTLY (one matmul chain,
// no power series). A single 32x32 Horner instead loses fp32 precision on harder blocks.
//   Bi00=(I-N00)^-1 (top-left), Bi11=(I-N11)^-1 (bottom-right), off=Bi11@N10@Bi00 (bottom-left).
//   out = [[Bi00,0],[off,Bi11]].
// cb_eye = identity; masks (single tiles): cb_mask[0]=Qtl, [1]=Qbr, [2]=Q10 (bottom-left).
// tmpN/tmpT = scratch. Private scratch A..D: single-tile-capable fp32 CBs, NOT drained by any
// writer while this runs, and none may alias src, out, tmpN, or tmpT.
inline void invert_block(
    uint32_t src,
    uint32_t tile,
    uint32_t out,
    uint32_t tmpN,
    uint32_t tmpT,
    uint32_t cb_eye,
    uint32_t cb_mask,
    uint32_t A,
    uint32_t B,
    uint32_t C,
    uint32_t D) {
    // Every CB this function touches is fp32, so one packer+unpacker format config up front
    // serves the whole body; the per-call reconfigs inside cpy_t/ewt/mm are skipped (they are
    // unconditional register writes and this body issues ~9 such calls per invocation, twice per
    // chunk). Op order and pack boundaries are unchanged — bit-exact with the unhoisted form.
    if (kGdnHoistReconfig) {
        pack_reconfig_data_format(out);
        reconfig_data_format(cb_eye, src);
    }
    cpy_t(src, tile, tmpN, kGdnHoistReconfig);
    CircularBuffer(tmpN).wait_front(1);  // negN -> tmpN[0]
    // Bi00 = (I-N00)^-1  (N00 = top-left quadrant of negN; top-right is already 0)
    ewt(tmpN, 0, cb_mask, 0, A, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);  // N00
    invert16(A, B, tmpT, cb_eye);
    CircularBuffer(B).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi00 -> B
    // Bi11 = (I-N11)^-1  (N11 = bottom-right quadrant)
    ewt(tmpN, 0, cb_mask, 1, A, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);  // N11
    invert16(A, C, tmpT, cb_eye);
    CircularBuffer(C).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi11 -> C
    // off = Bi11 @ N10 @ Bi00  (N10 = bottom-left quadrant; result lives only there)
    ewt(tmpN, 0, cb_mask, 2, A, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);  // N10
    CircularBuffer(tmpN).pop_front(1);
    mm(C, A, tmpT, 1, 1, 1, false, kGdnHoistReconfig);
    CircularBuffer(tmpT).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi11@N10
    mm(tmpT, B, A, 1, 1, 1, false, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);
    CircularBuffer(tmpT).pop_front(1);  // @Bi00 -> A(off)
    // out = Qtl*Bi00 + Qbr*Bi11 + off
    ewt(B, 0, cb_mask, 0, D, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(D).wait_front(1);
    CircularBuffer(B).pop_front(1);  // Bi00_tl -> D
    ewt(C, 0, cb_mask, 1, B, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(B).wait_front(1);
    CircularBuffer(C).pop_front(1);  // Bi11_br -> B
    ewt(D, 0, B, 0, C, EwOp::Add, true);
    CircularBuffer(C).wait_front(1);
    CircularBuffer(D).pop_front(1);
    CircularBuffer(B).pop_front(1);
    ewt(C, 0, A, 0, out, EwOp::Add, true);
    CircularBuffer(C).pop_front(1);
    CircularBuffer(A).pop_front(1);  // + off -> out
}

#if defined(GDN_TINV_SFPU)
// T_inv = (I + strictly_lower(N))^-1 of one 32x32 tile by forward substitution.
//   n      : fp32 CB whose front tile holds N in its strict lower triangle (wy_factor_lower; the diagonal and the
//            upper triangle are not read), front-waited by the caller.
//   cb_eye : CB holding the identity tile.
//   out    : fp32 CB that receives T_inv (one tile pushed).
inline void sfpu_tinv(uint32_t n, uint32_t cb_eye, uint32_t out) {
    CircularBuffer(out).reserve_back(1);
    // copy_tile reads the identity through SrcA; the solve's DEST loads and stores are fp32 under fp32 dest
    // accumulation.
    reconfig_data_format_srca(cb_eye);
    pack_reconfig_data_format(out);
    copy_init(cb_eye);
    constexpr uint32_t DST_RHS = 0, DST_X = 1;
    tile_regs_acquire();
    copy_tile(cb_eye, 0, DST_RHS);  // RHS = I
    triangle_solve_tile_init();
    triangle_solve_tile<DataFormat::Float32, false /*L_NEGATED*/>(n, 0 /*l_tile_idx*/, DST_RHS, DST_X);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(DST_X, out, 0);
    tile_regs_release();
    CircularBuffer(out).push_back(1);
}
#endif

// out[1,Ct] row-form = transpose of col[Ct,1]; produces Ct tiles (each row0 = a 32-chunk of col).
inline void transpose_col(uint32_t in, uint32_t o, uint32_t Ct) {
    CircularBuffer(o).reserve_back(Ct);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);  // unary: in->srcA
    transpose_init(in);
    for (uint32_t i = 0; i < Ct; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    CircularBuffer(o).push_back(Ct);
}

// CB map for prep_chunk — one field per CB the body touches. dl/mask are the prep kernel's
// aliases (cb_dl = the vnew slot, cb_mask = the u slot); the map carries the resolved ids.
struct GdnPrepCbs {
    uint32_t q, k, v, g, beta;
    uint32_t eye, tril, ones, S;
    uint32_t decay, decay_exp, decayfac, lmask, Tinv, vbeta, kbeta;
    uint32_t w, qdecay, intra, s2, ointer, kdec_t, supd, stmp, final_s;
    uint32_t scr1, scr2, scr3, s3;
    uint32_t dl;    // alias of the vnew slot in prep (1 tile used)
    uint32_t mask;  // alias of the u slot in prep (3 quadrant-mask tiles)
};

// CB map for scan_step — one field per CB the body touches (the state CBs S/s2/s3/final are
// selected per chunk by the caller and passed as cur_S/dst).
struct GdnScanCbs {
    uint32_t dl, Tinv, out;
    uint32_t vbeta, nkd, qdecay, intra;
    uint32_t vnew, ointer, kdec_t;
    uint32_t eye;  // one 32x32 identity tile (fp32), written by the reader at kernel start
};

// PHASE A (prep): one state-independent (head, chunk) work-item. No recurrent state here; the
// sequential state scan lives in scan_step. Outputs (per chunk) v_beta, nkd(->cb.w), T_inv,
// k_dec_t, q_decay, intra, dl are pushed to their CBs and streamed to DRAM by the prep writer.
// Ct/Kt/Vt/qk_norm are TEMPLATE parameters (not runtime args): the shape branches below must compile
// out exactly as the monolithic kernel's `if constexpr` did, or the Ct==2 prep program overflows the
// 70 KB kernel-config buffer (found on QB2 at chunk_size=64: 73344 > 70656 bytes).
template <uint32_t Ct, uint32_t Kt, uint32_t Vt, bool qk_norm>
inline void prep_chunk(const GdnPrepCbs& cb, uint32_t scale_bits, uint32_t eps_bits) {
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t C = Ct * 32;
    // Runtime copies of the tile counts for the helper calls (identical code at Ct == 1; keeps the Ct == 2
    // build's per-tile loops rolled -- see tiles_dim).
    const uint32_t ct = tiles_dim<Ct>(Ct);
    const uint32_t cc_rt = ct * ct;
    const uint32_t ck_rt = ct * Kt;

    WAIT(cb.q, ck);
    WAIT(cb.k, ck);
    WAIT(cb.v, cv);
    WAIT(cb.g, Ct);
    WAIT(cb.beta, Ct);

    // Block order. Each block below unpacks its operands, computes in DST and packs; a block that reads another
    // block's packed result must WAIT for it, and a WAIT placed right after the producing block idles this thread's
    // unpacker for that block's math + pack latency (~0.2-0.3 us on a one-tile block). So the blocks are ordered to
    // put an independent block between a producer and its first consumer wherever one exists: by the time the
    // WAIT is reached the tile has long been pushed. The order changes no block's operands or arithmetic, and the
    // writer waits for all seven outputs of a chunk before it sends any, so their push order is free. Pops are
    // ordered after this thread's unpacks by the CB protocol, so each input is popped after its last consumer.
    //   decay -> [D_q] -> v_beta -> [q_normed] -> decay_row -> [D_k] -> L_mask -> [k_normed] -> q_decay -> k_beta
    //   -> intra -> WY factor -> T_inv -> nkd -> k_dec -> dl -> k_dec_t          ([..]: in-kernel qk-norm only)
    // T_inv is the exception: it stays right behind its factor (see pp_tinv).
    // Scratch: cb.scr3 holds D_q, D_k, then the WY factor; cb.scr1 holds decay_row, then the Horner inverse's
    // temporaries; k_dec goes through cb.kbeta's ring once nkd has consumed k_beta (see pp_kdec).
    uint32_t Q = cb.q, Kk = cb.k;

    {
        GDN_ZONE("pp_decay");
        // ---- decay = tril@g, decay_exp, decayfac = exp(g_sum - decay), dl = exp(g_sum): one DST pass ----
        decay_all(cb.tril, cb.ones, cb.g, cb.decay, cb.decay_exp, cb.decayfac, ct);
        POP(cb.g, Ct);
    }

    // In-kernel L2-norm of q,k over K (fold q's scale). Consumes the raw reader q/k and produces normalized
    // q->cb.supd, k->cb.stmp (both free in Ct==1); the rest of the chunk then reads Q/Kk instead of cb.q/cb.k.
    // Each operand takes two blocks: D = diag(rsqrt(rowsum(x^2) + eps) [* scale]) from x @ x^T in one DST pass
    // (cb.scr3), then x_normed = D @ x.
    if constexpr (qk_norm) {
        GDN_ZONE("pp_norm_dq");
        inv_rms_diag</*DO_SCALE=*/true>(cb.q, cb.eye, cb.scr3, ct, Kt, eps_bits, scale_bits);
    }

    {
        GDN_ZONE("pp_vbeta");
        bcast_cols_mul(cb.v, cb.beta, cb.vbeta, ct, Vt);  // v_beta -> the writer
        POP(cb.v, cv);
    }

    if constexpr (qk_norm) {
        GDN_ZONE("pp_norm_q");
        WAIT(cb.scr3, Ct);
        mm_diag(cb.scr3, cb.q, cb.supd, ct, Kt);
        POP(cb.scr3, Ct);
        POP(cb.q, ck);
        Q = cb.supd;
    }

    {
        GDN_ZONE("pp_decay_row");
        WAIT(cb.decay, Ct);
        transpose_col(cb.decay, cb.scr1, ct);  // decay_row in scr1
    }

    if constexpr (qk_norm) {
        GDN_ZONE("pp_norm_dk");
        inv_rms_diag</*DO_SCALE=*/false>(cb.k, cb.eye, cb.scr3, ct, Kt, eps_bits, scale_bits);
    }

    {
        GDN_ZONE("pp_lmask");
        // ---- L_mask = tril(exp(decay_i - decay_j)): one DST pass per tile (was five packed blocks) ----
        WAIT(cb.scr1, Ct);
        lmask_fused(cb.ones, cb.decay, cb.scr1, cb.tril, cb.lmask, ct);
        POP(cb.scr1, Ct);  // decay_row done
        POP(cb.decay, Ct);
    }

    if constexpr (qk_norm) {
        GDN_ZONE("pp_norm_k");
        WAIT(cb.scr3, Ct);
        mm_diag(cb.scr3, cb.k, cb.stmp, ct, Kt);
        POP(cb.scr3, Ct);
        POP(cb.k, ck);
        Kk = cb.stmp;
    }

    {
        GDN_ZONE("pp_qdecay");
        // ---- q_decay = q*decay_exp ----
        WAIT(Q, ck);
        WAIT(cb.decay_exp, Ct);
        bcast_cols_mul(Q, cb.decay_exp, cb.qdecay, ct, Kt);
    }

    {
        GDN_ZONE("pp_kbeta");
        WAIT(Kk, ck);
        bcast_cols_mul(Kk, cb.beta, cb.kbeta, ct, Kt);
        POP(cb.beta, Ct);
    }

    {
        GDN_ZONE("pp_intra");
        // ---- intra = (q@k^T) * L_mask ----
        WAIT(cb.lmask, cc);
        intra_fused(Q, Kk, cb.lmask, cb.intra, ct, Kt);  // intra = (q @ k^T) * L_mask, one DST pass per tile
        POP(Q, ck);
    }

    {
        GDN_ZONE("pp_negn");
        // ---- N = strictly_lower(k_beta@k^T * L_mask); T_inv = (I + strictly_lower)^-1 ----
        // The WY inverse. With GDN_TINV_SFPU (FORWARD_SUBSTITUTION, AUTO on Blackhole at chunk 32) it is one
        // SFPU forward substitution on N, see sfpu_tinv. HORNER mirrors FLA's solve_tril: block down to 16x16
        // (invert_block splits each 32x32 tile into 16-quadrants), invert the small diagonal blocks with bounded
        // Horners, and merge off-diagonal blocks exactly. This keeps every intermediate bounded, unlike a single
        // 32x32/full-matrix Horner whose deep power series loses fp32 precision on harder chunks.
        // The factor is kept in cb.scr3, one DST pass per tile (kk from the matmul, the masks applied on the SFPU)
        // instead of four packed blocks: negN = -(strictly_lower(kk * L_mask)) = -A_strict for Horner, which reads
        // the whole tile; only the strict lower triangle of kk * L_mask for the solve, which reads nothing else.
        WAIT(cb.kbeta, ck);
#if defined(GDN_TINV_SFPU)
        static_assert(Ct == 1, "the SFPU WY-inverse solve is a single-tile routine (chunk_size == 32)");
        wy_factor_lower(cb.kbeta, Kk, cb.lmask, cb.scr3, Kt);
#else
        negn_fused(cb.kbeta, Kk, cb.lmask, cb.eye, cb.scr3, ct, Kt);
#endif
        POP(cb.lmask, cc);
    }

    {
        GDN_ZONE("pp_tinv");
        // T_inv directly follows its factor, with no other block in between. The forward-substitution solve
        // stashes its rows with SFPSTOREs into DST rows that no FPU instruction has written, so every earlier DST
        // section must already be released when it starts (see the SFPU pass note). This WAIT guarantees it: the
        // factor is pushed by the block just before, after that block's own release.
        WAIT(cb.scr3, cc);
        // invert_block's private scratch A..D = cb.S/cb.final_s/cb.s2/cb.s3 — all fp32 and NOT drained
        // by the prep writer (unlike the output CBs cb.w/cb.qdecay/cb.intra, whose scratch pushes the
        // writer would wrongly consume). None alias src (cb.scr3), out, or the Ct==2 persistents
        // (cb.supd/cb.stmp).
        if constexpr (Ct == 1) {
#if defined(GDN_TINV_SFPU)
            // One SFPU forward-substitution solve, reading N in place.
            sfpu_tinv(cb.scr3, cb.eye, cb.Tinv);
            POP(cb.scr3, cc);
#else
            // Single 32x32 block: T_inv is just its inverse.
            invert_block(cb.scr3, 0, cb.Tinv, cb.scr1, cb.scr2, cb.eye, cb.mask, cb.S, cb.final_s, cb.s2, cb.s3);
            WAIT(cb.Tinv, cc);
            POP(cb.scr3, cc);
#endif
        } else if constexpr (Ct == 2) {
            // 2x2 tile-block lower-triangular. negN tiles: 0=(0,0), 2=(1,0), 3=(1,1); (0,1)=0.
            // Diagonal inverses Mi11, Mi22, then off-diagonal Mi21 = -Mi22 @ A21 @ Mi11.
            // (A21 = -negN21, so -Mi22@A21@Mi11 = Mi22 @ negN21 @ Mi11.)
            // Mi11 -> cb.supd, Mi22 -> cb.stmp, Mi21 -> cb.ointer (all free in prep).
            // Mi11 -> cb.supd (negN tile 0), Mi22 -> cb.stmp (negN tile 3). ONE inlined invert_block body
            // serves both through a 2-iteration loop the compiler must not unroll: inlining it twice
            // put the Ct==2 prep program over the kernel-config buffer (70752 > 70656 B on QB2), and the
            // LLK's inline asm forbids the out-of-line (noinline / -Os) alternatives. Same ops, same
            // order, same pack boundaries -> bit-exact with the unrolled form.
            {
                const uint32_t neg_tile[2] = {0, 3};
                const uint32_t inv_out[2] = {cb.supd, cb.stmp};
#pragma GCC unroll 1
                for (uint32_t i = 0; i < 2; i++) {
                    invert_block(
                        cb.scr3,
                        neg_tile[i],
                        inv_out[i],
                        cb.scr1,
                        cb.scr2,
                        cb.eye,
                        cb.mask,
                        cb.S,
                        cb.final_s,
                        cb.s2,
                        cb.s3);
                }
            }
            cpy_t(cb.scr3, 2, cb.s3);  // negN21 -> cb.s3 (invert_block's scratch D, free again)
            WAIT(cb.s3, 1);
            mm(cb.s3, cb.supd, cb.scr2, 1, 1, 1, false);  // tmp = negN21 @ Mi11
            WAIT(cb.scr2, 1);
            POP(cb.s3, 1);
            mm(cb.stmp, cb.scr2, cb.ointer, 1, 1, 1, false);  // Mi21 = Mi22 @ tmp
            WAIT(cb.ointer, 1);
            POP(cb.scr2, 1);
            POP(cb.scr3, cc);  // negN done
            // T_inv = [[Mi11, 0], [Mi21, Mi22]]  (cb.eye[1] is the zero block)
            asm4(cb.supd, 0, cb.eye, 1, cb.ointer, 0, cb.stmp, 0, cb.Tinv);
            WAIT(cb.Tinv, cc);
            POP(cb.supd, 1);
            POP(cb.stmp, 1);
            POP(cb.ointer, 1);
        } else {
            // Fallback (C>64, currently xfail): full-matrix Horner.
            ew(cb.eye, cb.scr3, cb.Tinv, cc, EwOp::Add);
            WAIT(cb.Tinv, cc);
            for (uint32_t m = 2; m < C; m++) {
                mm(cb.scr3, cb.Tinv, cb.scr1, Ct, Ct, Ct, false);
                WAIT(cb.scr1, cc);
                POP(cb.Tinv, cc);
                ew(cb.eye, cb.scr1, cb.Tinv, cc, EwOp::Add);
                WAIT(cb.Tinv, cc);
                POP(cb.scr1, cc);
            }
            POP(cb.scr3, cc);
        }
    }

    {
        GDN_ZONE("pp_kd");
        // ---- un-premultiplied WY hand-off: output v_beta (cb.vbeta), nkd = -(k_beta*decay_exp) (cb.w),
        // T_inv (cb.Tinv). The scan computes v_new = T_inv @ (v_beta + nkd@S), applying the inverse
        // AFTER the subtraction so its fp error is not amplified by the u - w@S cancellation. The
        // operand is handed off NEGATED so the scan forms v_beta + nkd@S as one DST accumulation
        // (nkd @ S, then I @ v_beta accumulated onto it): the negation is an exact SFPU sign flip of
        // the broadcast product before it is packed.
        bcast_cols_mul_neg(cb.kbeta, cb.decay_exp, cb.w, ct, Kt);  // nkd -> cb.w (output, no wait)
        POP(cb.kbeta, ck);
    }

    // cb.vbeta (v_beta) and cb.Tinv (T_inv) remain pushed for the writer; NOT popped here.

    {
        GDN_ZONE("pp_kdec");
        // k_dec = k * exp(decay_last - decay), staged in cb.kbeta's ring (k_beta was popped at pp_kd). A ck-tile
        // block needs a ring whose pushes are all ck tiles: tiles are addressed linearly from the ring pointers,
        // so in a ring that also takes single-tile pushes (cb.scr1) the block would run past the ring's end.
        WAIT(cb.decayfac, Ct + 1);
        bcast_cols_mul(Kk, cb.decayfac, cb.kbeta, ct, Kt);
        POP(Kk, ck);
    }

    {
        GDN_ZONE("pp_dl");
        // ---- dl*I: dl = exp(g_sum) (decayfac's extra tile Ct, the same value in every row of column 0)
        // broadcast down the identity -> one tile with dl on the diagonal. The scan decays the state as
        // the matmul (dl*I) @ S_tile so the update S <- dl*S + k_dec_t@v_new accumulates in one DST pass.
        dl_tile(cb.eye, cb.decayfac, Ct, cb.dl);
        POP(cb.decayfac, Ct + 1);
        POP(cb.decay_exp, Ct);
    }

    {
        GDN_ZONE("pp_kdec_t");
        // k_dec_t = transpose(k_dec) [K,C]: transpose each [Ct,Kt] tile block into [Kt,Ct].
        WAIT(cb.kbeta, ck);
        CircularBuffer(cb.kdec_t).reserve_back(Kt * Ct);
        pack_reconfig_data_format(cb.kdec_t);
        reconfig_data_format_srca(cb.kbeta);  // unary: in->srcA
        transpose_init(cb.kbeta);
        for (uint32_t ki = 0; ki < Kt; ki++) {
            for (uint32_t ci = 0; ci < Ct; ci++) {
                tile_regs_acquire();
                transpose_tile(cb.kbeta, ci * Kt + ki, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb.kdec_t, ki * Ct + ci);
                tile_regs_release();
            }
        }
        CircularBuffer(cb.kdec_t).push_back(Kt * Ct);
        POP(cb.kbeta, ck);
    }
    // v_beta, nkd, T_inv, k_dec_t, q_decay, intra, dl remain pushed in their CBs for the writer, which drains them
    // per chunk; they are NOT popped here.
}

// PHASE B (scan): one chunk of the sequential recurrence. cur_S = the state input CB for this
// chunk (reader-fed cb_S at chunk 0, then the compute-only ping-pong), dst = where the updated
// state goes (the other ping-pong CB, or the final-state CB on the last chunk).
template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
inline void scan_step(const GdnScanCbs& cb, uint32_t cur_S, uint32_t dst) {
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t kc = Kt * Ct;
    // Every CB the scan touches is fp32 (hand-off intermediates, state, scratch, o), so the
    // unpacker/packer formats set once at kernel start (compute_kernel_hw_startup on fp32 CBs) hold
    // for the whole step: skip the per-call reconfigs (the producer does the same under
    // GDN_HOIST_RECONFIG). Formats are identical either way => bit-exact.
    constexpr bool H = true;

    // v_new = T_inv @ (v_beta + nkd@S), nkd = -(k_beta*decay_exp) from prep -- apply the inverse
    // after the subtraction so the WY inverse's fp error is not amplified by the cancellation (vs the
    // u - w@S form). diff is formed in ONE DST pass of matmul accumulation: nkd @ S, then I @ v_beta
    // onto the same DST tile (MVMUL adds into DST; only the packer zeroes it at release). This replaces
    // the packed kd@S, its re-unpack and the eltwise block of the two-block form; the identity matmul
    // keeps the window inside one op class (no eltwise init or reconfig between the products). One
    // rounding moves: v_beta enters the sum through srcA like every other matmul operand instead of
    // through the eltwise add. diff itself must be packed: it is the in1 operand of T_inv @ diff, and
    // the WAIT on cb.ointer below is what keeps that block's unpacker behind this block's packer.
    {
        GDN_ZONE("st_diff");
        WAIT(cb.nkd, ck);
        WAIT(cb.vbeta, cv);
        WAIT(cur_S, kv);
        CircularBuffer(cb.ointer).reserve_back(cv);
        matmul_init(cb.nkd, cur_S, 0);  // one init serves both products: matmul_tiles binds its CBs per call
        for (uint32_t t0 = 0; t0 < cv; t0 += kDstTiles) {
            const uint32_t nb = (cv - t0 < kDstTiles) ? (cv - t0) : kDstTiles;
            tile_regs_acquire();
            for_row_runs(t0, nb, Vt, [&](uint32_t mi, uint32_t ni, uint32_t j, uint32_t n) {
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    mm_tiles_row(cb.nkd, cur_S, mi * Kt + ki, ki * Vt + ni, j, n);  // DST[j..]  = nkd @ S
                }
                mm_tiles_row(cb.eye, cb.vbeta, 0, t0 + j, j, n);  // DST[j..] += I @ v_beta
            });
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nb; j++) {
                pack_tile(j, cb.ointer, t0 + j);
            }
            tile_regs_release();
        }
        CircularBuffer(cb.ointer).push_back(cv);
        POP(cb.nkd, ck);
        WAIT(cb.ointer, cv);  // the next block's unpacker must not run ahead of this block's packer
        POP(cb.vbeta, cv);
    }
    {
        GDN_ZONE("st_vnew");
        WAIT(cb.Tinv, cc);
        mm(cb.Tinv, cb.ointer, cb.vnew, Ct, Ct, Vt, false, H);  // v_new = T_inv @ diff -> vnew
        WAIT(cb.vnew, cv);
        POP(cb.Tinv, cc);
        POP(cb.ointer, cv);
    }

    // The two outputs are each a sum of two matmuls; both sums stay in DST: the second matmul
    // accumulates onto the first (MVMUL adds into DST, which only the packer zeroes at release), so
    // there is no packed intermediate, no re-unpack and no eltwise block. matmul_tiles binds its
    // operand CBs per call, so one matmul_init serves both products of a block.
    // o = q_decay @ S + intra @ v_new -> cb_out (drained by the writer)
    {
        GDN_ZONE("st_o");
        WAIT(cb.qdecay, ck);
        WAIT(cb.intra, cc);
        CircularBuffer(cb.out).reserve_back(cv);
        matmul_init(cb.qdecay, cur_S, 0);
        for (uint32_t t0 = 0; t0 < cv; t0 += kDstTiles) {
            const uint32_t nb = (cv - t0 < kDstTiles) ? (cv - t0) : kDstTiles;
            tile_regs_acquire();
            for_row_runs(t0, nb, Vt, [&](uint32_t mi, uint32_t ni, uint32_t j, uint32_t n) {
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    mm_tiles_row(cb.qdecay, cur_S, mi * Kt + ki, ki * Vt + ni, j, n);
                }
                for (uint32_t kc_ = 0; kc_ < Ct; kc_++) {
                    mm_tiles_row(cb.intra, cb.vnew, mi * Ct + kc_, kc_ * Vt + ni, j, n);
                }
            });
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nb; j++) {
                pack_tile(j, cb.out, t0 + j);
            }
            tile_regs_release();
        }
        CircularBuffer(cb.out).push_back(cv);
        POP(cb.qdecay, ck);
        POP(cb.intra, cc);
    }
    // S_new = (dl*I) @ S + k_dec_t @ v_new -> dst (the next chunk's cur_S, or the final state).
    // The decay is block-diagonal, so every state tile (i,j) uses the single dl*I tile as in0.
    {
        GDN_ZONE("st_snew");
        WAIT(cb.kdec_t, kc);
        WAIT(cb.dl, 1);
        CircularBuffer(dst).reserve_back(kv);
        matmul_init(cb.dl, cur_S, 0);
        for (uint32_t t0 = 0; t0 < kv; t0 += kDstTiles) {
            const uint32_t nb = (kv - t0 < kDstTiles) ? (kv - t0) : kDstTiles;
            tile_regs_acquire();
            mm_tiles_row(cb.dl, cur_S, 0, t0, 0, nb);  // the one dl*I tile against the batch's consecutive state tiles
            for_row_runs(t0, nb, Vt, [&](uint32_t mi, uint32_t ni, uint32_t j, uint32_t n) {
                for (uint32_t kc_ = 0; kc_ < Ct; kc_++) {
                    mm_tiles_row(cb.kdec_t, cb.vnew, mi * Ct + kc_, kc_ * Vt + ni, j, n);
                }
            });
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nb; j++) {
                pack_tile(j, dst, t0 + j);
            }
            tile_regs_release();
        }
        CircularBuffer(dst).push_back(kv);
        POP(cb.kdec_t, kc);
        POP(cb.dl, 1);
    }
    // v_new and the input state fed both blocks; release them once everything is packed.
    POP(cb.vnew, cv);
    POP(cur_S, kv);
}
