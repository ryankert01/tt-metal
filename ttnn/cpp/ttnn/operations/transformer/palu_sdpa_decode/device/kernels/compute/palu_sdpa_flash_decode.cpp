// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This kernel reconfigs ~30x; force the Src zero-flag DEFAULT configurator out-of-line (one shared copy
// reached by a call at each reconfig/init site, instead of an inlined fast path at every one) to reclaim
// kernel-config-buffer space. Must be defined before the compute API includes. Perf-neutral (init-time only).
#define LLK_ZEROFLAG_OUTLINE 1

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define MAX_TREE_REDUCTION_ROUNDS 6

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "api/compute/eltwise_binary_sfpu.h"
#include "ttnn/operations/transformer/palu_sdpa_decode/device/kernels/compute/palu_compute_common.hpp"
#include "api/compute/pack_untilize.h"
constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    // Compile time arguments

    // Input dimensions in tiles
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);

    // Matmul configs
    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(13);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(16);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(19);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(20);

    // Attention-specific parameters
    constexpr bool is_causal = get_compile_time_arg_val(21) == 1;
    constexpr bool use_attention_mask = get_compile_time_arg_val(22) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(24);
    constexpr bool tilize_q = get_compile_time_arg_val(25) == 1;
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(26);
    constexpr bool use_half_tile = get_compile_time_arg_val(27);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(28);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(29);
    constexpr uint32_t num_tree_reduction_rounds = get_compile_time_arg_val(30);
    constexpr uint32_t original_block_size = get_compile_time_arg_val(31);
    // ---------------- Palu (stage 2) ----------------
    constexpr uint32_t Rt = get_compile_time_arg_val(32);        // latent width in tiles
    constexpr bool recon_v = get_compile_time_arg_val(33) == 1;  // V rebuilt on core from u_v
    constexpr uint32_t Rvt = get_compile_time_arg_val(34);       // V latent width in tiles
    constexpr uint32_t Wt = 2 * DHt;
    constexpr bool has_block_padding = original_block_size > 0 && original_block_size < 32;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    constexpr bool untilize_output = tilize_q;

    // CB index definitions
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_sliding_window_mask_in = tt::CBIndex::c_13;  // Separate buffer for sliding window mask
    constexpr uint32_t cb_block_pad_mask = tt::CBIndex::c_14;          // Block padding mask (block_size < TILE_HEIGHT)
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_11;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;
    // #44366: compute reads cur_pos from c_15 (writer reads from c_8) — see reader_decode_all.cpp.
    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_15;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_accumulate_im = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_1 = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_2 = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_1 = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_2 = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_prev_sum_2 = tt::CBIndex::c_21;
    constexpr uint32_t cb_exp_max_diff_2 = tt::CBIndex::c_22;
    constexpr uint32_t cb_out_accumulate_im_2 = tt::CBIndex::c_23;

    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_out_final = tt::CBIndex::c_20;

    // Palu CBs (spec §4). The stock owners of these slots must be compiled out; the program factory
    // refuses to build otherwise, and these assert the kernel side of that contract. (cb_zero_in above
    // still names c_12, but matmul_blocks only reads it when it fuses a mask into the QK matmul, which
    // needs DYNAMIC_CHUNK_SIZE and never happens here -- the writer no longer produces the zero tile.)
    static_assert(!use_attention_sink, "Palu reuses the attention-sink CB (c_4) for w_rot");
    static_assert(!tilize_q, "Palu reuses the row-major-Q CB (c_10) for the AB block; Q must arrive tilized");
    static_assert(sliding_window_size == 0, "Palu reuses the sliding-window CB (c_13) for [cos | sin]");
    static_assert(!has_block_padding, "Palu reuses the block-padding CB (c_14) for rot_step");
#ifdef DYNAMIC_CHUNK_SIZE
#error "palu_sdpa_decode: c_12 is the Phase-2 query state (the stock zero tile is gone); dynamic chunking's mask fusion would read it"
#endif
    constexpr uint32_t cb_lat_in = tt::CBIndex::c_9;
    constexpr uint32_t cb_w_rot = tt::CBIndex::c_4;
    constexpr uint32_t cb_ab = tt::CBIndex::c_10;
    constexpr uint32_t cb_cs_in = tt::CBIndex::c_13;
    constexpr uint32_t cb_qrot = tt::CBIndex::c_12;      // Phase 2: [q~ | q~ T], fp32
    constexpr uint32_t cb_rot_step = tt::CBIndex::c_14;  // Phase 2: [cos(c f) | sin(c f)], fp32
#ifdef PALU_PHASE2
    constexpr uint32_t cb_q_src = cb_qrot;  // QK reads the on-core rotated query
#define PALU_HAS_PHASE2 1
#else
    constexpr uint32_t cb_q_src = cb_q_in;
#define PALU_HAS_PHASE2 0
#endif
    // The per-head resident weight block in c_4 is [w_rot | u_v (in-kernel V) | trans_mat (Phase 2)];
    // the reader pushes it as ONE block, so these offsets and the total must match read_w_rot exactly.
    constexpr uint32_t uv_tile_base = Rt * Wt;                            // u_v follows w_rot in the block
    constexpr uint32_t t_tile_idx = uv_tile_base + (recon_v ? Rvt * DHt : 0);  // trans_mat is last
    constexpr uint32_t w_rot_cb_tiles = t_tile_idx + (PALU_HAS_PHASE2 ? 1 : 0);
    // In the V mode the rebuilt V takes c_1 (the QK matmul has already popped K_rot out of it).
    constexpr uint32_t cb_v_src = recon_v ? cb_k_in : cb_v_in;

    // Runtime arguments
    uint32_t arg_idx = 0;
    const bool do_reduce = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool apply_mask_at_last_chunk = do_reduce && is_causal;
    const bool do_output = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    // Tree reduction runtime arguments
    const bool is_tree_root = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t parent_core_in_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t send_at_round = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_children = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_active_rounds = get_arg_val<uint32_t>(arg_idx++);
    const bool has_parent = parent_core_in_group != UINT32_MAX;

    // Read children_per_round array
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        children_per_round[r] = get_arg_val<uint32_t>(arg_idx++);
    }

    // Idle core
    // get_arg_val<uint32_t>(0) can go from 0-63 for the core_num; for active cores 65 is out of range so 65 indicates
    // an idle_core
    if (get_arg_val<uint32_t>(0) == 65) {
        return;
    }

    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    if constexpr (is_causal) {
        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            // Read cur_pos from CB using mailbox-based synchronization (issue #27979).
            CircularBuffer(cb_cur_pos).wait_front(1);
            cur_pos = read_tile_value(cb_cur_pos, 0, cur_batch / q_heads_parallel_factor);
            CircularBuffer(cb_cur_pos).pop_front(1);
        }
        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    // When block_size < TILE_HEIGHT, convert cur_pos to padded tile space.
    // Only for causal mode; non-causal cur_pos is already in padded space.
    if constexpr (has_block_padding && is_causal) {
        cur_pos = (cur_pos / original_block_size) * 32 + (cur_pos % original_block_size);
    }

    // Get dynamic chunk size for K in tiles
    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Get the sequence length assignment
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            cur_batch,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);

    // Check if this core has local data to process
    const bool has_local_data = (k_chunk_start != k_chunk_end);

    // Cores without data don't participate in tree reduction at all
    // They just exit early - no sending, no receiving
    if (!has_local_data) {
        return;
    }

    // Determine which children actually participate in reduction (based on chunk allocation)
    // A child at core_num is active or has data if core_num < k_num_chunks
    // E.g. k_num_chunks = 2, num_cores_per_head = 4
    // | core 0 | core 1 | core 2 | core 3 |
    //   chunk 0  chunk 1   NA       NA
    // core 0 would have core 1 and core 2 as children, but only core 1 is active to perform reduction with
    uint32_t num_active_children = 0;
    uint32_t active_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t num_active_rounds = 0;

    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        uint32_t child_id = children_per_round[r];
        if (child_id != UINT32_MAX && child_id < k_num_chunks) {
            // This child has data
            active_children_per_round[r] = child_id;
            num_active_children++;
            num_active_rounds = r + 1;
        } else {
            active_children_per_round[r] = UINT32_MAX;
        }
    }

    // We tilize input Q if it is in ROW MAJOR layout
    if constexpr (tilize_q) {
        compute_kernel_hw_startup(cb_q_rm, cb_q_in);
        // Keep InitAndUninit: the helper picks fast- vs regular-tilize at compile time and its
        // teardown must match (fast_tilize_uninit vs tilize_uninit). tilize_q with a FULL-tile Q
        // (use_half_tile==false, e.g. >16 heads) can take the fast-tilize path, so we must not
        // hand-roll the uninit here.
        compute_kernel_lib::tilize<
            q_chunk_tiles,
            cb_q_rm,
            cb_q_in,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        matmul_init(cb_q_in, cb_k_in);
        // #49266: The Q tilize runs on SrcA; on galaxy Q is a half-tile (num_faces=2), and
        // tilize_uninit correctly restores SrcA to Q's geometry. But the QK matmul reads operands
        // REVERSED (SrcA <- in1 = cb_k_in, num_faces=4), and the per-k_chunk reconfig below is
        // IGNORE (format-only). Reprogram SrcA/SrcB tile geometry ONCE here for the matmul
        // operands (is_tile_dim_reconfig_en=true) so K is unpacked with the correct num_faces.
        // One-time, not per-chunk: nothing after this re-establishes Q's geometry on SrcA (K and V
        // are both full tiles), so the per-chunk reconfig can stay IGNORE. Without this, SrcA stays
        // at num_faces=2 and the matmul reads K wrong -> Top-1 0%. (Full-tile Q: this is a no-op
        // re-assert of num_faces=4.)
        reconfig_full_operand(cb_k_in, cb_q_in);
    } else {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
        matmul_init(cb_q_in, cb_k_in);
    }
    CircularBuffer(cb_q_in).wait_front(q_chunk_tiles);

    // Wait for block padding mask (generated once by writer, reused every chunk without popping)
    if constexpr (has_block_padding) {
        uint32_t block_pad_mask_tiles = Sq_chunk_t * Sk_chunk_t_dynamic;
        CircularBuffer(cb_block_pad_mask).wait_front(block_pad_mask_tiles);
    }

    // Define dynamic matmul configs
#ifdef DYNAMIC_CHUNK_SIZE
    const uint32_t qk_subblock_h_dynamic = 1;
    const uint32_t qk_subblock_w_dynamic = Sk_chunk_t_dynamic;  // Guaranteed < DST
    const uint32_t qk_in0_num_subblocks_dynamic = 1;
    const uint32_t qk_in1_num_subblocks_dynamic = 1;
    const uint32_t out_in0_block_w_dynamic = Sk_chunk_t_dynamic;
    const uint32_t out_num_blocks_dynamic = 1;
    const uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Sk_chunk_t_dynamic;
#else
    constexpr uint32_t qk_subblock_h_dynamic = qk_subblock_h;
    constexpr uint32_t qk_subblock_w_dynamic = qk_subblock_w;
    constexpr uint32_t qk_in0_num_subblocks_dynamic = qk_in0_num_subblocks;
    constexpr uint32_t qk_in1_num_subblocks_dynamic = qk_in1_num_subblocks;
    constexpr uint32_t out_in0_block_w_dynamic = out_in0_block_w;
    constexpr uint32_t out_num_blocks_dynamic = out_num_blocks;
    constexpr uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Sk_chunk_t;
#endif

    // - VectorMode::RC is equivalent to 32x32 tiles
    // - VectorMode::R is equivalent to 16x32 tiles
    // NOTE: Using VectorMode::RC for 16x32 tiles will be correct accuracy, just slower due to unnecessary math
    constexpr VectorMode vector_mode = use_half_tile ? VectorMode::R : VectorMode::RC;

    // We set up Ping Pong intermediate buffers between loops
    uint32_t cb_cur_max = cb_max_1;
    uint32_t cb_prev_max = cb_max_2;
    uint32_t cb_cur_sum = cb_sum_1;
    uint32_t cb_prev_sum = cb_sum_2;

#ifdef PALU_GROUP_SHARED
    /******************************************************************************
     *          GROUP-SHARED MODE (stage 2c): chunk outer, head inner              *
     ******************************************************************************
     * The num_heads_per_core KV heads this core owns all sit in one latent group, so the reader
     * pushes each latent chunk once and every head rebuilds its own K (and V) from it with its own
     * resident weight block in c_4. Everything else is the per-head kernel in the #else branch, in
     * the same order: head h's math and chunk order are untouched, so its output is bit-identical.
     *
     * The per-head flash state (m in c_28, l in c_30, O in c_26) lives in those CBs as a ring of H
     * blocks: every head runs the identical wait/pop/reserve/push sequence on each of them per chunk,
     * so the ring hands head h its own block back at the front on the next chunk. Two in-place
     * helpers rotate the block they update to the back of the ring in the middle of a head's step and
     * are read again right after (mul_block_inplace on l, mul_block_bcast_cols on O); after those the
     * ring is closed back up with palu_rotate_blocks, which is pointer-only -- the data never moves.
     * The other state CBs (c_24, c_25, c_27, c_29, c_31) are emptied inside a head's step.
     *
     * Q is per (batch, core), not per head: the reader reads one Q block for the core and the writer
     * picks head h's rows out of each result, so all H heads share c_0 and, in Phase 2, the single
     * rotated-query state in c_12 -- same seed, same per-chunk step for every head, so computing it
     * once is bit-identical to computing it H times.
     */
    static_assert(recon_v, "group-shared mode rebuilds V on core: grouped latents always carry u_v");
    static_assert(is_causal && !use_attention_mask, "group-shared mode is causal-only (Palu envelope)");
    static_assert(!use_attention_sink, "group-shared mode has no attention-sink finalize");
    static_assert(!untilize_output, "group-shared mode emits tilized output only");
    constexpr uint32_t H = num_heads_per_core;
    CircularBuffer(cb_w_rot).wait_front(H * w_rot_cb_tiles);  // the core's H weight blocks, resident

#ifdef PALU_PHASE2
    // q~ = q R_{-k_chunk_start c}: seeded once for the core (trans_mat is the same tile in every block)
    palu_init_q_state<cb_q_in, cb_qrot, cb_w_rot, q_chunk_tiles, w_rot_cb_tiles>(t_tile_idx);
    for (uint32_t s = 0; s < k_chunk_start; ++s) {
        palu_step_q<cb_qrot, cb_rot_step, DHt, Sq_chunk_t>();
    }
#endif

    uint32_t cb_out_mm = cb_out_accumulate_im;  // the first chunk seeds O, later chunks add into it
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
#ifdef PALU_PHASE2
        if (k_chunk > k_chunk_start) {
            palu_step_q<cb_qrot, cb_rot_step, DHt, Sq_chunk_t>();  // q~ <- q~ R_{-c}
        }
#endif
        for (uint32_t h = 0; h < H; ++h) {
            const uint32_t w_base = h * w_rot_cb_tiles;  // this head's [u_k | u_k T | u_v | T] block

            // ---- Palu: rebuild this head's K for the shared chunk into cb_k_in ----
            palu_build_k_chunk<cb_lat_in, cb_w_rot, cb_ab, cb_cs_in, cb_k_in, Rt, DHt, PALU_DST_TILES, false>(
                Sk_chunk_t_dynamic, /*cs_tile_base=*/0, w_base);

            /* QK = Q_CHUNK @ K_CHUNK */
            reconfig_data_format(cb_k_in, cb_q_src);
            pack_reconfig_data_format(cb_qk_im);
            matmul_blocks(
                cb_q_src,
                cb_k_in,
                cb_qk_im,
                Sq_chunk_t,
                Sk_chunk_t_dynamic,
                DHt,
                qk_num_blocks,
                qk_in0_num_subblocks_dynamic,
                qk_in1_num_subblocks_dynamic,
                qk_in0_block_w,
                qk_subblock_h_dynamic,
                qk_subblock_w_dynamic,
                true,
                false,
                cb_mask_in,
                cb_zero_in);

            /* QK += MASK: causal decode masks the last chunk only, and never pops the mask CB */
            if (k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk) {
                reconfig_data_format(cb_qk_im, cb_mask_in);
                add_block_inplace<false>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
            }

            /* CUR_MAX = eltwise_max(PREV_MAX, max(QK, dim=-1)) */
            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
            pack_reconfig_data_format(cb_cur_max);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                cb_cur_max, cb_prev_max, Sk_chunk_t_dynamic, k_chunk > k_chunk_start);

            /* QK = exp((QK - CUR_MAX) * scale) */
            reconfig_data_format(cb_qk_im, cb_cur_max);
            pack_reconfig_data_format(cb_qk_im);
            sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true, false, vector_mode>(
                cb_cur_max, cb_cur_sum, Sk_chunk_t_dynamic);
            CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles_dynamic);

            /* CUR_SUM = sum(QK, dim=-1) */
            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
            pack_reconfig_data_format(cb_cur_sum);
            reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);

            // V_h for this chunk: the QK matmul has consumed cb_k_in, so the rebuilt V takes its slot
            palu_build_v_chunk<cb_v_in, cb_w_rot, cb_k_in, Rvt, vDHt, PALU_DST_TILES, false>(
                Sk_chunk_t_dynamic, w_base + uv_tile_base);

            /* OUT_IM = QK @ V_CHUNK */
            reconfig_data_format(cb_v_src, cb_qk_im);
            pack_reconfig_data_format(cb_out_im);
            matmul_blocks(
                cb_qk_im,
                cb_v_src,
                cb_out_mm,
                Sq_chunk_t,
                vDHt,
                Sk_chunk_t_dynamic,
                out_num_blocks_dynamic,
                out_in0_num_subblocks,
                out_in1_num_subblocks,
                out_in0_block_w_dynamic,
                out_subblock_h,
                out_subblock_w,
                false /*transpose*/,
                false,
                cb_mask_in,
                cb_zero_in);
            reconfig_data_format_srca(cb_out_im);
            CircularBuffer(cb_qk_im).pop_front(qk_chunk_tiles_dynamic);

            if (k_chunk > k_chunk_start) {
                /* EXP_MAX_DIFF = exp(PREV_MAX - CUR_MAX) */
                reconfig_data_format(cb_prev_max, cb_cur_max);
                pack_reconfig_data_format(cb_exp_max_diff);
                sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);

                /* PREV_SUM *= EXP_MAX_DIFF (rotates this head's block to the back: close the ring) */
                mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);
                palu_rotate_blocks(cb_prev_sum, Sq_chunk_t, H - 1);

                /* OUT_ACC *= EXP_MAX_DIFF (same rotation) */
                reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff);
                pack_reconfig_data_format(cb_out_accumulate_im);
                mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(
                    cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im);
                palu_rotate_blocks(cb_out_accumulate_im, out_chunk_tiles, H - 1);

                /* CUR_SUM += PREV_SUM */
                reconfig_data_format(cb_cur_sum, cb_prev_sum);
                pack_reconfig_data_format(cb_cur_sum);
                add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

                /* OUT_ACC += OUT_IM */
                reconfig_data_format(cb_out_accumulate_im, cb_out_im);
                pack_reconfig_data_format(cb_out_accumulate_im);
                add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
            }

            // PREV_MAX <- CUR_MAX, PREV_SUM <- CUR_SUM (this head's block goes back to the ring)
            reconfig_data_format(cb_cur_max, cb_cur_max);
            pack_reconfig_data_format(cb_prev_max);
            move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
            move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

            if (h + 1 == H) {  // the shared chunk is spent: release it for the reader's next push
                CircularBuffer(cb_lat_in).pop_front(Sk_chunk_t_dynamic * Rt);
                CircularBuffer(cb_v_in).pop_front(Sk_chunk_t_dynamic * Rvt);
#ifndef PALU_PHASE2
                CircularBuffer(cb_cs_in).pop_front(Sk_chunk_t_dynamic * Wt);  // Phase 1: streamed rows
#endif
            }
        }
        cb_out_mm = cb_out_im;
    }

    /* END OF FLASH ATTENTION LOOP */
    CircularBuffer(cb_w_rot).pop_front(H * w_rot_cb_tiles);
#ifdef PALU_PHASE2
    CircularBuffer(cb_qrot).pop_front(2 * q_chunk_tiles);  // the last [q~ | q~ T]
#endif

    /******************************************************************************
     *      PER-HEAD TREE REDUCTION AND FINALIZE (Phase B): strictly sequential     *
     ******************************************************************************
     * A core can own H heads *and* share the sequence split with num_cores_per_head cores. Head h's
     * reduction and its output have to be interleaved, not run as an H-long pass each: the writer's
     * loop is "receive head h, then send/write head h", so a compute that finished every head's
     * reduction before emitting head 0's payload would deadlock against a writer waiting for head 0's
     * payload before receiving head 1's.
     *
     * The head's state is *peeled out of the rings first*. c_26 (O), c_28 (m) and c_30 (l) hold one
     * block per head with head h at the front, and this loop pops exactly one block from each, in head
     * order -- the rings are never rotated here. Everything after the peel is the per-head kernel's
     * reduction and finalize, verbatim, on single-block CBs: no FIFO bookkeeping to get wrong, and the
     * numbers go through the same helpers in the same order, so head h's result is bit-identical.
     *
     * The three scratch CBs are the chunk loop's own, provably drained once the loop is over: c_25 held
     * the PV product (and is exactly one O block), c_24 the QK scores, c_10 the AB rows. All three are
     * bf16, like the state they take over.
     *
     * The payload CBs (cb_out_o / cb_out_m / cb_out_l) and the reduction scratch (c_21, c_22, c_23,
     * c_27, c_29, c_31) are single-block for the same reason: head h's payload is consumed by the
     * writer before head h + 1's is produced, and that single block is the back-pressure keeping the
     * two in step. cb_intermed_out is the one exception and IS sized per head (see the program
     * factory): its producer is a *remote* core, which nothing here can back-pressure.
     */
    constexpr uint32_t cb_o_head = cb_out_im;  // c_25: exactly out_chunk_tiles, i.e. one O block
    constexpr uint32_t cb_m_head = cb_qk_im;   // c_24
    constexpr uint32_t cb_l_head = cb_ab;      // c_10
    static_assert(Sq_chunk_t * Sk_chunk_t >= Sq_chunk_t, "c_24 must hold one statistics block");
    static_assert(2 * Wt >= Sq_chunk_t, "c_10 must hold one statistics block");

    for (uint32_t head = 0; head < H; ++head) {
        // Peel head's O, m and l out of the rings (front = head, in head order).
        reconfig_data_format(cb_out_accumulate_im, cb_out_accumulate_im);
        pack_reconfig_data_format(cb_o_head);
        move_block<true>(cb_out_accumulate_im, cb_o_head, out_chunk_tiles);
        reconfig_data_format(cb_prev_max, cb_prev_max);
        pack_reconfig_data_format(cb_m_head);
        move_block<true>(cb_prev_max, cb_m_head, Sq_chunk_t);
        pack_reconfig_data_format(cb_l_head);
        move_block<true>(cb_prev_sum, cb_l_head, Sq_chunk_t);

        // Tree reduction: merge each child's (l, m, O) for this head. The writer has already moved the
        // child's payload into cb_l_in / cb_m_in / cb_out_o, one round at a time.
        if (num_active_children > 0) {
            for (uint32_t round = 0; round < num_active_rounds; ++round) {
                if (active_children_per_round[round] == UINT32_MAX) {
                    continue;
                }
                move_block<true>(cb_l_in, cb_prev_sum_2, Sq_chunk_t);

                // Softmax correction, spelled out with the primitives the flash loop itself uses rather
                // than the fused correction_block: cur_max = max(m, m_child), the two
                // exp((m - cur_max) * scale) factors, then l and O rescaled and added.
                //
                // correction_block does exactly this in one dst acquisition (five operands through dst
                // registers 0..4, fused_max_sub_exp_add_tile), and it is what the per-head branch below
                // uses -- but in *this* branch it returns the local max as the merged max, which sends
                // the child's exp factor to zero and drops its contribution entirely. Measured: the op's
                // output was then the reducer core's own slice, normalised by its own l, to the last bit
                // (PCC 1.0000 against a host reference over that slice) while the child's l, m and O were
                // all verified to arrive correctly in c_7 / c_6 / c_16. Why the fused helper misbehaves
                // here and not in the per-head branch was not established; the primitives below are the
                // same arithmetic in the same order and cost four extra dst acquisitions per child.
                max_block<vector_mode>(cb_m_head, cb_m_in, cb_cur_max, Sq_chunk_t);
                sub_exp_block<scale_fp32>(cb_m_head, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                sub_exp_block<scale_fp32>(cb_m_in, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);

                // l = l * EXP_MAX_DIFF + l_child * EXP_MAX_DIFF_2
                mul_block_inplace(cb_l_head, cb_exp_max_diff, Sq_chunk_t);
                mul_block_inplace(cb_prev_sum_2, cb_exp_max_diff_2, Sq_chunk_t);
                add_block_inplace<true>(cb_l_head, cb_prev_sum_2, Sq_chunk_t);

                // OUT_ACC_2 <- CHILD_OUT; O = O * EXP_MAX_DIFF + O_child * EXP_MAX_DIFF_2
                move_block<true>(cb_out_o, cb_out_accumulate_im_2, out_chunk_tiles);
                mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_o_head, cb_exp_max_diff);
                mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im_2, cb_exp_max_diff_2);
                add_block_inplace<true>(cb_o_head, cb_out_accumulate_im_2, out_chunk_tiles);

                // m <- cur_max
                CircularBuffer(cb_m_head).pop_front(Sq_chunk_t);
                CircularBuffer(cb_m_in).pop_front(Sq_chunk_t);
                move_block<true>(cb_cur_max, cb_m_head, Sq_chunk_t);
            }
        }

        if (is_tree_root) {
            /* SUM = 1.0 / SUM */
            reconfig_data_format(cb_l_head, cb_l_head);
            pack_reconfig_data_format(cb_l_head);
            recip_block_inplace(cb_l_head, Sq_chunk_t);

            /* OUT_ACC *= 1/SUM (cb_l_head is consumed and popped) */
            reconfig_data_format(cb_o_head, cb_l_head);
            pack_reconfig_data_format(cb_o_head);
            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_o_head, cb_l_head);

            pack_reconfig_data_format(cb_out_final);
            CircularBuffer(cb_m_head).pop_front(Sq_chunk_t);
            move_block<true>(cb_o_head, cb_out_final, out_chunk_tiles);
        } else if (has_parent) {
            // Emit this head's payload for the writer to send to the parent; the three CBs hold one
            // block each, so the next head waits here until the writer has sent and popped them.
            move_block<true>(cb_o_head, cb_out_o, out_chunk_tiles);
            move_block<true>(cb_m_head, cb_out_m, Sq_chunk_t);
            move_block<true>(cb_l_head, cb_out_l, Sq_chunk_t);
        }
    }

    // Free up cb_q_in after Q chunks
    CircularBuffer(cb_q_in).pop_front(q_chunk_tiles);
#else
    // Loop through all heads assigned to core
    for (uint32_t cur_head_work = 0; cur_head_work < num_heads_per_core; ++cur_head_work) {
        // Reset ping-pong buffer assignments at the start of each head iteration
        cb_cur_max = cb_max_1;
        cb_prev_max = cb_max_2;
        cb_cur_sum = cb_sum_1;
        cb_prev_sum = cb_sum_2;

#ifdef PALU_PHASE2
        // q~ = q R_{-k_chunk_start * c}: seed [q | q T] from Q, then one R_{-c} step per chunk this
        // core skips (cores that own a later slice of the context start at their first chunk).
        palu_init_q_state<cb_q_in, cb_qrot, cb_w_rot, q_chunk_tiles, w_rot_cb_tiles>(t_tile_idx);
        for (uint32_t s = 0; s < k_chunk_start; ++s) {
            palu_step_q<cb_qrot, cb_rot_step, DHt, Sq_chunk_t>();
        }
#endif

        /******************************************************************************
         *                           FLASH ATTENTION LOOP                             *
         ******************************************************************************/
        /**
         * Compute Parameters (most are compile time but some are dynamic):
         * @tparam St - Total sequence length in tiles
         * @tparam DHt - Head dimension in tiles
         * @tparam Sq_chunk_t - Query chunk size in tiles
         * @tparam Sk_chunk_t - Key chunk size in tiles (dynamic)
         * @tparam qk_in0_block_w - QK matmul block width
         * @tparam qk_subblock_w - QK matmul subblock width (dynamic)
         * @tparam qk_subblock_h - QK matmul subblock height (dynamic)
         * @tparam qk_in0_num_subblocks - QK input0 subblocks (dynamic)
         * @tparam qk_in1_num_subblocks - QK input1 subblocks (dynamic)
         * @tparam qk_num_blocks - QK number of blocks
         * @tparam out_in0_block_w - Output matmul block width (dynamic)
         * @tparam out_subblock_w - Output matmul subblock width
         * @tparam out_subblock_h - Output matmul subblock height
         * @tparam out_in0_num_subblocks - Output input0 subblocks
         * @tparam out_in1_num_subblocks - Output input1 subblocks
         * @tparam out_num_blocks - Output number of blocks (dynamic)
         * @tparam is_causal - Whether to use causal attention (if mask is applied)
         * @tparam use_attention_mask - Whether to use attention mask for non-causal attention
         *
         * Circular Buffer Parameters:
         * @tparam cb_q_in - Query input buffer
         * @tparam cb_k_in - Key input buffer
         * @tparam cb_v_in - Value input buffer
         * @tparam cb_mask_in - Mask input buffer
         * @tparam cb_scale_in - Scale input buffer
         * @tparam cb_identity_scale_in - Identity scale buffer
         * @tparam cb_qk_im - QK intermediate buffer
         * @tparam cb_out_im - Output intermediate buffer
         * @tparam cb_out_accumulate_im - Output accumulate buffer
         * @tparam cb_cur_max - Current max buffer
         * @tparam cb_prev_max - Previous max buffer
         * @tparam cb_cur_sum - Current sum buffer
         * @tparam cb_prev_sum - Previous sum buffer
         * @tparam cb_exp_max_diff - Exp max diff buffer
         * @tparam cb_out_o - Output O buffer
         * @tparam cb_out_m - Output M buffer
         * @tparam cb_out_l - Output L buffer
         *
         * Runtime Parameters:
         * @param k_chunk_start - Start index of key chunk
         * @param k_chunk_end - End index of key chunk
         * @param do_reduce - Whether to perform reduction
         * @param qk_chunk_tiles - Number of QK chunk tiles (dynamic)
         * @param out_chunk_tiles - Number of output chunk tiles
         */
        /* START OF FLASH ATTENTION LOOP */
        uint32_t cb_out_mm = cb_out_accumulate_im;

        // Loop through all K chunks
        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
#ifdef PALU_PHASE2
            if (k_chunk > k_chunk_start) {
                palu_step_q<cb_qrot, cb_rot_step, DHt, Sq_chunk_t>();  // q~ <- q~ R_{-c}
            }
#endif
            // ---------------- Palu: rebuild this chunk's K on core into cb_k_in ----------------
            palu_build_k_chunk<cb_lat_in, cb_w_rot, cb_ab, cb_cs_in, cb_k_in, Rt, DHt, PALU_DST_TILES>(
                Sk_chunk_t_dynamic, /*cs_tile_base=*/0);
#ifndef PALU_PHASE2
            CircularBuffer(cb_cs_in).pop_front(Sk_chunk_t_dynamic * Wt);  // Phase 1: streamed rows, consumed
#endif

            // Reconfig register DF
            reconfig_data_format(cb_k_in, cb_q_src);
            pack_reconfig_data_format(cb_qk_im);

            // OPTIMIZATION: Add the attention mask directly on top of DST if chunk sizes are dynamic
#ifdef DYNAMIC_CHUNK_SIZE
                bool add_causal_mask_fusion = is_causal && k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk;
                bool add_sliding_window_mask_fusion = k_chunk == window_start_chunk && window_start_unaligned > 0;
                bool add_mask_fusion = add_causal_mask_fusion || use_attention_mask || add_sliding_window_mask_fusion;
#else
                bool add_mask_fusion = false;
                bool add_sliding_window_mask_fusion = false;
#endif

                /* QK = Q_CHUNK @ K_CHUNK */
                // Determine which mask buffer to use for fusion
                uint32_t mask_cb_to_use = cb_mask_in;  // Default to causal mask buffer
                if (add_sliding_window_mask_fusion) {
                    mask_cb_to_use = cb_sliding_window_mask_in;  // Use sliding window mask buffer
                }

                matmul_blocks(
                    cb_q_src,
                    cb_k_in,
                    cb_qk_im,
                    Sq_chunk_t,
                    Sk_chunk_t_dynamic,
                    DHt,
                    qk_num_blocks,
                    qk_in0_num_subblocks_dynamic,
                    qk_in1_num_subblocks_dynamic,
                    qk_in0_block_w,
                    qk_subblock_h_dynamic,
                    qk_subblock_w_dynamic,
                    true,
                    add_mask_fusion,
                    mask_cb_to_use,
                    cb_zero_in);

                /* QK += MASK */
                // Apply block padding mask for every chunk when block_size < TILE_HEIGHT.
                // Uses <false> to NOT pop the mask CB so it can be reused for subsequent chunks.
                // Applied outside mask_fusion conditional since it's always needed independently.
                if constexpr (has_block_padding) {
                    reconfig_data_format(cb_qk_im, cb_block_pad_mask);
                    add_block_inplace<false>(cb_qk_im, cb_block_pad_mask, qk_chunk_tiles_dynamic);
                }

                if (!add_mask_fusion) {
                    if constexpr (is_causal) {
                        // For decode, we only apply mask at the last chunk for causal mode
                        if (k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk) {
                            reconfig_data_format(cb_qk_im, cb_mask_in);
                            add_block_inplace<false>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                        }
                    } else {
                        if constexpr (use_attention_mask) {
                            reconfig_data_format(cb_qk_im, cb_mask_in);
                            add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                        }
                    }

                    // Apply sliding window mask to the first chunk (only on the core that processes it)
                    if (k_chunk == window_start_chunk && window_start_unaligned > 0) {
                        reconfig_data_format(cb_qk_im, cb_sliding_window_mask_in);
                        add_block_inplace<false>(cb_qk_im, cb_sliding_window_mask_in, qk_chunk_tiles_dynamic);
                    }
                }

                /**
                 * OPTIMIZATION
                 * Typically, scores are multiplied by a scalar here, but an optimization was employed
                 * where the scaling is fused into exp both in exp(x - max) and exp(prev_max - cur_max).
                 * This gives us scaling for free on the performance-critical exp(x - max) computation.
                 */

                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_max);

                /**
                 * OPTIMIZATION
                 * reduce_c can perform both reduce_max and eltwise max with previous result.
                 * if do_eltwise_max:
                 *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
                 * else:
                 *  cur_max = max(qk, dim=-1)
                 */
                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_max, cb_prev_max, Sk_chunk_t_dynamic, k_chunk > k_chunk_start);
                /* QK -= cb_cur_max */
                /* QK = exp(QK)*/
                reconfig_data_format(cb_qk_im, cb_cur_max);
                pack_reconfig_data_format(cb_qk_im);

                /**
                 * sub_exp performs `QK = exp((QK - cur_max) * scale)`
                 */
                sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true, false, vector_mode>(
                    cb_cur_max, cb_cur_sum, Sk_chunk_t_dynamic);
                CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles_dynamic);

                // Reconfig register DF
                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_sum);

                /* reduce_c performs CUR_SUM = sum(QK, dim = -1) */
                reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);

#ifdef PALU_RECON_V
                // V_h for this chunk: the QK matmul above has consumed cb_k_in, so the rebuilt V takes its slot
                palu_build_v_chunk<cb_v_in, cb_w_rot, cb_k_in, Rvt, vDHt, PALU_DST_TILES>(
                    Sk_chunk_t_dynamic, uv_tile_base);
#endif
                /* OUT_IM = QK @ V_CHUNK */
                reconfig_data_format(cb_v_src, cb_qk_im);  // DEBUG
                pack_reconfig_data_format(cb_out_im);
                matmul_blocks(
                    cb_qk_im,
                    cb_v_src,
                    cb_out_mm,
                    Sq_chunk_t,
                    vDHt,
                    Sk_chunk_t_dynamic,
                    out_num_blocks_dynamic,
                    out_in0_num_subblocks,
                    out_in1_num_subblocks,
                    out_in0_block_w_dynamic,
                    out_subblock_h,
                    out_subblock_w,
                    false /*transpose*/,
                    false,
                    cb_mask_in,
                    cb_zero_in);

                // Reconfig register DF
                reconfig_data_format_srca(cb_out_im);
                CircularBuffer(cb_qk_im).pop_front(qk_chunk_tiles_dynamic);

                /* OUT_ACC += OUT_IM */
                if (k_chunk == k_chunk_start) {
                    cb_out_mm = cb_out_im;
                } else {
                    // When there is more than 1 chunk, we perform Lazy Softmax
                    // Reconfig register DF
                    reconfig_data_format(cb_prev_max, cb_cur_max);
                    pack_reconfig_data_format(cb_exp_max_diff);

                    /* EXP_MAX_DIFF = exp(PREV_MAX - CUR_MAX) */
                    sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                    CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);

                    /* PREV_SUM *= EXP_MAX_DIFF */
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                    /* OUT_ACC *= EXP_MAX_DIFF */
                    reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(
                        cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im);

                    /* CUR_SUM += PREV_SUM */
                    reconfig_data_format(cb_cur_sum, cb_prev_sum);
                    pack_reconfig_data_format(cb_cur_sum);
                    add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

                    /* OUT_ACC += OUT_IM */
                    reconfig_data_format(cb_out_accumulate_im, cb_out_im);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                }

                // More local chunks to process - move intermediate sum and max values to ping-pong buffers
                reconfig_data_format(cb_cur_max, cb_cur_max);
                pack_reconfig_data_format(cb_prev_max);

                // PREV_MAX <- CUR_MAX
                move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);

                // PREV_SUM <- CUR_SUM
                move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
        }

        /* END OF FLASH ATTENTION LOOP */
        CircularBuffer(cb_w_rot).pop_front(w_rot_cb_tiles);  // this head's [u_k | u_k T] (+ trans_mat)
#ifdef PALU_PHASE2
        CircularBuffer(cb_qrot).pop_front(2 * q_chunk_tiles);  // the last [q~ | q~ T]
#endif

        /******************************************************************************
         *                      TREE REDUCTION LOGIC                                  *
         ******************************************************************************/
        /**
         * Tree reduction reduces the online softmax results in O(log n) rounds.
         *
         * For each round r (0 to my_active_rounds-1):
         *   - If children_per_round[r] != UINT32_MAX, receive from that child
         *   - Combine received data with local accumulator using softmax correction
         *
         * After all receives:
         *   - If is_tree_root: finalize (1/sum normalization) and output
         *   - Else: output intermediate results for writer to send to parent
         */
        // Tree reduction: receive from children and combine
        // Buffer state entering tree reduction:
        //   - cb_out_accumulate_im: local O (output accumulator)
        //   - cb_prev_max: local M (max of logits)
        //   - cb_prev_sum: local L (sum of exp)
        // Only receive from children that actually have data
        if (num_active_children > 0) {
            // Iterate through each round and receive from child if one exists AND has data
            for (uint32_t round = 0; round < num_active_rounds; ++round) {
                uint32_t child_id = active_children_per_round[round];
                if (child_id != UINT32_MAX) {
                    // Writer kernel handles the semaphore wait and data transfer to cb_m_in, cb_l_in, cb_out_o
                    // Data arrives in order: l, m, o

                    // Combine child with existing local/accumulated data
                    // Move child's L to cb_prev_sum_2 for correction
                    move_block<true>(cb_l_in, cb_prev_sum_2, Sq_chunk_t);
                    // Fused Softmax Correction
                    // * Fused Correction is a fused operation that performs the following steps:
                    // * 1. CUR_MAX = max(PREV_MAX, WORKER_MAX)
                    // * 2. EXP_MAX_DIFF_2 = exp((WORKER_MAX - CUR_MAX)*scale)
                    // * 3. PREV_SUM_2 *= EXP_MAX_DIFF_2
                    // * 4. EXP_MAX_DIFF = exp((PREV_MAX - CUR_MAX)*scale)
                    // * 5. PREV_SUM *= EXP_MAX_DIFF
                    // * 6. CUR_SUM = PREV_SUM_2 + PREV_SUM
                    correction_block<scale_fp32, vector_mode>(
                        cb_m_in,        // cb child max
                        cb_prev_sum_2,  // cb child sum
                        cb_cur_max,
                        cb_prev_max,
                        cb_cur_sum,
                        cb_prev_sum,
                        cb_exp_max_diff,
                        cb_exp_max_diff_2,
                        Sq_chunk_t);

                    // OUT_ACC_2 <- CHILD_OUT
                    move_block<true>(cb_out_o, cb_out_accumulate_im_2, out_chunk_tiles);

                    // OUT_ACC *= EXP_MAX_DIFF (scale local accumulator)
                    // OUT_ACC_2 *= EXP_MAX_DIFF_2 (scale child's accumulator)
                    mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
                    mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im_2, cb_exp_max_diff_2);

                    // OUT_ACC = OUT_ACC + OUT_ACC_2
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_accumulate_im_2, out_chunk_tiles);

                    // Update prev buffers for next round
                    // PREV_MAX <- CUR_MAX
                    // PREV_SUM <- CUR_SUM
                    CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
                    CircularBuffer(cb_m_in).pop_front(Sq_chunk_t);
                    move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
                    move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                }
            }
        }

        // Finalize output based on tree role
        if (is_tree_root) {
            // Root node: perform final normalization and output
            // Determine which sum/max buffer to use based on whether we did tree reduction
            // If we had children with data, results are in cb_prev_sum/cb_prev_max after tree reduction
            // If single core (no children with data), results are in cb_cur_sum/cb_cur_max from FA loop

            /* SUM = 1.0 / SUM */
            reconfig_data_format(cb_prev_sum, cb_prev_sum);
            pack_reconfig_data_format(cb_prev_sum);

            // Handle attention sink here
            if constexpr (use_attention_sink) {
                // Use appropriate max buffer based on tree reduction
                uint32_t max_cb_for_sink = cb_prev_max;

                // m_new
                max_block<vector_mode>(cb_attention_sink, max_cb_for_sink, cb_cur_max, Sq_chunk_t);

                // exp(m - m_new)
                sub_exp_block<scale_fp32>(max_cb_for_sink, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);

                // l -> l * exp(m - m_new)
                mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                // exp(sink - m_new)
                sub_exp_block<scale_fp32>(cb_attention_sink, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
                CircularBuffer(cb_cur_max).pop_front(Sq_chunk_t);

                // l -> l + exp(sink - m_new)
                add_block_inplace<true>(cb_prev_sum, cb_exp_max_diff_2, Sq_chunk_t);

                // O -> O * exp(m - m_new)
                mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
            }

            reconfig_data_format(cb_prev_sum, cb_prev_sum);
            pack_reconfig_data_format(cb_prev_sum);
            recip_block_inplace(cb_prev_sum, Sq_chunk_t);

            /* OUT_ACC *= 1/SUM */
            reconfig_data_format(cb_out_accumulate_im, cb_prev_sum);
            pack_reconfig_data_format(cb_out_accumulate_im);

            // cb_prev_sum is consumed and popped by mul_block_bcast_cols_inplace
            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_prev_sum);
            pack_reconfig_data_format(cb_out_final);

            // Pop the max buffer that still has data
            CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);

            // Untilize output to ROW MAJOR if input Q was also ROW MAJOR
            if constexpr (untilize_output) {
                // Unified untilize - auto-dispatches based on out_chunk_tiles vs DEST limit
                compute_kernel_lib::untilize<
                    out_chunk_tiles,
                    cb_out_accumulate_im,
                    cb_out_final,
                    compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                    compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                    compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
            } else {
                // Move output to buffer for the writer
                move_block<true>(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);
            }

        } else if (has_parent) {
            // Non-root node with parent: send intermediate results
            // We have data (checked at function start), so send it
            // After tree reduction (if any), results are in:
            //   - cb_out_accumulate_im: O
            //   - cb_prev_sum: L
            //   - cb_prev_max: M
            // Move O to output CB
            move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
            // Move M to output CB
            move_block<true>(cb_prev_max, cb_out_m, Sq_chunk_t);
            // Move L to output CB
            move_block<true>(cb_prev_sum, cb_out_l, Sq_chunk_t);
        }
    }

    // Free up cb_q_in after Q chunks
    CircularBuffer(cb_q_in).pop_front(q_chunk_tiles);
#endif
}
