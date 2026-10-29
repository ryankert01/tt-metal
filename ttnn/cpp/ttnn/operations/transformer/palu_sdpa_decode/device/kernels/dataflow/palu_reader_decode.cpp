// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include <vector>

#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "palu_dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    /*
    In DRAM, Q is (B, PNHt, DHt), K is (B, St, DHt), V is (B, St, DHt), mask is (B, PNHt, PSt)
    We want to read for a particular batch cur_batch, and sequence length up to padded layer length.
    We read Q: (cur_batch, PNHt, DHt), K: (cur_batch, PSt, DHt), V: (cur_batch, PSt, DHt), mask: (cur_batch, PNHt, PSt)
    */
    constexpr uint32_t B = get_compile_time_arg_val(0);           // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);        // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);          // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);         // head dim
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);        // head dim of V
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr bool is_q_sharded = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(8);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(9);
    constexpr uint32_t index_stick_size_B = get_compile_time_arg_val(10);
    constexpr bool is_paged_attention = get_compile_time_arg_val(11) == 1;
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(12);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(13);
    constexpr uint32_t Bkv = get_compile_time_arg_val(14);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(15);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(16);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(17);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(18);
    constexpr bool is_causal = get_compile_time_arg_val(19) == 1;
    constexpr bool use_attention_mask = get_compile_time_arg_val(20) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(22);
    constexpr bool tilize_q = get_compile_time_arg_val(23) == 1;
    constexpr bool reuse_k = get_compile_time_arg_val(24) == 1;
    constexpr bool use_half_tile = get_compile_time_arg_val(25);
    constexpr uint32_t q_chunk_size_bytes = get_compile_time_arg_val(26);
    constexpr bool is_cur_pos_tensor_sharded = get_compile_time_arg_val(27);
    constexpr bool is_page_table_sharded = get_compile_time_arg_val(28);
    constexpr uint32_t q_page_size_bytes = get_compile_time_arg_val(29);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(30);
    constexpr uint32_t original_block_size = get_compile_time_arg_val(31);
    constexpr bool has_block_padding = is_paged_attention && original_block_size > 0 && original_block_size < 32;
    constexpr uint32_t k_mcast_semaphore_id = get_compile_time_arg_val(32);
    constexpr bool q_locally_available = get_compile_time_arg_val(33) == 1;
    constexpr bool use_k_mcast = get_compile_time_arg_val(34) == 1;
    constexpr uint32_t Bmask = get_compile_time_arg_val(35);
    // 0 = unbounded cache (legacy); nonzero = wrap virtual tile index mod this value
    // before page_table lookup. Value is in TILE rows (= cache_position_modulo /
    // TILE_HEIGHT). Validated to be a multiple of block_size_t at op level.
    constexpr uint32_t capacity_t = get_compile_time_arg_val(36);
    // ---------------- Palu (stage 2) ----------------
    constexpr uint32_t Rt = get_compile_time_arg_val(37);       // latent width in tiles
    constexpr bool phase2 = get_compile_time_arg_val(38) == 1;  // chunk-relative rotation (Task 4)
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(39);  // KV heads sharing one latent (G-LRD)
    constexpr bool recon_v = get_compile_time_arg_val(40) == 1;         // V rebuilt on core from u_v
    constexpr uint32_t Rvt = get_compile_time_arg_val(41);              // V latent width in tiles
    constexpr uint32_t Wt = 2 * DHt;
    constexpr uint32_t n_lat_heads = num_kv_heads / heads_per_group;  // latent groups per batch (G-LRD: < num_kv_heads)

    constexpr auto q_args = TensorAccessorArgs<42>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto attention_sink_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    constexpr auto w_rot_args = TensorAccessorArgs<attention_sink_args.next_compile_time_args_offset()>();
    constexpr auto cs_args = TensorAccessorArgs<w_rot_args.next_compile_time_args_offset()>();
    constexpr auto rot_step_args = TensorAccessorArgs<cs_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<rot_step_args.next_compile_time_args_offset()>();
    constexpr auto u_v_args = TensorAccessorArgs<trans_mat_args.next_compile_time_args_offset()>();
    static_assert(u_v_args.next_compile_time_args_offset() > 0);  // keeps the arg chain checked
    // The Palu CBs below reuse slots whose stock owners must be compiled out (spec §4); the program
    // factory refuses to build otherwise, and these assert the kernel side of that contract.
    static_assert(!use_attention_sink, "Palu reuses the attention-sink CB (c_4) for w_rot");
    // Stage 3: paged latent caches. c_9 stays the latent chunk; the page-table row rides behind rot_step
    // in c_14, which exists in Phase 2 only (see palu_read_page_table_row).
    static_assert(!is_paged_attention || phase2, "paged palu_sdpa_decode needs Phase 2 (rot_step): the page table lives behind it in c_14");
    static_assert(!is_paged_attention || !is_page_table_sharded, "paged palu_sdpa_decode: interleaved page table only");
    static_assert(sliding_window_size == 0, "Palu reuses the sliding-window CB (c_13) for [cos | sin]");
    static_assert(!has_block_padding, "Palu reuses the block-padding CB (c_14) for rot_step");

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
    // #44366: cur_pos is consumed by both the writer (c_8) and compute (c_15).
    // Using one shared CB races — whichever consumer pops first drains the
    // count and the other hangs waiting for tiles. Each consumer gets its own CB.
    constexpr uint32_t cb_writer_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_compute_cur_pos = tt::CBIndex::c_15;
    // Palu CBs (spec §4); the stock owners of these slots are compiled out in the envelope.
    constexpr uint32_t cb_lat_in = tt::CBIndex::c_9;
    constexpr uint32_t cb_w_rot = tt::CBIndex::c_4;
    constexpr uint32_t cb_cs_in = tt::CBIndex::c_13;
    constexpr uint32_t cb_rot_step = tt::CBIndex::c_14;

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t attention_sink_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t w_rot_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cs_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rot_step_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t trans_mat_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t u_v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_page_size = get_arg_val<uint32_t>(arg_idx++);
    const bool is_worker = get_arg_val<uint32_t>(arg_idx++) == 0;
    const bool is_output_core = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);
    const bool do_k_mcast = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_y0 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_y1 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_dests = get_arg_val<uint32_t>(arg_idx++);

    // idle core
    if (q_addr == 0) {
        return;
    }

#ifdef PALU_TABLE_MCAST
    // ---------------- Phase 2 angle table, multicast per grid column (stage 2e) ----------------
    // The reader's multicast args (do_k_mcast, mcast_x/y0/y1, num_dests) describe this multicast; the
    // K multicast they were copied for never runs in the Palu envelope. This sits above every early
    // exit below -- a skipped batch (cur_pos == -1) and a sequence split that leaves a core no chunk
    // both return before the K/V loop, and a leader that took one of those exits would leave the
    // receivers in its column waiting on the semaphore forever. Idle cores (returned just above) are
    // not in any column's active set, so no leader ever multicasts to one.
    //
    // The table is chunk-relative, so its height is the pinned K chunk. A per-core dynamic chunk would
    // make the leader's and the receivers' tile counts disagree; the op always resolves a chunk, so
    // Sk_chunk_t is nonzero here.
    if constexpr (phase2) {
        static_assert(Sk_chunk_t > 0, "palu_sdpa_decode: table_mcast needs a pinned K chunk");
        constexpr uint32_t table_barrier_threshold =
            get_barrier_read_threshold<get_tile_size(cb_q_in), num_cores>();
        palu_read_table_mcast<cb_cs_in, cb_rot_step, table_barrier_threshold>(
            Sk_chunk_t * Wt,
            Wt,
            TensorAccessor(cs_args, cs_addr),
            TensorAccessor(rot_step_args, rot_step_addr),
            do_k_mcast,
            num_dests,
            mcast_x,
            mcast_y0,
            mcast_y1,
            k_mcast_semaphore_id);
    }
#endif

    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    if constexpr (is_causal) {
        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            // Reader fills cb_writer_cur_pos (c_8) first (from DRAM, or via the
            // aliased sharded buffer) then copies the same stick into
            // cb_compute_cur_pos (c_15) via an L1->L1 read.
            CircularBuffer cb_writer(cb_writer_cur_pos);
            cb_writer.reserve_back(1);
            uint32_t index_cb_wr_ptr = cb_writer.get_write_ptr();
            if constexpr (!is_cur_pos_tensor_sharded) {
                const auto addrg = TensorAccessor(pos_args, pos_addr);
                // index_tensor has one page to read
                noc.async_read(addrg, CoreLocalMem<uint32_t>(index_cb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
                noc.async_read_barrier();
            }
            CircularBuffer cb_compute(cb_compute_cur_pos);
            cb_compute.reserve_back(1);
            uint32_t index_cb_compute_wr_ptr = cb_compute.get_write_ptr();
            const uint8_t noc_id = noc.get_noc_id();
            const uint32_t my_noc_x = my_x[noc_id];
            const uint32_t my_noc_y = my_y[noc_id];
            UnicastEndpoint pos_src;
            noc.async_read(
                pos_src,
                CoreLocalMem<uint32_t>(index_cb_compute_wr_ptr),
                index_stick_size_B,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = index_cb_wr_ptr},
                {});
            noc.async_read_barrier();
            cb_writer.push_back(1);
            cb_compute.push_back(1);
            volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
            cur_pos = index_ptr[cur_batch / q_heads_parallel_factor];
        }
        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    // When block_size < TILE_HEIGHT, each tile has zero-padded rows. Convert cur_pos from
    // the original sequence space to the padded tile space so get_runtime_args computes the
    // correct number of tiles to process. Only needed for causal mode where cur_pos comes
    // from user input; non-causal uses cur_pos_base which is already in the padded space.
    if constexpr (has_block_padding && is_causal) {
        cur_pos = (cur_pos / original_block_size) * 32 + (cur_pos % original_block_size);
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Sequence length assignment
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            cur_batch,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);

    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    tt_l1_ptr uint32_t* all_output_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t* all_output_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
    uint32_t output_core_noc_y = all_output_noc_y[cur_batch];

    constexpr uint32_t q_chunk_tiles = PNHt * DHt;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t attention_sink_tile_bytes = get_tile_size(cb_attention_sink);
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    // Read Q entirely - always read into cb_q_in
    // When tilize_q is true, compute will tilize back to cb_q_in
    // When tilize_q is false, Q is already tilized
    const uint32_t q_batch_offset = cur_batch * q_chunk_tiles;

    // Read Q
    read_q<cb_q_in, cb_q_rm, q_tile_bytes, q_chunk_tiles, is_q_sharded, tilize_q, use_half_tile, barrier_threshold>(
        q_locally_available,
        is_output_core,
        q_addr,
        output_core_noc_x,
        output_core_noc_y,
        q_chunk_size_bytes,
        q_args,
        q_page_size_bytes,
        q_batch_offset);

    const auto k_reader = TensorAccessor(k_args, k_addr);

    const auto v_reader = TensorAccessor(v_args, v_addr);

    const auto w_rot_reader = TensorAccessor(w_rot_args, w_rot_addr);
    const auto cs_reader = TensorAccessor(cs_args, cs_addr);
    const auto trans_mat_reader = TensorAccessor(trans_mat_args, trans_mat_addr);
    const auto u_v_reader = TensorAccessor(u_v_args, u_v_addr);

#ifndef PALU_TABLE_MCAST
    if constexpr (phase2) {
        // Chunk-relative [cos | sin] rows t = 0..c-1: the first Sk_chunk_t row tiles of cs_tab, and the
        // per-chunk rotation constants. Both are read once per core and stay resident for the run.
        // (stage 2e: with PALU_TABLE_MCAST one core per grid column does this read instead, near the
        // top of the kernel, and multicasts the result down its column.)
        palu_read_tiles<cb_cs_in, barrier_threshold>(0, Sk_chunk_t_dynamic * Wt, cs_reader);
        const auto rot_step_reader = TensorAccessor(rot_step_args, rot_step_addr);
        palu_read_tiles<cb_rot_step, barrier_threshold>(0, Wt, rot_step_reader);
    }
#endif

    // Read attention sink
    if constexpr (use_attention_sink) {
        const auto attention_sink_reader = TensorAccessor(attention_sink_args, attention_sink_addr);

        CircularBuffer cb_sink(cb_attention_sink);
        cb_sink.reserve_back(PNHt);
        uint32_t attention_sink_write_ptr = cb_sink.get_write_ptr();

        for (uint32_t tile = 0; tile < PNHt; ++tile) {
            // Use noc.async_read with explicit size instead of noc.async_read_page because
            // the CB may use half tiles (16x32) while the DRAM buffer stores full tiles (32x32).
            // noc.async_read_page would read buffer->aligned_page_size() bytes, overflowing the CB.
            noc.async_read(
                attention_sink_reader,
                CoreLocalMem<uint32_t>(attention_sink_write_ptr),
                attention_sink_tile_bytes,
                {.page_id = tile},
                {});
            attention_sink_write_ptr += attention_sink_tile_bytes;
        }
        noc.async_read_barrier();
        cb_sink.push_back(PNHt);
    }

    // Stage 3: this batch's page-table row, resident behind rot_step in c_14 (rot_step has been pushed
    // by now on both the multicast and the direct path above).
    volatile tt_l1_ptr uint32_t* page_table_ptr = nullptr;
    if constexpr (is_paged_attention) {
        constexpr uint32_t rs_tile_bytes = get_tile_size(cb_rot_step);
        const uint32_t pages = (page_table_page_size + rs_tile_bytes - 1) / rs_tile_bytes;
        page_table_ptr = palu_read_page_table_row<cb_rot_step>(
            pages, cur_batch / q_heads_parallel_factor, page_table_args, page_table_addr, page_table_page_size);
    }

#ifdef PALU_GROUP_SHARED
    // ---------------- Group-shared mode (stage 2c) ----------------
    // The core's heads are contiguous inside one latent group (the host guarantees heads_per_core
    // divides heads_per_group), so lat_head = cur_head / heads_per_group is the same for all of them
    // and every latent chunk is read ONCE for the whole group instead of once per head. Q is per
    // (batch, core) and was already read above. The weight blocks are pushed head by head and stay
    // resident in c_4 for the run; compute indexes block h from the CB base.
    {
        const uint32_t head_base = cur_head_group * num_heads_per_core;
        const uint32_t lat_head = head_base / heads_per_group;  // the group's latent
        ASSERT((head_base + num_heads_per_core - 1) / heads_per_group == lat_head);
        for (uint32_t h = 0; h < num_heads_per_core; ++h) {
            read_w_rot<cb_w_rot, Rt, Wt, Rvt, DHt, barrier_threshold, phase2, recon_v>(
                head_base + h, w_rot_reader, u_v_reader, trans_mat_reader);
        }
        if constexpr (is_paged_attention) {
            read_palu_chunks_paged<Rt, Rvt, barrier_threshold, cb_lat_in, cb_v_in, n_lat_heads, block_size_t>(
                k_chunk_start, k_chunk_end, lat_head, Sk_chunk_t_dynamic, page_table_ptr, k_reader, v_reader);
        } else {
            const uint32_t lat_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * n_lat_heads * St * Rt;
            const uint32_t lat_start_tile_id =
                lat_batch_offset + lat_head * St * Rt + k_chunk_start * Sk_chunk_t_dynamic * Rt;
            const uint32_t v_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * n_lat_heads * St * Rvt;
            const uint32_t v_start_tile_id =
                v_batch_offset + lat_head * St * Rvt + k_chunk_start * Sk_chunk_t_dynamic * Rvt;
            read_palu_chunks<Rt, Rvt, Wt, barrier_threshold, cb_lat_in, cb_v_in, cb_cs_in, phase2>(
                k_chunk_start, k_chunk_end, lat_start_tile_id, v_start_tile_id, Sk_chunk_t_dynamic,
                k_reader, v_reader, cs_reader);
        }
    }
#else
    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        // ---------------- Palu: latent chunks + rotation tables instead of K chunks ----------------
        // G-LRD: this head's latents live in its group's slice of the cache; the weights stay per head.
        const uint32_t lat_head = cur_head / heads_per_group;                 // the group's latent
        read_w_rot<cb_w_rot, Rt, Wt, Rvt, DHt, barrier_threshold, phase2, recon_v>(
            cur_head, w_rot_reader, u_v_reader, trans_mat_reader);
        if constexpr (is_paged_attention) {
            // Stage 3: the same chunks, each tile row found through the page table.
            read_palu_chunks_paged<Rt, Rvt, barrier_threshold, cb_lat_in, cb_v_in, n_lat_heads, block_size_t>(
                k_chunk_start, k_chunk_end, lat_head, Sk_chunk_t_dynamic, page_table_ptr, k_reader, v_reader);
        } else {
            const uint32_t lat_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * n_lat_heads * St * Rt;
            const uint32_t lat_head_offset = lat_head * St * Rt;
            const uint32_t lat_start_tile_id = lat_batch_offset + lat_head_offset + k_chunk_start * Sk_chunk_t_dynamic * Rt;
            const uint32_t v_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * n_lat_heads * St * Rvt;
            const uint32_t v_head_offset = lat_head * St * Rvt;
            const uint32_t v_start_tile_id = v_batch_offset + v_head_offset + k_chunk_start * Sk_chunk_t_dynamic * Rvt;
            // read_palu_chunks's v_width_t is the V row width in DRAM: Rvt (the V latent's own width), not
            // vDHt (the compile-time V head dim of the stock op).
            read_palu_chunks<Rt, Rvt, Wt, barrier_threshold, cb_lat_in, cb_v_in, cb_cs_in, phase2>(
                k_chunk_start, k_chunk_end, lat_start_tile_id, v_start_tile_id, Sk_chunk_t_dynamic,
                k_reader, v_reader, cs_reader);
        }
    }
#endif
}
