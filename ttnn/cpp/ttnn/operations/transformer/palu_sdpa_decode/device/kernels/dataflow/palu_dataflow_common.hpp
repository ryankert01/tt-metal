// SPDX-License-Identifier: Apache-2.0
// Dataflow helpers for palu_sdpa_decode (Palu fused-K flash decode).
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// n consecutive DRAM tiles [first_tile, first_tile + n) -> the CB write pointer wp, which is advanced
// and returned. Caller owns the reserve_back/push_back around it (a push must stay one block).
template <uint32_t cb_id, uint32_t barrier_threshold, typename ReaderT>
uint32_t palu_read_tiles_at(uint32_t wp, uint32_t first_tile, uint32_t n, const ReaderT& reader) {
    Noc noc;
    constexpr uint32_t tile_bytes = get_tile_size(cb_id);
    uint32_t barrier_count = 0;
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(reader, CoreLocalMem<uint32_t>(wp), tile_bytes, {.page_id = first_tile + i}, {});
        if (++barrier_count == barrier_threshold) {
            noc.async_read_barrier();
            barrier_count = 0;
        }
        wp += tile_bytes;
    }
    noc.async_read_barrier();
    return wp;
}

// n consecutive DRAM tiles [first_tile, first_tile + n) -> one push of n tiles into cb_id.
template <uint32_t cb_id, uint32_t barrier_threshold, typename ReaderT>
void palu_read_tiles(uint32_t first_tile, uint32_t n, const ReaderT& reader) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(n);
    palu_read_tiles_at<cb_id, barrier_threshold>(cb.get_write_ptr(), first_tile, n, reader);
    cb.push_back(n);
}

// Per-head resident weight block: [w_rot (Rt x Wt) | u_v (Rvt x DHt, in-kernel V) | trans_mat (1, Phase 2)],
// ONE push per head (the block is the CB's granule; a push writes linearly and cannot wrap mid-block).
// w_rot is [1, n_kv, r_k, 2*hd] TILE and u_v [1, n_kv, r_v, hd] TILE, both head-major and row-major within
// a head. The rotation weight and u_v are per KV head even when the latents are shared by a group.
template <uint32_t cb_w, uint32_t Rt, uint32_t Wt, uint32_t Rvt, uint32_t DHt, uint32_t barrier_threshold, bool phase2,
          bool recon_v, typename ReaderT, typename UvReaderT, typename TReaderT>
void read_w_rot(uint32_t head, const ReaderT& reader, const UvReaderT& u_v_reader, const TReaderT& trans_mat_reader) {
    constexpr uint32_t block = Rt * Wt + (recon_v ? Rvt * DHt : 0) + (phase2 ? 1 : 0);
    CircularBuffer cb(cb_w);
    cb.reserve_back(block);
    uint32_t wp = palu_read_tiles_at<cb_w, barrier_threshold>(cb.get_write_ptr(), head * Rt * Wt, Rt * Wt, reader);
    if constexpr (recon_v) {
        wp = palu_read_tiles_at<cb_w, barrier_threshold>(wp, head * Rvt * DHt, Rvt * DHt, u_v_reader);
    }
    if constexpr (phase2) {
        palu_read_tiles_at<cb_w, barrier_threshold>(wp, 0, 1, trans_mat_reader);
    }
    cb.push_back(block);
}

// Per K chunk, in this order: the latent rows [Sk_chunk_t x Rt] (contiguous in DRAM), the absolute
// [cos | sin] rows for the same positions [Sk_chunk_t x Wt] (Phase 1 only; Phase 2 uses a resident
// chunk-relative table read once by the caller), then V [Sk_chunk_t x v_width_t]. v_width_t is the V
// row width actually stored per position in DRAM: the V head dim (DHt) on the absorbed-V path, or the
// V latent's own width (Rvt) when V is rebuilt on core from u_v (G-LRD groups the latent, not DHt).
template <
    uint32_t Rt,
    uint32_t v_width_t,
    uint32_t Wt,
    uint32_t barrier_threshold,
    uint32_t cb_lat,
    uint32_t cb_v,
    uint32_t cb_cs,
    bool phase2,
    typename LatReaderT,
    typename VReaderT,
    typename CsReaderT>
void read_palu_chunks(
    uint32_t k_chunk_start,
    uint32_t k_chunk_end,
    uint32_t lat_start_tile_id,
    uint32_t v_start_tile_id,
    uint32_t Sk_chunk_t,
    const LatReaderT& lat_reader,
    const VReaderT& v_reader,
    const CsReaderT& cs_reader) {
    const uint32_t lat_chunk_tiles = Sk_chunk_t * Rt;
    const uint32_t v_chunk_tiles = Sk_chunk_t * v_width_t;
    const uint32_t cs_chunk_tiles = Sk_chunk_t * Wt;
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        palu_read_tiles<cb_lat, barrier_threshold>(lat_start_tile_id, lat_chunk_tiles, lat_reader);
        lat_start_tile_id += lat_chunk_tiles;
        if constexpr (!phase2) {
            palu_read_tiles<cb_cs, barrier_threshold>(k_chunk * cs_chunk_tiles, cs_chunk_tiles, cs_reader);
        }
        palu_read_tiles<cb_v, barrier_threshold>(v_start_tile_id, v_chunk_tiles, v_reader);
        v_start_tile_id += v_chunk_tiles;
    }
}

// Phase 2 angle table and rot_step, multicast down a grid column (stage 2e). The column's leader reads
// both from DRAM into its own CB region and writes them to the same L1 addresses on the num_dests
// cores in the vertical range (x, y0..y1), then bumps their multicast semaphore; the receivers wait
// for it. Both CBs are empty on every core at this point, so every write pointer is the CB base and
// the addresses coincide. A core whose column has no other active core (num_dests == 0) reads for
// itself. Pieces of at most 32 KB per multicast call keep each NoC transaction well inside the
// hardware's burst limit.
template <uint32_t cb_cs, uint32_t cb_rs, uint32_t barrier_threshold, typename CsReaderT, typename RsReaderT>
void palu_read_table_mcast(
    uint32_t cs_tiles, uint32_t rs_tiles, const CsReaderT& cs_reader, const RsReaderT& rs_reader,
    bool leader, uint32_t num_dests, uint32_t mcast_x, uint32_t mcast_y0, uint32_t mcast_y1, uint32_t sem_id) {
    Noc noc;
    CircularBuffer cs(cb_cs), rs(cb_rs);
    cs.reserve_back(cs_tiles);
    rs.reserve_back(rs_tiles);
    const uint32_t cs_wp = cs.get_write_ptr();
    const uint32_t rs_wp = rs.get_write_ptr();
    if (leader || num_dests == 0) {
        palu_read_tiles_at<cb_cs, barrier_threshold>(cs_wp, 0, cs_tiles, cs_reader);
        palu_read_tiles_at<cb_rs, barrier_threshold>(rs_wp, 0, rs_tiles, rs_reader);
        if (num_dests > 0) {
            auto mcast = [&](uint32_t addr, uint32_t bytes) {
                constexpr uint32_t piece = 32 * 1024;
                for (uint32_t off = 0; off < bytes; off += piece) {
                    const uint32_t n = (bytes - off < piece) ? (bytes - off) : piece;
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(addr + off),
                        MulticastEndpoint{},
                        n,
                        num_dests,
                        {},
                        {.noc_x_start = mcast_x, .noc_y_start = mcast_y0, .noc_x_end = mcast_x, .noc_y_end = mcast_y1,
                         .addr = addr + off},
                        false);
                }
            };
            mcast(cs_wp, cs_tiles * get_tile_size(cb_cs));
            mcast(rs_wp, rs_tiles * get_tile_size(cb_rs));
            noc.async_write_barrier();
            constexpr uint32_t VALID = 1;
            Semaphore<> sem(sem_id);
            sem.set(VALID);
            sem.set_multicast(noc, mcast_x, mcast_y0, mcast_x, mcast_y1, num_dests);
            noc.async_write_barrier();
        }
    } else {
        Semaphore<> sem(sem_id);
        sem.wait(1);
        sem.set(0);
        noc.async_atomic_barrier();
    }
    cs.push_back(cs_tiles);
    rs.push_back(rs_tiles);
}
