// SPDX-License-Identifier: Apache-2.0
// Dataflow helpers for palu_sdpa_decode (Palu fused-K flash decode).
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

// n consecutive DRAM tiles [first_tile, first_tile + n) -> one push of n tiles into cb_id.
template <uint32_t cb_id, uint32_t barrier_threshold, typename ReaderT>
void palu_read_tiles(uint32_t first_tile, uint32_t n, const ReaderT& reader) {
    Noc noc;
    CircularBuffer cb(cb_id);
    constexpr uint32_t tile_bytes = get_tile_size(cb_id);
    cb.reserve_back(n);
    uint32_t wp = cb.get_write_ptr();
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
    cb.push_back(n);
}

// w_rot for one KV head: Rt x Wt tiles, row-major in DRAM ([1, n_kv, r_k, 2*hd] TILE), pushed once per head.
template <uint32_t cb_w, uint32_t Rt, uint32_t Wt, uint32_t barrier_threshold, typename ReaderT>
void read_w_rot(uint32_t head, const ReaderT& reader) {
    palu_read_tiles<cb_w, barrier_threshold>(head * Rt * Wt, Rt * Wt, reader);
}

// Per K chunk, in this order: the latent rows [Sk_chunk_t x Rt] (contiguous in DRAM), the absolute
// [cos | sin] rows for the same positions [Sk_chunk_t x Wt] (Phase 1 only; Phase 2 uses a resident
// chunk-relative table read once by the caller), then V [Sk_chunk_t x vDHt] exactly as stock.
template <
    uint32_t Rt,
    uint32_t vDHt,
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
    const uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
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
