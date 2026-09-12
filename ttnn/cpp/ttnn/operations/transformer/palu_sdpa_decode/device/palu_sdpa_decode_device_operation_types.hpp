// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct PaluSdpaDecodeParams {
    bool is_causal = false;
    bool paged_attention = false;
    std::vector<uint32_t> cur_pos;
    std::optional<float> scale = std::nullopt;
    std::optional<uint32_t> sliding_window_size = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config = std::nullopt;
    DeviceComputeKernelConfig compute_kernel_config;
    uint32_t k_chunk_size = 0;
    // Share cache is only meaningful for some unpaged configurations; default is false.
    std::optional<bool> share_cache = std::nullopt;
    // When true, enables multi-latent attention (MLA) path where V is derived from K.
    std::optional<bool> use_mla = std::nullopt;
    std::optional<uint32_t> head_dim_v = std::nullopt;
    // Paired paged-cache geometry overrides (shared with chunked prefill SDPA). See
    // ttnn::operations::transformer::PagedCacheGeometryOverride.
    ttnn::operations::transformer::PagedCacheGeometryOverride paged_cache_geometry{};
    // Optional circular-buffer capacity (in tokens) for the K/V cache view. When set,
    // the kernel computes ``virtual_pos %= cache_position_modulo`` before resolving the
    // page_table entry for every tile read, so a bounded sliding-window cache of
    // capacity N can be addressed by absolute positions ≥ N. Required for layers using
    // vLLM's SlidingWindowSpec, which sizes the per-layer page_table to
    // sliding_window/block_size entries and zero-pads the rest. Must be a multiple of
    // the effective block_size and ≥ sliding_window_size when both are set. Paged-mode
    // only (validated in validate_on_program_cache_miss).
    std::optional<uint32_t> cache_position_modulo = std::nullopt;
    // Palu: chunk-relative rotation with the query advanced on core (Phase 2). Set iff rot_step is given.
    bool cs_recurrence = false;
    // Palu G-LRD: KV heads per shared latent (k_latent/v_latent have n_kv / heads_per_group heads).
    uint32_t heads_per_group = 1;
    // Palu: V rebuilt on core per head from the V latent and u_v (head_dim_v is then 128).
    bool recon_v = false;
};

struct PaluSdpaDecodeInputs {
    // Mandatory tensors
    Tensor q;
    Tensor k;

    // Optional V tensor; when MLA is enabled, V is derived from K and this may be nullopt.
    std::optional<Tensor> v;

    // Optional auxiliary tensors
    std::optional<Tensor> cur_pos_tensor;
    std::optional<Tensor> page_table_tensor;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> attention_sink;

    // Palu fused-K inputs (spec §3)
    Tensor w_rot;                      // [1, n_kv, r_k, 2*hd] bf16: [u_k | u_k T]
    Tensor cs_tab;                     // [1, 1, S_cs >= S, 2*hd] bf16: [cos | sin] per position
    std::optional<Tensor> rot_step;    // [1, 1, 32, 2*hd] fp32: [cos(c f) | sin(c f)] (Phase 2)
    std::optional<Tensor> trans_mat;   // [1, 1, 32, 32] bf16 pair-rotation tile (Phase 2)
    std::optional<Tensor> u_v;         // [1, n_kv, r_v_g, 128] bf16 (in-kernel V)
};

}  // namespace ttnn::prim
