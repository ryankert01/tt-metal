// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

// Palu fused-K flash decode (stage 2). K is never materialised: each K chunk is rebuilt on core
// from the latent cache as RoPE(lat @ u_k). See docs/superpowers/specs/2026-09-10-palu-tt-stage2-design.md.
ttnn::Tensor palu_sdpa_decode(
    const ttnn::Tensor& input_tensor_q,          // [1, B, nh, 128] bf16 TILE, already rotated
    const ttnn::Tensor& k_latent,                // [B, n_groups, S, r_k]  (r_k a multiple of 32 in 32..512)
    const ttnn::Tensor& w_rot,                   // [1, n_kv, r_k, 256] bf16: [u_k | u_k T], n_kv = heads_per_group * n_groups
    const ttnn::Tensor& cs_tab,                  // [1, 1, >=S, 256] bf16: [cos | sin]
    const ttnn::Tensor& v_latent,                // [B, n_groups, S, r_v]
    uint32_t head_dim_v,
    const ttnn::Tensor& cur_pos_tensor,          // [B] int32
    const std::optional<const Tensor>& rot_step = std::nullopt,   // [1, 1, 32, 256] fp32 -> Phase 2
    const std::optional<const Tensor>& trans_mat = std::nullopt,  // [1, 1, 32, 32] bf16 (Phase 2)
    const std::optional<const Tensor>& u_v = std::nullopt,        // [1, n_kv, r_v_g, 128] bf16 (optional) -> in-kernel V reconstruction
    std::optional<uint32_t> heads_per_core = std::nullopt,        // KV heads per core (None = the stock work split)
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
