// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "palu_sdpa_decode.hpp"

#include <cstdint>
#include <optional>
#include <utility>

#include <tt-logger/tt-logger.hpp>

#include "device/palu_sdpa_decode_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
using namespace tt::tt_metal;

namespace {

inline uint32_t palu_get_chunk_size(uint32_t s) {
    /*
    # find maximum power of 2 divisor of s
    for i in range(1, s):
        if s % (2**(i+1)) != 0:
            break
    */
    uint32_t i = 1;
    for (; i < s; i++) {
        if (s % (1 << (i + 1)) != 0) {
            break;
        }
    }
    return std::min(512, 1 << i);
}
}  // namespace

namespace ttnn::transformer {

ttnn::Tensor palu_sdpa_decode(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& k_latent,
    const ttnn::Tensor& w_rot,
    const ttnn::Tensor& cs_tab,
    const ttnn::Tensor& v_latent,
    const uint32_t head_dim_v,
    const ttnn::Tensor& cur_pos_tensor,
    const std::optional<const Tensor>& rot_step,
    const std::optional<const Tensor>& trans_mat,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    uint32_t s = k_latent.logical_shape()[-2];
    uint32_t k_chunk_size = palu_get_chunk_size(s);
    if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
        k_chunk_size = program_config.value().k_chunk_size;
        // assert chunk size must be power of 2 and multiple of 32
        TT_FATAL(
            (k_chunk_size & (k_chunk_size - 1)) == 0,
            "User provided k_chunk_size must be power of 2, got: {}",
            k_chunk_size);
        TT_FATAL(k_chunk_size % 32 == 0, "User provided k_chunk_size must be multiple of 32, got: {}", k_chunk_size);
    } else {
        TT_FATAL(
            k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: {}",
            k_chunk_size);
    }

    TT_FATAL(
        v_latent.padded_shape()[-1] == head_dim_v,
        "Head dimension of the V tensor must be equal to head_dim_v parameter value, got {} and {}",
        v_latent.padded_shape()[-1],
        head_dim_v);
    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::palu_sdpa_decode(
        input_tensor_q, k_latent, w_rot, cs_tab, v_latent, cur_pos_tensor, rot_step, trans_mat, head_dim_v, scale,
        memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG), program_config, kernel_config_val, k_chunk_size);
}

}  // namespace ttnn::transformer
