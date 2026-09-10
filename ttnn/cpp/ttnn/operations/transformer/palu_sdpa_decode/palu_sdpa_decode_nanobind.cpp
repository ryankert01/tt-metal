// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "palu_sdpa_decode_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include "palu_sdpa_decode.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::transformer {

void bind_palu_sdpa_decode(nb::module_& mod) {
    const auto* const doc =
        R"doc(
        Palu fused-K flash decode (stage 2 of the Palu-on-TT study). Task 1 state: a verbatim clone of
        flash_multi_latent_attention_decode; the K-reconstruction inputs are added in Task 3.
        )doc";

    ttnn::bind_function<"palu_sdpa_decode", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::palu_sdpa_decode,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("input_tensor_v") = nb::none(),
        nb::arg("head_dim_v").noconvert(),
        nb::kw_only(),
        nb::arg("is_causal").noconvert() = true,
        nb::arg("attn_mask") = nb::none(),
        nb::arg("cur_pos") = nb::cast(std::vector<uint32_t>()),
        nb::arg("cur_pos_tensor") = nb::none(),
        nb::arg("attention_sink") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("sliding_window_size") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
}  // namespace ttnn::operations::transformer
