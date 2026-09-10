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
        Palu fused-K flash decode (stage 2 of the Palu-on-TT study). K is never materialised: every
        K chunk is rebuilt on core from the low-rank latent cache as RoPE(k_latent @ u_k) and then
        fed to the stock flash-decode math. See
        docs/superpowers/specs/2026-09-10-palu-tt-stage2-design.md.

        Tensors (spec section 3):

        +---------------------+--------------------------------+---------+------------------------------------------+
        | Tensor              | Shape                          | dtype   | Notes                                    |
        +=====================+================================+=========+==========================================+
        | input_tensor_q      | [1, B, n_q_pad, 128]           | bf16    | already RoPE'd, TILE layout              |
        | k_latent            | [B, n_kv, max_seq_len, r_k]    | bf16    | r_k in {32, 64, 96}                      |
        | w_rot               | [1, n_kv, r_k, 256]            | bf16    | [u_k | u_k * T] per KV head              |
        | cs_tab              | [1, 1, >= max_seq_len, 256]    | bf16    | [cos | sin] per position, interleaved    |
        | v_latent            | [B, n_kv, max_seq_len, r_v]    | bf16    | as for flash_multi_latent_attention_decode|
        | cur_pos_tensor      | [B]                            | int32   | current position per batch row           |
        | rot_step (optional) | [1, 1, 32, 256]                | fp32    | [cos(c f) | sin(c f)]; selects Phase 2   |
        | trans_mat (optional)| [1, 1, 32, 32]                 | bf16    | required with rot_step (Phase 2)         |
        +---------------------+--------------------------------+---------+------------------------------------------+

        Envelope: head_dim 128, batch <= 32, causal, unpaged, no mask / sink / sliding window.
        Returns [1, B, n_q, head_dim_v].
        )doc";

    ttnn::bind_function<"palu_sdpa_decode", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::palu_sdpa_decode,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("k_latent").noconvert(),
        nb::arg("w_rot").noconvert(),
        nb::arg("cs_tab").noconvert(),
        nb::arg("v_latent").noconvert(),
        nb::arg("head_dim_v").noconvert(),
        nb::arg("cur_pos_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("rot_step") = nb::none(),
        nb::arg("trans_mat") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
}  // namespace ttnn::operations::transformer
