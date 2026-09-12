// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "palu_sdpa_decode_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <cstdint>

#include <tt-metalium/constants.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt_stl/reflection.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {
void PaluSdpaDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    bool use_mla = operation_attributes.use_mla.value_or(false);

    if (use_mla) {
        TT_FATAL(
            operation_attributes.head_dim_v.has_value(), "Must provide head_dim_v for multi-latent attention decode");
        TT_FATAL(operation_attributes.is_causal, "Multi-latent attention decode only tested for causal!");
    } else {
        TT_FATAL(tensor_args.v.has_value(), "Must have 3 input tensors and mask");
    }

    std::vector<Tensor> input_tensors{{tensor_args.q}, {tensor_args.k}};
    if (tensor_args.v.has_value()) {
        input_tensors.emplace_back(tensor_args.v.value());
    }
    for (const auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device!");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device!");
        TT_FATAL(
            input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B ||
                input_tensor.dtype() == DataType::BFLOAT4_B,
            "Unsupported data type {}.",
            input_tensor.dtype());
    }

    for (size_t i = 1; i < input_tensors.size(); i++) {
        TT_FATAL(input_tensors.at(i).layout() == Layout::TILE, "Inputs to SDPA must be tilized except for Q");
    }

    const auto q_shape = input_tensors.at(0).padded_shape();
    const auto q_shape_unpadded = input_tensors.at(0).logical_shape();
    const auto k_shape = input_tensors.at(1).padded_shape();
    const auto v_shape = input_tensors.size() > 2 ? input_tensors.at(2).padded_shape() : k_shape;

    // When using multi-latent attention, the V tensor is the same as K tensor, but of smaller head dimension.
    // If V tensor is not provided, we assume that V is subset of K in the hidden dimension. Hence we use K shape for V.
    // If V tensor is provided, we take the V tensor and use that to validate head_dim_v. (hidden dim of V must be equal
    // to head_dim_v)

    if (use_mla) {
        // Head dim v validation
        if (tensor_args.v.has_value()) {
            if (operation_attributes.recon_v) {
                // Palu in-kernel V: c_2 holds the V latent, whose width is u_v's row count; the output
                // width head_dim_v (128) is u_v's column count instead.
                TT_FATAL(
                    v_shape[3] == tensor_args.u_v->padded_shape()[2],
                    "palu_sdpa_decode: V latent width ({}) must equal u_v rows ({})",
                    v_shape[3],
                    tensor_args.u_v->padded_shape()[2]);
            } else {
                TT_FATAL(
                    v_shape[3] == operation_attributes.head_dim_v.value(),
                    "Head dimension of V must be equal to head_dim_v, got {} and {}",
                    v_shape[3],
                    operation_attributes.head_dim_v.value());
            }
        }
        // Palu: K in DRAM is the latent [B, nkv, S, r_k]; the head dim comes from Q, not from K.
        // Checked on the logical width: a 48-wide latent is tile-padded to 64 and would slip through.
        const auto r_k_logical = input_tensors.at(1).logical_shape()[-1];
        TT_FATAL(
            r_k_logical % 32 == 0 && 32 <= r_k_logical && r_k_logical <= 512 && k_shape[-1] == r_k_logical,
            "palu_sdpa_decode: r_k (latent width) must be a multiple of 32 in 32..512, got {}",
            r_k_logical);
        TT_FATAL(
            operation_attributes.head_dim_v.value() <= q_shape[3],
            "Head dimension of V must be less than or equal to head dim of Q, got {} and {}",
            operation_attributes.head_dim_v.value(),
            q_shape[3]);
    }

    // Input 0 must be sharded by height or DRAM interleaved. All other inputs must be in DRAM.
    const auto Q_memcfg = input_tensors.at(0).memory_config();
    if (input_tensors.at(0).is_sharded()) {
        TT_FATAL(
            Q_memcfg.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Q tensor memory layout must be HEIGHT_SHARDED when sharded but got {}",
            Q_memcfg.memory_layout());
    } else {
        TT_FATAL(
            Q_memcfg.buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Q tensor buffer type must be DRAM when not sharded but got {}",
            Q_memcfg.buffer_type());
    }

    for (std::size_t i = 1; i < input_tensors.size(); i++) {
        TT_FATAL(
            input_tensors.at(i).buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Input tensor {} buffer type must be DRAM but got {}",
            i,
            input_tensors.at(i).buffer()->buffer_type());
    }
    // Output memconfig must be height sharded or DRAM
    if (operation_attributes.output_mem_config.is_sharded()) {
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Output memory config layout must be HEIGHT_SHARDED when sharded but got {}",
            operation_attributes.output_mem_config.memory_layout());
    }

    if (!operation_attributes.is_causal) {
        if (tensor_args.attn_mask.has_value()) {
            // Causal attention verification
            const auto& mask_tensor = tensor_args.attn_mask.value();
            const auto mask_shape = mask_tensor.padded_shape();
            const auto mask_shape_unpadded = mask_tensor.logical_shape();

            TT_FATAL(
                mask_shape[2] == q_shape[2],
                "Expect same number of padded heads in mask as in Q, got {} and {}",
                mask_shape[2],
                q_shape[2]);
            TT_FATAL(
                mask_shape_unpadded[2] == q_shape_unpadded[2],
                "Expect same number of heads in mask as in Q, got {} and {}",
                mask_shape_unpadded[2],
                q_shape_unpadded[2]);
            if (!operation_attributes.paged_attention) {
                TT_FATAL(
                    mask_shape[3] == k_shape[2],
                    "Expect same sequence length in mask as in K, got {} and {}",
                    mask_shape[3],
                    k_shape[2]);
            }

            TT_FATAL(operation_attributes.k_chunk_size > 0, "Must provide k_chunk_size if using attention mask!");
            TT_FATAL(
                mask_shape[3] % operation_attributes.k_chunk_size == 0,
                "Mask sequence length must be multiple of chunk size, got: {} and {}",
                mask_shape[3],
                operation_attributes.k_chunk_size);

            TT_FATAL(
                mask_tensor.dtype() == DataType::BFLOAT16 || mask_tensor.dtype() == DataType::BFLOAT8_B ||
                    mask_tensor.dtype() == DataType::BFLOAT4_B,
                "Unsupported data type for mask tensor: {}.",
                mask_tensor.dtype());
        }
    } else {
        // Uncausal attention verification
        TT_FATAL(not tensor_args.attn_mask.has_value(), "Must not have attn_mask tensor for non-causal attention");
    }

    const auto& paged_geo = operation_attributes.paged_cache_geometry;
    if (paged_geo.active()) {
        TT_FATAL(
            tensor_args.page_table_tensor.has_value(),
            "PagedCacheGeometryOverride is only supported in paged mode (when page_table is provided)");
        TT_FATAL(paged_geo.block_size > 0, "PagedCacheGeometryOverride.block_size must be > 0");
        TT_FATAL(paged_geo.num_kv_heads > 0, "PagedCacheGeometryOverride.num_kv_heads must be > 0");
    }

    if (operation_attributes.cache_position_modulo.has_value()) {
        TT_FATAL(
            tensor_args.page_table_tensor.has_value(),
            "cache_position_modulo is only supported in paged mode (when page_table is provided)");
        const uint32_t modulo = operation_attributes.cache_position_modulo.value();
        TT_FATAL(modulo > 0, "cache_position_modulo must be > 0 when provided");
        // Wrap must align with block boundaries (otherwise a wrapped position would split
        // across two physical blocks and the kernel can't address it). Pull
        // effective_block_size from the same fallback the program factory uses.
        // block_size>0 is already enforced above when the override is active.
        const uint32_t effective_block_size = paged_geo.active() ? paged_geo.block_size : k_shape[2];
        TT_FATAL(
            modulo % effective_block_size == 0,
            "cache_position_modulo ({}) must be a multiple of effective block_size ({})",
            modulo,
            effective_block_size);
        if (operation_attributes.sliding_window_size.has_value() &&
            operation_attributes.sliding_window_size.value() > 0) {
            TT_FATAL(
                modulo >= operation_attributes.sliding_window_size.value(),
                "cache_position_modulo ({}) must be >= sliding_window_size ({}); a tighter "
                "capacity would let the SDPA attention range wrap multiple times and "
                "double-count positions.",
                modulo,
                operation_attributes.sliding_window_size.value());
        }
    }

    if (operation_attributes.paged_attention) {
        // Paged attention verification
        TT_FATAL(
            !operation_attributes.share_cache.value_or(false), "Share cache feature not supported for paged attention");
        TT_FATAL(tensor_args.page_table_tensor.has_value(), "Must have page_table tensor for paged attention");
        const auto& page_table_tensor = tensor_args.page_table_tensor.value();
        const auto B = page_table_tensor.is_sharded() ? q_shape[1] : page_table_tensor.padded_shape()[0];

        if (operation_attributes.is_causal) {
            // Check cur pos tensor for causal mode
            TT_FATAL(
                tensor_args.cur_pos_tensor.has_value(), "Must have cur_pos tensor for paged attention in causal mode");
            const auto& cur_pos_tensor = tensor_args.cur_pos_tensor.value();
            TT_FATAL(
                cur_pos_tensor.dtype() == DataType::INT32,
                "Expect cur_pos to be INT32, got {}",
                cur_pos_tensor.dtype());
            TT_FATAL(
                cur_pos_tensor.layout() == Layout::ROW_MAJOR,
                "Expect cur_pos to be ROW_MAJOR, got {}",
                cur_pos_tensor.layout());
            const auto cur_pos_shape = cur_pos_tensor.padded_shape();

            if (!cur_pos_tensor.is_sharded()) {
                TT_FATAL(
                    cur_pos_shape[-1] == B,
                    "cur_pos must have batch size equal to Q, got {} and {}",
                    cur_pos_shape[0],
                    B);
            }
        }

        if (page_table_tensor.is_sharded()) {
            TT_FATAL(
                page_table_tensor.dtype() == DataType::UINT16,
                "Error: SDPA currently only supports UINT16 datatype for sharded configurations");
        } else {
            TT_FATAL(
                page_table_tensor.dtype() == DataType::INT32, "Error: SDPA currently only supports INT32 datatype");
        }

        TT_FATAL(
            page_table_tensor.layout() == Layout::ROW_MAJOR,
            "Page table tensor layout must be ROW_MAJOR but got {}",
            page_table_tensor.layout());

        const auto page_table_shape = page_table_tensor.padded_shape();

        if (page_table_tensor.is_sharded()) {
            uint32_t num_cores = page_table_tensor.memory_config().shard_spec()->grid.num_cores();
            TT_FATAL(
                page_table_shape[0] / num_cores == B,
                "Page_table must have shard height batch_size {} equal to Q on {} cores",
                B,
                num_cores);
        } else {
            TT_FATAL(page_table_shape[0] == B, "Page_table must have batch size equal to Q");
        }

        TT_FATAL(k_shape[2] == v_shape[2], "K and V must have same block size");

        // Geometry overrides + MLA not yet exercised; reject until needed. Also closes the
        // asymmetry where num_kv_heads could be applied to V under MLA with no elems/block check.
        const auto& geo = operation_attributes.paged_cache_geometry;
        if (geo.active()) {
            TT_FATAL(!use_mla, "PagedCacheGeometryOverride is not supported with multi-latent attention");
        }

        if (use_mla) {
            TT_FATAL(
                k_shape[3] == q_shape[3], "Q and K must have same hidden size, got {} and {}", k_shape[3], q_shape[3]);
            if (tensor_args.v.has_value()) {
                TT_FATAL(
                    v_shape[3] == operation_attributes.head_dim_v.value(),
                    "V must have hidden size equal to head_dim_v, got {} and {}",
                    v_shape[3],
                    operation_attributes.head_dim_v.value());
            }
        } else if (geo.active()) {
            // Shared-buffer path: the K/V cache was allocated for a different layer's
            // (block_size, head_dim) shape, and this call reads through its own view.
            // Q's last dim drives head_dim; geo.block_size drives block_size.
            // K and V caches still share their per-layer allocation pair, and the
            // per-block element count (num_kv_heads * block_size * head_dim) must equal
            // what the cache was allocated for.
            //
            // Sanity guard only — not a correctness proof. Equal elems/block catches a
            // *size* mismatch but not a *geometry* mismatch: a caller that fills with the
            // declared geometry and reads with an override of equal elems/block still
            // passes and gets silent garbage. Read/write view geometry must match by
            // contract (same PagedCacheGeometryOverride on fill and SDPA).
            TT_FATAL(
                k_shape[3] == v_shape[3],
                "K and V cache must have same hidden size with geometry overrides, got {} and {}",
                k_shape[3],
                v_shape[3]);
            const uint32_t cache_num_kv_heads = k_shape[1];
            const uint32_t cache_block_size = k_shape[2];
            const uint32_t cache_head_dim = k_shape[3];
            const uint32_t q_head_dim = q_shape[3];
            // Both fields are required plain values whenever the override is active.
            const uint32_t effective_block_size = geo.block_size;
            // num_kv_heads on the call-view side may differ from the cache when the
            // caller is reading an HMA cross-group buffer (e.g. Gemma4-26B-A4B sliding
            // kv=8 cache read by a full layer with kv=2). Earlier versions of this
            // check used cache_num_kv_heads on both sides, so the kv factor cancelled
            // and the per-kv-heads dimension was unchecked — masking real mismatches.
            const uint32_t view_num_kv_heads = geo.num_kv_heads;
            const uint64_t cache_elems_per_block =
                static_cast<uint64_t>(cache_num_kv_heads) * cache_block_size * cache_head_dim;
            const uint64_t view_elems_per_block =
                static_cast<uint64_t>(view_num_kv_heads) * effective_block_size * q_head_dim;
            TT_FATAL(
                view_elems_per_block == cache_elems_per_block,
                "paged_scaled_dot_product_attention_decode geometry mismatch: cache has {} elems/block "
                "(kv_heads={}, block_size={}, head_dim={}) but call view is {} "
                "(kv_heads={}, block_size={}, head_dim={} from Q.padded_shape[-1]).",
                cache_elems_per_block,
                cache_num_kv_heads,
                cache_block_size,
                cache_head_dim,
                view_elems_per_block,
                view_num_kv_heads,
                effective_block_size,
                q_head_dim);
            TT_FATAL(
                effective_block_size % tt::constants::TILE_HEIGHT == 0,
                "effective block_size ({}) must be a multiple of TILE_HEIGHT ({})",
                effective_block_size,
                tt::constants::TILE_HEIGHT);
            TT_FATAL(
                q_head_dim % tt::constants::TILE_WIDTH == 0,
                "Q last dim ({}) must be a multiple of TILE_WIDTH ({})",
                q_head_dim,
                tt::constants::TILE_WIDTH);
        } else {
            TT_FATAL(k_shape[3] == v_shape[3] && k_shape[3] == q_shape[3], "Q, K, V must have same hidden size");
        }
        // Validate chunk size for paged version
        // k_chunk_size can also be zero; if k_chunk_size = 0, figure it out in kernels
        TT_FATAL(
            operation_attributes.k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, got: {}",
            operation_attributes.k_chunk_size);
        if (!operation_attributes.is_causal) {
            TT_FATAL(operation_attributes.k_chunk_size > 0, "Must provide k_chunk_size if paged and non-causal!");
            TT_FATAL(
                (page_table_shape[1] * k_shape[2]) % operation_attributes.k_chunk_size == 0,
                "K sequence length must be multiple of chunk size, got: {} and {}",
                page_table_shape[1] * k_shape[2],
                operation_attributes.k_chunk_size);
        }
    } else {
        // Unpaged attention verification
        TT_FATAL(
            not tensor_args.page_table_tensor.has_value(), "Must not have page_table tensor for unpaged attention");
        // Assert we are in decode mode if not causal
        // Q: [1, B, NH, D]
        // K: [1, B, S, D]
        // V: [1, B, S, D]

        // Batch must match
        const auto B = q_shape[1];
        if (operation_attributes.share_cache.value_or(false)) {
            TT_FATAL(k_shape[0] == 1, "Share cache expects K to have batch size of 1, but got {}", k_shape[0]);
            TT_FATAL(v_shape[0] == 1, "Share cache expects V to have batch size of 1, but got {}", v_shape[0]);
        } else {
            TT_FATAL(k_shape[0] == B, "K tensor batch size ({}) must equal B ({})", k_shape[0], B);
            TT_FATAL(v_shape[0] == B, "V tensor batch size ({}) must equal B ({})", v_shape[0], B);
        }
        // TT_FATAL(Q_memcfg.shard_spec.value().grid.num_cores() == B, "Q must be height sharded by batch ");

        // Q seqlen must be 1 if we are running decode mode
        TT_FATAL(q_shape[0] == 1, "Q tensor batch size must be 1 for decode mode but got {}", q_shape[0]);

        // Check sequence lengths
        TT_FATAL(
            k_shape[-2] == v_shape[-2],
            "K and V tensors must have the same sequence length. K: {}, V: {}",
            k_shape[-2],
            v_shape[-2]);

        // Validate chunk size for unpaged version
        TT_FATAL(operation_attributes.k_chunk_size > 0, "Must provide k_chunk_size if non-causal!");
        TT_FATAL(
            operation_attributes.k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, got: {}",
            operation_attributes.k_chunk_size);
        TT_FATAL(
            k_shape[2] % operation_attributes.k_chunk_size == 0,
            "K sequence length must be multiple of chunk size, got: {} and {}",
            k_shape[2],
            operation_attributes.k_chunk_size);

        // Check hidden size
        const auto D = q_shape[-1];
        // Palu: K in DRAM is the latent [B, nkv, S, r_k]; the head dim comes from Q. The latent width
        // itself is checked once, in the use_mla block above (on the logical width, which fires first).
        TT_FATAL(D == 128, "palu_sdpa_decode: head_dim must be 128, got {}", D);

        if (!use_mla) {
            TT_FATAL(
                v_shape[-1] == D,
                "V tensor hidden dimension ({}) must equal Q tensor hidden dimension ({})",
                v_shape[-1],
                D);
        }
        // Check valid seqlen
        for (unsigned int cur_pos_val : operation_attributes.cur_pos) {
            TT_FATAL(cur_pos_val < k_shape[-2], "cur_pos must be <= K sequence dim");
        }
    }

    // Check gqa specific validation
    TT_FATAL(
        k_shape[1] == v_shape[1],
        "Flash decode expects K and V to have same number of heads, but got {} and {}",
        k_shape[1],
        v_shape[1]);
    bool is_gqa = (k_shape[1] > 1);
    if (is_gqa) {
        TT_FATAL(!operation_attributes.output_mem_config.is_sharded(), "Sharded output not supported for GQA");
        TT_FATAL(
            input_tensors.at(0).dtype() == DataType::BFLOAT16,
            "GQA expects BFLOAT16 input tensor, but got {}",
            input_tensors.at(0).dtype());
        TT_FATAL(
            q_shape_unpadded[2] % k_shape[1] == 0,
            "GQA expects Q to have a multiple of K heads, but got {} and {}",
            q_shape_unpadded[2],
            k_shape[1]);
    }

    // Check attention sink
    if (tensor_args.attention_sink.has_value()) {
        const auto& attention_sink = tensor_args.attention_sink.value();

        const auto& sink_shape = attention_sink.padded_shape();
        TT_FATAL(sink_shape.size() == 2, "Attention sink must have 2 dimensions");
        TT_FATAL(
            sink_shape[0] == q_shape[2],
            "Attention sink must have the same padded num heads as Q but got {}",
            sink_shape[0]);
        TT_FATAL(
            sink_shape[1] == tt::constants::TILE_WIDTH,
            "Attention sink must be a single tile wide, but got {}",
            sink_shape[1]);
        TT_FATAL(
            attention_sink.dtype() == DataType::BFLOAT16,
            "Attention sink must by a BF16 tensor, but got {}",
            attention_sink.dtype());
        TT_FATAL(
            attention_sink.layout() == Layout::TILE,
            "Attention sink must be in TILE layout, but got {}",
            attention_sink.layout());
        TT_FATAL(
            attention_sink.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Attention sink must be in DRAM memory, but got {}",
            attention_sink.memory_config().buffer_type());
    }

    // ---------------- Palu envelope (stage 2 spec §3) ----------------
    TT_FATAL(operation_attributes.is_causal && !operation_attributes.paged_attention, "palu_sdpa_decode: causal, unpaged only");
    TT_FATAL(tensor_args.cur_pos_tensor.has_value(), "palu_sdpa_decode: cur_pos_tensor is required");
    TT_FATAL(
        !tensor_args.attn_mask.has_value() && !tensor_args.attention_sink.has_value() &&
            !tensor_args.page_table_tensor.has_value() && !operation_attributes.sliding_window_size.has_value() &&
            !operation_attributes.share_cache.value_or(false),
        "palu_sdpa_decode: mask, sink, page table, sliding window and share_cache are outside the envelope");
    TT_FATAL(tensor_args.q.layout() == Layout::TILE, "palu_sdpa_decode: Q must be TILE layout (c_10 is reused)");
    TT_FATAL(q_shape[1] <= 32, "palu_sdpa_decode: batch must be <= 32, got {}", q_shape[1]);
    const uint32_t r_k = k_shape[-1];
    const uint32_t S = k_shape[2];
    const uint32_t D_palu = q_shape[-1];
    auto expect = [](const Tensor& t, const char* name, std::initializer_list<uint32_t> shape, DataType dt) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr, "palu_sdpa_decode: {} must be on device", name);
        TT_FATAL(t.layout() == Layout::TILE, "palu_sdpa_decode: {} must be TILE layout", name);
        TT_FATAL(t.dtype() == dt, "palu_sdpa_decode: {} has the wrong dtype", name);
        TT_FATAL(t.padded_shape().rank() == 4, "palu_sdpa_decode: {} must be rank 4", name);
        uint32_t i = 0;
        for (uint32_t want : shape) {
            TT_FATAL(want == 0 || t.padded_shape()[i] == want, "palu_sdpa_decode: {} dim {} is {}, expected {}", name, i, t.padded_shape()[i], want);
            i++;
        }
    };
    const uint32_t n_kv = tensor_args.w_rot.padded_shape()[1];
    const uint32_t hpg = operation_attributes.heads_per_group;
    TT_FATAL(
        hpg >= 1 && n_kv == hpg * k_shape[1],
        "palu_sdpa_decode: heads_per_group {} x latent heads {} != w_rot heads {}",
        hpg,
        k_shape[1],
        n_kv);
    expect(tensor_args.w_rot, "w_rot", {1, n_kv, r_k, 2 * D_palu}, DataType::BFLOAT16);
    if (operation_attributes.recon_v) {
        expect(*tensor_args.u_v, "u_v", {1, n_kv, v_shape[3], D_palu}, DataType::BFLOAT16);
        TT_FATAL(
            operation_attributes.head_dim_v.value() == D_palu,
            "palu_sdpa_decode: with u_v head_dim_v must be {} (got {})",
            D_palu,
            operation_attributes.head_dim_v.value());
        TT_FATAL(
            v_shape[3] % 32 == 0 && v_shape[3] <= 512,
            "palu_sdpa_decode: V latent width must be a multiple of 32 up to 512, got {}",
            v_shape[3]);
    } else {
        TT_FATAL(
            hpg == 1,
            "palu_sdpa_decode: grouped latents (heads_per_group {}) need u_v: the absorbed-V path is per "
            "head only",
            hpg);
    }
    expect(tensor_args.cs_tab, "cs_tab", {1, 1, 0, 2 * D_palu}, DataType::BFLOAT16);
    TT_FATAL(tensor_args.cs_tab.padded_shape()[2] >= S, "palu_sdpa_decode: cs_tab has {} rows, cache has {}", tensor_args.cs_tab.padded_shape()[2], S);
    if (operation_attributes.cs_recurrence) {
        TT_FATAL(tensor_args.trans_mat.has_value(), "palu_sdpa_decode: rot_step and trans_mat are optional together");
        expect(*tensor_args.rot_step, "rot_step", {1, 1, 32, 2 * D_palu}, DataType::FLOAT32);
        expect(*tensor_args.trans_mat, "trans_mat", {1, 1, 32, 32}, DataType::BFLOAT16);
        auto [fid, approx, fp32_dest, l1_acc, full_sync] =
            get_compute_kernel_config_args(tensor_args.q.device()->arch(), operation_attributes.compute_kernel_config);
        TT_FATAL(fp32_dest, "palu_sdpa_decode: Phase 2 (rot_step) requires fp32_dest_acc_en");
    } else {
        TT_FATAL(
            !tensor_args.trans_mat.has_value(), "palu_sdpa_decode: rot_step and trans_mat are optional together");
        // Phase 1 streams a [c, 2*D] cos/sin block per chunk; at the op's default chunk that CB alone
        // is 512 KB and the program does not fit in L1. Fail here instead of in the L1 allocator.
        TT_FATAL(
            operation_attributes.k_chunk_size <= 128,
            "palu_sdpa_decode: Phase 1 (no rot_step) needs k_chunk_size <= 128 (it streams the [cos | sin] "
            "table per chunk), got {}",
            operation_attributes.k_chunk_size);
    }
}

tt::tt_metal::TensorSpec PaluSdpaDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.q;
    const auto q_shape = input.padded_shape();
    const auto q_shape_unpadded = input.logical_shape();
    const bool use_mla = operation_attributes.use_mla.value_or(false);
    uint32_t B;
    if (operation_attributes.paged_attention && tensor_args.page_table_tensor.has_value()) {
        const auto& page_table = tensor_args.page_table_tensor.value();
        if (page_table.is_sharded()) {
            uint32_t num_cores = page_table.memory_config().shard_spec()->grid.num_cores();
            const uint32_t total_batch = page_table.padded_shape()[0];
            TT_FATAL(
                total_batch % num_cores == 0,
                "Paged attention with sharded page table expects batch dimension batch size to be divisible by number "
                "of cores");
            B = total_batch / num_cores;
        } else {
            B = page_table.padded_shape()[0];
        }
    } else {
        B = tensor_args.k.padded_shape()[0];
    }
    bool q_locally_available = false;
    uint32_t num_q_heads = q_shape_unpadded[2];
    if (input.is_sharded() && use_mla && operation_attributes.program_config.has_value()) {
        const uint32_t q_shard_height = input.memory_config().shard_spec()->shape[0];
        const uint32_t max_cores = operation_attributes.program_config->max_cores_per_head_batch;
        const uint32_t num_q_shards = input.memory_config().shard_spec()->grid.num_cores();
        const uint32_t num_groups = num_q_shards / max_cores;
        const uint32_t q_heads_parallel_factor = num_groups / B;
        q_locally_available = (q_shape[2] == B * q_shard_height * q_heads_parallel_factor * max_cores);
        if (q_locally_available) {
            num_q_heads = q_heads_parallel_factor * q_shard_height;
        }
    }
    Layout output_layout = input.layout() == Layout::ROW_MAJOR ? Layout::ROW_MAJOR : Layout::TILE;
    ttnn::Shape output_shape = input.logical_shape();
    output_shape[1] = B;
    output_shape[2] = num_q_heads;

    // Preserve logical head_dim by default; only override in MLA mode.
    if (use_mla) {
        output_shape[3] = operation_attributes.head_dim_v.value();
    }
    return tt::tt_metal::TensorSpec(
        output_shape, TensorLayout(input.dtype(), PageConfig(output_layout), operation_attributes.output_mem_config));
}

Tensor PaluSdpaDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

Tensor palu_sdpa_decode(
    const Tensor& input_tensor_q,
    const Tensor& k_latent,
    const Tensor& w_rot,
    const Tensor& cs_tab,
    const Tensor& v_latent,
    const Tensor& cur_pos_tensor,
    const std::optional<const Tensor>& rot_step,
    const std::optional<const Tensor>& trans_mat,
    const std::optional<const Tensor>& u_v,
    uint32_t head_dim_v,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    uint32_t k_chunk_size) {
    using OperationType = PaluSdpaDecodeDeviceOperation;
    // Palu G-LRD: the rotation weight is per KV head, the latent cache is per group.
    const uint32_t n_kv = w_rot.padded_shape()[1];
    const uint32_t n_groups = k_latent.padded_shape()[1];
    TT_FATAL(
        n_groups > 0 && n_kv % n_groups == 0,
        "palu_sdpa_decode: heads_per_group must be an integer: w_rot has {} heads, k_latent {}",
        n_kv,
        n_groups);
    // Stock attributes pinned to the envelope: causal, unpaged, MLA-style separate V, cur_pos from tensor.
    auto operation_attributes = OperationType::operation_attributes_t{
        .is_causal = true,
        .paged_attention = false,
        .cur_pos = {},
        .scale = scale,
        .sliding_window_size = std::nullopt,
        .output_mem_config = output_mem_config,
        .program_config = program_config,
        .compute_kernel_config = compute_kernel_config,
        .k_chunk_size = k_chunk_size,
        .share_cache = std::nullopt,
        .use_mla = true,
        .head_dim_v = head_dim_v,
        .paged_cache_geometry = ttnn::operations::transformer::PagedCacheGeometryOverride{},
        .cache_position_modulo = std::nullopt,
        .cs_recurrence = rot_step.has_value(),
        .heads_per_group = n_kv / n_groups,
        .recon_v = u_v.has_value(),
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = input_tensor_q,
        .k = k_latent,
        .v = v_latent,
        .cur_pos_tensor = cur_pos_tensor,
        .page_table_tensor = std::nullopt,
        .attn_mask = std::nullopt,
        .attention_sink = std::nullopt,
        .w_rot = w_rot,
        .cs_tab = cs_tab,
        .rot_step = rot_step,
        .trans_mat = trans_mat,
        .u_v = u_v,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
