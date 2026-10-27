# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
One-page report bench: paged vs non-paged SDPA decode at Llama-3.1-8B shapes.

Same Q/K/V values, same SDPAProgramConfig, two paths:
  - non-paged: ttnn.transformer.scaled_dot_product_attention_decode
  - paged:     ttnn.transformer.paged_scaled_dot_product_attention_decode
                with random page_table permutation (realistic fragmented case)

Run:
  pytest tests/ttnn/unit_tests/operations/sdpa/bench_paged_vs_nonpaged_decode.py -s

Output: /tmp/paged_vs_nonpaged_decode.csv
"""

import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand, nearest_n, nearest_pow_2

CSV_PATH = Path(os.environ.get("PAGED_VS_NONPAGED_CSV", "/tmp/paged_vs_nonpaged_decode.csv"))

LLAMA31_8B = dict(nh=32, nkv=8, d=128)
BLOCK_SIZE = 32

SMOKE = os.environ.get("SDPA_BENCH_SMOKE", "0") == "1"
if SMOKE:
    BATCH_SWEEP = [1, 8]
    SEQLEN = 2048
    WARMUP_ITERS = 2
    TIMED_ITERS = 5
else:
    BATCH_SWEEP = [1, 8, 16, 32]
    SEQLEN = 8192  # long context: where paging matters most
    WARMUP_ITERS = 5
    TIMED_ITERS = 20


def _to_paged(cache, b, nkv, blocks_per_seq, block_size, head_dim):
    return (
        cache.reshape(b, nkv, blocks_per_seq, block_size, head_dim)
        .transpose(1, 2)
        .reshape(b * blocks_per_seq, nkv, block_size, head_dim)
    )


def _build_inputs(device, b, s, kv_dtype, q_dtype):
    nh, nkv, d = LLAMA31_8B["nh"], LLAMA31_8B["nkv"], LLAMA31_8B["d"]
    blocks_per_seq = s // BLOCK_SIZE
    assert blocks_per_seq * BLOCK_SIZE == s
    total_blocks = b * blocks_per_seq

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    dram = ttnn.DRAM_MEMORY_CONFIG

    # non-paged contiguous tensors
    tt_K_np = ttnn.as_tensor(K, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_V_np = ttnn.as_tensor(V, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)

    # paged layout: shuffle blocks via random permutation, page_table maps logical -> physical
    paged_k = _to_paged(K, b, nkv, blocks_per_seq, BLOCK_SIZE, d)
    paged_v = _to_paged(V, b, nkv, blocks_per_seq, BLOCK_SIZE, d)
    permutation = torch.randperm(total_blocks)
    page_table = torch.argsort(permutation).reshape(b, blocks_per_seq)
    paged_k_shuf = paged_k[permutation]
    paged_v_shuf = paged_v[permutation]
    tt_K_p = ttnn.as_tensor(paged_k_shuf, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_V_p = ttnn.as_tensor(paged_v_shuf, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_pt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    Q = fa_rand(1, b, nh, d)
    tt_Q = ttnn.as_tensor(Q[:, :, :nh], device=device, dtype=q_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)

    cur_pos = np.full(b, s - 1, dtype=np.int32).tolist()
    tt_cur_pos = ttnn.Tensor(torch.tensor(cur_pos), ttnn.int32).to(device)

    return tt_Q, tt_K_np, tt_V_np, tt_K_p, tt_V_p, tt_pt, tt_cur_pos


def _make_program_config(grid_size, padded_num_heads):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=0,  # auto
        exp_approx_mode=False,
    )


def _make_compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _time_op(call_fn):
    for _ in range(WARMUP_ITERS):
        out = call_fn()
        ttnn.deallocate(out)
    samples_us = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter_ns()
        out = call_fn()
        ttnn.synchronize_device(out.device())
        t1 = time.perf_counter_ns()
        samples_us.append((t1 - t0) / 1000.0)
        ttnn.deallocate(out)
    return samples_us


def _run_pair(device, b, s, kv_dtype=ttnn.bfloat8_b, q_dtype=ttnn.bfloat16):
    nh, nkv, d = LLAMA31_8B["nh"], LLAMA31_8B["nkv"], LLAMA31_8B["d"]
    grid_size = (8, 8)

    tt_Q, tt_K_np, tt_V_np, tt_K_p, tt_V_p, tt_pt, tt_cur_pos = _build_inputs(device, b, s, kv_dtype, q_dtype)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    program_config = _make_program_config(grid_size, padded_num_heads)
    compute_kernel_config = _make_compute_config()
    dram = ttnn.DRAM_MEMORY_CONFIG
    scale = d**-0.5

    def call_nonpaged():
        return ttnn.transformer.scaled_dot_product_attention_decode(
            tt_Q,
            tt_K_np,
            tt_V_np,
            cur_pos_tensor=tt_cur_pos,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )

    def call_paged():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_Q,
            tt_K_p,
            tt_V_p,
            tt_pt,
            cur_pos_tensor=tt_cur_pos,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )

    ttnn.synchronize_device(device)
    np_samples = _time_op(call_nonpaged)
    p_samples = _time_op(call_paged)

    for t in (tt_Q, tt_K_np, tt_V_np, tt_K_p, tt_V_p, tt_pt, tt_cur_pos):
        ttnn.deallocate(t)

    return np_samples, p_samples


@pytest.mark.timeout(0)
def test_paged_vs_nonpaged_decode(device):
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["batch", "seqlen", "variant", "iters", "p50_us", "p95_us", "min_us", "status"])
        for b in BATCH_SWEEP:
            label = f"b={b} s={SEQLEN}"
            try:
                np_samples, p_samples = _run_pair(device, b, SEQLEN)
                for variant, samples in (("nonpaged", np_samples), ("paged", p_samples)):
                    p50 = statistics.median(samples)
                    p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]
                    mn = min(samples)
                    logger.info(f"{label} {variant}: p50={p50:.1f}us p95={p95:.1f}us min={mn:.1f}us")
                    w.writerow([b, SEQLEN, variant, len(samples), f"{p50:.2f}", f"{p95:.2f}", f"{mn:.2f}", "ok"])
            except (RuntimeError, AssertionError) as e:
                msg = str(e).splitlines()[0][:120]
                logger.warning(f"{label}: FAILED ({msg})")
                w.writerow([b, SEQLEN, "fail", 0, "nan", "nan", "nan", f"fail: {msg}"])
            f.flush()
    logger.info(f"Wrote results to {CSV_PATH}")
