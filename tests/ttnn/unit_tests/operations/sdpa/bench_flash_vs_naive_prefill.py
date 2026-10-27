# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
One-page report bench: flash prefill vs naive 3-pass prefill, Llama-3.1-8B shapes, b=1.

Same Q/K/V values, two paths:
  - flash: ttnn.transformer.scaled_dot_product_attention(Q, K, V, is_causal=True)
  - naive: ttnn.matmul(Q, K^T)*scale -> +causal_mask -> ttnn.softmax -> ttnn.matmul(., V)
           (each intermediate writes to DRAM; this is exactly what flash fuses away)

GQA is unrolled on the host (K,V repeated to nh) so the naive matmuls are well-defined.

Run:
  pytest tests/ttnn/unit_tests/operations/sdpa/bench_flash_vs_naive_prefill.py -s

Output: /tmp/flash_vs_naive_prefill.csv
"""

import csv
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

CSV_PATH = Path(os.environ.get("FLASH_VS_NAIVE_CSV", "/tmp/flash_vs_naive_prefill.csv"))

LLAMA31_8B = dict(nh=32, nkv=8, d=128)

SMOKE = os.environ.get("SDPA_BENCH_SMOKE", "0") == "1"
if SMOKE:
    SEQLEN_SWEEP = [512, 1024]
    WARMUP_ITERS = 2
    TIMED_ITERS = 5
else:
    # naive intermediate is [1, nh=32, s, s] * 2 bytes; at s=4096 -> 4 GB. 8192 OOM-risky.
    SEQLEN_SWEEP = [512, 1024, 2048, 4096]
    WARMUP_ITERS = 3
    TIMED_ITERS = 10


def _build_inputs(device, b, s, dtype):
    nh, nkv, d = LLAMA31_8B["nh"], LLAMA31_8B["nkv"], LLAMA31_8B["d"]
    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    # GQA-expanded K, V for naive path
    K_rep = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_rep = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    K_rep_T = K_rep.transpose(-1, -2).contiguous()  # [b, nh, d, s] for matmul

    # causal mask: [1, 1, s, s], -inf on upper triangle (j > i)
    mask = torch.zeros(1, 1, s, s)
    mask.masked_fill_(torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1), float("-inf"))

    dram = ttnn.DRAM_MEMORY_CONFIG
    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram, device=device, pad_value=0.0)
    tt_K_rep_T = ttnn.from_torch(
        K_rep_T, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram, device=device, pad_value=0.0
    )
    tt_V_rep = ttnn.from_torch(
        V_rep, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram, device=device, pad_value=0.0
    )
    tt_mask = ttnn.from_torch(
        mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram, device=device, pad_value=0.0
    )
    return tt_Q, tt_K, tt_V, tt_K_rep_T, tt_V_rep, tt_mask


def _make_program_config(device, q_chunk, k_chunk):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def _make_compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _time_op(call_fn, device, warmup, iters):
    for _ in range(warmup):
        out = call_fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    samples_us = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = call_fn()
        ttnn.synchronize_device(device)
        t1 = time.perf_counter_ns()
        samples_us.append((t1 - t0) / 1000.0)
        ttnn.deallocate(out)
    return samples_us


def _run_pair(device, s, dtype=ttnn.bfloat16):
    nh, nkv, d = LLAMA31_8B["nh"], LLAMA31_8B["nkv"], LLAMA31_8B["d"]
    b = 1
    scale = d**-0.5

    tt_Q, tt_K, tt_V, tt_K_rep_T, tt_V_rep, tt_mask = _build_inputs(device, b, s, dtype)

    # flash chunk sizes (must be tile-aligned and divide s; pick small but legal)
    q_chunk = min(128, s)
    k_chunk = min(128, s)
    flash_pc = _make_program_config(device, q_chunk, k_chunk)
    compute_cfg = _make_compute_config()

    def call_flash():
        return ttnn.transformer.scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            is_causal=True,
            program_config=flash_pc,
            compute_kernel_config=compute_cfg,
        )

    def call_naive():
        # Step 1: QK^T * scale
        qk = ttnn.matmul(tt_Q, tt_K_rep_T, compute_kernel_config=compute_cfg)
        qk_scaled = ttnn.multiply(qk, scale)
        ttnn.deallocate(qk)
        # Step 2: + causal mask
        qk_masked = ttnn.add(qk_scaled, tt_mask)
        ttnn.deallocate(qk_scaled)
        # Step 3: softmax
        p = ttnn.softmax(qk_masked, dim=-1)
        ttnn.deallocate(qk_masked)
        # Step 4: P @ V
        out = ttnn.matmul(p, tt_V_rep, compute_kernel_config=compute_cfg)
        ttnn.deallocate(p)
        return out

    flash_samples = _time_op(call_flash, device, WARMUP_ITERS, TIMED_ITERS)
    naive_samples = _time_op(call_naive, device, WARMUP_ITERS, TIMED_ITERS)

    for t in (tt_Q, tt_K, tt_V, tt_K_rep_T, tt_V_rep, tt_mask):
        ttnn.deallocate(t)

    return flash_samples, naive_samples


@pytest.mark.timeout(0)
def test_flash_vs_naive_prefill(device):
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["batch", "seqlen", "variant", "iters", "p50_us", "p95_us", "min_us", "status"])
        for s in SEQLEN_SWEEP:
            label = f"b=1 s={s}"
            try:
                flash_samples, naive_samples = _run_pair(device, s)
                for variant, samples in (("flash", flash_samples), ("naive", naive_samples)):
                    p50 = statistics.median(samples)
                    p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]
                    mn = min(samples)
                    logger.info(f"{label} {variant}: p50={p50:.1f}us p95={p95:.1f}us min={mn:.1f}us")
                    w.writerow([1, s, variant, len(samples), f"{p50:.2f}", f"{p95:.2f}", f"{mn:.2f}", "ok"])
            except (RuntimeError, AssertionError) as e:
                msg = str(e).splitlines()[0][:120]
                logger.warning(f"{label}: FAILED ({msg})")
                w.writerow([1, s, "fail", 0, "nan", "nan", "nan", f"fail: {msg}"])
            f.flush()
    logger.info(f"Wrote results to {CSV_PATH}")
