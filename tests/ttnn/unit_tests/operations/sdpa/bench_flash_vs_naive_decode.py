# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
One-page report bench: flash decode vs naive 3-pass decode, Llama-3.1-8B shapes, b=1.

Same Q/K/V values, two paths:
  - flash: ttnn.transformer.scaled_dot_product_attention_decode (FlashDecoding)
  - naive: ttnn.matmul(Q, K^T)*scale -> softmax -> ttnn.matmul(., V)
           (no mask: cur_pos=s-1 so all keys are valid)

GQA is unrolled on the host (K,V repeated to nh) for the naive matmul.
For decode the s² intermediate doesn't exist (Q has length 1 -> [b, nh, 1, s]),
so the flash advantage here is multi-core split-K reduction + kernel tuning,
NOT HBM intermediate savings (that's the prefill story).

Run:
  pytest tests/ttnn/unit_tests/operations/sdpa/bench_flash_vs_naive_decode.py -s

Output: /tmp/flash_vs_naive_decode.csv
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

CSV_PATH = Path(os.environ.get("FLASH_VS_NAIVE_DECODE_CSV", "/tmp/flash_vs_naive_decode.csv"))

LLAMA31_8B = dict(nh=32, nkv=8, d=128)

SMOKE = os.environ.get("SDPA_BENCH_SMOKE", "0") == "1"
if SMOKE:
    SEQLEN_SWEEP = [512, 2048]
    WARMUP_ITERS = 2
    TIMED_ITERS = 5
else:
    SEQLEN_SWEEP = [512, 2048, 8192]
    WARMUP_ITERS = 5
    TIMED_ITERS = 20


def _build_inputs(device, b, s, dtype):
    nh, nkv, d = LLAMA31_8B["nh"], LLAMA31_8B["nkv"], LLAMA31_8B["d"]

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)
    Q_full = fa_rand(1, b, nh, d)  # FlashDecode layout

    # naive layout: Q [b, nh, 1, d], K_rep_T [b, nh, d, s], V_rep [b, nh, s, d]
    Q_naive = Q_full.permute(1, 2, 0, 3).contiguous()
    K_rep = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_rep = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    K_rep_T = K_rep.transpose(-1, -2).contiguous()

    dram = ttnn.DRAM_MEMORY_CONFIG
    tt_Q_flash = ttnn.as_tensor(Q_full, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_K_flash = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_V_flash = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_Q_naive = ttnn.from_torch(
        Q_naive, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram, device=device, pad_value=0.0
    )
    tt_K_rep_T = ttnn.from_torch(
        K_rep_T, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram, device=device, pad_value=0.0
    )
    tt_V_rep = ttnn.from_torch(
        V_rep, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram, device=device, pad_value=0.0
    )

    cur_pos = np.full(b, s - 1, dtype=np.int32).tolist()
    tt_cur_pos = ttnn.Tensor(torch.tensor(cur_pos), ttnn.int32).to(device)

    return tt_Q_flash, tt_K_flash, tt_V_flash, tt_cur_pos, tt_Q_naive, tt_K_rep_T, tt_V_rep


def _make_compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
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

    tt_Q_flash, tt_K_flash, tt_V_flash, tt_cur_pos, tt_Q_naive, tt_K_rep_T, tt_V_rep = _build_inputs(
        device, b, s, dtype
    )

    grid_size = (8, 8)
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    flash_pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=0,
        exp_approx_mode=False,
    )
    compute_cfg = _make_compute_config()
    dram = ttnn.DRAM_MEMORY_CONFIG

    def call_flash():
        return ttnn.transformer.scaled_dot_product_attention_decode(
            tt_Q_flash,
            tt_K_flash,
            tt_V_flash,
            cur_pos_tensor=tt_cur_pos,
            scale=scale,
            program_config=flash_pc,
            compute_kernel_config=compute_cfg,
            memory_config=dram,
        )

    def call_naive():
        # QK^T  ->  [b, nh, 1, s]
        qk = ttnn.matmul(tt_Q_naive, tt_K_rep_T, compute_kernel_config=compute_cfg)
        qk_scaled = ttnn.multiply(qk, scale)
        ttnn.deallocate(qk)
        # softmax (no mask: full context, every key is valid)
        p = ttnn.softmax(qk_scaled, dim=-1)
        ttnn.deallocate(qk_scaled)
        # P @ V -> [b, nh, 1, d]
        out = ttnn.matmul(p, tt_V_rep, compute_kernel_config=compute_cfg)
        ttnn.deallocate(p)
        return out

    flash_samples = _time_op(call_flash, device, WARMUP_ITERS, TIMED_ITERS)
    naive_samples = _time_op(call_naive, device, WARMUP_ITERS, TIMED_ITERS)

    for t in (tt_Q_flash, tt_K_flash, tt_V_flash, tt_cur_pos, tt_Q_naive, tt_K_rep_T, tt_V_rep):
        ttnn.deallocate(t)

    return flash_samples, naive_samples


@pytest.mark.timeout(0)
def test_flash_vs_naive_decode(device):
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
