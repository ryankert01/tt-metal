# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Op-level ablation: paged SDPA decode latency vs k_chunk_size at Llama-3.1-8B shapes.

k_chunk_size=0 -> auto-tile (flash). Larger values approach a one-shot softmax
(non-flash). The kernel asserts non-zero k_chunk_size is a power of 2 AND a
multiple of 32; the sweep grid only feeds legal values.

Run:
  pytest tests/ttnn/unit_tests/operations/sdpa/bench_paged_sdpa_decode_ablation.py -s

Output: /tmp/sdpa_ablation_oplevel.csv
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

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand, num_to_corerange, nearest_n, nearest_pow_2

CSV_PATH = Path(os.environ.get("SDPA_ABLATION_CSV", "/tmp/sdpa_ablation_oplevel.csv"))

# (batch, n_heads, n_kv_heads, head_dim, max_seqlen) — Llama-3.1-8B GQA
LLAMA31_8B = dict(nh=32, nkv=8, d=128)
SMOKE = os.environ.get("SDPA_ABLATION_SMOKE", "0") == "1"
if SMOKE:
    BATCH_SWEEP = [1]
    SEQLEN_SWEEP = [512]
    K_CHUNK_SWEEP = [0, 128]
    WARMUP_ITERS = 2
    TIMED_ITERS = 5
else:
    BATCH_SWEEP = [1, 8, 16, 32]
    SEQLEN_SWEEP = [512, 2048, 8192, 32768]
    # legal: power-of-2 AND multiple of 32. 0 = auto.
    K_CHUNK_SWEEP = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    WARMUP_ITERS = 5
    TIMED_ITERS = 20
BLOCK_SIZE = 32  # paged-cache block size (tile-aligned)


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
    paged_k = _to_paged(K, b, nkv, blocks_per_seq, BLOCK_SIZE, d)
    paged_v = _to_paged(V, b, nkv, blocks_per_seq, BLOCK_SIZE, d)

    permutation = torch.randperm(total_blocks)
    page_table = torch.argsort(permutation).reshape(b, blocks_per_seq)
    paged_k_shuf = paged_k[permutation]
    paged_v_shuf = paged_v[permutation]

    dram = ttnn.DRAM_MEMORY_CONFIG
    tt_K = ttnn.as_tensor(paged_k_shuf, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_V = ttnn.as_tensor(paged_v_shuf, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_pt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    Q = fa_rand(1, b, nh, d)
    tt_Q = ttnn.as_tensor(Q[:, :, :nh], device=device, dtype=q_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)

    cur_pos = np.full(b, s - 1, dtype=np.int32).tolist()
    tt_cur_pos = ttnn.Tensor(torch.tensor(cur_pos), ttnn.int32).to(device)

    return tt_Q, tt_K, tt_V, tt_pt, tt_cur_pos


def _run_one(device, b, s, k_chunk_size, kv_dtype=ttnn.bfloat8_b, q_dtype=ttnn.bfloat16):
    nh, nkv, d = LLAMA31_8B["nh"], LLAMA31_8B["nkv"], LLAMA31_8B["d"]
    grid_size = (8, 8)

    tt_Q, tt_K, tt_V, tt_pt, tt_cur_pos = _build_inputs(device, b, s, kv_dtype, q_dtype)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram = ttnn.DRAM_MEMORY_CONFIG
    scale = d**-0.5

    def call():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_Q,
            tt_K,
            tt_V,
            tt_pt,
            cur_pos_tensor=tt_cur_pos,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )

    for _ in range(WARMUP_ITERS):
        out = call()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples_us = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter_ns()
        out = call()
        ttnn.synchronize_device(device)
        t1 = time.perf_counter_ns()
        samples_us.append((t1 - t0) / 1000.0)
        ttnn.deallocate(out)

    ttnn.deallocate(tt_Q)
    ttnn.deallocate(tt_K)
    ttnn.deallocate(tt_V)
    ttnn.deallocate(tt_pt)
    ttnn.deallocate(tt_cur_pos)

    return samples_us


def _legal_chunks(seqlen):
    out = []
    for k in K_CHUNK_SWEEP:
        if k == 0:
            out.append(k)
            continue
        if k > seqlen:
            continue
        if k % 32 != 0:
            continue
        if (k & (k - 1)) != 0:
            continue
        out.append(k)
    return out


def _run_with_cores(device, b, s, max_cores_per_head_batch, kv_dtype=ttnn.bfloat8_b, q_dtype=ttnn.bfloat16):
    """Same as _run_one but at a fixed (auto) k_chunk_size, varying max_cores_per_head_batch."""
    nh, nkv, d = LLAMA31_8B["nh"], LLAMA31_8B["nkv"], LLAMA31_8B["d"]
    grid_size = (8, 8)

    tt_Q, tt_K, tt_V, tt_pt, tt_cur_pos = _build_inputs(device, b, s, kv_dtype, q_dtype)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=0,
        exp_approx_mode=False,
        max_cores_per_head_batch=max_cores_per_head_batch,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram = ttnn.DRAM_MEMORY_CONFIG
    scale = d**-0.5

    def call():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_Q, tt_K, tt_V, tt_pt,
            cur_pos_tensor=tt_cur_pos,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )

    for _ in range(WARMUP_ITERS):
        out = call(); ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples_us = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter_ns()
        out = call()
        ttnn.synchronize_device(device)
        t1 = time.perf_counter_ns()
        samples_us.append((t1 - t0) / 1000.0)
        ttnn.deallocate(out)

    for t in (tt_Q, tt_K, tt_V, tt_pt, tt_cur_pos):
        ttnn.deallocate(t)
    return samples_us


@pytest.mark.timeout(0)
def test_paged_sdpa_decode_max_cores_sweep(device):
    """Per FlashDecode tech report §4.2: at small (bsz*n_kv_heads) the cap matters.
    Llama-3.1-8B single-chip: n_kv_heads=8 -> at b=1, 8 works on 64 cores (8 cores/work).
    Sweep cap to see where the kernel stops being able to add cores or starts NoC-thrashing."""
    csv_path = Path("/tmp/sdpa_ablation_max_cores.csv")
    cells = [(1, 512), (1, 8192), (8, 512), (8, 2048)]
    caps = [1, 2, 4, 8, 16, 32, 64]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["batch", "seqlen", "max_cores_per_head_batch", "iters", "p50_us", "p95_us", "min_us", "status"])
        for b, s in cells:
            for cap in caps:
                label = f"b={b} s={s} cap={cap}"
                try:
                    samples = _run_with_cores(device, b, s, cap)
                    p50 = statistics.median(samples)
                    p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]
                    mn = min(samples)
                    logger.info(f"{label}: p50={p50:.1f}us p95={p95:.1f}us min={mn:.1f}us")
                    w.writerow([b, s, cap, len(samples), f"{p50:.2f}", f"{p95:.2f}", f"{mn:.2f}", "ok"])
                except (RuntimeError, AssertionError) as e:
                    msg = str(e).splitlines()[0][:120]
                    logger.warning(f"{label}: FAILED ({msg})")
                    w.writerow([b, s, cap, 0, "nan", "nan", "nan", f"fail: {msg}"])
                f.flush()


@pytest.mark.timeout(0)
def test_paged_sdpa_decode_ablation(device):
    rows = []
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["batch", "seqlen", "k_chunk_size", "iters", "p50_us", "p95_us", "min_us", "status"])

        for b in BATCH_SWEEP:
            for s in SEQLEN_SWEEP:
                for k_chunk in _legal_chunks(s):
                    label = f"b={b} s={s} k={k_chunk}"
                    try:
                        samples = _run_one(device, b, s, k_chunk)
                        p50 = statistics.median(samples)
                        p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]
                        mn = min(samples)
                        status = "ok"
                        logger.info(f"{label}: p50={p50:.1f}us p95={p95:.1f}us min={mn:.1f}us")
                        w.writerow([b, s, k_chunk, len(samples), f"{p50:.2f}", f"{p95:.2f}", f"{mn:.2f}", status])
                    except (RuntimeError, AssertionError) as e:
                        msg = str(e).splitlines()[0][:120]
                        logger.warning(f"{label}: FAILED ({msg})")
                        w.writerow([b, s, k_chunk, 0, "nan", "nan", "nan", f"fail: {msg}"])
                    f.flush()
                    rows.append(label)

    logger.info(f"Wrote {len(rows)} rows to {CSV_PATH}")
