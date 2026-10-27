# SPDX-License-Identifier: Apache-2.0
"""
Quantify the prefill perf hit of power-of-2 ISL padding (current policy) vs the
old next-2048-multiple policy. Context: https://github.com/tenstorrent/tt-metal/issues/34117

Key idea: prefill latency depends only on the *padded* length L, not on token
content. So we measure T(L) once per distinct padded length, then derive every
ISL's perf hit by lookup:  perf_hit(isl) = T(new_pad(isl)) / T(old_pad(isl)) - 1

We force prefill to run at an exact length by monkeypatching get_padded_prefill_len
to the identity, then feeding a synthetic prompt of exactly that width.

Run (Llama-3.1-8B on N300):
    HF_MODEL=meta-llama/Llama-3.1-8B-Instruct MESH_DEVICE=N300 \
    pytest models/tt_transformers/tests/test_padding_perf.py -s

Results are printed and written to /tmp/padding_perf.json for offline analysis
(see analyze_padding.py).
"""
import json
import math
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
import models.tt_transformers.tt.generator as generator_mod
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.tt_transformers.demo.simple_text_demo import prepare_generator_args


# --- the two padding policies (exact, recovered from git 39d81812) ---
def old_pad(n):
    """Pre-#34117: min(next power of 2, next multiple of 2048)."""
    if n <= 128:
        return 128
    return min(2 ** math.ceil(math.log2(n)), 2048 * math.ceil(n / 2048))


def new_pad(n):
    """Current code (common.py:641): next power of 2, floor 1024."""
    if n <= 128:
        return 128
    if n <= 1024:
        return 1024
    return 2 ** (n - 1).bit_length()


# Representative ISLs in the SINGLE-CHUNK regime only (padded <= max_prefill_chunk_size
# = 65536 for Llama-3.1-8B / N300), chosen just above powers of 2 to maximize the gap.
# The chunked regime (e.g. 70k) is a separate phase; see notes at bottom.
ISLS = [int(x) for x in os.getenv("PAD_ISLS", "5000,9000,17000,33000").split(",")]
# PAD_LENGTHS lets gen_isl_trace.py request exact buckets; else derive from ISLS.
_LEN_ENV = os.getenv("PAD_LENGTHS")
LENGTHS = (
    sorted({int(x) for x in _LEN_ENV.split(",")})
    if _LEN_ENV
    else sorted({f(n) for n in ISLS for f in (old_pad, new_pad)})
)

ITERS = int(os.getenv("PAD_ITERS", "10"))
WARMUP = int(os.getenv("PAD_WARMUP", "3"))
ENABLE_TRACE = os.getenv("PAD_TRACE", "0") == "1"  # eager by default: avoids trace-region OOM; ratio is trace-insensitive


@pytest.mark.timeout(3600)  # measuring up to 65536 (eager ~13s each) exceeds the default 300s
@pytest.mark.parametrize(
    "device_params",
    # fabric_config=True is REQUIRED: n300 prefill does cross-die ops over the fabric;
    # without it the workload fabric is "dispatch only" and prefill deadlocks. (demo's value)
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device", [{"N150": (1, 1), "N300": (1, 2)}.get(os.getenv("MESH_DEVICE", "N300"), (1, 2))], indirect=True
)
def test_padding_perf(mesh_device, monkeypatch):
    assert LENGTHS, "no lengths to measure"
    max_l = max(LENGTHS)
    assert max_l <= 65536, f"{max_l} exceeds single-chunk regime (65536); use the chunked-regime variant"

    # Size max_seq_len / paging to the lengths we actually measure (128k OOM'd the
    # device). max_seq_len = next power of 2 >= max_l (warmup requires a power-of-2 cap).
    model_max_seq_len = min(1 << (max_l - 1).bit_length(), 128 * 1024)
    block_size = 64
    page_params = {"page_block_size": block_size, "page_max_num_blocks_per_dp": model_max_seq_len // block_size + 64}
    model_args, model, _page_table, tt_kv_cache, tokenizer, _processor, _dp, _idx = prepare_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        instruct=True,
        global_batch_size=1,
        optimizations=lambda ma: DecodersPrecision.performance(ma.n_layers, ma.model_name),
        max_seq_len=model_max_seq_len,
        page_params=page_params,
        paged_attention=True,
        num_layers=None,
        use_prefetcher=False,
        use_hf_rope=False,
    )
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)
    # Force the sampling_params=None warmup path (the demo's proven one). On-device
    # sampling (penalties+log-probs) during prefill warmup deadlocks the device.
    for m in generator.model:
        m._supports_on_device_sampling = False

    # Force prefill to run at exactly the requested length (bypass the policy under test).
    monkeypatch.setattr(generator_mod, "get_padded_prefill_len", lambda n: n)

    # Run the model's own warmup once (compiles/traces supported buckets + one-time
    # device setup) — exactly as the demo does. Skipping this deadlocked prefill.
    warm_args = generator._mock_tokens(1, 2048, tt_kv_cache, 0)
    generator.prefill_forward_text(**warm_args, kv_cache=tt_kv_cache, enable_trace=ENABLE_TRACE, warmup_prefill=True)

    def time_prefill(L):
        # Build inputs exactly like the model's own warmup (_mock_tokens): correct
        # page-table shape/dtype, prompt_lens tensor, empty_slots. Content is irrelevant
        # for latency. (Hand-built inputs deadlocked the device.)
        args = generator._mock_tokens(1, L, tt_kv_cache, 0)
        call = lambda: generator.prefill_forward_text(
            **args,
            kv_cache=tt_kv_cache,
            enable_trace=ENABLE_TRACE,  # traced for [128,1024,2048,4096,8192]; eager otherwise (both work)
            warmup_prefill=False,  # already warmed once above
        )
        for _ in range(WARMUP):  # compile (+ capture trace if enabled)
            call()
        ttnn.synchronize_device(mesh_device)
        samples = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            call()
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - t0) * 1e3)  # ms
        return statistics.median(samples), statistics.pstdev(samples)

    # Optional sanity check: same L, different content -> latency should be ~identical.
    if os.getenv("PAD_SANITY"):
        a, _ = time_prefill(LENGTHS[0])
        torch.manual_seed(LENGTHS[0] + 1)
        b, _ = time_prefill(LENGTHS[0])
        logger.info(f"[sanity] content-independence at L={LENGTHS[0]}: {a:.1f} vs {b:.1f} ms")

    T = {}
    for L in LENGTHS:
        med, sd = time_prefill(L)
        T[L] = med
        logger.info(f"L={L:>6}  prefill = {med:8.2f} ms  (sd {sd:.2f})")

    # --- per-ISL perf hit ---
    print("\n=== prefill padding perf hit (Llama-3.1-8B / N300, single-chunk) ===")
    print(f"{'ISL':>7} {'old L':>7} {'old ms':>9} {'new L':>7} {'new ms':>9} {'hit':>8}")
    rows = []
    for n in ISLS:
        ol, nl = old_pad(n), new_pad(n)
        hit = T[nl] / T[ol] - 1.0
        rows.append({"isl": n, "old_L": ol, "new_L": nl, "old_ms": T[ol], "new_ms": T[nl], "hit": hit})
        print(f"{n:>7} {ol:>7} {T[ol]:>9.2f} {nl:>7} {T[nl]:>9.2f} {hit:>+7.1%}")

    # --- warmup cost (the thing the new policy buys): distinct buckets to compile ---
    rng = range(2048, max_l + 1, 2048)
    old_buckets = len({128} | {old_pad(x) for x in rng})
    new_buckets = len({128} | {new_pad(x) for x in rng})
    print(f"\nwarmup buckets <= {max_l}:  old={old_buckets}  new={new_buckets}  (fewer = faster warmup)")

    json.dump({"T_ms": T, "rows": rows, "max_prefill_chunk_size": 65536}, open("/tmp/padding_perf.json", "w"), indent=2)
    print("wrote /tmp/padding_perf.json")
