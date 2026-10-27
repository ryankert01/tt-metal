#!/usr/bin/env python3
"""Static-batching comparison driver.

Reads the request trace, splits it into fixed batches of size B, and invokes
simple_text_demo.py once per batch via subprocess. Captures per-batch wall-clock
and the demo's reported avg TTFT / decode tok/s. From those it derives per-request
TTFT and E2EL distributions assuming all requests "arrive" at t=0 (the same
assumption vllm bench serve makes when --request-rate=inf).

Output: JSON summary written to /tmp/cb_vs_static_static_summary.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEMO_PATH = "models/tt_transformers/demo/simple_text_demo.py"
RE_TTFT = re.compile(r"Time to First Token \(TTFT\): ([0-9.]+)ms")
RE_AVG_SPEED = re.compile(r"Average speed: ([0-9.]+)ms @ ([0-9.]+) tok/s/user \(([0-9.]+) tok/s throughput\)")


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run_one_batch(batch_prompts, batch_size, max_seq_len, hf_model, tt_metal_home):
    """Write the batch to a temp JSON, invoke simple_text_demo with the right
    parametrize selector, parse out timings.
    Returns: (batch_wall_s, ttft_ms, itl_ms, tok_s_user, tok_s_throughput)"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"prompt": r["prompt"]} for r in batch_prompts], f)
        prompts_file = f.name

    # Use exact match against the parametrize id to avoid catching e.g. "batch-32-log-probs".
    selector = f"performance and batch-{batch_size} and not log-probs"
    env = os.environ.copy()
    env["HF_MODEL"] = hf_model
    env["TT_METAL_HOME"] = tt_metal_home
    env["ARCH_NAME"] = "wormhole_b0"
    env["PYTHONPATH"] = tt_metal_home
    env["TT_DEMO_PROMPTS_OVERRIDE"] = prompts_file

    cmd = [
        "pytest", "-s",
        DEMO_PATH,
        "-k", selector,
        "--capture=tee-sys",
    ]

    # Stream output to a per-batch log so progress is visible in real time.
    live_log = Path(env.get("STATIC_LIVE_LOG_DIR", "/tmp/static_logs")) / f"_live_pid{os.getpid()}.log"
    live_log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with live_log.open("w") as lf:
        proc = subprocess.run(cmd, cwd=tt_metal_home, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    log = live_log.read_text()
    ttft = float(m.group(1)) if (m := RE_TTFT.search(log)) else None
    sp = RE_AVG_SPEED.search(log)
    itl = float(sp.group(1)) if sp else None
    tok_s_u = float(sp.group(2)) if sp else None
    tok_s = float(sp.group(3)) if sp else None

    Path(prompts_file).unlink(missing_ok=True)
    return wall, ttft, itl, tok_s_u, tok_s, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, default=Path("/tmp/cb_vs_static_trace.json"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=Path("/tmp/cb_vs_static_static_summary.json"))
    ap.add_argument("--hf-model", default=os.environ.get("HF_MODEL"))
    ap.add_argument("--tt-metal-home", default=os.environ.get("TT_METAL_HOME", "/home/user01/tt-data/tt-metal"))
    args = ap.parse_args()

    if not args.hf_model:
        print("ERROR: HF_MODEL not set (use --hf-model or env var)", file=sys.stderr)
        sys.exit(1)

    trace = json.loads(args.trace.read_text())
    print(f"Loaded {len(trace)} requests from {args.trace}", flush=True)

    batches = list(chunks(trace, args.batch_size))
    print(f"Splitting into {len(batches)} batches of <= {args.batch_size}", flush=True)

    per_batch = []
    cum_wall = 0.0
    per_request_ttft_ms = []
    per_request_e2e_ms = []
    log_dir = args.out.parent / "static_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for bi, batch in enumerate(batches):
        # Need batch-size==32 fit; if last batch is smaller, repeat the first prompt
        # to pad to 32 — matches what the demo does internally for sub-32 batches.
        eff_batch = batch
        if len(batch) < args.batch_size:
            pad = [batch[0]] * (args.batch_size - len(batch))
            eff_batch = batch + pad

        print(f"\n[batch {bi+1}/{len(batches)}] {len(batch)} real prompts (padded to {len(eff_batch)})", flush=True)
        wall, ttft, itl, tok_s_u, tok_s, log = run_one_batch(
            eff_batch, args.batch_size, args.max_seq_len, args.hf_model, args.tt_metal_home
        )
        (log_dir / f"batch_{bi}.log").write_text(log)
        per_batch.append({
            "batch_index": bi,
            "n_real": len(batch),
            "wall_s": wall,
            "ttft_ms": ttft,
            "itl_ms": itl,
            "tok_s_user": tok_s_u,
            "tok_s": tok_s,
        })
        print(f"  wall={wall:.1f}s ttft={ttft} itl={itl} tok_s={tok_s}", flush=True)

        # Per-request: this batch's *real* requests started at cum_wall (all-at-once arrivals).
        # All requests in the batch get the same TTFT (batch's TTFT) and same E2EL (cum_wall + wall).
        ttft_for_user_ms = (cum_wall * 1000) + (ttft if ttft is not None else 0)
        e2e_for_user_ms = (cum_wall + wall) * 1000
        for _ in range(len(batch)):  # only count REAL requests, not padding
            per_request_ttft_ms.append(ttft_for_user_ms)
            per_request_e2e_ms.append(e2e_for_user_ms)
        cum_wall += wall

    summary = {
        "system": "static_batching",
        "batch_size": args.batch_size,
        "n_requests": len(trace),
        "n_batches": len(batches),
        "total_wall_s": cum_wall,
        "per_batch": per_batch,
        "per_request_ttft_ms": per_request_ttft_ms,
        "per_request_e2e_ms": per_request_e2e_ms,
        "throughput_tok_s_aggregate": (
            sum(r.get("output_tokens", 0) + r.get("input_tokens", 0) for r in trace) / cum_wall
            if cum_wall > 0 else 0
        ),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {args.out}\n  total_wall={cum_wall:.1f}s tok/s_agg={summary['throughput_tok_s_aggregate']:.1f}")


if __name__ == "__main__":
    main()
