#!/usr/bin/env python3
"""V2 static driver: passes per-batch max_generated_tokens (= max output_tokens
in trace) and stop_at_eos=0 so static demo really does keep slots occupied
for the longest user — exposing the inefficiency CB avoids.

Output: /tmp/cb_vs_static_static_summary_v2.json
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


def run_one_batch(batch_prompts, batch_size, max_gen_tokens, hf_model, tt_metal_home, log_dir, bi):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"prompt": r["prompt"]} for r in batch_prompts], f)
        prompts_file = f.name

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
        f"--max_generated_tokens={max_gen_tokens}",
        "--stop_at_eos=0",
    ]

    live_log = log_dir / f"batch_{bi}.log"
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
    return wall, ttft, itl, tok_s_u, tok_s, proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, default=Path("/tmp/cb_vs_static_trace_v2.json"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("/tmp/cb_vs_static_static_summary_v2.json"))
    ap.add_argument("--hf-model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--tt-metal-home", default=os.environ.get("TT_METAL_HOME", "/home/user01/tt-data/tt-metal"))
    args = ap.parse_args()

    trace = json.loads(args.trace.read_text())
    print(f"Loaded {len(trace)} requests from {args.trace}", flush=True)

    log_dir = args.out.parent / "static_logs_v2"
    log_dir.mkdir(parents=True, exist_ok=True)

    batches = list(chunks(trace, args.batch_size))
    per_batch = []
    cum_wall = 0.0
    per_request_ttft_ms = []
    per_request_e2e_ms = []
    total_target_output = 0

    for bi, batch in enumerate(batches):
        # Static demo doesn't shrink batch; max_gen_tokens applies globally.
        # Use the longest target in this batch — that's what static is forced to run for.
        max_gen = max(r["output_tokens"] for r in batch)
        eff_batch = batch
        if len(batch) < args.batch_size:
            pad = [batch[0]] * (args.batch_size - len(batch))
            eff_batch = batch + pad

        print(f"\n[batch {bi+1}/{len(batches)}] real={len(batch)} max_gen_tokens={max_gen}", flush=True)
        wall, ttft, itl, tok_s_u, tok_s, rc = run_one_batch(
            eff_batch, args.batch_size, max_gen, args.hf_model, args.tt_metal_home, log_dir, bi
        )
        per_batch.append({
            "batch_index": bi,
            "n_real": len(batch),
            "max_gen_tokens": max_gen,
            "wall_s": wall,
            "ttft_ms": ttft,
            "itl_ms": itl,
            "tok_s_user": tok_s_u,
            "tok_s": tok_s,
            "returncode": rc,
        })
        print(f"  wall={wall:.1f}s ttft={ttft} itl={itl} tok_s={tok_s} rc={rc}", flush=True)

        ttft_for_user_ms = (cum_wall * 1000) + (ttft if ttft is not None else 0)
        e2e_for_user_ms = (cum_wall + wall) * 1000
        for r in batch:
            per_request_ttft_ms.append(ttft_for_user_ms)
            per_request_e2e_ms.append(e2e_for_user_ms)
            total_target_output += r["output_tokens"]
        cum_wall += wall

    summary = {
        "system": "static_batching_v2",
        "batch_size": args.batch_size,
        "n_requests": len(trace),
        "n_batches": len(batches),
        "total_wall_s": cum_wall,
        "per_batch": per_batch,
        "per_request_ttft_ms": per_request_ttft_ms,
        "per_request_e2e_ms": per_request_e2e_ms,
        "target_output_tokens_total": total_target_output,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {args.out}\n  total_wall={cum_wall:.1f}s target_out={total_target_output}")


if __name__ == "__main__":
    main()
