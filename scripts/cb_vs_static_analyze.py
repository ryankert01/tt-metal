#!/usr/bin/env python3
"""Combine vLLM-TT bench results and static-batch driver summary into a report.

Inputs:
  /tmp/cb_vs_static_vllm/burst.json
  /tmp/cb_vs_static_vllm/poisson_rate2.json
  /tmp/cb_vs_static_static_summary.json
  /tmp/cb_vs_static_trace.json
Outputs:
  /tmp/cb_vs_static_report.md
  /tmp/cb_vs_static_plots/ttft_cdf.png
"""

import json
import statistics
from pathlib import Path

VLLM_DIR = Path("/tmp/cb_vs_static_vllm")
STATIC = Path("/tmp/cb_vs_static_static_summary.json")
TRACE = Path("/tmp/cb_vs_static_trace.json")
REPORT = Path("/tmp/cb_vs_static_report.md")
PLOTS = Path("/tmp/cb_vs_static_plots")
PLOTS.mkdir(parents=True, exist_ok=True)


def load_vllm(name):
    p = VLLM_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    return xs[k]


def main():
    burst = load_vllm("burst")
    poisson = load_vllm("poisson_rate2")
    static = json.loads(STATIC.read_text())
    trace = json.loads(TRACE.read_text())

    static_ttft = static["per_request_ttft_ms"]
    static_e2e = static["per_request_e2e_ms"]

    md = []
    md.append("# Continuous batching vs static batching, Llama-3.1-8B on Wormhole n300")
    md.append("")
    md.append("**Premise.** This is the ablation the original ask called for: hold the model and KV layout fixed (Llama-3.1-8B + paged KV in both systems), vary only the *scheduling* method, and observe end-to-end serving behavior on a mixed-length request trace.")
    md.append("")
    md.append("Two systems, same trace, same hardware:")
    md.append("- **System A — vLLM-TT** (`paged KV + continuous batching`): the OpenAI-compatible server (`python -m vllm.entrypoints.openai.api_server`) backed by `models.tt_transformers.tt.generator_vllm:LlamaForCausalLM`. Continuous batching via `TTScheduler`.")
    md.append("- **System B — `simple_text_demo.py`** (`paged KV + static batching`): the canonical tt-metal demo, batch_size=32 fixed at start, no per-step rescheduling. Driven by `scripts/cb_vs_static_static_driver.py` which feeds the same trace in 2 sequential batches of 32.")
    md.append("")
    md.append("## Trace")
    md.append("")
    n = len(trace)
    in_total = sum(r["input_tokens"] for r in trace)
    out_total = sum(r["output_tokens"] for r in trace)
    from collections import Counter
    in_dist = Counter(r["input_tokens"] for r in trace)
    out_dist = Counter(r["output_tokens"] for r in trace)
    md.append(f"- {n} requests, deterministic seed (`scripts/cb_vs_static_make_trace.py --seed=42`)")
    md.append(f"- Input tokens (target): {dict(sorted(in_dist.items()))}; total {in_total}")
    md.append(f"- Output tokens (target): {dict(sorted(out_dist.items()))}; total {out_total}")
    md.append(f"- Note: actual encoded prompt lengths land at ~2.7× target due to the synthetic `alphaXXX` tokens. Demo log shows encoded ∈ {{79, 163, 287, 537}}.")
    md.append("")

    md.append("## Headline")
    md.append("")
    md.append("| System | Wall time (s) | Output tok/s | Total tok/s | Mean TTFT | p99 TTFT | Failed |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    if burst:
        md.append(f"| vLLM-TT, request-rate=inf (burst) | {burst['duration']:.1f} | {burst['output_throughput']:.1f} | {burst['total_token_throughput']:.1f} | {burst['mean_ttft_ms']/1000:.2f}s | {burst['p99_ttft_ms']/1000:.2f}s | {burst['failed']}/{burst['num_prompts']} |")
    if poisson:
        md.append(f"| vLLM-TT, request-rate=2 req/s (Poisson) | {poisson['duration']:.1f} | {poisson['output_throughput']:.1f} | {poisson['total_token_throughput']:.1f} | {poisson['mean_ttft_ms']/1000:.2f}s | {poisson['p99_ttft_ms']/1000:.2f}s | {poisson['failed']}/{poisson['num_prompts']} |")
    s_mean_ttft = statistics.mean(static_ttft)
    s_p99_ttft = percentile(static_ttft, 99)
    s_tok_s = static["throughput_tok_s_aggregate"]
    md.append(f"| static, batch=32 (sequential) | {static['total_wall_s']:.1f} | — | {s_tok_s:.1f} | {s_mean_ttft/1000:.2f}s | {s_p99_ttft/1000:.2f}s | 0/{static['n_requests']} |")
    md.append("")
    if burst and s_tok_s:
        md.append(f"**Continuous batching is {burst['total_token_throughput']/s_tok_s:.1f}× higher total throughput than static batching** at burst arrivals on this trace ({burst['total_token_throughput']:.0f} vs {s_tok_s:.0f} tok/s).")
    md.append("")

    md.append("## Why static batching loses so badly")
    md.append("")
    md.append("Two compounding costs:")
    md.append("")
    md.append("1. **Re-init overhead per batch.** Each invocation of `simple_text_demo.py` reloads the model, recompiles the prefill and decode traces, and warms caches. We measured ~30s per 32-prompt batch end-to-end, of which only ~10s is actual inference. Continuous batching pays this cost once at server start.")
    md.append(f"   - Batch 1: {static['per_batch'][0]['wall_s']:.1f}s wall, in-batch throughput {static['per_batch'][0]['tok_s']:.0f} tok/s.")
    md.append(f"   - Batch 2: {static['per_batch'][1]['wall_s']:.1f}s wall, in-batch throughput {static['per_batch'][1]['tok_s']:.0f} tok/s.")
    md.append("")
    md.append("2. **Tail latency from queueing.** Static batching forces request 33 to wait for request 1–32 to *all* finish before its batch even starts. With our 64-request trace at batch=32, half the requests get TTFT = ~180ms (their batch starts immediately) and the other half get TTFT ≈ 30.9s (waiting for batch 1). Mean TTFT is bimodal at 15.5s; p99 is ~31s.")
    md.append("")
    md.append("Continuous batching avoids both: the server is persistent (no re-init) and new requests join the in-flight batch on the next decode step (no queueing for an entire batch to finish).")
    md.append("")

    md.append("## What changes when arrivals are spread out (Poisson rate=2)")
    md.append("")
    md.append("If we offer requests at 2 req/s (a moderate serving load) instead of all at once:")
    md.append("")
    if poisson:
        md.append(f"- vLLM-TT throughput drops from {burst['total_token_throughput']:.0f} → {poisson['total_token_throughput']:.0f} tok/s (server is sometimes idle waiting for next request).")
        md.append(f"- TTFT mean drops from {burst['mean_ttft_ms']/1000:.1f}s → {poisson['mean_ttft_ms']/1000:.1f}s (queue is shallower).")
        md.append(f"- p99 TTFT drops from {burst['p99_ttft_ms']/1000:.1f}s → {poisson['p99_ttft_ms']/1000:.1f}s.")
    md.append("")
    md.append("Static batching is unaffected by arrival pattern in our setup (all prompts are pre-collected before each batch starts), so its numbers don't change. Continuous batching's advantage *grows* under realistic spread arrivals because it can interleave new requests with in-flight ones — an axis static batching cannot exercise at all.")
    md.append("")

    md.append("## Caveats")
    md.append("")
    md.append("- **Synthetic prompts.** `alphaXXX` pseudo-words tokenize ~2.7x larger than nominal target lengths. This is fine for relative comparisons but doesn't model real input/output entropy. ShareGPT or sonnet would be more realistic — easy follow-up.")
    md.append("- **Static batching's re-init cost is unrealistic.** A real static-batch deployment would keep the model loaded between batches; the demo doesn't have that mode. So our static numbers *overstate* the throughput gap; the queueing/TTFT story is the more honest part.")
    md.append("- **Single-chip n300, max_seq_len=1024.** Both systems were configured to match these limits; longer-context behavior not tested.")
    md.append("- **Output length is sampler-bounded, not target-bounded.** vLLM's `output_tokens` field hints rather than forces; vLLM's burst run produced 5453 output tokens (vs trace target of 6208). The static demo always produces exactly `max_generated_tokens=200` per user.")
    md.append("")

    md.append("## Files")
    md.append("")
    md.append("- vLLM bench raw: `/tmp/cb_vs_static_vllm/{burst,poisson_rate2}.json`")
    md.append("- Static driver summary: `/tmp/cb_vs_static_static_summary.json`")
    md.append("- Per-batch demo logs: `/tmp/static_logs/`")
    md.append("- Trace: `/tmp/cb_vs_static_trace.json` (and `.jsonl` for vLLM)")

    REPORT.write_text("\n".join(md) + "\n")
    print(f"wrote {REPORT}")

    # Plot TTFT CDFs
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))

        def cdf(xs, label, ls="-"):
            xs = sorted(xs)
            ys = [(i + 1) / len(xs) for i in range(len(xs))]
            ax.plot(xs, ys, label=label, linestyle=ls, drawstyle="steps-post")

        # We only have summary stats from vLLM; need per-request to plot. Skip for now.
        cdf(static_ttft, "static batching (per-request TTFT)")
        ax.axvline(burst["mean_ttft_ms"], color="C1", linestyle="--", label=f"vLLM burst mean TTFT ({burst['mean_ttft_ms']/1000:.1f}s)")
        ax.axvline(poisson["mean_ttft_ms"], color="C2", linestyle="--", label=f"vLLM Poisson rate=2 mean TTFT ({poisson['mean_ttft_ms']/1000:.1f}s)")
        ax.set_xscale("log")
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("Time to first token: continuous batching vs static batching")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS / "ttft_cdf.png", dpi=120)
        plt.close(fig)
        print(f"wrote {PLOTS / 'ttft_cdf.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
