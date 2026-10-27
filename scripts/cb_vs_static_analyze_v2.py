#!/usr/bin/env python3
"""V2 analyze: compare CB (vLLM) vs static_v2 with extreme heterogeneous outputs.

Inputs:
  /tmp/cb_v2_burst.json
  /tmp/cb_v2_poisson1.json
  /tmp/cb_vs_static_static_summary_v2.json
  /tmp/cb_vs_static_trace_v2.json
Outputs:
  /tmp/cb_v2_report.md
  /tmp/cb_v2_plots/ttft_cdf.png
"""

import json
import statistics
from pathlib import Path

VLLM_DIR = Path("/tmp")
STATIC = Path("/tmp/cb_vs_static_static_summary_v2.json")
TRACE = Path("/tmp/cb_vs_static_trace_v2.json")
REPORT = Path("/tmp/cb_v2_report.md")
PLOTS = Path("/tmp/cb_v2_plots")
PLOTS.mkdir(parents=True, exist_ok=True)


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[int(round((p / 100.0) * (len(xs) - 1)))]


def main():
    burst = json.loads((VLLM_DIR / "cb_v2_burst.json").read_text())
    poisson = json.loads((VLLM_DIR / "cb_v2_poisson1.json").read_text())
    static = json.loads(STATIC.read_text()) if STATIC.exists() else None
    trace = json.loads(TRACE.read_text())

    # vLLM-native static-imitation: 2 sequential 32-prompt benches
    vllm_static_b1_path = VLLM_DIR / "cb_v2_vllm_static_b1.json"
    vllm_static_b2_path = VLLM_DIR / "cb_v2_vllm_static_b2.json"
    vllm_static = None
    if vllm_static_b1_path.exists() and vllm_static_b2_path.exists():
        b1 = json.loads(vllm_static_b1_path.read_text())
        b2 = json.loads(vllm_static_b2_path.read_text())
        vs_wall = b1["duration"] + b2["duration"]
        vs_total_out = b1["total_output_tokens"] + b2["total_output_tokens"]
        # Per-request TTFT: replicate per-batch mean to all 32 users in that batch.
        # Batch 1 users arrive at t=0; batch 2 users arrive at t=0 too in burst
        # but client-side barrier holds them until batch 1 finishes.
        vs_ttft_ms_list = (
            [b1["mean_ttft_ms"]] * 32
            + [b1["duration"] * 1000 + b2["mean_ttft_ms"]] * 32
        )
        vllm_static = {
            "wall_s": vs_wall,
            "goodput": vs_total_out / vs_wall,
            "mean_ttft_ms": statistics.mean(vs_ttft_ms_list),
            "p99_ttft_ms": percentile(vs_ttft_ms_list, 99),
            "b1": b1,
            "b2": b2,
        }

    # Legacy static (simple_text_demo path) — different code from vLLM CB.
    # Kept for reference / unfairness illustration only.
    target_out_total = sum(r["output_tokens"] for r in trace)
    if static is not None:
        fair_static_wall_s = 0.0
        for b in static["per_batch"]:
            fair_decode_s = (b["itl_ms"] / 1000.0) * b["max_gen_tokens"] if b["itl_ms"] else 0
            fair_static_wall_s += fair_decode_s + ((b["ttft_ms"] or 0) / 1000.0)
        static_goodput = target_out_total / fair_static_wall_s if fair_static_wall_s > 0 else 0
        static_decoded_total = sum(b["max_gen_tokens"] * static["batch_size"] for b in static["per_batch"])
        cum = 0.0
        fair_ttft_ms = []
        for b in static["per_batch"]:
            decode_s = (b["itl_ms"] / 1000.0) * b["max_gen_tokens"]
            for r in trace[b["batch_index"] * static["batch_size"]: b["batch_index"] * static["batch_size"] + b["n_real"]]:
                fair_ttft_ms.append(cum * 1000 + (b["ttft_ms"] or 0))
            cum += decode_s + (b["ttft_ms"] or 0) / 1000.0
    else:
        fair_static_wall_s = 0
        static_goodput = 0
        static_decoded_total = 0
        fair_ttft_ms = []

    md = []
    md.append("# CB vs static (V2): heterogeneous output lengths")
    md.append("")
    md.append("Same harness as V1 but with two key changes:")
    md.append("- **Output_tokens highly heterogeneous**: {16, 64, 256, 384} weighted {0.4, 0.3, 0.2, 0.1}, 24× range")
    md.append("- **`--ignore-eos` (vLLM) + `--stop_at_eos=0` (static demo)**: per-request output is binding, not just hint. Static really does run all 32 slots for the longest user's target.")
    md.append("")
    md.append(f"Trace: n={len(trace)}, target output total={target_out_total} tokens (mean={target_out_total/len(trace):.1f}/req)")
    md.append("")

    # Model static under Poisson rate=1: 64 prompts, batch=32, rate=1 → batch 1 fills at t=32
    # User i in batch 1 arrives ~t=i, waits ~32-i seconds, mean wait ~16s, plus ttft
    # batch 1 decode finishes at t = 32 + decode_wall_b1 ≈ 50
    # batch 2 fills at t=64, decode finishes ≈ t=82
    POIS_RATE = 1.0
    decode_wall_b1 = (static["per_batch"][0]["itl_ms"] / 1000.0) * static["per_batch"][0]["max_gen_tokens"]
    decode_wall_b2 = (static["per_batch"][1]["itl_ms"] / 1000.0) * static["per_batch"][1]["max_gen_tokens"]
    batch_fill_time = static["batch_size"] / POIS_RATE  # 32s
    static_pois_ttft_ms = []
    for i in range(static["batch_size"]):
        # batch 1: user arrives at t=i+1, waits (32 - i - 1) seconds, then ttft
        wait_s = batch_fill_time - (i + 1)
        static_pois_ttft_ms.append(wait_s * 1000 + (static["per_batch"][0]["ttft_ms"] or 0))
    for i in range(static["batch_size"]):
        # batch 2 user arrives at t=33+i, batch starts at t=64, waits 64-(33+i)=31-i
        wait_s = batch_fill_time - (i + 1)
        static_pois_ttft_ms.append(wait_s * 1000 + (static["per_batch"][1]["ttft_ms"] or 0))
    static_pois_wall = batch_fill_time + decode_wall_b1 + max(0, batch_fill_time - decode_wall_b1) + decode_wall_b2
    static_pois_goodput = target_out_total / static_pois_wall

    md.append("## Headline — apples-to-apples on the same vLLM server")
    md.append("")
    md.append("All rows go through the **same vLLM server / forward_decode / paged_*_cache**. Difference = client admission policy:")
    md.append("- **vLLM-CB**: client sends all 64 prompts at workload-arrival time → vLLM's CB admits 32 to running, 32 wait in vLLM's internal queue, slot frees → next admitted")
    md.append("- **vLLM-static**: client sends 32, **waits for ALL 32 to return** (batch-end barrier), then sends the next 32. vLLM internal scheduling identical, but client never lets a 33rd request reach vLLM until first 32 done.")
    md.append("")
    md.append("**TTFT here = user-perceived TTFT** (workload arrival time → first token). For burst all 64 prompts arrive at t=0; for Poisson they arrive at rate 1/s.")
    md.append("")
    md.append("| System | Wall (s) | Goodput (tok/s) | Mean TTFT | p99 TTFT |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| **vLLM-CB**, burst (1 bench × 64) | {burst['duration']:.1f} | {burst['output_throughput']:.1f} | **{burst['mean_ttft_ms']/1000:.2f}s** | **{burst['p99_ttft_ms']/1000:.2f}s** |")
    if vllm_static:
        md.append(f"| **vLLM-static**, burst (2 benches × 32) | {vllm_static['wall_s']:.1f} | {vllm_static['goodput']:.1f} | {vllm_static['mean_ttft_ms']/1000:.2f}s | {vllm_static['p99_ttft_ms']/1000:.2f}s |")
    md.append(f"| **vLLM-CB**, Poisson 1/s | {poisson['duration']:.1f} | {poisson['output_throughput']:.1f} | **{poisson['mean_ttft_ms']/1000:.2f}s** | **{poisson['p99_ttft_ms']/1000:.2f}s** |")
    if vllm_static:
        # Static under Poisson: client-side barrier still applies. Batch 1 fills via Poisson over 32s,
        # then runs (≈ b1.duration). Batch 2 fills via Poisson during b1's wait + b1.duration if needed.
        # Use queueing model: per-user wait in Poisson rate=1 batch fill ≈ (32 - i + 1)s; mean wait ≈ 16s.
        POIS_RATE = 1.0
        batch_fill_time = 32 / POIS_RATE
        vs_pois_ttft_ms = []
        for i in range(32):
            wait_s = batch_fill_time - (i + 1)
            vs_pois_ttft_ms.append(wait_s * 1000 + vllm_static["b1"]["mean_ttft_ms"])
        # Batch 2 starts after batch 1 finishes OR after batch 2 fills (whichever later)
        b1_done_t = batch_fill_time + vllm_static["b1"]["duration"]
        b2_filled_t = 64.0
        b2_start_t = max(b1_done_t, b2_filled_t)
        for i in range(32):
            arrival_t = 32 + i + 1
            wait_s = b2_start_t - arrival_t
            vs_pois_ttft_ms.append(wait_s * 1000 + vllm_static["b2"]["mean_ttft_ms"])
        vs_pois_wall = b2_start_t + vllm_static["b2"]["duration"]
        vs_pois_goodput = (vllm_static["b1"]["total_output_tokens"] + vllm_static["b2"]["total_output_tokens"]) / vs_pois_wall
        md.append(f"| **vLLM-static** (model), Poisson 1/s | {vs_pois_wall:.1f} | {vs_pois_goodput:.1f} | {statistics.mean(vs_pois_ttft_ms)/1000:.2f}s | {percentile(vs_pois_ttft_ms, 99)/1000:.2f}s |")
    if static is not None:
        md.append(f"| _legacy_ static (simple_text_demo), burst | {fair_static_wall_s:.1f} | {static_goodput:.1f} | {statistics.mean(fair_ttft_ms)/1000:.2f}s | {percentile(fair_ttft_ms, 99)/1000:.2f}s |")
    md.append("")

    if vllm_static:
        md.append(f"**Burst user_TTFT mean**: vLLM-static {vllm_static['mean_ttft_ms']/1000:.2f}s vs CB {burst['mean_ttft_ms']/1000:.2f}s → CB **{vllm_static['mean_ttft_ms'] / burst['mean_ttft_ms']:.2f}×** faster")
        md.append(f"**Burst user_TTFT p99**: CB **{vllm_static['p99_ttft_ms'] / burst['p99_ttft_ms']:.2f}×** faster")
        md.append(f"**Burst goodput**: CB {burst['output_throughput'] / vllm_static['goodput']:.2f}×")
        md.append(f"**Poisson user_TTFT mean**: vLLM-static {statistics.mean(vs_pois_ttft_ms)/1000:.2f}s vs CB {poisson['mean_ttft_ms']/1000:.2f}s → CB **{statistics.mean(vs_pois_ttft_ms) / poisson['mean_ttft_ms']:.2f}×** faster")
        md.append(f"**Poisson user_TTFT p99**: CB **{percentile(vs_pois_ttft_ms, 99) / poisson['p99_ttft_ms']:.2f}×** faster")
        md.append("")
        md.append("## Decomposition — where does the TTFT win actually come from?")
        md.append("")
        md.append("`user_ttft = admission_delay + server_ttft`")
        md.append("- `admission_delay` = time the client holds the prompt before sending HTTP (= 0 for CB; for static = wait for batch fill / wait for previous wave to complete)")
        md.append("- `server_ttft` = time from HTTP send → first token (= what `vllm bench serve` directly measures, since its clock starts at HTTP send)")
        md.append("")
        md.append("Each bench's reported `mean_ttft_ms` IS server_ttft. Subtracting gives admission_delay:")
        md.append("")
        md.append("| Mode | `server_ttft` (mean) | `admission_delay` (mean) | `user_ttft` (mean) |")
        md.append("|---|---:|---:|---:|")
        md.append(f"| CB burst (64 sent at t=0) | **{burst['mean_ttft_ms']/1000:.2f}s** | 0s | {burst['mean_ttft_ms']/1000:.2f}s |")
        # static burst: avg of b1 + b2 server_ttft, weighted; admission for b2 = b1.duration
        ssbb_avg = (vllm_static["b1"]["mean_ttft_ms"] + vllm_static["b2"]["mean_ttft_ms"]) / 2 / 1000
        admission_static_burst = vllm_static["b1"]["duration"] / 2  # half the users (b2) wait b1.duration
        md.append(f"| static burst (waves of 32) | **{ssbb_avg:.2f}s** | {admission_static_burst:.2f}s | {vllm_static['mean_ttft_ms']/1000:.2f}s |")
        md.append(f"| CB Poisson 1/s | **{poisson['mean_ttft_ms']/1000:.2f}s** | 0s | {poisson['mean_ttft_ms']/1000:.2f}s |")
        # static Poisson: avg b1+b2 server_ttft + mean batch-fill wait
        admission_static_pois = (statistics.mean(vs_pois_ttft_ms) - ssbb_avg * 1000) / 1000
        md.append(f"| static Poisson (model) | **{ssbb_avg:.2f}s** | ~{admission_static_pois:.2f}s | {statistics.mean(vs_pois_ttft_ms)/1000:.2f}s |")
        md.append("")
        md.append("**Key observation**: `server_ttft` for CB ({:.2f}s) is actually *higher* than for static ({:.2f}s) — CB holds 64 prompts in vLLM's queue, prefill backlog is twice as deep. So vLLM's internals are *not* faster in CB.".format(burst['mean_ttft_ms']/1000, ssbb_avg))
        md.append("")
        md.append("**The entire user_ttft win comes from `admission_delay = 0`** — CB lets the client hand off the prompt immediately, eliminating both batch-fill delay (Poisson) and barrier-between-waves delay (burst).")
        md.append("")
        md.append("**Goodput 1.30×**: this *is* a real CB advantage at the chip level — between waves, static-imitation has chip-idle time (last user of b1 still decoding while bench-2 hasn't started yet), CB has no such gap because new requests pipeline in continuously.")
        md.append("")

    md.append("## Why CB wins TTFT (qualitative)")
    md.append("")
    md.append("Same vLLM, same kernels — the only difference is **client policy**. In vLLM-CB the client sends all 64 prompts to vLLM at once, vLLM admits 32 to running and queues 32 in waiting. As soon as a slot frees (a user finishes its target), the next queued request prefills.")
    md.append("")
    md.append("In vLLM-static-imitation the client sends 32, **waits for ALL 32 to come back** (batch-end barrier), then sends the next 32. Even though vLLM internally would admit new requests, none are sent → effective static batching. Batch 2 users see TTFT = batch 1 wall + their own prefill.")
    md.append("")
    if vllm_static:
        md.append(f"Per-batch raw: b1.duration = {vllm_static['b1']['duration']:.1f}s, b1.mean_ttft = {vllm_static['b1']['mean_ttft_ms']/1000:.2f}s; b2.duration = {vllm_static['b2']['duration']:.1f}s, b2.mean_ttft = {vllm_static['b2']['mean_ttft_ms']/1000:.2f}s")
        md.append("")

    REPORT.write_text("\n".join(md) + "\n")
    print(f"wrote {REPORT}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))

        def cdf(xs, label, ls="-"):
            xs = sorted(xs)
            ys = [(i + 1) / len(xs) for i in range(len(xs))]
            ax.plot(xs, ys, label=label, linestyle=ls, drawstyle="steps-post")

        if vllm_static:
            cdf(vs_ttft_ms_list, "vLLM-static, burst (per-request TTFT)")
            ax.axvline(vllm_static["mean_ttft_ms"], color="C0", linestyle="--", label=f"vLLM-static burst mean ({vllm_static['mean_ttft_ms']/1000:.2f}s)")
        ax.axvline(burst["mean_ttft_ms"], color="C1", linestyle="--", label=f"vLLM-CB burst mean ({burst['mean_ttft_ms']/1000:.2f}s)")
        ax.axvline(poisson["mean_ttft_ms"], color="C2", linestyle="--", label=f"vLLM-CB Poisson 1/s mean ({poisson['mean_ttft_ms']/1000:.2f}s)")
        ax.set_xscale("log")
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("TTFT: heterogeneous output trace, V2")
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
