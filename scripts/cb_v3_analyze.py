#!/usr/bin/env python3
"""Analyze CB v3 experiment outputs.

Ingests:
  - /tmp/cb_v3_<EXP_ID>_burst.json (new v3 runs)
  - /tmp/cb_v2_burst.json (v2 baseline, reused as A2)

Emits:
  - /tmp/cb_v3_summary.csv
  - /tmp/cb_v3_plots/A_burst_size_vs_p99itl.png
  - /tmp/cb_v3_plots/B_out_heterogeneity_vs_goodput.png
  - /tmp/cb_v3_plots/B_out_heterogeneity_vs_ttft.png
  - /tmp/cb_v3_plots/C_input_len_vs_goodput.png
  - /tmp/cb_v3_plots/D_mmbt_pareto.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Experiment metadata (must match run_cb_v3.sh definitions)
EXPERIMENTS = {
    "A1": {"exp": "A", "n": 32,  "in_spread": "{16,48,96}",       "out_spread": "24x (115/24)", "mml": 1024, "mmbt": "default"},
    "A2": {"exp": "A", "n": 64,  "in_spread": "{16,48,96}",       "out_spread": "24x",          "mml": 1024, "mmbt": "default"},
    "A3": {"exp": "A", "n": 128, "in_spread": "{16,48,96}",       "out_spread": "24x",          "mml": 1024, "mmbt": "default"},
    "A4": {"exp": "A", "n": 256, "in_spread": "{16,48,96}",       "out_spread": "24x",          "mml": 1024, "mmbt": "default"},
    "B1": {"exp": "B", "n": 64,  "in_spread": "{16,48,96}",       "out_spread": "homogeneous (256)",     "mml": 1024, "mmbt": "default", "out_max_div_min": 1.0},
    "B2": {"exp": "B", "n": 64,  "in_spread": "{16,48,96}",       "out_spread": "4x (128-512)",          "mml": 1024, "mmbt": "default", "out_max_div_min": 4.0},
    "A2_as_B3": {"exp": "B", "n": 64, "in_spread": "{16,48,96}",  "out_spread": "24x (16-384)",          "mml": 1024, "mmbt": "default", "out_max_div_min": 24.0},
    "B4": {"exp": "B", "n": 64,  "in_spread": "{16,48,96}",       "out_spread": "64x (16-1024)",         "mml": 1152, "mmbt": "default", "out_max_div_min": 64.0},
    "C2": {"exp": "C", "n": 64,  "in_spread": "{256,512,1024}",   "out_spread": "homogeneous (256)",     "mml": 2048, "mmbt": "default", "in_mean": 597},
    "C3": {"exp": "C", "n": 64,  "in_spread": "{2048,4096}",      "out_spread": "homogeneous (256)",     "mml": 8192, "mmbt": "default", "in_mean": 3072},
    "D1": {"exp": "D", "n": 64,  "in_spread": "{16,48,96}",       "out_spread": "24x", "mml": 1024, "mmbt": 2048},
    "D2": {"exp": "D", "n": 64,  "in_spread": "{16,48,96}",       "out_spread": "24x", "mml": 1024, "mmbt": 4096},
    "D3": {"exp": "D", "n": 64,  "in_spread": "{16,48,96}",       "out_spread": "24x", "mml": 1024, "mmbt": 8192},
    "D4": {"exp": "D", "n": 64,  "in_spread": "{16,48,96}",       "out_spread": "24x", "mml": 1024, "mmbt": 16384},
    "A2_as_D5": {"exp": "D", "n": 64, "in_spread": "{16,48,96}",  "out_spread": "24x", "mml": 1024, "mmbt": "default"},
}

METRICS = [
    "duration", "completed", "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "p99_e2el_ms",
]


def load_run(json_path: Path):
    if not json_path.exists() or json_path.stat().st_size == 0:
        return None
    with json_path.open() as f:
        return json.load(f)


def collect():
    rows = []
    out_dir = Path("/tmp")
    v2_burst = out_dir / "cb_v2_burst.json"

    for exp_id, meta in EXPERIMENTS.items():
        if exp_id.startswith("A2_as"):
            data = load_run(v2_burst)
            display_id = exp_id.replace("A2_as_", "") + " (=v2_burst)"
        elif exp_id == "A2":
            data = load_run(v2_burst) or load_run(out_dir / f"cb_v3_{exp_id}_burst.json")
            display_id = "A2"
        else:
            data = load_run(out_dir / f"cb_v3_{exp_id}_burst.json")
            display_id = exp_id
        if data is None:
            continue
        row = {"exp_id": display_id, **meta}
        for k in METRICS:
            row[k] = data.get(k)
        rows.append(row)
    return rows


def write_csv(rows, path: Path):
    if not rows:
        print("no rows; skipping CSV", file=sys.stderr)
        return
    keys = list(rows[0].keys())
    # ensure stable column order even if later rows have different keysets
    extra = set()
    for r in rows:
        extra |= set(r.keys()) - set(keys)
    keys += sorted(extra)
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)")


def plot_A(rows, out_dir: Path):
    a_rows = [r for r in rows if r["exp"] == "A"]
    if len(a_rows) < 2:
        return
    a_rows.sort(key=lambda r: r["n"])
    xs = [r["n"] for r in a_rows]
    p99_itl = [r["p99_itl_ms"] for r in a_rows]
    median_itl = [r["median_itl_ms"] for r in a_rows]
    p99_ttft = [r["p99_ttft_ms"] for r in a_rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(xs, p99_itl, "o-", label="p99 ITL")
    ax1.plot(xs, median_itl, "s--", label="median ITL")
    ax1.set_xlabel("num_prompts (admit-count proxy = N − 32)")
    ax1.set_ylabel("Inter-token latency (ms)")
    ax1.set_title("(A) Phase-separation tax vs burst size")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(xs, p99_ttft, "o-", color="C2", label="p99 TTFT")
    ax2.set_xlabel("num_prompts")
    ax2.set_ylabel("TTFT (ms)")
    ax2.set_title("(A) TTFT vs burst size")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "A_burst_size_vs_p99itl.png", dpi=120)
    plt.close(fig)
    print(f"  wrote A plot ({len(xs)} points)")


def plot_B(rows, out_dir: Path):
    b_rows = [r for r in rows if r["exp"] == "B" and r.get("out_max_div_min") is not None]
    if len(b_rows) < 2:
        return
    b_rows.sort(key=lambda r: r["out_max_div_min"])
    xs = [r["out_max_div_min"] for r in b_rows]
    goodput = [r["request_throughput"] for r in b_rows]
    mean_ttft = [r["mean_ttft_ms"] for r in b_rows]
    p99_ttft = [r["p99_ttft_ms"] for r in b_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, goodput, "o-", color="C0")
    ax.set_xlabel("Output length spread (max/min)")
    ax.set_ylabel("Goodput (req/s)")
    ax.set_xscale("log")
    ax.set_title("(B) CB chip-level Goodput vs output heterogeneity")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "B_out_heterogeneity_vs_goodput.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, mean_ttft, "o-", label="mean TTFT")
    ax.plot(xs, p99_ttft, "s--", label="p99 TTFT")
    ax.set_xlabel("Output length spread (max/min)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_xscale("log")
    ax.set_title("(B) CB user-perceived TTFT vs output heterogeneity")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "B_out_heterogeneity_vs_ttft.png", dpi=120)
    plt.close(fig)
    print(f"  wrote B plots ({len(xs)} points)")


def plot_C(rows, out_dir: Path):
    c_rows = [r for r in rows if r["exp"] == "C" and r.get("in_mean") is not None]
    # include B1 (homogeneous 256/256) as the "short input" anchor for C
    b1 = next((r for r in rows if r["exp_id"] == "B1"), None)
    points = list(c_rows)
    if b1 is not None:
        b1_copy = dict(b1); b1_copy["in_mean"] = 53  # ~mean of {16,48,96}
        points.append(b1_copy)
    points = [p for p in points if p.get("in_mean") is not None]
    if len(points) < 2:
        return
    points.sort(key=lambda r: r["in_mean"])
    xs = [r["in_mean"] for r in points]
    goodput = [r["request_throughput"] for r in points]
    tput = [r["output_throughput"] for r in points]
    p99_ttft = [r["p99_ttft_ms"] for r in points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(xs, goodput, "o-", label="req/s")
    ax1b = ax1.twinx()
    ax1b.plot(xs, tput, "s--", color="C1", label="tok/s")
    ax1.set_xlabel("Mean input length (tokens)")
    ax1.set_ylabel("Goodput (req/s)", color="C0")
    ax1b.set_ylabel("Output throughput (tok/s)", color="C1")
    ax1.set_xscale("log")
    ax1.set_title("(C) Goodput vs input length")
    ax1.grid(alpha=0.3, which="both")
    ax2.plot(xs, p99_ttft, "o-", color="C2")
    ax2.set_xlabel("Mean input length (tokens)")
    ax2.set_ylabel("p99 TTFT (ms)")
    ax2.set_xscale("log")
    ax2.set_title("(C) p99 TTFT vs input length")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "C_input_len_vs_goodput.png", dpi=120)
    plt.close(fig)
    print(f"  wrote C plot ({len(xs)} points)")


def plot_D(rows, out_dir: Path):
    d_rows = [r for r in rows if r["exp"] == "D"]
    if len(d_rows) < 2:
        return

    def mmbt_key(r):
        v = r.get("mmbt", "default")
        return 1_000_000 if v == "default" else int(v)

    d_rows.sort(key=mmbt_key)
    labels = [str(r["mmbt"]) for r in d_rows]
    p99_itl = [r["p99_itl_ms"] for r in d_rows]
    mean_ttft = [r["mean_ttft_ms"] for r in d_rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_ttft, p99_itl, s=80)
    for i, lbl in enumerate(labels):
        ax.annotate(f"MMBT={lbl}", (mean_ttft[i], p99_itl[i]),
                    xytext=(7, 7), textcoords="offset points")
    ax.set_xlabel("mean TTFT (ms)")
    ax.set_ylabel("p99 ITL (ms)")
    ax.set_title("(D) Admission cadence Pareto: TTFT vs ITL tax")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "D_mmbt_pareto.png", dpi=120)
    plt.close(fig)
    print(f"  wrote D plot ({len(labels)} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", type=Path, default=Path("/tmp/cb_v3_summary.csv"))
    ap.add_argument("--plots-dir", type=Path, default=Path("/tmp/cb_v3_plots"))
    ap.add_argument("--partial", action="store_true",
                    help="emit plots even if some experiments missing (default behavior)")
    args = ap.parse_args()

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    rows = collect()
    print(f"loaded {len(rows)} experiment rows")
    for r in rows:
        print(f"  {r['exp_id']:12s} dur={r.get('duration'):7.1f}s "
              f"thr={r.get('request_throughput'):.2f} req/s  "
              f"p99_itl={r.get('p99_itl_ms'):7.1f}ms  "
              f"p99_ttft={r.get('p99_ttft_ms'):7.1f}ms")

    write_csv(rows, args.out_csv)
    plot_A(rows, args.plots_dir)
    plot_B(rows, args.plots_dir)
    plot_C(rows, args.plots_dir)
    plot_D(rows, args.plots_dir)


if __name__ == "__main__":
    sys.exit(main())
