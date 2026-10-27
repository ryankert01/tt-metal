#!/usr/bin/env python3
"""V2 trace for cb-vs-static: extreme output-length heterogeneity.

Each request gets its own output_tokens. Combined with vLLM `--ignore-eos`,
this makes per-request output binding — so CB really does release each slot
at the requested length, while static keeps the slot occupied until the
batch's longest user finishes.

Output formats:
  - JSON:  list of {"id", "prompt", "input_tokens", "output_tokens"}
  - JSONL: one {"prompt": ..., "output_tokens": N} per line (vLLM custom dataset)
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Encoded prompt + max_output ≤ 1024 (demo max_seq_len)
INPUT_LEN_BUCKETS = [
    (16,  0.40),   # encoded ~60
    (48,  0.40),   # encoded ~180
    (96,  0.20),   # encoded ~360
]

# Extreme heterogeneity — variance is the point.
# Mean ≈ 16*0.4 + 64*0.3 + 256*0.2 + 384*0.1 = 6.4 + 19.2 + 51.2 + 38.4 = 115
# Range: 16x between shortest and longest
OUTPUT_LEN_BUCKETS = [
    (16,  0.40),
    (64,  0.30),
    (256, 0.20),
    (384, 0.10),
]


def weighted_choice(buckets, rng):
    r = rng.random()
    cum = 0.0
    for v, p in buckets:
        cum += p
        if r <= cum:
            return v
    return buckets[-1][0]


def make_prompt(target_tokens: int, rng) -> str:
    n_words = int(target_tokens * 1.3)
    words = [f"alpha{rng.randint(0, 999)}" for _ in range(n_words)]
    return "Question: " + " ".join(words) + " Answer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=Path("/tmp/cb_vs_static_trace_v2.json"))
    ap.add_argument("--out-jsonl", type=Path, default=Path("/tmp/cb_vs_static_trace_v2.jsonl"))
    ap.add_argument("--out-b1-jsonl", type=Path, default=Path("/tmp/cb_v2_static_b1.jsonl"))
    ap.add_argument("--out-b2-jsonl", type=Path, default=Path("/tmp/cb_v2_static_b2.jsonl"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    requests = []
    for i in range(args.n):
        in_t = weighted_choice(INPUT_LEN_BUCKETS, rng)
        out_t = weighted_choice(OUTPUT_LEN_BUCKETS, rng)
        requests.append({
            "id": i,
            "prompt": make_prompt(in_t, rng),
            "input_tokens": in_t,
            "output_tokens": out_t,
        })

    args.out_json.write_text(json.dumps(requests, indent=2))

    def _write_jsonl(path, items):
        with path.open("w") as f:
            for r in items:
                f.write(json.dumps({"prompt": r["prompt"], "output_tokens": r["output_tokens"]}) + "\n")

    _write_jsonl(args.out_jsonl, requests)
    bs = args.batch_size
    _write_jsonl(args.out_b1_jsonl, requests[:bs])
    _write_jsonl(args.out_b2_jsonl, requests[bs:bs * 2])

    from collections import Counter
    in_dist = Counter(r["input_tokens"] for r in requests)
    out_dist = Counter(r["output_tokens"] for r in requests)
    out_total = sum(r["output_tokens"] for r in requests)
    out_max = max(r["output_tokens"] for r in requests)
    out_mean = out_total / len(requests)
    print(f"wrote {len(requests)} requests")
    print(f"  input_tokens dist: {dict(sorted(in_dist.items()))}")
    print(f"  output_tokens dist: {dict(sorted(out_dist.items()))}")
    print(f"  output_tokens: total={out_total} mean={out_mean:.1f} max={out_max}")
    print(f"  -> JSON  {args.out_json}")
    print(f"  -> JSONL {args.out_jsonl} (full)")
    print(f"  -> JSONL {args.out_b1_jsonl} (batch 1, n={bs})")
    print(f"  -> JSONL {args.out_b2_jsonl} (batch 2, n={bs})")


if __name__ == "__main__":
    sys.exit(main())
