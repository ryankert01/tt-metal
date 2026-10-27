#!/usr/bin/env python3
"""Generate a deterministic mixed-length request trace for the cb-vs-static comparison.

Output: JSON list of {"prompt": str, "input_tokens": int, "output_tokens": int}.
"""

import argparse
import json
import random
import sys
from pathlib import Path


# Workload mix sized so encoded prompt + output fits demo max_seq_len=1024.
# Pseudo-words "alphaXXX" tokenize to ~3 tokens each, so encoded ≈ target * 2.7.
# Need encoded + max_output ≤ 1024 → target ≤ ~300.
INPUT_LEN_BUCKETS = [
    (16,  0.30),   # short query: encoded ~60 tokens
    (48,  0.40),   # typical question: encoded ~180
    (96,  0.20),   # paragraph: encoded ~360
    (192, 0.10),   # short doc: encoded ~720
]
OUTPUT_LEN_BUCKETS = [
    (32,  0.25),
    (96,  0.50),
    (160, 0.25),
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
    """Build a prompt that tokenizes to roughly target_tokens. We use space-separated
    short pseudo-words; the Llama-3 tokenizer averages ~0.75 tokens per such word.
    Slight over-shoot is fine — the demo will truncate or pad as needed."""
    n_words = int(target_tokens * 1.3)
    words = [f"alpha{rng.randint(0, 999)}" for _ in range(n_words)]
    return "Question: " + " ".join(words) + " Answer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="Number of requests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("/tmp/cb_vs_static_trace.json"))
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

    args.out.write_text(json.dumps(requests, indent=2))
    print(f"wrote {len(requests)} requests to {args.out}")
    # Summary
    from collections import Counter
    print(f"  input_tokens: {sorted(Counter(r['input_tokens'] for r in requests).items())}")
    print(f"  output_tokens: {sorted(Counter(r['output_tokens'] for r in requests).items())}")
    total_in = sum(r["input_tokens"] for r in requests)
    total_out = sum(r["output_tokens"] for r in requests)
    print(f"  total: {total_in} input tokens, {total_out} output tokens")


if __name__ == "__main__":
    sys.exit(main())
