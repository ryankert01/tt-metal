#!/usr/bin/env python3
"""V3 trace generator for CB gap-filling sweep.

Forked from cb_vs_static_make_trace_v2.py. Differences:
  - INPUT_LEN_BUCKETS / OUTPUT_LEN_BUCKETS now configurable via CLI
  - Drops b1/b2 static-batch splits (v3 is CB-only)

Bucket spec format: "len1:prob1,len2:prob2,..."  e.g. "16:0.4,48:0.4,96:0.2"
Probabilities must sum to ~1.0 (tolerance 1e-3).
"""

import argparse
import json
import random
import sys
from pathlib import Path


def parse_buckets(spec: str) -> list[tuple[int, float]]:
    buckets = []
    total = 0.0
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        length_s, prob_s = pair.split(":")
        length = int(length_s)
        prob = float(prob_s)
        buckets.append((length, prob))
        total += prob
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"bucket probabilities sum to {total}, expected 1.0 ± 1e-3")
    return buckets


def weighted_choice(buckets, rng):
    r = rng.random()
    cum = 0.0
    for v, p in buckets:
        cum += p
        if r <= cum:
            return v
    return buckets[-1][0]


_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import os
        from transformers import AutoTokenizer
        snaps = os.path.expanduser(
            "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
        )
        snap = sorted(os.listdir(snaps))[0]
        _TOKENIZER = AutoTokenizer.from_pretrained(snaps + snap)
    return _TOKENIZER


def make_prompt(target_tokens: int, rng, exact_tokens: bool = False) -> str:
    """Generate a synthetic prompt.

    When exact_tokens=False (legacy v2 behavior): generates ~target_tokens*1.3
    words; actual token count is ~2.6× target. Use for back-compat with v2 runs
    where the bucket labels are nominal "target word counts".

    When exact_tokens=True: tokenizes and trims to hit `target_tokens` real
    Llama-3.1 tokens (±1).
    """
    n_words = int(target_tokens * 1.3)
    if exact_tokens:
        n_words = int(target_tokens * 0.6)  # initial over-shoot starting point
    words = [f"alpha{rng.randint(0, 999)}" for _ in range(n_words)]
    prompt = "Question: " + " ".join(words) + " Answer:"
    if not exact_tokens:
        return prompt
    tok = _get_tokenizer()
    ids = tok(prompt).input_ids
    # Adjust word count until we hit target tokens (linear convergence)
    for _ in range(8):
        diff = target_tokens - len(ids)
        if abs(diff) <= 1:
            break
        # ~2 tokens per word for these synthetic strings
        delta_words = max(1, abs(diff) // 2) if diff < 0 else max(1, diff // 2)
        if diff > 0:
            words.extend(f"alpha{rng.randint(0, 999)}" for _ in range(delta_words))
        else:
            words = words[:-delta_words]
        prompt = "Question: " + " ".join(words) + " Answer:"
        ids = tok(prompt).input_ids
    # Final truncation to exact length if still over
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
        prompt = tok.decode(ids, skip_special_tokens=True)
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--in-buckets",
        type=str,
        default="16:0.4,48:0.4,96:0.2",
        help="Input length buckets, e.g. '16:0.4,48:0.4,96:0.2'",
    )
    ap.add_argument(
        "--out-buckets",
        type=str,
        default="16:0.4,64:0.3,256:0.2,384:0.1",
        help="Output length buckets",
    )
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--exact-tokens", action="store_true",
                    help="Generate prompts whose actual Llama-3.1 token count matches "
                         "the input bucket value (slower; needed for input-length sweeps).")
    args = ap.parse_args()

    in_buckets = parse_buckets(args.in_buckets)
    out_buckets = parse_buckets(args.out_buckets)

    rng = random.Random(args.seed)
    requests = []
    for i in range(args.n):
        in_t = weighted_choice(in_buckets, rng)
        out_t = weighted_choice(out_buckets, rng)
        requests.append({
            "id": i,
            "prompt": make_prompt(in_t, rng, exact_tokens=args.exact_tokens),
            "input_tokens": in_t,
            "output_tokens": out_t,
        })

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        for r in requests:
            f.write(json.dumps({"prompt": r["prompt"], "output_tokens": r["output_tokens"]}) + "\n")

    if args.out_json:
        args.out_json.write_text(json.dumps(requests, indent=2))

    from collections import Counter
    in_dist = Counter(r["input_tokens"] for r in requests)
    out_dist = Counter(r["output_tokens"] for r in requests)
    out_total = sum(r["output_tokens"] for r in requests)
    out_max = max(r["output_tokens"] for r in requests)
    out_mean = out_total / len(requests)
    print(f"wrote {len(requests)} requests -> {args.out_jsonl}")
    print(f"  in_dist:  {dict(sorted(in_dist.items()))}")
    print(f"  out_dist: {dict(sorted(out_dist.items()))}")
    print(f"  output total={out_total} mean={out_mean:.1f} max={out_max}")


if __name__ == "__main__":
    sys.exit(main())
