#!/usr/bin/env python3
"""Prefix-sharing dataset for paged-attention showcase.

Every prompt = <SHARED ~8K-token system prompt> + <unique 256-token suffix>.
With `--enable-prefix-caching` on the vLLM server, blocks covering the shared
prefix are hashed and reused across users (`block_pool.py` ref-counting).
Without it, every user re-prefills the full prompt.

Output: JSONL with {"prompt": str, "output_tokens": int} per line, consumable
by `vllm bench serve --dataset-name custom --custom-output-len -1`.
"""

import argparse
import json
from transformers import AutoTokenizer


# Long deterministic filler. ~30 unique tokens per line; `*N` to extend.
FILLER_LINE = (
    "You are an AI assistant deployed at ACME Corporation. Your role is to "
    "help engineers debug distributed systems, write code, and analyze logs. "
    "Always cite specific function names and line numbers when relevant. "
)

UNIQUE_TEMPLATE = (
    "User question #{i}: please describe in detail step number {i} of the "
    "standard onboarding procedure, including any caveats specific to "
    "configuration profile {i}. Provide concrete examples. "
)


def trim_to_tokens(tok, text: str, n: int) -> str:
    ids = tok(text, add_special_tokens=False).input_ids[:n]
    return tok.decode(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="HF model path for tokenizer")
    p.add_argument("--out", required=True, help="output .jsonl path")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--shared-tokens", type=int, default=8192)
    p.add_argument("--unique-tokens", type=int, default=256)
    p.add_argument("--output-tokens", type=int, default=256)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Build shared prefix to exact token length.
    shared_source = FILLER_LINE * (args.shared_tokens // 8 + 16)
    shared_text = trim_to_tokens(tok, shared_source, args.shared_tokens)
    shared_re_tokens = len(tok(shared_text, add_special_tokens=False).input_ids)
    print(f"shared prefix: {shared_re_tokens} tokens (target {args.shared_tokens})")

    with open(args.out, "w") as f:
        for i in range(args.num_prompts):
            unique_source = UNIQUE_TEMPLATE.format(i=i) * 32
            unique_text = trim_to_tokens(tok, unique_source, args.unique_tokens)
            # Newline boundary keeps BPE merges from crossing shared/unique.
            prompt = shared_text + "\n\n" + unique_text
            f.write(json.dumps({"prompt": prompt, "output_tokens": args.output_tokens}) + "\n")

    print(f"wrote {args.num_prompts} prompts to {args.out}")
    print(f"  per-prompt = {args.shared_tokens} shared + ~{args.unique_tokens} unique")
    print(f"  expected output per prompt = {args.output_tokens} (use --ignore-eos)")


if __name__ == "__main__":
    main()
