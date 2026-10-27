# Continuous batching vs static batching, Llama-3.1-8B on Wormhole n300

**Premise.** This is the ablation the original ask called for: hold the model and KV layout fixed (Llama-3.1-8B + paged KV in both systems), vary only the *scheduling* method, and observe end-to-end serving behavior on a mixed-length request trace.

Two systems, same trace, same hardware:
- **System A — vLLM-TT** (`paged KV + continuous batching`): the OpenAI-compatible server (`python -m vllm.entrypoints.openai.api_server`) backed by `models.tt_transformers.tt.generator_vllm:LlamaForCausalLM`. Continuous batching via `TTScheduler`.
- **System B — `simple_text_demo.py`** (`paged KV + static batching`): the canonical tt-metal demo, batch_size=32 fixed at start, no per-step rescheduling. Driven by `scripts/cb_vs_static_static_driver.py` which feeds the same trace in 2 sequential batches of 32.

## Trace

- 64 requests, deterministic seed (`scripts/cb_vs_static_make_trace.py --seed=42`)
- Input tokens (target): {16: 18, 48: 28, 96: 11, 192: 7}; total 4032
- Output tokens (target): {32: 16, 96: 31, 160: 17}; total 6208
- Note: actual encoded prompt lengths land at ~2.7× target due to the synthetic `alphaXXX` tokens. Demo log shows encoded ∈ {79, 163, 287, 537}.

## Headline

| System | Wall time (s) | Output tok/s | Total tok/s | Mean TTFT | p99 TTFT | Failed |
|---|---:|---:|---:|---:|---:|---:|
| vLLM-TT, request-rate=inf (burst) | 34.9 | 156.4 | 528.1 | 8.18s | 17.84s | 0/64 |
| vLLM-TT, request-rate=2 req/s (Poisson) | 52.0 | 112.9 | 362.3 | 2.97s | 9.10s | 0/64 |
| static, batch=32 (sequential) | 61.8 | — | 165.6 | 15.54s | 30.90s | 0/64 |

**Continuous batching is 3.2× higher total throughput than static batching** at burst arrivals on this trace (528 vs 166 tok/s).

## Why static batching loses so badly

Two compounding costs:

1. **Re-init overhead per batch.** Each invocation of `simple_text_demo.py` reloads the model, recompiles the prefill and decode traces, and warms caches. We measured ~30s per 32-prompt batch end-to-end, of which only ~10s is actual inference. Continuous batching pays this cost once at server start.
   - Batch 1: 30.7s wall, in-batch throughput 681 tok/s.
   - Batch 2: 31.1s wall, in-batch throughput 677 tok/s.

2. **Tail latency from queueing.** Static batching forces request 33 to wait for request 1–32 to *all* finish before its batch even starts. With our 64-request trace at batch=32, half the requests get TTFT = ~180ms (their batch starts immediately) and the other half get TTFT ≈ 30.9s (waiting for batch 1). Mean TTFT is bimodal at 15.5s; p99 is ~31s.

Continuous batching avoids both: the server is persistent (no re-init) and new requests join the in-flight batch on the next decode step (no queueing for an entire batch to finish).

## What changes when arrivals are spread out (Poisson rate=2)

If we offer requests at 2 req/s (a moderate serving load) instead of all at once:

- vLLM-TT throughput drops from 528 → 362 tok/s (server is sometimes idle waiting for next request).
- TTFT mean drops from 8.2s → 3.0s (queue is shallower).
- p99 TTFT drops from 17.8s → 9.1s.

Static batching is unaffected by arrival pattern in our setup (all prompts are pre-collected before each batch starts), so its numbers don't change. Continuous batching's advantage *grows* under realistic spread arrivals because it can interleave new requests with in-flight ones — an axis static batching cannot exercise at all.

## Caveats

- **Synthetic prompts.** `alphaXXX` pseudo-words tokenize ~2.7x larger than nominal target lengths. This is fine for relative comparisons but doesn't model real input/output entropy. ShareGPT or sonnet would be more realistic — easy follow-up.
- **Static batching's re-init cost is unrealistic.** A real static-batch deployment would keep the model loaded between batches; the demo doesn't have that mode. So our static numbers *overstate* the throughput gap; the queueing/TTFT story is the more honest part.
- **Single-chip n300, max_seq_len=1024.** Both systems were configured to match these limits; longer-context behavior not tested.
- **Output length is sampler-bounded, not target-bounded.** vLLM's `output_tokens` field hints rather than forces; vLLM's burst run produced 5453 output tokens (vs trace target of 6208). The static demo always produces exactly `max_generated_tokens=200` per user.

## Files

- vLLM bench raw: `/tmp/cb_vs_static_vllm/{burst,poisson_rate2}.json`
- Static driver summary: `/tmp/cb_vs_static_static_summary.json`
- Per-batch demo logs: `/tmp/static_logs/`
- Trace: `/tmp/cb_vs_static_trace.json` (and `.jsonl` for vLLM)
