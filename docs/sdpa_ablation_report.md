# Paged SDPA Decode Sensitivity Study: Llama-3.1-8B on Wormhole n300

**Scope.** This is *not* a flash-on/flash-off ablation — that comparison cannot be made on tt-metal because no "non-flash" SDPA decode op ships. The only `paged_scaled_dot_product_attention_decode` is the FlashDecode kernel described in [`tech_reports/FlashAttention/FlashDecode.md`](../tech_reports/FlashAttention/FlashDecode.md) (a generative-decode-tuned variant of the FlashAttention-2 algorithm in [`tech_reports/FlashAttention/FlashAttention.md`](../tech_reports/FlashAttention/FlashAttention.md)).

Instead, this study sweeps the two performance hyperparameters of `SDPAProgramConfig` that the FlashDecode tech report explicitly says "should be swept for a given input size":
- `k_chunk_size` — controls how K is tiled inside the streaming softmax
- `max_cores_per_head_batch` — caps cores assigned per (batch, kv_head) work, mitigating NoC traffic in the reduction step

Both default to values picked by `model_config.py:get_attn_sdpa_decode_program_config` (`k_chunk_size=0` → kernel-auto; `max_cores_per_head_batch=16`).

## TL;DR

- **Largest within-FlashDecode speedup of auto-tile vs. coarsest legal k_chunk:** 1.78x at batch=1, seqlen=512 (vs k_chunk=512).
- **Auto-tile is *worse* than a fixed k_chunk in 3 of 16 (batch, seqlen) cells**, by up to 26%. The heuristic in `models/tt_transformers/tt/model_config.py:1486` is suboptimal in the transitional parallelization regime (b∈{8,16} on n300 single-chip, where bsz·n_kv_heads is comparable to core count).
- **`max_cores_per_head_batch` matters most at low batch + long context** — at b=1 s=8192, cap=1 is **2.6x slower** than the default cap=16. NoC reduction cost dominates when too few cores share work; reduction-fanout overhead dominates when too many do. The default 16 sits at the U-curve minimum for this shape (matches FlashDecode tech report §4.2).
- **Why the b=32 results look uniform across k_chunk:** at b=32, n_kv_heads=8 → 256 works ≫ 64 cores, so each core handles ≥4 entire works with no inter-core reduction (FlashDecode §3.1, "extreme case 1"). The kernel becomes memory-bound in a regime where K-iteration granularity barely matters — *not* because the override is being ignored.
- **L1 capacity caps `k_chunk_size` at ≤1024** at every (b, s) cell tested. We never measured `k_chunk = seqlen` at long contexts, so the largest "streaming step" in our sweep is always still flash-style.
- **End-to-end (batch-32) throughput** drops 9.0% going k_chunk 0 → 1024 (762.02 → 693.39 tok/s). Even in the b=32 regime, large k_chunk does have a measurable e2e cost.

## Setup

- Op: `ttnn.transformer.paged_scaled_dot_product_attention_decode`
- Shapes: Llama-3.1-8B GQA (n_q_heads=32, n_kv_heads=8, head_dim=128), bfloat8_b KV / bfloat16 Q
- Hardware: Wormhole n300, single-chip path (8×8 = 64 cores)
- Constraint: kernel asserts non-zero `k_chunk_size` is a power of 2 AND multiple of 32
- PERF.md baseline (default config): 44.2 tok/s, TTFT=67.0 ms

## k_chunk_size sweep (op level)

Headline table: `k_chunk=0` (kernel-auto, the default) vs the largest k_chunk that fits in L1 at each (batch, seqlen):

| batch | seqlen | auto p50 (µs) | largest-k p50 (µs) | largest-k | ratio |
|------:|-------:|--------------:|-------------------:|----------:|------:|
| 1 | 512 | 61.9 | 110.1 | 512 | 1.78x |
| 1 | 2048 | 108.3 | 110.4 | 1024 | 1.02x |
| 1 | 8192 | 181.1 | 209.0 | 1024 | 1.15x |
| 1 | 32768 | 456.9 | 455.1 | 1024 | 1.00x |
| 8 | 512 | 127.0 | 93.5 | 512 | 0.74x |
| 8 | 2048 | 256.2 | 222.8 | 1024 | 0.87x |
| 8 | 8192 | 725.9 | 755.3 | 1024 | 1.04x |
| 8 | 32768 | 2854.9 | 2862.8 | 1024 | 1.00x |
| 16 | 512 | 137.7 | 162.4 | 512 | 1.18x |
| 16 | 2048 | 440.5 | 413.0 | 1024 | 0.94x |
| 16 | 8192 | 1408.5 | 1448.3 | 1024 | 1.03x |
| 16 | 32768 | 5513.3 | 5626.1 | 1024 | 1.02x |
| 32 | 512 | 224.7 | 225.0 | 512 | 1.00x |
| 32 | 2048 | 731.4 | 733.4 | 1024 | 1.00x |
| 32 | 8192 | 2855.2 | 2859.9 | 1024 | 1.00x |
| 32 | 32768 | 10990.3 | 10994.9 | 1024 | 1.00x |

Ratio >1.0 = auto is faster (good); ratio <1.0 = a *worse* k_chunk gives a better number (heuristic is leaving performance on the table).

Per-(batch, seqlen) full curves:
- `sdpa_ablation_plots/oplevel_b1_s512.png`
- `sdpa_ablation_plots/oplevel_b1_s2048.png`
- `sdpa_ablation_plots/oplevel_b1_s8192.png`
- `sdpa_ablation_plots/oplevel_b1_s32768.png`
- `sdpa_ablation_plots/oplevel_b8_s512.png`
- `sdpa_ablation_plots/oplevel_b8_s2048.png`
- `sdpa_ablation_plots/oplevel_b8_s8192.png`
- `sdpa_ablation_plots/oplevel_b8_s32768.png`
- `sdpa_ablation_plots/oplevel_b16_s512.png`
- `sdpa_ablation_plots/oplevel_b16_s2048.png`
- `sdpa_ablation_plots/oplevel_b16_s8192.png`
- `sdpa_ablation_plots/oplevel_b16_s32768.png`
- `sdpa_ablation_plots/oplevel_b32_s512.png`
- `sdpa_ablation_plots/oplevel_b32_s2048.png`
- `sdpa_ablation_plots/oplevel_b32_s8192.png`
- `sdpa_ablation_plots/oplevel_b32_s32768.png`

## max_cores_per_head_batch sweep (op level)

Spot check at the operating points where, per FlashDecode §3.1 "extreme case 2," too many cores per work cause the NoC reduction to dominate. Default cap = 16.

| batch | seqlen | cap | p50 (µs) | vs cap=16 |
|------:|-------:|----:|---------:|----------:|
| 1 | 512 | 1 | 72.9 | 0.88x |
| 1 | 512 | 2 | 72.1 | 0.87x |
| 1 | 512 | 4 | 76.1 | 0.92x |
| 1 | 512 | 8 | 81.7 | 0.99x |
| 1 | 512 | 16 | 82.6 | 1.00x |
| 1 | 512 | 32 | 61.5 | 0.74x |
| 1 | 512 | 64 | 89.2 | 1.08x |
| 1 | 8192 | 1 | 395.1 | 2.62x |
| 1 | 8192 | 2 | 221.0 | 1.46x |
| 1 | 8192 | 4 | 168.2 | 1.11x |
| 1 | 8192 | 8 | 160.0 | 1.06x |
| 1 | 8192 | 16 | 151.1 | 1.00x |
| 1 | 8192 | 32 | 172.1 | 1.14x |
| 1 | 8192 | 64 | 154.4 | 1.02x |
| 8 | 512 | 1 | 89.3 | 0.98x |
| 8 | 512 | 2 | 113.4 | 1.25x |
| 8 | 512 | 4 | 90.0 | 0.99x |
| 8 | 512 | 8 | 91.2 | 1.00x |
| 8 | 512 | 16 | 90.9 | 1.00x |
| 8 | 512 | 32 | 113.6 | 1.25x |
| 8 | 512 | 64 | 90.3 | 0.99x |
| 8 | 2048 | 1 | 218.3 | 0.99x |
| 8 | 2048 | 2 | 219.8 | 1.00x |
| 8 | 2048 | 4 | 220.5 | 1.00x |
| 8 | 2048 | 8 | 217.7 | 0.99x |
| 8 | 2048 | 16 | 220.0 | 1.00x |
| 8 | 2048 | 32 | 252.2 | 1.15x |
| 8 | 2048 | 64 | 225.3 | 1.02x |

- `sdpa_ablation_plots/max_cores_b1_s512.png`
- `sdpa_ablation_plots/max_cores_b1_s8192.png`
- `sdpa_ablation_plots/max_cores_b8_s512.png`
- `sdpa_ablation_plots/max_cores_b8_s2048.png`

## End-to-end demo (full prefill+decode, max_seq_len=1024)

`HF_MODEL=Llama-3.1-8B-Instruct`, `pytest models/tt_transformers/demo/simple_text_demo.py -k 'performance and {selector}'` with `TT_SDPA_K_CHUNK_DECODE` env-var override.

| k_chunk | selector | TTFT (ms) | ITL (ms) | tok/s/user | tok/s |
|--------:|:---------|----------:|---------:|-----------:|------:|
| 0 | batch-1 | 92.78 | 40.13 | 24.92 | 24.92 |
| 256 | batch-1 | 88.21 | 40.21 | 24.87 | 24.87 |
| 1024 | batch-1 | 92.35 | 40.89 | 24.45 | 24.45 |
| 0 | batch-32 | 25.21 | 41.99 | 23.81 | 762.02 |
| 256 | batch-32 | 25.16 | 42.29 | 23.64 | 756.61 |
| 1024 | batch-32 | 25.09 | 46.15 | 21.67 | 693.39 |

Observations: at batch-1 the spread is <2% (chunk size doesn't dominate at this batch + max_seq_len). At batch-32, k_chunk=1024 (i.e., one chunk per cache, the `k=max_seq_len` case in this demo) costs **~9% throughput** vs auto.

## Caveats and what this study does not measure

- **Not a flash vs. non-flash comparison.** The 20x (FlashAttention) and 2-10x (FlashDecode) speedups reported in the tech reports are vs. the *naive matmul → softmax → matmul with DRAM-spilling intermediates*, an op tt-metal no longer ships. Within FlashDecode, all my data points still keep intermediates in L1.
- **L1 capacity prevents `k_chunk = seqlen`** at any seqlen ≥ 2048; OOMs uniformly above k=1024. So I never measured a true single-chunk softmax at long context.
- **Single-chip n300 path** (one of the n300's two chips); multi-chip mesh parallelization (T3K-style with `n_kv_heads=1` per chip) is exactly the regime where `max_cores_per_head_batch` matters most per FlashDecode §4.2, and is *not* covered here.
- **No Tracy device-timer breakdown.** Times are host wall-clock with `ttnn.synchronize_device` barriers — accurate for end-to-end op latency but does not separate K-gather, QK^T, softmax, V-matmul, and reduction phases. Useful next step.

## Code changes

- `models/tt_transformers/tt/model_config.py:1486` — adds `TT_SDPA_K_CHUNK_DECODE` env-var override; default behavior unchanged.
- `tests/ttnn/unit_tests/operations/sdpa/bench_paged_sdpa_decode_ablation.py` — pytest harness (k_chunk and max_cores sweeps).
- `run_e2e_ablation.sh` — driver looping demo over k_chunk values.
- `scripts/sdpa_ablation_analyze.py` — generates this report.
