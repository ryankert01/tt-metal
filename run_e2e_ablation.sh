#!/bin/bash
# End-to-end Llama-3.1-8B ablation: paged SDPA decode k_chunk_size sweep.
# Loops over a small set of k_chunk values and demo selectors, captures TTFT/ITL/tok-s.
# Each invocation is a clean process so the @lru_cache picks up the env var fresh.

set -u
cd "$(dirname "$0")"

: "${HF_MODEL:=meta-llama/Llama-3.1-8B-Instruct}"
: "${TT_METAL_HOME:=$PWD}"
: "${ARCH_NAME:=wormhole_b0}"
export HF_MODEL TT_METAL_HOME ARCH_NAME

source python_env/bin/activate

OUT_DIR="${OUT_DIR:-/tmp/sdpa_ablation_e2e}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/summary.csv"

# K_CHUNKS: 0=auto-flash; 256=mid; 1024=non-flash bound (rounded for safety; raise to seqlen if L1 allows)
K_CHUNKS="${K_CHUNKS:-0 256 1024}"
SELECTORS="${SELECTORS:-batch-1 batch-32}"

echo "k_chunk,selector,ttft_ms,itl_ms,decode_tok_s_user,decode_tok_s,log_path" > "$CSV"

for sel in $SELECTORS; do
    for k in $K_CHUNKS; do
        log="$OUT_DIR/k${k}_${sel}.log"
        echo "=== k_chunk=$k selector=$sel ===" | tee -a "$CSV.run"
        TT_SDPA_K_CHUNK_DECODE=$k pytest -s \
            models/tt_transformers/demo/simple_text_demo.py \
            -k "performance and ${sel}" \
            --capture=tee-sys \
            > "$log" 2>&1
        rc=$?

        # Demo log format (simple_text_demo.py:1461-1463):
        #   Average Time to First Token (TTFT): 67.34ms
        #   Average speed: 22.61ms @ 44.21 tok/s/user (44.21 tok/s throughput)
        ttft=$(grep -oE "Time to First Token \(TTFT\): [0-9.]+ms" "$log" | tail -1 | grep -oE "[0-9.]+")
        itl=$(grep -oE "Average speed: [0-9.]+ms" "$log" | tail -1 | grep -oE "[0-9.]+")
        tok_s_u=$(grep -oE "@ [0-9.]+ tok/s/user" "$log" | tail -1 | grep -oE "[0-9.]+")
        tok_s=$(grep -oE "\([0-9.]+ tok/s throughput\)" "$log" | tail -1 | grep -oE "[0-9.]+")

        echo "  rc=$rc ttft=${ttft:-NA} itl=${itl:-NA} tok_s_u=${tok_s_u:-NA} tok_s=${tok_s:-NA}"
        echo "$k,$sel,${ttft:-NA},${itl:-NA},${tok_s_u:-NA},${tok_s:-NA},$log" >> "$CSV"
    done
done

echo "Done. Summary: $CSV"
