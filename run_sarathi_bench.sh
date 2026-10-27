#!/bin/bash
# Run the SARATHI contention benchmark against an already-running TT server.
# Usage: run_sarathi_bench.sh <label> [request_rate]
set -u
ROOT=/home/user01/tt-data/tt-metal
cd "$ROOT"
LABEL="${1:-baseline}"
RATE="${2:-4}"
TRACE=/tmp/sarathi_trace.jsonl
OUT_JSON="/tmp/sarathi_${LABEL}.json"

HF_PATH=$(ls -d "$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"*/ 2>/dev/null | head -1)
HF_PATH="${HF_PATH%/}"

source python_env/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

rm -f "$OUT_JSON"
echo "[bench] label=$LABEL rate=$RATE trace=$TRACE"
vllm bench serve \
    --backend vllm \
    --model "$HF_PATH" \
    --tokenizer "$HF_PATH" \
    --dataset-name custom \
    --dataset-path "$TRACE" \
    --custom-output-len -1 \
    --num-prompts 24 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --request-rate "$RATE" \
    --result-filename "$OUT_JSON"
echo "[bench] exit=$? -> $OUT_JSON"
