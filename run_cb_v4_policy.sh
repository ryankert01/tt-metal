#!/bin/bash
# CB v4: 3-way admission-policy comparison on the same scattered workload.
#
# Same chip, same vLLM, same model, same workload (B3/A2 24x scattered output).
# Only TT_ADMISSION_POLICY differs:
#   - greedy: vLLM TT default (admit on every End)
#   - batched: accumulate K Ends before admit (K=8 default)
#   - defer: never admit mid-stream (= static behavior, server-side)
#
# Single bench client submits all 64 prompts at t=0, so TTFT clock is unified
# across all three policies (no admission_delay accounting needed).
#
# Usage: bash run_cb_v4_policy.sh [POLICY1 POLICY2 ...]
# Default: runs all three sequentially.

set -u
cd "$(dirname "$0")"
ROOT="$PWD"

POLICIES=("$@")
if [ ${#POLICIES[@]} -eq 0 ]; then
    POLICIES=(greedy batched defer)
fi
BATCH_K="${TT_ADMISSION_BATCH_K:-8}"

HF_PATH=$(ls -d "$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"*/ 2>/dev/null | head -1)
HF_PATH="${HF_PATH%/}"
[ -f "$HF_PATH/tokenizer.json" ] || { echo "ERROR: HF model not cached at $HF_PATH"; exit 1; }
HF_MODEL="$HF_PATH"

source python_env/bin/activate
export TT_METAL_HOME="$ROOT"
export ARCH_NAME=wormhole_b0
export MESH_DEVICE=N300
export VLLM_TARGET_DEVICE=tt
export PYTHONPATH="$ROOT:$ROOT/../vllm"
export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
export HF_MODEL
export VLLM_RPC_TIMEOUT=300000
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TT_ADMISSION_BATCH_K

log() { echo "[$(date '+%H:%M:%S')] [v4] $*"; }

# Reuse the v3 scattered-output trace (same workload as A2/v2 burst).
TRACE_JSONL=/tmp/cb_v4_scattered_trace.jsonl
if [ ! -s "$TRACE_JSONL" ]; then
    log "generating scattered workload trace (n=64, OUT={16,64,256,384})"
    python3 scripts/cb_v3_make_trace.py \
        --n 64 \
        --in-buckets "16:0.4,48:0.4,96:0.2" \
        --out-buckets "16:0.4,64:0.3,256:0.2,384:0.1" \
        --out-jsonl "$TRACE_JSONL" \
        --seed 42
fi

overall=0
for POLICY in "${POLICIES[@]}"; do
    OUT_JSON="/tmp/cb_v4_${POLICY}_burst.json"
    SERVER_LOG="/tmp/cb_v4_${POLICY}_server.log"
    SERVER_PID="/tmp/cb_v4_${POLICY}_server.pid"
    BENCH_LOG="/tmp/cb_v4_${POLICY}_bench.log"

    if [ -s "$OUT_JSON" ]; then
        log "$POLICY: skip (output exists)"
        continue
    fi

    log "=== POLICY=$POLICY ==="
    log "starting server (TT_ADMISSION_POLICY=$POLICY TT_ADMISSION_BATCH_K=$BATCH_K)"
    rm -f "$SERVER_LOG"

    TT_ADMISSION_POLICY="$POLICY" python3 ../vllm/examples/server_example_tt.py \
        --model "$HF_MODEL" \
        --max_num_seqs 32 \
        --block_size 64 \
        --max-model-len 1024 \
        --disable-log-requests \
        > "$SERVER_LOG" 2>&1 &
    echo $! > "$SERVER_PID"
    log "  pid=$(cat $SERVER_PID), waiting /health (max 8min)"

    elapsed=0
    while [ $elapsed -lt 480 ]; do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            log "  server ready after ${elapsed}s"
            break
        fi
        if ! kill -0 "$(cat $SERVER_PID)" 2>/dev/null; then
            log "  ERROR: server died, last 30 lines:"
            tail -30 "$SERVER_LOG"
            overall=1; break
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done
    if [ $elapsed -ge 480 ]; then
        log "  TIMEOUT"
        tail -30 "$SERVER_LOG"
        kill -9 "$(cat $SERVER_PID)" 2>/dev/null
        overall=1
    fi

    if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        log "$POLICY skipped due to server failure"
        continue
    fi

    log "  bench: 64 prompts, single client, request-rate=inf"
    rm -f "$OUT_JSON"
    vllm bench serve \
        --backend vllm \
        --model "$HF_MODEL" \
        --tokenizer "$HF_MODEL" \
        --dataset-name custom \
        --dataset-path "$TRACE_JSONL" \
        --custom-output-len -1 \
        --num-prompts 64 \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --request-rate inf \
        --result-filename "$OUT_JSON" \
        > "$BENCH_LOG" 2>&1
    rc=$?
    log "  bench exit=$rc"

    log "  stopping server"
    kill "$(cat $SERVER_PID)" 2>/dev/null
    sleep 5
    kill -9 "$(cat $SERVER_PID)" 2>/dev/null
    rm -f "$SERVER_PID"

    [ $rc -eq 0 ] || overall=1
done

log "DONE (overall exit=$overall)"
exit $overall
