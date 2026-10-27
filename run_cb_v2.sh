#!/bin/bash
# Orchestrator for CB vs static V2:
# 1. Generate v2 trace (extreme output heterogeneity)
# 2. Run vLLM server + 2 benches (burst + Poisson rate=1) with --ignore-eos
# 3. Run static driver with max_generated_tokens=max(output) and stop_at_eos=0
# 4. Caller runs analyze script
#
# Logs:   /tmp/cb_v2_*.log
# Output: /tmp/cb_v2_burst.json, /tmp/cb_v2_poisson1.json,
#         /tmp/cb_vs_static_static_summary_v2.json

set -u
cd "$(dirname "$0")"

ROOT="$PWD"
HF_MODEL_NAME="${HF_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
HF_PATH=$(ls -d "$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"*/ 2>/dev/null | head -1)
HF_PATH="${HF_PATH%/}"
[ -n "$HF_PATH" ] && [ -f "$HF_PATH/tokenizer.json" ] || { echo "ERROR: cached HF model not found at HF_PATH=$HF_PATH"; exit 1; }
# Use the LOCAL snapshot path so we never hit HF (gated repo, 401)
HF_MODEL="$HF_PATH"

SERVER_LOG=/tmp/cb_v2_server.log
SERVER_PID=/tmp/cb_v2_server.pid
BENCH_BURST_LOG=/tmp/cb_v2_bench_burst.log
BENCH_POIS_LOG=/tmp/cb_v2_bench_poisson1.log
BURST_JSON=/tmp/cb_v2_burst.json
POIS_JSON=/tmp/cb_v2_poisson1.json
TRACE_JSONL=/tmp/cb_vs_static_trace_v2.jsonl

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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# === Step 1: Trace ===
log "(1/4) Generating v2 trace"
python3 scripts/cb_vs_static_make_trace_v2.py
[ -s "$TRACE_JSONL" ] || { log "trace JSONL missing"; exit 1; }

# === Step 2: Start vLLM server ===
log "(2/4) Starting vLLM server (HF_MODEL=$HF_MODEL)"
rm -f "$SERVER_LOG"
python3 ../vllm/examples/server_example_tt.py \
    --model "$HF_MODEL" \
    --max_num_seqs 32 \
    --block_size 64 \
    --max-model-len 1024 \
    --disable-log-requests \
    > "$SERVER_LOG" 2>&1 &
echo $! > "$SERVER_PID"
log "  server pid=$(cat $SERVER_PID), waiting for /health (max 5min)"

# Wait up to 5 min for /health
elapsed=0
while [ $elapsed -lt 300 ]; do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        log "  server ready after ${elapsed}s"
        break
    fi
    if ! kill -0 "$(cat $SERVER_PID)" 2>/dev/null; then
        log "  ERROR: server died, see $SERVER_LOG"
        tail -40 "$SERVER_LOG"
        exit 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
done
if [ $elapsed -ge 300 ]; then
    log "  TIMEOUT waiting for server"
    tail -40 "$SERVER_LOG"
    kill -9 "$(cat $SERVER_PID)" 2>/dev/null
    exit 1
fi

# === Step 3: Run benches ===
COMMON_ARGS=(
    --backend vllm
    --model "$HF_MODEL"
    --tokenizer "$HF_MODEL"
    --dataset-name custom
    --dataset-path "$TRACE_JSONL"
    --custom-output-len -1
    --num-prompts 64
    --ignore-eos
    --percentile-metrics ttft,tpot,itl,e2el
    --save-result
)

log "  bench 1/4: burst (request-rate=inf)"
rm -f "$BURST_JSON"
vllm bench serve \
    "${COMMON_ARGS[@]}" \
    --request-rate inf \
    --result-filename "$BURST_JSON" \
    > "$BENCH_BURST_LOG" 2>&1
log "  burst exit=$? (log=$BENCH_BURST_LOG)"

log "  bench 2/4: Poisson rate=1 req/s"
rm -f "$POIS_JSON"
vllm bench serve \
    "${COMMON_ARGS[@]}" \
    --request-rate 1 \
    --burstiness 1.0 \
    --result-filename "$POIS_JSON" \
    > "$BENCH_POIS_LOG" 2>&1
log "  poisson exit=$? (log=$BENCH_POIS_LOG)"

# vLLM-native static-imitation: 2 sequential 32-prompt bench invocations
# against the SAME vLLM server. Identical kernel/scheduler path as CB; only
# the client policy differs (batch-end barrier between b1 and b2).
STATIC_B1_JSON=/tmp/cb_v2_vllm_static_b1.json
STATIC_B2_JSON=/tmp/cb_v2_vllm_static_b2.json
STATIC_B1_LOG=/tmp/cb_v2_bench_static_b1.log
STATIC_B2_LOG=/tmp/cb_v2_bench_static_b2.log
STATIC_BENCH_ARGS=(
    --backend vllm
    --model "$HF_MODEL"
    --tokenizer "$HF_MODEL"
    --dataset-name custom
    --custom-output-len -1
    --num-prompts 32
    --ignore-eos
    --request-rate inf
    --percentile-metrics ttft,tpot,itl,e2el
    --save-result
)

log "  bench 3/4: vLLM-static batch 1 (32 prompts, request-rate=inf)"
rm -f "$STATIC_B1_JSON"
vllm bench serve \
    "${STATIC_BENCH_ARGS[@]}" \
    --dataset-path /tmp/cb_v2_static_b1.jsonl \
    --result-filename "$STATIC_B1_JSON" \
    > "$STATIC_B1_LOG" 2>&1
log "  static b1 exit=$? (log=$STATIC_B1_LOG)"

log "  bench 4/4: vLLM-static batch 2 (32 prompts, request-rate=inf)"
rm -f "$STATIC_B2_JSON"
vllm bench serve \
    "${STATIC_BENCH_ARGS[@]}" \
    --dataset-path /tmp/cb_v2_static_b2.jsonl \
    --result-filename "$STATIC_B2_JSON" \
    > "$STATIC_B2_LOG" 2>&1
log "  static b2 exit=$? (log=$STATIC_B2_LOG)"

# === Step 4: Stop server ===
log "(3/4) Stopping vLLM server"
kill "$(cat $SERVER_PID)" 2>/dev/null
sleep 5
kill -9 "$(cat $SERVER_PID)" 2>/dev/null

# === Step 5: Static driver ===
log "(4/4) Running static driver"
python3 scripts/cb_vs_static_static_driver_v2.py \
    --trace /tmp/cb_vs_static_trace_v2.json \
    --hf-model "$HF_PATH" \
    --tt-metal-home "$ROOT"

log "DONE — run analyze: python3 scripts/cb_vs_static_analyze_v2.py"
