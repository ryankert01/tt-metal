#!/bin/bash
# Non-paged proxy: same workload as run_prefix_share.sh, but cap max_num_seqs=8
# to simulate non-paged's per-user max_seq_len pre-reservation memory budget.
#
# Math: KV pool ~8.3 GB; non-paged reserves 1.0 GB/user (max_seq_len=16384,
# 64KB/token); ⇒ ~8 concurrent users max under true non-paged.
#
# Compare against existing /tmp/prefix_share_off.json (paged, no caching,
# max_num_seqs=32 → 15 admitted by KV pressure).

set -u
cd "$(dirname "$0")"

ROOT="$PWD"
HF_PATH=$(ls -d "$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"*/ 2>/dev/null | head -1)
HF_PATH="${HF_PATH%/}"
[ -n "$HF_PATH" ] && [ -f "$HF_PATH/tokenizer.json" ] || { echo "ERROR: cached HF model not found"; exit 1; }
HF_MODEL="$HF_PATH"

DATASET=/tmp/prefix_share_8k.jsonl     # reuse from run_prefix_share.sh
SERVER_PID=/tmp/nonpaged_server.pid
SERVER_LOG=/tmp/nonpaged_server.log
JSON_OUT=/tmp/nonpaged_proxy.json
BENCH_LOG=/tmp/nonpaged_bench.log

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

[ -f "$DATASET" ] || { log "ERROR: dataset $DATASET not found - run run_prefix_share.sh first"; exit 1; }

log "Starting server: --max_num_seqs 8 --no-enable-prefix-caching (non-paged proxy)"
rm -f "$SERVER_LOG"
python3 ../vllm/examples/server_example_tt.py \
    --model "$HF_MODEL" \
    --max_num_seqs 8 \
    --block_size 64 \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    > "$SERVER_LOG" 2>&1 &
echo $! > "$SERVER_PID"
log "  pid=$(cat $SERVER_PID), waiting /health (max 5min)"

elapsed=0
while [ $elapsed -lt 300 ]; do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        log "  ready after ${elapsed}s"; break
    fi
    if ! kill -0 "$(cat $SERVER_PID)" 2>/dev/null; then
        log "  ERROR: server died"; tail -40 "$SERVER_LOG"; exit 1
    fi
    sleep 10; elapsed=$((elapsed + 10))
done
[ $elapsed -ge 300 ] && { log "  TIMEOUT"; tail -40 "$SERVER_LOG"; kill -9 "$(cat $SERVER_PID)" 2>/dev/null; exit 1; }

log "Running bench: 64-prompt burst against capped batch=8"
rm -f "$JSON_OUT"
vllm bench serve \
    --backend vllm \
    --model "$HF_MODEL" \
    --tokenizer "$HF_MODEL" \
    --dataset-name custom \
    --dataset-path "$DATASET" \
    --custom-output-len -1 \
    --num-prompts 64 \
    --ignore-eos \
    --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result \
    --result-filename "$JSON_OUT" \
    > "$BENCH_LOG" 2>&1
log "  bench exit=$? (log=$BENCH_LOG)"

kill "$(cat $SERVER_PID)" 2>/dev/null
sleep 5
kill -9 "$(cat $SERVER_PID)" 2>/dev/null

log "Summary"
python3 - <<PY
import json
d = json.load(open("$JSON_OUT"))
print(f"  non-paged proxy (max_num_seqs=8, caching OFF):")
print(f"    duration       : {d['duration']:.1f} s")
print(f"    goodput        : {d['output_throughput']:.1f} tok/s")
print(f"    mean TTFT      : {d['mean_ttft_ms']:.0f} ms")
print(f"    p99 TTFT       : {d['p99_ttft_ms']:.0f} ms")
print(f"    mean TPOT      : {d['mean_tpot_ms']:.1f} ms")
print(f"    p99 ITL        : {d['p99_itl_ms']:.1f} ms")
print()
on  = json.load(open("/tmp/prefix_share_on.json"))
off = json.load(open("/tmp/prefix_share_off.json"))
print(f"  vs paged+caching ON (max_num_seqs=32): {d['duration']/on['duration']:.2f}x slower wall")
print(f"  vs paged+caching OFF (max_num_seqs=32): {d['duration']/off['duration']:.2f}x slower wall")
PY

log "DONE — raw at $JSON_OUT"
