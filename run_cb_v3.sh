#!/bin/bash
# CB v3 gap-filling experiment runner.
#
# Usage: run_cb_v3.sh <GROUP_ID>
# Groups: S1 S2 S3 S4 S4_FALLBACK S5 S6 S7 S8
#
# Each group = one server config (MML, MNS, MMBT) + a list of benches.
# Script starts server once, runs all benches in the group, kills server.
# Idempotent: skips a bench if its output JSON already exists.
#
# Outputs: /tmp/cb_v3_<EXP_ID>_burst.json + /tmp/cb_v3_<GROUP_ID>_server.log
# Bench logs: /tmp/cb_v3_<EXP_ID>_bench.log

set -u
cd "$(dirname "$0")"
ROOT="$PWD"

GROUP_ID="${1:-}"
[ -n "$GROUP_ID" ] || { echo "usage: $0 <GROUP_ID>  (S1..S8)"; exit 2; }

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

log() { echo "[$(date '+%H:%M:%S')] [$GROUP_ID] $*"; }

# Bench params per group: parallel arrays indexed by bench number.
# EXP_IDS, NS, IN_BUCKETS_LIST, OUT_BUCKETS_LIST, RATES
EXP_IDS=(); NS=(); IN_BUCKETS_LIST=(); OUT_BUCKETS_LIST=(); RATES=()

add_bench() {
    EXP_IDS+=("$1"); NS+=("$2"); IN_BUCKETS_LIST+=("$3"); OUT_BUCKETS_LIST+=("$4"); RATES+=("$5")
}

IN_DEFAULT="16:0.4,48:0.4,96:0.2"
OUT_HET24="16:0.4,64:0.3,256:0.2,384:0.1"

case "$GROUP_ID" in
    S1)
        SRV_MML=1024; SRV_MNS=32; SRV_MMBT=""
        add_bench A1 32  "$IN_DEFAULT" "$OUT_HET24" inf
        add_bench A3 128 "$IN_DEFAULT" "$OUT_HET24" inf
        add_bench A4 256 "$IN_DEFAULT" "$OUT_HET24" inf
        add_bench B1 64  "$IN_DEFAULT" "256:1.0" inf
        add_bench B2 64  "$IN_DEFAULT" "128:0.25,256:0.25,384:0.25,512:0.25" inf
        ;;
    S2)
        # MML must be power of 2 (TT warmup constraint). 2048 covers 96-input + 1024-output.
        SRV_MML=2048; SRV_MNS=32; SRV_MMBT=""
        add_bench B4 64 "$IN_DEFAULT" "16:0.5,1024:0.5" inf
        ;;
    S3)
        SRV_MML=2048; SRV_MNS=32; SRV_MMBT=""
        add_bench C2 64 "256:0.34,512:0.33,1024:0.33" "256:1.0" inf
        ;;
    S4)
        SRV_MML=8192; SRV_MNS=32; SRV_MMBT=""
        add_bench C3 64 "2048:0.5,4096:0.5" "256:1.0" inf
        ;;
    S4_FALLBACK)
        SRV_MML=4096; SRV_MNS=32; SRV_MMBT=""
        add_bench C3 64 "1024:0.5,2048:0.5" "256:1.0" inf
        ;;
    S5) SRV_MML=1024; SRV_MNS=32; SRV_MMBT=2048;  add_bench D1 64 "$IN_DEFAULT" "$OUT_HET24" inf ;;
    S6) SRV_MML=1024; SRV_MNS=32; SRV_MMBT=4096;  add_bench D2 64 "$IN_DEFAULT" "$OUT_HET24" inf ;;
    S7) SRV_MML=1024; SRV_MNS=32; SRV_MMBT=8192;  add_bench D3 64 "$IN_DEFAULT" "$OUT_HET24" inf ;;
    S8) SRV_MML=1024; SRV_MNS=32; SRV_MMBT=16384; add_bench D4 64 "$IN_DEFAULT" "$OUT_HET24" inf ;;
    *) echo "unknown group $GROUP_ID"; exit 2 ;;
esac

# Idempotency: skip group if all outputs already exist
all_exist=1
for EXP_ID in "${EXP_IDS[@]}"; do
    if [ ! -s "/tmp/cb_v3_${EXP_ID}_burst.json" ]; then
        all_exist=0; break
    fi
done
if [ $all_exist -eq 1 ]; then
    log "all bench outputs already exist; skipping group"
    exit 0
fi

# === Start server ===
SERVER_LOG="/tmp/cb_v3_${GROUP_ID}_server.log"
SERVER_PID="/tmp/cb_v3_${GROUP_ID}_server.pid"
log "starting server: MML=$SRV_MML MNS=$SRV_MNS MMBT=${SRV_MMBT:-default}"
rm -f "$SERVER_LOG"

SRV_ARGS=(
    --model "$HF_MODEL"
    --max_num_seqs "$SRV_MNS"
    --block_size 64
    --max-model-len "$SRV_MML"
    --disable-log-requests
)
if [ -n "$SRV_MMBT" ]; then
    SRV_ARGS+=(--max-num-batched-tokens "$SRV_MMBT")
fi

python3 ../vllm/examples/server_example_tt.py "${SRV_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
echo $! > "$SERVER_PID"
log "  pid=$(cat $SERVER_PID), waiting /health (max 8min)"

elapsed=0
while [ $elapsed -lt 480 ]; do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        log "  server ready after ${elapsed}s"
        break
    fi
    if ! kill -0 "$(cat $SERVER_PID)" 2>/dev/null; then
        log "  ERROR: server died, last 40 lines:"
        tail -40 "$SERVER_LOG"
        exit 1
    fi
    sleep 10; elapsed=$((elapsed + 10))
done
if [ $elapsed -ge 480 ]; then
    log "  TIMEOUT"; tail -40 "$SERVER_LOG"
    kill -9 "$(cat $SERVER_PID)" 2>/dev/null; exit 1
fi

# === Run benches ===
n_benches=${#EXP_IDS[@]}
overall_exit=0
for ((i=0; i<n_benches; i++)); do
    EXP_ID="${EXP_IDS[i]}"
    N="${NS[i]}"
    IN_B="${IN_BUCKETS_LIST[i]}"
    OUT_B="${OUT_BUCKETS_LIST[i]}"
    RATE="${RATES[i]}"
    OUT_JSON="/tmp/cb_v3_${EXP_ID}_burst.json"
    BENCH_LOG="/tmp/cb_v3_${EXP_ID}_bench.log"
    TRACE_JSONL="/tmp/cb_v3_${EXP_ID}_trace.jsonl"

    if [ -s "$OUT_JSON" ]; then
        log "  $EXP_ID: skip (output exists)"
        continue
    fi

    # C-series experiments need actual token counts (not v2's word-count approximation)
    EXACT_FLAG=""
    if [[ "$EXP_ID" == C* ]]; then
        EXACT_FLAG="--exact-tokens"
    fi
    log "  $EXP_ID: trace n=$N in='$IN_B' out='$OUT_B' exact=${EXACT_FLAG:-no}"
    python3 scripts/cb_v3_make_trace.py \
        --n "$N" \
        --in-buckets "$IN_B" \
        --out-buckets "$OUT_B" \
        --out-jsonl "$TRACE_JSONL" \
        --seed 42 \
        $EXACT_FLAG \
        >> "$BENCH_LOG" 2>&1
    if [ ! -s "$TRACE_JSONL" ]; then
        log "  $EXP_ID: trace gen failed (see $BENCH_LOG)"
        overall_exit=1
        continue
    fi

    log "  $EXP_ID: bench n=$N rate=$RATE"
    vllm bench serve \
        --backend vllm \
        --model "$HF_MODEL" \
        --tokenizer "$HF_MODEL" \
        --dataset-name custom \
        --dataset-path "$TRACE_JSONL" \
        --custom-output-len -1 \
        --num-prompts "$N" \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --request-rate "$RATE" \
        --result-filename "$OUT_JSON" \
        >> "$BENCH_LOG" 2>&1
    bench_exit=$?
    log "  $EXP_ID: bench exit=$bench_exit"
    [ $bench_exit -eq 0 ] || overall_exit=1
done

# === Stop server ===
log "stopping server"
kill "$(cat $SERVER_PID)" 2>/dev/null
sleep 5
kill -9 "$(cat $SERVER_PID)" 2>/dev/null
rm -f "$SERVER_PID"

exit $overall_exit
