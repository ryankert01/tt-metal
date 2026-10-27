#!/bin/bash
# CB v5 — fragmentation-reduction boundary test.
#
# Question: when does paged-KV's fragmentation reduction matter on TT?
# §II.3 found "frag ≈ 0" on prefill-heavy + shared-prefix. §V.4 asks: does that
# generalize, or is it workload-specific?
#
# Method: for each workload, run paged baseline (MNS=32) and "non-paged proxy"
# (MNS=8 caps admission to simulate non-paged's max_seq_len reservation budget).
# If A/B throughput ratio > 1.2× → frag reduction matters.
#
# Cells:
#   F1 = decode-heavy + heterogeneous output (key test; expects A > B)
#   F3 = prefill-heavy + heterogeneous input, no shared prefix (mirror of §II.3
#        without sharing's amplification)
#
# Usage: bash run_cb_v5_frag.sh                # runs all 4 groups
#        bash run_cb_v5_frag.sh F1_paged F1_proxy   # specific groups
#
# Output: /tmp/cb_v5_<group>_burst.json + server / bench logs.

set -u
cd "$(dirname "$0")"
ROOT="$PWD"

# NOTE: positional args ("$@") cannot be trusted in this environment — the
# harness injects a UID as $1. Hardcode the group list; comment/uncomment to
# subset.
EXP_GROUPS=(F1_paged F1_proxy F3_paged F3_proxy)

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

log() { echo "[$(date '+%H:%M:%S')] [v5] $*"; }

# Workload definitions (input buckets, output buckets, MML)
# F1: decode-heavy + heterogeneous output
F1_IN="64:0.5,96:0.5"
F1_OUT="128:0.25,512:0.25,1024:0.25,2048:0.25"
F1_MML=4096
# F3: prefill-heavy + heterogeneous input, no shared prefix
F3_IN="256:0.25,1024:0.25,2048:0.25,4096:0.25"
F3_OUT="128:1.0"
F3_MML=8192

overall=0
for GROUP in "${EXP_GROUPS[@]}"; do
    case "$GROUP" in
        F1_paged) IN_B=$F1_IN; OUT_B=$F1_OUT; MML=$F1_MML; MNS=32 ;;
        F1_proxy) IN_B=$F1_IN; OUT_B=$F1_OUT; MML=$F1_MML; MNS=8  ;;
        F3_paged) IN_B=$F3_IN; OUT_B=$F3_OUT; MML=$F3_MML; MNS=32 ;;
        F3_proxy) IN_B=$F3_IN; OUT_B=$F3_OUT; MML=$F3_MML; MNS=8  ;;
        *) log "unknown group $GROUP"; overall=1; continue ;;
    esac

    OUT_JSON="/tmp/cb_v5_${GROUP}_burst.json"
    SERVER_LOG="/tmp/cb_v5_${GROUP}_server.log"
    SERVER_PID="/tmp/cb_v5_${GROUP}_server.pid"
    BENCH_LOG="/tmp/cb_v5_${GROUP}_bench.log"
    TRACE_JSONL="/tmp/cb_v5_${GROUP}_trace.jsonl"

    if [ -s "$OUT_JSON" ]; then
        log "$GROUP: skip (output exists)"
        continue
    fi

    log "=== GROUP=$GROUP MML=$MML MNS=$MNS ==="
    log "  IN='$IN_B'  OUT='$OUT_B'"

    # Generate trace with --exact-tokens (real token counts)
    log "  generating trace"
    python3 scripts/cb_v3_make_trace.py \
        --n 64 \
        --in-buckets "$IN_B" \
        --out-buckets "$OUT_B" \
        --out-jsonl "$TRACE_JSONL" \
        --exact-tokens \
        --seed 42 \
        >> "$BENCH_LOG" 2>&1
    if [ ! -s "$TRACE_JSONL" ]; then
        log "  $GROUP: trace gen failed (see $BENCH_LOG)"
        overall=1
        continue
    fi

    # Start server
    log "  starting server"
    rm -f "$SERVER_LOG"
    python3 ../vllm/examples/server_example_tt.py \
        --model "$HF_MODEL" \
        --max_num_seqs "$MNS" \
        --block_size 64 \
        --max-model-len "$MML" \
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
            overall=1
            break
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done

    if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        log "  $GROUP skipped (server not ready)"
        kill -9 "$(cat $SERVER_PID)" 2>/dev/null
        rm -f "$SERVER_PID"
        continue
    fi

    # Run bench (single bench, all 64 prompts at t=0, request-rate=inf)
    log "  bench (64 prompts, request-rate=inf)"
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
        >> "$BENCH_LOG" 2>&1
    rc=$?
    log "  bench exit=$rc"
    [ $rc -eq 0 ] || overall=1

    # Stop server
    log "  stopping server"
    kill "$(cat $SERVER_PID)" 2>/dev/null
    sleep 5
    kill -9 "$(cat $SERVER_PID)" 2>/dev/null
    rm -f "$SERVER_PID"
done

log "DONE (overall exit=$overall)"
exit $overall
