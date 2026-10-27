#!/bin/bash
# Prefix-sharing benchmark: paged KV with prefix caching ON vs OFF.
#
# Goal: quantify the fragmentation + ref-counted-sharing advantage of paged
# attention under a realistic shared-system-prompt workload (the dominant
# mini-datacenter pattern: chat/agent serving with one shared template).
#
# 64 prompts, each = 8K shared prefix + 256-token unique suffix + 256 output.
# Same vLLM server, same kernels, same dataset — only `--enable-prefix-caching`
# differs.
#
# Outputs: /tmp/prefix_share_{on,off}.json + comparison printed to stdout.

set -u
cd "$(dirname "$0")"

ROOT="$PWD"
HF_PATH=$(ls -d "$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"*/ 2>/dev/null | head -1)
HF_PATH="${HF_PATH%/}"
[ -n "$HF_PATH" ] && [ -f "$HF_PATH/tokenizer.json" ] || { echo "ERROR: cached HF model not found"; exit 1; }
HF_MODEL="$HF_PATH"

DATASET=/tmp/prefix_share_8k.jsonl
WARMUP_DATASET=/tmp/prefix_share_warmup.jsonl
SERVER_PID=/tmp/prefix_share_server.pid
ON_SERVER_LOG=/tmp/prefix_share_server_on.log
OFF_SERVER_LOG=/tmp/prefix_share_server_off.log
ON_JSON=/tmp/prefix_share_on.json
OFF_JSON=/tmp/prefix_share_off.json
ON_BENCH_LOG=/tmp/prefix_share_bench_on.log
OFF_BENCH_LOG=/tmp/prefix_share_bench_off.log
WARMUP_ON_JSON=/tmp/prefix_share_warmup_on.json
WARMUP_OFF_JSON=/tmp/prefix_share_warmup_off.json

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

start_server() {
    local prefix_flag="$1"
    local server_log="$2"
    rm -f "$server_log"
    python3 ../vllm/examples/server_example_tt.py \
        --model "$HF_MODEL" \
        --max_num_seqs 32 \
        --block_size 64 \
        --max-model-len 16384 \
        $prefix_flag \
        --disable-log-requests \
        > "$server_log" 2>&1 &
    echo $! > "$SERVER_PID"
    log "  server pid=$(cat $SERVER_PID) [$prefix_flag], waiting /health (max 5min)"
    local elapsed=0
    while [ $elapsed -lt 300 ]; do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            log "  ready after ${elapsed}s"
            return 0
        fi
        if ! kill -0 "$(cat $SERVER_PID)" 2>/dev/null; then
            log "  ERROR: server died"
            tail -40 "$server_log"
            return 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    log "  TIMEOUT"
    tail -40 "$server_log"
    kill -9 "$(cat $SERVER_PID)" 2>/dev/null
    return 1
}

stop_server() {
    kill "$(cat $SERVER_PID)" 2>/dev/null
    sleep 5
    kill -9 "$(cat $SERVER_PID)" 2>/dev/null
}

run_bench() {
    local dataset="$1"
    local num_prompts="$2"
    local out_json="$3"
    rm -f "$out_json"
    vllm bench serve \
        --backend vllm \
        --model "$HF_MODEL" \
        --tokenizer "$HF_MODEL" \
        --dataset-name custom \
        --dataset-path "$dataset" \
        --custom-output-len -1 \
        --num-prompts "$num_prompts" \
        --ignore-eos \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-filename "$out_json"
}

# === Step 1: Generate dataset ===
log "(1/3) Generating prefix-shared dataset (64 prompts) + warmup (1 prompt)"
python3 scripts/make_prefix_share_dataset.py \
    --tokenizer "$HF_MODEL" \
    --out "$DATASET" \
    --num-prompts 64 \
    --shared-tokens 8192 \
    --unique-tokens 256 \
    --output-tokens 256

python3 scripts/make_prefix_share_dataset.py \
    --tokenizer "$HF_MODEL" \
    --out "$WARMUP_DATASET" \
    --num-prompts 1 \
    --shared-tokens 8192 \
    --unique-tokens 256 \
    --output-tokens 16

# === Step 2a: prefix caching ON (warmup + measure) ===
log "(2/3) Run A: --enable-prefix-caching"
start_server "--enable-prefix-caching" "$ON_SERVER_LOG" || exit 1
log "  warmup: 1 prompt to populate cache"
run_bench "$WARMUP_DATASET" 1 "$WARMUP_ON_JSON" > /tmp/prefix_share_warmup_on.log 2>&1
log "  warmup-on exit=$? (cache should now hold the 8K prefix)"
sleep 3
log "  measure: 64-prompt burst against warm cache"
run_bench "$DATASET" 64 "$ON_JSON" > "$ON_BENCH_LOG" 2>&1
log "  bench-on exit=$? (log=$ON_BENCH_LOG)"
stop_server
sleep 10

# === Step 2b: prefix caching OFF (same warmup pattern for fairness) ===
log "(2/3) Run B: --no-enable-prefix-caching"
start_server "--no-enable-prefix-caching" "$OFF_SERVER_LOG" || exit 1
log "  warmup: 1 prompt (no caching, no benefit, kept for symmetry)"
run_bench "$WARMUP_DATASET" 1 "$WARMUP_OFF_JSON" > /tmp/prefix_share_warmup_off.log 2>&1
log "  warmup-off exit=$?"
sleep 3
log "  measure: 64-prompt burst"
run_bench "$DATASET" 64 "$OFF_JSON" > "$OFF_BENCH_LOG" 2>&1
log "  bench-off exit=$? (log=$OFF_BENCH_LOG)"
stop_server

# === Step 3: Summary ===
log "(3/3) Summary"
python3 - <<'PY'
import json, re, subprocess
on = json.load(open('/tmp/prefix_share_on.json'))
off = json.load(open('/tmp/prefix_share_off.json'))
def row(d, label):
    print(f"  {label}:")
    print(f"    duration       : {d['duration']:.1f} s")
    print(f"    output_throughput: {d['output_throughput']:.1f} tok/s")
    print(f"    mean TTFT      : {d['mean_ttft_ms']:.0f} ms")
    print(f"    p99 TTFT       : {d['p99_ttft_ms']:.0f} ms")
    print(f"    mean TPOT      : {d['mean_tpot_ms']:.1f} ms")
    print(f"    p99 ITL        : {d['p99_itl_ms']:.1f} ms")
row(on, "prefix-caching ON")
row(off, "prefix-caching OFF")
print()
print(f"  TTFT speedup (off/on)   : {off['mean_ttft_ms']/on['mean_ttft_ms']:.2f}x")
print(f"  Goodput speedup (on/off): {on['output_throughput']/off['output_throughput']:.2f}x")
print(f"  Wall speedup (off/on)   : {off['duration']/on['duration']:.2f}x")
print()
# Surface prefix-cache hit rate from server log
def max_hit_rate(path):
    try:
        with open(path) as f:
            hits = [float(m.group(1)) for line in f
                    for m in [re.search(r"Prefix cache hit rate: ([\d.]+)%", line)] if m]
        return max(hits) if hits else None
    except FileNotFoundError:
        return None
print(f"  Max prefix-cache hit-rate (ON server log):  {max_hit_rate('/tmp/prefix_share_server_on.log')}")
print(f"  Max prefix-cache hit-rate (OFF server log): {max_hit_rate('/tmp/prefix_share_server_off.log')}")
PY

log "DONE — raw JSON at $ON_JSON $OFF_JSON"
