#!/bin/bash
# Launch a TT vLLM OpenAI server for the SARATHI chunked-prefill experiment.
# Usage: run_sarathi_server.sh <admission_policy> [extra server args...]
set -u
ROOT=/home/user01/tt-data/tt-metal
cd "$ROOT"

POLICY="${1:-greedy}"; shift || true

HF_PATH=$(ls -d "$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"*/ 2>/dev/null | head -1)
HF_PATH="${HF_PATH%/}"

source python_env/bin/activate
export TT_METAL_HOME="$ROOT"
export ARCH_NAME=wormhole_b0
export MESH_DEVICE=N300
export VLLM_TARGET_DEVICE=tt
export PYTHONPATH="$ROOT:$ROOT/../vllm"
export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
export VLLM_RPC_TIMEOUT=300000
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_MODEL="$HF_PATH"
export TT_ADMISSION_POLICY="$POLICY"

echo "[server] policy=$POLICY model=$HF_PATH args=$*"
exec python3 ../vllm/examples/server_example_tt.py \
    --model "$HF_PATH" \
    --max_num_seqs 32 \
    --block_size 64 \
    --max-model-len 2048 \
    --disable-log-requests \
    "$@"
