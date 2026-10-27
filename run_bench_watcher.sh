#!/bin/bash
# Background watcher: polls every 10 min for n300 to be free of foreign tt-metal users,
# then runs the two report benches sequentially.
#
# Done sentinel: /tmp/bench_watcher.done (exists on completion)
# Log:           /tmp/bench_watcher.log

set -u
cd "$(dirname "$0")"

LOG=/tmp/bench_watcher.log
DONE=/tmp/bench_watcher.done
POLL_INTERVAL=600  # 10 min

ME=$(id -u)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

chip_busy() {
    # Any process owned by a uid != me, with a tt-metal-ish cmdline,
    # is treated as holding the chip.
    ps -eo uid,pid,cmd \
        | awk -v me="$ME" '$1 != me && /run_vllm_api_server|tt_metal|ttnn/' \
        | grep -q .
}

rm -f "$DONE"
: > "$LOG"
log "watcher started (pid=$$), poll=${POLL_INTERVAL}s"

while chip_busy; do
    holder=$(ps -eo uid,pid,cmd | awk -v me="$ME" '$1 != me && /run_vllm_api_server|tt_metal|ttnn/ {print; exit}')
    log "chip busy, sleeping ${POLL_INTERVAL}s. holder: $holder"
    sleep "$POLL_INTERVAL"
done

log "chip free, starting benches"

source python_env/bin/activate
export TT_METAL_HOME="$PWD" ARCH_NAME=wormhole_b0

# Bench 1: paged vs non-paged decode
log "=== bench 1: paged vs non-paged decode ==="
rm -f /tmp/paged_vs_nonpaged_decode.csv
PAGED_VS_NONPAGED_CSV=/tmp/paged_vs_nonpaged_decode.csv \
    timeout 1800 pytest tests/ttnn/unit_tests/operations/sdpa/bench_paged_vs_nonpaged_decode.py -s \
    >> "$LOG" 2>&1
log "bench 1 exit=$?"

# Bench 2: flash vs naive prefill
log "=== bench 2: flash vs naive prefill ==="
rm -f /tmp/flash_vs_naive_prefill.csv
FLASH_VS_NAIVE_CSV=/tmp/flash_vs_naive_prefill.csv \
    timeout 1800 pytest tests/ttnn/unit_tests/operations/sdpa/bench_flash_vs_naive_prefill.py -s \
    >> "$LOG" 2>&1
log "bench 2 exit=$?"

# Bench 3: flash vs naive decode
log "=== bench 3: flash vs naive decode ==="
rm -f /tmp/flash_vs_naive_decode.csv
FLASH_VS_NAIVE_DECODE_CSV=/tmp/flash_vs_naive_decode.csv \
    timeout 1800 pytest tests/ttnn/unit_tests/operations/sdpa/bench_flash_vs_naive_decode.py -s \
    >> "$LOG" 2>&1
log "bench 3 exit=$?"

log "watcher done"
echo "done $(date)" > "$DONE"
