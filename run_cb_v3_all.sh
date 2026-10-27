#!/bin/bash
# CB v3 outer driver — runs all groups in order.
#
# Order (cheapest+most informative first):
#   1. S1: main block (A1,A3,A4,B1,B2)
#   2. [stop-gate: check A's p99-ITL monotonicity]
#   3. S5,S6,S7,S8: admission cadence sweep (D1..D4)
#   4. S2: B4 extreme heterogeneity
#   5. S3: C2 medium input
#   6. S4: C3 long input (fall back to S4_FALLBACK on failure)
#
# Each group has its own server startup. Re-running is idempotent (skips done benches).

set -u
cd "$(dirname "$0")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [driver] $*"; }

GROUPS=("${@:-S1 S5 S6 S7 S8 S2 S3 S4}")
if [ $# -eq 0 ]; then
    GROUPS=(S1 S5 S6 S7 S8 S2 S3 S4)
fi

overall=0
for G in "${GROUPS[@]}"; do
    log "=== starting group $G ==="
    bash ./run_cb_v3.sh "$G"
    rc=$?
    log "=== group $G exit=$rc ==="

    if [ $rc -ne 0 ] && [ "$G" = "S4" ]; then
        log "S4 failed (likely OOM at MML=8192); trying S4_FALLBACK (MML=4096)"
        bash ./run_cb_v3.sh S4_FALLBACK
        rc=$?
        log "=== group S4_FALLBACK exit=$rc ==="
    fi

    [ $rc -eq 0 ] || overall=1
done

log "ALL DONE (overall exit=$overall)"
log "run: python3 scripts/cb_v3_analyze.py"
exit $overall
