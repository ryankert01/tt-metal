# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device-side duration of the GDN program from profile_log_device.csv: for every core that ran a chunk_gdn kernel,
take its TRISC-KERNEL / BRISC-KERNEL / NCRISC-KERNEL zones whose source is not the firmware idle path and report, per
program run, max(end) - min(start) over those cores (us). Prints the largest run (the GDN op) and the per-chunk period.

prof_opdur.py <csv> <num_chunks> [all]
  all : print the two longest program runs instead (the phased path's prep and scan programs)
"""
import csv
import sys
from collections import defaultdict


def main():
    path, nc = sys.argv[1], int(sys.argv[2])
    with open(path) as f:
        header = f.readline()
        freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.reader(f))
    cols = [c.strip() for c in rows[0]]
    ix = {c: i for i, c in enumerate(cols)}
    # cores that ran GDN zones
    gdn_cores = set()
    for r in rows[1:]:
        if len(r) < len(cols) - 1:
            continue
        z = r[ix["zone name"]].strip()
        if z in ("prep_item", "scan_step"):
            gdn_cores.add((r[ix["core_x"]].strip(), r[ix["core_y"]].strip()))
    runs = defaultdict(lambda: [None, None])  # run id -> [min start, max end]
    span = {}
    for r in rows[1:]:
        if len(r) < len(cols) - 1:
            continue
        z = r[ix["zone name"]].strip()
        core = (r[ix["core_x"]].strip(), r[ix["core_y"]].strip())
        if not z.endswith("-KERNEL") or core not in gdn_cores:
            continue
        rid = r[ix["run host ID"]].strip()
        t = int(r[ix["time[cycles since reset]"]])
        typ = r[ix["type"]].strip()
        if typ == "ZONE_START":
            runs[rid][0] = t if runs[rid][0] is None else min(runs[rid][0], t)
        elif typ == "ZONE_END":
            runs[rid][1] = t if runs[rid][1] is None else max(runs[rid][1], t)
    for rid, (a, b) in runs.items():
        if a is not None and b is not None:
            span[rid] = (b - a) / freq
    if not span:
        print("no GDN kernel zones found")
        return
    if len(sys.argv) > 3 and sys.argv[3] == "all":
        top = sorted(span.items(), key=lambda kv: -kv[1])[:2]
        print("  ".join(f"run {rid}: {us:9.1f} us" for rid, us in top))
        return
    rid = max(span, key=span.get)
    us = span[rid]
    print(
        f"GDN program device time: {us:9.1f} us over {len(gdn_cores)} cores  ->  {us / nc:6.2f} us per chunk (NC={nc})"
    )


if __name__ == "__main__":
    main()
