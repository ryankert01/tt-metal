# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Summarize device-profiler zones from profile_log_device.csv.

prof_zones.py <csv> [--zones PREFIX,PREFIX...] [--risc TRISC_1] [--skip-first N] [--json OUT]
Pairs ZONE_START/ZONE_END per (core, RISC, zone) with a stack, converts cycles to us with the file's CHIP_FREQ,
and prints per-zone count / median / mean / min / max per RISC type, aggregated over cores.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--zones", default="pp_,st_,prep_,scan_,tx_,rx_")
    ap.add_argument("--risc", default=None)
    ap.add_argument("--skip-first", type=int, default=0, help="drop the first N instances of each zone per core")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    prefixes = tuple(p for p in args.zones.split(",") if p)

    with open(args.csv) as f:
        header = f.readline()
        freq_mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.reader(f))
    cols = [c.strip() for c in rows[0]]
    ix = {c: i for i, c in enumerate(cols)}
    stacks = defaultdict(list)
    durs = defaultdict(lambda: defaultdict(list))  # (risc, zone) -> core -> [cycles]
    for r in rows[1:]:
        if len(r) < len(cols) - 1:
            continue
        zone = r[ix["zone name"]].strip()
        if not zone.startswith(prefixes):
            continue
        risc = r[ix["RISC processor type"]].strip()
        if args.risc and risc != args.risc:
            continue
        core = (r[ix["core_x"]].strip(), r[ix["core_y"]].strip())
        t = int(r[ix["time[cycles since reset]"]])
        typ = r[ix["type"]].strip()
        key = (core, risc, zone)
        if typ == "ZONE_START":
            stacks[key].append(t)
        elif typ == "ZONE_END" and stacks[key]:
            durs[(risc, zone)][core].append(t - stacks[key].pop())

    out = {}
    order = sorted(durs.keys(), key=lambda k: (k[1], k[0]))
    print(f"CHIP_FREQ {freq_mhz:.0f} MHz   (us; aggregated over cores, first {args.skip_first} per core dropped)")
    print(f"{'zone':18s} {'risc':8s} {'n':>5s} {'cores':>5s} {'median':>9s} {'mean':>9s} {'min':>9s} {'max':>9s}")
    for risc, zone in order:
        vals = []
        for core, v in durs[(risc, zone)].items():
            vals.extend(v[args.skip_first :])
        if not vals:
            continue
        us = [x / freq_mhz for x in vals]
        med, mean = statistics.median(us), statistics.fmean(us)
        print(
            f"{zone:18s} {risc:8s} {len(us):5d} {len(durs[(risc, zone)]):5d} {med:9.3f} {mean:9.3f} {min(us):9.3f} {max(us):9.3f}"
        )
        out[f"{zone}|{risc}"] = {"n": len(us), "median": med, "mean": mean, "min": min(us), "max": max(us)}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
