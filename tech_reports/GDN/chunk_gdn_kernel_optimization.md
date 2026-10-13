# GDN `chunk_gated_delta_rule` compute kernels: fused SFPU passes and scan operand reuse

Measured 2026-10-12 on top of `20e872e3b6`. Scope: the two compute kernels of
`ttnn.transformer.chunk_gated_delta_rule` (`chunk_gdn_prep.cpp`, `chunk_gdn_scan.cpp`, both implemented in
`device/kernels/compute/chunk_gdn_math.hpp`) and the two writers that drain the prep kernel.

Update 2026-10-13: for upstreaming, the change is split into the branches `gdn-prep-fixes` (the two bug fixes)
and `gdn-prep-scan-perf` (the performance work, stacked on it). Both trees were re-profiled back to back; see
the note under the timing table.

## Summary

- On a Wormhole n150 the prep kernel's per-chunk item is 16% faster (56.9 to 47.8 us), the scan step is up to 28%
  faster, and the fused op end to end is 15% to 24% faster across five geometries.
- At the production chunk size (32) every output is bit-identical to the baseline, on n150 silicon and on simulated
  Blackhole.
- Two existing bugs are fixed: `chunk_size=64` produced wrong results, and a scratch circular buffer was overrun
  on every chunk.
- Every program is smaller than before, by 2 to 7 KB, against the 70,656-byte kernel-config buffer.
- **Not measured: Blackhole silicon.** The only hardware available was a Wormhole n150. Blackhole was covered for
  correctness and program size with the `ttsim` simulator, which is bit-exact but not cycle-accurate. The Blackhole
  timing gain and one timing-dependent hazard (see [Open items](#open-items)) still need a run on hardware.

## Test setup

| | |
|---|---|
| Silicon | one Wormhole n150 L, 8x8 Tensix grid, 1000 MHz; timing from the device profiler (`TT_METAL_DEVICE_PROFILER=1`) |
| Blackhole | `ttsim` v1.11.3 (the version pinned in `tt_metal/ttsim-version`), slow dispatch |
| Op shape | K = V = 128 (4 tiles each), `chunk_size` 32 unless stated, fp32 hand-off tensors |
| Input forms | "flat": raw flat q/k/v with the L2 norm of q and k done in the kernel (the production form); "split": host-normalised q/k |

Two differences between the test beds matter when reading the numbers:

- The triangular inverse inside the prep item has two implementations. Blackhole at chunk 32 uses one SFPU
  forward-substitution solve (`GDN_TINV_SFPU`). Everything else, including every Wormhole run, uses the blocked Horner
  inverse. So the n150 timings below all include the Horner inverse, which this work did not touch.
- Wormhole silicon and simulated Blackhole produce bit-identical outputs for the same kernel code path, so the n150
  is a valid arithmetic check as well as the timing source.

Terms used below: a *producer* core runs the prep kernel and emits one *item* per (head, chunk), seven hand-off
tensors. A *receiver* core runs the scan kernel and does one *step* per chunk. In the fused program each value head
gets NP producers and NV receivers, and a receiver handles a slice of the V dimension that is 4 / NV tiles wide.

## Results

### Timing on the n150

Per-kernel times are medians of the `prep_item` and `scan_step` profiler zones on the unpack thread, first two
instances per core dropped, fused program with 4 value heads, 8 chunks, NP = 2.

| | Baseline | New | Change |
|---|---|---|---|
| Prep item | 56.9 us | 47.8 us | -16% |
| Scan step, slice 4 tiles wide (NV = 1) | 12.01 us | 8.65 us | -28% |
| Scan step, slice 2 tiles wide (NV = 2) | 6.51 us | 4.74 us | -27% |
| Scan step, slice 1 tile wide (NV = 4) | 3.15 us | 3.03 us | -4% |

- The Horner inverse takes 31 to 32 us of the prep item in both builds. Everything else in the item dropped from
  roughly 25 to 26 us to 15.5 to 16.7 us, which is -36% to -39%. The range reflects whether the wait for the
  inverse's input is charged to the inverse zone or to the block before it.
- The baseline figures for the 2-tile and 1-tile slices come from a build with the baseline scan code and an
  intermediate version of the prep code. Re-measured on 2026-10-13 against the bug-fix-only tree, three runs per
  tree: prep item 56.9 to 47.8 us; scan step 12.03 to 8.67, 6.51 to 4.75 and 3.16 to 3.03 us; the fused-op
  times below reproduce within 0.3%. The bug fixes alone change no timing.
- A 1-tile slice gains almost nothing because the scan change removes repeated unpacks within a row of output
  tiles, and a 1-tile slice has one tile per row.

Fused op, device time from the first kernel start to the last kernel end, T = 2048 (64 chunks), one profiled run
per geometry and build:

| Value heads | NV | NP | Cores | Baseline | New | Change |
|---|---|---|---|---|---|---|
| 4 | 1 | 4 | 20 | 983.7 us | 823.6 us | -16.3% |
| 4 | 1 | 7 | 32 | 905.4 us | 690.2 us | -23.8% |
| 4 | 2 | 6 | 32 | 685.8 us | 579.6 us | -15.5% |
| 8 | 1 | 6 | 56 | 927.3 us | 716.0 us | -22.8% |
| 8 | 2 | 5 | 56 | 817.6 us | 694.8 us | -15.0% |

Phased path, 8 value heads, T = 2048, two runs per build:

| Program | Baseline | New |
|---|---|---|
| Prep | 581.7, 584.9 us | 503.1, 502.7 us (-14%) |
| Scan | 676.3, 658.7 us | 640.6, 669.9 us (no clear change) |

### What to expect on Blackhole

Unmeasured, so this is reasoning only.

- The repo's fused cost model (`chunk_gdn_device_operation.cpp`) records the Blackhole scan step at 2.4, 2.9 and
  5.0 us for 1-, 2- and 4-tile slices, against 3.2, 6.5 and 12.0 us measured here on Wormhole. Blackhole's scan is
  therefore much less dominated by unpacking, and the scan gain should be smaller than on Wormhole.
- The prep gain should carry over in direction, because the removed SFPU work is the same on both architectures and
  the production Blackhole path additionally drops the masking and negation of the solve's input. Its size depends
  on Blackhole's SFPU and unpacker speeds, which were not measured.

### Program size

Sum of all kernel binaries of the largest program per configuration, in bytes, from
`TT_METAL_LOG_JIT_TELEMETRY=1` (`program_config_size.total.TENSIX`). The limit is 70,656.

Wormhole (n150), baseline to new:

| Configuration | Normal | Device profiler | Watcher |
|---|---|---|---|
| fused, flat, chunk 32 | 57,872 to 52,272 | 68,592 to 62,400 | 65,152 to 59,232 |
| fused, split, chunk 32 | 47,760 to 45,728 | 58,704 to 58,000 | 54,704 to 50,720 |
| fused, split, chunk 64 | 62,064 to 58,576 | 68,944 to 66,352 | 69,072 to 62,256 |
| phased, flat, chunk 32 | 60,528 to 54,032 | 70,192 to 63,344 | 66,432 to 59,536 |
| phased, split, chunk 64 | 61,488 to 56,672 | 68,416 to 63,664 | 68,336 to 60,240 |

Blackhole (simulated), baseline to new. The first row is the production configuration.

| Configuration | Normal | Watcher |
|---|---|---|
| fused, flat, chunk 32 (SFPU solve) | 42,832 to 40,560 | 49,488 to 47,504 |
| fused, flat, chunk 32, Horner forced | 58,528 to 52,960 | 65,536 to 59,728 |
| fused, split, chunk 32 | 34,272 to 31,952 | 41,264 to 38,176 |
| fused, split, chunk 64 | 62,576 to 56,624 | 70,064 to 63,776 |
| phased, flat, chunk 32 | 41,856 to 39,584 | not measured to 45,344 |
| phased, split, chunk 64 | 62,928 to 53,856 | not measured to 59,984 |

Blackhole profiler builds are too slow under the simulator to sweep. The one baseline data point obtained is that
the fused, split, chunk-64 program does **not** fit with the device profiler on (70,768 bytes). The new code was not
re-measured in that build. It is 6.3 KB smaller in the watcher build, so it very likely fits, but that is an
inference.

### Correctness

All comparisons are bitwise on the fp32 words, new build against baseline build, same inputs.

| Check | n150 silicon | Simulated Blackhole |
|---|---|---|
| Seven prep hand-off tensors, chunk 32, five configurations (flat and split inputs, two numeric regimes, both inverses) | identical | identical |
| Final output and final state of the whole op, six configurations (fused and phased, flat and split, both inverses) | identical | identical |
| Chunk 64, prep hand-off tensors | only `k_dec_t` and `dl` differ (the fix below) | only `k_dec_t` and `dl` differ |
| Chunk 64, whole op against a torch reference | PCC 0.999995 (output), 0.999999 (final state) | same |
| 16 repeated fused runs on each of the five geometries above, T = 2048 | identical run to run | not run |
| LLK asserts and watcher enabled, chunk 32 | plain prep prim passes; the other assert-enabled builds do not fit the kernel-config buffer | prep (SFPU solve and Horner) and fused op pass |

Repo test files under the simulator: `test_chunk_gdn_prims.py` 9 passed, `test_chunk_gdn_fused_geometry.py` 271
passed and 8 skipped, `test_chunk_gdn_fused.py` 71 passed and 15 skipped, `test_chunk_gated_delta_rule.py` 88 passed
and 108 skipped. The skips are tests the suite marks hardware-only when it runs under the simulator, and geometries
that do not fit the simulated grid.

## What changed

### Prep kernel: one fused SFPU pass per block, over only the lanes that are read

Each prep block lands FPU results in the destination register (DST) and then finishes on the SFPU. The baseline
chained LLK one-op tile passes (`sub_binary_tile`, `exp_tile`, `mul_binary_tile`, `rsqrt_tile`, ...). Each of those
loads, computes and stores all 32 SFPU vectors of a tile, so a five-op chain made five round trips over the whole
tile.

The new `gdn_sfpu` namespace has one hand-written pass per block. A pass loads each vector once, applies the whole
chain in registers, and stores once. It also skips lanes nothing reads:

| Block | What is skipped | Why it is safe |
|---|---|---|
| Decay vectors (`decay_exp`, `decayfac`, `dl`) | all but column 0: 8 of 32 vectors | they are column vectors, and only column 0 is read downstream |
| Decay mask `L_mask`, diagonal tile | the upper-right face is stored as zero instead of computed | the chain leaves exactly +0 there |
| q and k normalisers | the two off-diagonal faces are stored as zero | the result is a diagonal matrix |
| Solve input (Blackhole path) | the upper-right face, the diagonal mask and the negation | see next section |

Per chunk on the production path this takes the SFPU from about 800 vector round trips (96 of them `exp`, 64
`rsqrt`) to 248 plus 40 plain zero stores (40 `exp`, 32 `rsqrt`). These are static counts from the code. The lanes
that are read go through the same SFPU operations in the same order as before, which is why the results are
bit-identical.

### Blackhole solve path: un-negated factor

The SFPU forward-substitution solve reads only the strict lower triangle of its input. The baseline still built the
full masked, negated matrix the Horner inverse needs and asked the solve to treat it as negated. The new
`wy_factor_lower` produces only the lower-triangle faces, un-negated, and the solve is instantiated with
`L_NEGATED = false`, which flips the sign bit as it loads. The Horner path keeps the full matrix.

### Prep kernel: block order

A block that reads another block's packed result has to wait for it, and a wait placed directly after the
producing block idles the unpacker for that block's math and pack latency. The blocks are reordered so an unrelated
block usually sits between a producer and its first consumer. No block's operands or arithmetic change.

The triangular solve is the deliberate exception. It stays directly behind the block that produces its input, with
the wait in between. See [Hardware notes](#hardware-and-toolchain-notes) for why.

Because the push order of the seven outputs changed, `writer_chunk_gdn_prep.cpp` and `writer_chunk_gdn_fused.cpp`
now drain and wait in the new order.

### Scan kernel: unpack a shared operand once per row

The scan step's three matmul blocks computed each output tile with its own `matmul_tiles` call, which unpacks the
left operand tile every time even though one row of output tiles shares it. The new `mm_tiles_row` unpacks the left
operand once and streams the row's right operand tiles past it. At a 4-tile slice this cuts the tile unpacks per
step from 152 to 95. The same helper is used in the generic `mm()` for non-transposed products.

It is built from `llk_unpack_AB_matmul` and `llk_math_matmul` directly, not from `matmul_block`. On Blackhole
`matmul_block` can switch to a firmware-throttled sequence that reorders the fidelity phases, which would make the
sums depend on throttle state. Each DST tile receives exactly the instruction sequence `matmul_tiles` would issue.

### Code size

GCC inlined and duplicated the new SFPU bodies at first, which grew the chunk-64 programs past the limit in
instrumented builds. Each pass is now a `[[gnu::noinline]]` function template whose DST tile indices are template
parameters, so there is exactly one copy and the load and store addresses are instruction immediates. `ewt`, which
the Horner inverse calls at seven sites, is also out of line.

## Bugs fixed

### `chunk_size=64` produced wrong results

`pack_tile` ignores its tile index unless out-of-order packing is enabled and fills the buffer in call order. The
decay block packed the `dl` tile from the first tile's window, so at two tiles per chunk it landed in slot 1 instead
of slot 2. `dl` and half of `k_dec_t` were wrong, and the final state had no correlation with the reference (PCC
about -0.004).

Nothing caught it because the existing chunk-64 tests compare device paths with each other, and every path shared
the bug. `dl` is now packed from the last window. `test_prep_outputs_vs_torch` gains a `chunk64` case that fails on
the baseline (`k_dec_t` PCC 3.4e-05) and passes now. Chunk 32, the production size, was never affected.

### Scratch ring overrun

Circular-buffer pointers wrap only when they land exactly on the ring limit, and a multi-tile block is addressed
linearly from the write pointer. The prep kernel pushed a 4-tile block into `scr1` after single-tile pushes, so the
block periodically ran past the end of the ring into the neighbouring scratch buffers. It was harmless only because
those buffers happened to be idle at that moment.

With LLK asserts and the watcher on, the baseline trips `CB push_back: fifo_wr_ptr would exceed fifo_limit` on the
prep kernel's pack thread under Blackhole ttsim. On the n150 the baseline's assert-enabled build does not fit the
kernel-config buffer; the trip was first seen there on an intermediate build with the same push sequence.
The 4-tile block now goes through `kbeta`'s ring, which only ever takes 4-tile pushes, and the chunk-64 path's one
extra single-tile copy goes through `s3`. The assert-enabled chunk-32 runs are clean. Chunk-64 builds with asserts
exceed the buffer on both architectures, so that path's routing rests on the alignment argument, not on a run.

## Hardware and toolchain notes

These are the rules that shaped the change. The first one comes from reading the Tensix ISA documentation
(`tenstorrent/tt-isa-documentation`) and the LLK sources. It was not observed failing on silicon.

- **SFPU stores into undefined DST rows.** Each DST row has a valid bit. An SFPU store into a row no FPU
  instruction has written must set that bit, and the update can be lost if it lands in the cycle the pack thread
  zeroes and releases the other DST half. The packer then reads the row as zero. The new passes only store into
  rows an FPU instruction wrote earlier in the same acquire. That is why the decay block issues each matmul twice.
- **The Blackhole triangular solve does store into undefined rows** (LLK `triangle_solve_tile`). It is safe only
  if every earlier DST section has been released, which the wait directly before it guarantees. Any block inserted
  between that wait and the solve reintroduces the race. The simulator cannot show it.
- **SFPU geometry.** One load or store moves 4 rows by 8 columns, even or odd columns. A 16x16 face is 8 vectors
  and a tile is 32, so restricting a pass to a column or to some faces is a direct saving.
- **`pack_tile` is sequential** by default, whatever index it is given.
- **Circular buffers** must only see pushes of one size per ring when multi-tile blocks are used.
- **Kernel-config buffer.** All kernel binaries of a program share 70,656 bytes. Watcher, profiler and
  LLK-assert builds are several KB larger than normal builds, so all of them need checking.
- **`ttsim`** is bit-exact against silicon and strict about undefined behaviour, which makes it a good check for
  uninitialised DST reads. It says nothing about timing or timing-dependent races.

## Open items

1. **Blackhole silicon run.** Profile the fused op and run the hardware-only repeat tests. This gives the real
   gain and exercises the solve-ordering hazard above on hardware.
2. **Fused cost model.** `w_p_us`, `t_step_us` and `t_phased_us` in `chunk_gdn_device_operation.cpp` were
   calibrated on Blackhole hardware for the old kernels. Both producers and receivers are faster now, so the
   geometry the model picks may no longer be the best one. They were left unchanged because they cannot be
   recalibrated without the hardware.
3. **Blackhole profiler build at chunk 64** was not size-checked with the new code.
4. Possible follow-ups, not attempted: fold the negation of `nkd` into its neighbours to remove 128 SFPU vector
   round trips per chunk, and make `triangle_solve_tile` itself write its scratch rows on the FPU first so callers
   do not depend on a preceding drain.

## Reproducing

The scripts are in `tech_reports/GDN/tools/`. They need a built tree, `TT_METAL_HOME`, and
`PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools`. Tensor dumps go to `$GDN_OUT`.

```bash
T=tech_reports/GDN/tools
CSV=generated/profiler/.logs/profile_log_device.csv

# Per-kernel timing (prep_item, scan_step zones); vary --nv for the slice width
TT_METAL_DEVICE_PROFILER=1 python $T/gdn_harness.py --tag zones --path fused --bh 4 --hk 2 --t 256 --nv 1 --np 2
python $T/prof_zones.py $CSV --zones prep_item,scan_step --skip-first 2

# Fused op device time, 64 chunks
TT_METAL_DEVICE_PROFILER=1 python $T/gdn_harness.py --tag e2e --path fused --bh 8 --hk 4 --t 2048 --nv 2 --np 5
python $T/prof_opdur.py $CSV 64

# Bitwise comparison of two builds: dump with one build, then --compare with the other
python $T/gdn_prep_dump.py --tag base_norm --norm                  # seven prep hand-off tensors
python $T/gdn_prep_dump.py --tag new_norm --norm --compare base_norm
python $T/gdn_harness.py --tag base_fused --path fused --nv 1 --np 2
python $T/gdn_harness.py --tag new_fused --path fused --nv 1 --np 2 --compare base_fused

# Program size against the 70,656-byte limit; repeat with TT_METAL_WATCHER=10 and TT_METAL_DEVICE_PROFILER=1
TT_METAL_LOG_JIT_TELEMETRY=1 python $T/gdn_harness.py --tag sz --path fused --bh 2 --hk 1 --t 64 --nv 1 --np 1 \
    | grep program_config_size.total.TENSIX
```

For Blackhole without hardware, download `libttsim_bh.so` for the tag in `tt_metal/ttsim-version` from the
`tenstorrent/ttsim` releases, place it as `libttsim.so` next to a copy of
`tt_metal/soc_descriptors/blackhole_140_arch.yaml` named `soc_descriptor.yaml`, and export
`TT_METAL_SIMULATOR=<dir>/libttsim.so TT_METAL_SIMULATOR_HOME=<dir> TT_METAL_SLOW_DISPATCH_MODE=1
TT_METAL_DISABLE_SFPLOADMACRO=1`.

The raw profiler logs and tensor dumps behind the tables (about 730 MB) are not checked in.
