# DMA is bytes-linear at line rate: the descriptor axis is closed (2026-09-27)

This file closes the "we are DMA-descriptor-bound" hypothesis with a direct hardware cost measurement, and it
closes the attention-layout repack (`ATT_V4` and its successors) permanently. Ledger rows **660**
(`R_BTE_RETRACT`) and **661** (`M_DMACOST`, which corrects row 659) are the index entries.

## The governing fact

**DMA cost on this box is ~95% bytes and the marginal rate is the hardware line rate.** Measured from
`runs/prof_V55INSTR/parquet/prof/DmaPacket.parquet`, 13,603,860 individual packets with per-packet
`transfer_bytes` / `duration_ns` / `queue_name` (19.51 GB, 1006.1 ms summed over 16 DMA engines):

| bytes | packets | mean ns | ns/KiB | summed ms |
|---|---|---|---|---|
| 4 | 1,932,473 | 14.20 | 3634 | 27.4 |
| 256 | 4,940,160 | 17.83 | 71.3 | 88.1 |
| 1024 | 2,467,129 | 55.91 | 55.9 | 137.9 |
| 2048 | 1,920,256 | 114.26 | 57.1 | 219.4 |
| 4096 | 1,869,456 | 192.18 | 48.0 | 359.3 |
| 8192 | 45,072 | 343.24 | 42.9 | 15.5 |
| 12288 | 292,352 | 507.31 | 42.3 | 148.3 |

A linear fit over 256–12288 B gives **7.4 ns fixed per packet and 41.7 ns/KiB marginal**. Over a 48× size range
ns/KiB improves only **1.69×**. And 1 KiB / 41.7 ns = 24.6 GB/s per engine × 16 = **393 GB/s**, against the
368 GB/s per-core peak — the engines are at line rate, not at 60% efficiency. Load is balanced to within 0.1%
across all 16 engines (629.8 MB each on Q1), so there is no straggler queue either.

Total DMA wall ≈ 1006 ms / 16 engines ≈ **63 ms of a 424 ms step**.

No new capture was needed for any of this: the parquet was already on disk and the analysis is CPU-only pyarrow,
run with the device still training. **Check `runs/*/parquet/prof/` before queueing a profiling job.**

## What this kills

**A descriptor-count or packet-count lever cannot exist.** With a 7.4 ns fixed cost against 41.7 ns/KiB, a
packet has to be under ~180 B before overhead dominates. Everything we move is 256 B or more per partition, so
deleting descriptors at constant bytes buys ~nothing.

Byte shares from the `src_elem_size` field of all 102,556 `DMA_DIRECT2D` instructions (it is parseable out of
the `operands` string: `src_elem_size=12288 src_pattern=[16384,1][128,1]`):

| bytes/partition | share of instructions | share of bytes |
|---|---|---|
| 256 (the `(128, HD=128)` attention tiles) | **39%** | **7.48%** |
| 1024 | | 30.12% |
| 2048 | | 16.06% |
| 4096 | | 22.22% |
| 12288 | | 23.01% |

The 256 B tiles are 39% of DMA instructions but take 8.75% of packet time to carry 6.5% of bytes. The excess is
2.25 points = 22.6 ms summed / 16 engines = **~1.4 ms of wall.**

### The attention-layout repack is permanently closed

The plan was to read four heads in one `(128, 512)` DMA instead of four `(128, 128)` DMAs, which needs the
`(T, D)` layout, which needs either a strided DMA on a graph-produced tensor (**tracer rule #11 — NaN'd twice
as `ATT_V4`, including with `SIDECAR_SYNC=1`**) or a full repack after the vendor attention backward that
`layer_bwd`'s own comment prices at ~12,288 instructions per step.

**Its entire prize is 1.4 ms, and the repack would cost bytes it does not save.** Do not reopen this. The
descriptor count was never the cost; three separate attempts were paying to fix a number that does not bill.

### The 4 B index tiles are also not a lever

1.93 M packets, **14% of all packets, 0.04% of all bytes**, and **~1.7 ms of wall if every one of them
vanished.** The `(128, 1)` uint32 index loads in `embed_gather_v2` / `embed_scatter_add_v2` are the worst-filled
transfers in the codebase (3634 ns/KiB, 87× the marginal rate) and they still do not matter, because there are
no bytes behind them. An `embed_gather_v2_x4` sharing one index load across the four token tables is still
mildly worth doing for its *launch* count, but its DMA prize is a fraction of 1.7 ms, not the 0.3–0.8 ms I
first estimated on packet-rate reasoning.

## Two corrections to how we had been reading the earlier profile

**1. `dynamic_dma_active_time_percent` = 119–131% was never evidence of saturation.** It is a *sum over
queues*, so 131% means about 1.3 queues busy on average. The metric that carries any argument is
`dma_active_time_percent` (72.6% forward / 74.2% backward) — the least-idle resource, but not saturated. The
"DMA is the only saturated resource" framing was an artifact of adding up per-queue percentages.

**2. The 78.7%-padding scatter plan is already fixed, and its two knobs are now dead levers.** That measurement
(`step-split-2026-09-26.md`) is of `scatter_rounds`, the *old* planner. v70/v71's `scatter_packed` is the fix
that file proposed and it shipped: the token plan is 128 tiles / 16,384 slots for 16,384 real occurrences,
about 0% padding, down from 76,928 slots.

It also cannot be improved further. The slot count is `max(ceil(m/128), max_multiplicity)` rounded up to a
multiple of `SCATTER_BINS`, and `ceil(16384/128) = 128` is now the binding term — the plan is **capacity-bound,
not multiplicity-bound**. So raising `SCATTER_DENSE_K` above 32 or changing `SCATTER_BINS` changes the tile
count by zero. Both knobs are at their floor; do not sweep them.

## The one correction the cost model forces on our bf16 intuition

I had written that bf16 on a `(128, HD)` attention tile "would go 512 → 256 B and buy nothing." **That is wrong
under a bytes-linear cost**: 512 B interpolates to ~28.2 ns/packet and 256 B measures 17.8, so halving the bytes
saves ~37% of that transfer's time regardless of packet fill. **Bytes are the only thing that ever buys
anything here.** Which is also the reason `BTE_BF16` is directionally right — see
`findings/bte-bf16-2026-09-27.md`.

## What is left on this axis: essentially nothing

19.51 GB/step/core, 63 ms of DMA wall in a 424 ms step. **Halving every byte in the step buys 31 ms = 7.4% =
about −0.0026.** Getting there means fewer or smaller activation tensors, i.e. SBUF-resident fusion — already
gated **2.4% slower**. There is no partial version of that which pays.

One item survives, and only on byte accounting, because it is too small to gate:

**`adamw_kernel` and `muon_kernel` each write the updated parameter twice** — once in place to `p_in`/`w_in`
(the real update) and once to a full-size `shared_hbm` output tensor whose return value both call sites
**discard**. That is ~109 MB written to HBM every step for nothing (77.6 MB AdamW + 31.4 MB Muon per rank),
0.56% of the step's 19.51 GB, so **~0.3 ms** — unmeasurable in a gate and worth taking only as a passenger on
another change. It also frees ~109 MB of HBM allocation, which is worth something against the compile-time
peak-memory estimate (`NCC_EOOM002` at 28.56 GB killed T15). The safe form follows
`embed_scatter_add_v2`'s established precedent: mutate the input in place and return a small dummy sourced from
the mutated buffer.

## Attribution limits of this capture, stated so nobody re-derives them

- `kernel_instruction_name` and `hlo_name` are **null for every DMA instruction** in this capture, and
  `DmaPacket.id` is a per-packet id that does **not** join to `Instruction.id`. **Per-kernel byte attribution
  is not available from this profile.** A capture with DGE notifications enabled would be needed.
- `prof_V55INSTR` predates v70/v71's `scatter_packed`, so the byte *totals* are stale. Every conclusion above
  rests on the ns-per-byte cost model — a hardware property — plus share ratios, not on v55 absolute counts.

## Method lesson

I derived a packet-rate cost model from two consistent calibrations (a static DMA-instruction count that
matched the profile's packet count to 12%, and a per-tile cost implied by the measured `embed_scatter` time) and
it was **wrong by 10×** in its prediction for the attention-layout lever — 58 ms against the measured 1.4 ms.
Both calibrations were consistent with each other *and* with the truth; what they could not do is distinguish
"cost per packet" from "cost per byte", because in our workload the two are nearly proportional. The
discriminating measurement was a per-packet size-vs-duration table, and **it already existed on disk.**

The rule: a cost model calibrated on aggregates cannot separate two factors that co-vary in the aggregate.
Before building on one, find the measurement where they come apart — here, a 48× size sweep — and check whether
it is already captured.
