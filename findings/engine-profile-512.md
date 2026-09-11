# Per-engine profile at 512-wide (read 2026-09-07 from profiles captured 2026-09-06)

Source: `~/tf/runs/prof_fwd512/summary.txt` and `~/tf/runs/prof_bwd512/summary.txt` on the Trn2 box (test4),
captured 2026-09-06 19:43-19:55 UTC against the 512-wide compiled graph, `nc_idx 9`, LNC=2. The summaries were
produced by `neuron-explorer view --output-format=summary-text` (the tool formerly called `neuron-profile`,
still installed under both names at `/opt/aws/neuron/bin/`).

**This file exists because the per-engine data we kept saying we could not see was already on disk.** Nothing
new was captured to write it; `neuron-top`'s single utilization number is an aggregate, but the profiler
reports every engine separately and always has.

## How to regenerate / extend

```bash
neuron-explorer view --output-format=summary-text --output-file=summary.txt <flags for NEFF + NTFF>
# also supports: summary-json | json | parquet | perfetto | db
# `neuron-explorer view --help` - ALL inputs must be passed as flags, positional args are rejected
```
Existing artifacts on the box: `runs/prof_fwd512/`, `runs/prof_bwd512/` (NTFF + summary + a 7.3 GB profile.json),
`runs/nprof/<instance>_pid_<pid>/<id>/profile_nc_{0..3}_session_*.ntff` (4 cores, from `neuron_profile_step.py`).

## The numbers

| metric | forward | backward |
|---|---|---|
| total_time | 0.1037 s | 0.2538 s |
| **dma_active_time_percent** | **72.6%** | **74.2%** |
| dynamic_dma_active_time_percent (sums over queues) | 131.0% | 119.0% |
| tensor engine (PE) active | 41.5% | 47.1% |
| scalar engine (Act) active | 46.6% | 36.7% |
| vector engine (Pool) active | 35.5% | 41.1% |
| gpsimd engine active | 42.3% | 26.4% |
| sync engine active | 23.8% | 23.1% |
| **total_active_time_percent** | **99.9%** | **99.8%** |
| mfu_estimated_percent | 18.6% | 28.5% |
| **mfu_max_achievable_estimated_percent** | **13.3%** | **14.8%** |
| **transpose_flops / hardware_flops** | **23.1%** (227 / 984 GFLOP) | 6.8% (212 / 3103 GFLOP) |
| mm_arithmetic_intensity | 29.2 | 32.6 |
| dma_packet_time (cumulative over queues) | 1.438 s | 4.403 s |
| hardware dynamic DMA packets | 16.34 M | - |
| sbuf_read_bytes | 80.1 GB | 248.1 GB |
| psum_write_bytes | 32.9 GB | 103.6 GB |
| hbm_read_bytes | 9.5 GB | 53.2 GB |
| throttle_activity_1_active_time_nc5 | 72.6% | 61.2% |

fwd + bwd total_time = 0.357 s, consistent with PR7's measured fwd_bwd 0.385 s at 512-wide.

## What it says

1. **The "70% utilization" in `neuron-top` is the DMA engines, not compute.** DMA is active 72.6% / 74.2% of the
   time; the tensor engine is active 41.5% / 47.1%. The number we had been reading as headroom is data movement.
2. **No engine is saturated and none is idle** - all five sit between 23% and 47% *simultaneously*, with
   `total_active_time_percent` at 99.8%. That is the signature of a **data-movement-bound** graph, not a
   compute-bound one and not a scheduling problem.
3. **The graph's own ceiling is ~15% MFU.** `mfu_max_achievable_estimated_percent` = 13.3% fwd / 14.8% bwd is the
   profiler's estimate of the best this graph could do if perfectly scheduled. We are not losing 85% of the
   machine to bad luck; the graph is shaped so 85% is unreachable. Raising it requires changing what the graph
   does, not how fast it runs.
4. **23% of forward tensor-engine FLOPs are transposes.** A quarter of the forward's matmul budget is spent on
   layout rather than model math. The backward is only 6.8%, so this is specifically a *forward* problem - look at
   `tr = lambda x: x.transpose(-2, -1).contiguous()` (`train3_submission.py:392`) and the (b,h,t,d) <-> (b,h,d,t)
   conversions around attention.
5. **Arithmetic intensity is 29-33** against 328 GB of SBUF reads and 136 GB of PSUM writes per profiled unit.
   Raising it means more work per SBUF load - which is exactly what the RMSNorm and RoPE kernels do, and they are
   gated to eager by `torch.compiler.is_compiling()`, so the compiled (submitted) line gets none of them.

## What it does NOT say

- It does not say the model is too small or the batch too small. Memory is ~50% per core (5,954 MB of a 12 GB
  LNC=1 core), but the compile-time peak estimate is the real gate (T15: `NCC_EOOM002` at 28.56 GB estimated),
  and more batch costs steps.
- It does not indict any single kernel. Every engine is at partial occupancy; there is no hot spot to delete,
  which is why per-kernel attribution (`neuron_profile_step.py`) found nothing decisive earlier.

## Consequence for the plan

Every architecture screen in `queue_NKI4/5` operates on the ~15% the graph can reach; nothing in the queue
touches the ~85% it cannot. That matches the run history: every large win came from step time (width 256->512
-0.040, NKI attention eager -0.010, fused optimizer -0.008/-0.010), while loss-side architecture has delivered
0.002-0.008 apiece, sublinearly stacked (L4r3).

Priorities implied, in order:
1. **Transposes and layout in the forward** - 23% of forward PE FLOPs, the largest single identified waste.
2. **Close the eager/compiled split** - the submitted compiled file forfeits NKI attention (S_NKI3: the custom-op
   HOP is not inlined, ~3600 vs ~430 execs/step), the eager line forfeits the fused NEFF. `S_FLEX` is the
   experiment that decides whether one line can hold both; it is the highest-variance item queued.
3. **Fusion for arithmetic intensity** - get `nki_rmsnorm` / `nki_rope` reachable from the compiled path, or
   accept the eager line and fuse there.
4. **Drop `S_WIN` / `S_WINWARM` as queued** - on the explicit path a window is a mask, so they buy the quality
   restriction of a 256-window on a 1024 sequence and none of the compute saving. They only become real through
   the NKI kernel's `bound_min`/`bound_max`.
