# Per-engine profile at 512-wide (read 2026-09-07 from profiles captured 2026-09-06)

Source: `runs/prof_fwd512/summary.txt` + `runs/prof_bwd512/summary.txt` on the Trn2 box, captured 2026-09-06 19:43-19:55
UTC against the 512-wide compiled graph (`nc_idx 9`, LNC=2) via `neuron-explorer view --output-format=summary-text` (the
tool formerly `neuron-profile`; both names at `/opt/aws/neuron/bin/`; all inputs must be flags — positional args are
rejected; also supports summary-json | json | parquet | perfetto | db). Existing artifacts: `runs/prof_fwd512/`,
`runs/prof_bwd512/` (NTFF + summary + a 7.3 GB profile.json), `runs/nprof/<instance>_pid_<pid>/<id>/profile_nc_{0..3}_session_*.ntff`.

**This file exists because the per-engine data we kept saying we could not see was already on disk:** `neuron-top`'s
single utilization number is an aggregate, but the profiler reports every engine separately and always has.

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
1. **`neuron-top`'s "70%" is the DMA engines, not compute** (72.6/74.2% DMA vs 41.5/47.1% tensor). The number we had
   been reading as headroom is data movement.
2. **No engine is saturated and none is idle** — all five sit between 23% and 47% *simultaneously*, with
   `total_active_time_percent` at 99.8%: a data-movement-bound graph, not a compute-bound one and not a scheduling
   problem. There is no hot spot to delete, which is why per-kernel attribution (`neuron_profile_step.py`) found
   nothing decisive.
3. **The graph's own ceiling is ~15% MFU** (`mfu_max_achievable_estimated_percent` 13.3% fwd / 14.8% bwd). We are not
   losing 85% of the machine to bad luck; the graph is shaped so 85% is unreachable. Raising it means changing what the
   graph does, not how fast it runs.
4. **23% of forward tensor-engine FLOPs are transposes** (backward only 6.8% — a forward-specific problem): the
   (b,h,t,d) <-> (b,h,d,t) conversions around attention (`train3_submission.py:392`). The later attention-layout kernels
   (v40) attacked exactly this.
5. **Arithmetic intensity is 29-33** against 328 GB of SBUF reads and 136 GB of PSUM writes per profiled unit. Raising it
   means more work per SBUF load — what the RMSNorm and RoPE kernels do, but they are gated to eager by
   `torch.compiler.is_compiling()`, so the compiled (submitted) line gets none of them.
- Not said here: the model/batch is too small (memory is ~50% per core, 5,954 MB of a 12 GB LNC=1 core; the real gate is
  the compile-time peak estimate — T15 died at `NCC_EOOM002`, 28.56 GB estimated — and more batch costs steps).

## Consequences that still stand
- Every loss-side architecture screen operates on the ~15% the graph can reach: every large win in the run history came
  from step time (width 256->512 -0.040, NKI attention eager -0.010, fused optimizer -0.008/-0.010), while loss-side
  architecture delivered 0.002-0.008 apiece, sublinearly stacked (L4r3).
- The window/curriculum family buys compute only when it rides the NKI kernel's per-query `bound_min`/`bound_max`; on the
  explicit path a window is just a mask (quality restriction, no compute saving).
- Whether one line can hold both the fused NEFF and the NKI custom ops was queued here as `S_FLEX` and decided later.
