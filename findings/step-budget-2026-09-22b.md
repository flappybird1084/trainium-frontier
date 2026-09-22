# The v46 step budget, measured (2026-09-22, session 2) — and the one axis it opens

Fresh `PROFILE=1 PROFILE_SKEW=1 OPT_PROF=1` screen of the v46 config (`P_V46`, vehicle `train3_ko.py`, 300 s, 8-way
LNC=1). This supersedes the coarser split quoted in `mfu-profile-2026-09-21.md` (fwd_bwd 0.416 / grad_sync 0.044 /
opt 0.056), which did not break the optimizer down.

```
PROFILE  (avg s/step, 30 steps): fwd_bwd=0.416  grad_sync=0.053  optimizer=0.053  total=0.522  skew=0.034
OPTPROF  (avg s/step):           muon=0.0051  adamw=0.0053  adamw_flat=0.0089  gather=0.0306
SKEW     per-rank fwd+bwd:       r0=0.416 r1=0.429 r2=0.419 r3=0.418 r4=0.418 r5=0.417 r6=0.419 r7=0.430
```

## Read

1. **The optimizer is 58% collective.** `gather` (the ZeRO param `all_gather`) is 30.6 ms of the 53 ms optimizer;
   the actual arithmetic (muon 5.1 + adamw 5.3 + adamw_flat 8.9 = 19 ms) is the small part.
2. **Per-step fixed overhead = grad_sync 44-53 ms + optimizer 53 ms ≈ 97 ms of a 516 ms step = 19%.** It is
   dominated by two collectives' worth of traffic over the *whole* parameter set, and it is paid **once per
   optimizer step regardless of how many tokens that step covers**.
3. **Rank skew is small** (fwd+bwd spread 416-430 ms = 3%); the 34 ms `skew` figure is mostly device-queue drain
   forced by the profiling barrier, not load imbalance. Nothing to win here.
4. The sharded parameter set is **218.1 M elements**, of which the **bigram table is 134.2 M (62%)** — 537 MB fp32
   reduce-scattered and 537 MB all-gathered every step. The bigram table dominates communication.

## The axis this opens: amortize the fixed 97 ms over more tokens (grad accumulation)

`TOTAL_BATCH` has not been re-tested since the v20/v21 era (2026-09-08), when the step was ~0.94 s and the fixed
overhead was a much smaller fraction. At 0.52 s/step the arithmetic is:

| TOTAL_BATCH | accum | predicted s/step | predicted tok/s | vs now |
|---|---|---|---|---|
| 131 072 (v46) | 1 | 0.516 | 254 k | — |
| 262 144 | 2 | 0.929 | 282 k | **+11%** |
| 524 288 | 4 | 1.761 | 298 k | **+17%** |

(`fwd_bwd` is strictly per-token, so accum *n* costs `n x 0.416 + 0.097`.)

The only question is whether quality per token degrades at the larger batch. The one historical data point says it
does **not** at this scale: 65 k -> 131 k gave +17% tokens and **-0.006 bpb** at the full 1800 s gate (T32 1.040894
@3376 steps -> T34 1.034802 @1978), i.e. the whole throughput gain converted to bpb with no large-batch penalty.
Exchange rate on this line: **1% step time ~ 0.0003-0.0005 bpb.**

Gates launched (queue `queue_batch.txt`, vehicle `train3_ko.py`, full 1800 s): `R_REF` (131 072, paired anchor),
`R_B262`, `R_B524`, then `P_SPLIT` (PROFILE=2 micro-step split).

## Why the other levers in this step budget are not worth taking first

- **Sparse / bf16 bigram gradient comm** (modded-nanogpt record #71 does exactly this) would cut the 62% of comm
  the bigram table causes, worth ~40 ms/step ~ -0.004. **But grad accumulation amortizes the same 97 ms**, so the
  two overlap: if `R_B262`/`R_B524` win, sparse comm is worth only a third of that. Re-price it after the gates.
- **fp8 double-row matmul**: `MLPBENCH` measured **x1.30-1.41** on the exact production MLP shapes and `S_V29_FQ8`
  showed numerics parity, so the old "fp8 ~1%" verdict understates the kernel-level gain. It is still the wrong
  lever: the backward graph is **Vector-bound (Vector 82% busy, Tensor 51%)**, so a Tensor-engine speedup cannot
  shorten the wall. Additionally the backward operands (gradients ~1e-3) sit below e4m3's smallest normal (0.0156)
  and would need per-tile dynamic scaling — more Vector work, on the engine that is already the constraint.
- **Width 640** would trade Vector work (scales with D) for Tensor work (scales with D^2) and so fits the measured
  engine imbalance — but it is closed on implementation, not just on score: **the kernels cap D at 512**
  (`plan-097.md`), and width 640 lost four separate times.
- `SYNC_EVERY`, engine rebalance, coalesced collectives, optimizer/comm overlap, backward pre-transposes (<1%),
  compiler flags: all previously closed with device numbers. Not re-litigated.

## MFU, for calibration

Trn2 = 8 NeuronCore-v3; NC-v3 bf16 ~79 TFLOPS => ~630 TFLOPS/chip. Model FLOPs (dense matmul + attention) are
~5.98e12 per micro-step per core, i.e. ~115 TFLOP/s for the chip = **~18% whole-step MFU** (the compiled graphs
alone measure ~25%). Closing the ~0.03 bpb gap to the leaders needs ~2x tokens, i.e. ~36% MFU. Nothing in this
budget delivers that incrementally; grad accumulation is worth ~11-17% and is free.
