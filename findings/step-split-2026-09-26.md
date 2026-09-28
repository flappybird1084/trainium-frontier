# The whole step, measured for the first time (2026-09-26)

Every previous step-time analysis on this project profiled only the **forward and backward graphs** — about
two-thirds of the step. On 2026-09-25 I used that partial picture to argue that 0.9650 was unreachable
("78 ms is the entire addressable budget against 83 ms needed"). **That argument was wrong, and it was wrong
because the other third had never been measured.** This file is the measurement.

## The measurement

`train3_s19.py` with the exact 47-variable v55 reference env (`OPT_SHARD=zero`, `GATHER_BF16=1`, `ADAMW=nki`,
`MUON=fused`, `EMBED=sidecar`, `LAYER=op`, `PRE_BWD=fused`, `WVIEWS=1`), `PROFILE=2 OPT_PROF=1`, 300 s.
306,432 tok/s. Startup confirms `grad_accum_steps=1` and `OPT_SHARD=zero: 9 params (218,103,808 elements)
updated 1/8 per worker`.

```
PROFILE  : fwd_bwd=0.363  grad_sync=0.060  optimizer=0.049  total=0.472
OPTPROF  : muon=0.0050  adamw=0.0052  adamw_flat=0.0084  gather=0.0284
PROFILE2 : embed_gather=3.4  fwd=123.0  bwd=202.5  host=18.3  embed_scatter=12.9   (ms/step)
```

PROFILE2's sections sum to 360.1 ms against the 363 ms phase, so this passes the rule from
`handoff-2026-09-24b.md` ("before believing a per-section profile, check the sections sum to the phase").
All three blocks are sync-bracketed, so total reads 472 ms vs the 461 ms unprofiled step — treat the splits as
indicative and 461 ms as the real step.

### The step, reconciled against the two graph profiles (forward span 103 ms, backward span 200 ms)

| region | ms | note |
|---|---|---|
| forward graph | 103 | 22.1 exclusive, **8.1 idle** |
| forward launch/gap | ~20 | 123.0 measured − 103 span |
| backward graph | 200 | 42.9 exclusive, 4.9 idle |
| backward launch/gap | ~2.5 | already tight |
| host | 18.3 | |
| embedding sidecar scatter + gather | 16.3 | |
| gradient reduce_scatter | 60 | ~44 of it cross-rank barrier wait |
| **parameter all_gather** | **28.4** | **58% of the optimizer phase** |
| Muon / AdamW / small-param flat AdamW | 5.0 / 5.2 / 8.4 | the optimizer *math* is only 18.6 ms total |

**The optimizer math is not the problem** — AdamW is 5.2 ms over 27.3M elements/rank against a ~0.7 ms element
floor (VectorE at 256 fp32 elem/cycle, 0.96 GHz). I briefly hypothesised a fused flat AdamW was worth tens of
ms; it is worth ~4. The optimizer's cost is its **collective**.

## Why the two collectives (88 ms) are immovable

- **They cannot be overlapped with compute.** Ledger `F_PROBE` (2026-09-16): async reduce_scatter/all_gather
  plus a matmul loop measures the **sum** of the two, at every size (50 MB RS 1.16+2.94=4.10 ms;
  537 MB RS 9.47+2.93=12.30; AG 1.09+2.98=4.09). And with one fused forward graph and one fused backward
  graph, gradients never become available incrementally anyway, so there is nothing to bucket.
- **Not byte-bound** (gradient side): halving all 218.1M elements moved `grad_sync` by 0.000.
- **Not byte-bound** (gather side either): `GATHER_BF16` halved its bytes for only ~2.8 ms.
- **Not call count, not ordering, not launch count** — all three separately refuted.
- What is left is the **cross-rank skew** that each of the ~18 rendezvous per step exposes.

## The straggler: the ledger's closure was invalid, and the lever is dead anyway — for a NEW reason

`Q_SKEW`/`Q_OMP1` (2026-09-24) found exactly **two** ranks 19 ms slow, reproducibly, not thread contention.
Still true on v55 (`P_SKEW`, 2026-09-26): `r0=0.353 r1=0.371 r2=0.354 r3=0.354 r4=0.353 r5=0.356 r6=0.355
r7=0.373`, i.e. six ranks at 0.353-0.356 and **r1, r7 at +18 ms**. `skew=0.044`.

**The ledger marked this "DECISIVE - AREA CLOSED / CONFIRMS DEAD" on invalid reasoning.** `Q_ROT`
(`SHARD_ROTATE=3`) removed **r1's** straggle (0.390 → 0.371, confirmed in its SKEW line) and measured 646
steps against the reference's 646. That zero is not evidence the area is dead: **r7 was still at 0.391, so the
barrier still waited for r7.** Removing one of two equal stragglers gains nothing *by construction*. The
experiment tested nothing.

`Q_ROT` did establish something useful the ledger did not draw out: **r1's slowness is DATA-caused** (it moved
when shards rotated) while **r7's follows the rank** (it did not).

**But the lever is dead anyway, on arithmetic nobody had done.** Reducing r7's work means fewer rows, and rows
come in groups of `PACK = 4` (`submit/train.py:81`, asserted at `:3606`), so 16 rows can only drop to 12.
That is a **25% cut on a rank that is 4% slow**, which costs 4 of 128 rows = **−3.1% of the batch** to buy
**+4% steps**. Total tokens ≈ steps × batch, so the net is ~+1%, worth ~−0.0003 — inside the noise. Finer
granularity (`PACK=2` → 14 rows, `PACK=1` → 15) still overshoots the 4% target and degrades packing.

**Rule this yields: a straggler is only worth fixing if you can make the slow rank FASTER. Giving it less data
trades away exactly the tokens you saved.** That is the correct reason this axis is closed, and it generalises.

Also retracted, my own: I hypothesised the two stragglers were hyperthread siblings, since the box is 12 vCPUs
= **6 physical cores** running **8 ranks**, so two cores must host two ranks each. The arithmetic kills it:
host work is 18.3 ms inside a 466 ms step = a **4% CPU duty cycle**, so two ranks sharing a core would collide
rarely and could not produce a consistent 18 ms. Not tested; not worth testing.

## The embedding scatter plan is 78.7% padding (measured, and its docstring is wrong by 3.5x)

`scatter_rounds` (`submit/train.py:3033`) documents "~150 rounds for token ids". Measured on the real cached
corpus at the production shape (12 micro-steps, `make_dataloader(tok, 4, 4096, 'train', tokenizer_threads=1)`,
16,384 real tokens each):

```
ROUNDS         min/mean/max : 403 / 520 / 573        <- docstring says ~150
PADDED SLOTS   mean         : 76,928 for 16,384 real  -> 78.7% PADDING
distinct tokens             : 3,308 of vocab 8192
max multiplicity            : 520                     <- this is what forces the round count
```

The mechanism: one round per occurrence-rank, each padded up to 128 slots, so the token that appears 520 times
forces 520 nearly-empty tiles. Alternatives, optimally packed into 128-distinct tiles
(`max(ceil(rows/128), max_multiplicity)`):

| approach | optimal slots | vs current 76,928 |
|---|---|---|
| bin-pack only | 67,627 | 88% (only −12%) |
| + dense top-16 | 12,800 | 17% |
| **+ dense top-32** | **11,755** | **15%** |
| + dense top-64 | 10,432 | 14% |

**[2026-09-28: SHIPPED in v70 (commit 0c534f9, gated -0.00088) as scatter_packed + SCATTER_DENSE_K=32; the live plan is 16,512 slots, ~0.8%% padding, capacity-bound at 128 tiles. Nothing left to win here. The 78.7%% figure below is the OLD scatter_rounds planner.]**

**Bin-packing alone is nearly worthless** — the 520-multiplicity is the binding constraint. Bin-packing *plus*
a dense path for the ~32 most frequent tokens is what collapses the plan. At the ~4.4 µs/tile implied by the
measured 12.9 ms embed_scatter (601 tiles x 4 tables = 2,404 tile-tables), this deletes ~2,036 tile-tables =
**~9 ms**, i.e. ~2% of the 424 ms step ≈ **−0.0006**, for an estimated 4-6 h build reusing the existing
`embed_scatter_add_v2_x4` kernel with a shorter plan (no new NKI kernel). Exact in real arithmetic: fp32
accumulation either way, and PSUM accumulation over a dense tile is arguably better-conditioned than 520
sequential adds.

Also note **the step is 424 ms, not 461**: 4243 steps / 1800 s. The 461-485 ms figures throughout `findings/`
are all from `PROFILE=1` runs, which inflate the step ~10%. Every per-section number in
`exclusive-time-method-2026-09-25.md` sits on that inflated step, so scale percentages accordingly.

## What this leaves

Attackable, after the 88 ms of collectives and the 18 ms of skew are excluded: forward launch gap ~20,
forward internal idle 8.1, host 18.3, embedding scatter 12.9, small-param flat AdamW 8.4, backward exclusive
top items ~3.6 = **~71 ms nominal**. Against 83 ms needed for 0.9650, and against this project's historical
record of paired phase deltas **over-predicting gated step counts by ~5x**. So the step-time axis alone does
not reach 0.9650, and the honest expectation from it is single-digit milliseconds of real gain.

**Conclusion: the route to 0.9650 is quality-side, not throughput-side.** See
`findings/quality-levers-2026-09-26.md`.

## Method notes worth keeping

- **A vehicle run with default knobs is a different model.** My first attempt at this measurement used
  `train3_s19.py` with only `PROFILE`/`OPT_PROF` set and no config env, which silently ran a different
  hyperparameter set: 52,584 tok/s against the real 306,432. Always pass the full reference env (recover it
  from a known-good run's `runs/<ID>.env` rather than retyping it).
- **Same-session paired reproducibility is ~0.00005, far tighter than the 0.00025 per-gate figure.** Two
  independent 300 s screens of the identical config: 673 steps / 1.101130 and 673 steps / 1.101177. The
  0.00025 number is cross-session; paired same-session arms can resolve much smaller effects than the ledger's
  warnings imply.

## CORRECTION (2026-09-27): the "4% CPU duty cycle" argument above is wrong

The retraction two sections up kills the hyperthread-sibling hypothesis with "host work is 18.3 ms inside a
466 ms step = a **4% CPU duty cycle**". Measured directly against a live 8-rank run (`/proc/<pid>/stat`,
`ps -o %cpu`):

```
8 rank processes, EVERY one at 96-101% CPU      (utime ~14000 ticks vs stime ~1800 -> ~89% USER time)
main thread parked in futex_do_wait             -> the CPU is burned by a helper thread, not the main thread
/sys/.../thread_siblings_list: 0,6  1,7  2,8  3,9  4,10  5,11
Cpus_allowed_list = 0-11 for all 8 ranks        -> no affinity pinning by torchrun or the Neuron runtime
```

So the host is **saturated, not 4% busy**: eight continuously-busy threads on **6 physical cores**, which forces
two physical cores to carry two ranks each. The duty-cycle arithmetic was measuring the wrong thing - `host=18.3`
is the *loader* section of the step, not the process's CPU consumption.

**This does not resurrect the lever, and the axis stays closed** - for the reasons that were always the real
ones: (a) the `PACK=4` arithmetic (a 25% row cut to fix a 4% deficit costs the tokens it saves), and (b) 8 ranks
on 6 cores is *forced*, so no pinning arrangement exists that removes the sharing, and `LNC=2`/`NPROC=4` (which
would) is separately dead at **0.654 vs 0.485 s/step** (`Q_LNC2`). Tokenizer threads 32 -> 8 also left skew
unchanged. Prize is bounded at ~18 ms ≈ −0.0012 with no known intervention.

**What to take from it:** do not reuse "the host is only 4% busy" to dismiss any host-side mechanism. The host
is pegged. The reason host latency does not matter is the *measured* one - `D_QDEPTH` found 311.6 ms of device
work already queued when the host leaves to fetch a batch - not a duty-cycle estimate.
