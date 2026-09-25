# The step is LATENCY-bound. It is NOT element-, FLOP-, byte- OR instruction-bound (2026-09-24, night)

> **READ THIS BOX FIRST — the original title of this file was "The step is INSTRUCTION-bound" and that was
> WRONG.** I deleted **176,640 of the 461,449 VectorE instructions (38%)** with verified bit-identical math
> (`ROPE_WIDE`, `train3_s18.py`) and the step did not move: **646 steps vs 646, total 0.482 vs 0.482 s**.
> That is ~11 ns recovered per deleted instruction, against the 104 ns the AWS docs state and the 360 ns the
> profile average implied.
>
> **The error to learn from: per-instruction DURATION in the profile is not critical-path time.** An
> instruction can take 374 ns and cost the wall nothing, because no engine is saturated (vector 69%, tensor
> 49%, scalar 48%, dma 40%) and the wall is set by the **dependency chain through the graph**, not by any
> engine's throughput. Section 2(a)/(b) below reason from instruction duration and are therefore unsafe;
> sections 1, 3, 4 and 5 are measurements and stand.
>
> This retro-explains every other null of 2026-09-24 — bf16 gradients, back-to-back collectives, the collapsed
> scalar AdamW, the bf16 bigram table — all were THROUGHPUT optimisations on a LATENCY-bound machine. It also
> agrees with the oldest note in the project memory ("Sep 10 profile: graphs latency-bound, all engines
> 30-60%"), which I rediscovered the expensive way.
>
> **The only lever consistent with this diagnosis is FEWER, LARGER KERNELS** (a shorter dependency chain).
> This file has already had that program applied (`LAYER=op` fused layers, `PRE_BWD=fused`). What remains is
> the vendor attention backward (101.5 ms, 21% of the step, no external lever) and the graph's serial
> structure. `ROPE_WIDE` is kept in `train3_s18.py`, knob-off by default, as a verified-correct dead end.

This file answers "why is MFU ~15% and is that a ceiling?" with a real NEFF instruction profile plus the
documented hardware numbers. It supersedes the engine-utilisation reasoning in
`findings/mfu-profile-2026-09-21.md` and the element-pass model in `findings/vector-deletions-2026-09-23.md`
(both are still correct about *ranking* deletions; they were wrong about *why* the step costs what it does).

## 1. The hardware, from AWS's own docs (not inferred)

Per **NeuronCore-v3** (8 per Trainium2 chip; `trn2.3xlarge` = 1 chip, 96 GB HBM @ 2.9 TB/s, 12 vCPU):

| unit | figure |
|---|---|
| Tensor engine (PE) | 128x128 systolic array @ 2.4 GHz; **79 TFLOP/s dense bf16**, 158 cFP8, 20 fp32 |
| Vector engine | 0.96 GHz, 128 lanes; **512 elements/cycle bf16/fp16, 256 for fp32**; ~1 TFLOP/s fp32-class |
| Scalar engine | 1.2 GHz, 128 elements/cycle |
| SBUF | 28 MiB = 128 partitions x **224 KiB/partition** |
| PSUM | 2 MiB, 8 banks, free dim capped at **512 fp32** per bank |

**The Vector engine is ~66-80x weaker than the PE.** Any material share of step time landing on VectorE dominates
wall clock long before the PE saturates — so at hidden 512, where elementwise work scales as O(width) and matmul
as O(width^2), being vector-bound is *structural*, not a kernel-writing failure.

**The documented instruction tax** (NKI Performance Guide, quoted): "When VectorE is running many back-to-back
instructions either with N << 128 [elements/partition] or with every instruction depending on the output tile of
the previous instruction, a static instruction overhead of **~100 engine cycles** should be added"; and "engines
typically require at least **128 elements/partition** to be efficient." ~100 cycles @ 0.96 GHz = **~104 ns**.

AWS publishes **no** achieved-MFU figure for transformer training on Trn2. Nearest points: a customer quote of
80% on *Trainium3* (no methodology), and an academic *inference* matmul kernel at ~85% of Tensor-Engine peak on
Trn1. Mature-CUDA reference is 40-55%. SemiAnalysis attributes Trainium's lower MFU to the software-managed
SBUF, deliberately lower arithmetic intensity (225.9 bf16 FLOP/byte vs 300-560 on H100/GB200), and cross-engine
dependency bubbles between the four specialised engines.

## 2. The profile (`trn2_profile_one.sh` on a v52-config training NEFF, 2nd execution)

```
engine     instr-time   instructions        opcode                ns/instr
Tensor      238.07 ms    1,067,870          MATMUL                   337
Vector      166.00 ms      461,449          TENSOR_TENSOR             374
Scalar      132.44 ms      349,397          LDWEIGHTS                 108
GpSimd       39.52 ms       77,694          COPY                      339
Sync         26.85 ms       58,021          ACTIVATE                  449
                                            SCALAR_TENSOR_TENSOR      545
whole-graph active: vector 69%,             TENSOR_REDUCE             399
tensor 49%, scalar 48%, dma 40%,            MEMSET                    459
gpsimd 18%                                  RANGE_SELECT              805
```

### The two findings that matter

**(a) The ~100-cycle tax is charged INSIDE the instruction, not in gaps between instructions.**
`TENSOR_TENSOR` measures 374 ns, against ~267 ns of pure element time for a 512-wide fp32 pass. Independent
confirmation from this repo's own history: `MLP_WG_GROUP` removed 55,296 `tensor_tensor` at N=512 fp32 and
measured **-22 ms = 398 ns each** — the model fits only with the overhead included.
**Consequence: hand-interleaving independent tiles to hide dependency stalls is worth ZERO.** Only *fewer,
wider* instructions pay. This killed a planned interleaving project before a line was written.

**(b) Element throughput explains only ~38 ms of the 166 ms of Vector time.** The rest — ~128 ms, 77% — is
per-instruction cost, ~277 ns averaged over 461,449 instructions. **Instruction COUNT is the step's currency.**

Note `LDWEIGHTS` (55 ms) **overlaps** `MATMUL` in the PE pipeline: the two sum to 238 ms of duration but the
engine reports 49% active (~118 ms), while Vector's 69% matches its instruction sum exactly. So the 1:1
LDWEIGHTS:MATMUL ratio (no stationary-operand reuse anywhere) costs nothing extra — do not chase it.

Even the PE is overhead-bound: a 128x128x128 matmul needs ~53 ns of array time and measures 337 ns. **That, not
idle silicon, is why MFU reads ~15%.** 540,012 matmuls per step is the real number to attack, and it is set by
the PE's 128-deep contraction, so it is not directly reducible.

## 3. Where the instructions are (cross-tab by `nki_source_location` x `opcode`)

| source | instrs | ms |
|---|---|---|
| **nkilib `attention_bwd.py` (VENDOR)** | **346,410** | **101.53 = 21% of the step** |
| compiler `private_nkl/transpose.py` | ~113,000 | 26.5 |
| our fused layer forward + its glue | ~135,000 | 58.5 |

The vendor attention backward is the **single largest block in the step** and was invisible to every
element-pass audit, because it is library code. Its internals: MATMUL 39.23 ms/126,691 @310 ns, LDWEIGHTS
11.51/101,376, SCALAR_TENSOR_TENSOR 10.93/19,194, COPY 8.28, ACTIVATE 8.05, RANGE_SELECT 7.11/8,840 @805 ns.
It is exactly the ~95 ms of Vector time an independent instruction budget could not account for.

**It has no external lever** (all three checked tonight): its signature has **no tile-size parameters**;
`sliding_window` does not help at seqlen 1024 (below); and our own attention backward measured *slower* than
the vendor twice (`ATTNBWD`, `ATTNBWD_V2`).

## 4. A method warning: one line of the profile nearly cost me a day

`train3_s13.py:5107` is the top single source line — 29.70 ms, 65,777 instructions — and the code on it is
`ws = [lin._stack[lin.idx].to(bf16) for lin in (...)]`. I read that as "we re-cast 50 M weight elements from
fp32 to bf16 every step, 6.2% of the step" and was about to rebuild the weight-stack dtype plumbing.

It is **not** a cast. The opcode cross-tab shows TENSOR_TENSOR 11.32 + MEMSET 7.49 (15,506 instrs) + ACTIVATE
3.01 + MATMUL 2.96 + DMA 1.61 + COPY 1.06 — a *fused attribution* covering several graph ops. The arithmetic
check that caught it: 50 M elements tiled at (128, 512) is ~768 instructions, not 65,777.
**Rule: a per-line attribution in this profile is a bucket, not an operation. Cross-tab by opcode before
believing any of it, and sanity-check the implied elements-per-instruction.**

## 5. Closed tonight, each on a measurement

- **LNC=2 (two-core logical)**: total 0.654 vs 0.485 s/step; 256 steps vs 646. Both configs do the same total
  work (8x16 vs 4x32 rows), so it is **48% worse per unit work** — the hand-written kernels do not exploit the
  two fused physical cores, so half the chip idles. The switch to LNC=1 8-way was right and remains right.
- **Sliding window (vendor native)**: the kernel genuinely skips — `tile_required` is a **compile-time** list
  (`attention_bwd.py:890`), not a mask. But its key tiles are far coarser than 128, so at seqlen 1024 a 256
  window has almost nothing to skip: **649 steps vs 646 (no gain) and bpb +0.0108**. The old `S_C8_WIN`
  discard was right for the wrong reason (it was mask-based); the axis is dead for the *right* reason now.
- **Narrow QK** (`QK_DIM=64`): fails to compile in 34 s — the fused pre-attention kernel hardcodes q/k as four
  128-row blocks (`wq` index range [256,383] exceeds size 256). `QK_DIM` is a torch-path-only knob, which is
  why the ledger had zero rows for it. A code comment claims -0.0075 via +27% steps on a 4080; untestable here
  without teaching the kernels a separate qk tile count.
- **Layer-0 attention drop** (`DROP_ATTN0`): the fused fast path is guarded by `and not skip_attn`, so the flag
  pushes layer 0 onto the *eager* path. 631 steps vs 646. This also retro-explains `G_SMEARDROP` (+0.0032):
  that bundle probably lost because layer 0 went eager, not because the architecture was worse.
- **Bigram sign trick**: already measured a wash twice (`F_V30_BGSIGN`, full gate, -0.00003). The leaderboard's
  +1.8% does not transfer — our table is far larger relative to an 8192 vocab, so collisions are already rare.

## 6. What survives — REVISED after ROPE_WIDE measured zero

**Nothing in the "fewer, wider Vector instructions" family.** The largest member of it was built, verified
bit-identical in the simulator and in fp32 numpy, and measured **exactly zero** (646 steps vs 646). Do not
build the rest of the family (the per-head scalar-chain batching, the `head_bwd` blend folds, the glue fold):
they are smaller versions of a change that just demonstrated a ~0% conversion rate.

The superseded reasoning was:

Fewer, wider Vector instructions. The ranked, modelled set (~28 ms bit-identical, ~55 ms including
statistical items), largest first: **rotary as a handful of 512-wide masked ops instead of ~48 narrow
(32-element/lane) ones per token tile in both pre-attention kernels (-129,024 instructions)**; batching the
eight per-head (128,1) scalar chains into one (128,8) tile and folding `*1/HD + eps` into `activation(scale=,
bias=)`; two `head_bwd` blend pairs into one two-op `tensor_scalar` each; folding the autograd glue's per-layer
scalar passes into the pre-attention store. At this file's historical marginal conversion (27%) that family is
worth ~1.7%; at 60% it is ~7%.

Architecture is NOT the lever — see `handoff-2026-09-24b.md` and the shape analysis: 12x512 sits in an interior
optimum with 0.002-0.005 margins in all four directions, width >512 **fails to compile** ("Matmul moving free
dimension 640 exceeds max 512"), and the asymmetry that explains it is that *lookup* parameters cost only
latency-bound data movement (nearly free at the margin here) while *dense* parameters cost elementwise work,
which is the wall. That is the opposite of the GPU intuition and it is why a 134 M-parameter bigram table is a
bargain and buying width is not.
