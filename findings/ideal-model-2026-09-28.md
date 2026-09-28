# The ideal model for Trn2, and what can be built by Sep 30

**Bottom line.** A 100%-MFU step would take 77 ms, not 400 ms. That is 5.5x the tokens, worth about -0.050 bpb. Even
with perfect graphs, the collectives, optimizer and launch costs (about 136 ms) cap the gain at about -0.020. The
hardware prefers wide, shallow layers (d = 1024 and fewer layers). Two measured depth results make shallow layers lose
more quality than they win back in speed. **No shape change reaches -0.0045 by Sep 30.** Best buildable throughput work:
about -0.0005 to -0.0015.

## 0. A premise to correct first

The "instruction-count bound" title of `instruction-profile-2026-09-24.md` is **retracted inside that same file**.
ROPE_WIDE deleted 38% of Vector instructions and the step did not move. Ledger row M_UNIONBUSY (09-27) measured busy
time per engine as merged wall-clock intervals. In the backward graph (181.9 ms span): Vector 123.5 ms (68%), PE 98.2 ms
(54%), Scalar 95.8 ms (53%), about 2.1x parallelism, and nothing saturated. Summed per-instruction durations come to
581 ms for that 182 ms graph, because the per-instruction `duration_ns` overcounts by 2.4x on the PE. So "count times
measured ns per instruction" **cannot** reproduce the step. The model below uses union-busy time instead. Also, the
vendor attention costs **56.5 ms/step** of real wall time (ATTNPLACE), not 101 ms.

## 1. Cost model and calibration

**Model FLOPs per core per step** (16,384 tokens, causal):
- Forward: 16,384 x [L(4+2r)d² + dV + L·T·d] = 9.96e11 MAC.
- Backward: 2x the dense part plus 2.5x the attention part = 2.04e12 MAC.
- Total 6.08e12 FLOP. At the 79 TFLOP/s bf16 peak that is **77 ms**.

**PE (irreducible class).** One MATMUL does at most 128 (contraction) x 128 x 512 (moving free dimension, the NC-v3
cap) = 8.39M MAC in 213 ns of array time. So the instruction floor is MAC / 8.39M.
- Backward: 243.7k full instructions minimum, against 533.9k measured. About 221k of those are transposes (a transpose
  is an identity MATMUL with N = 128).
- Fit: 51.8 ms array time + 221k x 53 ns (transposes) + 533.9k x **65 ns fixed per instruction** = 98.2 ms. This point
  is exact by construction.
- Validation on the forward graph: predicted 47.7 ms of PE union time, measured about 42 ms (41% of 103). **+13%.**
- Consequence: even with perfect tiling, the PE tops out at 213/(213+65) = **77% efficiency**. That is the floor set by
  the 128-deep contraction and the per-instruction tax.

**Step reconciliation, in ms:**

| forward graph | backward graph | launch gaps | collectives (RS 60 + AG 28) | optimizer math | embedding | sum | measured |
|---|---|---|---|---|---|---|---|
| 103 | 182 | 22 | 88 | 18.6 | 7 | 421 | 424 (1%) |

The fixed part (collectives, optimizer, embedding, launch) is about 136 ms and does not depend on layer shape. The
layer stack is about 288 ms.

**Two ways to price a shape change:**
- **F (FLOP-priced).** Step = 136 + 3.815 ms x M-MAC/token, with M = L[(4+2r)d² + 4Td]. It reproduces MLP ratio 6→8 on
  the fused line (+11.4% predicted, +11.3% measured). It also reproduces width 640 on the old v9 line (+31.1% vs
  +30.0%).
- **E (engine-split, my construction).** The 288 ms splits into three parts:
  - Vendor attention, 56.5 ms, scaling with heads x L.
  - PE GEMMs, 105 ms: 68 ms of math scaling with MAC, plus 37 ms of instruction overhead scaling with instruction
    count, with the free dimension chunked at 512.
  - Elementwise/glue, 126.5 ms: half scales with L·r·d. Half scales with L·(100·⌈d/512⌉ + d/2) cycles, i.e. the
    per-instruction tax plus 512-wide PSUM chunks.

  It reproduces ratio 6→8 (+47 ms vs +48 measured). It is my construction; no width measurement exists to test it.

## 2. What 100% MFU would take, and the realistic ceiling

| scenario | step | tokens | bpb (-0.0293 / -0.035 per e-fold) |
|---|---|---|---|
| today | 424 ms | 1x | 0 |
| perfect scheduling, today's instruction mix (graph = busiest engine, about 175 ms) | 311 | 1.36x | -0.009 / -0.011 |
| 100% graph MFU, fixed costs unchanged | 213 | 1.99x | -0.020 / -0.024 |
| 100% whole-step MFU | 77 | 5.5x | -0.050 / -0.060 |

**What is irreducible:**
- The MATMUL count (MAC / 8.39M), which caps the PE at 77%.
- The Vector-only opcodes: TENSOR_TENSOR, SCALAR_TENSOR_TENSOR, TENSOR_REDUCE and RANGE_SELECT, about 86 ms of Vector
  time in the backward (M_ISAENGINE). Only COPY and CAST can move, about 17 ms.
- The 88 ms of collectives, which cannot be overlapped in Phase 1 (CCMIN8).

**What the ideal hardware model looks like:**
- d a multiple of 512 (1024 is the clean choice; 640, 768 and 896 leave ragged 128/256 chunks).
- d_head 128 or 256, never 64.
- Every GEMM at N = 512, with no transposes.
- Elementwise work folded into the PSUM drains, on Scalar.
- Few layers.

Model E puts 1024x4 at 35% graph MFU (25% whole-step), against 25% today. **The hardware wants wide and shallow; the
loss curve does not.**

## 3. Alternatives, ranked by predicted bpb at fixed time

Quality terms used:
- **Width dose −0.0132 per e-fold of dense params at fixed steps.** Measured: it comes from the MLP-ratio and width
  gates.
- **Extra depth penalty when going shallower.** Measured twice:
  - G_D10 (12→10 layers on the v46 line): +0.0044 at fixed time despite +16% steps, which implies 0.036 per e-fold of
    L beyond the parameter loss.
  - F_V34_D8: implies 0.017.
  - Applied only downward; depth 14 also lost.
- Exchange rate −0.0293 per e-fold of steps. Measured (C_2400).

| design | step F / E (ms) | net bpb, F / E, depth penalty 0.017 | E with no depth penalty (best case, guessed) |
|---|---|---|---|
| **512x12 r6 (today)** | 424 / 424 | 0 | 0 |
| 512x12 r4 / r8 | 375 / 473 | +0.0002 / +0.0003 | same |
| 512x14 r4 | 415 | +0.0012 | +0.0012 |
| 640x10 | 489 / 506 | +0.004 / +0.005 | +0.0017 |
| 768x8 | 526 / 507 | +0.008 / +0.007 | -0.0001 |
| 768x6 | 427 / 413 | +0.011 / +0.010 | -0.0023 |
| 1024x4 | 464 / 406 | +0.018 / +0.014 | **-0.0050** |
| 1024x8 r4 | 663 / 577 | +0.011 / +0.007 | -0.0001 |
| 640x12 / 768x12 | 561 / 582, 724 / 695 | +0.002..+0.005 | same |
| d_head 64 (8x64) | 469 (E, guessed 1.8x attention) | +0.003 + unknown quality | |
| **d_head 256 (2x256)** | about 400-410 (guessed: softmax work scales with heads, 0.6-0.75x attention) | speed −0.001..−0.0016; quality guessed 0..+0.003 | |
| remove or merge norms | 0 to −1% (ROPE_WIDE-class deletions convert at about 0%) | ≈0 speed, stability risk | |
| bigger matmul free dimension | none available: 512 is the NC-v3 cap, and _FMAX=256 measured −14% | 0 | |
| our strided attention forward | −8.6 ms (measured standalone) | −0.0006, but NaN-blocked | |
| our attention backward | +22..+42% slower than the vendor's now | loses | |
| pin PSUM drains to Scalar (mlp_bwd, pre_attn_bwd, head_bwd) | −5..−10 ms (up to 17 ms movable) | **−0.0004..−0.0008** | |

**Is the flat-plane result confounded?**
- **Yes on speed.** All three width-640 runs (T13, S_C8_640, F_V23_W640) used compiler-generated layer paths. The fused
  line has **zero** width measurements: S_V29_W640 did not compile ("moving free dimension 640 exceeds max 512"). The
  "−0.0001..+0.0024 plane" in ledger row 587 is model output, not runs.
- **But the confound cuts both ways.** Model E prices 640 and 896 *worse* than FLOPs do, because of ragged chunks.
- **The plane also left out the measured depth penalty**, so its shallow points (768x8, 896x7) were optimistic.
- **The only winners in the table need the depth penalty to be zero.** Both measurements say it is not, and 1024x4 is
  far outside the tested depth range (7-16).

## 4. Buildability by Sep 30 (about 30 h)

1. **PSUM drains pinned to Scalar.** 2-4 h of work: `engine=` is passed at 0 of 50 sites. Bit-identical output.
   - Risk: the tracer rules. Kernel-argument variables are unresolvable, so literals are needed.
   - First measurement: a standalone A/B of mlp_bwd with pinned vs unpinned drains in one process (the placement-harness
     pattern), then a paired 240 s throughput screen. The screen's noise floor is about 1%, so expect to resolve only
     the upper half of the prediction.
2. **d_head 256 (2 heads).** The vendor kernels accept head dim up to 512 (`_MAX_HEAD_DIM`); the backward supports
   asymmetric dims.
   - Our pre/post-attention kernels hard-code n_head == D/128. A rewrite of the four-block Q/K layout, rope and QK-norm
     is 8-14 h.
   - Risk: quality loss from fewer heads (unmeasured), and the silent-wrong-head bug listed in handoff-09-26. Add the
     assert first.
   - **Cheapest first measurement, about 30 min with no model code:** vendor attention_cte + attention_bwd standalone at
     (bs·H = 32, d = 256) vs (64, 128). Kill it if the combined wall drops by less than 15%. Quality can be pre-screened
     at fixed steps on the local GPU (`gpu_gdn.sh`).
3. **Width port (1024x8 r4 or 768x8).** 28-40 h: PSUM D-chunking in the pre/post/head kernels plus an mlp_bwd SBUF
   restructure. **Not buildable in Phase 1**, and the predicted value is at best about 0.
   - For Phase 2 only: the cheapest confirmation is an mlp fwd+bwd kernel at D = 1024 vs 512, per token. The MLP is 62%
     of FLOPs. This tests whether model E or model F is right.

**Honest total.** Items 1 and 2, if both land: about −0.0005 to −0.0024. Item 2 is the only one with an upside near the
target, and its quality term is a guess. The −0.0045 is not available from throughput or shape in this window.
