# Architectural backlog: nanogpt-speedrun + parameter-golf lineage (compiled 2026-09-06)

Purely architectural ideas with a track record in the modded-nanogpt speedrun and OpenAI parameter-golf,
filtered against what our stack already has and against the Trn2 profile. Companion to
`external-techniques.md` (raw mining notes) and `train2-experiments.md` (our own E-/L-series ledger).
Sources at the bottom.

## The filter (read before picking anything)

PR7 profile at 512-wide: **launch/DMA-bound, not FLOP-bound** - ~430 model submits and ~5,000 small DMA
copy-ins per step, ~9% of matmul peak; width 256 -> 512 cost ~0% step time (0.48 -> 0.49 s). So:

- Ideas that add **FLOPs on tensors we already materialize** are close to free.
- Ideas that add **graph nodes / kernel launches** are expensive - they hit the actual bottleneck.
- Ideas that add **new shapes** are worst: a cold NEFF compile at 512-wide is 500-650 s (cap 900 s).
- Anything touching attention masking is cheap **if** it goes through the NKI kernel's per-query
  `bound_min`/`bound_max` (see `handoff-2026-09-06.md` section 5) - that tensor is computed anyway.

Already in the stack, do not re-list: RoPE (200k), QK-norm, ReLU^2, zero-init `c_proj`, U-net skips,
3x value-embedding tables, untied small-init lm_head, asymmetric logit softcap, x0 residual lambdas,
doc-masked attention, sinks/registers, bigram hash table, Muon+AdamW, trapezoid LR schedule.

## A. Highest value - composes with the NKI attention kernel

1. **Long-short sliding-window attention** (speedrun record 20). Half the layers/heads see a 128-256
   window, the rest global. Elsewhere a mask trick; here it is *free* - `attention_cte` already takes
   per-query `bound_min`/`bound_max`, so a window is a value change in a tensor we compute anyway, zero
   extra ops, and it cuts attention FLOPs.
2. **Attention window warmup** (record 13). Grow the window over training (short early, full late).
   On Neuron this is a value change inside `bounds`, **not a shape change** -> no recompile.
3. **Narrow QK** (`NEURON_COMPETITION_R1_QK_DIM=64`, v stays 128). Our own L2 run: **-0.0075 via +27%
   steps** on the 4080. Knob is already wired in the vehicle but never screened at 512-wide on Trn2.
   Cheapest untested win we have.
4. **Merged QKV projection.** One matmul instead of three per layer. Not a loss lever - a *submit-count*
   lever aimed straight at the 430-submits/step bottleneck. Same family: fusing the MLP matmul launches.
5. **Parallel residuals** (attn and MLP both read the block input). Discarded on loss at fixed-time at
   width 256 on GPU - but it halves the serial dependency chain per block, which is what a launch-bound
   device pays for. Re-test as a **step-time** experiment, not a loss one.

## B. Speedrun classics not yet in the stack

6. **Multi-token prediction** (record 53). Aux head predicting t+2, dropped at eval so `load_for_eval`
   stays stateless/pure. Vocab 8192 -> the extra head is ~1% of step FLOPs. Big historical win, small cost.
7. **Prefix / backward token prediction** (record 88). Aux loss predicting *earlier* tokens from the same
   hidden state. Same cost profile as MTP; stacks with it.
8. **Value-residual learning.** Blend layer-0 V into every layer's V with a learned per-layer lambda.
   Distinct from our ve tables (token-indexed); this one is depth-indexed. Two adds per layer + scalars.
9. **Backout.** Subtract an earlier layer's contribution from the final residual with a learned scalar
   before lm_head. One add, one scalar, zero-init -> transparent at step 0.
10. **XSA / exclusive self-attention** (record 82, current era). Remove the self-aligned component of the
    attention output via a learned per-(layer, head) tanh gate. Cheap, zero-init transparent. Reported
    **incompatible with paired-head attention** (#11).
11. **Paired-head attention** (record 58). Value reshaping across head pairs. Pure layout change, no new params.
12. **MUDD / DenseFormer depth-wise mixing** (record 81; our queued E28a). Each block's input = learned-gate
    mixture over all earlier block outputs; generalizes our U-net skips. **Slow-gate rule applies** -
    dedicated lr-0.02 group; E28a vs E28b was 0.025 bpb on exactly this failure mode.
13. **Sparse / low-rank value embeddings** (record 17). We carry 3 dense 8192x512 ve tables (~12.6M params
    each at 512-wide). Sparsify or restrict to a subset of layers and respend the params - the speedrun
    found dense ve overparameterized.
14. **Smear gate** (our E15, queued, never run). `x_t += lam * sigmoid(W x_t) * x_{t-1}`, zero-init, and
    **BOS-masked** so it cannot leak across packed documents (that fix is from parameter-golf PR #1953).
15. **Per-layer embeddings** (our E30, Gemma-PLE style). Token-indexed additive vector injected into the
    residual per layer. Pure table capacity, ~zero step time - and "tables pay" is the most-replicated
    finding in our own ledger.
16. **Dynamic Tanh (DyT)** in place of RMSNorm. Removes a reduction per norm - 20 norms/step of fewer ops
    on a device where ops, not FLOPs, are the currency. Recent lineage item, low loss risk.

## C. Parameter-golf lineage

17. **Learnable per-head QK gain** (our E20; parameter-golf PR #1953). Restores attention temperature under
    QK-norm; init 1.0 = transparent. Slow-gate group mandatory - this is exactly the sensitive
    multiplicative-scalar class that burned E28a.
18. **Within-row in-context n-gram boost** (our E18). Boost the logit of the token that followed the most
    recent match of the current context *within the same row's strict prefix*, learned scalar. Stateless
    per-row -> legal under `load_for_eval`. The legal, portable shadow of their illegal online n-gram tilt.
19. **Cross-sparse attention.** Their "architecture" category headliner alongside depth recurrence and
    parallel residuals. Worth one screen; maps onto the same `bounds` mechanism as #1.
20. **Depth recurrence / block weight-tying with per-layer scalars.** Top golf submissions cluster at 10-11
    layers (we are at 10 - good sign), but recurrence is *size*-motivated and we have no size cap; under
    fixed time it buys FLOPs we already cannot use. **Low priority, listed for completeness.**
21. **GLU-family activation (SwiGLU).** The calibration paper ranks it among the few *real* effects. We
    measured it as a loss under fixed-time at width 256 - worth exactly one re-screen at 512, then close
    the axis permanently.

## D. Re-checks - our own rule says token-budget verdicts expire

22. **Trigram table** (`TRIGRAM=1`, wired, off). E31c said "worse" at 27.5M tokens; L1 said **-0.0042** at
    ~185M. Full runs at 512-wide are now ~118M tokens - squarely in the zone where it flipped. But L4r3
    showed trigram + narrow QK is a **wash**, so test the pair jointly, never additively.
23. **Key-embedding table** (our E19) - the ve trick applied to keys. Same family as the wins that have paid
    most consistently for us.

## Do-not / known-dead

- **Sequence-length curriculum** - a new shape per stage is a fresh 500-650 s NEFF compile each time.
  WR#82 abandoned it for compile overhead on *GPU*; strictly worse here.
- **Batch-size ramp** - only viable via grad-accum count, never via row count, or we recompile.
- **Test-time training / LoRA TTT** - the biggest golf lever and flatly illegal under our stateless
  `load_for_eval` contract (and the no-reading-validation-data rule).
- **Quantization chain (AWQ/GPTQ/LQER, ternary, int5/6)** - serves their 16 MB artifact cap, no analog here.
- **NorMuon / AdaMuon and friends** - optimizer, not architecture, and the calibration paper puts most
  optimizer modifications in the noise.

## Suggested order, given the kernel work underway

**1, 2, 3** first (they ride the NKI bounds/QK plumbing being written now), then **4, 6, 8, 9** (cheap, high
historical hit rate), then the table re-checks **22, 15**.

## Sources

- modded-nanogpt record history: https://github.com/KellerJordan/modded-nanogpt
- NanoGPT Speedrun Worklog (Tyler Romero): https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/
- Hunting for headroom on modded-nanoGPT (WR #82): https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html
- What Parameter Golf taught us (OpenAI): https://openai.com/index/what-parameter-golf-taught-us/
- Parameter Golf: What Really Works? (arXiv 2607.01517): https://arxiv.org/pdf/2607.01517

---

A 2026-09-08 research sweep for counterintuitive/reproduced wins and novel architectures, filtered
against our regime, is in `strange-wins-research.md`. It supersedes item #16 (Dynamic Tanh) of this file as dead,
and adds the modded-nanogpt **Track 3** step-minimization leaderboard as an evidence source we were not using.
