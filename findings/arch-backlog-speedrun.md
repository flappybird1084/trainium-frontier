# Architectural backlog: nanogpt-speedrun + parameter-golf lineage (compiled 2026-09-06)

Purely architectural ideas with a track record in the modded-nanogpt speedrun and OpenAI parameter-golf, filtered against
what our stack already has and against the Trn2 profile. Companion to `external-techniques.md` (raw mining notes) and
`train2-experiments.md` (our own E-/L-series ledger). Sources at the bottom. **Trn2 verdicts are marked inline where the
campaign log has one; unmarked = untried on Trn2.**

## The filter (read before picking anything)
- Ideas that add **FLOPs on tensors we already materialize** are close to free (256->512-wide cost ~0% step time).
- Ideas that add **graph nodes / kernel launches** are expensive — they hit the actual bottleneck (the compiled step was
  launch/DMA-bound, and on the kernel line custom-op bodies run eagerly at runtime).
- Ideas that add **new shapes** are worst: a cold NEFF compile is 500-650 s (cap 900 s).
- Anything touching attention masking is cheap **if** it goes through the NKI kernel's per-query `bound_min`/`bound_max`
  (computed anyway); on the explicit path a window is just a mask (quality restriction, no compute saving).
- Already in the stack, do not re-list: RoPE, QK-norm, ReLU^2, zero-init `c_proj`, U-net skips, value-embedding tables,
  untied small-init lm_head, asymmetric logit softcap, x0 residual lambdas, doc-masked attention, sinks/registers, bigram
  hash table, Muon+AdamW, trapezoid LR schedule.

## A. Highest value — composes with the attention kernel
1. **Long-short sliding-window attention** (speedrun record 20). Half the layers/heads see a 128-256 window, the rest
   global; here it is *free* — `attention_cte` already takes per-query `bound_min`/`bound_max`. **Trn2: screened — window
   256 on odd layers -0.0017 (free) on the eager kernel line, +0.0007 on the compile line; not adopted.**
2. **Attention window warmup** (record 13). Grow the window over training: a value change inside `bounds`, not a shape
   change -> no recompile. **Trn2: adds nothing (identical to the fixed window).**
3. **Narrow QK** (`QK_DIM=64`, v stays 128). Our 4080 run: **-0.0075 via +27% steps**. **Trn2: closed — `QK_DIM` breaks
   the 128-wide kernels, and the 4080 win was throughput-driven (we are throughput-rich here).**
4. **Merged QKV projection.** One matmul instead of three — a *submit-count* lever, not a loss lever. **Trn2: +0.0005
   (eager kernel line) / +0.010 (compile line).**
5. **Parallel residuals** (attn and MLP both read the block input). Halves the serial dependency chain per block.
   **Trn2: +0.005 at equal steps (closed).**

## B. Speedrun classics not yet in the stack
6. **Multi-token prediction** (record 53). Aux head predicting t+2, dropped at eval so `load_for_eval` stays pure; vocab
   8192 -> ~1% of step FLOPs. **Trn2: +0.040 at 900 s (discard).**
7. **Prefix / backward token prediction** (record 88). Aux loss predicting *earlier* tokens from the same hidden state;
   same cost profile as MTP, stacks with it. Untried (needs two target logits in the fused head kernel).
8. **Value-residual learning.** Blend layer-0 V into every layer's V with a learned per-layer lambda (depth-indexed, unlike
   our token-indexed ve tables). **Trn2: +0.0010 (discard).**
9. **Backout.** Subtract an earlier layer's contribution from the final residual with a learned scalar before lm_head.
   **Trn2: +0.0013 (discard).**
10. **XSA / exclusive self-attention** (record 82). Remove the self-aligned component of the attention output via a learned
    per-(layer, head) tanh gate. **Trn2: closed — 2x step as torch glue (custom-op bodies run eagerly at runtime, so extra
    elementwise ops are launch-bound); expected gain ~0.001.**
11. **Paired-head attention** (record 58). Value reshaping across head pairs; pure layout change, no new params. Untried;
    reported incompatible with XSA.
12. **MUDD / DenseFormer depth-wise mixing** (record 81; our queued E28a). Each block's input = a learned-gate mixture over
    all earlier block outputs; generalizes U-net skips. **Slow-gate rule applies** (dedicated lr-0.02 group; E28a vs E28b
    was 0.025 bpb on exactly this failure mode).
13. **Sparse / low-rank value embeddings** (record 17). We carry 3 dense 8192x512 ve tables; sparsify or restrict to a
    subset of layers and respend the params (the speedrun found dense ve overparameterized).
14. **Smear gate** (our E15). `x_t += lam * sigmoid(W x_t) * x_{t-1}`, zero-init, BOS-masked so it cannot leak across
    packed documents. **Trn2: +0.0032 (no step gain) — closed.**
15. **Per-layer embeddings** (E30, Gemma-PLE style). Token-indexed additive vector into the residual per layer; pure table
    capacity, ~zero step time. **Trn2: +0.016 at 900 s (discard).**
16. **Dynamic Tanh (DyT)** in place of RMSNorm. **Superseded as dead** by `strange-wins-research.md` (2026-09-08 sweep).

## C. Parameter-golf lineage
17. **Learnable per-head QK gain** (E20; parameter-golf PR #1953). Restores attention temperature under QK-norm; init 1.0 =
    transparent. Slow-gate group mandatory (the E28a failure class). Untried on Trn2.
18. **Within-row in-context n-gram boost** (E18). Boost the logit of the token that followed the most recent match of the
    current context *within the same row's strict prefix*, learned scalar; stateless per-row -> legal under
    `load_for_eval`. **Trn2: +0.0008/+0.0012 (closed).**
19. **Cross-sparse attention.** Their "architecture" headliner alongside depth recurrence and parallel residuals; maps onto
    the same `bounds` mechanism as #1. Untried.
20. **Depth recurrence / block weight-tying with per-layer scalars.** Golf submissions cluster at 10-11 layers (we are at
    10 — good sign), but recurrence is *size*-motivated and we have no size cap; under fixed time it buys FLOPs we cannot
    use. **Low priority, listed for completeness.**
21. **GLU-family activation (SwiGLU).** The calibration paper ranks it among the few *real* effects. Lost under fixed-time
    at width 256; worth exactly one re-screen at 512, then close the axis permanently.

## D. Re-checks — our own rule says token-budget verdicts expire
22. **Trigram table** (`TRIGRAM=1`, wired, off). "Worse" at 27.5M tokens; **-0.0042** at ~185M. **Trn2: +0.0035 (eager
    kernel line), +0.0050 (via the sidecar), a wash on the compile line — closed.** Trigram + narrow QK was a wash, so
    test such pairs jointly, never additively.
23. **Key-embedding table** (E19) — the ve trick applied to keys; same family as our most consistent wins. Untried.

## Do-not / known-dead
- **Sequence-length curriculum** — a new shape per stage is a fresh 500-650 s NEFF compile each time.
- **Batch-size ramp** — only viable via grad-accum count, never row count, or we recompile.
- **Test-time training / LoRA TTT** — flatly illegal under the stateless `load_for_eval` contract (and the no-reading-
  validation-data rule).
- **Quantization chain (AWQ/GPTQ/LQER, ternary, int5/6)** — serves their 16 MB artifact cap, no analog here.
- **NorMuon / AdaMuon and friends** — optimizer, not architecture, and the calibration paper puts most optimizer
  modifications in the noise.

## Sources
- modded-nanogpt record history: https://github.com/KellerJordan/modded-nanogpt
- NanoGPT Speedrun Worklog (Tyler Romero): https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/
- Hunting for headroom on modded-nanoGPT (WR #82): https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html
- What Parameter Golf taught us (OpenAI): https://openai.com/index/what-parameter-golf-taught-us/
- Parameter Golf: What Really Works? (arXiv 2607.01517): https://arxiv.org/pdf/2607.01517

---
The 2026-09-08 research sweep for counterintuitive/reproduced wins and novel architectures is in
`strange-wins-research.md` (it supersedes item #16 above as dead and adds the modded-nanogpt Track 3
step-minimization leaderboard as an evidence source we were not using).
