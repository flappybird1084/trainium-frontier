# train2.py experiment findings (loss-per-step at fixed 420 steps × 65,536 tok)

Raw per-run data lives in `train2_results.tsv`; this file is the distilled lessons.
Baseline R0 = 1.3988 val_bpb; goal ladder 1.30 ✓ → 1.25 ✓ → 1.20 (open, gap 0.0275).

**⚠ PROTOCOL PIVOT (2026-09-02 02:11 PT):** the portal scored `train2_best_submission.py` WORSE than the stock baseline — fixed-STEP wins did not survive the official fixed-TIME budget (215M params + portable-but-slow attention ⇒ too few steps in 30 min). All experiments from T1 onward run under **FIXED WALL-TIME: 720 s of training, steps float, warmdown keyed to elapsed time**. Judge configs by val_bpb per 720 s; always record steps and tok/s. Fixed-step results below remain valid as loss-per-step physics but NOT as competition value.

**FIXED-TIME CAMPAIGN RESULT (overnight 2026-09-02, T1–T12):** champion **1.1427 / 720 s** (T10):
depth 10 × width 256, all-attention, batch 32768 + ×0.71 LRs, packing reshape, full E-era free
stack (tables/registers/doc-mask/AsymLogit/rope 200k). Key inversions vs fixed-step: size optimum
768→256 (steps beat capacity); KDA hybrid flips from −0.038 keep to −0.026 LOSS (step tax);
MLP5, parallel residuals, SwiGLU all lose; batch halving wins ONLY with √-scaled LRs.
`train2_best_submission.py` v2 carries this config; all validation gates green.

**CAMPAIGN STATE (fixed-step era, 2026-09-02, ~45 runs):** champion **1.2275** (E25d: KDA 5:5 hybrid on the
E23b stack; `train2_best.py`, fla-dependent). Submittable best: `train2_best_submission.py`
(1.2657-era stack, fla-free, all local gates passed). Win waves: tables (spent) →
attention-mass mechanics (spent) → linear-recurrence hybrids (active; GDN −0.025, KDA −0.013).
The user-directed technique batch is fully executed; remaining credible arch headroom is
mostly deeper recurrence variants — projected landing ~1.21–1.22 without new families.

## What pays (in order discovered)

- **Value embeddings** (token-indexed table added to attention values, learnable scalar gate/layer): −0.054, the single largest win, ~zero step-time cost. Splitting into 2 then 3 per-depth tables adds ~−0.002 each.
- **U-net skips** (encoder half pushes activations, decoder half pops with learned scalar weights): −0.017, free. Odd depth: middle layer neither pushes nor pops.
- **Doc-masked attention** via FlexAttention block mask (BOS-cumsum doc ids, causal ∧ same-doc): −0.009 *and* ~15% faster than dense SDPA (block sparsity beats mask cost).
- **Width** at depth 10: 640→768 gave −0.005; 768→896 measured WORSE (+0.0017, E16 on a 5090). Width optimum = 768. All three capacity axes are now closed: depth 10, width 768, tables bigram-only.
- **MLP ratio 5** (vs 4): ~noise at width 640, real (+0.0025) at width 768.
- **LR warmup+warmdown schedule** (in R0 already): the biggest known win over the shipped flat schedule — not an architecture change, noted for the submission.

## The slow-gate rule (E28a vs E28b, controlled)

Identical architecture (100 zero-init cross-layer gates), only the gate LR changed: scalar group 0.2 → dedicated group 0.02 recovered **0.025 bpb**. High-LR scalar groups are safe for a handful of well-conditioned gates (resid/x0/ve/skip) but toxic for many or sensitive ones (E8 per-channel, E20 QK temperature, E28a). Any new mechanism with >~10 gates or multiplicative/temperature-like scalars gets the lr-0.02 slow-gate group.

## The 420-step regime rule (learned the hard way)

Three independent failures — per-channel gates (E8, +0.020), oversized single-hash table (E13, +0.0028), full DeepSeek Engram port (E31, +0.0127) — plus NoPE's catastrophe (E21, +0.238) point the same way: **at 27.5M training tokens, simple direct additive capacity with immediate gradient flow wins; gated/multi-stage/finer-grained machinery bootstraps too slowly to pay for itself.** Test complex mechanisms as their simplest additive distillation first.

## What doesn't pay

- **Depth, either direction**: 12 slightly worse, 8 much worse (+0.086 — shallow stacks break the shortcut mechanisms). Optimum firmly at 10 for this budget.
- **Per-channel (vector) gates** instead of scalars for resid/x0/ve/skip: +0.020 worse — gate noise dominates in 420 steps.

## Portal failure log (3 slots burned — treat every untested assumption as hostile)

- **v1** (fixed-step 215M stack): `no_checkpoint`, died ~0.03 s. Original "flex import" diagnosis was WRONG (the import was guarded). Cause still unknown; class = instant startup death.
- **v2** (fixed-time champion): `no_run_meta`, no log at all — died even earlier than v1 from the harness's view. Cause unknown.
- **v3 responses**: lazy CUDA-only flex import (native import crashes bypass try/except), `SUBMISSION_BOOT` banners at module start / post-imports / main() so the next failure log discriminates where death occurred, dropped `from __future__` (banner placement), py_compile added to local gates (`prepare.py check` only ASTs — it passed a file that could not compile!).
- **Open ask to organizers (user action): expose stderr for failed submissions** — blind debugging has cost 3 of 5 weekly slots.

## Submission engineering rules (paid for in failed runs)

1. **Always run a timed confidence check of the EXACT submission file** before submitting — correctness gates don't measure throughput, and both submission failures (v1 portal, v2 first confidence run) were pure speed traps with a healthy model.
2. **`torch.set_float32_matmul_precision("high")`** — one missing line cost 20x step rate on GPU (fp32 graph segments without TF32). Harmless on Neuron; always set it.
3. ~~Compile is on by default with a pre-clock warmup~~ **SUPERSEDED (upstream c907da4, 2026-09-03):** compile is still default-on, but the warmup is gone — the runner now excludes launch→5th `record_step_boundary()` (cap 900 s). Submissions must call `prepare.record_step_boundary(step)` after every synchronized optimizer step (check_submission rejects otherwise), restart the budget clock at boundary 5, and stop via a rank-synchronized control collective gated behind ≥5 completed steps. `--compile-only` flag required (an unknown-flag argparse exit = instant death if the runner passes it). See findings/competition-rulings.md §2.
4. Per-worker micro-batch is sized for the 8-worker Trn2 launch (DEVICE_BATCH_SIZE=4 post-pack rows); dev boxes override via NEURON_COMPETITION_R1_DEVICE_BATCH.

## Hardware/portability notes

- RTX 3090 (sm86, ~99KB smem/SM): FlexAttention default Triton blocks OOM; needs `kernel_options` — 64×64 fwd, 32×32 bwd blocks, `num_stages: 1`. The bwd needs the extra shrink because Inductor fuses zero-init into the template inside a fully compiled model.
- Fixed tokenizer BOS id = 8188 (`<|reserved_0|>`).
- All of this measures **loss per step on NVIDIA**; seconds-per-step must be re-measured on Trn2 before any of it goes in the submission (fixed-step wins ≠ fixed-time wins).

- **Hashed bigram table** (2^17 entries, (prev,cur)-pair hash gated into the input embedding): −0.0084, time-free. Took champion to 1.2989 — first goal (1.30) cleared. The token-pair axis extends the "tables pay" finding.
- **Attention sinks → registers** (4 always-visible per-head kv slots, zero-init learnable k AND v): sinks −0.0213 (biggest win since ve), registers +−0.0029 more. Off-by-one softmax semantics from step 0.
- **GatedDeltaNet hybrid** (fla chunked kernels in layers 1,3,5,7,9; full attention elsewhere; deltanet params in AdamW lr 0.01, expand_v=1): **−0.0251, champion 1.2405 — second goal (1.25) cleared.** Linear-attention hybrid beats pure attention at this budget. Caveats: fla dependency (not in competition env — needs a self-contained port + eventually an NKI chunked-scan kernel); recurrent state not reset at doc boundaries; ~15% slower per step than flex attention on the 3090.

## L-series (2026-09-02 PM): NanoGPT-speedrun claim testing, local 4080 SUPER

Source: an unverified "39.9 s NanoGPT record" announcement (ANVIL optimizer / sampled softmax /
trigram tables / narrow QK / fp8). Local fixed-time baseline **L0 = 1.1260 @ 5138 steps / 720 s**
(idle 4080S ≈ 1.35× the 3090; never compare L-numbers to T-numbers directly).

- **Trigram hashed table 2^17 (L1): −0.0042 → 1.1218.** REVERSES E31c ("trigram worse") — that
  verdict was an artifact of the fixed-step era's 27.5M tokens; at ~185M tokens/run the third
  n-gram order pays. Lesson: **table-capacity verdicts are token-budget-dependent — recheck the
  spent axes whenever the effective token budget grows.**
- **Narrow QK (L2, qk_dim 64, v 128): −0.0075 → 1.1185, NEW BEST.** +27% steps at equal quality;
  flex_attention handles E≠Ev natively. k_sink shape follows QK_DIM; rope precomputed at QK_DIM.
- **Dual fast/slow Muon momentum lanes (L3, mean of 0.95/0.98 nesterov blends): +0.0054, discard.**
  The distilled ANVIL claim does not replicate; slow lane too stale at ~5k steps.
- Sampled softmax: SKIPPED with cause — vocab 8192 (head ≈1% of step FLOPs); the claim's 8 s
  saving exists only at 50k vocab. fp8-through-backward: skipped, no Trainium training analog
  (E33: quantization tax ≈ 0, so on-chip fp8 is free when available).
- **Combo (L4r3, clean): trigram + narrow QK = wash vs narrow QK alone** (1.1183 vs 1.1185, noise).
  The two wins do NOT stack — trigram's gain disappears on the narrow-QK model. Champion = L2
  (narrow QK only, 1.1185, 51.3M params). Win-stacking must be re-measured, never assumed.
- Local-box hazards for benchmarking (cost 3 run slots tonight): (a) user's other jobs (jepa,
  thonk-v2, a 14.5GB anon process) claim the GPU unpredictably — gate launches on a fully-empty
  GPU, treat any contended fixed-time run as garbage; (b) **never launch a run from inside a
  Monitor task** — when the monitor stream ends, task teardown reaps its whole process tree,
  setsid included (silent mid-run SIGKILL). Launch from direct Bash; monitors watch only.

## Open/queued (user-directed batch, 2026-09-01 evening — takes priority)

> **⚠ Research before implementing (user directive):** for every technique below, do a fresh web/paper lookup BEFORE writing code — model training data likely predates the current form of these methods (e.g. KDA and AttnRes are 2026 publications; DeepSeek engram, MoBA revisions, MLA variants evolve fast). Verify the mechanism from the primary source (paper or official repo), not from memory, and note the source consulted in the TSV description or here.

- E21 **NoPE** — drop rotary (keep QK norm); if full NoPE loses, partial variant (NoPE on half the heads).
- E22 **per-layer/per-head attention budgets** — flex mask_mod with per-layer/per-head sliding windows (e.g. local-128 early layers, global late); may also be a speed win that funds width.
- E23 **attention sinks** — one always-visible extra KV slot per head: learned k_sink, v_sink=0 → pure denominator sink (off-by-one softmax), implemented as a mask extension in flex.
- E24 **MLA** (multi-head latent attention) — low-rank KV compression with decoupled-RoPE dims; param-efficiency play, KV-cache benefit is Phase-2/inference-side.
- E25 **Gated DeltaNet** — replace attention with gated delta rule in a subset of layers, trained in **block-parallel/chunked form** (WY representation: intra-chunk parallel, inter-chunk recurrent carry — per user note, this is the efficiency-viable training mode and the form that maps to Trainium's TensorEngine). Research runs may use the `fla` library on the pod; a submission needs a self-contained port (NKI chunked-scan kernel — high effort).
- E26 **AdaLN-Zero-style modulation** — token-conditioned low-rank scale/shift/gate on block norms, gates zero-init.
- E27 **compressed conv attention** — causal strided depthwise-conv downsampling of K/V (factor 2–4) so some layers/heads attend to a compressed sequence; doc/causality masking must allow only fully-past compressed blocks. Pairs naturally with E22's local/global budget split (local heads full-res, global heads compressed).
- E28 **gated cross-layer attention** — (a) cheap variant first: DenseFormer-style depth-weighted averaging — each block's input is a learned-gate mixture over ALL earlier block outputs (generalizes our U-net skips, near-zero compute); (b) heavier variant: layer i's queries attend to earlier layer j's K/V with per-head gates blending own-layer vs cross-layer reads.
- E29 **GQA** — grouped-query attention (e.g. 6 Q heads sharing 2–3 KV heads at width 768): frees K/V projection params to respend on width/tables; at fixed steps the question is whether the loss cost is smaller than what the freed params buy. Also directly relevant to Phase-2 inference (CORE timeout) as a KV-cache saver.
- E30 **per-layer embeddings** (Gemma-3n-PLE-style) — token-indexed additive vectors injected into the residual stream per layer (generalizes the per-depth vte split into the residual path); pure table capacity, ~zero step-time cost.
- E31 **DeepSeek-engram-style n-gram memory** — multi-order hashed n-gram tables (bigram + trigram) with per-order learned gates, injected at a middle layer rather than only at the input. Direct generalization of E12's winning bigram table; strongest prior of the whole batch given session evidence. Trigram hash over (t-2, t-1, t).
- E33 **QAT probe (int8 weights+activations, fake-quant)** — user-suggested (int4/int8/q5/q6). Frame note: at FIXED STEPS, QAT can only cost loss (no size cap in this competition; evaluate_bpb reads fp logits), so a full precision sweep can't win here. The real payoff is fixed-TIME on Trn2: FP8-family matmuls (natively accelerated; Trn3 adds MXFP8/FP4) buy more steps per 30 min — int4/5/6 aren't accelerated for training on Trainium. Run ONE int8 probe to measure the quality tax as calibration for an fp8-on-chip decision; sweep further precisions only if the tax is surprisingly small.
- E32 **Kimi/MoBA block-sparse attention** — queries route to top-k KV blocks via mean-pooled block keys (gated). Expectation flag: at seq 1024 (8 blocks of 128) the long-context efficiency motive mostly vanishes; testing whether the routing prior itself helps loss. Related Kimi mechanisms map to existing slots: KDA = vector-gated delta rule → fold into E25 as the gating variant; AttnRes (layers attend over earlier layers) ≈ E28(b), treat it as the reference design.

## Open/queued

- E13 bigram table 2^18 (doubling) — running.
- E14 AsymLogit softcap (learned pos/neg cap scalars — see external-techniques.md; subsumes softcap tuning).
- E15 SmearGate (input-gated prev-token mix, zero-init).
- E16 width 896 (needs MLP ratio 4 to fit 15-min runs, or faster steps).
- E17 rope base 500k vs 2M (current value adopted untested from autoresearch).
- E18 within-row in-context n-gram logit boost (stateless adaptation — see external-techniques.md).
- E19 key-embedding table (ve trick on keys).
- E20 learnable per-head QK gain (attention temperature under QK-norm; init 1.0, sweep init if it wins — from parameter-golf PR #1953).
- Packing/tail-truncation experiment (organizer reshape trick — see competition-rulings.md); data-sampling not arch, needs user OK.

- Frozen best config: `train2_best.py` (champion 1.265658, E23b stack) — do not edit; experiments continue in train2.py.
