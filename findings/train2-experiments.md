# train2.py experiment findings (the 3090 / 4080 line, pre-Trn2 era)

Raw per-run data in `train2_results.tsv`; this file is the distilled lessons. Baseline R0 = 1.3988 val_bpb; goal ladder
1.30 ✓ -> 1.25 ✓ -> 1.20 (open).

**PROTOCOL PIVOT (2026-09-02):** the portal scored the fixed-step champion WORSE than the stock baseline — fixed-STEP wins
did not survive the official fixed-TIME budget (215M params + portable-but-slow attention = too few steps in 30 min). From
that point everything ran under **fixed wall-time (720 s, steps float, warmdown keyed to elapsed time)**; judge by val_bpb
per 720 s and always record steps and tok/s. Fixed-step results stay valid as loss-per-step physics, not as competition
value. Fixed-time champion: **1.1427 / 720 s** — depth 10 x width 256, all-attention, batch 32768 + x0.71 LRs, packing
reshape, the full E-era free stack. Key inversions vs fixed-step: size optimum 768->256 (steps beat capacity); KDA hybrid
flips from -0.038 keep to **-0.026 LOSS** (step tax); MLP5, parallel residuals, SwiGLU all lose; batch halving wins ONLY
with sqrt-scaled LRs. Fixed-step era for the record (~45 runs): champion **1.2275** (E25d: KDA 5:5 hybrid on the E23b
stack); win waves were tables (spent) -> attention-mass mechanics (spent) -> linear-recurrence hybrids (active then).

## What pays (in order discovered)
- **Value embeddings** (token-indexed table added to attention values, learnable scalar gate/layer): -0.054, the single
  largest win, ~zero step-time cost; splitting into 2 then 3 per-depth tables adds ~-0.002 each.
- **U-net skips** (encoder half pushes activations, decoder half pops with learned scalars): -0.017, free; at odd depth the
  middle layer neither pushes nor pops.
- **Doc-masked attention** (BOS-cumsum doc ids, causal ∧ same-doc): -0.009 *and* ~15% faster than dense SDPA (block
  sparsity beats mask cost).
- **Width** at depth 10: 640->768 -0.005; 768->896 WORSE (+0.0017 on a 5090). Width optimum 768. Capacity axes closed:
  depth 10, width 768, tables bigram-only.
- **MLP ratio 5** (vs 4): ~noise at width 640, real (+0.0025) at 768.
- **LR warmup+warmdown**: the biggest known win over the shipped flat schedule (not an architecture change).
- **Hashed bigram table** (2^17 entries, (prev,cur) hash gated into the input embedding): -0.0084, time-free -> 1.2989,
  first goal cleared. The token-pair axis extends "tables pay".
- **Attention sinks -> registers** (4 always-visible per-head kv slots, zero-init learnable k AND v): sinks -0.0213
  (biggest win since ve), registers a further -0.0029; off-by-one softmax semantics from step 0.
- **GatedDeltaNet hybrid** (chunked kernels in layers 1,3,5,7,9, full attention elsewhere; deltanet params in AdamW at
  lr 0.01): **-0.0251 -> 1.2405**, second goal cleared. Caveats: fla dependency (not in the competition env — needs a
  self-contained port and eventually an NKI chunked-scan kernel); recurrent state not reset at doc boundaries; ~15% slower
  per step than flex attention on the 3090.

## Rules this era paid for
- **The slow-gate rule (E28a vs E28b, controlled):** identical architecture (100 zero-init cross-layer gates), only the
  gate LR changed — scalar group 0.2 -> dedicated group 0.02 recovered **0.025 bpb**. High-LR scalar groups are safe for a
  handful of well-conditioned gates (resid/x0/ve/skip) but toxic for many or sensitive ones (E8 per-channel, E20 QK
  temperature, E28a). Any new mechanism with >~10 gates or multiplicative/temperature-like scalars gets the lr-0.02 group.
- **The 420-step regime rule:** per-channel gates (E8 +0.020), oversized single-hash table (E13 +0.0028), the full Engram
  port (E31 +0.0127) and NoPE's catastrophe (E21 +0.238) all point the same way — **at 27.5M training tokens, simple direct
  additive capacity with immediate gradient flow wins; gated/multi-stage/finer-grained machinery bootstraps too slowly to
  pay for itself. Test complex mechanisms as their simplest additive distillation first.**
- **Table-capacity verdicts are token-budget-dependent** (trigram: "worse" at 27.5M tokens, -0.0042 at ~185M) — recheck
  spent axes whenever the effective token budget grows.
- **Win-stacking must be re-measured, never assumed** (L4r3: trigram + narrow QK = wash vs narrow QK alone).
- Depth, either direction: 12 slightly worse, 8 much worse (+0.086 — shallow stacks break the shortcut mechanisms).
- Per-channel (vector) gates instead of scalars for resid/x0/ve/skip: +0.020 — gate noise dominates in 420 steps.
- Dual fast/slow Muon momentum lanes (L3, mean of 0.95/0.98 nesterov blends): +0.0054, discard — the distilled ANVIL claim
  does not replicate; the slow lane is too stale at ~5k steps. Sampled softmax skipped: vocab 8192 -> the head is ~1% of
  step FLOPs, and the saving exists only at ~50k vocab.

## L-series (2026-09-02 PM, local 4080 SUPER): NanoGPT-speedrun claim testing
Baseline L0 = 1.1260 @ 5138 steps / 720 s (idle 4080S ~1.35x the 3090; never compare L-numbers to T-numbers).
- Trigram 2^17 (L1): **-0.0042** -> 1.1218 (reverses E31c — see the token-budget rule above).
- **Narrow QK (L2, qk_dim 64, v 128): -0.0075 -> 1.1185, NEW BEST.** +27% steps at equal quality; flex_attention handles
  E != Ev natively; k_sink shape follows QK_DIM; rope precomputed at QK_DIM.
- Combo (L4r3, clean): trigram + narrow QK = wash. Champion = L2 (narrow QK only, 51.3M params).
- Local-box hazards (cost 3 run slots): gate launches on a fully-empty GPU (other jobs claim it unpredictably; any
  contended fixed-time run is garbage); **never launch a run from inside a Monitor task** — when the monitor stream ends,
  task teardown reaps its whole process tree, setsid included (silent mid-run SIGKILL).

## Portal failures and submission engineering (this era)
- **v1** (fixed-step 215M stack): `no_checkpoint`, died ~0.03 s — the original "flex import" diagnosis was WRONG (the
  import was guarded); class = instant startup death, cause unknown. **v2** (fixed-time champion): `no_run_meta`, no log at
  all. **v3 responses:** lazy CUDA-only flex import, `SUBMISSION_BOOT` banners at module start / post-imports / main() so
  the next failure log discriminates where death happened, dropped `from __future__`, py_compile added to the local gates
  (`prepare.py check` only ASTs — it once passed a file that could not compile). Open ask to organizers (user action):
  expose stderr for failed submissions — blind debugging cost 3 of 5 weekly slots.
- **Always run a timed confidence check of the EXACT submission file** before submitting — correctness gates do not measure
  throughput, and both failures were pure speed traps over a healthy model.
- **`torch.set_float32_matmul_precision("high")`** — one missing line cost 20x step rate on GPU (fp32 graph segments
  without TF32); harmless on Neuron; always set it.
- Compile is default-on, but the old pre-clock warmup is SUPERSEDED (upstream c907da4) — see `competition-rulings.md` §2.
- Per-worker micro-batch is sized for the 8-worker Trn2 launch (4 post-pack rows); dev boxes override via
  `NEURON_COMPETITION_R1_DEVICE_BATCH`.
- Hardware: RTX 3090 FlexAttention default Triton blocks OOM — needs `kernel_options` (64x64 fwd, 32x32 bwd,
  `num_stages: 1`); the bwd needs the extra shrink because Inductor fuses zero-init into the template inside a fully
  compiled model. Fixed tokenizer BOS id = 8188 (`<|reserved_0|>`).
- Everything here measures loss per step on NVIDIA; seconds-per-step must be re-measured on Trn2 before it goes in a
  submission (fixed-step wins != fixed-time wins).

## Untried ideas from this era (for several of these, this is the only record)
> **Research before implementing (user directive):** for every technique below, do a fresh web/paper lookup BEFORE writing
> code — training data likely predates their current form (KDA and AttnRes are 2026 publications; Engram, MoBA, MLA variants
> evolve fast). Verify the mechanism from the primary source (paper or official repo), not from memory, and note the source
> consulted in the TSV description or here.
- **NoPE** (E21) — drop rotary, keep QK-norm; if full NoPE loses, try NoPE on half the heads. (Catastrophic at 420 steps,
  +0.238; re-test only at a long token budget.)
- **Per-layer / per-head attention budgets** (E22) — flex mask_mod sliding windows (local-128 early layers, global late);
  may also be a speed win that funds width.
- **MLA** (E24) — low-rank KV compression with decoupled-RoPE dims; param-efficiency play (the KV-cache benefit is
  Phase-2/inference-side).
- **AdaLN-Zero-style modulation** (E26) — token-conditioned low-rank scale/shift/gate on block norms, gates zero-init.
- **Compressed conv attention** (E27) — causal strided depthwise-conv downsampling of K/V (factor 2-4) so some layers/heads
  attend to a compressed sequence; doc/causal masking must allow only fully-past compressed blocks. Pairs with E22.
- **Gated cross-layer attention** (E28) — (a) cheap variant: DenseFormer-style depth-weighted averaging over ALL earlier
  block outputs (generalizes U-net skips, near-zero compute); (b) heavier: layer i's queries attend to earlier layer j's
  K/V with per-head gates blending own-layer vs cross-layer reads (= AttnRes, treat it as the reference design). Slow-gate
  rule applies.
- **GQA** (E29) — grouped-query attention (e.g. 6 Q heads sharing 2-3 KV heads at width 768): frees K/V projection params
  to respend on width/tables; also a Phase-2 KV-cache saver.
- **DeepSeek-engram-style n-gram memory** (E31) — multi-order hashed tables with per-order learned gates, injected at a
  middle layer rather than only at the input. Full port +0.0127 at 420 steps; the distilled form (multi-head hashing — see
  `ideas.md`, DeepSeek Engram) is untried on Trn2.
- **Kimi/MoBA block-sparse attention** (E32) — queries route to top-k KV blocks via mean-pooled block keys (gated); at seq
  1024 (8 blocks of 128) the long-context efficiency motive mostly vanishes, so this tests whether the routing prior
  itself helps loss.
- **QAT probe** (E33) — ONE int8 fake-quant probe to measure the quality tax as calibration for an fp8-on-chip decision;
  at fixed steps QAT can only cost loss (no size cap; `evaluate_bpb` reads fp logits), so the payoff is fixed-time fp8
  matmuls. int4/5/6 are not accelerated for training on Trainium.
- Also open from the E-list: rope base 500k vs 2M (E17, the current value was adopted untested); key-embedding table (E19,
  the ve trick on keys). Closed on Trn2 since: per-layer embeddings (E30: +0.016 at 900 s), SmearGate (E15: +0.0032),
  within-row in-context n-gram boost (E18: +0.0008/+0.0012), trigram ("worse" then a wash), AsymLogit softcap (E14, in the
  stack), packing via the organizer reshape trick (superseded by PACK=4).
- Frozen best config: `train2_best.py` (champion 1.265658, E23b stack) — do not edit; experiments continue in train2.py.
