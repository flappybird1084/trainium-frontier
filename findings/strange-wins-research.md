# Strange wins & novel architectures (research sweep, 2026-09-08)

Reproduced gains whose mechanism was NOT understood or ran against theory, plus novel 2024-26 architectures at ~150M
params / <1B tokens; sources primary (papers, official repos, record changelogs). **Verdict labels are the researcher's,
filtered against our regime** (30 min wall-clock, ~2300-5500 steps, seq 1024, vocab 8192, Trainium2, compute-bound at
56.8% MFU so added FLOPs cost steps directly; stateless pure-forward eval contract); "free"/"zero"/"cheap" below means
so under that regime. `PR #N` = modded-nanogpt PR N.

## The meta-finding, which matters more than any single item

The modded-nanogpt speedrun (124M, ~400M tokens, wall-clock scored, multi-seed, adversarial) is the only public
benchmark in our regime, and **in two years not one famous architecture paper has ever landed there.** What lands is
near-zero-FLOP, shape-preserving, zero-init-transparent tweaks - our own 420-step-regime rule, externally confirmed.
Also: a **Track 3 "optimization" leaderboard** (fixed architecture, minimize *steps*, unlimited wall-clock, n=4-30
seeds, 0.0045 loss ~ 100 steps) - the most in-distribution evidence source available to us. **~134M of our 187M params
are the bigram table** (~80% of params, ~0 FLOPs); the real compute model is a ~36M-param dense net whose LM head is
~10% of forward FLOPs.

## Tier 1 - strange, free, reproduced

**1. Prefix-token-prediction aux loss.** T' = the token whose bytes are the longest proper prefix of T; add
`prefix_w*(lse - z_T')` on the SAME logits, 0.25 -> 0 linear, off at eval. PR #337 (record #88): **-15 steps / -0.8 s**
at 124M, speedrun-verified. Cost **free**. **PLAUSIBLE - top pick**; stacks with MTP, so our MTP failure does not rule
it out.

**2. Canonical token masking.** BPE is deterministic, so most next-tokens are unreachable given the previous one; mask
non-canonical continuations, renormalize. PR #350 · arXiv 2506.07956: **-0.0017 nats over 20 runs -> -15 steps**; open
PR, not merged; provable, not empirical. Cost **free** (an 8192x8192 bitmask, 8.4 MB). **PLAUSIBLE - highest expected
value per line of code in the report.**

**3. Bigram hash "sign trick".** Multiply each retrieved row by a frozen random ±1 pattern keyed to the bigram
(`sign_idx = (prev ^ cur) % 8192`); paired with **narrow rows**: 5x vocab @ 768-dim -> 15x vocab @ 192-dim. PR #299
(record #83), basis Chen et al. 2015: **-1.5 s of an 80 s record** (sign ~-0.5 s, dim ~-0.5 s), speedrun-verified; VIP
buckets for top-16k bigrams and reducing slots both FAILED. Cost **free**. ~~**PLAUSIBLE - most on-target for us**~~
**TRN2: CLOSED - wash (plan-097)** (the table is 72% of params and memory-capped at 2^19, which is why it looked so
on-target.)

**4. Hash-table size must NOT be an integer multiple of vocab size.** With a polynomial rolling hash, collisions spike
near an integer multiple of the base vocab (the paper's rule: *"should significantly deviate from integer multiples of
the base vocabulary size"*). arXiv 2601.21204 (LongCat-Flash-Lite); single paper, not separately quantified. Cost
**free**. ~~**PLAUSIBLE, ACT ON IT:** our BIGRAM=2^17 is exactly 16x vocab and 2^18 exactly 32x~~ **TRN2: CLOSED - prime
table size measured WORSE (plan-097).** CAVEAT (ours, verified): our hash is Knuth multiplicative
(`pair * -1640531527 >> 13 & (SIZE-1)`), not the polynomial hash analysed.

**5. Mimetic V-O init - stop zero-initing the attention out-projection.** Per head `W_V = +sqrt(b) I`, `W_O = -sqrt(b) I`,
b=0.05 plus sigma=0.01 noise, so `V@O^T ~ -bI` at init instead of `O = 0`. PR #214 (record #64) · arXiv 2305.09828:
**-20 steps of 1575, n=7 interleaved, p=0.0036**, matched-step control failing (p=0.115). **Contradicts the zero-init
doctrine we use.** Cost **exactly zero**. ~~**PLAUSIBLE - a two-line swap.**~~ **TRN2: CLOSED - worse (plan-097).**
b>=0.10 destabilizes; attention `c_proj` only.

**6. Partial key offset + half-truncate RoPE.** Zero the top half of `inv_freq` so half of each head's dims are
unrotated, then shift the STATIONARY key dims one forward: `k[:,1:,:,D//4:D//2] = k[:,:-1,:,D//4:D//2]`. *"Enables
1-layer induction."* PR #169 (record #49) · RoPE-fraction ablation repo discussions/150. **-50 steps (~3.6%), n=11,
p=0.0004**; full-dim RoPE worse than 50%; corroborated (LlamaCanon, MHA2MLA arXiv 2502.14837). Cost **free**.
**PLAUSIBLE - strongest in-regime statistics in the report.**

**7. Tail-EMA checkpoint blend.** Last ~15% of steps: `ema += (w - ema)/tau` (tau~150) for every param EXCEPT
token-embedding tables; save `w_final = 0.65*w + 0.35*ema`. PRs #325 / #328 (Track-3 records #45/#46), Track-1 port
#364: **-42 steps of 2745 (-1.6%)**, "the single biggest lever"; Track-1 port **-0.0024 val loss**. **Reproduced by
three independent parties** plus IMU-1 (arXiv 2602.02522). Full SWA/LAWA and forward BEMA both rejected; only the
PARTIAL blend works. Cost **free**, tail only. **PLAUSIBLE - do first.** Use fp32 EMA state (a bf16 EMA at tau=150
silently stalls, update weight 6.7e-3).

**8. Cautious Weight Decay.** `mask = 1[update*w > 0]; w *= (1 - lr*lambda*mask)` - decay only coordinates the update is
already shrinking. arXiv 2510.12402 · records #43, #50, Track-3 #46: Track-1 -1.25% then -0.85% wall-clock, Track-3
isolated -19 steps, dose inverted-U (0.0125 weak, **0.025 optimal**, 0.05 over-shrinks). **Reproduced four times.**
Cost **free**. STATUS (ours, verified): our NKI Muon kernel ALREADY does this (`train3_submission.py:847,860`). **The
remaining move is extending it to the AdamW groups.**

**9. EMA-Nesterov outer wrapper.** EMA (gamma=0.99) of realized parameter DISPLACEMENT; jump forward along it before the
forward pass, the inner optimizer stepping from there. Lookahead 0.3*lr/max_lr, prefill 300 steps, off for the last
third. arXiv 2605.25395 · Track-3 record #39: **plain tuned Muon 3325 -> 3125 steps (-6.0%), n=20**. Still active
while Contra-Muon, Soft-Muon, Circuit-Muon and Aurora were all removed as neutral. Cost **cheap** - one param buffer,
two adds, no extra matmul. **PLAUSIBLE.**

## Tier 2 - cheap, reproduced, smaller

**10. Gated XSA (exclusive self-attention).** `y -= tanh(alpha[l,h])*((y*v).sum(-1)/(v.square().sum(-1).clamp_min(
1e-8)))*v`, alpha zero-init => bit-identical at step 0. arXiv 2603.09078 · PR #264 (records #82/#86): **-0.0045 nats at
matched steps, n=10, Welch p=0.0014** -> -30 steps. **Strongest reproduction chain in the report** (2.7B paper ->
independent 124M implementation -> re-derived by a fourth party -> composed by a fifth); the ungated fixed-alpha version
FAILED twice. Cost **cheap** (~1% of step). **PLAUSIBLE.** PR #317 DELETED the head gate once XSA landed - **we already
have learned sinks**, so test XSA with and against the sinks.

**11. Primer MDHA / static short conv (= Canon-B).** Width-3/4 causal depthwise conv on Q, K, V post-projection,
pre-RoPE; 3 shifted per-channel multiply-adds; identity-init. arXiv 2109.08668 · 2026 repro vs a RoPE-era baseline
arXiv 2606.03825 (MIT/MIT-IBM): 305M/15B, Nemotron PPL 19.12 -> **18.66 (-2.4%)** for +0.2M params (0.07%); 1.82B/100B:
11.71 -> 11.50. BLaLM (arXiv 2511.05560): 20.01 -> 12.37. **Reproduced twice, both non-Google, one against a modern
baseline.** Cost 4,608 MACs/token/layer ~ **0.13% FLOPs**, but 5.8-12% wall-clock on GPU. **NEEDS-ADAPTATION** (kernel
problem = NKI work); never landed in the speedrun in 2 years, so discount. Leaks 3 tokens across doc boundaries - mask
at BOS.

**12. Smear embeddings -> then DELETE layer-0 attention (as a pair).**
`x_t += lambda*sigmoid(W*x_t[:12])*x_{t-1}`, W zero-init, 12 params; then remove the first attention layer. PR #130
(#34), PR #131 (#35). Smear alone ~-5 steps of 1650 (marginal); **the removal is -0.8% wall-clock, n=10, p=0.0005**.
Cost **free then negative**. **PLAUSIBLE AS A PAIR, ruled out individually.** Mask smear at BOS.

**13. Drop the FIRST MLP block - never the last.** PR #120 (#30): **-3% step time, +1.5% steps => net -0.86%
wall-clock**, n=10 vs n=10 interleaved. The author's first claim (first AND last help) was a BUG - dropping the last MLP
makes training loss "much worse". Cost **negative** (-8d^2 params, -16d^2 FLOPs/token). **PLAUSIBLE BUT MARGINAL**;
consistent with our F_V23_D12 (depth 12 loses to 10).

**14. Knocking-Heads Attention.** One shared `d_h x d_h` matrix across all heads after the per-head V projection:
`V_i' = V_i T_v`. **Diagonal-initialized**; V-only carries nearly all the gain; random init destroys it. arXiv
2510.23052: at 100B tokens **-0.010 (MHA) / -0.020 (GQA-4)** loss, gains growing with scale. **Claimed only, single
paper.** `V_i T_v = X(W_v T_v)` - algebraically absorbable into W_v, ZERO added expressivity; it works purely through
optimization geometry. Cost **~0.0006%** if folded in weight space (recompute `W_v_eff = W_v @ T_v` per optimizer step)
vs 1.8%/layer in activation space. **PLAUSIBLE - one run to settle.** Optimizer-dependent: **Muon orthogonalizes
per-matrix gradients**, so the effect could amplify or vanish.

**15. Trimmed MUDD (last two layers, V + residual only).** A rank-64 GELU bottleneck emits PER-TOKEN mixing weights over
`{x0, h_{L-4}, h_{L-2}}`; W2 and biases zero-init => exact no-op at step 0. arXiv 2502.12170 · PR #259 (#81), PR #315
(#85): **-65 steps (-3.0%), n=10, p=0.0001**; gated follow-on another -4.3%. Paper (to 2.8B) + two independent 124M
records with matched-step controls. The PR deletes `backout_lambda` and folds it into a MUDD bias init - the DYNAMIC
version of the thing we tested and failed is what works. README verbatim: the full version (Q/K/V/residual every layer)
is *"too expensive at this model scale"*. Cost **cheap** (~1.2%/step, bought back 3x). **NEEDS-ADAPTATION.** Our
queued E28a burned on this failure mode (slow gates).

**16. Retie embedding to lm_head for the first ~2/3, then untie.** PR #175 (#51), PR #178 (#53): **retying alone -55
steps at identical per-step time (~-2.6%), n=9, p=0.0002.** modded-nanogpt UNTIED at record #8 (a big win); fourteen
months later RETYING was a bigger one. Cost **zero forward, negative optimizer/comm** (one fewer 4.2M matrix to shard
and all-reduce for 2/3 of the run). **PLAUSIBLE, NEEDS CARE.** Our `lm_head` init is std 0.001 vs `wte` std 0.8 - test
plain retying with a retuned init first.

## Tier 3 - free constants we are on the wrong side of

**17. Attention logit scale is a free tunable.** We use `1/sqrt(128) = 0.0884`; the speedrun uses a tuned constant `0.1`
under QK-norm, and nGPT derives the same target from unrelated first principles. Cost **zero**. **PLAUSIBLE - cheapest
experiment in the report.** Keep it a CONSTANT: the learnable per-head version (our queued E20) at 124M *"pushed every
head back to ~1.0 within 200 steps"*; SSMax failed the same way, catastrophically at 1.2B.

**18. Schedule constants** - four one-line changes: **cooldown fraction** ours 0.90 (~pure linear decay) vs Track-3
baseline 0.70, master 0.60, nanochat 0.65 - Schaipp et al. arXiv 2501.18965: with a fully-tuned peak LR the optimum IS
linear. **LR floor** ours 0.02x peak vs speedrun 0.10-0.15, nanochat 0.05 - the pro-decay-to-zero paper's Fig. 5 says
D2Z LOSES below ~4 tokens-per-param, which is us. **split cooldown fractions** (aux Adam 0.4 / matrix 0.8): Track-3
3325 -> 3175, -4.5%, n=10. **Muon momentum warmup** 0.85 -> 0.95 over ~300 steps then back down over the last ~50
(ours fixed 0.95). We screened FINAL_LR 0.0 (wash) and 0.10 (worse) at our current 0.90 cooldown - but these interact.

**19. Per-group LR ratios.** Track-3 baseline runs embed lr 0.7 / proj 0.004 (**175:1**); master uses lr_mul 75x on
bigram+value embeds, 5x on scalars. Kalra & Barkeshli arXiv 2605.21486: the entire practical muP-vs-SP gap IS the
embedding LR, concentrated early - so it helps MORE at short horizons. "Learning Less Is More" arXiv 2605.10504
multiplies **upper-half-layer W_Q/W_K LR by 0.25** for the first ~3-12% (270M, 2.5B tokens, 3 seeds): PPL 26.798 ->
**26.301**; halving the GLOBAL LR is much worse, so it is specifically upper-QK. Prime Intellect's search landed on
"attention LR 0.62x vs MLP 1.0x". Cost free. **PLAUSIBLE.** We screened EMBED_LR_SCALE 3.0 (+0.0009) and 1.4 (+0.0038)
around 2.0 - the UNTESTED part is the attention-vs-MLP split and the upper-layer QK damping.

**20. Adam on alternate steps only.** `if step % 2: adam.step()`. Record #39, -1.2% wall-clock. Cost **negative** -
halves AdamW work, and our profile shows the optimizer at 7-24% of step time. **PLAUSIBLE, free money.**

**21. Drop global gradient clipping.** Absent in BOTH speedrun codebases at 124M-560M with QK-norm + Muon; also removes
an all-reduce under ZeRO-1. Cost **negative**. "Clipping is mandatory" traces entirely to pre-QK-norm AdamW; no paper
ablates clipping in a modern QK-norm + Muon setup. **PLAUSIBLE, low risk.** Counter-evidence (ZClip, arXiv 2504.02507)
is 1B/50B without QK-norm.

**22. Variable-width ("bowtie") transformer.** Vary d per layer geometrically to a bottleneck at l* = 0.75L, d_l* = 0.3d;
resize PARAMETER-FREE by truncating down and copying forward the last-processed coordinates up. arXiv 2606.18246 ·
github.com/ZhaofengWu/variable-width-transformers. At **200M**: loss 3.452 -> 3.430 **while cutting FLOPs 3.2%**; holds
200M->2B; scaling-law fit says baseline quality at 77.8% of FLOPs. **Claimed only, single paper.** The BOTTLENECK shape
won (intuition favoured a diamond with compute in the middle). Cost **negative**. **NEEDS-ADAPTATION** - kernels for
many shapes, a NEFF per distinct width.

## Explicitly ruled out - do not re-litigate

- **nGPT** (Franke et al., arXiv 2505.22014, 32M-1B): the only independent test measures **1.29x** against a QK-normed
  baseline, not 4-20x; NVIDIA's own repo admits its baseline was bf16-handicapped. ~9% runtime cost.
- **Dynamic Tanh** (arXiv 2604.23434): benefit is entirely a function of tokens/param - +27.3% at T/P 0.016, **-18.8%
  (worse) at T/P 1.84**; we are at T/P ~8.4 on FLOP-bearing params; *"diverged to NaN at every alpha-init"* in the
  speedrun. **Downgrade arch-backlog item #16 to dead.**
- **Hyperball / MuonH**: Track-3 3325 vs tuned Muon 3250, submitted by the Hyperball author himself; arXiv 2607.22444
  shows the gain is an effective-step-size artifact appearing only late.
- **PSGD-Kron (3375) and Shampoo (4100) both LOSE to tuned Muon (3250)** at our step count; "PSGD is strong at small
  scale" does not survive a tuned multi-seed comparison.
- **AdEMAMix / Schedule-Free / AdaMuon / Dion**: all need >1500 steps for their slow state to warm up. Everett & Qiu
  (arXiv 2609.04577): optimal optimizer memory rises 20 -> 1280 as overtraining goes 1x -> 128x; we are ~0.1x.
- **Differential attention** (-9% throughput by the authors' own measurement; negative at 1.2B iso-compute, z=-2.45),
  **sigmoid attention** (worst of 20, z=-77.68), **true Softpick** (authors' own numbers worse on loss at 340M and 1.8B;
  the 2026 study's win is for a different formula and it deletes the sinks we built), **hyper-connections** (z=-9.79 at
  1.2B; *"didn't help convergence at all"* at 124M), **stick-breaking / selective attention / talking-heads / MLA / GQA /
  MoE / label smoothing / z-loss / MQA** - each ruled out on cost, regime, or metric.

## Two methodological warnings worth more than several items above

1. **Optimizer and schedule rankings flip mid-training** because of LR decay (arXiv 2509.02046): our 5-minute
   `autoresearch` proxy with a compressed schedule mis-ranks every schedule/optimizer item, and specifically PENALIZES
   Tail-EMA, which only pays in the tail.
2. **Single-LR ablations measure "did this break my tuned LR", not "is this better"** (arXiv 2608.11859: with 4 HP
   configs per condition, a scaling law is completely absent). Of our 8
   recent failures, **value-residual learning and per-channel gates are the most likely false negatives** - both change
   effective update scale. The speedrun's per-channel gate reads only the first **12** residual dims and sits in the
   **Adam** group; widening it to the full residual measured worse than no gate at all, so a full-width or Muon-group
   version would be a gate-input-width failure, not a gating failure.
3. Run-to-run sigma here is **0.0010-0.0019 val loss**; the speedrun community requires n~10 and p<0.01. Several results
   above were first published as wins and later retracted as noise; our 900 s screens resolve ~0.002 - at or above that
   sigma, so single screens cannot separate the Tier-2 items from noise.

## If we only do five

Open: canonical token masking (#2), prefix-token-prediction loss (#1), then the Tier-3 constants (#19-22).
Already landed: #7 (tail-EMA blend is in the vehicle since v26). Closed on Trn2: #3/#4 (plan-097), #5 (plan-097).
**Status conflict to resolve on #2:** git bd48268 reports a closed "canonical mask eval-only -0.0007" run, while
`wild-ideas-2026-09-15.md` still lists it as never built.
