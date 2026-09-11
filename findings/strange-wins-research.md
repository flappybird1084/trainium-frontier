# Strange wins & novel architectures (research sweep, 2026-09-08)

Commissioned after 8 consecutive loss-side architecture failures. Brief: find techniques with a documented,
reproduced gain whose mechanism was NOT understood or ran against theory, plus novel 2024-26 architectures
that hold at ~150M params / <1B tokens. Sources are primary (papers, official repos, record changelogs).

**Verdict labels are the researcher's, filtered against our regime** (30 min wall-clock, ~2300-5500 steps,
seq 1024, vocab 8192, Trainium2, compute-bound at 56.8% MFU so added FLOPs cost steps directly).

## The meta-finding, which matters more than any single item

The modded-nanogpt speedrun (124M, ~400M tokens, wall-clock scored, multi-seed, adversarial) is the only public
benchmark in our regime, and **in two years not one famous architecture paper has ever landed there.** What lands
is near-zero-FLOP, shape-preserving, zero-init-transparent tweaks. That is our own 420-step-regime rule,
externally confirmed. There is also a **Track 3 "optimization" leaderboard** (fixed architecture, minimize
*steps*, unlimited wall-clock, n=4-30 seeds, 0.0045 loss ~ 100 steps) which is the most in-distribution evidence
source available to us and which we were not using.

Also established: **~134M of our 187M params are the bigram table** - ~80% of parameters cost ~0 FLOPs. The real
compute model is a ~36M-param dense net where the LM head is ~10% of forward FLOPs.

## Tier 1 - strange, free, reproduced

**1. Prefix-token-prediction aux loss.** For target T, a static table gives T' = the token whose bytes are the
longest proper prefix of T's bytes; add `prefix_w * (lse - z_T')` using the SAME logits. Weight 0.25 for the first
third, linear to 0 over the second, off after; disabled at eval.
https://github.com/KellerJordan/modded-nanogpt/pull/337 (record #88). **-15 steps / -0.8 s** at 124M. Speedrun-
verified. Strange: injects sub-token BYTE structure through the existing head; nobody predicted the tokenizer's
merge tree carried usable signal. Cost **free**. **PLAUSIBLE - top pick.** Author reports it stacks with MTP
rather than replacing it, so our MTP failure does not rule it out.

**2. Canonical token masking.** BPE is deterministic, so most next-tokens are unreachable given the previous
token; mask non-canonical continuations and renormalize. Still a valid probability model.
https://github.com/KellerJordan/modded-nanogpt/pull/350 · paper https://arxiv.org/abs/2506.07956.
**-0.0017 nats over 20 runs -> -15 steps.** **Open PR, not merged** - but the mechanism is provable, not
empirical. Strange: pure free-energy recovery from a modeling error shipped since GPT-2. Cost **free**; at vocab
8192 it is an 8192x8192 bitmask (8.4 MB) keyed on the previous token only. Our scorer recomputes CE from
`model(x)` logits, so the mask inside the forward is legal, stateless, causal. **PLAUSIBLE - highest expected
value per line of code in the report.**

**3. Bigram hash "sign trick".** Multiply each retrieved row by a frozen random +-1 pattern keyed to the bigram
(`sign_idx = (prev ^ cur) % 8192`) so colliding bigrams cancel in expectation instead of summing. Paired with
**narrow rows**: 5x vocab @ 768-dim -> 15x vocab @ 192-dim at equal memory, injected partial-width.
https://github.com/KellerJordan/modded-nanogpt/pull/299 (record #83). Basis: Chen et al. 2015 hashing trick.
**-1.5 s of an 80 s record** (sign ~-0.5 s, dim reduction ~-0.5 s). Speedrun-verified. Strange: the PR author
says outright *"I do NOT understand the dynamics of bigrams sharing a slot"* - VIP buckets for top-16k bigrams
and reducing slots both FAILED; only random-sign decorrelation worked. Cost **free**.
**PLAUSIBLE - the single most on-target item for our architecture**, since our table is 72% of params and
memory-capped (2^19 OOMs). At 128-dim rows we get 4x the rows in the same memory.

**4. Hash-table size must NOT be an integer multiple of vocab size.** With a polynomial rolling hash, collision
counts spike when the table size is near an integer multiple of the base vocab. Design rule verbatim: *"The
vocabulary size of N-gram Embedding should significantly deviate from integer multiples of the base vocabulary
size."* https://arxiv.org/abs/2601.21204 (LongCat-Flash-Lite, Fig. 3). Single paper; effect shown as collision-
count spikes, not separately quantified. Cost **free**.
**PLAUSIBLE, ACT ON IT: our BIGRAM=2^17 is exactly 16x vocab and 2^18 is exactly 32x vocab.** Try a prime near
2^18 (e.g. 262139) - a one-constant test.
CAVEAT (ours, verified): our hash is Knuth multiplicative (`pair * -1640531527 >> 13 & (SIZE-1)`), not the
polynomial rolling hash the paper analyses. The aliasing argument may not transfer. Cheap enough to test anyway.

**5. Mimetic V-O init - stop zero-initing the attention out-projection.** Per head `W_V = +sqrt(b) I`,
`W_O = -sqrt(b) I` with b=0.05 plus sigma=0.01 noise, so `V@O^T ~ -bI` at init instead of `O = 0`.
https://github.com/KellerJordan/modded-nanogpt/pull/214 (record #64) · paper https://arxiv.org/abs/2305.09828.
**-20 steps of 1575, n=7 interleaved, p=0.0036**, with a matched-step control explicitly failing (p=0.115).
Paper is vision-only; the LM application is one careful 124M study plus two independent "increase attention init"
signals. Strange: **directly contradicts the zero-init-output-projection doctrine we currently use** - zero-init
makes attention a no-op at step 0 and the model burns early steps breaking symmetry. Cost **exactly zero**.
**PLAUSIBLE - a two-line swap.** b>=0.10 destabilizes; mimetic on Q-K measured negative; attention `c_proj`
only, leave `mlp.c_proj` zero-init.

**6. Partial key offset + half-truncate RoPE.** Zero the top half of `inv_freq` so half of each head's dims are
unrotated, then shift the STATIONARY key dims one position forward:
`k[:,1:,:,D//4:D//2] = k[:,:-1,:,D//4:D//2]`. Record comment: *"Enables 1-layer induction."*
https://github.com/KellerJordan/modded-nanogpt/pull/169 (record #49) · RoPE-fraction ablation
https://github.com/KellerJordan/modded-nanogpt/discussions/150. **-50 steps (~3.6%), n=11, p=0.0004**; full-dim
RoPE on all layers measured clearly worse than 50%. Corroborated independently (LlamaCanon, MHA2MLA
arXiv 2502.14837). Strange: nobody has explained why a partial-WIDTH key shift beats a full one; it converts a
learned induction circuit into free structure. Cost **free** - strided copy + zeroing half a frequency table, no
new shapes. **PLAUSIBLE - strongest in-regime statistics in the report.** We use full-dim RoPE, so this is two
changes in one. Mask the shift at document starts.

**7. Tail-EMA checkpoint blend.** Over the last ~15% of steps keep `ema += (w - ema)/tau` (tau~150) for every
param EXCEPT token-embedding tables; save `w_final = 0.65*w + 0.35*ema`. Training untouched.
PRs https://github.com/KellerJordan/modded-nanogpt/pull/325 / #328 (Track-3 records #45/#46), Track-1 port #364.
**-42 steps of 2745 (-1.6%)**, called "the single biggest lever" in that record; Track-1 port **-0.0024 val loss**.
**Reproduced by three independent parties** plus IMU-1 (arXiv 2602.02522) - best-replicated item here. Strange:
folk theory says LR-decay-to-near-zero already IS weight averaging, so EMA should be redundant. It is not - full
SWA/LAWA (lags) and forward BEMA (overshoots) were both tested and rejected; only the PARTIAL blend works.
Cost **free**, tail only, no forward change. **PLAUSIBLE - do first.** Fits `load_for_eval` exactly.
Use fp32 EMA state: at tau=150 the update weight (6.7e-3) is below 2x bf16 epsilon and a bf16 EMA silently
stalls. Our 5-min autoresearch proxy will systematically PENALIZE this - validate at full length.

**8. Cautious Weight Decay.** `mask = 1[update*w > 0]; w *= (1 - lr*lambda*mask)` - decay only coordinates the
update is already shrinking. https://arxiv.org/abs/2510.12402 · speedrun records #43, #50, Track-3 #46.
Track-1 -1.25% then -0.85% wall-clock (two contributors); Track-3 isolated -19 steps. Dose is an inverted U:
0.0125 weak, **0.025 optimal**, 0.05 over-shrinks. **Reproduced four times.** Strange: isotropic WD is nearly a
no-op on the Muon path since Muon renormalizes update magnitude; the sign gate changes update SHAPE, the one axis
norm control leaves untouched. Cost **free**.
STATUS (ours, verified): our NKI Muon kernel ALREADY does this - `mask = (g*w >= 0)` at the weight-decay step
(`train3_submission.py:847,860`). **The remaining move is extending it to the AdamW groups.**

**9. EMA-Nesterov outer wrapper.** Keep an EMA (gamma=0.99) of realized parameter DISPLACEMENT; jump forward
along it before the forward pass and let the inner optimizer step from there. Lookahead 0.3*lr/max_lr, prefill
300 steps, off for the last third. https://arxiv.org/abs/2605.25395 · Track-3 record #39.
**Plain tuned Muon 3325 -> 3125 steps (-6.0%), n=20** - the relevant number, since our stack is closer to plain
Muon than to the 10-technique Track-3 stack. Still active in the current record while Contra-Muon, Soft-Muon,
Circuit-Muon and Aurora were all removed as neutral. Strange: classic Lookahead was thought redundant with
momentum; EMA-ing the UPDATE rather than the parameters fixes the noise problem. Cost **cheap** - one param
buffer, two adds, no extra matmul. **PLAUSIBLE.** Measure jointly with #7 (both smooth the trajectory).

## Tier 2 - cheap, reproduced, smaller

**10. Gated XSA (exclusive self-attention).** Remove the component of attention output along the token's own
value vector: `dot=(y*v).sum(-1); den=v.square().sum(-1).clamp_min(1e-8); y -= tanh(alpha[l,h])*(dot/den)*v`,
alpha zero-init => bit-identical at step 0. https://arxiv.org/abs/2603.09078 ·
https://github.com/KellerJordan/modded-nanogpt/pull/264 (records #82/#86). **-0.0045 nats at matched steps,
n=10, Welch p=0.0014** -> -30 steps. **Strongest reproduction chain in the report** (paper to 2.7B -> independent
124M implementation -> re-derived by a fourth party -> composed by a fifth). Strange: softmax is supposed to learn
to downweight self-attention; XSA says the self-value component is structurally redundant (already in the
residual) and needs a projection, not a reweighting. The ungated fixed-alpha version FAILED twice; the learnable
gate is the whole trick. Cost **cheap** (~1% of step, no matmul). **PLAUSIBLE.**
Redundancy risk: PR #317 DELETED the head gate once XSA landed, on the grounds both provide attention-sinking -
**we already have learned sinks**, so test XSA with and against the sinks; do not assume additivity.

**11. Primer MDHA / static short conv (= Canon-B).** Width-3/4 causal depthwise conv on Q, K, V after projection,
before RoPE. Implement as 3 shifted per-channel multiply-adds, not a conv op. Identity-init.
https://arxiv.org/abs/2109.08668 · **independent 2026 reproduction against a RoPE-era baseline**:
https://arxiv.org/abs/2606.03825 (MIT/MIT-IBM), 305M/15B tokens, Nemotron PPL 19.12 -> **18.66 (-2.4%)** for
+0.2M params (0.07%); 1.82B/100B: 11.71 -> 11.50. Also BLaLM (arXiv 2511.05560) at ~270M on a 10M-word budget:
strict-small PPL 20.01 -> 12.37. **Reproduced twice, both non-Google, one against a modern baseline.** Strange:
attention was assumed to subsume local mixing; the whole Mamba-2/Based/GDN line ships this exact conv and nobody
ablated it. Related: fixed RANDOM per-channel weights already work (a shared scalar kernel does not). Cost 4,608
MACs/token/layer ~ **0.13% FLOPs**, but measured 5.8-12% wall-clock on GPU because it is pure bandwidth +
launches. **NEEDS-ADAPTATION** - the FLOP/wall-clock gap is a kernel problem, which is what NKI is for. But a
depthwise conv has never landed in the speedrun in two years, and our 2^18 bigram table already feeds local
n-gram structure into the stream, so discount the marginal gain. Leaks 3 tokens across doc boundaries; mask at BOS.

**12. Smear embeddings -> then DELETE layer-0 attention (as a pair).**
`x_t += lambda*sigmoid(W*x_t[:12])*x_{t-1}`, W zero-init, 12 params; then remove the first attention layer.
https://github.com/KellerJordan/modded-nanogpt/pull/130 (#34) and /pull/131 (#35). Smear alone ~-5 steps of 1650
(self-reported as marginal); **the removal is -0.8% wall-clock, n=10, p=0.0005.** Strange: the first attention
layer turned out to be doing nothing but previous-token attention; once smear does that for 12 parameters, a whole
attention layer becomes deletable. Cost **free then negative**.
**PLAUSIBLE AS A PAIR, ruled out individually.** Mask smear at BOS.

**13. Drop the FIRST MLP block - never the last.** https://github.com/KellerJordan/modded-nanogpt/pull/120 (#30).
**-3% step time, +1.5% steps => net -0.86% wall-clock**, n=10 vs n=10 interleaved. Strange: the PR's own
correction is the finding - the author first claimed first AND last helped; that was a BUG, and dropping the last
MLP makes training loss "much worse." The asymmetry is the result. Cost **negative** (-8d^2 params,
-16d^2 FLOPs/token). **PLAUSIBLE BUT MARGINAL** - removing a whole matmul is one of the few FLOP cuts that
converts cleanly to wall-clock at 56.8% MFU. Consistent with our F_V23_D12 (depth 12 loses to 10).

**14. Knocking-Heads Attention.** One shared `d_h x d_h` matrix across all heads after the per-head V projection:
`V_i' = V_i T_v`. **Diagonal-initialized**; V-only carries nearly all the gain; random init destroys it.
https://arxiv.org/abs/2510.23052. At 100B tokens: **-0.010 (MHA) / -0.020 (GQA-4)** loss; gains grow with scale.
**Claimed only, single paper.** Strange, and this is the point: `V_i T_v = X(W_v T_v)`, so it is algebraically
absorbable into W_v and adds ZERO expressivity - it works purely by changing optimization geometry through tying
all heads' gradients to a shared factor. Cost **~0.0006%** if folded in weight space (recompute
`W_v_eff = W_v @ T_v` once per optimizer step) rather than in activation space (1.8%/layer).
**PLAUSIBLE - one run to settle.** Because the effect is pure reparameterization it is optimizer-dependent, and
**Muon orthogonalizes per-matrix gradients** - the effect could amplify or vanish. Unresolvable from literature.

**15. Trimmed MUDD (last two layers, V + residual only).** A rank-64 GELU bottleneck emits PER-TOKEN mixing
weights over `{x0, h_{L-4}, h_{L-2}}`; W2 and biases zero-init => exact no-op at step 0.
https://arxiv.org/abs/2502.12170 · https://github.com/KellerJordan/modded-nanogpt/pull/259 (#81), /pull/315 (#85).
**-65 steps (-3.0%), n=10, p=0.0001**; gated follow-on another -4.3%. Paper (to 2.8B) + two independent 124M
records with matched-step controls. Strange: **the PR deletes `backout_lambda` and folds it into a MUDD bias
init** - the DYNAMIC version of the thing we tested and failed is what works, and static dense skips measured
slightly worse. README verbatim: the paper's full version (Q/K/V/residual every layer) is *"too expensive at this
model scale"*. Cost **cheap** (~1.2%/step, bought back 3x). **NEEDS-ADAPTATION.** Our queued E28a burned on
exactly this failure mode (slow gates); use a dedicated low-LR group and bias-init to our current static skip
values so step 0 is a no-op. Heaviest implementation here - do it last.

**16. Retie embedding to lm_head for the first ~2/3, then untie.**
https://github.com/KellerJordan/modded-nanogpt/pull/175 (#51), /pull/178 (#53). **Retying alone -55 steps at
identical per-step time (~-2.6%), n=9, p=0.0002.** Strange, genuinely: modded-nanogpt UNTIED at record #8 and it
was a big win; fourteen months later RETYING was a bigger one. Something in the intervening stack - value
embeddings, bigram hash tables, sparsified embeddings, **all of which we have** - removed the need for a separate
input embedding. Cost **zero forward, negative optimizer/comm** (one fewer 4.2M matrix to shard and all-reduce for
2/3 of the run). **PLAUSIBLE, NEEDS CARE.** Our `lm_head` init is std 0.001 while `wte` is std 0.8 - retying
forces one init, so test plain retying with a retuned init first. Allocate both tensors up front and flip an alias
flag, or the mid-run materialization triggers a NEFF compile.

## Tier 3 - free constants we are on the wrong side of

**17. Attention logit scale is a free tunable.** We use `1/sqrt(128) = 0.0884`; the speedrun uses a tuned constant
`0.1` under QK-norm, and nGPT derives essentially the same target from unrelated first principles. Cost **zero**.
**PLAUSIBLE - cheapest experiment in the report.** Keep it a CONSTANT: the learnable per-head version (our queued
E20) was tried at 124M and *"the optimizer pushed every head back to ~1.0 within 200 steps"*; SSMax failed the
same way and catastrophically at 1.2B.

**18. Schedule constants** - four one-line changes, each a merged record or published result:
- **cooldown fraction**: ours is 0.90 (~pure linear decay); Track-3 baseline 0.70, master 0.60, nanochat 0.65.
  Schaipp et al. https://arxiv.org/abs/2501.18965 prove that with a fully-tuned peak LR the optimum IS linear -
  so sweep 0.3-0.7 JOINTLY with peak LR, not alone.
- **LR floor**: ours 0.02x peak; speedrun 0.10-0.15, nanochat 0.05. The pro-decay-to-zero paper's own Fig. 5 says
  D2Z LOSES below ~4 tokens-per-param, which is us.
- **split cooldown fractions** (aux Adam 0.4 / matrix 0.8): Track-3 3325 -> 3175, -4.5%, n=10.
- **Muon momentum warmup** 0.85 -> 0.95 over ~300 steps then back down over the last ~50 (ours is fixed 0.95).
We screened FINAL_LR 0.0 (wash) and 0.10 (worse) at our current 0.90 cooldown - but these interact, so a joint
sweep is not the same experiment.

**19. Per-group LR ratios.** Track-3 baseline runs embed lr 0.7 / proj 0.004 (**175:1**); master uses lr_mul 75x
on bigram+value embeds, 5x on scalars. Kalra & Barkeshli https://arxiv.org/abs/2605.21486 show the entire
practical muP-vs-SP gap IS the embedding LR, concentrated early - so it helps MORE at short horizons.
Independently, "Learning Less Is More" https://arxiv.org/abs/2605.10504 multiplies **upper-half-layer W_Q/W_K LR
by 0.25** for the first ~3-12%: 270M, seq 1024, 2.5B tokens, 3 seeds, PPL 26.798 -> **26.301**; halving the GLOBAL
LR is much worse, so it is specifically upper-QK. Prime Intellect's autonomous search independently landed on
"attention LR 0.62x vs MLP 1.0x". Cost free. **PLAUSIBLE.** We screened EMBED_LR_SCALE 3.0 (+0.0009) and 1.4
(+0.0038) around 2.0, so embedding LR may be tuned - the UNTESTED part is the attention-vs-MLP split and the
upper-layer QK damping.

**20. Adam on alternate steps only.** `if step % 2: adam.step()`. Record #39, -1.2% wall-clock. Cost **negative** -
halves AdamW work, and our profile shows the optimizer at 7-24% of step time. **PLAUSIBLE, free money.**

**21. Drop global gradient clipping.** Absent in BOTH speedrun codebases at 124M-560M with QK-norm + Muon; also
removes an all-reduce under ZeRO-1. Cost **negative**. Strange: "clipping is mandatory" traces entirely to
pre-QK-norm AdamW; no paper ablates clipping in a modern QK-norm + Muon setup. **PLAUSIBLE, low risk** - spike
risk falls with shorter runs. Counter-evidence (ZClip, arXiv 2504.02507) is 1B/50B without QK-norm.

**22. Variable-width ("bowtie") transformer.** Vary d per layer geometrically to a bottleneck at l* = 0.75L,
d_l* = 0.3d; resize PARAMETER-FREE by truncating down and copying forward the last-processed coordinates up.
https://arxiv.org/abs/2606.18246 · github.com/ZhaofengWu/variable-width-transformers. At **200M**: loss
3.452 -> 3.430 **while cutting FLOPs 3.2%**; holds 200M->2B; scaling-law fit says baseline quality at 77.8% of
FLOPs. **Claimed only, single paper.** Strange: intuition favoured a diamond shape with compute in the middle;
the BOTTLENECK won, apparently as a structural regularizer against low-rank collapse in middle layers.
Cost **negative**. **NEEDS-ADAPTATION** - genuine negative cost, but needs kernels for many shapes and each
distinct width is a NEFF shape (compile time is excluded from our budget, so viable but not free to develop).

## Explicitly ruled out - do not re-litigate

- **nGPT**: the only genuinely independent test (Franke et al., https://arxiv.org/abs/2505.22014, 32M-1B) measures
  **1.29x** against a QK-normed baseline, not 4-20x; NVIDIA's own repo admits the baseline was bf16-handicapped
  and *"it also could be that something is wrong in this implementation."* ~9% runtime cost.
- **Dynamic Tanh**: https://arxiv.org/abs/2604.23434 shows the benefit is entirely a function of tokens/param:
  +27.3% at T/P 0.016, **-18.8% (worse) at T/P 1.84**; we are at T/P ~8.4 on FLOP-bearing params. Also *"diverged
  to NaN at every alpha-init"* in the speedrun. **Downgrade arch-backlog item #16 to dead.**
- **Hyperball / MuonH**: Track-3 3325 vs tuned Muon 3250, submitted by the Hyperball author himself;
  arXiv 2607.22444 shows the gain is an effective-step-size artifact appearing only late.
- **PSGD-Kron (3375) and Shampoo (4100) both LOSE to tuned Muon (3250)** at our step count - the "PSGD is strong
  at small scale" reputation does not survive a tuned multi-seed comparison.
- **AdEMAMix / Schedule-Free / AdaMuon / Dion**: all need >1500 steps for their slow state to warm up. Everett &
  Qiu (arXiv 2609.04577): optimal optimizer memory rises 20 -> 1280 as overtraining goes 1x -> 128x; we are ~0.1x.
- **Differential attention** (-9% throughput by the authors' own measurement; negative at 1.2B iso-compute,
  z=-2.45), **sigmoid attention** (worst of 20, z=-77.68), **true Softpick** (authors' own numbers worse on loss
  at 340M and 1.8B; the 2026 study's win is for a different formula and it deletes the sinks we built),
  **hyper-connections** (z=-9.79 at 1.2B; *"didn't help convergence at all"* at 124M), **stick-breaking /
  selective attention / talking-heads / MLA / GQA / MoE / label smoothing / z-loss / MQA** - each ruled out on
  cost, regime, or metric.

## Two methodological warnings worth more than several items above

1. **Optimizer and schedule rankings flip mid-training** because of LR decay (arXiv 2509.02046). Our 5-minute
   `autoresearch` proxy with a compressed schedule will mis-rank every schedule/optimizer item and will
   specifically PENALIZE Tail-EMA, which only pays in the tail.
2. **Single-LR ablations measure "did this break my tuned LR", not "is this better"** (arXiv 2608.11859: with 4 HP
   configs per condition a scaling law is completely absent). Of our 8 recent failures, **value-residual learning
   and per-channel gates are the most likely false negatives** - both change effective update scale. Also: the
   speedrun's per-channel gate reads only the first **12** residual dims and sits in the **Adam** group; widening
   it to the full residual measured worse than no gate at all. If ours was full-width or in the Muon group, that
   was a gate-input-width failure, not a gating failure.
3. Run-to-run sigma in this regime is **0.0010-0.0019 val loss**; the speedrun community requires n~10 and p<0.01.
   Several results here were first published as wins and later retracted as noise. Our 900 s screens resolve
   ~0.002 - at or above that sigma, so single screens cannot separate the Tier-2 items from noise.

## If we only do five

Canonical token masking (#2), tail-EMA blend (#7), mimetic V-O init (#5), bigram modulus + sign trick (#3/#4),
prefix-token-prediction loss (#1). All five are free or negative cost, all five contradict something we currently
do, and four of the five aim straight at the bigram table that dominates our parameter budget.
