# The repeat regime - the data axis, and three untried moves (2026-09-14)

Written after the PACK=4 result. Corrects two claims carried in `plan-097.md` and the Sep 13 handoffs: the
"out of data" framing and the 1800 s noise band. Ideas here are **hypotheses with citations, not results** -
nothing below has been gated. Gate plan at the bottom.

## 1. CORRECTION: we are not out of data

`plan-097.md` and `handoff-2026-09-13b.md` treat the corpus as a wall and conclude that further throughput buys
only repetition. The Sep 14 data work already found empirically that "data repetition (~2 passes) is ~free."
The literature puts a number on how much further that goes, and it is **~2.8x current usage**.

- [Muennighoff et al., *Scaling Data-Constrained Language Models*](https://arxiv.org/abs/2305.16264): repeating
  data for up to **~4 epochs yields negligible degradation** vs equivalent fresh data at fixed compute; gains
  continue (with decay) to ~16 epochs and only vanish near 40.
- Measured here (Sep 14): corpus = **315M tokens** (5 shards); a v40-class run = **446.7M tokens = 1.42 epochs**.
- 4 epochs of 315M = **~1.26B tokens**. We use 0.45B. **~2.8x more throughput still converts at approximately
  the old exchange rate** (1% step time ~ 0.0003 bpb) before repetition meaningfully bites.

Consequence: the kernel/step-time levers (plan-097 B, D, F) are **not** devalued by the epoch boundary the way
the Sep 13 handoff claimed. They are devalued by being hard, which is a different problem.

Also stale in `plan-097.md`: "the leaderboard top (~0.93 portal) is ~0.06 away." Sep 14 portal reading has
**leaders at 0.967** and v40 scored 0.9915 (8th). At portal = local + ~0.0065 that is ~0.960 local, so v42
(0.9812) is **~0.021 local** off the lead, not 0.06. The lead is roughly 5x closer than the plan assumes.

## 2. What PACK=4 actually established (and what it did not)

| run | val_bpb | vs A_REF 0.985741 |
|---|---|---|
| A_REF (v40 config, same day) | 0.985741 | - |
| A_STREAM (100% coverage, eval-matched rows) | 0.987395 | **+0.0017 LOSS** |
| A_PACK2 | 0.981913 | -0.0038 |
| A_PACK4 / A_PACK4b / T59_gate (v42) | 0.981426 / 0.981073 / **0.981197** | **-0.0045** |

**-0.0045 from one env var at zero step cost is the largest single gain since v40** (v37->v40, four versions of
kernel work, totalled -0.0114). After ~30 versions of step-time engineering, the data axis paid more per
box-hour than anything else on the board.

Do not over-read it. The two obvious data hypotheses both **failed**:
- **Coverage is not the mechanism.** A_STREAM had the *most* coverage (100% vs 72% provided vs 87% PACK=4) and
  lost. It also delayed epoch 2 the longest (step 2410 vs 1720) and still lost.
- **Eval-distribution matching is not the mechanism.** A_STREAM's rows were 99.8% mid-doc, matching the eval
  rows' 99.9%, and lost.

What is left as the mechanism: **document integrity in the training row** - best-fit whole documents, long
documents placed whole. The provided PACK=1 packer never trained on 28% of the corpus (18% truncated, 10% never
placed: docs > 1025 never fit); PACK=4 places documents up to 4k whole. Cutting every row (stream) destroys
integrity even at 100% coverage. Treat "keep documents intact" as the live variable, **not** "cover more corpus"
and **not** "look like eval."

Open: the curve is PACK 1 (0.9857) -> 2 (0.9819) -> 4 (0.9812) -> ? and **PACK=8 crashes the eager sidecar at
batch 1** (unknown cause, Sep 14). Since PACK 2->4 is nearly flat, the optimum is plausibly at or just past 4 and
the remaining value here is small - but fixing the PACK=8 sidecar crash is the only way to know, and that crash
may share a root cause with the OOB flake (plan-097 lever E, also the portal-crash risk).

## 3. CORRECTION: the 1800 s noise band is ~0.0004, not 0.001-0.002

`plan-097.md` states "1800 s noise band ~0.001-0.002 (v40 vs v39 = 0.0014)." That number came from comparing
**two different configs**, which measures config difference + noise, not noise.

We now have a genuine same-config triple:

- PACK=4 at full length: **0.981426, 0.981073, 0.981197** -> spread **0.00035**, sd ~0.00018.
- v40 config: gate 0.985161 vs A_REF 0.985741 -> 0.00058 (different days/vehicle shas).

**The instrument is ~3-5x sharper than the plan assumes.** Every lever discarded as "within the band" at
0.001-0.002 should be re-read against ~0.0004. That directly affects lever A's verdicts: `P4_MLR085` (+0.0002)
and `P4_MLR055` (-0.0001) are genuinely flat, but anything previously called a wash at 0.001-0.0015 was not
necessarily a wash.

**No further sharpening is available from seeding** (corrected): every run already uses `manual_seed(42)` and the
same data order, so all A/B comparisons are *already* effectively paired. The ~0.0004 spread is therefore the
**paired residual** - device nondeterminism in the collectives plus step-count jitter - not seed or ordering
variance. That is the floor. Levers below ~0.0008 stay unresolvable by a single gate; the only way to see them
is repeats (n=3 gets the standard error to ~0.0001), which costs 1.5 h per verdict.

## 4. The thesis: every recipe we inherited is a single-epoch recipe

The organizers' baseline, the 3090 ladder, and the whole modded-nanogpt lineage we borrowed from
(value embeddings, sinks, U-net skips, bigram tables, Muon) are tuned on FineWeb-style budgets where **no token
is ever seen twice**. We are at 1.42 epochs and heading higher as the step gets faster.

Nobody else in this competition is in the repeat regime, because nobody else is this fast. The repeat regime
wants different hyperparameters and a different objective, and that space is unexplored **by construction** -
the lineage we copy from never had to explore it. PACK=4 is the first evidence that the data axis is live.

Three moves follow. All three are **review-clean**: no eval-mode branch, no extra head at eval, no change to
`load_for_eval` or the forward path's purity (the v12/v32 rejection modes).

> **VERDICT (Sep 15, 06:15 UTC): the thesis is FALSIFIED on this line.** Ideas 1 and 3 were gated at full length
> on the PACK=4 base and both lost monotonically - see the CLOSED blocks below. Regularization does not pay at
> 1.4 epochs here, and the packing win was *not* a repeat-regime effect. Idea 2 (self-distillation) remains
> untried and is the only survivor. Keep this file for the two corrections in s1-s3, which stand independently.

### Idea 1 - CLOSED (lost). Repeat-regime regularization: strong WD + masked-input aux loss

The heresy: the speedrun does not use dropout or heavy WD, and `nki-kernel-playbook-2.md` 4.11 records that as
our reason to skip it. Correct for one epoch; inverted for several.

- [Xu et al., *Data-Constrained LM Pretraining: Improved Regularization and Scaling Laws*](https://arxiv.org/abs/2606.06888)
  (Jun 2026): **MIR** (masked-input regularization) = an auxiliary next-token loss on randomly masked inputs,
  **on top of strong weight decay**. Improves validation loss across 72M-1.4B in the data-constrained regime.
  We are 197M. No architectural change, **no inference overhead** - eval path untouched.
- **Re-open WD.** WD 0.20 was rejected at +0.0032 and WD 0.05 as a wash - both on **900 s screens** at the
  65k-batch config, before epoch 2 was a factor. Per the v41 lesson (screen LR wins do not transfer, +0.009 at
  full length), those verdicts do not bind. Strong WD is also the *precondition* for MIR in the paper.
- **Knobs:** `WD` re-gated at 0.1 / 0.2 / 0.3; then `MIR_FRAC` (0.05-0.15 of input positions masked) and
  `MIR_WEIGHT`. Implement masking in-place on a fraction of rows of the existing micro-batch - elementwise work
  on a Vector engine at 47-61% occupancy, so ~zero step cost. No second forward pass.
- **Expected:** -0.002 .. -0.005. Highest confidence, lowest effort of the three.
- **Falsified if:** WD is flat 0.1-0.3 at full length on the PACK=4 base *and* MIR at 0.1 is a wash.

**RESULT - worse than falsified, both curves are monotonically bad** (1800 s gates vs T59 0.981197):

| gate | val_bpb | vs T59 |
|---|---|---|
| WD 0.1 (= current) | 0.981197 | - |
| P4_WD02 | 0.985687 | **+0.0045** |
| P4_WD03 | 0.992145 | **+0.0109** |
| P4_MIR005 | 0.986213 | **+0.0050** |
| P4_MIR010 | 0.990758 | **+0.0096** |

MIR also measured +0.014 at fixed steps locally, so this is not a step-time artifact - the regularization is
genuinely harmful here. Reading: at 1.4 epochs on a 197M model with a 30-minute budget we are **undertrained,
not overfit**, so the paper's data-constrained regime (where the model has capacity to spare and repetition is
the binding cost) does not describe us. Regularization only pays once repetition actually starts hurting, and
at 1.4 epochs it does not. **Do not re-litigate WD or input masking below ~3 epochs.**

### Idea 2 - Cached-logit self-distillation on the repeat pass

The repeat-pass problem is not that tokens recur; it is that they recur with **hard one-hot targets we have
already fit**. Low bits per token.

During epoch 1 we compute a full distribution over 8192 vocab entries for every token and discard it. Cache it,
and train the repeat pass against a blend of the hard label and our own cached soft distribution
(born-again / self-distillation: a teacher no better than the student still transfers which wrong answers were
plausible). **Zero extra forward cost - we are reading a log, not running a teacher.**

- **Storage:** top-8 per token = 315M x 8 x 4 B ~ **10 GB host RAM**, streamed with the loader (bf16 logits +
  int16 ids halves it to ~5 GB). Per micro-step: 16k tokens x 8 x 4 B = 512 KB. Trivial DMA.
- **Plumbing exists:** the embedding sidecar already moves per-token gather/scatter outside the compiled graph.
  Same shape of problem.
- **Knobs:** `SELFDIST_K` (8), `SELFDIST_W` (soft-target blend weight), `SELFDIST_FROM` (only cache from the
  last ~40% of epoch 1, so the teacher is not an early-training checkpoint).
- **Risk:** the teacher is strictly weaker than the student by epoch 2. Mitigate via `SELFDIST_FROM`, or use the
  tail-EMA weights as teacher. Judge on **eval bpb only** - training loss is meaningless under a mixed objective
  (the hop/EMA lesson).
- **Expected:** unknown. This is the speculative pick; it either does something or it does nothing, cheaply.

### Idea 3 - CLOSED (lost). Parallel structure: convert idle engines into step time

`engine-profile-v36.md`: every engine at 30-60%, MFU 17-19%, dependency-bound. **FLOPs are free; serial depth is
expensive.** Every architecture change we have tried adds depth (serial) rather than branches (parallel).

`ideas.md` (former arch-backlog #5) already said it: *"Parallel residuals - discarded on loss at
fixed-time at width 256 on GPU, but it halves the serial dependency chain per block, which is what a launch-bound
device pays for. **Re-test as a step-time experiment, not a loss one.**"* The verdict that killed it was measured
on a machine that was not dependency-bound - the same re-test rule that flipped depth-12 (four sign changes) and
trigram.

- **Ladder:** (a) parallel residuals (attn and MLP both read the block input) measured as **s/step first, bpb
  second**; (b) if the chain shortens materially, two towers of 6 layers summed (same params, same FLOPs, half
  the serial depth); (c) re-spend the savings on depth 14 (currently blocked by the OOB flake, lever E).
- **Expected:** -0.003 .. -0.008 if the chain-shortening is real; 0 if the compiler was already overlapping it.
  With the section-1 correction, a step-time win now converts to score instead of dead repetition.
- **Falsified if:** parallel residuals do not move s/step. Then the dependency chain is not per-block and the
  whole branch closes.

**RESULT - closed (PARRES_T).** Parallel residuals are stable at lower LR but cost **+0.005 at equal steps**,
which exceeds the ~10% step-time overlap bound the branch would have had to clear. The loss-per-step penalty the
GPU measured in the 3090 era was real and survives the regime change; the dependency-chain saving does not cover
it. The two-tower escalation (b) and the depth-14 re-spend (c) are moot and were not run.

## 5. Gate plan (PACK=4 base, ref T59 0.981197) - SUPERSEDED, see the verdicts above

1. ~~`WD` 0.1 / 0.2 / 0.3~~ **DONE, lost.** WD 0.2 +0.0045, WD 0.3 +0.0109.
2. **Instrument:** nothing to do - runs are already paired (§3). Budget n=3 repeats for any verdict that lands
   inside ~0.0008, and do not spend a gate on anything whose predicted effect is smaller than that.
3. ~~Idea 1 full (MIR)~~ **DONE, lost.** MIR 0.05 +0.0050, MIR 0.10 +0.0096.
4. ~~Idea 3 (parallel residuals)~~ **DONE, lost.** +0.005 at equal steps.
5. **Idea 2 (self-distillation) is the only survivor** and is now the whole of this file's forward value. It is
   also the largest implementation. Gate it only if the box has nothing better queued.
6. PACK=8 result decides whether the packing curve has more in it (§2) - still blocked by the sidecar crash.

**What actually paid on this base, from the same sweep:** embedding LR **x4.5** = -0.0012, confirmed by two
paired gates (0.980139 / 0.980029 vs 0.981197), curve sharp below it (x3.5 = +0.0039, x6 = current). That is
lever A, not anything in this file. Clean **v43** (sha d1b78be9a0285c8d) is built and queued as T60_gate.

## 6. Method notes carried forward

- One change per gate; 1800 s only; reference every 4-8 runs (and now: prefer a **paired** reference).
- Judge only by eval bpb. Anything with train/eval asymmetry makes training loss a liar.
- All three ideas are training-only and leave the eval forward path identical - keep it that way, the review
  rejected us twice on exactly that surface.

Sources: [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) ·
[Data-Constrained LM Pretraining: Improved Regularization and Scaling Laws](https://arxiv.org/abs/2606.06888) ·
[modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt) ·
[hunting for headroom on modded-nanoGPT (WR #82)](https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html)
