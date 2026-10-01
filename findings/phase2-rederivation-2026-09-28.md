# Phase 2, re-derived from scratch (2026-09-28)

**Research + local-GPU only. Nothing was run on the Trainium box. Nothing in `submit/` or
`submission_raw/train3_canon.py` was touched.**

---

## 0. THE HEADLINE, AND THE TWO THINGS THE BRIEF GOT WRONG

The brief asked "at 4 hours, what changes?" on the premise that Phase 2 is **8x the wall clock**, taking us
from 1.82 to ~15 epochs. Reading `README.md:132-148` and `:310-316` changes both halves of that premise:

1. **Phase 2 is `trn2.48xlarge` — 16 chips — not one chip.** 8x the time AND 16x the hardware = **128x the
   compute.** At the Phase 1 model that is **71 B tokens over a 316 M-token corpus = 225 epochs**, not 15.
2. **Phase 2 is scored 50% `val_bpb` + 50% CORE**, both min-max normalised across the ten-team pool, plus an
   `inference.py` and a 60-minute CORE timeout. **CORE appears zero times in 698 ledger rows and in all 46
   files of `findings/`.** Half the Phase 2 score is an axis this project has never measured, never optimised
   and never even defined.

Those two facts together reorder everything.

**The one-line answer.** At 128x compute the binding constraint stops being time and becomes **the 316 M-token
corpus**. Repeating it can never exceed **16.387 x 316 M = 5.18 B effective tokens** no matter how much compute
you throw at it. Run the Phase-1 model unchanged for the whole Phase 2 budget and it reaches **96% of that
ceiling in the first 22% of the clock**; the remaining 78% of the compute buys 4% more effective data. So the
compute has to go into **capacity** — the model-FLOP size grows from 54.5 M dense to **0.4-0.9 B**, an optimum
three independent parameterisations agree on, which survives an 8x range of the one imported constant (§1.5b),
and which is **the same in both shard branches** (§1.6).

**The one thing this session measured rather than derived**, on the local GPU at 1/160th the parameter scale:
at fixed tokens and fixed steps, the weight-decay optimum moves from **exactly the box's 0.1 at 1 epoch** (a
clean interior optimum, both neighbours ~0.011 worse) to **beyond 0.4 and still falling at 8 and 32 epochs**
(−0.0196 and −0.4042 for 0.4 over 0.1). The regularisation verdict flips sign between 1 and 8 epochs.
`ideas.md`'s "do not re-litigate WD or input masking below ~3 epochs" was correct, and its converse is now
measured.

**And the thing nobody has priced, which changes the recommendation inside that band.** At ~3.6 bits per
parameter, **Phase 1's 205 M-parameter model cannot memorise its 154 MB of training-set content (0.60x), and
every Phase 2 configuration in that band can, 1.9-3.2x over.** The whole 698-row ledger was measured on the safe
side of a threshold Phase 2 crosses. §6's own small-scale measurement lands in the same place from a different
direction: the published repetition law tracks our corpus to ~8 epochs and then **under-predicts the penalty by
3-6x**, with the knee at ~1 unique token per parameter — and both Phase 2 candidate shapes sit past it.

**So the recommendation is: go much bigger, but take the CONSERVATIVE end of the band, and spend the surplus
compute on FLOPs-per-token rather than parameters-per-token** — longer context, a weight-shared (looped) stack,
and auxiliary objectives, all of which add compute and *no* memorisation capacity (§1.5d). That plus the kernel
**width port** — which `handoff-2026-09-26.md` explicitly told us not to build for Phase 1 — is the Phase 2
critical path, and both are buildable today without any device time.

**Confidence summary, up front:**

| claim | confidence | why |
|---|---|---|
| 128x compute, not 8x | **certain** | README, two lines |
| CORE is 50% and unmeasured here | **certain** | README + `grep` over the whole repo |
| corpus is the binding constraint at 128x | **high** | our own measured corpus size + one published law + our own WD/MIR losses agreeing with the law's regime |
| dense optimum 0.4-0.9 B, i.e. 8-16x today | **medium-high** | three parameterisations agree, and the answer survives an 8x range of `R_D*` (§1.5b) |
| ridge tilts ~2.2 e-folds and stays flat | **medium-high** | same |
| Phase 2 crosses the memorisation threshold | **medium-high** | two independent routes (bits/parameter, and §6's measured knee) agree |
| **regularisation inverts with epoch count** | **high on the sign, medium on the size** | §6: the wd argmin is 0.1 at 1 epoch (interior) and past 0.4 at 8 and 32 epochs; wd 0.4 vs 0.1 is **+0.0122 / −0.0196 / −0.4042** |
| the published repetition law breaks past ~8-16 epochs | **medium** | measured at 1/160th the parameter scale, one seed per point |
| the specific (width, depth) point inside the band | **low** | our data cannot resolve aspect ratio; resolve with a throughput screen |
| looped depth is the right way to spend surplus compute | **low, highest value** | one ledger wash (+0.0011) plus a derivation; never tested in this regime |
| absolute Phase 2 `val_bpb` ~0.90 | **low** | 2.2 e-folds of extrapolation from a 1.8-epoch operating point |
| tau ~ 0.31 x averaging window | **medium-high** | two of our own points, with a mechanism |
| Muon WD will *not* invert (it is not a regulariser here) | **medium-high** | our own LR curve already predicted the WD rows out of sample to ±0.001 |

---

## 1. THE ARITHMETIC THAT DECIDES PHASE 2

### 1.1 The compute budget

| | Phase 1 | Phase 2 | ratio |
|---|---|---|---|
| hardware | `trn2.3xlarge`, 1 chip, 8 NC-v3 | `trn2.48xlarge`, 16 chips, 128 NC-v3 | **16x** |
| training budget | 1800 s | 14 400 s | **8x** |
| compute at our measured 20% MFU | 2.28e17 FLOPs | **2.9e19 FLOPs** | **128x** |
| tokens at the *Phase 1 model* | 575 M | **71.2 B** | 124x |
| epochs over the 316 M corpus | 1.82 | **225** | |

(2.28e17 cross-checks against `gap-analysis-2026-09-27.md` §3.4's independently derived `C ~ 6ND = 2.2e17`.)

### 1.2 The data ceiling — the single most important number in this file

Muennighoff et al., *Scaling Data-Constrained Language Models* (arXiv 2305.16264), with the fit `R_D* = 15.387`
that `data-axis-2026-09-27.md` §1 already adopted for this project:

```
D_effective = U_D · (1 + R_D* · (1 − exp(−R_D / R_D*)))       R_D = epochs − 1
```

`D_effective → U_D · (1 + R_D*)` as `R_D → ∞`. With our **measured** `U_D = 315 947 189` tokens:

> **No amount of Phase 2 compute can buy more than 5.18 B effective tokens from the five public shards.**
> Phase 1 already realises 568.6 M of them. **The entire remaining data axis, forever, is 2.21 e-folds.**

Marginal value of one more repeated token (this answers the brief's question directly):

| epochs | marginal repeated token | cumulative efficiency of the nominal tokens |
|---|---|---|
| **1.82 (Phase 1)** | **94.8%** | 98.8% |
| 4 | 82.3% | 93.1% |
| 8 | 63.4% | 82.8% |
| **15 (the brief's premise)** | **40.3%** | 67.9% |
| **22 (recommended operating point)** | **25.5%** | 56.6% |
| 30 | 15.2% | 46.8% |
| 64 | 1.7% | 25.2% |
| **225 (naive 128x scale-out)** | **0.0%** | 16.4% |

**So the brief's "what is a repeated token worth at 15 epochs" is 40%, down from 95%** — and at the actual
Phase 2 optimum it is ~25%. The naive scale-out (same model, 16x batch, 8x time) throws away **84% of the
Phase 2 compute budget.** That is the mistake to avoid, and it is the default behaviour of the provided
multi-chip baseline.

**Which bound this is.** `R_D*` was fitted on C4 at larger unique-data pools than ours, so the number is
imported. It is however the *conservative* direction for this argument: a smaller `R_D*` would make the ceiling
lower and the case for capacity *stronger*, not weaker. Two independent internal lines say we are in the
large-`R_D*` corner: WD 0.2/0.3 and MIR 0.05/0.10/0.15 were monotonically harmful at 1.82 epochs, which only
happens when repetition is not yet biting. §6 reports our own small-scale measurement of the same curve.

### 1.3 The exchange rate, re-derived as a function of epoch count

`gap-analysis-2026-09-27.md` §2.5 split our −0.0353/e-fold time slope into

```
s_T = 0.028  bpb per e-fold of TOKENS  at fixed updates
s_S = 0.007  bpb per e-fold of UPDATES at fixed tokens
```

Only `s_T` decays with repetition, and it decays by exactly `d ln D_eff / d ln D_nominal`:

| epochs | `d ln D_eff / d ln D_nom` | slope per e-fold of steps (calibrated to the measured −0.0293) |
|---|---|---|
| **1.82** | 0.959 | **−0.0293** ← the measured `v55` vs `C_2400` pair, reproduced |
| 5 | 0.853 | −0.0267 |
| 15 | 0.592 | −0.0204 |
| **22** | 0.416 | **−0.0159** |
| 30 | 0.324 | −0.0139 |
| 100 | 0.010 | −0.0063 |

The `E = 1.82` row is not fitted — it falls out of the model and lands on the measured value. That is the only
self-check available for this construction and it passes.

### 1.4 The capacity trade, re-priced

`handoff-2026-09-26.md`'s **governing fact**: dense capacity is **−0.0132 per e-fold at fixed steps** but
costs **~0.37 e-folds of step time per e-fold of parameters**, netting **−0.0024 at fixed time**. Only the
second term is budget-dependent, and it scales with the step slope from §1.3:

| epochs | net capacity per e-fold **at fixed time** | vs Phase 1 |
|---|---|---|
| **1.82** | **−0.0024** ← reproduces the governing fact | 1.0x |
| 5 | −0.0033 | 1.4x |
| 15 | −0.0056 | 2.3x |
| **22** | **−0.0073** | **3.0x** |
| 30 | −0.0081 | 3.4x |
| 100 | −0.0109 | 4.5x |

> **The ridge tilts. Extra parameters are worth 3-4.5x more at Phase 2 than at Phase 1, for the single reason
> that the step time they cost now buys tokens that are worth 25% of a fresh one instead of 95%.**

### 1.5 Solving for the optimum

Budget constraint: `D_nominal(m) = 71.2 B / m` where `m = N_dense / 54.5 M` (time ∝ params; the honest
direction at large `N`, where the model becomes genuinely FLOP-bound). Three parameterisations:

- **(a) constant slopes** — our measured −0.0132/e-fold capacity and 0.028/e-fold effective tokens, applied flat.
- **(b) decayed slopes** — the same, with each slope decayed over the path at Chinchilla exponents (α=0.34, β=0.28).
- **(c) parametric** — `L = E0 + A·N^−α + B·D_eff^−β` with `A`, `B` **calibrated to our two measured slopes at our
  operating point** rather than imported. (This fit also says our loss surface is ~20x flatter than a
  Chinchilla-form law predicts at our token count — which is the quantitative statement of "we are on a flat
  ridge", and the reason no *absolute* Phase-2 number here deserves much trust.)

| m | N_dense | epochs | D_eff | Δbpb (a) | Δbpb (b) | Δbpb (c) |
|---|---|---|---|---|---|---|
| 1 | 54 M | 225 | 5.18 B | −0.0619 | −0.0462 | −0.0461 |
| 2 | 109 M | 113 | 5.17 B | −0.0710 | −0.0526 | −0.0543 |
| 4 | 218 M | 56 | 5.04 B | −0.0794 | −0.0585 | −0.0603 |
| 6 | 327 M | 38 | 4.73 B | −0.0829 | −0.0609 | −0.0624 |
| **8** | **436 M** | **28** | 4.35 B | −0.0844 | −0.0618 | **−0.0631** |
| **10** | **545 M** | **23** | 3.98 B | **−0.0849** | **−0.0621** | **−0.0631** |
| 12 | 654 M | 19 | 3.65 B | −0.0848 | −0.0619 | −0.0627 |
| 16 | 872 M | 14 | 3.10 B | −0.0841 | −0.0612 | −0.0615 |
| 22 | 1.20 B | 10 | 2.51 B | −0.0824 | −0.0598 | −0.0593 |
| 32 | 1.74 B | 7 | 1.89 B | −0.0794 | −0.0574 | −0.0555 |

**All three agree: `m = 8-12`, `N_dense ≈ 440-650 M`, `22-28 epochs`, and the surface is flat to ±0.001 from
`m = 6` to `m = 16` (327 M to 872 M).** So Phase 1's flat ridge *survives as a flat ridge* — it just moved
**2.2 e-folds to the right.**

(The `epochs` column here scales throughput exactly as `1/m` from the Phase-1 measurement of 309 k tok/s —
which is 19.7% MFU once attention and the fused head are counted, so the column is a 20%-MFU line. §3.2 prices
the recommended shape from the FLOP budget with an explicit MFU band instead; the two agree on where the
optimum is, and disagree only on how many tokens you actually get there, which is an MFU question.)

Expected total: **Δ ≈ −0.062 bpb**, i.e. local `val_bpb` around **0.90**. Treat that as an order of magnitude,
not a number: it is a 2.2-e-fold extrapolation from a 1.8-epoch operating point, and the parametric fit's own
verdict is that our surface is much flatter than the law it borrows its exponents from.

**One systematic bias in this table, and it matters for the choice.** Every row assumes the effective-data law
holds, so loss can only *saturate* with more epochs. §6 measures that past ~8-16 epochs it **rises** instead. So
**the low-`m`, high-epoch rows are optimistic and the high-`m`, low-epoch rows are not** — which shifts the true
optimum to the right of `m = 8-12` and reinforces §1.5d's conclusion that the epoch count should be pushed down
by spending compute per *token*. Read the table as a lower bound on how far right to go.

### 1.5b Sensitivity to `R_D*` — the direction is robust, the magnitude is not

`R_D* = 15.387` is the one imported constant the optimum depends on. Re-solving parameterisation (c) across an
8x range:

| `R_D*` | D_eff ceiling | optimum `N_dense` | optimum epochs | Δ bpb | flat band |
|---|---|---|---|---|---|
| **15.387** (published) | 5.18 B | **436 M** | 28 | −0.063 | m = 6-16 |
| 8.0 | 2.84 B | **654 M** | 19 | −0.057 | m = 8-22 |
| 4.0 | 1.58 B | **1.20 B** | 10 | −0.048 | m = 12-32 |
| 2.0 | 0.95 B | **1.74 B** | 7 | −0.039 | m = 16-48 |

> **The direction is invariant and the sign of every recommendation in §3 survives: a *smaller* `R_D*` makes
> repetition saturate sooner, which makes capacity worth even more and pushes the model BIGGER and the epoch
> count LOWER.** The whole 8x range keeps the answer inside **0.4-1.7 B dense**, against today's 54.5 M. What
> `R_D*` actually sets is where in that band to sit. §6 spent local GPU time trying to pin it on our own corpus
> and **could not identify it from seven points** — but it did establish the more useful thing: that past ~8-16
> epochs the law's *functional form* stops applying at all, because loss rises rather than saturating. Read the
> table above as "the answer is 0.4-1.7 B and the epoch count should be at the low end of whatever that implies".
>
> A smaller `R_D*` would also *weaken* §5.1 and §5.2 (fewer epochs ⇒ less data pressure ⇒ auxiliary objectives
> and regularisation invert less) while *strengthening* §3.1 (shape) and §5.3-5.4. The engineering is robust;
> the objective changes are the part that is conditional.

### 1.5c The regime change nobody has priced: Phase 2 crosses the memorisation threshold

The Muennighoff law is fitted where models are much smaller than their data. Phase 2 is not, and the crossing
point is computable.

- **Corpus content.** 316 M tokens at our current 2.694 nats/token = 1.23 Gbit = **154 MB** of residual
  information, i.e. what a model would still have to store to fit the training set exactly.
- **Model capacity.** The empirical figure for transformer memorisation is **~3.6 bits per parameter**
  (Morris et al. 2025, *How much do language models memorize?*).

| | total params | capacity | / corpus content |
|---|---|---|---|
| **Phase 1 (ships)** | 205 M | 92 MB | **0.60x** — below the threshold |
| shape **C** (201 M unique dense, looped; table at 2^19) | 512 M | 230 MB | **1.50x** |
| shape **A** (403 M dense distinct; table at 2^19) | 713 M | 321 MB | **2.1x** |
| shape **B** (755 M dense distinct; table at 2^19) | **1.09 B** | **490 MB** | **3.2x** |

> **Phase 1 physically cannot memorise its training set. Every Phase 2 configuration in §1.5's flat band can,
> 1.5 to 3.2 times over.** That is a regime the entire 698-row ledger was measured *outside*, and it is the
> mechanism behind both the §5.2 regularisation inversion and the caution below. (§3.2 defines shapes A/B/C;
> only *unique* parameters count toward capacity, which is the whole point of the looped variant.)

This is the strongest argument for **not** taking the very top of the capacity band, and for treating
regularisation as a first-class part of the Phase 2 recipe rather than an afterthought. It also explains why
§6's small-scale arms blow up the way they do: at 4 M unique tokens against 8.9 M parameters they sit at 0.45
unique tokens per parameter, which is *deeper* into this regime than shape C (0.62) and close to shape A (0.44).

### 1.5d Where surplus compute should go once parameters become dangerous

§1.5c changes the shape of the problem. The naive reading of §1.5 is "spend the 128x on parameters"; §1.5c says
total parameters past ~360 M start buying *memorisation capacity we cannot feed* (that is the 154 MB / 3.6 bits
crossing point). The resolution is to distinguish two ways of spending compute:

- **Compute per *parameter*** — a bigger model. Adds memorisation capacity. Priced at −0.0132/e-fold at fixed
  steps and dangerous past §1.5c's threshold.
- **Compute per *token*** — more FLOPs per token at **fixed parameter count**. Adds no memorisation capacity at
  all, and in a data-bound regime that is exactly the currency we want.

**Four ways to buy compute-per-token, ranked, all with ledger evidence:**

1. **Longer context.** Attention FLOPs are `12·L·T·d`, linear in `T` at fixed parameters: on the recommended
   1024 x 24 shape, seq 1024 → 2048 takes attention from 11% to 20% of the FLOP budget and 4096 would take it to
   33%. Zero extra parameters, and it is the one axis that also directly serves CORE (§5.4).
2. **Weight-shared / looped depth.** Run a 12-block stack twice for an effective depth of 24 at half the
   parameters: full FLOPs, full serial depth, **half the memorisation capacity.** Our ledger has
   **`stack recurrence +0.0011`** — a *wash*, recorded as a rejection. At Phase 1 halving the dense parameters
   should have cost about +0.0017 at fixed time, so **a +0.0011 wash means weight sharing was already
   free-to-slightly-favourable, and it was dropped for being uninteresting rather than for being bad.** In a
   regime where parameters are the liability and FLOPs are the surplus, that flips from uninteresting to
   central. **This is the highest-value unmeasured item in this file.**
3. **Auxiliary objectives** (§5.1) — more loss terms per token is literally more signal per token, which is the
   §5.1 argument restated as a compute-allocation one.
4. **Output-space ensembling** (§5.6) — two forward passes per token; the cleanest form of "more compute, same
   parameters seeing the same data".

**Consequence for §3.1: take the CONSERVATIVE end of the capacity band, not the top.** ~400-500 M dense with
looped depth and seq 2048 lands at ~1.5-2.1x the memorisation threshold instead of 3.2x, keeps the full FLOP
budget in use, and keeps effective depth high for CORE. The open question — whether the looped model's lost
parameters cost more than the memorisation they avoid — is worth exactly one Phase 2 gate and it is the one I
would spend first after the objective changes.

### 1.6 The sensitivity that decides the whole plan: does Phase 2 open more shards?

`prepare.py:53-54` pins `MAX_SHARD = 6542` with `PUBLIC_TRAIN_SHARDS = list(range(5))`. At ~64.6 M tokens per
shard the *dataset* holds ~423 B tokens; we are given 0.075% of it. Phase 2 ships a new `prepare.py` and we do
not know what it will pin.

- **With 316 M unique** (branch A): optimum as above, `Δ ≈ −0.063`.
- **With unlimited unique** (branch B): Chinchilla for `C = 2.9e19` gives `N = 492 M`, `D = 9.84 B`,
  `Δ ≈ −0.075`.

> **On the `val_bpb` axis, opening the corpus is worth only ~0.012 bpb — and `N_dense ≈ 0.5 B` is the answer in
> BOTH branches.** Every piece of Phase 2 *engineering* (the kernel width port, the memory plan, the
> parallelism strategy) can therefore be committed to **now**, before the question is answered.
>
> **But that 0.012 is the naive parametric answer and it understates the difference**, because branch A's
> optimum is *constrained by overfitting* (§1.5c) while branch B's is not: with 9.8 B unique tokens the
> memorisation threshold is never approached, so the top of the capacity band becomes safe, and §5.1/§5.2's
> whole objective-and-regularisation programme becomes unnecessary.
>
> **On the CORE axis it is worth far more still**, because in-context learning and world knowledge scale with
> *unique* data, and repetition buys memorisation, which CORE does not reward. **This is a rules/logistics
> question, not a research question, and it is the single highest-value thing to ask the organisers before
> Phase 2 opens.**

---

## 2. CORE — HALF THE SCORE, AND WE HAVE NEVER LOOKED AT IT

`grep -ri core` over `findings/` and `trn2_results.tsv` returns only "NeuronCore". There is no measurement, no
baseline, and no plan. Phase 2's score is

```
0.5 · minmax(−val_bpb over the 10-team pool)  +  0.5 · minmax(CORE over the 10-team pool)
```

Four consequences that change the recipe, ordered by size:

**(a) Min-max normalisation makes relative rank on each axis worth the same.** Being best-in-pool on CORE is
worth exactly as much as being best-in-pool on `val_bpb`. Nine of the ten finalists will have spent six weeks
tuning `val_bpb` and none tuning CORE, so **the CORE axis is where the pool will spread most and where a given
amount of effort converts to the most normalised points.**

**(b) `val_bpb` and CORE want different models, and repetition is where they diverge.** `val_bpb` on a held-out
ClimbMix shard rewards fitting the ClimbMix distribution — which deep repetition does, right up to the point
where it memorises. CORE rewards in-context learning and world knowledge, which repetition does **not** buy.
**The repetition that is cheap for `val_bpb` is plausibly neutral-to-harmful for CORE**, and outright
memorisation (§1.5c) is unambiguously bad for it.

Since the `val_bpb` ridge is flat to ±0.001 across a 2.7x range of model size, the `val_bpb` axis is nearly
indifferent to where inside the band we sit — so **the shape choice should be made on CORE and on overfitting
risk, and they point at the same resolution: more effective DEPTH (which CORE rewards) at fewer PARAMETERS
(which §1.5c requires).** That is exactly what §1.5d item 2's looped stack buys, and it is why the looping
screen is the first thing on the Phase 2 list rather than a curiosity.

**(c) The 2^18 x 512 hashed bigram table contributes ~nothing to CORE.** It is a local-statistics memoriser. On
a 50/50 score it is being asked to justify 65% of the parameters against half the metric it cannot move — and
the Phase 2 tie-break is *fewer total parameters*. §4 gives the full re-pricing.

**(d) Inference cost is not a constraint, so it should not be used to argue against size.** CORE is
log-likelihood scoring over candidate completions — forward only, no generation. A ~1 B model over an
optimistic 10^8 scored tokens is ~2e17 forward FLOPs = **~2 minutes** on 16 chips against a 60-minute timeout.
The timeout binds on *compilation and KV/shape handling*, not on FLOPs. Build `inference.py` with fixed shapes
and it is a non-issue.

**What to do, in order, and none of it needs the box:**
1. **Download the pinned CORE bundle the moment it is published (2026-10-07) and score `/tmp/v76_final.pt` on
   it.** That single number — our current model's CORE — is the most valuable missing fact in this project. If
   it is ~0.00, CORE is an emergence problem and the size argument in §1 gets much stronger; if it is ~0.05,
   it is a tuning problem.
2. Build `inference.py` early and score every Phase 2 candidate on both axes. A candidate scored only on
   `val_bpb` is scored on half the metric.
3. Treat **context length** as a CORE parameter, not a `val_bpb` parameter (§5.4).

---

## 3. THE CONCRETE PHASE 2 RECIPE

Each line carries its derivation and its confidence. Where the existing code already implements the correct
scaling I say so, because those are the constants that need *no* work.

### 3.1 Shape

| knob | Phase 1 | **Phase 2 recommendation** | derivation | confidence |
|---|---|---|---|---|
| model-FLOP size | 54.5 M dense | **~800 M effective** | §1.5's band is 327-872 M and this is at its top, where the FLOP budget is fully used | medium-high |
| **unique** dense params | 54.5 M | **~200 M** (12 distinct blocks, looped) | §1.5c: unique parameters are what memorise, and 200 M keeps total capacity at 1.5x the corpus content instead of 3.2x | medium on the principle, **low on the loop working** |
| width | 512 | **1024** (8 heads x 128) | power of two, keeps `HD = 128` and the kernels' `n_head == n_embd/128` invariant, 2x tiling rather than 3x | medium |
| depth | 12 | **12 distinct blocks, stack looped 4x = effective depth 48** (fall back to 24 distinct if looping loses) | §1.5d item 2 and the shape table in §3.2; the loop buys FLOPs and serial depth at zero memorisation capacity | **low — this is the one recommendation resting on a derivation rather than a measurement** |
| MLP ratio | 6 | **6** | `F_V31_MLP6` vs `MLP8` = +0.00006, i.e. flat upward; no reason to move | medium |
| heads | 4 | **8** (HD 128) | head count is FLOP-invariant by algebra (`Σ_h T·d_h = T·d`); HD must stay 128 for the kernels | high |
| seq len | 1024 | **2048** | §5.4 (CORE) and §1.5d item 1 (compute per token at zero parameters) | medium |

`16·L·d²` at 1024 x 12 ratio 6 = **201 M unique dense parameters**; looped 4x they do the FLOP work of an
805 M-parameter, 48-layer model. Plus the four token/value tables (4 x 8192 x 1024 = 33.5 M), the head (8.4 M)
and the lookup table at 2^19 x 512 (268 M): **~512 M total = 1.50x the memorisation threshold**, against
Phase 1's 205 M (0.60x) and an unlooped 1536 x 20's 1.09 B (3.2x). §3.2 tabulates all four candidates against
both constraints.

**All of these sit inside the same flat ridge and differ by ~0.002 bpb of predicted quality, so the choice is
being made on overfitting risk, epoch count and kernel-engineering cost — not on predicted bpb, because the
ridge cannot tell them apart.** That is the honest state of the shape question.

**The aspect ratio is the one thing our data cannot resolve, and three mechanisms pull in different
directions.** (i) **The device pulls WIDE**: `engine-profile-512.md` measures DMA 72-74% active against the
tensor engine at 41-47% and a ~15% graph MFU ceiling — we are data-movement-bound, and arithmetic intensity per
weight-byte grows linearly in width. A width-2048 x depth-12 model (805 M dense) would keep the layer count,
the U-net skip indices, the value-embedding depth schedule and the whole kernel launch structure *identical to
Phase 1*, which is far the smaller engineering delta. (ii) **CORE pulls DEEP**, because in-context learning and
multi-step reasoning are the parts of the metric that respond to depth. (iii) **§1.5c pulls toward looping**,
which gives depth without parameters but multiplies the serial dependency chain on a device that is already
latency-bound.

1024 x 24 with the stack looped is the point that serves (ii) and (iii) and sacrifices (i). **Resolve it with a
day-one throughput screen over {1024 x 24 distinct, 1024 x 12 looped 2x, 2048 x 12} at fixed steps** — a
step-time question, and step-time screens are the one class `session-2026-09-27.md` §12 leaves valid. Take the
widest shape that does not lose MFU to the launch-bound path.

**Two blockers that must be fixed before any width change, both already written down and neither fixed**
(`handoff-2026-09-26.md`, last section): `head_bwd` uses `kt = D//128` in four loops where it means
`_HVC//128 = 4`; and `post_attn_fwd_h` writes `outT_out[row*n_head + i]` under an undocumented,
unasserted invariant `n_head == n_embd//128`. **Both are silently wrong at any width but 512.** Add the assert
first, fix `head_bwd` second, port third.

**A third, new blocker, from SBUF arithmetic.** `nki-kernel-playbook-2.md` §34: at width 512 a layer's weights
are QKV 1.5 MiB + MLP 4 MiB + out-proj 0.5 MiB = **6 MiB, comfortably inside the 24 MiB SBUF**. Weights scale
as `d²`, so at width 1024 that is 24 MiB — **exactly the SBUF capacity, with nothing left for activations** —
and at 1536 it is 54 MiB. **Every fused kernel must gain a weight-tiling loop.** This is not a port, it is a
redesign of the inner loop, and it is the single largest engineering item in Phase 2. Start it now.

One consolation: **a looped stack is kinder to SBUF than a deep one.** Twelve distinct blocks re-read the same
12 weight sets twice per forward instead of streaming 24, which halves the weight working set and is the same
direction the tiling loop wants.

### 3.2 Batch, steps, and parallelism

The most useful structural fact, and it is a lucky one:

> **128 NeuronCores x 1 row = 131 072 tokens at seq 1024, or 262 144 at seq 2048.** The Phase 2 hardware's
> *minimum possible global batch at `DEVICE_BATCH=1`* is **exactly our Phase 1 batch, or twice it at the longer
> context.** You cannot go below it and you never need to.

Our own measured batch curve (`R_B262`, `R_B524`, full 1800 s gates) puts the Phase 1 critical batch between
131 k and 262 k: 2x cost +0.0017, 4x cost +0.0113 with the 4x point *under-predicted*, the signature of
crossing `B_crit`. Critical batch grows as the reducible loss falls; our reducible loss roughly halves at
Phase 2, which moves `B_crit` up by ~1.9-3.6x.

| knob | Phase 1 | **Phase 2** | derivation |
|---|---|---|---|
| global batch | 131 072 | **524 288** (4x) | 4x sits at the upper edge of the grown `B_crit`; it is also the cheapest way to amortise a 16-chip collective, which is the opposite constraint pulling the same way |
| `DEVICE_BATCH` | 4 | **1** | 128 cores x 1 row; at seq 2048 that is 262 144 tokens per micro-step |
| grad accumulation | 1 | **2** | 2 x 262 144 = 524 288 |
| FLOPs per token | 0.40 G | **A 3.02 G / B 5.29 G / C 6.04 G** | `6N_eff + 12·L_eff·T·d` |
| total tokens | 575 M | **A 9.6-14.5 B / B 5.5-8.3 B / C 4.8-7.2 B** | `C / FLOPs-per-token` at 20-30% MFU |
| **epochs** | 1.82 | **A 30-46 / B 17-26 / C 15-23** | over the 316 M corpus |
| optimizer steps | 4390 | **A 18 000-28 000 / B 10 500-15 800 / C 9 200-13 800** | tokens / 524 288 |
| step time | 0.424 s | **A 0.5-0.8 s / B 0.9-1.4 s / C 1.0-1.6 s** | 14 400 s / steps |
| parallelism | ZeRO-1 over 8 cores | **ZeRO-1 over 128 cores, pure DP** | see below |

**A** = the conservative shape (1024 x 24 distinct blocks, 403 M dense); **B** = the large shape (1536 x 20,
755 M dense). Attention is 20% of A's FLOPs and 14% of B's at seq 2048, against 11% and 8% at seq 1024 — that is
the honest price of the context decision in §5.4, and it is already inside the token counts.

**Note the awkward consequence and do not hide it: neither A nor B is good on both constraints at once.**
The epoch count is an *output* of the shape choice, not an independent knob: at a fixed compute budget the only
things that lower it are more FLOPs per token or more parameters, and more parameters is the thing §1.5c says to
avoid. Laid out against the two numbers that matter:

| shape | unique dense | total params | memorisation (§1.5c) | epochs | verdict |
|---|---|---|---|---|---|
| Phase 1 | 54.5 M | 205 M | 0.60x | 1.8 | wastes 84% of the budget |
| **A** 1024 x 24 distinct | 403 M | 713 M | 2.1x | **30-46** | too many epochs (§6's collapse zone) |
| **B** 1536 x 20 distinct | 755 M | 1.09 B | **3.2x** | 17-26 | epochs fine, memorisation worst |
| **C** 1024 x 12 **looped 4x** | **201 M** | **512 M** | **1.50x** | **15-23** | **best on both** |

**Shape C is the one configuration that is simultaneously under 2x the memorisation threshold and under
25 epochs, and it is completely unmeasured.** Twelve distinct blocks at width 1024 are 201 M parameters; running
the stack four times gives the FLOP profile of an 805 M-parameter model (effective depth 48) — `6N_eff = 4.83`
GFLOP/token plus 1.21 of attention at seq 2048 = **6.04 GFLOP/token**, i.e. 4.8-7.2 B tokens, **15-23 epochs**,
9 200-13 800 steps at 1.0-1.6 s each.

**Its two real risks, both cheap to price:** (i) effective depth 48 at width 1024 is `d/L_eff = 21`, very deep,
so training stability is a genuine concern in a run with no retries; (ii) it multiplies the serial dependency
chain **4x** on a device that `engine-profile-512.md` shows is latency-bound at 30-60% engine occupancy, so the
FLOPs may simply not land. **Risk (ii) is a throughput screen and it should be the first thing measured on the
16-chip box**; risk (i) argues for looping 2x from a 24-block base as the intermediate rung.

**Which shape to build is therefore decided by one step-time screen and one CORE baseline, not by a bpb gate** —
because §1.5's ridge is flat across all four rows above and cannot tell them apart.

**Phase 2 does NOT need tensor or pipeline parallelism, and reaching for it would be a mistake.** The reason
to add model parallelism is that data parallelism forces the batch too high — and here it does not, because
the batch floor is exactly where we already are. The scaling is absorbed by the **model**, not the batch.
Memory confirms it: even at the largest candidate (~1.09 B params) bf16 replicated weights are 2.2 GB/core
against the ~10.8 GB/core measured on the Phase 1 chip, and ZeRO-1 optimizer state (fp32 master + two moments)
sharded 128 ways is 102 MB/core. Shape C is half that.

**The one thing that does force a design choice is the bigram table's share of the collective.** At Phase 1 it
is 62% of the sharded parameter set; at 2^19 rows it is 26% of shape B and **52% of shape C** (whose unique
dense body is only 201 M), and at 2^20 rows it dominates both. **So the looped shape and the bigger table pull
against each other through the collective**, and the table's rows should be gated *after* the shape is settled,
not before.

**The one genuine unknown is the 16-chip collective.** Phase 1 measures grad_sync + optimizer at 97 ms of a
424 ms step (23%), of which 58% of the optimizer phase is the parameter `all_gather`, and `F_PROBE` established
that collectives here **do not hide behind compute** — they cost the sum. Across NeuronLink, at 4.4x the
parameter bytes, this could plausibly be 200-400 ms. **Decision rule for day one of Phase 2: measure the
all-reduce of the full parameter vector across 16 chips; if it exceeds ~15% of the step, raise gradient
accumulation until it does not.** Accumulation is free in collective terms and costs only batch size, and §3.2
shows we have 2 e-folds of batch headroom before `B_crit` binds. That is the whole tuning loop.

### 3.3 Schedules

**Three of the four LR scalings are already correct in the code and need no re-derivation.** This is worth
stating plainly because it is where a re-tune would otherwise burn Phase 2 gates:

- `_BATCH_LR = (TOTAL_BATCH/32768)**0.5` — sqrt-batch scaling, automatic. Moving to 524 288 scales every LR by
  x2 relative to today with no edit.
- `dmodel_lr_scale = (n_embd/768)**-0.5` — muP-style width scaling for the AdamW groups, automatic.
- `lr_eff = lr * sqrt(max(1, out/in))` in `_muon_step` — the modded-nanogpt Muon width scaling, automatic.

| knob | Phase 1 | **Phase 2** | derivation | confidence |
|---|---|---|---|---|
| `WARMUP_STEPS` | 35 | **50-60** | the *measured* Phase-1 gradient is that shorter is better (50→100 = +0.0021; 100 on v43 = +0.0021), so do **not** lengthen it on principle. At 4x batch, 35 steps is already 4x more warmup *tokens*. The +15-25 steps are an insurance premium against a divergence in a 14 400 s run with no retries, priced at ~+0.0007 by the Phase-1 gradient scaled to ~11 500 steps — and at effective depth 48 (shape C) the premium is worth paying | medium |
| `WARMDOWN_TIME_FRAC` | 0.90 | **0.90, unchanged** | it is indexed on `elapsed_frac`, not steps — budget-invariant by construction | high |
| `FINAL_LR_FRAC` | 0.034 (B2) | **0.034, unchanged** | same; and the B2 derivation that set it is a *temperature* argument, which §5.2 shows is horizon-invariant | medium |
| momentum ramp | 0.85→0.97 over 300 steps | **0.85→0.97 over 300 steps** | 300 of ~11 500 is 2.6%; the ramp is a stability device, not a schedule | medium |
| momentum peak | 0.97 | **0.97, with 0.94 as one arm** | at 4x batch, a fixed β quadruples the effective averaging horizon `batch/(1−β)`. Holding that horizon fixed would need β=0.88 — but `G_MOMON` at β=0.93 **lost** at Phase 1, so the measured direction is up, not down. Genuinely two-sided; worth exactly one arm | low |
| `EMA_START` | **0.60** (the shipped B2 value; 0.75 on the pre-B2 canon line) | **0.60, unchanged** | it is a *fraction*, and the 0.85→0.75→0.65 sweep on the pre-B2 line showed a plateau (−0.00044 then +0.00002) rather than a sharp optimum. Unlike the decay and the blend, the window is **not** offline-sweepable — it decides what gets accumulated — so leave it where it was measured and spend the run elsewhere | medium |
| **EMA `tau`** | 556 steps (decay 0.9982) | **~1050 steps (decay 0.99905)** at shape C's ~11 500 steps | see below | medium-high |
| `EMA_BLEND` / per-group blends | 0.65 / 0.65 | **sweep offline, do not guess** | see below | **high** |
| `WEIGHT_DECAY` (Muon) | 0.10 | **0.10 first; sweep it second if the embedding sweep is flat** | §5.2 — its Phase-1 outcome is predicted by the LR curve, so it reads as a temperature knob; but §6 shows regularisation of *some* kind inverts hard, and if the group that should carry it does not, this is the next place to look | medium |
| AdamW `weight_decay` on the **embedding group** (incl. the bigram table) | 0.001 | **sweep 0.001 / 0.01 / 0.05** | §5.2 — never swept in 698 rows, and it is the only decay in the model applied to parameters that can actually overfit | med-high that it matters, low on the value |
| `EMBED_LR_SCALE` | 4.5 | **4.5, with 3.0 as one arm** | embedding rows are visited ~20x more often at 22 epochs, which argues the per-visit LR should fall; but the Phase-1 curve is flat above the optimum (6 = +0.0011, 8 = +0.0008) and sharp below (3.5 = +0.0039) | low |

**The EMA horizon, derived rather than guessed.** Our two measured (window, optimal-tau) pairs:

| averaging window | window length at 4390 steps | measured optimal tau | tau / window |
|---|---|---|---|
| 0.75 (pre-B2 canon line) | 1098 steps | 333 | **30.3%** |
| **0.60 (the shipped B2 line)** | 1756 steps | 556 | **31.7%** |

**tau ≈ 0.31 x the averaging window**, with a mechanism: the stated reason short horizons win is that the
terminal rampdown makes late weights improve fast, so a fixed *fraction* of the window is the horizon-invariant
quantity. At a window of 0.70 over shape C's ~11 500 steps (3450 steps of averaging), **tau ≈ 1050, decay ≈
0.99905**; scale it with whatever step count the final shape produces.

**But do not ship that number — ship the instrument.** The tail-EMA blend is a pure post-training linear
combination, so `SAVE_EMA_PARTS=1` plus `EMA_DECAY_EXTRA="0.99810,0.99905,0.99952"` (tau ~525 / 1050 / 2100)
carries three parallel accumulators at a measured cost inside run-to-run scatter, and
`experimental/blend_sweep.py` / `blend_groups.py` then sweep decay x blend x group **exactly, offline, on the
local GPU, with zero gate noise.** This is the highest-confidence recommendation in the whole file: it converts
the single most horizon-sensitive part of the recipe from a guess into an exact offline computation, and the
project already owns the tooling. **Weight-averaging gains also generally grow with run length**, so this
subsystem is likely to be worth *more* than the −0.0009 it is worth at Phase 1.

### 3.4 The capacity split

| component | Phase 1 | **Phase 2 recommendation** | why |
|---|---|---|---|
| dense transformer | 54.5 M (27%) | **600-800 M (~75%)** | §1.4-1.5 |
| hashed bigram table | 134.2 M (65%) | **2^19 x 512 = 268 M via a 512→d projection**, with 2^20 (537 M) as the first capacity gate | §4 |
| token/value tables | 4 x 4.19 M | 4 x 8192 x d | scales with width automatically |
| head | 4.19 M | 8192 x d | ditto |

**Do not scale the bigram table's rank with the model width.** At width 1536 a naively-scaled `2^18 x 1536`
table is 402 M parameters of pure lookup that CORE cannot use and the collective must move every step. Hold
the row width at 512 and add a `512 → d` projection (0.8 M parameters); spend the capacity on **rows** instead
(§4).

### 3.5 Two practical constraints that will bite before any of the science does

1. **The 262 144-byte portal source limit.** `submit/train.py` is **232 204 bytes = 88.6% of the cap**, and
   v78 was already rejected once for exceeding it. The width port adds a weight-tiling loop to every fused
   kernel, and Phase 2 adds `inference.py`. **There is ~30 KB of headroom and the port will not fit in it.**
   Budget a deliberate size pass (the file carries very large explanatory comment blocks) *before* the port,
   not after, or the work will be unshippable the way v78's was.
2. **The reviewer's surface.** `review-rejection-2026-09-28.md`: v52 (accepted) read the environment 5 times,
   v78 (rejected) read it 50, and the rejection reason was that *the architecture of the scored model could be
   changed without changing the file*. Phase 2 multiplies the number of things one would like to make a knob
   (parallelism degree, accumulation, shape, seq len). **Every one of them must be hard-coded in the shipped
   file**, with the knobs living only in the dev vehicle. This is the third time the project has drifted the
   same way; plan for it rather than rediscovering it in November.

### 3.6 What explicitly does NOT change, and why that matters

The brief warns against the error of thinking a +4% step-time mechanism gets cheaper at 8x the budget. It does
not. The correct discriminator is **whether the cost is a percentage or a fixed number of seconds**:

- **Percentage costs are invariant.** The Canon conv's +4.7% in-kernel, the ~95 ms-per-eager-torch-op between
  fused layer ops, fp8, the 38%-of-vector-instructions family — all unchanged. Do not reopen any of them on
  budget grounds.
- **Fixed-second costs get 8x cheaper as a fraction.** This is where the real inversions live, and §5.3 lists
  them.
- **The one subtlety**: a cost quoted *per call* rather than per byte gets relatively cheaper when the step
  gets longer. MoE's `all_to_all` floor is ~3 ms per call independent of size; 48 calls is 34% of a 424 ms step
  but 15% of a 940 ms one. Still far too much. See §5.5.

**Also unchanged: the optimizer family verdict.** `ideas.md` rules out SOAP/Kron because they "only overtake Muon
at ≥8x Chinchilla". Phase 2 at the recommended point is **8 B tokens / 755 M params = 10.6 tokens/param = 0.53x
Chinchilla** — *further* from the crossover, not closer. **Muon stays.** (This flips only in branch B with a
much smaller model, which §1.5 says is the wrong choice anyway.)

---

## 4. THE BIGRAM TABLE IN THE REPEAT REGIME

The brief asks which way 15 epochs pushes a memorisation device that is 65% of the parameters. The answer is
in two parts and they point in opposite directions, which is why it needs measuring rather than reasoning.

### 4.1 A new measured fact: the table is 37:1 oversubscribed by type, not saturated

`handoff-2026-09-26.md` closes the rows axis with "the measured intra-row collision rate is already 0.29%, so
addressing is saturated, not scarce." Measured this session on the real corpus (5 prefix points from 2 M to
66 M tokens, Heaps fit `b = 0.5420`, extrapolated to the full 316 M):

| | value |
|---|---|
| distinct bigram **types** in 2 M / 8 M / 16 M / 32 M / 66 M tokens | 511 596 / 1 234 449 / 1 875 628 / 2 790 191 / 4 130 717 |
| **extrapolated distinct types at 316 M tokens** | **8.4 - 9.7 M** |
| types sharing each row at 2^18 rows (ships) | **32 - 37** |
| at 2^19 | 16 - 18 |
| at 2^20 | 8 - 9 |
| at 2^21 | 4 - 5 |

(The Heaps exponent is itself decaying across the five points — 0.635, 0.603, 0.573, 0.542 on successive
doublings — so the 9.7 M figure from the last pair is an **upper** bound and 8.4 M assumes the decay continues.
Either way the conclusion is the same to within 15%.)

**Both statements are true and they measure different things.** By *token mass* the table is saturated — a
row's traffic is dominated by its one frequent bigram, giving the 0.29% figure. By *type count* it is
**37:1 oversubscribed**: the tail types are all colliding, all the time. At Phase 1 that did not matter,
because a bigram occurring twice in the corpus is seen only **3.6 times** in a whole training run and cannot be
learned at all. **At 22 epochs it is seen 44 times.** The tail becomes learnable, and learnable tail types are
what more rows buy.

### 4.2 The horizon-dependent sign flip is already in the ledger, twice

The rows knob has **already reversed sign once with horizon**, and the mechanism was written down at the time:

| measurement | horizon | 2^17 → 2^18 |
|---|---|---|
| `E13` (train2 ladder) | 420 steps, 27.5 M tokens | **+0.0028 — recorded as "entries undertrained"** |
| `F_V23_BG18` / `T36_gate` | full 1800 s, 575 M tokens | **−0.00277** |

And `2^18 → 2^19` at full 1800 s is **+0.0009** — the same "entries undertrained" signature, one doubling
later, at 20x fewer visits per row than Phase 2 will have. Visits per row: Phase 1 is 575 M / 262 144 = 2194;
Phase 2 at 8 B tokens and 2^20 rows is **7600**, i.e. **3.5x more visits per row than Phase 1 gets today at a
quarter of the rows.**

> **PREDICTION, and the first Phase 2 capacity gate to run: `2^19` and `2^20` rows invert from loss to win.**
> Falsifier: if `2^19` is still ≥0 at Phase 2 scale, the rows axis is closed for good and the table should be
> frozen at 2^18 and de-prioritised.

### 4.3 What pushes the other way

1. **CORE.** The table is worth 0 on half the score, and the Phase 2 tie-break is fewer total parameters.
2. **Communication.** The table is **62% of the sharded parameter set**; at 16 chips its reduce-scatter and
   all-gather cross NeuronLink every step. A 2^20-row table is 537 M parameters and would dominate the
   collective budget. This is a real constraint on growing it, independent of quality — and it is exactly what
   §3.2's accumulation rule trades against.
3. **The dense body now covers the n-gram job itself.** `session-2026-09-27.md`'s stated mechanism for the MTP
   failure is "this model already has a dedicated 2^18 x 512 hashed bigram table doing that job". That argument
   runs both ways: a 7-14x larger, far better trained transformer also does that job.
4. **Averaging.** `handoff-2026-09-26.md`'s rule — *average densely-updated parameters, never sparse lookup
   tables* — was derived from the table touching ~6% of rows per micro-step. At 4x the batch it touches ~4x as
   many rows per step, which moves it toward the "dense" side and makes the +0.00028 EMA loss worth re-testing.
5. **Memorisation.** §5.2 identifies the table as the single most overfittable object in the model: ~9 M types
   into 262 144 rows at `weight_decay = 0.001`, and in Phase 2 each row's types recur tens of times. **Growing
   the rows and raising that decay are the same experiment and should be run as a 2-D pair, not separately** —
   more rows without more decay is exactly the configuration §1.5c warns about.

**Net recommendation:** keep the table, hold its **rank** at 512 behind a projection, gate its **rows** at 2^19
and 2^20 as the first capacity experiment, and be ready to cut it entirely if the day-one CORE baseline shows
CORE is where the score is.

---

## 5. WHAT INVERTS IN THE REPEAT REGIME — RANKED

This is the list to re-test first when Phase 2 opens. The **#n** in each heading is the rank by expected size x
confidence; the section order is narrative.

### 5.1 #2 — auxiliary objectives that extract more supervision per token. **INVERTS.**

MTP lost **+0.0051**; prefix-aux lost **+0.00349**. The recorded explanation was "the bigram table already
supplies that local structure." That explanation is incomplete, and the complete one is a clean inversion
argument:

> **At 1.82 epochs a marginal token is worth 94.8% of a fresh one, so data is effectively free and there is no
> reason to extract more bits from each token. At 22 epochs it is worth 25.5%, so extracting more supervision
> per token is worth roughly what four extra tokens are worth.** When compute binds you buy more tokens; when
> data binds you buy more signal per token. Phase 1 was the first regime and Phase 2 is the second.

Also: the MTP that lost was the **stochastic** form with 43% of targets replaced, which adds gradient variance
the exact weighted-CE sum does not — so the mechanism was never cleanly tested even at Phase 1. And MTP's
published value is disproportionately on *downstream and in-context* tasks rather than on next-token
perplexity, which is precisely the CORE half of the Phase 2 metric.

**Expected: −0.002 … −0.008 on bpb, plausibly more on CORE.** Falsifier: exact (non-stochastic) 2-token MTP at
Phase 2 scale; if it is still ≥0 the whole auxiliary-objective family closes for good. Ranked below §5.2 only
because §5.2 now has a direct measurement behind it and this does not.

### 5.2 #1 — regularisation. **INVERTS — now MEASURED — and the Phase 1 verdict tested the wrong parameters.**

WD 0.2/0.3 lost +0.0045/+0.0109 and MIR lost +0.0050/+0.0096/+0.0139, with the explicit note *"do not
re-litigate WD or input masking below ~3 epochs."* Phase 2 is well past that, so the note's own condition is met
— and §6 now measures the sign flip directly on our own corpus: at a fixed total token budget,
**the weight-decay argmin is 0.1 at 1 epoch (interior, both neighbours ~0.011 worse) and beyond 0.4 at both 8
and 32 epochs, where the curve is monotone decreasing across the whole grid; 0.4 vs 0.1 is +0.0122 / −0.0196 /
−0.4042.**
Same code, same data, same 3906 steps, same 64 M tokens; only the epoch count differs. That is the clearest
single piece of evidence in this file, and it is ours rather than imported — which is why this is ranked #1
despite MTP having the larger headline number at Phase 1.

But reading the code changes *which* decay should be re-litigated:

**`WEIGHT_DECAY = 0.10` applies only to the Muon matrix stacks** (`train3_canon.py:3355`). The B2 analysis at
`:340-372` argues those stacks are read through RMSNorm everywhere, so what the loss feels is not the weight
norm but the temperature `T = lr/||W||`, with `||W||_eq ~ sqrt(lr/wd)` reached in 82-201 steps. **The decisive
evidence is not the algebra, it is an out-of-sample prediction that already succeeded:** the up/down slopes of
the v29 `MATRIX_LR` curve, fitted against `ln T` using LR rows *only*, predict the WD rows — wd 0.20 predicted
+0.0042 against measured +0.0032 / +0.0045, wd 0.05 predicted +0.0016 against measured +0.0022 / +0.0003.

> **A knob whose outcome is predicted to ±0.001 by the learning-rate curve is not delivering an independent
> regularisation effect. The Muon WD gates measured a temperature, not a regulariser, and they will not
> invert.** Leave `WEIGHT_DECAY` at 0.10.

The parameters that *can* actually overfit are the ones whose scale the forward can feel, and they sit in the
AdamW groups at decay values **nobody has ever swept**: `embedding_params` (including the **134 M-element
bigram table**) at **wd = 0.001**, and `lm_head` at **wd = 0.01**. A lookup table with ~9 M bigram types
crammed into 262 144 rows, each row absorbing ~35 types' gradients tens of times over, at wd = 0.001, is the
single most overfittable object in the model.

> **Re-test as: AdamW `weight_decay` on the embedding group (0.001 → 0.01 / 0.05), NOT Muon `WEIGHT_DECAY`.**
> Expected −0.001 … −0.005. This is the sharpest correction in this file: the axis was closed by a measurement
> that could not have detected the effect.

MIR / input masking / dropout are genuine regularisers and their verdicts *are* horizon-dependent — §6 is a
direct demonstration that *something* in this family inverts by a large margin. **Re-test dropout too: it has
never been tested here at all, in 698 rows, rejected purely by inheritance from a single-epoch lineage.**
And note the ordering that §6 implies: at 32 epochs one knob at one setting recovered 38% of a 1.088 collapse
and the optimum was beyond the grid, so **expect the Phase 2 regularisation optimum to be several times the
Phase 1 value, and sweep coarsely (x3, x10) rather than in x1.5 steps.**

### 5.3 #3 — everything rejected on the 900 s compile cap. **INVERTS, by pure arithmetic.**

`quality-levers-2026-09-26.md` §6: *"Sequence-length and batch-size ramps are dead on the compile cap, not on
the idea. Each distinct micro-step shape is a new NEFF; cold compile is 503.6 s against a hard 900 s startup
cap, and over-cap compile is back-dated against the 1800 s budget (400 s charged = −22% steps = +0.0067)."*

At Phase 2 the *same* 400 s of over-cap compile is back-dated against **14 400 s = −2.8% of steps = +0.0004**,
a **16x** reduction in the price of a shape. And the 900 s allowance itself goes from 50% of the budget to
6.25%.

**Reopened by this, and all of them are things we want anyway:**
- **Sequence-length ramp** (512 → 1024 → 2048). `data-axis-2026-09-27.md` §4 calls it "the one curriculum
  family with real published wins" and kills it *solely* on the compile cap.
- **Batch-size ramp** — start at 131 k and ramp to 524 k+. This is standard practice precisely because it keeps
  you under `B_crit` early and amortises collectives late, which is exactly the Phase 2 tension in §3.2.
- **`torch.compile` and multi-shape specialised kernels** generally.

### 5.4 #4 — context length 2048. **INVERTS, mostly for CORE.**

`S_V24_T2048` lost **+0.0039** at Phase 1. Three things change: (a) CORE's reading-comprehension and
long-passage tasks need it; (b) 11.35% of documents exceed 1025 tokens and hold **38.5% of all tokens**, so at
2048 far more documents are seen whole; (c) `Q_SWA512` already measured that shrinking the window buys ~nothing
here, and the vendor attention backward is only ~3.5 ms exclusive, so the attention-cost objection is weak.
**Pair it with the previously-shelved free win**: `window-256-on-odd-layers` measured **−0.0017 and free** and
was "not adopted" — it is worth much more at 2048 than at 1024.

### 5.5 #5 — MoE. **THE QUALITY CASE INVERTS; THE COLLECTIVE COST DOES NOT.**

Both Phase 1 kills need re-reading:
- *Kill 1 (stack-specific):* `all_to_all_single` has a **~3 ms floor independent of size**; 48 calls/step = 26-37%
  of a 424 ms step. At a 940 ms step that is **15%** — better, still disqualifying.
- *Kill 2 (general):* `F_V31_MLP6` 0.996552 vs `MLP8` 0.996495 — *"+33% MLP params bought 0.00006, this model
  is not MLP-capacity-limited."* **That measurement was taken on a 54.5 M-dense model at 1.82 epochs, where
  capacity was worth −0.0024/e-fold at fixed time. At 22 epochs capacity is worth −0.0073/e-fold and CORE
  rewards raw parameter count directly. Kill 2 no longer binds.**

**Verdict: do not start MoE on spec.** It needs a collective design that avoids the per-call floor (e.g. one
fused all-gather of the routed hidden states rather than 48 `all_to_all`s), and that is a research project.
But the "this model has nothing to buy" argument is retired, and if the day-one CORE baseline is near zero,
MoE is the highest-ceiling item on the board.

### 5.6 #6 — output-space ensembling. **PLAUSIBLY INVERTS, and it is cheap to settle.**

`quality-levers-2026-09-26.md` §6c measured **−0.020 for output-space ensembling of two independently-trained
same-config checkpoints**, and rejected it because splitting the budget in half cost **+0.025**. The halving
cost scales with the step slope from §1.3, which falls from −0.0293 to −0.0159, a factor 0.543. Scaling the
**recorded** +0.025 (not my model's 0.020) by that factor gives **+0.0136 against a −0.020 gain: net −0.006,
a sign flip.** Two caveats that could eat all of it: ensemble gains shrink as the constituent models get better
and larger, and the −0.020 was measured between two *Phase-1-scale* checkpoints. The Phase 2 form is natural:
split the 16 chips into two groups of 8, train two ~400 M models
on different data orders for the full 4 hours, and ship a log-prob mixture as one `nn.Module`. That satisfies
the eval contract literally (pure, stateless, causal, consolidated, single-device) though it costs on the
"fewer total parameters" tie-break, and CORE's response to ensembling versus raw size is genuinely unknown.
**Cheap to settle exactly and offline from two Phase 2 checkpoints. Do that before building anything.**

### 5.7 #7 — weight-shared / looped depth. **INVERTS from "uninteresting wash" to central.**

Fully derived in §1.5d item 2 and repeated here because it belongs on this list: the ledger's
**`stack recurrence +0.0011`** was a wash, and at Phase 1 halving the dense parameters should have cost ~+0.0017
at fixed time, so **the wash means weight sharing was already free-to-favourable and it was dropped for being
boring.** In Phase 2 it is the only tool that buys FLOPs and serial depth while *reducing* memorisation
capacity, which §1.5c says is the binding risk. **Risk to price first: a looped stack multiplies the serial
dependency chain, and this device is latency-bound (all engines 30-60% busy), so the FLOPs may not actually
land.** That is a throughput screen, which is valid, and it should be run before the quality gate.

### 5.8 #8 — AdEMAMix. **INVERTS on the stated objection.**

`ideas.md` rules out "AdEMAMix/Schedule-Free/AdaMuon/Dion (need >1500 steps to warm up)". AdEMAMix's whole
mechanism is a second, very slow gradient EMA at `β₃ ≈ 0.9999` — a ~10 000-step horizon that **cannot engage at
all in a 4390-step run** and engages comfortably in 10 000-14 000. It is the one optimizer item whose rejection reason
is literally "not enough steps". Low expected size (−0.001?) but it is now a legitimate arm.

### 5.9 Things that will be wrongly reopened — do not

- **Weight decay on the Muon group** (§5.2) — its outcome is already predicted by the LR curve out of sample,
  so it is a temperature knob. **Caveat on my own evidence:** §6's toy has no Muon group (it is AdamW
  throughout, with the head and embeddings in the same group), so the toy shows *that* regularisation inverts,
  not *which* group's decay does the work. Sweep the embedding group first because it is the one with no prior
  measurement at all; if that is flat, only then revisit the Muon group.
- **The 900 s "giant first five steps" trick.** Worth −0.0058 at Phase 1 by `gap-analysis`'s own pricing. At
  Phase 2 five giant steps add ~100 M tokens against 8 B = **~−0.0003**. It prices itself out, which
  conveniently retires the ethical question too.
- **Label smoothing** — raises CE, and CE is the metric. Invariant.
- **Coverage / packing / dedup / ordering.** `data-axis-2026-09-27.md` caps the whole family at −0.0007 at
  Phase 1. At Phase 2 the cap *rises* (100% coverage is worth `ln(316/275) = 0.139` e-folds of unique data, and
  unique data is worth more when repetition is deep) but only to roughly **−0.002**, and the eight gates that
  already lost on this axis all lost for reasons that do not change. Still not first.
- **The optimizer family** (§3.6) — we move *away* from the SOAP/Kron crossover, not toward it.

---

## 6. OUR OWN MEASUREMENT OF THE REPETITION CURVE (local GPU)

Everything above leans on `R_D* = 15.387`, a constant fitted on C4 by someone else. This session measured the
same curve on **our** corpus and **our** tokenizer, at small scale, on the local RTX 4080, at no device cost.

**What it set out to do and what it actually did.** It did *not* pin down `R_D*` — the fit is not identified from
seven points (RESULT 1 reading 1), and §1.5b already shows the recommendation survives the whole plausible range
anyway. What it did do is more useful: it **bounded where the published law stops applying** (somewhere between 8
and 16 epochs at this data-to-parameter ratio, and past that the behaviour is overfitting rather than diminishing
returns — an object the law cannot represent), and it **measured the regularisation inversion directly**, which
is the single most actionable result in this file.

**Design.** A 4.7 M-dense / 8.9 M-total GPT (width 256, depth 6, 4 heads, ReLU² MLP ratio 4, RMSNorm, RoPE,
QK-norm, logit softcap, seq 512), AdamW, **64 M training tokens and 3906 optimizer steps in every arm**, with
only the size of the unique pool changing: 64 M (1 epoch), 32 M (2), 16 M (4), 8 M (8), 4 M (16), 2 M (32),
1 M (64). Scored with the exact `evaluate_bpb` formula (`nats / (ln2 · bytes)` using `prepare.get_token_bytes`)
on **1.6 M tokens of held-out train-shard documents** — the val shard was never opened. Tokens/param = 13.6,
comparable to our 10.6 at Phase 1.

**Known confounds, stated.** (a) Each arm's pool is a *prefix* of the same stream, so arms see different
documents; `data-axis-2026-09-27.md` §3.5 measured the corpus to be an i.i.d. shuffle with zero length
autocorrelation and no content block structure at any lag, so this is noise, not bias. (b) The model is ~160x
smaller than the Phase 2 target and uses AdamW on **every** parameter rather than our Muon/AdamW split, so the
weight-decay arms show *that* regularisation inverts, not *which* parameter group's decay does the work.
(c) `R_D*` is fitted as scale-independent in the source, but that is exactly the assumption under test here.
(d) Single seed per point except where noted.

**RESULT 1 — the published law agrees within ~2x up to 8 epochs and under-predicts by 3-6x beyond 16.**

| epochs | unique pool | unique / total params | val_bpb | Δ vs 1 epoch | law's prediction* | ratio |
|---|---|---|---|---|---|---|
| 1 | 64.0 M | 7.18 | 1.363721 | — | — | — |
| 2 | 32.0 M | 3.59 | 1.367888 | **+0.0042** | (calibration point) | — |
| 4 | 16.0 M | 1.80 | 1.372578 | **+0.0089** | +0.0184 | law **over** by 2.1x |
| 8 | 8.0 M | 0.90 | 1.421128 | **+0.0574** | +0.0504 | law right to 1.1x |
| 16 | 4.0 M | 0.45 | 1.649634 | **+0.286** | +0.111 | law **under** by 2.6x |
| 32 | 2.0 M | 0.22 | 2.451879 | **+1.088** | +0.215 | law **under** by 5.1x |
| 64 | 1.0 M | 0.11 | 3.429744 | **+2.066** | +0.369 | law **under** by 5.6x |

\* `R_D* = 15.387` with the bpb-per-e-fold slope calibrated on the 2-epoch point (0.268 bpb per e-fold of
effective tokens, which is ~10x our Phase-1 slope because a 4.7 M-parameter model at 64 M tokens is far from
converged).

**Two readings, and the second is the important one.**

1. **A naive re-fit prefers a smaller `R_D*`, but it is not identified.** Fitting
   `bpb = c + B·D_eff^(−β)` with free `(c, B, β, R_D*)` over all seven points gives `R_D* = 4.85, β = 0.89`
   at rms 0.070 — but the rms profile over `R_D*` is nearly flat (0.070 at 4.85, 0.072 at 8, 0.077 at 15.387),
   and restricting to `E ≤ 8` drives `β` to the grid edge. **Do not quote a re-fitted `R_D*` from this
   experiment.** §1.5b already shows the recommendation survives the whole 2-15 range anyway.
2. **The law's functional form cannot describe what happens past the knee, because loss *increases* and a
   saturating-effective-data law can only flatten.** `D_effective` is monotone in epochs; our bpb is not. That
   is not a bad fit, it is the wrong object. **The regime the law describes ends somewhere between 8 and 16
   epochs at this data-to-parameter ratio**, and beyond it the behaviour is overfitting, not diminishing
   returns.

**The knee sits at ~1 unique token per parameter — and every Phase 2 candidate shape is past it.**

| | unique tokens / total params | nearest arm |
|---|---|---|
| **Phase 1 (ships)** | **1.54** | between the 4- and 8-epoch arms, i.e. in the benign region — which is exactly what the ledger measures |
| shape A (403 M dense, 713 M total) | **0.44** | the **16**-epoch arm (+0.286) |
| shape B (755 M dense, 1.09 B total) | **0.29** | between the 16- and 32-epoch arms |
| **shape C (201 M unique, 512 M total, looped 4x)** | **0.62** | between the 8- and 16-epoch arms — the least bad, which is why §3.2 recommends it |

**How much of this transfers, stated honestly.** Not literally: 316 M tokens of diverse web text contain far
more generalisable structure per parameter than 4 M do, and the alignment of the ratios is partly coincidence.
Two things make it worth acting on anyway: (a) it is the *same* diagnostic §1.5c reaches independently from
bits-per-parameter, and two independent routes to "Phase 2 crosses a threshold Phase 1 was safely below" is
more than a coincidence; (b) **every arm ran one fixed recipe**, tuned for the 1-epoch point — no extra weight
decay, no dropout, no input masking. The collapse is therefore an upper bound on the penalty for an
*unadapted* recipe, and the whole question is how much of it regularisation recovers. That is what the
weight-decay arms below measure, and it is the direct small-scale test of §5.2.

**RESULT 2 — the regularisation inversion, measured, with a sign flip.** Same code, same data, same 3906 steps,
same 64 M tokens; only the epoch count and the AdamW `weight_decay` differ.

| epochs | wd = 0.0 | wd = 0.1 | wd = 0.4 | argmin | wd 0.4 − wd 0.1 |
|---|---|---|---|---|---|
| **1** | 1.374465 | **1.363721** | 1.375889 | **0.1** — a clean interior optimum | +0.0122 (hurts) |
| **8** | 1.448341 | 1.421128 | **1.401558** | **≥0.4** — monotone decreasing in wd | −0.0196 (helps) |
| **32** | 2.658279 | 2.451879 | **2.047710** | **≥0.4** — monotone, steeply | −0.4042 (helps a lot) |

> **This is the clearest result in the file and it is ours, not imported.** At 1 epoch the decay curve has an
> interior optimum at exactly the value the box ships (0.1) and both neighbours are worse by ~0.011-0.012. At 8
> and 32 epochs the curve is **monotone decreasing over the whole grid** — the optimum has left the grid
> upward. `ideas.md`'s "do not re-litigate WD or input masking below ~3 epochs" was right, and **its converse is
> now measured rather than argued: above ~4 epochs the regularisation optimum is several times the Phase 1
> value and rising.**

Two further readings:

- **Regularisation recovers a large fraction of the collapse but not all of it.** At 32 epochs the unadapted
  recipe costs +1.088; with wd 0.4 the penalty against *its own* 1-epoch arm is +0.672, so raising one knob by
  4x recovered 38% of it — and the optimum is off the top of the grid, so more is available. **Deep repetition
  is expensive even with an adapted recipe.** That is a second, independent argument for not choosing a shape
  that forces 30+ epochs (§1.5d, §3.2).
- **Sweep coarsely.** The optimum moves from 0.1 to beyond 0.4 over one and a half orders of magnitude in epoch
  count, so the Phase 2 search should be x3/x10 steps from the Phase 1 value, not x1.5 steps. A x1.5 sweep would
  read as "flat and slightly worse" and close the axis incorrectly.

**Noise floor, measured.** Seed-2 replicates of the wd 0.1 arms:

| epochs | seed 1 | seed 2 | spread |
|---|---|---|---|
| 1 | 1.363721 | 1.366163 | **0.0024** |
| 8 | 1.421128 | 1.419976 | **0.0012** |

Every delta quoted above clears that floor: the 1-epoch weight-decay deltas by ~5x, the 8-epoch one by 16x, the
32-epoch one by >100x. **The one number I would not lean on hard is the 1-epoch +0.0122** — real but only ~5σ at
n=1 per arm; happily it is also the one the box has independently confirmed (+0.0045 for wd 0.2 at full length,
same sign).
- **Conversely, the Phase-1 verdicts are vindicated, not overturned.** At 1 epoch extra decay costs +0.0122 here
  and +0.0045 on the box — same sign, same order. The ledger was not wrong; it was measured in the regime where
  it was right.



---

## 6b. THE CONTINGENCY: what changes if Phase 2 opens the corpus

Branch B (more than 5 shards) moves surprisingly little on `val_bpb` and a great deal on CORE. Explicitly:

| | branch A (316 M unique) | branch B (unlimited) |
|---|---|---|
| `N_dense` | 400-650 M | **492 M** (Chinchilla for `C = 2.9e19`) — *the same answer* |
| nominal tokens | 4.8-8.3 B (shape-dependent) | 9.8 B |
| epochs | 15-26 | ~1 |
| Δ bpb (parametric) | −0.063 | −0.075 |
| **§5.1 MTP / auxiliary objectives** | **invert (data binds)** | **stay dead** (data is free again) |
| **§5.2 embedding-group weight decay** | **invert** | stay dead |
| **§4 bigram rows 2^19 / 2^20** | **invert** | unchanged, entries still undertrained |
| §5.3 compile-cap items | invert either way | invert either way |
| §5.4 seq 2048 | invert (CORE) | invert (CORE) |
| **§1.5c memorisation threshold** | **crossed 1.5-3.2x — the binding risk** | **gone entirely** (9.8 B unique tokens is ~4.8 GB of content against ~490 MB of capacity) |
| §1.5d looped depth / compute-per-token | central | unnecessary; spend on parameters |
| CORE | severely capped by 1.27 GB of unique text | the real opportunity |

**Read this table as the reason to ask the question early.** Three of the seven inversions in §5 are
*conditional on the corpus staying closed*, and one of them (exact MTP) costs a day of kernel work to build.
**The shape and the kernel port are unconditional; the objective changes are not.**

---

## 7. THE ORDERED PLAN

**Precondition.** All of this assumes we advance. Phase 1 closes 2026-09-30 with ten entries and ten places,
and `submit/train.py` (v79) is still not on the portal — see the current handoff. Nothing in this file matters
if that upload does not happen.

**Before 2026-10-07 (no device needed):**
1. **Ask the organisers whether Phase 2 opens more than 5 train shards.** It is worth ~0.012 bpb and
   considerably more CORE, and it is a one-email question. §1.6.
2. **Start the kernel width port.** Add the `n_head == n_embd//128` assert, fix `head_bwd`'s `kt`, then design
   the weight-tiling loop the >512 widths force (§3.1). This is the Phase 2 critical path and it needs no box.
3. **Write `inference.py`** against the published CORE harness shape.

**Day one of Phase 2:**
4. **Score `/tmp/v76_final.pt` on the pinned CORE bundle.** Most valuable missing number in the project.
5. **Measure the 16-chip parameter all-reduce** and set gradient accumulation from it (§3.2).
6. **One throughput screen over the four candidate shapes** — A (1024 x 24 distinct), B (1536 x 20 distinct),
   C (1024 x 12 **looped 4x**) and a wide control (2048 x 12) — at fixed steps (§3.1-§3.2, §5.7). This decides
   the shape, it is a step-time question, and step-time screens are the one class `session-2026-09-27.md` §12
   leaves valid. **C is the arm that could surprise**: it multiplies the serial dependency chain 4x on a
   latency-bound device, and if its FLOPs do not land, its whole case collapses.

**First Phase 2 quality gates, in order:**
1. **AdamW embedding-group weight decay** 0.001 → 0.01 → 0.05 (§5.2). The cheapest of the inversions, the one
   with a measurement *gap* behind it rather than a judgement call, and the one §6 measured a sign flip for.
   **Sweep coarsely (x10, x50), not in x1.5 steps** — §6's argmin moved by more than 4x over the epoch range.
2. **Exact (non-stochastic) MTP** (§5.1). Largest expected size; needs head-kernel work, so start it early.
3. **Bigram rows 2^19 / 2^20 crossed with the embedding decay** (§4.2, §4.3 item 5) — one 2-D pair, not two
   1-D sweeps, because more rows without more decay is the configuration §1.5c warns about. **Run it after the
   shape is settled**: at 2^19 rows the table is 26% of shape B's sharded parameters but 52% of shape C's, so
   its collective cost depends on which shape won.
4. **Seq 2048 + windowed odd layers** (§5.4).
5. **The offline EMA decay x blend x group sweep** (§3.3) — costs no gate at all; run it on the first
   checkpoint that exists.

---

## 8. METHOD NOTES

- **Nothing was run on the Trainium box.** Numbers are: existing full 1800 s gates cited by their ledger ids,
  exact local measurements on the cached train shards, one small-scale GPU experiment (§6), and arithmetic.
- **Every extrapolation is labelled.** The §1.5 optimum extrapolates our measured slopes ~2.2 e-folds; all
  three parameterisations bracket it and I report the spread rather than one number. The absolute Phase 2
  `val_bpb` is the *least* trustworthy number here and is given only as an order of magnitude.
- **§6 is a 1/160th-scale experiment and its absolute numbers do not transfer.** Its slope is ~10x ours because
  a 4.7 M-parameter model at 64 M tokens is far from converged. What transfers is a **sign** (the weight-decay
  argmin moves with epoch count) and a **shape** (the law's validity ends between 8 and 16 epochs at ~1 unique
  token per parameter). Neither claim needs the magnitudes to carry.
- **What I did NOT establish, stated so nobody quotes it as settled:** (i) `R_D*` for our corpus — the fit is not
  identified; (ii) which parameter group's decay carries the regularisation (§6's toy has one group); (iii)
  whether the looped stack's FLOPs actually land on this device; (iv) any CORE number at all, for any of our
  models; (v) the 16-chip collective cost. Items (iii)-(v) are the first three measurements of Phase 2.
- **Direction of every imported conversion.** `R_D* = 15.387` is imported from a C4 fit; a *smaller* `R_D*`
  would strengthen the capacity argument, so importing it is the conservative direction for §1. The
  Chinchilla exponents in §1.5(b)/(c) are used only to *decay* our own measured slopes, never to set them.
- **No short quality screen is proposed as a decision procedure anywhere.** The screens recommended (§7 item 6,
  and the throughput half of §3.1) measure step time, which `session-2026-09-27.md` §12 explicitly leaves valid.
  §6 is not a screen for a Phase 2 decision — it is a *regime* measurement whose arms differ by 0.02-1.1 bpb,
  three orders of magnitude above any noise floor, and it is used only for the sign of an effect.
- **Prior-test-of-the-wrong-version checks**, as the brief requires: (a) the bigram **rows** verdict rests on a
  full-length gate at 2194 visits/row, and Phase 2 has 3.5x that — the knob has already flipped sign once with
  horizon (§4.2); (b) the **WD** verdict swept the group whose outcome the LR curve already predicts out of
  sample, and never touched the group that can overfit (§5.2); (c) **MTP** was tested only in its stochastic form (§5.1); (d) the **batch**
  axis has full-length gates only from 131 072 upward, which is the direction Phase 2 moves, so those gates are
  the relevant ones for once.

**Sources:** `README.md` (Phase 2 hardware and scoring), `prepare.py` (corpus pinning, budget constants),
`trn2_results.tsv` (698 rows), `findings/handoff-2026-09-26.md`, `gap-analysis-2026-09-27.md`,
`data-axis-2026-09-27.md`, `quality-levers-2026-09-26.md`, `session-2026-09-27.md`, `repeat-regime.md`,
`ideas.md`, `step-budget-2026-09-22b.md`, `engine-profile-512.md`, `nki-kernel-playbook-2.md`,
`submission_raw/train3_canon.py`;
[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264);
Morris et al., *How much do language models memorize?* (the ~3.6 bits/parameter figure used in §1.5c).

**Reproducing or extending §6:** the experiment is `experimental/repeat_law.py` (a dev tool, never part of a
submission), run from the repo root as

```bash
.venv/bin/python experimental/repeat_law.py --total 64000000 --epochs 1,2,4,8,16,32,64 --wd 0.1 --seed 1
```

~3.2 min per arm on the local 4080; it tokenises its own pools into `~/.cache/repeat_law` on first use and never
opens the val shard. **The two extensions worth running next, both free:** (a) a second model size (`--d 384
--L 8`) to test whether the knee really tracks unique-tokens-per-parameter rather than epoch count — that is the
assumption the whole transfer argument rests on; (b) `--wd 1.0` and `--wd 2.0` at 8 and 32 epochs, to find the
high-epoch argmin this grid did not contain.
