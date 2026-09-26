# The quality axis reopened, and the leaderboard (2026-09-26)

Two conclusions this project had treated as settled are both false. This file records what replaced them.

## 1. We ARE on the portal, at 10th of 10, and the offset is +0.0069

Verified directly against the public leaderboard JSON (no sign-in needed):
`https://d3n62br6wsx257.cloudfront.net/api/leaderboard`, `lastUpdated 2026-09-25T22:51:38 UTC`.

| # | team | portal val_bpb | local-equiv (−0.0069) |
|---|---|---|---|
| 1 | Them Bones | 0.94635 | 0.9394 |
| 2 | Team Awesome | 0.95746 | 0.9506 |
| 3 | Trainium of Thought | 0.96086 | 0.9540 |
| 4 | msamuel | 0.96428 | 0.9574 |
| 5 | Team Quanta | 0.96959 | 0.9627 |
| 6 | Next Token | 0.96994 | 0.9631 |
| 7 | Team DLYog | 0.97485 | 0.9680 |
| 8 | Eldrun | 0.97627 | 0.9694 |
| 9 | Yuntian Deng | 0.97733 | 0.9704 |
| 10 | **competitive larping (us)** | **0.97854** | 0.9717 |

- **The `States.Timeout` saga is over.** Submission `sub-907467dd` (2026-09-25) scored 0.97854, which is v52's
  local mean 0.971674 + 0.006866. A resubmission simply worked, confirming the handoff's "resubmit unchanged
  first" call and that v52 was never at fault.
- **The portal offset is +0.0069, not +0.0060 or +0.0065.** Use 0.0069 for all leaderboard arithmetic.
- **Top 10 advance and we are 10th**, 0.0012 behind 9th, with Phase 1 closing **2026-09-30**.
- **v55 does not secure the place**: 0.970871 + 0.0069 = **0.97774** portal, still 10th, ~0.0004 behind 9th.
  Upload it anyway (it is strictly better and free), but it is not the answer.
- **0.9650 local = 0.9719 portal = 7th on today's board.** The user's target is, to within rounding, the price
  of a safe qualifying position rather than a stretch goal.
- The leader is **0.0314 local** ahead of us, not the 0.015 that `leaders-gap-2026-09-22.md` implies (that file
  compares against a stale board). They have dropped ~0.002/day for eleven days.
- The Devpost project gallery is still **unpublished** (449 registrants), so there is still no writeup from any
  competitor. Only the numbers are public.

## 2. `leaders-gap-2026-09-22.md`'s "we already run the frontier stack" is five records out of date

That recon predates modded-nanogpt records 86-91 and **PR #360 (2026-08-30, "ANVIL2")**, which is the current
record at 39.914 s against record #89's 73.889 s on the same machine — **1.85x faster than the record our stack
was copied from** — with n=18 certification, independent reproduction, published per-mechanism `valΔ` ablations
and published coefficients. `findings/` had none of it.

**Re-run this recon monthly.** Twice now, "no public disclosure exists / we are at the ceiling" has been
recorded as a conclusion when it was really a statement about when someone last looked.

## 3. The largest single lever is an oversight in our own code

**Our tail EMA excludes every `nn.Embedding` table.** `EMA_SKIP_EMBED=1` (which the submission wires on) builds
`skip = {id(m.weight) for n, m in transformer.items() if isinstance(m, nn.Embedding)}` and drops all five. That
is ~69% of parameters and, critically, the four token/value tables (`wte`, `vte`, `vte_mid`, `vte2`, 8192x512 =
4.19M each) which train at **`EMBED_LR_SCALE = 4.5`**. The parameters with the largest per-step updates — the
ones a terminal average helps most — are the only ones never averaged.

**There is no ledger row testing the alternative.** The axis was closed by an implementation detail, not a
measurement: `EMA_SKIP_EMBED=0` would also sweep in the 2^18 x 512 = 134.2M bigram table (a 537 MB fp32 clone
plus a 134M-element lerp every step), which is presumably why nobody flipped it.

The record treats terminal averaging as a **per-group** schedule with the embedding/head group at a *stronger*
blend than the matrix banks, and rates the family its single largest quality item (20.2 millinats for +0.7%
wall). Conversion here: 1 millinat ≈ 0.0004 bpb (4.0196 bytes/token, measured on shard 0 — see below).

`submission_raw/make_s21.py` → `train3_s21.py` implements it as a second EMA group over embedding tables under
a numel ceiling (`EMA_TABLE_MAX = 3.4e7` admits the four 4.19M tables, excludes the 134M bigram) with its own
`EMA_TABLE_BLEND` (0.65). +67 MB of state, four extra lerps/step, throughput-neutral. Gated as a matched
same-file same-session pair `G_S20REF` / `G_S20TAB`.

## 4. Corpus and metric constants, measured rather than assumed

- **4.0196 bytes/token** (4000 documents of train shard 0: 12,067,797 bytes / 3,002,213 tokens).
- Shard ≈ 64.6M tokens; `PUBLIC_TRAIN_SHARDS = list(range(5))` is pinned in `prepare.py:54` ⇒ **323M tokens
  total**, so our 556M tokens is **1.72 epochs**. More data is not obtainable.
- `evaluate_bpb` is `nats / (ln2 · bytes)`, i.e. exactly uniform per-token cross-entropy. **There is no
  byte-reweighting lever.**
- v55 = 0.970871 bpb = 2.7051 nats/token = ppl 14.96. Target 0.9650 = ppl 14.72. Leader ≈ ppl 13.71.
  The leader is 3.2% better in bpb but **8.4% better in perplexity** — an ordinary modelling gap, not an
  asymptote effect.
- Good models on general web text reach ~0.6-0.75 bits/byte, so **the corpus is nowhere near a floor**. What
  we are fighting is our own recipe's asymptote.

## 5. The exchange rate is flattening, which reprices everything

v46→v55 is a **pure-throughput series** — every one of those changes is exact/bit-identical, model math
unchanged — so it is the cleanest steps-vs-bpb data available:

| version | steps | val_bpb | | local slope (bpb/e-fold) |
|---|---|---|---|---|
| v46 | 3652 | 0.976163 | v46→v47 | −0.0380 |
| v47 | 3942 | 0.973256 | v47→v48 | −0.0293 |
| v48 | 4008 | 0.972769 | v48→v49 | −0.0428 |
| v49 | 4085 (n=4) | 0.971955 | v49→v52 | −0.0575 |
| v52 | 4105 (n=4) | 0.971674 | v52→v55 | −0.0243 |
| v55 | 4243 | 0.970871 | **end-to-end** | **−0.0353** |

The per-pair slopes are noise-dominated (each Δ is 0.0003-0.003 against ±0.00025 gates), so the end-to-end
−0.0353 is the only trustworthy figure, and a curvature fit is only bounded to an asymptote in
**[0.943, 0.967]**. Consequences:

- **+22% steps buys 0.9650 only if the curve is log-linear.** Under the best fit it needs +57%; at the
  pessimistic edge no amount of throughput gets there.
- **As the rate flattens, capacity gets relatively cheaper.** Re-price any capacity lever that was rejected
  when the rate was −0.0404 (e.g. depth 13/14 computes to a wash now rather than a loss).
- A run of v55 unchanged at a **2400 s** budget (queued as `C_2400`) discriminates directly: log-linear
  predicts 0.9607-0.9623, a strong bend predicts ~0.9654.

## 6. Ranked quality plan (expected bpb at FIXED TIME)

| # | lever | expected | cost | notes |
|---|---|---|---|---|
| 1 | **Per-group terminal averaging** — embedding tables into their own EMA group at blend 0.65; then a short tail-average replacing the matrix banks, + decontraction (renorm each shipped tensor to its pre-ship Frobenius norm) + a Richardson lag correction | **−0.002 … −0.004** | 4-8 h, no kernel | §3. The whole plan hinges on this gate |
| 2 | 6-map Polar Express cascade (published ANVIL coefficients, envelope [0.9951, 1.00041] vs our ~[0.86, 1.14]) | −0.0005 … −0.0015 | 1 h | literal constant swap at `submit/train.py:124-131` |
| 3 | Prefix-token-prediction aux loss (2nd CE arm on the same logits, target = longest in-vocab proper byte-prefix of the true target; weight 0.25 annealed to 0) | −0.0010 … −0.0015 | 1 day | **96.9% of our vocab (7932/8188) has an in-vocab byte-prefix**; only the 256 single-byte tokens do not. ~0 step cost: one extra gather of an already-softcapped logit |
| 4 | Twin-rail Muon velocity (fast β=0.85 + slow β=0.98 rails, engaging at ~43% of the run) | −0.0010 … −0.0020 | 1 day | **refutes** `ideas.md`'s "AdEMAMix/Schedule-Free/AdaMuon need >1500 steps to warm up": ANVIL's slow rail engages at step 514 of 1194 |
| 5 | LR/schedule re-tune at 4243 steps | ≤0.0005 … −0.002 | 4 gates | the two consultations disagree; every swept scalar sits on a plateau, but the constants were tuned at ~3300 steps |
| 6 | Canon-style short causal depthwise conv (width 3-4) on Q/K/V | −0.002 … −0.005 fixed-step | 2-3 days | the one untested architecture item with a same-family win already in our data: `KEY_OFFSET` (a 1-position shift on a quarter of the key dims) paid |
| 7 | Canonical / infeasible-continuation logit mask, applied in train AND eval | −0.0007 | 1 day | **13,522 single-merge-infeasible (prev,next) pairs = 0.020% of the 8192² grid**, max 1.65 per token ⇒ an (8192,16) int16 table. Changes the scored forward, so review risk |

### Closed cleanly (saves days)

- **Sampled-softmax loss — the biggest item in the frontier stack (11% of their wall) — is dead here on vocab
  arithmetic.** V=8192 with 16,384 targets per micro-step means the batch's own targets already cover
  8192·(1−e⁻²) ≈ **87% of the vocabulary**, so "all targets + negatives" *is* the full softmax.
- **Sequence-length and batch-size ramps are dead on the compile cap, not on the idea.** Each distinct
  micro-step shape is a new NEFF; cold compile is 503.6 s against a hard 900 s startup cap, and over-cap
  compile is back-dated against the 1800 s budget (400 s charged = −22% steps = +0.0067).
- **Trigram / more lookup capacity: dead on saturation.** Vocab 8192 ⇒ only 67.1M possible bigrams, so 262,144
  rows is already 256:1 — we sit on the *saturated* end of the record's own dose curve, which is why their
  2.65-5.3 millinats/doubling does not transfer (ours measures ~1.5). Trigram needs 8192³ = 5.5e11.
- **z-loss:** logits are already bounded by a learned asymmetric tanh softcap and `head_fwd` relies on that
  bound to delete the online max, so there is no logsumexp drift to penalise.
- **A giant first-5-steps batch to use the free startup allowance:** the allowance caps *optimizer updates* at
  5, and 5 updates are 5 updates (~20M extra tokens against 556M).

## 6b. A SCORED RUN HAS BEEN OVER THE CHARGED BUDGET (fixed in v56)

`prepare.record_step_boundary`'s docstring is the contract and it is explicit: the excluded allowance is launch
through the **5th** boundary, and **"everything after it is charged against the training budget."** Our stop rule
reserves `CHECKPOINT_RESERVE_SECONDS = 5.0` for the checkpoint save and **nothing** for what follows: an in-run
`causality_check` plus a full `evaluate_bpb`, both run in-process after the last step.

From the v55 gate (`runs/T73_gate_v55.log`, a **warm** box):

```
training_seconds 1794.7 | charged_seconds_est 1795.3 | over_budget False | eval_seconds 5.4
```

1795.3 + 5.4 = **1800.7 s against an 1800 s budget.** Our `over_budget` flag reads False only because it is
computed at `submit/train.py:3776`, *before* the self-checks at `:3780-3786`, so it never counts them.
`eval_seconds` ranges **6.3-147.3 s** across 433 box logs and 22-41 s is typical **cold**, because the eval-shape
graph (`targets=None`, `emb=None`, 4 rows, eager) is compiled for the first time there and nothing traced it
earlier — `check_fused_loss` covers only the 16-row with-targets path. So on the scoring host we are plausibly
**23-142 s over the charged budget**.

**The in-run copy earns nothing, and we can now prove the scorer does not read it.** The portal **re-trains on its
own hardware**: the local→portal offset is not constant (v43 0.979941 → 0.9824 = +0.0025; v52 0.971674 → 0.97854
= +0.0069). A rescoring of our checkpoint cannot vary like that; a fresh run with its own step count can. **This
retires the open question written into `make_v54.py`**, which deliberately kept the in-run eval because "we do NOT
know whether the portal reads val_bpb from our stdout". It does not.

It is also an **unguarded crash surface**: `evaluate_bpb(split="public_val")` → `list_shard_paths` has no
`try/except` around it, so a scoring host missing the public-val shard kills a run that already trained fine.

**v56** (`submission_raw/make_v56.py`) defaults `--eval-public` off and drops `WALL_RESERVE_SECONDS` 300 → 90.
Dev gates already pass `NEURON_COMPETITION_R1_EVAL_PUBLIC=1` explicitly, so gating is unaffected. `prepare.py
check` passes. No math changes, so the local score should match v55 within noise; the gain is that a scored run
stops being over budget, and the smaller reserve lets the total-wall guard fire later on a slow-compiling host.

## 6c. Eval-side items measured by the red-team pass (small but free)

- **Never-scored tokens still consume softmax mass.** `prepare.py:266` sets `token_bytes = 0` for the four
  special tokens and `evaluate_bpb` masks with `valid & (nbytes > 0)`, so **BOS-as-target is never scored** — yet
  training scores it with full weight (the train loader emits no `-1`) and the head softmaxes over all 8192
  classes. Measured **−0.000381** from a finite −30 offset on those four logits at eval. **Trap: `-inf` is fatal**
  — `evaluate_bpb` computes `loss * mask` and `inf * 0.0 = NaN` ⇒ `val_bpb = NaN`. Use a finite offset.
  The unambiguous half is the training side: `ignore_index` on BOS targets.
- **Dead, measured, do not revisit:** logit temperature (optimum exactly 1.00 on val *and* held-out train),
  per-class bias recalibration (+0.0046 transfer penalty), **fp32 at eval (exactly 0.000000, twice** — the
  weights the scorer sees are bf16-precision regardless, since `gather_params` writes bf16-rounded values into
  the fp32 replica), consolidating fp32 optimizer masters, best-checkpoint-instead-of-last, huge first-5-step
  batches, splitting the budget into two ensembled halves (+0.025 by the length ladder).
- **Output-space ensembling is worth −0.020** between two independently-trained same-config checkpoints, while
  **weight-space averaging of the same pair is +0.035**. That asymmetry retro-explains this project's EMA
  anomaly (`S_V24_EMA995` +0.0145, `EMA998` +0.0388, and why `EMA_BLEND` had to be held at 0.35): weight
  interpolation needs one basin, log-prob mixing does not. **But the free version is dead** — the tail-EMA
  snapshot and the final weights are 7e-5 apart in relative L2, i.e. the same point, which also explains why
  `EMA_BLEND` 0.35 vs 0.5 was a wash. The one untested variant is an **asymmetric** mixture of the final weights
  with a **mid-run** snapshot (~55% of budget, before the warmdown collapses the LR), tilted heavily toward the
  final weights. Costs ~1-2 s of training and one extra 336 MB of checkpoint; the sweep is offline and exact.
- **DISCARDED as against the rules, and recorded so it is not rediscovered as a clever trick:** in
  `make_eval_dataloader` (`prepare.py:552-556`) eval rows are consecutive slices of one continuous stream, so
  row *i*+1 is the literal continuation of row *i* **inside the same scored batch**. Using that left context
  would be batch-dependence, which the eval contract forbids in as many words, and `causality_check` would not
  catch it. **Do not do this.** The legitimate version is `A_STREAM`, already tested and lost (+0.0017).

## 7. Flagged, NOT built: test-time adaptation in `load_for_eval`

The highest single number found (−0.002 … −0.007) is dynamic evaluation: split each eval row at 512, take one
Adam step on the model's own loss over tokens 0-511, score 512-1023 with the nudged copy and 0-511 with the
base weights, discard the copy. It costs **zero training seconds**, and `evaluate_bpb`'s timeout is 1200 s
against our 28-156 s of actual eval — a ~40x unused compute budget. Done per row it is batch-independent and
passes `causality_check` (which flips tokens ≥512 and compares logits <512) exactly.

**Not built, deliberately.** It updates weights using validation input, which is a genuine gray area against
"off-limits: reading validation data" and plausibly reads as circumventing the 30-minute training budget. This
project already has two review rejections on eval-path surfaces. **Ask the organizers before building it.**
