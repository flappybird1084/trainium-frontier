# The gap to the leader, decomposed (2026-09-27, research only — nothing run on the box)

**HEADLINE: the gap is 0.033, not 0.027 — the leader moved −0.0068 in two days — and no single structural
difference accounts for it. Our own 2400 s run proves it cannot be throughput, and 92 millinats is ~6x a whole
generation of published quality work, so it cannot be one technique either. The largest single *identifiable*
block is not modelling at all: it is budget-edge usage we declined and best-of-N submission selection, worth
0.008–0.017 together.**

This file answers four questions: decompose the gap; attack the bigram-table assumption; check the vocab/bpb
interaction; sanity-check the gap itself. It closes with one bet that costs one screen.

---

## 0. The live board (fetched 2026-09-27, `lastUpdated 2026-09-27T20:55:56Z`)

`https://d3n62br6wsx257.cloudfront.net/api/leaderboard` — public, no sign-in. **Check it every session; it moved
a lot in two days.**

| # | team | portal val_bpb | Δ vs 2026-09-25 |
|---|---|---|---|
| 1 | Them Bones | **0.9396011** | **−0.00675** |
| 2 | Team Awesome | 0.9526343 | −0.00483 |
| 3 | Trainium of Thought | 0.9575115 | −0.00335 |
| 4 | msamuel | 0.9642813 | 0.00000 |
| 5 | Team DLYog | 0.9674456 | −0.00535 |
| 6 | Team Quanta | 0.9678769 | −0.00171 |
| 7 | Next Token | 0.9699352 | −0.00001 |
| 8 | shiv | 0.9699651 | new entry |
| 9 | FrontierForge | 0.9705457 | new entry |
| 10 | **competitive larping (us)** | **0.9759244** (v58) | 0.00000 |

- **Ten entries, top ten advance, 449 registrants, Phase 1 closes 2026-09-30.** We are last of ten. Two new
  names appeared in two days (`shiv`, `FrontierForge`) and `Eldrun`/`Yuntian Deng` are gone from the top ten.
  The only way we fail to advance is an eleventh entry scoring below 0.9759244 before the close.
- Our best local is **0.9671** (v72 0.968134 − B2 0.00064 − eps 0.000065 − temp 0.000331) ⇒ **~0.9726 portal**.
  That is still 10th (FrontierForge 0.97055), but it is **0.0033 more cushion**, and `submit/train.py` is already
  byte-identical to `submission_v76_clean_evaltemp.py`. **Uploading is a one-click action that has now been the
  top recommendation of three consecutive handoffs.**

**The gap: 0.9726 − 0.9396 = 0.0330 portal.** In local-equivalent units (offset +0.0055) the leader is ~0.9341
and the gap is the same 0.0330, because **a gap is offset-invariant** — both numbers travel through the same
portal.

Unit conversion used throughout: 4.0196 bytes/token ⇒ `bpb = nats_per_token / 2.7860`, so **1 millinat/token =
0.000359 bpb**. Our 0.9671 = 2.6939 nats = ppl 14.79. The leader's 0.9341 = 2.6018 nats = ppl 13.48, i.e.
**8.85% better in perplexity, 92 millinats of cross-entropy.**

---

## 1. THE DECISIVE ARGUMENT: it cannot be throughput, and our own data proves it

Same file, same box, same session, two budgets: **v55 at 1800 s = 0.970871 @4243** and **`C_2400` at 2400 s =
0.962251 @5694**. That is −0.0299 per e-fold of wall clock, and the curvature fit over the pure-throughput
v46→v55 series bounds this recipe's asymptote **below 0.9623**.

Take the most optimistic possible reading of our own data — log-linear, no bend at all — and reaching 0.9341
from 0.9671 needs

```
exp(0.0330 / 0.0299) = 3.0x the wall clock  =  5400 s of an 1800 s budget
```

Take the fitted bend and **it is unreachable at any budget on this recipe.**

For scale, our step is 424 ms for 131,072 tokens = 309 k tok/s at ~20–25% MFU (measured: forward graph 25.0%,
backward 24.8%). A 3.0x throughput gain means ~64% MFU on a 512-wide model. The counter-evidence is stronger
still: `CLEAN_LT4` measured the **compiler-fused layer path at 5.2x SLOWER** than our hand kernels, so a team
that did *not* write NKI is behind us on throughput, not ahead.

**Bound on the throughput axis, stated as a bound:** the busiest engine over the step is Vector at 67.9% union-busy
occupancy, so a *perfectly scheduled* step is at most 1.47x faster = 0.385 e-folds = **−0.0135**. That is the
ceiling with unlimited engineering time. A realistic two-day yield is ≤1.05x = **−0.0017**.

**Conclusion: at most 40% of the gap is throughput, and most of that is out of reach. The majority is
quality-per-token.**

---

## 2. Ranked decomposition of 0.0330

| # | candidate | Δ bpb | which bound | confidence | evidence |
|---|---|---|---|---|---|
| A | **Budget-edge usage we declined** | **0.006 … 0.013** | our own arithmetic | medium | §2.1 |
| B | **Throughput engineering** | 0.008 … 0.014 | hard upper bound | high on the bound | §1 |
| C | **Best-of-N portal selection** | 0.002 … 0.004 | order-statistic estimate | medium | §2.2 |
| D | **Recipe quality, many small items** | 0.005 … 0.015 | literature-calibrated | medium | §2.3 |
| E | **Something we do not know** | 0.000 … 0.015 | residual | — | §2.4 |
| | **total** | **0.021 … 0.061** | | | brackets 0.033 |

No row is larger than 0.013. **That is the answer to the question as asked: no single structural difference
plausibly accounts for 0.033, and the decomposition that does is four or five blocks of comparable size.**

### 2.1 A — budget-edge usage, 0.006 … 0.013 (the largest identifiable block, and we chose not to take it)

Two items, both already priced in `findings/`, both deliberately declined:

1. **The 900 s excluded startup allowance** (`prepare.TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS = 900`; the excluded
   window is launch → 5th `record_step_boundary`). Our cold compile uses ~500 s and we leave **~360–400 s
   unused**. Five gradient-accumulated giant steps inside that window add ~+130 M tokens = **+23% tokens** with
   only 5 extra optimizer updates. Priced at the token-only slope derived in §2.5 (`s_T = 0.028`/e-fold):
   **−0.0058**. `handoff-2026-09-26.md` declined this on stated organizer intent ("warm up to compile, then the
   clock starts"), and its stated price of "5 updates are 5 updates" understated it — the constraint caps
   *updates*, not *tokens*, and tokens are what this model is starved of.
2. **Per-row test-time adaptation in `load_for_eval`** — our own estimate **−0.002 … −0.007**, costing zero
   training seconds against a 1200 s eval allowance we use 6–29 s of. Flagged NOT BUILT pending an organizer
   ruling.

**This is the single most decision-relevant row in the table**, because it is worth more than every remaining
code lever combined and it is a *rules question*, not a research question. It costs an email, not a run.
I am not recommending either action unilaterally — item 1 in particular reads to me as against the stated
intent — but the honest position is that 0.006–0.013 of the gap may simply be teams taking clock we left on the
table, and that the question deserves to be asked out loud before the close.

### 2.2 C — best-of-N portal selection, 0.002 … 0.004 (free, legitimate, and stated in the rules)

README: **"5 submissions per team per week in Phase 1 … your best score counts."** Our three uploads give
portal offsets **+0.00246 (v43), +0.00687 (v52), +0.00554 (v58)**, sd ≈ 0.0022. The private-shard bias is fixed
across uploads, so that spread is almost entirely portal-side step-count variance (0.0044 of bpb = ~12% of
steps — cold vs warm compile on their host).

Order statistics at sd = 0.0022: best-of-3 sits 0.85σ = 0.0019 below the mean; best-of-30 (5/week × ~6 weeks of
Phase 1) sits 2.04σ = **0.0045** below. A team that used its quota therefore holds **~0.0026 of pure selection**
over us. We have used **one** of this week's five.

### 2.3 D — recipe quality, 0.005 … 0.015

Three independent calibrations of how big a *modelling* gap can be at this scale:

- **`R_ANVIL2`:** the whole modded-nanogpt generation after the one our stack was copied from decomposes as 72%
  wall clock + 28% quality, and its portable quality content at our scale is **~15 millinats = 0.005 bpb**.
  One full generation of their record is 58 millinats *on their setup*.
- **Our own 680 rows:** nothing since v40 has moved a single change by more than −0.0009. The largest quality
  items ever measured here are the bigram table (~0.017 at 500 local steps), document masking (−0.009), MLP
  ratio 6 (−0.0056), depth 10→12 (−0.0032).
- 92 millinats is therefore **~6x a full generation of published quality work**, or ~20 of our best remaining
  items stacked perfectly.

**Direction of the conversion, stated because it is biased:** converting their millinats to our bpb is the
*optimistic* bound (they train at ~1.2 tokens/param and 2.65 steps/millinat; we at ~7.8 and ~43), so the true
transferable content is smaller than 0.005.

### 2.4 E — the residual

What would qualify structurally, and why each is unlikely here:

- **MoE.** Dead twice, and the second kill is the general one: `F_V31_MLP6` 0.996552 vs `MLP8` 0.996495 says
  **+33% MLP params bought 0.00006**. This model is not MLP-capacity-limited, so capacity routing has nothing
  to buy. (The first kill — a 2–3 ms `all_to_all` floor independent of size — is stack-specific and could be
  wrong for a different implementation. The second is not.)
- **A much better attention kernel** enabling longer context or more heads. Head count is dead by algebra
  (Σ_h T·d_h = T·d). Context 2048 lost (+0.0039) and the vendor attention backward is only ~3.5 ms exclusive,
  so there is no attention time to recover.
- **A strong practitioner who tuned thirty things.** This is my modal explanation for rows 5–9 of the board,
  who sit within 0.004 of each other and 0.005 of us.

### 2.5 The slope decomposition this analysis needed, derived from two full-length gates

Our −0.035/e-fold is a *time* slope: it moves tokens and updates together. The batch gates separate them.
`R_B262` (2x batch: tokens ×1.093, steps ×0.547) = **+0.0017**; `R_B524` (4x batch: ×1.15, ×0.288) = **+0.0113**.
Solving the two-term model near the operating point against `s_T + s_S = 0.0353`:

```
s_T (bpb per e-fold of TOKENS at fixed updates)   = 0.028
s_S (bpb per e-fold of UPDATES at fixed tokens)   = 0.007
```

`R_B524` is then under-predicted (+0.0046 vs +0.0113), which is the expected signature of the update penalty
growing past the critical batch size — so 0.007 is a *lower* bound on `s_S` at larger batch, and the fit is
local.

**Two consequences worth keeping.** (1) **This model is token-starved, not update-starved** — extra tokens are
worth 4x extra optimizer steps, which is exactly why the giant-first-steps trick prices at −0.0058 and why pure
throughput remains the highest-value axis in principle. (2) **The batch axis is closed downward too, by
arithmetic.** Halving to 65,536 (DEVICE_BATCH=8) cuts only `fwd_bwd`, not the ~0.10 s of batch-independent
comm + optimizer + gather, so the step goes 0.41 → ~0.255 s: steps ×1.61, tokens ×0.804. That is
+0.218 e-folds of token loss (+0.0076 at `s_T`) against −0.476 e-folds of update gain (−0.0033 at `s_S`) =
**+0.004, a loss**. The only full-length batch gates on the modern line are *upward*; this closes the downward
leg without spending one.

---

## 3. Q2 — is the 2^18 x 512 hashed bigram table a good use of 65% of the parameters?

**Yes, and the trade you are worried about is priced at +0.0020 from two full-length measurements. But the
table's BASE value has never been measured at full length, and you should know that.**

### 3.1 What the ledger actually says it was worth

| measurement | horizon | value of the table |
|---|---|---|
| `E12` (train2, the 3090 ladder: 10L x 768, 2^17) | **420 steps, 27.5 M tokens** | −0.0084 |
| `E13` 2^18 at that scale | 420 steps | **+0.0028 — "entries undertrained"** |
| local GPU, depth 12, no-bigram floor 1.206291 vs table 1.188775 | **500 fixed steps** | −0.0175 |
| `S_V21_BG0` vs `S_V21_REF6` on the box | 600 s | **~0 net** (1.109948 @807 vs 1.118470 @648: the 0.184 s/step the table costs eats its whole gain) |
| `S_V21_BG14` (2^14 rows) | 600 s | a dead wash against BG0 (1.109832 vs 1.109948) |
| **full 1800 s, `BIGRAM=0`** | — | **NO SUCH ROW IN 680** |

So the table's presence rests on a −0.0084 at 420 steps on a different model at 4.8% of our token budget, a
−0.0175 at 500 local fixed steps, and an argument ("its value shows only at full length"). By the
`session-2026-09-27.md` §12 rule those are exactly the instruments that mis-ranked four mechanisms. **The base
value is not measured. The derivative is, and the derivative is what the decision needs.**

### 3.2 What IS measured at full length: the table is at an interior optimum in both dimensions

| full 1800 s gate | change | Δ bpb |
|---|---|---|
| `F_V23_BG18` / `T36_gate` | rows 2^17 → 2^18 | **−0.00277** |
| `F_V34_BG19` | rows 2^18 → 2^19 | **+0.0009** |
| **`S_B1FG`** | **width 512 → 256 at the same rows** | **+0.0039 step-adjusted** (and it *gained* 1.7% steps from deleting 67.1 M params) |
| `F_V34_BGS2` | 2 x (2^18 x 256), independent hashes, equal params | wash (+0.0001) |
| `F_V30_BGSIGN` | frozen ±1 sign per pair | wash (−0.00003) |
| `F_V31_BGP` | prime hash modulus | +0.0013 |
| `Q_BGPOS` | position folded into the hash | +0.0117 |
| `S_HDCb` | frozen HDC binding instead of the table | +0.0132 |
| `R_BGCONF64` | Witten–Bell confidence on the rows | wash (+0.0003) |
| `G_REINJ2` / `G_REINJ4` | re-inject the row at 2 / 4 more block inputs | +0.0008 / +0.0012 |
| `S_V24_TRI` | hashed trigram alongside | +0.0050 |

Eleven variants. Rows are at an interior optimum (dose decays 5.2x per doubling; measured intra-row collision
rate already **0.29%**, so addressing is saturated, not scarce). **Width is load-bearing at full length** —
and note the sign reversal: the width cut read **−0.00206 at 600 steps** and **+0.0039 at 1800 s**, which is the
cleanest example in the project of the anti-predictive screen.

### 3.3 The "smaller table, bigger transformer" trade, priced

`S_B1FG` deleted half the table (134.2 M → 67.1 M) and **did not respend** the freed parameters. Respend them
perfectly in the dense body:

```
dense params 54.5 M -> 121.6 M          = 0.80 e-folds
at the measured FIXED-TIME capacity slope -0.0024/e-fold   = -0.0019
against the measured cost of the deletion                  = +0.0039
NET                                                        = +0.0020
```

**The trade is a measured loss, not a different ridge.** The −0.0024/e-fold slope already contains the FLOP
price of the added parameters, which is the expensive half: 67 M extra dense parameters roughly doubles the
matmul work, whereas the 67 M table parameters cost only lookup bandwidth. That asymmetry — **lookup capacity
is FLOP-free and dense capacity is not** — is the whole reason the allocation is 65/25 and it is the right way
round.

### 3.4 And the ridge is not conditional on the table

The flat (width, depth) ridge is explained by one number that the table does not appear in: extra dense
capacity is worth −0.0132 per e-fold at fixed *steps* but costs ~0.37 e-folds of step time per e-fold of
parameters, netting **−0.0024 per e-fold at fixed TIME**. Chinchilla-optimal for our compute
(C ≈ 6ND = 2.2e17) is N ≈ 43 M dense against our 54.5 M — 1.27x over, which on a shallow isoFLOP surface is
worth single-digit millinats. Both accounts of the flat ridge are independent of the table.

### 3.5 The one thing the table leaves genuinely unexplored

Our table enters the **input** embedding only (`x = wte(idx) + bigram_lambda * bte(hash(prev, cur))`), so the
(prev, cur) signal must survive 12 layers and then be read through a **rank-512** head. There is no logit-space
n-gram channel at all. And because V = 8192, our output layer is **4.19 M parameters = 2.0% of the model and
~2% of its FLOPs**, against 20–30% in a 50 k-vocab model — **output-side capacity is ~10x cheaper for us than
for a normal LM, and we spend nothing there.**

An exact output-side bigram, `logits += B[prev]` with `B` of shape (8192, 8192) = 67.1 M parameters, needs no
hashing and no collisions and is *smaller* than the table we have. Honest price: **−0.002 … −0.006 at fixed
steps**, 1–2 days of head-kernel surgery (the fused head never materialises logits; the bias would have to ride
the 512-column tile loop with a per-token gather), and **no valid screen exists for it**. Not doable before
2026-09-30. Recorded so it is not lost.

**MEASURED 2026-09-28 — CLOSED on price, not on redundancy.** Offline, on the trained full-length `v72_final.pt`
(no screen): freeze the model, learn only `B` (zero-init, `logits = softcap(z + B[x_t])`, bias INSIDE the cap) on
fresh train tokens by gradient, score on public_val (2M tokens, same tokens both arms).
`experimental/offline_outbigram_fit.py`. The loss is convex in `B`, so this is the ceiling of what the converged
model leaves on the table for an exact current-token output bias:

| train tokens for B | val_bpb delta |
|---|---|
| 20M / 40M / 80M (lr decayed) | +0.000001 / −0.00016 / −0.00029 |
| 50M / 100M / 200M (lr decayed over 200M) | −0.00023 / −0.00045 / **−0.00059** |

Extrapolated to a run's ~570M tokens: roughly **−0.0008**. Not zero — the model misses a little — but the
cheapest honest device build (sidecar gather + scatter of a (16384, 8192) row block per core, +268 MB of grads
through reduce_scatter/all_gather, AdamW on +67M) prices at ~+15–25 ms ≈ **+0.0012–0.0020**. An in-kernel
scatter could cut that to ~10 ms but needs indirect DMA writes inside the compiled graph (tracer rules #9/#11).
Net ≈ 0 at best. **Optimizer trap if reopened:** plain Adam at lr 3e-3 on `B` DIVERGED even with the model
frozen (+0.023 at 20M tokens: Adam sign-amplifies the tiny dense `p` gradients on every column of a hit row);
the embedding group runs at lr ≈ 2.35 and the head group at ≈ 0.010, both far above that. Only row-count-
normalised SGD (`B[v] -= lr * mean_{t: x_t=v} dL/dz`) fit cleanly.

---

## 4. Q3 — the small vocab against a byte-normalised metric

**There is no lever here, and I checked the metric line by line rather than trusting the note.**

`prepare.evaluate_bpb` (`:632-651`): `nbytes = token_bytes[safe_y]`, `mask = valid & (nbytes > 0)`,
`total_nats += (loss * mask).sum()`, `total_bytes += (nbytes * valid).sum()`,
`val_bpb = nats / (log(2) * nbytes)`.

- **The denominator is a property of the text**, not of the model. So minimising `val_bpb` is *exactly*
  minimising uniform per-token cross-entropy over the scored targets. There is no byte-reweighting lever, and a
  long token is worth no more than a short one. (Third independent confirmation.)
- **The four `token_bytes == 0` specials** are excluded from the numerator *and* contribute 0 to the
  denominator, so they cannot be traded against anything: measured **exactly 0.000000** at logit offsets of
  −10, −30 and −80.
- **BPE-feasibility structure** is exhausted: 13,522 single-merge-infeasible (prev, next) pairs = 0.020% of the
  8192² grid, only **1 of 65,528** real adjacent train pairs is infeasible, and masking them measured +0.000398.
- **Train/eval row-structure mismatch** (eval rows are continuous stream cuts, so ~3% of scored positions have
  <32 tokens of context) is already measured: `P_ROLL` at f=1.0 is +0.0007 and `A_STREAM` is +0.0017/+0.0062.

**The one real asymmetry** the small vocab creates is the cheap output layer in §3.5 — and note that the
*obvious* exploit of it cuts the wrong way. The softmax bottleneck is **less** severe for us than for GPT-2
(d/V = 512/8192 = 6.25% against 768/50257 = 1.5%), so mixture-of-softmaxes, which is what `B3`'s escalation
is, is a **weaker** lever here than in the literature it comes from. That matches B3's own author's prior that
it loses, and it is a reason not to spend a gate on the B3 diagnostic with two days left.

---

## 5. Q4 — is 0.033 the right gap?

**Yes, ±0.003, and the offset arithmetic cannot shrink it.**

- **A gap is offset-invariant.** Both numbers are produced by the same portal on the same private shard. The
  offset only matters if *their* offset differs from *ours*.
- **What would make it smaller.** Our offset decomposes into a fixed private-shard bias (cancels) plus a
  portal-side step-loss term. Their step-loss term is smaller than ours if their compile is cheaper, and
  *also* smaller because the exchange-rate curve is flatter at 0.94 than at 0.97 (ours already flattened to
  −0.0243 over v52→v55), so a given loss of portal steps costs them less bpb. Both push their true
  local-equivalent up (worse) and shrink the gap — **by at most ~0.003**.
- **What would make it larger.** Our 0.9671 is an *offline local-GPU* score, and local-GPU eval sits ~0.001
  **above** the box's in-run number, so we are already quoting the conservative side. And 0.9671 is not on the
  portal — the board says 0.9759244.
- **Irreducible uncertainty.** Three offsets, sd ≈ 0.0022. Any single portal number carries ±0.002.
- **What actually changed the number.** Not arithmetic — the leader improved 0.00675 in two days and has been
  dropping ~0.002/day for two weeks. **Re-fetch the board before quoting any gap.**

---

## 6. THE BET: delete the fp32 residual stream — one shape-(1,) parameter costs 37 casts per step

**−0.0005 … −0.002, bit-identical, one paired 300 s throughput screen, and the instrument is valid.**

### The mechanism

`self.bigram_lambda = nn.Parameter(torch.full((1,), 0.2))` is a **shape-(1,)** fp32 tensor, not a 0-dim one.
PyTorch's weak-scalar promotion spares only 0-dim operands, so at `submit/train.py:3251`

```python
x = emb["wte"] + self.bigram_lambda * emb["bte"]     # bf16 + fp32(1,) * bf16  ->  FLOAT32
```

the residual stream is promoted to fp32 and **stays fp32 for the whole stack**. Verified locally:
shape-(1,) fp32 × bf16 → fp32, while `self.skip_lambdas[i]` (0-dim, from integer indexing) → bf16. So there is
**exactly one** promotion site and it is a tensor *shape*, not a dtype decision.

Consequence, at `:3072` and `:3079`, in every one of the 12 blocks:

```python
x2 = x.reshape(T, D).to(bf16).contiguous(); x02 = x0.reshape(T, D).to(bf16).contiguous()
...
return outs[0].view(b, t, D).to(x.dtype)      # bf16 -> fp32, straight back up
```

plus one more at the fused head (`:3227`). **37 casts forward on (16384, 512) tensors, and their gradient casts
back.**

### Counted, not guessed

Per rank per optimizer step (accum = 1), T·D = 8,388,608 elements ⇒ fp32 33.55 MB, bf16 16.78 MB:

| | traffic |
|---|---|
| forward, per block: x cast (50.33) + x0 cast (50.33) + output cast (50.33) | 151.0 MB |
| x 12 blocks | 1.812 GB |
| the fused head's own x cast | 0.050 GB |
| backward: the 37 gradient casts | ~1.812 GB |
| backward: x0's gradient accumulating 12x in fp32 instead of bf16 | +0.402 GB |
| **total deleted** | **~4.06 GB / step** |

At the measured DMA line rate of **393 GB/s** that is **10.3 ms of a 424 ms step = 2.43%** = 0.0246 e-folds =
**−0.00086 bpb** at −0.035/e-fold. Upside if any of it sits on the dependency chain rather than hidden behind
another engine; downside to ~0 if DMA's 65% idle absorbs all of it. **That is precisely what the screen
resolves.**

### Why it is safe

`bf16 -> fp32 -> bf16` is exact (verified). The kernels already compute in bf16 and return bf16, so the fp32
storage carries **no information** — it is a widening of a bf16 value. Casting **once**, after the input
`norm`, instead of 37 times inside the loop, is therefore **bit-identical on the op path**. The single
non-exact spot is the U-net skip add in blocks 6–11 (one rounding instead of two); keep it exact by doing that
one add in fp32 locally, or accept 1 bf16 ulp and confirm with a step-matched pair.

It also does **not** interact with the v73 eps fix: `norm()` now pins `eps=0.0078125` explicitly, so it is
dtype-independent.

### The edit

One line at `submit/train.py:3255`:

```python
x = norm(x).to(COMPUTE_DTYPE)      # was: x = norm(x)
```

`x` and `x0` become bf16, every per-block `to(bf16)` becomes a no-op, and `outs[0].view(...).to(x.dtype)`
becomes a no-op. Nothing else changes. The one non-exact consequence is the U-net skip add in blocks 6–11
(`x = x + self.skip_lambdas[j] * skips.pop()`), which now rounds twice instead of once; the bit-exact variant
is

```python
x = (x.float() + self.skip_lambdas[j] * skips.pop().float()).to(COMPUTE_DTYPE)
```

which keeps 18 of the 37 casts and roughly half the saving. Measure the cheap version first — if the screen
shows nothing, neither variant matters.

Verified locally, on the shipping file's own constructs: `COMPUTE_DTYPE` is `bfloat16` on Neuron
(`compute_dtype_for`), the sidecar gathers bf16 rows (`:3591`, `.to(COMPUTE_DTYPE)`), shape-(1,) fp32 × bf16
promotes to fp32, 0-dim fp32 × bf16 does not, and `bf16 -> fp32 -> bf16` is bit-exact over 65,536 random values.

### Why nobody found it

**Every profiling instrument this project built attributes time to `nki_source_location` — NKI kernel lines.**
These casts are torch ops *between* the custom ops and are invisible to `critpath.py`, `excl_attrib.py` and the
instruction cross-tab. The one instrument that did see them was the **eager** per-op profile, where `_to_copy`
was the single largest host item (`OPPROF4`: 495 calls, 125 ms) — and the fix that followed (`WVIEWS`) deleted
the **weight** casts and left the **activation** casts. The promotion itself was written down this morning in
the `D_EPS` row ("bigram_lambda is an fp32 Parameter so the residual stream stays fp32-typed") purely as the
explanation for an eps mismatch; its throughput consequence was never drawn.

### Falsification

One paired 300 s screen, arm against same-file control in the same session, step counts only — the instrument
`S_PACK0/1` and `S_K128/32` used, and the one class of screen `session-2026-09-27.md` §12 explicitly leaves
valid (throughput is horizon-independent). **Require ≥1.5% more steps** (≈ +10 on ~690, ~3σ on the sd-3.6
step-count noise); or read `PROFILE=1 fwd_bwd`, which resolves 2 ms directly and needs no step-count statistics.
If it gains nothing, the casts are fully hidden **and that closes the whole "delete torch glue between the
fused kernels" family**, which is worth the screen by itself. Bit-identity is provable offline on CPU in
minutes by the exact protocol `R_B2PORT` already used (60 real steps, losses compared in hex float).

**Free rider in the same edit:** `cos_full`, `sin_full` (`:3064`), `dstart` (`:3076`) and `scal` are rebuilt
**identically in all 12 blocks**. `cos_full`/`sin_full` alone are 12 x 2 x 4.19 MB of redundant fp32 repeats.
Hoisting all four to `GPT.forward` deletes ~150 MB and ~50 redundant graph ops, bit-identically.

---

## 7. What to do with the remaining machine time (~15 runs, 2 days)

**Zero runs, in this order:**

1. **Upload `submit/train.py` (= v76) now.** 0.97592 → ~0.9726 portal. It does not change rank but it buys
   0.0033 of cushion against an eleventh entry, which is the only failure mode left. Three handoffs have
   recommended this.
2. **Use the rest of this week's submission quota on it.** 5/week, best counts, portal sd ≈ 0.0022 ⇒ four extra
   draws are worth ~0.002 for zero work and zero risk.
3. **Ask the organizers about the 900 s startup allowance and about per-row test-time adaptation.** Worth
   0.006–0.013 together, more than every remaining code lever combined, and it is a rules question.

**One screen:**

4. **§6, the fp32 residual stream.** −0.0005 … −0.002, bit-identical, one 300 s pair.

**Gates, ranked by expected bpb per run (all step-matched pairs against the `S_SM0` anchor 0.9696259 @4330):**

| | arm | expected | why |
|---|---|---|---|
| 5 | EMA-Nesterov lookahead | −0.0005 | built, verified to 7e-08, β=0 reduces exactly to baseline; published ≈ −0.0018 at n=20 on plain Muon |
| 6 | `NS_STEPS` 3/4 vs 5 | −0.0003 | zero rows in 680; selects a *prefix* of `POLAR_EXPRESS_COEFFS` so lower is also **cheaper**; the ANVIL 6-map cascade's +0.00062 says tighter orthogonalisation hurts here, so looser may help |
| 7 | `VE_T1`/`VE_T2` depth schedule (2/8/2, 5/2/5) | −0.0003 | never swept; the 4/4/4 split is inherited; free in step time |
| 8 | `C_KO0` — `KEY_OFFSET` off at the current config | −0.0002 | measured only as a *package* with `ROPE_FRAC` at v24, 50+ versions ago; if it is now neutral, `ROPE_FRAC` is freed and RF=0.75 becomes admissible. Diagnostic value exceeds score value |
| 9 | `ROPE_BASE` 10k/50k, `SINK_INIT`, `LOGIT_SOFTCAP` init | −0.0002 ea | zero rows as values; free; **must carry the arm's env into the scoring job** (the retracted `ROPE_FRAC` result) |

**Explicitly not worth a run:** `B3` tower mixture (its author's prior is that it loses, and §4 shows the
softmax-bottleneck argument is *weaker* at V=8192 than in the literature it comes from, while its escalation
needs kernel work we do not have); an output-side bigram (§3.5, 1–2 days and unscreenable); anything on the
data axis (capped at −0.0007); anything on shape; MoE; fp8; a full-length `BIGRAM=0` ablation (§3 fixes the
sign of the derivative, which is what the decision needs).

**Honest expectation for the code work: −0.002 to −0.004 from all of items 4–9 together.** The free items
1–3 are worth more. **Nothing on this list closes 0.033, and nothing I found suggests anything could in two
days.**

---

## 8. Method notes

- Nothing was run on the box. Numbers are: existing full 1800 s gates (labelled with their ledger ids), exact
  local measurements, one live HTTP fetch of the public leaderboard, and arithmetic.
- Every cross-domain conversion is labelled with its direction. §2.3's millinat→bpb conversion is the
  **optimistic** bound; §1's throughput ceiling is a **hard upper** bound; §2.5's `s_T`/`s_S` split is a local
  fit and `s_S` is a **lower** bound at larger batch.
- No short quality screen is proposed as evidence anywhere. The one screen recommended (§6) measures step time,
  which `session-2026-09-27.md` §12 leaves valid.
- Three prior-test-of-the-wrong-version checks, as asked: (a) the bigram table's base value rests on a
  **420-step, 27.5 M-token, different-model** measurement and a **500-fixed-step** one — the two instrument
  classes since ruled anti-predictive; (b) the batch axis has full-length gates only **upward** from 131,072,
  and the 65 k-vs-131 k comparison that set the current value was made at v19/v21 with a 0.94 s step and a
  different model (§2.5 closes the downward leg by arithmetic instead); (c) the `900 s` allowance was priced in
  `handoff-2026-09-26.md` as "5 updates are 5 updates", which bounds the wrong quantity — the cap is on
  updates, not tokens, and §2.5 shows tokens are worth 4x updates here.
- Sources: `trn2_results.tsv` (680 rows), `findings/session-2026-09-27.md`, `handoff-2026-09-26.md`,
  `quality-levers-2026-09-26.md`, `data-axis-2026-09-27.md`, `mfu-profile-2026-09-21.md`,
  `exclusive-time-method-2026-09-25.md`, `leaders-gap-2026-09-22.md`, `train2_results.tsv`,
  `flow-inference-compute.md`, `prepare.py`, `README.md`, `submit/train.py` (= v76),
  `submission_raw/train3_canon.py`.
