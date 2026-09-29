# In-row context cache: offline ceiling on V86 (2026-09-28)

**Verdict: CLOSED. The best cache fitted on train improves val by -0.00018 bpb, and the ceiling fitted on val is -0.00019.
That is about 1/20 of the -0.004 we need and 1/10 of the 0.002 bar. V86's attention already does exact induction, so
the cache adds almost no information.**

## Setup
- Checkpoint `/tmp/V86.pt`, loaded with `load_for_eval` from `submission_raw/submission_v86_gate.py` on the RTX 4080 (fp32 logits).
- **Val:** the first 2048 rows (2,097,152 targets) of `public_val`, built by `prepare.make_eval_dataloader` itself at
  seq 1024. That is exactly the 4x524288-token window `eval-public` scores, because the loop stops at 2048 full rows. bpb
  follows `evaluate_bpb` exactly: nats are counted where `token_bytes>0` and divided by ln2 times the sum of bytes.
- **Train (used for fitting):** 2048 rows packed the same way from the last 4 row groups of shard 4. In a ~1.8-epoch
  run these groups are seen only once. The model is barely memorized on them: train bpb 0.9610 vs val 0.9668.
- Every cache uses only tokens of the same row before the predicted position, in two scopes: the whole row, or the
  current document only. Parameters are fitted on train rows and applied to val. The "ceiling" is fitted on val and is
  a diagnostic only.
- The per-position scores are exact. The script builds the T x T exact-match matrices and suffix-match lengths up to 8
  on the GPU, and computes candidate counts with one matmul of match matrix @ one-hot(successor), the same op a
  Trainium version would use.

## Results (val bpb; baseline **0.966782**, which matches the official V86 score of 0.966782)

| cache variant | val, train-fit | ceiling, val-fit |
|---|---|---|
| unigram, row, decay none/1/64/1/256 | -0.00011 / -0.00013 / -0.00013 | -0.00015 |
| unigram, doc, no decay or 1/256 | -0.00015 | -0.00016 |
| induction (bigram) row, global lambda (fitted lambda = 0.001) | -0.00010 | -0.00010 |
| induction row, trigram+ weighted (1,3,9,27), global lambda | -0.00010 | -0.00010 |
| induction doc, (1,3,9,27), global lambda | -0.00014 | -0.00014 |
| induction row, lambda(max match len, n matches) | -0.00013 | -0.00014 |
| **induction doc (1,3,9,27), lambda(max match len, n matches)** | **-0.00018** | **-0.00019** |
| successor of most recent longest match (point mass), lambda(len up to 8) | -0.00007 | -0.00007 |
| lambda(len up to 8, n), row | -0.00012 | -0.00014 |
| **logit-space bias** b_v = sum of log1p(c_k), [c_k>0], row and doc, recency, vote share (21 weights, LBFGS) | -0.00002 | -0.00002 |
| logit-space log1p(c1) only | 0.00000 (fitted beta = 0.00) | 0.00000 |

The fitted lambdas are tiny: 0.001-0.002 globally. Even with an 8-token suffix match and 17 or more matches, lambda is
only about 0.05-0.09. The logit-space weights fit to about 0. The model's logits are already calibrated with respect to
cache evidence. The small gain from probability-space mixing is only floor insurance: it rescues rare cases where the
model gives a cached token near-zero probability.

## Why: the model already copies
- **54% of positions have at least one induction match** in the row (49% within the document). The target is among
  the induction candidates at 18% of positions, which is 33% of matched positions.
- Average gain on matched positions: 0.64 millinats per token. Unmatched positions gain 0.
- When the successor of the most recent longest match is correct, V86 already gives the target:

| match length | share of positions | successor correct | model p(target) when correct: mean / median |
|---|---|---|---|
| 1 | 35.8% | 13% | 0.69 / 0.85 |
| 2 | 8.8% | 36% | 0.81 / 0.96 |
| 3 | 3.6% | 48% | 0.84 / 0.97 |
| 4-5 | 2.9% | 55% | 0.86 / 0.98 |
| 6-8 | 2.5% | 73% | 0.90 / 0.99 |

  When the cache is wrong, the model gives the target about 0.19-0.21, so it knows when not to copy. This is the
  attention induction circuit working as intended. The papers cited (Grave 2016, pointer-sentinel, kNN-LM) used LSTMs
  or models without an in-context copy mechanism, which is why they saw large gains.

## Where the (tiny) gain sits
- By token type: "space+word" tokens give 66% of the gain (55% of tokens), "space+Capital" names give 16% (7% of
  tokens), word pieces 19%, and digits and punctuation slightly negative. So names are over-represented about 2x, but
  the absolute amount is negligible.
- By row: the top 5% of rows hold 70% of the gain. The best row gains 0.004 bpb within itself. These rows are
  long technical or patent prose that repeats multi-word terms, not code.

## Recommendation
Do not build it. The train-fitted gain is 0.00018, below every gate's noise and 10x under the 0.002 bar. The cheap,
kernel-friendly form (a logit-space log-count bias) is worth 0.00002. A trained in-model version could do somewhat
better than a post-hoc mix, but the ceiling shows there is almost no unused in-row information to capture. Cost
reference, in case it is revisited: per 1024-token row it takes one T x T x d_hash equality matmul plus one T x T x V
count matmul (about 17 GFLOP per row at V=8192; tens of ms per step on Trainium at our batch). A logit-space bias would
add one extra (T,V) addend inside the fused head/CE tile loop. Neither is justified by -0.0002.
Close this item alongside the other eval-side items in `findings/`.

Scripts and logs are in this directory: `dump.py` (GPU pass), `analyze.py` (probability-space fits), `logit_fit.py`
(logit-space fit), `diag.py`, `analyze.log`, and `logit.log`.
