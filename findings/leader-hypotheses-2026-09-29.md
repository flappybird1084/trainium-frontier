# How is someone at 0.937? Five ranked hypotheses (2026-09-28)

**Bottom line.** No team has published how it works. No team name turns up anything public. So everything below is inference. The gap is 0.9722 − 0.9374 = **0.035 portal**, which puts the leader near 0.932 on our local scale. Our measured exchange rate is −0.029 bpb per e-fold of wall clock (C_2400). At that rate the gap equals **~3.3x our compute**, or about 2.7x at the −0.035 slope. That is too big for recipe tweaks, but plausible for **a systems gap plus one mechanism we have never tested**.

## 1. They get 2–3x our useful FLOPs per second (most likely, 0.015–0.030)

- **Mechanism.** We run at about 18% model-FLOP utilisation: 6.08e12 FLOP per core per step in 0.424 s, against a 79 TF/s peak. About 136 ms of each step is fixed cost that does not depend on the model shape (collectives, optimizer, launch). The hardware favours wide layers (d = 1024) with every matmul at N = 512, fp8 on the matrix engine (2x the bf16 peak), and far fewer device ops.
  - The fastest modded-nanogpt record did exactly this on GPUs (fp8, graph capture, sparse communication): 1.85x faster than our ancestor record.
- **Why our "throughput can't explain it" argument is weak.**
  - Its 1.47x ceiling bounds *our* instruction mix, not a redesign.
  - "The compiler path is 5.2x slower" only shows that compiler-only teams are slower. It says nothing about expert kernel teams.
  - The bent-curve fit that says the gap is "unreachable at any budget" was fitted to three points. Our only direct measurement, C_2400, is log-linear.
- **Rules:** allowed. **Ledger:** our own throughput levers are exhausted, but "what does 2–3x compute buy?" has never been measured.
- **Cheapest test (one device arm, about 75 min).** Run v86 unchanged at `MAX_TRAIN_SECONDS=3600` and score it officially.
  - If the result is **≤ 0.947 local** (log-linear predicts 0.9465), compute explains at least two thirds of the gap. Phase 2 work should then be a from-scratch wide/fp8 redesign, not more knobs.
  - If **≥ 0.952**, compute explains at most half.
  - A 5400 s arm (3x) settles it outright if a slot is free.

## 2. A memory built from the training data, read at eval time (worth 0.003–0.015; never tested here)

- **Mechanism.** Store `(final hidden state, next token)` pairs from the training data in the checkpoint. At eval, `logits = log(λ·p_kNN + (1−λ)·p_LM)`.
  - This is kNN-LM. A training-set datastore took WikiText-103 perplexity from 18.65 to 16.12, which is −0.146 nats/token.
  - The token-only version is an n-gram / infini-gram mixture over the 316M training tokens.
  - In Parameter Golf, n-gram mixing was the only technique still giving clear gains among already-competitive entries. Most headline numbers there were invalidated (unnormalised mixtures or the validation stream); our scorer's log_softmax rules out the first.
- **Against.** Near-duplicates are only 0.03% of tokens, and our exact output-side bigram fit gave only −0.0008. Datastore gains shrink as the base model gets stronger, and on web text that doesn't repeat.
- **Cost if it works.** The datastore can be captured for free: save hidden states during the last ~15% of training forwards. That is about 85M tokens, or about 22 GB at 128 dimensions in bf16. Brute-force kNN over 2M eval tokens takes about 1–2 minutes of the 20-minute eval allowance (my estimate, not measured).
- **Rules:** allowed as written: "assets produced within your own training loop on the provided data", and the forward stays pure, causal and frozen. The reviewer could still object, so ask the organisers before shipping. **Ledger:** no rows.
- **Cheapest test (offline, local GPU, 3–4 h).**
  1. Take the v86 checkpoint and build datastores of 2M, 8M and 16M training tokens (PCA to 128 dimensions, fp16, which keeps RAM under 5 GB).
  2. Query 262k held-out tokens: train-shard documents that are not in the store, plus the public dev shard.
  3. Fit λ, temperature and k on one half and score the other half.
  4. Fit the gain against log(store size) and extrapolate to 85M and 316M tokens.
  5. Run a hashed Kneser-Ney 2–6-gram mixture through the same script.
  - **Kill** if the extrapolation to 85M is smaller than −0.003.

## 3. Non-modelling margin: portal variance, uploads, and the startup window (worth 0.004–0.012 combined)

- **Mechanism, three parts.**
  1. **Best-of-N uploads.** The rules allow 5 uploads a week and the best counts. Our three portal offsets span 0.0044 (sd about 0.0022). Best of about 25 uploads sits about 0.0045 below the mean; best of our few sits near the mean.
  2. **Different portal offsets.** Ours is +0.0054, mostly lost steps on the portal. A team with a short, deterministic compile may lose nothing (≤ 0.003).
  3. **The startup window.** The first 5 optimizer steps are excluded from the clock, up to a 900 s cap, and we use about 430 s of it. Doing real optimizer updates in the unused ~470 s would be worth about −0.007 (+26% time).
- **Rules and ledger.**
  - Part 3 as ordinary updates is against the stated intent. Our legal form (giant accumulated steps, v84) lost +0.0020.
  - Parts 1 and 2 are legitimate. They are already in the proposals file but not acted on.
- **Cheapest test (no device).**
  1. Upload v86 now, and re-upload the same file each time the quota resets. Three draws measure our own portal spread.
  2. Re-fetch the leaderboard every few hours: trendless jumps mean lottery draws; steady drops mean real work.
  3. Email the organisers one question: are optimizer updates before boundary 5 allowed?

## 4. A much larger n-gram table with sparse updates and a trigram channel (worth 0.002–0.006)

- **Mechanism.** modded-nanogpt PR #360 (ANVIL2) grew its hashed n-gram table 224x, to 84.6M rows, and added a trigram channel. It pulled only the rows touched by each batch, sent back segment-summed gradients, and ran Adam on touched rows only. That made optimizer and communication cost almost independent of table size. Credited with 7 s of a 74 s run (about −0.003 for us).
- **Our evidence.** Our table-size results were all measured with dense AdamW and dense collectives.
  - BG20 lost 10.9% tokens/s.
  - Even at matched steps, BG20 (−0.00156) was worse than BG19 (−0.00225). Dense Adam keeps pushing rows that were not touched this step with stale momentum. With about 6% of rows touched per step, that plausibly hurts bigger tables more.
  - The trigram lost three times, each time on step cost.
- **Rules:** allowed. **Ledger:** table size and trigram are closed only under the dense implementation. Lazy Adam is built (`LAZY_BTE`) but never gated.
- **Cheapest test (one step-matched device arm, about 40 min).** `BIGRAM_BITS=20 LAZY_BTE=1` at 4330 steps against the BG19 anchor.
  - If BG20 with lazy Adam beats BG19 by ≥ 0.0008, a sparse-row table is a real Phase 2 lever.
  - If not, the table axis is closed for good.

## 5. A better per-token recipe from many small items (worth 0.003–0.010)

- **Mechanism.** Parameter Golf's best non-TTT entries run nearly our shape (11 layers × 512, vocab 8192). Their stack carries items we either lack or measured under a different parameterisation:
  - learned per-head QK gain, initialised at about 5 on top of QK-norm;
  - MuonEq-R row normalisation before Newton–Schulz;
  - layer-scaled norms;
  - LeakyReLU² with slope 0.75;
  - 3-layer recurrence plus parallel residuals.

  
- **Against.** Seven of our last seven knob arms lost, and parallel residuals (+0.005) and recurrence (+0.0011) are closed here.
- **Rules:** allowed. **Ledger:** mostly closed item by item. It has never been tested as a whole-stack comparison.
- **Cheapest test (local GPU, about 4 h).** Run Parameter Golf's public record `train_gpt.py` (pre-quantisation, no TTT) on our tokenised ClimbMix with our dataloader, against our vehicle via `gpu_gdn.sh`. Match tokens at about 1500 steps.
  - If theirs is ≥ 0.005 better per token, port the stack one diff at a time.
  - Within ±0.002: dead.

## Candidates from the brief that I rule out
- **Illegal:** cross-row context or cache (depends on the batch), and test-time training (the forward must be "frozen").
- **Already there:** in-row cache (attention does it; E18 +0.0008).
- **No lever:** tokenizer and byte factorisation (bpb is exactly per-token cross-entropy); weight tying (the head is 2% of FLOPs).
- **Closed below 0.001:** data order, dedup and filtering (near-duplicates are 0.03% of tokens); multi-epoch (−0.00065 in total).
- **Forbidden and worth ~0 anyway:** more ClimbMix shards.
- **Closed:** other optimizers (all three ANVIL pieces lost), depth recurrence (+0.0011), MoE (+33% MLP bought 0.00006).
- **Part of #1, not separate:** a larger model trained in quantised precision.

## Order to run in the next 24 h
1. Hypothesis 3 uploads and the organiser email: free.
2. Hypothesis 2 offline kNN / n-gram study, on the local GPU in parallel.
3. Hypothesis 1: the 3600 s device arm.
4. Hypothesis 4: the lazy-BG20 device arm.
5. Hypothesis 5: the Parameter Golf stack comparison, on the local GPU once hypothesis 2 finishes.

## Sources
- Leaderboard JSON: https://d3n62br6wsx257.cloudfront.net/api/leaderboard
- Competition rules: https://github.com/aws-neuron/trainium-frontier
- Parameter Golf repo and leaderboard: https://github.com/openai/parameter-golf/
- Parameter Golf n-gram and TTT history, including invalidations: https://github.com/openai/parameter-golf/issues/140
- "Parameter Golf: What Really Works?": https://arxiv.org/html/2607.01517v1
- modded-nanogpt records (#92 ANVIL2): https://github.com/KellerJordan/modded-nanogpt
- ANVIL2 ablation, sparse-row table: https://github.com/KellerJordan/modded-nanogpt/pull/360
- kNN-LM: https://arxiv.org/abs/1911.00172
- Infini-gram: https://arxiv.org/abs/2401.17377
- Why kNN-LMs work (gain shrinkage): https://arxiv.org/abs/2301.02828
- NanoGPT Slowrun (data-limited regime, not our analogue): https://github.com/qlabs-eng/slowrun
- Internal: findings/gap-analysis-2026-09-27.md, ideal-model-2026-09-28.md, lazy-bte-2026-09-28.md, comm-bytes-2026-09-24.md, data-axis-2026-09-27.md, ideas.md §3, trn2_results.tsv (C_2400, BG19/BG20, V84, V86)
