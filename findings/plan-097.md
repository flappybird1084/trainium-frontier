# Plan to 0.97 local (written 2026-09-13; supersedes the "levers" sections of older handoffs)

> **Three numbers below are stale - see `findings/repeat-regime.md` (2026-09-14):** (1) the corpus is not a wall
> at ~1.4 epochs (~4 epochs is near-free, so ~2.8x more throughput still converts); (2) the 1800 s noise band is
> **~0.0004**, not 0.001-0.002 (three same-config PACK=4 gates spread 0.00035) - re-read past "wash" verdicts
> against that; (3) the leaderboard top is **0.967 portal (~0.960 local)**, not ~0.93, so the lead is ~0.021
> local away. That file also carries three untried levers on the data/repeat axis.

## Where we are, in numbers
- `submit/train.py` = v40, **0.985161 local** (portal = local + ~0.0065 -> ~0.992). Gap to 0.97 local = **0.015**.
- Step 0.53 s, ~2900 steps in 1800 s, 131k tokens/step (~380M tokens). Exchange rate: **1% of step time ~ 0.0003 bpb**.
- The leaderboard top (~0.93 portal) is ~0.06 away. **Nothing measured on this line reaches it**; that would need a 2-3x
  faster step or a different training regime. This plan is about 0.97 local, and even that needs most levers to land.

## Exhausted - do not re-litigate (each with its number)
compiler flags (9 configs, all 0.52-0.53 s; the 'corrupt eval' was the cold-start EMA bug, see ema-cold-start-bug.md); width 640 (lost 4x, +28% step;
kernels cap D at 512); depth 14 (OOB flake, 2 deaths); fp8 (~1%); comm flattening (worse) and async reduce-scatter (no
gain, grad_sync 43 ms); own attention with strided DMAs (NaN in-model, rule #11); multi-hop attention at fixed time
(+0.0074 at the gate; -0.008 at fixed steps); prehead denoise (+0.006); output-side logit flows, stack recurrence, latent
scratchpad (wash/worse); DeltaNet (unstable); 600 s LR screens (do not transfer: v41 +0.009); bigram sign trick (wash),
prime modulus (worse), mimetic V-O init (worse); tinygrad / fusion search (no Neuron code generator exists).

## Levers with evidence, ranked by expected gain per box-hour
| # | lever | expected | cost | confidence |
|---|---|---|---|---|
| 0 | Data/packing: **A_PACK4 = 0.981426 vs 0.985741 (-0.0043), the biggest gain since v40**, one env var, zero step cost. Stream packer (all rows cut) lost +0.0017: best-fit whole docs + long docs placed whole is what pays. Repeat + PACK=2 gating; then clean v42 = v40 + PACK=4 | -0.004 | done if the repeat confirms | high |
| A | Full-length scalar re-tune at 1800 s gates. WARMDOWN 0.7 = wash (+0.0001). The remaining 7 gates (MATRIX_LR 0.85/0.55, EMBED_LR 4.5/8, EMA blend 0.5, EMA start 0.75, WARMDOWN 0.8) were RE-BASED on PACK=4 (queue_A7, ref T59 0.981197) so any win stacks into v43 | -0.001 .. -0.003 | running, unattended | medium-low |
| B | Own attention: plain fwd + our bwd + hoisted per-step glue. MEASURED: clean and correct, 0.53 s vs 0.52 ref = PARITY (the bwd kernel saves ~9 ms/step; remaining per-layer glue + in-graph scheduling eat it). A win needs the attention fused INTO the layer kernels (no per-layer glue at all) - see D-adjacent megakernel | ~0 as is; -0.001..-0.002 if fused into the layer op | 2-3 days for the fusion | low-medium |
| C | Untried free items. DONE Sep 15 on the PACK=4 base: WD 0.2/0.3 (+0.005/+0.011) and MIR 0.05/0.10 (+0.005/+0.010) are monotonically WORSE - the repeat-regime regularization thesis fails here. Parallel residuals: stable at lower LR but +0.005 at equal steps (closed). Still untried: canonical token masking, prefix-token aux loss, cautious WD, EMA-Nesterov | -0.001 .. -0.003 | 1 day + 4 gates | low |
| D | **Cheap multi-hop attention**: reuse K/V in the hop (recompute only Q + attention + out-proj). Hop cost ~15% -> ~10%; the fixed-step gain (-0.005 with box semantics) then nets ~-0.002 .. -0.003 | -0.002 .. -0.003 | 1-2 days (new fused op variant) + 2 gates | medium-low |
| E | Depth 14, blocked by the OOB flake ("scatter/gather via vector DGE"; 10 dev-shape deaths, 0 on production shapes). Understanding the flake is worth doing anyway: it is the portal-crash risk | -0.001 .. -0.002 | 1 day investigation + 1 gate | low |
| F | Overlap the optimizer (10% of step) with the backward. Comm overlap already measured at zero; only the optimizer part remains; requires splitting the monolithic compiled graph | up to -0.003 | 2-3 days, high risk | low |

Best case if everything lands: ~-0.017 -> ~0.968. Realistic (A + half of C + one of D/E; B is at parity): ~-0.007 -> ~0.978.
**0.97 is reachable only if nearly every lever lands; it is not a safe target.** Say so when asked.

## Sequence (the box runs one thing at a time; keep it saturated with gates, do the coding in parallel)
1. **Tonight:** queue the lever-A gate sweep (8 x 1800 s, one scalar per run, a reference every 4). The queue runner survives
   individual crashes. Read only the 1800 s numbers. CORRECTED (repeat-regime.md s3): same-config full-length spread is ~0.0004 (PACK=4 triple 0.981426/0.981073/0.981197); every run is already same-seed (manual_seed(42)) and same data order, so all comparisons are effectively paired.
2. **Meanwhile (local):** implement C as default-off knobs with CPU smokes (loss == CE(head logits), eval-hook signature).
3. **Day 2:** gate each C item on top of the best A config, one at a time. B measured at parity (0.53 vs 0.52 s): keep the vendor attention unless the attention gets fused into the layer kernels.
4. **Day 3-4:** D (K/V-reuse hop op) if A+C are still short of 0.97; E in parallel (flake). F only if everything else landed.
5. Freeze **v42** = best gated combination (make_v4x.py generator pattern), gate the clean file, promote, commit + push
   `personal-remote` (memory `git-personal-remote`).

## Method rules that this campaign paid for
- Anything touching LR/schedule: **gates only** (1800 s). One change per gate. Reference every ~4-8 runs.
- Judge only by eval bpb: training loss lies for anything with train/eval asymmetry (hops, EMA).
- Never ship the vehicle while a queue has unlaunched lines; never run `prepare.py eval-public` standalone on the box.
- ~~A compiler flag or -O3 can corrupt eval silently~~ (retracted 2026-09-16: cold-start EMA bug). A cold screen shorter than compile/EMA_START is invalid without the step-0 elapsed_frac guard.
