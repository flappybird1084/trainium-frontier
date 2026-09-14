# Plan to 0.97 local (written 2026-09-13; supersedes the "levers" sections of older handoffs)

## Where we are, in numbers
- `submit/train.py` = v40, **0.985161 local** (portal = local + ~0.0065 -> ~0.992). Gap to 0.97 local = **0.015**.
- Step 0.53 s, ~2900 steps in 1800 s, 131k tokens/step (~380M tokens). Exchange rate: **1% of step time ~ 0.0003 bpb**.
- The leaderboard top (~0.93 portal) is ~0.06 away. **Nothing measured on this line reaches it**; that would need a 2-3x
  faster step or a different training regime. This plan is about 0.97 local, and even that needs most levers to land.

## Exhausted - do not re-litigate (each with its number)
compiler flags (9 configs, all 0.52-0.53 s; -O3 and any NEURON_CC_FLAGS corrupt eval); width 640 (lost 4x, +28% step;
kernels cap D at 512); depth 14 (OOB flake, 2 deaths); fp8 (~1%); comm flattening (worse) and async reduce-scatter (no
gain, grad_sync 43 ms); own attention with strided DMAs (NaN in-model, rule #11); multi-hop attention at fixed time
(+0.0074 at the gate; -0.008 at fixed steps); prehead denoise (+0.006); output-side logit flows, stack recurrence, latent
scratchpad (wash/worse); DeltaNet (unstable); 600 s LR screens (do not transfer: v41 +0.009); bigram sign trick (wash),
prime modulus (worse), mimetic V-O init (worse); tinygrad / fusion search (no Neuron code generator exists).

## Levers with evidence, ranked by expected gain per box-hour
| # | lever | expected | cost | confidence |
|---|---|---|---|---|
| A | **Full-length scalar re-tune** at 1800 s gates (WARMDOWN 0.9, MATRIX_LR, EMBED_LR, EMA decay/start/blend). Every current value was set by 600 s screens, which we now KNOW over-favor high LR; the full-length optimum has never been swept | -0.002 .. -0.004 | 8 gates = 4 box-hours, unattended | medium |
| B | Own attention: plain fwd + our bwd + hoisted per-step glue. MEASURED: clean and correct, 0.53 s vs 0.52 ref = PARITY (the bwd kernel saves ~9 ms/step; remaining per-layer glue + in-graph scheduling eat it). A win needs the attention fused INTO the layer kernels (no per-layer glue at all) - see D-adjacent megakernel | ~0 as is; -0.001..-0.002 if fused into the layer op | 2-3 days for the fusion | low-medium |
| C | Untried free modeling items from the strange-wins list: canonical token masking (#2, provable), prefix-token aux loss (#1), cautious weight decay (#8), EMA-Nesterov outer loop (#9: -6% steps on plain Muon in the calibration -> ~-0.002 at fixed time) | -0.002 .. -0.004 total | 1 day to implement (CPU smoke each) + 4 gates | medium-low (each individually small) |
| D | **Cheap multi-hop attention**: reuse K/V in the hop (recompute only Q + attention + out-proj). Hop cost ~15% -> ~10%; the fixed-step gain (-0.005 with box semantics) then nets ~-0.002 .. -0.003 | -0.002 .. -0.003 | 1-2 days (new fused op variant) + 2 gates | medium-low |
| E | Depth 14, blocked by the OOB flake ("scatter/gather via vector DGE"; 10 dev-shape deaths, 0 on production shapes). Understanding the flake is worth doing anyway: it is the portal-crash risk | -0.001 .. -0.002 | 1 day investigation + 1 gate | low |
| F | Overlap the optimizer (10% of step) with the backward. Comm overlap already measured at zero; only the optimizer part remains; requires splitting the monolithic compiled graph | up to -0.003 | 2-3 days, high risk | low |

Best case if everything lands: ~-0.017 -> ~0.968. Realistic (A + half of C + one of D/E; B is at parity): ~-0.007 -> ~0.978.
**0.97 is reachable only if nearly every lever lands; it is not a safe target.** Say so when asked.

## Sequence (the box runs one thing at a time; keep it saturated with gates, do the coding in parallel)
1. **Tonight:** queue the lever-A gate sweep (8 x 1800 s, one scalar per run, a reference every 4). The queue runner survives
   individual crashes. Read only the 1800 s numbers; 1800 s noise band ~0.001-0.002 (v40 vs v39 = 0.0014).
2. **Meanwhile (local):** implement C as default-off knobs with CPU smokes (loss == CE(head logits), eval-hook signature).
3. **Day 2:** gate each C item on top of the best A config, one at a time. B measured at parity (0.53 vs 0.52 s): keep the vendor attention unless the attention gets fused into the layer kernels.
4. **Day 3-4:** D (K/V-reuse hop op) if A+C are still short of 0.97; E in parallel (flake). F only if everything else landed.
5. Freeze **v42** = best gated combination (make_v4x.py generator pattern), gate the clean file, promote, commit + push
   `personal-remote` (memory `git-personal-remote`).

## Method rules that this campaign paid for
- Anything touching LR/schedule: **gates only** (1800 s). One change per gate. Reference every ~4-8 runs.
- Judge only by eval bpb: training loss lies for anything with train/eval asymmetry (hops, EMA).
- Never ship the vehicle while a queue has unlaunched lines; never run `prepare.py eval-public` standalone on the box.
- A compiler flag or -O3 can corrupt eval silently; the harness cannot catch it. Stay at the default optimization level.
