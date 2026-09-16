# The "corrupted eval" was a vehicle bug: tail-EMA started at step 0 on cold short screens (found 2026-09-16)

## Symptom (seen 6 times, misattributed 4 times)
A screen whose eval bpb is far above what its training loss implies while training looks normal and causality passes:
FL_O3c 1.1195, FL_IDC 1.1210, FL_FCS 1.1204, ATT_PLAIN_HOIST 1.1254 (600 s; clean runs ~1.053-1.056), S_HDC 1.1508
(600 s), I_SYNC1 1.2605 (300 s; clean 1.120). Every one of them logs `EMA of the weights started at step 0` (or step 1);
every clean run starts it at ~85% of the steps (I_SYNC4: step 491; S_SMEARDROP: step 966).

## Mechanism
`elapsed_frac` at step 0 is computed AFTER the cold compile but BEFORE the budget clock is restarted at the end of
the excluded startup step (`step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED`). With a 300 s / 600 s screen and a 500-775 s
compile, elapsed_frac >= EMA_START (0.85) at step 0, so the EMA (decay 0.9933, ~150-step horizon) starts from the
INIT weights and ends as a ~150-step-lagged average of a rapidly-changing early model; the final blend (35%) then
mixes that in. The checkpoint itself is damaged (I_SYNC1 vs I_SYNC4: matrix norms 20% smaller, cosine 0.83-0.89;
non-EMA'd embeddings equal), so it is not an eval-graph problem. The trigger condition is compile >= 0.85 x budget:
600 s screens with startup ~510-550 s sit on the edge (ATT_PLAIN 442 s clean, S_SMEARDROP 510.4 s clean,
ATT_PLAIN_HOIST 546 s hit), 300 s screens always hit when cold, 1800 s gates never did (would need a 1530 s compile).

## Fix (vehicle, 2026-09-16)
`if step < TRAIN_STARTUP_STEPS_EXCLUDED: elapsed_frac = 0.0` right after elapsed_frac is computed (the clock is
restarted at the end of that step anyway; warmup is step-based so the LR is unaffected). Goes into v44.

## Corrections to earlier verdicts
- "-O3 / any NEURON_CC_FLAGS corrupts the eval graph" (handoff-2026-09-13b, plan-097): FALSE. Those were cold 600 s
  screens with 549-775 s compiles. The flags remain useless for speed (all 0.52-0.53 s/step) but are not hazardous.
- "Per-step glue hoist recovers ~1 point but corrupts eval": the hoist is clean; own attention + hoist = 0.53 s
  vs 0.52 s reference = parity. Still no reason to leave the vendor attention.
- S_HDC "+0.098": INVALID. The HDC binding (wild-ideas 3.1) was never measured; re-screened warm as S_HDCb against
  S_REF600 (v43 config, 600 s). RESULT: S_HDCb 1.064601 vs S_REF600 1.051451 = +0.0132 - HDC still loses, just honestly.
- Multi-hop "eval collapse = tail-EMA blend with hop-coupled weights" (Sep 12-13): probably this bug as well
  (cold 600 s screens); the full-length hop gate (T58, 0.9926) was 1800 s and stands.
- Portal risk: the portal compiles cold at 1800 s; the trigger needs a >= 1530 s compile, so v40-v43 were not hit
  (consistent +0.006 portal offset agrees). v44 carries the fix regardless.

## Rule
A cold screen shorter than compile / EMA_START is invalid without this fix. Always check the
`EMA of the weights started at step N` line: N must be ~EMA_START x num_steps.
