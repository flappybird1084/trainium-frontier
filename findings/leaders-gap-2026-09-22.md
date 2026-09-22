# The 0.02 gap to the leaders (0.956-0.958) — recon 2026-09-22 (goal: 0.96 local)

No public disclosure exists: Devpost gallery is a bare portal, no team writeups, no Trn2 training MFU numbers, no hits
for team names + trainium/nki/bpb. We cannot copy anyone directly. Two actionable leads from adjacent evidence:

## Lead 1 (throughput): our hand-NKI kernels may be capping MFU, not raising it
General Neuron guidance (Transformer Lab, lab.cloud/research/custom-kernels-on-trainium/, 2026): hand kernels help only
where the compiler can't fuse; they do NOT beat the compiler's own fusion/scheduling once it handles an op. A torch.compile
Neuron training pipeline hit 68% MFU (Trn1 LoRA, not a clean comparison) vs our 15-25%. => TEST whether LAYER=torch
(compiler-fused layer) is faster than our LAYER=op hand-kernels. If yes, a compiler-first rewrite could ~2-3x tokens ->
the throughput needed for 0.96. (CLEAN_LT screen launched 2026-09-22.)

## Lead 2 (quality): MUDDFormer dynamic dense connections — the one credible technique we lack
arXiv 2502.12170 (ICML 2025), adopted by modded-nanogpt record 85 (05/27/26). Per-position, per-stream (Q/K/V/residual)
dynamic dense connection weights across layers (depth-wise multi-head). Claims 1.8-2.4x compute-equivalent quality at
scale; a small extra projection/gating, portable, not hardware-specific. RISK: adds compute -> like multi-hop, a
fixed-step win can be a fixed-time loss at our tiny scale; must gate at 1800s, not fixed steps.

## Already have / tested (do not re-port): XSA (knob), smear/drop-attn0, bigram-hash, value-embeddings, QK-norm,
## doc-masking, softcap, Muon+NorMuon+cautious-WD+momentum-warmup, window-warmup ("adds nothing"), TOP/prefix-pred head.
## Unknown: "bimaxwell momentum" (Muon extension, kmaxwell-sota) — title only, needs the diff.

## Bottom line: no smoking gun. The path to 0.96 is (1) a compiler-first throughput rewrite if CLEAN_LT shows the
## compiler beats our kernels, and/or (2) MUDDFormer as a fixed-time quality gate. Both are real bets, neither guaranteed.

## Lead 1 (compiler-clean rewrite): DEAD (2026-09-22, CLEAN_LT4)
LAYER=torch (compiler-fused layer) OOMs at DB16 (192MB alloc fail) and at DB4+accum4 runs 2.694 s/step vs v46's 0.516 s
for the same 131072 tokens = **5.2x slower**. Our hand-NKI kernels are far faster AND more memory-efficient than the
compiler path. The recon's "hand kernels may be why we're behind" hypothesis is refuted for our setup. => the throughput
gap to the leaders is NOT because we hand-wrote kernels; our step is already near the kernel-achievable floor. Only
Lead 2 (MUDDFormer quality) and pure-quality architecture levers remain for 0.96.

## Lead 2 (MUDDFormer): FIXED-STEP WIN -0.0062 (2026-09-22, local 500-step eager)
MUDD_ON (ways=4) 1.181899 vs MUDD_OFF 1.188077 = -0.0062 at 500 fixed steps. +17% step time (4.75 vs 4.05 s). Implemented
in submission_raw/train3_mudd.py (eager only, knob MUDD, MuddMixer per block, dynamic dense weights = static + RMSNorm->MLP
dynamic, multiway feeds Q/K/V/residual; stability fix = RMSNorm the DDW input + fp32 sum). KEY DIFFERENCE from multi-hop
(which won fixed-step but INVERTED at fixed-time): MUDD's mixers start at identity and are UNDERTRAINED at 500 steps, so its
gain should GROW at the full 3400-step budget, potentially surviving the +17% step penalty. Decisive test = fixed-time box
gate, which needs MUDD on the op/kernel path. Residual-only (ways=1) is portable torch-only (layer op takes one input);
full multiway (ways=4) needs the layer op to accept separate q/k/v inputs (kernel signature change).

## MUDD fixed-time box gate: LOSS as eager glue; hinges on gain-scaling (2026-09-22)
MUDD_W1b (ways=1, op path, MUDD=1) box gate = 1.012563 at only 1875 steps vs baseline 0.979874 at 3409. The mixer runs
as EAGER torch glue between the compiled layer ops -> +62% step time (0.85 vs 0.52 s; local eager was only +7% because
local eager is uniformly slow, hiding the relative cost). +62% -> 45% fewer steps -> undertrained -> hard loss. The -0.007
quality gain is real but drowned by the step penalty.
=> MUDD is viable ONLY if the mixer is FUSED into the kernel path (kill +62% -> ~+7%, like the key-offset fold). Even then
it reaches ~0.975 UNLESS its quality gain SCALES with training length (MUDDFormer's benefit compounds as mixers train;
measured -0.007 at 500 steps is likely an underestimate at the 3400-step budget). DECISIVE TEST (running): local fixed-step
MUDDSCALE OFF vs ON at 1200 steps -> if the gain widened past ~-0.012, a fused mixer at full budget plausibly hits ~0.96
local (0.9762 - ~0.015); if flat, fused-MUDD caps ~0.975 and won't reach 0.96. Build the fused mixer only if the gain scales.

## MUDDFormer: CLOSED — gain SHRINKS with training length (2026-09-22, decisive)
Gain-scaling test (local eager fixed-step, ways=1): delta -0.0070 at 500 steps -> -0.0039 at 1200 steps (ON 1.119527 vs
OFF 1.123438). The advantage FADES as training proceeds (mixers help most early), opposite to the hoped-for growth. At the
full 3400-step budget the gain projects to ~-0.002 or less -> even a fused mixer (+7% step) nets ~break-even; nowhere near
the -0.015 needed for 0.96. MUDDFormer is a small early-training helper, not a path to the leaders. Not worth the fused
kernel. 

## FINAL (2026-09-22): 0.96 local is not reachable with any technique found. v46 (0.9762) is the ceiling of this line.
Throughput: all levers closed (kernels at the floor; compiler 5x slower; comm/optimizer irreducible; GpSimd hint + coalescing
unimplemented). Quality: multi-hop, flow, DeltaNet, TOP, distillation, MUDDFormer all fade at fixed-time. Recon (x2): we
already run the frontier technique stack; no public disclosure of the leaders' method (gallery empty until Sep 30). The
leaders' ~0.956 remains unexplained and unreachable with current knowledge.
