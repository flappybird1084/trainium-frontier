# Flow-idea scoreboard (local GPU, depth-8, 500 steps) vs FF_BASE 1.197446
| id | idea | val_bpb | vs base | status |
|---|---|---|---|---|
| FF_BASE | no flow (baseline) | 1.197446 | - | ref |
| B1_logit_res | #2 logit correction, zero-init, LM-loss | 1.201948 | +0.0045 | worse |
| B1_logit_rf | #1 logit rectified-flow (boost5 lr.01 aux1, hot) | diverged@60 | - | crash |
| B1b_logit_rf_s | #1 logit rectified-flow (boost2 lr.003 aux.3, cool) | 1.197333 | -0.0001 | WASH |
| B1b_draft | #3 draft-refine (CE floor aux.5) | 1.199956 | +0.0025 | worse |
| B2_stack | #8 stack recurrence (P=3 passes, DB2) | 1.198520 | +0.0011 | worse |
| B2_mlpflow | #19 MLP-as-flow (P=3 extra steps, ckpt) | 1.195776 | -0.0017 | small WIN |
| B2_mhattn | #14 multi-hop attention (P=3 hops, DB2) | 1.190852 | -0.0066 | **WIN** |
| B3_mhattn5 | #14 multi-hop attention (P=5 hops, DB2) | 1.190295 | -0.0072 | **WIN** |
| B3_predenoise | #6 prehead denoising aux (noise.1 aux1) | 1.195039 | -0.0024 | small WIN |
| B3_latent | #10 latent scratchpad (Z=256) | 1.199602 | +0.0022 | worse |
| B4_base_db2 | CONTROL: baseline at DB2 | 1.198017 | +0.0006 vs DB4 base | control (~=base) |
| B4_mhattn7 | #14 multi-hop 7 hops (DB2) | ~1.190 (eval OOM@end; train loss=mhattn5) | -0.008 | saturates |
| B5_mhattn_mlp | #14+#19 multi-hop + MLP-flow (P=3, DB2) | 1.189262 | -0.0088 | **BEST** |
| B5_mhattn2 | #14 multi-hop 2 hops (DB2) | 1.190810 | -0.0072 | win (cheap) |
| B6_triple | #14+#19+#6 all three (DB2) | diverged@100 | - | unstable (mlpflow+predenoise together) |
| B6_mhattn_pre | #14+#6 multi-hop + prehead denoise (DB2) | 1.188719 | -0.0093 | **BEST (stable)** |

## DEPTH-12 TRANSFER (500 steps, DB2)
| id | idea | val_bpb | vs d12 base | status |
|---|---|---|---|---|
| B7_d12_base | depth-12 baseline (no flow) | 1.188775 | - | ref |
| B7b_d12_champ | #14+#6 multi-hop + prehead denoise | 1.180481 | -0.0083 | **WIN TRANSFERS** |
Note: training loss looked TIED (training samples 1..P hops); the EVAL win comes from using MAX hops at inference -- the thesis.
| B8_d12_hop1 | #14+#6 depth-12, FIXED 1 hop (train=eval=1) | 1.181265 | -0.0075 | 1 hop = 90% of 3-hop gain -> cheapest config keeps the win |
| B9_d12_hop1_every2 | #14+#6 depth-12, 1 hop on EVERY OTHER block | 1.181823 | -0.0070 | 93% of every-block gain at half the hop cost -> box config |

## TRN2 BOX (600 s screens, PROFILE=1, vehicle sha 276784aa931c0a27 = hop w/ detached cached weights)
| id | config | step dt | val_bpb | note |
|---|---|---|---|---|
| REF_FLOW0 | v40 config, flow off, DB16 | 0.53 s | 1.052904 | same-day control (edited vehicle reproduces v40 path) |
| S_FLOW_HOP1 | hop every block, DB16 | - | OOM@step0 | device HBM: hop activations x12 blocks |
| S_FLOW_HOP1_DB8 | hop every block, DB8, old sha (stack-grad scatter) | 4.04 s | 1.534304 | 0 errors, ~150 steps: 7.6x slow = full-stack zero-grad per stacked-weight use x36/step |
| REF_DB8 | flow off, DB8 | 0.57 s | 1.059170 | DB8 alone costs ~7.5% step + ~0.006 at 600 s |
| S_FLOW_HOPE2_DB8 | hop every OTHER block, DB8, detached weights | ? | ? | running: must land near 0.57 s and beat 1.059170 |
| S_FLOW_HOPE2_DB8 | hop every other block, DB8, sha 276784aa (detached via cached() dict lookup) | 14.5 s | 2.084242 | ~40 steps; WORSE than 4.0 s: cache/dict lookup in trace -> more breaks (hypothesis) |
| S_FLOW_HOPE2_V3 | same config, sha 68b8f357 (explicit bf16 hop_w tensors, no lookups in trace) + TORCH_LOGS=graph_breaks | ? | ? | running: step time + exact break list |
| S_FLOW_HOPE2_V3 | sha 68b8f357 explicit bf16 hop_w | - | compile FAIL | NCC_INKI016: attention_bwd got q,k bf16 but v fp32 (vv + fp32-scaled ve promotes) |
| S_FLOW_HOPE2_V4 | sha 2138a34f = V3 + v cast to bf16 in hop | ? | ? | running with TORCH_LOGS=graph_breaks |
| B10_d12_hopdet_every2 | #14+#6 depth-12, 1 hop every other block, hop weights DETACHED (box semantics) | 1.184084 | -0.0047 | detaching hop wgrad keeps ~67% of B9's -0.0070 |
| S_FLOW_HOPE2_V4 | sha 2138a34f (bf16 hop_w + v cast) + TORCH_LOGS | 1.57 s | (600 s, ~380 steps) | NO dynamo breaks: 1.57 s is the true cost of compiled EAGER attention glue (~170 ms/hop x 6) -> dead end |
| S_FLOW_PRE_DB16 | prehead denoise ONLY, DB16 (fits), sha c8e423c6 | ? | ? | cheap win candidate: must beat REF_FLOW0 1.052904 |
| S_FLOW_HOPE2_V5 | hop as FUSED custom op tf_nki::hop_fwd/bwd (pre-attn kernel -> vendor attn -> out-proj), every other block, DB8 | ? | ? | the real port |
| S_FLOW_PRE_DB16 (result) | prehead denoise only, DB16 | 0.55 s | 1.058766 | +0.006 vs REF_FLOW0 1.052904 -> NOT a win at 600 s |
| B11_d12_pre_only | #6 prehead denoise ONLY, depth-12 | 1.187640 | -0.0011 | shrinks with depth; cannot pay its ~4% step cost at fixed time -> drop prehead, multi-hop is the lever |
| S_FLOW_HOPE2_V5 (result) | fused hop op, every other block, DB8, +prehead, TRAIN 1 hop / EVAL 2 hops | 0.69 s | 1.077842 | +0.019 vs REF_DB8: far beyond the step-count cost; training loss was AHEAD of ref -> eval-graph mismatch suspected (never trained with 2 hops; local wins all had eval=train hops) |
| S_FLOW_HOPE2_DB16 | fused hop op, every other block, DB16 (train 1 / eval 2, mismatched) | 0.61 s | (pending, informational) | FITS at DB16; +15% vs v40 0.53 -> DB8 penalty gone |
| S_FLOW_HOPE2_DB16 (result) | fused hop, every other block, DB16, train 1 / eval 2 | 0.62 s | 1.154657 | +0.10 vs REF 1.052904: eval pass BROKEN for train1/eval2 (training loss normal). Discriminate: S_HOP1E2_DB16 (eval=train=1) vs local B12 |
| S_HOP1E2_DB16 | fused hop, every other block, DB16, eval=train=1 hop (matched) | 0.62 s | 1.064928 | +0.012 vs REF 1.052904: mismatch explained 0.09 of the 0.10, but matched hops still LOSE at 600 s. Gaps vs local winner: no ve term, no hop wgrad |
| B12_d12_train1_eval2 | LOCAL eager, detached, every other block, train 1 hop / eval 2 | - | 1.183768 | -0.0050 = same as eval=train=1 (B10 -0.0047): the mismatch is HARMLESS in eager -> the box's 2-hop eval graph of the fused op is buggy; keep eval=train on the box |
| S_HOP1E2F_DB16 | FAITHFUL fused hop (ve term + hop wgrad), every other block, DB16, eval=train=1, sha ca23f2d4f3216a4d | ? | ? | queued |
| S_HOP1E2F_DB16 (result) | FAITHFUL fused hop (ve + wgrad), every other block, DB16, eval=train=1 | 0.62 s | 1.179141 | +0.126 vs REF: BEST training loss (2.929@800) but eval broken -> train/eval discrepancy in the fused op itself (not hop-count mismatch) |
| S_HOP1E3_DB16 | every 3rd block, eval=train=1 | - | KILLED | launched after the faithful (broken) build was shipped -> uninformative; killed to free the device |
| DIAG_HOPCHK | eager COMPILE=0, op-vs-eager hop check at step 0 | - | vacuous | both sides exactly 0: out-proj is zero-initialized at step 0 -> compare on TRAINED weights instead (EVALF_CHECK) |
| EVALF_HOP / EVALF_NOHOP / EVALF_CHECK | re-eval of the faithful checkpoint out_S_HOP1E2F_DB16/final.pt: hop on / hop off / eager op-vs-eager check | - | pending | discriminates "weights fine, eval hop application broken" vs "weights trained into a hop-dependent regime" |
| EVALF_* (standalone eval-public) | re-eval of the faithful checkpoint | - | HUNG 2.5 h | standalone prepare.py eval-public hangs on the box with the kernel env; killed. Replaced by training runs with FLOW_EVAL_HOPS |
| RUN_EVAL0 | faithful hop training (same as S_HOP1E2F_DB16), EVAL with hop OFF (FLOW_EVAL_HOPS=0), sha c8725b29 | ? | ? | discriminator: sane (~1.06) = weights fine / eval hop application wrong; bad = weights hop-dependent |
| RUN_CHK | eager COMPILE=0, op-vs-eager HOP_CHECK after ~100 trained steps | - | ? | numeric forward check on trained weights |
| RUN_EVAL0 (result) | faithful training, eval with hop OFF | 0.62 s | 1.153573 | still +0.10: the WEIGHTS score badly at eval with or without the hop -> train-fwd vs eval-fwd difference on same weights; prime suspect = EMA blend (35%) on a sharper hop-trained trajectory |
| RUN_NOEMA | faithful training, eval hop 1, EMA_BLEND=0 (live weights at eval) | ? | ? | EMA-blend hypothesis: sane -> blend is the culprit (fix: skip EMA for hop-trained stacks / blend less) |
| RUN_CHK (result) | eager op-vs-eager HOP_CHECK on trained weights (layer 10, 6 calls) | - | PASS | max|diff| 0.25-0.31 vs ref max ~22; mean 0.0078 vs 0.70 (~1% = bf16 precision) -> fused op FORWARD IS CORRECT; discrepancy is a training-regime effect (EMA blend?) |
| RUN_NOEMA | faithful, EMA_BLEND=0 | - | CRASH@compile (49 s) | tracer: pre_attn_fwd_h "argument not supplied" at first compiled step; ran the new bisect-knob source (sha 80eda3a5). Relaunched as RUN_NOEMA2 to test source-vs-transient. NOTE: RUN_EVAL0 ran EMA_BLEND=1.0 (default; C string lacked it), not 0.35 |
| RUN_NOEMA2 | faithful, EMA_BLEND=0 (relaunch, same source 80eda3a5) | ? | running | passed the 49 s crash point with 0 tracer errors -> RUN_NOEMA's crash was TRANSIENT (compile-cache race right after the source swap) |
| RUN_NOEMA2 (result) | faithful hop, EMA_BLEND=0 (live weights at eval) | 0.62 s | 1.063662 | SANE: the EMA weight blend was the eval killer (1.179 -> 1.064). vs REF+EMA 1.052904 = +0.011, but not apples-to-apples |
| REF_NOEMA / HOP_EMA01 | ref with EMA off; hop with blend 0.1 | ? | queued | EMA's own worth at 600 s; does a gentle blend survive with the hop |
| REF_NOEMA (result) | flow off, EMA_BLEND=0 | 0.52 s | 1.053373 | EMA blend worth only ~0.0005 at 600 s (vs 1.052904). FAIR PAIR: hop/noEMA 1.063662 vs ref/noEMA 1.053373 = +0.010 -> hop LOSES at fixed time at 600 s (ahead on train loss at equal steps, but 18% fewer steps) |
| HOP_EMA01 (result) | faithful hop, EMA_BLEND=0.1 | 0.62 s | 1.063084 | blend 0.1 survives (no collapse), ~= blend 0 -> gate setting |
| T58_gate_hop | FULL 1800 s gate: faithful fused hop, every other block, DB16, EMA_BLEND=0.1, PROFILE=0 | ? | running | vs v40 0.985161 |
| T58_gate_hop (result) | FULL 1800 s gate, faithful fused hop, every other block, DB16, EMA_BLEND=0.1 | 0.62 s | **0.992602** | **+0.0074 vs v40 0.985161 -> DISCARD.** Port correct; fixed-time budget eats the fixed-step gain |
| HOP_E3_NOEMA (result) | hop every 3rd block, EMA_BLEND=0.1, 600 s | 0.61 s | 1.066940 | worse than every-other (1.0631) and ref (1.0534): fewer hop blocks = less gain, ~same cost. Closes the "fewer blocks" route |

## COMPILER-FLAG SWEEP (v40 config, flow off, PROFILE=0, DB16, 600 s; fresh compile-cache dir per run)
| id | NEURON_CC_FLAGS | first-step delay | step dt | val_bpb | note |
|---|---|---|---|---|---|
| FL_REF | (none) | 326 s (cache hit) | 0.52 s | 1.052747 | same-day reference; noise floor vs prior refs ~0.001 |
| FL_TRANSF | --model-type=transformer | 681 s (cold) | 0.52 s | 1.119822 | +0.067 WORSE with IDENTICAL step dt, ~same steps (1120 vs 1131) and a slightly BETTER final train loss (2.808 vs 2.831): the flag miscompiles the EVAL graph (deterministic -> causality still passes). DISCARD. HAZARD: flags can break eval silently; only the score is trustworthy |
| FL_O3 | -O3 (via CC_FLAGS) | 935 s (cold) | 0.52 s | 1.118588 | +0.066: SAME broken-eval signature as FL_TRANSF -> systematic with NEURON_CC_FLAGS set (also recompiles the NKI custom ops), not flag-specific. Step dt unchanged either way |
| FL_O1 | -O1 | - | - | killed | no-op (default is -O1); auto-killed by watcher |
| FL_MPA | --enable-mixed-precision-accumulation | 606 s (cold) | 0.52 s | 1.120903 (broken eval, 3rd of 3 under CC_FLAGS) | loss bit-identical to ref -> no effect on this graph |
| FL_SATINF | --enable-saturate-infinity | 588 s (cold) | 0.52 s | 1.121086 (broken eval, 4 of 4) | loss bit-identical -> no effect |
| FL_FCS | --enable-fast-context-switch | 571 s (cold) | 0.52 s | 1.120434 (broken eval, 5 of 5) | the one runtime-side flag; no launch-latency effect |
| FL_IDC | --enable-implicit-downcast | 690 s (cold) | 0.52 s | 1.120986 (broken eval, 6 of 6) | loss bit-identical -> no effect |
Notes: FL_REF's 326 s first-step delay was a CACHE HIT on REF_NOEMA's graph (no new NEFF written), not a cold compile; flags are
hashed into the NEFF cache key (CompilationCacheKey.cpp: hash(base_key || compiler_config)) so every flagged run cold-compiles
(compile time is outside the 600 s budget). Cache-dir override is TORCH_NEURONX_NEFF_CACHE_DIR (NEURON_COMPILE_CACHE_URL is inert).
Lever 2 (kernel-schedule grid) inventory: vendor attention made _AB_KB/_AF_QG dead; _FMAX=512 is the matmul free-dim limit;
the only live tunable is _HVC (head vocab chunk, 512) with 256 as the sole valid alternative -> now env NEURON_COMPETITION_R1_HVC.
| FL_O2c | NEURON_COMPILER_OPT_LEVEL=-O2 (canonical) | recompiled (217 NEFFs) | 0.52 s | 1.054609 | SANE eval (within noise of ref); loss bit-identical -> -O2 = no-op for step time. Proves the eval corruption is specific to NEURON_CC_FLAGS, and that eval_seconds ~28 s is just a cold eval-graph compile, not a brokenness fingerprint |
| FL_O3c | NEURON_COMPILER_OPT_LEVEL=-O3 (canonical) | recompiled (193 NEFFs) | 0.53 s | 1.119461 | BROKEN eval via the canonical route too -> -O3 itself corrupts eval (-O2 canonical was sane). Step dt unchanged |
Note: torch_neuronx passes optimization_level = NEURON_COMPILER_OPT_LEVEL or "-O1" by default (NeuronCompiler.cpp) -> FL_O1 is a
no-op (kill on launch); the real upside is -O2/-O3.
**LEVER 1 VERDICT: 9 compiler configurations (7 flags via NEURON_CC_FLAGS incl. -O3, plus -O2/-O3 via NEURON_COMPILER_OPT_LEVEL):
every single one ran at 0.52-0.53 s/step = the reference. Zero step-time movement. The step is latency-bound, not
schedule-bound; the compiler's optimizer has nothing left to give this graph. Side finding: -O3 (either route) and every
NEURON_CC_FLAGS setting corrupt the eval-mode graph (~1.12 vs 1.053) while training stays intact; -O2 is clean.**
| KN_HVC256 | head vocab chunk _HVC=256 (lever 2's only live knob) | - | CRASH@compile | head kernel has a 2nd hard-coded 512 (sub-block loop indexes [256,383] in a 256-wide chunk) -> not a free knob; rewrite not worth a sub-% head-kernel tweak. LEVER 2 CLOSED |

## OWN FLASH ATTENTION, Phase A (2026-09-13 evening; v40 config, DB16, PROFILE=0, 600 s; ref FL_REF 1.052747 @0.52 s)
| id | fwd / bwd | result | note |
|---|---|---|---|
| ATT_V4 | v4 strided-load fwd + our bwd (inside the v40 head-major layer op) | NaN early (bad loss) | head-major native layout does NOT fix it: the strided INPUT DMAs themselves race. v4 dead unless sidecar sync helps |
| ATT_V4_SYNC | same + SIDECAR_SYNC=1 | NaN early | sidecar hypothesis REFUTED: strided input DMAs are intrinsically unsafe in-graph (same class as rule #9 outputs). v4 dead |
| ATT_PLAIN | per-tile-load fwd (attn_fwd4_plain) + our strided bwd, OLD per-layer glue | CLEAN, 0.54 s (+4%), 1.056006 on 1088 steps (ref 1131) | correct; the glue cost is the whole deficit (= AFB_B). Hoisted-glue screen decides |
| ATT_PLAIN_HOIST | same kernels + per-step hoisted glue (pack (4,B,t,1) via start/end inputs; raw lse passthrough), sha e641750822f35107 | CLEAN, 0.53 s (vs 0.54 unhoisted, 0.52 ref); loss bit-identical to ATT_PLAIN (pack path correct); score pending | hoist recovered ~1% of the 4%; remaining per-layer glue (sink expand x24, dsink reduce x12, zeros x12) + in-graph kernel scheduling ~= the bwd kernel's 9 ms/step saving -> PARITY, not a win |
| ATT_PLAIN_HOIST (result) | | 1.125406 on 1094 steps | EVAL CORRUPTED (the ~1.12 signature) with bit-identical training -> the hoisted pack is right in the train graph, wrong in the eval graph; same unexplained class as -O3 / CC_FLAGS. Unhoisted plain+ours stays clean (+4%). Verdict for own attention: parity at best; keep the vendor kernels |

## LEVER A / LEVER 0 GATES (full 1800 s, v40 config, PROFILE=0; 2026-09-14)
| id | change | val_bpb | vs A_REF | note |
|---|---|---|---|---|
| A_REF | none (v40 config, vehicle) | 0.985741 | - | v40 gate was 0.985161 -> 1800 s noise band ~0.001 |
| A_PACK8 | PACK=8 (rows of 8193 reshaped) | CRASH | - | OOB in the first sidecar.prepare (deterministic); superseded by A_STREAM |
| A_STREAM | PACK_MODE=stream (eval-style rows; 100% coverage, 99.8% mid-doc) | 0.987395 | +0.0017 | LOSS. 40% more unique tokens + eval-matched rows did not help; stream-cut rows are a slightly worse training signal than full-doc best-fit rows. Lever 0 CLOSED |
| A_PACK4 | PACK=4 (rows of 4097 reshaped; 87% coverage, 75% mid-doc rows) | **0.981426** | **-0.0043** | WIN, beyond the ~0.001 band; -0.0037 vs the v40 gate. Repeat (A_PACK4b) + PACK2 queued |
| A_WD07..A_EMAS075 | 8 scalar re-tunes | queued | | lever A |
| B13_d12_pack4 (local, 500 fixed steps, DB4) | PACK=4 vs B7_d12_base 1.188775 | 1.188717 | -0.0001 | WASH at fixed steps (no 2nd epoch at 65M tokens; the eval-like rows did not lower eval loss) -> the M4 "BOS-less penalty" is mostly inherent (little context / harder long-doc content), not a trainable mismatch. What remains of lever 0 is COVERAGE (28% more unique tokens), which only shows past epoch 1 -> A_STREAM full gate decides |
| A_WD07 | WARMDOWN 0.9 -> 0.7 | 0.985861 | +0.0001 | wash |
| A_PACK4b | PACK=4 repeat | **0.981073** | **-0.0047** | CONFIRMED (two gates 0.9811 / 0.9814 vs ref 0.9857) |
| T59_gate | CLEAN v42 = v40 + PACK=4 (sha 95dfedd9cc359b97) | **0.981197** | -0.0045 | **NEW BEST -> submit/train.py**; causality passed |
| A_PACK2 | PACK=2 | 0.981913 | -0.0038 | PACK=2 captures ~80% of the PACK=4 gain; PACK=4 stays best (PACK=8 crashes the sidecar) |
| P4_* (first attempt) | 7 scalar gates on PACK=4 | - | malformed queue lines (field order), all exited in 35 s | relaunched as queue_A7b |
| P4_MLR085 | PACK=4 + MATRIX_LR_SCALE 0.7 -> 0.85 | 0.981396 | +0.0002 vs T59 | wash |
| P4_MLR055 | PACK=4 + MATRIX_LR_SCALE 0.7 -> 0.55 | 0.981077 | -0.0001 vs T59 | wash (matrix LR flat 0.55-0.85) |
| P4_ELR45 | PACK=4 + EMBED_LR_SCALE 6 -> 4.5 | 0.980139 | -0.0011 vs T59 | marginal win (band ~0.001); P4_ELR8 decides the direction |
| (repeat-regime.md gate plan) | appended to queue_A7b: P4_WD02, P4_WD03 (WD 0.2 / 0.3 on PACK=4; 0.1 = T59 already). Item 2 "paired same-seed repeat" needs no run: every run uses manual_seed(42) and a deterministic rank-strided data order, so PACK4 / PACK4b / T59 (0.981426 / 0.981073 / 0.981197) ARE a paired triple -> residual spread 0.00035 | | | items 3-5 (MIR aux loss, parallel residuals, self-distillation) need code before they can be queued |
| (repeat-regime idea 1) | MIR implemented: MIR_FRAC / MIR_ROWS, loader-side, single forward, BOS kept, targets untouched (unit test + CPU smoke pass); vehicle sha 899fdc6ca4aa811e shipped | | | gates appended: P4_MIR005 / P4_MIR010 / P4_MIR015 (MIR_ROWS 0.5, PACK=4, WD 0.1) |
| (repeat-regime idea 3) | PAR_RESID implemented in the EAGER block path (attn + MLP both read the block input); local fixed-step quality screen B14_d12_parresid running vs 1.188775 | | | the Trn2 layer op is sequential by construction: a step-time test needs a kernel variant, so quality is checked first |
| P4_ELR8 | PACK=4 + EMBED_LR 8.0 | LOST | - | trained all 3400 steps (loss 2.755, healthy) then the END-OF-RUN eval compile failed with the tracer "argument not supplied" error: I shipped the vehicle (MIR/PAR_RESID) mid-run; NKI's tracer re-reads kernel source from disk via inspect at trace time, so a changed file corrupts any later trace in a running process. Same mechanism as the RUN_NOEMA "transient" crash. Re-queued as P4_ELR8b |
| B14_d12_parresid (local, 500 fixed steps) | PAR_RESID=1 (eager) vs 1.188775 | 1.306967 | +0.118 | training tracked the baseline to step 300 (3.42 vs 3.37) then rose to 3.84 by step 490: late-run INSTABILITY, not a train/eval bug. Parallel residuals are unstable/worse at fixed steps with this optimizer config -> the Trn2 layer kernel is NOT worth building unless the overlap test (PARRES_T, queued) shows a large gain AND a retune stabilizes it |
| P4_EMAB05 | PACK=4 + EMA_BLEND 0.35 -> 0.5 | 0.980991 | -0.0002 vs T59 | wash (band ~0.0004) |
| B15_d12_mir010 (local, 500 fixed steps, 1 epoch) | MIR_FRAC 0.10, MIR_ROWS 0.5 vs 1.188775 | 1.202383 | +0.0136 | MIR costs a lot at a single-epoch fixed-step horizon (as regularization does); the repeat-regime claim is that it pays past epoch 1 -> the three full-length box gates decide; a +0.014 fixed-step deficit is a high bar |
| P4_EMAS075 | PACK=4 + EMA_START 0.85 -> 0.75 | 0.981511 | +0.0003 vs T59 | wash |
| B16_d12_base_rep (local, same config/seed as B7_d12_base) | baseline repeat | 1.188077 | -0.0007 vs 1.188775 | LOCAL fixed-step band ~0.0007 (GPU nondeterminism; box same-config spread is 0.00035). Re-read: B13 PACK=4 (-0.0001) wash; multi-hop (-0.005..-0.008) real; the depth-8 mlpflow (-0.0017) / predenoise (-0.0024) "small wins" were 2-3 bands: marginal, as suspected |
| P4_WD08 | PACK=4 + WARMDOWN 0.9 -> 0.8 | 0.981819 | +0.0006 vs T59 | slightly worse (1.5 bands); 0.9 stays |
| B17_d12_parresid_lr045 (local, 500 fixed steps) | PAR_RESID=1 + MATRIX_LR_SCALE 0.45 vs 1.188775 | 1.193635 | +0.0049 | STABLE (step 490 loss 3.293, no late rise): the B14 blow-up was LR-driven. A stable parallel block is ~0.005 worse at equal steps = ~16% of step time at the exchange rate, above the ~10% overlap upper bound -> IDEA 3 CLOSED on arithmetic (PARRES_T job kept: 5 min, general fact) |
| P4_ELR35 | PACK=4 + EMBED_LR_SCALE 6 -> 3.5 | 0.985104 | +0.0039 vs T59 | clearly worse: the embed-LR curve on PACK=4 is 3.5: 0.9851 / 4.5: 0.9801 / 6.0: 0.9812; the 4.5 repeat (running) decides whether -0.0011 is real |
| P4_ELR45b | PACK=4 + EMBED_LR_SCALE 4.5 (repeat) | **0.980029** | **-0.0012** vs T59 | CONFIRMED with P4_ELR45 0.980139 (two paired gates, ~3 bands) -> v43 = v42 + embed LR x4.5, clean-file gate T60 queued |
| T60_gate | CLEAN v43 = v42 + EMBED_LR x4.5 (sha d1b78be9a0285c8d), queued last in queue_A7b | ? | | expect ~0.9801 (P4_ELR45 0.980139 / P4_ELR45b 0.980029); promote if <= ~0.9805 with causality passed |
| P4_WD02 | PACK=4 + WD 0.1 -> 0.2 | 0.985687 | +0.0045 vs T59 | clearly WORSE at full length (repeat-regime.md idea 1's precondition "strong WD" fails; WD 0.3 running, expected worse) |
| P4_WD03 | PACK=4 + WD 0.3 | 0.992145 | +0.0109 vs T59 | monotonically worse: WD 0.1 (0.9812) / 0.2 (0.9857) / 0.3 (0.9921). Strong WD is harmful at full length on this line -> repeat-regime idea 1's "strong WD" precondition is FALSE here |
| P4_MIR005 | PACK=4 + MIR_FRAC 0.05 (MIR_ROWS 0.5) | 0.986213 | +0.0050 vs T59 | worse at full length too (local fixed-step: MIR 0.10 +0.014). 0.10/0.15 running, expected worse |
| P4_MIR010 | PACK=4 + MIR_FRAC 0.10 | 0.990758 | +0.0096 vs T59 | monotone: 0.05 +0.005, 0.10 +0.010. Idea 1 CLOSED (0.15 runs out the queue) |
| P4_MIR015 | PACK=4 + MIR_FRAC 0.15 | 0.995062 | +0.0139 vs T59 | completes the monotone curve (0.05/0.10/0.15 = +0.005/+0.010/+0.014) |
| P4_ELR8b | PACK=4 + EMBED_LR_SCALE 8.0 | 0.981913 | +0.0007 vs T59 | slightly worse: embed-LR curve on PACK=4 = 3.5: 0.9851 / 4.5: 0.9801 / 6: 0.9812 / 8: 0.9819 -> 4.5 is the optimum |
| PARRES_T (device job) | nested vendor _attention_cte + MLP body in ONE kernel | (a) attn 2.19 ms, (b) MLP 1.48 ms, (c) fused 3.77 ms | - | bit-exact fusion works, but NO overlap (3.77 >= 3.67 serial): the NKI scheduler executes nested kernel bodies serially. Intra-kernel co-scheduling is not a lever on this stack |
| T60_gate | CLEAN v43 = v42 + EMBED_LR x4.5 (sha d1b78be9a0285c8d) | **0.979941** | -0.0013 vs T59 | **NEW BEST -> submit/train.py**; causality passed |
