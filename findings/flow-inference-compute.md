# Flow-matching / inference-compute sweep (2026-09-12)

User thesis: "more compute for inference is a-ok." Tested ~20 placements of flow-matching / iterative refinement in the
model, all on the **local CUDA GPU** via `gpu_gdn.sh` at **depth 8, 500 steps**, compared to the no-flow baseline
**FF_BASE = 1.197446** (matched DB=2 baseline = 1.198017; DB is a memory/speed knob only, total batch held by grad-accum,
so DB=2 and DB=4 baselines agree to ~0.0006). Vehicle: `submission_raw/train3_submission.py`. Knobs added:
`FLOW_SITE` (now comma-separated multi-site), `FLOW_MODE` (refine|rf|denoise|draft), `FLOW_STEPS`, `FLOW_LR`, `FLOW_RANK`,
`FLOW_AUX`, `FLOW_NOISE`, `FLOW_BOOST`, `FLOW_STARTS`. Eval contract: `model(x)` must return logits; eval has a hard
20-min wall (`EVAL_TIMEOUT_SECONDS`), so unbounded per-token integration is out, but a large multiple of the fast head is fine.

## Verdict
**Inference-compute refinement works, but only when it re-uses the real attention/MLP weights - not as a separate velocity
MLP on logits, and not as whole-stack feedback.** The winners are (nearly) zero-new-parameter and stack.

| idea | # | val_bpb | vs matched base | note |
|---|---|---|---|---|
| **multi-hop attention + prehead denoising** | 14+6 | **1.188719** | **-0.0093** | best, stable; gains stack |
| multi-hop attention + MLP-as-flow | 14+19 | 1.189262 | -0.0088 | gains stack |
| all three (mhattn+mlpflow+prehead) | 14+19+6 | diverged@100 | - | unstable together |
| multi-hop attention, 5 hops | 14 | 1.190295 | -0.0077 | saturates ~5 |
| multi-hop attention, 3 hops | 14 | 1.190852 | -0.0072 | |
| multi-hop attention, 2 hops | 14 | 1.190810 | -0.0072 | ~full gain, cheap |
| prehead denoising (denoise aux) | 6 | 1.195039 | -0.0024 | needs the denoise target; deterministic prehead was the WORST earlier |
| MLP-as-flow (extra weight-tied MLP steps /P) | 19 | 1.195776 | -0.0017 | |
| logit rectified-flow (cool: boost2 lr.003 aux.3) | 1 | 1.197333 | -0.0001 | wash |
| logit correction (zero-init, LM-loss) | 2 | 1.201948 | +0.0045 | worse |
| logit draft-refine (CE floor) | 3 | 1.199956 | +0.0025 | worse |
| logit rectified-flow (hot: boost5 lr.01 aux1) | 1 | diverged@60 | - | too-hot aux |
| stack recurrence (P passes, feedback blend) | 8 | 1.198520 | +0.0011 | worse |
| latent scratchpad (Z=256) | 10 | 1.199602 | +0.0022 | worse; adds competing params |

## What multi-hop attention (#14) is
In the eager block (CUDA path), after the normal `x = x + attn(norm(x))`, run P extra weight-tied attention reads, each
rebuilding the query from the updated x: `for _ in range(P): x += attn(norm(x))/P`. Training samples P in 1..FLOW_STEPS,
eval uses max. Zero new parameters. `FLOW_SITE=mhattn`, memory-heavy in training (stores each hop's activations) so run at
`DEVICE_BATCH=2`. The gain is captured by just 2 extra reads and saturates by ~5.

## Why the others failed
- **Output-side logit flow (#1,#2,#3):** a velocity MLP on the 8192-logit vector just competes with the LM head; wash at
  best, and the aggressive rectified-flow target (boost 5, lr .01) diverges (the velocity blows up the integrated logits).
- **Stack recurrence (#8) / latent scratchpad (#10):** whole-stack feedback and extra latent state don't help this small
  model; latent adds parameters that compete.
- **Lesson:** the value is in spending more compute through the *existing* attention/MLP transforms, not in bolting on new
  refiners or refining the output distribution.

## Memory / running notes
- Re-applying sublayers with backprop multiplies stored activations by P -> OOM at DB=4. Fix: DB=2 (math identical, grad
  accum holds total batch) for multi-hop; gradient-checkpoint the MLP-flow steps (`torch.utils.checkpoint`) so it fits at DB=4.
- The 24 GB `ulimit -v` in `gpu_gdn.sh` caps host virtual memory (incl. CUDA mappings); 7-hop multi-hop OOM'd host memory at
  the eval step (training was fine). Keep hops <= 5.
- Each screen ~25-55 min (DB=2 multi-hop slower). Runs driven by `$S/scratchpad/flow_batch.sh` (isolated subprocess per run,
  a crash of one does not stop the rest) + a persistent Monitor (step-10 loss, divergence, final bpb). Scoreboard:
  `runs/gpu/_flow_scoreboard.md`.

## TRANSFER TO DEPTH 12 - CONFIRMED WIN (2026-09-12)
depth-12, 500 steps, DB2: baseline **1.188775** -> multi-hop + prehead denoise **1.180481 = -0.0083**. The gain transfers.
**Key subtlety:** at depth 12 the *training* loss looked TIED (base 3.194 vs champ 3.184 at step 490) so an early read
looked like no transfer -- but training samples 1..P hops per step while **eval uses MAX P hops**, so the training loss
understates the eval benefit. Always judge these by eval bpb, never training loss. This is precisely the "more inference
compute" payoff: the model is trained cheap (few hops) and evaluated rich (max hops).

## Open / next
- **Port multi-hop attention (+ prehead denoise) to the compiled Trainium path** - this is the real prize now that transfer
  is confirmed. The winning code is EAGER-only; the submission runs the compiled `_NKILayer` path, which bypasses it. Options:
  add the extra attention reads around the vendor attention op, or in the block glue outside the fused layer kernel.
- Re-check the exact hop count / prehead-denoise weight at depth 12 (2 vs 3 hops; the depth-8 curve saturated ~2-5).
- Memory: depth-12 multi-hop needs DB2 + raised host vmem cap (`R1_VMEM_KB` in gpu_gdn.sh) for the eval pass.
- **Trainium:** the winning code is in the EAGER block path only; the submission runs the compiled `_NKILayer` path, which
  bypasses it. Landing multi-hop in the compiled/Trainium path is the real (larger) work if transfer holds.
- Untried from the 20 (deprioritized after the pattern was clear): attention-weight flow (#12), KV refine (#15),
  bigram-anchored output flow (#5), x0 transport (#9), block-ODE (#7), cascade (#20), consistency head (#17), CNF exact
  likelihood (#4, too heavy for the 20-min eval wall), multi-start/self-ensemble/adaptive-step (#16/#18/#11, eval toggles on
  output flow which was a wash).

## Trn2 port log (2026-09-12 evening)
Goal: land multi-hop attention (+ prehead denoise) in the compiled v40 path (`LAYER=op`, torch.compile backend=neuron).
- **v1 - hop as compiled eager glue after the fused layer op** (`self.attn(norm(xo), ...)` inside the op branch):
  DB16 -> device OOM at step 0 (hop activations x12 blocks); DB8 fits but **4.0 s/step vs 0.53** (REF_DB8 = 0.57 s: DB8 alone
  costs ~7.5%). Detaching the hop weights via `cached()` dict lookups made it **14.5 s** (lookups/indexing inside the trace).
  Passing explicit bf16 weight tensors + casting v to bf16: **1.57 s, with `TORCH_LOGS=graph_breaks` showing NO breaks** ->
  1.57 s is the genuine cost of compiler-generated eager attention (~170 ms per hop vs ~44 ms for a whole fused layer). Dead end.
- **Lessons:** (1) plain autograd through a stacked-weight slice `stack[idx]` allocates a full (n,out,in) zero grad per use
  under compile (docstring of `_StackSliceLinear`); (2) vendor attention kernels reject mixed q/k/v dtypes
  (`NCC_INKI016 "Got multiple input dtypes"`): `vv + ve` promotes to fp32 because ve is fp32-scaled -> cast v; (3) `trn2_run.sh`
  REFUSES (prints the owning pids and exits) when the device is busy - it does not wait; use `trn2_queue.sh`, which waits;
  (4) `pkill -f`/`pgrep -f` from an ssh one-liner matches the ssh shell's own cmdline (self-kill / false busy) -> use `[p]attern`.
- **v2 - hop as a fused custom op** `tf_nki::hop_fwd/hop_bwd` defined inside `build_layer_ops` (shares pre_kh / attn_fwd /
  attn_bwd / pre_bwd_k closures): fwd = pre-attn kernel (scal=[1,0,0,0], raw xo in) -> key-offset glue -> vendor attention ->
  `F.linear(y2, wo)`; bwd = out-proj dgrad -> attention_bwd -> pre_bwd_k with the pass-through grad zeroed (the caller's
  residual add supplies it) and wgrads discarded (detached weights). Registered with `register_autograd`. Vehicle sha
  c8e423c67fa43378. Screen S_FLOW_HOPE2_V5 (every other block, DB8) pending.
- **Local sizing of the box semantics (depth 12, 500 steps, base 1.188775):** 1 hop every block 1.181265 (-0.0075); every other
  block 1.181823 (-0.0070); every other block with DETACHED hop weights 1.184084 (-0.0047). Prehead-only sizing pending (B11).
- **Fixed-time arithmetic:** ~1% step time ~ 0.0003 bpb. DB8 costs ~7.5% (~0.002). So the fused hop must add well under ~10%
  step time for the -0.0047 (detached, every other block) to net positive; prehead-only is the near-free fallback.

### Fused hop op results (2026-09-12 late evening)
- `tf_nki::hop_fwd/hop_bwd` works: **0.69 s/step at DB8**, **0.61-0.62 s at DB16 (fits!)** vs v40 0.53 -> every-other-block hop
  costs ~+15% at DB16 (~20 ms per hop, kernel speed). Prehead denoise on the box is a LOSS (+0.006 at 600 s; only -0.0011
  locally at depth 12) -> dropped.
- **Eval-graph bug for eval hops > train hops on the box:** train 1 hop / eval 2 hops = **1.154657 (+0.10)** while the SAME
  training run evaluated with 1 hop = 1.064928; locally (eager) the same mismatch is harmless (B12 1.183768 vs B10 1.184084).
  The 2-hop eval graph of the fused op is a valid, deterministic (causality passes) but wrong model. Not diagnosed; **keep
  eval = train hop count on the box.**
- **Matched hops (eval=train=1), every other block, DB16, detached weights, no ve term: 1.064928 vs REF 1.052904 = +0.012 at
  600 s** -> a loss; ~half is the 14% step-count cost, the rest the hop not delivering its local gain in this form.
- Faithful version (ve term + hop weight-grads through the per-step bf16 cache copies, like the layer op) = sha
  ca23f2d4f3216a4d, screen S_HOP1E2F_DB16 pending; every-3rd-block S_HOP1E3_DB16 pending. If neither beats 1.052904 at 600 s,
  the verdict is: multi-hop attention wins at FIXED STEPS (-0.005..-0.008 locally) but does not pay under Trn2's FIXED-TIME
  budget at ~20 ms/hop; a full 1800 s gate only if a 600 s screen wins.
- **Gate gotcha (2):** a `JOB` line whose command contains `native_venv/bin/python` makes the *launching ssh shell's* cmdline
  match the queue/job gate (`pgrep -f 'native_venv/bin/[p]ython'`) -> "GATE BLOCKED"/refused, and a refused JOB ends the
  runner. Start the runner from a shell whose command line contains neither `python` nor `torchrun` (write the queue file
  first, then `setsid nohup bash trn2_queue.sh <file> &` in a separate ssh).
- **Gate bug in `trn2_job.sh`:** its gate greps `[s]ubmission_raw/train3_submission.py`; any job whose command mentions the
  vehicle path (e.g. `prepare.py eval-public --train-py submission_raw/train3_submission.py`) matches the job script's OWN
  cmdline -> always "GATE BLOCKED". Run such evals directly (detached script) after checking `neuron-ls` owners = 0.
- **Standalone `prepare.py eval-public` on the box HANGS** with the full kernel env (LAYER=op, ATTN=nki, EMBED=sidecar,
  OPT_SHARD=zero) in a single process: stuck after "SUBMISSION_BOOT: all imports done" for 2.5 h, no compiler procs. The
  campaign only ever scores inside the training run (torchrun). Re-score via a training run + `FLOW_EVAL_HOPS` instead.
- **RUN_EVAL0 (faithful training, eval with hop OFF) = 1.153573**: the hop-trained WEIGHTS score badly at eval with or without
  the hop, despite the best training loss -> a training-forward vs eval-forward difference on the same weights. Prime suspect:
  the tail-EMA blend (final = 0.65 w + 0.35 ema) on a sharper hop-trained trajectory. Test: RUN_NOEMA (EMA_BLEND=0).
- **RUN_CHK: fused hop op forward == eager hop on trained weights to ~1% (bf16 precision)** (layer 10: mean|diff| 0.0078 vs
  mean|ref| 0.70; max 0.25 vs 22). The op is correct; its wgrad path mirrors the layer op's (`ws` = grad-carrying casts of
  stack slices; WCACHE is OFF in the v40 env, so no detached cache is involved). The eval discrepancy is a training-regime
  effect. Knobs added to bisect: FLOW_HOP_VE (ve term + grad), FLOW_HOP_WGRAD (weight grads), EMA_BLEND=0 (RUN_NOEMA).
- **ROOT CAUSE of the eval collapse: the tail-EMA weight blend.** Faithful hop with EMA_BLEND=0 (live weights) = **1.063662**
  vs 1.179 with blend 0.35 (and 1.154 with pure EMA + hop off). A hop-coupled net reads the same weights twice, so it is far
  more sensitive to weight perturbation; a linear blend of two weight snapshots (live vs ~150-step-lagged EMA) lands
  off-manifold. The op is correct; the trained weights are fine. Next: REF with EMA off (EMA's worth) and hop with blend 0.1.
- **Fair 600 s pair on the box (both EMA_BLEND=0):** hop (faithful, every other block, DB16) **1.063662** vs reference
  **1.053373** = **+0.010**. The EMA blend itself is worth only ~0.0005 at 600 s. The hop leads on training loss at equal
  steps (2.929 vs 2.976 at step 800) but the reference runs ~18% more steps in the same wall time. Under Trn2's fixed-time
  budget the hop loses at this horizon; the 1800 s gate is the definitive test.

## VERDICT (2026-09-13 04:10 UTC): multi-hop attention on Trn2 = 0.992602 vs v40 0.985161 (+0.0074). DISCARD.
The full gate confirms the 600 s fair pair (+0.010). Multi-hop attention is a genuine FIXED-STEP win (-0.0083 locally at depth
12; -0.0047 with the box's detached semantics; the faithful fused hop leads the reference on training loss at equal steps) but
under Trn2's FIXED-TIME budget the fused hop's +15% step time (0.62 vs 0.53 s, ~15% fewer steps) outweighs it. To net a win
the hop would have to cost < ~8% (roughly half the fused op: K/V reuse - only Q + attention + out-proj - is the next lever if
this is ever revisited). Everything else stands: op correct (forward == eager to ~1%), eval collapse = tail-EMA blend (use 0.1
or 0 with hops), keep eval hops = train hops on the box. submit/train.py stays v40.

## Compiler-flag sweep (2026-09-13 morning) - running log
- torch_neuronx default optimization level is **-O1** (`NeuronCompiler.cpp`: `NEURON_COMPILER_OPT_LEVEL` or "-O1");
  `NEURON_CC_FLAGS` is appended as additional args; both are hashed into the NEFF cache key (every flag set cold-compiles).
- **`--model-type=transformer`: 1.119822 vs ref 1.052747 (+0.067) with identical step time, same step count, slightly better
  final train loss -> the flag miscompiles the eval-mode graph.** Causality passes (deterministic), so the harness cannot
  catch it. Lesson: a compiler flag can leave training intact and silently corrupt eval; judge flags ONLY by val_bpb.
  (Re-reading last night: the hop eval collapse may share this mechanism - eval graph vs train graph - rather than being
  purely the EMA blend; EMA_BLEND=0 fixed it empirically either way.)
- **-O3 via NEURON_CC_FLAGS: 1.118588 (+0.066), the SAME broken-eval signature as --model-type=transformer**, with normal
  training (1123 steps, final loss 2.835 vs ref 2.831) and the same eval fingerprint (eval_seconds ~30 vs 7.7). Two unrelated
  flags -> the breakage is systematic when NEURON_CC_FLAGS is set, not flag-specific. Lead: `KernelExecution.cpp` also reads
  NEURON_CC_FLAGS, so the flags recompile the NKI custom ops as well. The canonical `NEURON_COMPILER_OPT_LEVEL` route (a
  different code path) is the discriminator (FL_O2c/FL_O3c). **Step-time verdict so far: -O3 changed the code (longer
  compile, different rounding) but step dt stayed 0.52 s -> the step is latency-bound, not schedule-bound.**
- **Canonical route is clean:** `NEURON_COMPILER_OPT_LEVEL=-O2` recompiled everything (217 NEFFs, cold eval graph) and
  evaluated SANE (1.054609 vs ref 1.052747). So: (a) the eval corruption is specific to `NEURON_CC_FLAGS` (6 of 6 flags via
  that variable broke eval; the variable also feeds the NKI custom-op kernel compile in `KernelExecution.cpp` - suspected
  but not proven mechanism); (b) eval_seconds ~28-30 s is merely a cold eval-graph compile, not a corruption fingerprint;
  (c) -O2 gives 0.52 s/step with bit-identical loss = no-op. Never ship a submission that relies on NEURON_CC_FLAGS.
- **-O3 via the canonical route (FL_O3c): 1.119461, broken eval too** (recompiled, 193 NEFFs; 0.53 s/step). So -O3 itself
  corrupts eval regardless of route, while -O2 canonical is clean; and every NEURON_CC_FLAGS setting corrupts eval. At least
  two triggers, shared mechanism unknown (untested hypotheses: an eval-graph fusion legal under -O1/-O2 only; the custom-op
  kernel recompile path). Not pursued: no configuration improved step time, so nothing here would ever be adopted.

## LEVER 1 VERDICT (compiler-flag sweep, 2026-09-13 06:33-09:38 UTC)
9 configurations, all at **0.52-0.53 s/step**, identical to the reference (v40 config, 0.52 s). The Neuron compiler's
optimization level and every relevant enable-flag leave the step time untouched: the graph is latency-bound (launch/collective/
DMA-issue bound), exactly as the Sep 10 engine profile said (all engines 30-60% busy). There is no schedule-search gain to be
had from the compiler on this graph. Lever 2 (kernel-schedule grid) has one live knob (head vocab chunk _HVC 512->256, run
KN_HVC256). Lever 3 (beam-searching the fusion partition) would require a NKI code generator - a compiler project - and the
partition space around this architecture has been walked by hand (each remaining fusion measured <= 0.001).
- **Lever 2 closed:** `_HVC=256` fails the NKI tracer (a second hard-coded 512 in the fused head kernel's sub-block loop). The
  only live schedule knob is not free without a kernel rewrite whose ceiling is a fraction of a percent of step time. Not pursued.

## tinygrad as an alternative code generator (2026-09-13, forked subtask) - NOT VIABLE
tinygrad 0.14.0 (cloned to `tinygrad/`, untracked, 70 MB; also `/tmp/tinygrad` on the box) has zero Neuron/Trainium/NKI
references; backends are all SIMT GPUs or scalar CPU/LLVM (on the box it selects CPU). Its kernel model is per-thread SIMT code
(blockIdx/threadIdx, warp-level mma intrinsics); the NeuronCore has no C/LLVM target, only HLO (neuronx-cc) and the NKI tile
DSL. "Adding Neuron ops" = a renderer that re-vectorizes scalar loop nests into 128-partition tile instructions with engine
selection and SBUF/PSUM/DMA management, plus a runtime: a compiler for a non-SIMT target (weeks-months). Even built, it would
emit one generic tile kernel per reduce (the same class as neuronx-cc glue, measured 8x slower than the fused hand kernels),
without the cross-op fusion the hand kernels win on; its BEAM autotuner searches SIMT axes that do not exist on Neuron.

## Own flash attention, Phase A (2026-09-13 evening) - written, integrated, at parity
- **v4 (strided-DMA loads) is dead:** NaN in-model under v40's native head-major layout (ATT_V4) and with SIDECAR_SYNC=1
  (ATT_V4_SYNC) -> the sidecar-race hypothesis is refuted; strided INPUT DMAs are intrinsically unsafe in the compiled
  graph, like strided output stores (memory rule #11). Deterministic in isolation, wrong in-graph.
- **Plain per-tile-load fwd + our strided-backward:** clean, causality passes, 0.54 s/step (+4% vs 0.52 ref), 1.056006
  vs 1.052747 on 1088 vs 1131 steps (the deficit is exactly the step-time cost).
- **Per-step glue hoist** (vehicle sha e641750822f35107): the kernels' per-row inputs [doc end, min(end,q+1), doc start,
  key index] built ONCE per step as a (4, B, t, 1) pack passed through the op's existing start/end inputs (free views per
  layer), raw lse passed through the neg_max slot when both kernels are ours, batch size derived from the pack. Bit-identical
  loss to the unhoisted run (correct); **0.53 s/step** -> recovered ~1 of the 4 points. **Verdict: parity, not a win.**
  Remaining per-layer glue: sink-row expand (x24), dsink reduction (x12), the recip placeholder (x12), plus in-graph
  scheduling of two extra kernels per layer. The backward kernel's isolated saving (0.74 ms x 12 = 9 ms) is real but is
  consumed by integration cost. A net win requires fusing attention INTO the layer kernels (one kernel per layer
  direction, no per-layer glue), i.e. the megakernel direction (est. -1..-2%).
- **ATT_PLAIN_HOIST scored 1.125406** (training bit-identical to ATT_PLAIN's 1.056): the hoisted pack corrupts the EVAL
  graph while the training graph is correct. Third distinct trigger of the same ~1.12 eval-corruption signature (-O3,
  NEURON_CC_FLAGS, now a graph-produced per-step input pack read by kernels). Common thread is unknown; the harness
  cannot catch it (causality is deterministic). Not pursued further: lever B's ceiling is ~0.0005 and even the working
  combo is only at parity. **Keep the vendor attention.** The hoist code stays default-off (ATTN_FWD=vendor).

## DATA PIPELINE (2026-09-14) - measured on the real shards/tokenizer (local, scripts data_m123.py / data_m4.py)
- **Corpus:** ~63M tokens/shard x 5 public shards = **~315M tokens**; the v40 gate consumed 446.7M tokens = **1.42 epochs**
  (the gate log shows epoch 2 from ~step 1720). train.py cannot download more (requests/urllib are forbidden imports), so
  every team has the same 315M tokens.
- **Doc lengths:** mean 733, median 617, p90 1117, max 39k. Docs > 1025 tokens = 11.4% of docs but **38.1% of all tokens**
  (docs > 2050 = 24.2% of tokens).
- **The provided packer (prepare.make_dataloader, best-fit, row = 1025) never fits a doc > 1025:** long docs pile up in the
  buffer (2500-row sim: 522 long docs = 1.55M tokens of residue vs 2.56M placed) until the buffer is all long docs, then
  each is TRUNCATED to its first 1025 tokens and the rest discarded. Plus ~1 short-doc truncation per row discarding
  6.9% of consumed tokens. Net: roughly a quarter of the corpus (the tails of long docs) is never trained on, and the
  effective unique set is ~220-240M tokens -> the run is ~2 passes over the short docs and ~0 over long-doc tails.
- **Eval rows are a different distribution:** make_eval_dataloader concatenates all docs and cuts every 1024 tokens ->
  **99.9% of eval rows start mid-document** (no BOS at position 0); the BOS-less prefix averages 513 tokens, so **50.0% of
  eval targets are predicted with no document start in context**, a situation the training rows (always BOS at pos 0)
  never contain. Only 15.9% of eval targets are within 128 tokens after a BOS.
- **Lever:** the organizer-sanctioned PACK trick (make_dataloader with seq_len = PACK*1024, reshape to 1024 rows; vehicle
  knob PACK, never tested on the compiled line) fixes both at once: rows 2..PACK start mid-doc like eval, and docs up to
  PACK*1024+1 fit, recovering most of the discarded long-doc tokens. Gates A_PACK8 / A_PACK4 queued first in queue_A2;
  local fixed-step B13_d12_pack4 running. M4 (per-position eval loss of the depth-12 checkpoint) quantifies the BOS-less
  penalty directly.
- **M5, full-shard steady state (shard 1, 63.2M tokens), exact copy of the provided packer:**
  | PACK | row capacity | corpus placed | truncation-discarded | never placed (long docs) | 1024-rows starting mid-doc | targets in BOS-less prefix |
  |---|---|---|---|---|---|---|
  | 1 (v40) | 1025 | **72.3%** | 18.1% | 9.6% | 0.0% | 0.0% |
  | 4 | 4097 | 87.1% | 3.3% | 9.6% | 74.9% | 36.5% |
  | 8 | 8193 | **92.1%** | 0.7% | 7.2% | 87.4% | 42.5% |
  | eval | stream | - | - | - | 99.9% | 50.0% |
  So v40 trains on ~228M unique tokens (~2 passes in 446.7M) with zero exposure to the eval's dominant condition; PACK=8
  trains on ~290M unique tokens (+27%, ~1.5 passes) with 42.5% of targets in the eval-like BOS-less condition. Both
  effects are in the same direction and cost no step time (the packer runs on the host; per-step token count unchanged).
- **M4, per-position eval loss of the depth-12 checkpoint (B7_d12_base, 1.1888 bpb) on the full public val shard**
  (61,888 rows, 99.9% start mid-doc, 63.4M targets, mean 3.304 nats):
  | context length | after-BOS nll (n) | BOS-less prefix nll (n) | delta |
  |---|---|---|---|
  | [0,8) | 4.321 (0.68M) | 4.836 (0.49M) | **+0.515** |
  | [8,32) | 3.711 (2.0M) | 3.903 (1.45M) | **+0.192** |
  | [32,128) | 3.447 (7.3M) | 3.495 (5.3M) | +0.047 |
  | [128,512) | 3.254 (16.7M) | 3.235 (15.0M) | -0.019 |
  | [512,1024) | 2.974 (4.6M) | 3.126 (10.0M) | **+0.151** |
  Excess nll of the BOS-less condition summed over buckets = 2.0M nats = **0.032 nats/token = ~1.0% of the loss (~0.010
  bpb at our level)**. Decomposition: the first ~128 tokens carry ~0.78M nats (0.4%) = the pure "no document start in
  context" effect (position 0 alone: 6.62 nats vs ~4.3 after a BOS - the model never saw a non-BOS token at position 0);
  the [512,1024) bucket carries 1.5M nats (0.7%) = rows lying entirely inside LONG documents, i.e. the long-doc content
  the v40 packer never trains on (M5: 9.6% never placed + 18% truncated). PACK=8 addresses both. Expected: -0.005..-0.010
  if the mismatch is fully removed, at zero step cost. Gates A_PACK8 / A_PACK4 pending; local fixed-step B13 pending.
- **Stream packer implemented** (vehicle `PACK_MODE=stream`, `STREAM_RESHUFFLE`): training rows built exactly like the eval
  loader (one BOS-prepended stream, row = next 1025 tokens, advance 1024), rank-strided row groups, seeded row-group
  permutation on epoch >= 2. Smoke on rank 0/8: 0 alignment violations, **99.8% mid-doc starts, 50.4% BOS-less targets**
  (eval: 99.9% / 50.0%), 100% corpus coverage, 6.9M tok/s per rank (200x the box's need). Tiny CPU end-to-end run: trains,
  causality passes, evals. Gate A_STREAM queued after A_PACK8 (queue_A3).
- **A_PACK8 died at its FIRST `sidecar.prepare(x)`** (before any step) with the "scatter/gather (indirect memory copy via
  vector DGE) out-of-bound access" error, deterministically; A_STREAM (my stream loader, same mid-doc rows) passed the same
  point and went on to compile. M6 refutes the plan-length theory: every packing mode (PACK=1 included) produces ~78k-slot
  scatter plans (max repetition ~530 in a 16k-token micro-batch), so the chunked >32768 path is ALWAYS used and is not
  what distinguishes PACK=8. Cause unknown (something in the provided loader's PACK>1 rows as consumed by the eager
  sidecar gather; the old ledger also has PACK>1 dying at start on an eager line). PACK=8/4 superseded by stream mode.
- **B13 (local, depth 12, 500 fixed steps): PACK=4 = 1.188717 vs 1.188775 base -> a wash.** Training on eval-like rows did
  NOT reduce eval loss at fixed steps, so most of M4's BOS-less excess is inherent to the eval rows (first tokens of a
  mid-doc fragment are unpredictable with or without a BOS; long-doc content is harder), not a trainable train/eval
  mismatch. Revised expectation for lever 0: the coverage gain (72% -> 100% of the corpus) only pays where the v40 run
  repeats data (epoch 2 starts at ~step 1720 of ~3400) -> the A_STREAM full gate is the decisive number; local
  fixed-step screens cannot see it.
- **Coverage confirmed on the box:** A_STREAM entered its second data pass at step 2410 vs A_REF at step 1720 (ratio 1.40 =
  the predicted 1/0.72): the stream loader delivers 40% more unique tokens per pass at the same 0.52 s/step. The 1800 s run
  becomes ~1.4 passes instead of ~2.0. Final score pending.

## DATA LEVER VERDICT (2026-09-14 19:31 UTC): A_STREAM 0.987395 vs A_REF 0.985741 = +0.0017. CLOSED.
Full-length gate of the eval-style stream packer (100% coverage, 99.8% mid-doc rows, epoch-2 reshuffle, same 0.52 s/step,
40% more unique tokens per pass - confirmed by the epoch-2 step 2410 vs 1720) LOSES slightly. Combined with B13 (eval-like
rows are a wash at fixed steps): (1) the measured BOS-less eval excess is inherent to the eval rows, not a trainable
mismatch; (2) ~2 passes over 72% of the corpus is not worse than ~1.4 passes over 100% (consistent with data-constrained
scaling results: <= 4 epochs of repetition is nearly free); (3) best-fit full-document rows are a slightly better training
signal per token than arbitrary stream cuts. PACK=8/4 via the provided loader crash the eager sidecar at the first batch
(cause unknown; superseded). Keep the v40 loader. The stream packer stays in the vehicle, default-off (PACK_MODE=bestfit).

## CORRECTION (2026-09-14 20:05 UTC): A_PACK4 = 0.981426 vs A_REF 0.985741 = -0.0043. THE DATA LEVER IS REAL.
The organizer-sanctioned PACK trick at 4 (best-fit rows of 4097 tokens reshaped to 4 x 1024; 87% corpus coverage, 75% of
rows start mid-doc; same 0.52 s/step) beats the reference by 0.0043 (band ~0.001) and the v40 gate by 0.0037 - the largest
single gain since v39 -> v40, from one env var. My stream packer (100% coverage, ALL rows arbitrary cuts) lost by 0.0017,
so the mechanism is not coverage or eval-likeness per se: **best-fit keeps short docs whole and places long docs (<= 4k)
whole at the head of a group** (BOS at row start, continued as BOS-less sub-rows) - i.e. long-document content gets trained
in full without fragmenting the short documents that stream cuts destroy. PACK=8 crashes the eager sidecar at batch 1
(cause unknown). Repeat gate A_PACK4b and PACK=2 queued first (queue_A5); if confirmed -> clean v42 = v40 + PACK=4.
The "lever 0 closed" verdict above is retracted.

## v42 = v40 + PACK=4: gated 0.981197 (T59, causality passed) -> submit/train.py (2026-09-14 21:53 UTC)
Three independent full gates of PACK=4: 0.981426 (vehicle), 0.981073 (vehicle repeat), 0.981197 (clean file) vs the
same-day reference 0.985741 and the v40 gate 0.985161: **-0.0040 .. -0.0047**, 4x the 1800 s band, zero step cost. Portal
expectation ~0.9875 (portal = local + ~0.0065; v40 scored 0.9915 from 0.9852). Mechanism (measured): the provided 1025-row
packer never fits a document longer than a row (28% of the corpus never trained on, 38% of tokens live in such docs);
4097-token best-fit rows place documents up to 4k whole (long-document content trained in full, short documents kept
whole, 75% of sub-rows start mid-document like the eval rows). Stream-cut rows (100% coverage but every doc fragmented)
LOSE +0.0017, so whole-document rows are the ingredient, not coverage per se. PACK=8 crashes the eager sidecar at batch 1.
- Tooling lesson (Sep 14): a queue line must be `<ID> <secs|full> ENV=..`; a generated line with the env before `full`
  makes trn2_run.sh exec `env ... full ...` ("env: 'full': No such file") and the runner logs a 35 s "finished: 0 errors".
  Always `awk '{print $2}' queue | sort -u` == full/secs before launching. Re-launched as queue_A7b (7 PACK=4 scalar gates).
- **A_PACK2 = 0.981913 (-0.0038):** PACK=2 already captures ~80% of the PACK=4 gain; PACK=4 stays the choice.
- **ROOT CAUSE of the "transient" tracer crashes (RUN_NOEMA, P4_ELR8):** NKI's tracer reads kernel source from DISK
  (inspect.getsource) at trace time. Overwriting the vehicle while a run is IN PROGRESS shifts the line offsets, and any
  kernel traced afterwards in that process (the end-of-run eval graph is compiled at the end!) fails with "argument 'x_in'
  not supplied". P4_ELR8 trained all 3400 steps and lost its score this way. **Rule: never overwrite the vehicle while any
  run of it is in progress - not only while a queue has unlaunched lines.** Ship only between runs (or use a versioned
  filename per run).

## Parallel residuals on Trainium (repeat-regime idea 3) - the actual constraint and the experiment (2026-09-15)
On a NeuronCore separate kernel launches execute back-to-back; the compiler cannot co-schedule two opaque kernels. So
parallel residuals buy step time ONLY if the MLP's matmuls sit in the SAME instruction stream as the attention, where the
NKI scheduler can fill the attention's non-PE phases (softmax on Vector/Scalar/GpSimd) with the MLP's PE work. Options:
(a) our own attention in a layer megakernel (parked: strided-DMA corruption, parity at best), or (b) nest the VENDOR
attention: `nkilib...attention_cte` is a thin @nki.jit entrypoint over an undecorated `_attention_cte(...)` that exposes
`out_o_hbm/out_neg_max_hbm/out_sum_hbm` and `init_sbuf_addr` - i.e. it is built to be called from inside another kernel.
Upper bound: attention fwd+bwd ~3.8 ms/layer and MLP fwd+bwd ~3.6 ms/layer of a ~37 ms layer -> perfect overlap saves up
to ~10% of the step (~0.003 bpb); realistic maybe half (both need the PE during attention's matmuls).
**Experiment queued (JOB PARRES_T after the gates, `nki/nki_parres_test.py`):** time (a) attention_cte alone, (b) mlp_fwd
alone, (c) one kernel = nested `_attention_cte` + the MLP body inline, at production shapes. time(c) ~ max(a,b) -> build
the parallel-residual layer kernel (fwd: pre kernel also emits the MLP branch from norm(xm); bwd: mlp_bwd on norm(xm) ->
nbwd -> fold into pre_bwd_k's pass-through dx1 - no pre_bwd_k change needed); time(c) ~ a+b -> the branch closes.
Quality precondition: local fixed-step B14_d12_parresid (eager PAR_RESID) vs 1.188775 - a loss there makes the kernel moot.
- **B14 (parallel residuals, eager, depth 12, 500 fixed steps): 1.306967 vs 1.188775 (+0.118).** Training loss matched
  the baseline through step 300 (3.42 vs 3.37) then ROSE to 3.84 by step 490 - a late-run instability with the current
  optimizer scalars (Muon LR, per-layer lambdas tuned for sequential blocks), not an eval bug. Idea 3's quality
  precondition fails as-is; the intra-kernel overlap experiment (PARRES_T) stays queued because it is 5 minutes of device
  time and answers a general question (can vendor-attention + MLP co-schedule in one kernel), but the parallel-residual
  layer kernel is not worth building unless that overlap is large AND a retune makes the block stable.
- **Local fixed-step noise band measured:** same-config, same-seed depth-12 repeat = 1.188077 vs 1.188775 -> **~0.0007** (vs
  the box's ~0.0004 same-config spread). Local screen verdicts below ~0.0015 are not signals.
- **B17 (parallel residuals at matrix LR 0.45): stable, 1.193635 vs 1.188775 (+0.0049).** The instability was LR-driven; a
  stable parallel block costs ~0.005 at equal steps, i.e. it would need >~16% step-time savings to break even, above the
  ~10% intra-kernel overlap upper bound. **Idea 3 closed on arithmetic.** The PARRES_T overlap job stays queued as a
  5-minute general measurement (vendor attention + MLP co-scheduling in one kernel) for future fusion decisions.
- **WD on the PACK=4 base at full length: 0.1 -> 0.981197, 0.2 -> 0.985687, 0.3 -> 0.992145.** Monotonically worse; the
  "strong WD is the precondition for MIR" premise of repeat-regime idea 1 does not hold on this line (1.4 epochs, Muon).
  MIR gates (0.05/0.10/0.15 at WD 0.1) follow; locally MIR 0.10 cost +0.014 at fixed steps.
- **MIR at full length on PACK=4: 0.05 -> 0.986213 (+0.005), 0.10 -> 0.990758 (+0.010).** Monotonically worse, matching the
  local fixed-step result. Repeat-regime idea 1 (strong WD + masked-input aux loss) is CLOSED on this line: at 1.4 epochs
  with Muon there is no under-regularization to fix; every added regularizer costs bpb roughly in proportion to its strength.
- **PARRES_T (intra-kernel overlap experiment, production shapes):** vendor attention alone 2.19 ms, our MLP alone 1.48 ms,
  ONE kernel with the nested `_attention_cte` followed by the MLP body 3.77 ms (bit-exact outputs). **No overlap at all**
  (3.77 >= 3.67 serial sum): the NKI instruction scheduler runs the two bodies serially even in one instruction stream.
  General conclusion: engine-level co-scheduling of independent work is NOT available by nesting kernels; the idle-engine
  time in the profile cannot be reclaimed by parallel branches. (Nesting the vendor kernel itself works and is bit-exact -
  useful for future fusion for DMA/launch reasons, not for overlap.)
- **Embed-LR curve on PACK=4 complete: x3.5 0.9851 / x4.5 0.9801 / x6 0.9812 / x8 0.9819 -> x4.5 optimum (v43).**

## v43 = v42 + EMBED_LR x4.5: gated 0.979941 (T60, causality passed) -> submit/train.py (2026-09-15 08:00 UTC)
Three paired gates of embed LR x4.5 on the PACK=4 base: 0.980139, 0.980029, 0.979941 (clean file) vs v42 0.981197 ->
-0.0013, three bands. Cumulative since v40: 0.985161 -> 0.979941 = -0.0052, both gains from the data/scalar axis at zero
step cost. Portal expectation ~0.9865. Everything else gated on the PACK=4 base this cycle was flat or worse: WARMDOWN
0.7/0.8, MATRIX_LR 0.55/0.85, EMA blend 0.5, EMA start 0.75 (flat); WD 0.2/0.3, MIR 0.05/0.10/0.15, embed LR 3.5/8 (worse).

## User ideas (2026-09-15): bigram re-injection, smear + drop layer-0 attention
- **Bigram re-injection** (`BIGRAM_REINJ`, the gathered bigram row re-added at 2 or 4 block inputs via zero-init scalars;
  zero extra gathers/scatters): local fixed-step -0.0019 (x2) / -0.0023 (x4) vs 1.188775, but full gates on the v43 base
  **0.980789 (x2) / 0.981184 (x4) vs T60 0.979941 = +0.0008 / +0.0012**. Another short-horizon gain that inverts at 1800 s
  (like the 600 s LR wins, MIR-free regularizers). Closed.
- **Smear + drop layer-0 attention** (`SMEAR`, `DROP_ATTN0`): step time 0.53 s = no gain (one of twelve attentions is < 1% of
  the step; the smear gate's host elementwise work cancels it). 600 s 1.053269; full gate G_SMEARDROP pending.
- **Smear + drop layer-0 attention full gate: 0.983132 vs 0.979941 (+0.0032)** at the same step time. Closed: the step-time
  premise fails (one attention of twelve is < 1% of the step) and layer-0 attention is worth ~0.003-0.01 that the smear gate
  cannot replace. Smear alone (B22, local) pending as a footnote.
- **PACK=8 (2026-09-15):** the A_PACK8 step-0 crash was the intermittent OOB flake, not PACK=8 (host plans valid; torch-sidecar
  and NKI-sidecar 300 s runs both clean; full gate clean). Full gate on the v43 base: **0.980043 vs 0.979941 = wash**. The
  packing curve is flat from PACK=4 on; v43 (PACK=4) stays. Smear alone: +0.0022 locally.
