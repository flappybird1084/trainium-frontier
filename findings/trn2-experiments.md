# Trn2 experiment campaign (real Trainium, fixed 1800 s wall-time)

Started 2026-09-05 on test4 (`trn2-access.md`). Master copy = this project dir; the box's `~/tf` is an rsync mirror (never
edit there). **Per-run rows live in `trn2_results.tsv` (append-only; run ids here match it)** — this file carries the
narrative, the mechanisms and the durable numbers. Hardware: `trn2-hardware.md`; per-engine reads: `engine-profile-512.md`,
`engine-profile-v36.md`; kernel reference: `nki-kernel-playbook-2.md`; submission-contract history: `playbook.md`.
Counterpart of `train2-experiments.md` (the 3090 ladder): one change per run, keep only on improvement, log washes and
crashes too.

## Protocol and discipline (all of this was paid for)
- **Scoring run:** exact competition shape (`NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8`,
  1800 s budget, `prepare.py`'s own `public_val_bpb`) — ~35 min wall. **Eager LNC=2** (4 workers x 2-core logical NCs) is
  the other sanctioned shape (`NEURON_LOGICAL_NC_CONFIG=2 … --nproc_per_node=4`) and was often the faster one.
- **Nothing is kept until confirmed at 1800 s** (the metric is loss-per-wall-time). Noise bands: 600 s screen ~0.005,
  900 s ~0.001-0.002, **1800 s ~0.0004** (three same-config gates spread 0.00035; every run is same-seed and same data
  order, so comparisons are effectively paired). **600 s screens over-favor high-LR and high-step-count configs and do not
  transfer** (v41: screens 1.0578 vs the full gate 0.994234 vs incumbent 0.985161).
- Launch from a plain detached shell (`setsid nohup … &`), never from a monitor/wait loop (a run got SIGKILLed on monitor
  teardown). **Hard-gate every launch** on zero procs AND zero `neuron-ls` owners (memory `trn2-launch-gating`).
  Never call a run hung from log silence (memory `trn2-startup-silent-gaps`). Never run `prepare.py eval-public`
  standalone on the box (hangs).
- **Queue discipline:** append only while the runner is several items from the end; insert AFTER the running line, never
  before it (the runner re-reads by offset); never overwrite a running bash script; restart the runner for script changes.
  For long unattended queues pre-clean `out_*` (each screen writes a ~228 MB checkpoint; a full disk stops the runner).
- **Cold compile is 500-650 s** (~900 s for 640-wide at LNC=2) against the 900 s startup allowance. First run of a new
  shape only warms the NEFF cache — measure on the second. Cold-compile every new version once with a fresh cache dir and
  confirm startup << 900 s before it goes to the portal. The NEFF cache is keyed on the **graph (MLIR) content**, not the
  file: training-loop/constant edits are free, graph changes recompile. Compiler-flag experiments need BOTH stores fresh
  (`TORCH_NEURONX_NEFF_CACHE_DIR`, `TORCH_NEURONX_HLO_CACHE_DIR`) or the cached NEFF is silently reused (T17).
- **Whoever ran first on a box owns `/tmp/{hlo,local,neff}_cache`**; a different user must `chown` them or run as that user.

## The version ladder (every number a full-length gate unless marked)
| version | change | local val_bpb | notes |
|---|---|---|---|
| v5 | eager champion, LNC=2 launch (T4_gate) | 1.140051 | portal FAILED `no_run_meta` |
| v6 | compile default OFF + fallback, ASCII (T6) | 1.155026 | portal FAILED identically |
| v7 | PACK=1, per-worker rows capped at 8, 8-way (T7) | 1.176902 | **portal SCORED 1.1827** -> portal = local + ~0.006 |
| v9 | stacked parameters + function-mode backward (T9) | 1.128411 | compile + LNC=2; T10 = 1.160509 eager 8-way |
| v10 | width 512 / 4 heads (T14) | 1.120626 | portal-safe shape; fresh-process eval 1.121571 |
| v11 | embed LR x2 + warmdown 0.8 (T20) | 1.112172 | |
| v12 | NKI attention + fused Muon/AdamW kernels, eager LNC=2 (T26) | 1.069919 | 0.40 s/step |
| v13 | the same kernels on compile + 8-way (T25) | 1.070259 | 0.41 s/step; tie, portal-proven shape |
| v14 | clean minimal file, same math, eager LNC=2 (T27) | 1.069640 | all 6 gates + fresh-process |
| v15 | clean + compile 8-way (T28) | 1.052909 | explicit attention (kv 1024 tiles well), kernels |
| v16 | + fused RoPE kernel (T29) | 1.064665 | eager LNC=2 |
| v17/v18 | + concat-free sink (SINK_IMPL=lse), then warmdown 0.9 (T31) | 1.051593 | v17 is the lse-only variant |
| v19 | + 65k tokens/step with sqrt-batch LR (T32) | 1.040894 | -0.0107 |
| v20 | + flat scalar optimizer group + ZeRO-1 sharding (T33) | 1.036095 | -0.0048, no model change |
| v21 | + 131k tokens/step (T34) | 1.034802 | epoch 2 starts ~step 1720 |
| v23 | + embedding sidecar (NKI gather/scatter) (T35) | 1.026999 | first sub-1.03 |
| v24 | + bigram 2^18 + warmup 50 (T36) | 1.025514 | **portal 1.0311, 9th** |
| v25 | scatter plans capped 32768 + index_add fallback | (= v24 to 5e-5 at 600 s) | robustness, same math |
| v26 | + partial RoPE 0.5 + key offset + tail-EMA (T37) | 1.021352 | |
| v27 | + fused head kernel (HEAD=nki) (T38) | 1.019536 | |
| v28 | + layer kernels as custom ops (LAYER=op) (T39) | 1.012167 | |
| v29 | + 8-row micro-batches (T40) | 1.011566 | |
| v30 | + matrix LR x0.7 + attention scale 0.1 (T41) | 1.010664 | |
| v31 | + embedding LR x6.0 (T42) | 1.002410 | largest schedule gain of the campaign (-0.0042) |
| v32 | + MLP ratio 6 (T43) | 0.996429 | first sub-1.00 |
| v33 | review-safe rewrite (T45) | 1.000418 | **portal 1.0069, 8th** |
| v34 | + logits-returning fused head (HEAD=nki2) (T46) | 0.998477 | review-safe |
| v35 | + fused loss presented for review (T49) | 0.996308 | on-device fused-loss check |diff| 9.5e-7 |
| v36 | + plan-buffer ring for the async scatter (T50) | 0.996520 | same numerics as v35; robustness |
| v37 | + fused pre-attention backward (T51) | 0.992056 | -13% step |
| v38 | + 16-row micro-steps (T52) | 0.989999 | 0.46 s/step |
| v39 | + depth 12 (T53) | 0.986544 | the re-spend finally pays |
| v40 | + attention-layout kernels (T56) | 0.985161 | head-major q/k/v; this file's last promotion |
| (v41) | LR re-tune | 0.994234 @gate | **discarded** — screen win did not transfer || v42 | + PACK=4 data packing | 0.981197 | after this file's period (handoff-13b) |
| v43 | + embed LR x4.5 | 0.979941 | current submit |

Frozen-file shas (`submission_raw/`): v5 912e282cc04dd38b, v6 4691c9d5f266fb0d, v7 71404a2b30fe0426, v9 8b8e5210fa136c53,
v10 6dd554d720ae421e, v11 c6cd05b896ff290b, v12 58eccae335e3a435, v13 d1e208cc43871694, v15 fc1f33c77b5cf519,
v16 90fe1fedbcdb52e5, v17 2e8ad23e03a68697, v18 b737878e6c23ef70, v19 ca78aae3f672f3df, v20 a508d2b77c4224e1,
v21 3c3a39bdda5bdb22, v23 34155e0b773e407b, v24 f387d28aed03940b, v25 c2652cba20c66cbd, v26 3d623c8a0065fbca,
v27 a396e2dd36910e3a, v28 67b631e733b97665, v29 c6bebb0660b45081, v30 d61d510fcb37867d, v31 22d2a0749a9854cc,
v32 e17aa1e0d787b1a4, v34 cf1ca1b04a501e8d, v35 73049474159aa77d, v36 0474724dc02cf4d5, v37 aac729e027fa1c09,
v38 d19cff0aec87b2bf, v39 423feb03b6782053, v40 a9594ed53945831f.

**Naming collision:** "v41" appears twice in the record — a `v39 + own attention` candidate (sha c4a9e81e7c4270bf, `$S/make_v41.py`)
whose gate T55 NaN'd at step 3 (DO NOT promote), and later the discarded LR re-tune above.

## Sep 5: the port (T1-T4) — nothing ran, then everything did
- **S1a/S1b, instant crashes:** root-owned `/tmp/{hlo,local,neff}_cache` from the user's baseline run -> `Permission denied`
  on their lock files. `sudo chown -R ubuntu:ubuntu` fixes it; the ownership rule above.
- **S1c/S1f, the dynamo alias guard:** `ValueError: Output 2 already aliases input 123` with COMPILE=1. Local CPU AOT
  probe: 124 graph inputs (input 0 = `idx`, input 123 = `targets`), zero mutations, **34 pass-through returns** (inputs
  saved for backward and returned unchanged) merged with the backend's post-AOT pass-through list into one
  `AliasingInfo` -> raises on an output-index collision. Fix: `idx.clone()` / `targets.clone()` (saved tensors become
  intermediates) — the `x0 = x.clone()` first guess was harmless but not the cause. Compile default flipped to eager, since
  the portal sets no env.
- **S1d, the LR-in-NEFF discovery:** eager champion ran at 0.68 s/step, but **every distinct python-float LR value
  recompiles the NEFF** — warmup 4.05 s/step, and the warmdown (last 60% of the budget) ~4-5 s/step. Fix (S1e):
  `MuonAdamW._dev_scalar` keeps lr/momentum as 0-d device tensors refreshed by host->device `copy_`; AdamW `p.sub_(lr_t*update)`;
  bias corrections left as python floats (empirically no recompile, CUDA numerics byte-identical). 294 steps vs 88, flat
  0.79 s/step through warmdown, zero recompiles. **This would have hit the portal.**
- **S1g, compile:** 433 steps (eager 294), 0.52-0.55 s/step (~62k tok/s), 240 s val 1.3637 vs 1.4797; cold whole-graph
  compile 184.6 s. util.log: NC ~5-6%, CPU 85-95% — eager is op-by-op dispatch (thousands of tiny launches, chip ~95%
  idle), the compiler fuses forward+backward but the optimizer stays eager.
- T1 (eager, portal-exact) **1.174335**, -0.0846 vs the 1.258923 baseline; the 3090 ladder's projection was within 0.004 —
  **the ladder transfers.**
- **Fixed-cost hunt:** eager cost model = 0.42 s fixed + 0.37 s per 32k tokens. Gradient bucketing = wash (S2 0.56 vs 0.53);
  compiled optimizer (S3b) = 0.65 vs 0.53 **slower** because the dynamo backend partitions into many small graphs —
  compiling did not cut the ~29k device executions/s. Conclusion: the fixed per-step cost is not removable; **do more
  useful work per step at ~flat cost: wider model and/or bigger batch** (the opposite of the 3090 optimum).
- Width: 512-wide 0.68 s/step at 119.6M params vs 0.53 at 51.9M (+28% time for 2.3x model); dev mem 74.5/96 GB.
  65k batch: +22% tokens/s but 39% fewer steps = wash at 240 s.
- **LNC=2 (L2/L2b): 0.47 s/step vs 0.53, NC util 17% (from 6%), CPU 70% (from 95%)** — with 8 workers on 12 vCPUs the
  *host* was the bottleneck. Sanctioned launch (README; `prepare.py check` passes).
- T3 1.139926 (target met), T4_gate 1.140051 on the exact submission shape -> frozen v5.
- **Portal saga:** v5 FAILED `USER_TRAINING_CRASHED exit=1 status=no_run_meta (no train.log)`; v6 (compile off + fallback +
  ASCII — the file had 11 non-ASCII chars) FAILED identically, falsifying the compile hypothesis; v7 (PACK=1, rows capped
  at 8, the 8-way launch that has scored twice) **SCORED 1.1827**. Portal calibration **+0.006** over local. The standalone
  scorer path (`prepare.py eval-public`, fresh process, `load_for_eval`) was verified on the device (1.15536 vs 1.15503
  in-run) — checkpoint format and the hook are fine on Neuron.

## Sep 6: throughput campaign (T9-T20)
- **Stacked-parameter refactor** (matrix weights in one (n, out, in) Parameter per shape; per-block scalars in one tensor):
  bit-exact vs the pre-stack reference on CPU; optimizer tensors 120 -> 18 at 10 layers; 0.373 s/step (from 0.475),
  optimizer 0.181 -> 0.074, grad_sync 0.019 -> 0.007. Under eager the select-backward allocated a full stack-sized zeros
  per layer per step -> fixed with `_StackSliceLinear` accumulating into `stack.grad[i]` (function mode).
- Compiled-graph profiling: the compiled model is TWO regions — a custom autograd Function behind `F.cross_entropy`
  breaks the graph; an eager `aten::ne` costs ~38 ms/call, and each of the optimizer's 14 scalar H2D copies synchronizes
  (~280 ms/step). `CE_IMPL=manual` + one packed H2D scalar copy: 0.362 s/step (small — the copies were overlapped).
- Width curve (compile+LNC=2, 1800 s): 256 1.1284 -> 384 1.1065 (0.48 s/step) -> **512 1.0887 (0.49 — the 256->384 jump
  was the head-count change; FLOPs are still ~free)** -> 640 1.0945 (0.63 s/step, cold compile 893 s = at the cap). 512 is
  the optimum; anything wider would charge compile time to the budget. 65k batch at 512-wide does not fit an LNC=2 core
  (`NCC_EOOM002`, 28.56 GB peak estimate > 24 GB).
- PR7 profile: fwd_bwd 0.385 / grad_sync 0.012 / optimizer 0.094 = 0.491 s/step, ~61 TFLOP/s = ~9% of the chip.
- Screens: MLP ratio 5 -0.0019 (wash at full length), depth 12 +0.011 (at that speed), 8x64 heads +0.020 (halved head_dim
  doubles the small attention matmuls; the overhead-bound NEFF pays for every one).
- Compiler settings: `NEURON_COMPILER_OPT_LEVEL` defaults to **-O1**; T17b (-O2, real recompile, 568 s) and T18b
  (`--model-type=transformer`, 534 s) both gave identical step time and bit-identical loss traces. Closed.
- **LR/schedule screens at 512** (900 s, ref 1.140360): Muon LR x1.5 +0.0033 / x0.7 +0.0072 (at optimum); warmdown
  50/70/80% = +0.0015 / -0.0007 / -0.0011; embedding LR x2 **-0.0028** (x4 -0.0017); head LR x2 +0.0003; compiled
  optimizer +3% step. Scalar LR x2 was +0.013. T19 = 1.084527; T20 = v11 1.112172 on the portal shape.

## Sep 6-8: NKI kernel campaign (v23-v29)
- **Device trace at 512-wide:** ~430 model submits and ~5,000 small DMA copy-ins per step -> launch/DMA-bound.
- **Vendor flash attention (`attention_cte`/`attention_bwd`, via `wrap_nki`):** numerics bf16-level vs an fp32 reference;
  timing at one worker's shape, eager — explicit fp32 fwd 10.3 ms / fwd+bwd 15.6 ms vs **NKI fwd 1.3 / fwd+bwd 2.9 ms
  (5.4x)**. Decision rule (>= 2x) passed. Gotcha: the kernel's sequence-packing path asserts `scale == 1.0` — pre-scale q
  by 1/sqrt(d) in bf16 and run with scale 1.0.
- **Dynamo integration:** an `autograd.Function` around the kernels graph-breaks inside every Block (fatal with stacked
  weights, `recompile_limit`); the supported route is the **`nki_op(name)` decorator** (a `torch.library` custom op whose
  fake/functional dispatch re-traces the inner function, so the HOP appears only after AOT) built at module level before
  `torch.compile`.
- **Compiled-line verdict:** S_NKI3 ran 1.73 s/step at 512-wide with ~3,600 device executions/step (vs ~430 explicit) —
  the custom-op HOP is NOT inlined in the fused NEFF. **Keep the explicit attention on the compiled line; NKI attention
  pays on the eager line** (S_E8_NKI2 1.163167 vs 1.173359 base, -0.010). NKIF1 later showed the backend's own
  flex_attention route DOES inline (4.6 vs 5.6 ms/layer) but attention is only ~56 ms of the 385 ms fwd_bwd — **not the
  compiled bottleneck**; S_FLEX killed. The static doc-mask encoding (`DOCMASK=score`: doc id in 6 spare q/k dims as
  `q_ext·k_ext = -1024*((da)^2+(db)^2)`, 0 same-doc, <= -1024 otherwise, exact in bf16) stays available but unused.
- **Fused Muon step kernel (`MUON=fused`, NS-only variant `MUON=nki`):** 15-16 -> 2.0-2.4 ms per stack (512x512 15.9 ->
  3.1 ms, 2048x512 10.0 -> 2.5, 512x2048 11.4 -> 2.8); **fused AdamW (`ADAMW=nki`, `ADAMW_INPLACE=1`):** 2.43 -> 0.34 ms
  (8192x512), 20.1 -> 5.8 ms (131072-row table); optimizer ~94 -> ~25 ms/step.
- **NKI tracer rules learned the hard way** (memory `nki-tracer-rules`; the simulator accepts what the tracer rejects):
  kernels must be module-level (nested definitions break when the file around them moves); the kernel's **module globals
  are the only namespace** (no closure constants, no `global`/`import` statements inside); no list comprehensions or
  `for a, b, c in …` tuple unpacking; `scalar_tensor_tensor` cannot read both operands from PSUM; **inside custom ops keep
  every DMA a 128-row-aligned block** (build shifted/masked slices in torch) — unaligned DMAs are lowered differently
  in-graph and caused the OOB crash; strided-pattern OUTPUT DMAs are unsafe (rule #11).
- T21 (compile + optimizer kernels) 1.076414; T23 (eager, all kernels) 1.092707; **T26 = v12 1.069919 at 0.40 s/step** —
  eager + kernels + 8 rows/worker + 4 dispatching processes is the fastest configuration. Fused RoPE: 61 -> 0.59 ms per
  q/k call (-0.0064 at 900 s). Fused RMSNorm was slower than `F.rms_norm` (shelved).
- **Backward profile (PROF_BWD):** 0.254 s/exec, DMA 74%, ~115 GB of HBM traffic per worker-step (~30x the model's
  inherent activation traffic) — the compiled line is bandwidth-bound on compiler-materialized intermediates, so it was
  parked in favor of the eager kernel line.
- **Backlog screens on the eager kernel line** (900 s, ref 1.112149): window 256 on odd layers **-0.0017 (free, the only
  winner)**; window warmup adds nothing; trigram +0.0035; merged QKV +0.0005; MTP +0.040; value residual +0.0010; backout
  +0.0013; PLE +0.016; QK64 crashed (the NKI attention op assumes equal q/k/v head dims).
- **Compile 8-way line screens** (900 s, ref 1.096323): concat-free sink (`SINK_IMPL=lse`) -0.0010; window +0.0007 (the
  eager gain did not transfer); bf16 scores (`ATTN_BF16`) wash; SDPA +0.0011; width 640 +0.024; depth 12 +0.010; WEIGHT_T +0.0014;
  merged QKV +0.010; 2 rows/worker +0.005; **65k batch -0.0087** -> T32.
- **Optimizer phase** (S_C8_PROF: fwd_bwd 0.208 / grad_sync 0.027 / optimizer 0.073 / total 0.307 s/step; the optimizer is
  24% of the step but only ~12 ms of real work): flat scalar group 0.277, then **ZeRO-1 (reduce_scatter grads, one eighth
  of each stack updated per worker, all_gather back) 0.266** — cheaper than the all_reduce it replaces (0.027 -> 0.016) and
  the optimizer falls to 41 ms. Two pure hardware changes, no model change: v20 -0.0048.
- **Data budget:** the provided training set is exactly the 5 public shards (~225M tokens); T34 entered **epoch 2 at step
  ~1720 of 1978**. Every further throughput gain buys repeated tokens.
- **Knob sweeps at 65k/131k** (refs 1.079482 / 1.081878): Muon LR x0.8 +0.0046, x1.25 -0.0003; embedding LR x0.7 +0.0038;
  WD 0.05 +0.0003, WD 0.20 +0.0032; warmdown 0.95/0.8 flat; final LR 0 +0.0004, 0.10 +0.0022; warmup 50 -0.0012 (kept).
- **The embedding backward was the compiled graph's waste.** Backward hardware FLOPs 3.10 TF vs ~1.7 TF expected; the
  excess = the embeddings' backward lowered as dense one-hot matmuls. Ablation (bigram 2^14) cut fwd_bwd 18% (0.860 ->
  0.705 s/step). **`EMBED=sidecar`:** two NKI kernels (`embed_gather` indirect DMA, `embed_scatter_add` via `nisa.dma_compute`
  RMW with `oob_mode.skip`), tables outside the compiled graph; `dma_compute` needs distinct indices per 128-row
  instruction, so the host groups rounds (token ids repeat ~140-190x per 4096-token micro-batch; bigram hashes 25-50x);
  **slot counts are bucketed to powers of two >= 4096** or `wrap_nki` recompiles per micro-step. Steady gain 12% at first
  (`S_V21_SC4` -0.0166 at 600 s, 0.76 s/step); the NKI gather/scatter is worth ~9% over the device's own index ops
  (which are exact and cheap: `index_add_` 2.7 / 4.0 ms). T35 = v23 1.026999.
- Discards on the v23 line: width 640 +0.0065; depth 12 +0.0024; MLP recompute wash; flex attention with the logit sink
  broken (1.618 at 600 s). Keeps: **bigram 2^18 -0.0023**; warmup 50 (inside v24). 2^19 OOMs on a 12 GB LNC=1 core —
  2^18 is the ceiling on that launch. Trigram via sidecar +0.0050. **`EPOCH2=rotate` +0.0176 with a LOWER train loss** —
  the second loader repeats recently seen rows (memorisation), not a win.
- PROFV24 (sidecar graph, 4 rows): fwd 0.051 / bwd 0.082 s; PE 62% / 57%, DMA 53% / 53%; hbm 2.9 / 10.8 GB read,
  6.8 / 3.5 GB write; the 1.4 TF one-hot is gone; MFU 9.4% / 11.7%, max achievable 36% / 48%. Context 2048 (+0.0039) and
  bf16 score matrices (+0.0051) both discard.
- **Sidecar crash class:** a micro-batch with one index repeated > 256 times doubled the scatter plan to 65536 slots and
  the first compile of that variant failed mid-run (`[NkiDispatch] lower_func failed`) — v25 caps plans at 32768 slots,
  cuts them on tile boundaries, and wraps every kernel call with an `index_add_` fallback. Separately, the intermittent
  **"scatter/gather (indirect memory copy via vector DGE) out-of-bound access"** flake killed long runs; v36's plan-buffer
  ring (keeping the last two micro-steps' index buffers referenced) closed that hypothesis for production-shaped runs
  (10 deaths, all dev-mode).
- **Tail-EMA + partial RoPE:** F_V24_TEMA -0.0013; F_V24_ROPE (ROPE_FRAC 0.5 + key offset) -0.0032; both in v26 (T37
  1.021352). The key offset is a permanent circuit (gating it to 0 late costs +0.012).
- **MoD closed on the 4080** (720 s fixed time): L5_BASE 1.117662 vs L5_MOD 1.144366 (+0.0267 with 18% more steps) —
  routing half the tokens around five blocks loses more quality than the compute it saves. No kernel line for it.
- **Megakernel pricing (the pivot that mattered):** the eager all-kernels line was 2.1x the compiled step (1.508 vs
  0.713 s per 131k step) but OPPROF2 showed the excess is cast/copy glue, not kernels; `post_attn_fwd` (out-proj +
  residual + rmsnorm + ReLU^2 MLP + residual as one kernel) ran 0.49 ms per 4096 tokens (~39 TFLOP/s, about half the
  core's peak) vs 1.06 ms as 6 eager ops vs ~5.1 ms for the compiled graph's entire forward. **The 3-kernel layer forward
  = 3.36 ms/layer (33.6 ms per micro-step) vs ~51 compiled; the backward chain = 7.51 ms/layer (75 vs 82).** But the
  eager chain line was **host-bound** (OPPROF5-7: the device finishes within ~7 ms of the host's last enqueue, 7,162
  dispatched ops/step), so kernel fusion inside the layer cannot move the step until the op count drops ~10x ->
  **layer kernels must go INSIDE the compiled graph as custom ops** (`LAYER=op`).
- **Fused head kernel** (`nki_head.py`: final rmsnorm + lm_head + asymmetric softcap + online CE, logits never reach
  HBM): 3.25 fwd + 7.62 ms bwd vs 28.9 ms eager. -0.0036 at 600 s -> F_V26_HEAD 1.019731 -> T38 = v27 1.019536.
- **`LAYER=op` real result (S_V24_OP2): fwd_bwd 0.466 vs 0.713 s/step, total 0.595 vs 0.862 (-31%)** -> F_V26_OPHEAD
  1.012237 -> T39 = v28 1.012167. (The first `S_V24_OP`/`OPHEAD` screens were INVALID: the branch needs `rope_tabs`, which
  `GPT.forward` only built for `ROPE=nki`; what they measured was the vendor NKI attention in-graph, 65 ms/step slower than
  the compiler's explicit attention.) The layer-op forward graph is +31 ms vs the isolated kernels while its backward is
  7 ms faster; ~45 ms/micro-step is spent outside the compiled graphs (sidecar + launch gaps).
- v29 = v28 + 8-row micro-batches (T40 1.011566). LOADER_DEV=cpu wash; `CKPT=mlp` (MLP recompute) wash;
GRAD_BF16 collectives no time gain and +0.0006;
  loss+grads as one graph via `torch.func.grad` fails under dynamo.

## Sep 8-9: knob campaign and review safety (v30-v36)
- Matrix LR x0.7 = 1.010506 (best) -> v30 1.010664 with attention scale 0.1 (-0.0006). Batch 196k wash, 262k +0.0038;
  width 640 blocked by the kernels' 512 free-dim cap; **`FAKE_FP8=1`** (e4m3 rounding with per-row scaling inside the
  kernels) trains at parity — so real fp8 is a pure speed question, and the kernels are vector-bound today.
- **Embedding LR sweep:** x3.0 -0.0042, then **x6.0 = 1.002104 -> v31 = 1.002410**. Head LR x2 +0.0015 / x0.5 +0.0004;
  matrix LR x1.0 +0.0008; scalar LR x2 +0.0025; depth 12 +0.0018; prime table +0.0013; sign trick wash; EMA_START 0.75
  +0.0001; EMA_BLEND 0.5 +0.0001; WARMUP 25 -0.00015; WD 0.05 +0.0022 / 0.2 +0.0023; EMA decay 0.99 +0.0007; FINAL_LR 0
  +0.0004 / 0.05 +0.0015; batch 65k +0.0044. **Run-to-run noise on this line ~0.0003** — treat < 0.0005 as noise.
- **MLP ratio curve** 3/4/5/6/8 = 1.0091 / 1.0021 / 0.9982 / 0.9966 / 0.9965 -> ratio 6 -> v32 0.996429 (T43).
- **Review safety saga:** v32 was REJECTED_BY_REVIEW (a claimed unsafe `torch.load` fallback the file never had); the
  response was a clean minimal file — config hard-coded, no compile path, no unused branches, NKI imports in one guarded
  block, kernels at module level, plain `torch.load(..., weights_only=True)`, torch fallbacks only where optional ->
  v33 (portal 1.0069, 8th). Then HEAD=nki2 (kernel returns bf16 softcapped logits, CE in torch) -0.0064 at 600 s -> v34
  0.998477; v35 presents the full fused loss for review (0.996308, on-device |diff| 9.5e-7); v36 adds the plan-buffer ring.
- Bigram-table closure: split tables (`BIGRAM_SPLIT=2`: two half-width, independent hashes) wash; the sign trick
  (`BIGRAM_SIGN`) wash; 2^19 (0.999346 vs 0.998477) discard —
  the table's own step cost (+5%) exceeds its capacity gain; an in-context collision repair could touch at most 0.29% of
  positions -> dead. Frozen-table ensembles would have to win on raw capacity alone.
- Depth 8 / 9 / 12 all lose to depth 10 at this speed (+0.0014 / +0.0014 / +0.0018).
- **fp8 closed with numbers:** plain matmul kernels x1.00-1.04 (HBM-traffic bound; DMA 71% active, GpSimd 97% issuing
  53k software DMA descriptors); the PE-bound fused MLP x1.30-1.41 (54-59 -> 82 bf16-equivalent TFLOP/s). The step is not
  matmul-bound (all dense matmul work is ~80-100 ms of the 560 ms step), so fp8 could touch only the forward MLP
  (~6 ms/step) — not worth the numerics risk. Also not buildable: inside `torch.compile` an e4m3fn cast fails to compile
  and an e5m2 round trip is folded away.

## Sep 10: the kernel line (v37-v40) and its two dead ends
- v36 engine profiles -> `engine-profile-v36.md`. Forward 60.0 ms (PE 31%, DMA 50%, Vector 47%, Scalar 47%, mfu 19%);
  backward 144.4 ms (PE 34%, **Vector 61%**): 239k matmuls (84 ms; the 128-wide ones at 15 TFLOP/s), 144k tensor_tensor
  (69 ms), 182k PE transposes (36 ms), 221k stationary loads (25 ms). The backward is still a kernel CHAIN with six torch
  matmuls and compiler transposes per layer -> the fused `pre_attn_bwd` is the biggest identifiable lever (~60 ms/step, 10%).
- **Fused `pre_attn_bwd`** (key-offset bwd + QK-norm bwd + rope bwd + Q/K/V dgrad into one PSUM bank + wgrad + rmsnorm
  bwd, static tile loop, per-head DMA reads — no permutes): 2.78 ms per layer-microstep vs ~50 ms for the chain;
  **in-graph -13% (fwd_bwd 0.429 vs 0.509 s/step)**, losses identical. First attempt died on all cores with the OOB flake
  — the kernel's unaligned DMAs are lowered differently inside the model NEFF -> the 128-row-aligned-DMA rule. The fused
  post half was a wash (mlp_bwd already carried it). v37 0.992056.
- **16-row micro-steps** (v38 0.989999, 0.46 s/step) then **depth 12 pays at the new speed** (v39 0.986544, 0.52-0.54
  s/step): the re-spend that lost at every earlier speed (+0.011 / +0.0024 / +0.0018) flips positive once kernels make
  steps cheap.
- **Own attention kernels: PARKED after two device-length investigations.** Standalone they win: bwd v3 (one strided DMA
  per tensor) 1.80 ms vs vendor 2.69; fwd v4 0.67 ms vs vendor 1.14; correct on realistic inputs and inside a compiled
  graph (`ATTNGRAPH`). **In the model they NaN at random early steps on random ranks**; the one screen that synchronized
  after every layer never did. Bisection: each kernel deterministic alone, fwd+bwd in one graph nondeterministic ->
  **strided-pattern OUTPUT DMAs are mis-tracked in-graph** (rule #11); per-tile stores fixed that pair (fwd 0.82 / bwd 1.95,
  still faster than the vendor) but the model still NaN'd. The plain per-tile-load variants (fwd 1.24 / bwd 2.37) are
  slower than the vendor in-graph once the per-layer torch glue that builds their per-row inputs (sink/bounds expands,
  arange, -lse/ones) is counted. **Keep the vendor attention.** Also closed here: host-side doc bounds (+4%), scan-based
  doc bounds (no Neuron lowering for cummax), depth 14 (died to the flake twice -> dropped), FLAT_COMM, per-rank skew
  (ranks are equal; the cost is per-collective latency), XSA as torch glue (2x step — custom-op bodies execute eagerly at
  runtime, so extra elementwise ops are launch-bound), ADAM_ALT (halving the x6-LR embeddings' updates hurts).
- **DeltaNet hybrid = NOT VIABLE on this stack.** Chunked WY math is exact (1e-15 vs the recurrence); on the device:
  8/16 rows OOM at graph load, 4 rows compile but NaN at step 0, every mixer stage compiles and is finite ALONE but the
  whole mixer hits a neuronx-cc internal error (`NCC_IMGN901 MacroGeneration`) and silently miscompiles to NaN; eager mode
  is finite for 4 steps then NaN at 19 s/step. A Trainium DeltaNet needs a full NKI chunked fwd+bwd kernel (days) with
  unknown step cost. **The GPU-line gain (-0.0251 fixed steps; KDA a further -0.0131) has NOT been reproduced here**, and
  the local RTX port also diverges (with the params in the boosted scalar group at step 5, and at step 20 with a dedicated
  0.01 group) — **the port is the blocker, not Trainium**; the instability is in the training numerics.
- **v40 = v39 + attention-layout kernels** (T56 0.985161): `pre_attn_fwd_h` / `post_attn_fwd_h` emit/consume head-major
  (b*NH, t, HD) with the attention scale folded into q, removing four 16 MB compiler copies per layer per step (fwd_bwd
  0.451 -> 0.444); GLUE_PROF proved custom-op Python bodies run only during tracing, so the glue is compiler copies — the
  fix is layout, not code. ASYNC_COMM (async reduce_scatter) no gain.

## Sep 11-12: layout sweep, then flow (this file's end)
- **Overnight scalar sweep at depth 12** (82 lines, 600 s each; disk guard stopped it at line 23 — each screen writes a
  ~228 MB checkpoint): at depth 12 the matrix LR wants to be HIGHER and the embedding LR LOWER than the depth-10-tuned
  defaults — MATRIX_LR_SCALE 1.25 1.05867 / 1.0 1.05937 / 0.9 1.05994 / 0.8 1.06031 (all beat ref 1.062); EMBED_LR_SCALE
  4.0 1.05868 / 3.0 1.05911 (6-10 at or above ref); ATTN_SCALE 0.11 1.06031 (0.1 is fine). ~0.003 = ~3x noise.
- **Flow / rectified diffusion (user request): PARKED.** Implemented as a time-conditioned velocity field
  (`FLOW_STEPS`, `FLOW_SITE` = attn | mlp | block | prehead). First RunPod run diverged at step 4 — the flow params had
  inherited the boosted SCALAR_LR (the same bug as DeltaNet); fixed with a `FLOW_LR` knob (default 0.01, own AdamW group),
  after which flow is **stable but does not help** — four depth-8, 500-step local screens: base 1.197446; block lr0.01
  1.196959 (-0.0005); block lr0.003 1.197603; prehead 1.199086 (+0.0016). Best case a wash inside the noise band; the
  prehead site hurts. Knobs stay in the vehicle, off by default.
- **RunPod Flash pipeline** (`flash_app/app.py`, memory `runpod-flash-pipeline`): serverless RTX Pro 6000 Blackwell
  (102 GB VRAM, 2 TB RAM, torch 2.9 + cu128, ~11 s cold start); ships the code base64 in the request (private repo),
  runs `prepare.py setup` on the worker, returns val_bpb per config. Gotchas: needs the `{"input": …}` wrapper; training
  needs ALL of `PUBLIC_TRAIN_SHARDS` (range 5), not just `--num-train-shards`; kill stale uvicorns and `rm .flash/` when
  the endpoint changes (no hot reload). The flow sanity-run measured 11 GB VRAM, so full-scale runs were done locally;
  the worker is stopped.
- Handoff-era rules that ended the file: judge only by eval bpb (training loss lies under train/eval asymmetry); never
  ship the vehicle while a queue has unlaunched lines (they pick up the new source); never run `prepare.py eval-public`
  standalone on the box.
