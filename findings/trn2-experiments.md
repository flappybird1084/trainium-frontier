# Trn2 experiment campaign (real Trainium, fixed 1800 s wall-time) — target 1.15 val_bpb

Started 2026-09-05 on test4 (`trn2-access.md`). Master copy = this project dir; the box's `~/tf` is a rsync
mirror (never edit there). Results table: `trn2_results.tsv` (append-only). Hardware facts: `trn2-hardware.md`.
This is the Trainium counterpart of `train2-experiments.md` (the 3090 ladder) — same discipline: one change per
run, keep only on improvement, log washes and crashes too.

## Protocol
- **Scoring run:** full competition shape — `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8`,
  1800 s budget, `prepare.py`'s own `public_val_bpb`. ~35 min wall each (budget + ~4 min startup/eval).
- **Screening:** a shorter `MAX_TRAIN_SECONDS` proxy may be used to triage cheap ideas, but **nothing is
  "kept" until confirmed at the full 1800 s** (the metric is loss-per-wall-time; proxies can mislead).
- Launch from a plain detached shell on the box (`setsid nohup … &`) — never from inside a monitor/wait loop
  (an earlier run got SIGKILLed on monitor teardown). Log everything to `~/tf/runs/<id>.log`, pull back to `runs/`.
- Never call a run hung from log silence: ~2 min init + multi-min compile are normal (memory
  `trn2-startup-silent-gaps`). Tiebreaker `pgrep -af neuronx-cc`.

## Baseline (T0)
`public_val_bpb 1.258923` — unmodified train.py, 1000 steps, 65.5M tok, 1635 s charged, ≈1.55 s/step,
≈42k tok/s, **~40% NC utilization** (see trn2-hardware.md). Gap to target: **−0.109**.

## Plan (priority order — driven by the 40%-util headroom, then the 3090 ladder)
1. **Vehicle port.** Confirm the Trainium-ready submission vehicle runs on Neuron (flex_attention / torch.compile
   paths are CUDA-only and must not be on the Neuron path). Short smoke before any 30-min run.
2. **Utilization first (chip is latency-bound at 40%).** Micro-batch sweep 8192→16384→32768 tokens/rank (adjust
   grad_accum; try both "same total batch" and "bigger total batch"), watching tok/s and NC util in `util.log`.
3. **Width/depth to fill the cores.** 640→768/1024 wide, depth 10→12/16. Expect the Trainium optimum to be
   *wider* than the 256-wide 3090 champion.
4. **Port the 3090 ladder, one at a time, re-validated here:** value embeddings, hashed bigram table, AsymLogit
   softcap, Muon LR ×0.71 rule at larger batch, register tokens, trigram, narrow QK (expected NOT to transfer —
   it trades width for throughput, and we are throughput-rich here).
5. **Free micro-opts:** int32 token ids/positions (kills the int64→int32 autocast), `--nki-relu2` kernel.

## Log
| id | change | val_bpb | Δ vs T0 | verdict | notes |
|---|---|---|---|---|---|
| T0 | baseline | 1.258923 | — | anchor | 1000 steps, 40% util |
| T1 | champion v5 as-is, eager, portal-exact | **1.174335** | **−0.0846** | **keep — NEW BEST** | 2238 steps, 73.3M tok, 0.79 s/step, ~5% util |
| T2 | T1 + COMPILE=1 (torch.compile backend=neuron) | **1.151850** | **−0.1071** | **keep — NEW BEST** | 3235 steps, 106M tok, 0.54 s/step; 0.0019 from target |
| T3 | T2 + LNC=2 launch (4 workers × 2-core NCs) | **1.139926** | **−0.1190** | **keep — TARGET MET** | 3780 steps, 124M tok, 0.47 s/step, 17% util |
| T4 | FINAL GATE: exact submission shape, no env | **1.140051** | −0.1189 | **gate PASS** | reproduces T3; cold startup 273 s excluded |
| T9 | stacked params (v9-dev), compile + LNC=2 | **1.128411** | **−0.1305** | **keep — LOCAL BEST** | 4834 steps, 158M tok, 0.37 s/step; needs the portal to clear compile+LNC=2 |

### 2026-09-05 20:48 UTC — S1 smoke launched
`train3_submission.py` (the 3090 champion = whole kept ladder, 256-wide, batch 32768, DEVICE_BATCH 4 ×8 workers,
grad_accum 1) — **first ever run on real Neuron.** 240 s budget (`NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=240`,
`OUT_DIR=out_smoke`), 8-way LNC=1, detached via `setsid nohup` from `~/tf`, log `~/tf/runs/S1_smoke.log`.
Purpose: does the Neuron path (explicit fp32 attention + dense mask, int32 bigram hash, registers) run at all,
compile once (not per step), and what is its steady s/step and NC util vs the 640-wide baseline's 1.55 s / 40%.

**S1a/S1b (20:48–20:50): two instant crashes, both environmental, not code.** The user's baseline ran as `root`
and left root-owned compile caches in `/tmp` — `hlo_cache`, `local_cache`, `neff_cache` — so my `ubuntu` run died
on `Permission denied` opening their `.lock` files (S1a in NkiDispatch at 7 s; S1b as `COMPILATION FAILED: Failed to
open lock file /tmp/local_cache/locks/…` inside `init_process_group`). Fix: `sudo chown -R ubuntu:ubuntu` on all
three (root can still write to them). **Rule: whoever ran first on a box owns `/tmp/{hlo,local,neff}_cache`; a
different user must chown them or run as that user.** Logged in trn2-access.md.

**S1c (20:52): environment clean → first real code issue.** Got through init (`world_size=8`,
`micro_tokens=32,768 grad_accum_steps=1`) then `backend='neuron' raised: ValueError: Output 2 already aliases
input 123, cannot also alias input 0` — a dynamo aliasing guard on our forward graph. Suspect `x0 = x` (residual
skip) passed alongside `x` un-cloned. Investigating; fix must be math-neutral and keep the CUDA path identical.

**S1c root cause (20:55):** `COMPILE` defaults ON in train3 (`NEURON_COMPETITION_R1_COMPILE` unset → "1"), so
`torch.compile(backend="neuron", dynamic=False)` wraps the model — whereas the working baseline ran eager
(`compiled: False`). `x0 = x` in `GPT.forward` + compile graph breaks (python `skips` list, flex branch) → x/x0
become subgraph inputs → a pass-through output aliases two inputs → dynamo `ValueError`. **The portal sets no env
vars, so v4 as shipped would crash on the portal.** Fix in the file: `x0 = x.clone()` (math-neutral). Next:
S1d = eager (COMPILE=0) smoke; S1e = compiled smoke with the fix; pick the file's COMPILE default from measured
throughput, then the full 1800 s run.

### S1d (20:55–21:02) — **CHAMPION RUNS ON NEURON (eager).** 88 steps in 240 s, causality passed, val_bpb 1.810 (smoke only)
- `x0 = x.clone()` + `COMPILE=0` → clean run, `micro_tokens=32,768 grad_accum_steps=1`, 51.9M params.
- **True steady state 0.68 s/step ≈ 48k tok/s** (steps 20–50, `lrm`=1.000) — faster than the 640-wide baseline (1.55 s / 42k).
- **Recompile storm keyed to the LR value:** warmup steps 1–10 ≈ 4.05 s/step (lrm changes every step, neuronx-cc busy);
  step 60 (`lrm` 1.000→0.971, warmdown begins) = 5.0 s + neuronx-cc=8; every warmdown step ≈ 4 s. The LR is a python
  float baked into the NEFF as a constant → every distinct LR recompiles. train3's warmdown is the last 60% of the time
  budget → ~6× slowdown for 60% of the run. Would also have hit the portal. **Fix: tensor-valued LR (device 0-d tensor,
  in-place `fill_` per step) so the graph is invariant; same for any other per-step python scalar.**
- Startup: step 0 = 107.9 s compile (bigger graph than baseline's 78 s); startup_allowance_est 148.7 s.

**S1d utilization (util.log, 20:59–21:00, the 0.68 s/step phase): NC util ≈ 5% on all 4 cores, CPU ≈ 87%,
≈26k device executions/s (≈18k per step).** That is op-by-op eager dispatch: the champion's many small ops become
thousands of tiny kernel launches per step and the chip sits ~95% idle, starved by the CPU. (Baseline: 40% util —
simpler graph.) Device mem 34 GB of 96. **⇒ The dominant lever is graph fusion (`torch.compile`, which train3
defaulted ON for exactly this reason), now unblocked by the `x0` fix. Order: tensorize per-step scalars → S1e eager
(prove LR fix) → S1f compiled (the big one).** Per-step python scalars that can bake into NEFFs: `group["lr"]`
(AdamW `alpha=-lr`, `mul_(1-lr*wd)`; Muon `lr*g`), `muon_momentum(step)` (varies 300 steps), AdamW `1-beta**step`.

### S1e (21:07–21:12) — **LR/momentum tensorization WORKS.** eager, 240 s: **294 steps** (S1d: 88), val_bpb 1.4797 (smoke), causality OK
- Flat **≈0.79 s/step ≈ 41k tok/s through warmup AND the whole warmdown** — zero recompiles. Startup 46 s (warm cache; step 0 = 21.7 s).
- v5 patch: `MuonAdamW._dev_scalar` keeps lr/momentum as 0-d device tensors refreshed by host→device `copy_`; AdamW
  `p.sub_(lr_t*update)` (was `add_(alpha=-lr)`), Muon `lr_t`/`mom_t`. Bias corrections left as python floats (empirically
  didn't recompile; keeps CUDA numerics byte-identical). Loop unchanged.
- ~0.1 s/step slower than S1d's constant-LR phase (extra tiny scalar ops × eager per-op tax) — should vanish under compile.
- Projection, eager full run: (1800−46)/0.79 ≈ 2200 steps ≈ 72M tok @32768. 3090 ladder at ~2400 steps/32768 (T6b) gave 1.170.

### S1f (21:13) — COMPILE=1 still fails: same `Output 2 already aliases input 123, cannot also alias input 0`
So `x0 = x.clone()` was **not** the cause (harmless; kept). The message means two graph inputs share one storage and
the forward mutates/returns it — suspect a parameter registered under two names (tied weights: input 0 = first-used
param, input 123 = near-last). Investigating offline. **File default flipped to `COMPILE=0` (eager)** — the portal
sets no env, and compile crashes; eager is proven (S1e). v5 passes `prepare.py check` on the box.

### T1 launched (21:1x UTC) — FULL 1800 s, champion v5, eager, NO env overrides (exact portal shape)
`NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train3_submission.py`, log `~/tf/runs/T1_full.log`.
Expect ≈2200 steps ≈ 72M tok. Baseline to beat: 1.258923. Target 1.15.

**Alias root cause pinned (local CPU AOT probe, 21:30):** 124 graph inputs — input 0 = `idx`, input 123 = `targets`
(placeholders ordered by first use); **zero mutations**; **34 pass-through returns** (inputs saved for backward and
returned unchanged: `idx` for the embedding gather, `targets` for CE, scalar lambdas, matmul weights). The neuron
backend merges a pre-AOT alias analysis and a post-AOT pass-through list into one `AliasingInfo` (backend.py ~L204)
and raises on an output-index collision — here on the two *user* inputs. Workaround in v5: `idx.clone()`/
`targets.clone()` at the top of `GPT.forward` when training → saved tensors become intermediates. Test = S1g
(COMPILE=1, 240 s) right after T1.

**Tooling (21:35):** `train3_submission.py` gained dev-only env knobs (defaults == champion; portal sets none):
`NEURON_COMPETITION_R1_{DEPTH,N_EMBD,N_HEAD,MLP_RATIO,TOTAL_BATCH,DEVICE_BATCH}`. `TOTAL_BATCH` auto-applies the
T6b sqrt-LR rule (`_BATCH_LR = sqrt(TOTAL/32768)`). Box launcher `trn2_run.sh <ID> <secs|full> [ENV=..]`, status
`trn2_status.sh <ID>` (both rsynced to `~/tf`).

**Plan after T1 (device is serialized — one run at a time):**
1. **S1g** — COMPILE=1 240 s with the idx/targets clone. If compile works, measure s/step vs eager 0.79; if it's a big
   win, all sweeps below run under compile.
2. **Throughput screens (120–240 s each, read steady s/step + NC util from util.log):** the chip is ~95% idle in
   eager, so bigger micro-batches / wider models should be nearly free in wall time until util saturates.
   U1 `DEVICE_BATCH=8 TOTAL_BATCH=65536` (accum 1, 65k tok/micro) · U2 `DEVICE_BATCH=16 TOTAL_BATCH=131072` ·
   W1 `N_EMBD=512 N_HEAD=4` · W2 `N_EMBD=768 N_HEAD=6` · D1 `DEPTH=16` · combos of the winners.
3. **Full 1800 s runs** of the configs whose steady s/step stayed ~flat while params/tokens grew → val_bpb vs T1.
4. Then the ladder ablations whose verdict is regime-dependent: narrow QK, trigram, batch/LR rule.

### T1 (21:16–21:47) — **1.174335. Champion beats baseline by 0.085 on real Trn2, portal-exact launch.**
2238 steps / 73.3M tok in 1794.9 s charged (startup_allowance 27.8 s on a warm cache, eval 9.7 s), causality OK,
`over_budget: False`. Flat 0.75–0.81 s/step through the entire 60% warmdown (tensor-LR fix holds at full length).
Projection from the 3090 ladder (T6b, 2427 steps @32768 → 1.170) was within 0.004 — **the ladder transfers.**
Gap to target 1.15: **−0.024.** The chip is still ~95% idle (eager op-by-op), so the headroom is real: compile (S1g)
and/or bigger micro-batch / wider model at ~flat wall cost.

### S1g (21:47–21:55) — **torch.compile(backend="neuron") WORKS with the idx/targets clone. 1.5× step rate.**
240 s smoke: **433 steps** (eager S1e: 294), steady **0.52–0.55 s/step ≈ 62k tok/s** (eager 0.79 / 41k), val_bpb
1.3637 vs 1.4797 at equal wall time. Flat through warmup + warmdown (tensor-LR fix holds under dynamo). Cold
whole-graph compile: step 0 = 184.6 s → startup_allowance_est 210.3 s (< 900 s cap; excluded on the runner).
util.log during S1g: **NC ≈ 5–6%, CPU 85–95%, ~30k execs/s** — still dispatch-bound: forward/backward are fused
but the optimizer (Muon Polar-Express loops + AdamW per-param ops) is still eager op-by-op. More headroom.
**Decision:** compile is the mode. Next full run **T2 = champion + COMPILE=1** (projection ≈3300 steps → the 3090
ladder at that count gave ~1.14–1.15). File default stays eager until T2 passes all gates at full length.

### U1 (21:55–22:00) — eager batch screen: `DEVICE_BATCH=8 TOTAL_BATCH=65536` → 1.16 s/step @ 65,536 tok (56.5k tok/s)
Doubling the micro-batch cost +47% step time for 2× tokens (eager champion: 0.79 s @ 32k = 41.5k tok/s). NC util 7.2%,
device mem 57.6 GB (of 96). **Eager cost model: step ≈ 0.42 s fixed + 0.37 s per 32k tokens.** The fixed part is
~122 per-parameter `all_reduce` launches + ~300 eager optimizer ops + dispatch. Under compile (S1g 0.53 s @ 32k) the
compute part shrank but the fixed part remains → **attack the fixed overhead next:**
1. **Gradient bucketing** (installed in master 22:05, verified bit-equal to per-param AVG with a local 2-proc gloo test):
   one `all_reduce` per grad dtype. Screen S2 = COMPILE=1 + bucketing, 240 s, vs S1g 0.53 s/step.
2. If still overhead-bound: compile / foreach the optimizer step (Muon Polar-Express loops, AdamW per-param loop).
3. Then bigger micro-batch under compile (U1c) — larger steps get relatively cheaper as overhead falls.

### T2 launched 22:00 UTC — FULL 1800 s, champion v5 + COMPILE=1 (expected ~22:36)
- Bucketed `sync_gradients` verified **bit-exact** vs per-parameter `all_reduce(AVG)` and vs SUM/W truth (local 2-proc
  gloo, deterministic grads, every param both ranks: max|diff| = 0). (A first test showed False — harness RNG-replay
  artifact, not the code.)

**Compiled optimizer step (opt-in `NEURON_COMPETITION_R1_COMPILE_OPT=1`) validated locally (22:25):** with
`compiled=True, tensor_step=False` the compiled `MuonAdamW.step` matches eager **exactly (0.0)** over 4 one-step
comparisons from synced state (backend=eager and aot_eager). The tensor step counter (needed under compile so bias
corrections aren't python constants) contributes only fp32-`pow` rounding (~3e-7 rel); an earlier 1e-4 reading was
my harness computing grads before syncing state, amplified by AdamW/Muon normalization. Default path (knob off)
is unchanged: `init_state()` pre-allocation + eager `refresh_scalars()` produce identical numerics.

### T2 (22:00–22:31) — **1.151850. Compile at full length: 3235 steps / 106M tok, all gates green. 0.0019 from target.**
Champion v5 + `COMPILE=1`, portal launch shape. 1794.7 s charged (startup 39.6 s on a warm cache), eval 9.7 s,
causality OK, `over_budget: False`. −0.0225 vs T1 (eager) from 1.45× the steps. **File default flipped to
`COMPILE=1`** (validated end-to-end; cold compile ~210 s < 900 s cap; portal sets no env).
Next: S3 (running) = compile + bucketed all_reduce + compiled optimizer — targets the ~0.4 s/step fixed overhead.

### S3 (22:31) — compiled optimizer FAILED in the neuron Torch-MLIR lowering: `torch._foreach_copy_` (Muon write-back,
line 646) is captured as an opaque `torch.operator` that cannot be legalized to StableHLO (eager decomposes it; compile
does not). Fix: per-parameter `p.copy_(w[i])` loop (math-identical). Retest = S3b. S2 (compile + bucketing only)
auto-launched meanwhile to isolate bucketing's effect.

### S2 (22:33–22:38) — compile + bucketed all_reduce: **wash/slightly worse** (0.56 s/step, 416 steps vs S1g 0.53 / 433).
Reverted to per-parameter all_reduce (T2-validated). The fixed per-step overhead is therefore not the collectives;
it is the eager optimizer (~300 ops) + CPU dispatch (CPU ≈95%, NC ≈6%). S3b (compiled optimizer, foreach fix) is
the decisive test.

### S3b (22:38–22:44) — compiled optimizer works after the foreach fix but is **slower: 0.65 s/step vs 0.53** (discard)
Steps 1–4 ≈ 12 s each (optimizer-graph recompiles), then 0.62–0.67. util.log unchanged: NC ≈6%, CPU ≈95%, the same
~29k device executions/s (~1.2k per worker per step) — compiling did NOT cut the execution count, i.e. the neuron
dynamo backend partitions into many small graphs ("partial lowering"). **Conclusion: the fixed per-step cost is not
something compile/bucketing remove. The chip is ~94% idle → do more useful work per step at ~flat step cost:
wider model and/or bigger micro-batch.** This is the opposite of the 3090 optimum (256-wide, "smaller & faster").
Screens: W1 = `N_EMBD=512 N_HEAD=4` (head_dim stays 128) under compile; U1c = `DEVICE_BATCH=8 TOTAL_BATCH=65536`
under compile. Judge by steady s/step AND 240 s val_bpb (S1g 1.364 vs S1e 1.480 tracked the full-run ordering).

### W1 (22:46–22:52) — 512-wide under compile: **cold compile 362 s** ate the 240 s screen (5 steps, 0 train-s) — but the
5 steps say **0.68 s/step for 119.6M params vs 0.53 s for 51.9M** (+28% time for 2.3× model), NC 6.9%, dev mem
74.5 GB/96 (768-wide may not fit). Wider is ~free, as predicted for this overhead-bound regime. **Screen rule: a new
shape's first run only warms the NEFF cache; measure on the second run (W1b).** The 1800 s run is unaffected (cap 900 s).
- Why W1 stopped at 5 steps: the predictive stop tests `elapsed + max(recent_step_times[-20:]) + 5 ≥ budget`, and at
  boundary 5 that window still holds step 0's ~300 s cold-compile dt → `0 + 300 + 5 ≥ 240` stops a 240 s screen
  immediately. Irrelevant at 1800 s (300 ≪ 1800). Rule stands: first run of a new shape warms the cache, second measures.

### U1c (22:52–22:59) — 65k tok/step under compile, cold cache (5 steps): **0.87 s/step = 75k tok/s** (+22% vs 62k at
32k tok/step). Doubling the micro-batch costs +64% step time under compile (eager: +47%). Rerun warm = U1c-b.

### W1b (22:59–23:04) — 512-wide warm: 0.69 s/step, 334 steps, **240 s val 1.384 vs champion 1.364** (worse here).
The 240 s proxy favors step count (loss ≈3.8, per-step progress dominates). Not ruled out for 1800 s (step ratio 77%).
**Bigger insight from util.log: CPU ≈95% with 8 dispatching workers on 12 vCPUs — the host is the bottleneck, not the
chip.** → Try the sanctioned **LNC=2 launch (4 workers × 2-core logical NCs)**: half the dispatch loops, 4-way
collectives, same tokens/step (`DEVICE_BATCH=8` → 4×8×1024 = 32,768). Screen L2 (cold) then L2b (warm).
- LNC=2 is a sanctioned launch: README §"lnc=2" gives `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone
  --nproc_per_node=4 train.py`, the submission carries a participant-chosen launch-command string (README L204), and
  `prepare.py check` passes with it. `trn2_run.sh` now takes `LNC=` / `NPROC=`.

### U1c-b (23:04–23:09) — 65k tok/step warm: 0.89 s/step, 265 steps, 17.4M tok, **240 s val 1.3642 vs champion 1.3637** — wash.
Bigger batch buys +22% tokens/s but loses 39% of the steps; at this horizon they cancel exactly. Parked (possible
combo partner later, e.g. with a wider model where steps matter less).

### L2 (23:09–23:16) — **LNC=2 launch: 0.47 s/step (70k tok/s) vs 0.53 s at LNC=1** on the 5 cold-cache steps.
`world_size=4`, `micro_tokens=32,768 grad_accum 1` (DEVICE_BATCH=8). −12% step time for identical work: fewer
dispatching processes + 4-way collectives. Projects to ≈3700 steps / 1800 s (T2: 3235). Warm rerun L2b to confirm.

### L2b (23:16–23:21) — **LNC=2 confirmed: 0.47 s/step flat, 499 steps, 240 s val 1.3382 vs 1.3637 (−0.026).** causality OK.
util.log: NC ≈17% (was ≈6%), CPU ≈70% (was ≈95%) — the host dispatch was the bottleneck; 4 workers on 2-core logical
NCs halve it. **T3_full launched 23:21 UTC: LNC=2, 4 workers, DEVICE_BATCH=8, COMPILE=1, 1800 s** (ETA ~23:54).
Submission consequence: the launch-command string becomes `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone
--nproc_per_node=4 train.py` (sanctioned, check passes). File change: `DEVICE_BATCH` default now derived from world
size (8→4, 4→8) so the portal (no env) gets grad_accum 1 under either launch. Final gate = T4: exact portal shape,
no env, LNC=2.

### T3 (23:21–23:52) — **1.139926 — TARGET 1.15 MET.** LNC=2 launch, 3780 steps / 123.9M tok, all gates green.
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4`, DEVICE_BATCH=8 (32,768 tok/step, accum 1),
COMPILE=1. 1794.6 s charged (startup 35.6 s warm), eval 7.0 s, causality OK, `over_budget: False`. −0.0119 vs T2 from
+17% steps at identical per-step math. Ladder so far on real Trn2: baseline 1.2589 → eager v5 1.1743 → compile 1.1519
→ **LNC=2 1.1399**. **T4_gate launched 23:52: the exact submission shape (file defaults, NO env, LNC=2 launch).**

## Plan after the gate (beyond target; device serialized)
1. **W3L: width 384 (`N_EMBD=384 N_HEAD=3`, head_dim 128) under LNC=2** — regime-driven: the chip is still 83% idle,
   and 384 cost only ~10% of steps on the 3090 (T4 vs T5) where it was nearly a wash; here steps are cheaper still.
2. **Trigram hashed table port** (L1 on the 4080: −0.0042 alone; a gather is ~free here).
3. **65k batch under LNC=2** (`DEVICE_BATCH=16 TOTAL_BATCH=65536`) — was a wash at LNC=1; cheaper steps may tip it.
4. **Narrow QK re-validation** (L2 on the 4080: −0.0075, but the win came from +27% steps on a FLOP-bound GPU;
   expected wash here — the champion file does NOT include it).
Each: cold-cache warm run → warm 240 s screen → full 1800 s only if the proxy is ≤ the incumbent's.

**Trigram port (00:0x):** env-gated `NEURON_COMPETITION_R1_TRIGRAM=1` → `GPTConfig.use_trigram` (saved in the checkpoint
so `load_for_eval` rebuilds it): 2^17×n_embd table on the int32 hash of (prev2, prev, cur), own 0.2 gate, added to
the input embedding beside the bigram. Flag off is param-identical to v5 (verified locally); flag on adds 33.6M params.

### T4_gate (23:52–00:27) — **GATE PASSED: 1.140051.** Exact submission shape: file defaults, no env, LNC=2 launch.
`world_size=4`, `micro_tokens=32,768 grad_accum 1` (derived DEVICE_BATCH=8), compile on. 3776 steps / 123.7M tok,
causality OK, `over_budget: False`, startup 273.5 s (cold path, excluded under the 900 s cap; total 2053 s).
Reproduces T3 within 1e-4. **Frozen as `submission_v5_lnc2.py` (sha256 912e282cc04dd38b…)** with launch string
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`; `prepare.py check` passes as `train.py`.
(T4 ran the file minus the default-off trigram knob added afterwards; flag-off is param- and path-identical, verified.)

### TR1 (00:27–00:32) — trigram table under LNC=2, cold (5 steps): **0.68 s/step vs 0.47** (+45%). 85.5M params.
The 2^17×256 table's gather/scatter-add backward and its AdamW update add ~0.2 s of eager work per step in this
dispatch-bound regime (on the 4080 it was nearly free). Would need to buy back ~30% fewer steps. TR1b (warm) for the
240 s proxy vs L2b's 1.3382.

### TR1b (00:32–00:37) — trigram warm: 0.68 s/step, 351 steps, **240 s val 1.3904 vs 1.3382 — discard.**
L1's −0.0042 on the 4080 came with a near-free step; here the 2^17 table costs +45% step time. Rule of this regime:
**extra ops are never free — every addition must pay for its dispatch cost in steps.** Knob stays (default off).

### W3L (00:37–00:45) — 384-wide under LNC=2, cold (5 steps): **0.59 s/step (56k tok/s) vs 0.47** for 1.6× params (83.8M).
→ ~3000 steps at 1800 s = 79% of the champion's 3780. On the 3090, 384-wide at 69% of the steps lost only 0.002 (T4 vs
T5); here the step ratio is better, so a full-horizon win is plausible even if the 240 s proxy (step-biased) says
otherwise. W3Lb (warm) next; full run T5 if the proxy is within ~0.01 of 1.3382.

### W3Lb (00:45–00:50) — 384-wide warm: 0.58 s/step, 405 steps (81%), **240 s val 1.3519 vs 1.3382 (+0.014)**.
Outside the 0.01 auto-threshold, so U2L (65k batch under LNC=2) went next. Caveat: the 240 s proxy penalized
512-wide by +0.020 at 77% steps while the 3090 full-horizon penalty for 384 at 69% steps was only 0.002 — the proxy
is harsher on width than the full budget. A full 384-wide run (T5) remains the principled test; queued after the
batch screens.

**Narrow-QK knob (00:5x):** `NEURON_COMPETITION_R1_QK_DIM=64` → `GPTConfig.qk_dim` (checkpoint-saved): c_q/c_k at
n_head×qk_dim, v at head_dim, k_sink and rotary at qk_dim, scores scaled by √qk_dim. Flag off is param-identical to
v5 (verified); flag on trains + round-trips. Expected wash here (its 4080 win was throughput-driven).

### U2L (00:50–01:03) — 65k tok/step under LNC=2, cold (5 steps): **0.70 s/step = 94k tok/s** (+34% vs 70k). 68% of the
steps for 35% more tokens. **Cold compile 732 s** — under the 900 s cap but with only ~170 s margin; a slower portal box
could push the excess into charged time. Submission-risk flag on this config. U2Lb (warm) for the proxy.

### Incident 01:03–01:22 — double launch + wedged workers (my error; ~20 min of device time lost)
U2Lb (the "warm" 65k-batch rerun) was NOT warm: I had rsynced the narrow-QK edit between its cold and warm runs, and
**torch.compile's cache is keyed on the model source — any edit to the file forces a full recompile** (also why T4's
startup was 273 s after the DEVICE_BATCH edit). My poll chain's "wait for procs=0" was a soft timeout, not a gate, so
NQ1 launched on top of the still-compiling U2Lb; NQ1's workers then failed `NRT: Logical Neuron Core(s) not available
… cores busy` and sat wedged in futex (they never retry). Killed both; relaunched NQ1 alone.
**Rules adopted:** (1) launches are hard-gated on `pgrep train3_submission == 0` AND `neuron-ls` showing zero owners;
(2) never rsync model-code edits between a cold and a warm screen; (3) U2Lb still needs a clean warm rerun (its config
also carries the 732 s cold-compile risk, so it is low priority).

### Incident 01:22–01:4x — **kernel oops in the neuron driver; box rebooted**
The four NQ1 workers that failed core allocation ended up in state R spinning inside the driver
(`neuron_log_dump` ← `ncdev_misc_ioctl` [neuron]); dmesg shows `RIP: neuron_log_dump … [neuron]`. Processes spinning
in kernel mode never return to userspace, so SIGKILL cannot land and the module cannot be reloaded. Only remedy: reboot.
Trigger chain: a run launched while another held the cores → `NRT: Logical Neuron Core(s) not available` → the
driver's error-path log dump oopsed. (Vendor-reportable: capture in `runs/trn2/kernel_oops_20260906.txt`.)
Mitigation before reboot: `/tmp/{hlo,neff,local}_cache` copied to `~/neuron_cache_backup` (Ubuntu cleans /tmp at boot).
Risk accepted: the first-boot deadlock race may recur; the submission is already validated and frozen, so the downside
is bounded to the remaining bonus experiments (NQ1b, U2Lb, T5).

## Session 2 (2026-09-06 UTC, after the reboot) — experiments continue while the submission is pending
Housekeeping: local junk cleared (old checkpoint dirs, stale root logs, caches); `runs/` and `submit/` gitignored.
Box: `~/tf/out_*` screen checkpoints cleared except `out_T4_gate`. Rules in force: hard-gated launches; no source
edits between a cold and a warm screen (current source = v5 + trigram knob + QK knob, both default off).
Queue: H1 health check (champion, LNC=2; also warms the champion cache for this source) → NQ1/NQ1b (narrow QK) →
T5 (384-wide full) → U2Lb (65k batch warm) if time.

### PORTAL FAILURE sub-9ea276d1 (submitted 9/5 18:57 PDT): `USER_TRAINING_CRASHED exit=1 status=no_run_meta (no train.log)`
Diagnosis work (02:1x–02:4x UTC): ruled out `--compile-only` (exact bytes + exact launch run it with exit 0 here),
harness-set env collisions (baseline reads only COMPILE/EVAL_*/MAX_TRAIN_SECONDS/NKI_RELU/NUM_STEPS/OUT_DIR), launch-shape
validation (prepare.py's own validator accepts LNC 1|2, nproc 1|4|8), and a missing end-of-run artifact (baseline writes
only final.pt). **Pattern found: the two `no_run_meta` failures (v2 9/2, v5 9/5) both defaulted `torch.compile` ON; the
organizers' baseline is eager.** Compile needed two backend-bug workarounds on our …0731 build; the portal runs the
newer …0824 build. Leading hypothesis: compile crashes there. "(no train.log)" likely just means no log collection
after a crash. **v6 hardening:** compile default OFF; compile failures at torch.compile() or at the first step fall back
to eager; runner-contract import guarded. Frozen `submission_v6_eager.py`. Gate T6 = v6 bytes, eager, LNC=2, no env.
- 02:25: my first v6 patch script had a quoting bug and raised SyntaxError before applying anything; the "fallback
  test" therefore ran plain inductor and T6 launched with the UN-hardened file. Caught it, killed that run (device
  cleared cleanly), re-applied with per-replacement verification. Real v6 = `submission_v6_eager.py`
  (sha256 4691c9d5f266fb0d…): bogus-backend test degrades to eager and trains; no-env default runs eager.
- **T6_gate launched 02:29 UTC: v6 bytes, eager (default), LNC=2 launch, no env** → the validated number for an
  eager resubmission. ETA ~03:05.
- 02:5x: user noticed the file renders with mojibake in a plain-text viewer: 11 non-ASCII chars (em-dashes, arrows,
  minus signs) in comments/docstring. Baseline train.py is pure ASCII; the failed v2 also had non-ASCII. Not proven
  as the crash, but a harness reading the file under a C locale is a plausible trap. Stripped to pure ASCII
  (code AST verified identical). v6 re-frozen; `submit/train.py` updated. T6_gate (running) used the pre-strip
  bytes; comments do not affect execution, so its number stands for the ASCII file.

### T6_gate (02:29-03:00) - **v6 eager, LNC=2 launch, no env: 1.155026.** 2894 steps / 94.8M tok, 0.62 s/step, causality OK, exit 0.
### S8 (03:00-03:08) - ASCII v6, LNC=1 / 8 workers, eager, no env (except a 240 s budget): exit 0, world_size 8,
DEVICE_BATCH derived to 4 (micro 32,768, accum 1), 292 steps, causality OK. The 8-way path of the final bytes works.
**User's clue (7:54 PDT): the portal run sat in "executing" for a few minutes before dying.** That is the profile of
a compile-phase failure (cold torch.compile takes 3-5 min here, then a backend rejection raises -> exit 1), not a
launch/import error (seconds). Both files the user remembers scoring ran eager. v6 = eager default + fallback + ASCII.
Resubmission candidates (both validated on our box with v6 bytes): LNC=2/4-way = 1.155 (T6); LNC=1/8-way ~ 1.174 (T1
number, v5 bytes; S8 confirms the v6 8-way path). Top-10 cutoff on the board is ~1.164 -> LNC=2 lands 9th, 8-way 11th.

### 03:10 UTC - standalone scorer path verified on Neuron: `prepare.py eval-public` (fresh process, load_for_eval hook)
on T6's checkpoint with the v6 bytes: val_bpb 1.15536 vs 1.15503 in-run; 151 s single-process eval. Checkpoint
format + load_for_eval are fine on the device.

### PORTAL FAILURE #2: sub-39b23de5 (v6, eager, ASCII, LNC=2 launch, 8:15 PM PDT) - IDENTICAL status
`USER_TRAINING_CRASHED exit=1 status=no_run_meta (no train.log)`. **Compile hypothesis falsified** (v6 is eager);
encoding falsified. Shared by v2/v5/v6 and absent from the two scored files: v5+v6 = LNC=2 launch (v2 was LNC=1 and
had its own top-level flex import); PACK=4 loader rows of 4096 tokens (post-v1, worth -0.0013); per-step elapsed
broadcast. "Few minutes then exit 1" fits a failure at the first device touch (LNC=2 core allocation refused on a
runtime pinned to LNC=1 would look exactly like that). Own bug found: derived DEVICE_BATCH = 32//world_size puts 32
rows on one core if the harness runs nproc=1 (a sanctioned shape). v7 = v6 + PACK off + rows capped at 8, for the
8-way launch that has scored twice. Two slots are gone on untestable hypotheses: organizer stderr is now the priority.
- v7 built (03:2x): PACK default 1 (baseline loader shape; env knob keeps 4 for experiments), per-worker rows =
  min(8, 32//world_size). CPU: nproc=1 -> micro 8,192 accum 4; nproc=2 -> micro 16,384 accum 2. Frozen
  `submission_v7_eager_8way.py`; `submit/train.py` = v7. Next: box check + 8-way 240 s smoke (S9) + full 8-way gate (T7).
- S9 (03:29-03:34): v7 bytes, LNC=1 / 8 workers, eager, no env: exit 0, world_size 8, micro 32,768 accum 1, 292 steps,
  causality OK, 240 s val 1.445. Passes `prepare.py check` with the 8-way launch. **v7 is the resubmission
  candidate: `submit/train.py` (sha 71404a2b30fe0426) + `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone
  --nproc_per_node=8 train.py`.** Note: eager runs did not pay a recompile after the source edit (26.6 s startup):
  the op-by-op NEFF cache is keyed on ops, only torch.compile's cache is source-keyed. T7 = full 8-way gate of v7.

### T7_gate (03:34-04:0x) - v7 bytes, eager, PACK=1, LNC=1 / 8 workers, no env: **public_val_bpb 1.176902**, 2214 steps, 72.5M tok.
This is the exact number for the resubmission candidate (submit/train.py sha 71404a2b30fe0426 + 8-way launch).

### T8 launched 04:0x: 384-wide (N_EMBD=384 N_HEAD=3), eager, PACK=1, LNC=1 / 8 workers, full budget - the next
candidate in the portal-safe shape (to beat T7's 1.1769 for that shape). Uses train3_submission.py (= v7 bytes) + env.

### 04:1x UTC - PORTAL SCORED sub-ae5cf1a6 = v7: **1.1827** (local T7 1.1769; portal = local + 0.006). Bisection plan in
playbook. S10 = v7 bytes under LNC=2 (240 s smoke) to de-risk the next slot; T8 (384-wide 8-way) paused for it.

## Session 3 (2026-09-06 04:xx UTC) - throughput campaign after the portal score
- P1: compiled fwd+bwd lowers fully with `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1` -> no unsupported-op splits.
- dynamo: forward = 1 graph, 0 breaks, 503 ops (CPU `torch._dynamo.explain`).
- fp8: `torch._scaled_mm` works on the device (e4m3fn, e5m2); bf16 4096x1024x1024 matmul 0.18 ms (~48 TFLOP/s/core).
- **PR1b profile (compile, LNC=2, 0.475 s/step): fwd_bwd 0.276 (58%) | optimizer 0.180 (38%) | grad_sync 0.019 (4%).**
  Two targets: attention-heavy compiled graph (~5% MFU) and the eager optimizer.
- Optimizer fix (04:5x): compiled step no longer reads the per-step python floats (dynamo re-guarded and recompiled
  every step in S3b); reads only lr_t/momentum_t tensors. Screen S3c next.
- PR2 (launched 04:54): ATTN=sdpa + TORCH_NEURONX_ENABLE_NKI_SDPA=1, compile, LNC=2, profile, budget 600 (cold graph).
- PR2 (04:54-05:07): SDPA + NKI-SDPA flag, compile, LNC=2: **fwd_bwd 0.267 vs 0.276 explicit (-3%)**, total 0.467 s/step.
  Attention fusion is not the lever as implemented (kernel likely not engaged with a bool attn_mask, or attention is
  not the dominant cost). Next: S3c compiled optimizer (float-read fix), then op-level ablations of the forward
  (value-embedding tables, bigram gather, registers, U-net skips) to find what the 0.27 s actually is.
- S3c (05:09-05:13): compiled optimizer even with the float-read fix: **optimizer 0.218 s vs eager 0.181** (steps 1-4
  still 8-12 s). The backend runs a graph of many small in-place state updates slower than eager dispatch. Retired.
  Optimizer cost is launch count (~180 stacking copies for Muon + ~110 AdamW ops + 122 all_reduce + 14 scalar
  copies ~ 420 launches x ~0.4 ms). Structural fix = keep matrix params permanently stacked (views) so Muon and the
  all_reduce run on one tensor per shape group: saves ~180 copies + ~110 collectives, est. -0.1 s/step (~+25% steps).

### Stacked-parameter refactor (05:2x-05:4x UTC) - `submission_raw/train3_submission.py` (v9-dev)
Matrix weights live in one (n, out, in) Parameter per shape (`stacks.w256x256` etc., `StackedLinear` views); the five
per-block scalars live in one tensor each (`block_scalars.*`). Muon updates the stack in place (no 180 copies/step),
the all_reduce is one collective per stack, AdamW sees 9 tensors instead of 54. **Bit-exact vs the pre-stack reference
on CPU** (`train3_prestack_ref.py`): same param count, identical forward loss, identical params after 3 steps
(0.000e+00), dynamo 1 graph / 0 breaks, checkpoint round-trip OK. Optimizer tensors 120 -> 18 at 10 layers.
Checkpoint layout changed (self-consistent via load_for_eval). Screen PR3 = compile + LNC=2 + PROFILE.
- **PR3 (05:24-05:31): stacked vehicle, compile, LNC=2: 0.373 s/step (was 0.475). optimizer 0.074 (was 0.181), grad_sync
  0.007 (was 0.019), fwd_bwd 0.292 (was 0.276: index-backward scatter into the stacks costs ~0.016).** 87.7k tok/s.
  Projection at 1800 s: ~4800 steps -> ~1.135 local (3090 curve) -> ~1.14 portal (#9). fwd_bwd is now 78% of the step.
  Next: PR4 eager 8-way profile (portal-proven shape), T9 full run (compile+LNC=2), Neuron profiler on fwd_bwd.
- PR4 (05:40): stacked vehicle under EAGER 8-way: 0.854 s/step (v7 eager ~0.80). optimizer 0.135, sync 0.044, but
  fwd_bwd 0.675: eager autograd through `stack[i]` (select backward) allocates a full (n,out,in) zero tensor per
  layer per step; compile fuses this away. Fix in progress: custom autograd Function that accumulates the weight
  gradient directly into `stack.grad[i]` (eager path only; compile keeps the plain view).
- T9 launched: stacked vehicle, compile + LNC=2, full 1800 s (the number for when the portal clears that launch).
- Eager-efficient stacked backward (05:5x): `_StackSliceLinear` autograd Function accumulates each layer's weight
  grad into `stack.grad[idx]` (no full-size zeros); used when eager (`STACK_GRAD_MODE="function"`), compile keeps the
  plain view. **Bit-exact** vs view mode and vs the pre-stack reference over 3 CPU steps (0.00e+00). Screen PR5 (eager
  8-way profile) after T9.

### T9 (05:45-06:17 UTC) - **1.128411, local best.** Stacked vehicle, compile + LNC=2, full 1800 s: 4834 steps / 158.4M
tok at 0.37 s/step (T3: 3780 steps, 1.1399). Causality OK, over_budget False, startup 37.9 s. Portal projection ~1.134
(8th on the 2026-09-05 board) - conditional on the portal accepting compile and the LNC=2 launch (bisection pending).
- **PR5 (08:52): stacked vehicle + function-mode backward, EAGER 8-way: 0.587 s/step (v7: ~0.80).** fwd_bwd 0.412,
  optimizer 0.133, sync 0.043. Frozen as `submission_raw/submission_v9_eager_8way.py` (= the vehicle at this point).
  T10 = full 8-way eager run of it (no env) for the portal-shape number. User targets: 1.05 then 1.00 (local).

### Neuron profiler (09:40, single process, compile, LNC=2, 0.379 s/step) - `neuron_profile_step.py`, trace in runs/nprof
- `model_default` (compiled NEFF executions): 4 per step, ~269 ms/step = the fwd_bwd cost. The compiled model is
  TWO regions (0 and 2): `_CrossEntropyAutograd` (a custom autograd Function behind F.cross_entropy on Neuron) breaks
  the graph; an eager `aten::ne` (ignore-index mask) costs ~38 ms/call and ~180 neuron->cpu copies/step surround it.
- `neuron::copy::cpu_to_neuron` ~280 ms/step: each of the optimizer's 14 scalar H2D copies synchronizes with the device.
- `neuron::alloc::lazy` ~2,000/step at ~170 us; `RuntimeWorker::sleep` dominates CPU self time (host-bound waits).
Fixes (both verified on CPU): `CE_IMPL=manual` (log_softmax+gather+mask; mean diff 9.5e-7, per-token and grads exact)
and ONE packed H2D scalar copy per step with per-group views (bit-exact). Screen PR6.
- PR6 (09:48): manual CE + packed scalars, compile, LNC=2: **0.362 s/step** (fwd_bwd 0.284, optimizer 0.071). Small (-3%),
  so the profiler's copy/CE-region time was overlapped, not waiting. Both made default. The compiled fwd_bwd (~0.28 s,
  ~5% MFU) is intrinsic NEFF efficiency at d=256 (small K on 128x128 arrays), not graph structure. Next lever without
  kernel work: width (T11 = 384-wide full run at compile+LNC=2), then batch 65k (T12).

## Ideas backlog (2026-09-06 10:1x UTC) - from the NanoGPT speedrun and OpenAI parameter-golf tracks, assessed for Trn2
Already in the champion: Muon (Polar Express), QK-norm, RoPE, ReLU^2, zero-init projections, untied embeddings, value
embeddings (3 tables), U-net skips, logit softcap (asymmetric, learned), attention registers/sinks, doc-masked
attention, hashed bigram table (golf), tensor-LR warmdown keyed to time, batch 32768 with sqrt-LR.
Applicable, untried here, ranked by expected value / cost:
1. **Width/batch retune at Trn2 throughput** (T11 384-wide running; T12 65k batch next). The 3090 optimum (256, 32k)
   was set by GPU throughput; here matmul efficiency rises with width and per-kernel tokens.
2. **FP8 for the big matmuls** (`torch._scaled_mm` works on the device): only pays once MFU is above ~20%; today 5%.
3. **Batch-size warmup** (small total batch for the first ~10-15% of time, then 32k/65k): more optimizer steps early
   at the same tokens; speedrun reports a few thousandths. Cheap to implement (loader rows fixed; accumulate less).
4. **Muon on the embedding tables** / higher embedding LR schedules: speedrun keeps AdamW there; low EV.
5. **Sliding-window / long-short attention**: seq 1024, no; **sparse attention gates**: E20 blew up (LR); low EV.
6. **Trigram / larger hashed tables**: failed here (TR1b: +0.052 at the proxy) because extra launches are not free.
7. **NKI kernels for attention / fused MLP**: the real path to >20% MFU and the 1.0x range; multi-day.
Not applicable / disallowed: pre-trained weights, external data, test-time tricks (pure stateless forward required).

## T11 interim (10:15 UTC) + tonight's queue
T11 (384-wide, 3 heads, 83.8M params, compile + LNC=2) runs 0.48 s/step (~68k tok/s) vs T9's 0.37 s/step at
256-wide: +50% width = +125% matmul FLOPs for only +30% step time -> the compiled NEFF is still overhead-bound,
so width is cheap here. At matched WALL TIME (1336 s remaining) T11 loss 3.40 vs T9 3.53-3.58: clearly ahead.
CPU smoke (local, fp32, 2 steps): 384/3h = 83.8M, 512/4h = 119.6M, 640/5h = 159.3M params - all construct+train.
Queue (full 1800 s runs, compile + LNC=2, one variable each vs the running best; hard launch gate every time):
- T12: 512-wide / 4 heads (width lever continued).
- T13: best width + TOTAL_BATCH 65536 (DEVICE_BATCH 16 at 4 workers; sqrt-LR rule gives the 65k champion LRs).
- T14: best width, EAGER LNC=1 / 8 workers = the portal-proven shape (needed for a safe submission until the
  compile/LNC=2 bisection clears on the portal).
- Then: depth (12) at the best width; MLP ratio; 640-wide if 512 still improves.

## T11 RESULT (10:38 UTC): 384-wide = **1.106531** - NEW LOCAL BEST (T9 256-wide: 1.128411, -0.022)
3717 steps / 121.8M tokens in 1794.8 s charged (0.48 s/step); startup allowance 343.8 s (cold compile of the
new shapes); causality ok; eval 8.7 s. Width is the strongest single lever found on Trn2 so far, and it
confirms the profiler reading: the compiled step is overhead-bound, so FLOPs are nearly free. Next: T12 512-wide.

## T12 RESULT (11:18 UTC): 512-wide = **1.088742** - NEW LOCAL BEST (T11 384-wide 1.106531, -0.018; T9 256-wide 1.128411)
3649 steps / 119.6M tokens; **0.49 s/step - essentially the same as 384-wide (0.48)**, so the 256->384 jump
(0.37->0.48) was the head-count/shape change and the FLOPs themselves are still ~free. Startup allowance est
550.5 s (cold NEFF compile of the 512 shapes; the runner cap is 900 s - keep an eye on it for wider models:
a cold portal box pays this once, and anything over the cap is charged to the budget). Causality ok.
Width curve on Trn2 (compile+LNC=2, 1800 s): 256 1.1284 -> 384 1.1065 -> 512 1.0887. Still descending steeply.
Launched T13 = 640-wide / 5 heads (159.3M) at 11:19 UTC.

## v10 candidate staged (11:21 UTC) - portal-safe shape at the new width
`submission_raw/submission_v10_eager_8way.py` (sha256 6dd554d720ae421e) = the vehicle with N_EMBD 512 / N_HEAD 4 baked in as
defaults (2-line diff; COMPILE default stays 0 = eager; PACK 1; DEVICE_BATCH derives 4 rows at 8 workers). `prepare.py check`
passes. It is NOT in `submit/` yet: T14 must first run these exact bytes with the exact portal launch (eager, LNC=1, 8-way,
no env) and beat T10's 1.160509. Hypothesis: eager 8-way is CPU-dispatch-bound (95% CPU), so the wider ops are nearly free
there too. `trn2_run.sh` gained a `FILE=` override for running frozen files (default is still the vehicle).

## T13 RESULT (12:04 UTC): 640-wide = 1.094465 - WORSE than 512 (T12 1.088742). Width optimum = 512.
2808 steps / 92.0M tokens at 0.63 s/step (512: 0.49; 384: 0.48): at 640 the matmuls finally bind, and the fewer
steps outweigh the better loss/step. **Startup allowance est 893 s** (cold NEFF compile of the 640 shapes) = at the
runner's 900 s cap: anything wider would have compile time charged to the budget on a cold portal box. So the
Trn2 width curve at compile+LNC=2 / 32k batch is 256 1.1284 -> 384 1.1065 -> **512 1.0887** -> 640 1.0945.
Launch-gate lesson: my ssh one-liners self-matched their own `bash -c` command line (the launch string contains
the file name / the word torchrun), so the gate now lives INSIDE `trn2_run.sh` (pattern `native_venv/bin/[p]ython`
+ neuron-ls owners) and refuses before touching anything. T14 (v10 portal shape) launched 12:06 UTC.

## T14 interim (12:09 UTC): v10 eager 8-way at 512-wide steps at 0.74-0.75 s (T10 256-wide eager: 0.60 s)
So eager is NOT free with width (CPU 73% us / 20% sy = dispatch-bound, but the 8 workers' device time now shows):
+25% step time for 2x width vs +32% at compile+LNC=2 (0.37 -> 0.49). Projection ~2400 steps; needs < 1.1605 to ship.

## T14 RESULT (12:38 UTC): v10 (512-wide) on the PORTAL SHAPE = **1.120626** -> promoted to `submit/train.py`
Exact upload bytes (sha256 6dd554d720ae421e), exact portal launch (`NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone
--nproc_per_node=8 train.py`, no env): 2362 steps / 77.4M tokens at 0.75 s/step, startup allowance 135.8 s (eager: no
NEFF compile), causality ok, checkpoint 478 MB (119.6M params fp32; v7's was ~200 MB - watch the portal for any size
limit). Beats T10 (256-wide, same shape) 1.160509 by -0.040; the last portal score (v7) was 1.1827, so the expected
portal result is ~1.127 (local + ~0.006). `submit/` now holds train.py + submission_v10_eager_8way.py (v9 copy removed;
the archive copy stays in submission_raw/). `prepare.py check` passes.
Launched T15 = 512-wide + TOTAL_BATCH 65536 (DEVICE_BATCH 16 x 4 workers, grad_accum 1) at compile+LNC=2, 12:39 UTC.

## v10 fresh-process verification (12:44 UTC)
`python prepare.py eval-public --train-py submit/train.py --checkpoint <T14 final.pt> --device cuda` on the local 4080
(fresh process, the portal's load_for_eval path, config rebuilt from the checkpoint's model_config): **val_bpb 1.121571**
over 2,097,152 tokens / 8,418,678 bytes (in-run Neuron bf16 eval: 1.120626; the 0.001 gap is Neuron-vs-CUDA numerics,
same as for v7). Upload is validated end to end. Checkpoint copy deleted from the scratchpad afterwards.

## T15 CRASH (12:46 UTC): 65k batch via 16 rows/worker does not fit an LNC=2 core at 512-wide
`[NCC_EOOM002] Maximum peak HBM usage of 28.56GB exceeds HBM limit of 24.00GB for Trn2` - 32.60 GB I/O tensors + 12.25 GB
scratchpad in the fused forward+backward NEFF (every layer's activations are live in one graph). So at compile+LNC=2 the
per-core row budget at 512-wide is 8 (T12 fits); a 65k batch must use grad_accum 2 (2 x 8 rows), i.e. ~2x the fwd_bwd time
per step for only ~+15% tokens in the budget - a weak lever (3090 T6b/T6c: 32k was already the optimum with sqrt-LR).
Deprioritized; next device job = PR7 profile at 512-wide (600 s, warm cache) to see the fwd_bwd / optimizer split now.

## PR7 (12:48 UTC): profile at 512-wide, compile+LNC=2 (warm cache)
fwd_bwd 0.385 | grad_sync 0.012 | optimizer 0.094 | total 0.491 s/step (256-wide PR6: 0.284 / 0.007 / 0.071 / 0.362).
6 x 119.6M x 32768 = 2.35e13 FLOP per step / 0.385 s = ~61 TFLOP/s = ~9% of the chip: still overhead-bound inside the
NEFF, optimizer 19%. No cheap throughput lever left without kernel work, so the next runs are same-cost architecture
knobs at 512: 600 s screens (compile+LNC=2) of MLP_RATIO 5, N_HEAD 8 (head_dim 64), DEPTH 12, against PR7's own 600 s
val_bpb as the 512 reference (its profiling syncs cost <1% of steps). Queue script `trn2_queue.sh` (gated by
`trn2_run.sh`) runs them back-to-back.
PR7's own 600 s result: **1.176969** (1212 steps, 39.7M tokens) = the 512-wide screen reference. Queue started S_MLP5 at 12:58 UTC.

## S_MLP5 (13:18 UTC): 512-wide + MLP ratio 5 = 1.175024 at 600 s vs reference 1.176969 (-0.0019)
1174 steps at 0.51 s/step (ref 1212 at 0.49): the extra 4% step cost is nearly free, and the per-step quality edge covers it.
Marginal-positive at the screen scale (3090 E7 said the same: "marginal, +11% step time" - here the time cost is a
quarter of that). Candidate for a full run after the H8 / D12 screens.

## S_H8 (13:30 UTC) ABORTED at 5 steps - a screen-harness artifact, not a result
Cold compile of the 8-head shapes took 631 s (step-0 dt), and the predictive stop compares `elapsed + max(last 20 step
times) + 5 s` against the budget: 0 + 631 + 5 >= 600 fired on the first eligible step. Score 3.137 = an untrained model.
S_MLP5 survived only because its compile (560 s) was under the 600 s budget. **Latent hazard in the vehicle:** the
last-20 window should be reset when the budget clock restarts at the 5th boundary; harmless for 1800 s runs (a 500-600 s
compile never trips it) and irrelevant for eager (v10), so the fix waits for the next cold boundary - editing the source
now would invalidate every warm compile cache. Workaround: cold screens compile-then-abort, and the warm re-run is the
screen. New queue `queue_S512b`: S_H8b, S_D12b (warm), then T16_full (512 + MLP5).

## S_D12 (13:49 UTC): 512-wide, depth 12 = 1.188000 at 600 s vs reference 1.176969 -> DISCARD (+0.011)
1103 steps (9% fewer than the 10-layer reference): every extra layer adds ops to the overhead-bound NEFF, and the loss/step
gain is far too small. Same verdict as the 3090 (E3). Depth stays 10; width (and MLP ratio) are the levers. S_D12b dropped
from the queue; queue_S512c = T16_full (512 + MLP5) after S_H8b.

## Accelerator setting found (13:55 UTC): the native stack compiles at **-O1** by default
`torch_neuronx/csrc/core/compilation/NeuronCompiler.cpp` (the editable source under `~/workspace/torch_neuron_eager`):
`NEURON_COMPILER_OPT_LEVEL` (default `"-O1"`) is passed verbatim to `neuronx-cc compile ... --framework XLA --target <trn2>
--lnc <n> -O1`, and `NEURON_CC_FLAGS` appends extra args (e.g. `--model-type transformer`, `--auto-cast`,
`--enable-mixed-precision-accumulation`). neuronx-cc's own default is -O2 ("best balance"); -O3 "may incur higher compile
time and memory". The persistent NEFF cache key hashes the compiler config (`CompilationCacheKey.cpp`: hash(base_key ||
compiler_config), "compiler args included in the key"), so a different opt level compiles fresh - no stale-NEFF risk.
`TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR` only relocates the lock-file dir (`/tmp/local_cache`), not the NEFF store.
Portal note: these are plain env vars, so they cannot ride along on the portal's fixed launch string; if -O2 wins here,
the vehicle would have to set `os.environ["NEURON_COMPILER_OPT_LEVEL"]` itself before the first compile (allowed: it is
not a forbidden import/call) - and its extra compile time counts against the 900 s startup cap on a cold portal box.
Queued **T17_full = 512-wide base + -O2** (full 1800 s, compile+LNC=2) after T16; compare to T12 (1.088742, 0.49 s/step).

## S_H8b (14:00 UTC): 512-wide, 8 heads x 64 = 1.196710 at 600 s vs reference 1.176969 -> DISCARD (+0.020)
1029 steps at 0.58 s/step (4 x 128: 1212 at 0.49): halving head_dim doubles the count of small attention matmuls and the
overhead-bound NEFF pays for every one; no quality gain to offset it. Screen ledger at 512: MLP5 -0.002 (keep, full run
T16 launched 14:01), depth 12 +0.011, 8 heads +0.020. Architecture at 512 is basically settled; the remaining levers are
the compiler (T17 -O2) and kernels.

## T16 RESULT (14:32 UTC): 512-wide + MLP ratio 5 = 1.088372 vs T12 1.088742 -> WASH (-0.0004), MLP stays 4
The screen's -0.002 did not survive the full run; the extra 4% step time and 5M params buy nothing. Architecture at 512
is settled: 10 layers, 4 heads x 128, MLP 4x, 32k batch. T17 (base at -O2) launched 14:32 UTC.
Queued T18_full = 512 base at -O1 + `NEURON_CC_FLAGS=--model-type=transformer` (the `=` form because the queue splits on
spaces; neuronx-cc's argparse accepts it). Independent of the -O2 question; -O3 only if -O2 pays.

## T17 (-O2) was a cache hit, not a test - killed at step 450; T17b relaunched with fresh caches (14:38 UTC)
Evidence: step-0 dt 93.7 s (a cold 512 compile is ~520 s), step time 0.49 s, and the loss trace bit-identical to T12
(5.6440 at step 40, same to 4 decimals through step 450). The worker env did carry `NEURON_COMPILER_OPT_LEVEL=-O2`, and
186 small NEFFs (eager per-op kernels, <=330 KB) were recompiled at 14:33 - but the 50-73 MB fused fwd+bwd NEFFs kept
their 11:32-13:37 mtimes. So the torch.compile graph's NEFF was fetched from the persistent store under a key that does
not see the opt level (the dynamo path: `neuron_dynamo_backend/compile.py` content_key = sha256(mlir) + ":" +
global_version_hash -> `_C.compile_graph`; the C++ key builder nominally appends optimization_level, but the store
evidently hit anyway - not worth more C++ archaeology tonight). `/tmp/torchinductor_ubuntu` is empty (4 KB), so it is
not Torch's inductor cache. **Recipe that guarantees a real recompile: point BOTH stores at fresh dirs**
(`TORCH_NEURONX_NEFF_CACHE_DIR`, default /tmp/neff_cache, from PersistentCacheBackend.cpp; `TORCH_NEURONX_HLO_CACHE_DIR`,
default /tmp/hlo_cache, from python_ops/compilation/hlo_cache.py). Rule for any compiler-flag experiment from now on.
Side value: T17's first 450 steps are a T12 replicate (identical, so run-to-run noise on this stack is zero for
identical NEFFs - differences between runs come only from the schedule's time-keyed warmdown and eval).

## T17b (14:49 UTC): -O2 compiled for real (568 s cold, fresh 46 MB NEFF) -> **0.49-0.50 s/step, identical to -O1**; loss trace
bit-identical through step 190. The compiler's optimization level does nothing for this graph; -O3 not worth a slot.
Killed at step 190; queue moved on to T18b (`--model-type=transformer`, fresh caches). Plan: judge T18b by its early step
time and kill it too if unchanged, then take one cold boundary to batch vehicle edits (LR/schedule env knobs for warm
screens + the predictive-stop window reset).

## Vehicle edit batch (14:57 UTC) - synced to the box; T18b (already compiled) unaffected
1. Env knobs for warm screens: `NEURON_COMPETITION_R1_{MATRIX,EMBED,HEAD,SCALAR}_LR_SCALE` (multiplicative), `_WD`,
   `_WARMUP` (steps), `_WARMDOWN` (time fraction), `_FINAL_LR`. Defaults = champion; CPU 4-step trace bit-identical to
   the pre-edit file; a MATRIX_LR_SCALE=3 / WARMUP=2 run diverges as expected.
2. Predictive-stop fix: `recent_step_times.clear()` when the budget clock restarts at the 5th boundary, so a cold
   compile in step 0 can no longer trip a short screen. Static check passes. Whether the source edit invalidates the
   warm 512 NEFF will show in the next screen's step-0 dt.

## T18b (15:00 UTC): `--model-type=transformer` compiled for real (534 s) -> 0.49-0.50 s/step, identical trace -> no lever
Compiler settings closed out: -O2 (T17b) and model-type (T18b) both change nothing for this NEFF at 512-wide. Killed at
step 90. **Queue LR started 15:00 UTC** (900 s warm screens, S_BASE first): matrix LR x1.5 / x0.7, warmdown 0.5 / 0.7,
embedding LR x2, head LR x2. Screens at ~1800 steps are still short of the full run's ~3650, so an LR winner gets a
full-run confirmation before it is kept.

## S_BASE crash (15:09 UTC) - my bug in the predictive-stop fix; fixed and re-queued 15:11 UTC
`recent_step_times.clear()` ran after the step's own append, so the stop check on the same iteration hit
`max([])` (ValueError on every rank at step 4). My CPU equivalence test had used NUM_STEPS=4 and never reached the
5th-boundary path - test through the restart (NUM_STEPS >= 6) from now on. Fix: `recent_step_times[:] = [dt]` (keeps
the current normal step). Re-verified: 8-step CPU trace identical to the pre-edit file and no crash.
Also learned: the first run on the edited source compiled COLD (S_BASE step-0 dt 493.6 s) even though
`global_version_hash()` (torch_neuronx/utils.py) hashes only package versions - so the source-keyed-cache rule from
[trn2-launch-gating] holds empirically; every vehicle edit costs one ~8-9 min cold compile per graph shape. The re-queued
S_BASE pays it again; the six LR/schedule screens after it are env-only and warm.

## S_BASE (15:27 UTC): 512 base, 900 s = **1.140360** (1808 steps, 59.2M tokens) = the LR/schedule screen reference
Its step 0 took 17.8 s (warm) although the source had been edited again after the 493 s compile at 15:00 -> the NEFF
key is the graph (MLIR) content, not the file: training-loop / constant edits are free, only graph changes recompile.
The 15:00 cold compile therefore was not source keying; the likeliest cause is persistent-cache eviction
(`NEURON_COMPILATION_CACHE_MAX_ENTRIES` / `_MAX_SIZE_MB` exist in the C++; the many 50-70 MB screen NEFFs of the
afternoon could have pushed the base-512 NEFF out). Corrects the "source-keyed" belief in memory trn2-launch-gating.

## S_MLR15 (15:43 UTC): Muon LR x1.5 = 1.143694 vs 1.140360 -> DISCARD (+0.0033). Next: x0.7.
## S_MLR07 (15:59 UTC): Muon LR x0.7 = 1.147607 -> DISCARD (+0.0072). Muon LR sits at its optimum (x1.5 +0.0033). Next: warmdown 0.5 / 0.7.
## S_WDN05 (16:15 UTC): warmdown 50% = 1.141902 -> DISCARD (+0.0015). Next: warmdown 70%.
Appended S_OPTC (900 s, `COMPILE_OPT=1`: torch.compile the stacked-parameter optimizer step) to queue_LR: the optimizer is
19% of the step (PR7); the pre-stacking compiled optimizer lost to eager, but with ~20 stacked tensors it may now win.
## S_WDN07 (16:31 UTC): warmdown 70% = 1.139668 -> marginal -0.0007 (50% was +0.0015). Appended S_WDN08 (80%) to probe the trend.
## S_ELR2 (16:47 UTC): embedding LR x2 = 1.137544 -> **-0.0028**, the clearest LR win so far. Appended S_ELR4 (x4) to bracket.
The embedding LR was tuned at 256-wide on the 3090; at 512 the per-row update is spread over twice the dims, so a higher
LR is plausible. Plan: bracket (x4), then a full run of the best combination (embedding LR + warmdown 0.7/0.8).
## S_HLR2 (17:03 UTC): head LR x2 = 1.140683 -> wash (+0.0003). Next: S_OPTC (compiled optimizer), S_WDN08, S_ELR4.
## S_OPTC (17:07 UTC): compiled stacked optimizer = 0.50-0.51 s/step vs 0.49 eager (+3%), 12 recompile notices -> killed at step 110.
The optimizer's 19% is eager launches of ~20 stacked tensors; compiling them buys nothing on this stack. Retired for good.
## S_WDN08 (17:23 UTC): warmdown 80% = 1.139295 (-0.0011). Monotone: 50% +0.0015, 60% ref, 70% -0.0007, 80% -0.0011 -> T19 uses 0.8.
T19 pulled out of the queue: it launches by hand after S_ELR4 so the embedding scale (x2 vs x4) can be chosen first.

## S_ELR4 (17:39 UTC): embedding LR x4 = 1.138664 (-0.0017) - worse than x2 (-0.0028): the optimum is ~x2. LR screen set complete.
Ledger (900 s, ref 1.140360): Muon x1.5 +0.0033, x0.7 +0.0072, warmdown 50% +0.0015 / 70% -0.0007 / 80% -0.0011,
embedding x2 -0.0028 / x4 -0.0017, head x2 +0.0003, compiled optimizer +3% step time.
**T19_full launched 17:40 UTC** = 512 base + EMBED_LR_SCALE 2.0 + WARMDOWN 0.8 (warm compile). Screen deltas add to about
-0.004 at 900 s; the full run decides (T12 1.088742 to beat). If it wins, v11 = v10 + these two defaults, validated eager 8-way.

## T19 RESULT (18:11 UTC): 512 + embedding LR x2 + warmdown 80% = **1.084527** - NEW LOCAL BEST (T12 1.088742, -0.0042)
3617 steps / 118.5M tokens at 0.50 s/step, warm start (41.6 s), causality ok. Both 900 s screen deltas (embedding x2
-0.0028, warmdown 80% -0.0011) carried over to the full run, slightly better than additive. New vehicle defaults for the
portal line: v11 = v10 + EMBED_LR_SCALE 2.0 + WARMDOWN 0.8 baked in (plus the knob/predictive-stop code, which is
bit-exact with defaults). T20 = v11 on the exact portal shape (eager, LNC=1, 8-way, no env) must beat T14's 1.120626.

## T20 RESULT (18:43 UTC): v11 on the PORTAL SHAPE = **1.112172** -> promoted to `submit/train.py` (sha256 c6cd05b896ff290b)
Exact upload bytes, exact portal launch (eager, LNC=1, 8 workers, no env): 2340 steps / 76.7M tokens at 0.76 s/step,
startup 29.9 s, causality ok, checkpoint 478 MB. Beats v10 (T14 1.120626) by -0.0085 - the LR/schedule retune pays more
on the eager line (fewer steps, so the longer warmdown and hotter embeddings matter more). Expected portal ~1.118.
`submit/` = train.py + submission_v11_eager_8way.py. Fresh-process eval-public on the local GPU follows (checkpoint pulling).
Next on the device: warm 900 s screens of the knobs not yet swept, on top of the new defaults (reference S_BASE2):
scalar LR x2 / x0.5, weight decay 0.05 / 0.20, warmup 50.
v11 fresh-process check (18:47 UTC): `prepare.py eval-public --train-py submit/train.py --checkpoint <T20 final.pt> --device cuda`
= **1.113277** over 2,097,152 tokens (in-run Neuron eval 1.112172; the usual 0.001 numerics gap). Upload validated end to end.

## S_BASE2 (19:00 UTC): T19 defaults at 900 s = **1.135135** = reference for the second knob set (old base 1.140360: -0.0052)

## NKI scoping (19:10-19:25 UTC) - what the device trace says and what kernel would pay
Device profile at 512-wide (torch profiler + Neuron DEVICE mode; the per-instruction view could not be rendered: the
NTFF lacks NeffNodes / `neuron-profile view` cannot resolve the NEFFs by UID from this capture path). The runtime event
counts over 2 steps are the useful part: **nrt_model_submit 858, nc_model_switch 1624, dmem_buf_copyin 10,682,
nrt_dma_mem_alloc 3,300, nrt_tensor_free 5,038** - i.e. ~430 model submits and ~5,000 small DMA copy-ins per step.
The compiled forward+backward is 2 NEFFs; the rest are the eager optimizer (stacked Muon/AdamW, ~20 tensors x Polar
Express steps) and per-step scalar traffic. The chip is launch/DMA-bound, matching the ~9% MFU estimate.
NKI options on this stack (nkilib 0.5 shipped in the venv, wrapped by torch_neuronx via `wrap_nki`/`nki_op`):
- `nkilib.core.attention.attention_cte` (fwd) + `attention_bwd` (bwd): flash attention with causal mask, **sequence
  packing via bound_min/bound_max (= our document mask)**, a per-row denominator `sink`, LNC2 sharding, bf16 in / fp32
  softmax, returns max/sum for backward. Tested earlier via SDPA at 256-wide: -3% (PR2), because it saw a dense
  attn_mask (doc mask) and fell back; the flash path only engages with is_causal / bounds.
- Our registers are 4 learned k/v slots (contribute values), not a denominator-only sink -> an exact port needs either
  prefix-caching (k_prior/v_prior: not compatible with bounds in the fwd, absent in the bwd) or a modelling change
  (registers -> denominator sink). Decision pending on what the registers were worth on the 3090.
- Plan: standalone device test of attention_cte/attention_bwd with bounds vs the explicit path (fwd/bwd numerics),
  then an autograd.Function `NKIDocAttention` in the vehicle behind ATTN=nki, screen at 512 (compile+LNC=2 and eager
  8-way), full run if the step time drops materially. Device is on the LR2b queue until ~20:45 UTC; scoping meanwhile.

## Session 5 (2026-09-06 19:30 UTC) - NKI kernel campaign toward 1.065 (compile+LNC=2 line: T19 1.0845; eager 8-way: v11 1.1122)
Reference reading for this campaign: `findings/nki-kernel-playbook-2.md` (the CUDA-vs-NKI deltas, engine assignments,
what not to build). Box stack: nki 0.5.0, neuronx-cc 2.26, nkilib 0.5 (NOT the 0.6.0 the playbook was written for).
- S_SLR2 (scalar LR x2, 900 s) = **1.148098** vs S_BASE2 1.135135 -> +0.013, DISCARD. The remaining LR2b queue items
  (scalar x0.5, WD 0.05/0.20, warmup 50; each expected within +-0.002) were dropped: the queue runner was killed to give
  the device to kernel work. (Lesson: `pkill -f "bash trn2_queue.sh"` inside an ssh one-liner matches and kills the
  ssh session itself; use a `[t]rn2_queue.sh` pattern.)
- New box tool `trn2_job.sh <ID> <cmd>`: the same launch gate as `trn2_run.sh` for single-process device jobs (NKI tests,
  profiler captures); it must `source` the venv (`env PATH=... bash -c python` did not resolve `python` on this box).
- **NKI1 - vendor flash attention with doc bounds + sink, numerics and timing (nki_attn_test.py, fixed to the real bwd
  contract: bwd bounds are (bs, seqlen_q) float32 PER BATCH, sinks (bs, nheads); fwd bounds (B*H, L, 1) int32 work).**
  B=2/H=4/L=1024/E=128 vs an explicit fp32 reference: out err 4.1e-3 (|ref| 1.4), dq 3.6e-3, dk 3.6e-3, dv 1.3e-2
  (|ref| 3.65), dsink 2.0e-2 (|ref| 3.7) = bf16-level agreement; the sink gradient comes back as (B, H).
  Timing at one LNC=2 worker's shape (B=8, H=4, L=1024, E=128, eager): explicit fp32 fwd 10.3 ms / fwd+bwd 15.6 ms;
  **NKI fwd 1.3 ms / fwd+bwd 2.9 ms = 5.4x faster per layer** (decision rule was >= 2x -> attention IS worth porting).
- Vehicle (`submission_raw/train3_submission.py`) gained `NEURON_COMPETITION_R1_SINK=kv|logit` (GPTConfig.sink_mode,
  checkpoint-saved) and `ATTN=nki`: logit mode replaces the 4 learned k/v registers by one learned denominator-only sink
  logit per (layer, head), init log(4) so step-0 softmax mass matches the champion's 4 zero registers; the explicit path
  implements the same math (used off-Neuron, e.g. eval-public on CUDA/CPU); `doc_bounds(idx)` builds per-position
  [doc start, doc end) with scatter/gather only; `NKIDocAttention` (autograd.Function) wraps attention_cte / attention_bwd
  via `wrap_nki`, passes `scale` into the kernel, and returns dq/dk/dv/dsink. Defaults (SINK=kv) are bit-identical to the
  pre-edit file on a CPU 6-step trace; SINK=logit trains, round-trips through load_for_eval (causal), passes prepare.py check.
- Kernel gotcha: `attention_cte`'s sequence-packing (range_select) path asserts `scale == 1.0` at trace time
  (`[NCC_INKI016] range_select path doesn't support scale != 1.0`): the vehicle pre-scales q by 1/sqrt(d) in bf16, runs both
  kernels with scale 1.0 and multiplies dq back by the scale.
- **NKIV1 (19:47-19:52 UTC): vehicle integration test passes.** `doc_bounds` reproduces the explicit doc+causal mask exactly on
  real rows (17 BOS in 8 rows); `NKIDocAttention` vs the explicit sink-softmax: out 3.5e-3, dq/dk 3.9e-3, dv 1.6e-2 (|ref| 3.2),
  dsink 3.1e-2 (|ref| 9.4). Whole-model eager fwd+bwd (single process, 8 rows, LNC=2): explicit 409-426 ms -> **nki 343 ms
  (-17%)**, identical loss. Queue `runs/queue_NKI.txt` started: S_NKI (compile+LNC=2, 900 s, cold compile), S_E8_NKI, S_E8_BASE
  (eager 8-way 900 s pair). Reference for S_NKI: S_BASE2 1.135135.
- **S_NKI (19:49 UTC) INVALID, killed at ~3 min:** under torch.compile the `autograd.Function` around the `wrap_nki` kernels
  graph-breaks inside every Block (`torch_dynamo_resume_in_forward_at_499`), so dynamo compiles each layer's resumed frame
  separately, guards on the static `StackedLinear.idx` ints, hits `recompile_limit (8)` and runs the rest EAGER. Lesson:
  **the model must stay ONE dynamo graph** (any per-layer graph break is fatal with the stacked-weight design).
  Mechanism found in `torch_neuronx/nki_hop.py`: `NKIHOPCaller.__call__` takes an eager C++ fast path off-compile and the
  `nki_kernel_wrapper` HOP under compile; the supported way to expose a kernel to dynamo is the **`nki_op(name)` decorator**
  (a `torch.library` custom op whose fake impl and FunctionalTensorMode dispatch re-trace the inner function, so the HOP
  appears only after AOT), plus `torch.library.register_autograd` for the backward. The backend's own SDPA path instead
  registers aten decompositions (`neuron_dynamo_backend/decompositions.py`). Vehicle rewritten: `build_nki_ops()` defines
  `tf_nki::doc_attn_fwd` / `doc_attn_bwd` (must run before `torch.compile`; called in main() and load_for_eval),
  `nki_doc_attention()` replaces the Function. Tests: NKIV2 (eager, custom-op build), NKIC1 (`nki_compile_test.py`: compiled
  fwd+bwd at 256-wide, nki vs explicit, dynamo counters for frames/graph breaks), then S_NKI2 (900 s compile+LNC=2).
- Forward NEFF profile (512-wide, LNC=2 worker, 8 rows; `neuron-profile capture` on the cached fwd NEFF `3c71...`):
  **0.104 s/exec**; engine active: tensor 41.5% / scalar 46.6% / gpsimd 42.3% / vector 35.5% / DMA 72.6% (no engine saturated;
  DMA-bound); mfu_estimated 18.6%, **mm_arithmetic_intensity 29 FLOP/byte** (memory-bound matmuls); hardware_flops 9.8e11 of
  which transpose_flops 2.3e11 (23%); **hbm_write 16.4 GB + hbm_read 9.5 GB per forward** (the explicit attention's fp32
  (b,h,t,t) intermediates: scores, mask, softmax, saved for backward); 16.3M hardware-dynamic + 4.7M software-dynamic DMA
  packets, 2.7k static; spill 0. `neuron-profile view --output-format json` on the 1.1 GB NTFF did not finish in 15 min (killed);
  summary-text is fast. Profiles live in `~/tf/runs/prof_{fwd,bwd}512/` on the box.
- S_E8_NKI (eager 8-way, NKI attention via the old Function - valid in eager): 0.67-0.72 s/step vs 0.76 base (T20): only ~-7%
  with 8 dispatching workers; the CPU-bound eager line gains little from one fused op per layer.
- **Local NKI simulator set up (20:00 UTC):** the box's `nki-0.5.0` + `neuronx-cc` cp312 wheels install into `.venv-nki`
  (local Python 3.12.3 matches); `nki.simulate(kernel)(torch/numpy args)` runs a kernel on the CPU in ~1 s -> kernel numerics
  iterate locally, the device is only needed for timing. Simulator gotchas: index 3-D SBUF tensors in one step or, better,
  use python lists of 2-D tiles (`T[i][:, a:b]` on a 3-D `nl.ndarray` produced wrong views in `nc_matmul`).
- **Fused Polar Express (Muon) kernel `nki_muon.py`** (`polar_express_tall`: stack (n, m, k) bf16, m >= k = 512; X and X^T
  SBUF-resident, A = X^T X and A^2 via symmetry, X_new = aX + X B and X^T_new = aX^T + B X^T with B symmetric -> no transposes
  inside the iteration; Frobenius prescale via a ones-matmul all-partition sum; LNC sharding over slices by program id).
  Simulator vs the fp32 reference: 512x512 3.1e-3 (|ref| 0.245), 2048x512 1.6e-3 (the bf16 torch reference itself is 5.6e-3
  off fp32); orthogonality residual identical to the reference. Wired into the vehicle as `NEURON_COMPETITION_R1_MUON=nki`
  (wide stacks orthogonalized as their transpose; default `torch` path untouched). Device test = NKIM1 (`nki_muon_test.py`:
  numerics + timing vs the eager torch loop for 40x512x512, 10x2048x512, 10x512x2048).
- Rendering the 1.7 GB backward NTFF with `neuron-profile view` ran at 290% CPU for >8 min and slowed the concurrent eager
  8-way screen S_E8_NKI from 0.68 to 0.80-0.91 s/step (S_E8_NKI's val_bpb is therefore contaminated; its clean steady state
  before the render started was **0.68-0.69 s/step vs 0.76 base = -10%**). Rule: no CPU-heavy work on the box during a screen.
  Backward profile deferred (the forward profile already identifies the explicit attention's HBM traffic).
- **NKIV2 / NKIC1 (20:12-20:21 UTC):** the `nki_op` custom-op build keeps the model ONE dynamo graph (frames total 1, no graph
  breaks) and is numerically identical to the Function build (eager fwd+bwd 361 vs 412 ms explicit, -12%). But **under
  torch.compile the nki path is 3x SLOWER: 934 ms/step vs 305 ms explicit at 256-wide/8 rows** (compile 333 s vs 227 s).
  S_NKI2 killed on that evidence. Reading the backend: compiled graphs run NKI HOPs through `NKITorchMlirOpImpl`
  (`python_ops/torch_mlir/nki_op_impl.py`, a per-op TorchMlirKernel with `_prepare_execution_tensors` /
  `_copy_to_provided_output`), i.e. most likely each kernel call is a separate execution with copies around it, which
  partitions the fused NEFF at every layer. NKIC2 (`nki_compile_prof.py`: runtime event counts per step, explicit vs nki)
  will confirm. The backend's own route for attention is `fx/passes/flex_attention_nki_lowering.py`: `flex_attention` with a
  block mask is legalized into a generated (Jinja) NKI attention kernel with the mask_mod lowered via a pointwise
  FX->nki.isa mapping (`_nki/fx_to_nki.py`: add/mul/compare/logical only, **no index/gather**) and static tile elimination;
  the kernel takes only q, k, v -> data-dependent masks (our BOS document mask) cannot go that way. (Doc mask was worth
  -0.009 and 15% speed on the 3090 ledger.)
- S_E8_NKI (eager 8-way, contaminated by the profiler render): 1.177312 at 1104 steps (no clean reference yet; S_E8_BASE and
  a clean S_E8_NKI2 rerun queued). NKIM1 (Muon kernel on the device) failed at trace time: **list comprehensions are an
  "unsupported expression" for the NKI tracer** (the simulator accepted them) -> expanded to loops; NKIM2 queued.
- **User backlog `findings/arch-backlog-speedrun.md` (20:30 UTC) added to the experiment queue** as 900 s compile+LNC=2 screens
  vs S_BASE2 1.135135: env-only first (S_QK64 = narrow QK 64, S_TRI = trigram), then code knobs to implement: long-short
  sliding window + window warmup (tensor-valued window so the graph stays invariant), multi-token prediction aux head,
  value residual (layer-0 V blend), backout, merged QKV, per-layer embeddings. Plan per the user: after screening, pick the
  best couple and write kernels for them.
- Backlog knobs implemented in the vehicle (20:50 UTC), all env-gated, config-saved, CPU-verified (defaults bit-exact;
  each knob trains 6 steps; MTP/merged-QKV/all-on checkpoints reload via load_for_eval, causal): `WINDOW=<w>` (odd layers
  attend to the last w keys; masks and NKI bounds), `WINDOW_WARMUP=<frac>` (all windows grow 128 -> target in 128-buckets over
  that fraction of the budget; window values live in device int32 buffers refreshed by H2D copy, so no recompiles),
  `MTP=<w>` (t+2 aux head: zero-init 512x512 projection in the matrix stack + shared lm_head, training only),
  `VALUE_RESID=1` (v += lambda_l (v_0 - v), zero-init), `BACKOUT=1` (x -= lambda x_mid, zero-init), `MERGED_QKV=1`
  (one (3*n_embd, n_embd) stack), `PLE=1` (10 per-layer 8192x512 tables, gated 0.5). Note: the explicit mask now carries an
  extra `dist < win_global` AND even at defaults -> the base graph recompiles once. Queued as 900 s compile+LNC=2 screens
  after S_QK64/S_TRI: S_WIN (256), S_WINWARM (256 + warmup 0.5), S_QKV, S_MTP (0.3), S_VR, S_BO, S_PLE. Reference S_BASE2
  1.135135 (the base graph changed slightly, so a fresh S_BASE3 should be run before trusting +-0.002 deltas).
- **21:02 UTC results.** S_E8_NKI2 (clean eager 8-way, NKI attention): **1.163167 @1262 steps vs S_E8_BASE 1.173359 @1178 (-0.010)**;
  the eager (portal) line's first kernel win. NKIM3: the Muon kernel runs on the device - 512x512 stack 15.9 -> 3.1 ms,
  2048x512 10.0 -> 2.5 ms, 512x2048 11.4 -> 2.8 ms (errors 6-8e-3 vs the bf16 torch loop, |ref| 0.12-0.24); the tracer's
  second complaint was tuple-unpacking `for a, b, c in ...` ("expecting simple variable") -> index the coefficient list.
  NKIC3: **compiled nki == explicit (376 vs 375 ms/step at 256-wide) with 4 NEFF executions per step in both** -> the kernel IS
  inlined in the fused NEFF; NKIC1's 934 ms was a cold-cache/first-execution artifact (the timed steps ran right after the
  333 s compile). The 512-wide compiled screen (S_NKI3) and a MUON=nki screen (S_MUON) are queued right after S_BASE3.
- Box tooling: `trn2_job.sh` now pins jobs to a logical core (`NEURON_RT_VISIBLE_CORES`, lock files `runs/.core_<n>.lock`)
  and only refuses while a torchrun holds the device, so kernel tests run in parallel (user's suggestion: 4 logical cores at
  LNC=2); `trn2_queue.sh` gained `PJOB` (launch without waiting) and `WAIT` lines.
- **Fused AdamW kernel `nki_adamw.py`** (one pass per 128-row tile: decay, moment lerps, sqrt/reciprocal, update; per-step
  scalars lr / 1/bias1 / 1/sqrt(bias2) / lr*wd arrive as a [128, 4] tensor so the NEFF never bakes a value): exact on the
  simulator (2e-7), in-place writes to the HBM inputs honored there; device test NKIA1 (numerics, in-place through wrap_nki,
  timing for 8192x512 and the 131072x512 bigram table) queued as a parallel job. Vehicle knob `ADAMW=nki` (2-D fp32 params
  with rows % 256 == 0: the 5 vocab tables, bte, lm_head; copies the three outputs back unless `ADAMW_INPLACE=1`).
  T22_full (eager 8-way, SINK=logit ATTN=nki = the v12 candidate for the portal line) queued after S_MUON/NKIA1.
- **Fully fused Muon step kernel `muon_step_stack` (nki_muon.py, 21:40 UTC):** momentum lerp + nesterov blend, Frobenius
  prescale, Polar Express in both frames, per-row (tall) / per-column (wide, via a K=1 ones-matmul broadcast tile)
  second-moment scaling, masked weight decay update - one launch per stack, in place on w/momentum/second. Simulator vs the
  torch mirror of `_muon_step`: w err 4-6e-4 for |dw| 5-7e-3 (bf16 NS), momentum/second exact, both orientations. Vehicle
  knob `MUON=fused` (NS-only kernel stays `MUON=nki`); `ADAMW=nki` hook applied (defaults bit-exact). Device tests NKIM4
  (fused Muon) and NKIA1 (AdamW) run as parallel jobs after S_MUON. Kernel bodies in the vehicle are copied from the dev
  files at patch time - edit the dev file, re-copy.
- **Incident 21:26-22:03 UTC: 37 min of idle device.** The queue runner died with `trn2_queue.sh: line 20: $2: unbound variable`
  right after S_BASE3: I had rsynced an edited `trn2_queue.sh` while the old runner was still executing it (bash reads scripts
  incrementally, so the running process picked up bytes of the new file at its old offset). Rule: **never overwrite a running
  bash script; edit queue FILES in place, but restart the runner for script changes.** Restarted from S_NKI3 as `queue_NKI3.txt`.
- **S_NKI3 (22:02-22:16 UTC) ABORTED: compiled NKI attention at 512-wide/4 workers runs 1.73 s/step vs 0.49 explicit.** Live
  monitor: ~2100 device executions/s (~3600 per step vs ~430 for the explicit compiled graph), NC utilization 25% average
  with bursts of 0 / 27 / 100% (the user watched it: the 100% bursts are the flash kernels, the rest is the partitioned graph
  pieces, copies and eager optimizer). So the custom-op HOP is NOT inlined in the fused NEFF; NKIC1's 934 ms was real and
  NKIC3's "model_default = 4 per step" simply does not count NKI executions. Verdict: **on the compiled line keep the explicit
  attention; NKI attention pays on the eager line (S_E8_NKI2 -0.010)**. Inlining would need the backend's flex-attention
  route (static masks only) or a custom-call lowering - parked.
- **22:24 UTC device results:** fused Muon step (NKIM4) 15-16 ms -> 2.0-2.4 ms per stack (w err 5-9e-4 for |dw| 5-10e-3,
  momentum/second exact, in-place honored); fused AdamW (NKIA1) 2.43 -> 0.34 ms (8192x512), 20.1 -> 5.8 ms (131072x512
  bigram table), in-place honored (`ADAMW_INPLACE=1`). Optimizer estimate: ~94 ms -> ~25 ms/step at compile+LNC=2.
  S_MUON crashed: the NKI tracer cannot resolve closure constants (`P`, `FMAX`) of the enclosing builder ("unbound variable")
  -> module-level `_NKI_P` / `_FMAX`. Re-screen S_FUSED (MUON=fused ADAMW=nki ADAMW_INPLACE=1; graph unchanged -> warm NEFF),
  then T21_full (compile line, all optimizer kernels) and T23_full (eager 8-way, attention + optimizer kernels).
- **T22_full (22:25-22:57 UTC): eager 8-way + NKI attention (SINK=logit ATTN=nki) = 1.103320 @2527 steps, 0.70 s/step -
  NEW EAGER-LINE BEST (v11/T20 1.112172).** v12 candidate = these + the optimizer kernels (`submission_v12_eager_8way.py`,
  built by `freeze_submission.sh`), validated by T23 (env) and T24 (exact bytes) before promotion.
- NKIO1 / S_FUSED crashed on a third tracer rule: **the kernel's module globals are the only namespace the NKI tracer sees** -
  builder-local `import nki.language as nl` is a closure name ("failed to resolve name 'nl.ndarray'"). Fix: `global nki, nisa, nl`
  in the builders (module-level placeholders). The queue chain was stopped (T21 had auto-launched with the broken file), fixed,
  and restarted as queue_NKI4.
- NKIO2 (22:58 UTC): the vehicle's in-file kernels now trace and run on the device (fused Muon tall/wide in-place errors 4-5e-4,
  NS-only orthogonality residual 6-7e-2 as before, AdamW update lands). S_FUSED launched 22:59 on the fixed file; chain
  S_FUSED -> T21_full (compile line + optimizer kernels) -> T23_full (eager + all kernels) -> T24_gate (exact v12 bytes, sha
  f85ca4c8d66cbd45) -> backlog screens. Queue-runner fix (local, sync only when the runner is restarted): WAIT now also
  waits on the job core-lock files (a PJOB's python had not started when WAIT polled 5 s later).
- **S_FUSED (22:59-23:17 UTC): fused optimizer kernels = 1.125177 @2054 steps, 0.44 s/step vs S_BASE3 1.135172 @1821 (0.49):
  -0.010 at 900 s.** T21_full (compile line + optimizer kernels) running at 0.43 s/step (projection ~4100 steps, ~1.075).
  Next compiled-attention idea queued as NKIF1 (`nki_flex_test.py`): the backend's flex_attention lowering with a static
  causal mask_mod - if that route inlines the kernel (unlike the custom-op HOP route), a static-mask variant (drop the doc
  mask, -0.009 on the 3090 ledger; registers padded to a 128-block prefix) becomes a compiled-line candidate.
- **Fused RoPE kernel `nki_rope.py`** (split-half rotation, rows = b*t in partitions, per-head slices in the free dim; the
  backward is the same kernel with `sign=-1`): simulator exact up to bf16 output rounding. Vehicle knob `ROPE=nki`
  (`_NKIRope` autograd.Function, eager Neuron only - guarded by `torch.compiler.is_compiling()`). Eager motivation: the torch
  `apply_rotary_emb` is ~9 launches per q/k per layer (~180 fwd + ~360 bwd per step of the ~1400 launches in the 0.70 s eager
  step). Device test NKIR1 queued next to NKIF1; if it pays, an eager screen with ROPE=nki follows and v12 -> v13.
- **T21_full (22:59-23:52 UTC): compile + LNC=2 + fused optimizer kernels = 1.076414 @4115 steps (0.44 s/step) - NEW LOCAL
  BEST** (T19 1.084527 @3617). The optimizer kernels are worth -0.008 at full length. Gap to 1.065: 0.011. Remaining levers on
  this line: the compiled attention route (NKIF1 flex test), the backlog screens (narrow QK, trigram, window, MTP, value
  residual, backout, merged QKV, PLE), and the ~0.05 s/step still outside the fused fwd+bwd NEFF (collectives, scalars,
  Python dispatch).
- **DOCMASK=score (00:10 UTC): the document mask made STATIC.** Doc id d (mod 64, two base-8 digits a, b) is written into the
  last 6 q/k dims after QK-norm/RoPE as q_ext = -1024*[a^2, -2a, 1, b^2, -2b, 1], k_ext = [1, a, a^2, 1, b, b^2] so that
  q_ext.k_ext = -1024*((da)^2+(db)^2): exactly 0 for the same document, <= -1024 otherwise (all products exact in bf16, 1024 a
  power of two); register keys get zeros (no penalty). The attention mask is then causal(+register prefix) only, which the
  Neuron backend's flex_attention pass can lower to an inlined NKI kernel (`ATTN=flex`, static block mask built once on CPU).
  CPU check (3 layers, 4 docs/row, real rows): boolean mask vs score-encoded logits differ by 2e-4 (a no-doc-mask control
  differs by 3.1). Cost: 6 of 128 q/k dims are repurposed. Pending: NKIF1 (is the flex route inlined/fast?), then S_FLEX
  (compile+LNC=2, ATTN=flex DOCMASK=score) if yes.
- **T23_full (23:47-00:19 UTC): eager 8-way + all kernels = 1.092707 @3022 steps (0.57 s/step) - eager-line best** (T22 1.1033,
  v11 1.1122). Ladder on the portal shape: 1.1122 -> 1.1033 (NKI attention) -> 1.0927 (+ fused Muon/AdamW).
- User decisions (00:40-00:50 UTC): first compile ON + 8-way (v13 `submission_v13_compile_8way.py`, sha d1e208cc43871694, gate
  T25 queued), then **eager + LNC=2 launch: `submit/train.py` = v12 (sha 58eccae335e3a435) with
  `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`** - gate T26 running at **0.40 s/step**
  (faster than the compiled LNC=2 line's 0.44: the eager kernels + 8 rows/worker + 4 dispatching processes).
- Fused RMSNorm fwd/bwd kernels `nki_rmsnorm.py` simulate correctly (bf16-rounding-level errors); eager-line candidate after
  NKIR1 (RoPE) shows whether small fusions still pay at 4-8 workers.
- **T26_gate (00:22-00:55 UTC): v12 bytes, EAGER + LNC=2 launch (4 workers x 8 rows) = 1.069919 @4452 steps (0.40 s/step) -
  NEW LOCAL BEST**, beating the compiled LNC=2 line (T21 1.0764 @4115). Eager with NKI attention + fused optimizer kernels at
  8 rows/worker and 4 dispatching processes is now the fastest configuration on the box. Gap to 1.065: 0.005.
  T27 (v14 clean bytes, same launch) gates the submission file next.
- **01:56 UTC results.** T25_gate (v13: compile ON + 8-way LNC=1 + explicit attention + optimizer kernels) = **1.070259 @4320
  steps, 0.41 s/step** - tied with the eager LNC=2 line and on the portal-proven launch shape. NKIF1: the backend's flex route
  DOES inline (compiled fwd+bwd 4.6 ms vs explicit 5.6 ms per layer at one worker's shape) but that means compiled attention is
  only ~56 ms of the 385 ms fwd_bwd -> **attention is not the compiled bottleneck** (PR2 was right); S_FLEX killed. The static
  doc-mask encoding stays available. NKIR1: RoPE kernel **61 ms -> 0.59 ms per q/k call** in eager (the torch
  slice/cat/mul chain is far more expensive on Neuron eager than its op count suggests) -> screens S_E2_BASE / S_E2_ROPE
  (eager LNC=2, all kernels, +-ROPE=nki). NKIN1: RMSNorm kernel slower than F.rms_norm (0.91 -> 1.18 ms), shelved.
  PROF_BWD (backward NEFF summary) queued as a job to find where the compiled graph's time goes.
- v15 (`submission_v15_clean_compile_8way.py`, sha fc1f33c77b5cf519): CPU 3-step trace identical to v14 (9.0117 / 9.0043 /
  8.9899); static check passes with the 8-way launch; gate T28 queued after PROF_BWD. Queue 5 running: S_E2_BASE ->
  S_E2_ROPE -> PROF_BWD -> T28_gate -> S_QK64 -> S_TRI -> S_WIN -> S_WINWARM -> S_QKV -> S_MTP -> S_VR -> S_BO -> S_PLE.
- **S_E2_ROPE (02:17-02:33 UTC) = 1.112149 @2411 vs S_E2_BASE 1.118517 @2225: RoPE kernel -0.0064 at 900 s (0.37 vs 0.40
  s/step).** Ported into v16 (clean file + RoPE kernel), gate T29 queued; projection at full length ~1.064.
- **PROF_BWD (02:33-02:50 UTC): backward NEFF at 512-wide (LNC=2 worker, 8 rows) = 0.254 s/exec, DMA active 74%, tensor engine
  47%, vector 41%, scalar 37%; hbm read 53.2 GB + write 35.5 GB per execution (fwd: 9.5 + 16.4 GB) -> ~115 GB of HBM traffic per
  worker-step, ~30x the model's inherent activation traffic; mm_arithmetic_intensity 33, mfu 28%, 22.6M hardware-dynamic DMA
  packets, spills ~0.** The compiled line is bandwidth-bound on compiler-materialized intermediates, not on attention (NKIF1)
  and not on FLOPs. Without a per-op breakdown (JSON render too slow) the compiled line is parked; the eager line with kernels
  is faster (0.37 s/step) and is where the backlog screens now run (queue rewritten: S_E2_QK64 ... S_E2_PLE, eager LNC=2 with
  all kernels + ROPE, reference S_E2_ROPE 1.112149 at 900 s).
- **Backlog screens on the eager kernel line (900 s, LNC=2, ref S_E2_ROPE 1.112149 @2411, 0.37 s/step), 03:46-05:05 UTC:**
  trigram 1.115680 @2122 (+0.0035, -12% steps: discard); **window 256 on odd layers 1.110465 @2403 (-0.0017, free: keep?)**;
  window + warmup 0.5 1.110463 (identical to window alone: warmup adds nothing); merged QKV 1.112641 @2363 (+0.0005, the
  1536-row stack leaves the fused Muon kernel: discard); MTP 0.3 1.152245 @2038 (+0.040, -15% steps: discard); QK64 crashed
  (NKI attention op assumes equal q/k/v head dims). Pending: value residual, backout, PLE, then T30 (v15 bytes, LNC=2 launch).
- **05:20 UTC: user decision - hold submissions until ~1.03-1.04 local** (portal: 9th at 1.0752; top-5 ~1.02). Next campaign on
  the compile 8-way line (the v15 config, 0.31 s/step): queued S_C8_BASE, S_C8_WIN (window 256), S_C8_BF16 (bf16 q@k^T,
  new knob ATTN_BF16), S_C8_SDPA (sink-logit SDPA variant: extra zero key/value + additive float mask carrying the sink),
  S_C8_640 (640/5h), S_C8_D12 (depth 12) - 900 s each, cold compiles. Hypothesis for 640/depth: the register mis-tiling
  penalised width/depth in the earlier compile screens (T13 640-wide was 0.63 s/step with kv-len 1028).
- Backlog screens finished (eager kernel line, 900 s, ref 1.112149): value residual 1.113117 (+0.0010), backout 1.113406
  (+0.0013), PLE 1.128177 @2118 (+0.016, -12% steps). **Backlog verdict: only the 256-token window on odd layers helps
  (-0.0017, free); everything else is a wash or a loss on this device.** T30 (v15 bytes, LNC=2 launch) runs next, then the
  compile 8-way screens.
- **T30 (06:00 UTC) aborted: v15 bytes on the LNC=2 launch (4 workers x 8 rows) = 0.71 s/step** (8-way LNC=1: 0.31; the
  register version at LNC=2 was 0.44). The compiled sink-logit graph only tiles well at 4 rows/worker -> the compile line IS
  the 8-way launch. Second shape suspect: the sink-logit explicit path concatenates a sink column, so the compiled softmax
  is 1025 wide. New knob `SINK_IMPL=lse`: softmax over the 1024 real keys, output scaled by 1 - sigmoid(sink - logsumexp)
  (identical math and gradients on CPU); screen S_C8_LSE queued right after S_C8_BASE.
- **07:05 UTC: user's `findings/engine-profile-512.md` read** (fwd/bwd profiles: DMA 73%, all engines 25-47%, mfu ceiling
  ~15%, transposes = 23% of forward PE FLOPs, arithmetic intensity ~30). Queue additions on the compile 8-way line
  (ref S_C8_BASE 1.096323 @2714): `WEIGHT_T=1` (stacks and lm_head stored (in, out), applied as x @ W - removes the
  per-matmul weight transposes; CPU bit-exact vs default), merged QKV (fewer weight transposes; the 1536x512 stack fits the
  fused Muon kernel), and LR/schedule retunes for the ~5500-step regime: Muon LR x1.2, embedding LR x3, warmdown 0.9.
  Not queued: bf16 logits/CE (bf16 logit resolution ~0.06 would hurt the loss), flex attention (needs a register prefix,
  which mis-tiles), bigger per-worker batch (8 rows on a 12 GB LNC=1 core would exceed the compile-time HBM estimate).
- **07:40 UTC: eager hardware track started** (user: the engine reports show the hardware at ~15% MFU; the compiled graph's
  remaining 85% is inside the compiler, the eager line is where layout and fusion are ours). `nki_eager_opprof.py` (EPROF1,
  queued as a job) ranks the eager step's launches by time. `nki_mlp.py`: fused ReLU^2 MLP forward (hidden kept TRANSPOSED
  so the only transposes are 128x128 SBUF-local blocks of x; weights in the WEIGHT_T (in,out) layout = the stacks' storage)
  simulates at bf16 accuracy; backward (dx, dW1t, dW2t with fp32 SBUF accumulators, per-program partials summed in torch)
  written, simulating now. Planned next: `MLP=nki` autograd Function in the vehicle (eager, requires WEIGHT_T=1), device test,
  eager LNC=2 screen; then the fused head + softcap + CE kernel.
- **EPROF1 (07:05 UTC): eager per-op self-CPU time, single process, 8 rows, 567 ms/step, 8569 events.** `aten::_to_copy` 388
  launches / 257 ms (dtype casts: the fp32->bf16 cast of every stack slice at every layer call, casts around the attention
  op), `aten::ne` 2 / 112 ms (= the host blocking on the device at the loss mask, i.e. hidden device time), `aten::copy_` 295 /
  83 ms + `contiguous`/`clone` 238 (head-major relayouts around the attention kernels), `aten::mul` 186 / 26 ms; the NKI ops
  themselves cost 1-3 ms of dispatch each. Verdict: on the eager line the next win is launch REMOVAL (casts and copies), not
  another matmul kernel. Implemented `WCACHE=1`: one bf16 copy per stack per step (6 launches) reused by every layer call and
  by the MLP kernel; float bounds precomputed once (screen S_E2_WC queued). NKIML1 crashed on a VectorE rule the simulator does
  not enforce ("scalar_tensor_tensor: src0 and src1 cannot both read from PSUM") - fixed (ReLU^2 from the SBUF copy), NKIML2 +
  S_E2_MLP2 re-queued. S_C8_WIN (window on the compile line) +0.0007: the eager-line gain did not transfer; dropped.
- **Fourth NKI tracer rule (07:25 UTC):** S_E2_MLP's fused-Muon kernel failed to re-trace with "NKI does not support 'global'
  statements ... 'import' statements within a function" pointing at the BUILDER's lines: the tracer re-reads kernel source by
  line number, and kernel definitions nested inside builder functions break once the file around them moves. Fix: every kernel
  is now defined at module level inside one guarded `try: import nki ... NKI_AVAILABLE` block (regenerated from the dev
  files), builders only `wrap_nki` them - the structure the clean submission files already use. S_E2_MLP therefore ran with
  the torch MLP (kernel disabled by the fallback) + WEIGHT_T: 0.41 s/step (x @ W is slower than F.linear in eager).
- S_E2_MLP ran as WEIGHT_T-only (kernel disabled): 1.116724 @2186, 0.41 s/step vs 0.37 -> **the (in, out) layout is a loss in
  EAGER** (x @ W dispatches slower than F.linear); WEIGHT_T stays a compile-line knob (S_C8_WT). The MLP kernels are being
  reworked to take the natural (out, in) weights (128x128 in-kernel block transposes in the forward, none in the backward) so
  MLP=nki no longer needs WEIGHT_T; NKIML2 / S_E2_MLP2 retest with the standard layout.
- **Eager launch-removal screens (08:15 UTC): S_E2_MLP2 (fused MLP) 1.113318 @2374 (+0.0012), S_E2_WC (bf16 weight cache +
  float bounds + RoPE tables once) 1.112869 @2384 (+0.0007), both 0.37-0.38 s/step = no change.** Conclusion: the eager step at
  4 workers is device-bound, not launch-bound - EPROF1's big "cpu" numbers were host waits on the device. Fusions that pay are
  the ones that remove HBM traffic (attention: -10%, RoPE: 9 elementwise passes -> 1 kernel: -8%); a matmul-for-matmul MLP
  kernel or cast removal does not. Next eager targets must be traffic-removing fusions (norm + residual + lambda chains, the
  head + softcap + CE), or the eager line is at its plateau (~0.37 s/step, 1.0647 full).
- Queued two batch-shape screens on the compile 8-way line (both use the banked SINK_IMPL=lse): S_C8_R2 = 2 rows/worker with
  grad_accum 2 (the 8->4 rows/worker change was the single biggest compile-line speedup: does a smaller graph tile even
  better?), S_C8_B64 = 65k tokens/step via grad_accum 2 with the sqrt-LR rule (optimizer cost amortized over two micro-steps
  now that a run makes ~5500 steps).
- **Compile 8-way screens ledger (900 s, ref S_C8_BASE 1.096323 @2714, 0.31 s/step), 06:05-10:20 UTC:** concat-free sink
  (SINK_IMPL=lse) **1.095337 @2781 (-0.0010, keep)**; window 256 +0.0007; bf16 scores identical (wash); SDPA +0.0011; width 640
  1.120100 @1774 (+0.024); depth 12 1.106372 @2464 (+0.010); WEIGHT_T +0.0014 (the compiler handles weight transposes); merged
  QKV 1.106770 @2526 (+0.010, 0.35 s/step). Architecture and layout knobs are exhausted on this line; pending: Muon LR x1.2,
  embedding LR x3, warmdown 0.9, 2 rows/worker x accum 2, 65k batch.
- 11:10 UTC: S_C8_R2 (2 rows/worker x accum 2) +0.005 (discard: 4 rows/worker is the sweet spot). S_C8_B64 crashed: with
  TOTAL_BATCH=65536 the vehicle derives 8 rows/worker (min(8, 64/8)) and an LNC=1 core (12 GB) OOMs at compile -> re-queued as
  S_C8_B64b with DEVICE_BATCH=4 (grad_accum 2) after the v18 gate. Rule: on the 8-way launch keep DEVICE_BATCH at 4.
- **T31_gate (11:07-11:41 UTC): v18 = 1.051593 @5687 steps, 0.30-0.32 s/step - new best; `submit/train.py` = v18.** Queue-runner
  quirk: the runner exited ("QUEUE: done") right after T31 without seeing the two lines appended to its file while T31 ran
  (after earlier in-place rewrites of the same file); restarted as queue_NKI6 (S_C8_B64b, S_C8_PROF). Rule: append to a queue
  only while its runner is still several items away from the end, otherwise start a new queue file.
- **12:01 UTC: S_C8_B64b (65k tokens/step = two 4-row micro-steps per worker, all LRs x sqrt(2)) = 1.086675 @1675 vs the 32k
  reference 1.095337 @2781: -0.0087 at 900 s, 0.52 s/step**, the largest single compile-line screen gain so far (110M vs 91M
  tokens in the window; the sqrt-scaled LRs use them). Must be confirmed at full length because the ratio of optimizer steps
  changes over 30 min: T32_gate = v19 (v18 + TOTAL_BATCH 65536, `_BATCH_LR` sqrt scaling, rows capped at 4/worker), queued.
- **12:12 UTC: S_C8_PROF (v18 config, PROFILE=1, avg over steps 10-40): fwd_bwd 0.208 / grad_sync 0.027 / optimizer 0.073 /
  total 0.307 s/step.** The optimizer phase (NKI Muon + AdamW kernels, replicated on all 8 workers, plus the per-group scalar
  H2D copies and the elapsed-fraction broadcast) is 24% of the step. Its FLOPs are ~2 ms of PE time and its HBM traffic
  (~3 GB: 88M AdamW params x 8 passes, mostly the 2^17 x 512 bigram table) ~10 ms, so most of the 73 ms is something else
  (launch overhead, host syncs, or the kernels' DMA pattern). `nki_opt_bench.py` (MB_OPT, single core) times every component
  in isolation, including a sharded-Muon slice (1/8 of each stack) - if the kernel time scales with the slice, splitting the
  optimizer over the 8 data-parallel workers (each updates its slice, then broadcasts it: the same bytes as the gradient
  all-reduce moves today) buys up to ~20% more steps at fixed fwd/bwd.
- **12:50 UTC: T32_gate = v19 (v18 + 65k tokens/step + sqrt-batch LR) = 1.040894 @3376 steps, 0.52 s/step - NEW BEST, -0.0107
  vs v18's 1.051593.** The screen's -0.0087 at 900 s grew at full length (the 32k line's extra optimizer steps matter less late
  in the run than the sqrt-scaled LR on twice the tokens). `submit/train.py` = v19 (sha ca78aae3f672f3df); fresh-process eval-public of its checkpoint on the local GPU: 1.042488. This is inside the
  user's hold window (~1.03-1.04). Next on the same axis: 98k or 131k tokens/step (3-4 micro-steps) - the optimizer share of the
  step falls further, LR x1.73 / x2 by the same rule; screen S_C8_B96 / S_C8_B128.
- **13:10-13:30 UTC: optimizer-phase screens (S_C8_LSE line, PROFILE=1; ref fwd_bwd/grad_sync/optimizer/total =
  0.208/0.027/0.073/0.307 s).** S_C8_OPTS (scalar group as one flat tensor): 0.207/0.022/0.048/**0.277**, 1.086661 @3130
  (-0.0087 vs 1.095337 @2781, keep). S_C8_OPTZ (+ ZeRO-1: reduce_scatter grads, 7 params = 5 tables + lm_head + the 40-matrix
  Muon stack updated 1/8 per worker, all_gather back): 0.209/0.016/0.041/**0.266**, **1.082566 @3338 (-0.0128, keep)** - the reduce_scatter + all_gather pair is
  cheaper than the all_reduce it replaces (0.027 -> 0.016) AND the optimizer drops to 41 ms (the two n=10 Muon stacks, 8.7 ms,
  and the copies remain replicated). `submission_v20_clean_compile_8way.py` = v19 + both (no knobs; CPU 2-proc smokes of v19,
  v20-zero and a v20-scalar-only variant give identical losses / 2.961794). Gate: T33_gate, inserted before S_C8_B96/B128.
- **14:16 UTC: T33_gate = v20 (v19 + flat scalar group + ZeRO-1 optimizer sharding) = 1.036095 @3707 steps, 0.48 s/step - NEW
  BEST, -0.0048 vs v19.** `submit/train.py` = v20 (sha a508d2b77c4224e1; fresh-process eval-public on the local GPU 1.037430). Two pure hardware changes (no model change) delivered 10% more
  steps. S_C8_B96 (98k tokens/step) = 1.090275 @1190, +0.0036 vs 65k at 900 s: discard; S_C8_B128 running for completeness.
- 14:33 UTC: S_C8_B128 (131k tokens/step, grad_accum 4, LR x2) = 1.086538 @929 - equal to 65k at 900 s (1.086675 @1675) and
  better than 98k (1.090275 @1190; non-monotonic, single runs). Because the 65k gain grew from -0.0087 (screen) to -0.0107 (full),
  131k gets a full gate: **T34_gate = v21** (`submission_v21_clean_compile_8way.py` = v20 + TOTAL_BATCH 131072), inserted after
  S_V20_REF in queue NKI13 (then MATRIX_LR x0.8/x1.25, EMBED_LR x0.7, WD 0.05/0.20 at the 65k batch).
- **15:23 UTC: T34_gate = v21 (v20 + 131k tokens/step, LR x2) = 1.034802 @1978 steps - new best by 0.0013 over v20 (single
  runs; margin is small).** `submit/train.py` = v21 (sha 3c3a39bdda5bdb22; fresh-process eval-public on the local GPU 1.036173). Batch ladder at full length: 32k 1.0516 -> 65k 1.0409 (-0.0107)
  -> 131k 1.0348 (-0.0013 on top of the sharded optimizer). S_V20_REF (v20 config, 900 s) = 1.079482 @1855 is the screen
  reference for the LR/WD sweep now running.
- **16:35 UTC: data budget reached.** T34 (131k tokens/step, 146k tok/s) entered **epoch 2 at ~step 1720 of 1978**: the
  provided training set is exactly the 5 public shards (`PUBLIC_TRAIN_SHARDS = range(5)`, ~225M tokens; the eval node has the
  same, and more shards would be external data). At the current throughput (~260M tokens per 30 min) the last ~13% of a run
  repeats data at low LR. It did not hurt T34 (best so far), but every further throughput gain now buys repeated tokens, so
  the marginal value of speed is lower than before and data-order / second-epoch handling becomes a real knob.
- LR/WD sweep at the 65k config (ref S_V20_REF 1.079482 @1855): Muon LR x0.8 +0.0046, x1.25 -0.0003 (wash), embedding LR x0.7
  +0.0038, WD 0.05 +0.0003 (wash), WD 0.20 +0.0032. The optimizer hyperparameters are at their optimum for this batch.
  Queue NKI14 (v21 config, 131k): S_V21_REF, warmdown 0.95 / 0.8, final LR 0.0 / 0.10, warmup 50.
- 18:25 UTC: schedule sweep at the 131k config (ref S_V21_REF 1.081878 @982): warmdown 0.95 +0.0002, 0.8 -0.0001, final LR 0
  +0.0004, final LR 0.10 +0.0022, **warmup 50 steps -0.0012 (keep?)**. Everything but the warmup is flat; warmup 100 to screen,
  then v22 = v21 + the better warmup gates. Note: 900 s screens undervalue the 131k batch (S_V21_REF is +0.0024 vs the 65k
  reference at 900 s while v21 beat v20 at full length), so keeps here are directional only.
- **18:40-19:05 UTC: embedding-backward hypothesis and the sidecar.** engine-profile-512: backward hardware FLOPs 3.10 TF vs
  ~1.7 TF expected for the model (2x fwd 0.75 + transposes 0.2); the ~1.4 TF excess equals the cost of the embedding
  backwards lowered as dense one-hot matmuls (bigram 2^17 rows: 8192x131072x512x2 = 1.1 TF; 4 vocab tables 0.28 TF). Diagnostic
  screens queued (v21 config, 600 s, PROFILE=1): S_V21_REF6 / S_V21_BG14 (bigram 2^14) / S_V21_BG0 (no bigram).
  Independent of the result, the fix is written: **EMBED=sidecar** in the vehicle - two NKI kernels (`nki_embed.py`, module
  copies in the vehicle): `embed_gather` (indirect DMA, `table.ap(vector_offset=idx)`) and `embed_scatter_add`
  (`nisa.dma_compute` read-modify-write add into the fp32 table, `oob_mode.skip` for padding). dma_compute needs distinct
  indices within one 128-row instruction, so the host groups slots into rounds (r-th occurrence of each distinct index) padded
  to 128 with the out-of-bounds index; real data: token ids repeat up to ~140-190x per 4096-token micro-batch (-> ~150 rounds
  for wte/vte/vte_mid/vte2, which share one index vector), bigram hashes ~25-50x. The compiled graph takes the 5 dense bf16
  row tensors as inputs (`forward(..., emb=...)`), returns their grads, and the sidecar accumulates them into persistent
  fp32 table grads that are attached before the (sharded) sync. Simulator: both kernels exact incl. duplicates; CPU 2-proc
  smoke of the wiring (`EMBED=sidecar_torch`, index_select/index_add_) identical to the graph path (3.047688). Device test
  JOB NKIE1 and screen S_V21_SC queued after the diagnostics.
- **19:25 UTC: hypothesis confirmed by ablation.** v21 config, 600 s, PROFILE=1: S_V21_REF6 fwd_bwd 0.860 s/step (4 micro-steps),
  1.118470 @648; **S_V21_BG14 (bigram table 2^14 instead of 2^17) fwd_bwd 0.705 s (-18%), 1.109832 @778 (-0.0086 even with the
  smaller table)**. The embedding backward's cost scales with table rows = dense one-hot matmul lowering. F_V21_D12 (12 layers,
  full length) = 1.039185 @1738: +0.0044 vs v21, discard at this throughput. Queue: S_V21_BG0, JOB NKIE1 (sidecar kernels on
  device), JOB PROFV21 (per-op profile of the current fwd/bwd NEFFs -> runs/prof_{fwd4,bwd4}), S_V21_SC (sidecar screen),
  F_V21_W640, S_V21_WU100, JOB FP8PROBE, S_V21_BGK4.
- 19:50-20:00 UTC: NKIE2 (fixed padding) on device: scatter RMW exact - 128 rows 0.11 ms, 4096 distinct 0.28 ms, 1063 rounds /
  137k slots 7.6 ms (~7-9 us per 128-row tile); eager `index_add_` into an 8192-row table 2.7 ms and exact (variant e tests
  the 131072-row table). **S_V21_SC2 ran at 8 s/step: the scatter kernel's slot count is data-dependent, so wrap_nki
  recompiled it every micro-step** (PROFILE fwd_bwd 1.26 / grad_sync 1.24 = straggler waits) -> slot counts bucketed to
  powers of two >= 4096 (padding tiles ~9 us each); simulator exact; NKIE3 (variant f checks no recompile on a second batch)
  and S_V21_SC3 queued. Rule (also for PROFV21b): profiler `neuron-explorer view` is not a python process, so the queue's
  JOB wait returned while it still ran beside the next screen (load 9); profile jobs now run through `trn2_profile_ops.py`.
- **20:02 UTC: S_V21_SC3 (sidecar, bucketed slots) = 1.109105 @705 vs 1.118470 @648 (-0.0094 at 600 s), 0.81 s/step vs 0.92,
  full 2^17 bigram table kept.** Steady-state gain 12% (target 25%: the remaining ~0.1 s/step is host plan work, H2D copies,
  10 kernel launches and straggler waits per step). S_V21_SC4 = same with the host plan overlapped with the backward and one
  upload per index kind; NKIE4 = device timings incl. eager index_add_ on the 131072-row table.
- **20:15 UTC: S_V21_SC4 (sidecar, host plan overlapped with the backward, one upload per index kind) = 1.101897 @761 vs
  1.118470 @648 (-0.0166 at 600 s), 0.76 s/step, 173k tok/s.** S_V21_SCT (same wiring with eager index_select/index_add_, no
  kernels) 0.83 s/step: the NKI gather/scatter is worth ~9% over the device's own index ops, which are exact and cheap
  (index_add_ 2.7 ms / 4.0 ms for the 8192- / 131072-row tables, NKIE4). **v23 = v21 + sidecar** (`make_v23.py` extracts the
  screened segments from the vehicle; NKI on Neuron, index ops elsewhere; sha 34155e0b773e407b): CPU 2-proc smoke identical to
  v21 (2.857338). T35_gate queued after SCT.
- **21:00 UTC: T35_gate = v23 (v21 + embedding sidecar) = 1.026999 - NEW BEST, -0.0078 vs v21, first score under 1.03.**
  `submit/train.py` = v23 (sha 34155e0b773e407b; 2316 steps; fresh-process eval-public on the local GPU 1.028285). The compiled graph no longer contains the embedding tables; their gather and
  scatter-add run as eager NKI kernels around it (0.76 vs 0.91 s/step). Next: fresh-process eval, then the full-length
  re-spend runs (F_V23_W640, F_V23_BG18, F_V23_D12).
- Profiling note: `neuron-explorer view --output-format=parquet` returns immediately and fills the files from a background
  process (PROFV21b/c: 0-byte parquet at job exit, the worker then ran at 300% CPU beside the next screen twice; killed
  both times). The per-op profile is parked: the big waste was found by ablation. If revisited, run it in a dedicated
  window with the queue paused, and poll for the explorer process to exit.
- 21:35 UTC: F_V23_W640 (width 640 + sidecar, full) = 1.033460 @1575 vs v23 1.026999 @2316: +0.0065, discard (1.13 s/step,
  159M params). Width is not the lever at this throughput; depth 12 + sidecar and bigram 2^18 + sidecar queued at full length.
  Added to the vehicle and queued as 600 s PROFILE screens on the v23 config (ref S_V21_SC4 1.101897 @761, 0.76 s/step):
  `CKPT=mlp` (torch.utils.checkpoint around the MLP block: recompute instead of saving the 2048-wide intermediates; CPU
  numerics identical) and `ATTN=flex` + `DOCMASK=score` with the logit sink through flex_attention's returned logsumexp
  (the concat-free sink gate), i.e. the backend's inlined NKI attention with no materialised score matrices.
- 21:52 UTC: S_V21_WU100 = 1.083923 @974 (+0.0020 vs ref; warmup 50 stays the marginal keep). **FP8PROBE**: eager float8
  casts and `torch._scaled_mm` work on the device, but inside `torch.compile(backend="neuron")` an e4m3fn cast fails to
  compile and an e5m2 round trip compiles with zero error (the compiler folds it away). fp8 saved activations / fp8 matmuls /
  a quantized bigram table are therefore not buildable on this build - parked.
- 22:40-23:15 UTC full-length re-spend runs (ref v23 1.026999 @2316): **bigram 2^18 = 1.024717 @2278 (-0.0023, keep)**; depth 12 =
  1.029442 @2018 (+0.0024, discard; the gap halved vs before the sidecar - depth pays once steps are cheaper still). v24 =
  v23 + 2^18 + warmup 50 (sha f387d28aed03940b) gating as T36 (23:15-23:50 UTC).
- **23:48 UTC: T36_gate = v24 (v23 + bigram 2^18 + warmup 50) = 1.025514 @2274 - new best (-0.0015 vs v23); `submit/train.py`
  = v24 (sha f387d28aed03940b; fresh-process eval-public 1.026164).** The vehicle's 2^18-only run was 1.024717, so warmup 50 is worth nothing or slightly negative at
  this batch (noise ~0.001); not worth a separate gate.
- 00:00-00:21 UTC (Sep 8): S_V23_CKPT (MLP recompute) 1.101694 @758, 0.77 s/step = wash; S_V23_FLEX (flex_attention +
  return_lse sink gate, DOCMASK=score) 1.618199 @607, 0.95 s/step = broken and slow: with the logit sink the backend's flex
  lowering is not the inlined kernel and the numerics differ (doc-score dims + explicit scale + lse gate) - dead end.
  Queue NKI15 done; the box is idle.
- 00:25-00:40 UTC (Sep 8): F_V24_BG19 (2^19 bigram) = device OOM at step 0 on the 12 GB LNC=1 core -> 2^18 is the ceiling on
  this launch. The sidecar now also carries the hashed trigram table (`TRIGRAM=1`: host hash, own kind "tg"; CPU smoke
  identical to the graph path, 3.048361) - it was a compile-line discard before only because its backward was another one-hot
  matmul. Queue NKI17: S_V24_REF6 (v24 config, 600 s), S_V24_TRI, then F_V24_EP2 (full: `EPOCH2=rotate` = the second pass reads
  the 5 shards starting from the middle one instead of replaying the identical order).
- **Portal (2026-09-07 17:13 PT): v24 = 1.0311, 9th** (local 1.0255 in-run / 1.0262 fresh: the portal gap is +0.005, as for
  v14). Board: 1st 0.9848, 6th 0.9997, 7th 1.0127, 8th 1.0192, 10th 1.0315. Targets in local terms: 8th needs ~1.014, the
  1.00 range needs ~0.995 (0.03 below the current local best).
- 01:00-01:35 UTC: S_V24_TRI (trigram via sidecar) 1.106139 @715 vs ref 1.101172 @747: +0.0050, discard. F_V24_EP2 (rotated
  shard order for the second pass) 1.043139 @2258 vs v24 1.025514: +0.0176 with a LOWER train loss (2.61 vs 2.71 at the end) -
  the second loader repeats recently seen rows (rank-strided row groups shift with the rotation), i.e. memorisation; discard.
  **PROFV24 (sidecar graph, 4 rows): fwd 0.051 s / bwd 0.082 s; PE active 62% / 57%; DMA active 53% / 53%; hbm_read 2.9 / 10.8
  GB, hbm_write 6.8 / 3.5 GB per micro-step; hardware FLOPs 0.51 / 0.85 TF (the one-hot 1.4 TF is gone); MFU 9.4% / 11.7%,
  max achievable 36% / 48%.** The graph is now latency/scheduling-bound (neither engine saturated, 300k tensor-engine
  instructions per backward) and its HBM traffic is dominated by the fp32 (b,h,t,t) attention tensors (~6.8 GB written per
  forward). Queue NKI19: S_V24_T2048 (context 2048, 2 rows) and S_V24_SBF16 (bf16 score matrices) at 600 s.
- 01:50-02:04 UTC: S_V24_T2048 (context 2048, 2 rows) 1.105086 @620 (+0.0039, 0.96 s/step) and S_V24_SBF16 (bf16 scores)
  1.106278 @708 (+0.0051, 0.81 s/step): both discards. Box idle (queue NKI19 done). Local best remains v24 = 1.0255
  (portal 1.0311, 9th).
- 02:10 UTC: host plan cost measured at 0.5 + 0.2 ms per micro-step (not the straggler source; the profile-window sync waits
  are the bucket compiles). New vehicle knob **EMA** (`NEURON_COMPETITION_R1_EMA=<decay>`, `EMA_START=<budget fraction>`):
  an EMA of all weights from EMA_START on, swapped in before the checkpoint save and the evaluation (CPU smoke ok). Queue
  NKI20: decay 0.99 / 0.995 / 0.998 at 600 s (ref S_V24_REF6 1.101172 @747).
- 02:30 UTC (Sep 8): reviewed `findings/offline-compile-gap.md`. Verdict: cache pre-warm cannot reach the portal (fresh
  compile per submission; a NEFF cannot be shipped); the box keeps its cache so 4-row graphs pay a 2-3 min cold compile
  once; the 900 s startup cap only ever bit 640-wide at LNC=2 (T13), which is out on merit now. Offline `neuronx-cc` on a
  dumped HLO is the right way to screen compiler flags without queue slots, but no flag has ever changed the graph and the
  profile says the loss is scheduling/DMA latency, so the tool is not built. Rule kept: **cold-compile every new v# once
  with a fresh cache dir and confirm startup << 900 s before it goes to the portal.**
- **02:20 UTC: S_V24_EMA99 crashed mid-run on one worker: `[NkiDispatch] lower_func failed` for the scatter kernel at a
  never-seen slot count** (a micro-batch with one index repeated > 256 times doubles the plan to 65536 slots; the first
  compile of that variant failed). Rare (two full gates never hit it) but a mid-run crash the step-0 fallback does not
  cover = a portal risk in v24. Fix: plans capped at 32768 slots and cut into chunks on tile boundaries (unit-tested:
  per-tile uniqueness, no lost rows), and every kernel call wrapped so a failure falls back to the device's `index_add_`
  for that micro-step. **v25 = v24 + this fix (same math)**; CPU smoke vs v24 pending; it replaces `submit/train.py` after
  a device screen confirms it runs.
- 02:55 UTC (Sep 8): user direction - if the eager all-kernels pricing screen (S_V24_EAGERK: 8 workers, 4 rows, sidecar,
  ATTN/MLP/NORM/ROPE=nki, no compile) lands within ~2x of the compiled 0.76 s/step, build a per-layer NKI "megakernel" line
  (layer forward fused first, then the layer backward, then head + CE), gated at full length against v24 with the compiled
  graph kept as the in-file fallback. Rationale: the compiled graph reaches ~10% MFU against a 36-48% profiler ceiling; every
  win on this hardware so far came from replacing compiler-scheduled ops with one pipelined kernel.
- **02:57 UTC: S_V25_CHK = 1.101222 @747 (600 s) = the v24 reference to 5e-5; `submit/train.py` = v25 (sha c2652cba20c66cbd)** - same model
  as v24 (1.0255 local / 1.0311 portal) with the scatter plans capped at 32768 slots and an index_add_ fallback per call.
- 03:15 UTC (Sep 8): user asked to test a routing architecture off the compiler. Mixture-of-depths with sequence top-k is
  non-causal (fails causality_check and the rules), so a **causal static-capacity MoD** went into the GPU vehicle
  (`train2.py`, knobs MOD_LAYERS / MOD_CAP): per MoD layer a router r_i = w.x_i; the first k = cap*t tokens (position order)
  with r_i > 0 go through the block, the subset is padded to exactly k with the earliest unselected tokens, fillers are
  masked out of the keys and gated to 0, the update is x_i += sigmoid(r_i) * (block(x)_i - x_i). CPU smoke: runs, causality
  check exact. GPU (idle 4080S, 720 s fixed time, L2 champion config): L5_BASE reference then L5_MOD (layers 1,3,5,7,9, cap
  0.5). If MoD wins per wall-clock it justifies a static-capacity NKI kernel line; if not, routing is closed.
- **03:33 UTC: causal MoD on the 4080S (720 s fixed time): L5_BASE 1.117662 vs L5_MOD 1.144366 (+0.0267) with ~18% more
  steps.** The quality lost by routing half the tokens around five blocks is far larger than the compute it saves at this
  scale, even before any Trainium kernel work. Routing/MoD is closed; no custom kernel line for it.
- 04:01 UTC: OPPROF2 (eager, all kernels, one core, 4 rows): 391 ms per micro-batch step vs ~143 ms for the compiled graph +
  optimizer, i.e. the launch-by-launch kernel line is ~2.7x slower at this shape; the profile's host side is dominated by
  casts/copies between ops (_to_copy 162 ms, copy_ 72 ms), not by the kernels (attention/RoPE launches ~5 ms). A per-layer
  fused kernel removes that glue, but the kernels' own device time is not captured here; the 8-way S_V24_EAGERK screen gives
  the step-level number. S_V24_ASC (attention scale 0.1) -0.0010, S_V24_NOMLP0 +0.0006 with 3% more steps (wash).
- **04:25 UTC: megakernel pricing.** S_V24_EAGERK (eager, all NKI kernels launch-by-launch, sidecar, 8 workers x 4 rows):
  fwd_bwd 1.508 s per 131k step vs 0.713 compiled (2.1x); OPPROF2 says the eager excess is cast/copy glue between ops, not
  the kernels. Verdict against the agreed rule (< 1.5 s commit, > 2.5 s no): borderline-favourable - a per-layer fused kernel
  removes the glue and can plausibly reach parity; beating the compiler needs kernels that pipeline better than its code.
  S_V24_LNC2 (two physical cores per 8-row graph) 1.107144 @705, 0.84 s/step: closed.
- 04:35 UTC: stage-1 megakernel work started. `nki_layer.py::post_attn_fwd` = out-projection + residual + rmsnorm + ReLU^2 MLP
  + residual as one kernel (built on nki_mlp.mlp_fwd's tiling; saves x1 and preT for the backward). Simulator: rel error
  0.3-0.4% vs torch (bf16 outputs). Device timing vs the same math as 6 eager ops: JOB NKIL1 (queued after the tail-EMA run).
  Queue-edit lesson repeated: an insert right after the line the runner just launched shifts its read offset into the
  middle of the next line - insert AFTER the running line, never before it.
- **05:05 UTC: F_V24_TEMA = 1.024206 @2272 (-0.0013 vs v24 at no step cost, keep). NKIL1: `post_attn_fwd` (out-proj +
  residual + rmsnorm + ReLU^2 MLP + residual, one kernel) = 0.49 ms per 4096 tokens on device (~19 GFLOP -> ~39 TFLOP/s,
  about half the core's peak) vs 1.06 ms for the same math as 6 eager ops and ~5.1 ms per layer for the compiled graph's
  entire forward.** This is the stage-1 evidence: one pipelined kernel runs this slice of the layer an order of magnitude
  faster than the compiler's code. `pre_attn_fwd` (residual mix + rmsnorm + Q/K/V + RoPE + QK-norm + head layout) failed
  to lower for a tuple-unpacking loop (tracer rule), fixed and re-queued (NKIL2). Next: the fused backward halves
  (post_attn_bwd around nki_mlp.mlp_bwd, pre_attn_bwd), then a layer-kernel training line measured against v24.
- **05:40 UTC: F_V24_ROPE = 1.022315 @2258 (-0.0032 vs v24 at full length): partial RoPE + key offset confirmed.** NKIL2/NKIL3:
  pre_attn_fwd 0.85 ms, post_attn_fwd 0.46 ms, vendor attention fwd 1.30 ms; **the 3-kernel layer forward = 3.36 ms per layer
  = 33.6 ms per 10-layer micro-step vs ~51 ms compiled (1.5x, glue included)**. v26 = v25 + ROPE_FRAC 0.5 + KEY_OFFSET +
  tail-EMA (sha 3d623c8a0065fbca; CPU smoke ok, causality exact) -> T37_gate, moved ahead of the prime-modulus run.
- **06:40 UTC (Sep 8): T37_gate = v26 (v25 + partial RoPE/key offset + tail-EMA) = 1.021352 @2261 - NEW BEST, -0.0042 vs v24.**
  `submit/train.py` = v26 (sha 3d623c8a0065fbca). S_V24_ASCPH (phased attention scale) wash; S_V24_KOPH (key offset gated to 0 late)
  +0.012 vs the constant offset: the offset is a permanent circuit.
- **06:45 UTC: backward chain (NKIL5) = 7.51 ms per layer = 75 ms per 10-layer micro-step vs ~82 ms compiled** (post_bwd 1.50 ms
  via mlp_bwd + rmsnorm_bwd + 2 matmuls, vendor attention bwd 3.22, pre_bwd 2.54 via norm/rope bwd kernels + 6 matmuls).
  With the 3-kernel forward (33.6 vs 51 ms) the kernel line stands at ~109 ms per micro-step vs 133 compiled BEFORE fusing
  the backward halves; the remaining eager glue (permutes, contiguous copies, weight transposes) is the next target.
  v26 fresh-process eval-public: 1.022568.
- **07:00 UTC (Sep 8): LAYER_IMPL=chain in the vehicle (`_NKILayer`: pre_attn_fwd -> vendor attention -> post_attn_fwd
  forward; mlp_bwd/rmsnorm_bwd/rope_bwd kernels + attention bwd + matmuls backward; weight grads into the stack slices).**
  NKIL6: loss identical to the eager kernel path (9.01088), all weight grads within bf16 tolerance. S_V26_CHAIN (8 workers,
  sidecar, 300 s): fwd_bwd 1.376 s/step = 0.344 s per micro-step vs 0.713/4 = 0.178 compiled and 0.377 for the eager
  all-kernels line. The layer kernels are ~109 ms of that; the other ~235 ms per micro-step is the eager glue around the
  layers (lm_head + softcap + CE fwd/bwd, residual/skip/x0 bookkeeping, the final norm, value-embedding selection, sidecar
  launches). Verdict so far: the kernels beat the compiler inside the layer, the eager framework around them costs more than
  they save; a winning kernel line needs the head + CE and the residual bookkeeping fused too (OPPROF3 ranks the remainder).
- **07:05 UTC: chain line verdict.** S_V26_CHAIN = 1.274477 @205 (300 s; 1.45 s/step vs 0.77 compiled). OPPROF3 (single
  core, chain, no bigram): 251 ms per micro-batch of which `_to_copy` = 565 dtype casts = 125 ms, `ne` 33 ms, `mul` 130
  calls 19 ms, `copy_` 309 calls 13 ms, `empty` 1634 allocations; the 20 layer kernels + 20 attention ops cost ~13 ms of host
  time. On this runtime every eager op is a separate device program with ~0.2-0.5 ms of overhead, and a training step still
  has ~1500 of them around the layer kernels (casts, residual/skip bookkeeping, head + CE, optimizer glue). **The kernels beat
  the compiler inside the layer (fwd 33.6 vs 51 ms, bwd 75 vs 82 ms per micro-step) but the eager framework around them costs
  2x what they save.** Paths to a win, each substantial: (a) one kernel per layer INCLUDING the residual/skip bookkeeping and
  a fused head + CE kernel, with weight casts cached per step (removes ~90% of the ops); (b) torch.compile the glue with the
  whole layer as one custom op (untested; partition launches would likely land at parity). Parked for tonight; the compiled
  line (v26) stays the submission.
- 07:40 UTC: F_V26_D12 = 1.022222 @1953 vs v26 1.021352 @2261: +0.0009 - depth 12 is now within noise of depth 10 (the gap
  was +0.0044 on v21 and +0.0024 on v23). Any further step-time gain flips it to a keep; keep it as the first re-spend.
- 09:30 UTC (2026-09-08), megakernel line, decisive measurement: **the eager kernel chain is HOST-bound.** NKIL7 (stage-1 cast
  cleanup, bf16 weight-grad matmuls, WCACHE): loss identical, chain 159 ms vs eager torch 166 ms per micro-step at DEPTH=2.
  OPPROF5/6/7 (single proc, 4 rows, DEPTH=10): fwd+bwd enqueue 150-165 ms vs synced 157-171 ms, optimizer 27-40 vs 30-43 ms,
  step 183-200 ms with 7,162 dispatched ops. The device finishes within ~7 ms of the host's last enqueue, so no kernel fusion
  inside the layer can move the step time until the op count drops by ~10x. The torch profiler records no stacks on this build
  and a TorchDispatchMode breaks the nki_kernel_wrapper HOP (NKI Muon/AdamW silently disabled), so per-line attribution is
  not available; dropped. Consequence: the layer kernels go INSIDE the compiled graph as custom ops (`LAYER=op`:
  `tf_nki::layer_fwd` / `layer_bwd` built like the attention ops, weight grads back through autograd), which keeps the
  in-layer gains (fwd 33.6 vs 51 ms, bwd 75 vs 82 ms per micro-step) and removes the host from the critical path. Screen
  S_V24_OP (600 s, v24 config + COMPILE=1 ATTN=nki LAYER=op) queued against S_V24_REF6 (1.101172 @747, fwd_bwd 0.713 s).
- 09:30 UTC: fused head kernel (`nki_head.py`): final rmsnorm + lm_head + asymmetric softcap + online cross-entropy in one
  forward (per-row loss/lse, h, rstd out; W^T resident in SBUF; logits never reach HBM), backward recomputes each 512-wide
  vocab chunk (dh accumulated in SBUF fp32, dW chunk in PSUM over the token tiles, rmsnorm backward at the end). Simulator:
  forward exact vs torch, dx rel err 0.6%, dW 0.3% (bf16 level). Device timing (HEADT) queued after S_V24_OP.
- 10:10 UTC (2026-09-08): **first megakernel-line win: the fused head.** S_V24_HEAD (v24 config + HEAD=nki) = 1.097545 @~790
  vs S_V24_REF6 1.101172 @747: -0.0036 at 600 s, numerics identical through step 2, fwd_bwd 0.688 vs 0.713 and grad_sync 0.078
  vs 0.102 s/step (total 0.813 vs 0.862, -5.7%). On device (HEADT3, one core, 4096 tokens): head fwd 3.25 ms + bwd 7.62 ms
  vs the eager torch head 28.9 ms fwd+bwd (2.7x), dx/dW at bf16 error level. The (T, 8192) fp32 logits, the softcap
  elementwise chain and the log-softmax never reach HBM. Full gate F_V26_HEAD (v26 config + HEAD=nki) queued vs T37
  1.021352 @2261; clean file v27 (= v26 + the head route with a torch fallback) generated, passes `prepare.py check`,
  T38_gate queued behind it.
- 10:10 UTC: **LAYER=op is a discard as-is**: the layer kernels as custom ops inside the compiled graph run 0.85 vs 0.78 s/step
  (fwd_bwd 0.778 vs 0.713; S_V24_OP 1.108903 @~690 vs 1.101172 @747) although the same kernels beat the compiler in isolation
  (109 vs 133 ms per 10-layer micro-step). Not the stack grads (compile fuses the view path already). Suspects: the glue inside
  the op bodies (permutes / casts / contiguous copies scheduled as separate ops), 13 saved outputs per layer, and lost
  cross-layer overlap at the custom-call boundaries. Needs a NEFF profile of the op graph before another attempt; S_V24_OPHEAD
  (both routes) still runs for the record.
- 10:57 UTC (2026-09-08): **F_V26_HEAD = 1.019731 @~2340 (new local best)** vs T37 (v26) 1.021352 @2261: -0.0016 from ~3.5%
  more steps (0.77 vs 0.80 s/step), numerics identical through step 2. S_V24_OPHEAD (layer op + head) 1.105865: the head
  saves ~27 ms/step on top of the layer op too, the layer op costs ~65 ms/step -> discard. Clean file v27 (v26 + head route,
  torch fallback) in T38_gate; F_V26_HEAD_D12 queued behind it (depth 12 was +0.0009 at parity on v26; the freed 6% may flip it).
- 11:35 UTC (2026-09-08): **T38_gate (clean v27) = 1.019536 @~2350 -> `submit/train.py` = v27** (sha a396e2dd36910e3a).
  Fused head confirmed on the clean file ("fused head enabled" printed, numerics identical to the vehicle). F_V26_HEAD_D12 running.
- 12:15 UTC (2026-09-08), NEFF profiles (PROFOP/PROFHEAD + forward NEFFs): device time per micro-step, op route 78.5 (fwd)
  + 72.3 (bwd) = 150.8 ms vs head route 47.7 + 79.6 = 127.3 ms; the 23.5 ms gap equals the measured 90 ms/step. So the
  layer-op loss is entirely in the FORWARD graph (+31 ms although the forward kernels take ~34 ms per 10 layers in
  isolation); the op-route backward is 7 ms faster than the compiler's. Second finding: the head route's wall micro-step is
  172 ms vs 127 ms of NEFF time -> ~45 ms per micro-step (26% of the step) is spent outside the compiled graphs (eager sidecar
  gather/scatter kernels, host loader/plan, launch gaps). That is the next lever; `PROFILE=2` (synchronized per-micro-step
  split) added to the vehicle to attribute it.
- 13:20 UTC (2026-09-08) CORRECTION: **S_V24_OP / S_V24_OPHEAD never ran the layer ops.** The `LAYER=op` branch needs
  `rope_tabs`, which `GPT.forward` only builds when `ROPE=nki` (not in the v24 env). The NEFF profile of that graph shows
  only the vendor `attention_cte` and the explicit StackedLinear / F.linear matmuls, no custom-call layer op. What those
  screens measured is the vendor NKI attention kernel INSIDE the compiled graph (`ATTN=nki`): 0.778 vs 0.713 s/step, i.e.
  the in-graph vendor attention is 65 ms/step slower than the compiler's explicit attention (the clean line rightly keeps the
  explicit path). The "op forward graph +31 ms" reading above is therefore about ATTN=nki, not our kernels. Branch condition
  fixed (rope tables also built for LAYER=op); the real LAYER=op screen is S_V24_OP2.
- 13:25 UTC (2026-09-08): **S_V24_OP2 (the real LAYER=op): fwd_bwd 0.466 vs 0.713 s/step, total 0.595 vs 0.862 (-31%),**
  cold compile 138 s, step-2 loss 8.9903 vs 8.9902. With rope tables also built for LAYER=op the op branch fires and the
  compiled graph carries one custom-call pair per layer (3 fused NKI kernels forward, kernel chain backward, weight grads
  through autograd). Queued: S_V26_OPHEAD (600 s) + F_V26_OPHEAD (full, v26 config + layer ops + fused head) and the clean
  file v28 (= v27 + the layer-op route, training only, torch fallback; `$S/make_v28.py` extracts the kernels/builders from
  the vehicle) in T39_gate. PROFILE=2 of the head route: per micro-step gather 2.3 / fwd 59.0 / bwd 99.6 / host 7.3 /
  scatter 11.4 ms vs NEFF time 47.7 + 79.6 -> ~31 ms/micro-step of launch overhead + 11 ms scatter remain as the next levers.
- 13:50 UTC (2026-09-08): S_V26_OPHEAD (v26 config + layer ops + fused head) = 1.068671 @~1080 at 600 s, 0.606 s/step
  (v26 + head alone 0.77-0.80), clean finish with EMA + eval. Full F_V26_OPHEAD running, then T39_gate (clean v28), then
  S_V26_OPHEAD_CPUL (LOADER_DEV=cpu: the sidecar's host plan currently blocks on the device because prepare.py's loader
  creates batches on the device; PROFILE3 showed ~95 ms/step of host launch work the device does not overlap), then
  F_V26_OPHEAD_D12 (depth 12 re-spend at 0.6 s/step). The two dev-mode 300 s screens (PROFILE=2/3) crashed at their end
  with an async NRT error; no normal run has.
- 14:22 UTC (2026-09-08): **F_V26_OPHEAD = 1.012237 @~3240 (new local best; v27 1.019731, v26 1.021352)** - layer ops +
  fused head on the v26 line at 0.601 s/step. Expected portal ~1.017 (8th; 7th is 1.0127). T39_gate (clean v28) running,
  then S_V26_OPHEAD_CPUL and F_V26_OPHEAD_D12.
- 14:58 UTC (2026-09-08): **T39_gate (clean v28) = 1.012167 @~3240 -> `submit/train.py` = v28** (sha 67b631e733b97665); matches the vehicle.
- 16:05 UTC (2026-09-08), afternoon screens on the op+head line (v26 config, 0.60 s/step baseline): depth 12 full
  1.011644 vs 1.012237 (parity, stays 10); LOADER_DEV=cpu wash (fwd_bwd 0.495: H2D copies block like the D2H did);
  DEVICE_BATCH=8 (two micro-steps per step) 0.575 s/step, 600 s 1.067314 vs 1.068671 -> small keep, full F_V26_OPHEAD_DB8
  + clean v29 (T40_gate) queued; GRAD_BF16 collectives: no time gain (grad_sync 0.083, latency-bound) and +0.0006 ->
  discard; loss+grads as ONE compiled graph via torch.func.grad fails under dynamo on this build ("RuntimeError when
  making fake tensor call") -> parked. Four dev-mode runs today died with the async sidecar OOB; no production-shaped run has.
- 17:12 UTC (2026-09-08): **T40_gate (clean v29 = v28 + 8-row micro-batches) = 1.011566 @~3340 -> `submit/train.py` = v29** (sha c6bebb0660b45081).
- 18:20 UTC (2026-09-08): batch re-tune on the v29 line: 196k +0.0006 (wash), 262k +0.0038 (discard) -> 131k stays. Width
  640 blocked in the kernels ("matmul moving free dimension 640 exceeds max 512": D is used as a free size; >512 needs
  chunking in pre/post/head kernels). Running: matrix-LR x0.8 / x1.2 full runs, then the fake-fp8 numerics screen
  (FAKE_FP8=1: forward matmul operands rounded to e4m3 with per-row scaling inside the kernels; simulator-verified).
- 22:15 UTC (2026-09-08), fp8 numerics probe: FAKE_FP8=1 (e4m3 rounding with per-row scaling of every forward matmul
  operand inside the layer and head kernels; simulator- and device-verified after two portability fixes: no `abs_max`
  reduce on Trn2, no in-place scalar ops) trains at parity with bf16 at matched steps (600 s screen). Real fp8 is
  therefore a pure speed question; the kernels are vector-bound today, so it waits for the elementwise trimming.
  Also today on the v29 line (full runs): matrix LR x0.7 1.010506 (best), x0.8 1.010716, x1.2 1.013779, warmdown 0.95
  wash; attention scale 0.1 1.010756 (-0.0006); bigram sign trick and prime table size washes at 600 s; batch 196k/262k
  washes/discards; width 640 blocked by the kernels' 512 free-dim cap; one-graph loss+grads blocked by dynamo.
- 00:10 UTC (2026-09-09): **T41_gate (clean v30 = v29 + matrix LR x0.7 + attention scale 0.1) = 1.010664 -> `submit/train.py` = v30** (sha d61d510fcb37867d).
- 01:15 UTC (2026-09-09): **embedding LR x3.0 (EMBED_LR_SCALE 2.0 -> 3.0) on the v30 config = 1.006512 vs 1.010664**
  (-0.0042, the largest single schedule gain of the campaign); x1.5 = 1.012650. Curve still falling: x4.0 / x5.0 inserted
  after the running prime-table run; v31 = v30 + best embedding LR (`$S/make_v31.py <scale>`) to gate next.
- 04:00 UTC (2026-09-09): embedding-LR sweep on the v30 config bottomed at **x6.0 = 1.002104** (x5 1.0033, x8 1.0031;
  x10 running for the record). v31 = v30 + embedding LR x6.0 (sha 22d2a0749a9854cc) in T42_gate right after; then head-LR
  x2.0 / x0.5 on the x6.0 config and the sign trick at full length.
- 05:05 UTC (2026-09-09): **T42_gate (clean v31 = v30 + embedding LR x6.0) = 1.002410 -> `submit/train.py` = v31** (sha 22d2a0749a9854cc).
- 05:40 UTC (2026-09-09), bigram-collision measurement (local, 16,000 packed rows = 16.4M tokens, the vehicle's hash, 2^18
  rows): 1.91M distinct bigrams (load 7.3 per table row, still growing); tokens whose bigram shares a table row with a
  DIFFERENT bigram in the same 1024-token row: **0.29%** (70% of rows contain at least one). An in-context "fast-weight
  collision repair" can affect at most 0.3% of positions -> dead. Capacity (more rows) is the actual limit; 2^19 rows OOMs
  on the 12 GB core (F_V24_BG19), so a bf16 table or narrower rows would be the route. GPU-line facts for the
  delta-rule proposals: E25 GDN hybrid -0.0251, E25d KDA -0.0131 more; doc-boundary reset was WORSE (+0.0068, E25c);
  5:5 ratio optimum. Not started on Trainium: chunked delta fwd+bwd kernels are days of work.
- 07:00 UTC (2026-09-09): head LR x2.0 +0.0015 / x0.5 +0.0004 (stays 1.0); bigram sign trick full run wash (1.010632 vs
  1.010664 on the v30 config) -> discard. Running on the v31 config: prime table stack, matrix LR x1.0 (re-check at the new
  embedding LR), scalar LR x2.0.
- 09:10 UTC (2026-09-09), v31-config knob runs (baseline 1.002104): matrix LR x1.0 +0.0008 (x0.7 stays), scalar LR x2.0
  +0.0025, depth 12 +0.0018 (0.659 s/step), prime table +0.0013, head LR x2/x0.5 +0.0015/+0.0004. The schedule is now
  well-tuned on this line; remaining knob runs: EMA_START 0.75, EMA_BLEND 0.5, WARMUP 25.
- 10:45 UTC (2026-09-09): v31-config washes: EMA_START 0.75 (+0.0001), EMA_BLEND 0.5 (+0.0001), WARMUP 25 (-0.00015). The
  schedule is at its optimum on this line; remaining queue: WD 0.05 / 0.2, EMA decay 0.99. Next real lever = kernel work.
- 12:20 UTC (2026-09-09): v31-config discards: WD 0.05 +0.0022, WD 0.2 +0.0023 (0.1 stays), EMA decay 0.99 +0.0007
  (0.9933 stays). Remaining queue: FINAL_LR 0.0 / 0.05, TOTAL_BATCH 65k. `submit/train.py` = v31 (1.002410).
- 13:30 UTC (2026-09-09): FINAL_LR 0.0 +0.0004 (wash), 0.05 +0.0015 (discard); 0.02 stays. Running: TOTAL_BATCH 65k
  (one 8-row micro-step per step), then a v31-config repeat for the run-to-run noise estimate.
- 14:25 UTC (2026-09-09): run-to-run noise on the v31 config: vehicle 1.002104, gate 1.002410, repeat 1.002437 -> spread
  ~0.0003; treat differences under ~0.0005 as noise. Batch 65k +0.0044 (discard). Running: MLP ratio 6 and 3.
- 15:00 UTC (2026-09-09): **F_V31_MLP6 (MLP hidden 3072) = 0.996552 vs 1.002104** (-0.0056, first sub-1.00 local) at 0.614 s/step;
  v32 = v31 + MLP ratio 6 (sha e17aa1e0d787b1a4) in T43_gate; ratio 3 running (for the curve), ratio 8 queued.
- 16:05 UTC (2026-09-09): **T43_gate (clean v32 = v31 + MLP ratio 6) = 0.996429 -> `submit/train.py` = v32** (sha e17aa1e0d787b1a4).
- 16:45 UTC (2026-09-09): MLP ratio curve on v31: 3 1.0091 / 4 1.0021 / 6 0.9966 / 8 0.9965 (plateau from 6; ratio 6 = v32,
  submission). Queued on the v32 config: embedding LR x8 / x4 and matrix LR x0.8 re-checks, WARMDOWN 0.8 (never swept here).
- 18:05 UTC (2026-09-09): LR re-checks at MLP ratio 6 (v32 config, 0.996552): embedding LR x8 +0.0007, matrix LR x0.8
  +0.0004 -> the v31 schedule carries over. Queued: embedding LR x4, WARMDOWN 0.8, MLP ratio 5, TOTAL_BATCH 196k at ratio 6.
- 19:30 UTC (2026-09-09): v33 (review-safe) portal 1.0069 (8th), local T45 1.000418 -> `submit/train.py` = v33. Portal gap 0.0065.
- 20:40 UTC (2026-09-09): review-safe fused head (HEAD=nki2: kernel returns bf16 softcapped logits in both modes, CE in
  torch) on the v33 config: 600 s screen 1.068569 vs explicit head 1.074957 (-0.0064); fwd_bwd 0.515 vs 0.566 s/step
  (v32's fused CE was ~0.503). 196k batch at ratio 6 wash (+0.0010). v34 = v33 + nki2 head in T46_gate.
- 21:15 UTC (2026-09-09): **T46_gate (clean v34 = v33 + logits-returning fused head) = 0.998477 -> `submit/train.py` = v34** (sha cf1ca1b04a501e8d).
- 21:50 UTC (2026-09-09): MLP ratio 5 = 0.998222 (curve 3/4/5/6/8 = 1.0091/1.0021/0.9982/0.9966/0.9965; ratio 6 stays). Box
  queue NKI31 empty. Submission = v34 (0.998477 local, review-safe). Next: kernel efficiency work (no queue items until kernels exist).
- 22:10 UTC (2026-09-09): user idea - ensemble of bigram tables (residual-quantize style, freeze one by one) to beat the
  capacity limit without the 2^19 OOM (frozen tables need no grad buffer / optimizer state). First probe at equal memory:
  BIGRAM_SPLIT=2 = two 2^18 x 256 tables with independent hashes, concatenated (hash-embedding style; a collision in one
  leaves the other half clean). Vehicle knob added (CPU smoke ok), F_V34_BGS2 queued after the depth runs. If it helps,
  the freeze-and-stack version (K bf16 frozen tables + one live) is next.
- 22:35 UTC (2026-09-09): depth 9 on v34 = 0.999868 (+0.0014 vs depth 10); with depth 12 +0.0018 (v31), depth 10 is the
  optimum on this line. Running: F_V34_BGS2 (two half-width bigram tables, independent hashes); depth 8 rerun after.
- 23:10 UTC (2026-09-09): BIGRAM_SPLIT=2 (two half-width tables, independent hashes) = 0.998600 vs 0.998477 -> wash. Hash
  collisions are not what limits the bigram table; a frozen-table ensemble would have to win on raw capacity alone (2^17 ->
  2^18 was -0.0023 on v24; 2^19 OOMs; a second live table needs ~+0.5 GB with frozen-first). Deprioritised behind kernel work.
- 23:45 UTC (2026-09-09): depth 8 rerun 0.999917 (+0.0014): depth 8/9/12 all lose to depth 10 on the v34 line. Last queued:
  S_V34_BG19_DB4 (2^19-row bigram table with 4-row micro-steps, 600 s) - the capacity question.
- 00:45 UTC (2026-09-10): 2^19-row bigram table (fits now) = 0.999346 vs 0.998477 at 2^18 -> discard: the table's own
  step cost (scatter + AdamW + grad sync, +5% at 8 rows) exceeds its capacity gain at 2560 steps. Together with the
  split-table wash, the bigram table is closed as a lever unless its per-step cost drops (bf16 grad buffer, fused scatter).
- 01:55 UTC (2026-09-10): **T49_gate (v35 = v33 + fused loss presented for review) = 0.996308 -> `submit/train.py` = v35** (sha 73049474159aa77d); on-device fused-loss check |diff| 9.5e-7.
- 02:50 UTC (2026-09-10), launch-overhead probe on the v35 line: S_V35_REF (PROFILE=1) fwd_bwd 0.501 / grad_sync 0.058 /
  optimizer 0.053 / total 0.613 s/step; S_V35_PACK (layer ops emit packed saved-activation outputs, i.e. far fewer op
  outputs per layer) 0.516 / 0.055 / 0.054 / 0.625 -> ~2% SLOWER (the pack/unpack copies cost more than the outputs
  saved). The ~60 ms/step of launch gap is therefore not per-output marshalling; discard `LAYER_SAVE=pack`, keep `full`.
  S_V35_REF died with the async sidecar OOB right after its profile line (10th occurrence, again a dev-mode run).
  Mitigation queued as the clean file **v36 = v35 + plan-buffer ring** (the training loop keeps the last two micro-steps'
  scatter index buffers referenced so the async indirect DMA can never read a freed buffer; no numeric change; CPU smoke
  loss == CE(logits)); gate T50_gate at full length. Also queued: JOB ATTNBWD, the device test of our own attention
  backward kernel (`nki_attn_bwd.py`, simulator rel err 0.35-0.41%) against the vendor `attention_bwd` (~130 ms/step, 21%).
- 03:00 UTC (2026-09-10): **own attention backward = discard (parked).** Device test ATTNBWD: vendor `attention_bwd` 2.69 ms
  vs ours 3.43 ms for one layer x 8-row micro-step (x20 per step = +15 ms/step even if it were correct), and the device
  outputs disagree with the vendor (dq max |diff| 81 vs vendor max 11.5; dsink of the wrong sign/magnitude) although the
  simulator matched a torch reference to 0.4%. Note the vendor backward is 54 ms/step on this measurement (2.69 x 20),
  not the ~130 ms the NEFF profile attributed to it (that number includes the in-graph recompute and scheduling gaps),
  so the ceiling for an own backward is small. Kernel stays in `nki_attn_bwd.py` behind `ATTN_BWD=ours` (default vendor).
- 03:25 UTC (2026-09-10): **T50_gate (v36 = v35 + plan-buffer ring) = 0.996520 -> `submit/train.py` = v36** (sha
  0474724dc02cf4d5). Same numerics as v35 (0.996308; the 0.0002 is run noise), clean finish. Promoted for robustness: the
  ring keeps the last two micro-steps' scatter index buffers alive, closing the async-indirect-DMA hypothesis for the
  sidecar OOB flake (never seen on a production-shaped run, 10 dev-mode deaths). FP8BENCH (fp8 double-row matmul vs bf16)
  running next.
- 03:35 UTC (2026-09-10), **fp8 double-row = closed as a lever (numbers, not numerics).** FP8BENCH (plain matmul kernels):
  x1.00-1.04 because those kernels are HBM-traffic bound (the up-proj writes 100 MB at T=16384; profile: DMA 71% active,
  GpSimd 97% issuing 53k software DMA descriptors, PE 84%/60%); static unrolling of the tile loop changes nothing (the
  compiler still issues software DMAs). MLPBENCH (fused MLP, hidden in SBUF, PE-bound): bf16 54-59 TFLOP/s per core, fp8
  double-row x1.30-1.41 (82 bf16-equivalent TFLOP/s; transposes/evacuations do not speed up). But the step is not
  matmul-bound: layer matrix params are 42M (the 197M is mostly the 2^18 bigram table), so all dense matmul work per
  worker-step is ~4.5 TFLOP = 80-100 ms of the 560 ms step, and fp8 could touch only the forward MLP (~19 ms/step) ->
  ~6 ms/step. Not worth the numerics risk. The remaining step time is attention kernels, elementwise/DMA inside the
  graphs, launch gaps (~60 ms), grad reduce-scatter (~55), optimizer + gather (~50) and the embedding scatter (~22).
- 03:55 UTC (2026-09-10), engine profile of the v36 FORWARD graph (NEFF aa4b8833, one micro-step of 8 rows, neuron-profile
  2nd exec): total 60.0 ms; PE active 31%, DMA 50%, Vector 47%, Scalar 47%, GpSimd 19%, Sync 29%, mfu 19%. No engine is
  near saturation -> the graph is dependency/latency bound (many short ops with poor overlap), not compute or bandwidth
  bound. That is consistent with fp8 being worthless here and says the remaining in-graph lever is fewer, longer
  instructions (fusing across layers / larger tiles), not faster matmuls. Backward profile (f2682e40) in progress.
- 04:20 UTC (2026-09-10), engine profile of the v36 BACKWARD graph (f2682e40): 144.4 ms per micro-step; PE 34%, Vector
  61%, Scalar 49%, DMA 37%, GpSimd 21%. 239k matmuls (84 ms of instruction time; the 128-wide ones at 15 TFLOP/s), 144k
  tensor_tensor (69 ms), 182k PE transposes (36 ms), 221k stationary loads (25 ms). Full write-up + lever ranking in
  `findings/engine-profile-v36.md`: the backward is still a kernel CHAIN with six torch matmuls and compiler transposes
  per layer; a fused `pre_attn_bwd` kernel (norm bwd + rope bwd + Q/K/V dgrad/wgrad) is the biggest identifiable lever
  (~60 ms/step, 10%), then a fused `post_attn_bwd`. New box tool `trn2_profile_one.sh` (capture + summary + parquet with
  timeouts; the parquet exporter hangs after writing on this build).
- 07:40 UTC (2026-09-10), **fused backward kernels (the lever ranked first by the engine profile).** `nki_pre_bwd.py::
  pre_attn_bwd` = key-offset bwd + QK-norm bwd + rope bwd + Q/K/V dgrad (12 matmuls into one PSUM bank) + wgrad
  (token-partition contraction, SBUF fp32 accumulators) + rmsnorm bwd with the residual, static tile loop, per-head DMA
  reads of dq4/dk4/dv4 (no permutes). Simulator matched the torch chain first try; device: correct at bf16 level,
  **2.78 ms per layer-microstep = 27.8 ms per micro-step vs ~50 ms for the chain.** `nki_post_bwd.py::post_attn_bwd` =
  mlp_bwd (same code) + norm recompute + norm bwd with the residual + out-proj dgrad written per head + dWo; simulator
  matched. Vehicle knobs `PRE_BWD=fused`, `POST_BWD=fused` (fused pre path requires KEY_OFFSET). Queue NKI36: S_V35_PREB
  (pre fused) -> S_V35_REF2 (fresh reference) -> JOB POSTBWD (device test) -> S_V35_PBOTH (both fused).
- 08:15 UTC (2026-09-10): **fused pre-attention backward WINS in-graph: S_V35_PREB2 fwd_bwd 0.429 vs 0.509 s/step (REF2),
  total 0.538 vs 0.620 (-13%)**, losses identical to 3 decimals. The first in-graph attempt (S_V35_PREB) died on all
  cores with "scatter/gather (indirect memory copy via vector DGE) out-of-bound access" although the same kernel passed
  the eager device test: the kernel's only unaligned DMAs (a +1-row shifted read of dk4[:, hq:2hq] and a 127-partition
  partial write for the last tile) are lowered differently inside the model NEFF. Rule: **inside custom ops, keep every
  DMA a 128-row-aligned block; build shifted/masked slices in torch.** The fused post half (S_V35_POSTB) was a wash
  (fwd_bwd 0.515): mlp_bwd already carried it. v37 = v36 + pre_attn_bwd (sha aac729e027fa1c09, `$S/make_v37.py`) in
  T51_gate; F_V35_PREB_D12 (depth 12 at the new speed) and S_V35_DB16 (one 16-row micro-step per step) queued.
- 09:15 UTC (2026-09-10): **T51_gate (v37 = v36 + fused pre-attention backward) = 0.992056 -> `submit/train.py` = v37**
  (sha aac729e027fa1c09). -0.0045 vs v36 (0.996520) from the 13% faster step alone (~3650 steps), 0 errors. Next:
  S_V35_PREB_DB16 (fused + 16-row micro-steps, screen), T52_gate (v38 = v37 + DEVICE_BATCH_ROWS 16 + eval batch cap),
  F_V35_PREB_DB16_D12 (depth 12 at the new speed), S_V35_SKEW.
- 10:12 UTC (2026-09-10): **T52_gate (v38 = v37 + 16-row micro-steps) = 0.989999 -> `submit/train.py` = v38** (sha
  d19cff0aec87b2bf), -0.0021 vs v37, 0.46 s/step, 0 errors. Running: F_V35_PREB_DB16_D12 (depth 12 at this speed), then
  S_V35_SKEW / S_V35_OPTPROF (where the remaining 90 ms/step outside the graphs goes).
- 10:55 UTC (2026-09-10): **depth 12 pays now: F_V35_PREB_DB16_D12 = 0.986795 vs v38 0.989999 (-0.0032)** at 0.52-0.54
  s/step (3345 steps). Earlier depth-12 runs were at parity because the step was 20% slower at the old speed; the two
  kernel-line speedups (fused pre backward, 16-row micro-steps) turned the re-spend positive. v39 = v38 + DEPTH 12 + eval
  batch cap 4 (the in-process eval at 12 layers x 16 rows retried 256 MB allocations at cap 8) -> T53_gate.
- 12:12 UTC (2026-09-10): **T53_gate (v39 = v38 + depth 12) = 0.986544 -> `submit/train.py` = v39** (sha 423feb03b6782053).
  Day's ladder: v36 0.9965 -> v37 0.9921 (fused pre-attention backward) -> v38 0.9900 (16-row micro-steps) -> v39 0.9865
  (depth 12 re-spend). Probes: rank skew is per-collective latency (FLAT_COMM discard: flatten traffic > latency),
  optimizer 48 ms has no dominant piece. Next: own attention kernels (vendor fwd+bwd ~10 ms/layer-step at 16 rows vs a
  ~1 ms PE bound; ATTNBWD_SMALL localizes the device mismatch), depth 14 full run.
- 12:12-18:40 UTC (2026-09-10), **own attention kernels (the vendor attention is ~24% of the step at 10x its PE bound).**
  Backward `nki_attn_bwd.py`: correct vs a torch reference on device (the earlier "mismatch" was against the vendor's
  lse/sink conventions, ATTNBWD_SMALL); v2 range_select mask + fused dS 3.02 ms; profile PROF_ABWD showed GpSimd 74% =
  DMA descriptor generation for ~100 small per-tile DMAs per problem -> v3 with one strided DMA per tensor: **1.80 ms vs
  vendor 2.69**. Forward `nki_attn_fwd.py`: v1 online softmax 1.70 ms; v2 fixed softmax reference (QK-normed, 0.1-scaled
  logits are bounded by ~18; S^T by narrow matmuls, no transposes) 1.41; v3 grouped 4 query blocks (512-wide ops) 1.31;
  v4 single strided DMAs **0.67 ms vs vendor 1.14**. Both correct on realistic inputs (ATTNREAL) and inside a compiled
  graph (ATTNGRAPH). In the MODEL (S_V39_AFB2/AFB4, T55_gate) random ranks produce NaN losses at random early steps;
  the one screen that synchronized after every layer (AFB3) never did. Bisection: each kernel deterministic alone; fwd+bwd
  in one graph nondeterministic -> the graph compiler mis-tracks strided-pattern OUTPUT DMAs (per-tile stores fixed the pair:
  ATTNSENT3 0/30 differ; fwd 0.82 / bwd 1.95 ms, still faster than the vendor) but the model still NaNs (AFB5 step 19, T55
  step 3); strided loads of compiler-produced tensors are deterministic (ATTNPROD). Current hypothesis: a race between the
  eager sidecar's async indirect DMAs and the graph (also the mechanism behind the intermittent OOB flake, 2 deaths today
  incl. the production-shaped depth-14 run) -> S_V39_AFB6 with SIDECAR_SYNC=1 running. Plain per-tile-load variants:
  fwd 1.24 / bwd 2.37 ms (fallback: bwd-only gain ~8 ms/step). Rules recorded in memory nki-tracer-rules #8/#9.
  Also closed: host-side doc bounds (+4%), scan-based doc bounds (cummax has no Neuron lowering), depth 14 (crashed to
  the flake at step 330, 0.60 s/step; not re-run), FLAT_COMM (-), per-rank skew (ranks equal; per-collective latency).
- 19:20 UTC (2026-09-10): **own attention parked.** Bisection in the model: strided single-DMA forward + plain backward
  -> NaN (AFB_A); plain forward + strided backward -> clean (AFB_B) but slower than the vendor path in-graph (total 0.565 vs
  0.546 with the profiler): the per-layer torch glue that builds the kernels' per-row inputs (sink/bounds expands, arange,
  -lse/ones) costs more than the kernels save, and the forward that would have paid (0.82 vs 1.14 ms eager) is the one
  that corrupts in the model context (deterministic in every isolated test). Vendor attention stays. Re-queued depth 14
  (F_V35_PREB_DB16_D14b) after its flake death this morning.
- 19:30-21:10 UTC (2026-09-10), evening screens on v39 (all 600 s, discards): depth 14 rerun died to the OOB flake again
  (2 of 2; depth 12 passed 3 gates -> depth 14 dropped); GATHER_BF16 (bf16 param all_gather) -2 ms/step, scores within
  noise; XSA (gated exclusive self-attention as torch glue in the layer op) doubles the step - the custom-op bodies execute
  eagerly at runtime, so extra elementwise ops are launch-bound (kernel-internal only; expected gain ~0.001, dropped);
  ADAM_ALT (AdamW on odd steps) 1.18 vs 1.055: the AdamW groups are the x6-LR embeddings, halving their updates hurts.
  Never-screened tier-1 items now closed: XSA, alternate Adam; drop-clipping is already the state (no clipping); retie
  is awkward with the sidecar tables; prefix-token aux loss (needs two target logits in the fused head kernel, record
  gain ~1% of steps) and cautious WD (in-kernel) remain untried and small.
- 21:50 Sep 10 - 06:30 Sep 11 UTC, **DeltaNet hybrid, stage 1 (torch-op chunked gated delta rule) = NOT VIABLE on this
  stack.** `gdn_torch.py` (WY form, exact nilpotent triangular inverse) == token recurrence to 1e-15; vehicle
  `GatedDeltaMixer` (DELTANET=1,3,5,7,9,11, reusing the layer's stacked q/k/v/o); CPU model smoke fine. On the device:
  16 and 8 rows -> device OOM at graph load (the unrolled chunk loop's saved state; the graph's own scratchpad); 4 rows
  compiled but loss NaN at step 0 on every rank; `--auto-cast=none` no change; chunk 16 OOM; the chunked math alone is
  exact on the device (GDNDEV, 3e-6); every mixer stage compiles and is finite ALONE but the whole mixer compiled as one
  graph hits a neuronx-cc internal error (NCC_IMGN901 MacroGeneration 'Trying to vectorize') - in the model it silently
  miscompiles to NaN; eager mode (COMPILE=0, 4 rows) is finite for 4 steps then NaN, at 19 s/step (no signal possible).
  Conclusion: a DeltaNet layer on Trainium needs a full NKI chunked forward+backward kernel with careful numerics (days),
  and even then its step cost vs the fused attention layer is unknown; the GPU-line gain (-0.025 at fixed steps) has NOT
  been reproduced here. Lessons: fused compiler graphs of small-matrix chains are unreliable (internal errors / NaN);
  keep such math in NKI kernels or avoid it.
- 06:50 UTC (2026-09-11), housekeeping: all `nki_*.py` kernel modules and tests moved from the project root into `nki/`
  (no `__init__.py`, so the installed `nki` package is not shadowed); the box mirror `~/tf/nki/` moved the same way.
  Queue JOB lines now use `python nki/<test>.py`. Generators that extract kernel bodies read the vehicle, not these files;
  `make_v41.py`-style extractions from `nki_attn_*.py` would need the new path.
- 07:00 UTC (2026-09-11), cleanup: `freeze_submission.sh` deleted (versions come from the make_v## generators + gates);
  `experimental/gdn/` (gdn_torch.py, gdn_dev_test.py, gdn_compile_test.py), `experimental/neff/neff_ops_agg.py` (legacy,
  pandas), `experimental/neuron/` (neuron_probes.py, dist_probe.py, run_probes.sh, neuron_profile_step.py: fresh-box
  probes and the old step profiler), `experimental/monitor/nm_monitor.py`. Still in use at the root: neff_instr_agg.py
  (called by trn2_profile_one.sh), gpu_gdn.sh (local-GPU fixed-step runs), the trn2_* box tools (untouched).
  Box copies of the moved files stay in ~/tf (the legacy trn2_profile_ops.sh still expects neff_ops_agg.py there).
