# Trainium Frontier playbook

Operational source of truth for getting a submission onto the portal without losing a
slot. Detailed experiment history lives in `train2-experiments.md`; rule interpretations
in `competition-rulings.md`. This file: what to submit, how to validate it, and a dated
log of every submission-affecting event.

## Current submission state (as of 2026-09-04)

- **File to upload:** `train3_submission.py` (v4, 2026-09-03) — all 6 gates green.
- **Launch command (exact):** `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`
  - The command targets `train.py` because the portal saves the upload under that name.
  - No extra env vars: compile is default-on inside the file; `NEURON_COMPETITION_R1_DEVICE_BATCH` is dev-only.
- **Architecture inside:** the T10 fixed-time champion (depth 10 × width 256, 2 heads,
  all-attention, doc-mask + 4 registers/layer, triple ve tables, bigram 2^17, AsymLogit,
  rope 200k, batch 32768 with ×0.71 LRs, PACK=4 reshape, Muon/Polar-Express + AdamW).
- **Not yet folded in (needs Trn2 timing first):** narrow QK (L2, −0.0075 on GPU, throughput-driven
  — GPU speedups do not automatically transfer to the chip). Trigram table does NOT stack with it (L4r3).
- Phase 1 closes **September 30, 2026** (extended from Sept 15 by upstream README update).
  5 submission slots/week; best score counts.

## The submission contract (upstream c907da4, 2026-09-03)

1. The runner watches an append-only step log (`NEURON_COMPETITION_R1_STEP_LOG`) and takes
   its own clock reading as each line appears. Wall time from launch through the **5th**
   reported boundary is the excluded startup/compilation allowance, **capped at 900 s**.
2. Submissions MUST call `prepare.record_step_boundary(step)` exactly once per completed
   optimizer step, after device sync (rank 0 writes; other ranks no-op).
   `prepare.py check` REJECTS files that never call it; **< 5 reported boundaries = unscorable**.
3. Consequences implemented in v4: budget clock restarts at boundary 5 (back-dated by any
   over-cap excess + 30 s launch margin); time-based stop gated behind ≥ 5 completed steps and
   carried on a rank-synchronized control collective; `--compile-only` flag supported
   (one synthetic optimizer step, then exit — an unknown-flag argparse exit would be instant death
   if the runner passes it); `MAX_TRAIN_SECONDS` = full budget (the old −60 s safety margin and the
   self-managed pre-clock warmup are superseded).

## Pre-submission gate battery (run ALL of these on the exact file)

| # | Gate | Command sketch | Catches |
|---|------|----------------|---------|
| 1 | Byte-compile | `python -m py_compile <file>` | syntax legal-to-AST but uncompilable (prepare.py check only ASTs — it passed a broken file once) |
| 2 | Official check, portal-shaped | `cp <file> /tmp/train.py && python prepare.py check --train-py /tmp/train.py --launch-command "<exact command>"` | forbidden imports/calls, missing record_step_boundary, launch-command shape (validated against the uploaded filename) |
| 3 | CPU functional run | `--device-type cpu --num-steps 7 --no-compile --no-eval-public` with `NEURON_COMPETITION_R1_STEP_LOG` set | e2e training, checkpoint, one log line per step, boundary-5 clock reset |
| 4 | compile-only | `--no-compile --compile-only` | the runner's warm-cache invocation path |
| 5 | 2-rank torchrun | `torchrun --standalone --nproc_per_node=2 <file> --device-type cpu --num-steps 3 ...` | dist init, rank-0-only log writes, lockstep stop via control collective |
| 6 | **Timed confidence run** | GPU, exact file, short `--max-train-seconds`, step log on | throughput traps (TF32! compile!), predictive stop margin, causality, sane val_bpb — correctness gates do NOT measure speed; both early portal failures were speed-class |

Hard-earned rules: always `torch.set_float32_matmul_precision("high")` (one missing line = 20×
slowdown on GPU dev boxes, harmless on Neuron); never benchmark on a GPU shared with other jobs
(gate on a fully-empty GPU); launch long runs from a direct shell with setsid, never from inside a
monitor/watcher task (its teardown reaps the process tree).

## Submission log

### 2026-09-04 — v4 validated, ready for slot #4 (report logged verbatim from 2026-09-03 session)

The `git pull` was **not minor for us** — upstream `c907da4` changed the submission
contract, and `train3_submission.py` was ported to v4 with every gate green.

**What changed upstream:**
- The compile-time-exclusion ruling is now mechanized: the official runner watches a step
  log and excludes launch through the **5th** `record_step_boundary()` call, capped at
  900 s. `prepare.check_submission` now **rejects** any file that never calls it, and a
  run reporting fewer than 5 boundaries is unscorable — our v3 would have failed on both counts.
- Phase 1 now closes **September 30** (was Sept 15) — ~4 more submission weeks than assumed.
- New official DLAMI/venv pinned in the README (ami-0d5703e99c1589f8d,
  venv neurips-torchneuronx-native-beta-2.11.3-20260824); prizes reworded.

**The v4 port** mirrors the new baseline exactly: `record_step_boundary` after every
synchronized step, budget clock restart at boundary 5 with over-cap back-dating, the
time-stop gated behind 5 completed steps and riding a rank-synchronized control
collective, `--compile-only` support, and full-budget `MAX_TRAIN_SECONDS`. The custom
pre-clock warmup was deleted — the allowance replaces it.

**Gates, all passed:** py_compile · new prepare-check (as `train.py` with the real launch
command) · CPU 7-step run (step log exact, clock reset verified: 77.9 s excluded) ·
`--compile-only` · 2-rank torchrun (rank-0-only log writes, lockstep stop) · timed GPU
run — **1,910 steps / 235 s charged, 1,910 log lines, predictive stop landed 5 s under
budget, causality clean, val_bpb 1.1937 at the 240 s mini-budget** (V4T in
`train2_results.tsv`), consistent with the champion trajectory.

Diagnostic note: the step-log mechanism proves the runner reads our runtime artifacts —
if the portal still fails, the `SUBMISSION_BOOT` banners plus the boundary count will
finally localize where the process died.

### 2026-09-02 — three failed slots, post-mortems (summary)

- **v1** (`train2_best.py`, sub-f02c5ed1): `no_checkpoint`, died 0.032 s. Dev-only file
  (no Neuron/dist support). First "flex import" diagnosis later disproven; startup-death
  root cause never confirmed (portal exposes no stderr).
- **v2** (`train2_best_submission.py`, sub-286b3ddd): `no_run_meta`, no log at all.
  Cause unknown. Local confidence run separately caught the TF32 trap (20×) and
  eager-by-default — speed-class bugs invisible to correctness gates.
- **v3** response: lazy CUDA-only flex import, `SUBMISSION_BOOT` banners, py_compile gate,
  timed-confidence-run rule. Baseline `train.py` scored 1.4226 (sub-a18a1065) before
  being pulled; open user action: ask organizers on Devpost to expose stderr for failed runs.

## 2026-09-05 — first real-Trn2 validation: v5 = 1.174335 (portal-exact launch)
`train3_submission.py` **v5** ran on test4 (trn2.3xlarge, beta AMI …20260731) with the exact portal launch and no env:
**public_val_bpb 1.174335**, 2238 steps, 73.3M tok, 1794.9 s charged, causality OK. Baseline train.py on the same
box: 1.258923. v5 = v4 + three Trn2-necessary fixes, all math-neutral: `x0 = x.clone()`, tensor-valued per-step
LR/momentum (python-float LR baked into NEFFs → ~4 s recompile per distinct value → warmup + 60% warmdown ran 6×
slow), `idx/targets.clone()` (neuron dynamo alias-table collision), and **COMPILE default flipped to eager** (compile
crashed; the portal sets no env, so v4 as shipped would have died on the portal). v5 passes `prepare.py check`.
**v5 is the current submission candidate.** Campaign log: `findings/trn2-experiments.md`; hardware: `trn2-hardware.md`.

### 2026-09-05 22:31 — v5 + compile = **1.151850** (T2, full length, all gates). COMPILE default now ON.
`train3_submission.py` v5 with `torch.compile(backend="neuron")` on: 3235 steps / 106M tok in the budget, causality OK.
Baseline 1.2589 → eager v5 1.1743 → compiled v5 **1.1519**. This is the submission candidate. Cold-cache compile
costs ~210 s of startup (excluded by the runner, cap 900 s).

### 2026-09-05 23:52 — **TARGET MET: v5 + compile + LNC=2 launch = 1.139926 (T3).**
Submission = `train3_submission.py` v5 (COMPILE default ON, DEVICE_BATCH derived from world size) with the
**launch-command string `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`** (README-
sanctioned; `prepare.py check` passes). Why LNC=2: util.log showed CPU ≈95% with 8 dispatching workers — the host,
not the chip, was the bottleneck; 4 workers on 2-core logical NCs run 0.47 s/step vs 0.53 (NC util 17% vs 6%).
Ladder on hardware: baseline 1.2589 → eager v5 1.1743 → +compile 1.1519 → +LNC=2 **1.1399**. Final gate T4 (exact
submission shape, no env) running; do not submit until it reports.

### 2026-09-06 00:27 — **FINAL GATE PASSED (T4 1.140051). Submission frozen: `submission_v5_lnc2.py`** (sha256 912e282cc04dd38b…)
Upload it as `train.py` with launch string **`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`**.
Gate = exact shape (no env), 3776 steps, causality OK, cold startup 273 s (excluded). Passes `prepare.py check` as
`train.py`. Hardware ladder: 1.2589 → 1.1743 → 1.1519 → **1.1400**. Further experiments continue on
`train3_submission.py`; re-freeze only after a new full-length gate.
- 2026-09-05 ~19:00 PDT: **submitted** `submit/train.py` (= `submission_v5_lnc2.py`, sha256 912e282cc04dd38b…) with
  `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`. Pending. Leaderboard at submit
  time: #8 1.1376, #9 1.1557 → 1.140 expected to land #9.

### 2026-09-06 — sub-9ea276d1 (v5) FAILED on the portal: `USER_TRAINING_CRASHED exit=1 status=no_run_meta (no train.log)`
Same signature as v2 (9/2). Not reproducible here: the exact bytes + exact launch run to completion, pass
`--compile-only`, and pass the harness's own launch validator. **The one factor shared by both portal failures and
absent from the organizers' baseline: `torch.compile(backend="neuron")` on by default.** Our build needed two
backend-bug workarounds for it; the portal runs the newer …0824 build. "(no train.log)" most likely just means the
harness collected nothing after a crash.
**v6 (`submission_v6_eager.py`, sha256 4691c9d5f266fb0d…):** compile default OFF; compile failures at
`torch.compile()` or at the first step fall back to eager; runner-contract import guarded. `submit/train.py` = v6.
Resubmission options (user's call, one slot left this week by our count):
- **Safest:** v6 + `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py` — the only launch
  shape that has ever scored for us; validated here eager 8-way = 1.1743 (T1, v5 bytes).
- **Better if LNC=2 is fine on their box:** v6 + `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4
  train.py` — T6_gate (running) gives the eager LNC=2 number.
- Still open: ask the organizers for the failed run's stderr (v2's cause was never learned either).

### 2026-09-06 03:08 UTC - v6 validated for resubmission. `submit/train.py` = `submission_v6_eager.py`, ASCII-only,
sha256 ed7c7f687afcf737. Eager default; compile falls back to eager; runner-contract import guarded; passes
`prepare.py check` under both launch shapes. Gates: T6 = eager+LNC=2 full run 1.155026; S8 = eager 8-way smoke OK.
Recommended launch: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py` (1.155, top-10);
fallback shape: `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py` (~1.174).
- 2026-09-06 03:34 UTC: sub-39b23de5 (v6, eager, LNC=2) also FAILED `no_run_meta` -> compile and encoding ruled out.
  **v7** = v6 + packing off (PACK=1, baseline loader shape) + per-worker rows capped at 8. `submit/train.py` = v7
  (sha 71404a2b30fe0426). Submit with the 8-way launch **`NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone
  --nproc_per_node=8 train.py`** (the only shape that has scored for this team). Expected ~1.175. Ask organizers for
  stderr of sub-39b23de5.
- 2026-09-06 04:05 UTC: **T7 gate = 1.176902** for the v7 resubmission candidate (exact bytes, 8-way launch, no env,
  2214 steps, causality OK). Board placement ~11th. Submit `submit/train.py` (sha 71404a2b30fe0426) with
  `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`.

### 2026-09-05 20:43 PDT - **sub-ae5cf1a6 SCORED 1.1827** (v7, eager, PACK=1, 8-way). First score since v1's 1.5437.
Calibration: local T7 1.1769 vs portal 1.1827 -> **portal = local + ~0.006** (held-out shard). Rank ~11th.
**Bisection now possible.** v7 OK (compile 0, LNC=1, PACK=1); v6 FAIL (0, LNC=2, PACK=4); v5 FAIL (compile 1, LNC=2,
PACK=4). Next slot = v7 bytes + `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`:
- scores -> PACK=4 was the killer; then compile+LNC=2 (local 1.140 -> ~1.146 portal, #9) is the follow-up.
- fails  -> LNC=2 is the killer; then compile on the 8-way launch (local T2 1.152 -> ~1.158, ~#10).
- 2026-09-06: repo reorganized. `submit/` = upload set (train.py + the latest submission_v#); `submission_raw/` = archive of
  all submission_v# files and the train2*/train3 vehicles. Box launcher runs `submission_raw/train3_submission.py` with
  PYTHONPATH=~/tf. Local CPU tests: `PYTHONPATH=. ./.venv/bin/torchrun ... submission_raw/train3_submission.py --device-type cpu`.

- 2026-09-06 06:17 UTC: **T9 = 1.128411** (stacked-parameter vehicle `submission_raw/train3_submission.py`, compile,
  LNC=2 launch). Local best. Not yet a portal candidate: needs compile+LNC=2 cleared there (v7+LNC=2 bisection
  submitted ~9/5 late evening; result unknown to the assistant). Eager 8-way version pending PR5.
- 2026-09-06 10:05 UTC: **v9 eager 8-way** (`submission_raw/submission_v9_eager_8way.py`, sha 8b8e5210fa136c53; stacked
  parameters + function-mode backward) validated by T10 = **1.160509** on the portal-proven launch (v7: 1.1769).
  `submit/train.py` = v9 now. Launch: `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`.
  Passes `prepare.py check` as train.py. Compile+LNC=2 with the same vehicle = 1.1284 (T9), portal-conditional.

### 2026-09-06 10:40 UTC - width is the lever on Trn2: T11 384-wide (compile+LNC=2) = **1.106531**, local best
- `submit/train.py` is still v9 (256-wide eager 8-way, T10 1.1605 local): that stays the portal-safe upload until the
  384-wide eager 8-way full run (T14) validates; then freeze `submission_v10_eager_8way.py` (N_EMBD 384 / N_HEAD 3 baked
  in as defaults - the portal launch string carries no extra env) and copy it to `submit/train.py`.
- compile+LNC=2 candidates (T9 1.1284, T11 1.1065) remain portal-conditional on the v7+LNC=2 bisection result.

### 2026-09-06 12:07 UTC - width optimum found: **512-wide (T12) = 1.088742** at compile+LNC=2; 640 is worse (1.0945) and its
cold compile sits at the 900 s startup cap. Portal-shape validation of the 512-wide file (v10, eager 8-way, sha 6dd554d720ae421e)
running as T14; if it beats T10's 1.1605 it becomes `submit/train.py`. Launch string unchanged:
`NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`.

### 2026-09-06 12:40 UTC - **`submit/train.py` = v10 (`submission_v10_eager_8way.py`, sha256 6dd554d720ae421e), T14 = 1.120626**
on the exact portal shape. Upload `submit/train.py` with `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`.
Expected portal score ~1.127. Only change vs v9: N_EMBD 512 / N_HEAD 4 defaults. Checkpoint is 478 MB (was ~200 MB).
- 12:44 UTC: v10 verified fresh-process on the local 4080 via `prepare.py eval-public` = 1.121571 (in-run 1.120626). Ready to upload.

### 2026-09-06 14:35 UTC - architecture at 512 is settled; compiler opt level is the open accelerator setting
- Full-run ledger at compile+LNC=2: 512-wide base **1.088742** (T12, local best); MLP5 1.088372 (wash); 640-wide 1.0945;
  384-wide 1.1065; screens: depth 12 and 8x64 heads both clearly worse. 65k batch does not fit an LNC=2 core at 512.
- The native stack compiles at -O1 (`NEURON_COMPILER_OPT_LEVEL`); T17 tests -O2. If it wins, the vehicle must set the env
  var itself (the portal launch string is fixed) and the longer compile counts toward the 900 s startup cap.
- `submit/train.py` remains v10 (512-wide eager 8-way, T14 1.120626); unchanged by anything above.

### 2026-09-06 18:12 UTC - local best 1.084527 (T19: 512-wide + embedding LR x2 + warmdown 80%, compile+LNC=2)
v11 candidate (`submission_v11_eager_8way.py`) = those defaults on the portal shape; T20 validates it eager 8-way. Until
T20 reports, `submit/train.py` stays v10 (T14 1.120626).

### 2026-09-06 18:45 UTC - **`submit/train.py` = v11 (`submission_v11_eager_8way.py`, sha256 c6cd05b896ff290b), T20 = 1.112172**
on the exact portal shape. Upload with `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`.
Changes vs v10: embedding LR x2, warmdown over the last 80% of the budget, plus the LR/schedule env knobs (defaults = these)
and the predictive-stop window reset (no effect on a full run). Expected portal score ~1.118.
- 18:47 UTC: v11 verified fresh-process on the local 4080 via `prepare.py eval-public` = 1.113277. Ready to upload.

### 2026-09-07 00:15 UTC - `submit/train.py` = v12 (`submission_v12_eager_8way.py`, sha256 58eccae335e3a435) - GATE PENDING (T24)
Defaults baked in: 512/4h, embed LR x2, warmdown 0.8, SINK=logit + ATTN=nki (vendor flash attention with doc bounds and a
learned sink, eager only), MUON=fused + ADAMW=nki (in-place NKI optimizer kernels). Launch unchanged:
`NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`. Evidence: T22 (attention only) 1.103320 full
run; T23 (all kernels, env knobs on the vehicle) running at 0.58 s/step; the exact-bytes gate T24 is queued right after.
Last fully gated file: v11 (`submission_raw/submission_v11_eager_8way.py`, T20 1.112172). Kernel imports are guarded: if the
portal's nkilib/NKI differs, attention falls back to the explicit path and the optimizer kernels are simply not used... NOTE:
every optimizer kernel call is wrapped: a trace/runtime failure disables that kernel and continues on the torch path.

### 2026-09-07 00:40 UTC - USER DECISION: submit compile ON with the 8-way launch. `submit/train.py` = **v13**
(`submission_raw/submission_v13_compile_8way.py`, sha256 d1e208cc43871694): 512/4h, embed LR x2, warmdown 0.8, COMPILE default
ON (falls back to eager on failure), explicit attention + 4 registers (NKI attention stays off under compile: it partitions the
NEFF), fused Muon + AdamW kernels (torch fallback). Launch: `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`.
Gate T25_gate (exact bytes, no env, full budget) queued; cold compile of the 4-rows-per-worker graph expected ~5-9 min (excluded).
v12 (eager, NKI attention, sha 58eccae335e3a435) stays in submission_raw/ as the eager-line candidate (gate T24).

### 2026-09-07 00:50 UTC - USER DECISION (supersedes 00:40): eager + LNC=2. `submit/train.py` = **v12** (sha256 58eccae335e3a435)
Launch string: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py` (check passes). Eager, NKI attention
+ sink logit, fused Muon/AdamW kernels, 512/4h, embed LR x2, warmdown 0.8; each worker derives 8 rows from the 4-way world size.
Gate T26_gate (exact bytes, that launch, no env) queued first; T25 (v13 compile 8-way) kept after it for information.
Portal risk unchanged: both earlier LNC=2 submissions failed `no_run_meta` (cause unknown); v13 (compile, 8-way) remains the
alternative in submission_raw/.

### 2026-09-07 01:00 UTC - PORTAL REJECTED_BY_REVIEW (sub-017d7c26, submitted 9/6 17:18 PDT = v12 bytes)
Message (truncated in the portal UI): "multiple critical violations: (1) unsafe checkpoint loading via torch.load(...,
weights_only=True) with a fallback mechanism that could deserialize arbitrary objects, (2) eva..." - the file has no such
fallback (single weights_only=True call, same as the baseline), so the review is at least partly a misread; the rest is unknown.
Response: **v14 = a clean minimal submission** (`submission_raw/submission_v14_clean.py`, 1178 lines): the chosen config
hard-coded (no dev knobs beyond the baseline's NUM_STEPS/MAX_TRAIN_SECONDS/OUT_DIR/EVAL_*), no torch.compile path, no unused
architecture branches, NKI imports in one guarded block at the top (no runtime `global` rebinding), kernels defined plainly at
module level, custom ops `r1::doc_attention_fwd/bwd`, plain `torch.load(..., weights_only=True)` in load_for_eval, torch
fallbacks only where a kernel is optional. Same math as v12 (CPU 5-step trace compared against the vehicle). Launch:
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`. Gate T27 (exact bytes) queued on the box.
Round-1 criteria noted from the user: correctness gate (finish in budget, valid score, unmodified harness, **use the full chip**,
provided data only), score = val_bpb; tie-breakers fewer parameters, then simpler code.
- 01:35 UTC: **T27_gate = 1.069640** (v14 exact bytes, eager, LNC=2 launch, no env, 4473 steps, causality ok). Submission
  validated. Fresh-process eval-public on the T27 checkpoint follows (local 4080, explicit path).
- 01:50 UTC: fresh-process `prepare.py eval-public --train-py submit/train.py --checkpoint <T27 final.pt> --device cuda` on the
  local 4080 (explicit attention path, no NKI) = **1.069914** over 2,097,152 tokens (in-run Neuron 1.069640). v14 is validated
  end to end: bytes, launch string, load_for_eval, causality, fresh-process scoring.
- 02:10 UTC: insurance file **v15** `submission_raw/submission_v15_clean_compile_8way.py` = the clean v14 style + torch.compile
  (backend=neuron, eager fallback at compile time and at step 0) for the portal-proven 8-way launch
  `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`; under compile the attention takes the explicit
  path (sink logit), the optimizer kernels stay. To be gated as T28 (T25 with the old-style v13 gave 1.0703 on this shape).
- 02:40 UTC: S_E2_ROPE (RoPE kernel on the eager LNC=2 line) -0.0064 at 900 s -> **v16** `submission_raw/submission_v16_clean.py`
  (sha 90fe1fedbcdb52e5) = v14 + the fused RoPE kernel (`rope_kernel`, `_NKIRope`; used only on eager Trainium, plain
  apply_rotary_emb elsewhere). Static check passes with the LNC=2 launch. Gate T29 queued before T28 (v15 insurance).
  If T29 < T27's 1.0696 it becomes `submit/train.py`.

### 2026-09-07 03:20 UTC - **TARGET MET: T29_gate = 1.064665.** `submit/train.py` = **v16** (`submission_v16_clean.py`, sha256 90fe1fedbcdb52e5)
Exact bytes, launch `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`, no env, 4882 steps at
0.36-0.37 s/step, causality ok. Ladder on this line: v11 1.1122 -> +NKI attention 1.1033 (T22) -> +fused Muon/AdamW 1.0699 (T26/T27,
LNC=2 launch) -> +RoPE kernel **1.0647**. Insurance for the 8-way launch: v15 (clean, compile ON) gate T28 running; v13 1.0703.
- 03:30 UTC: fresh-process `prepare.py eval-public --train-py submit/train.py --checkpoint <T29 final.pt> --device cuda` (local
  4080, plain path) = **1.064823** (in-run 1.064665). v16 validated end to end; `submit/` = train.py + submission_v16_clean.py.

### 2026-09-07 03:50 UTC - **T28_gate = 1.052909 -> `submit/train.py` = v15** (`submission_v15_clean_compile_8way.py`, sha256 fc1f33c77b5cf519)
Launch: **`NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`** (the shape that scored on the portal).
Clean single-config file: torch.compile(backend=neuron) with eager fallback, sink-logit explicit attention (kv length 1024 - the
4-register version's 1028 mis-tiled the compiled graph: 0.41 -> 0.31 s/step), fused Muon + AdamW NKI kernels (torch fallback),
5559 steps, causality ok. Ladder: v11 1.1122 -> v16 eager LNC=2 1.0647 -> **v15 compile 8-way 1.0529**. Fresh-process eval follows.
Portal caveats: compile ON was never proven fatal (the eager LNC=2 file failed the same way; the 8-way eager v7 scored); if
compile is rejected, the file runs eager (v12-class ~1.09); v16 (eager LNC=2, 1.0647) stays in submission_raw/ as the alternative.
- 04:00 UTC: fresh-process eval-public on the T28 checkpoint via `submit/train.py` (local 4080, plain path) = **1.053831**
  (in-run 1.052909). v15 validated end to end.
- 11:20 UTC: **v17** `submission_raw/submission_v17_clean_compile_8way.py` (sha 2e8ad23e03a68697) = v15 + the concat-free sink
  (softmax over the 1024 real keys scaled by 1 - sigmoid(sink - logsumexp); -0.0010 at 900 s on the compile line, S_C8_LSE).
  Static check passes with the 8-way launch; gate T31 queued after the last screens. Compile-line ledger: every other knob
  (window, bf16 scores, SDPA, width 640, depth 12, transposed weights, merged QKV, Muon LR x1.2, embed LR x3) is a wash or a loss.
- 11:00 UTC: S_C8_WDN09 -0.0014 -> **v18** `submission_raw/submission_v18_clean_compile_8way.py` = v17 + WARMDOWN_TIME_FRAC 0.90
  (sha b737878e6c23ef70, static check passes); T31_gate now gates v18 (v17 kept as the lse-only variant).

### 2026-09-07 11:50 UTC - **T31_gate = 1.051593 -> `submit/train.py` = v18** (`submission_v18_clean_compile_8way.py`, sha256 b737878e6c23ef70)
Launch unchanged: `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`. v15 + concat-free sink + warmdown 0.9,
5687 steps at 0.30-0.32 s/step, causality ok. Still held (user threshold ~1.03-1.04 local).
- 12:05 UTC: fresh-process eval-public on the T31 checkpoint via `submit/train.py` (local 4080) = **1.052771** (in-run 1.051593).
  v18 validated end to end.
