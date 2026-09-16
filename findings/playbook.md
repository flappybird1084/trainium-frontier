# Trainium Frontier playbook

The submission contract, the 6-gate battery, and the v4-v18 log — getting a submission onto the portal
without losing a slot. Phase 1 closes **September 30, 2026**; 5 slots/week, best score counts. Rules in
`competition-rulings.md`; numbers in `trn2_results.tsv`.

## The submission contract (upstream c907da4, 2026-09-03)

Compile time is excluded by machine: the runner clocks the step log (`NEURON_COMPETITION_R1_STEP_LOG`) —
launch through the **5th** reported boundary is the excluded allowance, **capped at 900 s**. Files must
call `prepare.record_step_boundary(step)` once per completed optimizer step, after device sync (rank 0
writes only): `prepare.py check` REJECTS files that never call it, and **< 5 boundaries = unscorable**.
`--compile-only` (one synthetic step, then exit) must exist — an unknown-flag argparse exit would be
instant death if the runner passes it. Our clock restarts at boundary 5, back-dated by over-cap excess +
30 s; the stop needs >= 5 steps.

## Launch shapes

The portal records the launch string, so the vehicle must bake its own defaults (width/heads, compile on/off);
only dev knobs are env-gated. The command names `train.py` (the portal saves it under that name).

| shape | command |
|---|---|
| LNC=1, 8-way — the only shape that has ever scored for us | `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py` |
| LNC=2, 4-way — host-bound, faster when CPU is the bottleneck (T3) | `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py` |

**Current submission: v43** (`findings/handoff-2026-09-13b.md`).

## Pre-submission gate battery (run ALL on the exact file)

| # | Gate | Command | Catches |
|---|------|---------|---------|
| 1 | Byte-compile | `python -m py_compile <file>` | uncompilable-but-legal-to-AST syntax (prepare.py check only ASTs; it once passed a broken file) |
| 2 | Official check, portal-shaped | `cp <file> /tmp/train.py && prepare.py check --train-py /tmp/train.py --launch-command "<exact>"` | forbidden imports/calls, missing record_step_boundary, launch-command shape |
| 3 | CPU functional run | `--device-type cpu --num-steps 7 --no-compile --no-eval-public`, step log on | e2e training, checkpoint, per-step log line, clock reset |
| 4 | compile-only | `--no-compile --compile-only` | the runner's warm-cache path |
| 5 | 2-rank torchrun | `torchrun --standalone --nproc_per_node=2 <file> --device-type cpu --num-steps 3` | dist init, rank-0-only log writes, lockstep stop |
| 6 | **Timed confidence run** | GPU, exact file, short `--max-train-seconds`, step log on | throughput traps (TF32! compile!), stop margin, causality, sane val_bpb — correctness gates do NOT measure speed (speed-class: both early portal failures) |

Rules: `torch.set_float32_matmul_precision("high")` always (one missing line = 20× slowdown on GPU dev
boxes, harmless on Neuron); never benchmark on a GPU shared with other jobs (gate on a fully-empty GPU);
launch long runs with setsid, never from inside a monitor/watcher task (its teardown reaps the process
tree).

## Submission log — v4..v18 (09-02..09-07)

All gates below = exact bytes, no env, own launch shape, full budget, causality clean.

| ver | change vs previous | gate (local val_bpb) |
|---|---|---|
| v4 | contract port; pre-clock warmup deleted | 6 gates green; 1.1937 at a 240 s mini-budget (V4T); CPU 7-step run verified the clock reset |
| v5 | Trn2 fixes, all math-neutral (details below); COMPILE -> eager (compile crashed; no env on the portal) | 1.174335 (T1); baseline train.py same box 1.258923 |
| v5+compile | COMPILE ON | 1.151850 (T2) |
| v5+LNC2 | LNC=2 launch; DEVICE_BATCH from world size | 1.139926 (T3); T4 1.140051 |
| v6 | compile OFF, eager fallback (compile time and step 0); contract import guarded | 1.155026 (T6); smoke S8 OK |
| v7 | PACK=1 (baseline loader shape), rows/worker capped 8 | 1.176902 (T7) |
| v9 | stacked parameters + function-mode backward | 1.160509 (T10, eager 8-way); 1.128411 (T9, compile+LNC=2) |
| v10 | 512/4 heads baked in; 384 = 1.106531 (T11), **512 = 1.088742 (T12)**, 640 = 1.0945 | 1.120626 (T14, portal shape) |
| v11 | embed LR x2, warmdown 80%, LR/schedule env knobs (defaults = these) | 1.112172 (T20) |
| v12 | NKI attention (logit sink, eager only) + fused in-place Muon/AdamW kernels | 1.103320 attention only (T22); 1.0699 all (T26/T27) |
| v13 | compile ON, 8-way launch; explicit attention + 4 registers; fused Muon/AdamW (torch fallback) | 1.0703 (T25) |
| v14 | the clean minimal file (below), written after the review rejection; math == v12 | 1.069640 (T27) |
| v15 | v14 + torch.compile on the portal-proven 8-way launch; sink-logit attention at kv 1024 | 1.052909 (T28) |
| v16 | v14 + fused RoPE kernel (eager Trainium only) | 1.064665 (T29) |
| v17->v18 | v17 = concat-free sink (-0.0010 at 900 s, S_C8_LSE); v18 = + WARMDOWN_TIME_FRAC 0.90 (-0.0014, S_C8_WDN09) | 1.051593 (T31) |

Code hazards: python-float LRs bake into NEFFs (~4 s recompile per value; warmup + 60% warmdown ran 6×
slow) -> tensor-valued per-step LR/momentum; `x0` and `idx/targets` need `.clone()` (dynamo alias table);
the 4-register kv 1028 mis-tiled the compiled graph (0.41 -> 0.31 s/step) -> kv 1024; NKI attention
partitions the NEFF, so it stays off under compile; kernel imports guarded, every optimizer kernel call
wrapped (a failure disables that kernel only); v14's clean file is the last word — hard-coded config, no
compile path, one guarded NKI import block, plain `torch.load(..., weights_only=True)`. Compile-line knobs
screened elsewhere are washes or losses (`trn2_results.tsv`); narrow QK (L2, -0.0075 on GPU) and the trigram
table (L4r3) were never folded in: GPU speedups do not transfer.

## Portal failures and forensics

- **2026-09-02 — three failed slots.** v1 (`train2_best.py`, sub-f02c5ed1): `no_checkpoint`, died 0.032 s;
  a dev-only file, no Neuron/dist support. v2 (`train2_best_submission.py`, sub-286b3ddd): `no_run_meta`, no
  log at all. Root causes never confirmed (no portal stderr; the "flex import" diagnosis was disproven). v3
  response: lazy CUDA-only flex import, `SUBMISSION_BOOT` banners, py_compile gate, timed-confidence-run
  rule; baseline `train.py` scored 1.4226 (sub-a18a1065); open action: ask organizers for failed-run stderr.
- **2026-09-06 — sub-9ea276d1 (v5) FAILED `USER_TRAINING_CRASHED exit=1 status=no_run_meta` (no train.log)**
  — the v2 signature; not reproducible (bytes + launch ran to completion, pass `--compile-only` and the
  launch validator). **The one factor shared by both portal failures and absent from the organizers'
  baseline: `torch.compile(backend="neuron")` on by default** — our build needed two backend-bug
  workarounds; later, compile ON was never proven fatal (the eager LNC=2 file failed the same way; the
  8-way eager v7 scored).
- **2026-09-06 — sub-39b23de5 (v6, eager, LNC=2) also FAILED `no_run_meta`** -> compile and encoding ruled out.
- **2026-09-05 — sub-ae5cf1a6 SCORED 1.1827** (v7, eager, PACK=1, 8-way), first score since v1's 1.5437;
  local T7 1.1769 -> **portal = local + ~0.006** (held-out shard), rank ~11th.
- **Bisection.** v7 OK (compile 0, LNC=1, PACK=1); v6 FAIL (0, LNC=2, PACK=4); v5 FAIL (compile 1, LNC=2,
  PACK=4). v7 bytes on the LNC=2 launch: **scores** -> PACK=4 was the killer, then compile+LNC=2 (local
  1.140 -> ~1.146 portal, #9); **fails** -> LNC=2 is the killer, then compile on the 8-way launch (local T2
  1.152 -> ~1.158, ~#10).
- **2026-09-07 — PORTAL REJECTED_BY_REVIEW** (sub-017d7c26, 9/6 17:18 PDT = v12 bytes). Message (truncated):
  "multiple critical violations: (1) unsafe checkpoint loading via torch.load(..., weights_only=True) with a
  fallback mechanism that could deserialize arbitrary objects, (2) eva..." — the file has no such fallback
  (one `weights_only=True`, same as the baseline): at least partly a misread, rest unknown. Response: v14.
  Round-1 criteria from the user: correctness gate (finish in budget, valid score, unmodified harness, **use
  the full chip**, provided data only), score = val_bpb; **tie-breakers: fewer params, then simpler code**.
