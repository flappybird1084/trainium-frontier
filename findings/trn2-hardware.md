# Trn2 hardware — first real measurements (2026-09-05)

First working run on actual Trainium. Box "test4": the competition beta AMI
`neurips-torchneuronx-native-beta-2.11.3-20260731` (`ami-05b825677ea807d13`, confirmed via IMDS; its login
banner shows the "Deep Learning AMI Neuron (Ubuntu 24.04)" *base image* — same AMI, not a different one),
native stack (`~/workspace/native_venv` + `torch_neuron_eager`), trn2.3xlarge, 8-way `NEURON_LOGICAL_NC_CONFIG=1
torchrun --nproc_per_node=8 train.py`, baseline (unmodified) `train.py`. The same AMI deadlocked on 2 other
fresh boxes (intermittent fork race — see `trn2-access.md`). Access/SSH notes and the
deadlock post-mortem live in `trn2-access.md` (gitignored). The "don't call a hang from silence"
process rule is memory `trn2-startup-silent-gaps`.

## Baseline config that ran
- 10 layer, `n_head=5`, `n_embd=640`, seq 1024, vocab 8192 → **59.6M params**, `flops_per_token=4.05e8`.
- `micro_tokens=8192`, `grad_accum=8` → **65,536 tokens / optimizer step**. Budget 1800 s, target 1000 steps (time-bound).

## Startup cost (cold)
- ~2 min silent init (NCCL/OFI + graph build) **then** step 0 `dt=78 s` (NEFF compile folded in) → **~1–4 min "to hit the cores."**
- **Throughput ramps for ~20 steps** (graph-variant recompiles), *not* a compute spin-up:
  steps 1–10 ≈ **3.8 s/step (~17k tok/s)** → steps 20+ steady **≈1.55 s/step (~42k tok/s)**.
  The ramp exceeds the 5-step startup exclusion — worth remembering for submission timing (cold-cache runs pay it).
- Compile cache persists on disk → warm re-launches skip the 78 s. Set/keep a stable `CACHE_DIR` for fast dev iteration.

## Steady-state throughput + the big signal
- **≈1.55 s/step, ≈42,000 tok/s, cores at ~40% utilization** on the 640-wide baseline → **large headroom.**
- Loss healthy: 9.01 (step 0) → 4.2 (step 210) with LR warmup to `lrm=1.0` by step 20.

## Tuning implications (the actionable part)
- 40% util = **latency/overhead-bound, not compute-bound.** Cheapest lever to capture headroom is a **bigger
  micro-batch** (amortize per-step overhead, more tokens in budget); next is a **bigger/wider model** (more FLOPs/token).
  Goal is still min `val_bpb` = (quality per token) × (tokens seen in budget), but the chip is telling us it's starved.
- **Our champion is 256-wide — NARROWER than this 640-wide baseline** — so it would utilize *even less* of the chip.
  All prior architecture tuning was done on CPU/GPU dev boxes with a different compute/throughput profile; the
  Trainium optimum is very likely **bigger/wider + bigger batch**, and dev-box wins that *trade width for throughput*
  (e.g. narrow QK) may not pay off when we're throughput-rich here. **First Trainium experiment = batch/width sweep
  to climb utilization, NOT a blind champion port.**
- **int64 unsupported → auto-cast to int32** (repeated `dtype_autocast` warnings). Feeding int32 token ids/positions
  directly avoids the per-op cast — a concrete, free Trainium-specific micro-optimization to fold into train.py.

## Benign warnings on a single-node box (ignore)
- `aws-ofi-nccl … OFI plugin initNet() failed is EFA enabled?` — no EFA fabric on 1 node; local-transport fallback.
- `barrier(): using the device under current context` — cosmetic; can silence with `device_id` in `init_process_group`.

## Baseline result — the anchor (test4, 2026-09-05)

Unmodified `train.py`, 8-way LNC=1, full 1800 s budget, completed the 1000-step target:

| field | value |
|---|---|
| **public_val_bpb** | **1.258923** |
| num_steps / total_tokens | 1000 / 65.5M |
| training_seconds (charged) | 1635.1 (budget 1800, `over_budget: False`) |
| startup_allowance_est | 225.8 s excluded (~2 min init + 78 s compile + warmup) |
| total_seconds (incl. eval) | 1768.8 |
| eval_seconds | 28.9 (timeout 1200) |
| causality | passed (max_abs 0.000000) |
| compiled / nki_relu2 | False / False |
| steady state | ≈1.55 s/step, ≈42k tok/s, ~40% NC util |

**Target: 1.15 val_bpb** (−0.109 from baseline). Campaign log `findings/trn2-experiments.md`; results `trn2_results.tsv`.

## Per-engine profile

The per-engine (PE / Act / Pool / gpsimd / sync) and DMA utilization breakdown at 512-wide lives in
`engine-profile-512.md`. Headline: `neuron-top`'s ~70% is the DMA engines (72-74% active), the tensor engine is
41-47%, and the compiled graph's own MFU ceiling is ~15% - the workload is data-movement-bound, not compute-bound.

## Off-device compilation

We have no standalone `neuronx-cc` path; every compile fact we hold was learned by running on the device.
The gap, what it has cost (T15/T17/T17b/T18b/S_H8/F_V24_BG19), and a proposed `hlo_compile.sh` are in
`offline-compile-gap.md`.
