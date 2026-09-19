# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Starter kit for the NKI Frontier competition: train a language model from scratch on AWS Trainium under a fixed time budget (30 min Phase 1 / 4 hr Phase 2) and minimize validation bits-per-byte (`val_bpb`). Optimization can happen at any level — architecture, optimizer, data sampling, or custom Trainium kernels written in NKI (Neuron Kernel Interface).

## Reporting to me

Status reports must be **concise and terse**. I come up with the ideas; I did not write the code — assume I have no
knowledge of the logs, file names, env-var spellings, or code internals. Lead with the outcome in one line, use plain
language, and include only what changes what I would decide next (no knob tables, no hashes, no file inventories, no log
excerpts unless I ask for them).

## Commands

Assumes a Trainium DLAMI environment that already provides `torch`, `torch_neuronx`, and `neuronx-cc`. There is no test suite or linter.

```bash
pip install -r requirements.txt                 # extra deps: pyarrow, tiktoken, requests

# Download data shards + unpack the fixed tokenizer (cached in ~/.cache/neuron_competition_r1)
python prepare.py setup                         # --num-train-shards N to fetch fewer while iterating

# Train the baseline (checkpoint written to out/final.pt)
NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py
# lnc=2 variant: NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py

# Score a checkpoint with the exact official metric (public dev shard)
python prepare.py eval-public --train-py train.py --checkpoint out/final.pt

# Static-check a submission (forbidden imports/calls, launch-command shape)
python prepare.py check --train-py train.py --launch-command "NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py"

# Rapid-iteration loop: single core, fixed 5-minute training budget
cd autoresearch && bash run.sh
```

`train.py` auto-detects the device and also runs on CUDA/CPU dev boxes (the NKI import is guarded; the example kernel silently falls back to eager off-Neuron). Runtime knobs are env vars prefixed `NEURON_COMPETITION_R1_` (e.g. `NUM_STEPS`, `MAX_TRAIN_SECONDS`, `OUT_DIR`, `EVAL_TOKENS`, `COMPILE`, `CACHE_DIR`) mirrored by CLI flags — see `main()` in train.py.

## Layout of our own files (2026-09-06)

- `submit/` — what to upload now: `train.py` is a byte-identical renamed copy of the latest `submission_v#` file kept beside it.
- `submission_raw/` — archive of every frozen `submission_v#` plus the experiment vehicles (`train2*.py`, `train3_submission.py`).
  `train3_submission.py` is the live vehicle for Trn2 experiments; it imports `prepare.py` from the repo root, so run it with
  `PYTHONPATH=.` (the box launcher `trn2_run.sh` does this). Results logs stay at the root (`trn2_results.tsv`, `train2_results.tsv`).
- `nki/` (2026-09-11) — every NKI kernel module and its device/simulator test (`nki_*.py`: head, layer, mlp, rope, rmsnorm,
  attention fwd/bwd, pre/post backward, muon, adamw, embed, benches). No `__init__.py` on purpose: a regular package would
  shadow the installed `nki` library; a plain directory does not. Run tests from the box as `python nki/<file>.py` (sibling
  imports resolve because the script's own directory leads sys.path). The box mirror is `~/tf/nki/`. The vehicle embeds
  kernel bodies by text, so it does not import these files.
- `train.py` at the root is the organizers' baseline and stays where it is.
- Box tools at the root (master copies, rsynced to the box's `~/tf`): `trn2_run.sh` (gated launcher), `trn2_queue.sh` (gated
  queue runner), `trn2_job.sh` (gated single-core job), `trn2_status.sh`, `trn2_profile_one.sh` + `neff_instr_agg.py` (NEFF
  engine/instruction profiles). Older profile drivers live in `experimental/trn2_legacy/`.
- `experimental/` (2026-09-11) — parked or legacy material kept for reference: `gdn/` (chunked gated delta rule in torch and
  its device tests; DeltaNet stage 1, not viable on this stack), `neff/neff_ops_agg.py` (per-op aggregation, needs pandas),
  `neuron/` (fresh-box health probes `neuron_probes.py` / `dist_probe.py` / `run_probes.sh` from the deadlock
  investigation, `neuron_profile_step.py`), `monitor/nm_monitor.py` (neuron-monitor line formatter). `gpu_gdn.sh` at the
  root runs the vehicle on the local CUDA GPU at fixed steps for architecture questions.

## Trn2 dev instance

SSH/login details, environment layout, and usage notes for the Trn2 dev box live in
**`trn2-access.md`** (gitignored, together with the `trn2.pem` key — never commit either).
Check it before any Trainium-side work.

## Findings directory

`findings/` holds non-instruction, project-related knowledge. **Start with the newest `findings/handoff-*.md`
(`handoff-2026-09-19.md`: state, the v44-v46 chain, what is closed with numbers, rules), then `findings/tools.md` (what every tool does and
when to use it), then the ledger `trn2_results.tsv` and the newest entries of `findings/trn2-experiments.md`; read the rest of
`findings/` before working on the competition** - it is the source of truth for anything not derivable from this repo's code,
and new discoveries belong there, not in this file. Older handoffs (`handoff-2026-09-13b.md`, `handoff-2026-09-13.md`, `handoff-2026-09-11b.md`, `handoff-2026-09-11.md`, `handoff-2026-09-06*.md`) are history.

`findings/` holds non-instruction, project-related knowledge: organizer rule clarifications, experiment results and lessons, hardware notes. **Start with the newest `findings/handoff-*.md` (session handoff: state, box, rules, next steps), then read every file in `findings/` before working on the competition** — it is the source of truth for anything not derivable from this repo's code, and new discoveries belong there, not in this file.

## The one hard rule

**`prepare.py` is fixed and must never be modified** — it is the trusted competition harness (data download, tokenizer, dataloaders, `evaluate_bpb()` scoring, `causality_check()`, submission checks). Everything a participant edits lives in `train.py`. Also off-limits per the rules: pre-trained weights/tokenizers, external training data, and reading validation data. `check_submission()` in prepare.py rejects train.py files importing `socket`/`subprocess`/`requests`/`urllib`/`http`/`ftplib` or calling `eval`/`exec`/`__import__`/`compile`/`os.system` — keep those out of train.py.

## Architecture

Two self-contained files, plus a sandbox:

- **`train.py`** (participant-editable, ~760 lines): GPT-style model (RMSNorm, RoPE, QK norm, ReLU² MLP, logit softcap, x0 residual skip), a combined `MuonAdamW` optimizer (Muon with Polar Express orthogonalization for matrix params, AdamW for embeddings/head/scalars), gradient-accumulation training loop with manual `all_reduce` gradient sync (no DDP wrapper), and one example NKI kernel (fused `relu(x)²`, off by default, `--nki-relu2` to enable).
- **`prepare.py`** (fixed): constants (`TRAIN_TIME_BUDGET_SECONDS`, `TOKENIZER_VOCAB_SIZE=8192`), SHA256-verified tokenizer artifact loading (`artifacts/tokenizer_8192.zip`), ClimbMix parquet shard download, BOS-aligned best-fit-packed train dataloader + finite eval dataloader, and the trusted `evaluate_bpb()` which recomputes cross-entropy from `model(x)` logits — it never trusts model-reported loss.
- **`autoresearch/`** (dev tool only, never part of a submission): a simplified single-core copy of train.py with its own hyperparameters, run via `run.sh` with `PYTHONPATH=..` so it imports the shared `prepare.py`. `program.md` defines an autonomous experiment loop: branch `autoresearch/<tag>`, edit `autoresearch/train.py`, commit, `bash run.sh > run.log 2>&1`, grep `val_bpb` from the log, record to `results.tsv` (columns: commit, val_bpb, status keep/discard/crash, description), keep the commit only if val_bpb improved. Follow `program.md` when running experiments there.

### The eval contract

The scorer imports the submitted train.py and calls its **`load_for_eval(checkpoint_path, device) -> nn.Module`** hook. The returned model must be a consolidated, single-device, full replica with a pure, stateless, causal forward: logits depend only on input tokens (no collectives, no rank/batch dependence). Any parallelism is training-only — consolidate before saving, since the scorer does not merge shards. `causality_check()` smoke-tests causality after training.

### NKI kernel notes (from the example in train.py)

- Kernel pattern: DMA a tile from HBM into SBUF, compute with `nisa` ops, DMA back. The SBUF partition dim is capped at 128 rows (`_NKI_PARTITION`), so loop over row chunks.
- Raw NKI kernels are **not differentiable** — wrap forward+backward in a `torch.autograd.Function` or gradients silently stop flowing.
- On Neuron, changing tensor shapes triggers NEFF recompilation (~60s+); keep shapes fixed across steps. Do not use `torch.cuda.*`; `torch.compile` is allowed but compile time counts against the training budget.
