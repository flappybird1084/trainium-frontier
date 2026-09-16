# Tools index (2026-09-11)

Master copies live in the project root (or the listed subdirectory) and are rsynced/scp'd to the box's `~/tf`. "Box" tools
run ON the Trn2 instance from `~/tf`; "local" tools run here.

## Box: running things

| tool | purpose | use it when | in / out |
|---|---|---|---|
| `trn2_run.sh <ID> <secs\|full> [LNC= NPROC= FILE=] [ENV=VAL..]` | gated launcher for one training run (waits for a free device, disk guard, detaches torchrun) | one run, by hand | writes `runs/<ID>.log`, `.env`, `.start`; checkpoint in `out_<ID>/` |
| `trn2_queue.sh <queue-file>` | runs the lines of a queue file one after another (`<ID> <secs\|full> ENV=..`, `JOB <ID> <cmd>`, `PJOB`, `WAIT`) | several runs/tests unattended | `runs/queue_<name>.out`; **append only**, and only while the runner is several items from the end |
| `trn2_job.sh <ID> <cmd>` | gated single-core job (kernel tests, profiler captures); refuses while torchrun holds the device; pins a logical core via `NEURON_RT_VISIBLE_CORES` + `runs/.core_<n>.lock` | device tests, profiles | `runs/<ID>.log` |
| `trn2_status.sh <ID>` | one-line status of a run (procs, compiler running, steps, done/err) | quick check | stdout |

## Box: profiling

| tool | purpose | in / out |
|---|---|---|
| `trn2_profile_one.sh <tag> <neff>` | neuron-profile capture (2nd exec) + summary-text + parquet export (with timeouts: the exporter hangs after writing) + `neff_instr_agg.py` aggregation | `runs/prof_<tag>/summary.txt`, `parquet/`, aggregation in the job log |
| `neff_instr_agg.py <prof dir> [top]` | per-engine / per-opcode / per-source-line instruction-time aggregation (pyarrow only) | stdout |
| NEFFs to profile | `/tmp/neff_cache/xx/yy/<hash>.neff`; find the run's by mtime (`find /tmp/neff_cache -name "*.neff" -mmin -N`) | |
| `experimental/trn2_legacy/`, `experimental/neff/neff_ops_agg.py`, `experimental/neuron/` | older profile drivers (`trn2_profile_neff/ops/sum`), the pandas aggregator, the fresh-box probes | superseded |

## Local

| tool | purpose |
|---|---|
| `prepare.py check --train-py <file> --launch-command "..."` | the organizers' static check (forbidden imports/calls); run on every clean file |
| `prepare.py eval-public --train-py <file> --checkpoint <pt>` | exact official scoring of a checkpoint (never run standalone on the box) |
| `$S/smoke_v##.py` (scratchpad) | CPU smoke of a clean file: 2-layer model, loss == CE(head_logits), eval hook signature |
| `$S/make_v##.py` (scratchpad) | version generators: each derives `submission_raw/submission_v##_clean_compile_8way.py` from the previous clean file + kernel bodies extracted from the vehicle by text |
| `gpu_gdn.sh <tag> <steps> [ENV=..]` | run the vehicle on the local CUDA GPU at fixed steps (v39 config minus Trainium-only knobs) for architecture questions; logs in `runs/gpu/` |
| `.venv-nki/bin/python nki/<kernel>.py` | NKI simulator check of a kernel (each module's `__main__` compares against a torch reference) |
| `.venv/bin/python` | CUDA torch (local GPU); `.venv-nki` is CPU torch + the NKI simulator |

## Kernel modules and tests (`nki/`)

| file | kernel(s) | test |
|---|---|---|
| `nki_head.py` | fused linear cross-entropy fwd/bwd, logits-returning variant (HEAD=nki / nki2) | `nki_head_test.py`, `nki_head_fq_test.py` |
| `nki_layer.py` | pre_attn_fwd, post_attn_fwd (the layer forward as kernels; `_h` = head-major layout) | `nki_layer*_test.py` |
| `nki_pre_bwd.py` / `nki_post_bwd.py` | fused pre-attention backward (in v37+) / fused post-attention backward (correct, a wash; unused) | `nki_{pre,post}_bwd_test.py` |
| `nki_mlp.py`, `nki_rmsnorm.py`, `nki_rope.py` | mlp fwd/bwd, rmsnorm fwd/bwd, rope | `nki_mlp_test.py`, `nki_norm_test.py`, `nki_rope_test.py` |
| `nki_attn_fwd.py`, `nki_attn_bwd.py` | own attention (v1..v4 fwd, v3 bwd; parked — strided DMAs unsafe in-graph) | `nki_attn_*_test.py` (incl. the race-bisection suite: `_small`, `_det`, `_pair*`, `_sentinel`, `_prod`, `_real`, `_graph*`) |
| `nki_muon.py`, `nki_adamw.py`, `nki_embed.py` | optimizer + embedding sidecar kernels | `nki_muon_test.py`, `nki_adamw_test.py`, `nki_embed_test*.py` |
| `nki_fp8_bench.py`, `nki_mlp_bench.py`, `nki_opt_bench.py` | microbenchmarks (fp8 double-row, fused MLP, optimizer) | self-contained |

## Conventions
- **Screen** = 600 s (or 120 s for profiles) with `PROFILE=1` (adds per-step syncs: a flake risk, never on gates);
  **gate** = full 1800 s of the clean file, no profiler. Reference screens are re-run alongside each batch (noise bands in
  `trn2-experiments.md`).
- Queue IDs: `S_<ver>_<name>` screens, `F_<ver>_<name>` full vehicle runs, `T##_gate` clean-file gates, `JOB <ID>` tests.
- After a run: ledger row in `trn2_results.tsv`, narrative in `findings/trn2-experiments.md`, log copied to `runs/trn2/box/`.
