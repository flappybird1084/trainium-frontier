# Official rule clarifications (Devpost discussions, organizer "lping2024")

Noted 2026-09-01; item 2 updated 2026-09-03 (upstream c907da4).

1. **Custom sequence packing inside train.py is allowed.** Repacking the provided train shards with the provided
   Tokenizer counts as "data sampling strategy" (README line 163); the protected "data pipeline" (line 176) means
   organizer code — prepare.py, tokenizer, eval/scoring — not batch construction inside train.py. Constraints:
   train split only, provided tokenizer, eval path untouched.
   - Organizer-suggested low-effort variant: call `make_dataloader` with `seq_len = 8*1024` and `.reshape()` each
     yielded `(x, y)` to `(batch*8, 1024)`. Next-token alignment is preserved exactly (targets are the same row
     shifted by one, so chunk boundaries chain correctly); only documents longer than the enlarged row lose their tail.
2. **Compilation time is excluded from the Phase 1 30-minute budget — and the exclusion is MECHANIZED** (upstream
   c907da4, 2026-09-03; supersedes the earlier "train.py's own internal timer" reading). The runner watches an
   append-only step log (`NEURON_COMPETITION_R1_STEP_LOG`) and takes its own clock readings: wall time from launch
   through the **5th** reported optimizer-step boundary is the excluded startup/compilation allowance, **capped at
   900 s** (`TRAIN_STARTUP_STEPS_EXCLUDED=5`, `TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900`). Submissions MUST call
   `prepare.record_step_boundary(step)` exactly once per completed optimizer step (after device sync; rank 0 writes,
   other ranks no-op) — `check_submission` REJECTS files that never call it, and **a run that never reports 5
   boundaries is not scorable**. Self-managed pre-clock warmups are superseded: the baseline restarts its budget
   clock at boundary 5 (back-dating by any over-cap excess + 30 s launch margin) and adds `--compile-only` (one
   synthetic optimizer step, then exit). Consequence unchanged: NEFF compile cost is not a reason to avoid
   torch.compile or shape-specialized NKI kernels — but the allowance is finite (900 s), so total compile must
   fit inside it.
3. Every Phase 1 submission trains a language model; `val_bpb` is the sole Phase 1 metric.
