#!/usr/bin/env python3
"""Generate v56 = v55 + do not run the post-training self-checks on a SCORED run. No model or kernel change.

## The defect this fixes

`prepare.record_step_boundary`'s docstring is the contract, and it is explicit:

    "The measured wall time from launch through the fifth reported boundary is the excluded
     startup/compilation allowance - EVERYTHING AFTER IT IS CHARGED AGAINST THE TRAINING BUDGET."

So the runner's clock starts at boundary 5 and charges wall time from there. Our stop rule reserves
`CHECKPOINT_RESERVE_SECONDS = 5.0` for the checkpoint save and **nothing at all** for what follows it: a
`causality_check` plus a full `evaluate_bpb` over the public dev shard, both run in-process after the last step.

Measured on the v55 gate (a WARM box, `runs/T73_gate_v55.log`):

    training_seconds 1794.7 | charged_seconds_est 1795.3 | over_budget False | eval_seconds 5.4

1795.3 + 5.4 = **1800.7 s against an 1800 s budget** - already marginally over, and our own `over_budget` flag
reads False only because it is computed *before* the self-checks run (`submit/train.py:3776` vs `:3780-3786`),
so it does not count them. On a cold box the same block costs far more: `eval_seconds` ranges 6.3-147.3 s across
433 box logs, 22-41 s is typical cold, and the organizers' own baseline measured 28.9 s. That is **23-142 s over
budget on the scoring host**, where the eval-shape graph is compiled for the first time (`targets=None`,
`emb=None`, 4 rows instead of 16, eager rather than compiled - a genuinely different graph that nothing traced
earlier, and `check_fused_loss` does not cover it).

**The in-run copy earns nothing.** The scorer re-runs both itself from the checkpoint via `load_for_eval`, and we
now know it does not read our stdout: the portal RE-TRAINS on its own hardware, proved by the offset between our
local score and the portal's not being constant (v43 local 0.979941 -> portal 0.9824 = +0.0025; v52 local
0.971674 -> portal 0.97854 = +0.0069). A rescoring of our checkpoint could not vary like that; a fresh run with
its own step count can. This retires the open question written into `make_v54.py`, which kept the in-run eval
because "we do NOT know whether the portal reads val_bpb from our stdout". It does not.

It is also an **unguarded crash surface**: `evaluate_bpb(split="public_val")` calls `list_shard_paths` on the
public val shard with no `try/except` around it, so a scoring host without `shard_06542.parquet` takes the whole
run down after training has already succeeded.

## The change (two lines)

1. `--eval-public` now defaults **off**. Dev gates already pass `NEURON_COMPETITION_R1_EVAL_PUBLIC=1`
   explicitly (see any `runs/*_gate_*.env`), so gating is unaffected and still reports `public_val_bpb` and
   `causality_passed`. Only the scored run changes, and only in skipping self-checks the scorer repeats.
2. `WALL_RESERVE_SECONDS` 300 -> 90. The reserve existed to hold room for "checkpoint save + causality check +
   eval (156 s cold) + teardown"; with the self-checks gone only the ~820 MB save and teardown remain. A smaller
   reserve means the adaptive total-wall guard fires **later**, so on a slow-compiling host it gives back fewer
   training steps than v54/v55 would.

Neither touches the math, so the local gate score should be identical to v55's within noise; the gain is that a
scored run is no longer 5-142 s over the charged budget.

Run from the repo root:  .venv/bin/python submission_raw/make_v56.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v55_clean_wviews.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v56_clean_noselfcheck.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------- the self-checks default off on a scored run
sub("eval default", '''    parser.add_argument("--eval-public", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_EVAL_PUBLIC", "1") != "0")''',
    '''    # DEFAULT OFF on a scored run. record_step_boundary's contract charges "everything after" the 5th boundary
    # against the 1800 s budget, and the in-run causality_check + evaluate_bpb cost 5.4 s warm but 22-147 s cold
    # (the eval-shape graph - targets=None, emb=None, 4 rows, eager - is compiled for the first time there). The
    # scorer re-runs both itself from the checkpoint, and it does NOT read our stdout: the portal re-trains on
    # its own host, proved by the local->portal offset varying (v43 +0.0025, v52 +0.0069). Dev gates pass
    # NEURON_COMPETITION_R1_EVAL_PUBLIC=1 to get public_val_bpb and causality_passed back.
    parser.add_argument("--eval-public", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_EVAL_PUBLIC", "0") != "0")''')

# ---------------------------------------------------------------- shrink the wall reserve accordingly
sub("wall reserve", """WALL_RESERVE_SECONDS = 300.0    # checkpoint save (~820 MB) + causality check + eval (156 s cold) + teardown""",
    """WALL_RESERVE_SECONDS = 90.0     # checkpoint save (~820 MB) + teardown. Was 300 while the in-run causality
                                # check and eval still ran after training; those are off by default now, so the
                                # guard can fire later and give back fewer steps on a slow-compiling host.""")

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:16s} x{v}")
