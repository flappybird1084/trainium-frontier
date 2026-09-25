#!/usr/bin/env python3
"""Generate v54 = v52 + ONE infrastructure change for the portal: an adaptive total-wall guard. No model or kernel change whatsoever.

Why: v52 (sub-83850d03) failed TWICE on the portal with `States.Timeout` and no val_bpb, each time after
roughly an hour, while running correctly end to end locally (cold compile 504 s, trains its 1800 s,
over_budget False, causality passed, official `prepare.py eval-public` scores it in 156 s cold). Compile growth
is ruled out: v43, which scored on the portal, compiles SLOWER (603 s vs 504 s). So the run is being killed by a
TOTAL WALL limit we cannot observe, not by anything wrong with the model.

The change below bounds our process's wall clock. It does not touch the math, and on this box it does not
change the score (verify with a gate: training still ends at ~2346 s of wall, far under the cap).

**The adaptive total-wall guard.** Today training stops on ITS OWN clock: a full `max_train_seconds` after the
5th step boundary, however long compilation took. If the harness limits TOTAL wall, a slow-compile host
pushes us past it and we score nothing. `WALL_CAP_SECONDS` (default 2900) bounds wall since process start,
with `WALL_RESERVE_SECONDS` held back for the checkpoint, causality check and teardown. The guard rides the
existing stop-decision all_reduce, so every rank still breaks on the same step.

On this box: boundary 5 at ~546 s + 1800 s of training = ~2346 s of wall at the end of training, and the
guard triggers only above 2900 - 300 = 2600 s, so it NEVER FIRES and the score is unchanged. On a host whose
compile is ~250 s slower than ours it starts trading training steps for actually finishing - which is
strictly better than a timeout that returns nothing.

Run from the repo root:  .venv/bin/python submission_raw/make_v54.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v52_clean_compile_8way_both.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v54_clean_wallguard.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------- the wall cap constants
sub("wall cap consts", """CHECKPOINT_RESERVE_SECONDS = 5.0""",
    """CHECKPOINT_RESERVE_SECONDS = 5.0
# Bound on TOTAL wall since process start, not just on the training budget. The training budget is measured from
# the 5th step boundary, so a slow compile does not shorten training - it pushes the whole run later, and a
# harness wall limit then kills it with no score at all. This guard trades training steps for completion, and
# only when the run is actually late: on a fast-compiling host it never fires.
WALL_CAP_SECONDS = float(os.environ.get("NEURON_COMPETITION_R1_WALL_CAP_SECONDS", "2900"))
WALL_RESERVE_SECONDS = 300.0    # checkpoint save (~820 MB) + causality check + eval (156 s cold) + teardown""")

# ---------------------------------------------------------------- the stop condition
sub("wall stop", """                or elapsed_after_step + max(recent_step_times[-20:]) + CHECKPOINT_RESERVE_SECONDS >= args.max_train_seconds))""",
    """                or elapsed_after_step + max(recent_step_times[-20:]) + CHECKPOINT_RESERVE_SECONDS >= args.max_train_seconds
                # total-wall guard: stop while there is still room to checkpoint, score and tear down
                or ((time.monotonic() - _PROCESS_STARTED) + max(recent_step_times[-20:])
                    + WALL_RESERVE_SECONDS >= WALL_CAP_SECONDS)))""")

# NOT DONE, deliberately: turning the in-run eval OFF would save ~156 s of cold eval-graph compile on the
# scored run, but we do NOT know whether the portal reads val_bpb from our stdout or computes it itself via
# load_for_eval (the failed submission showed "val_bpb -", consistent with either). If it reads ours, disabling
# the eval would make a SUCCESSFUL run unscorable - a far worse outcome than 156 s of wall. Also, it is
# causality_check (not evaluate_bpb) that triggers the eval-shape compile, so skipping only the eval saves ~28 s.

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:24s} x{v}")
