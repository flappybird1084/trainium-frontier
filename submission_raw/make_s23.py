#!/usr/bin/env python3
"""Generate train3_s23.py = train3_s22.py + the ANVIL Muon-momentum schedule (knob MOM_SCHED=anvil).

## What this is

The 2026-08-30 modded-nanogpt record (PR #360) sets Muon's Nesterov momentum from a schedule that differs from
ours in two ways, both free:

    ours    : 0.85 -> 0.95 over MOM_WARMUP_STEPS (300 steps), then HOLD 0.95 to the end
    theirs  : 0.85 -> 0.93 over 240 of 1194 steps (20.1% of the run), hold 0.93,
              then COOL 0.93 -> 0.85 over the last 50 steps (4.19% of the run)

So they peak **lower** (0.93 vs 0.95) and, unlike us, **cool the momentum back down at the very end**. The
terminal cooldown is an item `findings/ideas.md` already lists as untried, and it arrives free here.

This is deliberately split out from the rest of the twin-rail port (`_RAIL_FAST_BETA` / `_RAIL_SLOW_BETA` /
`_RAIL_FAST_W`), because the rails need a second velocity buffer threaded through the **fused NKI Muon kernel**
(a real kernel change) while this is **pure Python on the host** - a per-group float assigned once per step,
with no graph, no shape and no recompile. Test the free half first.

## Why it is expressed in fractions, not steps

Our step count is time-bound and floats run to run (4085-4243 across recent gates), so their absolute 240/50
steps are ported as fractions of the run and read off `elapsed_frac`, which the training loop already computes
and broadcasts (`synced_elapsed_frac`). The warmup keeps a steps-based floor too, so behaviour at the very
start does not depend on an `elapsed_frac` that is pinned to 0.0 during the excluded startup steps.

At T=4243 the fractions land at: warmup 853 steps, cooldown over the last 178 steps.

Defaults leave `MOM_SCHED="hold"`, i.e. today's behaviour exactly.

Run from the repo root:  .venv/bin/python submission_raw/make_s23.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "submission_raw/train3_s22.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "submission_raw/train3_s23.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------- the knobs
sub("knobs", '''MOM_WARMUP_STEPS = int(_ENV("NEURON_COMPETITION_R1_MOM_WARMUP", "300"))  # Muon momentum ramp 0.85 -> 0.95 span, in STEPS''',
    '''MOM_WARMUP_STEPS = int(_ENV("NEURON_COMPETITION_R1_MOM_WARMUP", "300"))  # Muon momentum ramp 0.85 -> 0.95 span, in STEPS
# MOM_SCHED="anvil": the published ANVIL momentum schedule - peak 0.93 (not 0.95), warmup over 20.1% of the run,
# and a terminal COOLDOWN back to 0.85 over the last 4.19%. Ours holds its peak to the end; the cooldown is an
# item ideas.md lists as untried. Pure host-side floats: no graph, no shape change, no recompile.
MOM_SCHED = _ENV("NEURON_COMPETITION_R1_MOM_SCHED", "hold")          # "hold" = today's behaviour
MOM_MIN = float(_ENV("NEURON_COMPETITION_R1_MOM_MIN", "0.85"))
MOM_PEAK = float(_ENV("NEURON_COMPETITION_R1_MOM_PEAK", "0.93"))     # theirs; ours is 0.95
MOM_WARM_FRAC = float(_ENV("NEURON_COMPETITION_R1_MOM_WARM_FRAC", "0.201"))   # 240/1194
MOM_COOL_FRAC = float(_ENV("NEURON_COMPETITION_R1_MOM_COOL_FRAC", "0.0419"))  # 50/1194''')

# ---------------------------------------------------------------- the schedule itself
sub("schedule", """def muon_momentum(step: int) -> float:
    frac = min(step / max(1, MOM_WARMUP_STEPS), 1.0)
    return (1 - frac) * 0.85 + frac * 0.95""",
    """def muon_momentum(step: int, elapsed_frac: float = 0.0) -> float:
    if MOM_SCHED != "anvil":
        frac = min(step / max(1, MOM_WARMUP_STEPS), 1.0)
        return (1 - frac) * 0.85 + frac * 0.95
    # ANVIL: ramp up over MOM_WARM_FRAC of the run, hold MOM_PEAK, then cool back to MOM_MIN over the tail.
    # Driven by elapsed_frac ALONE, deliberately: the loop pins elapsed_frac to 0.0 through the excluded startup
    # steps, which already yields MOM_MIN at step 0 exactly as their schedule does, so no steps-based floor is
    # needed. (An earlier draft took max() with the 300-step ramp, which silently finished the warmup at 7% of
    # the run instead of the specified 20.1% and made the fraction dead.)
    if elapsed_frac >= 1.0 - MOM_COOL_FRAC:
        t = min(1.0, (elapsed_frac - (1.0 - MOM_COOL_FRAC)) / max(1e-9, MOM_COOL_FRAC))
        return MOM_PEAK + t * (MOM_MIN - MOM_PEAK)
    return MOM_MIN + min(elapsed_frac / max(1e-9, MOM_WARM_FRAC), 1.0) * (MOM_PEAK - MOM_MIN)""")

# ---------------------------------------------------------------- pass elapsed_frac at the call site
sub("call site", """                group["momentum"] = muon_momentum(step)""",
    """                group["momentum"] = muon_momentum(step, elapsed_frac)""")

# ---------------------------------------------------------------- prove which schedule is live
sub("startup print", '''    print0(f"POLAR: {len(POLAR_EXPRESS_COEFFS)}-map schedule, {POLAR_STEPS} iterations, prescale {POLAR_PRESCALE} "
           f"(sigma_max {1.0 / POLAR_PRESCALE:.6f}; the 6-map diverges above 0.952450)")''',
    '''    print0(f"POLAR: {len(POLAR_EXPRESS_COEFFS)}-map schedule, {POLAR_STEPS} iterations, prescale {POLAR_PRESCALE} "
           f"(sigma_max {1.0 / POLAR_PRESCALE:.6f}; the 6-map diverges above 0.952450)")
    print0(f"MOM_SCHED={MOM_SCHED}" + (f": {MOM_MIN}->{MOM_PEAK} over {MOM_WARM_FRAC:.3f} of the run, "
           f"cooling back to {MOM_MIN} over the last {MOM_COOL_FRAC:.4f}" if MOM_SCHED == "anvil"
           else f": 0.85->0.95 over {MOM_WARMUP_STEPS} steps, then held"))''')

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:16s} x{v}")
