#!/usr/bin/env python3
"""Generate train3_s25.py = train3_s24.py + ANVIL twin-rail Muon velocity (knob RAILS), TORCH PATH ONLY.

## The hypothesis this tests, and why it reframes three failures

On 2026-09-26 I gated three components of the 2026-08-30 modded-nanogpt record (ANVIL2) individually:

    per-group terminal weight averaging   -0.00029   (won, small)
    published 6-map Polar Express cascade +0.00062   (LOST)
    published Muon momentum schedule      +0.00068   (LOST)

**ANVIL is a tuned package, and testing its pieces against a different baseline is a textbook way to get false
negatives.** The clearest case is the momentum schedule: it cools 0.93 -> 0.85 over the last 4.19% of the run,
which only makes sense *because a slow velocity rail carries the long-horizon signal at the end*. Without the
rail, cooling momentum just makes the optimizer myopic exactly when it should be settling - which is the shape
of the small loss I measured. The cascade plausibly interacts too (its envelope is tuned for the spectra the
rails produce).

So the real experiment is the **package**, and the missing piece is the twin rail.

## The change (all constants PUBLISHED at train_gpt.py:224, :240-250, :3382-3394, :3617-3619)

    _RAIL_FAST_BETA  = 0.85     fast-rail beta AFTER engagement
    _RAIL_SLOW_BETA  = 0.98     slow rail, constant, zero-init from step 0
    _RAIL_FAST_W     = 0.4385   fast-rail weight AFTER engagement
    engagement       = step 514 of 1194 = 0.4305 of the run  (HARD switch, no ramp)

Per step, in order:

    v0 <- lerp(v0, g, 1 - beta_fast)        # beta_fast = 0.85 if engaged else the momentum schedule's value
    v1 <- lerp(v1, g, 1 - 0.98)             # always accumulating
    m_eff = w*v0 + (1-w)*v1                 # w = 0.4385 if engaged else 1.0  -> before engagement m_eff == v0
    g_used = lerp(g, m_eff, momentum)       # the existing Nesterov lookahead, unchanged

Before engagement `w = 1.0`, so the slow rail accumulates but is unused and the first 43% of the run is
**exactly today's plain Muon Nesterov**. That is a useful property: it means the arm cannot diverge early.

## Why TORCH-ONLY, deliberately

The fused NKI Muon kernel (`muon_step_stack`) would need a second HBM buffer threaded through its signature plus
two more scalars, in the most numerically delicate kernel in the file - 4-5 h with a real NaN/silent-wrongness
risk. **This version is ~10 lines in the torch fallback and costs no device time at all**, because it is meant
to run on the local RTX 4080 at FIXED STEPS via `gpu_gdn.sh`, where the NKI paths are unavailable anyway.

That makes it a cheap FALSIFIER: if the rails do not help at fixed steps on the local GPU, the package
hypothesis dies and the kernel work is never started. If they do help, the kernel port becomes worth its 4-5 h.
Bias to remember when reading the result: short fixed-step runs UNDERSTATE a lever whose benefit compounds, and
this one engages only at 43% of the run, so a null at 500 steps is weak evidence while a win is strong.

Defaults leave `RAILS=0`, i.e. today's behaviour exactly.

Run from the repo root:  .venv/bin/python submission_raw/make_s25.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "submission_raw/train3_s24.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "submission_raw/train3_s25.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------------------- knobs
sub("knobs", '''MOM_SCHED = _ENV("NEURON_COMPETITION_R1_MOM_SCHED", "hold")          # "hold" = today's behaviour''',
    '''MOM_SCHED = _ENV("NEURON_COMPETITION_R1_MOM_SCHED", "hold")          # "hold" = today's behaviour
# RAILS: the ANVIL twin-rail Muon velocity. A second, slower EMA of the reduced gradient accumulates from step 0
# but is UNUSED until engagement (fast weight 1.0), so the first ~43% of the run is exactly plain Muon Nesterov.
# TORCH PATH ONLY - this is a cheap falsifier for the local-GPU fixed-step test, not a shippable change; the
# fused NKI Muon kernel would need a second HBM buffer and two more scalars.
RAILS = _ENV("NEURON_COMPETITION_R1_RAILS", "0") != "0"
RAIL_FAST_BETA = float(_ENV("NEURON_COMPETITION_R1_RAIL_FAST_BETA", "0.85"))
RAIL_SLOW_BETA = float(_ENV("NEURON_COMPETITION_R1_RAIL_SLOW_BETA", "0.98"))
RAIL_FAST_W = float(_ENV("NEURON_COMPETITION_R1_RAIL_FAST_W", "0.4385"))
RAIL_ENGAGE = float(_ENV("NEURON_COMPETITION_R1_RAIL_ENGAGE", "0.4305"))   # fraction of the run (514/1194)''')

# ---------------------------------------------------------------------------- the second buffer
sub("state", '''        if "second_momentum_buffer" not in state:''',
    '''        if RAILS and "slow_rail" not in state:
            state["slow_rail"] = torch.zeros_like(p0)
        if "second_momentum_buffer" not in state:''')

# ---------------------------------------------------------------------------- the update
sub("update", """        mom_t = self._group_scalar(group, "momentum_t").to(momentum_buffer.dtype)
        momentum_buffer.lerp_(g, 1 - mom_t)
        g = g.lerp_(momentum_buffer, mom_t)""",
    """        mom_t = self._group_scalar(group, "momentum_t").to(momentum_buffer.dtype)
        if RAILS:
            # ANVIL twin rail. Engagement is a HARD switch on fraction-of-run, matching the published rule
            # (their step 514 of 1194). Before it, w = 1.0 so m_eff == the fast rail and this is plain Nesterov.
            engaged = getattr(self, "rail_frac", 0.0) >= RAIL_ENGAGE
            beta_fast = RAIL_FAST_BETA if engaged else float(mom_t)
            w_fast = RAIL_FAST_W if engaged else 1.0
            slow = state["slow_rail"]
            momentum_buffer.lerp_(g, 1 - beta_fast)
            slow.lerp_(g, 1 - RAIL_SLOW_BETA)
            m_eff = momentum_buffer if w_fast >= 1.0 else torch.lerp(slow, momentum_buffer, w_fast)
            g = g.lerp_(m_eff, mom_t)
        else:
            momentum_buffer.lerp_(g, 1 - mom_t)
            g = g.lerp_(momentum_buffer, mom_t)""")

# ---------------------------------------------------------------------------- feed the engagement clock
sub("clock", """                group["weight_decay"] = WEIGHT_DECAY
        optimizer.refresh_scalars()""",
    """                group["weight_decay"] = WEIGHT_DECAY
        optimizer.rail_frac = elapsed_frac      # RAILS: engagement is on fraction-of-run, not step index
        optimizer.refresh_scalars()""")

# ---------------------------------------------------------------------------- prove which path is live
sub("startup print", '''    print0(f"_FMAX={_FMAX} (chunks per 1024-token row: {1024 // _FMAX}; PSUM fp32 free-dim cap is 512)")''',
    '''    print0(f"_FMAX={_FMAX} (chunks per 1024-token row: {1024 // _FMAX}; PSUM fp32 free-dim cap is 512)")
    if RAILS:
        print0(f"RAILS: twin-rail Muon velocity, fast beta {RAIL_FAST_BETA} weight {RAIL_FAST_W}, "
               f"slow beta {RAIL_SLOW_BETA}, engaging at elapsed_frac {RAIL_ENGAGE} (TORCH PATH ONLY)")''')

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:16s} x{v}")
