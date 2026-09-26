#!/usr/bin/env python3
"""Generate train3_s22.py = train3_s21.py + the published 6-map ANVIL Polar Express cascade (knob POLAR6).

## What this swaps, and why the prescale MUST change with it

Muon orthogonalizes each matrix gradient by iterating `X <- a*X + (b*A + c*A@A)@X` (`A = X^T X`), which acts on
singular values as `s <- a*s + b*s^3 + c*s^5`. A good schedule drives every singular value to 1. Ours is the
classical 5-map schedule; the 2026-08-30 modded-nanogpt record (PR #360, "ANVIL2") publishes a 6-map schedule
that is far flatter.

**Verified locally in float64 before writing a line of this** (all three figures reproduce the record's README
exactly, which confirms the coefficients are transcribed correctly AND that our 5-map is the schedule they
benchmark against):

    product of the six a-values          = 643.1335        (README: "tail gain 643.13")
    6-map envelope on [3e-3, 1/1.05]     = [0.99507, 1.00040]   (README: "[0.9951, 1.00041]")
    our 5-map envelope on the same range = [0.85887, 1.14113]   (README: "~[0.86, 1.14]")

So on the working band the 6-map is ~560x tighter. It is *worse* in the deep tail (at s=1.6e-3 it returns
0.8152 where ours returns 1.1065), which is the one way this could lose; for a (512,512) slice with a flat
spectrum every s ~ 1/(prescale*sqrt(512)) ~ 0.04, comfortably inside the good band.

**The trap: the 6-map DIVERGES above s = 0.952450** (bisected in float64). The iteration is only safe if the
prescale keeps `s_max <= 1/prescale` below that. Our prescale is **1.01**, giving `s_max = 0.990099` — **past
the divergence point**. Swapping only the coefficients would NaN the run. Our current 5-map diverges only above
1.021817, which is why 1.01 has always been safe for us.

**Prescale choice: 1.08, not the record's 1.05.** The envelope is flat across the whole family, so the safer
constant costs nothing measurable, and 1.05 leaves a margin of only 0.007% on a cascade that runs in bf16:

    prescale 1.05: s_max 0.952381  margin +0.007%  envelope [0.99507, 1.00040]
    prescale 1.08: s_max 0.925926  margin +2.865%  envelope [0.99507, 1.00039]
    prescale 1.15: s_max 0.869565  margin +9.532%  envelope [0.99507, 1.00039]

## The change

`POLAR6=1` selects the 6-map list, sets the iteration count to 6, and sets the prescale from `POLAR_PRESCALE`
(default 1.08). Both the NKI kernel sites and all three torch-path sites read the module-level constant, so the
two paths cannot silently disagree — this project has twice built on a knob that reached a dead code path.

Cost: one extra quintic = ~+20% of the Muon cascade. Muon is 5.1 ms of a 424 ms step, so **+1.0 ms = +0.24%**,
worth ~-0.00007 bpb of throughput against an expected quality gain of -0.0005 to -0.0015. `ns_steps` is
unrolled in the NKI kernel so this retraces and recompiles, but it adds **no new tensor shapes**.

Run from the repo root:  .venv/bin/python submission_raw/make_s22.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "submission_raw/train3_s21.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "submission_raw/train3_s22.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------- the coefficient list + knob + prescale
sub("coeffs", """POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]""",
    '''# POLAR6: the published 6-map ANVIL cascade (modded-nanogpt PR #360, 2026-08-30) instead of the classical
# 5-map. Verified in float64: product of a-values 643.1335 (their "tail gain 643.13"), envelope
# [0.99507, 1.00040] on [3e-3, 1/1.05] (their "[0.9951, 1.00041]"), against our 5-map's [0.85887, 1.14113]
# (their "~[0.86, 1.14]") - ~560x tighter on the working band, worse below s=3e-3.
# The 6-map DIVERGES above s = 0.952450, so the prescale is load-bearing: 1.01 admits s_max = 0.990099 and
# would NaN. 1.08 admits 0.925926 (a 2.9% margin) at an identical envelope, so it is preferred over their 1.05
# (whose margin is 0.007% - too thin for a bf16 cascade).
POLAR6 = os.environ.get("NEURON_COMPETITION_R1_POLAR6", "0") != "0"
POLAR_EXPRESS_COEFFS_5 = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
POLAR_EXPRESS_COEFFS_6 = [
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
]
POLAR_EXPRESS_COEFFS = POLAR_EXPRESS_COEFFS_6 if POLAR6 else POLAR_EXPRESS_COEFFS_5
POLAR_STEPS = int(os.environ.get("NEURON_COMPETITION_R1_POLAR_STEPS", "6" if POLAR6 else "5"))
# the prescale that bounds s_max = 1/POLAR_PRESCALE below the cascade's divergence point
POLAR_PRESCALE = float(os.environ.get("NEURON_COMPETITION_R1_POLAR_PRESCALE", "1.08" if POLAR6 else "1.01"))
assert POLAR_STEPS <= len(POLAR_EXPRESS_COEFFS), f"POLAR_STEPS={POLAR_STEPS} exceeds the {len(POLAR_EXPRESS_COEFFS)}-map schedule"''')

# ---------------------------------------------------------------- the two NKI kernel prescales
sub("kernel prescale", """            nisa.tensor_scalar(dst=scale, data=scale, op0=nl.multiply, operand0=1.01, op1=nl.add, operand1=1e-6)""",
    """            nisa.tensor_scalar(dst=scale, data=scale, op0=nl.multiply, operand0=POLAR_PRESCALE, op1=nl.add, operand1=1e-6)""",
    expect=2)

# ---------------------------------------------------------------- the three torch-path prescales
sub("torch prescale", """                x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)""",
    """                x = x / (x.norm(dim=(-2, -1), keepdim=True) * POLAR_PRESCALE + 1e-6)""")
sub("torch prescale b", """            x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)""",
    """            x = x / (x.norm(dim=(-2, -1), keepdim=True) * POLAR_PRESCALE + 1e-6)""",
    expect=2)

# ---------------------------------------------------------------- the iteration count on the muon group
sub("group ns_steps", """            param_groups.append(dict(kind="muon", params=[self.stacks[key]], lr=MATRIX_LR, momentum=0.95, beta2=0.90, ns_steps=5, weight_decay=WEIGHT_DECAY))""",
    """            param_groups.append(dict(kind="muon", params=[self.stacks[key]], lr=MATRIX_LR, momentum=0.95, beta2=0.90, ns_steps=POLAR_STEPS, weight_decay=WEIGHT_DECAY))""")

# ---------------------------------------------------------------- prove which schedule is live
sub("startup print", '''    print0(f"TOK_THREADS={TOK_THREADS} per rank ({TOK_THREADS * world_size} tokenizer threads over {os.cpu_count()} vCPUs)")''',
    '''    print0(f"TOK_THREADS={TOK_THREADS} per rank ({TOK_THREADS * world_size} tokenizer threads over {os.cpu_count()} vCPUs)")
    print0(f"POLAR: {len(POLAR_EXPRESS_COEFFS)}-map schedule, {POLAR_STEPS} iterations, prescale {POLAR_PRESCALE} "
           f"(sigma_max {1.0 / POLAR_PRESCALE:.6f}; the 6-map diverges above 0.952450)")''')

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:20s} x{v}")
