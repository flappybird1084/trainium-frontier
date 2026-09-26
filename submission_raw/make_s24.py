#!/usr/bin/env python3
"""Generate train3_s24.py = train3_s23.py + `_FMAX` as an env knob (the kernels' free-dimension chunk depth).

## Why this is the one cheap probe of the graphs' dependency-chain slack

Exclusive engine time is a good FINDER and a bad BOUND: an instruction can sit on the critical path while another
engine is busy with unrelated work, and the sweep scores it zero. From the two v55 captures:

    backward: span 199.9 ms, busiest engine (Vector) 139.7 ms  -> 60.2 ms with the bottleneck engine idle
    forward : span 103.0 ms, busiest engine (Scalar)  62.8 ms  -> 40.2 ms likewise

So ~100 ms of the 303 ms of graph time is **dependency-chain slack**, invisible to both the exclusive total
(65 ms) and the idle total (13 ms). `ROPE_WIDE` proved you cannot harvest it by deleting Vector instructions in
overlapped windows (176,640 instructions deleted bit-identically, step moved 0). Nothing has yet tested whether
it can be harvested by giving the compiler's list scheduler **more independent work in flight**.

`_FMAX` is the free-dimension chunk depth of the fused layer kernels (83 use sites). `post_attn_fwd_h` holds
~6.5 MB of resident transposed weights plus ~4.5 MB per chunk against 24 MB of SBUF, and one PSUM bank per
`pps`, so only 2-3 of its chunks can be in flight at once. **Halving `_FMAX` doubles the number of independent
chunks and frees PSUM**, at the cost of a narrower moving free dimension and 2x the instruction count.
`ROPE_WIDE` says wider-and-fewer is worth nothing; this is the untested opposite direction.

**It has never been swept.** The only mention of `_FMAX` in 579 ledger rows is incidental (`S_MUON`, where the
NKI tracer needed it hoisted to module level).

## Constraints on the values

- **512 is the ceiling**, not a tuning choice: PSUM's free dimension is capped at 512 fp32 per bank, and
  `pps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)`. Do not try 1024.
- `T // _FMAX` must be exact (T = 1024) and `_FMAX // _NKI_P` must be exact (_NKI_P = 128), so the legal
  values are **512, 256, 128**.
- Each value is a distinct NEFF (the tile shapes change), so each arm pays a fresh ~500 s compile. Compile is
  excluded from the training budget, so this costs wall time on a screen but not budget.
- It stays a module-level constant, which the NKI tracer requires (closure constants are invisible to it).

Run from the repo root:  .venv/bin/python submission_raw/make_s24.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "submission_raw/train3_s23.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "submission_raw/train3_s24.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


sub("fmax knob", """_NKI_P = 128
_FMAX = 512""",
    """_NKI_P = 128
# _FMAX: the fused kernels' free-dimension chunk depth. 512 is a HARD CEILING (PSUM's free dim is capped at 512
# fp32 per bank and `pps` is a (128, _FMAX) fp32 PSUM tile), and T // _FMAX and _FMAX // _NKI_P must both be
# exact, so the legal values are 512 / 256 / 128. Smaller = more independent chunks in flight and less PSUM
# pressure, at 2x the instruction count - the untested opposite direction to ROPE_WIDE, and the only cheap probe
# of the ~100 ms of dependency-chain slack in the two graphs. Kept module-level: the NKI tracer cannot see
# closure constants (ledger S_MUON).
_FMAX = int(os.environ.get("NEURON_COMPETITION_R1_FMAX", "512"))
assert _FMAX in (128, 256, 512), f"_FMAX={_FMAX} must be 128, 256 or 512"
assert 1024 % _FMAX == 0 and _FMAX % _NKI_P == 0, f"_FMAX={_FMAX} must divide T=1024 and be a multiple of {_NKI_P}\"""")

sub("startup print", '''    print0(f"MOM_SCHED={MOM_SCHED}"''',
    '''    print0(f"_FMAX={_FMAX} (chunks per 1024-token row: {1024 // _FMAX}; PSUM fp32 free-dim cap is 512)")
    print0(f"MOM_SCHED={MOM_SCHED}"''')

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:16s} x{v}")
