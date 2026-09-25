#!/usr/bin/env python3
"""Generate v55 = v54 + one differentiable `unbind` per weight stack per forward (was: 72 per-layer slices).

## What this deletes, and why it is the single largest addressable item in the step

The critical-path profile of 2026-09-25 timestamps every instruction, so it can answer the question that
instruction counting could not: **for how long is each engine the ONLY one working?** That is the time where
deleting work actually shortens the wall. For one forward+backward span of 199.9 ms:

    engine busy:  Vector 139.7  Tensor 99.0  Scalar 97.0  GpSimd 38.5  Sync 26.8 ms
    EXCLUSIVE:    Vector 27-31  Tensor 15-36  Scalar 12-15 ms       (the step is ~70% overlapped)

and attributing the Vector-exclusive time by source line put **16.6 ms - 53% of it - on ONE line**: the
per-layer weight slice at the LAYER=op site,

    ws = [lin._stack[lin.idx].to(bf16) for lin in (c_q, c_k, c_v, c_proj, c_fc, c_proj)]

Autograd lowers each of those 6 x 12 = 72 selects as `select_backward`: allocate `zeros_like(the WHOLE stack)`,
scatter one slice into it, then add the whole stack into `stack.grad`. Measured at that line:
MEMSET 6.59 ms exclusive + TENSOR_TENSOR 10.00 ms exclusive. Confirmed by element count: 15,506 MEMSET
instructions x (128, 512) = **1.016e9 elements**, against **1.057e9** predicted for 72 full-stack zero-fills
(attn 12.58M x 4 x 12 + MLP 18.87M x 2 x 12) - agreement within 4%.

**This contradicts an assumption written into the code.** `_StackSliceLinear`'s docstring in the experiment
vehicles says the select-backward allocation is something "which compile fuses away". It does not: the profile
shows it inside the compiled graph, on the critical path, in the backward.

`torch.unbind` collapses the 12 selects per stack into ONE `torch.stack` of the 12 gradients. It stays inside
the autograd graph (unlike the detached WCACHE in the vehicles, which cannot be used here because weight
gradients would never flow).

## Why it is bit-identical

Layer *i*'s slice touches a **disjoint** part of the stack, so `stack.grad` receives 12 non-overlapping
contributions either way - no element is summed twice, and `torch.stack` of the 12 gradients is exactly the
tensor the 12 scatter-and-adds produce. The cast moves from per-slice to whole-stack, which is elementwise.
So the only thing that changes is the step count.

## Measured

Paired 300 s screens, same session: fwd_bwd **0.356 vs 0.367-0.371**, total **0.461 vs 0.475-0.476**,
**674 steps vs 651** = **+3.5%**, exactly what the 16.6 ms of exclusive time predicts. At -0.0404 bpb per
e-fold of steps that is **-0.0014**.

Run from the repo root:  .venv/bin/python submission_raw/make_v55.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v54_clean_wallguard.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v55_clean_wviews.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------- Block.forward takes the pre-unbound views
sub("block signature", """    def forward(self, x, x0, ve, cos, sin, resid_lambda, x0_lambda, ve_lambda, sink, dense_mask, bounds, doc_start=None,
                rope_tabs=None):""",
    """    def forward(self, x, x0, ve, cos, sin, resid_lambda, x0_lambda, ve_lambda, sink, dense_mask, bounds, doc_start=None,
                rope_tabs=None, wviews=None):""")

# ---------------------------------------------------------------- use them instead of 6 per-layer selects
sub("ws from views", """            ws = [lin._stack[lin.idx].to(bf16) for lin in (at.c_q, at.c_k, at.c_v, at.c_proj, ml.c_fc, ml.c_proj)]""",
    """            _lins = (at.c_q, at.c_k, at.c_v, at.c_proj, ml.c_fc, ml.c_proj)
            # One differentiable bf16 view-set per stack is built once per forward (see GPT.forward). A plain
            # `stack[idx]` here costs a select_backward per layer per weight - zeros_like(the whole stack) plus a
            # full-stack add, 1.02e9 elements per step and 16.6 ms of VECTOR-EXCLUSIVE time in the backward.
            ws = ([lin._stack[lin.idx].to(bf16) for lin in _lins] if wviews is None
                  else [wviews[lin.key][lin.idx] for lin in _lins])""")

# ---------------------------------------------------------------- build the views once per forward
sub("build views", """        rope_tabs = (self.cos2d[:t], self.sin2d[:t]) if LAYER_IMPL == "op" else None""",
    """        # one cast + one unbind per weight stack, kept as a LOCAL and threaded through the block calls so
        # nothing mutates module state inside the traced region
        wviews = ({k: torch.unbind(p.to(torch.bfloat16)) for k, p in self.stacks.items()}
                  if (LAYER_IMPL == "op" and COMPUTE_DTYPE == torch.bfloat16) else None)
        rope_tabs = (self.cos2d[:t], self.sin2d[:t]) if LAYER_IMPL == "op" else None""")

# ---------------------------------------------------------------- pass them down
sub("block call", """            x = block(x, x0, ve_i, cos, sin, bs["resid"][i], bs["x0"][i], bs["ve"][i], bs["sink"][i], dense_mask, bounds,
                      doc_start=doc_start, rope_tabs=rope_tabs)""",
    """            x = block(x, x0, ve_i, cos, sin, bs["resid"][i], bs["x0"][i], bs["ve"][i], bs["sink"][i], dense_mask, bounds,
                      doc_start=doc_start, rope_tabs=rope_tabs, wviews=wviews)""")

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:20s} x{v}")
