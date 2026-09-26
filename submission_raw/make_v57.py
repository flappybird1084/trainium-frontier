#!/usr/bin/env python3
"""Generate v57 = v56 + a second tail-EMA group over the four small token/value tables. No kernel change.

## What this is

v56 and every version before it exclude **every** `nn.Embedding` table from the tail EMA:

    tables = {id(m.weight) for m in orig_model.transformer.values() if isinstance(m, nn.Embedding)}
    ema_pairs = [p for p in orig_model.parameters() if id(p) not in tables]

That drops ~69% of the parameters, including the four token/value tables (`wte`, `vte`, `vte_mid`, `vte2`,
8192x512 = 4.19M each) which train at `EMBED_LR_SCALE = 4.5` - the largest per-step updates in the model, and
exactly the parameters a terminal average should help most. There was **no ledger row testing the alternative**,
because flipping the existing switch would also sweep in the 2^18 x 512 = 134.2M bigram table (a 537 MB fp32
clone plus a 134M-element lerp every step).

This adds a **second** EMA group holding only the embedding tables under a numel ceiling - the ceiling admits
the four 4.19M tables and excludes the 134.2M bigram table - with its own blend. State cost 67 MB, four extra
`lerp_` calls per step (~0.07 ms of a 424 ms step), so it is throughput-neutral.

## Measured, and honestly small

Gated as a matched same-file same-session pair in the vehicle (`train3_s20.py`, `G_S20REF` vs `G_S20TAB`):

    control 0.971569 @ 4123 steps   |   arm 0.971437 @ 4101 steps
    step-adjusted effect  -0.000289   (~1.2 sd against a per-gate noise of ~0.00025)

So this is a **positive but not firmly established** result, at roughly one sixth of the 20.2-millinat estimate
that the 2026-08-30 modded-nanogpt record reports for its (much more elaborate, four-ship) averaging family. It
is included because it is free, directionally right, and the only lever of four tested overnight that did not
lose: the published 6-map orthogonalization cascade came in +0.000622 (2.5 sd worse) and the published momentum
schedule +0.000675 (2.7 sd worse), both rejected.

`EMA_TABLE_BLEND` is 0.65 (the record's value for its embedding/head group) against the existing 0.35 for
everything else - the two groups are deliberately blended differently, which is the whole point of splitting
them. The startup line prints which tables were admitted and which were excluded, so the log proves the path is
live rather than silently dead.

Run from the repo root:  .venv/bin/python submission_raw/make_v57.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v56_clean_noselfcheck.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v57_clean_ematables.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------- the constants
sub("consts", """EMA_BLEND = 0.35                # ... blended 0.65 w + 0.35 ema before the checkpoint; token tables excluded""",
    """EMA_BLEND = 0.35                # ... blended 0.65 w + 0.35 ema before the checkpoint; the BIGRAM table excluded
# Second tail-EMA group, for the SMALL embedding tables only. Every version through v56 excluded all five
# nn.Embedding tables from the EMA, i.e. ~69% of parameters including the four 4.19M token/value tables that run
# at EMBED_LR_SCALE (x4.5) - the largest per-step updates in the model. The numel ceiling admits those four and
# excludes the 2^18 x 512 = 134.2M bigram table, so the state cost is 67 MB rather than 537 MB and the per-step
# cost is four lerps (~0.07 ms). Gated at -0.000289 step-adjusted (~1.2 sd): small, positive, not firmly
# established - and the only one of four levers tested on 2026-09-26 that did not lose.
EMA_TABLE_BLEND = 0.65          # this group is blended harder than the 0.35 used for everything else
EMA_TABLE_MAX = 34_000_000      # numel ceiling: admits the 4.19M tables, excludes the 134.2M bigram table""")

# ---------------------------------------------------------------- state init
sub("state init", """    ema_params = None
    ema_pairs = None""",
    """    ema_params = None
    ema_pairs = None
    ema_tab_params = None
    ema_tab_pairs = None""")

# ---------------------------------------------------------------- accumulate
sub("accumulate", """                if ema_params is None:
                    tables = {id(m.weight) for m in orig_model.transformer.values() if isinstance(m, nn.Embedding)}
                    ema_pairs = [p for p in orig_model.parameters() if id(p) not in tables]
                    ema_params = [p.detach().clone() for p in ema_pairs]
                    print0(f"tail-EMA of the weights started at step {step}")
                else:
                    for e, p in zip(ema_params, ema_pairs):
                        e.lerp_(p.detach(), 1.0 - EMA_DECAY)""",
    """                if ema_params is None:
                    tables = {id(m.weight) for m in orig_model.transformer.values() if isinstance(m, nn.Embedding)}
                    ema_pairs = [p for p in orig_model.parameters() if id(p) not in tables]
                    ema_params = [p.detach().clone() for p in ema_pairs]
                    _tabs = [(n, m.weight) for n, m in orig_model.transformer.items()
                             if isinstance(m, nn.Embedding) and m.weight.numel() <= EMA_TABLE_MAX]
                    ema_tab_pairs = [p for _n, p in _tabs]
                    ema_tab_params = [p.detach().clone() for p in ema_tab_pairs]
                    print0(f"tail-EMA of the weights started at step {step}")
                    # print what was admitted AND what was excluded: the log must prove which path is live
                    print0("tail-EMA table group (blend " + f"{EMA_TABLE_BLEND}): "
                           + ", ".join(f"{n}({p.numel():,})" for n, p in _tabs)
                           + " | excluded: " + ", ".join(
                               f"{n}({m.weight.numel():,})" for n, m in orig_model.transformer.items()
                               if isinstance(m, nn.Embedding) and m.weight.numel() > EMA_TABLE_MAX))
                else:
                    for e, p in zip(ema_params, ema_pairs):
                        e.lerp_(p.detach(), 1.0 - EMA_DECAY)
                    for e, p in zip(ema_tab_params, ema_tab_pairs):
                        e.lerp_(p.detach(), 1.0 - EMA_DECAY)""")

# ---------------------------------------------------------------- blend
sub("blend", """    if ema_params is not None:
        with torch.no_grad():
            for e, p in zip(ema_params, ema_pairs):
                p.lerp_(e, EMA_BLEND)
        print0("tail-EMA blended into the weights for the checkpoint and the evaluation")""",
    """    if ema_params is not None:
        with torch.no_grad():
            for e, p in zip(ema_params, ema_pairs):
                p.lerp_(e, EMA_BLEND)
            for e, p in zip(ema_tab_params, ema_tab_pairs):
                p.lerp_(e, EMA_TABLE_BLEND)
        print0(f"tail-EMA blended into the weights for the checkpoint and the evaluation "
               f"({len(ema_params)} tensors at {EMA_BLEND}, {len(ema_tab_params)} tables at {EMA_TABLE_BLEND})")""")

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:14s} x{v}")
