#!/usr/bin/env python3
"""Generate train3_s20.py = train3_s19.py + a SECOND tail-EMA group for the small token/value tables.

## What this is, and why it is the largest untested quality lever on the board

Our tail EMA excludes **every** `nn.Embedding` table (`EMA_SKIP_EMBED=1`, which the submission wires on).
That is not a small exclusion: the five tables are ~69% of all parameters, and four of them are the
token/value embeddings that run at **4.5x the base learning rate** (`EMBED_LR_SCALE=4.5`). The parameters
with the largest per-step updates — the ones a tail average helps most — are the ones we never average.

There is **no ledger row testing the alternative**: `EMA_SKIP_EMBED=0` would sweep the 134M-element bigram
table into the EMA too (a 537 MB fp32 clone plus a 134M-element lerp every step), which is presumably why it
was never tried. So the axis was closed by an implementation detail, not by a measurement.

The 2026-08-30 modded-nanogpt record (PR #360) treats terminal averaging as a **per-group** schedule, with
the embedding/head group averaged at a *stronger* blend than the matrix banks, and rates the family the
single largest quality item in its stack. This change makes that expressible here.

## The change

`EMA_TABLES=1` puts the embedding tables **whose numel is <= EMA_TABLE_MAX** (default 3.4e7, which admits
the four 8192x512 = 4.2M token/value tables and excludes the 2^18 x 512 = 134M bigram table) into a SECOND
EMA group with its own blend `EMA_TABLE_BLEND` (default 0.65), leaving the existing group and its
`EMA_BLEND` untouched. Cost: 4 x 4.2M = 16.8M extra fp32 of state (+67 MB) and four extra lerps per step
(~0.07 ms), i.e. throughput-neutral.

The startup line prints the name and size of every table actually admitted, so the log proves which
implementation is live (this project has twice built on a knob that reached a dead code path).

Run from the repo root:  .venv/bin/python submission_raw/make_s20.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "submission_raw/train3_s19.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "submission_raw/train3_s20.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------------------- the knobs
sub("knobs", '''EMA_SKIP_EMBED = os.environ.get("NEURON_COMPETITION_R1_EMA_SKIP_EMBED", "0") != "0"   # EMA everything except the token tables''',
    '''EMA_SKIP_EMBED = os.environ.get("NEURON_COMPETITION_R1_EMA_SKIP_EMBED", "0") != "0"   # EMA everything except the token tables
# EMA_TABLES: with EMA_SKIP_EMBED, put the SMALL embedding tables back into a second tail-EMA group with its own
# blend. The four token/value tables are 8192x512 = 4.2M each and run at EMBED_LR_SCALE (x4.5) - the largest
# per-step updates in the model, and until now the only parameters never averaged. The 2^18 x 512 = 134M bigram
# table stays out: EMA_TABLE_MAX admits the 4.2M tables and excludes it (state cost 67 MB, not 537 MB).
EMA_TABLES = os.environ.get("NEURON_COMPETITION_R1_EMA_TABLES", "0") != "0"
EMA_TABLE_BLEND = float(os.environ.get("NEURON_COMPETITION_R1_EMA_TABLE_BLEND", "0.65"))   # final = (1-b)*w + b*ema for that group
EMA_TABLE_MAX = int(float(os.environ.get("NEURON_COMPETITION_R1_EMA_TABLE_MAX", "3.4e7")))  # numel ceiling for admission''')

# ---------------------------------------------------------------------------- state init
sub("state init", """    ema_params = None
    nes_state = None; nes_pairs = None; nes_snap = None""",
    """    ema_params = None
    ema_tab_params = None; ema_tab_pairs = None
    nes_state = None; nes_pairs = None; nes_snap = None""")

# ---------------------------------------------------------------------------- accumulate the second group
sub("accumulate", """                if ema_params is None:
                    skip = set()
                    if EMA_SKIP_EMBED:
                        skip = {id(m.weight) for n, m in orig_model.transformer.items() if isinstance(m, nn.Embedding)}
                    ema_pairs = [p for p in orig_model.parameters() if id(p) not in skip]
                    ema_params = [p.detach().clone() for p in ema_pairs]
                    print0(f"EMA of the weights started at step {step} (decay {EMA_DECAY}, blend {EMA_BLEND}, {len(ema_params)} tensors)")
                else:
                    for e, p in zip(ema_params, ema_pairs):
                        e.lerp_(p.detach(), 1.0 - EMA_DECAY)""",
    """                if ema_params is None:
                    skip = set()
                    if EMA_SKIP_EMBED:
                        skip = {id(m.weight) for n, m in orig_model.transformer.items() if isinstance(m, nn.Embedding)}
                    ema_pairs = [p for p in orig_model.parameters() if id(p) not in skip]
                    ema_params = [p.detach().clone() for p in ema_pairs]
                    print0(f"EMA of the weights started at step {step} (decay {EMA_DECAY}, blend {EMA_BLEND}, {len(ema_params)} tensors)")
                    if EMA_TABLES and EMA_SKIP_EMBED:
                        _tabs = [(n, m.weight) for n, m in orig_model.transformer.items()
                                 if isinstance(m, nn.Embedding) and m.weight.numel() <= EMA_TABLE_MAX]
                        ema_tab_pairs = [p for _n, p in _tabs]
                        ema_tab_params = [p.detach().clone() for p in ema_tab_pairs]
                        # print what was ADMITTED and what was rejected: the log must prove which path is live
                        print0("EMA_TABLES: second group blend " + f"{EMA_TABLE_BLEND} over "
                               + ", ".join(f"{n}({p.numel():,})" for n, p in _tabs)
                               + " | excluded: " + ", ".join(
                                   f"{n}({m.weight.numel():,})" for n, m in orig_model.transformer.items()
                                   if isinstance(m, nn.Embedding) and m.weight.numel() > EMA_TABLE_MAX))
                else:
                    for e, p in zip(ema_params, ema_pairs):
                        e.lerp_(p.detach(), 1.0 - EMA_DECAY)
                    if ema_tab_params is not None:
                        for e, p in zip(ema_tab_params, ema_tab_pairs):
                            e.lerp_(p.detach(), 1.0 - EMA_DECAY)""")

# ---------------------------------------------------------------------------- blend the second group in
sub("blend", """    if EMA_DECAY > 0 and ema_params is not None:
        with torch.no_grad():
            for e, p in zip(ema_params, ema_pairs):
                p.lerp_(e, EMA_BLEND)
        print0("EMA weights blended in for the checkpoint and the evaluation")""",
    """    if EMA_DECAY > 0 and ema_params is not None:
        with torch.no_grad():
            for e, p in zip(ema_params, ema_pairs):
                p.lerp_(e, EMA_BLEND)
            if ema_tab_params is not None:
                for e, p in zip(ema_tab_params, ema_tab_pairs):
                    p.lerp_(e, EMA_TABLE_BLEND)
        print0("EMA weights blended in for the checkpoint and the evaluation"
               + (f" (+{len(ema_tab_params)} tables at blend {EMA_TABLE_BLEND})" if ema_tab_params is not None else ""))""")

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:14s} x{v}")
