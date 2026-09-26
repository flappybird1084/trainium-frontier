#!/usr/bin/env python3
"""Generate v59 = v58 + (dev-only) save the RAW and AVERAGED weights separately, so the tail-EMA blend
coefficients can be swept EXACTLY and FOR FREE offline, with zero gate noise.

## Why this instrument is worth building

v57 added a second tail-EMA group for the four token/value tables and it is the only lever from the ANVIL record
that transferred here (-0.00035, confirmed twice). But **I shipped all three of its constants at the record's
values without tuning any of them for our step count**: `EMA_BLEND = 0.35` (inherited, and swept long ago),
`EMA_TABLE_BLEND = 0.65` (theirs, never swept here at all), and the decay/window pair.

Two of those are now settled and need no further runs:

- **The decay and window are already right.** What matters is not window length but **window / decay horizon**.
  Ours at 1800 s is (0.15 x 4233) / (1/(1-0.997)) = 635 / 333 = **1.9**; the record's is 298 / 149.5 = **2.0**.
  We already sit at their ratio, which also explains why twelve earlier EMA rows in the ledger all found a
  plateau. Moving the decay to the "rescaled" 0.998113 would take the ratio to 1.27, i.e. away from it.
- **A 900 s screen CANNOT tune any of this**, which I learned by wasting a sweep on it. The decay is measured in
  STEPS while the window is a FRACTION of the run, so halving the budget changes their ratio: at 900 s the window
  is ~307 steps against a 333-step horizon, so the average is undertrained and blending toward it HURTS. Measured:
  `E_REF` 1.006719 @2049 vs `E_TAB65` 1.007139 @2040 - **the opposite sign to the 1800 s gates**. Any future
  EMA/averaging experiment must be run at the full budget.

That leaves the two **blend coefficients**, which are pure post-hoc mixing applied once at the end:

    p_final = (1 - b) * p_raw + b * p_avg

So if a run saves `p_raw` and `p_avg` separately, **every** value of `b` is reconstructible afterwards without
retraining. A 2-D grid over (EMA_BLEND, EMA_TABLE_BLEND) then costs one training run plus a handful of forward
passes, and - the real point - it is **deterministic**. Every hyperparameter sweep this project has ever run was
limited by the ~0.00025 per-gate noise floor; an offline sweep on one fixed checkpoint has **no noise at all**,
so it can resolve differences 5x smaller than a gate can and find the true optimum rather than a plateau.

## The change (dev-only, default OFF)

`SAVE_EMA_PARTS=1` stores two extra name-keyed dicts in the checkpoint - the pre-blend raw weights and the EMA
averages, for both groups - alongside the normal (already blended) `model_state`. To do that the two groups are
built from `named_parameters()` instead of `parameters()`, which is behaviour-identical: the selection predicate
and the iteration order are unchanged, so the EMA math and the shipped result are bit-identical whether the knob
is on or off. With the knob off, nothing extra is stored and v59 == v58 exactly.

Checkpoint growth with the knob on: the blended groups are ~71.3M parameters (54.5M matrices/head/scalars plus
the four 4.19M tables), so 2 x 285 MB on top of the usual 822 MB. Fine for a dev run; never shipped on.

Run from the repo root:  .venv/bin/python submission_raw/make_v59.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v58_clean_bosignore.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v59_clean_emaparts.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------- the dev knob
sub("knob", """EMA_TABLE_BLEND = 0.65          # this group is blended harder than the 0.35 used for everything else""",
    """EMA_TABLE_BLEND = 0.65          # this group is blended harder than the 0.35 used for everything else
# SAVE_EMA_PARTS (DEV ONLY, default off): also store the pre-blend raw weights and the EMA averages, name-keyed,
# so the two blend coefficients can be swept exactly offline (p_final = (1-b)*raw + b*avg) with zero gate noise.
# Never enabled on a scored run: it adds ~570 MB to the checkpoint and earns nothing at scoring time.
SAVE_EMA_PARTS = os.environ.get("NEURON_COMPETITION_R1_SAVE_EMA_PARTS", "0") != "0\"""")

# ---------------------------------------------------------------- name the two groups
sub("groups", """                if ema_params is None:
                    tables = {id(m.weight) for m in orig_model.transformer.values() if isinstance(m, nn.Embedding)}
                    ema_pairs = [p for p in orig_model.parameters() if id(p) not in tables]""",
    """                if ema_params is None:
                    tables = {id(m.weight) for m in orig_model.transformer.values() if isinstance(m, nn.Embedding)}
                    # named_parameters() here rather than parameters(): same predicate, same order, so the EMA
                    # math is untouched - the names exist only so SAVE_EMA_PARTS can key its dicts.
                    ema_names = [n for n, p in orig_model.named_parameters() if id(p) not in tables]
                    ema_pairs = [p for _n, p in orig_model.named_parameters() if id(p) not in tables]""")

# ---------------------------------------------------------------- name the table group
sub("table names", """                    _tabs = [(n, m.weight) for n, m in orig_model.transformer.items()
                             if isinstance(m, nn.Embedding) and m.weight.numel() <= EMA_TABLE_MAX]""",
    """                    _tabs = [(n, m.weight) for n, m in orig_model.transformer.items()
                             if isinstance(m, nn.Embedding) and m.weight.numel() <= EMA_TABLE_MAX]
                    ema_tab_names = [f"transformer.{n}.weight" for n, _p in _tabs]""")

# ---------------------------------------------------------------- capture raw + avg just before blending
sub("capture", """    if ema_params is not None:
        with torch.no_grad():
            for e, p in zip(ema_params, ema_pairs):
                p.lerp_(e, EMA_BLEND)""",
    """    ema_parts = None
    if ema_params is not None and SAVE_EMA_PARTS:
        with torch.no_grad():   # BEFORE the blend: everything needed to reconstruct any coefficient offline
            ema_parts = {
                "raw": {n: p.detach().cpu().clone() for n, p in zip(ema_names, ema_pairs)},
                "avg": {n: e.detach().cpu().clone() for n, e in zip(ema_names, ema_params)},
                "tab_raw": {n: p.detach().cpu().clone() for n, p in zip(ema_tab_names, ema_tab_pairs)},
                "tab_avg": {n: e.detach().cpu().clone() for n, e in zip(ema_tab_names, ema_tab_params)},
                "blend": EMA_BLEND, "table_blend": EMA_TABLE_BLEND,
            }
            print0(f"SAVE_EMA_PARTS: stored {len(ema_parts['raw'])} raw/avg pairs "
                   f"+ {len(ema_parts['tab_raw'])} table pairs for an offline blend sweep")
    if ema_params is not None:
        with torch.no_grad():
            for e, p in zip(ema_params, ema_pairs):
                p.lerp_(e, EMA_BLEND)""")

# ---------------------------------------------------------------- put them in the checkpoint
sub("save", """        save_checkpoint(checkpoint_path, orig_model, {
            "step": step,""",
    """        if ema_parts is not None:
            _SAVE_EXTRA.update(ema_parts)
        save_checkpoint(checkpoint_path, orig_model, {
            "step": step,""")

sub("save_checkpoint", """def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model.config),
        "meta": meta,
    }
    torch.save(payload, path)""",
    """_SAVE_EXTRA: dict = {}        # dev-only side channel for SAVE_EMA_PARTS; empty on a scored run


def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model.config),
        "meta": meta,
    }
    if _SAVE_EXTRA:
        payload["ema_parts"] = _SAVE_EXTRA
    torch.save(payload, path)""")

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:16s} x{v}")
