#!/usr/bin/env python3
"""Generate v66 = v62 + TWO parallel EMA tracks + SAVE_EMA_PARTS, so the decay AND both blends can be swept
exactly and offline at the CURRENT window. Dev instrument; knob default off.

## Why: the window plateau was me misreading my own mechanism

After v62 (window 0.85 -> 0.75, -0.00044) and v64 (0.75 -> 0.65, a wash) I concluded the averaging family was
closed, reasoning that "at 3.1 horizons the average is already converged, so the decay does not need retuning".
**That is backwards.** Converged means the WINDOW stopped being the binding constraint - so the decay horizon
tau is now the effective averaging length, and raising tau is the only remaining way to make the average longer.
Our 1060-step window is decoration around a 333-step exponential average.

The ledger's decay measurements never covered where we now sit:

    rows 421-431 (2026-09-15/16)  3400 steps, EMA_START 0.85  -> window 510   tau 333 = 1.53 horizons
                                                                 tau 667 = 0.76 horizons   (0.997 ~ 0.9985, tied)
    today (v62)                   4238 steps, EMA_START 0.75  -> window 1060  tau 333 = 3.18 horizons  NEVER MEASURED

The entire measured plateau lies between 0.76 and 1.53 horizons; we operate at 3.18, off the measured range on
the short-tau side, because throughput grew AND v62 doubled the window. Restoring ~1.5 horizons at a 1060-step
window wants **tau ~ 700, i.e. EMA_DECAY ~ 0.9986** - an effective average about 2x longer than anything this
project has measured.

Raising tau lengthens the effective average at the cost of convergence: at W=1060, tau=333 gives ~333 fully
converged, tau=700 gives 700*(1-e^-1.51) ~ 545, tau=1060 gives ~670. There is an interior optimum and it has
never been located.

## And a second thing the audit caught

**Every offline sweep so far used the v59 checkpoint, whose `EMA_START` was 0.85.** The 0.65 blend grid, the
per-group coordinate descent that returned exactly zero, and the decontraction test were all computed at the OLD
window. v62 moved it to 0.75, so the blend optimum may have moved again - the same window<->blend coupling that
produced v60 and v62, one crank later, and free to re-check.

## The change

1. `EMA_DECAY2` (default 0.9986) maintains a SECOND set of EMA buffers in parallel with the existing one, for
   both groups. Cost: two extra `lerp_` passes over 71.3M elements per step, ~0.07 ms of a 424 ms step (<0.05%),
   and +285 MB of device memory. v61 already proved a 537 MB addition fits.
2. `SAVE_EMA_PARTS` (ported from v59, default off) stores the pre-blend raw weights and BOTH averages,
   name-keyed, so `experimental/blend_sweep.py` / `blend_groups.py` can sweep blend x table-blend x tau jointly,
   exactly, offline, with zero gate noise.

With both knobs off this is byte-for-byte v62 behaviour: the second track is only allocated when
`EMA_DECAY2 > 0`, and nothing extra is saved. The run therefore doubles as a normal control.

Run from the repo root:  .venv/bin/python submission_raw/make_v66.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v62_clean_start075.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v66_clean_twotrack.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------- knobs
sub("knobs", """EMA_TABLE_MAX = 34_000_000      # numel ceiling: admits the 4.19M tables, excludes the 134.2M bigram table""",
    """EMA_TABLE_MAX = 34_000_000      # numel ceiling: admits the 4.19M tables, excludes the 134.2M bigram table
# EMA_DECAY2 (DEV): a second EMA track at a longer horizon, maintained in parallel so tau can be swept OFFLINE.
# tau = 1/(1-decay): 0.997 -> 333 steps, 0.9986 -> 714. At the v62 window of ~1060 steps the first is 3.18
# horizons (fully converged, so the window is no longer binding and tau IS the effective average length) and the
# second is ~1.5 horizons, which is where every decay measurement in the ledger was actually taken.
EMA_DECAY2 = float(os.environ.get("NEURON_COMPETITION_R1_EMA_DECAY2", "0"))
# SAVE_EMA_PARTS (DEV): store pre-blend raw weights and every average, name-keyed, for an exact offline sweep.
SAVE_EMA_PARTS = os.environ.get("NEURON_COMPETITION_R1_SAVE_EMA_PARTS", "0") != "0\"""")

# ---------------------------------------------------------------- state init
sub("state init", """    ema_params = None
    ema_pairs = None
    ema_tab_params = None
    ema_tab_pairs = None""",
    """    ema_params = None
    ema_pairs = None
    ema_tab_params = None
    ema_tab_pairs = None
    ema_params2 = None
    ema_tab_params2 = None
    ema_names = None
    ema_tab_names = None""")

# ---------------------------------------------------------------- name the groups + allocate track 2
sub("groups", """                    ema_pairs = [p for p in orig_model.parameters() if id(p) not in tables]""",
    """                    # named_parameters(): same predicate and order, so the EMA math is unchanged; the names
                    # exist only so SAVE_EMA_PARTS can key its dicts.
                    ema_names = [n for n, p in orig_model.named_parameters() if id(p) not in tables]
                    ema_pairs = [p for _n, p in orig_model.named_parameters() if id(p) not in tables]""")

sub("alloc2", """                    ema_tab_params = [p.detach().clone() for p in ema_tab_pairs]
                    print0(f"tail-EMA of the weights started at step {step}")""",
    """                    ema_tab_params = [p.detach().clone() for p in ema_tab_pairs]
                    ema_tab_names = [f"transformer.{n}.weight" for n, _p in _tabs]
                    if EMA_DECAY2 > 0:
                        ema_params2 = [p.detach().clone() for p in ema_pairs]
                        ema_tab_params2 = [p.detach().clone() for p in ema_tab_pairs]
                        print0(f"second EMA track at decay {EMA_DECAY2} (tau {1.0 / (1.0 - EMA_DECAY2):.0f} steps) "
                               f"alongside {EMA_DECAY} (tau {1.0 / (1.0 - EMA_DECAY):.0f})")
                    print0(f"tail-EMA of the weights started at step {step}")""")

# ---------------------------------------------------------------- accumulate track 2
sub("accumulate2", """                    for e, p in zip(ema_tab_params, ema_tab_pairs):
                        e.lerp_(p.detach(), 1.0 - EMA_DECAY)""",
    """                    for e, p in zip(ema_tab_params, ema_tab_pairs):
                        e.lerp_(p.detach(), 1.0 - EMA_DECAY)
                    if ema_params2 is not None:
                        for e, p in zip(ema_params2, ema_pairs):
                            e.lerp_(p.detach(), 1.0 - EMA_DECAY2)
                        for e, p in zip(ema_tab_params2, ema_tab_pairs):
                            e.lerp_(p.detach(), 1.0 - EMA_DECAY2)""")

# ---------------------------------------------------------------- capture parts before the blend
sub("capture", """    if ema_params is not None:
        with torch.no_grad():
            for e, p in zip(ema_params, ema_pairs):
                p.lerp_(e, EMA_BLEND)""",
    """    if ema_params is not None and SAVE_EMA_PARTS:
        with torch.no_grad():   # BEFORE the blend: everything needed to reconstruct any (blend, tau) offline
            _parts = {
                "raw": {n: p.detach().cpu().clone() for n, p in zip(ema_names, ema_pairs)},
                "avg": {n: e.detach().cpu().clone() for n, e in zip(ema_names, ema_params)},
                "tab_raw": {n: p.detach().cpu().clone() for n, p in zip(ema_tab_names, ema_tab_pairs)},
                "tab_avg": {n: e.detach().cpu().clone() for n, e in zip(ema_tab_names, ema_tab_params)},
                "blend": EMA_BLEND, "table_blend": EMA_TABLE_BLEND,
                "decay": EMA_DECAY, "ema_start": EMA_START,
            }
            if ema_params2 is not None:
                _parts["avg2"] = {n: e.detach().cpu().clone() for n, e in zip(ema_names, ema_params2)}
                _parts["tab_avg2"] = {n: e.detach().cpu().clone() for n, e in zip(ema_tab_names, ema_tab_params2)}
                _parts["decay2"] = EMA_DECAY2
            _SAVE_EXTRA.update(_parts)
            print0(f"SAVE_EMA_PARTS: stored raw + {'two' if ema_params2 is not None else 'one'} average track(s) "
                   f"for {len(ema_names)} tensors + {len(ema_tab_names)} tables")
    if ema_params is not None:
        with torch.no_grad():
            for e, p in zip(ema_params, ema_pairs):
                p.lerp_(e, EMA_BLEND)""")

# ---------------------------------------------------------------- side channel in the checkpoint
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
