#!/usr/bin/env python3
"""Sweep the tail-EMA BLEND (and, if the run kept extra tracks, the DECAY) exactly and offline. Zero gate time.

## Why this is free

The tail EMA is applied ONCE, after the last optimizer step, as a pure linear combination:

    p_final = (1 - b) * p_raw + b * p_avg

Nothing downstream of it trains. So a run that saved p_raw and p_avg SEPARATELY determines val_bpb for every b
exactly - the whole blend axis, per-group blends included, from one checkpoint, on this GPU, for free. That is how
v60 was found (EMA_BLEND 0.35 -> 0.65; the offline grid predicted the box gate to 0.00001). A run that also kept
accumulators at other decays makes the DECAY axis free the same way, because each accumulator is just another
p_avg and a convex mix of two of them interpolates between their horizons.

Produce the inputs with ONE arm of an ordinary gate:

    NEURON_COMPETITION_R1_SAVE_EMA_PARTS=1 [NEURON_COMPETITION_R1_EMA_DECAY_EXTRA=0.997,0.9990]

which writes `ema_parts.pt` beside `final.pt` and leaves `final.pt` byte-identical to a flag-off run.

## Relation to the three scripts this replaces

`blend_sweep.py` (2-D grid), `blend_groups.py` (per-group coordinate descent) and `blend_tracks.py` (decay x
blend) each did one of these against v59/v66, which embedded the parts INSIDE final.pt. This reads either layout,
does all three sweeps, and fixes one thing all three got wrong:

**THE EPS DEFECT.** Training normalises inside the NKI kernels with eps = finfo(bf16).eps = 0.0078125. The torch
`norm()` in the vehicle is `F.rms_norm` with NO eps, so off-Neuron it takes finfo(fp32).eps = 1.19e-7 - 65,000x
apart - and on the box `load_for_eval` BUILDS the layer ops, so the box's blocks use 0.0078125 while this GPU's
did not. Every earlier offline sweep therefore scored a forward the box does not run, which is exactly the caveat
the 2026-09-27 audit flagged against the blend grid that produced v60. `--norm-eps` defaults to 0.0078125 so this
script's forward matches the box's; pass `--norm-eps 0` to reproduce the old (wrong) numbers.

## Absolute numbers are not comparable to the box

The local-GPU eval sits ~0.001 above the box's in-run figure. Only compare WITHIN one invocation of this script -
that comparison is exact, because it is the same tokens, the same code and the same device with only the weights
changing.

## Usage

    # the decisive check: does the shipped blend reproduce final.pt?
    .venv/bin/python experimental/blend_reconstruct.py out/final.pt --verify

    # the blend grid (what settles "is 0.65 right?")
    .venv/bin/python experimental/blend_reconstruct.py out/final.pt --grid

    # per-group blends, coordinate descent
    .venv/bin/python experimental/blend_reconstruct.py out/final.pt --groups

    # decay x blend, needs EMA_DECAY_EXTRA on the run
    .venv/bin/python experimental/blend_reconstruct.py out/final.pt --tracks

Add `--eval-tokens 2097152` to confirm a winner at full eval length before shipping it; the smaller default is
fine for RANKING because every point sees identical tokens.
"""
import argparse
import importlib.util
import itertools
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare  # noqa: E402

# The four averaging groups. Every parameter the EMA touches falls in exactly one; `tables` is the token/value
# embedding group the vehicle already blends with its own coefficient, the other three share one coefficient
# today and are split here because they are different kinds of object (Muon-updated banks, the AdamW head, and
# ~90 scalars) and the reference recipe blends them separately.
GROUPS = ("matrices", "head", "scalars", "tables")


def group_of(name: str) -> str:
    if name.startswith("stacks."):
        return "matrices"
    if name == "lm_head.weight":
        return "head"
    return "scalars"


def load_parts(checkpoint: Path) -> tuple:
    """Parts live in a sibling `ema_parts.pt` (train3_b2 and later) or inside the checkpoint (v59/v66).

    Returns (parts, payload) and loads the checkpoint exactly once: the embedded layout makes the file 1.4-1.7 GB
    and reading it twice is 3 GB of host RAM for nothing.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sibling = checkpoint.parent / "ema_parts.pt"
    if sibling.exists():
        print(f"parts: {sibling}")
        return torch.load(sibling, map_location="cpu", weights_only=True), payload
    parts = payload.get("ema_parts")
    if not parts:
        sys.exit(f"no parts found: neither {sibling} nor an 'ema_parts' key in {checkpoint}.\n"
                 "Re-run one arm with NEURON_COMPETITION_R1_SAVE_EMA_PARTS=1.")
    print(f"parts: embedded in {checkpoint} (v59/v66 layout)")
    # v66 stored a single extra track as avg2/tab_avg2; normalise it onto the list layout used below.
    if "avg2" in parts:
        parts["avg_extra"] = [parts["avg2"]]
        parts["tab_avg_extra"] = [parts["tab_avg2"]]
        parts["extra_decays"] = [parts.get("decay2", float("nan"))]
    return parts, payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="the run's final.pt (ema_parts.pt is picked up from the same directory)")
    ap.add_argument("--train-py", default="submission_raw/train3_b2.py",
                    help="the vehicle whose load_for_eval/GPT built this checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-tokens", type=int, default=524288)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=4, help="eval rows; the forward materializes (rows, T, V) fp32")
    ap.add_argument("--norm-eps", type=float, default=0.0078125,
                    help="RMSNorm eps for the scored forward. Default matches the NKI kernels (and therefore the "
                         "box); 0 leaves the vehicle's F.rms_norm default, which is what the old sweeps used.")
    ap.add_argument("--verify", action="store_true",
                    help="reconstruct at the SHIPPED blend and diff against final.pt's weights (do this first)")
    ap.add_argument("--grid", action="store_true", help="2-D grid over (main blend, table blend)")
    ap.add_argument("--groups", action="store_true", help="coordinate descent over the four per-group blends")
    ap.add_argument("--tracks", action="store_true", help="decay x blend, using the extra accumulators")
    ap.add_argument("--main", default="0.35,0.5,0.65,0.8,0.95")
    ap.add_argument("--tables", default="0.35,0.5,0.65,0.8,0.95")
    ap.add_argument("--rounds", type=int, default=2, help="--groups: coordinate-descent sweeps")
    ap.add_argument("--mix", action="store_true", help="--tracks: also score convex mixes between accumulators")
    args = ap.parse_args()
    if not (args.verify or args.grid or args.groups or args.tracks):
        args.verify = args.grid = True

    ckpt = Path(args.checkpoint)
    parts, payload = load_parts(ckpt)
    base = payload["model_state"]
    extra_decays = list(parts.get("extra_decays") or [])
    print(f"as shipped: blend {parts['blend']} / tables {parts['table_blend']}, decay {parts.get('decay')} "
          f"(tau {1.0 / max(1e-9, 1.0 - float(parts.get('decay', 0.997))):.0f} steps), "
          f"EMA_START {parts.get('ema_start')}, B2 {parts.get('b2')}, steps {parts.get('steps')}")
    if extra_decays:
        print(f"extra tracks: {extra_decays} (tau "
              + ", ".join(f"{1.0 / max(1e-9, 1.0 - d):.0f}" for d in extra_decays) + " steps)")

    spec = importlib.util.spec_from_file_location("veh", args.train_py)
    veh = importlib.util.module_from_spec(spec)
    sys.modules["veh"] = veh
    spec.loader.exec_module(veh)

    if args.norm_eps > 0:
        # Pin every RMSNorm in the scored forward to the kernels' eps, so this GPU runs the box's function of the
        # weights. Patch the module attribute: head_logits/the blocks/QK norm all call it by name.
        _eps = args.norm_eps
        veh.norm = lambda x: F.rms_norm(x, (x.size(-1),), eps=_eps)
        print(f"norm eps pinned to {_eps} (matches the NKI kernels and the box)")
    else:
        print("norm eps left at F.rms_norm's default (1.19e-7 for fp32) - reproduces the OLD sweeps, not the box")

    device = torch.device(args.device)
    tokenizer = prepare.ensure_tokenizer()
    # Build the model through the official contract, then re-load weights per point. load_for_eval also sets the
    # vehicle's COMPUTE_DTYPE for this device, which a bare GPT(config) would not.
    model = veh.load_for_eval(str(ckpt), device)
    shipped_state = payload["model_state"] if args.verify else None

    # name -> (raw, group) and the list of average dicts per track (track 0 = the shipped decay)
    entries = [(n, parts["raw"][n], group_of(n)) for n in parts["raw"]]
    entries += [(n, parts["tab_raw"][n], "tables") for n in parts["tab_raw"]]
    avgs = [dict(parts["avg"], **parts["tab_avg"])]
    for a, ta in zip(parts.get("avg_extra", []), parts.get("tab_avg_extra", [])):
        avgs.append(dict(a, **ta))
    for g in GROUPS:
        sel = [e for e in entries if e[2] == g]
        print(f"  group {g:<9} {len(sel):2d} tensors  {sum(r.numel() for _n, r, _g in sel):,} elements")

    def reconstruct(blends: dict, w_extra=None) -> dict:
        """state_dict for per-group blends. w_extra weights the extra tracks; track 0 takes the remainder."""
        state = dict(base)
        for n, raw, g in entries:
            avg = avgs[0][n].float()
            if w_extra:
                avg = avg * (1.0 - sum(w_extra))
                for w, a in zip(w_extra, avgs[1:]):
                    avg = avg + w * a[n].float()
            state[n] = torch.lerp(raw.float(), avg, blends[g]).to(raw.dtype)
        return state

    cache = {}

    def score(blends: dict, w_extra=None) -> float:
        key = (tuple(blends[g] for g in GROUPS), tuple(w_extra or ()))
        if key in cache:
            return cache[key]
        model.load_state_dict(reconstruct(blends, w_extra), strict=True)
        with torch.no_grad():
            v = prepare.evaluate_bpb(model, tokenizer, args.rows, args.seq_len, device,
                                     split="public_val", max_tokens=args.eval_tokens)["val_bpb"]
        cache[key] = v
        return v

    shipped = {g: (parts["table_blend"] if g == "tables" else parts["blend"]) for g in GROUPS}

    if args.verify:
        # THE decisive test. If the parts are right, reconstructing at the shipped coefficients must reproduce the
        # weights final.pt actually holds, to float noise - the two computations differ only in where the lerp ran.
        rec = reconstruct(shipped)
        worst, worst_n = 0.0, ""
        for n, _raw, _g in entries:
            d = (rec[n].float() - shipped_state[n].float()).abs().max().item()
            scale = shipped_state[n].float().abs().max().item()
            rel = d / max(scale, 1e-12)
            if rel > worst:
                worst, worst_n = rel, n
        print(f"\nVERIFY reconstruction at the shipped blend {parts['blend']}/{parts['table_blend']} vs final.pt:")
        print(f"  worst relative max-abs difference {worst:.3e} on {worst_n} "
              f"over {len(entries)} tensors")
        print("  PASS - the parts reconstruct the shipped checkpoint" if worst < 1e-5 else
              "  *** FAIL - the parts do NOT reconstruct final.pt. Every number below is meaningless. ***")
        if worst >= 1e-5:
            sys.exit(1)

    if args.grid:
        mains = [float(x) for x in args.main.split(",")]
        tabs = [float(x) for x in args.tables.split(",")]
        print(f"\nGRID over (shared main blend, table blend), {len(mains) * len(tabs)} points, "
              f"{args.eval_tokens:,} eval tokens:")
        res = {}
        for bm, bt in itertools.product(mains, tabs):
            b = {"matrices": bm, "head": bm, "scalars": bm, "tables": bt}
            res[(bm, bt)] = score(b)
            print(f"  main={bm:<5} tables={bt:<5} -> {res[(bm, bt)]:.6f}", flush=True)
        print("\n        " + "".join(f"{t:>11}" for t in tabs))
        for bm in mains:
            print(f"  {bm:<5} " + "".join(f"{res[(bm, t)]:>11.6f}" for t in tabs))
        best = min(res, key=res.get)
        print(f"\nBEST main={best[0]} tables={best[1]} -> {res[best]:.6f}")
        sp = (parts["blend"], parts["table_blend"])
        if sp in res:
            print(f"SHIPPED main={sp[0]} tables={sp[1]} -> {res[sp]:.6f}   (gain {res[best] - res[sp]:+.6f})")

    if args.groups:
        vals = [float(x) for x in args.main.split(",")]
        cur = dict(shipped)
        best = score(cur)
        print(f"\nPER-GROUP coordinate descent, start {cur} -> {best:.6f}")
        for r in range(args.rounds):
            for g in GROUPS:
                row = [(v, score(dict(cur, **{g: v}))) for v in vals]
                vb, sb = min(row, key=lambda t: t[1])
                print(f"  round {r + 1} {g:<9} " + "  ".join(
                    f"{v}:{s:.6f}{'*' if v == vb else ''}" for v, s in row), flush=True)
                if sb < best:
                    best, cur[g] = sb, vb
            print(f"  -> after round {r + 1}: {cur} = {best:.6f}")
        print(f"\nBEST per-group {cur} -> {best:.6f}   "
              f"(gain over shipped {best - score(shipped):+.6f})")

    if args.tracks:
        if not extra_decays:
            print("\n--tracks needs extra accumulators: re-run with NEURON_COMPETITION_R1_EMA_DECAY_EXTRA set.")
        else:
            mains = [float(x) for x in args.main.split(",")]
            tabs = [float(x) for x in args.tables.split(",")]
            # one pure point per track, plus optional 50/50 mixes with track 0
            ws = [None] + [tuple(1.0 if j == i else 0.0 for j in range(len(extra_decays)))
                           for i in range(len(extra_decays))]
            if args.mix:
                ws += [tuple(0.5 if j == i else 0.0 for j in range(len(extra_decays)))
                       for i in range(len(extra_decays))]
            all_decays = [float(parts.get("decay", 0.997))] + extra_decays
            print(f"\nDECAY x BLEND, {len(ws) * len(mains) * len(tabs)} points:")
            res = {}
            for w, bm, bt in itertools.product(ws, mains, tabs):
                b = {"matrices": bm, "head": bm, "scalars": bm, "tables": bt}
                v = score(b, w)
                res[(w, bm, bt)] = v
                if w is None:
                    label = f"tau~{1.0 / max(1e-9, 1.0 - all_decays[0]):.0f} (shipped)"
                elif sum(w) == 1.0:
                    label = f"tau~{1.0 / max(1e-9, 1.0 - extra_decays[w.index(1.0)]):.0f}"
                else:
                    label = f"mix w={w}"
                print(f"  {label:<20} main={bm:<5} tables={bt:<5} -> {v:.6f}", flush=True)
            best = min(res, key=res.get)
            print(f"\nBEST {best} -> {res[best]:.6f}")
            sp = (None, parts["blend"], parts["table_blend"])
            if sp in res:
                print(f"SHIPPED {sp} -> {res[sp]:.6f}   (gain {res[best] - res[sp]:+.6f})")

    print("\nThis is the PUBLIC DEV shard - the same surface every gate uses - so picking the argmin is ordinary "
          "hyperparameter selection. The sweep is deterministic, so read small differences as real; absolute "
          "values are ~0.001 above the box and only within-run comparisons transfer.")


if __name__ == "__main__":
    main()
