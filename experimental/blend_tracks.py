#!/usr/bin/env python3
"""Sweep the EMA DECAY together with both blend coefficients, exactly and offline. Zero gate noise.

A v66 run with `EMA_DECAY2=0.9986 SAVE_EMA_PARTS=1` carries TWO parallel EMA tracks (tau = 333 and 714 steps)
plus the pre-blend raw weights. Since the blend is applied once after training, every combination of
(which track, main blend, table blend) is reconstructible from that one checkpoint:

    p_final = (1 - b) * p_raw + b * p_avg[track]

and a convex mix of the two tracks approximates intermediate horizons, so `--mix` also sweeps *between* them.

## Why the decay is the item

After v62 (window 0.85 -> 0.75, -0.00044) and v64 (0.75 -> 0.65, a wash) I wrongly concluded the averaging
family was closed, reasoning "at 3.1 horizons the average is converged so the decay is fine". That is backwards:
converged means the WINDOW stopped binding, so tau is now the effective averaging length and the only remaining
way to lengthen the average. Our ~1060-step window is decoration around a 333-step exponential average.

The ledger never measured tau where we now sit. Rows 421-431 (2026-09-15/16) ran 3400 steps at EMA_START 0.85, a
510-step window: tau 333 = 1.53 horizons, tau 667 = 0.76. Those tied ("plateau from 0.997 on"). We now run a
1060-step window at 3.18 horizons - off the measured range on the short-tau side, because throughput grew AND
v62 doubled the window.

Raising tau lengthens the effective average at the cost of convergence: at W=1060, tau=333 gives ~333 fully
converged, tau=714 gives 714*(1-e^-1.48) ~ 552, tau=1060 ~ 670. There is an interior optimum, never located.

**Second thing this fixes:** every earlier offline sweep used the v59 checkpoint, whose `EMA_START` was 0.85.
v62 moved it to 0.75, so the blend optimum may have moved again - the same window<->blend coupling that produced
v60 and v62, one crank later. This re-checks both blends at the CURRENT window.

Usage:
    .venv/bin/python experimental/blend_tracks.py out_T_V66/final.pt [--eval-tokens 524288] [--mix]

Confirm the winner at --eval-tokens 2097152 before shipping. Absolute values run ~0.001 above the box's in-run
number (the known local-GPU-vs-Neuron eval gap), so only compare within this script's own output.
"""
import argparse
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--train-py", default="submission_raw/submission_v66_clean_twotrack.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-tokens", type=int, default=524288)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--main", default="0.5,0.65,0.8")
    ap.add_argument("--tables", default="0.5,0.65,0.8")
    ap.add_argument("--mix", action="store_true",
                    help="also sweep convex mixes of the two tracks (approximates intermediate tau)")
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    parts = payload["ema_parts"]
    if "avg2" not in parts:
        sys.exit("this checkpoint has only one EMA track - rerun with NEURON_COMPETITION_R1_EMA_DECAY2 set")
    base = payload["model_state"]
    d1, d2 = parts["decay"], parts["decay2"]
    print(f"track 1 decay {d1} (tau {1/(1-d1):.0f} steps) | track 2 decay {d2} (tau {1/(1-d2):.0f})")
    print(f"as-shipped: main {parts['blend']}, tables {parts['table_blend']}, EMA_START {parts.get('ema_start')}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("subm", args.train_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["subm"] = mod
    spec.loader.exec_module(mod)

    device = torch.device(args.device)
    tokenizer = prepare.ensure_tokenizer()
    model = mod.GPT(mod.GPTConfig(**payload["model_config"]))
    model.init_weights()

    def score(bm: float, bt: float, w2: float) -> float:
        """w2 = weight on the LONGER track (0 = tau 333 only, 1 = tau 714 only)."""
        state = dict(base)
        for key, keyavg, key2, b in (("raw", "avg", "avg2", bm), ("tab_raw", "tab_avg", "tab_avg2", bt)):
            for n, raw in parts[key].items():
                avg = torch.lerp(parts[keyavg][n].float(), parts[key2][n].float(), w2) if w2 > 0 \
                    else parts[keyavg][n].float()
                state[n] = torch.lerp(raw.float(), avg, b).to(raw.dtype)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return prepare.evaluate_bpb(model, tokenizer, args.batch_size, args.seq_len, device,
                                    split="public_val", max_tokens=args.eval_tokens)["val_bpb"]

    mains = [float(x) for x in args.main.split(",")]
    tabs = [float(x) for x in args.tables.split(",")]
    w2s = [0.0, 0.5, 1.0] if args.mix else [0.0, 1.0]

    results = {}
    for w2, bm, bt in itertools.product(w2s, mains, tabs):
        v = score(bm, bt, w2)
        results[(w2, bm, bt)] = v
        tau = 1 / (1 - d1) if w2 == 0 else (1 / (1 - d2) if w2 == 1 else None)
        label = f"tau~{tau:.0f}" if tau else f"mix w2={w2}"
        print(f"  {label:<12} main={bm:<5} tables={bt:<5} -> {v:.6f}")

    best = min(results, key=results.get)
    shipped = (0.0, parts["blend"], parts["table_blend"])
    print(f"\nBEST  w2={best[0]} main={best[1]} tables={best[2]} -> {results[best]:.6f}")
    if shipped in results:
        print(f"SHIPPED (track 1, {shipped[1]}/{shipped[2]}) -> {results[shipped]:.6f}"
              f"   gain {results[best] - results[shipped]:+.6f}")
    print("\nConfirm the winner at --eval-tokens 2097152. The blend and the track are both post-hoc, so whatever "
          "this says is exact for this training run - no gate needed to adopt it.")


if __name__ == "__main__":
    main()
