#!/usr/bin/env python3
"""Coordinate-descent the tail-EMA blend PER PARAMETER GROUP, exactly and offline. Zero gate noise.

`blend_sweep.py` showed the single shared blend for the "main" group has an interior optimum at 0.65 rather than
the shipped 0.35 (worth -0.000475, and the gate then agreed to 0.00001). But that group is not one kind of
object - it is three, sharing one coefficient:

    matrices  stacks.w3072x512 + w512x3072 + w512x512      50.3M   Muon-updated
    head      lm_head.weight                                4.19M  AdamW
    scalars   skip_lambdas, bigram_lambda, softcap_*,        93     AdamW, tiny
              block_scalars.{resid,sink,ve,x0}
    tables    wte, vte, vte_mid, vte2                       16.8M  AdamW at 4.5x LR (already its own group)

The ANVIL record blends exactly these separately - `lm_head` with `embed` at one coefficient, the qk/vo/mlp banks
at another (and a REPLACE rather than a blend for some) - so one shared number is very likely leaving something
on the table. Since the blend is applied once after training, all of this is reconstructible from a single v59
checkpoint: p_final = (1-b_g) * p_raw + b_g * p_avg, with b_g per group.

Coordinate descent rather than a full grid: 4 coefficients x 6 values would be 1296 evaluations. Two sweeps over
the four coordinates is ~48 and finds the local optimum, which is all a smooth surface needs (the 2-D grid in
blend_sweep.py was smooth and monotone in both directions).

Usage:
    .venv/bin/python experimental/blend_groups.py out_T_V59/final.pt [--eval-tokens 524288] [--rounds 2]

Verify the winning point at --eval-tokens 2097152 before believing the magnitude; the coordinate search itself is
fine on a smaller fixed subset because every comparison uses the identical tokens.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare  # noqa: E402

GROUPS = {
    "matrices": lambda n: n.startswith("stacks."),
    "head": lambda n: n == "lm_head.weight",
    "scalars": lambda n: not n.startswith("stacks.") and n != "lm_head.weight",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--train-py", default="submission_raw/submission_v59_clean_emaparts.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-tokens", type=int, default=524288)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--values", default="0.35,0.5,0.65,0.8,0.95")
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    parts = payload["ema_parts"]
    base, config_d = payload["model_state"], payload["model_config"]

    import importlib.util
    spec = importlib.util.spec_from_file_location("subm", args.train_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["subm"] = mod
    spec.loader.exec_module(mod)

    device = torch.device(args.device)
    tokenizer = prepare.ensure_tokenizer()
    model = mod.GPT(mod.GPTConfig(**config_d))
    model.init_weights()

    # name -> (raw, avg, group)
    entries = []
    for n, raw in parts["raw"].items():
        g = next(g for g, pred in GROUPS.items() if pred(n))
        entries.append((n, raw, parts["avg"][n], g))
    for n, raw in parts["tab_raw"].items():
        entries.append((n, raw, parts["tab_avg"][n], "tables"))
    for g in list(GROUPS) + ["tables"]:
        tot = sum(r.numel() for _n, r, _a, gg in entries if gg == g)
        print(f"  group {g:<9} {sum(1 for e in entries if e[3] == g):2d} tensors  {tot:,} elements")

    cache = {}

    def score(b: dict) -> float:
        key = tuple(sorted(b.items()))
        if key in cache:
            return cache[key]
        state = dict(base)
        for n, raw, avg, g in entries:
            state[n] = torch.lerp(raw.float(), avg.float(), b[g]).to(raw.dtype)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        v = prepare.evaluate_bpb(model, tokenizer, args.batch_size, args.seq_len, device,
                                 split="public_val", max_tokens=args.eval_tokens)["val_bpb"]
        cache[key] = v
        return v

    vals = [float(x) for x in args.values.split(",")]
    cur = {"matrices": 0.65, "head": 0.65, "scalars": 0.65, "tables": 0.65}
    best = score(cur)
    print(f"\nstart {cur} -> {best:.6f}")

    for r in range(args.rounds):
        for g in ["matrices", "head", "scalars", "tables"]:
            row = []
            for v in vals:
                trial = dict(cur, **{g: v})
                s = score(trial)
                row.append((v, s))
            v_best, s_best = min(row, key=lambda t: t[1])
            marks = "  ".join(f"{v}:{s:.6f}{'*' if v == v_best else ''}" for v, s in row)
            print(f"round {r + 1} {g:<9} {marks}")
            if s_best < best:
                best, cur[g] = s_best, v_best
        print(f"  -> after round {r + 1}: {cur} = {best:.6f}")

    print(f"\nBEST {cur} -> {best:.6f}")
    flat = score({k: 0.65 for k in cur})
    print(f"all-0.65 reference -> {flat:.6f}   (per-group gain {best - flat:+.6f})")
    print("\nVerify the winning point at --eval-tokens 2097152 before shipping; the search subset is fine for "
          "ranking because every comparison uses identical tokens, but the magnitude should be confirmed.")


if __name__ == "__main__":
    main()
