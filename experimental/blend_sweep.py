#!/usr/bin/env python3
"""Sweep the two tail-EMA blend coefficients EXACTLY, offline, from one checkpoint. Zero gate noise.

A v59 run with NEURON_COMPETITION_R1_SAVE_EMA_PARTS=1 stores, alongside the usual (already blended)
`model_state`, the pre-blend raw weights and the EMA averages for both blended groups. The final weights are

    p_final = (1 - b) * p_raw + b * p_avg

so every coefficient is reconstructible without retraining. This script rebuilds the state for each point of a
2-D grid over (EMA_BLEND for the 12 matrix/head/scalar tensors, EMA_TABLE_BLEND for the four token/value tables)
and scores each with the OFFICIAL metric via `prepare.evaluate_bpb`.

Why bother: every hyperparameter sweep on this project has been limited by the ~0.00025 per-gate noise floor,
which is why so many of them read as flat plateaus. This sweep is deterministic - the same checkpoint, the same
eval shard, no training - so it resolves differences far below that floor and finds a real optimum.

Usage (local GPU is fine and free; the box is not needed):
    .venv/bin/python experimental/blend_sweep.py out_T_V59/final.pt [--device cuda] [--eval-tokens 2097152]

Reads nothing but the checkpoint and the public dev shard. Does not modify prepare.py or any submission file.
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
    ap.add_argument("--train-py", default="submission_raw/submission_v59_clean_emaparts.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-tokens", type=int, default=2 * 524288)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--main", default="0.0,0.2,0.35,0.5,0.65,0.8,1.0")
    ap.add_argument("--tables", default="0.0,0.35,0.5,0.65,0.8,1.0")
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    parts = payload.get("ema_parts")
    if not parts:
        sys.exit("checkpoint has no 'ema_parts' - rerun training with NEURON_COMPETITION_R1_SAVE_EMA_PARTS=1")
    print(f"raw/avg pairs: {len(parts['raw'])} main + {len(parts['tab_raw'])} tables "
          f"(as-shipped blends: main {parts['blend']}, tables {parts['table_blend']})")

    # Import the submission module the same way the official scorer does, so GPT/GPTConfig match the checkpoint.
    mod = prepare._import_train_module(args.train_py) if hasattr(prepare, "_import_train_module") else None
    if mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("subm", args.train_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["subm"] = mod
        spec.loader.exec_module(mod)

    device = torch.device(args.device)
    tokenizer = prepare.ensure_tokenizer()
    base = payload["model_state"]
    config = mod.GPTConfig(**payload["model_config"])

    mains = [float(x) for x in args.main.split(",")]
    tabs = [float(x) for x in args.tables.split(",")]
    results = {}

    model = None
    for bm, bt in itertools.product(mains, tabs):
        state = dict(base)
        for n, raw in parts["raw"].items():          # p = (1-bm)*raw + bm*avg
            state[n] = torch.lerp(raw.float(), parts["avg"][n].float(), bm).to(raw.dtype)
        for n, raw in parts["tab_raw"].items():
            state[n] = torch.lerp(raw.float(), parts["tab_avg"][n].float(), bt).to(raw.dtype)
        if model is None:
            model = mod.GPT(config)
            model.init_weights()
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        bpb = prepare.evaluate_bpb(model, tokenizer, args.batch_size, args.seq_len, device,
                                   split="public_val", max_tokens=args.eval_tokens)["val_bpb"]
        results[(bm, bt)] = bpb
        print(f"  main={bm:<5} tables={bt:<5} -> {bpb:.6f}")

    print("\ngrid (rows = EMA_BLEND, cols = EMA_TABLE_BLEND):")
    print("        " + "".join(f"{t:>11}" for t in tabs))
    for bm in mains:
        print(f"  {bm:<5} " + "".join(f"{results[(bm, t)]:>11.6f}" for t in tabs))

    best = min(results, key=results.get)
    shipped = (parts["blend"], parts["table_blend"])
    print(f"\nBEST  main={best[0]} tables={best[1]} -> {results[best]:.6f}")
    if shipped in results:
        print(f"SHIPPED main={shipped[0]} tables={shipped[1]} -> {results[shipped]:.6f}"
              f"   (gain {results[best] - results[shipped]:+.6f})")
    print("\nNOTE: this is the PUBLIC DEV shard, the same surface every gate uses, so picking the argmin here is "
          "ordinary hyperparameter selection - but the grid is deterministic, so read small differences as real "
          "and expect the usual ~0.0069 offset to the portal on top.")


if __name__ == "__main__":
    main()
