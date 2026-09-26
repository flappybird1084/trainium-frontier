#!/usr/bin/env python3
"""Test DECONTRACTION of the tail-EMA blend, exactly and offline. Post-hoc, so it costs nothing to evaluate.

Averaging shrinks a weight tensor's norm: ||lerp(raw, avg, b)|| <= max(||raw||, ||avg||), and for two nearby but
not identical points the blend lands strictly inside, so the shipped tensor is systematically smaller than the
weights the network was actually trained with. The ANVIL record corrects exactly this, per tensor, right after
its ships:

    dref = ||p||_F  BEFORE any averaging
    ...blend...
    p *= dref / ||p||_F          (a single scalar per tensor, full-tensor Frobenius, kept as a python float)

applied to its embed / lm_head / mlp_bank / qk_bank / vo_bank labels and NOT to its value-embedding or n-gram
labels. It is the one member of that family we have not tested, and like the blend itself it is applied once
after training, so a v59 checkpoint that saved raw and averaged weights determines its effect EXACTLY - no gate,
no noise.

This reports the norm shrinkage per tensor (so the magnitude of the effect is visible before the score), then
scores: no decontraction, decontraction on the matrix stacks only, on stacks+head, on the tables only, and on
everything. The float-vs-tensor detail from the record matters and is followed: the ratio is applied as a python
float so a bf16 parameter rounds once on the store rather than snapping the ratio to bf16's 2^-7 grid first.

Usage:
    .venv/bin/python experimental/decontract.py out_T_V59/final.pt [--eval-tokens 524288] [--blend 0.65]
"""
import argparse
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
    ap.add_argument("--eval-tokens", type=int, default=524288)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--blend", type=float, default=0.65)
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    parts = payload["ema_parts"]
    base = payload["model_state"]

    import importlib.util
    spec = importlib.util.spec_from_file_location("subm", args.train_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["subm"] = mod
    spec.loader.exec_module(mod)

    entries = [(n, r, parts["avg"][n], "main") for n, r in parts["raw"].items()]
    entries += [(n, r, parts["tab_avg"][n], "tables") for n, r in parts["tab_raw"].items()]

    print("norm shrinkage from blending at b=%.2f (this bounds how much decontraction can matter):" % args.blend)
    for n, raw, avg, _g in entries:
        if raw.numel() < 1000:
            continue
        nb = torch.lerp(raw.float(), avg.float(), args.blend).norm().item()
        nr = raw.float().norm().item()
        print(f"   {n:<34} ||raw|| {nr:12.4f}  ||blend|| {nb:12.4f}  ratio {nr / nb:.6f}")

    device = torch.device(args.device)
    tokenizer = prepare.ensure_tokenizer()
    model = mod.GPT(mod.GPTConfig(**payload["model_config"]))
    model.init_weights()

    def build(which: set) -> dict:
        state = dict(base)
        for n, raw, avg, _g in entries:
            p = torch.lerp(raw.float(), avg.float(), args.blend)
            if n in which:
                nb = p.norm().item()
                if nb > 0:
                    p = p * float(raw.float().norm().item() / nb)   # python float, per the record's note
            state[n] = p.to(raw.dtype)
        return state

    stacks = {n for n, _r, _a, _g in entries if n.startswith("stacks.")}
    head = {"lm_head.weight"}
    tabs = {n for n, _r, _a, g in entries if g == "tables"}
    big = {n for n, r, _a, _g in entries if r.numel() >= 1000}

    cases = [("none", set()), ("stacks", stacks), ("stacks+head", stacks | head),
             ("tables", tabs), ("all big tensors", big)]
    print()
    out = {}
    for name, which in cases:
        model.load_state_dict(build(which), strict=True)
        model.to(device).eval()
        v = prepare.evaluate_bpb(model, tokenizer, args.batch_size, args.seq_len, device,
                                 split="public_val", max_tokens=args.eval_tokens)["val_bpb"]
        out[name] = v
        print(f"  decontract {name:<16} -> {v:.6f}")

    best = min(out, key=out.get)
    print(f"\nBEST {best} -> {out[best]:.6f}   (vs none {out['none']:.6f}, gain {out[best] - out['none']:+.6f})")
    print("Confirm any winner at --eval-tokens 2097152 before shipping.")


if __name__ == "__main__":
    main()
