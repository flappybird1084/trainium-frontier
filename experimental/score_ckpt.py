"""Score one or more checkpoints with prepare.evaluate_bpb under IDENTICAL settings, on whatever device is here.

Why this exists: a gate's in-run eval can crash after training (C_MTPB died in causality_check with the
checkpoint already on disk), and two gates' in-run numbers also differ by which eval tokens each rank drew. Scoring
saved checkpoints here removes BOTH problems - same tokens, same code, same device for every checkpoint, so the
DELTA between two checkpoints is exact and the only remaining noise is the training runs themselves.

usage: .venv/bin/python experimental/score_ckpt.py a.pt b.pt [--tokens N] [--vehicle path]
"""
import argparse
import importlib.util
import sys

import torch

sys.path.insert(0, ".")
import prepare  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("checkpoints", nargs="+")
ap.add_argument("--vehicle", default="submission_raw/submission_v72_clean_dyfuse.py")
ap.add_argument("--tokens", type=int, default=1048576)
ap.add_argument("--rows", type=int, default=4)
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("veh", a.vehicle)
veh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(veh)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = prepare.ensure_tokenizer()
print(f"device={dev} vehicle={a.vehicle} tokens={a.tokens}", flush=True)

out = []
for ck in a.checkpoints:
    m = veh.load_for_eval(ck, dev)
    with torch.no_grad():
        r = prepare.evaluate_bpb(m, tok, a.rows, veh.SEQ_LEN, dev, split="public_val", max_tokens=a.tokens)
    out.append((ck, r["val_bpb"]))
    print(f"{ck:34s} val_bpb {r['val_bpb']:.6f}", flush=True)
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()

if len(out) > 1:
    base = out[0][1]
    print("\ndeltas vs the first checkpoint (exact: same tokens, same code, same device):")
    for ck, v in out[1:]:
        print(f"  {ck:32s} {v - base:+.6f}")
