"""Post-hoc sweep of the model's LEARNED SCALARS on a trained checkpoint. Exact, offline, zero gate time.

Same logic that made the 0.98 eval logit temperature worth -0.00033: the scalars are optimised jointly with
the weights on TRAINING loss, and nothing re-tunes them after the tail-EMA blends weights at the end of the run.
Where a scalar controls how much of a MEMORISING component is mixed in, its held-out optimum should sit BELOW
its training optimum, because memorisation generalises worse than learned features. `bigram_lambda` is the
sharpest case: it weights a 2^18 x 512 hashed bigram table that is 65% of all parameters.

Each point is an exact re-score of the same checkpoint on the same tokens with one scalar overwritten, so the
deltas carry no training noise at all. This is the public dev shard - the same surface every gate uses - so
picking an argmin is ordinary hyperparameter selection, exactly as depth/width/rope-frac/EMA-blend were picked.

usage: .venv/bin/python experimental/scalar_sweep.py <checkpoint> --param bigram_lambda --values 0.5,0.6,0.7377,0.8
"""
import argparse
import importlib.util
import sys

import torch

sys.path.insert(0, ".")
import prepare  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--vehicle", default="submission_raw/submission_v76_clean_evaltemp.py")
ap.add_argument("--param", required=True, help="state_dict key, e.g. bigram_lambda / softcap_pos")
ap.add_argument("--values", required=True, help="comma-separated absolute values, or R<mult> for relative")
ap.add_argument("--tokens", type=int, default=524288)
ap.add_argument("--rows", type=int, default=4)
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("veh", args.vehicle)
veh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(veh)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = prepare.ensure_tokenizer()
model = veh.load_for_eval(args.checkpoint, dev)

target = dict(model.named_parameters()).get(args.param)
if target is None:
    print(f"no such parameter {args.param!r}; candidates: "
          f"{[n for n, p in model.named_parameters() if p.numel() <= 64]}")
    raise SystemExit(1)
orig = target.detach().clone()
print(f"device={dev} param={args.param} shape={tuple(orig.shape)} "
      f"trained={[round(float(x), 4) for x in orig.flatten()[:6]]}", flush=True)


def score(label):
    with torch.no_grad():
        r = prepare.evaluate_bpb(model, tok, args.rows, veh.SEQ_LEN, dev,
                                 split="public_val", max_tokens=args.tokens)
    print(f"{label:28s} val_bpb {r['val_bpb']:.6f}", flush=True)
    return r["val_bpb"]


ref = score("TRAINED (baseline)")
best, best_lbl = ref, "TRAINED"
for spec_v in args.values.split(","):
    spec_v = spec_v.strip()
    if spec_v.upper().startswith("R"):
        m = float(spec_v[1:])
        target.data.copy_(orig * m)
        lbl = f"{args.param} x{m:g}"
    else:
        target.data.fill_(float(spec_v)) if orig.numel() == 1 else target.data.copy_(
            orig * (float(spec_v) / float(orig.flatten()[0])))
        lbl = f"{args.param} = {spec_v}"
    b = score(lbl)
    print(f"{'':28s}   delta {b - ref:+.6f}", flush=True)
    if b < best:
        best, best_lbl = b, lbl
    target.data.copy_(orig)
print(f"\nBEST {best_lbl} -> {best:.6f}   (gain {best - ref:+.6f} vs trained)")
print("Deltas are exact: same checkpoint, same tokens, one scalar changed. Absolute level is not the box's.")
