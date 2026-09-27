"""Exact, zero-gate-noise offline evaluation of pure OUTPUT-SPACE transforms on a trained checkpoint.

Gates cost 30 min and carry ~7e-5 of same-file noise, so a -0.0004 question needs n>=3. A transform that touches
only the logits does not need a gate at all: the same checkpoint and the same eval tokens give an EXACT delta.
This is the method that settled the EMA blend and the per-group blends offline.

Two candidates from findings/quality-levers-2026-09-26.md, neither shipped:

  SPECIAL  The four special tokens (BOS 8188 and three reserved) are NEVER SCORED - prepare.py zeroes their
           token_bytes and evaluate_bpb masks on nbytes > 0 - yet they still consume softmax mass at every
           position. A finite negative offset on those four logits frees that mass for real tokens.
           Trap from the findings: -inf is FATAL, because evaluate_bpb computes loss * mask and inf * 0.0 = NaN.

  MERGE    Single-merge-infeasible continuations: if bytes(prev) + bytes(next) is ITSELF a token, then a BPE
           tokenizer that had both merges available would usually have emitted the merged token instead, so the
           pair is (mostly) impossible in tokenized text. Masking those logits removes mass from outcomes that
           cannot occur. This is derived from the FIXED tokenizer, not fitted to data - but the criterion is a
           heuristic about merge ORDER, so this script also VERIFIES it against real train tokens before
           trusting it, and reports how often the "impossible" pairs actually occur.

usage: .venv/bin/python experimental/offline_logit_eval.py <checkpoint> [--tokens N] [--variants a,b,c]
Reports val_bpb per variant against the same baseline. Changes nothing; writes nothing.
"""
import argparse
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
import prepare  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--vehicle", default="submission_raw/submission_v72_clean_dyfuse.py")
ap.add_argument("--tokens", type=int, default=4 * 524288)
ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--offsets", default="10,30,80")
ap.add_argument("--temps", default="0.96,0.98,1.02,1.04,1.08")
args = ap.parse_args()

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("veh", args.vehicle)
veh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(veh)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = prepare.ensure_tokenizer()
V = tok.get_vocab_size()
base = veh.load_for_eval(args.checkpoint, dev)
print(f"device={dev} vocab={V} params={sum(p.numel() for p in base.parameters()):,}", flush=True)

# ---- which token ids are never scored? read it off prepare's own table rather than assuming ----------------
tb = prepare.get_token_bytes(dev)
never = (tb == 0).nonzero().flatten().tolist()
print(f"never-scored ids (token_bytes == 0): {never}", flush=True)


class Wrap(nn.Module):
    """Applies a logit transform. Kept pure and stateless so causality/batch-independence are untouched."""

    def __init__(self, inner, fn):
        super().__init__()
        self.inner = inner
        self.fn = fn

    def forward(self, idx, *a, **k):
        return self.fn(self.inner(idx, *a, **k), idx)


def run(fn, label):
    m = base if fn is None else Wrap(base, fn)
    with torch.no_grad():
        r = prepare.evaluate_bpb(m, tok, args.rows, veh.SEQ_LEN, dev,
                                 split="public_val", max_tokens=args.tokens)
    print(f"{label:28s} val_bpb {r['val_bpb']:.6f}", flush=True)
    return r["val_bpb"]


ref = run(None, "BASELINE")

# ---- TEMP: a scalar on the logits. WHY THIS IS NOT TRIVIALLY 1.0 -----------------------------------------
# Training minimises exactly this metric, so at the training optimum the calibration is already right and a
# temperature can only hurt. But the shipped recipe BLENDS WEIGHTS AFTER TRAINING ENDS (tail-EMA, blend 0.65 on
# the table group) and nothing re-calibrates afterwards. A convex average of two weight vectors generally has a
# SMALLER effective logit scale than either endpoint - averaging in weight space shrinks the margin - which makes
# the blended model systematically UNDER-confident on a metric that rewards confidence when correct. So the
# optimum here is expected slightly ABOVE 1.0, and it is free: one scalar, pure output space, exactly evaluable.
# The learned logit softcap (tanh) does not absorb this: it squashes the tail, it does not rescale the body.
for t in [float(x) for x in args.temps.split(",")]:
    def fn(lg, idx, t=t):
        return lg * t
    b = run(fn, f"TEMP x{t:g}")
    print(f"{'':28s}   delta {b - ref:+.6f}", flush=True)

# ---- SPECIAL: a finite negative offset on the never-scored ids -------------------------------------------
sp_idx = torch.tensor(never, device=dev, dtype=torch.long)
for off in [float(x) for x in args.offsets.split(",")]:
    def fn(lg, idx, off=off):
        lg = lg.clone()
        lg[..., sp_idx] -= off
        return lg
    b = run(fn, f"SPECIAL -{off:g}")
    print(f"{'':28s}   delta {b - ref:+.6f}", flush=True)

print("\n=== MERGE: build the single-merge-infeasible table from the tokenizer ===", flush=True)
enc = tok.enc
raw = []
for i in range(V):
    try:
        raw.append(enc.decode_single_token_bytes(i))
    except Exception:
        raw.append(None)
lookup = {b: i for i, b in enumerate(raw) if b is not None}
# Enumerate by SPLITTING each token instead of scanning the 8192^2 grid: a pair (a,b) is single-merge-infeasible
# exactly when bytes(a)+bytes(b) is itself a token, i.e. when some token splits into two in-vocab halves.
pairs = [[] for _ in range(V)]
n_pairs = 0
for t, bt in enumerate(raw):
    if bt is None or len(bt) < 2:
        continue
    for k in range(1, len(bt)):
        a = lookup.get(bt[:k])
        b = lookup.get(bt[k:])
        if a is not None and b is not None:
            pairs[a].append(b)
            n_pairs += 1
mx = max(len(x) for x in pairs)
print(f"infeasible (prev,next) pairs: {n_pairs:,} = {n_pairs/(V*V):.4%} of the grid; max per prev {mx}", flush=True)

# ---- VERIFY the heuristic on real TRAIN tokens before trusting it ----------------------------------------
setp = [set(x) for x in pairs]
loader = prepare.make_dataloader(tok, 8, 1024, "train", dev, tokenizer_threads=4)
seen = tot = 0
for bi, (x, _y, _s) in enumerate(loader):
    f = x.reshape(-1).tolist()
    for a, b in zip(f, f[1:]):
        tot += 1
        if b in setp[a]:
            seen += 1
    if bi >= 7:
        break
print(f"VERIFY: {seen:,} of {tot:,} real adjacent train pairs ({seen/max(1,tot):.4%}) are 'infeasible'", flush=True)
print("  (a rate far above ~0 means the single-merge criterion is WRONG and the mask would delete real mass)", flush=True)

if seen / max(1, tot) < 0.002:
    flat = torch.full((V, mx), -1, dtype=torch.long)
    for p in range(V):
        if pairs[p]:
            flat[p, :len(pairs[p])] = torch.tensor(pairs[p])
    flat = flat.to(dev)
    valid = (flat >= 0)
    flatc = flat.clamp(min=0)

    def fn_merge(lg, idx, off=30.0):
        lg = lg.clone()
        cols = flatc[idx]                      # (b, t, mx) candidate next-ids to suppress
        msk = valid[idx]
        add = torch.where(msk, torch.full(cols.shape, -off, dtype=lg.dtype, device=lg.device),
                          torch.zeros(cols.shape, dtype=lg.dtype, device=lg.device))
        lg.scatter_add_(-1, cols, add)
        return lg
    b = run(fn_merge, "MERGE -30")
    print(f"{'':28s}   delta {b - ref:+.6f}", flush=True)
else:
    print("SKIPPING the merge mask: the criterion is refuted by real data.", flush=True)
