#!/usr/bin/env python3
"""Generate v58 = v57 + ignore BOS targets in the training loss. One line. Aligns training with the scored metric.

## The misalignment

`prepare.py:262-264` sets `token_bytes = 0` for every special token, and `evaluate_bpb` masks its numerator with
`valid & (nbytes > 0)` (`:628-634`). **So a BOS target is NEVER SCORED at evaluation.**

But training scores it with full weight. The train dataloader emits no `-1` anywhere, so every one of the 1024
targets per row counts, including the document-separator BOS that `tokenizer.encode(..., prepend=bos)` puts at
the head of each document. Measured on the corpus: the average document is 739.4 tokens, so BOS is **0.135% of
training targets** — 0.135% of the model's capacity and gradient signal spent on a prediction the metric deletes.

This is not a metric trick; it is the definition of `ignore_index`. We are currently training on a task the
scorer does not ask about.

## Why it costs one line

**The fused NKI head already implements `ignore_index = -1` correctly**, in both directions:

    head_fwd  (:729-733)  valid = (tgt >= 0);  lossr *= valid
    head_bwd  (:790-792)  valid = (tgt >= 0);  g     *= valid

and the reduction divides by the valid count, not the row count (`:2928-2929`,
`loss_rows.sum() / valid.sum().clamp_min(1)`), with the docstring at `:2917` stating "ignore_index=-1" outright.
So nothing in the kernels, the loss scale or the gradient normalisation needs touching — the targets simply have
to carry `-1`.

It is also safe with respect to the document masking, which is computed from **BOS positions in the INPUT**
(`doc_bounds(idx)` via `:2964`, `idx == BOS_ID`), not from the targets. Changing `y` cannot affect attention
bounds, the key offset, or the bigram `prev` channel.

## Expected

The red-team measured the *eval-side* analogue (suppressing the four never-scored reserved logits at evaluation)
at **-0.000381**. The training-side version should be similar or slightly better, because it also stops spending
capacity on the boundary task rather than merely declining to emit it. Call it **-0.0004**, with the honest note
that this is below the ~0.00025 single-gate noise floor and will need to be read as part of a stack.

Cost: one `torch.where` on a (16, 1024) tensor per micro-step. Unmeasurable.

Run from the repo root:  .venv/bin/python submission_raw/make_v58.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v57_clean_ematables.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v58_clean_bosignore.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------- the constant
sub("const", """BOS_ID = 8188                   # the fixed tokenizer's <|reserved_0|> (document separator)""",
    """BOS_ID = 8188                   # the fixed tokenizer's <|reserved_0|> (document separator)
# BOS_IGNORE: map BOS targets to -1 so the training loss skips them, because evaluate_bpb NEVER SCORES THEM -
# prepare.py sets token_bytes=0 for every special token and masks the numerator with (nbytes > 0). BOS is 0.135%
# of training targets (mean document 739.4 tokens), so that fraction of capacity and gradient currently goes to a
# prediction the metric deletes. The fused head already implements ignore_index=-1 in both directions and divides
# by the valid count, so this needs no kernel change. Safe for the document masking, which reads BOS from the
# INPUT (doc_bounds(idx)), never from the targets.
BOS_IGNORE = True""")

# ---------------------------------------------------------------- the one line
sub("packed", """    def _packed():   # PACK-long best-fit rows -> PACK rows of SEQ_LEN each (the organizer-suggested reshape)
        for _x, _y, _st in _raw_loader:
            yield _x.reshape(-1, SEQ_LEN), _y.reshape(-1, SEQ_LEN), _st""",
    """    def _packed():   # PACK-long best-fit rows -> PACK rows of SEQ_LEN each (the organizer-suggested reshape)
        for _x, _y, _st in _raw_loader:
            _yr = _y.reshape(-1, SEQ_LEN)
            if BOS_IGNORE:
                _yr = torch.where(_yr == BOS_ID, -1, _yr)     # unscored at eval -> ignore_index in training too
            yield _x.reshape(-1, SEQ_LEN), _yr, _st""")

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:10s} x{v}")
