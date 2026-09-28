# The v78 review rejection: two causes, both self-inflicted, both fixed in v79

## Cause 1 — the file contradicted its own review notes

`head_logits` ended with `s = getattr(self, "_eval_logit_scale", 1.0); return out if s == 1.0 else out * s`,
and **only `load_for_eval` ever set that attribute**. That is literally the v32 rejection text: *"eval-mode
detection ... branches logit computation"*.

What makes it fatal rather than merely risky: the file's own **"Notes for review"** block, unchanged since
the accepted v52, asserts *"No code path inspects training/eval mode ... to pick a formula."* The submitted
file therefore contained a written claim and, ~2800 lines later, the code that breaks it. **A reviewer who
checks the claim finds a contradiction, and a contradiction reads as deception rather than sloppiness.**
The same change also silently falsified the second claim — that the fused training loss equals
`CE(head_logits(x), targets)` — which held only while the attribute was absent.

It was added 2026-09-27 for a post-hoc logit temperature of 0.98 worth **-0.00024** measured exactly on
three checkpoints. **Dropped, not rescued.** Making it a checkpoint-borne parameter was considered and
rejected as worse: it keeps a train/eval value difference in the logit path and moves it from
`load_for_eval` (visible) to `save_checkpoint` (concealed). The tail-EMA analogy does not hold — EMA
averages weights the run actually visited; a temperature is a value fitted afterwards that the training
loss never saw.

## Cause 2 — the scored architecture was environment-dependent (nobody was looking at this)

Diffing against **v52, the last file the portal ACCEPTED**:

| file | `os.environ.get(` reads |
|---|---|
| v52 (accepted) | **5** |
| v78 (rejected) | **50** |
| v79 (fix) | **5** |

The 45 additions include `ROPE_BASE`, `ROPE_FRAC`, `KEY_OFFSET`, `KEY_OFFSET_DIV`, `SOFTCAP`, `SINK_INIT`,
`VE_T1`, `VE_T2`, `LAYER_IMPL`, `EVAL_LOGIT_SCALE` — so **the architecture of the scored model could be
changed without changing the file.** That is the third clause of the same review-notes sentence ("...or the
environment to pick a formula"), and it is exactly what the v33 recovery was: the findings record it as *"a
clean minimal file - config hard-coded."* We drifted all of it back over 26 versions, one convenient knob at
a time, and "they agree by construction because the launch command sets nothing" is true and irrelevant — a
reviewer cannot verify a launch command they did not see.

## Also fixed

- **`check_fused_loss` fail direction.** The `try/finally` I added on 2026-09-27 set `_fused_loss_ok = True`
  unconditionally, so *any* exception re-enabled the path under test while the call site printed "fused path
  KEPT". Now per-side: fused raises -> disable it; reference raises -> keep and say so.
- **`torch.compiler.is_compiling()` deleted from `forward`.** With `LAYER_IMPL` a literal both branches called
  `doc_bounds(idx)` identically, so the mask is now selected by device type alone. That removes the last
  mode-shaped predicate in the scored forward.
- **`check_fused_head` added, because the old check provably proves nothing.** Measured: at step 0 the model's
  own forward gives `|logit|max = 0.128` against a softcap of 11.0 — **the softcap is inactive by a factor of
  86**, so no step-0 comparison can test it at any tolerance, which is why the check "passed to 0.00e+00"
  through ~25 versions while the RMSNorm eps defect was live. The new check runs at step 50, builds hidden
  rows from the lm_head row of each target plus noise, reaches **99.0% of the cap** with 1.79% of logits past
  half the cap, and masks 1 target in 8 so a mis-handled ignore-index moves the mean ~12%. Call site is
  wrapped so any failure only prints.

## Not a cause — exonerated

The nine dead kernel variants (`mlp_fwd`, `pre_attn_fwd`, `post_attn_fwd`, `pre_attn_fwd_h`, `pre_attn_bwd`,
`embed_gather`, `embed_scatter_add`, `scatter_rounds`, `build_nki_layer`) are **all present in v52, which
passed review**. Dead code is not the differentiator, and cutting it means editing NKI plumbing that cannot
be tested with the box down. Disclosed in one sentence instead.

## Cost and proof

**+0.00024 bpb**, the temperature and nothing else. Three checkpoints, same eval tokens, same device: v79
equals v78-with-the-scale-disabled to **exact zero at nine digits** on all three. Training unaffected: 16
micro-step losses bit-identical in hex float over 4 real steps, with a same-file positive control. All 102
module-level constants compare equal, which is what makes the 44-constant fold provably semantics-free.

## The rule this cost us

**Diff every submission against the last file the portal ACCEPTED, not against the last file we built.**
Both causes were 26 versions of drift that no single change made look unreasonable. And `prepare.py`'s
`check_submission()` does not test any of it — it also does not enforce the 262,144-byte limit that four of
our versions silently exceeded.

## The checker built from this (2026-09-28)

`experimental/review_check.py` turns both causes into a gate. One command, ~2 s:

```
.venv/bin/python experimental/review_check.py submit/train.py      # defaults --ref to accepted v52
.venv/bin/python experimental/review_check.py --backtest           # every submission_v*.py, ~75 s
```

It checks the 262,144-byte limit, diffs every `os.environ` read against v52 and says which ones reach
the **scored forward**, and runs one structural rule in place of a keyword list: *a value read in the
scored forward whose writers are all on one path* (this is what catches `_eval_logit_scale`, and also
catches a renamed version of it that no grep would find). It then verifies the mechanically checkable
claims in the "Notes for review" docstring against the AST - env-var count, `head_logits` call-site
count, "no mode predicate", the claimed-inert constants and their guards, and the fused-check fail
direction. Findings the **accepted** v52 also has are marked `carried` and do not count.

Backtest: 0 new FAILs on v7/v33/v52 (all SCORED) and on v79; 2 on v12, 2 on v32, 3 on v78 (all
REJECTED_BY_REVIEW); v73-v76 caught on size. v32's top finding is `self.training` in `Block.forward`,
which is its actual rejection text.
