"""Freeze train3_ck.py into submission_v73_clean_ck.py. Run ONLY if the CK full gate wins.

Two changes, both deliberate:

1. `CK_W` defaults 0 -> 3, so the in-kernel causal depthwise conv is ON without an env var. This is the mechanism
   measured at **-0.013664 at a matched 600 steps** (1.1077839 vs 1.1214480), implemented inside
   pre_attn_fwd_ko/pre_attn_bwd_ko with ZERO added kernel launches (per-tap doc masks ride as extra columns of the
   existing dstart tensor, conv weights as extra columns of the existing scal row, and the weight gradient returns
   via dscal row 0 into the flat AdamW group's existing all_reduce). Bit-exact against the torch reference in the
   simulator at b16 x t1024, forward and both gradients.

2. The RMSNorm epsilon fix, worth **-0.000065 exact** (measured offline on v72's weights): training normalises
   inside the kernels with finfo(bf16).eps = 0.0078125, but the scored `head_logits` calls F.rms_norm with NO eps
   and x is fp32-typed (bigram_lambda promotes the residual stream), so the scorer silently used 1.19e-7 - a
   65,000x mismatch. Pinning norm() to the kernels' eps changes NO training math on Neuron (training goes through
   the layer ops and head_fwd, which never call norm()); it only aligns the eval path and the torch fallbacks, and
   makes the module docstring's "the two are the same function of the same weights" true.

usage: .venv/bin/python submission_raw/make_v73.py
"""
import hashlib
import pathlib

src = pathlib.Path("submission_raw/train3_ck.py").read_text()
n0 = len(src)

old = 'CK_W = int(os.environ.get("NEURON_COMPETITION_R1_CK", "0"))'
assert src.count(old) == 1, "CK_W knob not found exactly once"
src = src.replace(old, 'CK_W = int(os.environ.get("NEURON_COMPETITION_R1_CK", "3"))   # v73: ON by default')

old = """def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))"""
assert src.count(old) == 1, "norm() not found exactly once"
src = src.replace(old, """def norm(x: torch.Tensor) -> torch.Tensor:
    # eps pinned to what the KERNELS use (finfo(bf16).eps). Without it F.rms_norm takes finfo(x.dtype).eps, and x
    # is fp32-typed here, so the scored path silently normalised with 1.19e-7 against the kernels' 0.0078125 -
    # a 65,000x train/eval mismatch worth -0.000065, measured exactly offline. Changes no training math on Neuron.
    return F.rms_norm(x, (x.size(-1),), eps=0.0078125)""")

out = pathlib.Path("submission_raw/submission_v73_clean_ck.py")
out.write_text(src)
sha = hashlib.sha256(src.encode()).hexdigest()[:20]
print(f"wrote {out} ({n0} -> {len(src)} bytes)  sha256[:20]={sha}")
print("next: prepare.py check, then copy byte-identically to submit/train.py")
