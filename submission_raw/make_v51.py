#!/usr/bin/env python3
"""Generate v51 = v49 + `head_fwd` drops the online maximum from the log-sum-exp.

The logits fed to the softmax are SOFTCAPPED: zc = s*tanh(z/s), so |zc| <= max(sp, sn) by construction. The
running max exists to keep exp() in range, but with a hard bound on zc it is unnecessary - subtract the constant
cap = max(sp, sn) instead. At the usual cap ~11 the exponent range is exp(-22)..exp(0), nowhere near fp32
underflow (min normal ~1e-38), and the smallest terms underflowing would be harmless in a SUM anyway.

Per vocab chunk this deletes a FULL-WIDTH `tensor_reduce(maximum)` over the logits plus six (128,1) instructions
(cmax, mnew, dm, exp(dm), s*=dm, negm, m=mnew). The tiny ops are not free: at 2048 (head, chunk, tile)
iterations they are ~12,000 instructions of pure issue overhead.

Mathematically identical: the code finishes with lse = log(s) + m, which becomes lse = log(s) + cap.

Run from the repo root:  .venv/bin/python submission_raw/make_v51.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v49_clean_compile_8way_headbwd.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v51_clean_compile_8way_nomax.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1, scope=None):
    """scope: a `def ` line prefix; the replacement is confined to that function's body."""
    global s
    if scope is None:
        n = s.count(old)
        assert n == expect, f"{name}: expected {expect}, found {n}"
        s = s.replace(old, new)
        applied[name] = n
        return
    lines = s.split("\n")
    st = next(i for i, l in enumerate(lines) if l.startswith(scope))
    en = next(i for i in range(st + 1, len(lines)) if lines[i].startswith("    def "))
    body = "\n".join(lines[st:en])
    n = body.count(old)
    assert n == expect, f"{name} (scoped): expected {expect}, found {n}"
    lines[st:en] = body.replace(old, new).split("\n")
    s = "\n".join(lines)
    applied[name] = n


# ---------------------------------------------------------------- hoist cap = max(sp, sn) and -cap
sub("cap hoisted", """        nisa.iota(dst=col, pattern=[[1, _HVC]], offset=0, channel_multiplier=0)""",
    """        nisa.iota(dst=col, pattern=[[1, _HVC]], offset=0, channel_multiplier=0)
        # |zc| <= max(sp, sn) because the logits are softcapped, so the log-sum-exp needs no running max:
        # subtracting this constant keeps exp() in range for every element (cap ~11 => exp(-22)..exp(0)).
        cap = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=cap, data1=sc[:, 0:1], data2=sc[:, 1:2], op=nl.maximum)
        negcap = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negcap, data=cap, op0=nl.multiply, operand0=-1.0)""", scope="    def head_fwd(")

# ---------------------------------------------------------------- no running max to initialise
sub("running max init removed", """            m = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)       # running max
            s = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)       # running sum of exp(zc - m)
            tl = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)      # target logit (capped)
            nisa.memset(m, -1e30)
            nisa.memset(s, 0.0)
            nisa.memset(tl, 0.0)""",
    """            s = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)       # running sum of exp(zc - cap)
            tl = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)      # target logit (capped)
            nisa.memset(s, 0.0)
            nisa.memset(tl, 0.0)""")

# ---------------------------------------------------------------- the per-chunk block
sub("online max deleted", """                # online logsumexp
                cmax = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=cmax, op=nl.maximum, data=zc, axis=(1,))
                mnew = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=mnew, data1=m, data2=cmax, op=nl.maximum)
                dm = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dm, data1=m, data2=mnew, op=nl.subtract)
                nisa.activation(dst=dm, op=nl.exp, data=dm)                                        # exp(m - mnew)
                nisa.tensor_tensor(dst=s, data1=s, data2=dm, op=nl.multiply)
                negm = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=negm, data=mnew, op0=nl.multiply, operand0=-1.0)
                e = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=e, op=nl.exp, data=zc, bias=negm)                              # exp(zc - mnew)
                es = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=es, op=nl.add, data=e, axis=(1,))
                nisa.tensor_tensor(dst=s, data1=s, data2=es, op=nl.add)
                nisa.tensor_copy(dst=m, src=mnew)""",
    """                # logsumexp against the CONSTANT cap: the softcap bounds |zc|, so no running max is needed.
                # This deletes a full-width tensor_reduce(maximum) and six (128,1) instructions per chunk.
                e = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=e, op=nl.exp, data=zc, bias=negcap)                            # exp(zc - cap)
                es = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=es, op=nl.add, data=e, axis=(1,))
                nisa.tensor_tensor(dst=s, data1=s, data2=es, op=nl.add)""")

# ---------------------------------------------------------------- finish with the constant
sub("lse uses cap", """            nisa.tensor_tensor(dst=lse, data1=lse, data2=m, op=nl.add)""",
    """            nisa.tensor_tensor(dst=lse, data1=lse, data2=cap, op=nl.add)""")

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:28s} x{v}")
