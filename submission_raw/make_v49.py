#!/usr/bin/env python3
"""Generate v49 = v48 + the 2026-09-23 (evening) VectorE deletions, hardcoded.

1 head_fwd / head_logits_fwd: delete the logits PSUM->SBUF copy (its three consumers each take a single PSUM
  operand) and the per-tile `col + c*_HVC` recompute (shift the (128,1) target instead; `col` is loop-invariant).
2 head_bwd: the softcap is s*tanh(z/s) with s selected by sign(z). Select the SCALAR first and take ONE tanh
  instead of computing both branches over the whole logits tensor and blending. The softcap-parameter gradient
  then follows from a single expression m = pr*th - zs*dzf, with dsp = rowsum(pos*m), dsn = rowsum(m) - dsp;
  the 1/sp and 1/sn factors cancel exactly. 32 VectorE passes -> 19. Algebra verified in float64 (1e-16).
  Measured: 650 steps vs 642 in a paired 300 s screen (+1.25%).
3 gather_params: all_gather the sharded parameters in bf16 (both the padded and the plain path). The graph casts
  weights to bf16 anyway and the fp32 master stays on the owning shard, so this is numerically free; it halves
  the bytes of the single largest collective (the 262144-row bigram table). Measured +0.6% on the vehicle, and
  the clean file gains more because its padded path was still fp32.

Run from the repo root:  .venv/bin/python submission_raw/make_v49.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v48_clean_compile_8way_scat4.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v49_clean_compile_8way_headbwd.py")

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


# ---------------------------------------------------------------- 1a. drop the logits PSUM->SBUF copy
sub("head logits copy deleted", """                z = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=z, src=zps)""",
    """                z = zps          # three consumers below, each taking a single PSUM operand: the copy is not needed""",
    expect=1)

# ---------------------------------------------------------------- 1b. shift the target, not the column indices
sub("head col recompute deleted", """                cm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=cm, data=col, op0=nl.add, operand0=float(c * _HVC))
                eqm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eqm, data=cm, op0=nl.equal, operand0=tg)""",
    """                # col + c*_HVC == tg  <=>  col == tg - c*_HVC, and `col` is loop-invariant (one iota), so
                # shifting the (128, 1) target kills a full (128, _HVC) VectorE pass per token tile per chunk
                tgc = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=tgc, data=tg, op0=nl.subtract, operand0=float(c * _HVC))
                eqm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eqm, data=col, op0=nl.equal, operand0=tgc)""",
    expect=1)

# ---------------------------------------------------------------- 2. head_bwd: one tanh + algebraic dcap
sub("head_bwd hoisted constants", """        nisa.reciprocal(dst=inv_sn, data=sc[:, 1:2])""",
    """        nisa.reciprocal(dst=inv_sn, data=sc[:, 1:2])
        d_inv = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d_inv, data1=inv_sp, data2=inv_sn, op=nl.subtract)
        d_s = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d_s, data1=sc[:, 0:1], data2=sc[:, 1:2], op=nl.subtract)""", expect=1, scope="    def head_bwd(")

OLD_BWD = """                    z = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=z, src=zps)
                    tp = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    tn = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=tp, op=nl.tanh, data=z, scale=inv_sp)
                    nisa.activation(dst=tn, op=nl.tanh, data=z, scale=inv_sn)
                    pos = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=pos, data=z, op0=nl.greater_equal, operand0=0.0)"""
NEW_BWD = """                    # The softcap is s*tanh(z/s) with s = sp where z >= 0 and sn elsewhere. Selecting the SCALAR
                    # first costs two (128,1)-operand passes and saves a second tanh and eight blend passes over
                    # the whole logits tile: invs = inv_sn + pos*(inv_sp - inv_sn), sv = sn + pos*(sp - sn).
                    pos = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=pos, data=zps, op0=nl.greater_equal, operand0=0.0)
                    invs = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=invs, data=pos, op0=nl.multiply, operand0=d_inv)
                    nisa.tensor_scalar(dst=invs, data=invs, op0=nl.add, operand0=inv_sn)
                    zs = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=zs, data1=zps, data2=invs, op=nl.multiply)
                    th = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=th, op=nl.tanh, data=zs)
                    sv = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=sv, data=pos, op0=nl.multiply, operand0=d_s)
                    nisa.tensor_scalar(dst=sv, data=sv, op0=nl.add, operand0=sc[:, 1:2])"""
sub("head_bwd single tanh", OLD_BWD, NEW_BWD, expect=1, scope="    def head_bwd(")

OLD_TAIL = """                    a = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    b = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=a, data=tp, op0=nl.multiply, operand0=sc[:, 0:1])
                    nisa.tensor_scalar(dst=b, data=tn, op0=nl.multiply, operand0=sc[:, 1:2])
                    zc = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=zc, data1=a, data2=b, op=nl.subtract)
                    nisa.tensor_tensor(dst=zc, data1=zc, data2=pos, op=nl.multiply)
                    nisa.tensor_tensor(dst=zc, data1=zc, data2=b, op=nl.add)
                    nisa.tensor_tensor(dst=a, data1=tp, data2=tp, op=nl.multiply)
                    nisa.tensor_tensor(dst=b, data1=tn, data2=tn, op=nl.multiply)
                    deriv = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=deriv, data1=a, data2=b, op=nl.subtract)
                    nisa.tensor_tensor(dst=deriv, data1=deriv, data2=pos, op=nl.multiply)
                    nisa.tensor_tensor(dst=deriv, data1=deriv, data2=b, op=nl.add)                 # tn^2 + pos*(tp^2 - tn^2)
                    nisa.tensor_scalar(dst=deriv, data=deriv, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)"""
NEW_TAIL = """                    zc = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=zc, data1=th, data2=sv, op=nl.multiply)
                    deriv = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=deriv, data1=th, data2=th, op=nl.multiply)
                    nisa.tensor_scalar(dst=deriv, data=deriv, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)"""
sub("head_bwd blend removed", OLD_TAIL, NEW_TAIL, expect=1, scope="    def head_bwd(")

OLD_DCAP = """                    dz = nl.ndarray((_NKI_P, _HVC), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=dz, data1=pr, data2=deriv, op=nl.multiply)
                    # softcap parameter grads: d zc/d sp = tp - (z/sp)(1 - tp^2) on z >= 0; d zc/d sn = tn - (z/sn)(1 - tn^2) on z < 0
                    u = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=u, data=a, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)   # 1 - tp^2
                    nisa.tensor_tensor(dst=u, data1=u, data2=z, op=nl.multiply)
                    nisa.tensor_scalar(dst=u, data=u, op0=nl.multiply, operand0=inv_sp)
                    nisa.tensor_tensor(dst=u, data1=tp, data2=u, op=nl.subtract)
                    nisa.tensor_tensor(dst=u, data1=u, data2=pos, op=nl.multiply)
                    nisa.tensor_tensor(dst=u, data1=u, data2=pr, op=nl.multiply)
                    part = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=part, op=nl.add, data=u, axis=(1,))
                    nisa.tensor_tensor(dst=dcap[:, 0:1], data1=dcap[:, 0:1], data2=part, op=nl.add)
                    nisa.tensor_scalar(dst=u, data=b, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)   # 1 - tn^2
                    nisa.tensor_tensor(dst=u, data1=u, data2=z, op=nl.multiply)
                    nisa.tensor_scalar(dst=u, data=u, op0=nl.multiply, operand0=inv_sn)
                    nisa.tensor_tensor(dst=u, data1=tn, data2=u, op=nl.subtract)
                    nisa.tensor_scalar(dst=pos, data=pos, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)  # 1 - pos
                    nisa.tensor_tensor(dst=u, data1=u, data2=pos, op=nl.multiply)
                    nisa.tensor_tensor(dst=u, data1=u, data2=pr, op=nl.multiply)
                    nisa.tensor_reduce(dst=part, op=nl.add, data=u, axis=(1,))
                    nisa.tensor_tensor(dst=dcap[:, 1:2], data1=dcap[:, 1:2], data2=part, op=nl.add)"""
NEW_DCAP = """                    dzf = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=dzf, data1=pr, data2=deriv, op=nl.multiply)
                    dz = nl.ndarray((_NKI_P, _HVC), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(dst=dz, op=nl.copy, data=dzf)      # fp32 -> bf16 on ScalarE (dst is a fresh tile)
                    # m = pr*th - zs*dzf is the d/dsp integrand where z >= 0 and the d/dsn integrand where z < 0
                    # (the 1/sp and 1/sn factors cancel identically), so dsp = rowsum(pos*m), dsn = rowsum(m) - dsp.
                    mm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=mm, data1=pr, data2=th, op=nl.multiply)
                    uu = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=uu, data1=zs, data2=dzf, op=nl.multiply)
                    nisa.tensor_tensor(dst=mm, data1=mm, data2=uu, op=nl.subtract)
                    nisa.tensor_tensor(dst=uu, data1=mm, data2=pos, op=nl.multiply)
                    partA = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    partB = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=partA, op=nl.add, data=uu, axis=(1,))
                    nisa.tensor_reduce(dst=partB, op=nl.add, data=mm, axis=(1,))
                    nisa.tensor_tensor(dst=dcap[:, 0:1], data1=dcap[:, 0:1], data2=partA, op=nl.add)
                    nisa.tensor_tensor(dst=partB, data1=partB, data2=partA, op=nl.subtract)
                    nisa.tensor_tensor(dst=dcap[:, 1:2], data1=dcap[:, 1:2], data2=partB, op=nl.add)"""
sub("head_bwd algebraic dcap", OLD_DCAP, NEW_DCAP, expect=1, scope="    def head_bwd(")

# ---------------------------------------------------------------- 3. bf16 parameter gather (both paths)
sub("bf16 padded gather", """                    spad = plan["spad"] = torch.zeros((plan["per"],) + tuple(p.shape[1:]), dtype=p.dtype, device=p.device)
                    plan["fpad"] = torch.empty((plan["npad"],) + tuple(p.shape[1:]), dtype=p.dtype, device=p.device)""",
    """                    # bf16 all_gather: the graph casts every weight to bf16 anyway and the fp32 master stays on the
                    # owning shard, so gathering the bf16 rounding is numerically the same and halves the bytes
                    spad = plan["spad"] = torch.zeros((plan["per"],) + tuple(p.shape[1:]), dtype=torch.bfloat16, device=p.device)
                    plan["fpad"] = torch.empty((plan["npad"],) + tuple(p.shape[1:]), dtype=torch.bfloat16, device=p.device)""",
    expect=1)

sub("bf16 plain gather", """            dist.all_gather_into_tensor(p.data, plan["pshard"])""",
    """            p16 = plan.get("p16")
            if p16 is None:
                p16 = plan["p16"] = torch.empty_like(p.data, dtype=torch.bfloat16)
            dist.all_gather_into_tensor(p16, plan["pshard"].to(torch.bfloat16))
            p.data.copy_(p16)""", expect=1)

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:30s} x{v}")
