#!/usr/bin/env python3
"""Generate the clean v47 submission from submit/train.py (= v46) by applying the 2026-09-23 VectorE deletions
with the optimized path HARDCODED (the clean file carries no dev knobs).

Changes, all mathematically exact and simulator-verified (findings/vector-deletions-2026-09-23.md):
  1 mlp_bwd    : PSUM-group the dW1/dW2 matmuls over the 4 token blocks -> one VectorE add per group   (-5.6% step)
  2 norm sites : tensor_tensor(square) + tensor_reduce -> one nisa.activation_reduce
  3 rope       : skip the ROPE_FRAC-truncated columns, where cos == 1 and sin == 0 exactly (fwd + bwd)
  4 mlp        : forward stores relu(pre); backward reads it and squares it (deletes the max pass, adds no op)
  5 rmsnorm bwd: x*scalar then tensor-subtract -> one scalar_tensor_tensor(reverse1=True)
  6 pre_attn_bwd_ko : the same PSUM grouping for the Q/K/V weight gradients
"""
import re
import sys
from pathlib import Path

SRC = Path("submit/train.py")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v47_clean_compile_8way_vecdel.py")

s = SRC.read_text()
applied = {}


def sub_exact(name, old, new, expect=1, which=None):
    """Replace `old`. expect = how many occurrences must exist; which = 1-based index to replace (None = all)."""
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect} occurrences, found {n}"
    if which is None:
        s = s.replace(old, new)
        applied[name] = n
    else:
        i = -1
        for _ in range(which):
            i = s.index(old, i + 1)
        s = s[:i] + new + s[i + len(old):]
        applied[name] = 1


# ---------------------------------------------------------------- 1. mlp_bwd weight-grad PSUM grouping
sub_exact("mlp_bwd wgrad PSUM grouping", """            for tb in range(_FMAX // _NKI_P):
                dpn = nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
                hn = nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
                for h in range(ht):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=dpreT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P])
                    nisa.tensor_copy(dst=dpn[:, h * _NKI_P:(h + 1) * _NKI_P], src=tps)
                    tps2 = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps2, data=hT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P])
                    nisa.tensor_copy(dst=hn[:, h * _NKI_P:(h + 1) * _NKI_P], src=tps2)
                # dW1[h-tile, :] += dpre_nat[t-block][:, h-tile]^T x[t-block]
                for h in range(ht):
                    gps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=gps, stationary=dpn[:, h * _NKI_P:(h + 1) * _NKI_P], moving=xn[tb])
                    nisa.tensor_tensor(dst=A1[h], data1=A1[h], data2=gps, op=nl.add)
                # dW2[o-tile, h-chunk] += dy[t-block][:, o-tile]^T h_nat[t-block][:, h-chunk]
                for o in range(kt):
                    for hc in range(H // _FMAX):
                        gps2 = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=gps2, stationary=dyn[tb][:, o * _NKI_P:(o + 1) * _NKI_P], moving=hn[:, hc * _FMAX:(hc + 1) * _FMAX])
                        nisa.tensor_tensor(dst=A2[o][:, hc * _FMAX:(hc + 1) * _FMAX], data1=A2[o][:, hc * _FMAX:(hc + 1) * _FMAX], data2=gps2, op=nl.add)""",
"""            # The weight-gradient sums over token blocks are accumulated by the PE in PSUM, so each fp32 SBUF
            # accumulator takes ONE VectorE add per 512-token chunk instead of one per 128-token block. Same math
            # (fp32 accumulation either way); the backward is VectorE-bound, so this is worth ~5.6% of the step.
            dpn_l = []
            hn_l = []
            for tb in range(_FMAX // _NKI_P):
                dpn = nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
                hn = nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
                for h in range(ht):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=dpreT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P])
                    nisa.tensor_copy(dst=dpn[:, h * _NKI_P:(h + 1) * _NKI_P], src=tps)
                    tps2 = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps2, data=hT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P])
                    nisa.tensor_copy(dst=hn[:, h * _NKI_P:(h + 1) * _NKI_P], src=tps2)
                dpn_l.append(dpn)
                hn_l.append(hn)
            # dW1[h-tile, :] += sum over token blocks of dpre_nat[block][:, h-tile]^T x[block]
            for h in range(ht):
                gps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for i in range(_FMAX // _NKI_P):
                    nisa.nc_matmul(dst=gps, stationary=dpn_l[i][:, h * _NKI_P:(h + 1) * _NKI_P], moving=xn[i], accumulate=(i > 0))
                nisa.tensor_tensor(dst=A1[h], data1=A1[h], data2=gps, op=nl.add)
            # dW2[o-tile, h-chunk] += sum over token blocks of dy[block][:, o-tile]^T h_nat[block][:, h-chunk]
            for o in range(kt):
                for hc in range(H // _FMAX):
                    gps2 = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                    for i in range(_FMAX // _NKI_P):
                        nisa.nc_matmul(dst=gps2, stationary=dyn[i][:, o * _NKI_P:(o + 1) * _NKI_P],
                                       moving=hn_l[i][:, hc * _FMAX:(hc + 1) * _FMAX], accumulate=(i > 0))
                    nisa.tensor_tensor(dst=A2[o][:, hc * _FMAX:(hc + 1) * _FMAX], data1=A2[o][:, hc * _FMAX:(hc + 1) * _FMAX], data2=gps2, op=nl.add)""")

# ---------------------------------------------------------------- 2. square + row-sum -> activation_reduce
pat = re.compile(
    r"([ \t]+)nisa\.tensor_tensor\(dst=(\w+), data1=(\w+), data2=\3, op=nl\.multiply\)\n"
    r"([ \t]+)(\w+) = nl\.ndarray\(\(_NKI_P, 1\), dtype=nl\.float32, buffer=nl\.sbuf\)\n"
    r"[ \t]+nisa\.tensor_reduce\(dst=\5, op=nl\.add, data=\2, axis=\(1,\)\)")
s, n = pat.subn(
    lambda m: (f"{m.group(4)}{m.group(5)} = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)\n"
               f"{m.group(1)}# square and row-sum in one ScalarE instruction instead of two VectorE passes\n"
               f"{m.group(1)}nisa.activation_reduce(dst={m.group(2)}, op=nl.square, data={m.group(3)}, "
               f"reduce_op=nl.add, reduce_res={m.group(5)})"), s)
assert n >= 4, f"activation_reduce: only {n} sites"
applied["norm square+row-sum fusion"] = n

# ---------------------------------------------------------------- 3. rope: skip the truncated (identity) columns
ROPE_FWD_OLD = """                    x1 = src_t[:, c0:c0 + half]
                    x2 = src_t[:, c0 + half:c0 + HD]
                    r1 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)           # y1 = x1 cos + x2 sin
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)      # y2 = x2 cos - x1 sin
                    rot = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rot[:, 0:half], src=r1)
                    nisa.tensor_copy(dst=rot[:, half:HD], src=r2)"""
ROPE_FWD_NEW = """                    # ROPE_FRAC truncates inv_freq, so on the upper columns cos == 1 and sin == 0 EXACTLY and the
                    # rotation is the identity: rotate only the active columns and pass the rest straight through.
                    _rh = max(1, min(half, int(round(half * ROPE_FRAC))))
                    x1 = src_t[:, c0:c0 + _rh]
                    x2 = src_t[:, c0 + half:c0 + half + _rh]
                    r1 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    _cs = cs if _rh == half else cs[:, 0:_rh]
                    _sn = sn if _rh == half else sn[:, 0:_rh]
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)           # y1 = x1 cos + x2 sin
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)      # y2 = x2 cos - x1 sin
                    rot = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rot[:, 0:_rh], src=r1)
                    nisa.tensor_copy(dst=rot[:, half:half + _rh], src=r2)
                    if _rh < half:
                        nisa.tensor_copy(dst=rot[:, _rh:half], src=src_t[:, c0 + _rh:c0 + half])
                        nisa.tensor_copy(dst=rot[:, half + _rh:HD], src=src_t[:, c0 + half + _rh:c0 + HD])"""
sub_exact("rope forward skip", ROPE_FWD_OLD, ROPE_FWD_NEW, expect=s.count(ROPE_FWD_OLD), which=None)

ROPE_BWD_OLD = """                    x1 = dn[:, 0:half]
                    x2 = dn[:, half:HD]
                    r1 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)
                    if j == 0:
                        nisa.tensor_copy(dst=dpq[:, c0:c0 + half], src=r1)
                        nisa.tensor_copy(dst=dpq[:, c0 + half:c0 + HD], src=r2)
                    else:
                        nisa.tensor_copy(dst=dpk[:, c0:c0 + half], src=r1)
                        nisa.tensor_copy(dst=dpk[:, c0 + half:c0 + HD], src=r2)"""
ROPE_BWD_NEW = """                    _rh = max(1, min(half, int(round(half * ROPE_FRAC))))
                    x1 = dn[:, 0:_rh]
                    x2 = dn[:, half:half + _rh]
                    r1 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    _cs = cs if _rh == half else cs[:, 0:_rh]
                    _sn = sn if _rh == half else sn[:, 0:_rh]
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)
                    _dd = dpq if j == 0 else dpk
                    nisa.tensor_copy(dst=_dd[:, c0:c0 + _rh], src=r1)
                    nisa.tensor_copy(dst=_dd[:, c0 + half:c0 + half + _rh], src=r2)
                    if _rh < half:   # identity on the truncated frequencies, in the transpose too
                        nisa.tensor_copy(dst=_dd[:, c0 + _rh:c0 + half], src=dn[:, _rh:half])
                        nisa.tensor_copy(dst=_dd[:, c0 + half + _rh:c0 + HD], src=dn[:, half + _rh:HD])"""
sub_exact("rope backward skip", ROPE_BWD_OLD, ROPE_BWD_NEW, expect=s.count(ROPE_BWD_OLD), which=None)

# ---------------------------------------------------------------- 4. store relu(pre) instead of pre
sub_exact("post_attn_fwd_h stores relu", """                pre = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pre, src=pps)
                nisa.dma_copy(dst=pre_out[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX], src=pre)
                nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)""",
"""                # relu on ScalarE straight off PSUM (drops the PSUM->SBUF VectorE copy), then one VectorE square.
                # What is saved for the backward is relu(pre), NOT pre: the backward needs relu and relu^2, and
                # both come from relu with no extra op, so a full VectorE pass over (H, T) disappears there too.
                rl = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=rl, op=nl.relu, data=pps)
                nisa.tensor_tensor(dst=hT[h], data1=rl, data2=rl, op=nl.multiply)
                nisa.dma_copy(dst=pre_out[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX], src=rl)""", expect=3, which=None)

sub_exact("mlp_bwd reads relu", """                pre = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=pre, src=pre_in[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX])
                relu = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=relu, data=pre, op0=nl.maximum, operand0=0.0)
                hh = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=hh, data1=relu, data2=relu, op=nl.multiply)      # h = relu^2 (recomputed)""",
"""                relu = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=relu, src=pre_in[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX])   # relu(pre), saved by the forward
                hh = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=hh, data1=relu, data2=relu, op=nl.multiply)      # h = relu^2""", expect=1)

# ---------------------------------------------------------------- 5. rmsnorm-backward tail fusion
for nm, (old, new) in {
    "stt fusion (tt/r/cc)": ("""                    nisa.tensor_scalar(dst=tt, data=r, op0=nl.multiply, operand0=cc)
                    nisa.tensor_tensor(dst=tt, data1=g, data2=tt, op=nl.subtract)""",
                             """                    nisa.scalar_tensor_tensor(dst=tt, data=r, op0=nl.multiply, operand0=cc, op1=nl.subtract, operand1=g, reverse1=True)"""),
    "stt fusion (sq/xm/cm)": ("""            nisa.tensor_scalar(dst=sq, data=xm, op0=nl.multiply, operand0=cm)
            nisa.tensor_tensor(dst=sq, data1=dh, data2=sq, op=nl.subtract)""",
                              """            nisa.scalar_tensor_tensor(dst=sq, data=xm, op0=nl.multiply, operand0=cm, op1=nl.subtract, operand1=dh, reverse1=True)"""),
}.items():
    sub_exact(nm, old, new, expect=s.count(old), which=None)

# ---------------------------------------------------------------- 6. pre_attn_bwd_ko Q/K/V wgrad PSUM grouping
PRE_WG_OLD = """            for o in range(kt):
                wps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=wps, stationary=dpq[:, o * _NKI_P:(o + 1) * _NKI_P], moving=h)
                nisa.tensor_tensor(dst=accq[o], data1=accq[o], data2=wps, op=nl.add)
                wps2 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=wps2, stationary=dpk[:, o * _NKI_P:(o + 1) * _NKI_P], moving=h)
                nisa.tensor_tensor(dst=acck[o], data1=acck[o], data2=wps2, op=nl.add)
                wps3 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=wps3, stationary=gvb[:, o * _NKI_P:(o + 1) * _NKI_P], moving=h)
                nisa.tensor_tensor(dst=accv[o], data1=accv[o], data2=wps3, op=nl.add)"""
PRE_WG_NEW = """            _wbuf.append((dpq, dpk, gvb, h))
            if len(_wbuf) >= 4 or tbi == tiles - 1:
                # as in mlp_bwd: the PE sums the group's token tiles in PSUM, so each accumulator takes one
                # VectorE add per group of 4 tiles instead of one per tile. Exact (fp32 PSUM accumulation).
                for o in range(kt):
                    wps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    for _i, _e in enumerate(_wbuf):
                        nisa.nc_matmul(dst=wps, stationary=_e[0][:, o * _NKI_P:(o + 1) * _NKI_P], moving=_e[3], accumulate=(_i > 0))
                    nisa.tensor_tensor(dst=accq[o], data1=accq[o], data2=wps, op=nl.add)
                    wps2 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    for _i, _e in enumerate(_wbuf):
                        nisa.nc_matmul(dst=wps2, stationary=_e[1][:, o * _NKI_P:(o + 1) * _NKI_P], moving=_e[3], accumulate=(_i > 0))
                    nisa.tensor_tensor(dst=acck[o], data1=acck[o], data2=wps2, op=nl.add)
                    wps3 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    for _i, _e in enumerate(_wbuf):
                        nisa.nc_matmul(dst=wps3, stationary=_e[2][:, o * _NKI_P:(o + 1) * _NKI_P], moving=_e[3], accumulate=(_i > 0))
                    nisa.tensor_tensor(dst=accv[o], data1=accv[o], data2=wps3, op=nl.add)
                _wbuf = []"""
nwg = s.count(PRE_WG_OLD)
assert nwg == 2, f"pre wgrad: found {nwg}"
sub_exact("pre_attn_bwd_ko wgrad PSUM grouping", PRE_WG_OLD, PRE_WG_NEW, expect=2, which=2)

# declare the group buffer ahead of pre_attn_bwd_ko's token-tile loop
lines = s.split("\n")
st = next(i for i, l in enumerate(lines) if l.startswith("    def pre_attn_bwd_ko"))
en = next(i for i in range(st + 1, len(lines)) if lines[i].startswith("    def "))
li = max(i for i in range(st, en) if lines[i] == "        for tbi in range(tiles):")
lines.insert(li, "        _wbuf = []")
s = "\n".join(lines)
applied["pre_attn_bwd_ko group buffer"] = 1

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:38s} x{v}")
