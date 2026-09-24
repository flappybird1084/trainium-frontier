#!/usr/bin/env python3
"""Generate v50 = v49 + `post_attn_fwd_h` also emitting the transposed attention output (kills `tr(out)`).

The vendor attention backward wants [b, h, d, seq], so the backward paid a full ~16.8 MB
`.transpose(-2,-1).contiguous()` of `out` per layer (12 per micro-step; the profile attributed ~23 ms of the
200 ms backward to the eight such transposes). But `post_attn_fwd_h` ALREADY builds exactly those transposed
tiles: for the out-projection it does, per 128-token tile and per column block i,
    nc_transpose(yt[:, i*128:(i+1)*128]) -> PSUM -> tensor_copy -> yTi -> nc_matmul
and because HD == _NKI_P == 128 and kt == n_head, block i IS head i. So `yTi` becomes a slice of a per-head
[HD, t_len] staging row, the existing `tensor_copy` fills it, and the row is flushed with ONE contiguous store.
Zero added compute, zero extra HBM bytes (outT replaces the saved `out`), and no strided kernel-output store
(writing outT per tile would be a strided output DMA - the pattern that produced NaNs here historically).

Run from the repo root:  .venv/bin/python submission_raw/make_v50.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v49_clean_compile_8way_headbwd.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v50_clean_compile_8way_outT.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1, nth=None):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    if nth is None:
        s = s.replace(old, new)
    else:
        i = -1
        for _ in range(nth):
            i = s.index(old, i + 1)
        s = s[:i] + new + s[i + len(old):]
    applied[name] = n if nth is None else 1


# ---------------------------------------------------------------- the extra HBM output
sub("outT output", """        y2_out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)     # token-major copy of y for the backward (dWo)""",
    """        y2_out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)     # token-major copy of y for the backward (dWo)
        outT_out = nl.ndarray((y_in.shape[0], HD, t_len), dtype=y_in.dtype, buffer=nl.shared_hbm)   # y^T per head: the vendor bwd's [b, h, d, seq]""")

# ---------------------------------------------------------------- per-head staging rows
sub("outT staging rows", """        for c in range(chunks):                       # static: tile rows/positions known at trace time""",
    """        # per-head [HD, t_len] staging rows: the yT tiles the out-projection already builds ARE the transposed
        # attention output (kt == n_head and HD == _NKI_P), so outT costs one contiguous store per (row, head)
        oTb = []
        for _j in range(kt):
            oTb.append(nl.ndarray((_NKI_P, t_len), dtype=nl.bfloat16, buffer=nl.sbuf))
        for c in range(chunks):                       # static: tile rows/positions known at trace time""")

# ---------------------------------------------------------------- yTi becomes a slice of the staging row
sub("outT fill + flush", """                for i in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=yt[:, i * _NKI_P:(i + 1) * _NKI_P])
                    yTi = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=yTi, src=tps)
                    nisa.nc_matmul(dst=aps, stationary=yTi, moving=WoT[i], accumulate=(i > 0))""",
    """                for i in range(kt):               # kt == n_head and HD == _NKI_P: column block i IS head i
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=yt[:, i * _NKI_P:(i + 1) * _NKI_P])
                    yTi = oTb[i][:, p0:p0 + _NKI_P]   # the [HD, 128 tokens] tile lands straight in head i's staging row
                    nisa.tensor_copy(dst=yTi, src=tps)
                    nisa.nc_matmul(dst=aps, stationary=yTi, moving=WoT[i], accumulate=(i > 0))
                    if p0 + _NKI_P == t_len:          # row complete: ONE contiguous (HD, t_len) store
                        nisa.dma_copy(dst=outT_out[row * n_head + i, :, :], src=oTb[i])""", expect=2, nth=1)

sub("post_attn_fwd_h return", """        return out, x1_out, pre_out, y2_out""",
    """        return out, x1_out, pre_out, y2_out, outT_out""")

# ---------------------------------------------------------------- plumbing
sub("post_kh unpack", """        o, x1, preT, y2 = post_kh[grid](out.reshape(b * NH, t, HD), wo, xm, w1, w2, eps=eps, n_head=NH, t_len=t)""",
    """        o, x1, preT, y2, outT = post_kh[grid](out.reshape(b * NH, t, HD), wo, xm, w1, w2, eps=eps, n_head=NH, t_len=t)""")

sub("layer_fwd return", """        return o, xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT""",
    """        return o, xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT""")

sub("setup_context", """        o, xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT = output""",
    """        o, xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT = output""")
sub("setup_context save", """                              xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT)""",
    """                              xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT)""")
sub("backward unpack", """         xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT) = ctx.saved_tensors""",
    """         xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT) = ctx.saved_tensors""")
sub("backward call", """            dout2, x2, x02, ve2, xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT, cos2d, sin2d, sink, start, end,""",
    """            dout2, x2, x02, ve2, xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT, cos2d, sin2d, sink, start, end,""")
sub("layer_bwd signature", """                  rk: torch.Tensor, q4: torch.Tensor, k4: torch.Tensor, v4: torch.Tensor, out: torch.Tensor, neg_max: torch.Tensor,""",
    """                  rk: torch.Tensor, q4: torch.Tensor, k4: torch.Tensor, v4: torch.Tensor, outT: torch.Tensor, neg_max: torch.Tensor,""")

# layer_bwd calls the LOCAL attn_bwd helper in the clean file (the vehicle uses a registered op instead -
# checking which implementation is live before editing is now a standing rule)
sub("attn_bwd call", """        dq4, dk4, dv4, dsink = attn_bwd(q4, k4, v4, out, dy4, neg_max, recip, sink, start, end)""",
    """        dq4, dk4, dv4, dsink = attn_bwd(q4, k4, v4, outT, dy4, neg_max, recip, sink, start, end)""")
sub("attn_bwd signature", """    def attn_bwd(q4, k4, v4, out, dy4, neg_max, recip, sink, start, end):""",
    """    def attn_bwd(q4, k4, v4, outT, dy4, neg_max, recip, sink, start, end):""")
# outT is contiguous (b*h, HD, t); .view(b, h, HD, t) has exactly the strides tr(out) had. Occurrence 2 is the
# live attn_bwd (occurrence 1 is in the unused doc_attention_bwd kernel).
sub("drop tr(out)", """tr(out), """, """outT.view(b, h, d, t), """, expect=2, nth=2)

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:28s} x{v}")
