#!/usr/bin/env python3
"""Key-offset glue folded into the layer kernels (2026-09-16, "hard queue" step 1).

KEY_OFFSET: the normed key dims hq:2hq (hq = HD/4) of position p come from position p-1, zeroed where p starts a document
(dstart[p] == 1). Until now the vehicle did this with 5 torch ops on k4 after `pre_attn_fwd_h` and 4 ops on dk4 before
`pre_attn_bwd` (the dksh slice), per layer and direction: pure launch/DMA overhead in a latency-bound step.

  pre_attn_fwd_ko = pre_attn_fwd_h + the shift inside the kernel: k_out[bh, p, hq:2hq] = k_normed[p-1] * (1 - dstart[p]).
  pre_attn_bwd_ko = pre_attn_bwd with dksh_in replaced by dstart_in: g[p, hq:2hq] = dk[p+1, hq:2hq] * (1 - dstart[p+1]).

The partition (row) shift is done on the PE with constant 0/1 shift matrices built in-kernel (affine_select), so every
DMA stays a plain 128-row-aligned block (tracer rules #8/#9/#11: no unaligned, partial or strided-pattern DMAs in-graph):
  fwd: dst = S @ tile + E @ prev_tile,  S[k, m] = 1 iff m == k+1 (row k moves to row k+1), E[k, m] = 1 iff k == P-1, m == 0
       (the last row of the previous 128-row tile of the same sequence row becomes row 0; zero at the sequence start).
  bwd: tiles are visited in reverse order; dst = S^T @ masked_tile + E^T @ masked_next_tile (row k+1 moves to row k; the
       first row of the next tile becomes row P-1; zero at the end of the sequence row).
A bf16 x {0, 1} product accumulated in fp32 is exact, so the shifted values are bit-identical to the torch glue.
dstart_in: (T, 1) fp32 = the vehicle's (b, t) doc-start tensor viewed flat (a free view). Everything else as in the
originals (nki_layout_test.py / vehicle `pre_attn_fwd_h`, nki_pre_bwd.py `pre_attn_bwd`). Vehicle copy: train3_ko.py."""
import torch
import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def pre_attn_fwd_ko(x_in: nl.ndarray, x0_in: nl.ndarray, ve_in: nl.ndarray, wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray,
                    cos_in: nl.ndarray, sin_in: nl.ndarray, scal: nl.ndarray, dstart_in: nl.ndarray, n_head: int = 4,
                    eps: float = 1e-6, t_len: int = 1024, q_scale: float = 1.0, hq: int = 32):
    """pre_attn_fwd_h (see the vehicle) with the KEY_OFFSET shift applied to k_out inside the kernel (dstart_in (T, 1) fp32).
    q_out/k_out/v_out (b*n_head, t_len, HD) bf16 (q pre-scaled, k offset), xm (T, D), rq/rk (n_head, T, HD) unshifted."""
    T, D = x_in.shape
    HD = D // n_head
    half = HD // 2
    q_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    v_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    tpt = t_len // P
    xm_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    rq_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    rk_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    kt = D // P
    tiles = T // P
    sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=scal)
    # key-offset shift matrices (0/1 bf16, built once): S[k, m] = 1 iff m == k + 1 ; E[k, m] = 1 iff k == P-1 and m == 0
    ones = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    Sm = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.affine_select(dst=Sm, pattern=[[1, P]], channel_multiplier=-1, offset=-1, on_true_tile=ones, on_false_value=0.0)
    Em = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.affine_select(dst=Em, pattern=[[1, P]], channel_multiplier=-1, offset=P - 1, on_true_tile=ones, on_false_value=0.0)
    kprev = []                                    # per head: unshifted normed k dims hq:2hq of the previous tile
    for hh in range(n_head):
        kp = nl.ndarray((P, hq), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(kp, 0.0)
        kprev.append(kp)
    # resident transposed weights: WqT[i] = [128 (in i), D (out)], same for k, v
    WqT = []
    WkT = []
    WvT = []
    for _j in range(kt):
        WqT.append(nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        WkT.append(nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        WvT.append(nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
    for o in range(kt):
        bq = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        bk = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        bv = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=bq, src=wq[o * P:(o + 1) * P, :])
        nisa.dma_copy(dst=bk, src=wk[o * P:(o + 1) * P, :])
        nisa.dma_copy(dst=bv, src=wv[o * P:(o + 1) * P, :])
        for i in range(kt):
            tq = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tq, data=bq[:, i * P:(i + 1) * P])
            nisa.tensor_copy(dst=WqT[i][:, o * P:(o + 1) * P], src=tq)
            tk = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tk, data=bk[:, i * P:(i + 1) * P])
            nisa.tensor_copy(dst=WkT[i][:, o * P:(o + 1) * P], src=tk)
            tv = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tv, data=bv[:, i * P:(i + 1) * P])
            nisa.tensor_copy(dst=WvT[i][:, o * P:(o + 1) * P], src=tv)
    for tb in range(tiles):                       # static: the (row, position) of each tile is known at trace time
        t0 = tb * P
        row = tb // tpt
        p0 = (tb - row * tpt) * P
        x = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        x0 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x, src=x_in[t0:t0 + P, :])
        nisa.dma_copy(dst=x0, src=x0_in[t0:t0 + P, :])
        xm = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xm, data=x0, op0=nl.multiply, operand0=sc[:, 1:2])
        nisa.scalar_tensor_tensor(dst=xm, data=x, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=xm)
        xmb = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xmb, src=xm)
        nisa.dma_copy(dst=xm_out[t0:t0 + P, :], src=xmb)
        # rmsnorm
        sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=xm, data2=xm, op=nl.multiply)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
        h = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
        # hT tiles for the projections
        hT = []
        for i in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=h[:, i * P:(i + 1) * P])
            hTi = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hTi, src=tps)
            hT.append(hTi)
        qps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
        kps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
        vps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
        for i in range(kt):
            nisa.nc_matmul(dst=qps, stationary=hT[i], moving=WqT[i], accumulate=(i > 0))
            nisa.nc_matmul(dst=kps, stationary=hT[i], moving=WkT[i], accumulate=(i > 0))
            nisa.nc_matmul(dst=vps, stationary=hT[i], moving=WvT[i], accumulate=(i > 0))
        q = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        k = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        v = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q, src=qps)
        nisa.tensor_copy(dst=k, src=kps)
        ve = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ve, src=ve_in[t0:t0 + P, :])
        nisa.scalar_tensor_tensor(dst=v, data=ve, op0=nl.multiply, operand0=sc[:, 2:3], op1=nl.add, operand1=vps)
        cs = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
        sn = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs, src=cos_in[t0:t0 + P, :])
        nisa.dma_copy(dst=sn, src=sin_in[t0:t0 + P, :])
        # key-offset mask of these rows: 1 - dstart
        ds = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds, src=dstart_in[t0:t0 + P, :])
        mask = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mask, data=ds, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        for hh in range(n_head):
            c0 = hh * HD
            for j in range(2):
                src_t = q if j == 0 else k
                dst_t = q_out if j == 0 else k_out
                rot_t = rq_out if j == 0 else rk_out
                x1 = src_t[:, c0:c0 + half]
                x2 = src_t[:, c0 + half:c0 + HD]
                r1 = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
                r2 = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
                tmp = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=r1, data1=x1, data2=cs, op=nl.multiply)
                nisa.tensor_tensor(dst=tmp, data1=x2, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)           # y1 = x1 cos + x2 sin
                nisa.tensor_tensor(dst=r2, data1=x2, data2=cs, op=nl.multiply)
                nisa.tensor_tensor(dst=tmp, data1=x1, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)      # y2 = x2 cos - x1 sin
                rot = nl.ndarray((P, HD), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rot[:, 0:half], src=r1)
                nisa.tensor_copy(dst=rot[:, half:HD], src=r2)
                rb = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rb, src=rot)
                nisa.dma_copy(dst=rot_t[hh, t0:t0 + P, :], src=rb)
                sq2 = nl.ndarray((P, HD), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sq2, data1=rot, data2=rot, op=nl.multiply)
                ms2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=ms2, op=nl.add, data=sq2, axis=(1,))
                rs2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=rs2, data=ms2, op0=nl.multiply, operand0=1.0 / HD, op1=nl.add, operand1=eps)
                nisa.activation(dst=rs2, op=nl.rsqrt, data=rs2)
                ob = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                if j == 0:
                    nisa.tensor_scalar(dst=ob, data=rot, op0=nl.multiply, operand0=rs2, op1=nl.multiply, operand1=q_scale)
                    nisa.dma_copy(dst=dst_t[row * n_head + hh, p0:p0 + P, :], src=ob)
                else:
                    nisa.tensor_scalar(dst=ob, data=rot, op0=nl.multiply, operand0=rs2)
                    # key offset: dims hq:2hq of row p <- normed k of row p-1 (row 0 <- last row of the previous tile), masked
                    # (two PSUM tiles summed on VectorE: the compiler rejects mixed matmul accumulate flags, NCC_ILMM003)
                    ps = nl.ndarray((P, hq), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=ps, stationary=Sm, moving=ob[:, hq:2 * hq])
                    ok = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ok, src=ob)
                    if p0 > 0:
                        ps2 = nl.ndarray((P, hq), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=ps2, stationary=Em, moving=kprev[hh])
                        car = nl.ndarray((P, hq), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=car, data=ps2, op0=nl.multiply, operand0=mask)
                        nisa.scalar_tensor_tensor(dst=ok[:, hq:2 * hq], data=ps, op0=nl.multiply, operand0=mask, op1=nl.add, operand1=car)
                    else:
                        nisa.tensor_scalar(dst=ok[:, hq:2 * hq], data=ps, op0=nl.multiply, operand0=mask)
                    nisa.tensor_copy(dst=kprev[hh], src=ob[:, hq:2 * hq])
                    nisa.dma_copy(dst=dst_t[row * n_head + hh, p0:p0 + P, :], src=ok)
            vb = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=vb, src=v[:, c0:c0 + HD])
            nisa.dma_copy(dst=v_out[row * n_head + hh, p0:p0 + P, :], src=vb)
    return q_out, k_out, v_out, xm_out, rq_out, rk_out


@nki.jit
def pre_attn_bwd_ko(dq_in: nl.ndarray, dk_in: nl.ndarray, dv_in: nl.ndarray, rq_in: nl.ndarray, rk_in: nl.ndarray,
                    xm_in: nl.ndarray, dx1_in: nl.ndarray, cos_in: nl.ndarray, sin_in: nl.ndarray, dstart_in: nl.ndarray,
                    wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray, n_head: int = 4, eps: float = 1e-6,
                    scale: float = 1.0, hq: int = 32):
    """pre_attn_bwd (nki_pre_bwd.py) with the key-offset backward folded in: dstart_in (T, 1) fp32 replaces dksh_in;
    g[p, hq:2hq] = dk[p+1, hq:2hq] * (1 - dstart[p+1]) (0 at the last position of a sequence row). Tiles run in reverse."""
    T, D = xm_in.shape
    BH, t, HD = dq_in.shape
    half = HD // 2
    kt = D // P
    tiles = T // P
    tpt = t // P
    dxm_out = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    gv_out = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    dwq_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
    dwk_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
    dwv_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
    # key-offset backward shift matrices: St[k, m] = 1 iff m == k - 1 ; Et[k, m] = 1 iff k == 0 and m == P-1
    ones = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    St = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.affine_select(dst=St, pattern=[[1, P]], channel_multiplier=-1, offset=1, on_true_tile=ones, on_false_value=0.0)
    Et = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.affine_select(dst=Et, pattern=[[1, P]], channel_multiplier=-1, offset=-(P - 1), on_true_tile=ones, on_false_value=0.0)
    dkprev = []                                   # per head: masked dk dims hq:2hq of the NEXT tile (tiles run in reverse)
    for hh in range(n_head):
        kp = nl.ndarray((P, hq), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(kp, 0.0)
        dkprev.append(kp)
    # resident weights in natural (out, in) layout: chunk o = rows o*128:(o+1)*128 = the dgrad moving operand [K=out, N=in]
    Wq = []
    Wk = []
    Wv = []
    for o in range(kt):
        a = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=a, src=wq[o * P:(o + 1) * P, :])
        Wq.append(a)
        a = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=a, src=wk[o * P:(o + 1) * P, :])
        Wk.append(a)
        a = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=a, src=wv[o * P:(o + 1) * P, :])
        Wv.append(a)
    # weight-grad accumulators (fp32 SBUF, rows chunk o of each (D, D) matrix)
    accq = []
    acck = []
    accv = []
    for o in range(kt):
        a = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(a, 0.0)
        accq.append(a)
        a = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(a, 0.0)
        acck.append(a)
        a = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(a, 0.0)
        accv.append(a)
    for tbi in range(tiles):
        tb = tiles - 1 - tbi                      # reverse order: the carry comes from tile tb + 1 (already visited)
        t0 = tb * P
        b = tb // tpt
        p0 = (tb - b * tpt) * P
        # ---- rmsnorm of xm (recompute h and rstd)
        xmb = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xmb, src=xm_in[t0:t0 + P, :])
        xm = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xm, src=xmb)
        sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=xm, data2=xm, op=nl.multiply)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
        h = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
        # ---- rope tables for these rows (sin negated: the transpose of the rotation)
        cs = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
        sn = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs, src=cos_in[t0:t0 + P, :])
        nisa.dma_copy(dst=sn, src=sin_in[t0:t0 + P, :])
        nisa.tensor_scalar(dst=sn, data=sn, op0=nl.multiply, operand0=-1.0)
        # ---- key-offset mask of these rows: 1 - dstart (applied at the source position p, before the shift to p-1)
        ds = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds, src=dstart_in[t0:t0 + P, :])
        mask = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mask, data=ds, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        # ---- per head: gq/gk -> QK-norm backward -> rope backward; gv passthrough
        dpq = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        dpk = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        gvb = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for hh in range(n_head):
            bh = b * n_head + hh
            c0 = hh * HD
            for j in range(2):
                g = nl.ndarray((P, HD), dtype=nl.float32, buffer=nl.sbuf)
                if j == 0:
                    gb = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=gb, src=dq_in[bh, p0:p0 + P, :])
                    nisa.tensor_scalar(dst=g, data=gb, op0=nl.multiply, operand0=scale)
                else:
                    gb = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=gb, src=dk_in[bh, p0:p0 + P, :])
                    nisa.tensor_copy(dst=g, src=gb)
                    # key-offset backward: dims hq:2hq of row p <- masked grad of row p+1 (row P-1 <- row 0 of the next tile)
                    dkm = nl.ndarray((P, hq), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dkm, data=gb[:, hq:2 * hq], op0=nl.multiply, operand0=mask)
                    # (two PSUM tiles summed on VectorE: the compiler rejects mixed matmul accumulate flags, NCC_ILMM003)
                    ps = nl.ndarray((P, hq), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=ps, stationary=St, moving=dkm)
                    if p0 + P < t:
                        ps2 = nl.ndarray((P, hq), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=ps2, stationary=Et, moving=dkprev[hh])
                        car = nl.ndarray((P, hq), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=car, src=ps2)
                        nisa.tensor_tensor(dst=g[:, hq:2 * hq], data1=ps, data2=car, op=nl.add)
                    else:
                        nisa.tensor_copy(dst=g[:, hq:2 * hq], src=ps)
                    nisa.tensor_copy(dst=dkprev[hh], src=dkm)
                rb = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                if j == 0:
                    nisa.dma_copy(dst=rb, src=rq_in[hh, t0:t0 + P, :])
                else:
                    nisa.dma_copy(dst=rb, src=rk_in[hh, t0:t0 + P, :])
                r = nl.ndarray((P, HD), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=r, src=rb)
                # rmsnorm backward over the head dim: d = rs * (g - r * rs^2 * mean(g*r))
                tt = nl.ndarray((P, HD), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tt, data1=r, data2=r, op=nl.multiply)
                m2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=m2, op=nl.add, data=tt, axis=(1,))
                nisa.tensor_scalar(dst=m2, data=m2, op0=nl.multiply, operand0=1.0 / HD, op1=nl.add, operand1=eps)
                rs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=rs, op=nl.rsqrt, data=m2)
                nisa.tensor_tensor(dst=tt, data1=g, data2=r, op=nl.multiply)
                cc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=cc, op=nl.add, data=tt, axis=(1,))
                rs2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=rs2, data1=rs, data2=rs, op=nl.multiply)
                nisa.tensor_tensor(dst=cc, data1=cc, data2=rs2, op=nl.multiply)
                nisa.tensor_scalar(dst=cc, data=cc, op0=nl.multiply, operand0=1.0 / HD)
                nisa.tensor_scalar(dst=tt, data=r, op0=nl.multiply, operand0=cc)
                nisa.tensor_tensor(dst=tt, data1=g, data2=tt, op=nl.subtract)
                dn = nl.ndarray((P, HD), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dn, data=tt, op0=nl.multiply, operand0=rs)
                # rope transpose: y1 = x1 cos - x2 sin ; y2 = x2 cos + x1 sin   (sn already negated)
                x1 = dn[:, 0:half]
                x2 = dn[:, half:HD]
                r1 = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
                r2 = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
                tmp = nl.ndarray((P, half), dtype=nl.float32, buffer=nl.sbuf)
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
                    nisa.tensor_copy(dst=dpk[:, c0 + half:c0 + HD], src=r2)
            nisa.dma_copy(dst=gvb[:, c0:c0 + HD], src=dv_in[bh, p0:p0 + P, :])
        nisa.dma_copy(dst=gv_out[t0:t0 + P, :], src=gvb)
        # ---- dgrad: dh[tok, in] = sum_o dpq[tok, o] Wq[o, in] + ... ; stationary = transposed activation chunks [K=out, M=tok]
        dh_ps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
        first = True
        for src_idx in range(3):
            for o in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                if src_idx == 0:
                    nisa.nc_transpose(dst=tps, data=dpq[:, o * P:(o + 1) * P])
                elif src_idx == 1:
                    nisa.nc_transpose(dst=tps, data=dpk[:, o * P:(o + 1) * P])
                else:
                    nisa.nc_transpose(dst=tps, data=gvb[:, o * P:(o + 1) * P])
                aT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=aT, src=tps)
                if src_idx == 0:
                    nisa.nc_matmul(dst=dh_ps, stationary=aT, moving=Wq[o], accumulate=(not first))
                elif src_idx == 1:
                    nisa.nc_matmul(dst=dh_ps, stationary=aT, moving=Wk[o], accumulate=(not first))
                else:
                    nisa.nc_matmul(dst=dh_ps, stationary=aT, moving=Wv[o], accumulate=(not first))
                first = False
        dh = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dh, src=dh_ps)
        # ---- rmsnorm backward of xm: dxm = dx1 + rstd * (dh - xm * rstd^2 * mean(dh*xm))
        nisa.tensor_tensor(dst=sq, data1=dh, data2=xm, op=nl.multiply)
        cm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cm, op=nl.add, data=sq, axis=(1,))
        r2c = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r2c, data1=rstd, data2=rstd, op=nl.multiply)
        nisa.tensor_tensor(dst=cm, data1=cm, data2=r2c, op=nl.multiply)
        nisa.tensor_scalar(dst=cm, data=cm, op0=nl.multiply, operand0=1.0 / D)
        nisa.tensor_scalar(dst=sq, data=xm, op0=nl.multiply, operand0=cm)
        nisa.tensor_tensor(dst=sq, data1=dh, data2=sq, op=nl.subtract)
        dx1b = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=dx1b, src=dx1_in[t0:t0 + P, :])
        dx1 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dx1, src=dx1b)
        nisa.scalar_tensor_tensor(dst=dx1, data=sq, op0=nl.multiply, operand0=rstd, op1=nl.add, operand1=dx1)
        dxb = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dxb, src=dx1)
        nisa.dma_copy(dst=dxm_out[t0:t0 + P, :], src=dxb)
        # ---- wgrad: dW[o-chunk, in] += act[tok, o-chunk]^T @ h[tok, in]   (contraction over the token partitions)
        for o in range(kt):
            wps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wps, stationary=dpq[:, o * P:(o + 1) * P], moving=h)
            nisa.tensor_tensor(dst=accq[o], data1=accq[o], data2=wps, op=nl.add)
            wps2 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wps2, stationary=dpk[:, o * P:(o + 1) * P], moving=h)
            nisa.tensor_tensor(dst=acck[o], data1=acck[o], data2=wps2, op=nl.add)
            wps3 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wps3, stationary=gvb[:, o * P:(o + 1) * P], moving=h)
            nisa.tensor_tensor(dst=accv[o], data1=accv[o], data2=wps3, op=nl.add)
    for o in range(kt):
        nisa.dma_copy(dst=dwq_out[o * P:(o + 1) * P, :], src=accq[o])
        nisa.dma_copy(dst=dwk_out[o * P:(o + 1) * P, :], src=acck[o])
        nisa.dma_copy(dst=dwv_out[o * P:(o + 1) * P, :], src=accv[o])
    return dxm_out, gv_out, dwq_out, dwk_out, dwv_out


def key_offset_fwd_glue(k4, dstart, hq):
    """The vehicle's `_fwd_impl` torch glue: k4 (b, NH, t, HD) bf16 -> dims hq:2hq from position p-1, zeroed at doc starts."""
    b, NH, t, HD = k4.shape
    prev = torch.cat([torch.zeros_like(k4[:, :, :1, hq:2 * hq]), k4[:, :, :-1, hq:2 * hq]], dim=2)
    prev = prev * (1.0 - dstart).to(k4.dtype).view(b, 1, t, 1)
    return torch.cat([k4[..., :hq], prev, k4[..., 2 * hq:]], dim=-1).contiguous()
