#!/usr/bin/env python3
"""ONE kernel per layer forward (2026-09-17, "hard queue" step 2).

Today a layer forward is 3 launches + HBM round trips: pre_attn_fwd_ko (nki_pre_ko.py) -> vendor attention_cte ->
post_attn_fwd_h (vehicle). layer_fwd_fused = the pre_ko body, then the vendor's undecorated `_attention_cte` body NESTED
(it exposes out_o_hbm / out_neg_max_hbm / out_sum_hbm exactly for this, see nki_parres_test.py), then the post body, in one
@nki.jit. q/k/v still go to HBM (the vendor body reads them from HBM per (batch*head), and the backward needs them anyway),
so the HBM traffic is unchanged; what disappears is two launches per layer and the glue between them. Outputs are exactly
`_fwd_impl`'s 13 (o, xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT) so `_bwd_impl` is untouched.
Helper bodies are module-level plain functions (tracer rule #3/#4: names resolve through the kernel module's globals; the
parres test's `mlp_body` proved a module-level helper nests on device). sink_in (b*NH, 1) fp32, bmin/bmax (b*NH, t, 1) fp32
are the same tensors the vehicle's `attn_fwd` builds for the vendor entry point."""
import torch
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.attention.attention_cte import _attention_cte

P = 128
FMAX = 512


def _pre_body(x_in, x0_in, ve_in, wq, wk, wv, cos_in, sin_in, scal, dstart_in, q_out, k_out, v_out, xm_out, rq_out, rk_out,
              n_head, eps, t_len, q_scale, hq):
    """pre_attn_fwd_ko's body (nki_pre_ko.py) writing into caller-allocated HBM outputs."""
    T, D = x_in.shape
    HD = D // n_head
    half = HD // 2
    tpt = t_len // P
    kt = D // P
    tiles = T // P
    sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=scal)
    ones = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    Sm = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.affine_select(dst=Sm, pattern=[[1, P]], channel_multiplier=-1, offset=-1, on_true_tile=ones, on_false_value=0.0)
    Em = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.affine_select(dst=Em, pattern=[[1, P]], channel_multiplier=-1, offset=P - 1, on_true_tile=ones, on_false_value=0.0)
    kprev = []
    for hh in range(n_head):
        kp = nl.ndarray((P, hq), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(kp, 0.0)
        kprev.append(kp)
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
    for tb in range(tiles):
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
        sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=xm, data2=xm, op=nl.multiply)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
        h = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
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
                nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)
                nisa.tensor_tensor(dst=r2, data1=x2, data2=cs, op=nl.multiply)
                nisa.tensor_tensor(dst=tmp, data1=x1, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)
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


def _post_body(y_in, wo, xm_in, w1, w2, out, x1_out, pre_out, y2_out, eps, n_head, t_len):
    """post_attn_fwd_h's body (vehicle) writing into caller-allocated HBM outputs; single program, no fq."""
    T, D = xm_in.shape
    H = w1.shape[0]
    HD = D // n_head
    tpt = t_len // P
    kt = D // P
    ht = H // P
    chunks = T // FMAX
    WoT = []
    for _j in range(kt):
        WoT.append(nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
    for o in range(kt):
        blk = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=wo[o * P:(o + 1) * P, :])
        for i in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, i * P:(i + 1) * P])
            nisa.tensor_copy(dst=WoT[i][:, o * P:(o + 1) * P], src=tps)
    W1 = []
    for _j in range(kt):
        W1.append(nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf))
    for h in range(ht):
        blk = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w1[h * P:(h + 1) * P, :])
        for k in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, k * P:(k + 1) * P])
            nisa.tensor_copy(dst=W1[k][:, h * P:(h + 1) * P], src=tps)
    W2 = []
    for _j in range(ht):
        W2.append(nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
    for o in range(kt):
        blk = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w2[o * P:(o + 1) * P, :])
        for h in range(ht):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, h * P:(h + 1) * P])
            nisa.tensor_copy(dst=W2[h][:, o * P:(o + 1) * P], src=tps)
    for c in range(chunks):
        t0 = c * FMAX
        h2T = []
        for _j in range(kt):
            h2T.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        x1_tiles = []
        for _j in range(FMAX // P):
            x1_tiles.append(nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf))
        for tb in range(FMAX // P):
            yt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            gt = (t0 + tb * P) // P
            row = gt // tpt
            p0 = (gt - row * tpt) * P
            for hh in range(n_head):
                nisa.dma_copy(dst=yt[:, hh * HD:(hh + 1) * HD], src=y_in[row * n_head + hh, p0:p0 + P, :])
            nisa.dma_copy(dst=y2_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=yt)
            aps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            for i in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=yt[:, i * P:(i + 1) * P])
                yTi = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=yTi, src=tps)
                nisa.nc_matmul(dst=aps, stationary=yTi, moving=WoT[i], accumulate=(i > 0))
            xmt = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=xmt, src=xm_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            x1 = x1_tiles[tb]
            nisa.tensor_tensor(dst=x1, data1=xmt, data2=aps, op=nl.add)
            x1b = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x1b, src=x1)
            nisa.dma_copy(dst=x1_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=x1b)
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=x1, data2=x1, op=nl.multiply)
            ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h2 = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h2, data=x1, op0=nl.multiply, operand0=rstd)
            for k in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=h2[:, k * P:(k + 1) * P])
                nisa.tensor_copy(dst=h2T[k][:, tb * P:(tb + 1) * P], src=tps)
        hT = []
        for _j in range(ht):
            hT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for h in range(ht):
            pps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
            for k in range(kt):
                nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * P:(h + 1) * P], moving=h2T[k], accumulate=(k > 0))
            pre = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pre, src=pps)
            nisa.dma_copy(dst=pre_out[h * P:(h + 1) * P, t0:t0 + FMAX], src=pre)
            nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)
        for tb in range(FMAX // P):
            mps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            for h in range(ht):
                nisa.nc_matmul(dst=mps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=W2[h], accumulate=(h > 0))
            o = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o, data1=x1_tiles[tb], data2=mps, op=nl.add)
            nisa.dma_copy(dst=out[t0 + tb * P:t0 + (tb + 1) * P, :], src=o)


@nki.jit
def layer_fwd_fused(x_in: nl.ndarray, x0_in: nl.ndarray, ve_in: nl.ndarray, wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray,
                    cos_in: nl.ndarray, sin_in: nl.ndarray, scal: nl.ndarray, dstart_in: nl.ndarray, sink_in: nl.ndarray,
                    bmin: nl.ndarray, bmax: nl.ndarray, wo: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, n_head: int = 4,
                    eps: float = 1e-6, t_len: int = 1024, q_scale: float = 1.0, hq: int = 32):
    """One layer forward: pre_ko body -> nested vendor _attention_cte (sequence packing + sink, cache_softmax) -> post body.
    Returns _fwd_impl's 13 tensors: o (T, D), xm (T, D), rq/rk (NH, T, HD), q/k/v (b*NH, t, HD), out (b*NH, t, HD),
    neg_max/recip (b*NH, 128, t/128) fp32, y2 (T, D), x1 (T, D), preT (H, T)."""
    T, D = x_in.shape
    HD = D // n_head
    H = w1.shape[0]
    BH = (T // t_len) * n_head
    q_out = nl.ndarray((BH, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray((BH, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    v_out = nl.ndarray((BH, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    xm_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    rq_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    rk_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    att_out = nl.ndarray((BH, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    neg_max = nl.ndarray((BH, P, t_len // P), dtype=nl.float32, buffer=nl.shared_hbm)
    recip = nl.ndarray((BH, P, t_len // P), dtype=nl.float32, buffer=nl.shared_hbm)
    o_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    x1_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((H, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    y2_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    _pre_body(x_in, x0_in, ve_in, wq, wk, wv, cos_in, sin_in, scal, dstart_in, q_out, k_out, v_out, xm_out, rq_out, rk_out,
              n_head, eps, t_len, q_scale, hq)
    _attention_cte(q_out, k_out, v_out, scale=1.0, causal_mask=True, sink=sink_in, tp_q=True, tp_k=True, cache_softmax=True,
                   bound_min=bmin, bound_max=bmax, out_o_hbm=att_out, out_neg_max_hbm=neg_max, out_sum_hbm=recip)
    _post_body(att_out, wo, xm_out, w1, w2, o_out, x1_out, pre_out, y2_out, eps, n_head, t_len)
    return o_out, xm_out, rq_out, rk_out, q_out, k_out, v_out, att_out, neg_max, recip, y2_out, x1_out, pre_out
