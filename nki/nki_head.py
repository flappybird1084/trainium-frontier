#!/usr/bin/env python3
"""Fused LM head + asymmetric softcap + cross-entropy (2026-09-08), so the (T, 8192) fp32 logits never touch HBM.
Forward per 128-token tile: h = rmsnorm(x); for each 512-wide vocab chunk: z = h W^T, zc = softcap(z) (sp for z>=0, sn
for z<0), online row max / sum, and the target logit via a column-index mask. Returns per-row loss (T, 1) fp32 (0 for
ignored rows), the per-row logsumexp (T, 1) fp32 and h (T, D) bf16 (both saved for the backward).
Inputs: x (T, D) bf16; wT (D, V) bf16 = W^T (built once per step by the caller); tgt (T, 1) int32 (-1 = ignore);
scal (128, 4) fp32 with columns [sp, sn, 0, 0]. T % 128 == 0, D = 512, V % 512 == 0."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
VC = 512


@nki.jit
def head_fwd(x_in: nl.ndarray, wT: nl.ndarray, tgt: nl.ndarray, scal: nl.ndarray, eps: float = 0.0078125):
    T, D = x_in.shape
    V = wT.shape[1]
    kt = D // P
    nchunk = V // VC
    loss_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    lse_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    h_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    rstd_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nprog = nl.num_programs(0)
    tiles = T // P
    per = tiles // nprog
    sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=scal)
    inv_sp = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    inv_sn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sp, data=sc[:, 0:1])
    nisa.reciprocal(dst=inv_sn, data=sc[:, 1:2])
    # column indices of a chunk: iota over the free dim (same for every partition), plus the chunk offset later
    col = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=col, pattern=[[1, VC]], offset=0, channel_multiplier=0)
    # whole W^T resident in SBUF (kt x V bf16 per partition = 64 KB at D=512, V=8192): read from HBM once per kernel
    wsb = nl.ndarray((P, kt, V), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(kt):
        nisa.dma_copy(dst=wsb[:, i, :], src=wT[i * P:(i + 1) * P, :])
    for ti in nl.affine_range(per):
        tb = nl.program_id(0) * per + ti
        t0 = tb * P
        xr = nl.ndarray((P, D), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xr, src=x_in[t0:t0 + P, :])
        x = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x, src=xr)
        sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
        nisa.dma_copy(dst=rstd_out[t0:t0 + P, :], src=rstd)
        h = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h, data=x, op0=nl.multiply, operand0=rstd)
        nisa.dma_copy(dst=h_out[t0:t0 + P, :], src=h)
        hT = []
        for i in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=h[:, i * P:(i + 1) * P])
            hTi = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hTi, src=tps)
            hT.append(hTi)
        tgi = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tgi, src=tgt[t0:t0 + P, :])
        tg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tg, src=tgi)
        m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)       # running max
        s = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)       # running sum of exp(zc - m)
        tl = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)      # target logit (capped)
        nisa.memset(m, -1e30)
        nisa.memset(s, 0.0)
        nisa.memset(tl, 0.0)
        for c in range(nchunk):
            zps = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.psum)
            for i in range(kt):
                nisa.nc_matmul(dst=zps, stationary=hT[i], moving=wsb[:, i, c * VC:(c + 1) * VC], accumulate=(i > 0))
            z = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=z, src=zps)
            # asymmetric softcap: zc = sp*tanh(z/sp) if z >= 0 else sn*tanh(z/sn)
            tp = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            tn = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=tp, op=nl.tanh, data=z, scale=inv_sp)
            nisa.tensor_scalar(dst=tp, data=tp, op0=nl.multiply, operand0=sc[:, 0:1])
            nisa.activation(dst=tn, op=nl.tanh, data=z, scale=inv_sn)
            nisa.tensor_scalar(dst=tn, data=tn, op0=nl.multiply, operand0=sc[:, 1:2])
            pos = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos, data=z, op0=nl.greater_equal, operand0=0.0)          # 1.0 where z >= 0
            zc = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=zc, data1=tp, data2=tn, op=nl.subtract)
            nisa.tensor_tensor(dst=zc, data1=zc, data2=pos, op=nl.multiply)
            nisa.tensor_tensor(dst=zc, data1=zc, data2=tn, op=nl.add)                        # tn + pos*(tp - tn)
            # target logit: mask (col + c*VC == tgt)
            cm = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=cm, data=col, op0=nl.add, operand0=float(c * VC))
            eqm = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=eqm, data=cm, op0=nl.equal, operand0=tg)                     # per-partition scalar compare
            sel = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sel, data1=zc, data2=eqm, op=nl.multiply)
            part = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=part, op=nl.add, data=sel, axis=(1,))
            nisa.tensor_tensor(dst=tl, data1=tl, data2=part, op=nl.add)
            # online logsumexp
            cmax = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=cmax, op=nl.maximum, data=zc, axis=(1,))
            mnew = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mnew, data1=m, data2=cmax, op=nl.maximum)
            dm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dm, data1=m, data2=mnew, op=nl.subtract)
            nisa.activation(dst=dm, op=nl.exp, data=dm)                                        # exp(m - mnew)
            nisa.tensor_tensor(dst=s, data1=s, data2=dm, op=nl.multiply)
            negm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=negm, data=mnew, op0=nl.multiply, operand0=-1.0)
            e = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=e, op=nl.exp, data=zc, bias=negm)                              # exp(zc - mnew)
            es = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=es, op=nl.add, data=e, axis=(1,))
            nisa.tensor_tensor(dst=s, data1=s, data2=es, op=nl.add)
            nisa.tensor_copy(dst=m, src=mnew)
        lse = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=lse, op=nl.log, data=s)
        nisa.tensor_tensor(dst=lse, data1=lse, data2=m, op=nl.add)
        nisa.dma_copy(dst=lse_out[t0:t0 + P, :], src=lse)
        valid = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=valid, data=tg, op0=nl.greater_equal, operand0=0.0)
        lossr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=lossr, data1=lse, data2=tl, op=nl.subtract)
        nisa.tensor_tensor(dst=lossr, data1=lossr, data2=valid, op=nl.multiply)
        nisa.dma_copy(dst=loss_out[t0:t0 + P, :], src=lossr)
    return loss_out, lse_out, h_out, rstd_out


@nki.jit
def head_bwd(h_in: nl.ndarray, rstd_in: nl.ndarray, lse_in: nl.ndarray, dl_in: nl.ndarray, w: nl.ndarray, wT: nl.ndarray,
             tgt: nl.ndarray, scal: nl.ndarray):
    """Backward of head_fwd. h (T, D) bf16 (normed input), rstd/lse/dl (T, 1) fp32 (dl = upstream grad of the per-row loss),
    w (V, D) bf16, wT (D, V) bf16, tgt (T, 1) int32, scal as in the forward. Returns dx (T, D) bf16 and dW (V, D) fp32.
    Token halves of TH rows are the outer loop; inside, vocab chunks (dW chunk accumulates in PSUM over the half's token
    tiles) and token tiles (z recomputed, dh accumulated in SBUF fp32); the rmsnorm backward closes each half."""
    T, D = h_in.shape
    V = w.shape[0]
    kt = D // P
    nchunk = V // VC
    TH = 2048 if T >= 2048 else T
    nh = T // TH
    th_tiles = TH // P
    dx_out = nl.ndarray((T, D), dtype=h_in.dtype, buffer=nl.shared_hbm)
    dw_out = nl.ndarray((V, D), dtype=nl.float32, buffer=nl.shared_hbm)
    dcap_out = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=scal)
    inv_sp = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    inv_sn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sp, data=sc[:, 0:1])
    nisa.reciprocal(dst=inv_sn, data=sc[:, 1:2])
    col = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=col, pattern=[[1, VC]], offset=0, channel_multiplier=0)
    dcap = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)       # per-partition partial d(loss)/d(sp), d(loss)/d(sn)
    nisa.memset(dcap, 0.0)
    for hh in range(nh):
        base = hh * TH
        # resident per half: h (bf16), hT (bf16, kt tiles per token tile), dh accumulator (fp32), per-row scalars
        h = nl.ndarray((P, th_tiles, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        hT = nl.ndarray((P, th_tiles, kt, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        dh = nl.ndarray((P, th_tiles, D), dtype=nl.float32, buffer=nl.sbuf)
        g = nl.ndarray((P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
        nlse = nl.ndarray((P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
        tgi = nl.ndarray((P, th_tiles), dtype=nl.int32, buffer=nl.sbuf)
        tg = nl.ndarray((P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dh, 0.0)
        for t in range(th_tiles):
            t0 = base + t * P
            nisa.dma_copy(dst=h[:, t, :], src=h_in[t0:t0 + P, :])
            for i in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=h[:, t, i * P:(i + 1) * P])
                nisa.tensor_copy(dst=hT[:, t, i, :], src=tps)
            nisa.dma_copy(dst=tgi[:, t:t + 1], src=tgt[t0:t0 + P, :])
            nisa.dma_copy(dst=g[:, t:t + 1], src=dl_in[t0:t0 + P, :])
            nisa.dma_copy(dst=nlse[:, t:t + 1], src=lse_in[t0:t0 + P, :])
        nisa.tensor_copy(dst=tg, src=tgi)
        valid = nl.ndarray((P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=valid, data=tg, op0=nl.greater_equal, operand0=0.0)
        nisa.tensor_tensor(dst=g, data1=g, data2=valid, op=nl.multiply)
        nisa.tensor_scalar(dst=nlse, data=nlse, op0=nl.multiply, operand0=-1.0)
        for c in range(nchunk):
            wc = nl.ndarray((P, kt, D), dtype=nl.bfloat16, buffer=nl.sbuf)       # W rows of this chunk: 4 tiles (128 vocab, D)
            wTc = nl.ndarray((P, kt, VC), dtype=nl.bfloat16, buffer=nl.sbuf)     # W^T columns of this chunk
            for j in range(kt):
                nisa.dma_copy(dst=wc[:, j, :], src=w[c * VC + j * P:c * VC + (j + 1) * P, :])
                nisa.dma_copy(dst=wTc[:, j, :], src=wT[j * P:(j + 1) * P, c * VC:(c + 1) * VC])
            cm = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=cm, data=col, op0=nl.add, operand0=float(c * VC))
            dwps = []
            for j in range(kt):
                dwps.append(nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum))
            for t in range(th_tiles):
                zps = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.psum)
                for i in range(kt):
                    nisa.nc_matmul(dst=zps, stationary=hT[:, t, i, :], moving=wTc[:, i, :], accumulate=(i > 0))
                z = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=z, src=zps)
                tp = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                tn = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=tp, op=nl.tanh, data=z, scale=inv_sp)
                nisa.activation(dst=tn, op=nl.tanh, data=z, scale=inv_sn)
                pos = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=pos, data=z, op0=nl.greater_equal, operand0=0.0)
                # zc = sn*tn + pos*(sp*tp - sn*tn); deriv = (1 - tn^2) + pos*((1 - tp^2) - (1 - tn^2)) = 1 - tn^2 - pos*(tp^2 - tn^2)
                a = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                b = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=a, data=tp, op0=nl.multiply, operand0=sc[:, 0:1])
                nisa.tensor_scalar(dst=b, data=tn, op0=nl.multiply, operand0=sc[:, 1:2])
                zc = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=zc, data1=a, data2=b, op=nl.subtract)
                nisa.tensor_tensor(dst=zc, data1=zc, data2=pos, op=nl.multiply)
                nisa.tensor_tensor(dst=zc, data1=zc, data2=b, op=nl.add)
                nisa.tensor_tensor(dst=a, data1=tp, data2=tp, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=tn, data2=tn, op=nl.multiply)
                deriv = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deriv, data1=a, data2=b, op=nl.subtract)
                nisa.tensor_tensor(dst=deriv, data1=deriv, data2=pos, op=nl.multiply)
                nisa.tensor_tensor(dst=deriv, data1=deriv, data2=b, op=nl.add)                 # tn^2 + pos*(tp^2 - tn^2)
                nisa.tensor_scalar(dst=deriv, data=deriv, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
                # p = exp(zc - lse); dzc = (p - onehot) * g; dz = dzc * deriv
                pr = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=pr, op=nl.exp, data=zc, bias=nlse[:, t:t + 1])
                eqm = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eqm, data=cm, op0=nl.equal, operand0=tg[:, t:t + 1])
                nisa.tensor_tensor(dst=pr, data1=pr, data2=eqm, op=nl.subtract)
                nisa.tensor_scalar(dst=pr, data=pr, op0=nl.multiply, operand0=g[:, t:t + 1])
                dz = nl.ndarray((P, VC), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dz, data1=pr, data2=deriv, op=nl.multiply)
                # softcap parameter grads: d zc/d sp = tp - (z/sp)(1 - tp^2) on z >= 0; d zc/d sn = tn - (z/sn)(1 - tn^2) on z < 0
                u = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=u, data=a, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)   # 1 - tp^2
                nisa.tensor_tensor(dst=u, data1=u, data2=z, op=nl.multiply)
                nisa.tensor_scalar(dst=u, data=u, op0=nl.multiply, operand0=inv_sp)
                nisa.tensor_tensor(dst=u, data1=tp, data2=u, op=nl.subtract)
                nisa.tensor_tensor(dst=u, data1=u, data2=pos, op=nl.multiply)
                nisa.tensor_tensor(dst=u, data1=u, data2=pr, op=nl.multiply)
                part = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
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
                nisa.tensor_tensor(dst=dcap[:, 1:2], data1=dcap[:, 1:2], data2=part, op=nl.add)
                # dh[t] += dz @ W_chunk  (contract over the chunk's 512 vocab entries: 4 sub-tiles)
                dhps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                for j in range(kt):
                    tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=dz[:, j * P:(j + 1) * P])
                    dzT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dzT, src=tps)
                    nisa.nc_matmul(dst=dhps, stationary=dzT, moving=wc[:, j, :], accumulate=(j > 0))
                nisa.tensor_tensor(dst=dh[:, t, :], data1=dh[:, t, :], data2=dhps, op=nl.add)
                # dW_chunk[j] += dz[:, j]^T @ h[t]  (contract over the 128 tokens of this tile)
                for j in range(kt):
                    nisa.nc_matmul(dst=dwps[j], stationary=dz[:, j * P:(j + 1) * P], moving=h[:, t, :], accumulate=(t > 0))
            for j in range(kt):
                r0 = c * VC + j * P
                if hh == 0:
                    dwsb = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dwsb, src=dwps[j])
                    nisa.dma_copy(dst=dw_out[r0:r0 + P, :], src=dwsb)
                else:
                    prev = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=prev, src=dw_out[r0:r0 + P, :])
                    nisa.tensor_tensor(dst=prev, data1=prev, data2=dwps[j], op=nl.add)
                    nisa.dma_copy(dst=dw_out[r0:r0 + P, :], src=prev)
        # rmsnorm backward: dx = rstd * (dh - h * mean_D(dh * h))
        for t in range(th_tiles):
            t0 = base + t * P
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rstd, src=rstd_in[t0:t0 + P, :])
            hf = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hf, src=h[:, t, :])
            prod = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=dh[:, t, :], data2=hf, op=nl.multiply)
            mdot = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mdot, op=nl.add, data=prod, axis=(1,))
            nisa.tensor_scalar(dst=mdot, data=mdot, op0=nl.multiply, operand0=-1.0 / D)
            nisa.tensor_scalar(dst=hf, data=hf, op0=nl.multiply, operand0=mdot)
            nisa.tensor_tensor(dst=hf, data1=hf, data2=dh[:, t, :], op=nl.add)
            dx = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dx, data=hf, op0=nl.multiply, operand0=rstd)
            nisa.dma_copy(dst=dx_out[t0:t0 + P, :], src=dx)
    nisa.dma_copy(dst=dcap_out, src=dcap)
    return dx_out, dw_out, dcap_out


@nki.jit
def head_logits_fwd(x_in: nl.ndarray, wT: nl.ndarray, scal: nl.ndarray, eps: float = 0.0078125):
    """Review-safe variant: returns the softcapped logits (T, V) bf16 (plus h, rstd) for BOTH training and eval; the loss is
    plain cross-entropy over these logits in torch."""
    T, D = x_in.shape
    V = wT.shape[1]
    kt = D // P
    nchunk = V // VC
    logits_out = nl.ndarray((T, V), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    h_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    rstd_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nprog = nl.num_programs(0)
    tiles = T // P
    per = tiles // nprog
    sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=scal)
    inv_sp = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    inv_sn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sp, data=sc[:, 0:1])
    nisa.reciprocal(dst=inv_sn, data=sc[:, 1:2])
    # whole W^T resident in SBUF (kt x V bf16 per partition = 64 KB at D=512, V=8192): read from HBM once per kernel
    wsb = nl.ndarray((P, kt, V), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(kt):
        nisa.dma_copy(dst=wsb[:, i, :], src=wT[i * P:(i + 1) * P, :])
    for ti in nl.affine_range(per):
        tb = nl.program_id(0) * per + ti
        t0 = tb * P
        xr = nl.ndarray((P, D), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xr, src=x_in[t0:t0 + P, :])
        x = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x, src=xr)
        sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
        nisa.dma_copy(dst=rstd_out[t0:t0 + P, :], src=rstd)
        h = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h, data=x, op0=nl.multiply, operand0=rstd)
        nisa.dma_copy(dst=h_out[t0:t0 + P, :], src=h)
        hT = []
        for i in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=h[:, i * P:(i + 1) * P])
            hTi = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hTi, src=tps)
            hT.append(hTi)
        for c in range(nchunk):
            zps = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.psum)
            for i in range(kt):
                nisa.nc_matmul(dst=zps, stationary=hT[i], moving=wsb[:, i, c * VC:(c + 1) * VC], accumulate=(i > 0))
            z = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=z, src=zps)
            # asymmetric softcap: zc = sp*tanh(z/sp) if z >= 0 else sn*tanh(z/sn)
            tp = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            tn = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=tp, op=nl.tanh, data=z, scale=inv_sp)
            nisa.tensor_scalar(dst=tp, data=tp, op0=nl.multiply, operand0=sc[:, 0:1])
            nisa.activation(dst=tn, op=nl.tanh, data=z, scale=inv_sn)
            nisa.tensor_scalar(dst=tn, data=tn, op0=nl.multiply, operand0=sc[:, 1:2])
            pos = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos, data=z, op0=nl.greater_equal, operand0=0.0)          # 1.0 where z >= 0
            zc = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=zc, data1=tp, data2=tn, op=nl.subtract)
            nisa.tensor_tensor(dst=zc, data1=zc, data2=pos, op=nl.multiply)
            nisa.tensor_tensor(dst=zc, data1=zc, data2=tn, op=nl.add)                        # tn + pos*(tp - tn)
            zcb = nl.ndarray((P, VC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=zcb, src=zc)
            nisa.dma_copy(dst=logits_out[t0:t0 + P, c * VC:(c + 1) * VC], src=zcb)
    return logits_out, h_out, rstd_out


@nki.jit
def head_logits_bwd(h_in: nl.ndarray, rstd_in: nl.ndarray, dl_in: nl.ndarray, w: nl.ndarray, wT: nl.ndarray, scal: nl.ndarray):
    """Backward of head_logits_fwd: dl (T, V) fp32 is d loss / d (capped logits) from torch cross-entropy. Recomputes
    z per 512-wide vocab chunk; dh accumulates in SBUF, dW chunks in PSUM; rmsnorm backward closes each token half."""
    T, D = h_in.shape
    V = w.shape[0]
    kt = D // P
    nchunk = V // VC
    TH = 2048 if T >= 2048 else T
    nh = T // TH
    th_tiles = TH // P
    dx_out = nl.ndarray((T, D), dtype=h_in.dtype, buffer=nl.shared_hbm)
    dw_out = nl.ndarray((V, D), dtype=nl.float32, buffer=nl.shared_hbm)
    dcap_out = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=scal)
    inv_sp = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    inv_sn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sp, data=sc[:, 0:1])
    nisa.reciprocal(dst=inv_sn, data=sc[:, 1:2])
    dcap = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)       # per-partition partial d(loss)/d(sp), d(loss)/d(sn)
    nisa.memset(dcap, 0.0)
    for hh in range(nh):
        base = hh * TH
        # resident per half: h (bf16), hT (bf16, kt tiles per token tile), dh accumulator (fp32), per-row scalars
        h = nl.ndarray((P, th_tiles, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        hT = nl.ndarray((P, th_tiles, kt, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        dh = nl.ndarray((P, th_tiles, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dh, 0.0)
        for t in range(th_tiles):
            t0 = base + t * P
            nisa.dma_copy(dst=h[:, t, :], src=h_in[t0:t0 + P, :])
            for i in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=h[:, t, i * P:(i + 1) * P])
                nisa.tensor_copy(dst=hT[:, t, i, :], src=tps)
        for c in range(nchunk):
            wc = nl.ndarray((P, kt, D), dtype=nl.bfloat16, buffer=nl.sbuf)       # W rows of this chunk: 4 tiles (128 vocab, D)
            wTc = nl.ndarray((P, kt, VC), dtype=nl.bfloat16, buffer=nl.sbuf)     # W^T columns of this chunk
            for j in range(kt):
                nisa.dma_copy(dst=wc[:, j, :], src=w[c * VC + j * P:c * VC + (j + 1) * P, :])
                nisa.dma_copy(dst=wTc[:, j, :], src=wT[j * P:(j + 1) * P, c * VC:(c + 1) * VC])
            dwps = []
            for j in range(kt):
                dwps.append(nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum))
            for t in range(th_tiles):
                t0 = base + t * P
                zps = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.psum)
                for i in range(kt):
                    nisa.nc_matmul(dst=zps, stationary=hT[:, t, i, :], moving=wTc[:, i, :], accumulate=(i > 0))
                z = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=z, src=zps)
                tp = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                tn = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=tp, op=nl.tanh, data=z, scale=inv_sp)
                nisa.activation(dst=tn, op=nl.tanh, data=z, scale=inv_sn)
                pos = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=pos, data=z, op0=nl.greater_equal, operand0=0.0)
                # zc = sn*tn + pos*(sp*tp - sn*tn); deriv = (1 - tn^2) + pos*((1 - tp^2) - (1 - tn^2)) = 1 - tn^2 - pos*(tp^2 - tn^2)
                a = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                b = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=a, data=tp, op0=nl.multiply, operand0=sc[:, 0:1])
                nisa.tensor_scalar(dst=b, data=tn, op0=nl.multiply, operand0=sc[:, 1:2])
                zc = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=zc, data1=a, data2=b, op=nl.subtract)
                nisa.tensor_tensor(dst=zc, data1=zc, data2=pos, op=nl.multiply)
                nisa.tensor_tensor(dst=zc, data1=zc, data2=b, op=nl.add)
                nisa.tensor_tensor(dst=a, data1=tp, data2=tp, op=nl.multiply)
                nisa.tensor_tensor(dst=b, data1=tn, data2=tn, op=nl.multiply)
                deriv = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deriv, data1=a, data2=b, op=nl.subtract)
                nisa.tensor_tensor(dst=deriv, data1=deriv, data2=pos, op=nl.multiply)
                nisa.tensor_tensor(dst=deriv, data1=deriv, data2=b, op=nl.add)                 # tn^2 + pos*(tp^2 - tn^2)
                nisa.tensor_scalar(dst=deriv, data=deriv, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
                pr = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)                     # d loss / d zc for this chunk
                nisa.dma_copy(dst=pr, src=dl_in[t0:t0 + P, c * VC:(c + 1) * VC])
                dz = nl.ndarray((P, VC), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dz, data1=pr, data2=deriv, op=nl.multiply)
                # softcap parameter grads: d zc/d sp = tp - (z/sp)(1 - tp^2) on z >= 0; d zc/d sn = tn - (z/sn)(1 - tn^2) on z < 0
                u = nl.ndarray((P, VC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=u, data=a, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)   # 1 - tp^2
                nisa.tensor_tensor(dst=u, data1=u, data2=z, op=nl.multiply)
                nisa.tensor_scalar(dst=u, data=u, op0=nl.multiply, operand0=inv_sp)
                nisa.tensor_tensor(dst=u, data1=tp, data2=u, op=nl.subtract)
                nisa.tensor_tensor(dst=u, data1=u, data2=pos, op=nl.multiply)
                nisa.tensor_tensor(dst=u, data1=u, data2=pr, op=nl.multiply)
                part = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
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
                nisa.tensor_tensor(dst=dcap[:, 1:2], data1=dcap[:, 1:2], data2=part, op=nl.add)
                # dh[t] += dz @ W_chunk  (contract over the chunk's 512 vocab entries: 4 sub-tiles)
                dhps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                for j in range(kt):
                    tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=dz[:, j * P:(j + 1) * P])
                    dzT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dzT, src=tps)
                    nisa.nc_matmul(dst=dhps, stationary=dzT, moving=wc[:, j, :], accumulate=(j > 0))
                nisa.tensor_tensor(dst=dh[:, t, :], data1=dh[:, t, :], data2=dhps, op=nl.add)
                # dW_chunk[j] += dz[:, j]^T @ h[t]  (contract over the 128 tokens of this tile)
                for j in range(kt):
                    nisa.nc_matmul(dst=dwps[j], stationary=dz[:, j * P:(j + 1) * P], moving=h[:, t, :], accumulate=(t > 0))
            for j in range(kt):
                r0 = c * VC + j * P
                if hh == 0:
                    dwsb = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dwsb, src=dwps[j])
                    nisa.dma_copy(dst=dw_out[r0:r0 + P, :], src=dwsb)
                else:
                    prev = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=prev, src=dw_out[r0:r0 + P, :])
                    nisa.tensor_tensor(dst=prev, data1=prev, data2=dwps[j], op=nl.add)
                    nisa.dma_copy(dst=dw_out[r0:r0 + P, :], src=prev)
        # rmsnorm backward: dx = rstd * (dh - h * mean_D(dh * h))
        for t in range(th_tiles):
            t0 = base + t * P
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rstd, src=rstd_in[t0:t0 + P, :])
            hf = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hf, src=h[:, t, :])
            prod = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=dh[:, t, :], data2=hf, op=nl.multiply)
            mdot = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mdot, op=nl.add, data=prod, axis=(1,))
            nisa.tensor_scalar(dst=mdot, data=mdot, op0=nl.multiply, operand0=-1.0 / D)
            nisa.tensor_scalar(dst=hf, data=hf, op0=nl.multiply, operand0=mdot)
            nisa.tensor_tensor(dst=hf, data1=hf, data2=dh[:, t, :], op=nl.add)
            dx = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dx, data=hf, op0=nl.multiply, operand0=rstd)
            nisa.dma_copy(dst=dx_out[t0:t0 + P, :], src=dx)
    nisa.dma_copy(dst=dcap_out, src=dcap)
    return dx_out, dw_out, dcap_out


def reference_fwd(x, w, tgt, sp, sn, eps=0.0078125):
    import torch, torch.nn.functional as F
    h = F.rms_norm(x.float(), (x.shape[-1],), eps=eps).bfloat16().float()
    z = h @ w.float().T
    zc = torch.where(z >= 0, sp * torch.tanh(z / sp), sn * torch.tanh(z / sn))
    lse = torch.logsumexp(zc, dim=-1)
    t = tgt.view(-1).long()
    valid = t >= 0
    tl = zc.gather(1, t.clamp_min(0).unsqueeze(1)).squeeze(1)
    return (lse - tl) * valid.float(), lse, h


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    T, D, V = 256, 512, 8192
    x = torch.randn(T, D).bfloat16(); w = (torch.randn(V, D) * 0.05).bfloat16()
    tgt = torch.randint(0, V, (T, 1), dtype=torch.int32); tgt[3] = -1
    sp, sn = 11.0, 9.0
    scal = torch.tensor([sp, sn, 0.0, 0.0]).expand(128, 4).contiguous()
    loss, lse, h, rstd = nki.simulate(head_fwd)(x, w.t().contiguous(), tgt, scal)
    rl, rlse, rh = reference_fwd(x, w, tgt, sp, sn)
    for name, got, ref in (("loss", loss.view(-1), rl), ("lse", lse.view(-1), rlse), ("h", h, rh)):
        g = (got if isinstance(got, torch.Tensor) else torch.as_tensor(np.asarray(got))).float()
        err = (g - ref).abs().max().item()
        print(f"{name}: max err {err:.4f} (ref max {ref.abs().max().item():.3f})")
    print("mean loss", loss.float().mean().item(), "ref", rl.mean().item())
    # backward check: upstream grad dl = 1/n_valid per row (mean over valid rows)
    xr = x.clone().float().requires_grad_(True); wr = w.clone().float().requires_grad_(True)
    spr = torch.tensor(sp, requires_grad=True); snr = torch.tensor(sn, requires_grad=True)
    hr = torch.nn.functional.rms_norm(xr, (D,), eps=0.0078125)
    zr = hr.bfloat16().float() @ wr.T
    zcr = torch.where(zr >= 0, spr * torch.tanh(zr / spr), snr * torch.tanh(zr / snr))
    t = tgt.view(-1).long(); valid = t >= 0
    lossr = torch.nn.functional.cross_entropy(zcr[valid], t[valid])
    lossr.backward()
    dl = torch.full((T, 1), 1.0 / valid.sum().item(), dtype=torch.float32)
    dx, dw, dcap = nki.simulate(head_bwd)(h, rstd, lse, dl, w, w.t().contiguous(), tgt, scal)
    cvt = lambda a: (a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))).float()
    dx = cvt(dx); dw = cvt(dw); dcap = cvt(dcap).sum(0)
    print(f"dsp: kernel {dcap[0].item():.5e} ref {spr.grad.item():.5e}; dsn: kernel {dcap[1].item():.5e} ref {snr.grad.item():.5e}")
    for name, got, ref in (("dx", dx, xr.grad), ("dW", dw, wr.grad)):
        err = (got - ref).abs().max().item(); print(f"{name}: max err {err:.3e} (ref max {ref.abs().max().item():.3e}) rel {err / ref.abs().max().item():.4f}")
