#!/usr/bin/env python3
"""Stage-1 megakernel work (2026-09-08): the POST-attention half of a transformer layer as ONE NKI kernel.
  a  = y @ Wo^T            (attention output projection; y (T, D) bf16 is the attention kernel's output, tokens in rows)
  x1 = xm + a              (residual; xm = the mixed residual stream that fed the attention)
  h2 = rmsnorm(x1)
  m  = relu(h2 @ W1^T)^2 @ W2^T
  out = x1 + m
Returns out (T, D), x1 (T, D) and preT (H, T) (saved for the backward). Weights in the natural (out, in) layout; T % 512 == 0,
D = 512, H = 2048. Built on nki_mlp.mlp_fwd (same tiling: 512-token chunks, hidden kept transposed). Dev harness: simulator."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
FMAX = 512


@nki.jit
def post_attn_fwd(y_in: nl.ndarray, wo: nl.ndarray, xm_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, eps: float = 1e-6):
    T, D = y_in.shape
    H = w1.shape[0]
    out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)
    x1_out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((H, T), dtype=y_in.dtype, buffer=nl.shared_hbm)
    kt = D // P
    ht = H // P
    nprog = nl.num_programs(0)
    chunks = T // FMAX
    per = chunks // nprog
    # resident transposed weight tiles: WoT[i] = [128 (i), D (o)], W1t[k] = [128 (k), H], W2t[h] = [128 (h), D]
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
    for ci in nl.affine_range(per):
        c = nl.program_id(0) * per + ci
        t0 = c * FMAX
        # ---- out-projection + residual + rmsnorm, per 128-token tile; h2T built for the MLP contraction
        h2T = []
        for _j in range(kt):
            h2T.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        x1_tiles = []
        for _j in range(FMAX // P):
            x1_tiles.append(nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf))
        for tb in range(FMAX // P):
            yt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=yt, src=y_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            # yT tiles [128 (i), 128 tokens] via transposes; a[t, :] = sum_i yT[i, t] WoT[i, :]
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
            nisa.tensor_tensor(dst=x1, data1=xmt, data2=aps, op=nl.add)        # x1 = xm + a (fp32)
            x1b = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x1b, src=x1)
            nisa.dma_copy(dst=x1_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=x1b)
            # rmsnorm over D
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
        # ---- MLP (as nki_mlp.mlp_fwd): hT[h] = relu(W1t[:, h]^T h2T)^2 ; m[t] = sum_h hT[h][:, t]^T W2[h]
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
            nisa.tensor_tensor(dst=o, data1=x1_tiles[tb], data2=mps, op=nl.add)   # out = x1 + m
            nisa.dma_copy(dst=out[t0 + tb * P:t0 + (tb + 1) * P, :], src=o)
    return out, x1_out, pre_out


def reference(y, wo, xm, w1, w2, eps=1e-6):
    import torch, torch.nn.functional as F
    a = y.float() @ wo.float().T
    x1 = xm.float() + a
    h2 = F.rms_norm(x1, (x1.shape[-1],), eps=eps)
    pre = h2 @ w1.float().T
    m = torch.relu(pre) ** 2 @ w2.float().T
    return x1 + m, x1, pre.T


def _main():
    import torch
    torch.manual_seed(0)
    T, D, H = 512, 512, 2048
    y = (torch.randn(T, D) * 0.5).bfloat16(); xm = (torch.randn(T, D)).bfloat16()
    wo = (torch.randn(D, D) * 0.03).bfloat16(); w1 = (torch.randn(H, D) * 0.04).bfloat16(); w2 = (torch.randn(D, H) * 0.02).bfloat16()
    out, x1, preT = nki.simulate(post_attn_fwd)(y, wo, xm, w1, w2)
    ro, rx1, rpre = reference(y, wo, xm, w1, w2)
    for name, got, ref in (("out", out, ro), ("x1", x1, rx1), ("preT", preT, rpre)):
        g = (got if isinstance(got, torch.Tensor) else torch.as_tensor(np.asarray(got))).float()
        err = (g - ref).abs().max().item(); scale = ref.abs().max().item()
        print(f"{name}: max err {err:.4f} (ref max {scale:.2f}, rel {err / scale:.4f})")


@nki.jit
def pre_attn_fwd(x_in: nl.ndarray, x0_in: nl.ndarray, ve_in: nl.ndarray, wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray,
                 cos_in: nl.ndarray, sin_in: nl.ndarray, scal: nl.ndarray, n_head: int = 4, eps: float = 1e-6):
    """PRE-attention half of a layer as one kernel (T tokens = b*t rows, D = 512, head dim 128):
      xm = s0*x + s1*x0                    (scal[:, 0], scal[:, 1]: resid_lambda, x0_lambda as a (128, 4) fp32 tensor)
      h  = rmsnorm(xm)
      q  = rope(h @ Wq^T), k = rope(h @ Wk^T), both rms-normed per head; v = h @ Wv^T + s2 * ve   (s2 = ve_lambda)
    Returns q, k, v as (n_head, T, 128) bf16 (the attention kernel's (h, t, d) layout per batch row), xm (T, D) bf16, and the
    pre-norm rotated q and k (rq, rk: (n_head, T, 128) bf16) that the QK-norm backward needs.
    cos/sin: (T, 64) fp32 tables of the split-half rotation (rows = token positions of the T rows)."""
    T, D = x_in.shape
    HD = D // n_head
    half = HD // 2
    q_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    v_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    xm_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    rq_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    rk_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
    kt = D // P
    nprog = nl.num_programs(0)
    tiles = T // P
    per = tiles // nprog
    sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=scal)
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
    for ti in nl.affine_range(per):
        tb = nl.program_id(0) * per + ti
        t0 = tb * P
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
                nisa.tensor_scalar(dst=ob, data=rot, op0=nl.multiply, operand0=rs2)
                nisa.dma_copy(dst=dst_t[hh, t0:t0 + P, :], src=ob)
            vb = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=vb, src=v[:, c0:c0 + HD])
            nisa.dma_copy(dst=v_out[hh, t0:t0 + P, :], src=vb)
    return q_out, k_out, v_out, xm_out, rq_out, rk_out


def reference_pre(x, x0, ve, wq, wk, wv, cos, sin, scal, n_head=4, eps=1e-6):
    import torch, torch.nn.functional as F
    s0, s1, s2 = (float(scal[0, i]) for i in range(3))
    xm = s0 * x.float() + s1 * x0.float()
    h = F.rms_norm(xm, (xm.shape[-1],), eps=eps).bfloat16().float()
    T, D = x.shape; HD = D // n_head; half = HD // 2

    def proj_rope(w):
        p = (h @ w.float().T).view(T, n_head, HD)
        x1, x2 = p[..., :half], p[..., half:]
        c, s = cos.view(T, 1, half), sin.view(T, 1, half)
        r = torch.cat([x1 * c + x2 * s, x2 * c - x1 * s], dim=-1)
        return F.rms_norm(r, (HD,), eps=eps).permute(1, 0, 2)
    q, k = proj_rope(wq), proj_rope(wk)
    v = ((h @ wv.float().T) + s2 * ve.float()).view(T, n_head, HD).permute(1, 0, 2)
    return q, k, v, xm


if __name__ == "__main__":
    import torch
    _main()
    torch.manual_seed(1)
    T, D = 256, 512
    x = torch.randn(T, D).bfloat16(); x0 = torch.randn(T, D).bfloat16(); ve = torch.randn(T, D).bfloat16()
    wq, wk, wv = ((torch.randn(D, D) * 0.03).bfloat16() for _ in range(3))
    pos = torch.arange(T, dtype=torch.float32); inv = 1.0 / (10000 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    ang = torch.outer(pos, inv); cos, sin = ang.cos(), ang.sin()
    scal = torch.tensor([1.0, 0.05, 0.5, 0.0]).expand(128, 4).contiguous()
    q, k, v, xm, rq, rk = nki.simulate(pre_attn_fwd)(x, x0, ve, wq, wk, wv, cos, sin, scal)
    rq, rk, rv, rxm = reference_pre(x, x0, ve, wq, wk, wv, cos, sin, scal)
    for name, got, ref in (("q", q, rq), ("k", k, rk), ("v", v, rv), ("xm", xm, rxm)):
        g = (got if isinstance(got, torch.Tensor) else torch.as_tensor(np.asarray(got))).float()
        err = (g - ref).abs().max().item(); scale = ref.abs().max().item()
        print(f"pre {name}: max err {err:.4f} (ref max {scale:.2f}, rel {err / scale:.4f})")
