#!/usr/bin/env python3
"""Fused PRE-attention backward of a layer as one NKI kernel (2026-09-10), replacing the chain in `_bwd_impl`:
    gq = dq4 * scale ; gk = dk4 with the partial key offset undone (dims hq:2hq come from position p+1, masked) ; gv = dv4
    drq = rmsnorm_bwd(rq, gq) per head ; drk = rmsnorm_bwd(rk, gk)         (QK-norm backward, eps inside the rsqrt)
    dpq = rope^T(drq) ; dpk = rope^T(drk)                                   (split-half rotation, sin negated)
    h   = rmsnorm(xm) ; dh = dpq @ Wq + dpk @ Wk + gv @ Wv                  (dgrad, one PSUM accumulation)
    dxm = dx1 + rmsnorm_bwd(xm, dh)
    dWq = dpq^T h, dWk = dpk^T h, dWv = gv^T h                              (wgrad, contraction over tokens = partitions)
Layouts: dq/dk/dv (b*n_head, t, hd) bf16 (the attention kernel's (b, h, t, d) tensors, a free view); rq/rk (n_head, T, hd)
bf16 with T = b*t rows b-major; xm, dx1 (T, D) bf16; cos/sin (T, hd/2) fp32 per row; dksh (b*n_head, t, hq) bf16 = the
key-offset backward's shifted slice dk4[p+1, hq:2hq] * (1 - dstart[p+1]) (0 at the last position of every row), built in
torch so every kernel DMA is a 128-row-aligned block; wq/wk/wv (D, D) bf16 (out, in).
Returns dxm (T, D) bf16, gv2 (T, D) bf16, dWq/dWk/dWv (D, D) fp32. T % 128 == 0, hd == 128, D == n_head * hd."""
import torch
import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def pre_attn_bwd(dq_in: nl.ndarray, dk_in: nl.ndarray, dv_in: nl.ndarray, rq_in: nl.ndarray, rk_in: nl.ndarray,
                 xm_in: nl.ndarray, dx1_in: nl.ndarray, cos_in: nl.ndarray, sin_in: nl.ndarray, dksh_in: nl.ndarray,
                 wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray, n_head: int = 4, eps: float = 1e-6,
                 scale: float = 1.0, hq: int = 32):
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
    for tb in range(tiles):
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
                    # key-offset backward: dims hq:2hq of position p receive position p+1's masked grad (precomputed slice)
                    shb = nl.ndarray((P, hq), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=shb, src=dksh_in[bh, p0:p0 + P, :])
                    nisa.tensor_copy(dst=g[:, hq:2 * hq], src=shb)
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


def ref_pre_bwd(dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart, wq, wk, wv, scale, eps, hq):
    """The torch chain of `_bwd_impl` (pre-attention part) in fp32, for the tests. dq4/dk4/dv4: (b, NH, t, HD) bf16;
    rq/rk: (NH, T, HD); xm, dx1: (T, D); cos2d/sin2d: (t, HD/2); dstart (b, t); weights (D, D) (out, in)."""
    import torch.nn.functional as F
    b, NH, t, HD = dq4.shape
    T, D = xm.shape
    bf = torch.bfloat16

    def rms_bwd(x, dy):
        x = x.float(); dy = dy.float()
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return r * (dy - x * (r * r) * (dy * x).mean(-1, keepdim=True))

    def rope_t(x):   # transpose of the split-half rotation, per head
        x = x.float().view(T, NH, HD)
        c = cos2d.float().repeat(b, 1).view(T, 1, HD // 2); s = sin2d.float().repeat(b, 1).view(T, 1, HD // 2)
        x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).reshape(T, D)

    HQ = hq
    g = dk4[:, :, :, HQ:2 * HQ] * (1.0 - dstart).to(dk4.dtype).view(b, 1, t, 1)
    sh = torch.cat([g[:, :, 1:], torch.zeros_like(g[:, :, :1])], dim=2)
    dk4 = torch.cat([dk4[..., :HQ], sh, dk4[..., 2 * HQ:]], dim=-1)
    gq2 = (dq4.permute(0, 2, 1, 3).reshape(T, D) * scale).to(bf)
    gk2 = dk4.permute(0, 2, 1, 3).reshape(T, D).to(bf)
    gv2 = dv4.permute(0, 2, 1, 3).reshape(T, D).to(bf).contiguous()
    rq2 = rq.permute(1, 0, 2).reshape(T * NH, HD); rk2 = rk.permute(1, 0, 2).reshape(T * NH, HD)
    drq = rms_bwd(rq2, gq2.reshape(T * NH, HD)).view(T, D)
    drk = rms_bwd(rk2, gk2.reshape(T * NH, HD)).view(T, D)
    dpq = rope_t(drq).to(bf); dpk = rope_t(drk).to(bf)
    hx = xm.float() * torch.rsqrt(xm.float().pow(2).mean(-1, keepdim=True) + eps)
    h = hx.to(bf)
    dh = dpq.float() @ wq.float() + dpk.float() @ wk.float() + gv2.float() @ wv.float()
    dWq = dpq.float().t() @ h.float(); dWk = dpk.float().t() @ h.float(); dWv = gv2.float().t() @ h.float()
    dxm = dx1.float() + rms_bwd(xm, dh.to(bf))
    return dxm, gv2, dWq, dWk, dWv


def make_inputs(b, t, NH, HD, seed=0, device="cpu"):
    torch.manual_seed(seed)
    D = NH * HD; T = b * t; bf = torch.bfloat16
    dq4 = (torch.randn(b, NH, t, HD) * 0.05).to(bf); dk4 = (torch.randn(b, NH, t, HD) * 0.05).to(bf)
    dv4 = (torch.randn(b, NH, t, HD) * 0.05).to(bf)
    rq = torch.randn(NH, T, HD).to(bf); rk = torch.randn(NH, T, HD).to(bf)
    xm = torch.randn(T, D).to(bf); dx1 = (torch.randn(T, D) * 0.1).to(bf)
    ang = torch.outer(torch.arange(t).float(), 1.0 / (10000 ** (torch.arange(HD // 2).float() / (HD // 2))))
    cos2d = torch.cos(ang).contiguous(); sin2d = torch.sin(ang).contiguous()
    cos2d[:, HD // 4:] = 1.0; sin2d[:, HD // 4:] = 0.0      # half-truncated rope: the upper quarter of the pairs is untouched
    dstart = torch.zeros(b, t); dstart[:, 0] = 1.0; dstart[0, 100] = 1.0; dstart[b - 1, t - 1] = 1.0
    wq = (torch.randn(D, D) * D ** -0.5).to(bf); wk = (torch.randn(D, D) * D ** -0.5).to(bf); wv = (torch.randn(D, D) * D ** -0.5).to(bf)
    return dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart, wq, wk, wv


def key_offset_shift(dk4, dstart, hq):
    """(b*NH, t, hq) bf16: position p gets dk4[p+1, hq:2hq] * (1 - dstart[p+1]); the last position of a row gets 0."""
    b, NH, t, HD = dk4.shape
    g = dk4[:, :, 1:, hq:2 * hq] * (1.0 - dstart[:, 1:].to(dk4.dtype)).view(b, 1, t - 1, 1)
    return torch.cat([g, torch.zeros_like(g[:, :, :1])], dim=2).reshape(b * NH, t, hq).contiguous()


def kernel_args(dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart):
    """Torch-side glue that the layer op will do: free views + the per-row rope tables + the shift mask."""
    b, NH, t, HD = dq4.shape
    T = b * t
    cos_full = cos2d.repeat(b, 1).contiguous(); sin_full = sin2d.repeat(b, 1).contiguous()
    dksh = key_offset_shift(dk4, dstart, HD // 4)
    return (dq4.reshape(b * NH, t, HD).contiguous(), dk4.reshape(b * NH, t, HD).contiguous(),
            dv4.reshape(b * NH, t, HD).contiguous(), rq.contiguous(), rk.contiguous(), xm.contiguous(), dx1.contiguous(),
            cos_full, sin_full, dksh)


if __name__ == "__main__":
    import sys
    b, t, NH, HD = 2, 256, 4, 128
    scale, eps, hq = 0.1, 1e-6, HD // 4
    ins = make_inputs(b, t, NH, HD)
    dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart, wq, wk, wv = ins
    ref = ref_pre_bwd(dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart, wq, wk, wv, scale, eps, hq)
    args = kernel_args(dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart)
    outs = nki.simulate(pre_attn_bwd)(*args, wq, wk, wv, n_head=NH, eps=eps, scale=scale, hq=hq)
    for name, o, r in zip(("dxm", "gv2", "dWq", "dWk", "dWv"), outs, ref):
        o = torch.as_tensor(o).float(); r = r.float()
        err = (o - r).abs().max().item(); rel = ((o - r).norm() / r.norm()).item()
        print(f"{name}: max |diff| {err:.3e} (ref max {r.abs().max().item():.3e}) rel-norm {rel:.5f}")
