#!/usr/bin/env python3
"""Fused POST-attention backward of a layer as one NKI kernel (2026-09-10): mlp_bwd (unchanged math) + the recomputed
rmsnorm + its backward with the residual + the out-projection dgrad (written per head for the attention backward) + dWo.
Replaces in `_bwd_impl`: h2 = nfwd(x1); mlp_bwd; dx1 = d2 + nbwd(x1, dh2); dy2 = dx1 @ wo; dWo = dx1^T y2; dy4 permute."""
import torch
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
FMAX = 512


@nki.jit
def post_attn_bwd(dy_in: nl.ndarray, x1_in: nl.ndarray, pre_in: nl.ndarray, y_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray,
                  wo: nl.ndarray, n_head: int = 4, eps: float = 1e-6, t_len: int = 1024):
    """Fused POST-attention backward: h2 = rmsnorm(x1) (recomputed), the ReLU^2 MLP backward (as mlp_bwd), the rmsnorm
    backward with the residual (dx1 = dy + rmsnorm_bwd(x1, dh2)), the out-projection dgrad dy_attn = dx1 @ Wo written per
    head as (b*n_head, t, hd) bf16, and the weight grads dW1 (H, D), dW2 (D, H), dWo (D, D) fp32 (natural layouts).
    dy/x1/y (T, D) bf16, preT (H, T) bf16, w1 (H, D), w2 (D, H), wo (D, D) bf16 (out, in). Single program."""
    T, D = x1_in.shape
    H = w1.shape[0]
    HD = D // n_head
    kt, ht, chunks = D // P, H // P, T // FMAX
    tpt = t_len // P
    dyh_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    dx1_out = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    dw1_out = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.shared_hbm)
    dw2_out = nl.ndarray((D, H), dtype=nl.float32, buffer=nl.shared_hbm)
    dwo_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
    # resident out-projection weight (natural rows chunk o = the dgrad moving operand [K=out, N=in])
    Won = []
    for o in range(kt):
        wn = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wn, src=wo[o * P:(o + 1) * P, :])
        Won.append(wn)
    Ao = []
    for _j in range(kt):
        a = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(a, 0.0)
        Ao.append(a)
    # resident weights straight from the natural layouts (no transposes): W2T[o][h] = w2[o-tile, h-block], W1n[h] = w1[h-tile, :]
    W2T = []
    for o in range(kt):
        row = []
        for h in range(ht):
            blk = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=blk, src=w2[o * P:(o + 1) * P, h * P:(h + 1) * P])
            row.append(blk)
        W2T.append(row)
    W1n = []
    for h in range(ht):
        wn = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wn, src=w1[h * P:(h + 1) * P, :])
        W1n.append(wn)
    # fp32 gradient accumulators (SBUF) in the natural layouts: dW1 as ht tiles [128 (h), D]; dW2 as kt tiles [128 (o), H]
    A1 = []
    for _j in range(ht):
        a = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(a, 0.0)
        A1.append(a)
    A2 = []
    for _j in range(kt):
        a = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(a, 0.0)
        A2.append(a)
    for c in range(chunks):
        t0 = c * FMAX
        # dy tiles (natural) and dyT [o-tile][128 x 512 tokens]; x tiles (natural)
        dyn = []
        xn = []
        x1n = []
        rsn = []
        dyT = []
        for _j in range(kt):
            dyT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for tb in range(FMAX // P):
            d = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=d, src=dy_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            dyn.append(d)
            x1b = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=x1b, src=x1_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            x1f = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x1f, src=x1b)
            x1n.append(x1f)
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=x1f, data2=x1f, op=nl.multiply)
            ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            rsn.append(rstd)
            xx = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=xx, data=x1f, op0=nl.multiply, operand0=rstd)      # h2 = rmsnorm(x1)
            xn.append(xx)
            for o in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=d[:, o * P:(o + 1) * P])
                nisa.tensor_copy(dst=dyT[o][:, tb * P:(tb + 1) * P], src=tps)
        # dpreT[h] = 2 relu(preT[h]) * dhT[h],  dhT[h] = sum_o W2T[o][h]^T dyT[o]
        dpreT = []
        hT = []
        for h in range(ht):
            dps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
            for o in range(kt):
                nisa.nc_matmul(dst=dps, stationary=W2T[o][h], moving=dyT[o], accumulate=(o > 0))
            pre = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=pre, src=pre_in[h * P:(h + 1) * P, t0:t0 + FMAX])
            relu = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=relu, data=pre, op0=nl.maximum, operand0=0.0)
            hh = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=hh, data1=relu, data2=relu, op=nl.multiply)      # h = relu^2 (recomputed)
            hT.append(hh)
            dp = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=dp, data=relu, op0=nl.multiply, operand0=2.0, op1=nl.multiply, operand1=dps)
            dpreT.append(dp)
        # dx[t-block] = sum_h dpreT[h][:, t-block]^T W1n[h]
        for tb in range(FMAX // P):
            xps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            for h in range(ht):
                nisa.nc_matmul(dst=xps, stationary=dpreT[h][:, tb * P:(tb + 1) * P], moving=W1n[h], accumulate=(h > 0))
            # rmsnorm backward with the residual: dx1 = dy + rstd * (dh2 - x1 * rstd^2 * mean(dh2 * x1))
            dh2 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dh2, src=xps)
            x1f = x1n[tb]
            rstd = rsn[tb]
            pr = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr, data1=dh2, data2=x1f, op=nl.multiply)
            cm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=cm, op=nl.add, data=pr, axis=(1,))
            r2c = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r2c, data1=rstd, data2=rstd, op=nl.multiply)
            nisa.tensor_tensor(dst=cm, data1=cm, data2=r2c, op=nl.multiply)
            nisa.tensor_scalar(dst=cm, data=cm, op0=nl.multiply, operand0=1.0 / D)
            nisa.tensor_scalar(dst=pr, data=x1f, op0=nl.multiply, operand0=cm)
            nisa.tensor_tensor(dst=pr, data1=dh2, data2=pr, op=nl.subtract)
            dres = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dres, src=dyn[tb])
            nisa.scalar_tensor_tensor(dst=dres, data=pr, op0=nl.multiply, operand0=rstd, op1=nl.add, operand1=dres)
            dx1b = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dx1b, src=dres)
            nisa.dma_copy(dst=dx1_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=dx1b)
            # out-projection dgrad: dy_attn[tok, in] = sum_o dx1[tok, o] Wo[o, in]  (stationary = dx1^T chunks)
            yps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            for o in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=dx1b[:, o * P:(o + 1) * P])
                dT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dT, src=tps)
                nisa.nc_matmul(dst=yps, stationary=dT, moving=Won[o], accumulate=(o > 0))
            dya = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dya, src=yps)
            row = (t0 + tb * P) // t_len
            pos = t0 + tb * P - row * t_len
            for hh in range(n_head):
                nisa.dma_copy(dst=dyh_out[row * n_head + hh, pos:pos + P, :], src=dya[:, hh * HD:(hh + 1) * HD])
            # dWo[o-chunk, in] += dx1[tok, o-chunk]^T y[tok, in]
            yt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=yt, src=y_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            for o in range(kt):
                ops = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ops, stationary=dx1b[:, o * P:(o + 1) * P], moving=yt)
                nisa.tensor_tensor(dst=Ao[o], data1=Ao[o], data2=ops, op=nl.add)
        # weight gradients: per t-block build natural dpre[t-block] [128, H] and h[t-block] [128, H] by transposing blocks
        for tb in range(FMAX // P):
            dpn = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            hn = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            for h in range(ht):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=dpreT[h][:, tb * P:(tb + 1) * P])
                nisa.tensor_copy(dst=dpn[:, h * P:(h + 1) * P], src=tps)
                tps2 = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps2, data=hT[h][:, tb * P:(tb + 1) * P])
                nisa.tensor_copy(dst=hn[:, h * P:(h + 1) * P], src=tps2)
            # dW1[h-tile, :] += dpre_nat[t-block][:, h-tile]^T x[t-block]
            for h in range(ht):
                gps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=gps, stationary=dpn[:, h * P:(h + 1) * P], moving=xn[tb])
                nisa.tensor_tensor(dst=A1[h], data1=A1[h], data2=gps, op=nl.add)
            # dW2[o-tile, h-chunk] += dy[t-block][:, o-tile]^T h_nat[t-block][:, h-chunk]
            for o in range(kt):
                for hc in range(H // FMAX):
                    gps2 = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=gps2, stationary=dyn[tb][:, o * P:(o + 1) * P], moving=hn[:, hc * FMAX:(hc + 1) * FMAX])
                    nisa.tensor_tensor(dst=A2[o][:, hc * FMAX:(hc + 1) * FMAX], data1=A2[o][:, hc * FMAX:(hc + 1) * FMAX], data2=gps2, op=nl.add)
    for h in range(ht):
        nisa.dma_copy(dst=dw1_out[h * P:(h + 1) * P, :], src=A1[h])
    for o in range(kt):
        nisa.dma_copy(dst=dw2_out[o * P:(o + 1) * P, :], src=A2[o])
        nisa.dma_copy(dst=dwo_out[o * P:(o + 1) * P, :], src=Ao[o])
    return dyh_out, dx1_out, dw1_out, dw2_out, dwo_out


def ref_post_bwd(d2, x1, preT, y2, w1, w2, wo, n_head, eps):
    """fp32 torch reference of the chain (same rounding points as the vehicle's mlp_bwd where it matters)."""
    T, D = x1.shape
    b = None
    bf = torch.bfloat16
    x1f = x1.float()
    r = torch.rsqrt(x1f.pow(2).mean(-1, keepdim=True) + eps)
    h2 = (x1f * r).to(bf)
    pre = preT.float().t()                         # (T, H)
    relu = torch.relu(pre).to(bf).float()
    hid = (relu * relu).to(bf).float()
    d = d2.float()
    dhid = d @ w2.float()                          # (T, H)
    dpre = (2.0 * relu * dhid).to(bf).float()
    dh2 = dpre @ w1.float()                        # (T, D)
    dW1 = dpre.t() @ h2.float()
    dW2 = d.t() @ hid
    dx1 = d + r * (dh2 - x1f * (r * r) * (dh2 * x1f).mean(-1, keepdim=True))
    dx1b = dx1.to(bf)
    dy2 = (dx1b.float() @ wo.float()).to(bf)
    dWo = dx1b.float().t() @ y2.float()
    return dy2, dx1b, dW1, dW2, dWo


def make_inputs(b, t, NH, HD, H, seed=0):
    torch.manual_seed(seed)
    D = NH * HD; T = b * t; bf = torch.bfloat16
    d2 = (torch.randn(T, D) * 0.1).to(bf); x1 = torch.randn(T, D).to(bf); y2 = torch.randn(T, D).to(bf)
    w1 = (torch.randn(H, D) * D ** -0.5).to(bf); w2 = (torch.randn(D, H) * H ** -0.5).to(bf); wo = (torch.randn(D, D) * D ** -0.5).to(bf)
    h2 = (x1.float() * torch.rsqrt(x1.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(bf)
    preT = (h2.float() @ w1.float().t()).to(bf).t().contiguous()
    return d2, x1, preT, y2, w1, w2, wo


if __name__ == "__main__":
    b, t, NH, HD, H = 2, 256, 4, 128, 3072
    eps = 1e-6
    d2, x1, preT, y2, w1, w2, wo = make_inputs(b, t, NH, HD, H)
    ref = ref_post_bwd(d2, x1, preT, y2, w1, w2, wo, NH, eps)
    outs = nki.simulate(post_attn_bwd)(d2, x1, preT, y2, w1, w2, wo, n_head=NH, eps=eps, t_len=t)
    dyh = torch.as_tensor(outs[0]).float().view(b, NH, t, HD).permute(0, 2, 1, 3).reshape(b * t, NH * HD)
    got = [dyh] + [torch.as_tensor(o).float() for o in outs[1:]]
    for name, o, r in zip(("dy2", "dx1", "dW1", "dW2", "dWo"), got, ref):
        r = r.float(); err = (o - r).abs().max().item(); rel = ((o - r).norm() / r.norm()).item()
        print(f"{name}: max |diff| {err:.3e} (ref max {r.abs().max().item():.3e}) rel-norm {rel:.5f}")
