#!/usr/bin/env python3
"""NKI kernels: fused ReLU^2 MLP forward and backward for the eager Trainium line.

Layouts: x (T, D) bf16, tokens in rows; w1 (H, D) and w2 (D, H) in the NATURAL (out, in) layout; T % 512 == 0, D = 512, H = 2048.
The forward builds the transposed weight tiles it needs with 128x128 SBUF-local TensorE transposes; the backward needs none.
Forward computes the hidden activation TRANSPOSED (hidden in partitions) so no big transposes are needed:
  preT[h, t] = sum_k W1t[k, h] x[t, k]      -> stationary = W1t[k-tile, h-block], moving = xT[k-tile, tokens]
  hT = relu(preT)^2 (bf16, SBUF)             y[t, o] = sum_h hT[h, t] W2t[h, o]  -> stationary = hT[h-tile, t-block], moving = W2t[h-tile, :]
The only transposes are x tiles (128x128 blocks on TensorE, SBUF-local). Saves preT (bf16) for the backward.
Backward (dy (T, D)):
  dhT[h, t] = sum_o W2t[h, o] dy[t, o]      -> stationary = W2 natural = W2t^T blocks: we transpose dy tiles instead and use
                                              stationary = dyT[o-tile, t-block]?  no: out[M=h, N=t] needs stationary[K=o, M=h] = W2t^T.
  We therefore keep W2t^T (= W2, (D, H)) as a second weight input; likewise W1t^T (= W1, (H, D)) for dx. The caller passes
  both layouts (cheap: the transposes of two 2 MB weights per step are done once in torch, or stored twice).
  dpreT = dhT * 2 relu(preT)
  dx[t, k]   = sum_h dpreT[h, t] W1[h, k]   -> stationary = dpreT[h-tile, t-block], moving = W1[h-tile, :]
  dW1t[k, h] = sum_t x[t, k] dpreT[h, t]    -> stationary = x[t-tile, k-block]... needs moving dpre[t-tile, h]: transpose dpreT blocks
  dW2t[h, o] = sum_t hT[h, t] dy[t, o]      -> stationary = hT^T? out[M=h, N=o] needs stationary[K=t, M=h] = h[t, h]: transpose hT blocks
Weight gradients accumulate over token tiles in fp32 SBUF accumulators (PSUM holds one [128, 512] bank per output tile).
Dev harness: python nki_mlp.py (simulator, small T)."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
FMAX = 512


@nki.jit
def mlp_fwd(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
    """x (T, D) bf16; w1 (H, D) and w2 (D, H) in the natural (out, in) layout. Returns y (T, D) bf16 and preT (H, T) bf16."""
    T, D = x_in.shape
    H = w1.shape[0]
    y_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((H, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    kt = D // P              # contraction tiles over D (4)
    ht = H // P              # hidden blocks of 128 (16)
    nprog = nl.num_programs(0)
    chunks = T // FMAX       # token chunks of 512
    per = chunks // nprog
    # weights resident in SBUF in the transposed tiles the matmuls need: W1t[k] = [128 (k), H], W2t[h] = [128 (h), D],
    # built from the natural (out, in) layouts with 128x128 TensorE block transposes (SBUF-local, cheap)
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
        # xT for this chunk: kt tiles of [128 (k), 512 (tokens)] via 128x128 TensorE transposes
        xT = []
        for _j in range(kt):
            xT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for tb in range(FMAX // P):
            xt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            for k in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=xt[:, k * P:(k + 1) * P])
                nisa.tensor_copy(dst=xT[k][:, tb * P:(tb + 1) * P], src=tps)
        # hT[h-block] = relu(W1t[:, h-block]^T xT)^2  -> ht tiles of [128 hidden, 512 tokens]
        hT = []
        for _j in range(ht):
            hT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for h in range(ht):
            pps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
            for k in range(kt):
                nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * P:(h + 1) * P], moving=xT[k], accumulate=(k > 0))
            pre = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pre, src=pps)
            nisa.dma_copy(dst=pre_out[h * P:(h + 1) * P, t0:t0 + FMAX], src=pre)
            nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)   # VectorE: at most one PSUM operand
            # relu(pre)^2 == max(pre,0)*pre ; wrong sign for pre<0? max(pre,0)*pre = 0 for pre<=0 and pre^2 for pre>0: correct
        # y[t-block] = sum_h hT[h][:, t-block]^T W2[h]  -> [128 tokens, D]
        for tb in range(FMAX // P):
            yps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            for h in range(ht):
                nisa.nc_matmul(dst=yps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=W2[h], accumulate=(h > 0))
            yt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=yt, src=yps)
            nisa.dma_copy(dst=y_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=yt)
    return y_out, pre_out


def reference_fwd(x, w1, w2):
    import torch
    pre = x.float() @ w1.float().t()
    h = torch.relu(pre) ** 2
    return (h @ w2.float().t()), pre.t()


if __name__ == "__main__":
    import torch, time
    torch.manual_seed(0)
    T, D, H = 1024, 512, 2048
    x = (torch.randn(T, D) * 0.5).bfloat16(); w1 = (torch.randn(H, D) * 0.03).bfloat16(); w2 = (torch.randn(D, H) * 0.02).bfloat16()
    t0 = time.time(); y, preT = nki.simulate(mlp_fwd)(x, w1, w2)
    ry, rpre = reference_fwd(x, w1, w2)
    print(f"fwd sim {time.time()-t0:.1f}s | y err {(y.float()-ry).abs().max():.2e} (|y| {ry.abs().max():.2f}) | preT err {(preT.float()-rpre).abs().max():.2e} (|pre| {rpre.abs().max():.2f})")


@nki.jit
def mlp_bwd(x_in: nl.ndarray, dy_in: nl.ndarray, pre_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
    """x (T, D), dy (T, D), preT (H, T) bf16; w1 (H, D), w2 (D, H) natural. Returns dx (T, D) bf16 and per-program fp32
    weight-gradient partials in the NATURAL layouts: dW1 (nprog, H, D), dW2 (nprog, D, H) (sum over the leading dim in torch)."""
    T, D = x_in.shape
    H = w1.shape[0]
    kt, ht, chunks = D // P, H // P, T // FMAX
    nprog = nl.num_programs(0)
    per = chunks // nprog
    pid = nl.program_id(0)
    dx_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    dw1_out = nl.ndarray((nprog, H, D), dtype=nl.float32, buffer=nl.shared_hbm)
    dw2_out = nl.ndarray((nprog, D, H), dtype=nl.float32, buffer=nl.shared_hbm)
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
    for ci in nl.sequential_range(per):
        c = pid * per + ci
        t0 = c * FMAX
        # dy tiles (natural) and dyT [o-tile][128 x 512 tokens]; x tiles (natural)
        dyn = []
        xn = []
        dyT = []
        for _j in range(kt):
            dyT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for tb in range(FMAX // P):
            d = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=d, src=dy_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            dyn.append(d)
            xx = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xx, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
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
            dxt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dxt, src=xps)
            nisa.dma_copy(dst=dx_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=dxt)
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
        nisa.dma_copy(dst=dw1_out[pid, h * P:(h + 1) * P, :], src=A1[h])
    for o in range(kt):
        nisa.dma_copy(dst=dw2_out[pid, o * P:(o + 1) * P, :], src=A2[o])
    return dx_out, dw1_out, dw2_out


def reference_bwd(x, dy, w1, w2):
    import torch
    xr = x.float().requires_grad_(True); a = w1.float().requires_grad_(True); b = w2.float().requires_grad_(True)
    y = torch.relu(xr @ a.t()) ** 2 @ b.t()
    y.backward(dy.float())
    return xr.grad, a.grad, b.grad


if __name__ == "__main__":
    import torch
    torch.manual_seed(1)
    T, D, H = 1024, 512, 2048
    x = (torch.randn(T, D) * 0.5).bfloat16(); w1 = (torch.randn(H, D) * 0.03).bfloat16(); w2 = (torch.randn(D, H) * 0.02).bfloat16()
    dy = (torch.randn(T, D) * 0.1).bfloat16()
    y, preT = nki.simulate(mlp_fwd)(x, w1, w2)
    t0 = time.time(); dx, dw1, dw2 = nki.simulate(mlp_bwd)(x, dy, preT, w1, w2)
    rdx, rdw1, rdw2 = reference_bwd(x, dy, w1, w2)
    e = lambda a, b: f"{(a.float()-b).abs().max():.2e} (|ref| {b.abs().max():.2e})"
    print(f"bwd sim {time.time()-t0:.1f}s | dx {e(dx, rdx)} | dW1 {e(dw1.sum(0), rdw1)} | dW2 {e(dw2.sum(0), rdw2)}")
