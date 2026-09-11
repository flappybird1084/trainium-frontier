#!/usr/bin/env python3
"""Device microbenchmark (one core): fp8 double-row matmul (NeuronCore-v3 `perf_mode=double_row`) vs bf16 at the MLP shapes
(T=4096 tokens x K=512 -> N=3072 and K=3072 -> N=512), including the in-kernel bf16 -> fp8 cast of both operands.
Both kernels read X^T (K, T) and W (K, N) from HBM, keep W resident in SBUF, and write y = X W (T, N) bf16.
usage: python nki_fp8_bench.py"""
import time
import torch
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
NC = 512


@nki.jit
def mm_bf16(xT: nl.ndarray, w: nl.ndarray):
    K, T = xT.shape
    N = w.shape[1]
    kt = K // P
    tiles = T // P
    nch = N // NC
    y = nl.ndarray((T, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    wsb = nl.ndarray((P, kt, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(kt):
        nisa.dma_copy(dst=wsb[:, i, :], src=w[i * P:(i + 1) * P, :])
    for tb in nl.affine_range(tiles):
        t0 = tb * P
        xs = nl.ndarray((P, kt, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(kt):
            nisa.dma_copy(dst=xs[:, i, :], src=xT[i * P:(i + 1) * P, t0:t0 + P])
        for c in range(nch):
            ps = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.psum)
            for i in range(kt):
                nisa.nc_matmul(dst=ps, stationary=xs[:, i, :], moving=wsb[:, i, c * NC:(c + 1) * NC], accumulate=(i > 0))
            ob = nl.ndarray((P, NC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob, src=ps)
            nisa.dma_copy(dst=y[t0:t0 + P, c * NC:(c + 1) * NC], src=ob)
    return y


@nki.jit
def mm_fp8(xT: nl.ndarray, w: nl.ndarray):
    # double-row layout: contraction index k = i*256 + j*128 + p lives at [p, i, j] in both operands (consistent on both
    # sides, so the hardware's (p, j) pairing order does not matter for the result)
    K, T = xT.shape
    N = w.shape[1]
    kh = K // (2 * P)
    tiles = T // P
    nch = N // NC
    y = nl.ndarray((T, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    wsb = nl.ndarray((P, kh, 2, N), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for i in range(kh):
        wb = nl.ndarray((P, 2, N), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(2):
            nisa.dma_copy(dst=wb[:, j, :], src=w[i * 2 * P + j * P:i * 2 * P + (j + 1) * P, :])
        nisa.tensor_copy(dst=wsb[:, i, :, :], src=wb)
    for tb in nl.affine_range(tiles):
        t0 = tb * P
        xb = nl.ndarray((P, kh, 2, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(kh):
            for j in range(2):
                nisa.dma_copy(dst=xb[:, i, j, :], src=xT[i * 2 * P + j * P:i * 2 * P + (j + 1) * P, t0:t0 + P])
        xs = nl.ndarray((P, kh, 2, P), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for i in range(kh):
            nisa.tensor_copy(dst=xs[:, i, :, :], src=xb[:, i, :, :])
        for c in range(nch):
            ps = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.psum)
            for i in range(kh):
                nisa.nc_matmul(dst=ps, stationary=xs[:, i, :, :], moving=wsb[:, i, :, c * NC:(c + 1) * NC],
                               accumulate=(i > 0), perf_mode=nisa.matmul_perf_mode.double_row)
            ob = nl.ndarray((P, NC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob, src=ps)
            nisa.dma_copy(dst=y[t0:t0 + P, c * NC:(c + 1) * NC], src=ob)
    return y


@nki.jit
def mm_bf16_s(xT: nl.ndarray, w: nl.ndarray):
    K, T = xT.shape
    N = w.shape[1]
    kt = K // P
    tiles = T // P
    nch = N // NC
    y = nl.ndarray((T, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    wsb = nl.ndarray((P, kt, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(kt):
        nisa.dma_copy(dst=wsb[:, i, :], src=w[i * P:(i + 1) * P, :])
    for tb in range(tiles):
        t0 = tb * P
        xs = nl.ndarray((P, kt, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(kt):
            nisa.dma_copy(dst=xs[:, i, :], src=xT[i * P:(i + 1) * P, t0:t0 + P])
        for c in range(nch):
            ps = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.psum)
            for i in range(kt):
                nisa.nc_matmul(dst=ps, stationary=xs[:, i, :], moving=wsb[:, i, c * NC:(c + 1) * NC], accumulate=(i > 0))
            ob = nl.ndarray((P, NC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob, src=ps)
            nisa.dma_copy(dst=y[t0:t0 + P, c * NC:(c + 1) * NC], src=ob)
    return y


@nki.jit
def mm_fp8_s(xT: nl.ndarray, w: nl.ndarray):
    # double-row layout: contraction index k = i*256 + j*128 + p lives at [p, i, j] in both operands (consistent on both
    # sides, so the hardware's (p, j) pairing order does not matter for the result)
    K, T = xT.shape
    N = w.shape[1]
    kh = K // (2 * P)
    tiles = T // P
    nch = N // NC
    y = nl.ndarray((T, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    wsb = nl.ndarray((P, kh, 2, N), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for i in range(kh):
        wb = nl.ndarray((P, 2, N), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(2):
            nisa.dma_copy(dst=wb[:, j, :], src=w[i * 2 * P + j * P:i * 2 * P + (j + 1) * P, :])
        nisa.tensor_copy(dst=wsb[:, i, :, :], src=wb)
    for tb in range(tiles):
        t0 = tb * P
        xb = nl.ndarray((P, kh, 2, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(kh):
            for j in range(2):
                nisa.dma_copy(dst=xb[:, i, j, :], src=xT[i * 2 * P + j * P:i * 2 * P + (j + 1) * P, t0:t0 + P])
        xs = nl.ndarray((P, kh, 2, P), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for i in range(kh):
            nisa.tensor_copy(dst=xs[:, i, :, :], src=xb[:, i, :, :])
        for c in range(nch):
            ps = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.psum)
            for i in range(kh):
                nisa.nc_matmul(dst=ps, stationary=xs[:, i, :, :], moving=wsb[:, i, :, c * NC:(c + 1) * NC],
                               accumulate=(i > 0), perf_mode=nisa.matmul_perf_mode.double_row)
            ob = nl.ndarray((P, NC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob, src=ps)
            nisa.dma_copy(dst=y[t0:t0 + P, c * NC:(c + 1) * NC], src=ob)
    return y


def run(T, K, N, reps=20, static=False):
    from torch_neuronx.nki_hop import wrap_nki
    dev = torch.device("neuron")
    torch.manual_seed(0)
    x = torch.randn(T, K).bfloat16()
    w = (torch.randn(K, N) * (K ** -0.5)).bfloat16()
    ref = (x.float() @ w.float())
    xT = x.t().contiguous().to(dev); wd = w.to(dev)
    kb, k8 = wrap_nki(mm_bf16_s if static else mm_bf16), wrap_nki(mm_fp8_s if static else mm_fp8)
    tag = "static" if static else "affine"
    yb = kb[(1,)](xT, wd); torch.neuron.synchronize()
    eb = (yb.float().cpu() - ref).abs().max().item()
    print(f"[{tag}] T={T} K={K} N={N}: bf16 kernel max err {eb:.3e} (ref max {ref.abs().max().item():.3e})", flush=True)
    try:
        y8 = k8[(1,)](xT, wd); torch.neuron.synchronize()
        e8 = (y8.float().cpu() - ref).abs().max().item()
        rel = ((y8.float().cpu() - ref).norm() / ref.norm()).item()
        print(f"[{tag}] T={T} K={K} N={N}: fp8 kernel max err {e8:.3e} rel-norm err {rel:.4f}", flush=True)
    except Exception as exc:
        print(f"[{tag}] T={T} K={K} N={N}: fp8 kernel FAILED: {str(exc)[:600]}", flush=True)
        y8 = None
    t0 = time.monotonic()
    for _ in range(reps): yb = kb[(1,)](xT, wd)
    torch.neuron.synchronize(); tb = (time.monotonic() - t0) / reps * 1000
    print(f"[{tag}] T={T} K={K} N={N}: bf16 {tb:.3f} ms ({2 * T * K * N / tb / 1e9:.1f} TFLOP/s)", flush=True)
    if y8 is not None:
        t0 = time.monotonic()
        for _ in range(reps): y8 = k8[(1,)](xT, wd)
        torch.neuron.synchronize(); t8 = (time.monotonic() - t0) / reps * 1000
        print(f"[{tag}] T={T} K={K} N={N}: fp8  {t8:.3f} ms ({2 * T * K * N / t8 / 1e9:.1f} TFLOP/s)  speedup x{tb / t8:.2f}", flush=True)


if __name__ == "__main__":
    import sys
    static = "static" in sys.argv[1:]
    for T in (4096, 16384):
        run(T, 512, 3072, static=static)
        run(T, 3072, 512, static=static)
    print("done")
