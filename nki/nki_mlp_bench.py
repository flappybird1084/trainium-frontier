#!/usr/bin/env python3
"""Device microbenchmark (one core): fused MLP y = relu(x W1)^2 W2 with the 3072-wide hidden kept in SBUF (the production
post_attn kernel's structure: up-proj matmuls, relu^2 on the PSUM evacuation, PE transposes of the hidden, down-proj
matmuls), bf16 vs fp8 double-row operands (in-kernel casts, no scaling). HBM traffic is only x in / y out, so the kernel is
tensor-engine bound: 4 T D H flops = 103 GFLOP at T=16384. usage: python nki_mlp_bench.py"""
import time
import torch
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
NC = 512


@nki.jit
def mlp_bf16(xT: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
    D, T = xT.shape
    H = w1.shape[1]
    kt = D // P
    ht = H // P
    hch = H // NC
    tiles = T // P
    y = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    w1s = nl.ndarray((P, kt, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(kt):
        nisa.dma_copy(dst=w1s[:, i, :], src=w1[i * P:(i + 1) * P, :])
    w2s = nl.ndarray((P, ht, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(ht):
        nisa.dma_copy(dst=w2s[:, j, :], src=w2[j * P:(j + 1) * P, :])
    for tb in nl.affine_range(tiles):
        t0 = tb * P
        xs = nl.ndarray((P, kt, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(kt):
            nisa.dma_copy(dst=xs[:, i, :], src=xT[i * P:(i + 1) * P, t0:t0 + P])
        h = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(hch):
            ps = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.psum)
            for i in range(kt):
                nisa.nc_matmul(dst=ps, stationary=xs[:, i, :], moving=w1s[:, i, c * NC:(c + 1) * NC], accumulate=(i > 0))
            r = nl.ndarray((P, NC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=r, op=nl.relu, data=ps)
            nisa.tensor_tensor(dst=h[:, c * NC:(c + 1) * NC], data1=r, data2=r, op=nl.multiply)
        hT = nl.ndarray((P, ht, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(ht):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=h[:, j * P:(j + 1) * P])
            nisa.tensor_copy(dst=hT[:, j, :], src=tps)
        ps2 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
        for j in range(ht):
            nisa.nc_matmul(dst=ps2, stationary=hT[:, j, :], moving=w2s[:, j, :], accumulate=(j > 0))
        ob = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ob, src=ps2)
        nisa.dma_copy(dst=y[t0:t0 + P, :], src=ob)
    return y


@nki.jit
def mlp_fp8(xT: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
    # double-row layout: contraction index k = i*256 + j*128 + p lives at [p, i, j] in both operands
    D, T = xT.shape
    H = w1.shape[1]
    kh = D // (2 * P)
    hh = H // (2 * P)
    hch = H // NC
    tiles = T // P
    y = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    w1s = nl.ndarray((P, kh, 2, H), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for i in range(kh):
        wb = nl.ndarray((P, 2, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(2):
            nisa.dma_copy(dst=wb[:, j, :], src=w1[i * 2 * P + j * P:i * 2 * P + (j + 1) * P, :])
        nisa.tensor_copy(dst=w1s[:, i, :, :], src=wb)
    w2s = nl.ndarray((P, hh, 2, D), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for i in range(hh):
        wb = nl.ndarray((P, 2, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(2):
            nisa.dma_copy(dst=wb[:, j, :], src=w2[i * 2 * P + j * P:i * 2 * P + (j + 1) * P, :])
        nisa.tensor_copy(dst=w2s[:, i, :, :], src=wb)
    for tb in nl.affine_range(tiles):
        t0 = tb * P
        xb = nl.ndarray((P, kh, 2, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(kh):
            for j in range(2):
                nisa.dma_copy(dst=xb[:, i, j, :], src=xT[i * 2 * P + j * P:i * 2 * P + (j + 1) * P, t0:t0 + P])
        xs = nl.ndarray((P, kh, 2, P), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for i in range(kh):
            nisa.tensor_copy(dst=xs[:, i, :, :], src=xb[:, i, :, :])
        h = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(hch):
            ps = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.psum)
            for i in range(kh):
                nisa.nc_matmul(dst=ps, stationary=xs[:, i, :, :], moving=w1s[:, i, :, c * NC:(c + 1) * NC],
                               accumulate=(i > 0), perf_mode=nisa.matmul_perf_mode.double_row)
            r = nl.ndarray((P, NC), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=r, op=nl.relu, data=ps)
            nisa.tensor_tensor(dst=h[:, c * NC:(c + 1) * NC], data1=r, data2=r, op=nl.multiply)
        hT = nl.ndarray((P, hh, 2, P), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for i in range(hh):
            for j in range(2):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=h[:, (i * 2 + j) * P:(i * 2 + j + 1) * P])
                nisa.tensor_copy(dst=hT[:, i, j, :], src=tps)
        ps2 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
        for i in range(hh):
            nisa.nc_matmul(dst=ps2, stationary=hT[:, i, :, :], moving=w2s[:, i, :, :], accumulate=(i > 0),
                           perf_mode=nisa.matmul_perf_mode.double_row)
        ob = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ob, src=ps2)
        nisa.dma_copy(dst=y[t0:t0 + P, :], src=ob)
    return y


def ref_mlp(x, w1, w2):
    h = torch.relu(x.float() @ w1.float()) ** 2
    return h.bfloat16().float() @ w2.float()


def run(T, D, H, reps=20):
    from torch_neuronx.nki_hop import wrap_nki
    dev = torch.device("neuron")
    torch.manual_seed(0)
    x = torch.randn(T, D).bfloat16()
    w1 = (torch.randn(D, H) * (D ** -0.5)).bfloat16()
    w2 = (torch.randn(H, D) * (H ** -0.5) * 0.5).bfloat16()
    ref = ref_mlp(x, w1, w2)
    xT = x.t().contiguous().to(dev); w1d = w1.to(dev); w2d = w2.to(dev)
    flops = 4 * T * D * H
    res = {}
    for name, kern in (("bf16", mlp_bf16), ("fp8", mlp_fp8)):
        k = wrap_nki(kern)
        try:
            y = k[(1,)](xT, w1d, w2d); torch.neuron.synchronize()
            d = y.float().cpu() - ref
            print(f"T={T} D={D} H={H}: {name} max err {d.abs().max().item():.3e} (ref max {ref.abs().max().item():.3e}) rel-norm {(d.norm() / ref.norm()).item():.4f}", flush=True)
            t0 = time.monotonic()
            for _ in range(reps): y = k[(1,)](xT, w1d, w2d)
            torch.neuron.synchronize(); ms = (time.monotonic() - t0) / reps * 1000
            res[name] = ms
            print(f"T={T} D={D} H={H}: {name} {ms:.3f} ms ({flops / ms / 1e9:.1f} TFLOP/s)", flush=True)
        except Exception as exc:
            print(f"T={T} D={D} H={H}: {name} FAILED: {str(exc)[:600]}", flush=True)
    if "bf16" in res and "fp8" in res:
        print(f"T={T} D={D} H={H}: fp8 speedup x{res['bf16'] / res['fp8']:.2f}", flush=True)


if __name__ == "__main__":
    for T in (4096, 16384):
        run(T, 512, 3072)
    print("done")
