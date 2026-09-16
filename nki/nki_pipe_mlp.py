#!/usr/bin/env python3
"""Tile-level software-pipelining probe (ideas-2026-09-16 #1): the MLP block rmsnorm(x) -> W1 -> relu^2 -> W2 as ONE
kernel, written four ways that differ only in buffer allocation / loop form, to see whether the Neuron compiler overlaps
TensorE matmuls with the Vector/Scalar glue across token chunks.

Layouts: x (T, D) bf16 tokens in rows; w1t (D, H) and w2t (H, D) pre-transposed so y = relu(rmsnorm(x) @ w1t)^2 @ w2t;
T % 512 == 0, D % 128 == 0, H % 128 == 0. Chunk = 512 tokens = 4 contiguous 128-row DMAs (every DMA is a 128-row
aligned block, no strided patterns). Per chunk (the production post_attn_fwd structure, nki_layer.py):
  rmsnorm per 128-token tile (VectorE/ScalarE) -> 4 PE transposes -> h2T[k] [128 d, 512 tok]
  hT[h] = relu(W1t[:, h]^T h2T)^2   (4 accumulating matmuls per h-block; ScalarE relu on the PSUM read, VectorE square)
  y[t-block] = sum_h hT[h][:, t-block]^T W2t[h]  (24 accumulating matmuls; VectorE evacuation; DMA out)
Variants:
  mlp_auto    fresh nl.ndarray per chunk, compiler-managed addresses, nl.sequential_range  (= what nki_mlp.py / the vehicle do)
  mlp_pipe    nl.affine_range + ALL SBUF/PSUM placed by hand: NBUF-deep ring of chunk buffers, PSUM banks rotated (0-3 up-proj,
              4-5 down-proj, 6 transposes) so chunk c+1 has no WAR dependence on chunk c
  mlp_seq1    same manual layout but ONE buffer set / one PSUM bank per role: every chunk carries a WAR dependence on the
              previous one (forced-serial control)
  mlp_seq1_nr mlp_seq1 inside nl.no_reorder() (program order pinned: the strictest control)
NOTE (nki 0.5.0): nl.affine_range == nl.sequential_range == nl.static_range == fully unrolled at trace time (both former
deprecated), so the loop form itself carries no scheduling meaning; only the dependence structure does.
Simulator test: python nki_pipe_mlp_test.py ; device bench: python nki_pipe_mlp_bench.py (idle device only)."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
FMAX = 512
EPS = 1e-6
NBUF = 3
BANK = 2048            # PSUM bank: 512 fp32 per partition
# manual SBUF map, bytes per partition (weights first, then NBUF chunk sets)
W1B = 0                # 4 x [128, 3072] bf16 = 4 x 6144
W2B = 24576            # 24 x [128, 512] bf16 = 24 x 1024
SETB = 49152
OFF_XS = 0             # 4 x 1024 bf16 x tiles
OFF_SQ = 4096          # 2 x 2048 fp32 squares
OFF_MS = 8192          # 4 x 64 fp32 [128,1] sums
OFF_RS = 8448          # 4 x 64 fp32 [128,1] rstd
OFF_H2 = 9216          # 2 x 1024 bf16 normed tile
OFF_H2T = 11264        # 4 x 1024 bf16 transposed normed chunk
OFF_R = 15360          # 2 x 1024 bf16 relu tmp
OFF_HT = 17408         # 24 x 1024 bf16 hidden (transposed)
OFF_OB = 41984         # 2 x 1024 bf16 output tile
SETSZ = 44032          # 43 KiB per set; 3 sets + 48 KiB weights = 177 KiB of 192 KiB


@nki.jit
def mlp_auto(x_in: nl.ndarray, w1t: nl.ndarray, w2t: nl.ndarray):
    T, D = x_in.shape
    H = w1t.shape[1]
    kt = D // P
    ht = H // P
    chunks = T // FMAX
    y_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    w1s = []
    for k in range(kt):
        w = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=w1t[k * P:(k + 1) * P, :])
        w1s.append(w)
    w2s = []
    for h in range(ht):
        w = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=w2t[h * P:(h + 1) * P, :])
        w2s.append(w)
    for c in nl.sequential_range(chunks):
        t0 = c * FMAX
        h2T = []
        for _j in range(kt):
            h2T.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for tb in range(FMAX // P):
            xs = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xs, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=xs, data2=xs, op=nl.multiply)
            ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h2 = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h2, data=xs, op0=nl.multiply, operand0=rstd)
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
                nisa.nc_matmul(dst=pps, stationary=w1s[k][:, h * P:(h + 1) * P], moving=h2T[k], accumulate=(k > 0))
            r = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=r, op=nl.relu, data=pps)
            nisa.tensor_tensor(dst=hT[h], data1=r, data2=r, op=nl.multiply)
        for tb in range(FMAX // P):
            mps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            for h in range(ht):
                nisa.nc_matmul(dst=mps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=w2s[h], accumulate=(h > 0))
            ob = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob, src=mps)
            nisa.dma_copy(dst=y_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=ob)
    return y_out


@nki.jit
def mlp_pipe(x_in: nl.ndarray, w1t: nl.ndarray, w2t: nl.ndarray):
    T, D = x_in.shape
    H = w1t.shape[1]
    kt = D // P
    ht = H // P
    chunks = T // FMAX
    y_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    w1s = []
    for k in range(kt):
        w = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, W1B + k * 6144))
        nisa.dma_copy(dst=w, src=w1t[k * P:(k + 1) * P, :])
        w1s.append(w)
    w2s = []
    for h in range(ht):
        w = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, W2B + h * 1024))
        nisa.dma_copy(dst=w, src=w2t[h * P:(h + 1) * P, :])
        w2s.append(w)
    for c in nl.affine_range(chunks):
        t0 = c * FMAX
        base = SETB + (c % NBUF) * SETSZ
        h2T = []
        for k in range(kt):
            h2T.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_H2T + k * 1024)))
        for tb in range(FMAX // P):
            xs = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_XS + tb * 1024))
            nisa.dma_copy(dst=xs, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf, address=(0, base + OFF_SQ + (tb % 2) * 2048))
            nisa.tensor_tensor(dst=sq, data1=xs, data2=xs, op=nl.multiply)
            ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf, address=(0, base + OFF_MS + tb * 64))
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf, address=(0, base + OFF_RS + tb * 64))
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h2 = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_H2 + (tb % 2) * 1024))
            nisa.tensor_scalar(dst=h2, data=xs, op0=nl.multiply, operand0=rstd)
            for k in range(kt):
                slot = (c * 4 + tb) * kt + k
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 6 * BANK + (slot % 8) * 256))
                nisa.nc_transpose(dst=tps, data=h2[:, k * P:(k + 1) * P])
                nisa.tensor_copy(dst=h2T[k][:, tb * P:(tb + 1) * P], src=tps)
        hT = []
        for h in range(ht):
            hT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_HT + h * 1024)))
        for h in range(ht):
            j = c * ht + h
            pps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum, address=(0, (j % 4) * BANK))
            for k in range(kt):
                nisa.nc_matmul(dst=pps, stationary=w1s[k][:, h * P:(h + 1) * P], moving=h2T[k], accumulate=(k > 0))
            r = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_R + (h % 2) * 1024))
            nisa.activation(dst=r, op=nl.relu, data=pps)
            nisa.tensor_tensor(dst=hT[h], data1=r, data2=r, op=nl.multiply)
        for tb in range(FMAX // P):
            mps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum, address=(0, (4 + tb % 2) * BANK))
            for h in range(ht):
                nisa.nc_matmul(dst=mps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=w2s[h], accumulate=(h > 0))
            ob = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_OB + (tb % 2) * 1024))
            nisa.tensor_copy(dst=ob, src=mps)
            nisa.dma_copy(dst=y_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=ob)
    return y_out


@nki.jit
def mlp_seq1(x_in: nl.ndarray, w1t: nl.ndarray, w2t: nl.ndarray):
    T, D = x_in.shape
    H = w1t.shape[1]
    kt = D // P
    ht = H // P
    chunks = T // FMAX
    y_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    w1s = []
    for k in range(kt):
        w = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, W1B + k * 6144))
        nisa.dma_copy(dst=w, src=w1t[k * P:(k + 1) * P, :])
        w1s.append(w)
    w2s = []
    for h in range(ht):
        w = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, W2B + h * 1024))
        nisa.dma_copy(dst=w, src=w2t[h * P:(h + 1) * P, :])
        w2s.append(w)
    for c in nl.affine_range(chunks):
        t0 = c * FMAX
        base = SETB
        h2T = []
        for k in range(kt):
            h2T.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_H2T + k * 1024)))
        for tb in range(FMAX // P):
            xs = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_XS))
            nisa.dma_copy(dst=xs, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf, address=(0, base + OFF_SQ))
            nisa.tensor_tensor(dst=sq, data1=xs, data2=xs, op=nl.multiply)
            ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf, address=(0, base + OFF_MS))
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf, address=(0, base + OFF_RS))
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h2 = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_H2))
            nisa.tensor_scalar(dst=h2, data=xs, op0=nl.multiply, operand0=rstd)
            for k in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 6 * BANK))
                nisa.nc_transpose(dst=tps, data=h2[:, k * P:(k + 1) * P])
                nisa.tensor_copy(dst=h2T[k][:, tb * P:(tb + 1) * P], src=tps)
        hT = []
        for h in range(ht):
            hT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_HT + h * 1024)))
        for h in range(ht):
            pps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
            for k in range(kt):
                nisa.nc_matmul(dst=pps, stationary=w1s[k][:, h * P:(h + 1) * P], moving=h2T[k], accumulate=(k > 0))
            r = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_R))
            nisa.activation(dst=r, op=nl.relu, data=pps)
            nisa.tensor_tensor(dst=hT[h], data1=r, data2=r, op=nl.multiply)
        for tb in range(FMAX // P):
            mps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum, address=(0, 4 * BANK))
            for h in range(ht):
                nisa.nc_matmul(dst=mps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=w2s[h], accumulate=(h > 0))
            ob = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_OB))
            nisa.tensor_copy(dst=ob, src=mps)
            nisa.dma_copy(dst=y_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=ob)
    return y_out


@nki.jit
def mlp_seq1_nr(x_in: nl.ndarray, w1t: nl.ndarray, w2t: nl.ndarray):
    T, D = x_in.shape
    H = w1t.shape[1]
    kt = D // P
    ht = H // P
    chunks = T // FMAX
    y_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
    w1s = []
    for k in range(kt):
        w = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, W1B + k * 6144))
        nisa.dma_copy(dst=w, src=w1t[k * P:(k + 1) * P, :])
        w1s.append(w)
    w2s = []
    for h in range(ht):
        w = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, W2B + h * 1024))
        nisa.dma_copy(dst=w, src=w2t[h * P:(h + 1) * P, :])
        w2s.append(w)
    with nl.no_reorder():
        for c in nl.affine_range(chunks):
            t0 = c * FMAX
            base = SETB
            h2T = []
            for k in range(kt):
                h2T.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_H2T + k * 1024)))
            for tb in range(FMAX // P):
                xs = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_XS))
                nisa.dma_copy(dst=xs, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
                sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf, address=(0, base + OFF_SQ))
                nisa.tensor_tensor(dst=sq, data1=xs, data2=xs, op=nl.multiply)
                ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf, address=(0, base + OFF_MS))
                nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
                rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf, address=(0, base + OFF_RS))
                nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
                nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
                h2 = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_H2))
                nisa.tensor_scalar(dst=h2, data=xs, op0=nl.multiply, operand0=rstd)
                for k in range(kt):
                    tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 6 * BANK))
                    nisa.nc_transpose(dst=tps, data=h2[:, k * P:(k + 1) * P])
                    nisa.tensor_copy(dst=h2T[k][:, tb * P:(tb + 1) * P], src=tps)
            hT = []
            for h in range(ht):
                hT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_HT + h * 1024)))
            for h in range(ht):
                pps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
                for k in range(kt):
                    nisa.nc_matmul(dst=pps, stationary=w1s[k][:, h * P:(h + 1) * P], moving=h2T[k], accumulate=(k > 0))
                r = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_R))
                nisa.activation(dst=r, op=nl.relu, data=pps)
                nisa.tensor_tensor(dst=hT[h], data1=r, data2=r, op=nl.multiply)
            for tb in range(FMAX // P):
                mps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum, address=(0, 4 * BANK))
                for h in range(ht):
                    nisa.nc_matmul(dst=mps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=w2s[h], accumulate=(h > 0))
                ob = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base + OFF_OB))
                nisa.tensor_copy(dst=ob, src=mps)
                nisa.dma_copy(dst=y_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=ob)
    return y_out


VARIANTS = (("auto", mlp_auto), ("pipe", mlp_pipe), ("seq1", mlp_seq1), ("seq1_nr", mlp_seq1_nr))


def reference(x, w1t, w2t, eps=EPS):
    """Same rounding points as the kernels: bf16 normed input, bf16 relu, bf16 hidden, fp32 accumulation."""
    import torch
    import torch.nn.functional as F
    h2 = F.rms_norm(x.float(), (x.shape[-1],), eps=eps).bfloat16().float()
    r = torch.relu(h2 @ w1t.float()).bfloat16().float()
    hh = (r * r).bfloat16().float()
    return (hh @ w2t.float()).bfloat16()


def make_inputs(T, D, H, seed=0):
    import torch
    torch.manual_seed(seed)
    x = (torch.randn(T, D) * 0.7).bfloat16()
    w1t = (torch.randn(D, H) * (D ** -0.5)).bfloat16()
    w2t = (torch.randn(H, D) * (H ** -0.5) * 0.5).bfloat16()
    return x, w1t, w2t
