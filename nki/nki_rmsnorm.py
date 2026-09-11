#!/usr/bin/env python3
"""NKI kernels: RMSNorm (no weight, eps=1e-6 like F.rms_norm's default? the vehicle uses F.rms_norm(x, (d,)) with eps=None ->
torch uses eps = finfo(dtype).eps) forward and backward for x (rows, d) bf16, rows % 128 == 0.
  fwd: r = rsqrt(mean(x^2) + eps); y = x * r
  bwd: dx = r * (dy - x * r^2 * mean(dy * x))
Eager-line kernels (F.rms_norm backward is several launches). Harness: python nki_rmsnorm.py."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def rmsnorm_fwd(x_in: nl.ndarray, eps: float = 1e-6) -> nl.ndarray:
    rows, d = x_in.shape
    out = nl.ndarray((rows, d), dtype=x_in.dtype, buffer=nl.shared_hbm)
    nprog = nl.num_programs(0)
    per = (rows // P) // nprog
    for ii in nl.affine_range(per):
        i = nl.program_id(0) * per + ii
        x = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x, src=x_in[i * P:(i + 1) * P, :])
        sq = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
        nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=eps)
        nisa.activation(dst=ms, op=nl.rsqrt, data=ms)
        y = nl.ndarray((P, d), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y, data=x, op0=nl.multiply, operand0=ms)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=y)
    return out


@nki.jit
def rmsnorm_bwd(x_in: nl.ndarray, dy_in: nl.ndarray, eps: float = 1e-6) -> nl.ndarray:
    rows, d = x_in.shape
    out = nl.ndarray((rows, d), dtype=dy_in.dtype, buffer=nl.shared_hbm)
    nprog = nl.num_programs(0)
    per = (rows // P) // nprog
    for ii in nl.affine_range(per):
        i = nl.program_id(0) * per + ii
        x = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        dy = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x, src=x_in[i * P:(i + 1) * P, :])
        nisa.dma_copy(dst=dy, src=dy_in[i * P:(i + 1) * P, :])
        t = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=x, data2=x, op=nl.multiply)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ms, op=nl.add, data=t, axis=(1,))
        nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=eps)
        r = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=r, op=nl.rsqrt, data=ms)                       # r
        nisa.tensor_tensor(dst=t, data1=dy, data2=x, op=nl.multiply)        # dy*x
        c = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=c, op=nl.add, data=t, axis=(1,))             # sum(dy*x)
        r2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r2, data1=r, data2=r, op=nl.multiply)
        nisa.tensor_tensor(dst=c, data1=c, data2=r2, op=nl.multiply)
        nisa.tensor_scalar(dst=c, data=c, op0=nl.multiply, operand0=1.0 / d)   # r^2 * mean(dy*x)
        nisa.tensor_scalar(dst=t, data=x, op0=nl.multiply, operand0=c)          # x * that
        nisa.tensor_tensor(dst=t, data1=dy, data2=t, op=nl.subtract)            # dy - ...
        dx = nl.ndarray((P, d), dtype=dy_in.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dx, data=t, op0=nl.multiply, operand0=r)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=dx)
    return out


if __name__ == "__main__":
    import torch, torch.nn.functional as F, time
    torch.manual_seed(0)
    rows, d = 512, 512
    x = (torch.randn(rows, d) * 1.5).bfloat16()
    eps = torch.finfo(torch.bfloat16).eps
    xr = x.clone().float().requires_grad_(True)
    y = F.rms_norm(xr, (d,), eps=eps); dy = torch.randn(rows, d); y.backward(dy)
    t0 = time.time()
    yk = nki.simulate(rmsnorm_fwd)(x, eps=eps)
    dxk = nki.simulate(rmsnorm_bwd)(x, dy.bfloat16(), eps=eps)
    print(f"sim {time.time()-t0:.1f}s | fwd err {(yk.float()-y).abs().max():.2e} (|y| {y.abs().max():.2f}) | bwd err {(dxk.float()-xr.grad).abs().max():.2e} (|dx| {xr.grad.abs().max():.2e})")
