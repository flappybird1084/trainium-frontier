#!/usr/bin/env python3
"""NKI kernel: fused AdamW step for one 2-D fp32 parameter (R, C) with fp32 moments, in one pass over the data.

  p *= 1 - lr*wd ; m += (1-b1)(g - m) ; v += (1-b2)(g*g - v) ; p -= lr * (m/bias1) / (sqrt(v/bias2) + eps)

Per-step scalars (lr, bias1, bias2) arrive as a tiny fp32 tensor `scal` = [lr, 1/bias1, 1/sqrt(bias2), lr*wd] so the
NEFF is invariant across steps (a python float would bake into the kernel). Results are written IN PLACE into p, m, v
(HBM inputs) and p is also returned. The eager torch version is ~10 launches per tensor; this is one.
Dev harness: `python nki_adamw.py` runs the CPU simulator against a numpy reference."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def adamw_step(p_in: nl.ndarray, g_in: nl.ndarray, m_in: nl.ndarray, v_in: nl.ndarray, scal: nl.ndarray,
               beta1: float = 0.8, beta2: float = 0.995, eps: float = 1e-10):
    R, C = p_in.shape
    rt = R // P
    out = nl.ndarray((R, C), dtype=p_in.dtype, buffer=nl.shared_hbm)
    out_m = nl.ndarray((R, C), dtype=p_in.dtype, buffer=nl.shared_hbm)
    out_v = nl.ndarray((R, C), dtype=p_in.dtype, buffer=nl.shared_hbm)
    # the 4 per-step scalars, pre-broadcast by the caller to [P, 4]
    sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=scal)
    nprog = nl.num_programs(0)
    per = rt // nprog
    for ii in nl.affine_range(per):
        i = nl.program_id(0) * per + ii
        rows = slice(i * P, (i + 1) * P)
        p = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        g = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        m = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        v = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=p, src=p_in[rows, :])
        nisa.dma_copy(dst=g, src=g_in[rows, :])
        nisa.dma_copy(dst=m, src=m_in[rows, :])
        nisa.dma_copy(dst=v, src=v_in[rows, :])
        # p *= 1 - lr*wd   (scal[3] = lr*wd per partition)
        one_minus = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=one_minus, data=sc[:, 3:4], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        nisa.tensor_scalar(dst=p, data=p, op0=nl.multiply, operand0=one_minus)
        # m = b1*m + (1-b1)*g
        tmp = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=tmp, data=g, op0=nl.multiply, operand0=1.0 - beta1)
        nisa.scalar_tensor_tensor(dst=m, data=m, op0=nl.multiply, operand0=beta1, op1=nl.add, operand1=tmp)
        # v = b2*v + (1-b2)*g*g
        nisa.tensor_tensor(dst=tmp, data1=g, data2=g, op=nl.multiply)
        nisa.tensor_scalar(dst=tmp, data=tmp, op0=nl.multiply, operand0=1.0 - beta2)
        nisa.scalar_tensor_tensor(dst=v, data=v, op0=nl.multiply, operand0=beta2, op1=nl.add, operand1=tmp)
        # denom = sqrt(v) * (1/sqrt(bias2)) + eps ; upd = m * (1/bias1) / denom ; p -= lr * upd
        den = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=den, op=nl.sqrt, data=v)
        nisa.tensor_scalar(dst=den, data=den, op0=nl.multiply, operand0=sc[:, 2:3], op1=nl.add, operand1=eps)
        nisa.reciprocal(dst=den, data=den)
        nisa.scalar_tensor_tensor(dst=tmp, data=m, op0=nl.multiply, operand0=sc[:, 1:2], op1=nl.multiply, operand1=den)
        nisa.scalar_tensor_tensor(dst=p, data=tmp, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.subtract, operand1=p, reverse1=True)
        nisa.dma_copy(dst=out[rows, :], src=p)
        nisa.dma_copy(dst=out_m[rows, :], src=m)
        nisa.dma_copy(dst=out_v[rows, :], src=v)
        nisa.dma_copy(dst=p_in[rows, :], src=p)     # in-place too (whether the torch wrapper honors it is tested on device)
        nisa.dma_copy(dst=m_in[rows, :], src=m)
        nisa.dma_copy(dst=v_in[rows, :], src=v)
    return out, out_m, out_v


def reference(p, g, m, v, lr, bias1, bias2, wd, beta1=0.8, beta2=0.995, eps=1e-10):
    p = p * (1 - lr * wd)
    m = m + (1 - beta1) * (g - m)
    v = v + (1 - beta2) * (g * g - v)
    return p - lr * (m / bias1) / (np.sqrt(v / bias2) + eps), m, v


if __name__ == "__main__":
    import sys, time, torch
    rng = np.random.default_rng(0)
    R, C = (int(a) for a in (sys.argv[1:3] or (256, 512)))
    p = rng.standard_normal((R, C)).astype(np.float32); g = (rng.standard_normal((R, C)) * 0.01).astype(np.float32)
    m = (rng.standard_normal((R, C)) * 0.01).astype(np.float32); v = (rng.random((R, C)) * 1e-4).astype(np.float32)
    lr, wd, step = 0.02, 0.001, 7
    bias1, bias2 = 1 - 0.8 ** step, 1 - 0.995 ** step
    scal = np.array([lr, 1 / bias1, 1 / np.sqrt(bias2), lr * wd], dtype=np.float32)
    rp, rm, rv = reference(p, g, m, v, lr, bias1, bias2, wd)
    tp, tg, tm, tv = (torch.tensor(a.copy()) for a in (p, g, m, v))
    t0 = time.time()
    out, om, ov = nki.simulate(adamw_step)(tp, tg, tm, tv, torch.tensor(scal).expand(P, 4).contiguous())
    e = lambda a, b: np.abs(np.asarray(a) - b).max()
    print(f"simulate {time.time()-t0:.1f}s | p err {e(out, rp):.2e} m {e(om, rm):.2e} v {e(ov, rv):.2e} | in-place p {e(tp.numpy(), rp):.2e} m {e(tm.numpy(), rm):.2e} v {e(tv.numpy(), rv):.2e} | |ref dp| {np.abs(rp-p).max():.2e}")
