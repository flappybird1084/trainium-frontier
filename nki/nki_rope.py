#!/usr/bin/env python3
"""NKI kernel: fused split-half RoPE for x (rows, h*d) bf16 with rows = b*t (t a multiple of 128), cos/sin (t, d/2) fp32.
   out[:, j:j+d/2] = x1*cos + x2*sin ; out[:, j+d/2:j+d] = -x1*sin + x2*cos   per head slice j = head*d
The backward is the same rotation with sin negated (`sign=-1.0`): dx1 = dy1*cos - dy2*sin ; dx2 = dy1*sin + dy2*cos.
Eager-line kernel (the vehicle's apply_rotary_emb is ~9 launches per call; this is 1). Harness: python nki_rope.py."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def rope_kernel(x_in: nl.ndarray, cos_in: nl.ndarray, sin_in: nl.ndarray, n_head: int, sign: float = 1.0) -> nl.ndarray:
    rows, hd = x_in.shape
    t, d2 = cos_in.shape
    d = hd // n_head
    out = nl.ndarray((rows, hd), dtype=x_in.dtype, buffer=nl.shared_hbm)
    tiles = rows // P
    tpt = t // P                         # tiles per sequence
    nprog = nl.num_programs(0)
    per = tiles // nprog
    for ii in nl.affine_range(per):
        i = nl.program_id(0) * per + ii
        ti = i % tpt                     # position tile within the sequence (rows are b-major, t-minor)
        x = nl.ndarray((P, hd), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x, src=x_in[i * P:(i + 1) * P, :])
        cs = nl.ndarray((P, d2), dtype=nl.float32, buffer=nl.sbuf)
        sn = nl.ndarray((P, d2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs, src=cos_in[ti * P:(ti + 1) * P, :])
        nisa.dma_copy(dst=sn, src=sin_in[ti * P:(ti + 1) * P, :])
        if sign < 0:
            nisa.tensor_scalar(dst=sn, data=sn, op0=nl.multiply, operand0=-1.0)
        o = nl.ndarray((P, hd), dtype=nl.float32, buffer=nl.sbuf)
        t1 = nl.ndarray((P, d2), dtype=nl.float32, buffer=nl.sbuf)
        t2 = nl.ndarray((P, d2), dtype=nl.float32, buffer=nl.sbuf)
        for h in range(n_head):
            a = h * d
            x1 = x[:, a:a + d2]
            x2 = x[:, a + d2:a + d]
            nisa.tensor_tensor(dst=t1, data1=x1, data2=cs, op=nl.multiply)          # x1*cos
            nisa.tensor_tensor(dst=t2, data1=x2, data2=sn, op=nl.multiply)          # x2*sin
            nisa.tensor_tensor(dst=o[:, a:a + d2], data1=t1, data2=t2, op=nl.add)    # y1
            nisa.tensor_tensor(dst=t1, data1=x2, data2=cs, op=nl.multiply)          # x2*cos
            nisa.tensor_tensor(dst=t2, data1=x1, data2=sn, op=nl.multiply)          # x1*sin
            nisa.tensor_tensor(dst=o[:, a + d2:a + d], data1=t1, data2=t2, op=nl.subtract)   # y2 = x2*cos - x1*sin
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=o)
    return out


def reference(x, cos, sin, n_head, sign=1.0):
    import torch
    rows, hd = x.shape; d = hd // n_head; d2 = d // 2
    xx = x.float().view(rows, n_head, d)
    c = cos.view(-1, 1, d2).expand(rows // cos.shape[0], -1, -1, -1).reshape(rows, 1, d2) if False else None
    b = rows // cos.shape[0]
    c = cos.unsqueeze(0).expand(b, -1, -1).reshape(rows, 1, d2); s = sin.unsqueeze(0).expand(b, -1, -1).reshape(rows, 1, d2) * sign
    x1, x2 = xx[..., :d2], xx[..., d2:]
    return torch.cat((x1 * c + x2 * s, x1 * (-s) + x2 * c), -1).reshape(rows, hd)


if __name__ == "__main__":
    import torch, time
    torch.manual_seed(0)
    b, t, h, d = 2, 256, 4, 128
    x = (torch.randn(b * t, h * d) * 0.5).bfloat16()
    freqs = torch.arange(0, d, 2).float() / d; inv = 1.0 / (200000 ** freqs)
    tab = torch.outer(torch.arange(t).float(), inv); cos, sin = tab.cos(), tab.sin()
    for sign in (1.0, -1.0):
        t0 = time.time(); out = nki.simulate(rope_kernel)(x, cos, sin, n_head=h, sign=sign)
        ref = reference(x, cos, sin, h, sign)
        print(f"sign {sign:+.0f}: sim {time.time()-t0:.1f}s | max abs err {(out.float()-ref).abs().max().item():.2e} (|ref| {ref.abs().max():.2f})")
