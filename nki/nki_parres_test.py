#!/usr/bin/env python3
"""Device experiment (ON the box, single core): can the NKI instruction scheduler OVERLAP an MLP with the vendor attention when
both live in ONE kernel? This is the precondition for parallel residuals paying on Trainium (repeat-regime.md idea 3):
separate kernel launches serialize on a NeuronCore, so the only overlap available is intra-kernel. Times three things at
production shapes (64 attention problems of t=1024, d=128; MLP on 16384 tokens, D=512, H=3072):
  (a) attention_cte alone (vendor entrypoint, the vehicle's call), (b) our mlp_fwd alone,
  (c) ONE kernel calling the vendor's undecorated inner `_attention_cte` (it exposes out_*_hbm / init_sbuf_addr for exactly
      this kind of nesting) followed by the MLP body inline.
Verdict: time(c) ~ max(a, b) -> overlap real, build the parallel-residual layer kernel; time(c) ~ a + b -> serial, close it."""
import os, sys, time, traceback
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
import nki, nki.language as nl, nki.isa as nisa
from torch_neuronx.nki_hop import wrap_nki
from nkilib.core.attention.attention_cte import attention_cte, _attention_cte
P, FMAX = 128, 512
dev = torch.device("neuron"); torch.manual_seed(0)
B, T, D_ATT = 64, 1024, 128                        # 16 rows x 4 heads
TT, D, H = 16384, 512, 3072

def mlp_body(x_in, w1, w2, y_out, pre_out):
    """mlp_fwd's body (nki/nki_mlp.py), un-decorated so it can run inside another kernel. Single program."""
    T, Dm = x_in.shape; Hm = w1.shape[0]; kt = Dm // P; ht = Hm // P; chunks = T // FMAX
    W1 = []
    for _j in range(kt):
        W1.append(nl.ndarray((P, Hm), dtype=nl.bfloat16, buffer=nl.sbuf))
    for h in range(ht):
        blk = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w1[h * P:(h + 1) * P, :])
        for k in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, k * P:(k + 1) * P])
            nisa.tensor_copy(dst=W1[k][:, h * P:(h + 1) * P], src=tps)
    W2 = []
    for _j in range(ht):
        W2.append(nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf))
    for o in range(kt):
        blk = nl.ndarray((P, Hm), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w2[o * P:(o + 1) * P, :])
        for h in range(ht):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, h * P:(h + 1) * P])
            nisa.tensor_copy(dst=W2[h][:, o * P:(o + 1) * P], src=tps)
    for c in nl.affine_range(chunks):
        t0 = c * FMAX
        xT = []
        for _j in range(kt):
            xT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for tb in range(FMAX // P):
            xt = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            for k in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=xt[:, k * P:(k + 1) * P])
                nisa.tensor_copy(dst=xT[k][:, tb * P:(tb + 1) * P], src=tps)
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
            nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)
        for tb in range(FMAX // P):
            yps = nl.ndarray((P, Dm), dtype=nl.float32, buffer=nl.psum)
            for h in range(ht):
                nisa.nc_matmul(dst=yps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=W2[h], accumulate=(h > 0))
            yt = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=yt, src=yps)
            nisa.dma_copy(dst=y_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=yt)

@nki.jit
def mlp_only(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
    T, Dm = x_in.shape; Hm = w1.shape[0]
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    mlp_body(x_in, w1, w2, y_out, pre_out)
    return y_out, pre_out

@nki.jit
def attn_mlp_fused(q: nl.ndarray, k: nl.ndarray, v: nl.ndarray, sink: nl.ndarray, bmin: nl.ndarray, bmax: nl.ndarray,
                   x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
    # vendor attention NESTED (undecorated inner function), then the MLP body: one instruction stream for the scheduler
    out, neg_max, recip = _attention_cte(q, k, v, scale=1.0, causal_mask=True, sink=sink, tp_q=True, tp_k=True,
                                         cache_softmax=True, bound_min=bmin, bound_max=bmax)
    T, Dm = x_in.shape; Hm = w1.shape[0]
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    mlp_body(x_in, w1, w2, y_out, pre_out)
    return out, neg_max, recip, y_out, pre_out

def time_it(fn, n=20):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n * 1000

q = (torch.randn(B, T, D_ATT) * 0.3).bfloat16().to(dev); kk = (torch.randn(B, T, D_ATT) * 0.3).bfloat16().to(dev); vv = torch.randn(B, T, D_ATT).bfloat16().to(dev)
sink = torch.zeros(B, 1, device=dev); pos = torch.arange(T, dtype=torch.float32).view(1, T, 1)
bmin = torch.zeros(B, T, 1, device=dev); bmax = (pos + 1).expand(B, T, 1).contiguous().to(dev)
x = (torch.randn(TT, D) * 0.5).bfloat16().to(dev); w1 = (torch.randn(H, D) * 0.03).bfloat16().to(dev); w2 = (torch.randn(D, H) * 0.02).bfloat16().to(dev)
grid = (1,)
att = wrap_nki(attention_cte); mlp = wrap_nki(mlp_only); fused = wrap_nki(attn_mlp_fused)
fa = lambda: att[grid](q, kk, vv, tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True, bound_min=bmin, bound_max=bmax, sink=sink)
fm = lambda: mlp[grid](x, w1, w2)
ta = time_it(fa); print(f"(a) attention_cte alone: {ta:.2f} ms", flush=True)
tm = time_it(fm); print(f"(b) mlp_fwd alone:       {tm:.2f} ms   (serial sum a+b = {ta+tm:.2f} ms, ideal overlap max(a,b) = {max(ta,tm):.2f} ms)", flush=True)
try:
    ff = lambda: fused[grid](q, kk, vv, sink, bmin, bmax, x, w1, w2)
    o1 = fa(); y1, _ = fm(); o2, nm2, rc2, y2, _ = ff(); torch_neuronx.synchronize()
    e = lambda a, b: (a.float() - b.float()).abs().max().item()
    print(f"(c) fused correctness: attn out max|diff| {e(o1[0], o2):.3e}, mlp y max|diff| {e(y1, y2):.3e}", flush=True)
    tf = time_it(ff); print(f"(c) fused attention+MLP: {tf:.2f} ms  -> overlap fraction = {(ta+tm-tf)/min(ta,tm)*100:.0f}% of the smaller ({'OVERLAP REAL' if tf < 0.9*(ta+tm) else 'SERIAL: no intra-kernel overlap'})", flush=True)
except Exception:
    print("(c) fused kernel FAILED (nesting not supported as written):", flush=True); traceback.print_exc()
