#!/usr/bin/env python3
"""Dev tool (ON the box, single proc, LNC=2, PYTHONPATH=~/tf): does torch.compile(backend="neuron") lower flex_attention with a
static causal mask_mod into an inlined NKI kernel (fx/passes/flex_attention_nki_lowering.py)? Times compiled fwd+bwd for
flex vs the explicit fp32 attention at one worker's shape and checks numerics. If flex is much faster than explicit here
(the custom-op HOP route was 3.5x SLOWER at 512-wide), the static-mask flex route is a viable compiled attention path."""
import os, sys, time, torch, torch_neuronx, torch._dynamo
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
dev = torch.device("neuron"); torch.manual_seed(0)
B, H, T, D = 8, 4, 1024, 128
q = (torch.randn(B, H, T, D) * 0.5).bfloat16().to(dev).requires_grad_(True)
k = (torch.randn(B, H, T, D) * 0.5).bfloat16().to(dev).requires_grad_(True)
v = (torch.randn(B, H, T, D) * 0.5).bfloat16().to(dev).requires_grad_(True)
dy = torch.randn(B, H, T, D).bfloat16().to(dev)
def causal(b, h, qi, ki): return qi >= ki
bm = create_block_mask(causal, B, None, T, T, device=dev, BLOCK_SIZE=128)
def explicit(q, k, v):
    s = (q.float() @ k.float().transpose(-2, -1)) * (D ** -0.5)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    s = s.masked_fill(~mask, float("-inf"))
    return (torch.softmax(s, -1) @ v.float()).to(v.dtype)
def flex(q, k, v): return flex_attention(q, k, v, block_mask=bm)
def time_it(fn, n=5):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n
ref = explicit(q, k, v); ref.backward(dy); gq, gk, gv = (t.grad.clone() for t in (q, k, v)); torch_neuronx.synchronize()
for name, fn in (("flex", flex), ("explicit", explicit)):
    torch._dynamo.reset(); cfn = torch.compile(fn, backend="neuron", dynamic=False)
    def step():
        for t in (q, k, v): t.grad = None
        out = cfn(q, k, v); out.backward(dy); torch_neuronx.synchronize(); return out
    t0 = time.time(); out = step(); print(f"[{name}] compile+first {time.time()-t0:.0f}s", flush=True)
    e = lambda a, b: (a.float() - b.float()).abs().max().item()
    print(f"[{name}] vs eager explicit: out {e(out, ref):.2e} dq {e(q.grad, gq):.2e} dk {e(k.grad, gk):.2e} dv {e(v.grad, gv):.2e}", flush=True)
    print(f"[{name}] compiled fwd+bwd {time_it(step)*1000:.1f} ms", flush=True)
