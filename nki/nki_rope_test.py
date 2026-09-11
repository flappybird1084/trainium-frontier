#!/usr/bin/env python3
"""Dev tool (ON the box, single proc, LNC=2, PYTHONPATH=~/tf): the vehicle's ROPE=nki path vs apply_rotary_emb (numerics incl.
backward) and eager timing at one worker's q shape (8 rows x 1024 x 4 heads x 128)."""
import os, sys, time, importlib.util
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py"))
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
dev = torch.device("neuron"); torch.manual_seed(0)
b, t, h, d = 8, 1024, 4, 128
x = (torch.randn(b, t, h, d) * 0.5).bfloat16().to(dev).requires_grad_(True)
m = t3.GPT(t3.GPTConfig(sequence_len=1024, vocab_size=8192, n_layer=1, n_head=h, n_embd=h * d)); cos, sin = m.cos.to(dev), m.sin.to(dev)
dy = torch.randn(b, t, h, d).bfloat16().to(dev)
ref = t3.apply_rotary_emb(x, cos, sin); ref.backward(dy); gref = x.grad.clone(); x.grad = None; torch_neuronx.synchronize()
cos2d = cos.reshape(t, -1).float().contiguous(); sin2d = sin.reshape(t, -1).float().contiguous()
out = t3._NKIRope.apply(x.reshape(b * t, -1), cos2d, sin2d, h).view(b, t, h, d); out.backward(dy); gk = x.grad.clone(); torch_neuronx.synchronize()
e = lambda a, c: (a.float() - c.float()).abs().max().item()
print(f"rope nki vs torch: out {e(out, ref):.2e} (|ref| {ref.float().abs().max().item():.2f}) grad {e(gk, gref):.2e}", flush=True)
def time_it(fn, n=10):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n
def f_torch():
    x.grad = None; t3.apply_rotary_emb(x, cos, sin).backward(dy)
def f_nki():
    x.grad = None; t3._NKIRope.apply(x.reshape(b * t, -1), cos2d, sin2d, h).view(b, t, h, d).backward(dy)
print(f"rope fwd+bwd eager: torch {time_it(f_torch)*1000:.2f} ms | nki {time_it(f_nki)*1000:.2f} ms", flush=True)
