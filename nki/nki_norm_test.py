#!/usr/bin/env python3
"""Dev tool (ON the box, single proc, LNC=2, PYTHONPATH=~/tf): vehicle NORM=nki (fused RMSNorm fwd/bwd) vs F.rms_norm: numerics + eager timing at (8*1024, 512)."""
import os, sys, time, importlib.util
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch.nn.functional as F, torch_neuronx
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py"))
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
dev = torch.device("neuron"); torch.manual_seed(0)
x = (torch.randn(8, 1024, 512) * 1.5).bfloat16().to(dev).requires_grad_(True); dy = torch.randn(8, 1024, 512).bfloat16().to(dev)
ref = F.rms_norm(x, (512,)); ref.backward(dy); gref = x.grad.clone(); x.grad = None; torch_neuronx.synchronize()
t3.NORM_IMPL = "nki"; out = t3.norm(x); out.backward(dy); gk = x.grad.clone(); torch_neuronx.synchronize()
e = lambda a, b: (a.float() - b.float()).abs().max().item()
print(f"norm nki vs torch: out {e(out, ref):.2e} (|y| {ref.float().abs().max().item():.2f}) grad {e(gk, gref):.2e} (|dx| {gref.float().abs().max().item():.2e})", flush=True)
def time_it(fn, n=10):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n
def f_torch():
    x.grad = None; F.rms_norm(x, (512,)).backward(dy)
def f_nki():
    x.grad = None; t3.norm(x).backward(dy)
print(f"rmsnorm fwd+bwd eager: torch {time_it(f_torch)*1000:.2f} ms | nki {time_it(f_nki)*1000:.2f} ms", flush=True)
