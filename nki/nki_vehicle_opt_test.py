#!/usr/bin/env python3
"""Dev tool (ON the box, single proc, LNC=2, PYTHONPATH=~/tf): do the vehicle's in-file optimizer kernels (build_nki_muon_fused,
build_nki_adamw, build_nki_muon) trace/compile/run on the device, and match the dev-file kernels?"""
import os, sys, time, importlib.util
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py"))
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
from nki_muon import reference_muon_step
dev = torch.device("neuron"); torch.manual_seed(0)
fused, grid = t3.build_nki_muon_fused(); adam, _ = t3.build_nki_adamw(); orth = t3.build_nki_muon()
e = lambda a, b: (a.float().cpu() - b.float().cpu()).abs().max().item()
for shape in [(40, 512, 512), (10, 512, 2048)]:
    n, o, i_ = shape
    w = torch.randn(shape) * 0.05; g = torch.randn(shape) * 0.02; mom = torch.randn(shape) * 0.01
    sec = torch.rand((n, o, 1) if o >= i_ else (n, 1, i_)) * 1e-3
    momentum, lr, wd = 0.95, 0.02, 0.1; lr_eff = lr * (max(1.0, o / i_) ** 0.5)
    scal = torch.tensor([momentum, lr_eff, lr_eff * wd, 0.0]).expand(128, 4).contiguous().to(dev)
    rw, rm, rs = reference_muon_step(w.to(dev), g.to(dev), mom.to(dev), sec.to(dev), momentum, lr, wd)
    kw, kg, km, ks = (t.clone().to(dev) for t in (w, g, mom, sec))
    fused[grid](kw, kg, km, ks, scal, ns_steps=5, beta2=0.9); torch_neuronx.synchronize()
    print(f"vehicle fused muon {shape}: in-place w err {e(kw, rw):.2e} mom {e(km, rm):.2e} sec {e(ks, rs):.2e}", flush=True)
    x = (g.to(dev).bfloat16()); y = orth(x, 5); torch_neuronx.synchronize()
    print(f"vehicle NS-only {shape}: |X^T X - I| {((y.float().mT @ y.float() if o >= i_ else y.float() @ y.float().mT) - torch.eye(min(o, i_), device=dev)).abs().max().item():.2e}", flush=True)
p = torch.randn(8192, 512).to(dev); gg = (torch.randn(8192, 512) * 0.01).to(dev); m = torch.zeros(8192, 512).to(dev); v = torch.zeros(8192, 512).to(dev)
scal = torch.tensor([0.02, 1 / (1 - 0.8), 1 / (1 - 0.995) ** 0.5, 0.02 * 0.001]).expand(128, 4).contiguous().to(dev)
p0 = p.clone(); adam[grid](p, gg, m, v, scal, beta1=0.8, beta2=0.995, eps=1e-10); torch_neuronx.synchronize()
print(f"vehicle adamw: |dp| {e(p, p0):.2e} (expected ~0.02*sign-ish, nonzero) m nonzero {bool((m != 0).any())}", flush=True)
