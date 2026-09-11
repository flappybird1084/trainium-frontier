#!/usr/bin/env python3
"""Dev tool (runs ON the box, single process, NEURON_LOGICAL_NC_CONFIG=2, PYTHONPATH=~/tf): the Polar Express NKI kernel
(nki_muon.polar_express_tall via wrap_nki) vs the vehicle's eager torch Newton-Schulz loop on the device, for the three
stack shapes at 512-wide (40x512x512, 10x2048x512, 10x512x2048 handled as its transpose): numerics + timing."""
import os, sys, time, math
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores
from nki_muon import polar_express_tall, COEFFS
dev = torch.device("neuron"); grid = (int(get_logical_neuron_cores()),)
kern = wrap_nki(polar_express_tall)
torch.manual_seed(0)

def torch_ns(x, ns_steps=5):   # the vehicle's _muon_step orthogonalization (bf16), both branches
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if x.shape[-2] > x.shape[-1]:
        for a, b, c in COEFFS[:ns_steps]:
            xtx = x.mT @ x; x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in COEFFS[:ns_steps]:
            xxt = x @ x.mT; x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    return x

def nki_ns(x):
    if x.shape[-2] >= x.shape[-1]:
        return kern[grid](x.contiguous())
    return kern[grid](x.mT.contiguous()).mT

def time_it(fn, n=5):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n

for shape in [(40, 512, 512), (10, 2048, 512), (10, 512, 2048)]:
    g = (torch.randn(*shape) * 0.02).to(torch.bfloat16).to(dev)
    ref = torch_ns(g.clone()); torch_neuronx.synchronize()
    got = nki_ns(g.clone()); torch_neuronx.synchronize()
    err = (got.float() - ref.float()).abs().max().item()
    k = min(shape[1:]); y = got.float() if shape[1] >= shape[2] else got.float().mT
    orth = (y.mT @ y - torch.eye(k, device=dev)).abs().max().item()
    t_ref = time_it(lambda: torch_ns(g)); t_nki = time_it(lambda: nki_ns(g))
    print(f"{shape}: max abs err {err:.3e} (|ref| {ref.float().abs().max().item():.3f}) |X^T X - I| {orth:.3e} | "
          f"torch eager {t_ref*1000:.1f} ms | NKI {t_nki*1000:.1f} ms", flush=True)

# ---- fully fused step (muon_step_stack) vs the vehicle's _muon_step on the device, in-place check + timing
from nki_muon import muon_step_stack, reference_muon_step
fused = wrap_nki(muon_step_stack)
for shape in [(40, 512, 512), (10, 2048, 512), (10, 512, 2048)]:
    n, o, i_ = shape
    w = torch.randn(shape) * 0.05; g = torch.randn(shape) * 0.02; mom = torch.randn(shape) * 0.01
    sec = torch.rand((n, o, 1) if o >= i_ else (n, 1, i_)) * 1e-3
    momentum, lr, wd = 0.95, 0.02, 0.1
    lr_eff = lr * (max(1.0, o / i_) ** 0.5)
    scal = torch.tensor([momentum, lr_eff, lr_eff * wd, 0.0]).expand(128, 4).contiguous().to(dev)
    rw, rm, rs = reference_muon_step(w.to(dev), g.to(dev), mom.to(dev), sec.to(dev), momentum, lr, wd); torch_neuronx.synchronize()
    kw, kg, km, ks = (t.clone().to(dev) for t in (w, g, mom, sec))
    ow, om, os_ = fused[grid](kw, kg, km, ks, scal); torch_neuronx.synchronize()
    e = lambda a, b: (a.float().cpu() - b.float().cpu()).abs().max().item()
    print(f"FUSED {shape}: w err {e(ow, rw):.2e} (|dw| {e(rw, w.to(dev)):.2e}) mom {e(om, rm):.2e} sec {e(os_, rs):.2e} | in-place w {e(kw, rw):.2e} mom {e(km, rm):.2e} sec {e(ks, rs):.2e}", flush=True)
    t_ref = time_it(lambda: reference_muon_step(kw, kg, km, ks, momentum, lr, wd)); t_k = time_it(lambda: fused[grid](kw, kg, km, ks, scal))
    print(f"FUSED {shape}: torch eager full step {t_ref*1000:.1f} ms | NKI fused {t_k*1000:.1f} ms", flush=True)
