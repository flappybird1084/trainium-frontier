#!/usr/bin/env python3
"""Dev tool (ON the box, single proc, LNC=2, PYTHONPATH=~/tf): fused AdamW NKI kernel vs the vehicle's eager AdamW step on
the device: numerics, whether in-place writes to the inputs are honored through wrap_nki, and timing per tensor shape."""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores
from nki_adamw import adamw_step
dev = torch.device("neuron"); grid = (int(get_logical_neuron_cores()),); kern = wrap_nki(adamw_step)
torch.manual_seed(0)
b1, b2, eps, lr, wd, step = 0.8, 0.995, 1e-10, 0.02, 0.001, 7
bias1, bias2 = 1 - b1 ** step, 1 - b2 ** step
scal = torch.tensor([lr, 1 / bias1, 1 / bias2 ** 0.5, lr * wd], dtype=torch.float32).expand(128, 4).contiguous().to(dev)
lr_t = torch.tensor(lr, device=dev)

def torch_step(p, g, m, v):   # the vehicle's _adamw_step body
    p.mul_(1 - lr_t * wd); m.lerp_(g, 1 - b1); v.lerp_(g.square(), 1 - b2)
    p.sub_(lr_t * (m / bias1 / ((v / bias2).sqrt() + eps)))

def time_it(fn, n=5):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n

for shape in [(8192, 512), (131072, 512)]:
    p0 = torch.randn(*shape); g0 = torch.randn(*shape) * 0.01; m0 = torch.randn(*shape) * 0.01; v0 = torch.rand(*shape) * 1e-4
    rp, rg, rm, rv = (t.clone().to(dev) for t in (p0, g0, m0, v0)); torch_step(rp, rg, rm, rv); torch_neuronx.synchronize()
    kp, kg, km, kv = (t.clone().to(dev) for t in (p0, g0, m0, v0))
    op, om, ov = kern[grid](kp, kg, km, kv, scal); torch_neuronx.synchronize()
    e = lambda a, b: (a.float().cpu() - b.float().cpu()).abs().max().item()
    print(f"{shape}: outputs err p {e(op, rp):.2e} m {e(om, rm):.2e} v {e(ov, rv):.2e} | in-place honored: p {e(kp, rp):.2e} m {e(km, rm):.2e} v {e(kv, rv):.2e} (|dp| {e(rp, p0.to(dev)):.2e})", flush=True)
    t_ref = time_it(lambda: torch_step(rp, rg, rm, rv)); t_k = time_it(lambda: kern[grid](kp, kg, km, kv, scal))
    print(f"{shape}: torch eager {t_ref*1000:.2f} ms | NKI {t_k*1000:.2f} ms", flush=True)
