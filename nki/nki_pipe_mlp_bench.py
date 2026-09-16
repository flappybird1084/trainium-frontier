#!/usr/bin/env python3
"""Device bench (ONE core, single process, idle device only): the four nki_pipe_mlp variants vs the eager per-op torch path
(rms_norm, linear, relu**2, linear) at T=16384, D=512, H=3072; 1 warm-up call (compile) + correctness, then 20 warm calls.
usage (box): cd ~/tf && source ~/workspace/native_venv/bin/activate && PYTHONPATH=~/tf python nki/nki_pipe_mlp_bench.py [T] [reps]"""
import sys, time
import torch
import torch.nn.functional as F
from torch_neuronx.nki_hop import wrap_nki
from nki_pipe_mlp import VARIANTS, reference, make_inputs, EPS

T = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
D, H = 512, 3072
dev = torch.device("neuron")
x, w1t, w2t = make_inputs(T, D, H)
ref = reference(x, w1t, w2t).float()
xd, w1d, w2d = (t.to(dev) for t in (x, w1t, w2t))
flops = 4 * T * D * H
res = {}
w1dt = w1d.t().contiguous(); w2dt = w2d.t().contiguous()
def eager2():
    h2 = F.rms_norm(xd, (D,), eps=EPS)
    return F.linear(torch.relu(F.linear(h2, w1dt)) ** 2, w2dt)
o = eager2(); torch.neuron.synchronize()
print(f"eager    max abs err {(o.float().cpu() - ref).abs().max().item():.3e}", flush=True)
t0 = time.monotonic()
for _ in range(REPS):
    o = eager2()
torch.neuron.synchronize(); ms = (time.monotonic() - t0) / REPS * 1000
res["eager"] = ms
print(f"eager    {ms:.3f} ms per call at T={T} (4 torch ops: rms_norm, linear, relu**2, linear)", flush=True)
for name, kern in VARIANTS:
    k = wrap_nki(kern)
    try:
        t0 = time.monotonic()
        y = k[(1,)](xd, w1d, w2d); torch.neuron.synchronize()
        tc = time.monotonic() - t0
        d = y.float().cpu() - ref
        print(f"{name:8s} first call {tc:.1f}s (compile)  max abs err {d.abs().max().item():.3e}  rel-norm {(d.norm() / ref.norm()).item():.2e}", flush=True)
        t0 = time.monotonic()
        for _ in range(REPS):
            y = k[(1,)](xd, w1d, w2d)
        torch.neuron.synchronize(); ms = (time.monotonic() - t0) / REPS * 1000
        res[name] = ms
        print(f"{name:8s} {ms:.3f} ms per call at T={T} ({flops / ms / 1e9:.1f} TFLOP/s matmul-only)", flush=True)
    except Exception as exc:
        print(f"{name:8s} FAILED: {str(exc)[:800]}", flush=True)
print("summary:", {k: round(v, 3) for k, v in res.items()})
print("done")
