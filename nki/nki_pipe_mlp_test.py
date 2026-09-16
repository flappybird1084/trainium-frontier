#!/usr/bin/env python3
"""Simulator correctness for the four nki_pipe_mlp variants vs the torch reference (max abs err, rel-norm err).
usage: python nki_pipe_mlp_test.py [T] [H]   (defaults T=2048, D=512, H=3072: 4 chunks so the 3-deep ring wraps)"""
import sys, time
import torch
import nki
from nki_pipe_mlp import VARIANTS, reference, make_inputs

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
H = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
D = 512
x, w1t, w2t = make_inputs(T, D, H)
ref = reference(x, w1t, w2t).float()
print(f"T={T} D={D} H={H} |ref| max {ref.abs().max():.3f}", flush=True)
for name, kern in VARIANTS:
    t0 = time.time()
    try:
        y = nki.simulate(kern)(x, w1t, w2t)
        d = y.float() - ref
        print(f"{name:8s} sim {time.time() - t0:6.1f}s  max abs err {d.abs().max():.3e}  rel-norm {(d.norm() / ref.norm()):.2e}  nan {torch.isnan(y.float()).sum().item()}", flush=True)
    except Exception as exc:
        print(f"{name:8s} FAILED after {time.time() - t0:.1f}s: {str(exc)[:400]}", flush=True)
