#!/usr/bin/env python3
"""Dev tool (ON the box, single proc, LNC=2, PYTHONPATH=~/tf): vehicle MLP=nki (fused ReLU^2 MLP fwd/bwd, natural weight layout) vs the
torch MLP on the device: numerics of y / dx / stack grads, and eager fwd+bwd timing at (8, 1024, 512)."""
import os, sys, time, importlib.util
os.environ["NEURON_COMPETITION_R1_N_EMBD"] = "512"; os.environ["NEURON_COMPETITION_R1_N_HEAD"] = "4"
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py"))
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
dev = torch.device("neuron"); t3.COMPUTE_DTYPE = torch.bfloat16; torch.manual_seed(0)
m = t3.GPT(t3.build_config()); m.init_weights()
with torch.no_grad():
    for key in m.stacks: m.stacks[key].normal_(0, 0.03)
m.to(dev); t3.STACK_GRAD_MODE = "function"
mlp = m.transformer.h[0].mlp
x = (torch.randn(8, 1024, 512) * 0.5).bfloat16().to(dev).requires_grad_(True); dy = (torch.randn(8, 1024, 512) * 0.1).bfloat16().to(dev)
def run(impl):
    t3.MLP_IMPL = impl; x.grad = None
    for key in m.stacks: m.stacks[key].grad = None
    y = mlp(x); y.backward(dy); torch_neuronx.synchronize()
    return y.detach().clone(), x.grad.clone(), mlp.c_fc._stack.grad[mlp.c_fc.idx].clone(), mlp.c_proj._stack.grad[mlp.c_proj.idx].clone()
ry, rdx, rg1, rg2 = run("torch"); ky, kdx, kg1, kg2 = run("nki")
e = lambda a, b: f"{(a.float()-b.float()).abs().max().item():.2e} (|ref| {b.float().abs().max().item():.2e})"
print(f"mlp nki vs torch: y {e(ky, ry)} | dx {e(kdx, rdx)} | dW_fc {e(kg1, rg1)} | dW_proj {e(kg2, rg2)}", flush=True)
def time_it(fn, n=10):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n
for impl in ("torch", "nki"):
    t3.MLP_IMPL = impl
    def f():
        x.grad = None; mlp(x).backward(dy)
    print(f"mlp fwd+bwd eager {impl}: {time_it(f)*1000:.2f} ms", flush=True)
