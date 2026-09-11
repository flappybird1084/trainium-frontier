"""Device test (one core): can torch.compile(backend="neuron") trace a function that computes loss AND gradients in ONE graph
(torch.func.grad / functional_call), so a training micro-step becomes one NEFF launch instead of fwd + bwd? Reports the
number of compiled graphs, the step time of the one-graph form vs the usual fwd + .backward() form, and grad agreement."""
import time, torch, torch.nn as nn, torch.nn.functional as F
import torch_neuronx
from torch.func import functional_call, grad
dev = torch.device("neuron")
torch.manual_seed(0)
class Tiny(nn.Module):
    def __init__(self, d=512, h=2048, v=8192, L=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, d)) for _ in range(L)])
        self.head = nn.Linear(d, v, bias=False)
    def forward(self, x):
        for l in self.layers:
            x = x + l(x)
        return self.head(x)
m = Tiny().to(dev).bfloat16()
x = torch.randn(4096, 512, device=dev, dtype=torch.bfloat16); y = torch.randint(0, 8192, (4096,), device=dev)
params = {k: v.detach() for k, v in m.named_parameters()}
def loss_fn(p, x, y):
    return F.cross_entropy(functional_call(m, p, (x,)).float(), y)
def step_onegraph(p, x, y):
    return grad(loss_fn)(p, x, y)
n_graphs = [0]
import torch._dynamo as dynamo
def counting_backend(gm, ex):
    n_graphs[0] += 1
    from torch_neuronx.dynamo.backend import neuron_backend  # may not exist under this name
    return neuron_backend(gm, ex)
try:
    cg = torch.compile(step_onegraph, backend="neuron", dynamic=False)
    g = cg(params, x, y); torch_neuronx.synchronize()
    print("one-graph compile OK:", len(g), "grads", flush=True)
    t0 = time.monotonic()
    for _ in range(10): g = cg(params, x, y)
    torch_neuronx.synchronize(); print(f"one-graph step: {(time.monotonic() - t0) / 10 * 1000:.1f} ms", flush=True)
except Exception as exc:
    print(f"one-graph FAILED: {type(exc).__name__}: {str(exc)[:300]}", flush=True); g = None
# reference: usual compiled forward + autograd backward (two graphs)
cm = torch.compile(m, backend="neuron", dynamic=False)
for _ in range(2):
    m.zero_grad(set_to_none=True); F.cross_entropy(cm(x).float(), y).backward()
torch_neuronx.synchronize()
t0 = time.monotonic()
for _ in range(10):
    m.zero_grad(set_to_none=True); F.cross_entropy(cm(x).float(), y).backward()
torch_neuronx.synchronize(); print(f"fwd+bwd (2 graphs) step: {(time.monotonic() - t0) / 10 * 1000:.1f} ms", flush=True)
if g is not None:
    worst = max((g[k].float() - p.grad.float()).abs().max().item() / (p.grad.float().abs().max().item() + 1e-9) for k, p in m.named_parameters())
    print(f"grad agreement: worst rel diff {worst:.4f}", flush=True)
print("done")
