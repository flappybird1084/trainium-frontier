"""Device test: the chunked gated delta rule's intermediates on Neuron (eager torch ops) vs CPU fp32, at the model's shapes
(B = 4 rows x 4 heads, T = 1024, d = 128), for chunk sizes 64 and 16. usage: python gdn_dev_test.py"""
import torch, sys
sys.path.insert(0, ".")
from gdn_torch import gdn_chunked, _unit_lower_inverse
dev = torch.device("neuron")
torch.manual_seed(0)
B, T, d = 16, 1024, 128
q = torch.nn.functional.normalize(torch.randn(B, T, d), dim=-1); k = torch.nn.functional.normalize(torch.randn(B, T, d), dim=-1)
v = torch.randn(B, T, d)
g = -torch.nn.functional.softplus(torch.randn(B, T) * 2.0 + 1.0)      # log-decay, typical init magnitudes
beta = torch.sigmoid(torch.randn(B, T))
for C in (64, 16):
    ref = gdn_chunked(q, k, v, g, beta, C)
    out = gdn_chunked(q.to(dev), k.to(dev), v.to(dev), g.to(dev), beta.to(dev), C).cpu()
    print(f"C={C}: device vs cpu max |diff| {(out - ref).abs().max().item():.3e}, nonfinite on device: {(~torch.isfinite(out)).sum().item()}, ref max {ref.abs().max().item():.3e}", flush=True)
    # stages
    n = T // C
    kc = k.view(B, n, C, d); gc = torch.cumsum(g.view(B, n, C), -1); bc = beta.view(B, n, C)
    idx = torch.arange(C); strict = idx.view(C, 1) > idx.view(1, C); causal = idx.view(C, 1) >= idx.view(1, C)
    dexp = gc.unsqueeze(-1) - gc.unsqueeze(-2); decay = torch.exp(torch.where(causal, dexp, torch.zeros_like(dexp)))
    A = torch.where(strict, ((kc * bc.unsqueeze(-1)) @ kc.transpose(-1, -2)) * decay, torch.zeros_like(decay))
    Tm_ref = _unit_lower_inverse(A)
    Tm_dev = _unit_lower_inverse(A.to(dev)).cpu()
    print(f"C={C}: T=(I+A)^-1 device vs cpu max |diff| {(Tm_dev - Tm_ref).abs().max().item():.3e}, |T| max {Tm_ref.abs().max().item():.3e}, nonfinite {(~torch.isfinite(Tm_dev)).sum().item()}", flush=True)
    A_dev = torch.where(strict.to(dev), ((kc.to(dev) * bc.to(dev).unsqueeze(-1)) @ kc.to(dev).transpose(-1, -2)) * decay.to(dev), torch.zeros(1, device=dev))
    print(f"C={C}: A device vs cpu max |diff| {(A_dev.cpu() - A).abs().max().item():.3e}", flush=True)
print("done")
