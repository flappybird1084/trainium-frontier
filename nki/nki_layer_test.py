"""Device test (one core): fused post-attention layer kernel vs the same math as eager torch ops, at T=4096, D=512, H=2048.
usage: python nki_layer_test.py"""
import time
import numpy as np, torch
from torch_neuronx.nki_hop import wrap_nki
from nki_layer import post_attn_fwd, reference
torch.manual_seed(0)
T, D, H = 4096, 512, 2048
dev = torch.device("neuron")
y = (torch.randn(T, D) * 0.5).bfloat16(); xm = torch.randn(T, D).bfloat16()
wo = (torch.randn(D, D) * 0.03).bfloat16(); w1 = (torch.randn(H, D) * 0.04).bfloat16(); w2 = (torch.randn(D, H) * 0.02).bfloat16()
ro, rx1, rpre = reference(y, wo, xm, w1, w2)
k = wrap_nki(post_attn_fwd)
yd, xmd, wod, w1d, w2d = (t.to(dev) for t in (y, xm, wo, w1, w2))
out, x1, preT = k[(1,)](yd, wod, xmd, w1d, w2d); torch.neuron.synchronize()
print(f"kernel: out max err {(out.float().cpu() - ro).abs().max().item():.4f} x1 {(x1.float().cpu() - rx1).abs().max().item():.4f} preT {(preT.float().cpu() - rpre).abs().max().item():.4f}", flush=True)
t0 = time.monotonic()
for _ in range(20): out, x1, preT = k[(1,)](yd, wod, xmd, w1d, w2d)
torch.neuron.synchronize(); print(f"kernel: {(time.monotonic() - t0) / 20 * 1000:.2f} ms per 4096 tokens", flush=True)
def eager():
    a = torch.nn.functional.linear(yd, wod)
    x1 = xmd + a
    h2 = torch.nn.functional.rms_norm(x1, (D,), eps=1e-6)
    pre = torch.nn.functional.linear(h2, w1d)
    m = torch.nn.functional.linear(torch.relu(pre) ** 2, w2d)
    return x1 + m
o = eager(); torch.neuron.synchronize()
print(f"eager : out max err {(o.float().cpu() - ro).abs().max().item():.4f}", flush=True)
t0 = time.monotonic()
for _ in range(20): o = eager()
torch.neuron.synchronize(); print(f"eager : {(time.monotonic() - t0) / 20 * 1000:.2f} ms per 4096 tokens (6 torch ops)", flush=True)
print("done")

# ---- pre-attention kernel
from nki_layer import pre_attn_fwd, reference_pre
torch.manual_seed(1)
x = torch.randn(T, D).bfloat16(); x0 = torch.randn(T, D).bfloat16(); ve = torch.randn(T, D).bfloat16()
wq, wk, wv = ((torch.randn(D, D) * 0.03).bfloat16() for _ in range(3))
pos = torch.arange(T, dtype=torch.float32) % 1024; inv = 1.0 / (10000 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
ang = torch.outer(pos, inv); cos, sin = ang.cos(), ang.sin()
scal = torch.tensor([1.0, 0.05, 0.5, 0.0]).expand(128, 4).contiguous()
rq, rk, rv, rxm = reference_pre(x, x0, ve, wq, wk, wv, cos, sin, scal)
kp = wrap_nki(pre_attn_fwd)
xd, x0d, ved, wqd, wkd, wvd, cosd, sind, scald = (t.to(dev) for t in (x, x0, ve, wq, wk, wv, cos, sin, scal))
q, k, v, xm, rq, rk = kp[(1,)](xd, x0d, ved, wqd, wkd, wvd, cosd, sind, scald); torch.neuron.synchronize()
print(f"pre-kernel: q err {(q.float().cpu() - rq).abs().max().item():.4f} k {(k.float().cpu() - rk).abs().max().item():.4f} v {(v.float().cpu() - rv).abs().max().item():.4f} xm {(xm.float().cpu() - rxm).abs().max().item():.4f}", flush=True)
t0 = time.monotonic()
for _ in range(20): q, k, v, xm, rq, rk = kp[(1,)](xd, x0d, ved, wqd, wkd, wvd, cosd, sind, scald)
torch.neuron.synchronize(); print(f"pre-kernel: {(time.monotonic() - t0) / 20 * 1000:.2f} ms per 4096 tokens", flush=True)
def eager_pre():
    xm = 1.0 * xd + 0.05 * x0d
    h = torch.nn.functional.rms_norm(xm, (D,), eps=1e-6)
    q = torch.nn.functional.linear(h, wqd).view(T, 4, 128); k = torch.nn.functional.linear(h, wkd).view(T, 4, 128)
    v = (torch.nn.functional.linear(h, wvd) + 0.5 * ved).view(T, 4, 128)
    c, s = cosd.view(T, 1, 64), sind.view(T, 1, 64)
    def rope(p):
        a, b = p[..., :64], p[..., 64:]
        return torch.nn.functional.rms_norm(torch.cat([a * c + b * s, b * c - a * s], dim=-1), (128,), eps=1e-6)
    return rope(q).permute(1, 0, 2).contiguous(), rope(k).permute(1, 0, 2).contiguous(), v.permute(1, 0, 2).contiguous(), xm
o = eager_pre(); torch.neuron.synchronize()
t0 = time.monotonic()
for _ in range(20): o = eager_pre()
torch.neuron.synchronize(); print(f"eager-pre : {(time.monotonic() - t0) / 20 * 1000:.2f} ms per 4096 tokens (~20 torch ops)", flush=True)
print("done2")
