"""Device test: the GatedDeltaMixer's forward pieces under torch.compile(backend="neuron") on bf16 inputs at the model's
4-row shape, checking each stage's output for non-finite values (the model NaNs at step 0 while the eager chunked rule is
exact on the device). usage: python gdn_compile_test.py"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from gdn_torch import gdn_chunked
dev = torch.device("neuron")
torch.manual_seed(0)
b, t, D, H = 4, 1024, 512, 4
d = D // H
x = (torch.randn(b, t, D) * 1.0).bfloat16().to(dev)
Wq, Wk, Wv, Wo = [(torch.randn(D, D) * D ** -0.5).bfloat16().to(dev) for _ in range(4)]
a_proj = (torch.randn(D, H) * 0.02).to(dev); b_proj = (torch.randn(D, H) * 0.02).to(dev)
A_log = torch.log(torch.empty(H).uniform_(1.0, 16.0)).to(dev); dt_bias = torch.zeros(H, device=dev)
conv = torch.zeros(D, 4); conv[:, 0] = 1.0; conv = conv.to(dev)


def causal_conv(x, w):
    y = x * w[:, 0]
    for j in range(1, 4):
        y = y + F.pad(x, (0, 0, j, 0))[:, :-j] * w[:, j]
    return y


def stage1(x):   # projections + conv + silu
    q = F.silu(causal_conv(x @ Wq.t(), conv.to(x.dtype))); k = F.silu(causal_conv(x @ Wk.t(), conv.to(x.dtype))); v = F.silu(causal_conv(x @ Wv.t(), conv.to(x.dtype)))
    return q, k, v


def stage2(x):   # gate and beta
    xf = x.float()
    g = -F.softplus(A_log).view(1, 1, H) * F.softplus(xf @ a_proj + dt_bias)
    beta = torch.sigmoid(xf @ b_proj)
    return g, beta


def stage3(q, k, v, g, beta):   # normalize + chunked rule
    q = F.normalize(q.float().view(b, t, H, d), dim=-1).permute(0, 2, 1, 3).reshape(b * H, t, d)
    k = F.normalize(k.float().view(b, t, H, d), dim=-1).permute(0, 2, 1, 3).reshape(b * H, t, d)
    v = v.float().view(b, t, H, d).permute(0, 2, 1, 3).reshape(b * H, t, d)
    return gdn_chunked(q, k, v, g.permute(0, 2, 1).reshape(b * H, t), beta.permute(0, 2, 1).reshape(b * H, t), 64)


def stage4(o, x):   # rms norm + out proj
    o = F.rms_norm(o, (d,), eps=1e-6).view(b, H, t, d).permute(0, 2, 1, 3).reshape(b, t, D).to(x.dtype)
    return o @ Wo.t()


def full(x):
    q, k, v = stage1(x); g, beta = stage2(x); o = stage3(q, k, v, g, beta); return stage4(o, x)


def chk(name, *ts):
    for i, tt in enumerate(ts):
        tt = tt.float().cpu()
        print(f"{name}[{i}]: nonfinite {(~torch.isfinite(tt)).sum().item()} max {tt[torch.isfinite(tt)].abs().max().item() if torch.isfinite(tt).any() else float('nan'):.3e}", flush=True)


# eager on device, stage by stage
q, k, v = stage1(x); torch.neuron.synchronize(); chk("eager stage1", q, k, v)
g, beta = stage2(x); torch.neuron.synchronize(); chk("eager stage2", g, beta)
o = stage3(q, k, v, g, beta); torch.neuron.synchronize(); chk("eager stage3", o)
y = stage4(o, x); torch.neuron.synchronize(); chk("eager stage4", y)
# compiled pieces
c1 = torch.compile(stage1, backend="neuron", dynamic=False); q, k, v = c1(x); torch.neuron.synchronize(); chk("compiled stage1", q, k, v)
c2 = torch.compile(stage2, backend="neuron", dynamic=False); g, beta = c2(x); torch.neuron.synchronize(); chk("compiled stage2", g, beta)
c3 = torch.compile(stage3, backend="neuron", dynamic=False); o = c3(q, k, v, g, beta); torch.neuron.synchronize(); chk("compiled stage3", o)
c4 = torch.compile(stage4, backend="neuron", dynamic=False); y = c4(o, x); torch.neuron.synchronize(); chk("compiled stage4", y)
cf = torch.compile(full, backend="neuron", dynamic=False); y = cf(x); torch.neuron.synchronize(); chk("compiled full", y)
print("done", flush=True)
