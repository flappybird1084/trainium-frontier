"""Device test (one core): fused pre-attention backward kernel (nki_pre_bwd.py) vs the torch chain reference at the training
shape (8 rows x 1024 tokens, 4 heads x 128, D=512), plus timing. usage: python nki_pre_bwd_test.py"""
import time, torch
from torch_neuronx.nki_hop import wrap_nki
from nki_pre_bwd import pre_attn_bwd, ref_pre_bwd, make_inputs, kernel_args
b, t, NH, HD = 8, 1024, 4, 128
scale, eps, hq = 0.1, 1e-6, HD // 4
dev = torch.device("neuron")
ins = make_inputs(b, t, NH, HD)
dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart, wq, wk, wv = ins
ref = ref_pre_bwd(dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart, wq, wk, wv, scale, eps, hq)
args = [a.to(dev) for a in kernel_args(dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart)]
wqd, wkd, wvd = (w.to(dev) for w in (wq, wk, wv))
k = wrap_nki(pre_attn_bwd)
outs = k[(1,)](*args, wqd, wkd, wvd, n_head=NH, eps=eps, scale=scale, hq=hq); torch.neuron.synchronize()
for name, o, r in zip(("dxm", "gv2", "dWq", "dWk", "dWv"), outs, ref):
    o = o.float().cpu(); r = r.float()
    err = (o - r).abs().max().item(); rel = ((o - r).norm() / r.norm()).item()
    print(f"{name}: max |diff| {err:.3e} (ref max {r.abs().max().item():.3e}) rel-norm {rel:.5f}", flush=True)
t0 = time.monotonic()
for _ in range(20): outs = k[(1,)](*args, wqd, wkd, wvd, n_head=NH, eps=eps, scale=scale, hq=hq)
torch.neuron.synchronize(); ms = (time.monotonic() - t0) / 20 * 1000
print(f"pre_attn_bwd kernel: {ms:.3f} ms per layer-microstep (8192 tokens)  -> {ms * 10:.1f} ms per 10-layer micro-step", flush=True)
print("done")
