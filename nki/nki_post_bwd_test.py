"""Device test (one core): fused post-attention backward kernel (nki_post_bwd.py) vs the torch chain reference at the training
shape (8 rows x 1024 tokens, D=512, H=3072), plus timing. usage: python nki_post_bwd_test.py"""
import time, torch
from torch_neuronx.nki_hop import wrap_nki
from nki_post_bwd import post_attn_bwd, ref_post_bwd, make_inputs
b, t, NH, HD, H = 8, 1024, 4, 128, 3072
eps = 1e-6
dev = torch.device("neuron")
d2, x1, preT, y2, w1, w2, wo = make_inputs(b, t, NH, HD, H)
ref = ref_post_bwd(d2, x1, preT, y2, w1, w2, wo, NH, eps)
args = [a.to(dev) for a in (d2, x1, preT, y2, w1, w2, wo)]
k = wrap_nki(post_attn_bwd)
outs = k[(1,)](*args, n_head=NH, eps=eps, t_len=t); torch.neuron.synchronize()
dyh = outs[0].float().cpu().view(b, NH, t, HD).permute(0, 2, 1, 3).reshape(b * t, NH * HD)
got = [dyh] + [o.float().cpu() for o in outs[1:]]
for name, o, r in zip(("dy2", "dx1", "dW1", "dW2", "dWo"), got, ref):
    r = r.float(); err = (o - r).abs().max().item(); rel = ((o - r).norm() / r.norm()).item()
    print(f"{name}: max |diff| {err:.3e} (ref max {r.abs().max().item():.3e}) rel-norm {rel:.5f}", flush=True)
t0 = time.monotonic()
for _ in range(20): outs = k[(1,)](*args, n_head=NH, eps=eps, t_len=t)
torch.neuron.synchronize(); ms = (time.monotonic() - t0) / 20 * 1000
print(f"post_attn_bwd kernel: {ms:.3f} ms per layer-microstep (8192 tokens)  -> {ms * 10:.1f} ms per 10-layer micro-step", flush=True)
print("done")
