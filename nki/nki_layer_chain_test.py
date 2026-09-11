"""Device test (one core): a full transformer-layer FORWARD as a chain of 3 kernels (pre_attn_fwd -> vendor doc attention ->
post_attn_fwd) at the training shape (4 rows x 1024 tokens, D=512, 4 heads x 128, MLP 2048), timed over 10 layers, vs the
compiled graph's ~51 ms forward per micro-step. usage: python nki_layer_chain_test.py"""
import importlib.util, os, time
import numpy as np, torch
from torch_neuronx.nki_hop import wrap_nki
from nki_layer import pre_attn_fwd, post_attn_fwd
os.environ.setdefault("NEURON_COMPETITION_R1_SINK", "logit"); os.environ.setdefault("NEURON_COMPETITION_R1_ATTN", "nki")
spec = importlib.util.spec_from_file_location("veh", "submission_raw/train3_submission.py")
veh = importlib.util.module_from_spec(spec); spec.loader.exec_module(veh)
dev = torch.device("neuron")
torch.manual_seed(0)
B, Tt, D, NH, HD, H = 4, 1024, 512, 4, 128, 2048
T = B * Tt
x = torch.randn(T, D).bfloat16().to(dev); x0 = torch.randn(T, D).bfloat16().to(dev); ve = torch.randn(T, D).bfloat16().to(dev)
wq, wk, wv, wo = ((torch.randn(D, D) * 0.03).bfloat16().to(dev) for _ in range(4))
w1 = (torch.randn(H, D) * 0.04).bfloat16().to(dev); w2 = (torch.randn(D, H) * 0.02).bfloat16().to(dev)
pos = (torch.arange(T) % Tt).float(); inv = 1.0 / (10000 ** (torch.arange(0, HD, 2).float() / HD))
ang = torch.outer(pos, inv); cos, sin = ang.cos().to(dev), ang.sin().to(dev)
scal = torch.tensor([1.0, 0.05, 0.5, 0.0]).expand(128, 4).contiguous().to(dev)
sink = torch.zeros(NH).to(dev)
idx = torch.randint(1, 8192, (B, Tt), dtype=torch.int32); idx[:, 0] = veh.BOS_ID; idx = idx.to(dev)
start, end = veh.doc_bounds(idx)
pre_k = wrap_nki(pre_attn_fwd); post_k = wrap_nki(post_attn_fwd)
scale = 1.0 / (HD ** 0.5)

def layer_fwd(x):
    q, k, v, xm, rq, rk = pre_k[(1,)](x, x0, ve, wq, wk, wv, cos, sin, scal)    # (NH, T, HD) each
    q4 = q.view(NH, B, Tt, HD).permute(1, 0, 2, 3); k4 = k.view(NH, B, Tt, HD).permute(1, 0, 2, 3); v4 = v.view(NH, B, Tt, HD).permute(1, 0, 2, 3)
    y = veh.nki_doc_attention(q4.contiguous(), k4.contiguous(), v4.contiguous(), sink, start, end, scale)   # (B, NH, Tt, HD)
    y2 = y.permute(0, 2, 1, 3).reshape(T, D).contiguous()
    out, x1, preT = post_k[(1,)](y2, wo, xm, w1, w2)
    return out

out = layer_fwd(x); torch.neuron.synchronize()
print("layer forward ok", tuple(out.shape), flush=True)
t0 = time.monotonic()
for _ in range(10): out = layer_fwd(out)
torch.neuron.synchronize(); dt = (time.monotonic() - t0) / 10 * 1000
print(f"CHAIN layer forward: {dt:.2f} ms per layer (4x1024 tokens) -> {dt * 10:.1f} ms per 10-layer micro-step (compiled fwd ~51 ms)", flush=True)
# pieces
for name, fn in (("pre", lambda: pre_k[(1,)](x, x0, ve, wq, wk, wv, cos, sin, scal)),
                 ("post", lambda: post_k[(1,)](out, wo, x, w1, w2))):
    fn(); torch.neuron.synchronize(); t0 = time.monotonic()
    for _ in range(20): fn()
    torch.neuron.synchronize(); print(f"  {name}: {(time.monotonic() - t0) / 20 * 1000:.2f} ms", flush=True)
q, k, v, xm, rq, rk = pre_k[(1,)](x, x0, ve, wq, wk, wv, cos, sin, scal)
q4 = q.view(NH, B, Tt, HD).permute(1, 0, 2, 3).contiguous(); k4 = k.view(NH, B, Tt, HD).permute(1, 0, 2, 3).contiguous(); v4 = v.view(NH, B, Tt, HD).permute(1, 0, 2, 3).contiguous()
veh.nki_doc_attention(q4, k4, v4, sink, start, end, scale); torch.neuron.synchronize(); t0 = time.monotonic()
for _ in range(20): veh.nki_doc_attention(q4, k4, v4, sink, start, end, scale)
torch.neuron.synchronize(); print(f"  attention fwd (vendor): {(time.monotonic() - t0) / 20 * 1000:.2f} ms", flush=True)
print("done")
