"""Device test (one core): our attention backward vs the vendor attention_bwd at the training shape (8 rows x 4 heads,
T=1024, HD=128, doc bounds, logit sink). usage: python nki_attn_bwd_test.py"""
import time, numpy as np, torch
from torch_neuronx.nki_hop import wrap_nki
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.core.attention.attention_bwd import attention_bwd
from nki_attn_bwd import attn_bwd
torch.manual_seed(0)
b, h, t, d = 8, 4, 1024, 128
B = b * h
dev = torch.device("neuron")
q = (torch.randn(B, t, d) * 0.3).bfloat16(); k = (torch.randn(B, t, d) * 0.3).bfloat16(); v = torch.randn(B, t, d).bfloat16()
sink = torch.randn(h).float()
sink_rows = sink.view(1, h).expand(b, h).reshape(B, 1).contiguous()
start = torch.zeros(b, t, dtype=torch.int32); end = torch.full((b, t), t, dtype=torch.int32)
for r in range(b):                                   # one document boundary per row
    cut = 128 * (r + 1); start[r, cut:] = cut; end[r, :cut] = cut
bmin = start.unsqueeze(1).expand(b, h, t).reshape(B, t, 1).contiguous(); bmax = end.unsqueeze(1).expand(b, h, t).reshape(B, t, 1).contiguous()
qd, kd, vd, sinkd, bmind, bmaxd = (x.to(dev) for x in (q, k, v, sink_rows, bmin, bmax))
afwd, abwd, ours = wrap_nki(attention_cte), wrap_nki(attention_bwd), wrap_nki(attn_bwd)
out, neg_max, recip = afwd[(1,)](qd, kd, vd, tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True, bound_min=bmind, bound_max=bmaxd, sink=sinkd)
torch.neuron.synchronize()
lse = (-(neg_max.float() + torch.log(recip.float()))).reshape(b, h, *neg_max.shape[-2:]).contiguous()
do = torch.randn(B, t, d).bfloat16().to(dev)
tr = lambda x: x.transpose(-2, -1).contiguous()
q4, k4, v4, o4, do4 = (x.view(b, h, t, d) for x in (qd, kd, vd, out, do))
def vendor():
    res = abwd[(1,)](tr(q4), tr(k4), tr(v4), tr(o4), tr(do4), lse, sinks_ref=sink.view(1, h).expand(b, h).contiguous().to(dev),
                     bound_min=start.float().contiguous().to(dev), bound_max=end.float().contiguous().to(dev), use_causal_mask=True, mixed_precision=True, softmax_scale=1.0)
    return (r.transpose(-2, -1).contiguous() for r in res[:3]), res[3].float().sum(dim=0)
(vdq, vdk, vdv), vds = vendor(); torch.neuron.synchronize()
t0 = time.monotonic()
for _ in range(10): vendor()
torch.neuron.synchronize(); tv = (time.monotonic() - t0) / 10 * 1000
lse_rows = lse.reshape(B, t, 1).contiguous(); sk_rows = sink.view(1, h).expand(b, h).reshape(B, 1, 1).expand(B, t, 1).contiguous().to(dev)
odq, odk, odv, odsr = ours[(1,)](qd, kd, vd, out, do, lse_rows, sk_rows, bmind.float().contiguous(), bmaxd.float().contiguous()); torch.neuron.synchronize()
t0 = time.monotonic()
for _ in range(10): ours[(1,)](qd, kd, vd, out, do, lse_rows, sk_rows, bmind.float().contiguous(), bmaxd.float().contiguous())
torch.neuron.synchronize(); to = (time.monotonic() - t0) / 10 * 1000
print(f"vendor attention_bwd: {tv:.2f} ms | ours: {to:.2f} ms  (8 rows x 4 heads, T=1024, HD=128)", flush=True)
for name, a, r in (("dq", odq, vdq.reshape(B, t, d)), ("dk", odk, vdk.reshape(B, t, d)), ("dv", odv, vdv.reshape(B, t, d))):
    err = (a.float().cpu() - r.float().cpu()).abs().max().item(); print(f"{name}: max |ours - vendor| {err:.3e} (vendor max {r.float().abs().max().item():.3e})", flush=True)
ods = odsr.float().cpu().view(b, h, t).sum(dim=(0, 2))
print(f"dsink: ours {ods.tolist()} vendor {vds.cpu().tolist()}", flush=True)
print("done")
