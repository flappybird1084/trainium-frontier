"""Device test: own attention forward v4 / v1 and backward v3 (eager) on MODEL-LIKE inputs: q = rmsnorm(.) * 0.1, k = rmsnorm(.)
with the key offset (norm up to sqrt(256)), sink = 0, B = 16 rows x 4 heads = 64, real-looking BOS patterns (BOS at 0, dense
BOS rows, rows without BOS), checking for NaN/inf and agreement with the torch reference. usage: python nki_attn_real_test.py"""
import torch
from torch_neuronx.nki_hop import wrap_nki
from nki_attn_fwd import attn_fwd, attn_fwd4, fwd2_args
from nki_attn_bwd import attn_bwd, reference, bounds_lo_hi
dev = torch.device("neuron")
k1, k4, kb = wrap_nki(attn_fwd), wrap_nki(attn_fwd4), wrap_nki(attn_bwd)
torch.manual_seed(1)
b, h, t, d = 16, 4, 1024, 128
B = b * h
def rms(x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 0.0078125)
q = (rms(torch.randn(B, t, d)) * 0.1).bfloat16()
kk = rms(torch.randn(B, t, d))
kk[:, 1:, 32:64] = rms(torch.randn(B, t - 1, d))[:, :, 32:64]          # key offset: mixed normalized rows (norm up to sqrt(256))
k = kk.bfloat16(); v = torch.randn(B, t, d).bfloat16()
sink = torch.zeros(B, 1)
# doc bounds per ROW (shared by the 4 heads): BOS patterns
is_bos = torch.rand(b, t) < 0.02
is_bos[0, 0] = True; is_bos[1, :] = False; is_bos[2, ::7] = True; is_bos[3, -1] = True
pos = torch.arange(t)
start = torch.zeros(b, t); end = torch.full((b, t), float(t))
for r in range(b):
    cur = 0
    for p in range(t):
        if is_bos[r, p]: cur = p
        start[r, p] = cur
    nxt = t
    for p in range(t - 1, -1, -1):
        end[r, p] = nxt
        if is_bos[r, p]: nxt = p
start = start.unsqueeze(1).expand(b, h, t).reshape(B, t); end = end.unsqueeze(1).expand(b, h, t).reshape(B, t)
out_ref, lse_ref = reference(q.float(), k.float(), v.float(), sink, start, end)
print("logit bound check: max |s| =", (q.float() @ k.float().transpose(1, 2)).abs().max().item(), flush=True)
lo, hi = bounds_lo_hi(start, end)
sink_rows = sink.view(B, 1, 1).expand(B, t, 1).contiguous()
end_rows, kidx = fwd2_args(sink_rows, end, t)
qd, kd, vd, skd, lod, hid, erd, kxd = (a.to(dev) for a in (q, k, v, sink_rows, lo, hi, end_rows, kidx))
def cmp(name, got, ref):
    got = got.float().cpu(); ref = ref.float()
    bad = ~torch.isfinite(got)
    e = (got - ref).abs().max().item()
    print(f"{name}: max |diff| {e:.3e} (ref max {ref.abs().max().item():.3e}) nonfinite={bad.sum().item()}", flush=True)
    if bad.any():
        idx = bad.nonzero()[:5].tolist(); print("   first nonfinite at (problem, row, col):", idx, flush=True)
o1, l1 = k1[(1,)](qd, kd, vd, skd, lod, hid); torch.neuron.synchronize()
cmp("fwd v1 out", o1, out_ref); cmp("fwd v1 lse", l1, lse_ref)
o4, l4 = k4[(1,)](qd, kd, vd, skd, erd, kxd); torch.neuron.synchronize()
cmp("fwd v4 out", o4, out_ref); cmp("fwd v4 lse", l4, lse_ref)
qr = q.float().requires_grad_(True); kr = k.float().requires_grad_(True); vr = v.float().requires_grad_(True); skr = sink.clone().requires_grad_(True)
out_r, lse_r = reference(qr, kr, vr, skr, start, end); do = torch.randn(B, t, d) * 0.01; (out_r * do).sum().backward()
dq, dk, dv, ds = kb[(1,)](qd, kd, vd, out_r.detach().bfloat16().to(dev), do.bfloat16().to(dev), lse_r.detach().contiguous().to(dev), skd, lod, hid); torch.neuron.synchronize()
cmp("bwd v3 dq", dq, qr.grad); cmp("bwd v3 dk", dk, kr.grad); cmp("bwd v3 dv", dv, vr.grad); cmp("bwd v3 dsink rows", ds, ds.float().cpu())
print("done")
