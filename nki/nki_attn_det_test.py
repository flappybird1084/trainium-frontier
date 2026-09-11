"""Device test: determinism of each own attention kernel variant alone (eager wrap_nki, 8 runs each on identical inputs):
forward v1 (per-tile DMAs, online softmax), v2 (fixed ref, S^T), v3 (grouped), v4 (single DMAs); backward v3.
usage: python nki_attn_det_test.py"""
import torch
from torch_neuronx.nki_hop import wrap_nki
from nki_attn_fwd import attn_fwd, attn_fwd2, attn_fwd3, attn_fwd4, fwd2_args
from nki_attn_bwd import attn_bwd, reference, bounds_lo_hi
dev = torch.device("neuron")
torch.manual_seed(0)
B, T, D = 64, 1024, 128
q = (torch.randn(B, T, D) * 0.1).bfloat16(); k = torch.randn(B, T, D).bfloat16(); v = torch.randn(B, T, D).bfloat16()
sink = torch.zeros(B, 1); start = torch.zeros(B, T); end = torch.full((B, T), float(T))
for b in range(B):
    cut = 50 + 13 * b
    if cut < T: start[b, cut:] = float(cut); end[b, :cut] = float(cut)
lo, hi = bounds_lo_hi(start, end)
sink_rows = sink.view(B, 1, 1).expand(B, T, 1).contiguous(); end_rows, kidx = fwd2_args(sink_rows, end, T)
out_ref, lse_ref = reference(q.float(), k.float(), v.float(), sink, start, end)
do = (torch.randn(B, T, D) * 0.01).bfloat16()
qd, kd, vd, skd, lod, hid, erd, kxd, dod = (a.to(dev) for a in (q, k, v, sink_rows, lo, hi, end_rows, kidx, do))
od = out_ref.bfloat16().to(dev); lsed = lse_ref.contiguous().to(dev)


def check(name, fn, n=8):
    first = None; nd = 0; worst = 0.0
    for it in range(n):
        outs = [o.float().cpu() for o in fn()]
        if first is None: first = outs; continue
        diffs = [(a - b).abs().max().item() for a, b in zip(outs, first)]
        if max(diffs) > 0: nd += 1; worst = max(worst, max(diffs))
    print(f"{name}: {nd} of {n - 1} reruns differ (worst max diff {worst:.3e})", flush=True)


k1, k2, k3, k4, kb = (wrap_nki(f) for f in (attn_fwd, attn_fwd2, attn_fwd3, attn_fwd4, attn_bwd))
check("fwd v1", lambda: (lambda r: (torch.neuron.synchronize(), r)[1])(k1[(1,)](qd, kd, vd, skd, lod, hid)))
check("fwd v2", lambda: (lambda r: (torch.neuron.synchronize(), r)[1])(k2[(1,)](qd, kd, vd, skd, erd, kxd)))
check("fwd v3", lambda: (lambda r: (torch.neuron.synchronize(), r)[1])(k3[(1,)](qd, kd, vd, skd, erd, kxd)))
check("fwd v4", lambda: (lambda r: (torch.neuron.synchronize(), r)[1])(k4[(1,)](qd, kd, vd, skd, erd, kxd)))
check("bwd v3", lambda: (lambda r: (torch.neuron.synchronize(), r)[1])(kb[(1,)](qd, kd, vd, od, dod, lsed, skd, lod, hid)))
print("done", flush=True)
