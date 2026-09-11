"""Device test: determinism of each own attention kernel INSIDE its own compiled graph (nki_op custom op), 8 runs on identical
inputs: forward v1 (per-tile output DMAs) vs v4 (strided output DMAs), backward v3. usage: python nki_attn_graphdet_test.py"""
import torch
from torch_neuronx.nki_hop import wrap_nki, nki_op
from nki_attn_fwd import attn_fwd, attn_fwd4, fwd2_args
from nki_attn_bwd import attn_bwd, reference, bounds_lo_hi
dev = torch.device("neuron")
k1, k4, kb = wrap_nki(attn_fwd), wrap_nki(attn_fwd4), wrap_nki(attn_bwd)


@nki_op("d::afwd1", mutates_args={})
def afwd1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    o, l = k1[(1,)](q, k, v, sk, lo, hi)
    return o, l


@nki_op("d::afwd4", mutates_args={})
def afwd4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, er: torch.Tensor, kx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    o, l = k4[(1,)](q, k, v, sk, er, kx)
    return o, l


@nki_op("d::abwd3", mutates_args={})
def abwd3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, sk: torch.Tensor,
          lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b, c, d = kb[(1,)](q, k, v, o, do, lse, sk, lo, hi)
    return a, b, c, d


c1 = torch.compile(lambda *a: torch.ops.d.afwd1(*a), backend="neuron", dynamic=False)
c4 = torch.compile(lambda *a: torch.ops.d.afwd4(*a), backend="neuron", dynamic=False)
cb = torch.compile(lambda *a: torch.ops.d.abwd3(*a), backend="neuron", dynamic=False)
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
    first = None; nd = 0; worst = 0.0; cnt = 0
    for it in range(n):
        r = fn(); torch.neuron.synchronize()
        outs = [o.float().cpu() for o in r]
        if first is None:
            first = outs
            e = (outs[0] - out_ref).abs().max().item() if name.startswith("fwd") else 0.0
            continue
        diffs = [(a - b).abs().max().item() for a, b in zip(outs, first)]
        if max(diffs) > 0:
            nd += 1; worst = max(worst, max(diffs)); cnt = max(cnt, max(int((a != b).sum().item()) for a, b in zip(outs, first)))
    print(f"{name}: {nd} of {n - 1} reruns differ (worst max diff {worst:.3e}, up to {cnt} elements)", flush=True)


check("fwd v1 graph", lambda: c1(qd, kd, vd, skd, lod, hid))
check("fwd v4 graph", lambda: c4(qd, kd, vd, skd, erd, kxd))
check("bwd v3 graph", lambda: cb(qd, kd, vd, od, dod, lsed, skd, lod, hid))
print("done", flush=True)
