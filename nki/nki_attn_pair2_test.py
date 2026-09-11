"""Device test: which pairs of custom kernels in ONE compiled graph are nondeterministic (6 runs each):
(a) fwd v4 twice on independent inputs, (b) fwd v1 + bwd v3, (c) fwd v4 + bwd v3 with bwd on INDEPENDENT inputs (no data
dependency), (d) fwd v4 alone but called on two different inputs sequentially in one graph. usage: python nki_attn_pair2_test.py"""
import torch
from torch_neuronx.nki_hop import wrap_nki, nki_op
from nki_attn_fwd import attn_fwd, attn_fwd4, fwd2_args
from nki_attn_bwd import attn_bwd, reference, bounds_lo_hi
dev = torch.device("neuron")
k1, k4, kb = wrap_nki(attn_fwd), wrap_nki(attn_fwd4), wrap_nki(attn_bwd)


@nki_op("z::f1", mutates_args={})
def f1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    o, l = k1[(1,)](q, k, v, sk, lo, hi)
    return o, l


@nki_op("z::f4", mutates_args={})
def f4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, er: torch.Tensor, kx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    o, l = k4[(1,)](q, k, v, sk, er, kx)
    return o, l


@nki_op("z::b3", mutates_args={})
def b3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, sk: torch.Tensor,
       lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b, c, d = kb[(1,)](q, k, v, o, do, lse, sk, lo, hi)
    return a, b, c, d


torch.manual_seed(0)
B, T, D = 64, 1024, 128
def mk(seed):
    torch.manual_seed(seed)
    q = (torch.randn(B, T, D) * 0.1).bfloat16(); k = torch.randn(B, T, D).bfloat16(); v = torch.randn(B, T, D).bfloat16()
    sink = torch.zeros(B, 1); start = torch.zeros(B, T); end = torch.full((B, T), float(T))
    for b in range(B):
        cut = 50 + 13 * b
        if cut < T: start[b, cut:] = float(cut); end[b, :cut] = float(cut)
    lo, hi = bounds_lo_hi(start, end)
    sr = sink.view(B, 1, 1).expand(B, T, 1).contiguous(); er, kx = fwd2_args(sr, end, T)
    out_ref, lse_ref = reference(q.float(), k.float(), v.float(), sink, start, end)
    do = (torch.randn(B, T, D) * 0.01).bfloat16()
    return [a.to(dev) for a in (q, k, v, sr, lo, hi, er, kx, do, out_ref.bfloat16(), lse_ref.contiguous())]
A = mk(0); Bb = mk(1)
q, k, v, sr, lo, hi, er, kx, do, oref, lref = A
q2, k2, v2, sr2, lo2, hi2, er2, kx2, do2, oref2, lref2 = Bb
def g_a(): 
    o1, l1 = torch.ops.z.f4(q, k, v, sr, er, kx); o2, l2 = torch.ops.z.f4(q2, k2, v2, sr2, er2, kx2); return o1, l1, o2, l2
def g_b():
    o, l = torch.ops.z.f1(q, k, v, sr, lo, hi); return (o, l) + tuple(torch.ops.z.b3(q, k, v, o, do, l, sr, lo, hi))
def g_c():
    o, l = torch.ops.z.f4(q, k, v, sr, er, kx); return (o, l) + tuple(torch.ops.z.b3(q2, k2, v2, oref2, do2, lref2, sr2, lo2, hi2))
def g_d():
    return tuple(torch.ops.z.b3(q, k, v, oref, do, lref, sr, lo, hi)) + tuple(torch.ops.z.b3(q2, k2, v2, oref2, do2, lref2, sr2, lo2, hi2))
for name, g in (("a: fwd4 x2 independent", g_a), ("b: fwd1 + bwd3 dependent", g_b), ("c: fwd4 + bwd3 independent", g_c), ("d: bwd3 x2 independent", g_d)):
    c = torch.compile(g, backend="neuron", dynamic=False)
    first = None; nd = 0; worst = 0.0
    for it in range(6):
        r = c(); torch.neuron.synchronize(); outs = [o.float().cpu() for o in r]
        if first is None: first = outs; continue
        d = max((x - y).abs().max().item() for x, y in zip(outs, first))
        if d > 0: nd += 1; worst = max(worst, d)
    print(f"{name}: {nd} of 5 reruns differ (worst {worst:.3e})", flush=True)
print("done", flush=True)
