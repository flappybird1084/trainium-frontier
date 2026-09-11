"""Device test: fwd + bwd in ONE compiled graph, determinism over 6 runs, for three loop variants of the problem loop:
affine_range (current), sequential_range, static (unrolled). usage: python nki_attn_pairdet_test.py"""
import torch
from torch_neuronx.nki_hop import wrap_nki, nki_op
from nki_attn_fwd import attn_fwd4, fwd2_args
from nki_attn_bwd import attn_bwd, reference, bounds_lo_hi
from nki_attn_variants import attn_fwd4_seq, attn_fwd4_static, attn_bwd_seq, attn_bwd_static
dev = torch.device("neuron")
K = {"affine": (wrap_nki(attn_fwd4), wrap_nki(attn_bwd)), "seq": (wrap_nki(attn_fwd4_seq), wrap_nki(attn_bwd_seq)),
     "static": (wrap_nki(attn_fwd4_static), wrap_nki(attn_bwd_static))}
ops = {}
for name in K:
    def make(name):
        kf, kb = K[name]
        @nki_op(f"p::fwd_{name}", mutates_args={})
        def fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, er: torch.Tensor, kx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            o, l = kf[(1,)](q, k, v, sk, er, kx)
            return o, l
        @nki_op(f"p::bwd_{name}", mutates_args={})
        def bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, sk: torch.Tensor,
                lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            a, b, c, d = kb[(1,)](q, k, v, o, do, lse, sk, lo, hi)
            return a, b, c, d
        fo = getattr(torch.ops.p, f"fwd_{name}"); bo = getattr(torch.ops.p, f"bwd_{name}")
        def step(q, k, v, sk, er, kx, do, lo, hi):
            o, l = fo(q, k, v, sk, er, kx)
            return (o, l) + tuple(bo(q, k, v, o, do, l, sk, lo, hi))
        return torch.compile(step, backend="neuron", dynamic=False)
    ops[name] = make(name)
torch.manual_seed(0)
B, T, D = 64, 1024, 128
q = (torch.randn(B, T, D) * 0.1).bfloat16(); k = torch.randn(B, T, D).bfloat16(); v = torch.randn(B, T, D).bfloat16()
sink = torch.zeros(B, 1); start = torch.zeros(B, T); end = torch.full((B, T), float(T))
for b in range(B):
    cut = 50 + 13 * b
    if cut < T: start[b, cut:] = float(cut); end[b, :cut] = float(cut)
lo, hi = bounds_lo_hi(start, end)
sink_rows = sink.view(B, 1, 1).expand(B, T, 1).contiguous(); end_rows, kidx = fwd2_args(sink_rows, end, T)
do = (torch.randn(B, T, D) * 0.01).bfloat16()
args = [a.to(dev) for a in (q, k, v, sink_rows, end_rows, kidx, do, lo, hi)]
for name, fn in ops.items():
    first = None; nd = 0; worst = 0.0
    for it in range(6):
        r = fn(*args); torch.neuron.synchronize(); outs = [o.float().cpu() for o in r]
        if first is None: first = outs; continue
        d = max((a - b).abs().max().item() for a, b in zip(outs, first))
        if d > 0: nd += 1; worst = max(worst, d)
    print(f"{name}: {nd} of 5 reruns differ (worst {worst:.3e})", flush=True)
print("done", flush=True)
