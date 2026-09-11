"""Device test: repeated compiled-graph runs of the own attention kernels with fresh allocations between runs; checks that
(1) every output is finite, (2) outputs are bit-identical across runs (determinism), (3) sentinel buffers allocated around the
inputs/outputs keep their values (no out-of-bounds writes). usage: python nki_attn_sentinel_test.py"""
import torch
from torch_neuronx.nki_hop import wrap_nki, nki_op
from nki_attn_fwd import attn_fwd4, fwd2_args
from nki_attn_bwd import attn_bwd, reference, bounds_lo_hi
dev = torch.device("neuron")
k4, kb = wrap_nki(attn_fwd4), wrap_nki(attn_bwd)


@nki_op("s::afwd4", mutates_args={})
def afwd4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, er: torch.Tensor, kx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    o, l = k4[(1,)](q, k, v, sk, er, kx)
    return o, l


@nki_op("s::abwd3", mutates_args={})
def abwd3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, sk: torch.Tensor,
          lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b, c, d = kb[(1,)](q, k, v, o, do, lse, sk, lo, hi)
    return a, b, c, d


def step(q, k, v, sk, er, kx, do, lo, hi):
    o, l = torch.ops.s.afwd4(q, k, v, sk, er, kx)
    dq, dk, dv, ds = torch.ops.s.abwd3(q, k, v, o, do, l, sk, lo, hi)
    return o, l, dq, dk, dv, ds


cstep = torch.compile(step, backend="neuron", dynamic=False)
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
first = None
bad = 0
for it in range(30):
    pads = [torch.full((1 << 18,), 7.0, device=dev) for _ in range(3)]            # sentinels between allocations
    ins = [a.clone().to(dev) for a in (q, k, v, sink_rows, end_rows, kidx, do, lo, hi)]
    pads += [torch.full((1 << 18,), 7.0, device=dev)]
    outs = cstep(*ins); torch.neuron.synchronize()
    outs = [o.float().cpu() for o in outs]
    fin = all(torch.isfinite(o).all().item() for o in outs)
    if first is None:
        first = outs
        same = True
    else:
        same = all(torch.equal(a, b) for a, b in zip(outs, first))
    sent_ok = all(bool((p == 7.0).all().item()) for p in pads)
    if not (fin and same and sent_ok):
        bad += 1
        print(f"iter {it}: finite={fin} same_as_first={same} sentinels_ok={sent_ok}", flush=True)
        if not same:
            for name, a, b in zip(("out", "lse", "dq", "dk", "dv", "ds"), outs, first):
                d = (a - b).abs(); print(f"   {name}: max diff {d.max().item():.3e} at {d.argmax().item()} (#diff {(d > 0).sum().item()})", flush=True)
    del pads, ins
print(f"done: {bad} bad iterations of 30", flush=True)
