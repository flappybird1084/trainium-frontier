"""Device test: the own attention kernels INSIDE a torch.compile(backend="neuron") graph (as nki_op custom ops) vs their eager
wrap_nki results: forward v1 (per-tile DMAs, control), forward v4 and backward v3 (one strided DMA per tensor).
usage: python nki_attn_graph_test.py"""
import torch
from torch_neuronx.nki_hop import wrap_nki, nki_op
from nki_attn_fwd import attn_fwd, attn_fwd4, fwd2_args
from nki_attn_bwd import attn_bwd, reference, bounds_lo_hi
dev = torch.device("neuron")
k1, k4, kb = wrap_nki(attn_fwd), wrap_nki(attn_fwd4), wrap_nki(attn_bwd)


@nki_op("t::afwd1", mutates_args={})
def afwd1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    o, l = k1[(1,)](q, k, v, sk, lo, hi)
    return o, l


@nki_op("t::afwd4", mutates_args={})
def afwd4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, er: torch.Tensor, kx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    o, l = k4[(1,)](q, k, v, sk, er, kx)
    return o, l


@nki_op("t::abwd3", mutates_args={})
def abwd3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, sk: torch.Tensor,
          lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b, c, d = kb[(1,)](q, k, v, o, do, lse, sk, lo, hi)
    return a, b, c, d


def g1(q, k, v, sk, lo, hi): return torch.ops.t.afwd1(q, k, v, sk, lo, hi)
def g4(q, k, v, sk, er, kx): return torch.ops.t.afwd4(q, k, v, sk, er, kx)
def gb(q, k, v, o, do, lse, sk, lo, hi): return torch.ops.t.abwd3(q, k, v, o, do, lse, sk, lo, hi)
c1 = torch.compile(g1, backend="neuron", dynamic=False); c4 = torch.compile(g4, backend="neuron", dynamic=False); cb = torch.compile(gb, backend="neuron", dynamic=False)

torch.manual_seed(0)
B, T, D = 32, 1024, 128
q = (torch.randn(B, T, D) * 0.3).bfloat16(); k = (torch.randn(B, T, D) * 0.3).bfloat16(); v = torch.randn(B, T, D).bfloat16()
sink = torch.randn(B, 1) * 0.5
start = torch.zeros(B, T); end = torch.full((B, T), float(T))
for b in range(B):
    cut = 100 + 37 * b
    start[b, cut:] = float(cut); end[b, :cut] = float(cut)
out_ref, lse_ref = reference(q.float(), k.float(), v.float(), sink, start, end)
lo, hi = bounds_lo_hi(start, end)
sink_rows = sink.view(B, 1, 1).expand(B, T, 1).contiguous()
end_rows, kidx = fwd2_args(sink_rows, end, T)
qd, kd, vd, skd, lod, hid, erd, kxd = (a.to(dev) for a in (q, k, v, sink_rows, lo, hi, end_rows, kidx))


def cmp(name, got, ref):
    got = got.float().cpu(); ref = ref.float()
    e = (got - ref).abs().max().item(); print(f"{name}: max |diff| {e:.3e} (ref max {ref.abs().max().item():.3e}) nan={torch.isnan(got).any().item()}", flush=True)


o1e, l1e = k1[(1,)](qd, kd, vd, skd, lod, hid); torch.neuron.synchronize()
o1c, l1c = c1(qd, kd, vd, skd, lod, hid); torch.neuron.synchronize()
cmp("fwd v1 eager vs ref out", o1e, out_ref); cmp("fwd v1 GRAPH vs ref out", o1c, out_ref); cmp("fwd v1 GRAPH vs ref lse", l1c, lse_ref)
o4e, l4e = k4[(1,)](qd, kd, vd, skd, erd, kxd); torch.neuron.synchronize()
o4c, l4c = c4(qd, kd, vd, skd, erd, kxd); torch.neuron.synchronize()
cmp("fwd v4 eager vs ref out", o4e, out_ref); cmp("fwd v4 GRAPH vs ref out", o4c, out_ref); cmp("fwd v4 GRAPH vs ref lse", l4c, lse_ref)
# backward: torch reference grads
qr = q.float().requires_grad_(True); kr = k.float().requires_grad_(True); vr = v.float().requires_grad_(True); skr = sink.clone().requires_grad_(True)
out_r, lse_r = reference(qr, kr, vr, skr, start, end)
do = torch.randn(B, T, D); (out_r * do).sum().backward()
od = out_r.detach().bfloat16().to(dev); dod = do.bfloat16().to(dev); lsed = lse_r.detach().contiguous().to(dev)
dqe, dke, dve, dse = kb[(1,)](qd, kd, vd, od, dod, lsed, skd, lod, hid); torch.neuron.synchronize()
dqc, dkc, dvc, dsc = cb(qd, kd, vd, od, dod, lsed, skd, lod, hid); torch.neuron.synchronize()
cmp("bwd v3 eager vs ref dq", dqe, qr.grad); cmp("bwd v3 GRAPH vs ref dq", dqc, qr.grad)
cmp("bwd v3 GRAPH vs ref dk", dkc, kr.grad); cmp("bwd v3 GRAPH vs ref dv", dvc, vr.grad)
print("dsink graph", [round(x, 3) for x in dsc.float().cpu().sum(dim=(1, 2)).tolist()[:4]], "ref", [round(x, 3) for x in skr.grad.view(-1).tolist()[:4]], flush=True)
print("done")
