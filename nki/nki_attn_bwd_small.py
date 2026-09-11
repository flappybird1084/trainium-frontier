"""Device test: our attention backward (nki_attn_bwd.attn_bwd) vs the TORCH reference (not the vendor kernel), first at a
small shape (B=2, T=256, one doc boundary) and then at the training shape (B=32, T=1024, boundaries), to localize the
device-vs-simulator disagreement seen in ATTNBWD. usage: python nki_attn_bwd_small.py"""
import time, torch
from torch_neuronx.nki_hop import wrap_nki
from nki_attn_bwd import attn_bwd, reference, bounds_lo_hi
dev = torch.device("neuron")
kern = wrap_nki(attn_bwd)


def run(B, T, D, seed=0):
    torch.manual_seed(seed)
    q = (torch.randn(B, T, D) * 0.3).bfloat16(); k = (torch.randn(B, T, D) * 0.3).bfloat16(); v = torch.randn(B, T, D).bfloat16()
    sink = (torch.randn(B, 1) * 0.5)
    start = torch.zeros(B, T); end = torch.full((B, T), float(T))
    for b in range(B):
        cut = 100 + 37 * b
        if cut < T:
            start[b, cut:] = float(cut); end[b, :cut] = float(cut)
    qr = q.float().requires_grad_(True); kr = k.float().requires_grad_(True); vr = v.float().requires_grad_(True)
    skr = sink.clone().requires_grad_(True)
    out, lse = reference(qr, kr, vr, skr, start, end)
    do = torch.randn(B, T, D)
    (out * do).sum().backward()
    o_b = out.detach().bfloat16()
    sink_rows = sink.view(B, 1, 1).expand(B, T, 1).contiguous()
    lo, hi = bounds_lo_hi(start, end)
    args = [a.to(dev) for a in (q, k, v, o_b, do.bfloat16(), lse.detach().contiguous(), sink_rows, lo, hi)]
    dq, dk, dv, dsr = kern[(1,)](*args); torch.neuron.synchronize()
    dq, dk, dv, dsr = (t.float().cpu() for t in (dq, dk, dv, dsr))
    for name, got, ref in (("dq", dq, qr.grad), ("dk", dk, kr.grad), ("dv", dv, vr.grad)):
        err = (got - ref).abs().max().item()
        bad = (got - ref).abs().amax(dim=(1, 2))
        print(f"B={B} T={T}: {name} max err {err:.3e} (ref max {ref.abs().max().item():.3e}) rel {err / ref.abs().max().item():.4f}; per-problem max err {[round(x, 3) for x in bad.tolist()[:8]]}", flush=True)
    ds = dsr.sum(dim=(1, 2))
    print(f"B={B} T={T}: dsink kernel {[round(x, 3) for x in ds.tolist()[:8]]} ref {[round(x, 3) for x in skr.grad.view(-1).tolist()[:8]]}", flush=True)
    t0 = time.monotonic()
    for _ in range(10): kern[(1,)](*args)
    torch.neuron.synchronize(); print(f"B={B} T={T}: {(time.monotonic() - t0) / 10 * 1000:.2f} ms per call", flush=True)


run(2, 256, 128)
run(2, 1024, 128)
run(32, 1024, 128)
print("done")
