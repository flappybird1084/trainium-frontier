"""Device test: our attention forward (nki_attn_fwd.attn_fwd) vs the torch reference at B=2/T=256 and B=32/T=1024, plus
timing against the vendor attention_cte at the training shape. usage: python nki_attn_fwd_test.py"""
import time, torch
from torch_neuronx.nki_hop import wrap_nki
from nkilib.core.attention.attention_cte import attention_cte
from nki_attn_fwd import attn_fwd, attn_fwd2, attn_fwd3, attn_fwd4, fwd2_args
from nki_attn_bwd import reference, bounds_lo_hi
dev = torch.device("neuron")
kern = wrap_nki(attn_fwd); kern2 = wrap_nki(attn_fwd2); kern3 = wrap_nki(attn_fwd3); kern4 = wrap_nki(attn_fwd4); vend = wrap_nki(attention_cte)


def run(B, T, D, seed=0, time_vendor=False):
    torch.manual_seed(seed)
    q = (torch.randn(B, T, D) * 0.3).bfloat16(); k = (torch.randn(B, T, D) * 0.3).bfloat16(); v = torch.randn(B, T, D).bfloat16()
    sink = torch.randn(B, 1) * 0.5
    start = torch.zeros(B, T); end = torch.full((B, T), float(T))
    for b in range(B):
        cut = 100 + 37 * b
        if cut < T:
            start[b, cut:] = float(cut); end[b, :cut] = float(cut)
    out_ref, lse_ref = reference(q.float(), k.float(), v.float(), sink, start, end)
    lo, hi = bounds_lo_hi(start, end)
    sink_rows = sink.view(B, 1, 1).expand(B, T, 1).contiguous()
    args = [a.to(dev) for a in (q, k, v, sink_rows, lo, hi)]
    out, lse = kern[(1,)](*args); torch.neuron.synchronize()
    out, lse = out.float().cpu(), lse.float().cpu()
    e = (out - out_ref).abs().max().item(); print(f"B={B} T={T}: out max err {e:.3e} (ref max {out_ref.abs().max().item():.3e}) rel {e / out_ref.abs().max().item():.4f}", flush=True)
    e = (lse - lse_ref).abs().max().item(); print(f"B={B} T={T}: lse max err {e:.3e}", flush=True)
    t0 = time.monotonic()
    for _ in range(10): kern[(1,)](*args)
    torch.neuron.synchronize(); print(f"B={B} T={T}: ours {(time.monotonic() - t0) / 10 * 1000:.2f} ms per call", flush=True)
    end_rows, kidx = fwd2_args(sink_rows, end, T)
    args2 = [a.to(dev) for a in (q, k, v, sink_rows, end_rows, kidx)]
    out2, lse2 = kern2[(1,)](*args2); torch.neuron.synchronize()
    out2, lse2 = out2.float().cpu(), lse2.float().cpu()
    e = (out2 - out_ref).abs().max().item(); print(f"B={B} T={T}: v2 out max err {e:.3e} rel {e / out_ref.abs().max().item():.4f}", flush=True)
    e = (lse2 - lse_ref).abs().max().item(); print(f"B={B} T={T}: v2 lse max err {e:.3e}", flush=True)
    t0 = time.monotonic()
    for _ in range(10): kern2[(1,)](*args2)
    torch.neuron.synchronize(); print(f"B={B} T={T}: v2 {(time.monotonic() - t0) / 10 * 1000:.2f} ms per call", flush=True)
    if T % 512 == 0:
        out3, lse3 = kern3[(1,)](*args2); torch.neuron.synchronize()
        out3, lse3 = out3.float().cpu(), lse3.float().cpu()
        e = (out3 - out_ref).abs().max().item(); print(f"B={B} T={T}: v3 out max err {e:.3e} rel {e / out_ref.abs().max().item():.4f}", flush=True)
        e = (lse3 - lse_ref).abs().max().item(); print(f"B={B} T={T}: v3 lse max err {e:.3e}", flush=True)
        t0 = time.monotonic()
        for _ in range(10): kern3[(1,)](*args2)
        torch.neuron.synchronize(); print(f"B={B} T={T}: v3 {(time.monotonic() - t0) / 10 * 1000:.2f} ms per call", flush=True)
    if T % 512 == 0:
        out4, lse4 = kern4[(1,)](*args2); torch.neuron.synchronize()
        out4, lse4 = out4.float().cpu(), lse4.float().cpu()
        e = (out4 - out_ref).abs().max().item(); print(f"B={B} T={T}: v4 out max err {e:.3e} rel {e / out_ref.abs().max().item():.4f}", flush=True)
        e = (lse4 - lse_ref).abs().max().item(); print(f"B={B} T={T}: v4 lse max err {e:.3e}", flush=True)
        t0 = time.monotonic()
        for _ in range(10): kern4[(1,)](*args2)
        torch.neuron.synchronize(); print(f"B={B} T={T}: v4 {(time.monotonic() - t0) / 10 * 1000:.2f} ms per call", flush=True)
    if time_vendor:
        bmin = start.view(B, T, 1).contiguous().to(dev); bmax = end.view(B, T, 1).contiguous().to(dev)
        sk = sink.view(B, 1).contiguous().to(dev)
        qd, kd, vd = args[0], args[1], args[2]
        vend[(1,)](qd, kd, vd, tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True, bound_min=bmin, bound_max=bmax, sink=sk)
        torch.neuron.synchronize()
        t0 = time.monotonic()
        for _ in range(10): vend[(1,)](qd, kd, vd, tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True, bound_min=bmin, bound_max=bmax, sink=sk)
        torch.neuron.synchronize(); print(f"B={B} T={T}: vendor attention_cte {(time.monotonic() - t0) / 10 * 1000:.2f} ms per call", flush=True)


run(2, 256, 128)
run(32, 1024, 128, time_vendor=True)
print("done")
