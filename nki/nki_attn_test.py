#!/usr/bin/env python3
"""Dev tool (runs ON the box, single process, NEURON_LOGICAL_NC_CONFIG=2): does the vendor NKI flash attention
(nkilib attention_cte / attention_bwd via torch_neuronx.wrap_nki) reproduce our doc-masked causal attention with a
per-head denominator sink? Prints numerics (max abs err vs an explicit fp32 reference) and fwd+bwd timings."""
import os, sys, math, time, traceback
import torch
import torch_neuronx
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.core.attention.attention_bwd import attention_bwd

dev = torch.device("neuron")
LNC = int(get_logical_neuron_cores()); GRID = (LNC,)
fwd = wrap_nki(attention_cte); bwd = wrap_nki(attention_bwd)
torch.manual_seed(0)

def make_case(B, H, L, E, n_docs=3):
    q = torch.randn(B, H, L, E) * 0.5; k = torch.randn(B, H, L, E) * 0.5; v = torch.randn(B, H, L, E) * 0.5
    # document ids: each row cut into n_docs pieces at random BOS positions (first token is always a BOS)
    docs = torch.zeros(B, L, dtype=torch.long)
    starts_all = []
    for b in range(B):
        cuts = sorted(torch.randperm(L - 1)[: n_docs - 1].add(1).tolist()); starts = [0] + cuts
        starts_all.append(starts)
        for i, s in enumerate(starts): docs[b, s:] = i
    sink = torch.randn(B, H) * 0.5   # per-(row, head) sink logit
    return q, k, v, docs, starts_all, sink

def reference(q, k, v, docs, sink, scale):
    # explicit fp32 attention with causal + same-doc mask and a denominator-only sink logit
    B, H, L, E = q.shape
    s = (q.float() @ k.float().transpose(-2, -1)) * scale
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
    same = docs.unsqueeze(2) == docs.unsqueeze(1)
    mask = (causal & same).unsqueeze(1)
    s = s.masked_fill(~mask, float("-inf"))
    s_ext = torch.cat([s, sink.float().view(B, H, 1, 1).expand(B, H, L, 1)], dim=-1)
    p = torch.softmax(s_ext, dim=-1)[..., :L]
    return p @ v.float()

def bounds_from_docs(docs, starts_all, B, H, L):
    bmin = torch.zeros(B, L, dtype=torch.int32); bmax = torch.zeros(B, L, dtype=torch.int32)
    for b in range(B):
        starts = starts_all[b] + [L]
        for i in range(len(starts) - 1):
            bmin[b, starts[i]:starts[i + 1]] = starts[i]; bmax[b, starts[i]:starts[i + 1]] = starts[i + 1]
    # per (b,h) row for the kernel's flattened batch
    # fwd (attention_cte): per flattened (b,h) row, shape (B*H, L, 1); bwd (attention_bwd): per batch (B, L) float32
    fmin = bmin.unsqueeze(1).expand(B, H, L).reshape(B * H, L, 1).contiguous()
    fmax = bmax.unsqueeze(1).expand(B, H, L).reshape(B * H, L, 1).contiguous()
    return (fmin, fmax), (bmin.float().contiguous(), bmax.float().contiguous())

def run_kernel(q, k, v, bounds, sink, scale, want_grad_dy=None):
    B, H, L, E = q.shape
    (bmin, bmax), (wmin, wmax) = bounds
    qs = (q * scale).to(torch.bfloat16); kb = k.to(torch.bfloat16); vb = v.to(torch.bfloat16)
    out, neg_max, recip = fwd[GRID](qs.reshape(B * H, L, E), kb.reshape(B * H, L, E), vb.reshape(B * H, L, E),
                                    tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True,
                                    bound_min=bmin, bound_max=bmax, sink=sink.reshape(B * H, 1).float().contiguous())
    out = out.reshape(B, H, L, E)
    if want_grad_dy is None:
        return out, None
    lse_tiled = -(neg_max + torch.log(recip))                 # (B*H, 128, n_tiles) per the vendor wrapper
    lse_4d = lse_tiled.to(torch.float32).reshape(B, H, *lse_tiled.shape[-2:]).contiguous()
    dy = want_grad_dy.to(torch.bfloat16)
    res = bwd[GRID](qs.transpose(-2, -1).contiguous(), kb.transpose(-2, -1).contiguous(), vb.transpose(-2, -1).contiguous(),
                    out.to(torch.bfloat16).transpose(-2, -1).contiguous(), dy.transpose(-2, -1).contiguous(), lse_4d,
                    sinks_ref=sink.float().contiguous(), bound_min=wmin, bound_max=wmax,
                    use_causal_mask=True, mixed_precision=True, softmax_scale=1.0)
    dq, dk, dv = res[0].transpose(-2, -1) * scale, res[1].transpose(-2, -1), res[2].transpose(-2, -1)
    dsink = res[3] if len(res) > 3 else None
    return out, (dq, dk, dv, dsink)

def err(a, b): return (a.float() - b.float()).abs().max().item()

# ---------------- numerics (B=2) ----------------
B, H, L, E = 2, 4, 1024, 128; scale = 1.0 / math.sqrt(E)
q, k, v, docs, starts, sink = make_case(B, H, L, E)
qd, kd, vd, sd = (t.to(dev).requires_grad_(True) for t in (q, k, v, sink))
docs_d = docs.to(dev)
ref = reference(qd, kd, vd, docs_d, sd, scale)
dy = torch.randn_like(ref)
ref.backward(dy)
ref_out, ref_dq, ref_dk, ref_dv, ref_ds = ref.detach(), qd.grad.detach(), kd.grad.detach(), vd.grad.detach(), sd.grad.detach()
fb, wb = bounds_from_docs(docs, starts, B, H, L); bounds = (tuple(t.to(dev) for t in fb), tuple(t.to(dev) for t in wb))
try:
    out, grads = run_kernel(q.to(dev), k.to(dev), v.to(dev), bounds, sink.to(dev), scale, want_grad_dy=dy)
    torch_neuronx.synchronize()
    print(f"NUMERICS: out err {err(out, ref_out):.3e} (|ref| max {ref_out.abs().max().item():.2f})")
    dq, dk, dv, ds = grads
    print(f"  dq err {err(dq, ref_dq):.3e} (|ref| {ref_dq.abs().max().item():.2f}) | dk err {err(dk, ref_dk):.3e} (|ref| {ref_dk.abs().max().item():.2f})"
          f" | dv err {err(dv, ref_dv):.3e} (|ref| {ref_dv.abs().max().item():.2f})")
    if ds is not None: print(f"  dsink err {err(ds.reshape(B, H), ref_ds):.3e} (|ref| {ref_ds.abs().max().item():.3f}) shape {tuple(ds.shape)}")
    else: print("  dsink: kernel returned no sink gradient")
except Exception:
    print("KERNEL FAILED:"); traceback.print_exc(); sys.exit(1)

# ---------------- timing (B=8 = one LNC=2 worker's rows) ----------------
B = 8
q, k, v, docs, starts, sink = make_case(B, H, L, E)
qd, kd, vd, sd = (t.to(dev).requires_grad_(True) for t in (q, k, v, sink)); docs_d = docs.to(dev)
fb, wb = bounds_from_docs(docs, starts, B, H, L); bounds = (tuple(t.to(dev) for t in fb), tuple(t.to(dev) for t in wb))
dy = torch.randn(B, H, L, E, device=dev)
def time_it(fn, n=5):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n
def ref_step():
    for t in (qd, kd, vd, sd): t.grad = None
    reference(qd, kd, vd, docs_d, sd, scale).backward(dy)
def ker_step():
    run_kernel(qd.detach(), kd.detach(), vd.detach(), bounds, sd.detach(), scale, want_grad_dy=dy)
def ref_fwd():
    with torch.no_grad(): reference(qd, kd, vd, docs_d, sd, scale)
def ker_fwd():
    run_kernel(qd.detach(), kd.detach(), vd.detach(), bounds, sd.detach(), scale)
print(f"TIMING fwd only: explicit {time_it(ref_fwd)*1000:.1f} ms | NKI {time_it(ker_fwd)*1000:.1f} ms")
print(f"TIMING (B={B}, H={H}, L={L}, E={E}, LNC={LNC}, eager): explicit fp32 fwd+bwd {time_it(ref_step)*1000:.1f} ms | NKI flash fwd+bwd {time_it(ker_step)*1000:.1f} ms")
