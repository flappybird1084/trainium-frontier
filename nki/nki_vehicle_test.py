#!/usr/bin/env python3
"""Dev tool (runs ON the box, single process, NEURON_LOGICAL_NC_CONFIG=2, PYTHONPATH=~/tf): integration test of the
vehicle's ATTN=nki path (NKIDocAttention + doc_bounds) against its own explicit SINK=logit path on real token rows.
Checks: doc_bounds vs the dense mask, attention fwd/bwd errors (q, k, v, sink), then whole-model eager fwd+bwd timing
at 512/4h/8 rows for explicit vs nki. usage: python nki_vehicle_test.py [--rows 8]"""
import os, sys, time, math, argparse, importlib.util
ap = argparse.ArgumentParser(); ap.add_argument("--rows", type=int, default=8); args = ap.parse_args()
os.environ["NEURON_COMPETITION_R1_SINK"] = "logit"; os.environ["NEURON_COMPETITION_R1_ATTN"] = "nki"
os.environ.setdefault("NEURON_COMPETITION_R1_N_EMBD", "512"); os.environ.setdefault("NEURON_COMPETITION_R1_N_HEAD", "4")
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py"))
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
from prepare import ensure_tokenizer, make_dataloader
dev = torch.device("neuron"); t3.COMPUTE_DTYPE = torch.bfloat16; torch.manual_seed(0)
tok = ensure_tokenizer(build_if_missing=False)
loader = make_dataloader(tok, args.rows, t3.SEQ_LEN, "train", torch.device("cpu"))
x, y, _ = next(iter(loader)); x = x.to(torch.int32)
B, T = x.shape; H, D = t3.N_HEAD, t3.N_EMBD // t3.N_HEAD
n_bos = int((x == t3.BOS_ID).sum()); print(f"rows {B} x {T}, BOS tokens in batch: {n_bos}", flush=True)

# 1. doc_bounds vs the explicit mask
xd = x.to(dev)
start, end = t3.doc_bounds(xd); start, end = start.cpu(), end.cpu()
docs = (x == t3.BOS_ID).cumsum(1); pos = torch.arange(T)
real = (pos.view(1, T, 1) >= pos.view(1, 1, T)) & (docs.unsqueeze(2) == docs.unsqueeze(1))
kv = pos.view(1, 1, T)
from_bounds = (kv >= start.unsqueeze(2)) & (kv < torch.minimum(end, pos + 1).unsqueeze(2))
print("doc_bounds == explicit mask:", bool((real == from_bounds).all()), "| mismatches:", int((real != from_bounds).sum()), flush=True)

# 2. attention fwd/bwd: NKIDocAttention vs the explicit sink-softmax (fp32) on random q,k,v
q = (torch.randn(B, H, T, D) * 0.5).to(torch.bfloat16); k = (torch.randn(B, H, T, D) * 0.5).to(torch.bfloat16)
v = (torch.randn(B, H, T, D) * 0.5).to(torch.bfloat16); sink = torch.randn(H) * 0.5
scale = 1.0 / math.sqrt(D)
def explicit(q, k, v, sink, mask):
    s = (q.float() @ k.float().transpose(-2, -1)) * scale
    s = s.masked_fill(~mask, float("-inf"))
    s_col = sink.float().view(1, H, 1, 1).expand(B, -1, T, 1)
    p = torch.softmax(torch.cat([s, s_col], -1), -1)[..., :T]
    return p @ v.float()
qr, kr, vr, sr = (t.clone().requires_grad_(True) for t in (q, k, v, sink))
ref = explicit(qr, kr, vr, sr, real.unsqueeze(1)); dy = torch.randn_like(ref); ref.backward(dy)
qd, kd, vd, sd = (t.to(dev).requires_grad_(True) for t in (q, k, v, sink))
t3.build_nki_ops(); out = t3.nki_doc_attention(qd, kd, vd, sd, start.to(dev), end.to(dev), scale)
out.backward(dy.to(dev)); torch_neuronx.synchronize()
e = lambda a, b: f"{(a.float().cpu() - b.float()).abs().max().item():.3e} (|ref| {b.abs().max().item():.2f})"
print("nki_doc_attention vs explicit: out", e(out, ref.detach()), "| dq", e(qd.grad, qr.grad), "| dk", e(kd.grad, kr.grad),
      "| dv", e(vd.grad, vr.grad), "| dsink", e(sd.grad, sr.grad), flush=True)

# 3. whole-model eager fwd+bwd timing, explicit vs nki (same weights)
model = t3.GPT(t3.build_config()); model.init_weights(); model.to(dev); t3.STACK_GRAD_MODE = "function"
yd = y.to(dev)
def step():
    model.zero_grad(set_to_none=True); loss = model(xd, yd); loss.backward(); torch_neuronx.synchronize(); return loss
def time_it(n=5):
    step(); t0 = time.time()
    for _ in range(n): step()
    return (time.time() - t0) / n
for impl in ("explicit", "nki", "explicit", "nki"):
    t3.ATTN_IMPL = impl; l = step(); dt = time_it()
    print(f"eager fwd+bwd {impl:8s}: {dt*1000:.0f} ms/step  loss {l.item():.4f}", flush=True)
