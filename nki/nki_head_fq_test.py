"""Device test (one core): fused head (final norm + lm_head + asymmetric softcap + CE) fwd/bwd kernels vs the same math as eager
torch ops, at T=4096, D=512, V=8192 (one worker's micro-step). usage: python nki_head_test.py"""
import time
import numpy as np, torch, torch.nn.functional as F
from torch_neuronx.nki_hop import wrap_nki
import importlib.util, os
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py")); t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
head_fwd, head_bwd = t3.head_fwd, t3.head_bwd
torch.manual_seed(0)
T, D, V = 4096, 512, 8192
dev = torch.device("neuron")
x = torch.randn(T, D).bfloat16(); w = (torch.randn(V, D) * 0.05).bfloat16()
tgt = torch.randint(0, V, (T, 1), dtype=torch.int32); tgt[::37] = -1
sp, sn = 11.0, 9.0
scal = torch.tensor([sp, sn, 0.0, 0.0]).expand(128, 4).contiguous()
n_valid = int((tgt >= 0).sum())
dl = torch.full((T, 1), 1.0 / n_valid, dtype=torch.float32)
# CPU reference (fp32 math on the bf16 inputs)
xr = x.float().requires_grad_(True); wr = w.float().requires_grad_(True)
hr = F.rms_norm(xr, (D,), eps=0.0078125).bfloat16().float()
zr = hr @ wr.T
zcr = torch.where(zr >= 0, sp * torch.tanh(zr / sp), sn * torch.tanh(zr / sn))
t = tgt.view(-1).long(); valid = t >= 0
lossr = F.cross_entropy(zcr[valid], t[valid]); lossr.backward()
kf, kb = wrap_nki(head_fwd), wrap_nki(head_bwd)
xd, wd, wTd, tgtd, scald, dld = (a.to(dev) for a in (x, w, w.t().contiguous(), tgt, scal, dl))
loss, lse, h, rstd = kf[(1,)](xd, wTd, tgtd, scald); torch.neuron.synchronize()
lossq, _, _, _ = kf[(1,)](xd, wTd, tgtd, scald, fq=1); torch.neuron.synchronize()
print(f"fake-fp8 fwd: mean loss {lossq.float().sum().item() / n_valid:.5f} vs bf16 {loss.float().sum().item() / n_valid:.5f}", flush=True)
print(f"fwd kernel: mean loss {loss.float().sum().item() / n_valid:.5f} ref {lossr.item():.5f}", flush=True)
dx, dw, dcap = kb[(1,)](h, rstd, lse, dld, wd, wTd, tgtd, scald); torch.neuron.synchronize()
for name, got, ref in (("dx", dx.float().cpu(), xr.grad), ("dW", dw.float().cpu(), wr.grad)):
    err = (got - ref).abs().max().item(); print(f"bwd kernel: {name} max err {err:.3e} (ref max {ref.abs().max().item():.3e}) rel {err / ref.abs().max().item():.4f}", flush=True)
t0 = time.monotonic()
for _ in range(20): loss, lse, h, rstd = kf[(1,)](xd, wTd, tgtd, scald)
torch.neuron.synchronize(); print(f"fwd kernel: {(time.monotonic() - t0) / 20 * 1000:.2f} ms per 4096 tokens", flush=True)
t0 = time.monotonic()
for _ in range(20): dx, dw, dcap = kb[(1,)](h, rstd, lse, dld, wd, wTd, tgtd, scald)
torch.neuron.synchronize(); print(f"bwd kernel: {(time.monotonic() - t0) / 20 * 1000:.2f} ms per 4096 tokens", flush=True)
# eager torch head on device (what the vehicle does today: logits materialized in fp32)
xe = xd.clone().requires_grad_(True); we = wd.clone().requires_grad_(True); te = tgtd.view(-1).long(); ve = (te >= 0)
def eager():
    h = F.rms_norm(xe, (D,), eps=0.0078125)
    z = F.linear(h, we).float()
    zc = torch.where(z >= 0, sp * torch.tanh(z / sp), sn * torch.tanh(z / sn))
    lp = torch.log_softmax(zc, dim=-1)
    nll = -lp.gather(1, te.clamp_min(0).unsqueeze(1)).squeeze(1)
    return (nll * ve.float()).sum() / n_valid
l = eager(); l.backward(); torch.neuron.synchronize()
print(f"eager     : loss {l.item():.5f}; dx max err {(xe.grad.float().cpu() - xr.grad).abs().max().item():.3e}", flush=True)
t0 = time.monotonic()
for _ in range(10): xe.grad = None; we.grad = None; l = eager()
torch.neuron.synchronize(); print(f"eager fwd : {(time.monotonic() - t0) / 10 * 1000:.2f} ms per 4096 tokens", flush=True)
t0 = time.monotonic()
for _ in range(10): xe.grad = None; we.grad = None; l = eager(); l.backward()
torch.neuron.synchronize(); print(f"eager f+b : {(time.monotonic() - t0) / 10 * 1000:.2f} ms per 4096 tokens", flush=True)
print("done")
