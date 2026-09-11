"""Device test: own attention kernels fed by COMPILER-PRODUCED tensors inside one graph (q/k/v/o/do pass through a
permute+contiguous copy first, like the model), determinism over 6 runs: strided-load kernels vs plain per-tile-load
variants; plus timings of the plain variants eagerly. usage: python nki_attn_prod_test.py"""
import time, torch
from torch_neuronx.nki_hop import wrap_nki, nki_op
from nki_attn_fwd import attn_fwd4, fwd2_args
from nki_attn_bwd import attn_bwd, reference, bounds_lo_hi
from nki_attn_plain import attn_fwd4_plain, attn_bwd_plain
dev = torch.device("neuron")
KS = {"strided": (wrap_nki(attn_fwd4), wrap_nki(attn_bwd)), "plain": (wrap_nki(attn_fwd4_plain), wrap_nki(attn_bwd_plain))}
def make(name):
    kf, kb = KS[name]
    @nki_op(f"q::f_{name}", mutates_args={})
    def fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sk: torch.Tensor, er: torch.Tensor, kx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o, l = kf[(1,)](q, k, v, sk, er, kx)
        return o, l
    @nki_op(f"q::b_{name}", mutates_args={})
    def bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, sk: torch.Tensor,
            lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, c, d = kb[(1,)](q, k, v, o, do, lse, sk, lo, hi)
        return a, b, c, d
    fo = getattr(torch.ops.q, f"f_{name}"); bo = getattr(torch.ops.q, f"b_{name}")
    def step(qh, kh, vh, sk, er, kx, doh, lo, hi):
        # producers: (b, t, h, d) -> (b, h, t, d) -> (B, t, d) contiguous copies, like the model's glue
        q = qh.permute(0, 2, 1, 3).reshape(-1, qh.shape[1], qh.shape[3]).contiguous()
        k = kh.permute(0, 2, 1, 3).reshape(-1, kh.shape[1], kh.shape[3]).contiguous()
        v = vh.permute(0, 2, 1, 3).reshape(-1, vh.shape[1], vh.shape[3]).contiguous()
        o, l = fo(q, k, v, sk, er, kx)
        y = (o.view(qh.shape[0], qh.shape[2], qh.shape[1], qh.shape[3]).permute(0, 2, 1, 3) * 1.0).contiguous()   # a consumer copy
        do = doh.permute(0, 2, 1, 3).reshape(-1, doh.shape[1], doh.shape[3]).contiguous()
        return (y, l) + tuple(bo(q, k, v, o, do, l, sk, lo, hi))
    return torch.compile(step, backend="neuron", dynamic=False)
torch.manual_seed(0)
b, h, t, d = 16, 4, 1024, 128; B = b * h
qh = (torch.randn(b, t, h, d) * 0.1).bfloat16(); kh = torch.randn(b, t, h, d).bfloat16(); vh = torch.randn(b, t, h, d).bfloat16()
doh = (torch.randn(b, t, h, d) * 0.01).bfloat16()
sink = torch.zeros(B, 1); start = torch.zeros(B, t); end = torch.full((B, t), float(t))
for i in range(B):
    cut = 50 + 13 * i
    if cut < t: start[i, cut:] = float(cut); end[i, :cut] = float(cut)
lo, hi = bounds_lo_hi(start, end)
sr = sink.view(B, 1, 1).expand(B, t, 1).contiguous(); er, kx = fwd2_args(sr, end, t)
args = [a.to(dev) for a in (qh, kh, vh, sr, er, kx, doh, lo, hi)]
for name in ("strided", "plain"):
    fn = make(name); first = None; nd = 0; worst = 0.0
    for it in range(6):
        r = fn(*args); torch.neuron.synchronize(); outs = [o.float().cpu() for o in r]
        if first is None: first = outs; continue
        dd = max((x - y).abs().max().item() for x, y in zip(outs, first))
        if dd > 0: nd += 1; worst = max(worst, dd)
    print(f"{name} loads, compiler-produced inputs: {nd} of 5 reruns differ (worst {worst:.3e})", flush=True)
# eager timings of the plain variants
q = qh.permute(0, 2, 1, 3).reshape(B, t, d).contiguous().to(dev); k = kh.permute(0, 2, 1, 3).reshape(B, t, d).contiguous().to(dev); v = vh.permute(0, 2, 1, 3).reshape(B, t, d).contiguous().to(dev)
skd, erd, kxd, lod, hid = args[3], args[4], args[5], args[7], args[8]
kf, kb = KS["plain"]
o, l = kf[(1,)](q, k, v, skd, erd, kxd); torch.neuron.synchronize()
t0 = time.monotonic()
for _ in range(10): kf[(1,)](q, k, v, skd, erd, kxd)
torch.neuron.synchronize(); print(f"plain fwd: {(time.monotonic() - t0) / 10 * 1000:.2f} ms per call (B=64)", flush=True)
do = doh.permute(0, 2, 1, 3).reshape(B, t, d).contiguous().to(dev)
kb[(1,)](q, k, v, o, do, l, skd, lod, hid); torch.neuron.synchronize()
t0 = time.monotonic()
for _ in range(10): kb[(1,)](q, k, v, o, do, l, skd, lod, hid)
torch.neuron.synchronize(); print(f"plain bwd: {(time.monotonic() - t0) / 10 * 1000:.2f} ms per call (B=64)", flush=True)
print("done", flush=True)
