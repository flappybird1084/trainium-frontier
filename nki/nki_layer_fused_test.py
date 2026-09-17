"""Simulator / device test of the one-kernel layer forward (nki_layer_fused.py) against the 3-kernel path it replaces:
  layer_fwd_fused  vs  pre_attn_fwd_ko (nki_pre_ko.py) -> vendor attention_cte -> post_attn_fwd_h (extracted from the vehicle)
All 13 outputs compared (o, xm, rq, rk, q, k, v, out, neg_max, recip, y2, x1, preT). --time: 20 warm calls each on device.
usage: python nki/nki_layer_fused_test.py [sim|device] [small|prod] [--time] [--from-vehicle]   (prod = b 16 x t 1024, 4 heads x 128, D 512, H 3072)
Off-box the vendor package is not installed: pass its directory via NKILIB_DIR (a copy of site-packages/nkilib's parent)."""
import os, sys, time, torch, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
if os.environ.get("NKILIB_DIR"):
    sys.path.insert(0, os.environ["NKILIB_DIR"])
import nki
if "--from-vehicle" in sys.argv:   # the exact vehicle build: extract the embedded kernel block from submission_raw/train3_fused.py
    _vf = os.environ.get("VEHICLE_FUSED", os.path.join(here, "..", "submission_raw", "train3_fused.py"))
    _vs = open(_vf).read()
    _i = _vs.index("    def _pre_body("); _j = _vs.index("\n    _HVC = ", _i)
    _blk = "\n".join(l[4:] if l.startswith("    ") else l for l in _vs[_i:_j].split("\n"))
    import importlib.util, tempfile
    _fd = tempfile.mkdtemp(prefix="nki_fused_vehicle_"); _fp = os.path.join(_fd, "fused_vehicle.py")
    open(_fp, "w").write("import nki, nki.isa as nisa, nki.language as nl, numpy as np\n_NKI_P = 128\n_FMAX = 512\n"
                         "from nkilib.core.attention.attention_cte import _attention_cte\n" + _blk)
    _sp = importlib.util.spec_from_file_location("fused_vehicle", _fp); _fm = importlib.util.module_from_spec(_sp)
    sys.modules["fused_vehicle"] = _fm; _sp.loader.exec_module(_fm)
    layer_fwd_fused = _fm.layer_fwd_fused
    print("layer_fwd_fused: extracted from", _vf, flush=True)
else:
    from nki_layer_fused import layer_fwd_fused
from nki_pre_ko import pre_attn_fwd_ko
from nkilib.core.attention.attention_cte import attention_cte
dev = sys.argv[1] if len(sys.argv) > 1 else "sim"
shape = sys.argv[2] if len(sys.argv) > 2 else "small"
do_time = "--time" in sys.argv
b, t = (16, 1024) if shape == "prod" else (2, 512)
NH, HD, H = 4, 128, 3072
D = NH * HD; T = b * t; HQ = HD // 4; BH = b * NH
eps = 0.0078125; scale = 1.0 / HD ** 0.5
# --- the current post kernel, extracted from the vehicle (kernel bodies live there by text); materialized as a real module
vehicle = os.environ.get("VEHICLE", os.path.join(here, "..", "submission_raw", "train3_ko.py"))
src = open(vehicle).read()
i = src.index("    def post_attn_fwd_h("); i = src.rindex("    @nki.jit", 0, i); j = src.index("\n    @nki.jit", i + 20)
block = "\n".join(l[4:] if l.startswith("    ") else l for l in src[i:j].split("\n"))
import importlib.util, tempfile
_refdir = tempfile.mkdtemp(prefix="nki_post_h_ref_"); _refpath = os.path.join(_refdir, "post_h_ref.py")
open(_refpath, "w").write("import nki, nki.isa as nisa, nki.language as nl, numpy as np\n_NKI_P = 128\n_FMAX = 512\n_fq8 = None\n" + block)
_spec = importlib.util.spec_from_file_location("post_h_ref", _refpath); _mod = importlib.util.module_from_spec(_spec)
sys.modules["post_h_ref"] = _mod; _spec.loader.exec_module(_mod)
post_h = _mod.post_attn_fwd_h
if dev == "sim":
    run = lambda k, *a, **kw: nki.simulate(k)(*a, **kw)
    sync = lambda: None
    d = torch.device("cpu")
else:
    from torch_neuronx.nki_hop import wrap_nki
    d = torch.device("neuron")
    run = lambda k, *a, **kw: wrap_nki(k)[(1,)](*[v.to(d) if isinstance(v, torch.Tensor) else v for v in a], **kw)
    sync = torch.neuron.synchronize
cvt = lambda a: (a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))).float().cpu()
torch.manual_seed(0)
# docs: starts on/around tile boundaries and at the last position (carry + mask cases), plus a long doc per row
dstart = torch.zeros(b, t)
dstart[:, 0] = 1.0
for r in range(b):
    for p in (127, 128, 129, 255, 256, t - 1, (37 * r + 300) % t):
        dstart[r, p] = 1.0
dstart = dstart.contiguous()
dstart_col = dstart.reshape(T, 1).contiguous()
starts = torch.cumsum(dstart, dim=1) > 0   # doc id per position
doc_id = torch.cumsum(dstart, dim=1)
pos = torch.arange(t).view(1, t).expand(b, t)
start = torch.zeros(b, t, dtype=torch.int32); end = torch.zeros(b, t, dtype=torch.int32)
for r in range(b):
    ids = doc_id[r]
    for p in range(t):
        same = (ids == ids[p]).nonzero().flatten()
        start[r, p] = int(same.min()); end[r, p] = int(same.max()) + 1
start = start.contiguous(); end = end.contiguous()
x = torch.randn(T, D).bfloat16(); x0 = torch.randn(T, D).bfloat16(); ve = torch.randn(T, D).bfloat16()
wq, wk, wv, wo = [(torch.randn(D, D) * D ** -0.5).bfloat16() for _ in range(4)]
w1 = (torch.randn(H, D) * D ** -0.5).bfloat16(); w2 = (torch.randn(D, H) * H ** -0.5).bfloat16()
ang = torch.outer(torch.arange(t).float(), 1.0 / (10000 ** (torch.arange(HD // 2).float() / (HD // 2))))
cos = torch.cos(ang).repeat(b, 1).contiguous(); sin = torch.sin(ang).repeat(b, 1).contiguous()
scal = torch.tensor([0.9, 0.1, 0.5, 0.0]).view(1, 4).expand(128, 4).contiguous()
sink = torch.randn(NH) * 0.5
# the vehicle's attn_fwd glue for the vendor entry point (identical for both paths)
sink_rows = sink.float().view(1, NH).expand(b, NH).reshape(BH, 1).contiguous()
bmin = start.unsqueeze(1).expand(b, NH, t).reshape(BH, t, 1).contiguous()
bmax = end.unsqueeze(1).expand(b, NH, t).reshape(BH, t, 1).contiguous()
names = ("o", "xm", "rq", "rk", "q", "k", "v", "out", "neg_max", "recip", "y2", "x1", "preT")

def three_kernel(x, x0, ve, wq, wk, wv, cos, sin, scal, dstart_col, sink_rows, bmin, bmax, wo, w1, w2):
    qh, kh, vh, xm, rq, rk = run(pre_attn_fwd_ko, x, x0, ve, wq, wk, wv, cos, sin, scal, dstart_col, n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ)
    out, neg_max, recip = run_attn(qh, kh, vh, sink_rows, bmin, bmax)
    o, x1, preT, y2 = run(post_h, out, wo, xm, w1, w2, eps=eps, n_head=NH, t_len=t)
    return o, xm, rq, rk, qh, kh, vh, out, neg_max, recip, y2, x1, preT

def run_attn(qh, kh, vh, sink_rows, bmin, bmax):
    if dev == "sim":
        return nki.simulate(attention_cte)(qh, kh, vh, tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True,
                                           bound_min=bmin, bound_max=bmax, sink=sink_rows)
    from torch_neuronx.nki_hop import wrap_nki
    a = [v.to(d) if isinstance(v, torch.Tensor) else v for v in (qh, kh, vh, sink_rows, bmin, bmax)]
    return wrap_nki(attention_cte)[(1,)](a[0], a[1], a[2], tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True,
                                         bound_min=a[4], bound_max=a[5], sink=a[3])

args = (x, x0, ve, wq, wk, wv, cos, sin, scal, dstart_col, sink_rows, bmin, bmax, wo, w1, w2)
t0 = time.monotonic()
ref = [cvt(o) for o in three_kernel(*args)]
sync(); t1 = time.monotonic()
out = [cvt(o) for o in run(layer_fwd_fused, *args, n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ)]
sync(); t2 = time.monotonic()
ok = True
for name, a, r in zip(names, out, ref):
    err = (a - r).abs().max().item()
    tol = 1e-3 if name in ("xm", "rq", "rk", "q", "k", "v") else 2e-2 * max(1.0, r.abs().max().item())   # attention-dependent parts: bf16 noise
    ok &= err <= tol and tuple(a.shape) == tuple(r.shape)
    print(f"fused {name}: max |diff| {err:.3e} (ref max {r.abs().max().item():.3e}) shape {tuple(a.shape)} vs ref {tuple(r.shape)}", flush=True)
print(f"first-call wall: 3-kernel {t1 - t0:.1f}s  fused {t2 - t1:.1f}s", flush=True)

if do_time and dev != "sim":
    from torch_neuronx.nki_hop import wrap_nki
    def bench(fn, n=20):
        fn(); sync(); s = time.monotonic()
        for _ in range(n): fn()
        sync(); return (time.monotonic() - s) / n * 1000
    kpre = wrap_nki(pre_attn_fwd_ko)[(1,)]; katt = wrap_nki(attention_cte)[(1,)]; kpost = wrap_nki(post_h)[(1,)]; kf = wrap_nki(layer_fwd_fused)[(1,)]
    da = [v.to(d) for v in args]
    xd, x0d, ved, wqd, wkd, wvd, cosd, sind, scald, dsd, skd, bmind, bmaxd, wod, w1d, w2d = da
    def f_pre(): return kpre(xd, x0d, ved, wqd, wkd, wvd, cosd, sind, scald, dsd, n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ)
    qh, kh, vh, xm, rq, rk = f_pre(); sync()
    def f_att(): return katt(qh, kh, vh, tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True, bound_min=bmind, bound_max=bmaxd, sink=skd)
    outd, nm, rc = f_att(); sync()
    def f_post(): return kpost(outd, wod, xm, w1d, w2d, eps=eps, n_head=NH, t_len=t)
    def f_three():
        qh, kh, vh, xm, rq, rk = f_pre()
        out, nm, rc = katt(qh, kh, vh, tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True, bound_min=bmind, bound_max=bmaxd, sink=skd)
        return kpost(out, wod, xm, w1d, w2d, eps=eps, n_head=NH, t_len=t)
    def f_fused(): return kf(*da, n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ)
    res = {}
    for name, fn in (("pre_attn_fwd_ko", f_pre), ("attention_cte", f_att), ("post_attn_fwd_h", f_post), ("3-kernel chain", f_three), ("layer_fwd_fused", f_fused)):
        res[name] = bench(fn)
        print(f"time {name}: {res[name]:.3f} ms per call ({T} tokens)", flush=True)
    s3 = res["pre_attn_fwd_ko"] + res["attention_cte"] + res["post_attn_fwd_h"]
    print(f"sum of the three: {s3:.3f} ms ; chain {res['3-kernel chain']:.3f} ms ; fused {res['layer_fwd_fused']:.3f} ms "
          f"-> fused/chain = {res['layer_fwd_fused'] / res['3-kernel chain']:.3f} ({'>=5% faster: PASS' if res['layer_fwd_fused'] <= 0.95 * res['3-kernel chain'] else 'not 5% faster: KILL'})", flush=True)
print("PASS" if ok else "FAIL")
