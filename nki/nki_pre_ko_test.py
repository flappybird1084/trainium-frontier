"""Simulator / device test of the key-offset-folded layer kernels (nki_pre_ko.py) against the current kernels + torch glue:
  pre_attn_fwd_ko  vs  pre_attn_fwd_h (extracted from the vehicle) + `_fwd_impl`'s 5-op key-offset glue on k4
  pre_attn_bwd_ko  vs  pre_attn_bwd (nki_pre_bwd.py) + the dksh slice glue, and vs the fp32 torch reference (ref_pre_bwd)
usage: python nki/nki_pre_ko_test.py [sim|device] [small|prod] [--time]   (prod = b 16 x t 1024, 4 heads x 128, D 512)"""
import os, sys, time, torch, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import nki
from nki_pre_ko import pre_attn_fwd_ko, pre_attn_bwd_ko, key_offset_fwd_glue
from nki_pre_bwd import pre_attn_bwd, ref_pre_bwd, make_inputs, kernel_args
dev = sys.argv[1] if len(sys.argv) > 1 else "sim"
shape = sys.argv[2] if len(sys.argv) > 2 else "small"
do_time = "--time" in sys.argv
b, t = (16, 1024) if shape == "prod" else (2, 512)
NH, HD = 4, 128
D = NH * HD; T = b * t; HQ = HD // 4
eps = 0.0078125; scale = 0.1
# --- the current forward kernel, extracted from the vehicle (kernel bodies live there by text)
src = open(os.path.join(here, "..", "submission_raw", "train3_submission.py")).read()
i = src.index("    def pre_attn_fwd_h("); i = src.rindex("    @nki.jit", 0, i); j = src.index("\n    @nki.jit", i + 20)
block = "\n".join(l[4:] if l.startswith("    ") else l for l in src[i:j].split("\n"))
# the device tracer re-reads kernel source from a real file: materialize the extracted block as a module in a temp dir
import importlib.util, tempfile
_refdir = tempfile.mkdtemp(prefix="nki_pre_h_ref_"); _refpath = os.path.join(_refdir, "pre_h_ref.py")
open(_refpath, "w").write("import nki, nki.isa as nisa, nki.language as nl, numpy as np\n_NKI_P = 128\n_FMAX = 512\n_fq8 = None\n" + block)
_spec = importlib.util.spec_from_file_location("pre_h_ref", _refpath); _mod = importlib.util.module_from_spec(_spec)
sys.modules["pre_h_ref"] = _mod; _spec.loader.exec_module(_mod)
pre_h = _mod.pre_attn_fwd_h
if dev == "sim":
    run = lambda k, *a, **kw: nki.simulate(k)(*a, **kw)
    sync = lambda: None
else:
    from torch_neuronx.nki_hop import wrap_nki
    d = torch.device("neuron")
    run = lambda k, *a, **kw: wrap_nki(k)[(1,)](*[v.to(d) if isinstance(v, torch.Tensor) else v for v in a], **kw)
    sync = torch.neuron.synchronize
cvt = lambda a: (a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))).float().cpu()
torch.manual_seed(0)
# doc starts: every row start, plus starts on/around tile boundaries and at the last position (carry + mask cases)
dstart = torch.zeros(b, t)
dstart[:, 0] = 1.0
for r in range(b):
    for p in (127, 128, 129, 255, 256, t - 1, (37 * r + 300) % t):
        dstart[r, p] = 1.0
dstart = dstart.contiguous()
dstart_col = dstart.reshape(T, 1).contiguous()

# ---------------- forward
x = torch.randn(T, D).bfloat16(); x0 = torch.randn(T, D).bfloat16(); ve = torch.randn(T, D).bfloat16()
wq, wk, wv = [(torch.randn(D, D) * D ** -0.5).bfloat16() for _ in range(3)]
ang = torch.outer(torch.arange(t).float(), 1.0 / (10000 ** (torch.arange(HD // 2).float() / (HD // 2))))
cos = torch.cos(ang).repeat(b, 1).contiguous(); sin = torch.sin(ang).repeat(b, 1).contiguous()
scal = torch.tensor([0.9, 0.1, 0.5, 0.0]).view(1, 4).expand(128, 4).contiguous()
t0 = time.monotonic()
ref = [cvt(o) for o in run(pre_h, x, x0, ve, wq, wk, wv, cos, sin, scal, n_head=NH, eps=eps, t_len=t, q_scale=scale)]
sync(); t1 = time.monotonic()
ref[1] = key_offset_fwd_glue(ref[1].bfloat16().view(b, NH, t, HD), dstart, HQ).view(b * NH, t, HD).float()
out = [cvt(o) for o in run(pre_attn_fwd_ko, x, x0, ve, wq, wk, wv, cos, sin, scal, dstart_col, n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ)]
sync(); t2 = time.monotonic()
ok = True
for name, a, r in zip(("q", "k(offset)", "v", "xm", "rq", "rk"), out, ref):
    err = (a - r).abs().max().item(); ok &= err <= 1e-3
    print(f"fwd_ko {name}: max |diff| {err:.3e} (ref max {r.abs().max().item():.3e}) shape {tuple(a.shape)}", flush=True)
print(f"fwd first-call wall: ref {t1 - t0:.1f}s  ko {t2 - t1:.1f}s", flush=True)

# ---------------- backward
ins = make_inputs(b, t, NH, HD, seed=1)
dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, _ds, wq2, wk2, wv2 = ins
refb = ref_pre_bwd(dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart, wq2, wk2, wv2, scale, eps, HQ)
args = kernel_args(dq4, dk4, dv4, rq, rk, xm, dx1, cos2d, sin2d, dstart)     # ... + dksh built by the torch glue
t0 = time.monotonic()
cur = [cvt(o) for o in run(pre_attn_bwd, *args, wq2, wk2, wv2, n_head=NH, eps=eps, scale=scale, hq=HQ)]
sync(); t1 = time.monotonic()
new = [cvt(o) for o in run(pre_attn_bwd_ko, *args[:9], dstart_col, wq2, wk2, wv2, n_head=NH, eps=eps, scale=scale, hq=HQ)]
sync(); t2 = time.monotonic()
for name, o, c, r in zip(("dxm", "gv2", "dWq", "dWk", "dWv"), new, cur, refb):
    r = r.float()
    e_cur = (o - c).abs().max().item(); e_ref = (o - r).abs().max().item(); e_cr = (c - r).abs().max().item()
    ok &= e_ref <= max(1e-3, 2.0 * e_cr + 1e-6) if name != "dWq" else e_ref <= max(2e-2, 2.0 * e_cr)   # fp32 wgrads: sum-order noise
    print(f"bwd_ko {name}: max |ko - current| {e_cur:.3e}  |ko - torch ref| {e_ref:.3e}  (|current - ref| {e_cr:.3e}, ref max {r.abs().max().item():.3e})", flush=True)
print(f"bwd first-call wall: current {t1 - t0:.1f}s  ko {t2 - t1:.1f}s", flush=True)

if do_time and dev != "sim":
    def bench(fn, n=20):
        fn(); sync(); s = time.monotonic()
        for _ in range(n): fn()
        sync(); return (time.monotonic() - s) / n * 1000
    kh = wrap_nki(pre_h)[(1,)]; kko = wrap_nki(pre_attn_fwd_ko)[(1,)]
    bw = wrap_nki(pre_attn_bwd)[(1,)]; bko = wrap_nki(pre_attn_bwd_ko)[(1,)]
    fa = [v.to(d) for v in (x, x0, ve, wq, wk, wv, cos, sin, scal)]; dsd = dstart_col.to(d); dsb = dstart.to(d)
    ba = [v.to(d) for v in args]; wd = [v.to(d) for v in (wq2, wk2, wv2)]
    dk4d = dk4.to(d)
    def f_h(): return kh(*fa, n_head=NH, eps=eps, t_len=t, q_scale=scale)
    def f_h_glue():
        o = kh(*fa, n_head=NH, eps=eps, t_len=t, q_scale=scale)
        return key_offset_fwd_glue(o[1].view(b, NH, t, HD), dsb, HQ)
    def f_ko(): return kko(*fa, dsd, n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ)
    def b_cur(): return bw(*ba, *wd, n_head=NH, eps=eps, scale=scale, hq=HQ)
    def b_cur_glue():
        gsh = dk4d[:, :, 1:, HQ:2 * HQ] * (1.0 - dsb[:, 1:].to(dk4d.dtype)).view(b, 1, t - 1, 1)
        dksh = torch.cat([gsh, torch.zeros_like(gsh[:, :, :1])], dim=2).reshape(b * NH, t, HQ).contiguous()
        return bw(*ba[:9], dksh, *wd, n_head=NH, eps=eps, scale=scale, hq=HQ)
    def b_ko(): return bko(*ba[:9], dsd, *wd, n_head=NH, eps=eps, scale=scale, hq=HQ)
    for name, fn in (("pre_attn_fwd_h", f_h), ("pre_attn_fwd_h + eager glue", f_h_glue), ("pre_attn_fwd_ko", f_ko),
                     ("pre_attn_bwd", b_cur), ("pre_attn_bwd + eager dksh glue", b_cur_glue), ("pre_attn_bwd_ko", b_ko)):
        print(f"time {name}: {bench(fn):.3f} ms per call ({T} tokens)", flush=True)
print("PASS" if ok else "FAIL")
