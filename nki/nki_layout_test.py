"""Simulator/device test: pre_attn_fwd_h / post_attn_fwd_h (attention-layout variants extracted from the vehicle) reproduce
pre_attn_fwd / post_attn_fwd exactly (q pre-scaled, head-major q/k/v; per-head y reads + y2 copy). usage: python nki/nki_layout_test.py [device]"""
import sys, re, importlib.util, torch, numpy as np
sys.path.insert(0, ".")
import nki
src = open("submission_raw/train3_submission.py").read()
i = src.index("    def pre_attn_fwd("); i = src.rindex("    @nki.jit", 0, i)
j = src.index("    _HVC = 512   # vocab chunk width of the fused head kernels")
block = src[i:j]
block = "\n".join(l[4:] if l.startswith("    ") else l for l in block.split("\n"))
ns = {}
exec("import nki, nki.isa as nisa, nki.language as nl, numpy as np\n_NKI_P = 128\n_FMAX = 512\n_AB_KB = 512\n_AF_QG = 4\n_AF_MAX = 20.0\n" + block, ns)
pre, pre_h, post, post_h = ns["pre_attn_fwd"], ns["pre_attn_fwd_h"], ns["post_attn_fwd"], ns["post_attn_fwd_h"]
dev = sys.argv[1] if len(sys.argv) > 1 else "sim"
torch.manual_seed(0)
b, t, D, NH = 2, 512, 512, 4
T = b * t; HD = D // NH
x = torch.randn(T, D).bfloat16(); x0 = torch.randn(T, D).bfloat16(); ve = torch.randn(T, D).bfloat16()
wq, wk, wv, wo = [(torch.randn(D, D) * D ** -0.5).bfloat16() for _ in range(4)]
w1 = (torch.randn(6 * D, D) * D ** -0.5).bfloat16(); w2 = (torch.randn(D, 6 * D) * (6 * D) ** -0.5).bfloat16()
ang = torch.outer(torch.arange(t).float(), 1.0 / (10000 ** (torch.arange(HD // 2).float() / (HD // 2))))
cos = torch.cos(ang).repeat(b, 1).contiguous(); sin = torch.sin(ang).repeat(b, 1).contiguous()
scal = torch.tensor([0.9, 0.1, 0.5, 0.0]).view(1, 4).expand(128, 4).contiguous()
run = (lambda k, *a, **kw: nki.simulate(k)(*a, **kw)) if dev == "sim" else None
if dev != "sim":
    from torch_neuronx.nki_hop import wrap_nki
    d = torch.device("neuron")
    run = lambda k, *a, **kw: wrap_nki(k)[(1,)](*[v.to(d) if isinstance(v, torch.Tensor) else v for v in a], **kw)
cvt = lambda a: (a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))).float().cpu()
q, k, v, xm, rq, rk = [cvt(o) for o in run(pre, x, x0, ve, wq, wk, wv, cos, sin, scal, n_head=NH, eps=0.0078125)]
qh, kh, vh, xmh, rqh, rkh = [cvt(o) for o in run(pre_h, x, x0, ve, wq, wk, wv, cos, sin, scal, n_head=NH, eps=0.0078125, t_len=t, q_scale=0.1)]
q4 = q.view(NH, b, t, HD).permute(1, 0, 2, 3).reshape(b * NH, t, HD) * 0.1
k4 = k.view(NH, b, t, HD).permute(1, 0, 2, 3).reshape(b * NH, t, HD); v4 = v.view(NH, b, t, HD).permute(1, 0, 2, 3).reshape(b * NH, t, HD)
for name, a, r in (("q(scaled)", qh, q4), ("k", kh, k4), ("v", vh, v4), ("xm", xmh, xm), ("rq", rqh, rq), ("rk", rkh, rk)):
    print(f"pre_h {name}: max |diff| {(a - r).abs().max().item():.3e} (ref max {r.abs().max().item():.3e})", flush=True)
y4 = torch.randn(b * NH, t, HD).bfloat16()
y2 = y4.view(b, NH, t, HD).permute(0, 2, 1, 3).reshape(T, D).contiguous()
o, x1, preT = [cvt(z) for z in run(post, y2, wo, xm.bfloat16(), w1, w2, eps=0.0078125)]
oh, x1h, preTh, y2h = [cvt(z) for z in run(post_h, y4, wo, xm.bfloat16(), w1, w2, eps=0.0078125, n_head=NH, t_len=t)]
for name, a, r in (("out", oh, o), ("x1", x1h, x1), ("preT", preTh, preT), ("y2", y2h, y2.float())):
    print(f"post_h {name}: max |diff| {(a - r).abs().max().item():.3e} (ref max {r.abs().max().item():.3e})", flush=True)
print("done")
