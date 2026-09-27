#!/usr/bin/env python3
"""Simulator / device test of the CK conv folded into the layer kernels (submission_raw/train3_ck.py).

CK is a learnable causal depthwise conv on the block input, computed INSIDE pre_attn_fwd_ko / pre_attn_bwd_ko:
    u[p, d] = sum_i w[i, d] * x[p-i, d] * c_i[p],   c_i[p] = prod_{j<i} (1 - dstart[p-j])
i.e. the torch `canon_conv` the vehicle applies at the top of Block.forward, with the taps dropped wherever they
would reach across a document start. What is checked, against the SAME kernels run with ck_w=0 plus the torch
reference (so the reference is the vehicle's own `canon_conv` and its autograd, not a re-derivation):

  fwd   pre_attn_fwd_ko(x, ck_w=W)              vs  pre_attn_fwd_ko(canon_conv(x), ck_w=0)      - all 6 outputs
  bwd   pre_attn_bwd_ko(..., ck_w=W).dxm        vs  d(canon_conv)/dx   applied to ck_w=0's dxm
        pre_attn_bwd_ko(..., ck_w=W).dck        vs  d(canon_conv)/dw   applied to ck_w=0's dxm

Two data regimes:
  exact   x, dx1 small integers and w a power-of-2 per channel -> every product and sum is exact in bf16/fp32, so the
          agreement must be BIT-EXACT. This is what actually pins the shift / carry / mask / tile indexing down.
  random  normal data -> agreement to bf16 rounding of the tap terms (the kernel rounds (x*m_i)*w_i to bf16 to feed
          the PE row shift, and the ck_w=0 reference rounds the whole conv output to bf16 on its way through HBM).

Document starts are placed at a sequence-row start, MID-tile, exactly ON a 128-row tile boundary, one row before a
boundary, at two consecutive positions, and at the last position of a row - i.e. every cross-tile carry case.

usage: python nki/nki_ck_test.py [sim|device] [W] [prod]
  default sim W=3: the flag-off identity, the op-side algebra, and two small shapes (tiles%4 == 0 and != 0)
  prod:            the real training shape instead (b 16 x t 1024, 128 tiles, 8 tiles per sequence row; ~3 min)
  device:          the same checks through wrap_nki on the box - the ONLY way to test the tracer rules the
                   simulator does not enforce (module-level kernels, PSUM operands, DMA shapes, rules #8-#15)
"""
import os, sys, time, importlib.util, tempfile
import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import nki

dev = sys.argv[1] if len(sys.argv) > 1 else "sim"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 3
VEHICLE = os.path.join(here, "..", "submission_raw", "train3_ck.py")
src = open(VEHICLE).read()

# ---- the kernels and the torch reference, taken from the vehicle by text (kernel bodies live there, and the device
# tracer needs a real file: write the extracted block to one and import it, rule #13).
i = src.index("    def pre_attn_fwd_ko("); i = src.rindex("    @nki.jit", 0, i)
k = src.index("    def pre_attn_bwd_ko("); j = src.index("\n    @nki.jit", k + 20)
block = "\n".join(l[4:] if l.startswith("    ") else l for l in src[i:j].split("\n"))
_dir = tempfile.mkdtemp(prefix="nki_ck_")
_path = os.path.join(_dir, "ck_kernels.py")
open(_path, "w").write("import nki, nki.isa as nisa, nki.language as nl\n_NKI_P = 128\n_FMAX = 512\nROPE_FRAC = 0.5\n" + block)
_spec = importlib.util.spec_from_file_location("ck_kernels", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ck_kernels"] = _mod
_spec.loader.exec_module(_mod)
fwd_k, bwd_k = _mod.pre_attn_fwd_ko, _mod.pre_attn_bwd_ko

_ns = {"torch": torch, "CK_W": W}
exec(compile(src[src.index("def canon_conv("):src.index("class Block(")], "<vehicle>", "exec"), _ns)  # noqa: S102
canon_conv, ck_table = _ns["canon_conv"], _ns["ck_table"]

if dev == "sim":
    run = lambda kern, *a, **kw: nki.simulate(kern)(*a, **kw)
    sync = lambda: None
else:
    from torch_neuronx.nki_hop import wrap_nki
    _d = torch.device("neuron")
    run = lambda kern, *a, **kw: wrap_nki(kern)[(1,)](*[v.to(_d) if isinstance(v, torch.Tensor) else v for v in a], **kw)
    sync = torch.neuron.synchronize
import numpy as np
cvt = lambda a: (a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))).float().cpu()

NH, HD = 4, 128
D = NH * HD
half = HD // 2
eps = 0.0078125
scale = 0.1
HQ = HD // 4
ok = True


def doc_starts(b, t):
    """every carry case: row start, mid-tile, ON a tile boundary, one before it, two in a row, last position"""
    ds = torch.zeros(b, t)
    ds[:, 0] = 1.0
    for r in range(b):
        for p in (37, 127, 128, 129, 130, t - 1, (53 * r + 71) % t):
            if 0 <= p < t:
                ds[r, p] = 1.0
    return ds.contiguous()


def tables(ds, w):
    """the host-side tensors the op builds: the widened dstart (masks) and the widened scal (replicated weights)"""
    tab = ck_table(ds.bool(), ds.shape[0], ds.shape[1], ds.device).reshape(-1, 1 + 2 * (W - 1)).contiguous()
    srow = torch.tensor([0.9, 0.1, 0.5, 0.0]).view(1, 4)
    scal_ck = torch.cat([srow, w.reshape(1, W * D).float()], dim=1).expand(128, 4 + W * D).contiguous()
    return tab, srow.expand(128, 4).contiguous(), scal_ck


def check(name, got, ref, tol):
    global ok
    err = (got - ref).abs().max().item()
    good = err <= tol
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} {name:12s} max|diff| {err:.3e}  tol {tol:.1e}  "
          f"(ref max {ref.abs().max().item():.4g}, shape {tuple(got.shape)})", flush=True)


def one_shape(b, t, exact):
    global ok
    T = b * t
    tiles = T // 128
    tag = f"b{b} t{t} T{T} tiles{tiles} ({'exact' if exact else 'random'}, dck groups of {4 if tiles % 4 == 0 else 1})"
    print(f"\n=== {tag} ===", flush=True)
    torch.manual_seed(0)
    ds = doc_starts(b, t)
    if exact:   # small integers and power-of-two weights: every product/sum below is exact in bf16 and fp32
        x = torch.randint(-4, 5, (T, D)).float().bfloat16()
        w = torch.zeros(W, D)
        w[0] = 1.0
        for _i in range(1, W):
            w[_i] = torch.pow(2.0, -torch.randint(1, 4, (D,)).float()) * torch.randint(0, 2, (D,)).mul(2).sub(1).float()
        dx1 = torch.randint(-4, 5, (T, D)).float().bfloat16()
    else:
        x = torch.randn(T, D).bfloat16()
        w = torch.randn(W, D) * 0.2
        w[0] += 1.0
        dx1 = (torch.randn(T, D) * 0.1).bfloat16()
    x0 = torch.randn(T, D).bfloat16()
    ve = torch.randn(T, D).bfloat16()
    wq, wk, wv = [(torch.randn(D, D) * D ** -0.5).bfloat16() for _ in range(3)]
    ang = torch.outer(torch.arange(t).float(), 1.0 / (10000 ** (torch.arange(half).float() / half)))
    cos = torch.cos(ang).repeat(b, 1).contiguous()
    sin = torch.sin(ang).repeat(b, 1).contiguous()
    tab, scal4, scal_ck = tables(ds, w)
    dcol = ds.reshape(T, 1).contiguous()
    dsb = ds.bool().reshape(b, t)

    # ---------------- forward: the conv in torch feeding the ck_w=0 kernel vs the conv inside the kernel
    xc = canon_conv(x.float().view(b, t, D), w, dsb).view(T, D)
    t0 = time.monotonic()
    ref = [cvt(o) for o in run(fwd_k, xc.bfloat16().contiguous(), x0, ve, wq, wk, wv, cos, sin, scal4, dcol,
                               n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ, ck_w=0)]
    sync()
    got = [cvt(o) for o in run(fwd_k, x, x0, ve, wq, wk, wv, cos, sin, scal_ck, tab,
                               n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ, ck_w=W)]
    sync()
    print(f"  [fwd {time.monotonic() - t0:.0f}s]  conv max|u| {xc.abs().max().item():.4g}", flush=True)
    for nm in range(6):
        # random regime: the reference hands conv(x) to the kernel through bf16 HBM while the kernel keeps u in fp32,
        # so everything downstream may differ by a bf16 ULP of u (2^-8 relative) - allow two of them.
        tol = 0.0 if exact else 2.0 ** -7 * max(1.0, ref[nm].abs().max().item())
        check(("q", "k", "v", "xm", "rq", "rk")[nm], got[nm], ref[nm], tol)

    # ---------------- backward: dq = dk = dv = 0 in the exact regime, so the kernel's internal g is dx1 exactly
    z = torch.zeros(b * NH, t, HD).bfloat16()
    dq = z if exact else (torch.randn(b * NH, t, HD) * 0.05).bfloat16()
    dk = z if exact else (torch.randn(b * NH, t, HD) * 0.05).bfloat16()
    dv = z if exact else (torch.randn(b * NH, t, HD) * 0.05).bfloat16()
    rq = torch.randn(NH, T, HD).bfloat16()
    rk = torch.randn(NH, T, HD).bfloat16()
    xm = torch.randn(T, D).bfloat16()
    t0 = time.monotonic()
    base = [cvt(o) for o in run(bwd_k, dq, dk, dv, rq, rk, xm, dx1, cos, sin, dcol, wq, wk, wv, x, scal4,
                                n_head=NH, eps=eps, scale=scale, hq=HQ, ck_w=0)]
    sync()
    new = [cvt(o) for o in run(bwd_k, dq, dk, dv, rq, rk, xm, dx1, cos, sin, tab, wq, wk, wv, x, scal_ck,
                               n_head=NH, eps=eps, scale=scale, hq=HQ, ck_w=W)]
    sync()
    print(f"  [bwd {time.monotonic() - t0:.0f}s]", flush=True)
    # the conv is the only difference, so the reference is canon_conv's own autograd fed with ck_w=0's dxm
    g = base[0].view(b, t, D).clone()
    xr = x.float().view(b, t, D).clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    canon_conv(xr, wr, dsb).backward(g)
    # bf16 tolerances in the random regime: the kernel rounds each tap term to bf16 (2^-9 relative) before the PE
    # shift, and the reference's g is ck_w=0's bf16 output where the kernel used its fp32 internal value.
    tol_dx = 0.0 if exact else 6e-3 * max(1.0, g.abs().max().item())
    tol_dw = 0.0 if exact else 6e-3 * (g.abs().reshape(T, D) * x.float().abs()).sum(0).max().item()
    check("dxm (A^T g)", new[0], xr.grad.reshape(T, D), tol_dx)
    check("dck (dw)", new[5][0:1].reshape(W, D), wr.grad, tol_dw)
    for nm in range(1, 5):   # everything else must be untouched by the conv
        check(("", "gv2", "dWq", "dWk", "dWv")[nm], new[nm], base[nm], 0.0)
    if exact:   # and the conv-free part of the gradient must really be dx1 in this regime
        check("g == dx1", base[0], dx1.float(), 0.0)


def off_is_baseline(b=2, t=256):
    """ck_w = 0 must reproduce the pre-CK kernels BIT-EXACTLY (the flag-off guarantee, at kernel level)."""
    print("\n=== ck_w=0 vs the pre-CK kernels (train3_canon.py) ===", flush=True)
    base = os.path.join(here, "..", "submission_raw", "train3_canon.py")
    bsrc = open(base).read()
    i2 = bsrc.index("    def pre_attn_fwd_ko("); i2 = bsrc.rindex("    @nki.jit", 0, i2)
    k2 = bsrc.index("    def pre_attn_bwd_ko("); j2 = bsrc.index("\n    @nki.jit", k2 + 20)
    blk = "\n".join(l[4:] if l.startswith("    ") else l for l in bsrc[i2:j2].split("\n"))
    bp = os.path.join(_dir, "ck_kernels_base.py")
    open(bp, "w").write("import nki, nki.isa as nisa, nki.language as nl\n_NKI_P = 128\n_FMAX = 512\nROPE_FRAC = 0.5\n" + blk)
    sp = importlib.util.spec_from_file_location("ck_kernels_base", bp)
    md = importlib.util.module_from_spec(sp)
    sys.modules["ck_kernels_base"] = md
    sp.loader.exec_module(md)
    T = b * t
    torch.manual_seed(1)
    x = torch.randn(T, D).bfloat16(); x0 = torch.randn(T, D).bfloat16(); ve = torch.randn(T, D).bfloat16()
    wq, wk, wv = [(torch.randn(D, D) * D ** -0.5).bfloat16() for _ in range(3)]
    ang = torch.outer(torch.arange(t).float(), 1.0 / (10000 ** (torch.arange(half).float() / half)))
    cos = torch.cos(ang).repeat(b, 1).contiguous(); sin = torch.sin(ang).repeat(b, 1).contiguous()
    scal4 = torch.tensor([0.9, 0.1, 0.5, 0.0]).view(1, 4).expand(128, 4).contiguous()
    dcol = doc_starts(b, t).reshape(T, 1).contiguous()
    a = [cvt(o) for o in run(md.pre_attn_fwd_ko, x, x0, ve, wq, wk, wv, cos, sin, scal4, dcol,
                             n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ)]
    c = [cvt(o) for o in run(fwd_k, x, x0, ve, wq, wk, wv, cos, sin, scal4, dcol,
                             n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HQ, ck_w=0)]
    for nm in range(6):
        check("fwd " + ("q", "k", "v", "xm", "rq", "rk")[nm], c[nm], a[nm], 0.0)
    dq = (torch.randn(b * NH, t, HD) * 0.05).bfloat16(); dk = (torch.randn(b * NH, t, HD) * 0.05).bfloat16()
    dv = (torch.randn(b * NH, t, HD) * 0.05).bfloat16()
    rq = torch.randn(NH, T, HD).bfloat16(); rk = torch.randn(NH, T, HD).bfloat16()
    xm = torch.randn(T, D).bfloat16(); dx1 = (torch.randn(T, D) * 0.1).bfloat16()
    a = [cvt(o) for o in run(md.pre_attn_bwd_ko, dq, dk, dv, rq, rk, xm, dx1, cos, sin, dcol, wq, wk, wv,
                             n_head=NH, eps=eps, scale=scale, hq=HQ)]
    c = [cvt(o) for o in run(bwd_k, dq, dk, dv, rq, rk, xm, dx1, cos, sin, dcol, wq, wk, wv, x, scal4,
                             n_head=NH, eps=eps, scale=scale, hq=HQ, ck_w=0)]
    for nm in range(5):
        check("bwd " + ("dxm", "gv2", "dWq", "dWk", "dWv")[nm], c[nm], a[nm], 0.0)


def algebra(b=2, t=256):
    """the torch-side algebra the op body relies on, straight out of autograd (no kernels involved):
       dx = resid * A^T g ; d_resid = sum(A^T g * x) ; dw = resid * dw_raw(g) ; and the dscal row-0 routing."""
    print("\n=== op-side algebra (autograd on the vehicle's own canon_conv) ===", flush=True)
    torch.manual_seed(2)
    ds = doc_starts(b, t).bool()
    x = torch.randn(b, t, D)
    x0 = torch.randn(b, t, D)
    w = torch.randn(W, D) * 0.2
    w[0] += 1.0
    g = torch.randn(b, t, D) * 0.1
    rl = torch.tensor(0.9, requires_grad=True)
    xl = torch.tensor(0.1, requires_grad=True)
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    (rl * canon_conv(xr, wr, ds) + xl * x0).backward(g)          # what the layer really computes
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    canon_conv(xa, wa, ds).backward(g)                            # A^T g and dw_raw, as the kernel returns them
    check("dx = r*A^T g", xr.grad, 0.9 * xa.grad, 1e-5)
    check("dw = r*dw_raw", wr.grad, 0.9 * wa.grad, 1e-4)
    check("d_resid", rl.grad.reshape(1), (xa.grad * x).sum().reshape(1), 2e-3)
    # dscal row 0 -> canon_p: the expand/cat/reshape path the op's backward relies on
    cp = w.clone().requires_grad_(True)
    srow = torch.cat([torch.tensor([0.9, 0.1, 0.5, 0.0]).view(1, 4), cp.reshape(1, W * D)], dim=1)
    scal = srow.expand(128, 4 + W * D).contiguous()
    dck = torch.randn(1, W * D)
    dscal = torch.cat([torch.cat([torch.zeros(1, 4), dck * 0.9], dim=1), torch.zeros(127, 4 + W * D)], dim=0)
    scal.backward(dscal)
    check("dscal -> canon_p", cp.grad, (dck * 0.9).view(W, D), 0.0)


if "prod" in sys.argv:             # the real training shape: 8 tiles per sequence row, 128 tiles, groups of 4
    one_shape(16, 1024, True)
    one_shape(16, 1024, False)
else:
    off_is_baseline()
    algebra()
    for _exact in (True, False):
        one_shape(2, 256, _exact)  # tiles = 4  -> dck reduced in groups of 4
    one_shape(2, 384, True)        # tiles = 6  -> groups of 1 (uniform matmul count per PSUM tile, rule #12)
print(f"\n{'ALL PASS' if ok else 'FAILURES'} (W={W}, {dev})")
sys.exit(0 if ok else 1)
