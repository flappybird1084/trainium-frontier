#!/usr/bin/env python3
"""CAUTIOUS_MUON: simulator check of muon_cautious_kernel, extracted VERBATIM from submission_raw/train3_canon.py next to
the original muon_kernel, for the three stack shapes (512x512, tall 2048x512, wide 512x2048).
Reference: the ORIGINAL kernel run with wd = 0 exposes the update u = (w0 - w1) / lr exactly as the kernel computes it;
(bar a handful of coordinates whose update is below w's fp32 ulp); the cautious reference is then  w0 - lr * u * (u*g > 0) * numel/(kept+1) - lr*wd * w0 * (u*w0 >= 0).
Also checks that momentum / second moment are identical to the original kernel (states stay unmasked), and compares
against the vehicle's torch math (independent bf16 Newton-Schulz) as a sanity bound.
Run from outside the repo root:  cd /tmp && <repo>/.venv-nki/bin/python <repo>/nki/nki_muon_cautious_test.py"""
import os, re, sys, textwrap, tempfile, importlib.util
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(REPO, "submission_raw", "train3_canon.py")).read()


def extract(name):
    m = re.search(r"\n(    @nki\.jit\n    def " + name + r"\(.*?)(?=\n    @nki\.jit|\n    adamw_step_nki)", src, re.S)
    return textwrap.dedent(m.group(1))


coeffs = re.search(r"\nPOLAR_EXPRESS_COEFFS = \[.*?\n\]\n", src, re.S).group(0)
mod = ("import nki\nimport nki.isa as nisa\nimport nki.language as nl\nP = 128\nFMAX = 512\n" + coeffs + "\n"
       + extract("muon_kernel") + "\n\n" + extract("muon_cautious_kernel") + "\n")
path = os.path.join(tempfile.mkdtemp(), "cm_mod.py")
open(path, "w").write(mod)
spec = importlib.util.spec_from_file_location("cm_mod", path)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import nki

P = 128
ok = True
for n, o, i_ in [(1, 512, 512), (1, 2048, 512), (1, 512, 2048)]:
    torch.manual_seed(o + 3 * i_)
    w = torch.randn(n, o, i_) * 0.05
    g = torch.randn(n, o, i_) * 0.02
    mom = torch.randn(n, o, i_) * 0.01
    sec = torch.rand((n, o, 1) if o >= i_ else (n, 1, i_)) * 1e-3
    momentum, lr, wd = 0.95, 0.02, 0.1
    lr_eff = lr * (max(1.0, o / i_) ** 0.5)
    sc = lambda wdv: torch.tensor([momentum, lr_eff, lr_eff * wdv, 0.0]).expand(P, 4).contiguous()
    run = lambda kern, wdv: [t.clone() for t in (w, g, mom, sec)] + [sc(wdv)]
    a = run(K.muon_kernel, 0.0); nki.simulate(K.muon_kernel)(*a[:4], a[4])
    u = (w - a[0]) / lr_eff                                   # the kernel's own update (orth * NorMuon factor)
    c = run(K.muon_cautious_kernel, wd); nki.simulate(K.muon_cautious_kernel)(*c[:4], c[4])
    b = run(K.muon_kernel, wd); nki.simulate(K.muon_kernel)(*b[:4], b[4])
    cm = (u * g) > 0
    kept = cm.sum(dim=(-2, -1), keepdim=True).float()
    ref = w - lr_eff * u * cm * (o * i_ / (kept + 1.0)) - lr_eff * wd * w * ((u * w) >= 0)
    d = (c[0] - w).abs().max().item()
    # where lr*u is below w's fp32 ulp, w0 - w1 rounds to 0 and u is not recoverable (its sign drives the WD mask):
    # those few coordinates are excluded from the exact comparison and counted
    rec = u != 0
    err = (c[0] - ref)[rec].abs().max().item()
    # coordinates where the kernel's mask disagrees with the reference mask (only possible at |u*g| ~ 0)
    moved = (c[0] - w).abs() > (lr_eff * wd * w).abs() * 1.5 + 1e-9
    states_same = torch.equal(c[2], b[2]) and torch.equal(c[3], b[3])
    # sanity vs the vehicle's torch math (independent bf16 NS): relative error of the cautious update
    tw, tg, tm, ts = (t.clone() for t in (w, g, mom, sec))
    g_raw = tg.clone(); tm.lerp_(tg, 1 - momentum); gg = tg.lerp_(tm, momentum)
    x = gg.bfloat16(); x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for (ca, cb, cc) in K.POLAR_EXPRESS_COEFFS:
        if o > i_:
            xtx = x.mT @ x; x = ca * x + x @ (cb * xtx + cc * (xtx @ xtx))
        else:
            xxt = x @ x.mT; x = ca * x + (cb * xxt + cc * (xxt @ xxt)) @ x
    red = -1 if o >= i_ else -2
    vm = x.float().square().mean(dim=red, keepdim=True); rs = x.size(red)
    vn = (vm.sum(dim=(-2, -1), keepdim=True) * rs).sqrt(); ts.lerp_(vm, 0.1)
    st = ts.clamp_min(1e-10).rsqrt(); vnn = ((vm * rs) * st.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    ug = x * (st * (vn / vnn.clamp_min(1e-10))).to(x.dtype)
    wmask = (ug * tw) >= 0
    tcm = (x.float() * g_raw) > 0
    ug = ug * tcm * (tcm[0].numel() / (tcm.sum(dim=(-2, -1), keepdim=True).float() + 1.0))
    tref = tw - lr_eff * ug - lr_eff * wd * tw * wmask
    terr = ((c[0] - tref).norm() / (tref - w).norm()).item()
    mask_agree = (tcm == cm).float().mean().item()
    frac = cm.float().mean().item()
    good = err < 1e-6 and int((~rec).sum()) < 10 and states_same and terr < 0.05
    ok &= good
    print(f"({n},{o},{i_}): kept {frac:.3f}  |dw| {d:.2e}  kernel-vs-exact-ref max err {err:.2e} (unrecoverable coords {int((~rec).sum())})  "
          f"states == original kernel: {states_same}  vs torch path: rel err {terr:.2e}, mask agreement {mask_agree:.5f}"
          f"  -> {'OK' if good else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
