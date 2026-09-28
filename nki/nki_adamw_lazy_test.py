#!/usr/bin/env python3
"""LAZY_BTE: simulator check of adamw_lazy_kernel, extracted VERBATIM from submission_raw/train3_canon.py (so the text
tested is the text shipped), against a numpy reference of lazy AdamW; also re-checks that on an all-rows-touched
gradient it matches the vehicle's dense adamw_kernel bit for bit.
Run from outside the repo root:  cd /tmp && <repo>/.venv-nki/bin/python <repo>/nki/nki_adamw_lazy_test.py"""
import os, re, sys, textwrap, tempfile, importlib.util
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(REPO, "submission_raw", "train3_canon.py")).read()


def extract(name):
    m = re.search(r"\n(    @nki\.jit\n    def " + name + r"\(.*?)(?=\n    @nki\.jit|\n    adamw_step_nki)", src, re.S)
    return textwrap.dedent(m.group(1))


mod = "import nki\nimport nki.isa as nisa\nimport nki.language as nl\nP = 128\n\n" + extract("adamw_lazy_kernel") + "\n\n" + extract("adamw_kernel") + "\n"
path = os.path.join(tempfile.mkdtemp(), "lazy_mod.py")
open(path, "w").write(mod)
spec = importlib.util.spec_from_file_location("lazy_mod", path)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import nki

P = 128
b1, b2, eps = 0.8, 0.99, 1e-10


def ref_lazy(p, g, m, v, lr, ib1, ib2, lrwd):
    mk = (g != 0).any(axis=1, keepdims=True).astype(np.float32)
    p2 = p * (1 - lrwd * mk)
    m2 = m + (1 - b1) * mk * (g - m)
    v2 = v + (1 - b2) * mk * (g * g - v)
    return p2 - lr * mk * (m2 * ib1) / (np.sqrt(v2) * ib2 + eps), m2, v2


rng = np.random.default_rng(0)
R, C = 1024, 512
p = rng.standard_normal((R, C)).astype(np.float32)
m = (rng.standard_normal((R, C)) * 0.01).astype(np.float32)
v = (rng.random((R, C)) * 1e-4).astype(np.float32)
g = (rng.standard_normal((R, C)) * 0.01).astype(np.float32)
touched = rng.random(R) < 0.24
g[~touched] = 0.0
g[5] = 0.0; g[5, 7] = -3e-12          # one small negative element: must count as touched
g[6] = 0.0; g[6, :] = 2e-3            # a constant positive row
g[7] = 0.0; g[7, :] = -2e-3           # a constant negative row
g[8] = -0.0                            # negative zeros: untouched
touched[5:8] = True; touched[8] = False
v[:3] = 0.0; m[:3] = 0.0; g[:3] = 0.0; touched[:3] = False   # never-touched rows (v = 0: den = eps, m = 0)
lr, step, wd = 0.05, 37, 0.001
scal = np.array([lr, 1 / (1 - b1 ** step), 1 / np.sqrt(1 - b2 ** step), lr * wd], dtype=np.float32)
st = torch.tensor(scal).expand(P, 4).contiguous()

tp, tg, tm, tv = (torch.tensor(a.copy()) for a in (p, g, m, v))
nki.simulate(K.adamw_lazy_kernel)(tp, tg, tm, tv, st, beta1=b1, beta2=b2, eps=eps)
rp, rm, rv = ref_lazy(p, g, m, v, *scal)
un = ~touched
ok_exact = all(np.array_equal(a.numpy()[un].view(np.uint32), b[un].view(np.uint32)) for a, b in ((tp, p), (tm, m), (tv, v)))
e = lambda a, b: float(np.abs(a.numpy() - b).max())
print(f"untouched rows bit-exact (p, m, v): {ok_exact}  [{un.sum()} rows]")
print(f"touched rows vs numpy: p {e(tp, rp):.2e}  m {e(tm, rm):.2e}  v {e(tv, rv):.2e}   |dp| {np.abs(rp - p).max():.2e}")
print(f"rows 5-7 (tiny / constant rows) moved: {[bool((tp.numpy()[r] != p[r]).any()) for r in (5, 6, 7)]}")

# all rows touched: lazy must equal the dense kernel bit for bit
g2 = (rng.standard_normal((R, C)) * 0.01).astype(np.float32)
a = [torch.tensor(x.copy()) for x in (p, g2, m, v)]
b = [torch.tensor(x.copy()) for x in (p, g2, m, v)]
nki.simulate(K.adamw_lazy_kernel)(*a, st, beta1=b1, beta2=b2, eps=eps)
nki.simulate(K.adamw_kernel)(*b, st, beta1=b1, beta2=b2, eps=eps)
same = all(torch.equal(x.view(torch.int32), y.view(torch.int32)) for x, y in zip(a, b))
print(f"all-rows-touched: lazy == dense kernel bit-exact (p, m, v): {same}")
if not (ok_exact and same):
    sys.exit(1)
