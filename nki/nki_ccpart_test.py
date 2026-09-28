#!/usr/bin/env python3
"""Device experiment (ON the box, single core): IS THE CUSTOM-OP BOUNDARY A SCHEDULING BARRIER?

MEASURED 2026-09-28, production shape, A = mlp_body, B = 8 rmsnorm passes, both on independent data:
    A alone 1.47 ms   B alone 1.62 ms   serial sum 3.09   ideal max(A,B) 1.62
    eager, two separate NEFFs (two device executions)   3.11 ms  =   -2% of ideal overlap (fully serial)
    ONE hand-fused NKI kernel holding both bodies       2.83 ms  =   17% of ideal overlap
    torch.compile(backend="neuron"), ONE graph, TWO
      separate NKI custom calls                         2.82 ms  =   18% of ideal overlap
  So inside a compiled graph the custom-call boundary is NOT a barrier: two independent NKI ops overlap exactly
  as much as if you hand-fused them into one kernel. Hand-fusing kernels buys ZERO overlap over letting
  torch.compile put them in one graph. Separate *executions* (eager, one NEFF each) do serialise completely.

Original header: does ANY neuronx-cc flag or runtime env var change how much
independent work the toolchain overlaps inside one NKI kernel?

BASELINE TO MOVE (nki_coschedule_test.py, 2026-09-27): A = mlp_body 1.47 ms, B = 8 rmsnorm passes 1.62 ms,
serial sum 3.09 ms, fused 2.83 ms -> 19% of the ideal max(A,B) overlap is recovered, 82% unclaimed.

HOW THE FLAG GETS THERE (read from the installed stack, not recollection):
  torch_neuron_eager/torch_neuronx/kernels/neuronx_cc_wrapper.py:126-134 prepends $NEURON_CC_FLAGS to the
  neuronx-cc argv, and compiler_subprocess.py:42-56 puts `*sorted(config.extra_flags)` INTO the NEFF cache key,
  so every flag string automatically invalidates the cache and forces a real recompile. No stale-NEFF trap.
  The framework's own default (compiler_config.py) is `--model-type transformer -O1 --auto-cast=none`, i.e. it
  DOWNGRADES from the neuronx-cc default of -O2.

Usage (one variant per job; single token, commas become spaces):
    bash trn2_job.sh CCF_BASE  <python> nki/nki_ccflags_test.py
    R1_CC=-O2 bash trn2_job.sh CCF_O2 <python> nki/nki_ccflags_test.py
    R1_CC=--scheduler=none ...      # diagnostic: does the backend scheduler touch NKI code AT ALL?
    R1_CC=--internal-interleave-kind=pe_cycle ...
    R1_CC=BADFLAG ...               # positive control: must FAIL, proving the channel is live
"""
import os, sys, time, traceback

# Must happen BEFORE torch/nki import so every compiler subprocess inherits it.
_CC = os.environ.get("R1_CC", "")
if _CC == "BADFLAG":
    _CC = "--r1-no-such-flag"
if _CC:
    os.environ["NEURON_CC_FLAGS"] = (os.environ.get("NEURON_CC_FLAGS", "") + " " + _CC.replace(",", " ")).strip()
print(f"R1_CC={_CC!r}  ->  NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS','')!r}", flush=True)
for _k in sorted(os.environ):
    if _k.startswith(("NEURON_RT", "NEURON_LOGICAL", "NEURON_CC", "NKI_")):
        print(f"  env {_k}={os.environ[_k]}", flush=True)

sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
import nki, nki.language as nl, nki.isa as nisa
from torch_neuronx.nki_hop import wrap_nki

P, FMAX = 128, 512
dev = torch.device("neuron"); torch.manual_seed(0)
TT, D, H = 16384, 512, 3072
NR, ND = 16384, 512
PASSES = int(os.environ.get("R1_NORM_PASSES", "8"))


def mlp_body(x_in, w1, w2, y_out, pre_out):
    """Verbatim from nki/nki_coschedule_test.py. PE-dominated."""
    T, Dm = x_in.shape; Hm = w1.shape[0]; kt = Dm // P; ht = Hm // P; chunks = T // FMAX
    W1 = []
    for _j in range(kt):
        W1.append(nl.ndarray((P, Hm), dtype=nl.bfloat16, buffer=nl.sbuf))
    for h in range(ht):
        blk = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w1[h * P:(h + 1) * P, :])
        for k in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, k * P:(k + 1) * P])
            nisa.tensor_copy(dst=W1[k][:, h * P:(h + 1) * P], src=tps)
    W2 = []
    for _j in range(ht):
        W2.append(nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf))
    for o in range(kt):
        blk = nl.ndarray((P, Hm), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w2[o * P:(o + 1) * P, :])
        for h in range(ht):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, h * P:(h + 1) * P])
            nisa.tensor_copy(dst=W2[h][:, o * P:(o + 1) * P], src=tps)
    for c in nl.affine_range(chunks):
        t0 = c * FMAX
        xT = []
        for _j in range(kt):
            xT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for tb in range(FMAX // P):
            xt = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            for k in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=xt[:, k * P:(k + 1) * P])
                nisa.tensor_copy(dst=xT[k][:, tb * P:(tb + 1) * P], src=tps)
        hT = []
        for _j in range(ht):
            hT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for h in range(ht):
            pps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
            for k in range(kt):
                nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * P:(h + 1) * P], moving=xT[k], accumulate=(k > 0))
            pre = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pre, src=pps)
            nisa.dma_copy(dst=pre_out[h * P:(h + 1) * P, t0:t0 + FMAX], src=pre)
            nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)
        for tb in range(FMAX // P):
            yps = nl.ndarray((P, Dm), dtype=nl.float32, buffer=nl.psum)
            for h in range(ht):
                nisa.nc_matmul(dst=yps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=W2[h], accumulate=(h > 0))
            yt = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=yt, src=yps)
            nisa.dma_copy(dst=y_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=yt)


def norm_body(z_in, z_out, passes):
    """Verbatim from nki/nki_coschedule_test.py. No matmul, no transpose."""
    rows, d = z_in.shape
    tiles = rows // P
    for i in nl.affine_range(tiles):
        x = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x, src=z_in[i * P:(i + 1) * P, :])
        sq = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        for _p in range(passes):
            nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=1e-6)
            nisa.activation(dst=ms, op=nl.rsqrt, data=ms)
            nisa.tensor_scalar(dst=x, data=x, op0=nl.multiply, operand0=ms)
        y = nl.ndarray((P, d), dtype=z_out.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y, src=x)
        nisa.dma_copy(dst=z_out[i * P:(i + 1) * P, :], src=y)


@nki.jit
def a_only(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
    T, Dm = x_in.shape; Hm = w1.shape[0]
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    mlp_body(x_in, w1, w2, y_out, pre_out)
    return y_out, pre_out


@nki.jit
def b_only(z_in: nl.ndarray, passes: int = 8):
    rows, d = z_in.shape
    z_out = nl.ndarray((rows, d), dtype=z_in.dtype, buffer=nl.shared_hbm)
    norm_body(z_in, z_out, passes)
    return z_out


@nki.jit
def ab_fused(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, z_in: nl.ndarray, passes: int = 8):
    T, Dm = x_in.shape; Hm = w1.shape[0]; rows, d = z_in.shape
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    z_out = nl.ndarray((rows, d), dtype=z_in.dtype, buffer=nl.shared_hbm)
    mlp_body(x_in, w1, w2, y_out, pre_out)
    norm_body(z_in, z_out, passes)
    return y_out, pre_out, z_out


def time_it(fn, n=20):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n):
        fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n * 1000


x = (torch.randn(TT, D) * 0.5).bfloat16().to(dev)
w1 = (torch.randn(H, D) * 0.03).bfloat16().to(dev)
w2 = (torch.randn(D, H) * 0.02).bfloat16().to(dev)
z = (torch.randn(NR, ND) * 0.5).bfloat16().to(dev)
grid = (1,)
ka, kb, kab = wrap_nki(a_only), wrap_nki(b_only), wrap_nki(ab_fused)
fa = lambda: ka[grid](x, w1, w2)
fb = lambda: kb[grid](z, passes=PASSES)
fab = lambda: kab[grid](x, w1, w2, z, passes=PASSES)


def two_ops(xx, ww1, ww2, zz):
    """Two INDEPENDENT NKI ops in one python function. Under torch.compile(backend="neuron") this becomes ONE
    graph with two custom calls -> ONE NEFF. If neuronx-cc interleaves across the custom-call boundary, this
    lands near max(A,B); if each custom call is its own scheduling region, it lands at the serial sum."""
    y, pre = ka[grid](xx, ww1, ww2)
    zo = kb[grid](zz, passes=PASSES)
    return y, pre, zo


ta = time_it(fa); tb = time_it(fb); tab = time_it(fab)
t2e = time_it(lambda: (fa(), fb()))
print(f"(A) mlp_body alone, own NEFF      : {ta:6.2f} ms", flush=True)
print(f"(B) {PASSES} rmsnorm alone, own NEFF   : {tb:6.2f} ms", flush=True)
print(f"    serial sum {ta + tb:.2f}   ideal max(A,B) {max(ta, tb):.2f}", flush=True)
print(f"(D) EAGER: two separate NEFFs     : {t2e:6.2f} ms", flush=True)
print(f"(C) ONE kernel, two bodies        : {tab:6.2f} ms", flush=True)
try:
    c2 = torch.compile(two_ops, backend="neuron", dynamic=False)
    y3, _, z3 = c2(x, w1, w2, z); torch_neuronx.synchronize()
    t2c = time_it(lambda: c2(x, w1, w2, z))
    y1, _ = fa(); z1 = fb(); torch_neuronx.synchronize()
    e = lambda p, q: (p.float() - q.float()).abs().max().item()
    print(f"(E) torch.compile: ONE graph, two custom calls : {t2c:6.2f} ms  "
          f"maxdiff_mlp={e(y1, y3):.3e} maxdiff_norm={e(z1, z3):.3e}", flush=True)
    ideal, serial = max(ta, tb), ta + tb
    for nm, t in (("one-kernel-two-bodies", tab), ("eager-two-NEFFs", t2e), ("compiled-one-graph", t2c)):
        pct = (serial - t) / (serial - ideal) * 100.0
        print(f"RESULT {nm:24s} {t:6.2f} ms  = {pct:5.1f}% of ideal overlap", flush=True)
except Exception:
    print("torch.compile path FAILED:", flush=True); traceback.print_exc()
