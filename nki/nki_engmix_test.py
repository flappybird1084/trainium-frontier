#!/usr/bin/env python3
"""Device experiment (ON the box, single core): WHY does nki_coschedule_test only recover 19% of the "available"
overlap, and can Vector work be moved to the idle Scalar/GpSimd engines inside a real kernel?

THE FLAW I AM TESTING FOR. nki_coschedule_test calls body A "PE-dominated" and body B "Vector/Scalar, no PE", and
reads 2.83 ms against an ideal of max(A, B) = 1.62 ms. But counting free-elements per engine from the source says
BOTH bodies are VECTOR-bound:

  A (mlp_body), per 512-token chunk, on Vector:  16 xT drains x128  +  24 `pre` drains x512
                                               +  24 scalar_tensor_tensor x512  +  4 yt drains x512  = 28,672
      x 32 chunks = 917,504 free-elements. A measured 1.47 ms -> 1.60 ns per free-element.
  B (norm_body), per 128-row tile, on Vector:    8 passes x (tensor_tensor 512 + tensor_reduce 512) = 8,192
      x 128 tiles = 1,048,576 free-elements -> 1.55 ns at B's measured 1.62 ms.

Two independent bodies, one shared calibration to within 4%. So the honest ideal for the fused kernel is not
max(A, B) = 1.62 ms, it is the VECTOR sum = 1.42 + 1.55 = ~2.97 ms, and the measured 2.83 ms is AT that bound.
If that is right, the scheduler is not leaving 82% on the table - the two bodies were never on disjoint engines,
and the 19% figure is an artifact of the wrong denominator.

WHAT THIS SCRIPT MEASURES, in order of value:
  1. A0 vs A1: mlp_body with its PSUM->SBUF drains pinned to the SCALAR engine instead of the compiler's default.
     Scalar is otherwise IDLE inside mlp_body. This moves 16,384 of 28,672 Vector free-elements per chunk (57%)
     and is the exact change staged in submission_raw/train3_eng.py as ENG2. It prices that change at the real
     production shape in one minute instead of a 12-minute training screen.
  2. B0 vs B1: the same rmsnorm chain rewritten with ZERO Vector ops (activation_reduce + rsqrt on Scalar, the
     scale and the copy on GpSimd). B1 is what "disjoint from A" actually looks like.
  3. A0+B0 (reproduces the 2.83 ms) against A0+B1. If A0+B1 lands near max(A0, B1) while A0+B0 lands at the
     Vector sum, then co-scheduling works fine and the lever is ENGINE PLACEMENT, not source interleaving.
  4. Source order: unrolled-concatenated against unrolled-interleaved, at the same unrolling, so the comparison
     isolates interleaving. Run for both B0 (shared engine) and B1 (disjoint).

Usage on the box:  bash trn2_job.sh ENGMIX <python> nki/nki_engmix_test.py
                   R1_NORM_PASSES=8 R1_SKIP_ILV=1 ... to trim the compile list
"""
import os, sys, time, traceback
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
import nki, nki.language as nl, nki.isa as nisa
from torch_neuronx.nki_hop import wrap_nki

P, FMAX = 128, 512
dev = torch.device("neuron"); torch.manual_seed(0)
TT, D, H = 16384, 512, 3072                      # production MLP shape (one layer, one rank)
NR, ND = 16384, 512
PASSES = int(os.environ.get("R1_NORM_PASSES", "8"))
SKIP_ILV = os.environ.get("R1_SKIP_ILV", "0") != "0"


# ---------------------------------------------------------------------------- drain helpers (trace-time dispatch)
def _drain(dst, src, eng: int):
    """PSUM -> SBUF. eng 0 = compiler default (it picks Vector), 1 = Scalar via the engine hint,
    2 = Scalar via nisa.activation(op=nl.copy) in case the hint is rejected on this NeuronCore revision."""
    if eng == 1:
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.scalar)
    elif eng == 2:
        nisa.activation(dst=dst, op=nl.copy, data=src)
    else:
        nisa.tensor_copy(dst=dst, src=src)


# ---------------------------------------------------------------------------- body A, split into per-chunk units
def mlp_weights(w1, w2, eng: int):
    Dm = w1.shape[1]; Hm = w1.shape[0]; kt = Dm // P; ht = Hm // P
    W1 = [nl.ndarray((P, Hm), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(kt)]
    for h in range(ht):
        blk = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w1[h * P:(h + 1) * P, :])
        for k in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, k * P:(k + 1) * P])
            _drain(W1[k][:, h * P:(h + 1) * P], tps, eng)
    W2 = [nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(ht)]
    for o in range(kt):
        blk = nl.ndarray((P, Hm), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w2[o * P:(o + 1) * P, :])
        for h in range(ht):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, h * P:(h + 1) * P])
            _drain(W2[h][:, o * P:(o + 1) * P], tps, eng)
    return W1, W2


def mlp_chunk(c, x_in, W1, W2, y_out, pre_out, eng: int):
    """One 512-token chunk of mlp_fwd. Identical math to nki_coschedule_test.mlp_body's loop body."""
    T, Dm = x_in.shape; Hm = pre_out.shape[0]; kt = Dm // P; ht = Hm // P
    t0 = c * FMAX
    xT = [nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(kt)]
    for tb in range(FMAX // P):
        xt = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
        for k in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=xt[:, k * P:(k + 1) * P])
            _drain(xT[k][:, tb * P:(tb + 1) * P], tps, eng)
    hT = [nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(ht)]
    for h in range(ht):
        pps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
        for k in range(kt):
            nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * P:(h + 1) * P], moving=xT[k], accumulate=(k > 0))
        pre = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        _drain(pre, pps, eng)
        nisa.dma_copy(dst=pre_out[h * P:(h + 1) * P, t0:t0 + FMAX], src=pre)
        nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)
    for tb in range(FMAX // P):
        yps = nl.ndarray((P, Dm), dtype=nl.float32, buffer=nl.psum)
        for h in range(ht):
            nisa.nc_matmul(dst=yps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=W2[h], accumulate=(h > 0))
        yt = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
        _drain(yt, yps, eng)
        nisa.dma_copy(dst=y_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=yt)


# ---------------------------------------------------------------------------- body B, split into per-tile units
def norm_tile_vec(i, z_in, z_out, passes):
    """B0: the original. tensor_tensor + tensor_reduce are both VECTOR -> 2*d free-elements of Vector per pass."""
    rows, d = z_in.shape
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


def norm_tile_sc(i, z_in, z_out, passes):
    """B1: the SAME rmsnorm chain with ZERO Vector instructions. square+rowsum fuse into one Scalar
    activation_reduce (this is what the live post_attn_fwd_h already does), rsqrt is Scalar, and the two scaling
    ops plus the evacuating copy are pinned to GpSimd. Numerically equivalent, not bit-identical."""
    rows, d = z_in.shape
    x = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=z_in[i * P:(i + 1) * P, :])
    sq = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
    ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    for _p in range(passes):
        nisa.activation_reduce(dst=sq, op=nl.square, data=x, reduce_op=nl.add, reduce_res=ms)     # Scalar
        nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=1e-6,
                           engine=nisa.engine.gpsimd)                                            # GpSimd
        nisa.activation(dst=ms, op=nl.rsqrt, data=ms)                                            # Scalar
        nisa.tensor_scalar(dst=x, data=x, op0=nl.multiply, operand0=ms, engine=nisa.engine.gpsimd)  # GpSimd
    y = nl.ndarray((P, d), dtype=z_out.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=y, src=x, engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=z_out[i * P:(i + 1) * P, :], src=y)


_NTILE = {0: norm_tile_vec, 1: norm_tile_sc}


# ---------------------------------------------------------------------------- kernels
@nki.jit
def a_only(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, eng: int = 0):
    T, Dm = x_in.shape; Hm = w1.shape[0]
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    W1, W2 = mlp_weights(w1, w2, eng)
    for c in range(T // FMAX):
        mlp_chunk(c, x_in, W1, W2, y_out, pre_out, eng)
    return y_out, pre_out


@nki.jit
def b_only(z_in: nl.ndarray, passes: int = 8, form: int = 0):
    rows, d = z_in.shape
    z_out = nl.ndarray((rows, d), dtype=z_in.dtype, buffer=nl.shared_hbm)
    for i in range(rows // P):
        _NTILE[form](i, z_in, z_out, passes)
    return z_out


@nki.jit
def ab_cat(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, z_in: nl.ndarray, passes: int = 8,
           eng: int = 0, form: int = 0):
    """Unrolled, CONCATENATED: all of A, then all of B. Same unrolling as ab_ilv, so the pair isolates order."""
    T, Dm = x_in.shape; Hm = w1.shape[0]; rows, d = z_in.shape
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    z_out = nl.ndarray((rows, d), dtype=z_in.dtype, buffer=nl.shared_hbm)
    W1, W2 = mlp_weights(w1, w2, eng)
    for c in range(T // FMAX):
        mlp_chunk(c, x_in, W1, W2, y_out, pre_out, eng)
    for i in range(rows // P):
        _NTILE[form](i, z_in, z_out, passes)
    return y_out, pre_out, z_out


@nki.jit
def ab_ilv(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, z_in: nl.ndarray, passes: int = 8,
           eng: int = 0, form: int = 0):
    """Unrolled, INTERLEAVED in source: one A chunk, then the B tiles that belong beside it. Identical work and
    identical unrolling to ab_cat; the only difference is instruction ORDER in the traced stream."""
    T, Dm = x_in.shape; Hm = w1.shape[0]; rows, d = z_in.shape
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    z_out = nl.ndarray((rows, d), dtype=z_in.dtype, buffer=nl.shared_hbm)
    W1, W2 = mlp_weights(w1, w2, eng)
    chunks = T // FMAX; tiles = rows // P
    per = tiles // chunks
    for c in range(chunks):
        mlp_chunk(c, x_in, W1, W2, y_out, pre_out, eng)
        for j in range(per):
            _NTILE[form](c * per + j, z_in, z_out, passes)
    for i in range(chunks * per, tiles):
        _NTILE[form](i, z_in, z_out, passes)
    return y_out, pre_out, z_out


# ---------------------------------------------------------------------------- driver
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
ka, kb, kcat, kilv = wrap_nki(a_only), wrap_nki(b_only), wrap_nki(ab_cat), wrap_nki(ab_ilv)

# free-element budgets computed from the source, so the "ideal" uses the right denominator
kt, ht, chunks, tiles = D // P, H // P, TT // FMAX, NR // P
A_VEC = chunks * (16 * P + ht * FMAX + ht * FMAX + (FMAX // P) * D)      # drains + scalar_tensor_tensor
A_VEC_MOVABLE = chunks * (16 * P + ht * FMAX + (FMAX // P) * D)          # everything except scalar_tensor_tensor
B0_VEC = tiles * PASSES * 2 * ND
print(f"SOURCE BUDGET (free-elements on a [128, F] instruction, F summed):", flush=True)
print(f"  A mlp_body   Vector {A_VEC:,}   of which movable to Scalar {A_VEC_MOVABLE:,} "
      f"({100.0 * A_VEC_MOVABLE / A_VEC:.0f}%)", flush=True)
print(f"  B0 norm_vec  Vector {B0_VEC:,}    B1 norm_sc  Vector 0", flush=True)
print(f"  => the honest ideal for A+B0 is the VECTOR SUM, not max(A, B0).\n", flush=True)

res = {}


def run(label, fn, key=None):
    try:
        t = time_it(fn)
        res[key or label] = t
        print(f"{label:52s}: {t:6.2f} ms", flush=True)
        return t
    except Exception:
        print(f"{label:52s}: FAILED", flush=True); traceback.print_exc(); return None


fa0 = lambda: ka[grid](x, w1, w2, eng=0)
fa1 = lambda: ka[grid](x, w1, w2, eng=1)
fa2 = lambda: ka[grid](x, w1, w2, eng=2)
fb0 = lambda: kb[grid](z, passes=PASSES, form=0)
fb1 = lambda: kb[grid](z, passes=PASSES, form=1)

print("=== 1. ENGINE PLACEMENT inside one real kernel (this is the ENG2 change, priced) ===", flush=True)
ta0 = run("(A0) mlp_body, drains on the compiler's engine", fa0, "A0")
ta1 = run("(A1) mlp_body, drains pinned to SCALAR (hint)", fa1, "A1")
ta2 = run("(A2) mlp_body, drains pinned to SCALAR (activation)", fa2, "A2")
if ta0:
    y0, p0 = fa0(); torch_neuronx.synchronize()
    for nm, f in (("A1", fa1), ("A2", fa2)):
        try:
            y, p = f(); torch_neuronx.synchronize()
            print(f"     {nm} vs A0 bit-exactness: y max|diff| {(y.float() - y0.float()).abs().max().item():.3e} "
                  f"pre {(p.float() - p0.float()).abs().max().item():.3e}", flush=True)
        except Exception:
            print(f"     {nm} correctness check skipped (kernel failed)", flush=True)
    best = min(t for t in (ta0, ta1, ta2) if t)
    print(f"  --> best drain placement is {100.0 * (ta0 - best) / ta0:.1f}% faster than the default on the "
          f"production MLP forward shape.", flush=True)

print("\n=== 2. A TRULY DISJOINT body B ===", flush=True)
tb0 = run(f"(B0) {PASSES} rmsnorm passes, Vector+Scalar", fb0, "B0")
tb1 = run(f"(B1) {PASSES} rmsnorm passes, Scalar+GpSimd, NO Vector", fb1, "B1")

print("\n=== 3. CO-SCHEDULING: shared engine vs disjoint engines ===", flush=True)
for form, tb in ((0, tb0), (1, tb1)):
    f = lambda: kcat[grid](x, w1, w2, z, passes=PASSES, eng=0, form=form)
    t = run(f"(C{form}) A0 + B{form}, concatenated", f, f"C{form}")
    if t and ta0 and tb:
        ideal_max = max(ta0, tb)
        ideal_vec = ta0 + tb if form == 0 else max(ta0, tb)   # B1 adds no Vector, so its bound is max
        print(f"     serial {ta0 + tb:5.2f}  max(A,B) {ideal_max:5.2f}  Vector-aware ideal {ideal_vec:5.2f}  "
              f"-> recovered {100.0 * (ta0 + tb - t) / max(ta0 + tb - ideal_max, 1e-9):4.0f}% of max(A,B) "
              f"and {100.0 * (ta0 + tb - t) / max(ta0 + tb - ideal_vec, 1e-9):4.0f}% of the Vector-aware ideal",
                  flush=True)

if not SKIP_ILV:
    print("\n=== 4. SOURCE ORDER at identical unrolling: concatenated vs interleaved ===", flush=True)
    for form in (0, 1):
        fi = lambda: kilv[grid](x, w1, w2, z, passes=PASSES, eng=0, form=form)
        ti = run(f"(I{form}) A0 + B{form}, INTERLEAVED in source", fi, f"I{form}")
        tc = res.get(f"C{form}")
        if ti and tc:
            print(f"     interleave vs concatenate at the same unrolling: "
                  f"{100.0 * (tc - ti) / tc:+.1f}%  ({tc:.2f} -> {ti:.2f} ms)", flush=True)

print("\nREADING: if (1) shows a real speedup, engine placement is a live lever and ENG2 in train3_eng.py is "
      "worth a gate. If (3) shows C1 near max(A0, B1) while C0 sits at the Vector sum, the scheduler co-schedules "
      "fine and the 19% figure in nki_coschedule_test used max(A,B) as an ideal that the engine mix forbade. "
      "If (4) moves, source order is a second, independent lever.", flush=True)
