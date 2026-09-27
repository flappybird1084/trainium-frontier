#!/usr/bin/env python3
"""Device experiment (ON the box, single core, ~5 min): does the NKI instruction scheduler EVER overlap two
independent regions whose ENGINE MIXES ARE DISJOINT?

WHY THIS EXISTS. PARRES_T (nki_parres_test.py, 2026-09-15) nested the vendor attention and our MLP in one kernel
on independent data and measured 3.77 ms against a 3.67 ms serial sum - no overlap. That single measurement is
what closes the whole "expose concurrency in the architecture" family, including parallel attn+MLP blocks. But it
has TWO confounds that leave the general claim unproven:
  (1) SBUF CAPACITY. The vendor attention claims its own SBUF region via init_sbuf_addr and our mlp_body holds
      ~6.2 MB of transposed weights plus per-chunk tiles. If both working sets cannot be live at once in 24 MB,
      the compiler MUST serialise, and the null says nothing about the scheduler.
  (2) ENGINE MIX. The vendor attention is itself well balanced (PROF: GpSimd 74% / Vector 55% / PE 50% / DMA 52%
      all busy at once inside it), so even a perfect scheduler had little disjoint capacity to exploit.

THIS TEST REMOVES BOTH. Body A is our mlp_body: PE-dominated. Body B is a chain of RMSNorm passes with ZERO
matmul and ZERO transpose - Vector/Scalar/GpSimd only, and a working set of a few 128x512 tiles (~0.25 MB), so
SBUF pressure cannot be the explanation for a null. The inputs are completely independent.

READ THE RESULT LIKE THIS:
  time(A+B in one kernel) ~ max(A, B)  -> the scheduler DOES overlap disjoint-engine regions. PARRES_T's null was
      about the vendor kernel / SBUF, and the concurrency axis REOPENS: an ideal schedule is worth ~24% of the
      step (~-0.0099 bpb), which would be the largest single item on the board.
  time(A+B in one kernel) ~ A + B      -> the scheduler does not interleave separate program regions AT ALL, under
      any engine mix, with SBUF to spare. The axis is closed for good and no block redesign can reclaim the idle
      engine time: parallel attn+MLP, MUDD-style branches and every "widen the DAG" variant are all priced at
      zero refund. STOP LOOKING HERE.
Both orders are timed, because "A then B" and "B then A" differing would mean the scheduler has a lookahead
window rather than no interleaving at all - a third, actionable outcome (shrink the regions and interleave by hand).

Usage on the box (single core, gated like any other job):
    python nki/nki_coschedule_test.py
    R1_NORM_PASSES=16 python nki/nki_coschedule_test.py     # retune body B's size so B ~ A without editing code
"""
import os, sys, time, traceback
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
import nki, nki.language as nl, nki.isa as nisa
from torch_neuronx.nki_hop import wrap_nki

P, FMAX = 128, 512
dev = torch.device("neuron"); torch.manual_seed(0)
TT, D, H = 16384, 512, 3072                      # production MLP shape (one layer, one rank)
NR, ND = 16384, 512                              # body B tensor: the residual stream's own shape
PASSES = int(os.environ.get("R1_NORM_PASSES", "8"))


def mlp_body(x_in, w1, w2, y_out, pre_out):
    """mlp_fwd's body (nki/nki_mlp.py), undecorated so it can run inside another kernel. PE-dominated."""
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
    """`passes` back-to-back RMSNorm sweeps over (rows, d). ZERO matmul, ZERO transpose -> no PE, no PSUM.
    Live SBUF is three 128xd fp32 tiles (~0.79 MB at d=512), so a serial result here cannot be blamed on SBUF.
    Pattern copied verbatim from the device-proven rmsnorm_fwd (nki/nki_rmsnorm.py)."""
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
    """A then B, one instruction stream, fully independent inputs and outputs."""
    T, Dm = x_in.shape; Hm = w1.shape[0]; rows, d = z_in.shape
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    z_out = nl.ndarray((rows, d), dtype=z_in.dtype, buffer=nl.shared_hbm)
    mlp_body(x_in, w1, w2, y_out, pre_out)
    norm_body(z_in, z_out, passes)
    return y_out, pre_out, z_out


@nki.jit
def ba_fused(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, z_in: nl.ndarray, passes: int = 8):
    """B then A: same two regions, opposite program order. A difference vs ab_fused means the scheduler has a
    finite lookahead window rather than no interleaving at all."""
    T, Dm = x_in.shape; Hm = w1.shape[0]; rows, d = z_in.shape
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    z_out = nl.ndarray((rows, d), dtype=z_in.dtype, buffer=nl.shared_hbm)
    norm_body(z_in, z_out, passes)
    mlp_body(x_in, w1, w2, y_out, pre_out)
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
ka, kb = wrap_nki(a_only), wrap_nki(b_only)
kab, kba = wrap_nki(ab_fused), wrap_nki(ba_fused)
fa = lambda: ka[grid](x, w1, w2)
fb = lambda: kb[grid](z, passes=PASSES)
ta = time_it(fa); print(f"(A) mlp_body alone   [PE-dominated]          : {ta:6.2f} ms", flush=True)
tb = time_it(fb); print(f"(B) {PASSES} rmsnorm passes [Vector/Scalar, no PE] : {tb:6.2f} ms", flush=True)
print(f"    serial sum A+B = {ta + tb:.2f} ms      ideal overlap max(A,B) = {max(ta, tb):.2f} ms", flush=True)
if tb < 0.25 * ta or tb > 4.0 * ta:
    print(f"    WARNING: B is {tb / ta:.2f}x A - overlap would be hard to see. Re-run with R1_NORM_PASSES "
          f"~ {max(1, int(round(PASSES * ta / max(tb, 1e-6))))}.", flush=True)
for nm, kf in (("A-then-B", kab), ("B-then-A", kba)):
    try:
        ff = lambda: kf[grid](x, w1, w2, z, passes=PASSES)
        y1, _ = fa(); z1 = fb(); y2, _, z2 = ff(); torch_neuronx.synchronize()
        e = lambda p, q: (p.float() - q.float()).abs().max().item()
        tf = time_it(ff)
        frac = (ta + tb - tf) / min(ta, tb) * 100.0
        verdict = "OVERLAP REAL" if tf < 0.9 * (ta + tb) else "SERIAL: no intra-kernel overlap"
        print(f"(C) {nm} fused: {tf:6.2f} ms  | correctness max|diff| mlp {e(y1, y2):.3e} norm {e(z1, z2):.3e} "
              f"| overlap = {frac:4.0f}% of the smaller -> {verdict}", flush=True)
    except Exception:
        print(f"(C) {nm} fused FAILED to trace/run:", flush=True); traceback.print_exc()
print("\nIf BOTH orders are serial, the scheduler does not interleave program regions under ANY engine mix with "
      "SBUF to spare, and the 'expose concurrency in the block' family is closed for good.", flush=True)
