#!/usr/bin/env python3
"""NKI kernel: batched Polar Express (Muon) orthogonalization for a stack of TALL matrices X (n, m, k), m >= k = 512.

x <- x / (||x||_F * 1.01 + 1e-6);  5x: A = x^T x; B = b*A + c*A*A; x <- a*x + x @ B   (POLAR_EXPRESS_COEFFS)

Everything stays in SBUF per slice (bf16 operands, fp32 PSUM accumulation): the eager PyTorch version is ~40 launches
per stack per step and streams the (n, m, k) tensor through HBM ~30 times. Both X and X^T are maintained so no matmul
needs a transpose: A[r] = sum_p X[p][:, r]^T @ X[p]; A2 = A @ A via A's symmetry; X_new[p] = a X[p] + sum_r (Xt[r][:, p])^T @ B[r];
Xt_new[r] = a Xt[r] + sum_q (B[q][:, r])^T @ Xt[q] (B symmetric). Dev harness: `python nki_muon.py` runs the CPU simulator.
"""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128          # partitions
FMAX = 512       # psum free size (one bank of fp32)
COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@nki.jit
def polar_express_tall(x_in: nl.ndarray, ns_steps: int = 5) -> nl.ndarray:
    n, m, k = x_in.shape
    assert k == FMAX and m % P == 0, "kernel is specialised for k=512, m multiple of 128"
    mt = m // P                 # row tiles of X (contraction tiles for X^T X)
    kt = k // P                 # 128-blocks of the k axis (== 4)
    mc = m // FMAX              # 512-chunks of the m axis (moving free size for Xt updates)
    out = nl.ndarray((n, m, k), dtype=x_in.dtype, buffer=nl.shared_hbm)
    ones = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)

    nprog = nl.num_programs(0)          # LNC sharding: program j handles slices [j*n/nprog, (j+1)*n/nprog)
    per = n // nprog
    for ii in nl.affine_range(per):
        i = nl.program_id(0) * per + ii
        # ---- load X (rows in partitions) and build Xt (k in partitions) with TensorE transposes
        X = []
        for _j in range(mt):
            X.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
        Xt = []
        for _j in range(kt):
            Xt.append(nl.ndarray((P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
        for p in range(mt):
            nisa.dma_copy(dst=X[p], src=x_in[i, p * P:(p + 1) * P, :])
        # ---- Frobenius norm: per-partition sum of squares over all row tiles, then all-partition sum via ones-matmul
        sq = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
        rowsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        part = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(rowsum, 0.0)
        for p in range(mt):
            nisa.tensor_tensor(dst=sq, data1=X[p], data2=X[p], op=nl.multiply)
            nisa.tensor_reduce(dst=part, op=nl.add, data=sq, axis=(1,))
            nisa.tensor_tensor(dst=rowsum, data1=rowsum, data2=part, op=nl.add)
        rs_bf = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)   # NOTE: fp32 matmul operands are allowed too
        tot_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
        rs32 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rs32, src=rowsum)
        ones32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(ones32, 1.0)
        nisa.nc_matmul(dst=tot_ps, stationary=ones32, moving=rs32)      # every partition gets the full sum
        scale = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=scale, op=nl.sqrt, data=tot_ps)             # ||x||_F
        nisa.tensor_scalar(dst=scale, data=scale, op0=nl.multiply, operand0=1.01, op1=nl.add, operand1=1e-6)
        nisa.reciprocal(dst=scale, data=scale)
        for p in range(mt):
            nisa.tensor_scalar(dst=X[p], data=X[p], op0=nl.multiply, operand0=scale)
        # Xt blocks: Xt[c][:, p*128:(p+1)*128] = (X[p][:, c*128:(c+1)*128])^T
        for p in range(mt):
            for c in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=X[p][:, c * P:(c + 1) * P])
                nisa.tensor_copy(dst=Xt[c][:, p * P:(p + 1) * P], src=tps)

        for it in range(ns_steps):
            a = COEFFS[it][0]
            b = COEFFS[it][1]
            c = COEFFS[it][2]
            # ---- A = X^T X  (kt tiles of [128, k]) and B = b*A + c*A@A, both bf16 in SBUF
            A = []
            for _j in range(kt):
                A.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
            for r in range(kt):
                aps = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.psum)
                for p in range(mt):
                    nisa.nc_matmul(dst=aps, stationary=X[p][:, r * P:(r + 1) * P], moving=X[p], accumulate=(p > 0))
                nisa.tensor_copy(dst=A[r], src=aps)
            B = []
            for _j in range(kt):
                B.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
            for r in range(kt):
                a2ps = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.psum)
                for q in range(kt):
                    nisa.nc_matmul(dst=a2ps, stationary=A[q][:, r * P:(r + 1) * P], moving=A[q], accumulate=(q > 0))
                bA = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=bA, data=A[r], op0=nl.multiply, operand0=b)
                nisa.scalar_tensor_tensor(dst=B[r], data=a2ps, op0=nl.multiply, operand0=c, op1=nl.add, operand1=bA)
            # ---- X_new = a X + X B ; Xt_new = a Xt + B Xt
            Xn = []
            for _j in range(mt):
                Xn.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
            for p in range(mt):
                xps = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.psum)
                for r in range(kt):
                    nisa.nc_matmul(dst=xps, stationary=Xt[r][:, p * P:(p + 1) * P], moving=B[r], accumulate=(r > 0))
                nisa.scalar_tensor_tensor(dst=Xn[p], data=X[p], op0=nl.multiply, operand0=a, op1=nl.add, operand1=xps)
            Xtn = []
            for _j in range(kt):
                Xtn.append(nl.ndarray((P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
            for r in range(kt):
                for ch in range(mc):
                    tps2 = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
                    for q in range(kt):
                        nisa.nc_matmul(dst=tps2, stationary=B[q][:, r * P:(r + 1) * P],
                                       moving=Xt[q][:, ch * FMAX:(ch + 1) * FMAX], accumulate=(q > 0))
                    nisa.scalar_tensor_tensor(dst=Xtn[r][:, ch * FMAX:(ch + 1) * FMAX], data=Xt[r][:, ch * FMAX:(ch + 1) * FMAX],
                                              op0=nl.multiply, operand0=a, op1=nl.add, operand1=tps2)
            X, Xt = Xn, Xtn
        for p in range(mt):
            nisa.dma_copy(dst=out[i, p * P:(p + 1) * P, :], src=X[p])
    return out


def reference(x: np.ndarray, ns_steps: int = 5) -> np.ndarray:
    """numpy mirror of MuonAdamW._muon_step's Polar Express loop for tall matrices (fp32 math)."""
    x = x.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=(-2, -1), keepdims=True) * 1.01 + 1e-6)
    for a, b, c in COEFFS[:ns_steps]:
        xtx = np.swapaxes(x, -1, -2) @ x
        x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    return x


def reference_torch_bf16(x, ns_steps: int = 5):
    """exact mirror of the vehicle's _muon_step math (bf16 tensors, tall branch)."""
    import torch
    x = torch.as_tensor(x).bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in COEFFS[:ns_steps]:
        xtx = x.mT @ x
        x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    return x.float().numpy()


if __name__ == "__main__":
    import sys, time
    rng = np.random.default_rng(0)
    n, m, k = (int(v) for v in (sys.argv[1:4] or (2, 512, 512)))
    x = (rng.standard_normal((n, m, k)) * 0.02).astype(np.float32)
    ref = reference(x)
    t0 = time.time()
    import torch
    got = nki.simulate(polar_express_tall)(torch.tensor(x).to(torch.bfloat16))
    got = got.float().numpy()
    ref_bf = reference_torch_bf16(x)
    orth = lambda y: np.abs(np.swapaxes(y, -1, -2) @ y - np.eye(k)).max()
    print(f"simulate {time.time()-t0:.1f}s | n={n} m={m} k={k} | vs fp32 ref: max abs err {np.abs(got - ref).max():.3e} | "
          f"vs bf16 torch ref: {np.abs(got - ref_bf).max():.3e} (bf16 ref vs fp32: {np.abs(ref_bf - ref).max():.3e}, |ref| max {np.abs(ref).max():.3f}) | "
          f"|X^T X - I|: kernel {orth(got):.3e} ref {orth(ref):.3e}")


# ---------------------------------------------------------------------------------------------------------------------
# Fully fused Muon step for one stack (n, out, in) fp32: momentum lerp + nesterov blend -> Polar Express (in bf16, SBUF
# resident, both frames) -> per-row/col second-moment scaling -> masked weight decay update. In-place on w / mom / second
# (also returned). scal = [128, 4] fp32 = [momentum, lr_eff (= lr * max(1, out/in)^0.5), lr_eff*wd, 0].
# "tall" = out >= in: NS runs in the natural frame; "wide": the natural layout is the transposed frame (Xt), X is built by
# TensorE transposes, and the update uses Xt with per-column scaling (broadcast tile via a K=1 ones matmul).
@nki.jit
def muon_step_stack(w_in: nl.ndarray, g_in: nl.ndarray, mom_in: nl.ndarray, sec_in: nl.ndarray, scal: nl.ndarray,
                    ns_steps: int = 5, beta2: float = 0.9):
    n, o, i_ = w_in.shape
    tall = o >= i_
    m = o if tall else i_          # tall-frame rows
    k = i_ if tall else o          # tall-frame cols (== 512)
    mt, kt, mc = m // P, k // P, m // FMAX
    w_out = nl.ndarray((n, o, i_), dtype=nl.float32, buffer=nl.shared_hbm)
    mom_out = nl.ndarray((n, o, i_), dtype=nl.float32, buffer=nl.shared_hbm)
    sec_out = nl.ndarray(sec_in.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=scal)
    ones32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones32, 1.0)
    nprog = nl.num_programs(0)
    per = n // nprog
    for ii in nl.affine_range(per):
        i = nl.program_id(0) * per + ii
        # ---- natural-layout tiles: o // 128 tiles of [128, i_]; momentum update + nesterov blend, cast to bf16
        ot = o // P
        Gn = []
        for _j in range(ot):
            Gn.append(nl.ndarray((P, i_), dtype=nl.bfloat16, buffer=nl.sbuf))
        for p in range(ot):
            g = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
            mo = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g, src=g_in[i, p * P:(p + 1) * P, :])
            nisa.dma_copy(dst=mo, src=mom_in[i, p * P:(p + 1) * P, :])
            # mo = mo + (1-mom)(g - mo) = mom*mo + (1-mom)*g ;  gb = g + mom*(mo - g) = (1-mom)*g + mom*mo
            t1 = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
            one_m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=one_m, data=sc[:, 0:1], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            nisa.tensor_scalar(dst=t1, data=g, op0=nl.multiply, operand0=one_m)                         # (1-mom) g
            nisa.scalar_tensor_tensor(dst=mo, data=mo, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=t1)  # new momentum
            nisa.dma_copy(dst=mom_out[i, p * P:(p + 1) * P, :], src=mo)
            nisa.dma_copy(dst=mom_in[i, p * P:(p + 1) * P, :], src=mo)
            nisa.scalar_tensor_tensor(dst=Gn[p], data=mo, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=t1)  # blend, bf16
        # ---- Frobenius prescale over the natural tiles
        sq = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
        rowsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        part = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(rowsum, 0.0)
        for p in range(ot):
            nisa.tensor_tensor(dst=sq, data1=Gn[p], data2=Gn[p], op=nl.multiply)
            nisa.tensor_reduce(dst=part, op=nl.add, data=sq, axis=(1,))
            nisa.tensor_tensor(dst=rowsum, data1=rowsum, data2=part, op=nl.add)
        tot_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tot_ps, stationary=ones32, moving=rowsum)
        scale = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=scale, op=nl.sqrt, data=tot_ps)
        nisa.tensor_scalar(dst=scale, data=scale, op0=nl.multiply, operand0=1.01, op1=nl.add, operand1=1e-6)
        nisa.reciprocal(dst=scale, data=scale)
        for p in range(ot):
            nisa.tensor_scalar(dst=Gn[p], data=Gn[p], op0=nl.multiply, operand0=scale)
        # ---- tall frame: X (mt tiles of [128, k]) and Xt (kt tiles of [128, m]); one of them is Gn, the other transposed
        X = []
        Xt = []
        if tall:
            X = Gn
            for _j in range(kt):
                Xt.append(nl.ndarray((P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
            for p in range(mt):
                for c in range(kt):
                    tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=X[p][:, c * P:(c + 1) * P])
                    nisa.tensor_copy(dst=Xt[c][:, p * P:(p + 1) * P], src=tps)
        else:
            Xt = Gn
            for _j in range(mt):
                X.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
            for c in range(kt):
                for p in range(mt):
                    tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=Xt[c][:, p * P:(p + 1) * P])
                    nisa.tensor_copy(dst=X[p][:, c * P:(c + 1) * P], src=tps)
        for it in range(ns_steps):
            a = COEFFS[it][0]
            b = COEFFS[it][1]
            c = COEFFS[it][2]
            A = []
            for _j in range(kt):
                A.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
            for r in range(kt):
                aps = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.psum)
                for p in range(mt):
                    nisa.nc_matmul(dst=aps, stationary=X[p][:, r * P:(r + 1) * P], moving=X[p], accumulate=(p > 0))
                nisa.tensor_copy(dst=A[r], src=aps)
            B = []
            for _j in range(kt):
                B.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
            for r in range(kt):
                a2ps = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.psum)
                for q in range(kt):
                    nisa.nc_matmul(dst=a2ps, stationary=A[q][:, r * P:(r + 1) * P], moving=A[q], accumulate=(q > 0))
                bA = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=bA, data=A[r], op0=nl.multiply, operand0=b)
                nisa.scalar_tensor_tensor(dst=B[r], data=a2ps, op0=nl.multiply, operand0=c, op1=nl.add, operand1=bA)
            Xn = []
            for _j in range(mt):
                Xn.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
            for p in range(mt):
                xps = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.psum)
                for r in range(kt):
                    nisa.nc_matmul(dst=xps, stationary=Xt[r][:, p * P:(p + 1) * P], moving=B[r], accumulate=(r > 0))
                nisa.scalar_tensor_tensor(dst=Xn[p], data=X[p], op0=nl.multiply, operand0=a, op1=nl.add, operand1=xps)
            Xtn = []
            for _j in range(kt):
                Xtn.append(nl.ndarray((P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
            for r in range(kt):
                for ch in range(mc):
                    tps2 = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
                    for q in range(kt):
                        nisa.nc_matmul(dst=tps2, stationary=B[q][:, r * P:(r + 1) * P],
                                       moving=Xt[q][:, ch * FMAX:(ch + 1) * FMAX], accumulate=(q > 0))
                    nisa.scalar_tensor_tensor(dst=Xtn[r][:, ch * FMAX:(ch + 1) * FMAX], data=Xt[r][:, ch * FMAX:(ch + 1) * FMAX],
                                              op0=nl.multiply, operand0=a, op1=nl.add, operand1=tps2)
            X, Xt = Xn, Xtn
        # ---- second-moment scaling. Reduction axis = the tall-frame free axis (k): v_mean per tall-frame row.
        vmean = []                       # mt tiles of [128, 1] fp32
        for _j in range(mt):
            vmean.append(nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf))
        vsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(vsum, 0.0)
        sqk = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
        for p in range(mt):
            nisa.tensor_tensor(dst=sqk, data1=X[p], data2=X[p], op=nl.multiply)
            nisa.tensor_reduce(dst=vmean[p], op=nl.add, data=sqk, axis=(1,))
            nisa.tensor_scalar(dst=vmean[p], data=vmean[p], op0=nl.multiply, operand0=1.0 / k)
            nisa.tensor_tensor(dst=vsum, data1=vsum, data2=vmean[p], op=nl.add)
        vn_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=vn_ps, stationary=ones32, moving=vsum)                # sum of v_mean over all rows
        vnorm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=vnorm, data=vn_ps, op0=nl.multiply, operand0=float(k))
        nisa.activation(dst=vnorm, op=nl.sqrt, data=vnorm)                      # v_norm
        # second (per tall-frame row) lerp, step_size = rsqrt(max(second, 1e-10)), scaled = v_mean*k*step_size^2
        step = []
        ssum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(ssum, 0.0)
        for p in range(mt):
            sec = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            if tall:
                nisa.dma_copy(dst=sec, src=sec_in[i, p * P:(p + 1) * P, :])
            else:
                nisa.dma_copy(dst=sec, src=sec_in[i, :, p * P:(p + 1) * P].ap(pattern=[[1, P], [1, 1]], offset=0))
            t2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=t2, data=vmean[p], op0=nl.multiply, operand0=1.0 - beta2)
            nisa.scalar_tensor_tensor(dst=sec, data=sec, op0=nl.multiply, operand0=beta2, op1=nl.add, operand1=t2)
            if tall:
                nisa.dma_copy(dst=sec_out[i, p * P:(p + 1) * P, :], src=sec)
                nisa.dma_copy(dst=sec_in[i, p * P:(p + 1) * P, :], src=sec)
            else:
                nisa.dma_copy(dst=sec_out[i, :, p * P:(p + 1) * P].ap(pattern=[[1, P], [1, 1]], offset=0), src=sec)
                nisa.dma_copy(dst=sec_in[i, :, p * P:(p + 1) * P].ap(pattern=[[1, P], [1, 1]], offset=0), src=sec)
            st = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=st, data=sec, op0=nl.maximum, operand0=1e-10)
            nisa.activation(dst=st, op=nl.rsqrt, data=st)
            step.append(st)
            sc2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sc2, data1=st, data2=st, op=nl.multiply)
            nisa.scalar_tensor_tensor(dst=sc2, data=vmean[p], op0=nl.multiply, operand0=float(k), op1=nl.multiply, operand1=sc2)
            nisa.tensor_tensor(dst=ssum, data1=ssum, data2=sc2, op=nl.add)
        sn_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=sn_ps, stationary=ones32, moving=ssum)
        ratio = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=ratio, op=nl.sqrt, data=sn_ps)                       # v_norm_new
        nisa.tensor_scalar(dst=ratio, data=ratio, op0=nl.maximum, operand0=1e-10)
        nisa.reciprocal(dst=ratio, data=ratio)
        nisa.tensor_tensor(dst=ratio, data1=ratio, data2=vnorm, op=nl.multiply)  # v_norm / v_norm_new (all partitions equal)
        # per tall-frame row factor f[r] = step_size[r] * ratio ; the update runs in the NATURAL frame
        # tall: natural rows == tall rows -> per-partition scalar; wide: natural rows are k, factor varies along the free axis
        f_rows = []
        for p in range(mt):
            fr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=fr, data1=step[p], data2=ratio, op=nl.multiply)
            f_rows.append(fr)
        if not tall:
            # broadcast tile F[128, m]: F[:, r] = f[r]  via K=1 matmul: ones[1, 128]^T @ f_row[1, m]
            f_row = nl.ndarray((1, m), dtype=nl.float32, buffer=nl.sbuf)
            for p in range(mt):
                tp = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=f_rows[p])
                nisa.tensor_copy(dst=f_row[:, p * P:(p + 1) * P], src=tp)
            ones_row = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(ones_row, 1.0)
            Fb = nl.ndarray((P, m), dtype=nl.float32, buffer=nl.sbuf)
            for ch in range(mc):
                fps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=fps, stationary=ones_row, moving=f_row[:, ch * FMAX:(ch + 1) * FMAX])
                nisa.tensor_copy(dst=Fb[:, ch * FMAX:(ch + 1) * FMAX], src=fps)
        # ---- update in the natural frame: g = orth * f ; mask = (g*w >= 0) ; w -= lr*g + lr*wd*w*mask
        Gnat = X if tall else Xt
        for p in range(ot):
            w = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w, src=w_in[i, p * P:(p + 1) * P, :])
            gg = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
            if tall:
                nisa.tensor_scalar(dst=gg, data=Gnat[p], op0=nl.multiply, operand0=f_rows[p])
            else:
                nisa.tensor_tensor(dst=gg, data1=Gnat[p], data2=Fb, op=nl.multiply)
            prod = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=gg, data2=w, op=nl.multiply)
            mask = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=prod, op0=nl.greater_equal, operand0=0.0)     # 1.0 where g*w >= 0
            wdm = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=wdm, data1=w, data2=mask, op=nl.multiply)
            nisa.tensor_scalar(dst=wdm, data=wdm, op0=nl.multiply, operand0=sc[:, 2:3])    # lr*wd*w*mask
            nisa.scalar_tensor_tensor(dst=gg, data=gg, op0=nl.multiply, operand0=sc[:, 1:2], op1=nl.add, operand1=wdm)  # lr*g + ...
            nisa.tensor_tensor(dst=w, data1=w, data2=gg, op=nl.subtract)
            nisa.dma_copy(dst=w_out[i, p * P:(p + 1) * P, :], src=w)
            nisa.dma_copy(dst=w_in[i, p * P:(p + 1) * P, :], src=w)
    return w_out, mom_out, sec_out


def reference_muon_step(w, g, mom, sec, momentum, lr, wd, ns_steps=5, beta2=0.9):
    """torch mirror of MuonAdamW._muon_step on one stack (fp32 w/g/mom/sec, bf16 NS) -> new w, mom, sec."""
    import torch
    w, g, mom, sec = (torch.as_tensor(t).clone() for t in (w, g, mom, sec))
    shape = w.shape[1:]
    mom.lerp_(g, 1 - momentum)
    g = g.lerp_(mom, momentum)
    x = g.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if x.shape[-2] > x.shape[-1]:
        for a, b, c in COEFFS[:ns_steps]:
            xtx = x.mT @ x; x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in COEFFS[:ns_steps]:
            xxt = x @ x.mT; x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    g = x
    red_dim = -1 if shape[-2] >= shape[-1] else -2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    sec.lerp_(v_mean, 1 - beta2)
    step_size = sec.clamp_min(1e-10).rsqrt()
    scaled = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    lr_eff = lr * (max(1.0, shape[-2] / shape[-1]) ** 0.5)
    mask = (g * w) >= 0
    w.sub_(lr_eff * g + lr_eff * wd * w * mask)
    return w, mom, sec
