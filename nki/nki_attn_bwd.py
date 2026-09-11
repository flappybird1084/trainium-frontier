#!/usr/bin/env python3
"""Attention backward for the submission's attention (2026-09-10): causal + document bounds + per-head logit sink, head dim 128,
q pre-scaled (scale folded into q). One kernel program handles all B = batch*heads problems of T queries/keys.
Inputs (HBM): q, k, v, o, do (B, T, HD) bf16; lse (B, T, 1) fp32 = log(exp(sink) + sum_j exp(s_j)) per query row;
sink (B, T, 1) fp32 (the head's sink logit, broadcast per row by the caller); lo, hi (B, T, 1) fp32 = the valid key range
[lo, hi) per query row = [doc start, min(doc end, q + 1)) (the caller folds the causal bound into hi).
Outputs: dq, dk, dv (B, T, HD) bf16 and dsink_rows (B, T, 1) fp32 = per-row contribution -exp(sink - lse) * D (sum in torch).
Structure per problem: K natural, K^T, V^T resident in SBUF; for each 128-query block i and each 512-key block J up to the
diagonal: S = Q_i K_J^T (one matmul), P = exp(range_select(S, [lo, hi)) - lse) (one Vector op + one Scalar op), dP = dO_i V_J^T,
dS = P (dP - D_i) (one fused op reading PSUM),
dV_J += P^T dO_i and dK_J += dS^T Q_i (4 narrow matmuls each), dQ_i += dS K_J (4 matmuls on transposed dS slices, PSUM accumulate)."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
HD = 128
KB = 512          # key block (matmul moving free size)
# (masked logits are filled with -FP32_MAX by range_select; exp gives exactly 0)


@nki.jit
def attn_bwd(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, o_in: nl.ndarray, do_in: nl.ndarray, lse_in: nl.ndarray,
             sink_in: nl.ndarray, lo_in: nl.ndarray, hi_in: nl.ndarray):
    B, T, D = q_in.shape
    nq = T // P                       # query blocks of 128
    kb = KB if T >= KB else T         # key block width
    nk = T // kb                      # key blocks
    ks = kb // P                      # 128-key slices per key block
    dq_out = nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm)
    dk_out = nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm)
    dv_out = nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm)
    dsink_out = nl.ndarray((B, T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nprog = nl.num_programs(0)
    per = B // nprog
    for bi in nl.affine_range(per):
        b = nl.program_id(0) * per + bi
        # ---- one strided DMA per tensor for this problem: tile[p, j, :] = row j*128 + p
        NJ = T // P
        K3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=K3, src=k_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        V3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=V3, src=v_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        Q3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=Q3, src=q_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        dO3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=dO3, src=do_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        O3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=O3, src=o_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        lse3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lse3, src=lse_in[b].ap(pattern=[[1, P], [P, NJ], [1, 1]], offset=0))
        sk3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sk3, src=sink_in[b].ap(pattern=[[1, P], [P, NJ], [1, 1]], offset=0))
        lo3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lo3, src=lo_in[b].ap(pattern=[[1, P], [P, NJ], [1, 1]], offset=0))
        hi3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=hi3, src=hi_in[b].ap(pattern=[[1, P], [P, NJ], [1, 1]], offset=0))
        # K^T and V^T as (HD, T) via transposes of the natural slices
        KT = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        VT = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(NJ):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=K3[:, j, :])
            nisa.tensor_copy(dst=KT[:, j * P:(j + 1) * P], src=tps)
            tps2 = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps2, data=V3[:, j, :])
            nisa.tensor_copy(dst=VT[:, j * P:(j + 1) * P], src=tps2)
        dq3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        dsk3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        # fp32 accumulators for dK, dV (natural layout, one tile per 128 keys)
        dK = []
        dV = []
        for j in range(T // P):
            a = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            dK.append(a)
            a2 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a2, 0.0)
            dV.append(a2)
        for i in range(nq):
            t0 = i * P
            Q = Q3[:, i, :]
            dO = dO3[:, i, :]
            O = O3[:, i, :]
            lse = lse3[:, i, :]
            lo = lo3[:, i, :]
            hi = hi3[:, i, :]
            sk = sk3[:, i, :]
            # D_i = rowsum(dO * O); negatives of lse and D for the activation biases / subtractions
            prod = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=dO, data2=O, op=nl.multiply)
            Dr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=Dr, op=nl.add, data=prod, axis=(1,))
            nlse = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nlse, data=lse, op0=nl.multiply, operand0=-1.0)
            # dsink rows: -exp(sink - lse) * D
            ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ps, data1=sk, data2=lse, op=nl.subtract)
            nisa.activation(dst=ps, op=nl.exp, data=ps)
            nisa.tensor_tensor(dst=ps, data1=ps, data2=Dr, op=nl.multiply)
            nisa.tensor_scalar(dst=dsk3[:, i, :], data=ps, op0=nl.multiply, operand0=-1.0)
            # the transposes Q^T, dO^T (HD, 128q)
            tq = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tq, data=Q)
            QT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=QT, src=tq)
            tdo = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tdo, data=dO)
            dOT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dOT, src=tdo)
            dq_ps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            first = True
            for J in range(nk):
                if J * kb > t0 + P - 1:
                    continue                                    # key block entirely after the diagonal: causal skip
                k0 = J * kb
                # S = Q_i K_J^T  (128q x kb keys)
                s_ps = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s_ps, stationary=QT, moving=KT[:, k0:k0 + kb])
                sS = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=sS, src=s_ps)                   # PSUM -> SBUF first (range_select from PSUM raced)
                # mask: keep S where lo <= key < hi (key index = k0 + column), else -FP32_MAX; then P = exp(S - lse) in bf16
                sm = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.sbuf)
                nisa.range_select(dst=sm, on_true_tile=sS, comp_op0=nl.greater_equal, comp_op1=nl.less, bound0=lo, bound1=hi,
                                  range_start=k0)
                Pb = nl.ndarray((P, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=Pb, op=nl.exp, data=sm, bias=nlse)
                # dP = dO_i V_J^T ; dS = (dP - D) * P  (one fused op reading the PSUM tile)
                dp_ps = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=dp_ps, stationary=dOT, moving=VT[:, k0:k0 + kb])
                dpS = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dpS, src=dp_ps)
                dS = nl.ndarray((P, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=dS, data=dpS, op0=nl.subtract, operand0=Dr, op1=nl.multiply, operand1=Pb)
                for s in range(ks):
                    j = J * ks + s                                   # 128-key slice index
                    # dV_j += P[:, slice]^T dO_i ; dK_j += dS[:, slice]^T Q_i   (contraction over the 128 queries)
                    dv_ps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dv_ps, stationary=Pb[:, s * P:(s + 1) * P], moving=dO)
                    nisa.tensor_tensor(dst=dV[j], data1=dV[j], data2=dv_ps, op=nl.add)
                    dk_ps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dk_ps, stationary=dS[:, s * P:(s + 1) * P], moving=Q)
                    nisa.tensor_tensor(dst=dK[j], data1=dK[j], data2=dk_ps, op=nl.add)
                    # dQ_i += dS[:, slice] K_j   (contraction over the 128 keys: needs dS^T)
                    tds = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tds, data=dS[:, s * P:(s + 1) * P])
                    dST = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dST, src=tds)
                    nisa.nc_matmul(dst=dq_ps, stationary=dST, moving=K3[:, j, :], accumulate=(not first))
                    first = False
            nisa.tensor_copy(dst=dq3[:, i, :], src=dq_ps)
        dk3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        dv3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(NJ):
            nisa.tensor_copy(dst=dk3[:, j, :], src=dK[j])
            nisa.tensor_copy(dst=dv3[:, j, :], src=dV[j])
        for j in range(NJ):   # plain per-tile output stores (strided-pattern writes are mis-tracked by the graph compiler)
            nisa.dma_copy(dst=dq_out[b, j * P:(j + 1) * P, :], src=dq3[:, j, :])
            nisa.dma_copy(dst=dk_out[b, j * P:(j + 1) * P, :], src=dk3[:, j, :])
            nisa.dma_copy(dst=dv_out[b, j * P:(j + 1) * P, :], src=dv3[:, j, :])
            nisa.dma_copy(dst=dsink_out[b, j * P:(j + 1) * P, :], src=dsk3[:, j, :])
    return dq_out, dk_out, dv_out, dsink_out


def bounds_lo_hi(start, end):
    """(B, T, 1) fp32 key ranges per query row: lo = doc start, hi = min(doc end, q + 1) (causal bound folded in)."""
    import torch
    B, T = start.shape
    q1 = torch.arange(1, T + 1, dtype=torch.float32).view(1, T)
    return start.float().view(B, T, 1).contiguous(), torch.minimum(end.float(), q1).view(B, T, 1).contiguous()


def reference(q, k, v, sink, start, end):
    """Torch reference of the forward (returns out, lse) with the same semantics: softmax over [sink, masked scores]."""
    import torch
    B, T, D = q.shape
    s = q.float() @ k.float().transpose(1, 2)                               # (B, T, T)
    key = torch.arange(T).view(1, 1, T); qi = torch.arange(T).view(1, T, 1)
    mask = (key >= start.view(B, T, 1)) & (key < end.view(B, T, 1)) & (key <= qi)
    s = s.masked_fill(~mask, -1e9)
    full = torch.cat([sink.view(B, 1, 1).expand(B, T, 1), s], dim=-1)       # sink logit first
    lse = torch.logsumexp(full, dim=-1, keepdim=True)
    p = torch.exp(s - lse)
    out = p @ v.float()
    return out, lse


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    B, T, D = 2, 256, 128
    q = (torch.randn(B, T, D) * 0.3).bfloat16(); k = (torch.randn(B, T, D) * 0.3).bfloat16(); v = torch.randn(B, T, D).bfloat16()
    sink = torch.tensor([[0.5], [-1.0]])
    start = torch.zeros(B, T); end = torch.full((B, T), float(T))
    start[0, 100:] = 100.0; end[0, :100] = 100.0                              # problem 0: a document boundary at 100
    qr = q.float().requires_grad_(True); kr = k.float().requires_grad_(True); vr = v.float().requires_grad_(True)
    skr = sink.clone().requires_grad_(True)
    out, lse = reference(qr, kr, vr, skr, start, end)
    do = torch.randn(B, T, D)
    (out * do).sum().backward()
    o_b = out.detach().bfloat16()
    sink_rows = sink.view(B, 1, 1).expand(B, T, 1).contiguous()
    lo, hi = bounds_lo_hi(start, end)
    dq, dk, dv, dsr = nki.simulate(attn_bwd)(q, k, v, o_b, do.bfloat16(), lse.detach().contiguous(), sink_rows, lo, hi)
    cvt = lambda a: (a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))).float()
    dq, dk, dv, dsr = cvt(dq), cvt(dk), cvt(dv), cvt(dsr)
    for name, got, ref in (("dq", dq, qr.grad), ("dk", dk, kr.grad), ("dv", dv, vr.grad)):
        err = (got - ref).abs().max().item(); print(f"{name}: max err {err:.3e} (ref max {ref.abs().max().item():.3e}) rel {err / ref.abs().max().item():.4f}")
    ds = dsr.sum(dim=(1, 2))
    print(f"dsink: kernel {ds.tolist()} ref {skr.grad.view(-1).tolist()}")
