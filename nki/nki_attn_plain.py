"""Plain per-tile-DMA variants of the own attention kernels (generated) for the producer-race test."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
from nki_attn_fwd import P, HD, KB, QG, FIXED_MAX

@nki.jit
def attn_fwd4_plain(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, sink_in: nl.ndarray, end_in: nl.ndarray, kidx_in: nl.ndarray):
    """Forward v4 = v3 with ONE strided DMA per tensor per problem / query group (the profile of the backward showed GpSimd,
    which generates DMA descriptors, as the top engine: ~100 small DMAs per problem). K, V, doc ends are loaded as (128, J, .)
    tiles with tile[p, j] = row j*128 + p; the query group as (128, 4, D); outputs are written as (128, 4, .) tiles."""
    B, T, D = q_in.shape
    J = T // P
    ng = J // QG
    W = QG * P
    out = nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm)
    lse_out = nl.ndarray((B, T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nprog = nl.num_programs(0)
    per = B // nprog
    kx3 = nl.ndarray((P, J, 1), dtype=nl.float32, buffer=nl.sbuf)                 # key index per (partition, slice), once
    for j in range(J):
        nisa.dma_copy(dst=kx3[:, j, :], src=kidx_in[j * P:(j + 1) * P, :])
    for bi in nl.affine_range(per):
        b = nl.program_id(0) * per + bi
        K3 = nl.ndarray((P, J, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        V3 = nl.ndarray((P, J, D + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        ke3 = nl.ndarray((P, J, 1), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(J):
            nisa.dma_copy(dst=K3[:, j, :], src=k_in[b, j * P:(j + 1) * P, :])
            nisa.dma_copy(dst=V3[:, j, 0:D], src=v_in[b, j * P:(j + 1) * P, :])
            nisa.dma_copy(dst=ke3[:, j, :], src=end_in[b, j * P:(j + 1) * P, :])
        nisa.memset(V3[:, :, D:D + 1], 1.0)
        KT = []
        for j in range(J):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=K3[:, j, :])
            ktt = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ktt, src=tps)
            KT.append(ktt)
        for g in range(ng):
            q0 = g * W
            Q3 = nl.ndarray((P, QG, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            for c in range(QG):
                nisa.dma_copy(dst=Q3[:, c, :], src=q_in[b, q0 + c * P:q0 + (c + 1) * P, :])
            QT = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            for c in range(QG):
                tq = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tq, data=Q3[:, c, :])
                nisa.tensor_copy(dst=QT[:, c * P:(c + 1) * P], src=tq)
            sk3 = nl.ndarray((P, QG, 1), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(QG):
                nisa.dma_copy(dst=sk3[:, c, :], src=sink_in[b, q0 + c * P:q0 + (c + 1) * P, :])
            accs = []
            for c in range(QG):
                accs.append(nl.ndarray((P, D + 1), dtype=nl.float32, buffer=nl.psum))
            for j in range(QG * (g + 1)):
                sT = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sT, stationary=KT[j], moving=QT)
                sS = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=sS, src=sT)                     # PSUM -> SBUF first (range_select from PSUM raced)
                sm = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                nisa.range_select(dst=sm, on_true_tile=sS, comp_op0=nl.greater_equal, comp_op1=nl.less, bound0=kx3[:, j, :],
                                  bound1=ke3[:, j, :], range_start=q0)
                PT = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=PT, op=nl.exp, data=sm, bias=-FIXED_MAX)
                for c in range(QG):
                    if j > QG * g + c:
                        continue
                    nisa.nc_matmul(dst=accs[c], stationary=PT[:, c * P:(c + 1) * P], moving=V3[:, j, :], accumulate=(j > 0))
            ob3 = nl.ndarray((P, QG, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            ll3 = nl.ndarray((P, QG, 1), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(QG):
                l = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=l, op=nl.exp, data=sk3[:, c, :], bias=-FIXED_MAX)
                nisa.tensor_tensor(dst=l, data1=l, data2=accs[c][:, D:D + 1], op=nl.add)
                rl = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rl, data=l)
                nisa.tensor_scalar(dst=ob3[:, c, :], data=accs[c][:, 0:D], op0=nl.multiply, operand0=rl)
                nisa.activation(dst=ll3[:, c, :], op=nl.log, data=l)
            nisa.tensor_scalar(dst=ll3, data=ll3, op0=nl.add, operand0=FIXED_MAX)
            for c in range(QG):   # plain per-tile output stores: the compiler mis-tracks strided-pattern WRITES consumed downstream
                t0 = q0 + c * P
                nisa.dma_copy(dst=out[b, t0:t0 + P, :], src=ob3[:, c, :])
                nisa.dma_copy(dst=lse_out[b, t0:t0 + P, :], src=ll3[:, c, :])
    return out, lse_out



@nki.jit
def attn_bwd_plain(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, o_in: nl.ndarray, do_in: nl.ndarray, lse_in: nl.ndarray,
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
        for jj in range(NJ):
            nisa.dma_copy(dst=K3[:, jj, :], src=k_in[b, jj * P:(jj + 1) * P, :])
        V3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for jj in range(NJ):
            nisa.dma_copy(dst=V3[:, jj, :], src=v_in[b, jj * P:(jj + 1) * P, :])
        Q3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for jj in range(NJ):
            nisa.dma_copy(dst=Q3[:, jj, :], src=q_in[b, jj * P:(jj + 1) * P, :])
        dO3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for jj in range(NJ):
            nisa.dma_copy(dst=dO3[:, jj, :], src=do_in[b, jj * P:(jj + 1) * P, :])
        O3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for jj in range(NJ):
            nisa.dma_copy(dst=O3[:, jj, :], src=o_in[b, jj * P:(jj + 1) * P, :])
        lse3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        for jj in range(NJ):
            nisa.dma_copy(dst=lse3[:, jj, :], src=lse_in[b, jj * P:(jj + 1) * P, :])
        sk3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        for jj in range(NJ):
            nisa.dma_copy(dst=sk3[:, jj, :], src=sink_in[b, jj * P:(jj + 1) * P, :])
        lo3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        for jj in range(NJ):
            nisa.dma_copy(dst=lo3[:, jj, :], src=lo_in[b, jj * P:(jj + 1) * P, :])
        hi3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        for jj in range(NJ):
            nisa.dma_copy(dst=hi3[:, jj, :], src=hi_in[b, jj * P:(jj + 1) * P, :])
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


