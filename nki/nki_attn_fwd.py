#!/usr/bin/env python3
"""Attention forward for the submission's attention (2026-09-10): causal + document bounds + per-head logit sink, head dim
128, q pre-scaled. One kernel program handles all B = batch*heads problems of T queries/keys (natural (B, T, HD) layouts).
Inputs: q, k, v (B, T, HD) bf16; sink (B, T, 1) fp32 (the head's sink logit per row); lo, hi (B, T, 1) fp32 = valid key
range [lo, hi) per query row = [doc start, min(doc end, q + 1)).
Outputs: out (B, T, HD) bf16 and lse (B, T, 1) fp32 = log(exp(sink) + sum_j exp(s_j)) (what the backward needs).
Per 128-query block: online softmax over 512-key blocks up to the diagonal, the sink being the initial running max with
sum 1; S = Q_i K_J^T (one matmul), range_select masks and returns the row max, exp returns the row sum, P V_J via four
transposed 128-key slices accumulated in one PSUM tile, O = O * alpha + PV in one fused op."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
HD = 128
KB = 512


@nki.jit
def attn_fwd(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, sink_in: nl.ndarray, lo_in: nl.ndarray, hi_in: nl.ndarray):
    B, T, D = q_in.shape
    nq = T // P
    kb = KB if T >= KB else T
    nk = T // kb
    ks = kb // P
    out = nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm)
    lse_out = nl.ndarray((B, T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nprog = nl.num_programs(0)
    per = B // nprog
    for bi in nl.affine_range(per):
        b = nl.program_id(0) * per + bi
        # resident K^T (HD, T) and V natural tiles for this problem
        KT = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        Vn = []
        for j in range(T // P):
            kt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kt, src=k_in[b, j * P:(j + 1) * P, :])
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=kt)
            nisa.tensor_copy(dst=KT[:, j * P:(j + 1) * P], src=tps)
            vt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=vt, src=v_in[b, j * P:(j + 1) * P, :])
            Vn.append(vt)
        for i in range(nq):
            t0 = i * P
            Q = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=Q, src=q_in[b, t0:t0 + P, :])
            tq = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tq, data=Q)
            QT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=QT, src=tq)
            lo = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lo, src=lo_in[b, t0:t0 + P, :])
            hi = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=hi, src=hi_in[b, t0:t0 + P, :])
            # running max m (starts at the sink logit), running sum l (starts at exp(sink - m) = 1), accumulator O
            m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m, src=sink_in[b, t0:t0 + P, :])
            l = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(l, 1.0)
            O = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(O, 0.0)
            for J in range(nk):
                if J * kb > t0 + P - 1:
                    continue                                    # key block entirely after the diagonal
                k0 = J * kb
                s_ps = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s_ps, stationary=QT, moving=KT[:, k0:k0 + kb])
                # masked scores (-FP32_MAX outside [lo, hi)) and their row max
                sm = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.sbuf)
                bmax = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.range_select(dst=sm, on_true_tile=s_ps, comp_op0=nl.greater_equal, comp_op1=nl.less, bound0=lo, bound1=hi,
                                  range_start=k0, reduce_res=bmax, reduce_op=nl.maximum)
                # m_new = max(m, bmax); alpha = exp(m - m_new); m = m_new
                mnew = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=mnew, data1=m, data2=bmax, op=nl.maximum)
                nmnew = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=nmnew, data=mnew, op0=nl.multiply, operand0=-1.0)
                alpha = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=alpha, data1=m, data2=mnew, op=nl.subtract)
                nisa.activation(dst=alpha, op=nl.exp, data=alpha)
                nisa.tensor_copy(dst=m, src=mnew)
                # P = exp(sm - m_new) in bf16, with its row sum; l = l * alpha + rowsum
                Pb = nl.ndarray((P, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
                rs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=Pb, op=nl.exp, data=sm, bias=nmnew, reduce_op=nl.add, reduce_res=rs, reduce_cmd=nisa.reduce_cmd.reset_reduce)
                nisa.scalar_tensor_tensor(dst=l, data=l, op0=nl.multiply, operand0=alpha, op1=nl.add, operand1=rs)
                # PV = sum over 128-key slices of P[:, slice] V_slice  (contraction over keys: transposed P slices)
                pv_ps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                for s in range(ks):
                    j = J * ks + s
                    tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=Pb[:, s * P:(s + 1) * P])
                    PT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=PT, src=tps)
                    nisa.nc_matmul(dst=pv_ps, stationary=PT, moving=Vn[j], accumulate=(s > 0))
                # O = O * alpha + PV
                nisa.scalar_tensor_tensor(dst=O, data=O, op0=nl.multiply, operand0=alpha, op1=nl.add, operand1=pv_ps)
            # out = O / l ; lse = m + log(l)
            rl = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rl, data=l)
            ob = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ob, data=O, op0=nl.multiply, operand0=rl)
            nisa.dma_copy(dst=out[b, t0:t0 + P, :], src=ob)
            ll = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=ll, op=nl.log, data=l)
            nisa.tensor_tensor(dst=ll, data1=ll, data2=m, op=nl.add)
            nisa.dma_copy(dst=lse_out[b, t0:t0 + P, :], src=ll)
    return out, lse_out



FIXED_MAX = 20.0   # the QK-normed, 0.1-scaled logits are bounded (|s| <= 0.1 * sqrt(128) * sqrt(256) = 18.1 with the key offset)


@nki.jit
def attn_fwd2(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, sink_in: nl.ndarray, end_in: nl.ndarray, kidx_in: nl.ndarray):
    """Forward v2: no running max (fixed reference FIXED_MAX, valid because the logits are bounded), no transposes.
    Per 128-key slice j of a query block i: S^T = K_j Q_i^T (one narrow matmul: keys on partitions), the causal + document
    mask is ONE range_select on the query index per key row (valid iff key <= q < doc_end(key)), P^T = exp(S^T - M) in bf16,
    and one matmul with [V_j | 1] accumulates both P V and the row sums in one PSUM tile (129 columns).
    Inputs: q, k, v (B, T, HD) bf16; sink (B, T, 1) fp32 per query row; end (B, T, 1) fp32 = doc end per KEY position;
    kidx (T, 1) fp32 = the key index 0..T-1 (per-partition iota, built by the caller). Outputs: out (B, T, HD) bf16,
    lse (B, T, 1) fp32 = M + log(exp(sink - M) + sum_k exp(s_k - M))."""
    B, T, D = q_in.shape
    nq = T // P
    out = nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm)
    lse_out = nl.ndarray((B, T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nprog = nl.num_programs(0)
    per = B // nprog
    for bi in nl.affine_range(per):
        b = nl.program_id(0) * per + bi
        # resident K^T slices [HD, 128k], [V_j | 1] moving tiles [128k, HD + 1], per-key doc ends and key indices
        KT = []
        V1 = []
        ke = []
        kx = []
        for j in range(T // P):
            kt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kt, src=k_in[b, j * P:(j + 1) * P, :])
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=kt)
            ktt = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ktt, src=tps)
            KT.append(ktt)
            v1 = nl.ndarray((P, D + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=v1[:, 0:D], src=v_in[b, j * P:(j + 1) * P, :])
            nisa.memset(v1[:, D:D + 1], 1.0)
            V1.append(v1)
            e = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=e, src=end_in[b, j * P:(j + 1) * P, :])
            ke.append(e)
            x = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=kidx_in[j * P:(j + 1) * P, :])
            kx.append(x)
        for i in range(nq):
            t0 = i * P
            Q = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=Q, src=q_in[b, t0:t0 + P, :])
            tq = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tq, data=Q)
            QT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=QT, src=tq)
            acc = nl.ndarray((P, D + 1), dtype=nl.float32, buffer=nl.psum)    # [sum_k P V | sum_k P] per query row
            first = True
            for j in range(i + 1):                                            # key slices up to the diagonal
                sT = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)     # S^T slice: keys on partitions, queries free
                nisa.nc_matmul(dst=sT, stationary=KT[j], moving=QT)
                sm = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.range_select(dst=sm, on_true_tile=sT, comp_op0=nl.greater_equal, comp_op1=nl.less, bound0=kx[j], bound1=ke[j],
                                  range_start=t0)                              # keep where key <= q < doc_end(key)
                PT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=PT, op=nl.exp, data=sm, bias=-FIXED_MAX)
                nisa.nc_matmul(dst=acc, stationary=PT, moving=V1[j], accumulate=(not first))
                first = False
            # l = exp(sink - M) + rowsum ; out = (P V) / l ; lse = M + log(l)
            sk = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sk, src=sink_in[b, t0:t0 + P, :])
            l = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=l, op=nl.exp, data=sk, bias=-FIXED_MAX)
            nisa.tensor_tensor(dst=l, data1=l, data2=acc[:, D:D + 1], op=nl.add)
            rl = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rl, data=l)
            ob = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ob, data=acc[:, 0:D], op0=nl.multiply, operand0=rl)
            nisa.dma_copy(dst=out[b, t0:t0 + P, :], src=ob)
            ll = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=ll, op=nl.log, data=l, bias=None)
            nisa.tensor_scalar(dst=ll, data=ll, op0=nl.add, operand0=FIXED_MAX)
            nisa.dma_copy(dst=lse_out[b, t0:t0 + P, :], src=ll)
    return out, lse_out



QG = 4   # query blocks per group (512 queries share one S^T tile per key slice)


@nki.jit
def attn_fwd3(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, sink_in: nl.ndarray, end_in: nl.ndarray, kidx_in: nl.ndarray):
    """Forward v3 = v2 with query GROUPS of 4 blocks: per key slice j and group g, S^T = K_j [Q_g]^T is one 512-wide matmul,
    the mask one range_select and the exp one activation over (128 keys x 512 queries); then 4 narrow PV matmuls (one per
    query block) into 4 PSUM accumulators. Same inputs/outputs as attn_fwd2."""
    B, T, D = q_in.shape
    nq = T // P
    ng = nq // QG
    W = QG * P
    out = nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm)
    lse_out = nl.ndarray((B, T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nprog = nl.num_programs(0)
    per = B // nprog
    for bi in nl.affine_range(per):
        b = nl.program_id(0) * per + bi
        KT = []
        V1 = []
        ke = []
        kx = []
        for j in range(T // P):
            kt = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kt, src=k_in[b, j * P:(j + 1) * P, :])
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=kt)
            ktt = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ktt, src=tps)
            KT.append(ktt)
            v1 = nl.ndarray((P, D + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=v1[:, 0:D], src=v_in[b, j * P:(j + 1) * P, :])
            nisa.memset(v1[:, D:D + 1], 1.0)
            V1.append(v1)
            e = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=e, src=end_in[b, j * P:(j + 1) * P, :])
            ke.append(e)
            x = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=kidx_in[j * P:(j + 1) * P, :])
            kx.append(x)
        for g in range(ng):
            q0 = g * W
            # [Q_g]^T: (HD, 512 queries) from 4 transposed query blocks
            QT = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            for c in range(QG):
                Q = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=Q, src=q_in[b, q0 + c * P:q0 + (c + 1) * P, :])
                tq = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tq, data=Q)
                nisa.tensor_copy(dst=QT[:, c * P:(c + 1) * P], src=tq)
            accs = []
            for c in range(QG):
                accs.append(nl.ndarray((P, D + 1), dtype=nl.float32, buffer=nl.psum))
            first = True
            for j in range(QG * (g + 1)):                                   # key slices up to the group's last diagonal
                sT = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sT, stationary=KT[j], moving=QT)
                sm = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                nisa.range_select(dst=sm, on_true_tile=sT, comp_op0=nl.greater_equal, comp_op1=nl.less, bound0=kx[j], bound1=ke[j],
                                  range_start=q0)
                PT = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=PT, op=nl.exp, data=sm, bias=-FIXED_MAX)
                for c in range(QG):
                    if j > QG * g + c:
                        continue                                            # key slice entirely after this block's diagonal
                    nisa.nc_matmul(dst=accs[c], stationary=PT[:, c * P:(c + 1) * P], moving=V1[j], accumulate=(j > 0))
            for c in range(QG):
                t0 = q0 + c * P
                sk = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=sk, src=sink_in[b, t0:t0 + P, :])
                l = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=l, op=nl.exp, data=sk, bias=-FIXED_MAX)
                nisa.tensor_tensor(dst=l, data1=l, data2=accs[c][:, D:D + 1], op=nl.add)
                rl = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rl, data=l)
                ob = nl.ndarray((P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ob, data=accs[c][:, 0:D], op0=nl.multiply, operand0=rl)
                nisa.dma_copy(dst=out[b, t0:t0 + P, :], src=ob)
                ll = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=ll, op=nl.log, data=l)
                nisa.tensor_scalar(dst=ll, data=ll, op0=nl.add, operand0=FIXED_MAX)
                nisa.dma_copy(dst=lse_out[b, t0:t0 + P, :], src=ll)
    return out, lse_out



@nki.jit
def attn_fwd4(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, sink_in: nl.ndarray, end_in: nl.ndarray, kidx_in: nl.ndarray):
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
    nisa.dma_copy(dst=kx3, src=kidx_in.ap(pattern=[[1, P], [P, J], [1, 1]], offset=0))
    for bi in nl.affine_range(per):
        b = nl.program_id(0) * per + bi
        K3 = nl.ndarray((P, J, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=K3, src=k_in[b].ap(pattern=[[D, P], [P * D, J], [1, D]], offset=0))
        V3 = nl.ndarray((P, J, D + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=V3[:, :, 0:D], src=v_in[b].ap(pattern=[[D, P], [P * D, J], [1, D]], offset=0))
        nisa.memset(V3[:, :, D:D + 1], 1.0)
        ke3 = nl.ndarray((P, J, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ke3, src=end_in[b].ap(pattern=[[1, P], [P, J], [1, 1]], offset=0))
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
            nisa.dma_copy(dst=Q3, src=q_in[b, q0:q0 + W, :].ap(pattern=[[D, P], [P * D, QG], [1, D]], offset=0))
            QT = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            for c in range(QG):
                tq = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tq, data=Q3[:, c, :])
                nisa.tensor_copy(dst=QT[:, c * P:(c + 1) * P], src=tq)
            sk3 = nl.ndarray((P, QG, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sk3, src=sink_in[b, q0:q0 + W, :].ap(pattern=[[1, P], [P, QG], [1, 1]], offset=0))
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


def fwd2_args(sink_rows, end, T, device=None):
    """Caller-side inputs for attn_fwd2: doc end per key position (B, T, 1) fp32 and the key index column (T, 1) fp32."""
    import torch
    B = end.shape[0]
    kidx = torch.arange(T, dtype=torch.float32).view(T, 1).contiguous()
    return end.float().view(B, T, 1).contiguous(), kidx


if __name__ == "__main__":
    import torch
    from nki_attn_bwd import reference, bounds_lo_hi
    torch.manual_seed(0)
    B, T, D = 2, 256, 128
    q = (torch.randn(B, T, D) * 0.3).bfloat16(); k = (torch.randn(B, T, D) * 0.3).bfloat16(); v = torch.randn(B, T, D).bfloat16()
    sink = torch.tensor([[0.5], [-1.0]])
    start = torch.zeros(B, T); end = torch.full((B, T), float(T))
    start[0, 100:] = 100.0; end[0, :100] = 100.0
    out_ref, lse_ref = reference(q.float(), k.float(), v.float(), sink, start, end)
    lo, hi = bounds_lo_hi(start, end)
    sink_rows = sink.view(B, 1, 1).expand(B, T, 1).contiguous()
    out, lse = nki.simulate(attn_fwd)(q, k, v, sink_rows, lo, hi)
    cvt = lambda a: (a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))).float()
    out, lse = cvt(out), cvt(lse)
    e = (out - out_ref).abs().max().item(); print(f"out: max err {e:.3e} (ref max {out_ref.abs().max().item():.3e}) rel {e / out_ref.abs().max().item():.4f}")
    e = (lse - lse_ref).abs().max().item(); print(f"lse: max err {e:.3e} (ref max {lse_ref.abs().max().item():.3e})")
    end_rows, kidx = fwd2_args(sink_rows, end, T)
    out2, lse2 = nki.simulate(attn_fwd2)(q, k, v, sink_rows, end_rows, kidx)
    out2, lse2 = cvt(out2), cvt(lse2)
    e = (out2 - out_ref).abs().max().item(); print(f"v2 out: max err {e:.3e} rel {e / out_ref.abs().max().item():.4f}")
    e = (lse2 - lse_ref).abs().max().item(); print(f"v2 lse: max err {e:.3e}")
    B3, T3 = 2, 1024
    q3 = (torch.randn(B3, T3, D) * 0.3).bfloat16(); k3 = (torch.randn(B3, T3, D) * 0.3).bfloat16(); v3 = torch.randn(B3, T3, D).bfloat16()
    start3 = torch.zeros(B3, T3); end3 = torch.full((B3, T3), float(T3)); start3[0, 300:] = 300.0; end3[0, :300] = 300.0
    out_ref3, lse_ref3 = reference(q3.float(), k3.float(), v3.float(), sink, start3, end3)
    sink_rows3 = sink.view(B3, 1, 1).expand(B3, T3, 1).contiguous()
    end_rows3, kidx3 = fwd2_args(sink_rows3, end3, T3)
    out3, lse3 = nki.simulate(attn_fwd3)(q3, k3, v3, sink_rows3, end_rows3, kidx3)
    out3, lse3 = cvt(out3), cvt(lse3)
    e = (out3 - out_ref3).abs().max().item(); print(f"v3 out: max err {e:.3e} rel {e / out_ref3.abs().max().item():.4f}")
    e = (lse3 - lse_ref3).abs().max().item(); print(f"v3 lse: max err {e:.3e}")
    out4, lse4 = nki.simulate(attn_fwd4)(q3, k3, v3, sink_rows3, end_rows3, kidx3)
    out4, lse4 = cvt(out4), cvt(lse4)
    e = (out4 - out_ref3).abs().max().item(); print(f"v4 out: max err {e:.3e} rel {e / out_ref3.abs().max().item():.4f}")
    e = (lse4 - lse_ref3).abs().max().item(); print(f"v4 lse: max err {e:.3e}")
