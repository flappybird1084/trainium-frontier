"""Hardened embedding gather / scatter-add (lever E, 2026-09-16): identical to nki_embed.py except that every index tile
is passed through a compute-engine clamp before it drives the indirect DMA. Two effects: (1) an index the host never wrote
(whatever the cause) is clamped to the last table row (gather) or the padding tail (scatter) instead of faulting the
GPSIMD DGE ("scatter/gather (indirect memory copy via vector DGE) out-of-bound access", the flake that killed depth-14
runs); (2) the DGE now depends on a compute op that depends on the index DMA, so a DMA->DGE semaphore gap cannot let the
DGE read the index tile early. Cost: one 128x1 tensor_scalar per tile. Dev copy of the vehicle's EMBED_KERNEL=v2 text."""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def embed_gather_v2(table: nl.ndarray, idx: nl.ndarray):
    """out[i, :] = table[min(idx[i], R-1), :] (bf16 rows). idx (N, 1) uint32, N % 128 == 0; table (R, C) fp32 in HBM."""
    N = idx.shape[0]
    R = table.shape[0]
    C = table.shape[1]
    out = nl.ndarray((N, C), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for t in nl.affine_range(N // P):
        ix = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix, src=idx[t * P:(t + 1) * P, :])
        ixc = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ixc, data=ix, op0=nl.minimum, operand0=R - 1)
        rows = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=table.ap(pattern=[[C, P], [1, C]], vector_offset=ixc, indirect_dim=0))
        nisa.dma_copy(dst=out[t * P:(t + 1) * P, :], src=rows)
    return out


@nki.jit
def embed_scatter_add_v2(table: nl.ndarray, grad: nl.ndarray, perm: nl.ndarray, idx: nl.ndarray):
    """table[min(idx[s], Rp-1), :] += grad[min(perm[s], N-1), :] for S slots (S % 128 == 0), in place on the fp32 HBM
    buffer of Rp = R + 128 rows (the tail takes the padding, and now any stray index)."""
    S = idx.shape[0]
    Rp = table.shape[0]
    N = grad.shape[0]
    C = table.shape[1]
    out = nl.ndarray((1, C), dtype=table.dtype, buffer=nl.shared_hbm)
    for t in nl.sequential_range(S // P):
        pm = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pm, src=perm[t * P:(t + 1) * P, :])
        pmc = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pmc, data=pm, op0=nl.minimum, operand0=N - 1)
        ix = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix, src=idx[t * P:(t + 1) * P, :])
        ixc = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ixc, data=ix, op0=nl.minimum, operand0=Rp - 1)
        rows = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=grad.ap(pattern=[[C, P], [1, C]], vector_offset=pmc, indirect_dim=0))
        tgt = table.ap(pattern=[[C, P], [1, C]], vector_offset=ixc, indirect_dim=0)
        nisa.dma_compute(dst=tgt, srcs=[tgt, rows], reduce_op=nl.add)
    nisa.dma_copy(dst=out, src=table[0:1, :])
    return out
