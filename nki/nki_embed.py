"""Embedding sidecar kernels (2026-09-07): gather rows of an HBM table by dynamic index, and scatter-ADD dense gradient rows
into an HBM table by dynamic index. Used to take the embedding tables (wte, 3 value embeddings, 2^17-row bigram table) out of
the compiled graph, whose backward lowers their scatter-add as dense one-hot matmuls (~45% of backward FLOPs at 512-wide).

Scatter constraint (nisa.dma_compute RMW): indices must be unique within one instruction (one 128-row tile). The host groups
the rows into "rounds" (r-th occurrence of every distinct index) and pads each round to a multiple of 128 with DISTINCT dummy
rows R..R+127 of a (R+128)-row gradient buffer (out-of-bounds "skip" padding aborted the DMA on device: NKIE1 2026-09-07).
perm = which dense grad row feeds each slot (padding: row 0), idx = the buffer row per slot. Rounds run sequentially.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def embed_gather(table: nl.ndarray, idx: nl.ndarray):
    """out[i, :] = table[idx[i], :]. idx: (N, 1) uint32, N a multiple of 128; table (R, C) in HBM."""
    N = idx.shape[0]
    C = table.shape[1]
    out = nl.ndarray((N, C), dtype=table.dtype, buffer=nl.shared_hbm)
    for t in nl.affine_range(N // P):
        ix = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix, src=idx[t * P:(t + 1) * P, :])
        rows = nl.ndarray((P, C), dtype=table.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=table.ap(pattern=[[C, P], [1, C]], vector_offset=ix, indirect_dim=0))
        nisa.dma_copy(dst=out[t * P:(t + 1) * P, :], src=rows)
    return out


@nki.jit
def embed_scatter_add(table: nl.ndarray, grad: nl.ndarray, perm: nl.ndarray, idx: nl.ndarray):
    """table[idx[s], :] += grad[perm[s], :] for every slot s (S slots, multiple of 128), in place on `table` (HBM, fp32,
    R+128 rows: the tail takes the padding). idx / perm (S, 1) uint32, all in bounds. Slots within one 128-tile carry
    distinct indices (host guarantee); tiles are processed in order."""
    S = idx.shape[0]
    C = table.shape[1]
    out = nl.ndarray((1, C), dtype=table.dtype, buffer=nl.shared_hbm)   # dummy output (the update is in place)
    for t in nl.sequential_range(S // P):
        pm = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pm, src=perm[t * P:(t + 1) * P, :])
        ix = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix, src=idx[t * P:(t + 1) * P, :])
        rows = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=grad.ap(pattern=[[C, P], [1, C]], vector_offset=pm, indirect_dim=0))
        tgt = table.ap(pattern=[[C, P], [1, C]], vector_offset=ix, indirect_dim=0)
        nisa.dma_compute(dst=tgt, srcs=[tgt, rows], reduce_op=nl.add)
    nisa.dma_copy(dst=out, src=table[0:1, :])
    return out
