# ==== EXACT TEXT PROPOSED FOR train.py (planner block, replaces nothing; scatter_rounds/_pad_plan stay) ====
import numpy as np

SCATTER_BINS = 128        # bins (= 128-slot tiles) per launch: the ONE compiled slot count (128*SCATTER_BINS)
SCATTER_DENSE_K = 32      # one-hot width of the dense path for the token tables (0 = pure bin-packing, no matmul)
SCATTER_CHECK = True      # 0.17 ms/micro-step of host asserts on the plan (see the risk register)


def scatter_packed(idx, rows, bins=SCATTER_BINS, dense_k=SCATTER_DENSE_K):
    """Host plan for embed_scatter_add_v2[_x4], replacing scatter_rounds' occurrence-rank rounds.

    The kernel needs every 128-slot tile to hold DISTINCT destination rows (dma_compute is a read-modify-write, so
    a repeat inside a tile loses an update); a destination may repeat across tiles.  scatter_rounds gets that by
    occurrence rank, so the most frequent id sets the round count (measured: 403-701 rounds, 78% padding).  Here the
    ids whose count exceeds one tile's budget are pulled out into a DENSE path (their gradient rows are pre-summed
    by one matmul on device, then scattered as a single 128-slot tile) and the rest are dealt round-robin into
    `bins` bins, which is optimal packing.

    Returns (dense_ids, chunks):
      dense_ids  (dense_k,) int64, or None when dense_k == 0.  Column j of the one-hot <-> row j of the matmul
                 output <-> dense_ids[j].  -1 = unused column (destination: the dummy row rows+j).
      chunks     [(perm (S,1) uint32, sidx (S,) int64)], S == 128*bins for EVERY chunk - one kernel launch each.
                 sidx >= 0 is a table row; sidx == -1-j is the j-th padding slot of its tile (-> dummy row rows+j,
                 mapped by prepare()).  Only the NUMBER of chunks is data-dependent, never the shape, so no call
                 can trigger a recompile.
    """
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    n = idx.size
    cap = bins * 128
    order = np.argsort(idx, kind="stable")             # the occurrences of one id are contiguous in `order`
    sid = idx[order]
    first = np.r_[True, sid[1:] != sid[:-1]]
    starts = np.flatnonzero(first)
    cnt = np.diff(np.r_[starts, n])                    # occurrences per distinct id (ascending id)
    uid = sid[starts]
    dense_ids = None
    keep = np.ones(starts.size, dtype=bool)
    if dense_k > 0:
        o = np.argsort(-cnt, kind="stable")[:dense_k]  # most frequent first, ties by lowest id (deterministic)
        cs = cnt[o]
        cum = np.cumsum(cs)
        p = 0
        while p < o.size and (cs[p] > bins or n - (cum[p - 1] if p else 0) > cap):
            p += 1                                     # densify while a bin budget or the slot capacity is exceeded
        dense_ids = np.full(dense_k, -1, dtype=np.int64)
        dense_ids[:p] = uid[o[:p]]
        keep[o[:p]] = False
    sel = order[np.repeat(keep, cnt)]
    m = sel.size
    t = max(-(-m // 128), int(cnt[keep].max()) if m else 0, 1)
    t = -(-t // bins) * bins                           # whole chunks, so the slot count per launch stays `cap`
    i = np.arange(m)
    slot = (i % t) * 128 + (i // t)                    # tile = i % t, lane = i // t (<= 127, because m <= 128*t)
    perm = np.zeros(t * 128, dtype=np.int64)
    sidx = np.full(t * 128, -1, dtype=np.int64)
    perm[slot] = sel
    sidx[slot] = idx[sel]
    hole = sidx < 0
    sidx[hole] = -1 - (np.arange(t * 128) % 128)[hole]  # j = the slot's own lane: distinct inside the tile, j < 128
    chunks = [(perm[a:a + cap].astype(np.uint32).reshape(-1, 1), sidx[a:a + cap]) for a in range(0, t * 128, cap)]
    if SCATTER_CHECK:
        scatter_plan_check(idx, rows, dense_ids, chunks)
    return dense_ids, chunks


def scatter_dense_tile(dense_ids, rows):
    """The single 128-slot plan that scatters the dense matmul's (dense_k, C) output.  Lane j < dense_k carries
    dense_ids[j] (or the dummy row rows+j if that column is unused); lanes >= dense_k are padding."""
    k = len(dense_ids)
    lane = np.arange(128, dtype=np.int64)
    ids = np.full(128, -1, dtype=np.int64)
    ids[:k] = dense_ids
    perm = np.minimum(lane, k - 1)                     # every slot reads a real row of the (k, C) matmul output
    sidx = np.where(ids >= 0, ids, -1 - lane)
    return perm.astype(np.uint32).reshape(-1, 1), sidx


def scatter_onehot(idx, rows, dense_ids, dtype=None):
    """The (n, dense_k) bf16 one-hot whose column j selects the occurrences of dense_ids[j]; all-zero columns for
    the unused ones (their matmul row is then exactly 0 and lands on a dummy row)."""
    import torch
    n = idx.size
    k = len(dense_ids)
    p = int((dense_ids >= 0).sum())
    lut = np.full(rows, -1, dtype=np.int32)
    lut[dense_ids[:p]] = np.arange(p, dtype=np.int32)
    col = lut[idx]
    at = np.flatnonzero(col >= 0)
    oh = torch.zeros((n, k), dtype=dtype or torch.bfloat16)
    if at.size:
        oh[torch.from_numpy(at), torch.from_numpy(col[at].astype(np.int64))] = 1.0
    return oh


def scatter_plan_check(idx, rows, dense_ids, chunks):
    """Every way this plan can silently corrupt a gradient, checked on the host for 0.17 ms per micro-step."""
    n = idx.size
    real = 0
    for perm, sidx in chunks:
        assert len(sidx) % 128 == 0, "slot count is not a whole number of tiles"
        pm = perm.reshape(-1).astype(np.int64)
        assert pm.min() >= 0 and pm.max() < n, "perm out of the gradient buffer"          # else: clamped, wrong row
        neg = sidx < 0
        assert (sidx[neg] >= -128).all(), "padding index outside the 128-row tail"        # else: clamped onto a real row
        assert sidx[~neg].max(initial=0) < rows, "destination past the table"             # else: clamped onto row R-1
        dst = np.where(neg, rows - 1 - sidx, sidx).reshape(-1, 128)
        assert not (np.diff(np.sort(dst, axis=1), axis=1) == 0).any(), "repeat inside a tile"   # else: lost update
        real += int((~neg).sum())
    if dense_ids is not None:
        p = int((dense_ids >= 0).sum())
        assert np.unique(dense_ids[:p]).size == p, "duplicate id in the dense set"
        if p:
            hot = np.zeros(rows, dtype=bool)
            hot[dense_ids[:p]] = True
            for _perm, sidx in chunks:
                assert not hot[sidx[sidx >= 0]].any(), "id in BOTH paths: double-counted gradient"
            real += int(np.bincount(idx, minlength=rows)[dense_ids[:p]].sum())
    assert real == n, f"occurrences lost or duplicated: {real} placed, {n} in the batch"
