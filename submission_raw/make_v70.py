#!/usr/bin/env python3
"""Generate v70 = v68 + dense-top-K plus optimal bin-packing in the embedding gradient scatter. ~-9.9 ms/step.

## The defect, measured twice independently

`scatter_rounds` (`submit/train.py:3071`) achieves the kernel's per-tile distinctness requirement by OCCURRENCE
RANK: round *r* holds the r-th occurrence of every distinct id. So the most frequent token sets the round count.
Measured on the real cached corpus at the production shape (16,384 ids/micro-step):

    rounds        403 / 520 / 573  (min/mean/max)   <- the docstring claims "~150"
    padded slots  76,928 for 16,384 real ids        =  78.7% PADDING
    max multiplicity 520                            <- what forces it

Replacing it with a dense path for the few ids that exceed one tile's budget, plus optimal bin-packing for the
rest, was verified to fp64 exactness against `np.add.at` on 13 pathological cases and real corpus batches:

    tiles/micro-step   tok 638.7 -> 129.0    bg 256.0 -> 128.0
    tile-tables        2,811 -> 644  (22.9%)
    phase              12.9 ms -> 2.96 ms    = SAVES 9.94 ms of a 424 ms step = 2.3% = about -0.00065 bpb
    host planner       3.56 ms -> 2.21 ms    (FASTER, with every safety check on)

Three facts that shaped the design and corrected my own framing:
1. **Only 8-14 token ids per micro-step have count > 128.** The dense path needs K for HEADROOM, not coverage.
2. **Bin-packing alone halves the BIGRAM table** (256 -> 128 tiles), because its max multiplicity is 41-201. My
   earlier "bin-packing alone is nearly worthless" was true only for the token tables.
3. **bg multiplicity exceeded 128 twice in 60 micro-steps** (peak 201), so ~20 times in a 4,238-step run. Handled
   at zero shape cost by emitting a second chunk of the SAME shape.

## The structural win: the slot ladder disappears

`_pad_plan`'s power-of-two ladder and the 32,768-slot chunking exist to keep the compiled slot count out of the
data. Both become unnecessary. After removing the dense ids, sort so each id's occurrences are contiguous and
deal them round-robin into `T` bins: occurrences of one id land in distinct residues mod T, so **per-tile
distinctness is structural rather than achieved by padding**. Round T up to a whole multiple of `SCATTER_BINS`
and emit one launch per group.

So **every chunk has exactly 128 * SCATTER_BINS = 16,384 slots** and only the NUMBER of launches is
data-dependent. No call can trigger a recompile - this retires the hazard class behind the recorded
"65536-slot variant failed to lower mid-run", and it shrinks the compiled NEFF set (tok emitted 32768/32768/16384
= two shapes; now one).

## Exactness: numerically equivalent, NOT bit-identical - and not anywhere in the table

The hot rows change 520 sequential fp32 adds into a PE reduction, and bin-packing changes which tile each cold
occurrence lands in, so no subset stays bit-identical. **The loss curve therefore diverges from the reference at
step 1; do not use a step-1 loss match as the correctness check.** Measured relative error against fp64 on
realistic bf16 gradient rows: the current sequential path 7.4e-07, the fp32 dense path 7.4e-07 (same order, and
better conditioned - the sequential path degrades from 6e-08 at count 130 to 1.1e-05 at count 701), a bf16 dense
output 3.8e-03. So `SCATTER_DENSE_FP32` is kept on: it costs ~0.8 ms, which is 0.00005 bpb, below the revised
per-gate sd of 0.00007.

`SCATTER_DENSE_K = 128` rather than the 32 the design suggested: it makes the dense gradient exactly 128 rows, so
the indirect gather's `perm` is `arange(128)` with no clamping, which removes the one unverified question (whether
a 128-lane indirect DMA lowers from a 32-row tensor). The one-hot is then 8 MiB and the matmul 2.1 GFLOP per
table - still trivial on the least-busy engine in the step.

`SCATTER_BINS = 0` selects the old `scatter_rounds` path, so this file is its own A/B control.

Run from the repo root:  .venv/bin/python submission_raw/make_v70.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v68_clean_embbeta2lo.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v70_clean_scatterpack.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------------------- knobs + the new planner
sub("planner", """def scatter_rounds(idx):""",
    '''SCATTER_BINS = int(os.environ.get("NEURON_COMPETITION_R1_SCATTER_BINS", "128"))
SCATTER_DENSE_K = int(os.environ.get("NEURON_COMPETITION_R1_SCATTER_DENSE_K", "128"))
SCATTER_DENSE_FP32 = os.environ.get("NEURON_COMPETITION_R1_SCATTER_DENSE_FP32", "1") != "0"
SCATTER_CHECK = os.environ.get("NEURON_COMPETITION_R1_SCATTER_CHECK", "1") != "0"


def scatter_packed(idx, rows, bins=None, dense_k=None):
    """Host plan for embed_scatter_add_v2[_x4], replacing scatter_rounds' occurrence-rank rounds.

    The kernel needs every 128-slot tile to hold DISTINCT destination rows (dma_compute is a read-modify-write, so
    a repeat inside one tile silently loses an update); a destination may repeat ACROSS tiles. scatter_rounds gets
    distinctness by occurrence rank, so the most frequent id sets the tile count - measured 403-573 rounds and
    78.7% padding. Here the ids whose count exceeds one tile's budget go to a DENSE path (their gradient rows are
    pre-summed by one matmul, then scattered as a single 128-slot tile) and the rest are dealt round-robin into
    `bins` bins, which is optimal packing: occurrences of one id are contiguous after the sort and an id has
    count <= bins, so they land in distinct residues mod T - distinctness is STRUCTURAL, not padded.

    Every chunk has exactly 128*bins slots, so only the NUMBER of launches is data-dependent and no call can
    trigger a recompile. Returns (dense_ids, chunks); sidx == -1-j marks the j-th padding slot of its tile, with
    j == the slot's own lane so prepare()'s R-1-sidx mapping still gives a distinct dummy row per lane."""
    import numpy as np
    bins = SCATTER_BINS if bins is None else bins
    dense_k = SCATTER_DENSE_K if dense_k is None else dense_k
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    n = idx.size
    cap = bins * 128
    order = np.argsort(idx, kind="stable")             # occurrences of one id are contiguous in `order`
    sid = idx[order]
    first = np.r_[True, sid[1:] != sid[:-1]]
    starts = np.flatnonzero(first)
    cnt = np.diff(np.r_[starts, n])
    uid = sid[starts]
    dense_ids = None
    keep = np.ones(starts.size, dtype=bool)
    if dense_k > 0:
        o = np.argsort(-cnt, kind="stable")[:dense_k]   # most frequent first, ties by lowest id -> deterministic
        cs = cnt[o]
        cum = np.cumsum(cs)
        p = 0
        while p < o.size and (cs[p] > bins or n - (cum[p - 1] if p else 0) > cap):
            p += 1                                      # densify while a bin budget or the slot capacity is exceeded
        dense_ids = np.full(dense_k, -1, dtype=np.int64)
        dense_ids[:p] = uid[o[:p]]
        keep[o[:p]] = False
    sel = order[np.repeat(keep, cnt)]
    m = sel.size
    t = max(-(-m // 128), int(cnt[keep].max()) if m else 0, 1)
    t = -(-t // bins) * bins                            # whole chunks, so slots per launch stay `cap`
    i = np.arange(m)
    slot = (i % t) * 128 + (i // t)                     # tile = i % t, lane = i // t (<= 127 since m <= 128*t)
    perm = np.zeros(t * 128, dtype=np.int64)
    sidx = np.full(t * 128, -1, dtype=np.int64)
    perm[slot] = sel
    sidx[slot] = idx[sel]
    hole = sidx < 0
    sidx[hole] = -1 - (np.arange(t * 128) % 128)[hole]   # j = the slot's own lane: distinct in-tile, j < 128
    chunks = [(perm[a:a + cap].astype(np.uint32).reshape(-1, 1), sidx[a:a + cap]) for a in range(0, t * 128, cap)]
    if SCATTER_CHECK:
        scatter_plan_check(idx, rows, dense_ids, chunks)
    return dense_ids, chunks


def scatter_dense_tile(dense_ids, rows):
    """The single 128-slot plan that scatters the dense matmul's (K, C) output. Lane j carries dense_ids[j], or the
    dummy row rows+j when that column is unused."""
    import numpy as np
    k = len(dense_ids)
    lane = np.arange(128, dtype=np.int64)
    ids = np.full(128, -1, dtype=np.int64)
    ids[:min(k, 128)] = dense_ids[:128]
    perm = np.minimum(lane, k - 1)
    sidx = np.where(ids >= 0, ids, -1 - lane)
    return perm.astype(np.uint32).reshape(-1, 1), sidx


def scatter_onehot(idx, rows, dense_ids, dtype):
    """The (n, K) one-hot whose column j selects the occurrences of dense_ids[j]; unused columns are all-zero, so
    their matmul row is exactly 0 and lands on a dummy row."""
    import numpy as np
    n = idx.size
    k = len(dense_ids)
    p = int((dense_ids >= 0).sum())
    lut = np.full(rows, -1, dtype=np.int32)
    if p:
        lut[dense_ids[:p]] = np.arange(p, dtype=np.int32)
    col = lut[idx]
    at = np.flatnonzero(col >= 0)
    oh = torch.zeros((n, k), dtype=dtype)
    if at.size:
        oh[torch.from_numpy(at), torch.from_numpy(col[at].astype(np.int64))] = 1.0
    return oh


def scatter_plan_check(idx, rows, dense_ids, chunks):
    """Every way this plan can silently corrupt a gradient, checked on the host for ~0.17 ms per micro-step. The
    new planner is 1.35 ms FASTER than the old one, so the full check is free in net host time - and the invariant
    that corrupts silently (a repeat inside a tile) is exhaustively checkable, which is why it stays on."""
    import numpy as np
    n = idx.size
    real = 0
    for perm, sidx in chunks:
        assert len(sidx) % 128 == 0, "slot count is not a whole number of tiles"
        pm = perm.reshape(-1).astype(np.int64)
        assert pm.min() >= 0 and pm.max() < n, "perm out of the gradient buffer"       # else clamped -> wrong row
        neg = sidx < 0
        assert (sidx[neg] >= -128).all(), "padding index outside the 128-row tail"     # else clamped onto a real row
        assert sidx[~neg].max(initial=0) < rows, "destination past the table"          # else clamped onto row R-1
        dst = np.where(neg, rows - 1 - sidx, sidx).reshape(-1, 128)
        assert not (np.diff(np.sort(dst, axis=1), axis=1) == 0).any(), "repeat inside a tile"   # else lost update
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


def scatter_rounds(idx):''')

# ---------------------------------------------------------------------------- prepare builds the 5-tuple
sub("prepare", """        plans = {}
        for kind, ind in keys.items():
            chunks = scatter_rounds(ind)
            R = self.tables[self.kinds[kind][0]].shape[0]
            parts = [ind]
            for perm, sidx in chunks:
                parts.append(perm.reshape(-1)); parts.append(np.where(sidx < 0, R - 1 - sidx, sidx))   # padding -> R + j
            buf = np.concatenate(parts).astype(np.uint32).reshape(-1, 1)
            dbuf = torch.from_numpy(buf).to(dev)
            n = len(ind); off = n; views = []
            for perm, sidx in chunks:
                S = perm.shape[0]
                views.append((dbuf[off:off + S], dbuf[off + S:off + 2 * S])); off += 2 * S
            plans[kind] = (ind, dbuf[:n], views, None)
        return plans""",
    """        plans = {}
        for kind, ind in keys.items():
            R = self.tables[self.kinds[kind][0]].shape[0]
            dense_ids = None
            if SCATTER_BINS > 0:
                # the bigram table needs no dense path: its max multiplicity is 41-201, so bin-packing alone
                # already takes it from 256 tiles to 128
                dk = SCATTER_DENSE_K if (kind == "tok" and not getattr(self, "dense_off", False)) else 0
                dense_ids, chunks = scatter_packed(ind, R, dense_k=dk)
            else:
                chunks = scatter_rounds(ind)
            parts = [ind]
            for perm, sidx in chunks:
                parts.append(perm.reshape(-1)); parts.append(np.where(sidx < 0, R - 1 - sidx, sidx))   # padding -> R + j
            if dense_ids is not None:
                dperm, dsidx = scatter_dense_tile(dense_ids, R)
                parts.append(dperm.reshape(-1)); parts.append(np.where(dsidx < 0, R - 1 - dsidx, dsidx))
            buf = np.concatenate(parts).astype(np.uint32).reshape(-1, 1)
            dbuf = torch.from_numpy(buf).to(dev)
            n = len(ind); off = n; views = []
            for perm, sidx in chunks:
                S = perm.shape[0]
                views.append((dbuf[off:off + S], dbuf[off + S:off + 2 * S])); off += 2 * S
            dense = None
            if dense_ids is not None:
                oh = scatter_onehot(ind, R, dense_ids,
                                    torch.float32 if SCATTER_DENSE_FP32 else COMPUTE_DTYPE)
                dense = (oh.to(dev), dbuf[off:off + 128], dbuf[off + 128:off + 256])
            plans[kind] = (ind, dbuf[:n], views, None, dense)
        return plans""")

# ---------------------------------------------------------------------------- unpack sites
sub("gather unpack", """        for kind, (ind, ind_t, views, _unused) in plans.items():
            for name in self.kinds[kind]:""",
    """        for kind, (ind, ind_t, views, _unused, _dense) in plans.items():
            for name in self.kinds[kind]:""")

sub("scatter unpack", """        for kind, (ind, ind_t, views, _unused) in plans.items():
            names = self.kinds[kind]""",
    """        for kind, (ind, ind_t, views, _unused, dense) in plans.items():
            names = self.kinds[kind]""")

# ---------------------------------------------------------------------------- the dense launch
sub("dense launch", """                    gs = [emb[n].grad.reshape(-1, emb[n].grad.shape[-1]).contiguous() for n in names]
                    bs = [self.gbuf[n] for n in names]
                    k4 = build_nki_embed()[2]
                    for perm_t, sidx_t in views:
                        k4[(1,)](bs[0], bs[1], bs[2], bs[3], gs[0], gs[1], gs[2], gs[3], perm_t, sidx_t)
                    continue
                except Exception as exc:
                    print0(f"scatter x4 unavailable ({str(exc)[:80]}); per-table path")""",
    """                    gs = [emb[n].grad.reshape(-1, emb[n].grad.shape[-1]).contiguous() for n in names]
                    bs = [self.gbuf[n] for n in names]
                    k4 = build_nki_embed()[2]
                    if dense is not None:
                        oh, dperm_t, dsidx_t = dense
                        # the hot ids' rows pre-summed on the PE: (n,K)^T @ (n,C) -> (K,C), one tile to scatter.
                        # Deliberately BEFORE the sparse launches, so the new instantiation is the first to fail
                        # if it is going to fail, and gd is consumed immediately.
                        gd = [torch.matmul(oh.t(), g.to(oh.dtype)) for g in gs]
                        k4[(1,)](bs[0], bs[1], bs[2], bs[3], gd[0], gd[1], gd[2], gd[3], dperm_t, dsidx_t)
                    for perm_t, sidx_t in views:
                        k4[(1,)](bs[0], bs[1], bs[2], bs[3], gs[0], gs[1], gs[2], gs[3], perm_t, sidx_t)
                    continue
                except Exception as exc:
                    self.dense_off = True      # plans built two micro-steps out drop the dense path
                    print0(f"scatter x4 unavailable ({str(exc)[:80]}); per-table path")""")

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:18s} x{v}")
