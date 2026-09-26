"""Host-only measurement of the embed-scatter plan, on the real cached corpus.
Replicates prepare.make_dataloader(tok, 4, 4096, 'train') bit-for-bit without torch (best-fit packing copied
verbatim from prepare.py:490-514), then the train loop's reshape to (16, 1024)."""
import pickle, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

CACHE = Path.home() / ".cache/neuron_competition_r1"
SEQ_LEN, PACK, BOS_ID, BIGRAM_SIZE, VOCAB = 1024, 4, 8188, 2 ** 18, 8192
ROW_CAP = SEQ_LEN * PACK + 1


def loader(rank=0, world=8, nbatches=40):
    enc = pickle.load(open(CACHE / "tokenizer/tokenizer.pkl", "rb"))
    paths = sorted((CACHE / "data").glob("shard_0000*.parquet"))
    groups = []
    for p in paths:
        pf = pq.ParquetFile(p)
        groups += [(p, i) for i in range(pf.num_row_groups)]
    groups = groups[rank::world]
    buf = []
    gi = 0
    rows_out = 0

    def refill():
        nonlocal gi
        p, rg = groups[gi % len(groups)]
        gi += 1
        texts = pq.ParquetFile(p).read_row_group(rg).column("text").to_pylist()
        for i in range(0, len(texts), 128):
            for t in enc.encode_ordinary_batch(texts[i:i + 128]):
                buf.append([BOS_ID] + t)

    while rows_out < nbatches:
        rowb = np.zeros((4, ROW_CAP), dtype=np.int64)
        for r in range(4):
            pos = 0
            while pos < ROW_CAP:
                while len(buf) < 512:
                    refill()
                rem = ROW_CAP - pos
                bi, bl = -1, 0
                for i, d in enumerate(buf):
                    n = len(d)
                    if n <= rem and n > bl:
                        bi, bl = i, n
                if bi >= 0:
                    d = buf.pop(bi)
                    rowb[r, pos:pos + len(d)] = d
                    pos += len(d)
                else:
                    si = min(range(len(buf)), key=lambda i: len(buf[i]))
                    d = buf.pop(si)
                    rowb[r, pos:pos + rem] = d[:rem]
                    pos += rem
        rows_out += 1
        yield rowb[:, :-1]          # inputs, (4, 4096)


# --------------------------------------------------------- current planner (verbatim, submit/train.py:3033-3080)
def scatter_rounds(idx):
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    first = np.r_[True, sidx[1:] != sidx[:-1]]
    starts = np.flatnonzero(first)
    rank = np.arange(len(sidx)) - starts[np.cumsum(first) - 1]
    counts = np.bincount(rank)
    by_rank = np.argsort(rank, kind="stable")
    perm_l, idx_l = [], []
    pos = 0
    for n in counts:
        sel = order[by_rank[pos:pos + n]]
        pos += n
        pad = (-n) % 128
        perm_l.append(sel); idx_l.append(idx[sel])
        if pad:
            perm_l.append(np.zeros(pad, dtype=np.int64)); idx_l.append(-1 - np.arange(pad, dtype=np.int64))
    perm = np.concatenate(perm_l)
    sidx = np.concatenate(idx_l)
    if len(sidx) > 32768:
        chunks = []
        for a in range(0, len(sidx), 32768):
            chunks.extend(_pad_plan(perm[a:a + 32768], sidx[a:a + 32768]))
        return chunks
    return _pad_plan(perm, sidx)


def _pad_plan(perm, sidx):
    target = 4096
    while target < len(sidx):
        target *= 2
    pad = target - len(sidx)
    if pad:
        perm = np.concatenate([perm, np.zeros(pad, dtype=np.int64)])
        sidx = np.concatenate([sidx, -1 - (np.arange(pad, dtype=np.int64) % 128)])
    return [(perm.astype(np.uint32).reshape(-1, 1), sidx)]


# --------------------------------------------------------- candidate planner
def scatter_plan(idx, rows, tiles=128, dense_k=32):
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    n = idx.size
    cap = tiles * 128
    counts = np.bincount(idx, minlength=rows)
    present = np.flatnonzero(counts)
    cs = counts[present]
    o = np.argsort(-cs, kind="stable")
    cs_s = cs[o]
    cum = np.concatenate([[0], np.cumsum(cs_s)])
    tail_max = np.concatenate([cs_s, [0]])          # max multiplicity left after densifying p ids
    ok = (tail_max <= tiles) & ((n - cum) <= cap)
    p = int(np.argmax(ok))
    dense = present[o[:p]]
    if p > dense_k:
        return None, None, p, n
    dense_ids = np.full(dense_k, -1, dtype=np.int64)
    dense_ids[:p] = dense
    hot = np.zeros(rows, dtype=bool)
    hot[dense] = True
    pos = np.flatnonzero(~hot[idx])
    m = pos.size
    A = pos[np.argsort(idx[pos], kind="stable")]
    i = np.arange(m)
    slot = (i % tiles) * 128 + (i // tiles)
    perm = np.zeros(tiles * 128, dtype=np.int64)
    sidx = np.full(tiles * 128, -1, dtype=np.int64)
    perm[slot] = A
    sidx[slot] = idx[A]
    hole = sidx < 0
    sidx[hole] = -1 - (np.arange(tiles * 128) % 128)[hole]
    return dense_ids, [(perm.astype(np.uint32).reshape(-1, 1), sidx)], p, m


def check(idx, chunks, rows, tail=128):
    idx = np.asarray(idx, np.int64).reshape(-1)
    for perm, sidx in chunks:
        pm = perm.reshape(-1).astype(np.int64)
        dst = np.where(sidx < 0, rows - 1 - sidx, sidx)
        assert len(sidx) % 128 == 0
        assert pm.min() >= 0 and pm.max() < idx.size
        assert dst.min() >= 0 and dst.max() < rows + tail, (dst.min(), dst.max())
        d2 = dst.reshape(-1, 128)
        bad = [t for t in range(d2.shape[0]) if np.unique(d2[t]).size != 128]
        assert not bad, f"dup dst in tiles {bad[:4]}"


def sim(idx, chunks, g, rows, dense_ids=None, dg=None):
    b = np.zeros((rows + 128, g.shape[1]), dtype=np.float64)
    for perm, sidx in chunks:
        pm = perm.reshape(-1).astype(np.int64)
        dst = np.where(sidx < 0, rows - 1 - sidx, sidx)
        np.add.at(b, dst, g[pm])
    if dense_ids is not None:
        for j, d in enumerate(dense_ids):
            b[d if d >= 0 else rows + j] += dg[j]
    return b[:rows]


N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
rng = np.random.default_rng(0)
agg = {}
for s, xb in enumerate(loader(nbatches=N)):
    ids2 = xb.reshape(-1, SEQ_LEN)                       # (16, 1024)
    ids = ids2.reshape(-1)
    prev = np.concatenate([np.full((ids2.shape[0], 1), BOS_ID, dtype=np.int64), ids2[:, :-1]], 1).reshape(-1)
    pair = (prev.astype(np.int32) * np.int32(8192)) + ids.astype(np.int32)
    bh = (((pair * np.int32(-1640531527)) >> 13).astype(np.int64)) & (BIGRAM_SIZE - 1)
    for kind, ind, R in (("tok", ids, VOCAB), ("bg", bh, BIGRAM_SIZE)):
        c = np.bincount(ind, minlength=R)
        nz = np.sort(c[c > 0])[::-1]
        t0 = time.perf_counter(); cur = scatter_rounds(ind); t_cur = time.perf_counter() - t0
        cur_slots = sum(len(sx) for _, sx in cur)
        rec = agg.setdefault(kind, {k: [] for k in
              ("distinct", "maxmult", "cur_slots", "cur_chunks", "t_cur", "gt32", "gt64", "gt128", "gt256",
               "cov16", "cov32", "cov64", "cov128", "t_new", "p128", "p64", "p32", "m64", "bos")})
        rec["distinct"].append(nz.size); rec["maxmult"].append(nz[0])
        rec["cur_slots"].append(cur_slots); rec["cur_chunks"].append(len(cur)); rec["t_cur"].append(t_cur)
        for th in (32, 64, 128, 256):
            rec[f"gt{th}"].append(int((nz > th).sum()))
        for k in (16, 32, 64, 128):
            rec[f"cov{k}"].append(int(nz[:k].sum()))
        rec["bos"].append(int((ind == BOS_ID).sum()) if kind == "tok" else 0)
        t0 = time.perf_counter(); d, ch, p, m = scatter_plan(ind, R, tiles=128, dense_k=64); t_new = time.perf_counter() - t0
        rec["t_new"].append(t_new); rec["p128"].append(p)
        _, _, p64, m64 = scatter_plan(ind, R, tiles=64, dense_k=4096)
        rec["p64"].append(p64); rec["m64"].append(m64)
        _, _, p32, _ = scatter_plan(ind, R, tiles=32, dense_k=4096)
        rec["p32"].append(p32)
        if s < 3 and ch is not None:
            check(ind, ch, R); check(ind, cur, R)
            g = rng.standard_normal((ind.size, 8)).astype(np.float32).astype(np.float64)
            dg = np.zeros((len(d), 8))
            for j, dd in enumerate(d):
                if dd >= 0:
                    dg[j] = g[ind == dd].sum(0)
            ref = np.zeros((R, 8)); np.add.at(ref, ind, g)
            new = sim(ind, ch, g, R, d, dg)
            old = sim(ind, cur, g, R)
            print(f"  s{s} {kind}: new-ref {np.abs(new-ref).max():.2e}  old-ref {np.abs(old-ref).max():.2e}  p={p} m={m}")

for kind, rec in agg.items():
    print(f"\n=== {kind} ===")
    for k, v in rec.items():
        v = np.array(v, dtype=float)
        print(f"  {k:10s} min/mean/max  {v.min():10.1f} {v.mean():10.1f} {v.max():10.1f}")
