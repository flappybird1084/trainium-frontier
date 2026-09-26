import time, numpy as np
import deliverable as D
from probe import loader, scatter_rounds, SEQ_LEN, BOS_ID, BIGRAM_SIZE, VOCAB

def sim(idx, rows, dense_ids, chunks, grad, k):
    """exactly what the kernels do, including the clamps, in fp64"""
    n = idx.size
    buf = np.zeros((rows + 128, grad.shape[1]))
    if dense_ids is not None:
        oh = np.asarray(D.scatter_onehot(idx, rows, dense_ids, dtype=__import__("torch").float64), dtype=np.float64)
        gd = oh.T @ grad
        perm, sidx = D.scatter_dense_tile(dense_ids, rows)
        dst = np.minimum(np.where(sidx < 0, rows - 1 - sidx, sidx), rows + 127)
        pm = np.minimum(perm.reshape(-1).astype(np.int64), k - 1)
        assert np.unique(dst).size == 128
        np.add.at(buf, dst, gd[pm])
    for perm, sidx in chunks:
        pm = np.minimum(perm.reshape(-1).astype(np.int64), n - 1)
        dst = np.minimum(np.where(sidx < 0, rows - 1 - sidx, sidx), rows + 127)
        np.add.at(buf, dst, grad[pm])
    return buf[:rows]

def run(idx, rows, tag, k=32, bins=128, d=4):
    idx = np.asarray(idx, np.int64); n = idx.size
    di, ch = D.scatter_packed(idx, rows, bins=bins, dense_k=k)
    g = np.random.default_rng(1).standard_normal((n, d))
    got = sim(idx, rows, di, ch, g, k)
    ref = np.zeros((rows, d)); np.add.at(ref, idx, g)
    e = np.abs(got - ref).max()
    p = 0 if di is None else int((di >= 0).sum())
    print(f"  {tag:40s} launches={len(ch)}+{1 if di is not None else 0} tiles={sum(len(s) for _,s in ch)//128+(1 if di is not None else 0):5d} dense={p:3d} err={e:.1e}")
    assert e < 1e-9, tag

rng = np.random.default_rng(0); N = 16384
run(np.zeros(N, np.int64), VOCAB, "all one token")
run(np.full(N, 8191, np.int64), VOCAB, "all the LAST row")
run(np.arange(N) % VOCAB, VOCAB, "uniform 2 each")
run(rng.integers(0, VOCAB, N), VOCAB, "uniform random")
run(np.repeat(np.arange(128), 128), VOCAB, "128 ids x 128 (at the bin budget)")
run(np.concatenate([np.full(129, 5), rng.integers(0, VOCAB, N-129)]), VOCAB, "one id at 129")
run(np.concatenate([np.repeat(np.arange(64), 250), rng.integers(0, VOCAB, N-16000)]), VOCAB, "64 ids x250: K exhausted -> 2 chunks")
run(rng.integers(0, BIGRAM_SIZE, N), BIGRAM_SIZE, "bigram uniform", k=0)
run(np.concatenate([np.full(300, 7), rng.integers(0, BIGRAM_SIZE, N-300)]), BIGRAM_SIZE, "bigram bucket x300 -> 3 chunks", k=0)
run(np.zeros(N, np.int64), BIGRAM_SIZE, "bigram degenerate (all one bucket)", k=0)
run(rng.integers(0, VOCAB, 128), VOCAB, "tiny n=128")
run(np.array([3]), VOCAB, "n=1")
run(rng.integers(0, VOCAB, N), VOCAB, "bins=112 K=48", k=48, bins=112)

print("\n--- real corpus, 24 micro-steps ---")
oc=nc=ob=nb=0.0; to=tn=0.0
for s, xb in enumerate(loader(nbatches=24)):
    ids2 = xb.reshape(-1, SEQ_LEN); ids = ids2.reshape(-1)
    prev = np.concatenate([np.full((16,1), BOS_ID, np.int64), ids2[:, :-1]],1).reshape(-1)
    pair = (prev.astype(np.int32)*np.int32(8192)) + ids.astype(np.int32)
    bh = (((pair*np.int32(-1640531527))>>13).astype(np.int64)) & (BIGRAM_SIZE-1)
    for kind, ind, R, k in (("tok", ids, VOCAB, 32), ("bg", bh, BIGRAM_SIZE, 0)):
        t0=time.perf_counter(); cur = scatter_rounds(ind); to += time.perf_counter()-t0
        t0=time.perf_counter(); di, ch = D.scatter_packed(ind, R, dense_k=k); tn += time.perf_counter()-t0
        o = sum(len(sx) for _,sx in cur)//128; nn = sum(len(sx) for _,sx in ch)//128 + (1 if di is not None else 0)
        if kind=="tok": oc+=o; nc+=nn
        else: ob+=o; nb+=nn
        if s < 3: run(ind, R, f"s{s} {kind}", k=k)
n=24
ott = (oc*4+ob)/n; ntt = (nc*4+nb)/n
print(f"\ntiles/micro-step  tok {oc/n:.1f} -> {nc/n:.1f}   bg {ob/n:.1f} -> {nb/n:.1f}")
print(f"tile-tables       {ott:.0f} -> {ntt:.0f}  ({ntt/ott*100:.1f}%)   12.9 ms -> {12.9*ntt/ott:.2f} ms  (saves {12.9*(1-ntt/ott):.2f} ms)")
print(f"host planner+checks  {to/n*1e3:.2f} ms -> {tn/n*1e3:.2f} ms per micro-step")
