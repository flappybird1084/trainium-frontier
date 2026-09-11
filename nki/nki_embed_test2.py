"""Device test of nki_embed scatter variants, each in its own process (a DMA abort kills the process, not the test):
  a: 1 tile, 128 distinct indices        b: 32 tiles, 4096 distinct indices
  c: real-data-like duplicates via host rounds with distinct dummy-row padding (R+128 buffer)
  d: torch index_add_ in eager on the device (reference alternative)      e: same into a 131072-row table
  f: real-data-like token ids (max multiplicity ~150), bucketed slots, 4 tables sequentially (the per-micro-step cost)
usage: python nki_embed_test2.py            (driver)   |   python nki_embed_test2.py <a|b|c|d>"""
import sys, time, subprocess
import numpy as np
import torch

R, C, N = 8192, 512, 4096


def rounds(idx):
    order = np.argsort(idx, kind="stable"); sidx = idx[order]
    first = np.r_[True, sidx[1:] != sidx[:-1]]; starts = np.flatnonzero(first)
    rank = np.arange(len(sidx)) - starts[np.cumsum(first) - 1]
    counts = np.bincount(rank); by_rank = np.argsort(rank, kind="stable")
    perm_l, idx_l, pos = [], [], 0
    for n in counts:
        sel = order[by_rank[pos:pos + n]]; pos += n; pad = (-n) % 128
        perm_l.append(sel); idx_l.append(idx[sel])
        if pad:
            perm_l.append(np.zeros(pad, dtype=np.int64)); idx_l.append(R + np.arange(pad))
    perm = np.concatenate(perm_l); sidx = np.concatenate(idx_l)
    target = 4096
    while target < len(sidx):
        target *= 2
    pad = target - len(sidx)
    if pad:
        perm = np.concatenate([perm, np.zeros(pad, dtype=np.int64)]); sidx = np.concatenate([sidx, R + (np.arange(pad) % 128)])
    return perm.astype(np.uint32).reshape(-1, 1), sidx.astype(np.uint32).reshape(-1, 1), len(counts)


def run(variant):
    from nki_embed import embed_scatter_add
    from torch_neuronx.nki_hop import wrap_nki
    dev = torch.device("neuron")
    sk = wrap_nki(embed_scatter_add)
    torch.manual_seed(0); np.random.seed(0)
    RR = 131072 if variant == "e" else R
    table = torch.randn(RR + 128, C)
    if variant == "a":
        idx = np.random.permutation(R)[:128]
    elif variant in ("e", "f"):
        idx = np.minimum((np.random.zipf(1.1, N)), RR - 1).astype(np.int64)      # ~150x max multiplicity
    elif variant == "b":
        idx = np.random.permutation(R)[:N]
    else:
        idx = (np.random.zipf(1.3, N) % R).astype(np.int64)
    grad = torch.randn(len(idx), C)
    ref = table.clone(); ref[:RR].index_add_(0, torch.from_numpy(idx), grad)
    if variant in ("d", "e"):
        td = table.to(dev); gd = grad.to(dev); id_ = torch.from_numpy(idx).to(dev)
        td[:RR].index_add_(0, id_, gd); torch.neuron.synchronize()
        print(f"{variant}: index_add_ ({RR} rows) max err {(td.cpu()[:RR] - ref[:RR]).abs().max().item():.4f}")
        t0 = time.monotonic()
        for _ in range(10): td[:RR].index_add_(0, id_, gd)
        torch.neuron.synchronize(); print(f"{variant}: index_add_ ({RR} rows) {(time.monotonic() - t0) / 10 * 1000:.2f} ms"); return
    if variant in ("a", "b"):
        perm = np.arange(len(idx), dtype=np.uint32).reshape(-1, 1); sidx = idx.astype(np.uint32).reshape(-1, 1); nr = 1
    else:
        perm, sidx, nr = rounds(idx)
    td = table.to(dev); gd = grad.to(dev); pd = torch.from_numpy(perm).to(dev); sd = torch.from_numpy(sidx).to(dev)
    sk[(1,)](td, gd, pd, sd); torch.neuron.synchronize()
    err = (td.cpu()[:R] - ref[:R]).abs().max().item()
    t0 = time.monotonic()
    for _ in range(10): sk[(1,)](td, gd, pd, sd)
    torch.neuron.synchronize()
    print(f"{variant}: scatter max err {err:.4f}  {(time.monotonic() - t0) / 10 * 1000:.2f} ms  rounds={nr} slots={len(sidx)}")
    if variant == "f":   # a second batch with a different round count must NOT recompile (bucketed slots)
        idx2 = np.minimum((np.random.zipf(1.1, N)), R - 1).astype(np.int64); perm2, sidx2, nr2 = rounds(idx2)
        pd2 = torch.from_numpy(perm2).to(dev); sd2 = torch.from_numpy(sidx2).to(dev); gd2 = torch.randn(N, C).to(dev)
        t0 = time.monotonic(); sk[(1,)](td, gd2, pd2, sd2); torch.neuron.synchronize()
        print(f"{variant}: second batch rounds={nr2} slots={len(sidx2)} first call {(time.monotonic() - t0) * 1000:.1f} ms (recompile if >> 10)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        for v in "abcdef":
            try:
                out = subprocess.run([sys.executable, __file__, v], capture_output=True, text=True, timeout=150)
                lines = [l for l in (out.stdout + out.stderr).splitlines() if l.startswith(f"{v}:") or "Error" in l[:60] or "FATAL" in l]
                print(f"== {v}: rc={out.returncode}", *lines[:4], sep="\n  ", flush=True)
            except subprocess.TimeoutExpired:
                print(f"== {v}: TIMEOUT", flush=True)
        print("done")
