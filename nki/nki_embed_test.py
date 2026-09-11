"""Simulator / device test of nki_embed: gather vs torch indexing, scatter-add (with duplicate indices, via host rounds) vs
torch index_add_. usage: python nki_embed_test.py [sim|device]"""
import sys, time
import numpy as np
import torch
import nki
from nki_embed import embed_gather, embed_scatter_add, P


def scatter_plan(idx: np.ndarray, n_rows: int):
    """Group slots into rounds of distinct indices; pad rounds to multiples of 128 with the OOB index n_rows."""
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    # occurrence rank of each sorted element within its index run
    first = np.r_[True, sidx[1:] != sidx[:-1]]
    starts = np.flatnonzero(first)
    run_id = np.cumsum(first) - 1
    rank = np.arange(len(sidx)) - starts[run_id]
    perm_l, idx_l = [], []
    for r in range(int(rank.max()) + 1):
        sel = order[rank == r]
        pad = (-len(sel)) % P
        perm_l.append(np.concatenate([sel, np.zeros(pad, dtype=np.int64)]))
        idx_l.append(np.concatenate([idx[sel], np.full(pad, n_rows, dtype=np.int64)]))
    return np.concatenate(perm_l).astype(np.uint32), np.concatenate(idx_l).astype(np.uint32), int(rank.max()) + 1


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    torch.manual_seed(0); np.random.seed(0)
    R, C, N = 8192, 512, 4096
    table = torch.randn(R, C)
    idx_np = np.random.zipf(1.3, N).astype(np.int64) % R           # skewed: many duplicates
    idx = torch.from_numpy(idx_np.astype(np.uint32).reshape(N, 1).astype(np.int64)).to(torch.int32)  # placeholder dtype
    idx_u32 = torch.from_numpy(idx_np.astype(np.uint32).reshape(N, 1))
    grad = torch.randn(N, C)
    ref_g = table[idx_np]
    ref_t = table.clone(); ref_t.index_add_(0, torch.from_numpy(idx_np), grad)
    perm, sidx, rounds = scatter_plan(idx_np, R)
    print(f"N={N} distinct={len(np.unique(idx_np))} rounds={rounds} slots={len(sidx)}")
    if mode == "sim":
        g = nki.simulate(embed_gather)(table, idx_u32)
        print("gather max err", (torch.as_tensor(np.asarray(g)) - ref_g).abs().max().item())
        t2 = table.clone()
        nki.simulate(embed_scatter_add)(t2, grad, torch.from_numpy(perm.reshape(-1, 1)), torch.from_numpy(sidx.reshape(-1, 1)))
        print("scatter max err", (t2 - ref_t).abs().max().item())
    else:
        from torch_neuronx.nki_hop import wrap_nki
        dev = torch.device("neuron")
        gk = wrap_nki(embed_gather); sk = wrap_nki(embed_scatter_add)
        td = table.to(dev); id_ = idx_u32.to(dev); gd = grad.to(dev)
        pd = torch.from_numpy(perm.reshape(-1, 1)).to(dev); sd = torch.from_numpy(sidx.reshape(-1, 1)).to(dev)
        g = gk[(1,)](td, id_)
        torch.neuron.synchronize()
        print("gather max err", (g.cpu() - ref_g).abs().max().item())
        t0 = time.monotonic()
        for _ in range(10): g = gk[(1,)](td, id_)
        torch.neuron.synchronize(); print(f"gather {(time.monotonic()-t0)/10*1000:.2f} ms")
        t2 = table.clone().to(dev)
        sk[(1,)](t2, gd, pd, sd)
        torch.neuron.synchronize()
        print("scatter max err", (t2.cpu() - ref_t).abs().max().item())
        t0 = time.monotonic()
        for _ in range(10): sk[(1,)](t2, gd, pd, sd)
        torch.neuron.synchronize(); print(f"scatter {(time.monotonic()-t0)/10*1000:.2f} ms ({rounds} rounds, {len(sidx)} slots)")
        ref_d = ref_t.clone()
        for _ in range(10): ref_d.index_add_(0, torch.from_numpy(idx_np), grad)
        print("scatter x11 max err", (t2.cpu() - ref_d).abs().max().item())
    print("done")


if __name__ == "__main__":
    main()
