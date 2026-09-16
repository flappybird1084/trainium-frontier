"""Simulator / device test of nki_embed_v2 (clamped indices): in-bounds parity with nki_embed's semantics, plus injected
out-of-bound indices that must be clamped (gather -> last row; scatter -> padding tail, table unchanged).
usage: python nki_embed_v2_test.py [sim|device]"""
import sys, time
import numpy as np
import torch
import nki
from nki_embed_v2 import embed_gather_v2, embed_scatter_add_v2, P
from nki_embed_test import scatter_plan


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    torch.manual_seed(0); np.random.seed(0)
    R, C, N = 8192, 512, 4096
    table = torch.randn(R + 128, C)          # R real rows + 128 padding rows, like the vehicle's gbuf
    idx_np = np.random.zipf(1.3, N).astype(np.int64) % R
    grad = torch.randn(N, C)
    # gather: real indices plus a few out-of-bound ones that must clamp to row R-1
    g_idx = idx_np.copy(); bad = np.array([0, 7, 130, N - 1]); g_idx[bad] = np.array([R, R + 5, 2 ** 18 + 3, 0xFFFFFFF0])
    g_ref = table[:R][np.minimum(g_idx, R - 1)]
    g_u32 = torch.from_numpy(g_idx.astype(np.uint32).reshape(N, 1))
    # scatter: valid plan (rounds, padding rows R + j) plus injected stray slots aimed past the buffer
    perm, sidx, rounds = scatter_plan(idx_np, R)          # padding index = R (row R is padding, in bounds)
    sidx = sidx.astype(np.int64); sidx[5] = R + 500; sidx[300] = 2 ** 20   # must clamp to R + 127 (padding tail)
    keep = np.ones(len(sidx), dtype=bool); keep[[5, 300]] = False
    ref_t = table.clone()
    src = perm.astype(np.int64)[keep]; dst = sidx[keep]
    ref_t.index_add_(0, torch.from_numpy(dst), grad[torch.from_numpy(src)])
    ref_t[R:] = table[R:]                                # padding rows are don't-care: compare the real rows only
    print(f"N={N} rounds={rounds} slots={len(sidx)} (2 stray slots injected)")
    perm_t = torch.from_numpy(perm.reshape(-1, 1)); sidx_t = torch.from_numpy(sidx.astype(np.uint32).reshape(-1, 1))
    if mode == "sim":
        g = nki.simulate(embed_gather_v2)(table[:R].contiguous(), g_u32)
        print("gather max err (incl. 4 clamped rows)", (torch.as_tensor(g).float() - g_ref).abs().max().item())
        t2 = table.clone()
        nki.simulate(embed_scatter_add_v2)(t2, grad, perm_t, sidx_t)
        print("scatter real-rows max err (2 stray slots clamped into the tail)", (t2[:R] - ref_t[:R]).abs().max().item())
    else:
        from torch_neuronx.nki_hop import wrap_nki
        dev = torch.device("neuron")
        gk = wrap_nki(embed_gather_v2); sk = wrap_nki(embed_scatter_add_v2)
        td = table[:R].contiguous().to(dev); id_ = g_u32.to(dev); gd = grad.to(dev)
        g = gk[(1,)](td, id_); torch.neuron.synchronize()
        print("gather max err (incl. 4 clamped rows)", (g.cpu().float() - g_ref).abs().max().item())
        t0 = time.monotonic()
        for _ in range(10): g = gk[(1,)](td, id_)
        torch.neuron.synchronize(); print(f"gather {(time.monotonic() - t0) / 10 * 1000:.2f} ms")
        t2 = table.clone().to(dev); pd = perm_t.to(dev); sd = sidx_t.to(dev)
        sk[(1,)](t2, gd, pd, sd); torch.neuron.synchronize()
        print("scatter real-rows max err", (t2.cpu()[:R] - ref_t[:R]).abs().max().item())
        t0 = time.monotonic()
        for _ in range(10): sk[(1,)](t2, gd, pd, sd)
        torch.neuron.synchronize(); print(f"scatter {(time.monotonic() - t0) / 10 * 1000:.2f} ms ({rounds} rounds, {len(sidx)} slots)")
    print("done")


if __name__ == "__main__":
    main()
