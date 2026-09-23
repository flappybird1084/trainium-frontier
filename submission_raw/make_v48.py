#!/usr/bin/env python3
"""Generate v48 = v47 + the batched 4-table embedding scatter (SCAT4), hardcoded.

Why: PROFILE2 puts the embedding-gradient scatter at ~21.5 ms of the step, and it is LAUNCH-bound -
`scatter_rounds` emits ~150 index rounds and `EmbedSidecar.scatter` calls the kernel once per round PER TABLE,
so ~150 x 4 token tables = ~600 launches per micro-step. The four token tables (wte, vte, vte_mid, vte2) are
gathered with the SAME index plan, so one launch per round can serve all four and the perm/idx loads and clamps
happen once instead of four times. Verified bit-exact against both the per-table kernel and torch.index_add_.

Run from the repo root:  .venv/bin/python submission_raw/make_v48.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v47_clean_compile_8way_vecdel.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v48_clean_compile_8way_scat4.py")

s = SRC.read_text()

# ---------------------------------------------------------------- 1. the x4 kernel, after embed_scatter_add_v2
lines = s.split("\n")
st = next(i for i, l in enumerate(lines) if l.startswith("    def embed_scatter_add_v2("))
en = next(i for i in range(st + 1, len(lines)) if lines[i].startswith("    @nki.jit") or lines[i].startswith("    def "))
KERNEL = '''    @nki.jit
    def embed_scatter_add_v2_x4(t0: nl.ndarray, t1: nl.ndarray, t2: nl.ndarray, t3: nl.ndarray, g0: nl.ndarray,
                                g1: nl.ndarray, g2: nl.ndarray, g3: nl.ndarray, perm: nl.ndarray, idx: nl.ndarray):
        """embed_scatter_add_v2 for the four token tables in ONE launch. They are gathered with the same indices,
        so a single launch per round serves all four and the perm/idx loads and clamps are done once instead of
        four times: ~150 rounds x 4 tables = ~600 launches per micro-step becomes ~150. The scatter is
        launch-bound (~21.5 ms of the step), and this is bit-identical to four separate embed_scatter_add_v2
        calls (verified in the NKI simulator against torch.index_add_ on all four tables)."""
        S = idx.shape[0]
        Rp = t0.shape[0]
        N = g0.shape[0]
        C = t0.shape[1]
        out = nl.ndarray((1, C), dtype=t0.dtype, buffer=nl.shared_hbm)
        tabs = [t0, t1, t2, t3]
        grds = [g0, g1, g2, g3]
        for t in nl.sequential_range(S // _NKI_P):
            pm = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pm, src=perm[t * _NKI_P:(t + 1) * _NKI_P, :])
            pmc = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pmc, data=pm, op0=nl.minimum, operand0=N - 1)
            ix = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ix, src=idx[t * _NKI_P:(t + 1) * _NKI_P, :])
            ixc = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ixc, data=ix, op0=nl.minimum, operand0=Rp - 1)
            for k in range(4):
                rows = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=rows, src=grds[k].ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=pmc, indirect_dim=0))
                tgt = tabs[k].ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=ixc, indirect_dim=0)
                nisa.dma_compute(dst=tgt, srcs=[tgt, rows], reduce_op=nl.add)
        nisa.dma_copy(dst=out, src=t0[0:1, :])
        return out

'''
lines.insert(en, KERNEL)
s = "\n".join(lines)

# ---------------------------------------------------------------- 2. expose it
old = "        _NKI_EMBED = (wrap_nki(embed_gather_v2), wrap_nki(embed_scatter_add_v2))   # v45: clamped index tiles (findings/levers-def-2026-09-16.md, lever E)"
assert s.count(old) == 1
s = s.replace(old, "        _NKI_EMBED = (wrap_nki(embed_gather_v2), wrap_nki(embed_scatter_add_v2),\n"
                   "                      wrap_nki(embed_scatter_add_v2_x4))   # v45: clamped index tiles; x4: one launch for the 4 token tables", 1)

# ---------------------------------------------------------------- 3. batch the four token tables in the sidecar
# the same header appears in gather(); target the one inside scatter()
_sc = s.index("    def scatter(self, emb, plans):")
old2 = """        for kind, (ind, ind_t, views, _unused) in plans.items():
            for name in self.kinds[kind]:"""
new2 = """        for kind, (ind, ind_t, views, _unused) in plans.items():
            names = self.kinds[kind]
            # The four token tables share one index plan, so scatter them in a single launch per round: the
            # scatter is launch-bound (~150 rounds x 4 tables = ~600 launches per micro-step) and this makes it
            # ~150. Bit-identical to the per-table path below, which stays as the fallback.
            if (self.impl == "sidecar" and len(names) == 4
                    and all(emb[n].grad is not None for n in names)
                    and len({self.gbuf[n].shape for n in names}) == 1):
                try:
                    gs = [emb[n].grad.reshape(-1, emb[n].grad.shape[-1]).contiguous() for n in names]
                    bs = [self.gbuf[n] for n in names]
                    k4 = build_nki_embed()[2]
                    for perm_t, sidx_t in views:
                        k4[(1,)](bs[0], bs[1], bs[2], bs[3], gs[0], gs[1], gs[2], gs[3], perm_t, sidx_t)
                    continue
                except Exception as exc:
                    print0(f"scatter x4 unavailable ({str(exc)[:80]}); per-table path")
            for name in names:"""
_i = s.index(old2, _sc)
s = s[:_i] + new2 + s[_i + len(old2):]

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
