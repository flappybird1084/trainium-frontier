#!/usr/bin/env python3
"""Generate v53 = v52 + the BIGRAM TABLE and its sidecar gradient buffer held in bf16.

Why only this one change out of everything built on 2026-09-24: it is the only item in the non-forward/backward
22% of the step that measured a real gain. Paired 300 s screens (Q_ALL vs Q_REF) gave grad_sync 0.057 vs 0.059,
optimizer 0.046 vs 0.049 (gather 0.0264 vs 0.0286), total **0.480 vs 0.485 = -1.0%**. `FLAT_OPT2` (collapsing
the 93-element scalar AdamW from ~23 device launches to ~14) measured 0.0076 vs 0.0075 - nothing - and is
deliberately NOT included; neither is `RS_BATCH` (issuing the collectives back-to-back), also a wash.

The table is 134.2 M of the 218.1 M sharded elements. Holding it in bf16:
  - halves its reduce_scatter (rs_probe.py: 8.89 ms fp32 -> 4.62 ms bf16) with **no cast**, which is exactly why
    `GRAD_BF16` measured as a wash - its full-size cast pass costs ~8.4 ms against an ~8.5 ms saving;
  - lets the parameter all_gather write straight into `p.data`, deleting a ~805 MB/step converting copy.

Numerically the graph sees the same values it already saw: v52 already all_gathers this table in bf16 every
step, so `p.data` already holds bf16-rounded numbers stored in fp32. The fp32 master lives in `plan["pshard"]`
and is untouched. What IS new is that the sidecar accumulates the table's gradient in bf16; with 16384 hashes
over 262144 rows most rows take a single contribution, but the hottest bigram is hit ~30 times (see
`scatter_rounds`' docstring), so those rows carry ~1% of accumulation error. That is the one thing the gate has
to justify against a 1% step-time gain, and it is why this is gated rather than shipped on the screen.

Safe here only because the submission already excludes every `nn.Embedding` weight from the tail EMA
(`ema_pairs` at train.py:3683). A bf16 EMA buffer would make `e.lerp_(p, 1 - 0.997)` a silent no-op, since
0.003 is below bf16's 8-bit mantissa against `e`. **Re-check that before ever extending this to another param.**

The cast is placed before the compile-warmup block (which does its own synthetic forward/backward and its own
setup_sharding) so both traces see bf16 and the run pays ONE compile, not two. Consequence accepted: the fp32
master is seeded from bf16-rounded initial weights, a ~0.4% perturbation of a random normal init.

Run from the repo root:  .venv/bin/python submission_raw/make_v53.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "submission_raw/submission_v52_clean_compile_8way_both.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "submission_raw/submission_v53_clean_bte_bf16.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1, nth=None):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    if nth is None:
        s = s.replace(old, new)
    else:
        i = -1
        for _ in range(nth):
            i = s.index(old, i + 1)
        s = s[:i] + new + s[i + len(old):]
    applied[name] = n if nth is None else 1


# ---------------------------------------------------------------- the two embedding kernels become dtype-generic
# An indirect DMA between mismatched dtypes could reinterpret bytes rather than failing, so the staging buffer
# follows the tensor it is staged from. Both are no-ops for the four fp32 token tables. The _x4 scatter variant
# is deliberately left alone: it only ever sees those four fp32 tables.
sub("gather rows dtype", """            rows = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows, src=table.ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=ixc, indirect_dim=0))""",
    """            rows = nl.ndarray((_NKI_P, C), dtype=table.dtype, buffer=nl.sbuf)   # bf16 bigram table
            nisa.dma_copy(dst=rows, src=table.ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=ixc, indirect_dim=0))""")

sub("scatter rows dtype", """            rows = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows, src=grad.ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=pmc, indirect_dim=0))
            tgt = table.ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=ixc, indirect_dim=0)""",
    """            # `table` here is the gradient ACCUMULATOR and dma_compute adds into it in place, so all three
            # operands must share a dtype
            rows = nl.ndarray((_NKI_P, C), dtype=table.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows, src=grad.ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=pmc, indirect_dim=0))
            tgt = table.ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=ixc, indirect_dim=0)""")

# ---------------------------------------------------------------- the master shard stays fp32
sub("fp32 master", """                pshard = p.data[lo:hi].clone()""",
    """                pshard = p.data[lo:hi].clone().float()   # master stays fp32 even where the replica is bf16""")

# ---------------------------------------------------------------- cast the table, before the warmup trace
sub("bte cast", """        synthetic_y = torch.zeros((device_batch_size, SEQ_LEN), dtype=torch.long).to(device)""",
    """        synthetic_y = torch.zeros((device_batch_size, SEQ_LEN), dtype=torch.long).to(device)
        # The bigram table is 134.2 M of the 218.1 M sharded elements and is only ever read by the sidecar's
        # row-gather kernel, which already emits bf16 rows; the graph therefore sees the values it already saw
        # (v52 all_gathers this table in bf16 every step). Holding it in bf16 halves its reduce_scatter with no
        # cast pass and lets the all_gather write p.data directly. Done HERE so the warmup trace below and the
        # real trace agree and the run pays one compile. nn.Embedding weights are excluded from the tail EMA,
        # which a bf16 EMA buffer would silently break.
        _bte = orig_model.transformer.bte.weight
        _bte.data = _bte.data.to(torch.bfloat16)""")

# ---------------------------------------------------------------- reduce in the gradient's dtype, cast the shard up
sub("bf16 shard reduce", """            elif plan is not None:
                dist.reduce_scatter_tensor(plan["gshard"], p.grad, op=dist.ReduceOp.AVG)""",
    """            elif plan is not None and p.grad.dtype != plan["gshard"].dtype:
                # bf16 gradient buffer, fp32 optimizer shard: reduce at the gradient's dtype (half the bytes,
                # and no cast pass) and cast only the 1/8 result - build_nki_adamw requires an fp32 pair.
                gs = plan.get("gsmall")
                if gs is None:
                    gs = plan["gsmall"] = torch.empty_like(plan["gshard"], dtype=p.grad.dtype)
                dist.reduce_scatter_tensor(gs, p.grad, op=dist.ReduceOp.AVG)
                plan["gshard"].copy_(gs)
            elif plan is not None:
                dist.reduce_scatter_tensor(plan["gshard"], p.grad, op=dist.ReduceOp.AVG)""")

# ---------------------------------------------------------------- a bf16 replica is the all_gather's destination
sub("bf16 replica gather", """            p16 = plan.get("p16")
            if p16 is None:
                p16 = plan["p16"] = torch.empty_like(p.data, dtype=torch.bfloat16)
            dist.all_gather_into_tensor(p16, plan["pshard"].to(torch.bfloat16))
            p.data.copy_(p16)""",
    """            if p.data.dtype == torch.bfloat16:
                # the collective writes the replica directly: ~805 MB/step of converting copy deleted
                dist.all_gather_into_tensor(p.data, plan["pshard"].to(torch.bfloat16))
                continue
            p16 = plan.get("p16")
            if p16 is None:
                p16 = plan["p16"] = torch.empty_like(p.data, dtype=torch.bfloat16)
            dist.all_gather_into_tensor(p16, plan["pshard"].to(torch.bfloat16))
            p.data.copy_(p16)""")

# ---------------------------------------------------------------- the checkpoint the scorer loads stays fp32
sub("checkpoint fp32", """        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},""",
    """        # a bf16 replica is written back as fp32 so the checkpoint is dtype-identical to every earlier
        # version; the values are the same bf16 numbers training used
        "model_state": {k: (v.detach().cpu().float() if v.dtype == torch.bfloat16 else v.detach().cpu())
                        for k, v in model.state_dict().items()},""")

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:24s} x{v}")
