# Measure EXCLUSIVE engine time, not instructions (2026-09-25)

The one analysis that has been predictive on this machine. It replaces element counting
(`vector-deletions-2026-09-23.md`), instruction counting (`instruction-profile-2026-09-24.md`, whose title I had
to retract) and engine-utilisation percentages (`mfu-profile-2026-09-21.md`).

## The question those all got wrong

Two measured facts that look contradictory:
- `MLP_WG_GROUP` deleted 55,296 vector instructions -> **-5.6% step**.
- `ROPE_WIDE` deleted 176,640 vector instructions (38% of ALL of them), bit-identical -> **exactly 0**.

No count-based model explains both. The resolution: **the step is ~70% overlapped**, so work on a busy engine is
usually hidden behind another engine and costs no wall clock at all.

## The method

The NEFF profile has `start_ts` / `end_ts` / `engine` per instruction. So:

1. Merge each engine's instruction intervals into disjoint busy intervals.
2. Sweep-line all endpoints; keep the stretches where **exactly one** engine is busy.
3. For each instruction, intersect its interval with its own engine's exclusive stretches, and attribute that
   overlap to its `nki_source_location`.

That is the time where deleting the work shortens the WALL. Scripts: `experimental/critpath.py` and
`experimental/excl_attrib.py` (point them at a `runs/prof_<TAG>/parquet` directory).

**Caveat that matters:** the attribution over-counts on the Tensor engine, because `LDWEIGHTS` overlaps `MATMUL`
in the PE pipeline and both get credited the same window. Scale the per-line numbers to the sweep's true
exclusive total (e.g. 35.4 ms attributed vs 15.4 ms true => multiply by 0.44).

## What it found immediately

Backward graph, 199.9 ms span: Vector busy 139.7 ms but **exclusive only 27-31 ms**. Attributing that put
**16.6 ms - 53% of it - on ONE line**: the per-layer `stack[idx]` weight slice, which autograd lowers as
`select_backward` = `zeros_like(the WHOLE stack)` + scatter + full-stack add, 72 times per step. Confirmed by
size: 15,506 MEMSET instructions x (128, 512) = 1.016e9 elements vs 1.057e9 predicted from the stack shapes.

Fix (v55): one `torch.unbind(stack.to(bf16))` per stack per forward. Bit-identical - layer *i*'s slice touches a
disjoint part of the stack, so `torch.stack` of the 12 grads is exactly what the 12 scatter-adds produce.
**Predicted +3.5% steps, measured +3.37%, gate 0.971824 -> 0.970871.** First step-time prediction on this
project to match its measurement.

It also **refuted a claim written in the code**: `_StackSliceLinear`'s docstring calls this allocation one
"which compile fuses away". It does not - the profile shows it inside the compiled graph, on the critical path.

## And what it closed, cheaply

- **The vendor attention backward: ~3.5 ms exclusive** against 101.5 ms of instruction time - fully overlapped.
  On 2026-09-24 I called replacing it the only remaining structural lever and a multi-day build. It is worth
  almost nothing. One measurement instead of days.
- Deeper MLP weight-grad grouping (`MLP_WG_GROUP=8`): compiles, gains nothing.
- `BACKOUT`: dead wash once a same-session control removed a ~1.5% machine drift. **Always pair an arm with a
  control in the SAME session; never compare bpb across sessions.**
- `BIGRAM_POS=8`: +0.0117 (fragments the hash table's capacity ~8x).

## Where the step stands after v55

| graph | span | exclusive | idle | top item |
|---|---|---|---|---|
| backward | 200 ms | 42.9 ms (23.6%) | 4.9 ms | ~3.6 ms (mlp_bwd matmuls) |
| forward | 103 ms | 22.1 ms (21.4%) | **8.1 ms (7.8%)** | ~4.4 ms (out-proj matmul, v50:1444) |

The forward is **Scalar**-dominated (61% busy), the backward Vector-dominated - they need different treatment.

**No concentration remains.** 0.9700 needs ~10 ms = 3-5 separate kernel surgeries at ~0.0002 each.

The one structurally different lever left is the forward's **8.1 ms of pure idle** - inter-kernel launch/sync
gaps, ~48 kernel launches per forward at ~0.17 ms each. Recovering it means FUSING pre_attn + attention +
post_attn into fewer kernels (the megakernel direction). Multi-day, but it is the only remaining item of the
right magnitude and it is measured rather than guessed.

Second-best, concrete and much smaller: `post_attn_fwd_h`'s out-projection passes the ACTIVATION as `stationary`
and the weight as `moving`, so the PE array reloads every tile (LDWEIGHTS:MATMUL is 1:1). Swapping the operands
so the weight stays resident across token tiles is worth up to ~4.4 ms.
