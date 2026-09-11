# NKI Kernel Playbook II - What Flies, What Dies

Source: user-supplied reference document (2026-09-06), "companion to nki-trn2-playbook.md" (that companion is NOT in this
repo). Target: NeuronCore-v3 / Trainium2 / trn2.3xlarge, written against NKI 0.6.0 / Neuron 2.32.0.
**Our box runs NKI 0.5.0 + neuronx-cc 2.26 + nkilib 0.5 (competition beta AMI) - verify every API against the
installed SDK before use** (see the "Box-specific notes" section at the end for what was checked on 2026-09-06).
Code in the document is illustrative, NOT tested by its author. Treat every sketch as a starting point to profile.

## 1. The premise, and the two numbers that decide everything

Speedrun mindset transfers: measure wall clock end-to-end, find the real bottleneck, fuse aggressively, be willing to
change the model to fit the hardware. What does not transfer: the intuition about which changes are cheap.

**Ridge point** (compute-bound flips to memory-bound):

| Level | Peak | Bandwidth | Ridge |
|---|---|---|---|
| NeuronCore-v3, BF16 | 79 TFLOPS | 368 GB/s (16 DMA engines) | ~215 FLOP/byte |
| NeuronCore-v3, FP8 | 158 TFLOPS | 368 GB/s | ~429 FLOP/byte |
| Trainium2 chip, BF16 | 667 TFLOPS | 2.9 TB/s HBM | ~230 FLOP/byte |

8 cores x 368 GB/s = 2.94 TB/s = HBM: the DMA engines are provisioned to *just* saturate HBM. No slack - a wasteful
access pattern cannot be made up elsewhere. (H100 ridge ~295 FLOP/byte; the target is familiar, hitting it is harder
because there is no cache.)

**PSUM budget:** 8 banks x 512 fp32 per partition, 2 MiB total, TensorE-write-only.
One PSUM bank = 128 partitions x 512 fp32 = 256 KiB = exactly one attention score tile [128 queries x 512 keys] fp32.
The entire accumulator budget for a kernel is eight 128x512 tiles (seven if DMA-transpose-as-PE-transpose is active).

**SBUF budget in nanogpt terms** (BF16 weights for one transformer layer, vs 28 MiB SBUF):

| Model width | QKV | MLP (both) | + out-proj | verdict |
|---|---|---|---|---|
| d=768 | 3.4 MiB | 9.0 MiB | 13.5 MiB | fits - megakernel territory |
| d=1024 | 6.0 MiB | 16.0 MiB | 24.0 MiB | barely, no room for activations |
| d=1280 | 9.4 MiB | 25.0 MiB | 37.5 MiB | does not fit - must stream |

(Our 512-wide model: QKV 1.5 MiB + MLP 4 MiB + out-proj 0.5 MiB = 6 MiB per layer - comfortably SBUF-resident.)

## 2. The CUDA impedance mismatch - ten deltas (in order of damage)

2.1 **No warp shuffle.** Lanes are SBUF partitions; crossing them costs:
| Mechanism | Engine | Reach | Cost |
|---|---|---|---|
| `nc_stream_shuffle` | Vector | within a 32-partition quadrant only | one VectorE instruction |
| `tensor_partition_reduce` | GpSimd | all 128 | add/average/max/or/and only |
| `nc_matmul` vs a ones-vector | Tensor | all 128 | 1 of 128 PE columns (0.8% of TensorE) |
| `nc_transpose` then free-dim reduce | Tensor | all 128 | a full matmul against identity |
No general cross-partition permutation. **Rule: design so reductions run along the FREE axis.** `nisa.tensor_reduce`
reduces free axes only (axis 0 cannot be reduced; reduction axes must be the last contiguous free dims). Every layout
decision is downstream of this.

2.2 **Accumulators are scarce, physical, addressable.** 8 PSUM banks, only TensorE accumulates into them, a single
`nc_matmul` output must fit one bank (that is why `moving_fsize <= 512`). Second, smaller accumulator: ScalarE
`reduce_regs` (128 x 32-bit, one per lane); `nisa.activation` reduces into them (`reduce_cmd`), and
`nisa.tensor_scalar_reduce` can hold a running value in the VectorE accumulator across a chain.

2.3 **128 is physical, not a tuning knob.** Width of SBUF, PSUM, VectorE/ScalarE lanes, PE rows. head_dim=64 uses
half the machine; a 64-partition instruction costs as much as a 128-partition one. Escape hatch: TensorE row/column
tiling (3.7).

2.4 **Transposes burn the matmul engine.** `nc_transpose` on TensorE is a matmul against identity. Layout planning is
a first-class optimization pass. `nl.matmul(x, y, transpose_x=False)` silently inserts a transpose.

2.5 **No cache: bad access patterns are catastrophic.** Every byte moves under explicit DMA. DMA efficiency cliff:
`< 256 B/partition` severely overhead-bound; `2 KiB/partition` minimum acceptable; `4 KiB/partition` saturates. Each
transfer pays descriptor setup plus ~1300 ns cross-engine semaphore-to-start latency. A 1536-byte embedding row is
below the floor (4.6).

2.6 **No occupancy; heterogeneous pipelining replaces it.** Latency is hidden by software-pipelining across TensorE,
VectorE, ScalarE, GpSimdE plus DMA, each with an independent instruction stream. Idle ScalarE during a matmul is
throughput permanently lost.

2.7 **Per-instruction overhead ~100 cycles inside the kernel.** A `tensor_scalar` reading one element per partition
measured 189 ns initiation interval for one cycle of useful work. Free dim >= 128 elements per instruction, always.

2.8 **Dynamic control flow is a compiler feature.** `nl.fori_loop` / `nl.while_loop`: loop variable read-only (read
via `.ap(scalar_offset=i)`), state in SBUF/HBM mutated in place, `fori_loop` bodies return nothing, `while_loop`
bodies return only the next condition register, tensors declared inside cannot escape, neither inside
`nl.no_reorder()`. Across LNC cores, divergent dynamic control flow is undefined behavior.

2.9 **No atomics.** Nearest: `nisa.dma_compute` read-modify-write in the DMA engines
(`reduce_op` must be `nl.add`, all `scales` 1.0, and **`unique_indices=True` is the default - a promise, not a check**).
Colliding indices are the normal case in embedding backward (4.7).

2.10 **Stationary/moving asymmetry, backwards from intuition.** `nc_matmul(dst, stationary, moving)` computes
`stationary.T @ moving`; LoadStationary runs up to 4x faster than MultiplyMoving over the same free size.
**Put the LONG tensor in `stationary`, the short one in `moving`.** `A @ B = (B^T @ A^T)^T`, so swapping transposes
the output.

## 3. Kernels that fly

3.1 **ReLU^2 in one instruction.** `nisa.scalar_tensor_tensor` computes `(data op0 operand0) op1 operand1` on VectorE,
`operand0` a free compile-time scalar broadcast:
```python
# relu(x)^2 == max(x, 0) * x
nisa.scalar_tensor_tensor(dst=out, data=x, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=x)
# backward: grad_in = 2 * relu(x) * grad_out  (fold the 2 into the downstream GEMM scale)
nisa.scalar_tensor_tensor(dst=grad_in, data=x, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=grad_out)
```
Lands on VectorE (idle during the MLP GEMMs). NC-v3 VectorE gets an automatic 2-4x speedup when operands are BF16,
in SBUF, contiguous in the innermost free dim. GELU would occupy ScalarE (wanted for softmax exp) - ReLU^2 is cheaper
structurally.

3.2 **RMSNorm beats LayerNorm for a non-FLOPs reason.** LayerNorm's `bn_stats` caps the free dim at 512 (d=768 needs
two calls + `bn_aggr`, normalized axis must be in the free dim). RMSNorm is square, sum, rsqrt, scale:
```python
nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)          # x: [128 tokens, d], d in FREE dim
nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))            # VectorE free-axis reduce
nl.rsqrt(inv, ms)                                                    # GpSimdE since 0.5.0
nisa.activation(dst=out, op=nl.copy, data=x, scale=inv)              # per-partition vector scale, ScalarE
```
Better: `nisa.tensor_scalar_reduce` does the elementwise op AND the reduction in one VectorE instruction (0.6.0:
`reduce_res=None` on non-final chain iterations keeps the running sum in the accumulator). Gotchas: `nl.rsqrt` moved
to GpSimdE in 0.5.0 and no longer accepts PSUM inputs (copy out, or use `nisa.activation` rsqrt on ScalarE); this
layout needs d in the free dim - if x came out of a GEMM with d in partitions you have a layout collision (4.1).

3.3 **Muon / Newton-Schulz - the ISA's native shape.** `nc_matmul(dst, stationary=X, moving=X) == X.T @ X`: the Gram
matrix is the instruction's native form, so half of Newton-Schulz's matmuls are transpose-free. A [768,768] BF16
matrix (1.1 MiB) is SBUF-resident: contraction 6x128 accumulated in one PSUM bank, stationary free 6x128, moving free
512+256 (the 256 tile runs at partial efficiency), 72 `nc_matmul` per GEMM, ~15 GEMMs per orthogonalization all in
SBUF vs 15 HBM round trips in eager PyTorch. **Probably the highest-leverage single kernel.** Snags: (1) the
Frobenius prescale is a full-matrix reduction - free-axis reduce, then `tensor_partition_reduce` (GpSimd, keeps
TensorE free), then broadcast back (`nc_matmul` with a ones stationary + `is_stationary_onezero=True`, or
`broadcast_to`); (2) pad n to a multiple of 512 (768 -> 1024 may still win; measure). Distributed Muon needs
`nki.collectives` - untestable on trn2.3xlarge.
(Our stacks: 512x512 (x4 per layer: q,k,v,proj) and 2048x512 / 512x2048 (MLP) - all multiples of 512.)

3.4 **Fused attention: the engine assignment is the kernel.** Check the NKI Library first (Attention CTE/TKG,
Segmented CTE, KV-Parallel Segmented, MXFP8 TKG, Ring). The assignment template for every fused kernel:
| Stage | Engine | Instruction |
|---|---|---|
| QK^T, PV | Tensor | `nc_matmul` |
| causal / window mask | GpSimd | `affine_select`, `range_select` |
| row max | Vector | `tensor_reduce(op=nl.max, axis=(1,))` |
| exp(S - max) + row sum | Scalar | `activation(nl.exp, bias=neg_max, scale, reduce_op=nl.add, reduce_res=row_sum)` |
| 1/sum | Vector | `reciprocal` |
| rescale | Scalar | `activation` scale |
Fused softmax core: `tensor_reduce(..., negate=True)` (free) then one `activation` call gets the 1/sqrt(d) scale, the
max subtraction and the row sum for free. **`nisa.exponential` is NeuronCore-v4 (Trn3) only** - the single most
likely API to mislead; use the `activation` + `reduce_op` form on Trn2.

3.5 **Fused residual + value-residual + norm.** `x + lam*v0` is one VectorE `scalar_tensor_tensor`
(`op0=nl.multiply, operand0=lam, op1=nl.add, operand1=x`), chained straight into RMSNorm without touching HBM.
Never write a standalone elementwise kernel: an unfused residual add has arithmetic intensity 1/6 vs a ridge of 215.

3.6 **RoPE - change the convention.** Interleaved RoPE forces stride-2 access; the split-half convention (rotate first
half against second half, HF Llama style) gives two contiguous partition ranges:
```python
d2 = head_dim // 2; x1 = x[0:d2, :]; x2 = x[d2:head_dim, :]         # head_dim in partitions
nisa.tensor_tensor(dst=t0, data1=x1, data2=cos, op=nl.multiply)
nisa.scalar_tensor_tensor(dst=out1, data=x2, op0=nl.multiply, operand0=sin_bcast, op1=nl.subtract, operand1=t0, reverse1=True)
```
Hoist cos/sin tables: load once into SBUF, keep resident across layers. (Our vehicle already uses split-half RoPE.)

3.7 **TensorE row/column tiling - the head_dim=64 escape hatch.**
```python
nisa.nc_matmul(dst, stationary, moving, tile_size=(64, 128), tile_position=(0, 0))    # rows 0-63
nisa.nc_matmul(dst2, stationary2, moving2, tile_size=(64, 128), tile_position=(64, 0))  # rows 64-127, in parallel
```
Row tiling: 2x64 (NC-v2+) or 4x32 (NC-v3+); column tiling independently; each start a multiple of its tile size.
~2x recovery on QK^T/PV for head_dim=64, 4x for 32. `double_row` FP8 mode cannot combine with column tiling.
(Relevant to us only if we ever go back to 64-wide heads; the 8x64 screen S_H8b lost on step time for exactly the
reason 2.3 predicts.)

3.8 **Logit soft-capping.** `nisa.activation(dst=capped, op=nl.tanh, data=logits, scale=1.0/cap)` - scale/bias are
free in every `activation` (`func(data*scale + bias)`, same cost either way); fold the outer `* cap` into the
consumer. The problem is not the softcap but the tensor it operates on (4.4).

3.9 **FP8 head with double-pumping.** NC-v3 doubles FP8 matmul throughput with contraction 256: stationary
[128, 2, 128], moving [128, 2, 512], `perf_mode=nisa.matmul_perf_mode.double_row`; the major free dim must be
exactly 2 (pad K), FP8_E4/E5 may mix, accumulation fp32. Cannot combine with column tiling, sparse matmul,
transpose mode; FP8_E3 runs at BF16 speed. MX formats: `nisa.quantize_mx` + `nisa.nc_matmul_mx`.

3.10 **Fused optimizer step.** AdamW is pure bandwidth (read p,g,m,v; write p,m,v). Fused = one pass. Rules:
free dim >= 4 KiB per partition (1024 BF16 / 512 fp32 elements), all 128 partitions, m/v in BF16 if convergence
tolerates it, chain elementwise ops with `scalar_tensor_tensor` so each value is touched once.

## 4. The near-misses (what breaks, why it is invisible from CUDA, what to do instead)

4.1 **The layout collision: norm wants what matmul does not.** LC#1 (matmul): the contraction axis must be in the
partition dim. LC#2 (everything else): the parallel axis must be in the partition dim. Weights stationary +
activations moving gives a GEMM output [4*hidden, seq], but `bn_stats` needs 4*hidden in the FREE dim -> a
PF-transpose on TensorE between every linear and every norm. Fixes (from AWS's perf guide): (1) swap
stationary/moving - activations stationary, weights moving - so the output arrives in the layout the norm wants,
free; (2) pick an engine with different layout requirements - RMSNorm's sum can be done cross-partition on GpSimd
(`tensor_partition_reduce`) or on TensorE (ones-vector matmul), so keep weights stationary (what the NEXT linear
wants) and reduce cross-partition. **Layout is a global property of the operator chain; plan it end-to-end.**

4.2 **QK-norm hits it hardest.** Q leaves the QKV projection with head_dim in partitions (what QK^T wants) but QK-norm
normalizes over head_dim = a cross-partition reduction. Options: `tensor_partition_reduce` (GpSimd; pragmatic default,
GpSimd is the least-contended engine in attention), ones-matmul (0.8% of TensorE but blocks the pipeline), two
transposes, or restructure the QKV projection so head_dim lands in free (usually right; needs the QKV GEMM and the
attention GEMM planned together).

4.3 **Online softmax: the accumulator you want does not exist on Trn2.** `nisa.exponential` (exp(src - max) with a
running sum in VectorE reduce registers) is NC-v4 only; on Trn2 use `activation(nl.exp, bias=-max, reduce_op=nl.add,
reduce_res=...)` which writes the partial sum to SBUF. The running rescale (`acc *= exp(m_old - m_new)`) must happen
on PSUM, which only TensorE writes - evict to SBUF, rescale on Vector/Scalar, reload, or fold the rescale into a
matmul. Use the NKI Library's Attention CTE/TKG, which already solved this; if writing your own, budget the 8 PSUM
banks on paper first.

4.4 **The LM head: logits do not fit.** [128 tokens x 50257] fp32 = 24.5 MiB (SBUF is 28 MiB), 99 vocab tiles at
moving_fsize 512. Fix = online log-sum-exp over vocab tiles (running max/sum per token [128,1] SBUF-resident; per
tile: nc_matmul into one PSUM bank -> tanh softcap -> tile max -> new max -> exp with reduce_op=add). One PSUM bank
per vocab tile -> at most 8 tiles in flight = the pipeline depth. Target-logit gather is `nisa.nc_n_gather`
(ceil(elems_per_partition/512) instruction groups). Get BF16 right before FP8.
(Our vocab is 8192 = 16 tiles of 512: [128 x 8192] fp32 = 4 MiB, so the full logit tile DOES fit in SBUF here.)

4.5 **Sliding-window warmup: every window size is a separate binary.** NKI kernels compile AOT with static shapes;
each window size is a full compile (minutes). Fix: bucket 4-8 window sizes (multiples of 128/512), pre-warm the cache
with one step per bucket; or pass the bound in a register via `range_select`/`affine_select` (one binary, but no
skipping of fully-masked blocks).

4.6 **Embedding lookup is below the DMA floor.** A d=768 bf16 row is 1536 B < the 2 KiB floor (4 KiB saturates).
Fixes: `nisa.local_gather` with `num_elem_per_idx` (each of the 8 GpSimd cores gathers from its own 16 contiguous
SBUF partitions and the indices must come from those partitions - not a general gather), batch aggressively, keep
the table in SBUF if it fits (a sorted deduplicated batch vocabulary might), or accept and hide it (it overlaps with
nothing at the start of forward; if 2% of step time, move on). `nisa.nc_n_gather` is within-partition only;
cross-partition arbitrary gather = indirect DMA.

4.7 **Embedding backward: no atomicAdd.** `nisa.dma_compute` scatter RMW with `unique_indices=True` by default;
repeated tokens are the normal case. Fixes: sort-by-token then segmented reduce (costs a sort, 4.9); accumulate per
tile in SBUF then one scatter per unique index; or restructure so no scatter is needed.

4.8 **MoE routing: `nisa.topk` wants a 16-partition snake layout, BF16 in SBUF, partition dim multiple of 16, cost
4n+K GpSimd cycles.** For <= 8 candidates in the free dim use `nisa.max8` (+ `nisa.nc_find_index8` for indices).
The bigger wall is the data-dependent cross-partition permutation - use the library's MoE / Router Top-K kernels.

4.9 **Sorting.** No sort instruction; the library ships Argsort Unstable. Before sorting ask whether
`nisa.nonzero_with_count` or `nisa.sequence_bounds` (GpSimd) covers the case.

4.10 **NKI kernels are not differentiable.** A `@nki.jit` kernel is an opaque op: hand-write the backward and the
plumbing. Backward is ~2x forward FLOPs, so most of the win is in the part that costs double. Fusions that are their
own backward (elementwise chains, residual adds) and structurally similar backwards (ReLU^2) are cheap; anything with
a softmax or reduction is expensive - the library earns its keep there (Blockwise MM Backward, Ring Attention
Backward, ...). (On our box: `nkilib.core.attention.attention_bwd` exists; the MLP kernels are forward-only.)

4.11 **Dropout determinism.** No counter-based RNG; `nisa.dropout` on VectorE with explicit
`nisa.rand_set_state`/`rand_get_state` per engine. Materialize the mask, or do not use dropout (the speedrun does not).

4.12 **The megakernel temptation.** Real (NKI Library "Transformer TKG"), but d=768 fits, d=1024 barely, d=1280 does
not; over the cliff the compiler spills - **check `spill_save_bytes` in the profile before believing a megakernel
helped.** No grid-stride loop: `kernel[2](...)` + `nl.program_id(0)` across the LNC cores, divergent control flow UB.

4.13 **Small-batch decode inverts everything** (memory-bound weight streaming; weights stationary; MM initiation
floor 64 cycles regardless of N). Not our regime (training, N=8192 tokens per worker).

4.14 **Data-dependent trip counts.** `nl.dynamic_range` and bare `while reg:` are deprecated in 0.6.0. Prefer:
bucket the shapes; `nisa.sequence_bounds` + the library's Segmented CTE attention; compute-and-mask with
`affine_select`/`range_select`/`select_reduce` (wasteful but static); real dynamic loops last.

## 5. Speedrun scoreboard (order of attack from a working PyTorch model on Trn2)

| # | Kernel | Effort | Why now |
|---|---|---|---|
| 1 | Profile first - do nothing else | 1 day | You will be wrong about the bottleneck. Everyone is. |
| 2 | Adopt NKI Library kernels for attention, MLP, RMSNorm-quant, RoPE | 2-3 days | Free, tuned, ships a backward (attention). |
| 3 | Fused Muon / Newton-Schulz (3.3) | 3-5 days | Highest win/effort: native ISA shape, tiny working set, 15 round trips eliminated. |
| 4 | Fused residual + RMSNorm + ReLU^2 MLP (3.1, 3.2, 3.5) | 3-5 days | Kills elementwise round trips; uses idle Vector/Scalar engines. |
| 5 | Fused optimizer step (3.10) | 2 days | Pure bandwidth win, low risk. |
| 6 | Layout audit across the layer chain (4.1) | 1 week | Often the biggest single win; costs no FLOPs; needs a profile showing transpose fraction. |
| 7 | Fused LM head + softcap + CE (4.4) | 1-2 weeks | Big win, hard (online LSE over vocab tiles). |
| 8 | Row/column tiling for small heads (3.7) | 3 days | 2-4x on attention matmuls if head_dim < 128. |
| 9 | FP8 head with double-pump (3.9) | 1-2 weeks | Only after #7 works in BF16. |
| 10 | Custom backward kernels | ongoing | 2x FLOPs, 2x work, no autograd. |

Deprioritize: custom sort/argsort, scatter-add embedding backward, fully-dynamic block-sparse attention, anything
needing collectives (untestable on this instance).

## 6. Measurement harness

Compile in the `nki` env, time in the `nrtpy` env (cannot share a venv as of 2.32.0):
```bash
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_FRAMEWORK_DEBUG=1 python build_kernel.py   # -> ./compiler_workdir/graph.neff
source ~/nrtpy_venv/bin/activate; export LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH; python bench.py
neuron-explorer capture -n ./compiler_workdir/graph.neff -s profile.ntff --profile-nth-exec=2 --enable-dge-notifs
```
Four numbers per iteration (Neuron Explorer Summary): `mfu` (-> 100% for matmul kernels; low = transposes),
`mbu` (>= 60% for memory-bound; low = DMA payloads too small), `engine_active_time_percent` of the bottleneck engine
(>= 90%; low = semaphore waits), `spill_save_bytes / sb_read_bytes` (< 30%; high = SBUF over-committed). Profiles
export as Parquet (DuckDB). Roofline check before optimizing: algorithmic intensity vs 215 FLOP/byte per core -
above it chase engine utilization, below it chase DMA payload size.

## 7. CUDA -> NKI translation table

| CUDA | NKI | Notes |
|---|---|---|
| thread block | - | tiles are 128-partition, fixed |
| warp (32 lanes) | 32-partition quadrant | only for `nc_stream_shuffle` |
| `__shfl_down_sync` reduce | `tensor_partition_reduce` (GpSimd) or `nc_matmul` vs ones | vastly more expensive - restructure |
| shared memory | SBUF, 28 MiB, 128 x 224 KiB | software-managed, explicit DMA |
| registers / accumulators | PSUM, 8 banks x 512 fp32 | TensorE-write-only; + 128 `reduce_regs` on ScalarE |
| L2 cache | - | nothing; every byte is an explicit DMA |
| `atomicAdd` | `nisa.dma_compute` RMW | `nl.add` only, `unique_indices=True` default |
| `mma.sync` / WMMA | `nisa.nc_matmul` | stationary <= [128,128], moving <= [128,512]; contraction in partitions |
| FP8 tensor core | `perf_mode=double_row` | [128,2,128] / [128,2,512] layout, K even |
| tile quantization | `tile_size` / `tile_position` | 2x64 or 4x32 sub-arrays, NC-v3 |
| `cp.async` | `nisa.dma_copy` + compiler semaphores | sync automatic from data deps |
| coalescing (128 B) | >= 2 KiB, ideally 4 KiB per partition | 16x stricter |
| `__syncthreads()` | automatic | |
| grid-stride loop | - | `kernel[2]()` + `nl.program_id(0)` |
| occupancy | engine pipelining | 4 heterogeneous engines, manual |
| `if (tid < n)` | `affine_select` / `range_select` / `fori_loop` | |
| `torch.compile` recompile | full AOT compile per shape | minutes - bucket shapes |
| autograd for custom op | - | hand-write the backward |
| `nvidia-smi` / Nsight | `neuron-ls` / `neuron-explorer` | |
| Philox RNG | `rand_set_state` / `rand_get_state` | explicit state |

## Appendix: the five things most likely to burn a week
1. `nisa.exponential` does not exist on Trn2 (NC-v4 only). Use `activation(nl.exp, bias=..., reduce_op=nl.add)`.
2. `nl.matmul(x, y, transpose_x=False)` inserts a transpose. Use `transpose_x=True` with pre-transposed data or `nisa.nc_matmul`.
3. `unique_indices=True` is the default on `dma_compute` - a promise, not a check.
4. `bn_stats` free dim <= 512. RMSNorm sidesteps it.
5. `nisa.topk` wants a 16-partition snake layout and BF16; `nisa.max8` for <= 8 candidates.

## Box-specific notes (our trn2.3xlarge, checked 2026-09-06)
- Installed: `nki 0.5.0`, `neuronx-cc 2.26.6360`, `nkilib` (in `~/workspace/native_venv`), `torch-neuronx 2.11.3 (native beta)`.
  `nki.isa` exposes activation, scalar_tensor_tensor, tensor_reduce, tensor_scalar_reduce, nc_matmul, nc_transpose,
  dma_compute, tensor_partition_reduce, affine_select, range_select, bn_stats, activation_reduce, tensor_tensor,
  dma_copy, nc_n_gather, local_gather, reciprocal, iota, select_reduce, max8, nc_stream_shuffle, and also
  `exponential` (present as a symbol; per this document it targets NC-v4 - do not rely on it).
- `nl.tile_size` needs an active backend (`RuntimeError: No backend set`) outside a kernel - read constants inside a
  kernel or from the docs (128 partitions, PSUM 8 banks x 2 KiB/partition).
- Vendor kernels usable from torch via `torch_neuronx.nki_hop.wrap_nki(kernel)[grid](...)` (works eager and under
  torch.compile): `nkilib.core.attention.attention_cte` (fwd; `bound_min/bound_max` (batch, seqlen_q, 1) per-query
  KV range = sequence packing = our doc mask; `sink` (batch, 1) fp32 per-row denominator logit; `cache_softmax=True`
  returns neg_max and out_sum_recip) and `attention_bwd` (q/k/v/o/dy in [bs, nheads, d, seq] layout, lse
  [bs, nheads, 128, seq//128], `sinks_ref` [bs, nheads] (or [bs, nheads, num_sinks]), bounds [bs, seq_len_q] float32
  per BATCH (not per head), `use_causal_mask`, `mixed_precision`, `softmax_scale`). The dynamo backend's own SDPA
  lowering (`torch_neuronx/neuron_dynamo_backend/decompositions.py`, `_compute_attention_nki`) only engages with
  `attn_bias is None`, L and S multiples of 512, D <= 128, B*H <= 512 - a dense doc mask defeats it (PR2).
- Forward-only vendor kernels: `nkilib.core.mlp.mlp_cte` (with optional fused norm), `nkilib.core.rmsnorm`,
  `nkilib.core.embeddings.rope`/`rope_hf`, `nkilib.core.cumsum`, `nkilib.core.qkv.qkv_cte`, `output_projection`.
