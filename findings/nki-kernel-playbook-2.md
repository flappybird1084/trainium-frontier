# NKI Kernel Playbook II - What Flies, What Dies

Source: user-supplied reference document (2026-09-06), companion to `nki-trn2-playbook.md` (NOT in this repo). Target:
NeuronCore-v3 / Trainium2 / trn2.3xlarge, written against NKI 0.6.0 / Neuron 2.32.0. **Our box runs NKI 0.5.0 + neuronx-cc
2.26 + nkilib 0.5 (competition beta AMI) - verify every API against the installed SDK before use** ("Box-specific notes" =
what was checked on 2026-09-06). Code here is illustrative, NOT tested by its author.

## 1. The premise, and the two numbers that decide everything

The speedrun mindset transfers (measure wall clock, find the real bottleneck, fuse aggressively, change the model to fit the
hardware); the intuition about which changes are cheap does not. **Ridge point** (compute-bound -> memory-bound):

| Level | Peak | Bandwidth | Ridge |
|---|---|---|---|
| NeuronCore-v3, BF16 | 79 TFLOPS | 368 GB/s (16 DMA engines) | ~215 FLOP/byte |
| NeuronCore-v3, FP8 | 158 TFLOPS | 368 GB/s | ~429 FLOP/byte |
| Trainium2 chip, BF16 | 667 TFLOPS | 2.9 TB/s HBM | ~230 FLOP/byte |

8 cores x 368 GB/s = 2.94 TB/s = HBM: the DMA engines only *just* saturate HBM, so a wasteful access pattern cannot be made
up elsewhere. (H100 ridge ~295 FLOP/byte - familiar target, harder here: no cache.)

**PSUM budget:** 8 banks x 512 fp32 per partition, 2 MiB total, TensorE-write-only. One bank = 128 partitions x 512 fp32 =
256 KiB = exactly one attention score tile [128 queries x 512 keys] fp32; the whole accumulator budget is eight 128x512 tiles
(seven with DMA-transpose-as-PE-transpose).

**SBUF budget in nanogpt terms** (BF16 weights, one transformer layer, vs 28 MiB SBUF):

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
**Rule: run reductions along the FREE axis** - `nisa.tensor_reduce` reduces free axes only (axis 0 unreducible; reduction
axes = the last contiguous free dims).

2.2 **Accumulators are scarce, physical, addressable.** 8 PSUM banks, written only by TensorE; one `nc_matmul` output must fit
one bank (why `moving_fsize <= 512`). Secondary: ScalarE `reduce_regs` (128 x 32-bit, one per lane) - `nisa.activation`
reduces into them (`reduce_cmd`); `nisa.tensor_scalar_reduce` holds a running value in the VectorE accumulator across a chain.

2.3 **128 is physical, not a tuning knob** - SBUF, PSUM, VectorE/ScalarE lanes, PE rows. head_dim=64 uses half the machine; a
64-partition instruction costs as much as a 128-partition one. Escape hatch: TensorE row/column tiling (3.7).

2.4 **Transposes burn the matmul engine:** `nc_transpose` on TensorE = a matmul against identity, so layout planning is a
first-class pass. `nl.matmul(x, y, transpose_x=False)` silently inserts a transpose.

2.5 **No cache: bad access patterns are catastrophic.** Every byte moves under explicit DMA; each transfer pays descriptor
setup plus ~1300 ns cross-engine semaphore-to-start latency. Cliff: `< 256 B/partition` severely overhead-bound, `2
KiB/partition` minimum acceptable, `4 KiB/partition` saturates; a 1536-byte embedding row is below the floor (4.6).

2.6 **No occupancy; heterogeneous pipelining replaces it.** Five independent instruction streams (TensorE, VectorE, ScalarE,
GpSimdE, DMA) hide each other's latency; idle ScalarE during a matmul = throughput permanently lost.

2.7 **~100 cycles of per-instruction overhead:** a `tensor_scalar` reading one element per partition measured 189 ns
initiation interval for one cycle of useful work. Free dim >= 128 elements per instruction, always.

2.8 **Dynamic control flow is a compiler feature.** `nl.fori_loop` / `nl.while_loop`: loop variable read-only (read via
`.ap(scalar_offset=i)`); state in SBUF/HBM mutated in place; `fori_loop` bodies return nothing; `while_loop` bodies return
only the next condition register; tensors declared inside cannot escape; neither inside `nl.no_reorder()`. Divergent dynamic
control flow across LNC cores = undefined behavior.

2.9 **No atomics.** Nearest: `nisa.dma_compute` RMW in the DMA engines (`reduce_op` must be `nl.add`, all `scales` 1.0,
**`unique_indices=True` default - a promise, not a check**). Colliding indices are the normal case in embedding backward
(4.7).

2.10 **Stationary/moving asymmetry, backwards from intuition.** `nc_matmul(dst, stationary, moving)` = `stationary.T @
moving`; LoadStationary is up to 4x faster than MultiplyMoving at equal free size. **LONG tensor -> `stationary`, short ->
`moving`**; `A @ B = (B^T @ A^T)^T`, so swapping transposes the output.

## 3. Kernels that fly

3.1 **ReLU^2 in one instruction.** `nisa.scalar_tensor_tensor` = `(data op0 operand0) op1 operand1` on VectorE; `operand0` a
free compile-time scalar broadcast. `scalar_tensor_tensor(dst=out, data=x, op0=nl.maximum, operand0=0.0, op1=nl.multiply,
operand1=x)` is relu(x)^2; backward = the same call with `operand1=grad_out` (fold the 2 into the downstream GEMM scale).
VectorE is idle during the MLP GEMMs and is 2-4x faster on BF16 operands in SBUF contiguous in the innermost free dim. GELU
would occupy ScalarE (wanted for softmax exp).

3.2 **RMSNorm beats LayerNorm for a non-FLOPs reason:** `bn_stats` caps the free dim at 512 (d=768 = two calls + `bn_aggr`;
the normalized axis must be in the free dim). Chain: `tensor_tensor(x, x, op=multiply)` -> `tensor_reduce(op=add, axis=(1,))`
-> `rsqrt` -> `activation(op=copy, scale=inv)`; `tensor_scalar_reduce` instead fuses the elementwise op and the reduction in
one VectorE instruction (0.6.0: `reduce_res=None` on non-final chain iterations keeps the running sum in the accumulator).
Gotchas: `nl.rsqrt` is GpSimdE since 0.5.0 and rejects PSUM inputs (copy out, or `activation` rsqrt on ScalarE); d must be in
the free dim - d in partitions out of a GEMM is a layout collision (4.1).

3.3 **Muon / Newton-Schulz - the ISA's native shape.** `nc_matmul(dst, stationary=X, moving=X) == X.T @ X`; half of
Newton-Schulz's matmuls are transpose-free. [768,768] BF16 (1.1 MiB) is SBUF-resident: contraction 6x128 in one PSUM bank,
stationary free 6x128, moving free 512+256 (the 256 tile runs at partial efficiency), 72 `nc_matmul` per GEMM, ~15 GEMMs per
orthogonalization in SBUF vs 15 HBM round trips in eager PyTorch. Snags: the Frobenius prescale is a full-matrix reduction
(free-axis reduce -> `tensor_partition_reduce` GpSimd, TensorE stays free -> broadcast back via `nc_matmul` with a ones
stationary + `is_stationary_onezero=True`, or `broadcast_to`); pad n to a multiple of 512 (768 -> 1024 may still win;
measure); distributed Muon needs `nki.collectives` (untestable on trn2.3xlarge). Our stacks: 512x512 x4 per layer (q,k,v,proj)
and 2048x512 / 512x2048 MLP - all multiples of 512.

3.4 **Fused attention: the engine assignment is the kernel.** Check the NKI Library first (Attention CTE/TKG, Segmented CTE,
KV-Parallel Segmented, MXFP8 TKG, Ring). Template:
| Stage | Engine | Instruction |
|---|---|---|
| QK^T, PV | Tensor | `nc_matmul` |
| causal / window mask | GpSimd | `affine_select`, `range_select` |
| row max | Vector | `tensor_reduce(op=nl.max, axis=(1,))` |
| exp(S - max) + row sum | Scalar | `activation(nl.exp, bias=neg_max, scale, reduce_op=nl.add, reduce_res=row_sum)` |
| 1/sum | Vector | `reciprocal` |
| rescale | Scalar | `activation` scale |
Fused softmax core: `tensor_reduce(..., negate=True)` (free) then one `activation` gets the 1/sqrt(d) scale, the max
subtraction and the row sum for free. **`nisa.exponential` is NeuronCore-v4 (Trn3) only**; on Trn2 use the `activation` +
`reduce_op` form (4.3).

3.5 **Fused residual + value-residual + norm.** `x + lam*v0` is one VectorE `scalar_tensor_tensor` (`op0=nl.multiply,
operand0=lam, op1=nl.add, operand1=x`) chained straight into RMSNorm without touching HBM. Never a standalone elementwise
kernel: an unfused residual add has arithmetic intensity 1/6 vs a ridge of 215.

3.6 **RoPE - change the convention.** Interleaved RoPE forces stride-2 access; split-half (rotate first half against second
half, HF Llama style) gives two contiguous partition ranges: `x1 = x[0:d2, :]`, `x2 = x[d2:head_dim, :]` (head_dim in
partitions), then `tensor_tensor(x1, cos, op=multiply)` and `scalar_tensor_tensor(dst=out1, data=x2, op0=nl.multiply,
operand0=sin_bcast, op1=nl.subtract, operand1=t0, reverse1=True)`. Hoist cos/sin tables: load once into SBUF, keep resident
across layers. (Our vehicle already uses split-half RoPE.)

3.7 **TensorE row/column tiling - the head_dim=64 escape hatch.** Two `nc_matmul` calls with `tile_size=(64, 128)` and
`tile_position=(0, 0)` / `(64, 0)` run rows 0-63 and 64-127 in parallel: row tiling 2x64 (NC-v2+) or 4x32 (NC-v3+), column
tiling independently, each start a multiple of its tile size; ~2x recovery on QK^T/PV at head_dim=64, 4x at 32. `double_row`
FP8 cannot combine with column tiling, sparse matmul or transpose mode. (Ours only if we go back to 64-wide heads - the 8x64
screen S_H8b lost on step time for exactly the reason 2.3 predicts.)

3.8 **Logit soft-capping.** `nisa.activation(dst=capped, op=nl.tanh, data=logits, scale=1.0/cap)` - scale/bias are free in
every `activation` (`func(data*scale + bias)`, same cost either way); fold the outer `* cap` into the consumer. The problem is
not the softcap but the tensor it operates on (4.4).

3.9 **FP8 head with double-pumping.** NC-v3 doubles FP8 matmul throughput with contraction 256: stationary [128, 2, 128],
moving [128, 2, 512], `perf_mode=nisa.matmul_perf_mode.double_row`; the major free dim must be exactly 2 (pad K), FP8_E4/E5
may mix, accumulation fp32. FP8_E3 runs at BF16 speed. MX formats: `nisa.quantize_mx` + `nisa.nc_matmul_mx`.

3.10 **Fused optimizer step.** AdamW is pure bandwidth (read p,g,m,v; write p,m,v); fused = one pass. Free dim >= 4 KiB per
partition (1024 BF16 / 512 fp32 elements), all 128 partitions, m/v in BF16 if convergence tolerates it, chain elementwise ops
with `scalar_tensor_tensor` so each value is touched once.

## 4. The near-misses (what breaks, why it is invisible from CUDA, what to do instead)

4.1 **The layout collision: norm wants what matmul does not.** LC#1 (matmul): the contraction axis must be in the partition
dim. LC#2 (everything else): the parallel axis must be. Weights stationary + activations moving gives a GEMM output
[4*hidden, seq], but `bn_stats` needs 4*hidden in the FREE dim -> a PF-transpose on TensorE between every linear and every
norm. Fixes: swap stationary/moving (activations stationary, weights moving) so the output arrives in the layout the norm
wants, free; or keep weights stationary (what the NEXT linear wants) and reduce cross-partition - RMSNorm's sum on GpSimd
(`tensor_partition_reduce`) or on TensorE (ones-vector matmul, 0.8% of TensorE). **Layout is a global property of the operator
chain; plan it end-to-end.**

4.2 **QK-norm hits it hardest.** Q leaves the QKV projection with head_dim in partitions (what QK^T wants), but QK-norm
normalizes over head_dim = a cross-partition reduction. Options: `tensor_partition_reduce` (GpSimd; pragmatic default - the
least-contended engine in attention), ones-matmul (blocks the TensorE pipeline), two transposes, or restructure the QKV
projection so head_dim lands in free (usually right; the QKV and attention GEMMs must be planned together).

4.3 **Online softmax: the accumulator you want does not exist on Trn2.** No `nisa.exponential` (3.4) - use
`activation(nl.exp, bias=-max, reduce_op=nl.add, reduce_res=...)`, which writes the partial sum to SBUF. The running rescale
(`acc *= exp(m_old - m_new)`) must happen on PSUM, which only TensorE writes - evict to SBUF, rescale on Vector/Scalar,
reload, or fold the rescale into a matmul. Use the NKI Library's Attention CTE/TKG, which already solved this; writing your
own, budget the 8 PSUM banks on paper first.

4.4 **The LM head: logits do not fit.** [128 tokens x 50257] fp32 = 24.5 MiB (SBUF is 28 MiB); 99 vocab tiles at
`moving_fsize` 512. Fix = online log-sum-exp over vocab tiles (per token [128,1] running max/sum SBUF-resident; per tile:
`nc_matmul` into one PSUM bank -> tanh softcap -> tile max -> new max -> exp with `reduce_op=add`); one PSUM bank per vocab
tile -> at most 8 tiles in flight = the pipeline depth. Target-logit gather = `nisa.nc_n_gather`
(ceil(elems_per_partition/512) instruction groups). Get BF16 right before FP8. (Our vocab is 8192 = 16 tiles of 512:
[128 x 8192] fp32 = 4 MiB, so the full logit tile DOES fit in SBUF here.)

4.5 **Sliding-window warmup: every window size is a separate binary.** NKI kernels compile AOT with static shapes; each
window size is a full compile (minutes). Fix: bucket 4-8 window sizes (multiples of 128/512) and pre-warm the cache with one
step per bucket; or pass the bound in a register via `range_select`/`affine_select` (one binary, but no skipping of
fully-masked blocks).

4.6 **Embedding lookup is below the DMA floor.** A d=768 bf16 row is 1536 B < the 2 KiB floor (2.5). Fixes:
`nisa.local_gather` with `num_elem_per_idx` (each of the 8 GpSimd cores gathers from its own 16 contiguous SBUF partitions,
and the indices must come from those partitions - not a general gather), batch aggressively, keep the table in SBUF if it fits
(a sorted deduplicated batch vocabulary might), or accept and hide it (it overlaps nothing at the start of forward; if 2% of
step time, move on). `nisa.nc_n_gather` is within-partition only; cross-partition arbitrary gather = indirect DMA.

4.7 **Embedding backward: no atomicAdd.** `nisa.dma_compute` scatter RMW (2.9); repeated tokens are the normal case. Fixes:
sort-by-token then segmented reduce (costs a sort, 4.9); accumulate per tile in SBUF then one scatter per unique index; or
restructure so no scatter is needed.

4.8 **MoE routing:** `nisa.topk` wants a 16-partition snake layout, BF16 in SBUF, partition dim a multiple of 16, cost 4n+K
GpSimd cycles; `nisa.max8` (+ `nisa.nc_find_index8`) for <= 8 candidates in the free dim; the wall is the data-dependent
cross-partition permutation (use the library's MoE / Router Top-K kernels).

4.9 **Sorting:** no sort instruction; the library ships Argsort Unstable; check `nisa.nonzero_with_count` /
`nisa.sequence_bounds` (GpSimd) first.

4.10 **Not differentiable:** a `@nki.jit` kernel is an opaque op - hand-write the backward and the plumbing; backward is ~2x
forward FLOPs, so the win is in the doubly-expensive part. Cheap: fusions that are their own backward (elementwise chains,
residual adds), structurally similar backwards (ReLU^2); expensive: anything with a softmax or reduction - the library earns
its keep there (Blockwise MM Backward, Ring Attention Backward).

4.11 **Dropout determinism:** no counter-based RNG; `nisa.dropout` on VectorE with explicit
`nisa.rand_set_state`/`rand_get_state` per engine; materialize the mask, or do not use dropout (the speedrun does not).

4.12 **Megakernel:** real (NKI Library "Transformer TKG") but only d=768 fits (§1); over the cliff the compiler spills -
**check `spill_save_bytes` in the profile before believing a megakernel helped**; no grid-stride loop (`kernel[2](...)` +
`nl.program_id(0)` across the LNC cores, divergent control flow UB).

4.13 **Small-batch decode** (not our regime) inverts everything: memory-bound weight streaming, weights stationary, MM
initiation floor 64 cycles regardless of N (ours: training, N=8192 tokens per worker).

4.14 **Data-dependent trip counts:** `nl.dynamic_range` and bare `while reg:` are deprecated in 0.6.0; prefer bucketing
shapes, `nisa.sequence_bounds` + the library's Segmented CTE attention, or compute-and-mask with
`affine_select`/`range_select`/`select_reduce` (wasteful but static); real dynamic loops last.

## 5. Speedrun scoreboard (order of attack from a working PyTorch model on Trn2)

Already done on this line, in order: profile first; NKI Library kernels (attention/MLP/RMSNorm-quant/RoPE); fused Muon /
Newton-Schulz (3.3); fused residual + RMSNorm + ReLU^2 MLP (3.1/3.2/3.5); fused optimizer (3.10); layout audit (4.1); fused
LM head + softcap + CE (4.4). Still live: row/column tiling for small heads (3.7), FP8 double-pump (3.9, only after the BF16
head), custom backward kernels (ongoing, 2x FLOPs, no autograd). Deprioritize: custom sort/argsort, scatter-add embedding
backward, fully-dynamic block-sparse attention, anything needing collectives (untestable on this instance).

## 6. Measurement harness

Standalone kernel benchmarking on the box (never used by us — our path is the local simulator plus on-device NEFF
profiling — but it is the only off-model recipe we have): compile in the `nki` env, time in the `nrtpy` env (they cannot
share a venv as of 2.32.0) —
`NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_FRAMEWORK_DEBUG=1 python build_kernel.py` (-> `./compiler_workdir/graph.neff`),
then `source ~/nrtpy_venv/bin/activate; export LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH; python bench.py`,
then `neuron-explorer capture -n ./compiler_workdir/graph.neff -s profile.ntff --profile-nth-exec=2 --enable-dge-notifs`.

Four numbers per iteration (Neuron Explorer Summary): `mfu` (-> 100% for matmul kernels; low = transposes), `mbu` (>= 60% for
memory-bound; low = DMA payloads too small), `engine_active_time_percent` of the bottleneck engine (>= 90%; low = semaphore
waits), `spill_save_bytes / sb_read_bytes` (< 30%; high = SBUF over-committed). Roofline check: algorithmic intensity vs 215
FLOP/byte per core - above it chase engine utilization, below it chase DMA payload size.

## 7. CUDA -> NKI translation table

| CUDA | NKI | Notes |
|---|---|---|
| thread block | - | tiles are 128-partition, fixed |
| shared memory | SBUF, 28 MiB, 128 x 224 KiB | software-managed, explicit DMA |
| `mma.sync` / WMMA | `nisa.nc_matmul` | stationary <= [128,128], moving <= [128,512]; contraction in partitions |
| `cp.async` | `nisa.dma_copy` + compiler semaphores | sync automatic from data deps |
| coalescing (128 B) | >= 2 KiB, ideally 4 KiB per partition | 16x stricter |
| `nvidia-smi` / Nsight | `neuron-ls` / `neuron-explorer` | |

More of the mapping: shuffle/reduces 2.1, no cache 2.5, atomics 2.9, occupancy 2.6, tile quantization 3.7, FP8 3.9,
grid-stride loop and `if (tid < n)` 4.12/4.14, recompile 4.5, autograd 4.10, RNG 4.11.

## Box-specific notes (our trn2.3xlarge, checked 2026-09-06)
- Installed: `nki 0.5.0`, `neuronx-cc 2.26.6360`, `nkilib` (in `~/workspace/native_venv`), `torch-neuronx 2.11.3 (native
  beta)`. `nki.isa` exposes activation, scalar_tensor_tensor, tensor_reduce, tensor_scalar_reduce, nc_matmul, nc_transpose,
  dma_compute, tensor_partition_reduce, affine_select, range_select, bn_stats, activation_reduce, tensor_tensor, dma_copy,
  nc_n_gather, local_gather, reciprocal, iota, select_reduce, max8, nc_stream_shuffle, and also `exponential` (present as a
  symbol, but per this document NC-v4 - do not rely on it).
- `nl.tile_size` needs an active backend (`RuntimeError: No backend set`) outside a kernel - read constants inside a kernel or
  from the docs (128 partitions, PSUM 8 banks x 2 KiB/partition).
- Vendor kernels usable from torch via `torch_neuronx.nki_hop.wrap_nki(kernel)[grid](...)` (works eager and under
  torch.compile): `nkilib.core.attention.attention_cte` (fwd; `bound_min/bound_max` (batch, seqlen_q, 1) per-query KV range =
  sequence packing = our doc mask; `sink` (batch, 1) fp32 per-row denominator logit; `cache_softmax=True` returns neg_max and
  out_sum_recip) and `attention_bwd` (q/k/v/o/dy in [bs, nheads, d, seq] layout, lse [bs, nheads, 128, seq//128], `sinks_ref`
  [bs, nheads] (or [bs, nheads, num_sinks]), bounds [bs, seq_len_q] float32 per BATCH (not per head), `use_causal_mask`,
  `mixed_precision`, `softmax_scale`). The dynamo backend's own SDPA lowering
  (`torch_neuronx/neuron_dynamo_backend/decompositions.py`, `_compute_attention_nki`) only engages with `attn_bias is None`, L
  and S multiples of 512, D <= 128, B*H <= 512 - a dense doc mask defeats it (PR2).
- Forward-only vendor kernels: `nkilib.core.mlp.mlp_cte` (with optional fused norm), `nkilib.core.rmsnorm`,
  `nkilib.core.embeddings.rope`/`rope_hf`, `nkilib.core.cumsum`, `nkilib.core.qkv.qkv_cte`, `output_projection`.
