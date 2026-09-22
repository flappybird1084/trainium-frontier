# MFU profile of v46 (2026-09-21) — the loop is Vector-bound, ~25% MFU on the compiled graphs

Goal: whole-step MFU 15%->25% via a change that DECREASES bpb (faster step -> more tokens).

neuron-profile of the two v46 training NEFFs (trn2_profile_one.sh):
- **G0 (backward, 36MB): wall 200ms, MFU 24.8%.** Engines: Vector **82%**, Tensor 51%, Scalar 45%, GpSimd 24%, DMA 39%.
  Opcodes: MATMUL 166ms | **TENSOR_TENSOR 132ms (Vector)** | DMA_DIRECT2D 75ms | ACTIVATE 60ms | LDWEIGHTS 47ms | COPY 46ms
  | TENSOR_REDUCE 18ms | SCALAR_TENSOR_TENSOR 15ms | CAST 13ms.
- **G1 (forward, 26MB): wall 93ms, MFU 25.0%.** Engines: Scalar 55%, Vector 62%, Tensor 39%, GpSimd 25%. Opcodes:
  MATMUL 58 | DMA 43 | ACTIVATE 32 | TENSOR_TENSOR 28 | COPY 18 | LDWEIGHTS 16.
- Hottest source lines (G0, by nki_source_location): a cluster in `post_attn_fwd_h` (~2531-2571) ≈ 135ms — dominated by
  nc_transpose + tensor_copy pairs (building transposed matmul operands) and the rmsnorm/residual tensor_tensor; plus the
  vendor transpose.py (~23ms) and attention_bwd.py (~32ms).

## Read
- The compiled graphs are ALREADY ~25% MFU. Whole-step (~516ms = fwd_bwd 416 + grad_sync 44 + opt 56) is ~20% because
  grad_sync (comm-bound, lever F closed) + optimizer run at ~0% model-FLOP MFU.
- Both graphs are **Vector-bound** while Scalar/GpSimd are idle. So the both-goals lever = make fwd/bwd faster by
  **rebalancing Vector elementwise work onto Scalar/GpSimd and fusing tensor_tensor pairs into scalar_tensor_tensor**
  (math-neutral: same bpb-per-token, faster step -> more tokens -> lower bpb + higher MFU). Also: cut redundant
  transpose+copy (COPY 46+18ms + transpose 23ms ≈ 87ms is pure layout overhead).
- Throttle note: capture showed throttle_avg_util_limit ~66% — possible power throttling during capture; treat absolute ms
  as indicative, engine RATIOS as reliable.


## Engine-rebalance results (2026-09-21, 300s screens, baseline EB_OFF 556 steps)
- prebwd tensor_tensor -> GpSimd: **device COMPILATION FAILED** (a tensor_tensor in pre_attn_bwd; simulator accepted it).
  This is the kernel that holds the 132ms TENSOR_TENSOR, so the main lever is blocked at the compiler.
- postbwd tensor_tensor -> GpSimd: compiles, **556 = no gain** (post_attn_bwd elementwise is not on the critical path).
- backward tensor_scalar -> Scalar: **559 (+0.5%, noise)**.
Conclusion: moving engines does not unlock the win — the bottleneck op is stuck on Vector. Next: (1) make the failing
pre_attn_bwd tensor_tensor GpSimd-legal, (2) reduce tensor_tensor COUNT by fusing pairs into scalar_tensor_tensor. Measure
with PROFILE=1 (fwd_bwd seconds, ~0.002s resolution), not 300s step-counting (±0.5% noise).


## GpSimd engine-hint: DEAD (2026-09-21, minimal reproducer gp_probe.py on the box)
`nisa.tensor_tensor(..., engine=nisa.engine.gpsimd)` in a wrap_nki kernel FAILS to compile on neuronx-cc for BOTH bf16
and fp32 (vector + default compile fine). The simulator accepts it; the device backend rejects it. The compiler DOES use
GpSimd internally (24% busy in the bwd graph), but we cannot force ops there via the engine hint. => manual Vector->GpSimd
rebalancing is not available. Reducing the backward's 132ms Vector tensor_tensor must come from op-COUNT reduction (limited:
scalar_tensor_tensor only fuses tensor-op-scalar-then-tensor; the RoPE-bwd tensor*tensor products don't fold) or a
structural layout/megakernel change. i.e. the -100ms step goal's fwd_bwd portion needs structure, not hints.

## Coalesced collectives: DEAD (2026-09-22, 8-rank coal_probe.py)
dist._coalescing_manager exists in torch 2.11 but the neuron backend raises "does not implement startCoalescing" — so
grad/param collectives cannot be bucketed without the manual cat (what made FLAT_COMM lose). Comm (grad_sync 44ms +
param-gather 17ms) is irreducible on this backend: latency-bound, no overlap (lever F), no coalescing. (Raw torchrun
needs NEURON_LOGICAL_NC_CONFIG=1 or runtime init fails 'status code 2' on the LNC=2 box.)

## Net: -100ms step goal NOT reachable incrementally (all levers empirically closed this session)
Vector->GpSimd engine hint: device compile FAIL (bf16+fp32). Coalesced comm: backend unimplemented. FLAT_COMM: cat cost.
Optimizer overlap: serialized (lever F). fp8: not matmul-bound + won't compile. Batch: linear. Depth/width/ratio:
optimized. tensor_tensor fusion: limited. Only a structural layer megakernel (fwd+bwd, one kernel) could cut the step
enough — multi-day, low odds (fused-forward already lost to the compiler's cross-kernel scheduling).

## Megakernel moonshot: DEAD ON ARRIVAL (2026-09-22, Stage-1 feasibility, code-level)
User greenlit the megakernel; Stage-1 feasibility killed it before any build. The vendor `_attention_cte` (nkilib) is
structurally HBM-bound: Q is a required HBM positional (`_load_q_tile` only reads `q.ap()` from HBM, no q_sbuf branch);
output is hard-wired to HBM (`out_o_hbm`, `_write_back_o_impl` ends every path with dma_copy to shared_hbm; no SBUF-out
param exists). The `k_cache_sbuf`/`v_cache_sbuf` args are a decode KV-cache path forcing `bs_kv=1` (one sequence),
GQA-broadcast — not a training path. Our training CTE batch is b*NH=64 independent full-length seqs; q+k+v = 48MB > 24MB
SBUF, so 64-seq SBUF residency is physically impossible; and attention_bwd needs q/k/v/out in HBM regardless. So a fused
layer kernel removes launches+glue but NOT the ~18.7ms attention-output DMA round-trip — exactly why LAYER_FUSED lost.
The only route to SBUF-resident attention is our OWN flash-attention (rule #11 NaN history; parity-at-best when it worked).

## FINAL: the 0.42s step goal is unreachable on this stack
Every path closed with hard evidence: incremental levers (GpSimd hint compile-fail, coalescing unimplemented, fusion
limited, fp8 not matmul-bound, batch linear, depth/width optimal, optimizer-overlap serialized) AND the structural
megakernel (vendor attention HBM-bound by construction). v46 (0.9762 local, ~0.52s/step) is the ceiling of this line.
