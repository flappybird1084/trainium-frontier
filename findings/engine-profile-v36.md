# Engine profile of the v36 line (2026-09-10 03:30-04:20 UTC)

NEFFs from the T50_gate compile (v36 = v35 numerics: 10 layers, D=512, 4 heads, MLP 3072, 8-row micro-steps, layer ops +
fused head as NKI custom ops inside the compiled graph), profiled standalone on one core with `neuron-profile capture
--profile-nth-exec=2` and `neuron-explorer view` (summary-text + parquet; `neff_instr_agg.py` aggregation). Raw files in
`runs/trn2/box/prof_v36/`. Tool: `trn2_profile_one.sh <tag> <neff>` (capture + summary + parquet with a timeout, because
the parquet exporter hangs after writing its files on this build; kill it or let the timeout do it).

## Per micro-step (8 rows x 1024 tokens per worker)

| graph | total | PE active | Vector | Scalar | DMA | GpSimd | Sync | mfu |
|---|---|---|---|---|---|---|---|---|
| forward (aa4b8833) | 60.0 ms | 31% | 47% | 47% | 50% | 19% | 29% | 19% |
| backward (f2682e40) | 144.4 ms | 34% | 61% | 49% | 37% | 21% | 9% | 17% |

Two micro-steps per step -> ~408 ms of the ~560 ms step is inside the two graphs; the rest is launch gaps, the sidecar
gather/scatter, grad reduce-scatter (rank skew + collective latency, not bandwidth) and the sharded optimizer + gather.

Instruction time by opcode (sums over engines, overlapping):

| opcode | forward | backward |
|---|---|---|
| MATMUL | 31.3 ms (84k) | 84.3 ms (239k) |
| LDWEIGHTS (stationary loads) | 9.5 ms (83k) | 24.9 ms (221k) |
| DMA_DIRECT2D | 36.3 ms (58k) | 78.7 ms (120k) |
| TENSOR_TENSOR (Vector) | 14.1 ms (49k) | 69.2 ms (144k) |
| ACTIVATE (Scalar) | 16.7 ms (36k) | 29.6 ms (59k) |
| COPY | 8.2 ms | 24.3 ms |
| EVENT_SEMAPHORE | 6.6 ms | 24.9 ms |
| transposes (PE) | 8.3 ms (47k) | 36.0 ms (182k) |

Matmul efficiency: the 128x128x512 matmuls run at ~37 TFLOP/s (fwd 0.82 TFLOP in 22 ms; bwd 1.84 TFLOP in 49 ms); the
128x128x128 ones at ~15 TFLOP/s (bwd 0.54 TFLOP in 35 ms - the vendor attention backward's 128-key blocks and the
chunked weight grads). The fused-MLP microbenchmark reaches 54-59 TFLOP/s bf16, so the in-graph matmuls are 1.5-4x
below what the PE can do, but they are only ~1/3 of the engine time anyway.

## What this says

1. **Not matmul-bound, not bandwidth-bound: latency/dependency-bound with every engine at 30-60%.** The layer matrix
   params are 42M (the 197M is mostly the 2^18-row bigram table), so all dense matmul work per worker-step is ~4.5
   TFLOP = 80-100 ms of the 560 ms step. fp8 double-row (measured x1.30-1.41 on a PE-bound fused MLP, x1.00 on
   HBM-bound plain matmuls) would save ~6 ms/step. Closed.
2. **The backward is the target (144 vs 60 ms), and inside it the Vector engine (61% active, 69 ms of tensor_tensor)**
   plus 182k PE transposes (36 ms) and 221k stationary loads (25 ms). Per layer that is ~12-13 ms of backward vs ~4.5
   ms of forward. The backward of a layer is still a CHAIN (mlp_bwd, rmsnorm_bwd, rope kernels, vendor attention_bwd,
   and six torch matmuls for the Q/K/V/proj dgrad + wgrad with compiler-inserted transposes), while the forward is
   three fused kernels. The eager measurements of Sep 8 (4-row shapes) put pre_bwd at 2.54 ms/layer, attention bwd
   3.22, post_bwd 1.50 - pre_bwd (norm bwd + rope bwd + 6 matmuls) is the largest non-vendor piece and runs ~10x
   above its PE bound (25 GFLOP per micro-step for 10 layers = 0.5 ms of PE at bench rate).
3. **Lever ranking after this profile** (per step, 560 ms):
   - fused `pre_attn_bwd` kernel (norm bwd + rope bwd + Q/K/V dgrad and wgrad in one kernel, mirroring pre_attn_fwd;
     one concatenated [dq|dk|dv] x [Wq;Wk;Wv]^T dgrad matmul, one wgrad matmul with the token axis in PSUM
     accumulation): pre_bwd is ~50 ms per micro-step now; a kernel at forward-like efficiency would be ~15-20 ->
     **~60 ms/step (10%)**. Biggest identifiable item. Then a fused `post_attn_bwd` (mlp_bwd + rmsnorm_bwd + proj
     grads) for another ~20-30 ms/step.
   - the outside-graph pool (~150 ms/step = 27%): launch gaps ~60, grad sync ~55 (rank skew + per-collective latency;
     bf16 and bucketing were washes), optimizer + gather ~50, sidecar scatter ~22. Nothing here has a proven fix yet;
     a per-rank timeline (PROFILE=2 across ranks) would show whether the 55 ms is skew.
   - attention backward: vendor 2.69 ms per layer-microstep = 54 ms/step; our 512-key-block kernel was slower and wrong
     on device (parked). The 128-key blocks are why its matmuls run at 15 TFLOP/s; a correct own kernel could win
     ~20 ms/step at best.
   - packed layer-op outputs (-2%), fp8 (~1%), static DMA unrolling (0%): closed.
4. Speed-to-score on this line: ~1% step time = ~0.0003 val_bpb (DB8 evidence). Reaching ~0.992 local (1.00 portal)
   from 0.9965 by speed alone needs ~15% more throughput-equivalent; the fused backward kernels are the only
   identified path of that size, and they also unlock the depth-12 re-spend (parity at the current speed).
