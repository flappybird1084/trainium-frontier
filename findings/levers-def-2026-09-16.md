# Levers D / E / F - plans and status (2026-09-16, "do all")

## E. The "scatter/gather (vector DGE) out-of-bound" flake - REFRAMED
**Status 13:30:** H1 falsified (E_POISON ran clean: the plan copy is synchronous). Hardening shipped instead: EMBED_KERNEL=v2
(index tiles clamped through a compute op before the indirect DMA; nki/nki_embed_v2.py) gated at parity (E2_V2 0.979443,
same step time) and promoted as **v45** (T62_gate_v45 0.979285). The fault can no longer crash a run; its root cause stays open.
All ten crash logs (`runs/trn2/box/*OOB*`) carry `engine=GPSIMD`, a tiny NEFF (instruction index 34/43/66), a source
location inside the embedded NKI kernel bodies, and `operation=nki_kernel_global`: the fault is in an **eager wrap_nki
kernel**, i.e. the sidecar's `embed_gather` / `embed_scatter_add` (the only eager kernels using `vector_offset`), NOT
in the compiled graph; the other ranks then die in the next all-reduce. Every host-built plan is provably in bounds
(indices < R, padding rows < R+128, chunks <= 32768), so the device is reading index values the host did not write.
Top hypothesis H1: `torch.from_numpy(buf).to(dev)` is a lazy H2D copy and the numpy buffer is freed/re-used (the next
kind's plan, bigram-sized values up to 2^18) before the copy lands - exactly "valid-looking indices over an 8192-row
bound". Tests (vehicle knobs): `PLAN_POISON=1` overwrites the host buffer right after upload (lazy copy -> deterministic
death at step 0; queued as E_POISON, 120 s), `PLAN_KEEP_HOST=N` keeps the last N host buffers alive (the fix if H1 holds;
stress-gate with 2 x depth-14 full runs, historical death rate 2/2). Further hypotheses in order: offset views
materialised by wrap_nki and freed early (upload each view separately); `torch.neuron.synchronize()` not draining
wrap_nki kernels (ring depth); intra-kernel DMA->DGE race (pass indices through a compute op); device-side clamp +
canary as the shippable guard. No deterministic reproducer exists today (A_PACK8 was reclassified as the flake).

## D. K/V-reuse hop (cheaper multi-hop attention)
**CLOSED 2026-09-16 13:21:** K0 passed locally (R_HOPKV 1.183902 = R_HOP 1.183924: K/V reuse keeps the whole gain), but the
decision gate F_HOP44 (full-recompute fused hop on the v44 base, 1800 s) scored 0.985652 = +0.0065 at 0.62 s/step; the
K/V op saves at most ~1/3 of the hop cost (~0.002), leaving +0.004. Multi-hop cannot pay at fixed time on this hardware.
Cost accounting of the fused hop: ~16-17 ms/hop x 6 hops = 0.62 s/step; K/V reuse drops ~32% of the FLOPs but the
attention fwd+bwd (~5 ms) and out-proj (~1.5 ms) are a floor -> ~12 ms/hop, ~0.59 s/step (+13%). At 0.0003/1% that is
+0.004 against a fixed-step gain of -0.0047 (box semantics) -> net ~-0.001, inside the band. Kill test K0 before any
kernel work: eager `FLOW_HOP_KV=1` (the hop reuses the main read's K/V, fresh Q) at 500 fixed steps on the local GPU
vs the plain hop; stop unless the reuse keeps >= 0.004 of the gain. Running (R_HOP / R_HOPKV).

## F. Optimizer / backward overlap
**CLOSED 2026-09-16 09:12 by F_PROBE: the runtime serializes collectives and compute on the core's queue** - async
reduce_scatter (50 MB: 1.16 ms; 537 MB: 9.47 ms) or all_gather (1.09 ms) issued before a 2.9 ms matmul loop costs exactly
the sum (4.10 / 12.30 / 4.09 ms), and two async collectives back to back do not pipeline either (2.66 vs 2.25 ms summed).
No design that overlaps comm with compute can work here; only fewer launches/collectives (latency) or less replicated
compute (D2: shard the MLP stacks, ~1% = inside the band) remain, neither worth a gate.
Step tail after the backward NEFF: ~11 reduce collectives (36-43 ms) + optimizer (~48 ms: Muon 16, AdamW 12, param
all-gathers 17) = 17-19% of the step, of which real compute is ~10 ms; the rest is collective latency and launch gaps.
The earlier "async reduce-scatter: no gain" verdict never tested overlap: `synced_elapsed_frac`'s `.item()` sat between
issue and wait and drained the queue. Everything hinges on whether this runtime can run a collective while the core
computes: `experimental/neuron/cc_overlap_probe.py` (queued as F_PROBE) times a collective alone, a Muon-sized matmul
loop alone, and the async pair. If serialized -> close F. If overlap exists -> D1 "pipelined tail" (issue stack/head
reduce-scatters right after the backward so they overlap the ~180 sidecar scatter launches; move the elapsed broadcast
off the critical path; fold the per-kernel `scal` H2D copies into the single scalar buffer; stage the all-gathers under
the table AdamW), ceiling ~5%, target >= 575 steps in a 300 s warm screen (I_SYNC1b 560); then D2 shard the two MLP
stacks (n=12 padded to 16) to stop replicating their Muon on every rank (~1%).
