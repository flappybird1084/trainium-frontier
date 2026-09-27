"""Does all_to_all work on this Neuron runtime, and what does it cost at MoE shapes?

This is the gating unknown for the only remaining item that reaches 0.9650. An expert-parallel top-1 MoE gives
8x the MLP parameters at UNCHANGED per-rank memory (each rank already stores a full MLP replica; expert
parallelism just makes each different), for the cost of routing tokens to the rank that owns their expert. On the
routed-scaling-law ratio that prices at about -0.011 bpb, against the -0.0031 still needed.

Routing needs an all-to-all. Everything else in the design is ordinary work. So before spending days:

  1. Does `dist.all_to_all_single` exist and lower on the "neuron" backend at all?
  2. What does it cost at the real shapes? Per layer per micro-step we would dispatch (16*1024, 512) bf16 of
     hidden states and combine the same back: 8.4M elements = 16.8 MB each way, 12 layers, forward AND backward
     = 48 all-to-all per step. At the project's measured ~0.2 ms fixed cost per collective that is ~10 ms of
     latency alone before any bytes move.
  3. How does it compare to the collectives we already pay? `grad_sync` is ~60 ms and the parameter all_gather
     ~28 ms per step, so an extra ~25 ms (+6% step = about +0.0017 bpb of throughput cost) is the budget the
     -0.011 has to beat.

Measured against two facts already established on this box, both of which constrain the design:
  - Collectives are SERIALIZED against compute here (ledger F_PROBE): async collective + compute costs the SUM at
    every size. So the all-to-all cannot be hidden behind the MLP it feeds.
  - The step is LATENCY-bound, so the 48 launches matter as much as the 800 MB.

Run on the box with the device free:  cd ~/tf && torchrun --nproc_per_node=8 experimental/a2a_probe.py
Prints a table; changes nothing; touches no submission file.
"""
import time

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401

dist.init_process_group("neuron")
rank, ws = dist.get_rank(), dist.get_world_size()
dev = torch.device("neuron")


def sync():
    torch.neuron.synchronize()


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    sync()
    dist.barrier()
    sync()
    t = time.monotonic()
    for _ in range(iters):
        fn()
    sync()
    return (time.monotonic() - t) / iters * 1e3


def p0(*a):
    if rank == 0:
        print(*a, flush=True)


T, D = 16 * 1024, 512          # tokens per rank per micro-step, model width
p0(f"world={ws}  tokens/rank={T}  width={D}")
p0("")

# ---- 1. does it exist and lower at all? -------------------------------------------------------------------
have = hasattr(dist, "all_to_all_single")
p0(f"dist.all_to_all_single present: {have}")
if not have:
    p0("VERDICT: no all_to_all in this torch.distributed build -> expert parallelism needs a different primitive")
    dist.destroy_process_group()
    raise SystemExit(0)

x = torch.randn((T, D), dtype=torch.bfloat16, device=dev)
out = torch.empty_like(x)
ok = True
try:
    dist.all_to_all_single(out, x)
    sync()
except Exception as exc:                                     # noqa: BLE001
    ok = False
    p0(f"VERDICT: all_to_all_single FAILED to lower on neuron: {type(exc).__name__}: {str(exc)[:200]}")
p0(f"all_to_all_single lowers on neuron: {ok}")
if not ok:
    dist.destroy_process_group()
    raise SystemExit(0)

# correctness: rank r's chunk j must arrive as rank j's chunk r
probe = torch.full((T, D), float(rank), dtype=torch.float32, device=dev)
got = torch.empty_like(probe)
dist.all_to_all_single(got, probe)
sync()
chunk = T // ws
seen = [float(got[i * chunk].float().mean().item()) for i in range(ws)]
p0(f"exchange correctness (rank {rank} should see 0..{ws-1}): {[round(s, 1) for s in seen]}")
p0("")

# ---- 2. cost at the real MoE shapes ----------------------------------------------------------------------
p0(f"{'shape':>18} {'MB/rank':>8} {'ms':>8}   note")
for rows, label in ((T, "dispatch (1 layer)"), (T * 2, "2x capacity factor")):
    a = torch.randn((rows, D), dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    ms = bench(lambda: dist.all_to_all_single(b, a))
    p0(f"{str((rows, D)):>18} {a.numel() * 2 / 1e6:8.1f} {ms:8.3f}   {label}")

# ---- 3. the verdict arithmetic ---------------------------------------------------------------------------
a = torch.randn((T, D), dtype=torch.bfloat16, device=dev)
b = torch.empty_like(a)
one = bench(lambda: dist.all_to_all_single(b, a))
per_step = one * 12 * 2 * 2      # 12 layers x (dispatch+combine) x (fwd+bwd)
p0("")
p0(f"one all_to_all at (16384, 512) bf16 : {one:.3f} ms")
p0(f"48 of them per step (12 layers x dispatch+combine x fwd+bwd): {per_step:.1f} ms")
p0(f"that is {per_step / 424 * 100:.1f}% of the 424 ms step = about {0.0293 * (per_step / 424):+.4f} bpb of throughput cost")
p0(f"the MoE quality prize is about -0.011, so it survives if this stays under ~150 ms")
p0("NOTE: collectives are SERIALIZED against compute on this runtime (ledger F_PROBE), so none of this hides.")

dist.destroy_process_group()
