"""Why is one reduce_scatter 5x more expensive PER ELEMENT than another on this backend?

GRADPROF (Q_GPROF, 2026-09-24) found grad_sync is ~100% collective time, but wildly non-uniform: the two padded
MLP stacks move 50.3M elements in 6.2 ms while the seven even-split params move 167.8M in 112.7 ms. CCPROBE
already measured only ~0.14 ms of FIXED cost per call, so a per-call model cannot explain it - which points at
the cost being SUPERLINEAR in tensor size, the bigram table (134.2M elements, 537 MB) being the outlier.

This probe settles it with no correctness risk: it times `reduce_scatter_tensor` at the model's real shapes, then
times the bigram-sized one split into K contiguous row chunks. If chunking is much faster, the fix is to give the
bigram table an interleaved row ownership so it can be reduced in K chunks (ownership must match on both the
reduce_scatter and the all_gather side, which is why this is worth measuring before building).

Run on the box with the device free:  cd ~/tf && torchrun --nproc_per_node=8 experimental/rs_probe.py
"""
import time, torch, torch.distributed as dist
import torch_neuronx  # noqa: F401

dist.init_process_group("neuron")
rank, ws = dist.get_rank(), dist.get_world_size()
dev = torch.device("neuron")


def sync():
    if hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def bench(fn, reps=20, warm=5):
    for _ in range(warm):
        fn()
    sync()
    dist.barrier()
    t0 = time.monotonic()
    for _ in range(reps):
        fn()
    sync()
    dist.barrier()
    return (time.monotonic() - t0) / reps * 1000.0


def p0(msg):
    if rank == 0:
        print(msg, flush=True)


# ---- 1. per-element cost at the model's real shapes -------------------------------------------------------
p0("== reduce_scatter_tensor at the model's real shapes (fp32 unless noted)")
SHAPES = [
    ("bte          (262144,512)", (262144, 512), torch.float32),
    ("bte bf16     (262144,512)", (262144, 512), torch.bfloat16),
    ("mlp stack    (16,3072,512)", (16, 3072, 512), torch.float32),
    ("attn stack   (48,512,512)", (48, 512, 512), torch.float32),
    ("token table  (8192,512)", (8192, 512), torch.float32),
]
for name, shape, dt in SHAPES:
    src = torch.randn(*shape, device=dev, dtype=dt)
    out = torch.empty((shape[0] // ws,) + tuple(shape[1:]), device=dev, dtype=dt)
    ms = bench(lambda: dist.reduce_scatter_tensor(out, src, op=dist.ReduceOp.AVG))
    n = src.numel()
    p0(f"   {name:28s} {n/1e6:7.1f}M el  {src.numel()*src.element_size()/1e6:7.1f} MB  "
       f"{ms:7.2f} ms  {ms/(n/1e6):6.3f} ms/Mel")
    del src, out

# ---- 2. does splitting the big one into K chunks help? ---------------------------------------------------
p0("== the bigram-sized reduce_scatter split into K contiguous row chunks (timing only; ownership would change)")
R, C = 262144, 512
src = torch.randn(R, C, device=dev, dtype=torch.float32)
for K in (1, 2, 4, 8, 16, 32):
    per_in, per_out = R // K, R // K // ws
    outs = [torch.empty(per_out, C, device=dev, dtype=torch.float32) for _ in range(K)]

    def f(K=K, per_in=per_in, outs=outs):
        for i in range(K):
            dist.reduce_scatter_tensor(outs[i], src[i * per_in:(i + 1) * per_in], op=dist.ReduceOp.AVG)

    ms = bench(f, reps=10, warm=3)
    p0(f"   K={K:3d}  chunk {per_in:7d} rows ({per_in*C*4/1e6:6.1f} MB)  {ms:7.2f} ms total")
    del outs

# ---- 3. is all_gather (the other direction) size-sensitive the same way? ---------------------------------
p0("== all_gather_into_tensor for comparison (this one DID respond to bf16: GATHER_BF16 paid +0.6%)")
for dt in (torch.float32, torch.bfloat16):
    shard = torch.randn(R // ws, C, device=dev, dtype=dt)
    full = torch.empty(R, C, device=dev, dtype=dt)
    ms = bench(lambda: dist.all_gather_into_tensor(full, shard))
    p0(f"   bte {str(dt).split('.')[-1]:9s} {full.numel()*full.element_size()/1e6:7.1f} MB  {ms:7.2f} ms")
    del shard, full

p0("RSPROBE_DONE")
dist.destroy_process_group()
