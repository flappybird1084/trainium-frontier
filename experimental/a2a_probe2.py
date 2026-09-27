"""Is the 3.2 ms all_to_all fixed cost or bandwidth, and is all_to_all special?

a2a_probe.py found all_to_all_single works and costs ~3.2 ms at (16384,512) bf16, and that DOUBLING the bytes
made it slightly FASTER. That is the signature of fixed cost, not bandwidth -- but it could also be an artifact.
This separates the two and puts a2a next to the collectives we already pay, at identical byte counts.

If a2a is ~3.2 ms at 128 rows too -> pure per-launch cost, 48 launches = 154 ms, expert parallelism is dead and
no amount of byte reduction saves it.
If a2a is ~0.2 ms at 128 rows -> bandwidth-bound, and a narrower/lower-precision dispatch is worth designing.
If all_reduce costs the same ~3.2 ms at the same bytes -> a2a is not special, this is just what 17 MB costs here.
"""
import time
import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401

dist.init_process_group("neuron")
rank, ws = dist.get_rank(), dist.get_world_size()
dev = torch.device("neuron")
D = 512


def sync():
    torch.neuron.synchronize()


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    sync()
    t = time.monotonic()
    for _ in range(iters):
        fn()
    sync()
    return (time.monotonic() - t) / iters * 1e3


def p0(*a):
    if rank == 0:
        print(*a, flush=True)


p0(f"{'rows':>8} {'MB/rank':>8} {'a2a ms':>8}")
for rows in (128, 1024, 4096, 16384, 32768, 65536):
    a = torch.randn((rows, D), dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    p0(f"{rows:8d} {a.numel()*2/1e6:8.2f} {bench(lambda: dist.all_to_all_single(b, a)):8.3f}")

p0("")
p0("same bytes (16384x512 bf16 = 16.8 MB/rank), different collective:")
a = torch.randn((16384, D), dtype=torch.bfloat16, device=dev)
b = torch.empty_like(a)
p0(f"{'all_to_all':>16} {bench(lambda: dist.all_to_all_single(b, a)):8.3f} ms")
p0(f"{'all_reduce':>16} {bench(lambda: dist.all_reduce(a)):8.3f} ms")
g = torch.empty((16384*ws, D), dtype=torch.bfloat16, device=dev)
p0(f"{'all_gather':>16} {bench(lambda: dist.all_gather_into_tensor(g, a)):8.3f} ms")
s = torch.empty((16384//ws, D), dtype=torch.bfloat16, device=dev)
p0(f"{'reduce_scatter':>16} {bench(lambda: dist.reduce_scatter_tensor(s, a)):8.3f} ms")

p0("")
p0("tiny, all four -- this is the pure per-launch floor:")
a = torch.randn((128, D), dtype=torch.bfloat16, device=dev)
b = torch.empty_like(a)
p0(f"{'all_to_all':>16} {bench(lambda: dist.all_to_all_single(b, a)):8.3f} ms")
p0(f"{'all_reduce':>16} {bench(lambda: dist.all_reduce(a)):8.3f} ms")

dist.destroy_process_group()
