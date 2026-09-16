"""Lever-F probe (2026-09-16): can a collective overlap eager compute on this runtime, or does the core serialize them?

Run under torchrun x8 (queue line: `F_PROBE 60 FILE=experimental/neuron/cc_overlap_probe.py`). Times, per rank 0:
  A  reduce_scatter_tensor of a stack-sized fp32 gradient (48 x 512 x 512 = 50 MB) alone
  B  a Muon-sized matmul loop alone (~10 ms of TensorEngine work)
  C  async reduce_scatter issued, then the matmul loop, then wait()
  and the same for a bte-sized gradient (262144 x 512 fp32 = 537 MB) and an all_gather_into_tensor.
Verdict: C ~ max(A, B) -> collectives overlap compute (lever F is live); C ~ A + B -> serialized (close lever F).
"""
import os
import time

import torch
import torch.distributed as dist

RANK = int(os.environ.get("RANK", "0"))
REPS = 5


def log(msg):
    if RANK == 0:
        print(f"[probe] {msg}", flush=True)


def sync():
    torch.neuron.synchronize()


def timed(fn):
    sync(); t0 = time.monotonic(); fn(); sync(); return (time.monotonic() - t0) * 1000.0


dist.init_process_group("neuron")
world = dist.get_world_size()
dev = torch.device("neuron")
log(f"world {world}")

# Muon-sized compute: 12 matrices of (3072, 512) times (512, 512), a few rounds (Polar Express does ~5 matmul rounds per matrix)
A = torch.randn(12, 3072, 512, device=dev, dtype=torch.bfloat16)
W = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)


def compute():
    x = A
    for _ in range(6):
        x = torch.matmul(x, W)
    return x


def probe(name, g, out, use_gather=False):
    def coll(async_op=False):
        if use_gather:
            return dist.all_gather_into_tensor(out, g, async_op=async_op)
        return dist.reduce_scatter_tensor(out, g, op=dist.ReduceOp.AVG, async_op=async_op)

    # warm-up (first launches compile / allocate)
    for _ in range(2):
        coll(); compute()
    sync()
    a = sorted(timed(lambda: coll()) for _ in range(REPS))[REPS // 2]
    b = sorted(timed(lambda: compute()) for _ in range(REPS))[REPS // 2]

    def overlapped():
        w = coll(async_op=True)
        compute()
        w.wait()

    c = sorted(timed(overlapped) for _ in range(REPS))[REPS // 2]

    def overlapped_wait_first():   # control: the same, but compute issued after the collective completes
        w = coll(async_op=True)
        w.wait()
        compute()

    d = sorted(timed(overlapped_wait_first) for _ in range(REPS))[REPS // 2]
    verdict = "OVERLAP" if c < 0.8 * (a + b) else "SERIALIZED"
    log(f"{name}: coll {a:7.2f} ms | compute {b:7.2f} ms | async coll + compute {c:7.2f} ms | wait-then-compute {d:7.2f} ms "
        f"| a+b {a + b:7.2f} max {max(a, b):7.2f} -> {verdict}")


g_stack = torch.randn(48, 512, 512, device=dev, dtype=torch.float32)
o_stack = torch.empty(48 // world, 512, 512, device=dev, dtype=torch.float32)
probe("reduce_scatter stack 50MB", g_stack, o_stack)

g_bte = torch.randn(262144, 512, device=dev, dtype=torch.float32)
o_bte = torch.empty(262144 // world, 512, device=dev, dtype=torch.float32)
probe("reduce_scatter bte 537MB", g_bte, o_bte)

s_stack = torch.randn(48 // world, 512, 512, device=dev, dtype=torch.float32)
f_stack = torch.empty(48, 512, 512, device=dev, dtype=torch.float32)
probe("all_gather stack 50MB", s_stack, f_stack, use_gather=True)

# two collectives back to back vs one: does the runtime pipeline collectives among themselves?
def two():
    w1 = dist.reduce_scatter_tensor(o_stack, g_stack, op=dist.ReduceOp.AVG, async_op=True)
    w2 = dist.all_gather_into_tensor(f_stack, s_stack, async_op=True)
    w1.wait(); w2.wait()


sync(); t = sorted(timed(two) for _ in range(REPS))[REPS // 2]
log(f"two async collectives back to back: {t:7.2f} ms")
dist.barrier()
log("PROBE_DONE")
dist.destroy_process_group()
