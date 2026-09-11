"""Tier-2 distributed probe: the minimal init_process_group("neuron") + collective.

Every stage prints with flush and a timestamp so a hang localizes to a stage.
Run under torchrun; the driver wraps it in `timeout` so hangs become exit 124.
"""
import os
import time

import torch
import torch.distributed as dist

T0 = time.monotonic()
RANK = int(os.environ.get("RANK", "0"))


def log(msg):
    print(f"[r{RANK} +{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


log("boot")
log("init_process_group(neuron) ...")
dist.init_process_group("neuron")
log("init_process_group DONE")

dev = torch.device("neuron")
x = torch.ones(64, device=dev) * (RANK + 1)
log("tensor on device")

dist.all_reduce(x, op=dist.ReduceOp.SUM)
world = dist.get_world_size()
expect = world * (world + 1) / 2
val = float(x[0].cpu())
log(f"all_reduce DONE val={val} expect={expect} {'OK' if abs(val - expect) < 1e-3 else 'WRONG'}")

dist.barrier()
log("barrier DONE")

y = torch.zeros(4, device=dev)
if RANK == 0:
    y += 7
dist.broadcast(y, src=0)
log(f"broadcast DONE val={float(y[0].cpu())} {'OK' if float(y[0].cpu()) == 7.0 else 'WRONG'}")

dist.destroy_process_group()
log("destroy DONE — DIST_PROBE_SUCCESS")
