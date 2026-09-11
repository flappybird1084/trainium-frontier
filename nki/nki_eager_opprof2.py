#!/usr/bin/env python3
"""Dev tool (ON the box, single proc, LNC=1, 4 rows, PYTHONPATH=~/tf): eager per-op time breakdown of one training step of the
vehicle at the v16 config (all kernels + RoPE), using the torch profiler. Ranks aten ops / NKI launches by total device+host
time so the next fusion targets are chosen by measurement."""
import os, sys, time, importlib.util, collections
for k, v in {"SINK": "logit", "ATTN": "nki", "MUON": "fused", "ADAMW": "nki", "ADAMW_INPLACE": "1", "ROPE": "nki", "MLP": "nki", "NORM": "nki", "N_EMBD": "512", "N_HEAD": "4", "BIGRAM": "18", "SINK_IMPL": "lse"}.items():
    os.environ["NEURON_COMPETITION_R1_" + k] = v
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
from torch.profiler import ProfilerActivity, profile
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py"))
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
from prepare import ensure_tokenizer, make_dataloader
dev = torch.device("neuron"); t3.COMPUTE_DTYPE = torch.bfloat16; torch.manual_seed(0)
tok = ensure_tokenizer(build_if_missing=False)
x, y, _ = next(iter(make_dataloader(tok, 4, t3.SEQ_LEN, "train", torch.device("cpu"))))
xd, yd = x.to(torch.int32).to(dev), y.to(dev)
t3.build_nki_ops(); t3.STACK_GRAD_MODE = "function"
model = t3.GPT(t3.build_config()); model.init_weights(); model.to(dev)
opt = model.setup_optimizer(); opt.init_state()
def step(i):
    opt.zero_grad(set_to_none=True); loss = model(xd, yd); loss.backward()
    for g in opt.param_groups:
        g["lr"] = g["initial_lr"]
        if g["kind"] == "muon": g["momentum"] = t3.muon_momentum(i); g["weight_decay"] = t3.WEIGHT_DECAY
    opt.refresh_scalars(); opt.step(); torch_neuronx.synchronize()
for i in range(4): step(i)
t0 = time.time(); [step(i) for i in range(4, 8)]; print(f"eager step: {(time.time()-t0)/4*1000:.0f} ms (single proc, 4 rows, LNC=1, all kernels)", flush=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
    step(8); step(9)
ka = prof.key_averages()
rows = []
for e in ka:
    dev_t = getattr(e, "self_device_time_total", 0) or 0; cpu_t = e.self_cpu_time_total or 0
    rows.append((e.key, e.count // 2, dev_t / 2 / 1000, cpu_t / 2 / 1000))
rows.sort(key=lambda r: -(r[2] + r[3]))
tot_dev = sum(r[2] for r in rows); tot_cpu = sum(r[3] for r in rows)
print(f"per step: device {tot_dev:.0f} ms, self-cpu {tot_cpu:.0f} ms, {sum(r[1] for r in rows)} events", flush=True)
print(f"{'op':60s} {'n/step':>7s} {'dev ms':>8s} {'cpu ms':>8s}")
for k, n, d, c in rows[:45]:
    print(f"{k[:60]:60s} {n:7d} {d:8.2f} {c:8.2f}", flush=True)
