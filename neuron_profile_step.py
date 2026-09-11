#!/usr/bin/env python3
"""Dev tool (runs ON the box, single process): profile 2 training steps of the vehicle with the Neuron profiler.
usage: NEURON_LOGICAL_NC_CONFIG=2 PYTHONPATH=~/tf python neuron_profile_step.py [--eager]
Aggregates device/runtime trace events by name so the compiled forward+backward can be attributed per kernel."""
import os, sys, json, time, collections, argparse, importlib.util
ap = argparse.ArgumentParser(); ap.add_argument("--eager", action="store_true"); ap.add_argument("--rows", type=int, default=8); args = ap.parse_args()
os.environ.setdefault("NEURON_COMPETITION_R1_COMPILE", "0" if args.eager else "1")
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
from torch.profiler import ProfilerActivity, profile
from torch_neuronx.profiling import NeuronConfig, ProfileMode
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py"))
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
dev = torch.device("neuron"); t3.COMPUTE_DTYPE = torch.bfloat16
torch.manual_seed(0)
model = t3.GPT(t3.build_config()); model.init_weights(); model.to(dev)
t3.STACK_GRAD_MODE = "function" if args.eager else "view"
run_model = model if args.eager else torch.compile(model, backend="neuron", dynamic=False)
opt = model.setup_optimizer(); opt.init_state()
x = torch.randint(0, 8192, (args.rows, t3.SEQ_LEN), dtype=torch.int32).to(dev); y = torch.randint(0, 8192, (args.rows, t3.SEQ_LEN)).to(dev)
def step(i):
    opt.zero_grad(set_to_none=True); loss = run_model(x, y); loss.backward()
    for g in opt.param_groups:
        g["lr"] = g["initial_lr"]
        if g["kind"] == "muon": g["momentum"] = t3.muon_momentum(i); g["weight_decay"] = t3.WEIGHT_DECAY
    opt.refresh_scalars(); opt.step(); torch_neuronx.synchronize()
t0 = time.time()
for i in range(3): step(i)
print(f"warm steps done ({time.time()-t0:.0f}s incl. compile); timing 3 plain steps...", flush=True)
t0 = time.time(); [step(i) for i in range(3, 6)]; print(f"plain step: {(time.time()-t0)/3:.3f}s", flush=True)
out = os.path.expanduser("~/tf/runs/nprof"); os.makedirs(out, exist_ok=True)
cfg = NeuronConfig(modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME], profile_output_dir=out, max_events_per_nc=200000)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1], experimental_config=cfg) as prof:
    for i in range(6, 8): step(i)
tf = os.path.join(out, "trace.json"); prof.export_chrome_trace(tf); print("trace:", tf, os.path.getsize(tf)//1024, "KB", flush=True)
try:
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=25, top_level_events_only=False))
except Exception as e:
    print("key_averages(device sort) failed:", e); print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
ev = json.load(open(tf)).get("traceEvents", [])
by_name = collections.defaultdict(float); by_cat = collections.defaultdict(float); n = 0
for e in ev:
    if e.get("ph") == "X" and "dur" in e:
        by_name[(e.get("cat", "?"), e.get("name", "?")[:70])] += e["dur"]; by_cat[e.get("cat", "?")] += e["dur"]; n += 1
print(f"\n{n} complete events over 2 steps; total duration by category (ms):")
for c, d in sorted(by_cat.items(), key=lambda kv: -kv[1])[:8]: print(f"  {c:30s} {d/1000:9.1f}")
print("\ntop 30 events by total duration (ms, over 2 steps):")
for (c, nm), d in sorted(by_name.items(), key=lambda kv: -kv[1])[:30]: print(f"  {d/1000:8.1f}  [{c}] {nm}")
