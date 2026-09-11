#!/usr/bin/env python3
"""Dev tool (ON the box, single process, LNC=2, PYTHONPATH=~/tf): why is the compiled ATTN=nki path slow? Compiles the
vehicle at a small width (cached NEFFs from nki_compile_test.py) and counts Neuron runtime events per training step
(nrt_model_submit = NEFF executions, copy-ins, tensor allocs) for explicit vs nki attention under torch.compile."""
import os, sys, time, json, collections, argparse, importlib.util
ap = argparse.ArgumentParser(); ap.add_argument("--rows", type=int, default=8); ap.add_argument("--n-embd", default="256")
ap.add_argument("--n-head", default="2"); ap.add_argument("--impls", default="nki,explicit"); args = ap.parse_args()
os.environ["NEURON_COMPETITION_R1_SINK"] = "logit"; os.environ["NEURON_COMPETITION_R1_N_EMBD"] = args.n_embd
os.environ["NEURON_COMPETITION_R1_N_HEAD"] = args.n_head
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx, torch._dynamo
from torch.profiler import ProfilerActivity, profile
from torch_neuronx.profiling import NeuronConfig, ProfileMode
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py"))
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
from prepare import ensure_tokenizer, make_dataloader
dev = torch.device("neuron"); t3.COMPUTE_DTYPE = torch.bfloat16; torch.manual_seed(0)
tok = ensure_tokenizer(build_if_missing=False)
x, y, _ = next(iter(make_dataloader(tok, args.rows, t3.SEQ_LEN, "train", torch.device("cpu"))))
xd, yd = x.to(torch.int32).to(dev), y.to(dev)
t3.build_nki_ops()
for impl in args.impls.split(","):
    t3.ATTN_IMPL = impl; t3.STACK_GRAD_MODE = "view"; torch._dynamo.reset()
    model = t3.GPT(t3.build_config()); model.init_weights(); model.to(dev)
    cm = torch.compile(model, backend="neuron", dynamic=False)
    def step():
        model.zero_grad(set_to_none=True); loss = cm(xd, yd); loss.backward(); torch_neuronx.synchronize()
    t0 = time.time(); step(); step(); print(f"[{impl}] warm ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time(); step(); step(); dt = (time.time() - t0) / 2
    out = os.path.expanduser(f"~/tf/runs/nprof_{impl}"); os.makedirs(out, exist_ok=True)
    cfg = NeuronConfig(modes=[ProfileMode.RUNTIME], profile_output_dir=out, max_events_per_nc=200000)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1], experimental_config=cfg) as prof:
        step(); step()
    tf = os.path.join(out, "trace.json"); prof.export_chrome_trace(tf)
    ev = json.load(open(tf)).get("traceEvents", [])
    names = collections.Counter((e.get("cat", ""), e.get("name", "")) for e in ev if e.get("ph") == "X")
    summary = {k: names.get(("privateuse1_runtime", k), 0) // 2 for k in ("model_default", "neuron::alloc::lazy", "neuron::copy::cpu_to_neuron", "neuron::copy::neuron_to_cpu")}
    summary["cpu_ops"] = sum(c for (cat, n), c in names.items() if cat == "cpu_op") // 2
    print(f"[{impl}] compiled fwd+bwd {dt*1000:.0f} ms/step | per-step: {summary}", flush=True)
    names = collections.Counter(e.get("name", "") for e in ev if e.get("ph") == "X" and e.get("cat") == "privateuse1_runtime")
    print(f"[{impl}] top runtime events: {names.most_common(12)}", flush=True)
