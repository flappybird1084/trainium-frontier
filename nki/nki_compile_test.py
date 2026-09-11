#!/usr/bin/env python3
"""Dev tool (runs ON the box, single process, NEURON_LOGICAL_NC_CONFIG=2, PYTHONPATH=~/tf): does the vehicle stay ONE
dynamo graph with ATTN=nki under torch.compile(backend="neuron")? Compiles the model at a small width (fast compile),
counts dynamo recompiles/graph breaks, and times compiled fwd+bwd for explicit vs nki attention (SINK=logit both).
usage: python nki_compile_test.py [--rows 8] [--n-embd 256] [--n-head 2]"""
import os, sys, time, argparse, importlib.util, logging
ap = argparse.ArgumentParser(); ap.add_argument("--rows", type=int, default=8); ap.add_argument("--n-embd", default="256")
ap.add_argument("--n-head", default="2"); ap.add_argument("--impls", default="nki,explicit"); args = ap.parse_args()
os.environ["NEURON_COMPETITION_R1_SINK"] = "logit"; os.environ["NEURON_COMPETITION_R1_N_EMBD"] = args.n_embd
os.environ["NEURON_COMPETITION_R1_N_HEAD"] = args.n_head
sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx, torch._dynamo
spec = importlib.util.spec_from_file_location("t3", os.path.expanduser("~/tf/submission_raw/train3_submission.py"))
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
from prepare import ensure_tokenizer, make_dataloader
dev = torch.device("neuron"); t3.COMPUTE_DTYPE = torch.bfloat16; torch.manual_seed(0)
tok = ensure_tokenizer(build_if_missing=False)
x, y, _ = next(iter(make_dataloader(tok, args.rows, t3.SEQ_LEN, "train", torch.device("cpu"))))
xd, yd = x.to(torch.int32).to(dev), y.to(dev)
t3.build_nki_ops()
for impl in args.impls.split(","):
    t3.ATTN_IMPL = impl; t3.STACK_GRAD_MODE = "view"
    torch._dynamo.reset()
    model = t3.GPT(t3.build_config()); model.init_weights(); model.to(dev)
    cm = torch.compile(model, backend="neuron", dynamic=False)
    cnt = torch._dynamo.utils.counters
    t0 = time.time()
    def step():
        model.zero_grad(set_to_none=True); loss = cm(xd, yd); loss.backward(); torch_neuronx.synchronize(); return loss
    l0 = step(); print(f"[{impl}] first step (compile) {time.time()-t0:.0f}s loss {l0.item():.4f}", flush=True)
    step(); t0 = time.time()
    for _ in range(5): l = step()
    dt = (time.time() - t0) / 5
    gb = dict(cnt.get("graph_break", {})); frames = dict(cnt.get("frames", {})); uniq = dict(cnt.get("unique_graphs", {}))
    print(f"[{impl}] compiled fwd+bwd {dt*1000:.0f} ms/step | loss {l.item():.4f} | frames {frames} | unique_graphs {uniq} | graph_breaks {gb}", flush=True)
