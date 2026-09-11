#!/usr/bin/env python3
"""Dev tool (ON the box, single proc, LNC=1, 4 rows, PYTHONPATH=~/tf): stack-attributed eager op profile of one chain
(LAYER=chain, WCACHE, sidecar embedding) micro-step. Answers two questions OPPROF3/4 could not: (1) is the step host-bound
(enqueue time ~ synced time) or device-bound; (2) which SOURCE LINES of the vehicle issue the thousands of tiny ops."""
import os, sys, time, importlib.util, collections, re
for k, v in {"SINK": "logit", "ATTN": "nki", "MUON": "fused", "ADAMW": "nki", "ADAMW_INPLACE": "1", "ROPE": "nki", "MLP": "nki", "NORM": "nki", "N_EMBD": "512", "N_HEAD": "4", "SINK_IMPL": "lse", "LAYER": "chain", "BIGRAM": "0", "WCACHE": "1"}.items():
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
def fwd_bwd():
    model.refresh_weight_cache(); loss = model(xd, yd); loss.backward(); return loss
def opt_step(i):
    for g in opt.param_groups:
        g["lr"] = g["initial_lr"]
        if g["kind"] == "muon": g["momentum"] = t3.muon_momentum(i); g["weight_decay"] = t3.WEIGHT_DECAY
    opt.refresh_scalars(); opt.step()
def step(i):
    opt.zero_grad(set_to_none=True); fwd_bwd(); opt_step(i); torch_neuronx.synchronize()
for i in range(4): step(i)
# host-vs-device split: enqueue wall time (no sync) vs synced wall time, fwd+bwd only and optimizer only
def split(fn, n=4):
    enq = syn = 0.0
    for _ in range(n):
        torch_neuronx.synchronize(); t0 = time.time(); fn(); t1 = time.time(); torch_neuronx.synchronize(); t2 = time.time()
        enq += t1 - t0; syn += t2 - t0
    return enq / n * 1000, syn / n * 1000
opt.zero_grad(set_to_none=True)
e, s = split(fwd_bwd); print(f"fwd+bwd  : enqueue {e:.0f} ms, synced {s:.0f} ms  -> {'HOST-bound' if e > 0.8 * s else 'device-bound'}", flush=True)
e, s = split(lambda: opt_step(5)); print(f"optimizer: enqueue {e:.0f} ms, synced {s:.0f} ms", flush=True)
t0 = time.time(); [step(i) for i in range(6, 10)]; print(f"eager step: {(time.time()-t0)/4*1000:.0f} ms (single proc, 4 rows, LNC=1, chain+WCACHE, no bigram)", flush=True)
with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
    step(10); step(11)
ka = prof.key_averages(group_by_stack_n=12)
pat = re.compile(r"(train3_submission|nki_eager_opprof5)\.py\((\d+)\)")
by_line = collections.Counter(); t_line = collections.Counter(); by_op_line = collections.Counter()
for e in ka:
    if e.key.startswith("autograd::") or e.key.startswith("ProfilerStep"): continue
    src = "?"
    for fr in (e.stack or []):
        m = pat.search(fr)
        if m: src = f"{m.group(1)[:6]}:{m.group(2)}"; break
    n = e.count // 2; c = (e.self_cpu_time_total or 0) / 2 / 1000
    by_line[src] += n; t_line[src] += c; by_op_line[(e.key, src)] += n
print(f"\n{'source line':16s} {'ops/step':>8s} {'cpu ms':>8s}   top ops")
for src, n in by_line.most_common(45):
    tops = sorted(((k, v) for (k, s), v in by_op_line.items() if s == src), key=lambda kv: -kv[1])[:4]
    print(f"{src:16s} {n:8d} {t_line[src]:8.2f}   " + ", ".join(f"{k.replace('aten::','')}x{v}" for k, v in tops), flush=True)
