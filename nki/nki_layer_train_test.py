"""Device test (one core): the vehicle's LAYER_IMPL=chain path vs its eager kernel path on one real micro-batch (DEPTH=2,
512-wide, logit sink, NKI attention): loss, stack gradients, scalar gradients, and the per-micro-step time of each path.
usage: python nki_layer_train_test.py"""
import importlib.util, os, time
import torch
for k, v in {"SINK": "logit", "ATTN": "nki", "ROPE": "nki", "NORM": "nki", "MLP": "nki", "MUON": "fused", "ADAMW": "nki",
             "N_EMBD": "512", "N_HEAD": "4", "DEPTH": "2", "BIGRAM": "18", "SINK_IMPL": "lse", "LAYER": "torch", "WCACHE": "1"}.items():
    os.environ["NEURON_COMPETITION_R1_" + k] = v
spec = importlib.util.spec_from_file_location("veh", "submission_raw/train3_submission.py")
veh = importlib.util.module_from_spec(spec); spec.loader.exec_module(veh)
from prepare import ensure_tokenizer, make_dataloader
dev = torch.device("neuron"); torch.manual_seed(0)
tok = ensure_tokenizer(build_if_missing=False)
x, y, _ = next(iter(make_dataloader(tok, 4, veh.SEQ_LEN, "train", torch.device("cpu"))))
xd, yd = x.to(torch.int32).to(dev), y.to(dev)
veh.build_nki_ops(); veh.STACK_GRAD_MODE = "function"
model = veh.GPT(veh.build_config()); model.init_weights(); model.to(dev)
model.refresh_weight_cache() if hasattr(model, "refresh_weight_cache") else None

def run(impl):
    veh.LAYER_IMPL = impl
    for p in model.parameters():
        p.grad = None
    model.refresh_weight_cache()
    loss = model(xd, yd); loss.backward(); torch.neuron.synchronize()
    grads = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    t0 = time.monotonic()
    for _ in range(5):
        for p in model.parameters():
            p.grad = None
        model.refresh_weight_cache()
        l = model(xd, yd); l.backward()
    torch.neuron.synchronize()
    return float(loss.item()), grads, (time.monotonic() - t0) / 5 * 1000

l_t, g_t, ms_t = run("torch")
l_c, g_c, ms_c = run("chain")
print(f"loss torch {l_t:.5f} chain {l_c:.5f} diff {abs(l_t - l_c):.5f}", flush=True)
print(f"time per micro-step (fwd+bwd, DEPTH=2): torch {ms_t:.1f} ms  chain {ms_c:.1f} ms", flush=True)
worst = 0.0
for n in g_t:
    if n in g_c:
        a, b = g_t[n], g_c[n]
        rel = (a - b).abs().max().item() / (a.abs().max().item() + 1e-8)
        worst = max(worst, rel)
        if rel > 0.02:
            print(f"  GRAD MISMATCH {n}: rel {rel:.4f} (|torch| {a.abs().max().item():.3e}, |chain| {b.abs().max().item():.3e})", flush=True)
    else:
        print(f"  grad missing in chain: {n}", flush=True)
print(f"worst grad rel diff {worst:.4f} over {len(g_t)} params", flush=True)
print("done")
