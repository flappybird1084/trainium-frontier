"""Single-core microbenchmark of the optimizer phase of the compile 8-way line (v18 config).
Times each component in isolation on one logical core so the 0.073 s/step 'optimizer' phase from S_C8_PROF can be
attributed: full step, Muon kernel per stack, AdamW kernel per table, host->device scalar copies, a sharded-Muon
slice (the per-worker share if the optimizer were split over 8 workers), and the pure-torch fallback.
usage (box, via trn2_job.sh, LNC=1): python nki_opt_bench.py [submission file]"""
import importlib.util, sys, time
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "submission_raw/submission_v18_clean_compile_8way.py"
spec = importlib.util.spec_from_file_location("sub", path)
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)

device = torch.device("neuron")
torch.manual_seed(0)
model = sub.GPT(sub.build_config()).to(device)
opt = model.setup_optimizer()
for g in opt.param_groups:
    g["initial_lr"] = g["lr"]; g["momentum"] = 0.95; g["weight_decay"] = sub.WEIGHT_DECAY
for p in model.parameters():
    p.grad = torch.randn_like(p) * 1e-3
sync = lambda: sub.synchronize(device)

def timeit(name, fn, n=10, warm=3):
    for _ in range(warm): fn()
    sync(); t = time.monotonic()
    for _ in range(n): fn()
    sync(); dt = (time.monotonic() - t) / n
    print(f"{name:48s} {dt*1000:8.2f} ms", flush=True)
    return dt

print(f"NKI_AVAILABLE={sub.NKI_AVAILABLE} use_nki={opt.use_nki} grid={getattr(sub, 'NKI_GRID', None)}")
for g in opt.param_groups:
    shapes = [tuple(p.shape) for p in g["params"]]
    print(f"group kind={g['kind']} n={len(shapes)} shapes={shapes[:6]}")
print()
opt.refresh_scalars()
timeit("refresh_scalars (H2D)", opt.refresh_scalars)
timeit("_scal H2D copy", lambda: sub.MuonAdamW._scal([1.0, 2.0, 3.0, 4.0], device))
te = torch.tensor([0.5], device=device)
timeit("tensor(1).item() host sync", lambda: float(te.item()))
timeit("FULL optimizer.step (nki)", opt.step)
for g in opt.param_groups:
    if g["kind"] == "muon":
        w = g["params"][0]
        timeit(f"muon stack {tuple(w.shape)}", lambda g=g: opt._muon_step(g))
        n = w.shape[0]
        for share in sorted({max(1, -(-n // 8)), max(1, n // 4)}):
            sl = dict(g); sl["params"] = [w[:share].detach().requires_grad_(False)]
            sl["params"][0].grad = w.grad[:share]
            opt._muon_state(sl)
            timeit(f"  muon slice {share}/{n} matrices", lambda sl=sl: opt._muon_step(sl))
    else:
        for p in g["params"]:
            if p.dim() != 2: continue
            gg = dict(g); gg["params"] = [p]
            timeit(f"adamw {tuple(p.shape)}", lambda gg=gg: opt._adamw_step(gg))
        gg = dict(g); gg["params"] = [p for p in g["params"] if p.dim() != 2]
        if gg["params"]:
            timeit(f"adamw {len(gg['params'])} scalar params (torch path)", lambda gg=gg: opt._adamw_step(gg))
opt.use_nki = False
timeit("FULL optimizer.step (torch fallback)", opt.step)
for g in opt.param_groups:
    if g["kind"] == "muon":
        timeit(f"torch muon stack {tuple(g['params'][0].shape)}", lambda g=g: opt._muon_step(g))
print("done")
