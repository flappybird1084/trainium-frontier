"""Probe (box, one core): does this torch-neuronx build handle float8 tensors in eager and inside torch.compile(backend=
"neuron")? Decides the fp8-saved-activations idea and a quantized bigram table. usage: python nki_fp8_probe.py"""
import time, traceback
import torch

dev = torch.device("neuron")
x = torch.randn(2048, 512, dtype=torch.bfloat16, device=dev)
w = torch.randn(512, 2048, dtype=torch.bfloat16, device=dev)
res = {}
for name, f8 in [("e4m3fn", getattr(torch, "float8_e4m3fn", None)), ("e5m2", getattr(torch, "float8_e5m2", None))]:
    if f8 is None:
        res[f"eager_{name}"] = "dtype missing"; continue
    try:
        y = x.to(f8).to(torch.bfloat16); torch.neuron.synchronize()
        err = (y.float() - x.float()).abs().max().item()
        res[f"eager_cast_{name}"] = f"ok max_err {err:.3f}"
    except Exception as e:
        res[f"eager_cast_{name}"] = f"FAIL {type(e).__name__}: {str(e)[:100]}"
    try:
        def fn(a, b):
            return (a.to(f8).to(torch.bfloat16)) @ b
        cf = torch.compile(fn, backend="neuron", dynamic=False)
        y = cf(x, w); torch.neuron.synchronize()
        ref = x @ w
        err = (y.float() - ref.float()).abs().max().item()
        t0 = time.monotonic()
        for _ in range(20): y = cf(x, w)
        torch.neuron.synchronize()
        res[f"compiled_cast_{name}"] = f"ok max_err {err:.2f} {(time.monotonic()-t0)/20*1000:.2f} ms"
    except Exception as e:
        res[f"compiled_cast_{name}"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
try:
    def fn2(a, b):
        return a @ b
    cf2 = torch.compile(fn2, backend="neuron", dynamic=False)
    y = cf2(x, w); torch.neuron.synchronize()
    t0 = time.monotonic()
    for _ in range(20): y = cf2(x, w)
    torch.neuron.synchronize()
    res["compiled_bf16_matmul"] = f"{(time.monotonic()-t0)/20*1000:.2f} ms"
except Exception as e:
    res["compiled_bf16_matmul"] = f"FAIL {type(e).__name__}"
if hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn"):
    try:
        a8 = x.to(torch.float8_e4m3fn); b8 = w.t().contiguous().t().to(torch.float8_e4m3fn)
        s = torch.ones((), device=dev)
        y = torch._scaled_mm(a8, b8, scale_a=s, scale_b=s, out_dtype=torch.bfloat16); torch.neuron.synchronize()
        res["eager_scaled_mm"] = f"ok {(y.float() - (x @ w).float()).abs().max().item():.2f}"
    except Exception as e:
        res["eager_scaled_mm"] = f"FAIL {type(e).__name__}: {str(e)[:120]}"
for k, v in res.items():
    print(f"FP8PROBE {k}: {v}", flush=True)
print("done")
