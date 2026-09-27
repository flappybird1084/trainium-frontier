"""Does the train/eval RMSNorm epsilon mismatch matter?

Training normalises inside the fused kernels with eps = finfo(bf16).eps = 0.0078125 (deliberate; every layer
kernel is built with it). The scored forward's `norm()` is F.rms_norm with NO eps, so it takes
finfo(x.dtype).eps -- and x is fp32-typed (bigram_lambda is an fp32 Parameter and promotes the residual stream),
so the scorer uses 1.19e-7. 65,000x apart.

rstd = 1/sqrt(mean(x^2) + eps), so the relative logit-scale error is ~eps / (2*mean(x^2)). It is therefore
entirely a question of the activation scale at the head: at mean(x^2) ~ 75 it is 5e-5 and irrelevant; at
mean(x^2) ~ 0.1 it is 4% and a real systematic loss. Measure both the scale and the exact bpb effect.

usage: .venv/bin/python experimental/eps_probe.py ckpt/v72_final.pt
"""
import argparse
import importlib.util
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
import prepare  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--vehicle", default="submission_raw/submission_v72_clean_dyfuse.py")
ap.add_argument("--tokens", type=int, default=1048576)
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("veh", a.vehicle)
veh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(veh)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = prepare.ensure_tokenizer()

BF16_EPS = float(torch.finfo(torch.bfloat16).eps)
orig_norm = veh.norm
stats = []


def make_norm(eps):
    def f(x):
        if eps is None:
            return orig_norm(x)
        return F.rms_norm(x, (x.size(-1),), eps=eps)
    return f


# ---- 1. the activation scale at the head, which is the whole question -------------------------------------
model = veh.load_for_eval(a.checkpoint, dev)
orig_head = type(model).head_logits


def probing_head(self, x):
    stats.append((float(x.pow(2).mean().item()), str(x.dtype)))
    return orig_head(self, x)


type(model).head_logits = probing_head
loader = prepare.make_eval_dataloader(tok, 4, veh.SEQ_LEN, "public_val", dev)
with torch.no_grad():
    for i, (x, _y, _s) in enumerate(loader):
        model(x)
        if i >= 3:
            break
type(model).head_logits = orig_head
ms = sum(s[0] for s in stats) / len(stats)
print(f"mean(x^2) at the head: {ms:.4f}   dtype {stats[0][1]}   (n={len(stats)} batches)")
print(f"implied relative logit-scale error from eps=0.0078125 vs 1.19e-7: {BF16_EPS / (2 * ms):.3%}")
print("", flush=True)

# ---- 2. the exact bpb effect ------------------------------------------------------------------------------
for label, eps in (("scorer default (fp32 eps)", None), ("bf16 eps 0.0078125 (what TRAINING used)", BF16_EPS)):
    veh.norm = make_norm(eps)
    m = veh.load_for_eval(a.checkpoint, dev)
    with torch.no_grad():
        r = prepare.evaluate_bpb(m, tok, 4, veh.SEQ_LEN, dev, split="public_val", max_tokens=a.tokens)
    print(f"{label:42s} val_bpb {r['val_bpb']:.6f}", flush=True)
    del m
    torch.cuda.empty_cache() if dev.type == "cuda" else None
