#!/usr/bin/env python3
"""Device benchmark (one core): is the vendor NKI attention (nkilib attention_cte fwd + attention_bwd) at least 15% faster
per step with 2 heads x 256 than with today's 4 heads x 128 (and, for completeness, 8 x 64)?

Real shapes: 16 rows x 1024 tokens per core, bf16, d_model 512 (heads x head_dim = 512), causal + document mask passed as
per-query KV bounds, learned per-head softmax-denominator sink, q pre-scaled so scale = 1.0. The two kernel calls use
EXACTLY the arguments of the vehicle (submission_raw/train3_canon.py: doc_attention_fwd/bwd and build_layer_ops.attn_fwd/
attn_bwd): fwd on (b*h, t, d) with tp_q=tp_k=True, cache_softmax=True, bound_min/max (b*h, t, 1) int32, sink (b*h, 1) fp32;
bwd on HD-major (b, h, d, t) q/k/v/out/dy, lse (b, h, 128, t/128) fp32, sinks_ref (b, h) fp32, bound_min/max (b, t) fp32,
use_causal_mask=True, mixed_precision=True, softmax_scale=1.0.

Timing is kernels only: every input is laid out once, outside the timed region (the vehicle's layout glue is the same number
of elements at every head split). One rep = fwd call + bwd call, then a device sync. 10 warmup reps, N timed reps, medians.
A second, pipelined figure (8 reps per sync) shows how much of the per-rep time is host dispatch.

Correctness: at b=2 rows the kernels' out, dq, dk, dv and dsink are compared with an explicit fp32 torch reference (CPU)
for every head split, before any timing.

usage (device, from ~/tf):  python nki/nki_attn_hd_bench.py [--reps 60] [--cases 4x128,2x256,8x64]
usage (CPU simulator):      <venv with nki + nkilib>/python nki/nki_attn_hd_bench.py --sim [--sim-t 512]
                            (run from a directory other than the repo root: the repo's nki/ dir shadows the nki package)
"""
import argparse
import math
import statistics
import sys
import time

import torch

B_ROWS = 16        # rows per core
T = 1024           # tokens per row
D_MODEL = 512
WARMUP = 10
KILL_SPEEDUP = 0.15


# ---------------------------------------------------------------------------------------------------------------------
# inputs

def make_bounds(b, t, seed):
    """Document bounds per (row, position): [start, end) of the document holding each token, as the vehicle's doc_bounds.
    Row 0: BOS every ~300 tokens (typical); row 1: a single document; other rows: BOS with p = 1/300 plus one at 0."""
    g = torch.Generator().manual_seed(seed)
    is_bos = torch.rand(b, t, generator=g) < (1.0 / 300.0)
    is_bos[:, 0] = True
    if b > 1:
        is_bos[1, :] = False
        is_bos[1, 0] = True
    start = torch.zeros(b, t, dtype=torch.int32)
    end = torch.zeros(b, t, dtype=torch.int32)
    for r in range(b):
        cur = 0
        for p in range(t):
            if bool(is_bos[r, p]):
                cur = p
            start[r, p] = cur
        nxt = t
        for p in range(t - 1, -1, -1):
            end[r, p] = nxt
            if bool(is_bos[r, p]):
                nxt = p
    return start, end


def make_inputs(b, h, t, d, seed):
    """Model-like q/k/v: rms-normed q and k (q pre-scaled by 1/sqrt(d) as the vehicle does), unit-ish v, sink ~ log 4."""
    g = torch.Generator().manual_seed(seed)

    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    q = (rms(torch.randn(b, h, t, d, generator=g)) / math.sqrt(d)).bfloat16()
    k = rms(torch.randn(b, h, t, d, generator=g)).bfloat16()
    v = torch.randn(b, h, t, d, generator=g).bfloat16()
    dy = (torch.randn(b, h, t, d, generator=g) * 0.1).bfloat16()
    sink = (math.log(4.0) + 0.5 * torch.randn(h, generator=g)).float()
    start, end = make_bounds(b, t, seed + 1)
    return q, k, v, dy, sink, start, end


def kernel_layouts(q, k, v, dy, sink, start, end):
    """The exact tensors the vehicle hands the two kernels (fwd: flattened token-major; bwd: HD-major)."""
    b, h, t, d = q.shape
    fwd_args = dict(
        q=q.reshape(b * h, t, d).contiguous(), k=k.reshape(b * h, t, d).contiguous(), v=v.reshape(b * h, t, d).contiguous(),
        sink=sink.float().view(1, h).expand(b, h).reshape(b * h, 1).contiguous(),
        bound_min=start.unsqueeze(1).expand(b, h, t).reshape(b * h, t, 1).contiguous(),
        bound_max=end.unsqueeze(1).expand(b, h, t).reshape(b * h, t, 1).contiguous())
    bwd_args = dict(
        qT=q.transpose(-2, -1).contiguous(), kT=k.transpose(-2, -1).contiguous(), vT=v.transpose(-2, -1).contiguous(),
        dyT=dy.transpose(-2, -1).contiguous(),
        sinks=sink.float().view(1, h).expand(b, h).contiguous(),
        bound_min=start.float().contiguous(), bound_max=end.float().contiguous())
    return fwd_args, bwd_args


def lse_from_fwd(neg_max, recip, b, h):
    # vehicle: lse = -(neg_max + log(recip)), reshaped to (b, h, 128, t/128)
    return (-(neg_max.float() + torch.log(recip.float()))).reshape(b, h, *neg_max.shape[-2:]).contiguous()


def call_fwd(kfwd, fa):
    return kfwd(fa["q"], fa["k"], fa["v"], tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True,
                bound_min=fa["bound_min"], bound_max=fa["bound_max"], sink=fa["sink"])


def call_bwd(kbwd, ba, oT, lse):
    return kbwd(ba["qT"], ba["kT"], ba["vT"], oT, ba["dyT"], lse, sinks_ref=ba["sinks"],
                bound_min=ba["bound_min"], bound_max=ba["bound_max"],
                use_causal_mask=True, mixed_precision=True, softmax_scale=1.0)


# ---------------------------------------------------------------------------------------------------------------------
# reference (fp32, CPU)

def reference(q, k, v, dy, sink, start, end):
    """Causal + same-document attention with a denominator-only sink logit (the vehicle's torch fallback math).
    q is already scaled. Returns out and grads of sum(out * dy) w.r.t. q, k, v, sink."""
    b, h, t, d = q.shape
    qf, kf, vf = (x.float().detach().requires_grad_(True) for x in (q, k, v))
    sf = sink.float().detach().requires_grad_(True)
    j = torch.arange(t).view(1, 1, t)
    i = torch.arange(t).view(1, t, 1)
    allowed = (j >= start.long().unsqueeze(2)) & (j < end.long().unsqueeze(2)) & (j <= i)     # (b, t, t)
    s = (qf @ kf.transpose(-2, -1)).masked_fill(~allowed.unsqueeze(1), float("-inf"))
    s_ext = torch.cat([s, sf.view(1, h, 1, 1).expand(b, h, t, 1)], dim=-1)
    p = torch.softmax(s_ext, dim=-1)[..., :t]
    out = p @ vf
    (out * dy.float()).sum().backward()
    return out.detach(), qf.grad, kf.grad, vf.grad, sf.grad


def rel_err(got, ref):
    got = got.float().cpu()
    ref = ref.float().cpu()
    if not torch.isfinite(got).all():
        return float("inf")
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


def check_case(kfwd, kbwd, h, d, t, b, to_dev, sync, tol=3e-2):
    q, k, v, dy, sink, start, end = make_inputs(b, h, t, d, seed=100 + d)
    ro, rdq, rdk, rdv, rds = reference(q, k, v, dy, sink, start, end)
    fa, ba = kernel_layouts(q, k, v, dy, sink, start, end)
    fa = {n: to_dev(x) for n, x in fa.items()}
    ba = {n: to_dev(x) for n, x in ba.items()}
    out, neg_max, recip = call_fwd(kfwd, fa)
    out = torch.as_tensor(out).cpu() if not torch.is_tensor(out) else out.cpu()
    neg_max = torch.as_tensor(neg_max).cpu() if not torch.is_tensor(neg_max) else neg_max.cpu()
    recip = torch.as_tensor(recip).cpu() if not torch.is_tensor(recip) else recip.cpu()
    out4 = out.reshape(b, h, t, d)
    oT = to_dev(out4.transpose(-2, -1).contiguous().to(q.dtype))
    lse = to_dev(lse_from_fwd(neg_max, recip, b, h))
    res = call_bwd(kbwd, ba, oT, lse)
    sync()
    res = [torch.as_tensor(r).cpu() if not torch.is_tensor(r) else r.cpu() for r in res]
    dq, dk, dv = (r.transpose(-2, -1) for r in res[:3])
    ds = res[3].float().sum(dim=0) if len(res) > 3 else None
    errs = {"out": rel_err(out4, ro), "dq": rel_err(dq, rdq), "dk": rel_err(dk, rdk), "dv": rel_err(dv, rdv),
            "dsink": rel_err(ds, rds) if ds is not None else float("inf")}
    ok = all(e < tol for e in errs.values())
    print(f"CHECK {h}x{d} (b={b}, t={t}): " + " ".join(f"{n} {e:.2e}" for n, e in errs.items())
          + f"  (max|diff|/max|ref|, tol {tol:.0e}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ---------------------------------------------------------------------------------------------------------------------
# device

def run_device(args):
    import torch_neuronx
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores
    from nkilib.core.attention.attention_bwd import attention_bwd
    from nkilib.core.attention.attention_cte import attention_cte

    dev = torch.device("neuron")
    lnc = int(get_logical_neuron_cores())
    grid = (lnc,)
    kfwd = wrap_nki(attention_cte)[grid]
    kbwd = wrap_nki(attention_bwd)[grid]
    sync = torch_neuronx.synchronize
    to_dev = lambda x: x.to(dev)
    cases = [tuple(int(z) for z in c.split("x")) for c in args.cases.split(",")]
    print(f"LNC={lnc} grid={grid} rows={B_ROWS} t={T} d_model={D_MODEL} warmup={WARMUP} reps={args.reps}", flush=True)

    all_ok = True
    for h, d in cases:
        try:
            all_ok &= check_case(kfwd, kbwd, h, d, T, 2, to_dev, sync)
        except Exception as e:
            all_ok = False
            print(f"CHECK {h}x{d}: KERNEL FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)

    results = {}
    for h, d in cases:
        assert h * d == D_MODEL
        try:
            q, k, v, dy, sink, start, end = make_inputs(B_ROWS, h, T, d, seed=7)
            fa, ba = kernel_layouts(q, k, v, dy, sink, start, end)
            fa = {n: to_dev(x) for n, x in fa.items()}
            ba = {n: to_dev(x) for n, x in ba.items()}
            out, neg_max, recip = call_fwd(kfwd, fa)
            oT = out.reshape(B_ROWS, h, T, d).transpose(-2, -1).contiguous()
            lse = lse_from_fwd(neg_max, recip, B_ROWS, h)
            sync()

            def rep_fwd():
                call_fwd(kfwd, fa)

            def rep_bwd():
                call_bwd(kbwd, ba, oT, lse)

            def rep_both():
                call_fwd(kfwd, fa)
                call_bwd(kbwd, ba, oT, lse)

            def timed(fn, n, inner=1):
                for _ in range(WARMUP):
                    fn()
                sync()
                ts = []
                for _ in range(n):
                    t0 = time.perf_counter()
                    for _ in range(inner):
                        fn()
                    sync()
                    ts.append((time.perf_counter() - t0) * 1e3 / inner)
                return statistics.median(ts), min(ts), max(ts)

            r = {"fwd": timed(rep_fwd, args.reps), "bwd": timed(rep_bwd, args.reps), "both": timed(rep_both, args.reps),
                 "both_pipe8": timed(rep_both, max(10, args.reps // 4), inner=8)}
            results[(h, d)] = r
            print(f"TIME {h}x{d}: " + " | ".join(f"{n} med {m:.3f} ms (min {lo:.3f}, max {hi:.3f})" for n, (m, lo, hi) in r.items()),
                  flush=True)
        except Exception as e:
            print(f"TIME {h}x{d}: KERNEL FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)

    base = results.get((4, 128))
    if base is not None:
        for key in [c for c in results if c != (4, 128)]:
            r = results[key]
            s = {n: 1.0 - r[n][0] / base[n][0] for n in r}
            print(f"VS 4x128 {key[0]}x{key[1]}: " + " | ".join(f"{n} {100 * v:+.1f}% faster" for n, v in s.items()), flush=True)
        if (2, 256) in results:
            sp = 1.0 - results[(2, 256)]["both"][0] / base["both"][0]
            sp8 = 1.0 - results[(2, 256)]["both_pipe8"][0] / base["both_pipe8"][0]
            verdict = "GO" if min(sp, sp8) >= KILL_SPEEDUP else ("KILL" if max(sp, sp8) < KILL_SPEEDUP else "SPLIT (see pipe8)")
            print(f"VERDICT 2x256 vs 4x128 fwd+bwd: {100 * sp:+.1f}% (synced) / {100 * sp8:+.1f}% (pipelined), "
                  f"bar {100 * KILL_SPEEDUP:.0f}% -> {verdict}; correctness {'PASS' if all_ok else 'FAIL'}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# CPU simulator

def run_sim(args):
    import nki
    from nkilib.core.attention.attention_bwd import attention_bwd
    from nkilib.core.attention.attention_cte import attention_cte

    sfwd = nki.simulate(attention_cte)
    sbwd = nki.simulate(attention_bwd)
    cases = [tuple(int(z) for z in c.split("x")) for c in args.cases.split(",")]
    ok = True
    for h, d in cases:
        t0 = time.time()
        try:
            ok &= check_case(sfwd, sbwd, h, d, args.sim_t, 1, lambda x: x, lambda: None)
        except Exception as e:
            ok = False
            print(f"CHECK {h}x{d}: SIM FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
    print(f"SIM {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--cases", default="4x128,2x256,8x64")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--sim-t", type=int, default=512)
    a = ap.parse_args()
    (run_sim if a.sim else run_device)(a)
