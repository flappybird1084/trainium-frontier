#!/usr/bin/env python3
"""Device test (single core): what does pinning a kernel's PSUM->SBUF drains to a chosen engine do to its WALL TIME?
Measured on two LIVE kernels - mlp_fwd and mlp_bwd (nki/nki_mlp.py, the same bodies the submission runs) - at the
production shape T=16384 x D=512 x H=3072, without editing either kernel.

WHY. On our attention backward this was worth -17.5% (SCALAR vs VECTOR, replicated to +-0.5%, output bit-identical):
    ours bwd DEFAULT 2.103 / 1.873 ms   SCALAR 1.836 / 1.825 ms   VECTOR 2.225 / 2.213 ms      (ATTNPLACE2/3, B=32)
Note DEFAULT moved 12% between two runs while both PINNED variants held to 0.6%: the compiler's own engine choice for
PSUM drains is not stable run to run. Our LIVE kernels contain 50 such drain sites and pass `engine=` on NONE of them.
The vendor's attention_bwd pins all of its large drains to nisa.scalar_engine, which is part of why it is now ~2x
faster than it was on 2026-09-10.

HOW, and why there is no copied kernel body here. A module-level one-line shim replaces nisa.tensor_copy and forces
`engine=`, and each variant is a separately-jitted RENAMED CLONE of the live kernel's own function object, so a variant
cannot differ from the shipped kernel in anything except the engine argument. That makes this a reusable probe: point
it at any kernel we own and it reports the placement exchange rate with no code surgery.
The shims carry no branching - see the long comment at the shim definitions for the three tracer-only rules that
forced that, each of which the simulator happily accepts.
GpSimd is not offered: NKI's own validator rejects it for a PSUM operand ("tensor_copy src must be in ['sbuf'], got
psum"), so the chip's most idle engine (21% busy) cannot take drain work at all.

HOW TO READ IT.
  SCALAR < VECTOR on mlp_bwd  -> pin the backward kernels' drains to Scalar. mlp_bwd is the strongest candidate
        because it issues ZERO `activation` instructions, i.e. its Scalar engine is idle for the whole kernel.
  SCALAR > VECTOR on mlp_fwd  -> expected and fine: the forward graph is Scalar-dominated (61% busy vs Vector 50%),
        so the right pin differs per kernel and must be measured per kernel. That asymmetry IS the result.
  all three within ~1%        -> drains are not on this kernel's critical path; move on to the next kernel.
dx/dW must be bit-identical across variants (max|diff| 0.000e+00): the engine choice cannot change arithmetic, so any
difference is a bug in this file.

Usage on the box (gated, single core):
    cd ~/tf && bash trn2_job.sh MLPPLACE /home/ubuntu/workspace/native_venv/bin/python nki/nki_place_probe.py
Env: R1_ITERS (default 20). NKI_STRICT is set below.
"""
import os, sys, time, traceback, types

os.environ.setdefault("NKI_STRICT", "1")
sys.path.insert(0, os.path.expanduser("~/tf"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx.nki_hop import wrap_nki
import nki_mlp

ITERS = int(os.environ.get("R1_ITERS", "20"))
T, D, H = 16384, 512, 3072        # production MLP shape: one layer, one rank (b16 x t1024 tokens, 512 wide, 3072 hidden)
dev = torch.device("neuron")

# ---- the shim -------------------------------------------------------------------------------------------------
# MODE is read at TRACE time, so it must be set before a variant's first call and left alone until that variant has
# been timed. Each variant is a separately-jitted clone, so each traces exactly once.
_orig_tensor_copy = nisa.tensor_copy
_UNK = nisa.engine.unknown

# THE SHIMS ARE BRANCH-FREE ON PURPOSE. The device tracer COMPILES a substituted function as part of the kernel, so it
# accepts almost no Python in there: `**kwargs` is rejected ("variable keyword arguments are not supported in NKI"),
# keyword expansion is rejected, and even `getattr` is ("failed to resolve name 'builtins.getattr'"). The simulator
# accepts all three, so these are tracer-only rules - the same class of trap as "no inner function definitions".
# Consequence: no per-tile PSUM test inside the shim. Instead all tensor_copy calls are pinned. For nki_mlp.py that is
# EXACTLY equivalent to pinning only the drains, because every one of its tensor_copy sources is a PSUM tile
# (mlp_fwd lines 53/63/77/87/97, mlp_bwd 176/200/209/212 - checked). Re-check that before pointing this at a new kernel.


def _shim_scalar(dst, src, engine=_UNK, name=None):
    return _orig_tensor_copy(dst=dst, src=src, engine=nisa.scalar_engine, name=name)


def _shim_vector(dst, src, engine=_UNK, name=None):
    return _orig_tensor_copy(dst=dst, src=src, engine=nisa.vector_engine, name=name)


def _shim_off(dst, src, engine=_UNK, name=None):
    return _orig_tensor_copy(dst=dst, src=src, engine=engine, name=name)


def _clone(kernel, newname):
    """A renamed copy of a jitted kernel's underlying function, re-jitted. Renaming matters twice: wrap_nki derives
    its torch-op name from it (identical names would collide), and each clone must trace separately so the three
    variants get three NEFFs."""
    f = kernel.func
    g = types.FunctionType(f.__code__, f.__globals__, newname, f.__defaults__, f.__closure__)
    g.__annotations__ = dict(getattr(f, "__annotations__", {}))
    g.__doc__ = f.__doc__
    return nki.jit(g)


# module-level, as the tracer requires
mf_default, mf_scalar, mf_vector = (_clone(nki_mlp.mlp_fwd, n) for n in ("mf_default", "mf_scalar", "mf_vector"))
mb_default, mb_scalar, mb_vector = (_clone(nki_mlp.mlp_bwd, n) for n in ("mb_default", "mb_scalar", "mb_vector"))
W = {k: wrap_nki(v) for k, v in
     dict(mf_default=mf_default, mf_scalar=mf_scalar, mf_vector=mf_vector,
          mb_default=mb_default, mb_scalar=mb_scalar, mb_vector=mb_vector).items()}

MODES = [("DEFAULT (compiler picks)", "default", _shim_off),
         ("SCALAR  (drains -> Scalar)", "scalar", _shim_scalar),
         ("VECTOR  (drains -> Vector)", "vector", _shim_vector)]


def timed(fn, n=ITERS):
    fn()
    torch.neuron.synchronize()
    t0 = time.monotonic()
    for _ in range(n):
        fn()
    torch.neuron.synchronize()
    return (time.monotonic() - t0) / n * 1000


torch.manual_seed(0)
x = (torch.randn(T, D) * 0.5).bfloat16().to(dev)
dy = (torch.randn(T, D) * 0.1).bfloat16().to(dev)
w1 = (torch.randn(H, D) * 0.03).bfloat16().to(dev)
w2 = (torch.randn(D, H) * 0.02).bfloat16().to(dev)

print(f"NKI_STRICT={os.environ.get('NKI_STRICT')}  iters={ITERS}  T={T} D={D} H={H}\n"
      f"{'=' * 100}", flush=True)

# mlp_fwd first: it also produces preT, which mlp_bwd needs.
pre = None
for kern_prefix, label in (("mf", "mlp_fwd  (LIVE, forward)"), ("mb", "mlp_bwd  (LIVE, backward)")):
    print(f"\n{label}", flush=True)
    ref = None
    base = None
    for name, key, shim in MODES:
        try:
            nisa.tensor_copy = shim          # active for THIS variant's trace; each clone traces exactly once
            k = W[f"{kern_prefix}_{key}"]
            if kern_prefix == "mf":
                call = lambda kk=k: kk[(1,)](x, w1, w2)
            else:
                call = lambda kk=k: kk[(1,)](x, dy, pre, w1, w2)
            out = call()
            torch.neuron.synchronize()
            first = out[0].float().cpu()
            if kern_prefix == "mf" and pre is None:
                pre = out[1].contiguous()
            diff = "reference" if ref is None else f"max|diff| vs DEFAULT {(first - ref).abs().max().item():.3e}"
            if ref is None:
                ref = first
            ms = timed(call)
            tag = "" if base is None else f"  ({ms / base - 1:+.1%} vs DEFAULT)"
            if base is None:
                base = ms
            print(f"    {name:28s} {ms:8.3f} ms/call{tag:26s} {diff}", flush=True)
        except Exception as e:
            print(f"    {name:28s} FAILED: {type(e).__name__}: {str(e)[:250]}", flush=True)
            traceback.print_exc(limit=4)
    nisa.tensor_copy = _orig_tensor_copy

print("\ndone", flush=True)
