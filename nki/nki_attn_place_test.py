#!/usr/bin/env python3
"""Device test (single core): (1) HEAD-TO-HEAD our attention backward vs the VENDOR attention_bwd at the production
problem count, re-measured on today's toolchain; (2) ENGINE PLACEMENT - does pinning our backward's PSUM->SBUF drains
to the Scalar engine change the wall, and does GpSimd refuse as the NKI 0.5.0 docstrings say it must?

WHY THIS EXISTS.
The ledger says our own attention kernels BEAT the vendor's on device (2026-09-10, B=32 x T=1024 x HD=128):
  forward v4  0.67 ms vs vendor attention_cte 1.14 ms (x1.70)
  backward v3 1.80 ms vs vendor attention_bwd 2.69 ms (x1.49)
and they were parked for two reasons, BOTH of which have since expired: (a) "the glue eats the gains" - removed by
ATTN_LAYOUT=heads, outT, LAYER=op and in-kernel KEY_OFFSET; (b) a NaN, which bisection (S_V39_AFB_A/B) pinned on the
STRIDED-DMA FORWARD only - the strided BACKWARD ran clean in model context. Those numbers are 17 days and several
toolchain states old, so before anyone builds the swap they get re-measured.

Independently, the engine-placement axis: per NKI 0.5.0 docstrings only tensor_copy / tensor_tensor / tensor_scalar /
memset / nc_transpose take `engine=`; range_select, scalar_tensor_tensor, tensor_reduce and activation do NOT, and
GpSimd CANNOT ACCESS PSUM ("since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or Vector Engine must be
chosen when the input or output tile is in PSUM"). In our backward's inner loop that leaves exactly three movable
groups, all PSUM->SBUF drains via tensor_copy, and they are 1536 of ~3584 Vector element-units per (query block, key
block) = 43%. The vendor's own attention_bwd pins every one of ITS large drains to nisa.scalar_engine (4 sites), so
the technique is device-proven in this exact kernel family - it has just never been tried in ours.

VARIANTS (identical arithmetic, identical tile shapes, ONLY the engine= argument on the three drains differs):
  DEFAULT  no engine= : the compiler picks ("compiler selects best engine based on engine workload")
  SCALAR   engine=nisa.scalar_engine on all three drains
  VECTOR   engine=nisa.vector_engine on all three drains (control: if DEFAULT == VECTOR the compiler was already
           putting them on Vector, and SCALAR vs VECTOR is then the clean placement delta)
  GPSIMD   engine=nisa.gpsimd_engine (EXPECTED TO FAIL to trace/compile - the PSUM rule. A pass here would mean the
           docstring is wrong and the most idle engine on the chip, 21% busy, is reachable from attention.)

HOW TO READ THE RESULT.
  ours < vendor at B=64        -> the swap is a real throughput item; the per-step arithmetic is printed for you.
  SCALAR < DEFAULT by >3%      -> engine placement pays inside attention and the compiler's default is leaving it on
                                  the table; the same three-line change applies to every kernel we own that drains PSUM.
  SCALAR == DEFAULT == VECTOR  -> the compiler already balances drains and `engine=` is a no-op knob here; the
                                  placement axis is closed for OUR attention too, not just the vendor's.
  GPSIMD traces and runs       -> the docstring is wrong; report it, it reopens GpSimd for all PSUM work.
Correctness is printed against the torch reference for every variant; any variant whose max|diff| differs from
DEFAULT's is a bug in this file, not a finding.

Usage on the box (gated, single core):
    cd ~/tf && bash trn2_job.sh ATTNPLACE /home/ubuntu/workspace/native_venv/bin/python nki/nki_attn_place_test.py
Env: R1_BIG=0 skips the B=64 head-to-head (halves compile time); R1_ITERS (default 20).
Expect a multi-minute cold compile per variant (5 kernels total). NKI_STRICT is set below.
"""
import os, sys, time, traceback

os.environ.setdefault("NKI_STRICT", "1")
sys.path.insert(0, os.path.expanduser("~/tf"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx.nki_hop import wrap_nki
from nkilib.core.attention.attention_bwd import attention_bwd
from nkilib.core.attention.attention_cte import attention_cte
from nki_attn_bwd import reference, bounds_lo_hi
from nki_attn_fwd import attn_fwd4, fwd2_args
from nki_attn_plain import attn_fwd4_plain, attn_bwd_plain

P = 128
KB = 512
NH = 4                      # heads per batch row in the live model (b*NH = B problems; production b=16 -> B=64)
ITERS = int(os.environ.get("R1_ITERS", "20"))
BIG = os.environ.get("R1_BIG", "1") != "0"
# GpSimd is EXPECTED to fail (it cannot access PSUM). A failed compile can poison /tmp/nki_* for the rest of the
# process, so it is opt-in and should be run as its OWN job, after the real measurement has been taken.
GPS = os.environ.get("R1_GPSIMD", "0") != "0"
dev = torch.device("neuron")


# --------------------------------------------------------------------------------------------------------------
# our attention backward, body copied VERBATIM from nki/nki_attn_bwd.py:attn_bwd with ONE addition: `eng`, applied
# to the three PSUM->SBUF tensor_copy drains (sS, dpS, dST) and to the K^T/V^T setup drains. Everything else -
# every matmul, every range_select, every activation, every accumulate - is byte-identical to the shipped kernel.
# Undecorated so the four jitted wrappers below can each instantiate it with a different engine (tracer rule:
# @nki.jit functions must be at module level, so the VARIANTS are module-level and only the body is shared).
# --------------------------------------------------------------------------------------------------------------
def drain(dst, src, eng):
    """PSUM->SBUF copy, optionally pinned to an engine. MUST be module level: the device tracer rejects inner function
    definitions ("NKI does not support inner function definitions") and the SIMULATOR DOES NOT - this exact closure
    simulated fine locally and failed on device, which is tracer-rule territory."""
    if eng is None:
        nisa.tensor_copy(dst=dst, src=src)
    else:
        nisa.tensor_copy(dst=dst, src=src, engine=eng)


def _bwd_body(q_in, k_in, v_in, o_in, do_in, lse_in, sink_in, lo_in, hi_in,
              dq_out, dk_out, dv_out, dsink_out, eng):
    B, T, D = q_in.shape
    nq = T // P
    kb = KB if T >= KB else T
    nk = T // kb
    ks = kb // P
    nprog = nl.num_programs(0)
    per = B // nprog
    for bi in nl.affine_range(per):
        b = nl.program_id(0) * per + bi
        NJ = T // P
        K3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=K3, src=k_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        V3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=V3, src=v_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        Q3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=Q3, src=q_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        dO3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=dO3, src=do_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        O3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=O3, src=o_in[b].ap(pattern=[[D, P], [P * D, NJ], [1, D]], offset=0))
        lse3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lse3, src=lse_in[b].ap(pattern=[[1, P], [P, NJ], [1, 1]], offset=0))
        sk3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sk3, src=sink_in[b].ap(pattern=[[1, P], [P, NJ], [1, 1]], offset=0))
        lo3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lo3, src=lo_in[b].ap(pattern=[[1, P], [P, NJ], [1, 1]], offset=0))
        hi3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=hi3, src=hi_in[b].ap(pattern=[[1, P], [P, NJ], [1, 1]], offset=0))
        KT = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        VT = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(NJ):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=K3[:, j, :])
            drain(KT[:, j * P:(j + 1) * P], tps, eng)
            tps2 = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps2, data=V3[:, j, :])
            drain(VT[:, j * P:(j + 1) * P], tps2, eng)
        dq3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        dsk3 = nl.ndarray((P, NJ, 1), dtype=nl.float32, buffer=nl.sbuf)
        dK = []
        dV = []
        for j in range(T // P):
            a = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            dK.append(a)
            a2 = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a2, 0.0)
            dV.append(a2)
        for i in range(nq):
            t0 = i * P
            Q = Q3[:, i, :]
            dO = dO3[:, i, :]
            O = O3[:, i, :]
            lse = lse3[:, i, :]
            lo = lo3[:, i, :]
            hi = hi3[:, i, :]
            sk = sk3[:, i, :]
            prod = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=dO, data2=O, op=nl.multiply)
            Dr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=Dr, op=nl.add, data=prod, axis=(1,))
            nlse = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nlse, data=lse, op0=nl.multiply, operand0=-1.0)
            ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ps, data1=sk, data2=lse, op=nl.subtract)
            nisa.activation(dst=ps, op=nl.exp, data=ps)
            nisa.tensor_tensor(dst=ps, data1=ps, data2=Dr, op=nl.multiply)
            nisa.tensor_scalar(dst=dsk3[:, i, :], data=ps, op0=nl.multiply, operand0=-1.0)
            tq = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tq, data=Q)
            QT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            drain(QT, tq, eng)
            tdo = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tdo, data=dO)
            dOT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            drain(dOT, tdo, eng)
            dq_ps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
            first = True
            for J in range(nk):
                if J * kb > t0 + P - 1:
                    continue
                k0 = J * kb
                s_ps = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s_ps, stationary=QT, moving=KT[:, k0:k0 + kb])
                sS = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.sbuf)
                drain(sS, s_ps, eng)                                      # DRAIN 1 of 3 (512 units)
                sm = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.sbuf)
                nisa.range_select(dst=sm, on_true_tile=sS, comp_op0=nl.greater_equal, comp_op1=nl.less,
                                  bound0=lo, bound1=hi, range_start=k0)
                Pb = nl.ndarray((P, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=Pb, op=nl.exp, data=sm, bias=nlse)
                dp_ps = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=dp_ps, stationary=dOT, moving=VT[:, k0:k0 + kb])
                dpS = nl.ndarray((P, kb), dtype=nl.float32, buffer=nl.sbuf)
                drain(dpS, dp_ps, eng)                                    # DRAIN 2 of 3 (512 units)
                dS = nl.ndarray((P, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=dS, data=dpS, op0=nl.subtract, operand0=Dr,
                                          op1=nl.multiply, operand1=Pb)
                for s in range(ks):
                    j = J * ks + s
                    dv_ps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dv_ps, stationary=Pb[:, s * P:(s + 1) * P], moving=dO)
                    nisa.tensor_tensor(dst=dV[j], data1=dV[j], data2=dv_ps, op=nl.add)
                    dk_ps = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dk_ps, stationary=dS[:, s * P:(s + 1) * P], moving=Q)
                    nisa.tensor_tensor(dst=dK[j], data1=dK[j], data2=dk_ps, op=nl.add)
                    tds = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tds, data=dS[:, s * P:(s + 1) * P])
                    dST = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    drain(dST, tds, eng)                                  # DRAIN 3 of 3 (4 x 128 units)
                    nisa.nc_matmul(dst=dq_ps, stationary=dST, moving=K3[:, j, :], accumulate=(not first))
                    first = False
            drain(dq3[:, i, :], dq_ps, eng)
        dk3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        dv3 = nl.ndarray((P, NJ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(NJ):
            nisa.tensor_copy(dst=dk3[:, j, :], src=dK[j])
            nisa.tensor_copy(dst=dv3[:, j, :], src=dV[j])
        for j in range(NJ):
            nisa.dma_copy(dst=dq_out[b, j * P:(j + 1) * P, :], src=dq3[:, j, :])
            nisa.dma_copy(dst=dk_out[b, j * P:(j + 1) * P, :], src=dk3[:, j, :])
            nisa.dma_copy(dst=dv_out[b, j * P:(j + 1) * P, :], src=dv3[:, j, :])
            nisa.dma_copy(dst=dsink_out[b, j * P:(j + 1) * P, :], src=dsk3[:, j, :])


def _outs(q_in):
    B, T, D = q_in.shape
    return (nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm),
            nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm),
            nl.ndarray((B, T, D), dtype=q_in.dtype, buffer=nl.shared_hbm),
            nl.ndarray((B, T, 1), dtype=nl.float32, buffer=nl.shared_hbm))


@nki.jit
def bwd_default(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, o_in: nl.ndarray, do_in: nl.ndarray,
                lse_in: nl.ndarray, sink_in: nl.ndarray, lo_in: nl.ndarray, hi_in: nl.ndarray):
    a, b, c, d = _outs(q_in)
    _bwd_body(q_in, k_in, v_in, o_in, do_in, lse_in, sink_in, lo_in, hi_in, a, b, c, d, None)
    return a, b, c, d


@nki.jit
def bwd_scalar(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, o_in: nl.ndarray, do_in: nl.ndarray,
               lse_in: nl.ndarray, sink_in: nl.ndarray, lo_in: nl.ndarray, hi_in: nl.ndarray):
    a, b, c, d = _outs(q_in)
    _bwd_body(q_in, k_in, v_in, o_in, do_in, lse_in, sink_in, lo_in, hi_in, a, b, c, d, nisa.scalar_engine)
    return a, b, c, d


@nki.jit
def bwd_vector(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, o_in: nl.ndarray, do_in: nl.ndarray,
               lse_in: nl.ndarray, sink_in: nl.ndarray, lo_in: nl.ndarray, hi_in: nl.ndarray):
    a, b, c, d = _outs(q_in)
    _bwd_body(q_in, k_in, v_in, o_in, do_in, lse_in, sink_in, lo_in, hi_in, a, b, c, d, nisa.vector_engine)
    return a, b, c, d


@nki.jit
def bwd_gpsimd(q_in: nl.ndarray, k_in: nl.ndarray, v_in: nl.ndarray, o_in: nl.ndarray, do_in: nl.ndarray,
               lse_in: nl.ndarray, sink_in: nl.ndarray, lo_in: nl.ndarray, hi_in: nl.ndarray):
    a, b, c, d = _outs(q_in)
    _bwd_body(q_in, k_in, v_in, o_in, do_in, lse_in, sink_in, lo_in, hi_in, a, b, c, d, nisa.gpsimd_engine)
    return a, b, c, d


# wrap_nki registers a torch custom op per kernel: do it ONCE at module level (the pattern every working test here
# uses), never inside a loop.
K_DEFAULT = wrap_nki(bwd_default)
K_SCALAR = wrap_nki(bwd_scalar)
K_VECTOR = wrap_nki(bwd_vector)
K_GPSIMD = wrap_nki(bwd_gpsimd)
K_FWD4 = wrap_nki(attn_fwd4)
K_FWD4P = wrap_nki(attn_fwd4_plain)
K_BWDP = wrap_nki(attn_bwd_plain)
K_VBWD = wrap_nki(attention_bwd)
K_VFWD = wrap_nki(attention_cte)

VARIANTS = [("DEFAULT (compiler picks)", K_DEFAULT),
            ("SCALAR  (drains on Scalar)", K_SCALAR),
            ("VECTOR  (drains on Vector)", K_VECTOR)] + \
           ([("GPSIMD  (expected to FAIL)", K_GPSIMD)] if GPS else [])


def timed(fn, n=ITERS):
    fn()
    torch.neuron.synchronize()
    t0 = time.monotonic()
    for _ in range(n):
        fn()
    torch.neuron.synchronize()
    return (time.monotonic() - t0) / n * 1000


def build_inputs(B, T, D, seed=0):
    """Same generator as nki_attn_bwd_small.py / nki_attn_fwd_test.py (comparable to the ledger) with ONE change: the
    document cut depends on the BATCH ROW (b // NH), not the problem, because the vendor backward's bound_min/bound_max
    are per-batch-row (b, t) and cannot express per-head bounds. With row-uniform bounds both kernels see exactly the
    same problem, so the vendor's numerics are a real check rather than a diagnostic."""
    torch.manual_seed(seed)
    q = (torch.randn(B, T, D) * 0.3).bfloat16()
    k = (torch.randn(B, T, D) * 0.3).bfloat16()
    v = torch.randn(B, T, D).bfloat16()
    sink = torch.randn(B, 1) * 0.5
    start = torch.zeros(B, T)
    end = torch.full((B, T), float(T))
    for b in range(B):
        cut = 100 + 37 * (b // NH)
        if cut < T:
            start[b, cut:] = float(cut)
            end[b, :cut] = float(cut)
    qr = q.float().requires_grad_(True)
    kr = k.float().requires_grad_(True)
    vr = v.float().requires_grad_(True)
    skr = sink.clone().requires_grad_(True)
    out, lse = reference(qr, kr, vr, skr, start, end)
    do = torch.randn(B, T, D)
    (out * do).sum().backward()
    return dict(q=q, k=k, v=v, sink=sink, start=start, end=end, out=out.detach(), lse=lse.detach(), do=do,
                dq=qr.grad, dk=kr.grad, dv=vr.grad, dsink=skr.grad)


def run_shape(B, T, D, place=True):
    print(f"\n{'=' * 104}\nB={B} (b*NH problems)  T={T}  HD={D}   "
          f"[production is B=64: b=16 rows x NH=4 heads]\n{'=' * 104}", flush=True)
    r = build_inputs(B, T, D)
    lo, hi = bounds_lo_hi(r["start"], r["end"])
    sink_rows = r["sink"].view(B, 1, 1).expand(B, T, 1).contiguous()
    ours_args = [a.to(dev) for a in (r["q"], r["k"], r["v"], r["out"].bfloat16(), r["do"].bfloat16(),
                                     r["lse"].contiguous(), sink_rows, lo, hi)]
    refmax = {n: r[n].abs().max().item() for n in ("dq", "dk", "dv")}

    # ---- 1. OURS, the engine placements --------------------------------------------------------------------
    base = None
    for name, kern in (VARIANTS if place else VARIANTS[:1]):
        try:
            dq, dk, dv, dsr = kern[(1,)](*ours_args)
            torch.neuron.synchronize()
            errs = []
            for nm, got in (("dq", dq), ("dk", dk), ("dv", dv)):
                e = (got.float().cpu() - r[nm]).abs().max().item()
                errs.append(f"{nm} {e:.2e}/{refmax[nm]:.2e}")
            ms = timed(lambda kk=kern: kk[(1,)](*ours_args))
            tag = "" if base is None else f"  ({ms / base - 1:+.1%} vs DEFAULT)"
            if base is None:
                base = ms
            print(f"  ours bwd {name:28s} {ms:7.3f} ms/call{tag}   err(max|diff|/|ref|): {'  '.join(errs)}",
                  flush=True)
        except Exception as e:
            print(f"  ours bwd {name:28s} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            if "GPSIMD" not in name:
                traceback.print_exc()

    # ---- 2. VENDOR backward at the same problem count ------------------------------------------------------
    # The vehicle's call (submission_raw/train3_canon.py:attn_bwd): HD-major 4-D, bound_min/max as (b, t) fp32,
    # lse tiled (b, h, 128, t//128). b*h must equal B; production is b=16, h=4.
    b4 = B // NH
    try:
        tr = lambda x: x.view(b4, NH, T, D).transpose(-2, -1).contiguous().to(dev)
        lse4 = r["lse"].view(b4, NH, T // P, P).permute(0, 1, 3, 2).contiguous().to(dev)
        vb_args = (tr(r["q"]), tr(r["k"]), tr(r["v"]), tr(r["out"].bfloat16()), tr(r["do"].bfloat16()), lse4)
        vb_kw = dict(sinks_ref=r["sink"].view(b4, NH).float().contiguous().to(dev),
                     bound_min=r["start"][::NH].float().contiguous().to(dev),
                     bound_max=r["end"][::NH].float().contiguous().to(dev),
                     use_causal_mask=True, mixed_precision=True, softmax_scale=1.0)
        res = K_VBWD[(1,)](*vb_args, **vb_kw)
        torch.neuron.synchronize()
        vms = timed(lambda: K_VBWD[(1,)](*vb_args, **vb_kw))
        errs = []
        for i, nm in enumerate(("dq", "dk", "dv")):
            got = res[i].transpose(-2, -1).reshape(B, T, D).float().cpu()
            errs.append(f"{nm} {(got - r[nm]).abs().max().item():.2e}/{refmax[nm]:.2e}")
        print(f"  VENDOR attention_bwd                 {vms:7.3f} ms/call"
              f"                       err(max|diff|/|ref|): {'  '.join(errs)}", flush=True)
        # our PLAIN backward (per-tile loads instead of one strided DMA per tensor). The strided loads are what made
        # ours fast, and the NaN bisection (S_V39_AFB_A/B) pinned the model-context NaN on the strided FORWARD only -
        # the strided BACKWARD ran clean - so this row exists as the fallback, not the first choice.
        try:
            K_BWDP[(1,)](*ours_args)
            torch.neuron.synchronize()
            pms = timed(lambda: K_BWDP[(1,)](*ours_args))
            print(f"  ours bwd PLAIN (per-tile loads)      {pms:7.3f} ms/call   "
                  f"({pms / vms - 1:+.1%} vs vendor)   (ledger 2026-09-10 at B=32: plain 2.37 vs vendor 2.69)",
                  flush=True)
        except Exception as e:
            print(f"  ours bwd PLAIN FAILED: {type(e).__name__}: {str(e)[:250]}", flush=True)
        if base is not None:
            per_layer = vms - base
            print(f"  --> per call: vendor {vms:.3f} - ours {base:.3f} = {per_layer:+.3f} ms; "
                  f"x12 layers = {per_layer * 12:+.1f} ms/step (a 424 ms step => {per_layer * 12 / 424:+.1%} steps, "
                  f"{-0.035 * (per_layer * 12 / 424):+.4f} bpb at -0.035/e-fold)", flush=True)
    except Exception as e:
        print(f"  VENDOR attention_bwd FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc()

    # ---- 3. FORWARD head-to-head (ours v4 strided vs vendor attention_cte) --------------------------------
    try:
        omax = r["out"].abs().max().item()
        end_rows, kidx = fwd2_args(sink_rows, r["end"], T)
        f_args = [a.to(dev) for a in (r["q"], r["k"], r["v"], sink_rows, end_rows, kidx)]
        o4, l4 = K_FWD4[(1,)](*f_args)
        torch.neuron.synchronize()
        e4 = (o4.float().cpu() - r["out"]).abs().max().item()
        f_ms = timed(lambda: K_FWD4[(1,)](*f_args))
        print(f"  ours fwd v4 (strided DMA)            {f_ms:7.3f} ms/call"
              f"                       out max|diff|/|ref| {e4:.2e}/{omax:.2e}", flush=True)
        vfa = (f_args[0], f_args[1], f_args[2])
        vfk = dict(tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True,
                   bound_min=r["start"].view(B, T, 1).contiguous().to(dev),
                   bound_max=r["end"].view(B, T, 1).contiguous().to(dev),
                   sink=r["sink"].view(B, 1).contiguous().to(dev))
        vres = K_VFWD[(1,)](*vfa, **vfk)
        torch.neuron.synchronize()
        ev4 = (vres[0].float().cpu().reshape(B, T, D) - r["out"]).abs().max().item()
        vf_ms = timed(lambda: K_VFWD[(1,)](*vfa, **vfk))
        print(f"  VENDOR attention_cte                 {vf_ms:7.3f} ms/call"
              f"                       out max|diff|/|ref| {ev4:.2e}/{omax:.2e}   "
              f"(ledger 2026-09-10 at B=32: ours 0.67 vs vendor 1.14)", flush=True)
        # ours fwd PLAIN: the SHIPPABLE forward, because the strided one NaN'd in model context (S_V39_AFB_A).
        # Ledger 2026-09-10 at B=32 had plain fwd 1.24 vs vendor 1.16, i.e. the whole forward win was the strided DMA.
        K_FWD4P[(1,)](*f_args)
        torch.neuron.synchronize()
        fp_ms = timed(lambda: K_FWD4P[(1,)](*f_args))
        print(f"  ours fwd v4 PLAIN (per-tile loads)    {fp_ms:7.3f} ms/call   "
              f"({fp_ms / vf_ms - 1:+.1%} vs vendor)   (ledger 2026-09-10 at B=32: plain 1.24 vs vendor 1.16)",
              flush=True)
        print(f"  --> forward per call: vendor - ours_strided = {vf_ms - f_ms:+.3f} ms, "
              f"vendor - ours_plain = {vf_ms - fp_ms:+.3f} ms; x12 layers = "
              f"{(vf_ms - f_ms) * 12:+.1f} / {(vf_ms - fp_ms) * 12:+.1f} ms/step", flush=True)
    except Exception as e:
        print(f"  FORWARD head-to-head FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc()


print(f"NKI_STRICT={os.environ.get('NKI_STRICT')}  iters={ITERS}  R1_BIG={BIG}  R1_GPSIMD={GPS}", flush=True)
run_shape(32, 1024, 128, place=True)           # matches the 2026-09-10 ledger baselines exactly
if BIG:
    run_shape(64, 1024, 128, place=False)      # production problem count (b=16 x NH=4); head-to-head only, so this
    #                                            adds 2 NEFFs rather than 5 and keeps the cold compile bounded
print("\ndone", flush=True)
