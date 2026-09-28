#!/usr/bin/env python3
"""Device experiment (ON the box, single core): does ANY neuronx-cc flag or runtime env var change how much
independent work the toolchain overlaps inside one NKI kernel?

BASELINE TO MOVE (nki_coschedule_test.py, 2026-09-27): A = mlp_body 1.47 ms, B = 8 rmsnorm passes 1.62 ms,
serial sum 3.09 ms, fused 2.83 ms -> 19% of the ideal max(A,B) overlap is recovered, 82% unclaimed.

HOW THE FLAG GETS THERE (read from the installed stack, not recollection):
  torch_neuron_eager/torch_neuronx/kernels/neuronx_cc_wrapper.py:126-134 prepends $NEURON_CC_FLAGS to the
  neuronx-cc argv, and compiler_subprocess.py:42-56 puts `*sorted(config.extra_flags)` INTO the NEFF cache key,
  so every flag string automatically invalidates the cache and forces a real recompile. No stale-NEFF trap.
  The framework's own default (compiler_config.py) is `--model-type transformer -O1 --auto-cast=none`, i.e. it
  DOWNGRADES from the neuronx-cc default of -O2.

MEASURED 2026-09-28 (all on the box, production shape, 20 timed iters after warmup; FUSED baseline 2.82 ms):
  positive control  NEURON_CC_FLAGS=--r1-no-such-flag -> compile dies with NCC_EARG002 naming the flag. Channel live.
  neuronx-cc flags, all REAL recompiles (12-14 s vs 7.5 s cached), FUSED: -O2 2.82, -O3 2.88 (worse),
    --internal-interleave-kind=pe_cycle 2.82, --enable-internal-relaxed-order 2.82,
    --enable-internal-anti-dependence-reduction 2.81, --enable-parallel-queues 2.82,
    --internal-schedule-delayed-latency=0 2.82.  --scheduler=none and =experimental-loop-shift-left both die
    with NCC_IDVR006 (pipeline construction).
  NKI pipeline options, caches OFF (41-46 s real compiles): baseline 2.82; enable-instruction-scheduling=false
    2.82 (it is ALREADY false -- see below); =true dies with NCC_ILMM003.  address_rotation disabled: 2.82.
  runtime env: NEURON_RT_DBG_EARLY_COMPLETION=1 2.81, NEURON_RT_DBG_EARLY_POSTING=1 2.82,
    NEURON_RT_DBG_FORCE_STRICT_ORDERING=1 -> A 1.46->5.75, FUSED 2.82->7.50 (3.9x on A).  That last one is the
    control that matters: the runtime is ALREADY exploiting heavy reordering, so nothing is waiting for permission.
  VERDICT: no configuration knob moves the 19%. Turning the instruction scheduler OFF costs 0.00 ms, so the
  fused time is not set by instruction order at all -- it is an order-invariant per-engine throughput bound.

WHY enable-instruction-scheduling IS ALREADY FALSE: nki/framework/compiled.py:_compile_opts calls
disable_backend_optimizations() whenever _enable_backend_opt is False (its default), and that sets the NKI
pipeline options `enable-linear-scan-allocation=false enable-instruction-scheduling=false`. So NKI's own
scheduler and allocator never run on our kernels; neuronx-cc owns both. Verified locally that the LAST duplicate
pipeline option wins, so R1_NKI_PIPE below really does override the framework.

Usage (one variant per job; single token, commas become spaces):
    bash trn2_job.sh CCF_BASE  <python> nki/nki_ccflags_test.py
    R1_CC=-O2 bash trn2_job.sh CCF_O2 <python> nki/nki_ccflags_test.py
    R1_CC=--scheduler=none ...      # diagnostic: does the backend scheduler touch NKI code AT ALL?
    R1_CC=--internal-interleave-kind=pe_cycle ...
    R1_CC=BADFLAG ...               # positive control: must FAIL, proving the channel is live
"""
import os, sys, time, traceback

# Must happen BEFORE torch/nki import so every compiler subprocess inherits it.
_CC = os.environ.get("R1_CC", "")
if _CC == "BADFLAG":
    _CC = "--r1-no-such-flag"
if _CC:
    os.environ["NEURON_CC_FLAGS"] = (os.environ.get("NEURON_CC_FLAGS", "") + " " + _CC.replace(",", " ")).strip()
print(f"R1_CC={_CC!r}  ->  NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS','')!r}", flush=True)
for _k in sorted(os.environ):
    if _k.startswith(("NEURON_RT", "NEURON_LOGICAL", "NEURON_CC", "NKI_")):
        print(f"  env {_k}={os.environ[_k]}", flush=True)

sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx
import nki, nki.language as nl, nki.isa as nisa
from torch_neuronx.nki_hop import wrap_nki

P, FMAX = 128, 512
dev = torch.device("neuron"); torch.manual_seed(0)
TT, D, H = 16384, 512, 3072
NR, ND = 16384, 512
PASSES = int(os.environ.get("R1_NORM_PASSES", "8"))


def mlp_body(x_in, w1, w2, y_out, pre_out):
    """Verbatim from nki/nki_coschedule_test.py. PE-dominated."""
    T, Dm = x_in.shape; Hm = w1.shape[0]; kt = Dm // P; ht = Hm // P; chunks = T // FMAX
    W1 = []
    for _j in range(kt):
        W1.append(nl.ndarray((P, Hm), dtype=nl.bfloat16, buffer=nl.sbuf))
    for h in range(ht):
        blk = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w1[h * P:(h + 1) * P, :])
        for k in range(kt):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, k * P:(k + 1) * P])
            nisa.tensor_copy(dst=W1[k][:, h * P:(h + 1) * P], src=tps)
    W2 = []
    for _j in range(ht):
        W2.append(nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf))
    for o in range(kt):
        blk = nl.ndarray((P, Hm), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=blk, src=w2[o * P:(o + 1) * P, :])
        for h in range(ht):
            tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=blk[:, h * P:(h + 1) * P])
            nisa.tensor_copy(dst=W2[h][:, o * P:(o + 1) * P], src=tps)
    for c in nl.affine_range(chunks):
        t0 = c * FMAX
        xT = []
        for _j in range(kt):
            xT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for tb in range(FMAX // P):
            xt = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x_in[t0 + tb * P:t0 + (tb + 1) * P, :])
            for k in range(kt):
                tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=xt[:, k * P:(k + 1) * P])
                nisa.tensor_copy(dst=xT[k][:, tb * P:(tb + 1) * P], src=tps)
        hT = []
        for _j in range(ht):
            hT.append(nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
        for h in range(ht):
            pps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
            for k in range(kt):
                nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * P:(h + 1) * P], moving=xT[k], accumulate=(k > 0))
            pre = nl.ndarray((P, FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pre, src=pps)
            nisa.dma_copy(dst=pre_out[h * P:(h + 1) * P, t0:t0 + FMAX], src=pre)
            nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)
        for tb in range(FMAX // P):
            yps = nl.ndarray((P, Dm), dtype=nl.float32, buffer=nl.psum)
            for h in range(ht):
                nisa.nc_matmul(dst=yps, stationary=hT[h][:, tb * P:(tb + 1) * P], moving=W2[h], accumulate=(h > 0))
            yt = nl.ndarray((P, Dm), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=yt, src=yps)
            nisa.dma_copy(dst=y_out[t0 + tb * P:t0 + (tb + 1) * P, :], src=yt)


def norm_body(z_in, z_out, passes):
    """Verbatim from nki/nki_coschedule_test.py. No matmul, no transpose."""
    rows, d = z_in.shape
    tiles = rows // P
    for i in nl.affine_range(tiles):
        x = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x, src=z_in[i * P:(i + 1) * P, :])
        sq = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        for _p in range(passes):
            nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=1e-6)
            nisa.activation(dst=ms, op=nl.rsqrt, data=ms)
            nisa.tensor_scalar(dst=x, data=x, op0=nl.multiply, operand0=ms)
        y = nl.ndarray((P, d), dtype=z_out.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y, src=x)
        nisa.dma_copy(dst=z_out[i * P:(i + 1) * P, :], src=y)


@nki.jit
def a_only(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
    T, Dm = x_in.shape; Hm = w1.shape[0]
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    mlp_body(x_in, w1, w2, y_out, pre_out)
    return y_out, pre_out


@nki.jit
def b_only(z_in: nl.ndarray, passes: int = 8):
    rows, d = z_in.shape
    z_out = nl.ndarray((rows, d), dtype=z_in.dtype, buffer=nl.shared_hbm)
    norm_body(z_in, z_out, passes)
    return z_out


@nki.jit
def ab_fused(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, z_in: nl.ndarray, passes: int = 8):
    T, Dm = x_in.shape; Hm = w1.shape[0]; rows, d = z_in.shape
    y_out = nl.ndarray((T, Dm), dtype=x_in.dtype, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((Hm, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
    z_out = nl.ndarray((rows, d), dtype=z_in.dtype, buffer=nl.shared_hbm)
    mlp_body(x_in, w1, w2, y_out, pre_out)
    norm_body(z_in, z_out, passes)
    return y_out, pre_out, z_out


def time_it(fn, n=20):
    fn(); torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(n):
        fn()
    torch_neuronx.synchronize(); return (time.time() - t0) / n * 1000


x = (torch.randn(TT, D) * 0.5).bfloat16().to(dev)
w1 = (torch.randn(H, D) * 0.03).bfloat16().to(dev)
w2 = (torch.randn(D, H) * 0.02).bfloat16().to(dev)
z = (torch.randn(NR, ND) * 0.5).bfloat16().to(dev)
grid = (1,)
# Kernel-level NKI knob: address_rotation (nki/framework/kernel.py, default True) =
# "address rotation optimization for multi-buffered tensors". R1_NOROT=1 turns it off as a control.
if os.environ.get("R1_NOROT", "0") != "0":
    _a, _b, _ab = a_only.disable_address_rotation(), b_only.disable_address_rotation(), ab_fused.disable_address_rotation()
    print("address_rotation DISABLED on all three kernels", flush=True)
else:
    _a, _b, _ab = a_only, b_only, ab_fused
ka, kb, kab = wrap_nki(_a), wrap_nki(_b), wrap_nki(_ab)
fa = lambda: ka[grid](x, w1, w2)
fb = lambda: kb[grid](z, passes=PASSES)
fab = lambda: kab[grid](x, w1, w2, z, passes=PASSES)

# --------------------------------------------------------------------------- NKI's OWN MLIR scheduler knob.
# nki/compiler/_internal/_mlir_libs/_nki.so registers pipeline options on `nki-opt-pipeline`:
#   enable-instruction-scheduling=true|false   ("Enable instruction scheduling pass")
#   enable-linear-scan-allocation=true|false
#   nisa-list-scheduling-heuristic=trivial|dma-prefetch|asap|critical-path
#       trivial       "Prioritize instructions in original order"
#       dma-prefetch  "Prioritize DMA instructions by ready time"
#       asap          "Prioritize all instructions by ready time"
#       critical-path "Prioritize by longest path to output"
# They are a CompileOptions field (nki_opt_pipeline_options) with no env-var hook, so patch the frozen
# dataclass's __post_init__. torch_neuronx imports the same class object, so every kernel compile picks it up.
_PIPE = os.environ.get("R1_NKI_PIPE", "")
if _PIPE:
    import nki.compiler.ncc_driver as _nd
    _orig_post = _nd.CompileOptions.__post_init__

    def _patched_post(self):
        _orig_post(self)
        object.__setattr__(self, "nki_opt_pipeline_options",
                           tuple(self.nki_opt_pipeline_options) + tuple(_PIPE.split(",")))
    _nd.CompileOptions.__post_init__ = _patched_post
    print(f"PATCHED nki_opt_pipeline_options += {tuple(_PIPE.split(','))}", flush=True)
    assert _nd.CompileOptions(target="trn2").nki_opt_pipeline_options, "patch did not take"
    print(f"  verified: {_nd.CompileOptions(target='trn2').nki_opt_pipeline_options}", flush=True)

t_compile = time.time()
try:
    fa(); fb(); fab(); torch_neuronx.synchronize()
except Exception:
    print("COMPILE/RUN FAILED (this is the expected outcome for R1_CC=BADFLAG):", flush=True)
    traceback.print_exc(); sys.exit(3)
print(f"compile+first-run wall: {time.time() - t_compile:6.1f} s", flush=True)

ta = time_it(fa); tb = time_it(fb); tab = time_it(fab)
y1, _ = fa(); z1 = fb(); y2, _, z2 = fab(); torch_neuronx.synchronize()
e = lambda p, q: (p.float() - q.float()).abs().max().item()
ideal = max(ta, tb); serial = ta + tb
frac = (serial - tab) / (serial - ideal) * 100.0 if serial > ideal else float("nan")
print(f"(A) mlp_body            : {ta:6.2f} ms", flush=True)
print(f"(B) {PASSES} rmsnorm passes  : {tb:6.2f} ms", flush=True)
print(f"(C) A+B fused           : {tab:6.2f} ms   serial {serial:.2f}  ideal {ideal:.2f}", flush=True)
print(f"RESULT R1_CC={_CC or 'BASE'} PIPE={_PIPE or '-'} A={ta:.2f} B={tb:.2f} FUSED={tab:.2f} "
      f"PCT_OF_IDEAL_OVERLAP={frac:.1f}% maxdiff_mlp={e(y1, y2):.3e} maxdiff_norm={e(z1, z2):.3e}", flush=True)
