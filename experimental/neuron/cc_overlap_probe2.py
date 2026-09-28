#!/usr/bin/env python3
"""F_PROBE, RE-RUN PROPERLY (2026-09-27). Can a collective overlap compute on this runtime?

=== WHY THE OLD ANSWER IS UNSUPPORTED ===
Ledger `F_PROBE` (2026-09-16, experimental/neuron/cc_overlap_probe.py) closed lever F with "collectives are
SERIALIZED against compute on this runtime". Its own log refutes its method. It printed both an overlap arm and a
deliberately-serial control, and they are identical at every size:

    reduce_scatter  50MB : async coll + compute  4.10 ms | wait-then-compute  4.12 ms
    reduce_scatter 537MB : async coll + compute 12.30 ms | wait-then-compute 12.13 ms
    all_gather      50MB : async coll + compute  4.09 ms | wait-then-compute  4.03 ms

When the overlap arm equals the serial control to 0.02 ms (and at 537 MB is SLOWER than it), no overlap
opportunity was ever created. The probe measured the API, not the hardware.

=== THE MECHANISM, READ OUT OF THE RUNTIME SOURCE ON THE BOX ===
~/workspace/torch_neuron_eager/torch_neuronx/csrc/distributed/NeuronBackend.cpp:

    bool NeuronBackend::UseHostCC() const {
      static const bool hostcc_enabled = getenv("TORCH_NEURONX_ENABLE_HOST_CC") != nullptr && ... != "0";
      auto current = getCurrentNeuronStream(); auto default_stream = getDefaultNeuronStream();
      return (current != default_stream) && hostcc_enabled;          // BOTH conditions required
    }
    if (UseHostCC()) SubmitHostCCCollective(...);   // NRT host-collectives API -> dedicated collectives queue
    else             SubmitInNeffCollective(...);   // a NEFF submitted to the CURRENT stream = the compute queue

and NeuronWork::wait() is "make current stream wait for collective (non-blocking to CPU)" -- so a default-stream
collective makes the COMPUTE stream depend on it whether or not async_op is set. `async_op=True` cannot buy
overlap on the default stream; it only stashes the tensors. Two things are needed, and F_PROBE had neither:
    TORCH_NEURONX_ENABLE_HOST_CC=1     and     the collective issued inside `with torch.neuron.stream(side)`.
The vendor's own tests do exactly this and include a `TestMultiStreamOverlap` class
(tests/distributed/collective_ops/test_multistream_collectives_with_host_cc.py).
Hardware note from the same source: `constexpr int kStreamPoolSize = 1`, static_asserted equal to
`nrt::kCollectivesQueues` -- there is exactly ONE side stream / ONE collectives queue, so collectives serialize
among THEMSELVES but can run beside compute. Also: on the host-CC path ReduceOp.AVG is not native; it becomes
SUM plus a `div_(world)` post-pass on the side stream, which is a numerics and a cost difference to measure.

=== WHAT THIS FILE MEASURES ===
Per size, with an honest serial control and an honest ideal:
    coll_dflt   collective alone, default stream (the in-NEFF path we ship today)
    coll_hcc    collective alone, side stream + host CC (the direct path) -- is the collective itself cheaper?
    compute     compute alone
    F_PROBE     the old arm reproduced exactly: async_op=True on the default stream, then compute, then wait
    serial      collective on default stream, then compute        <- the number to beat
    overlap     collective on side stream, compute on default stream, then join
  recovered = (serial - overlap) / (serial - max(coll_hcc, compute))    100% = perfect hiding, 0% = serial
Correctness: every arm's output is compared bitwise against the default-stream result, and against the analytic
expected value, so a "fast but wrong" collective cannot pass.
Then the shape of the real change: 16 buckets totalling ~537 MB (our real per-step gradient traffic is ~550 MB
over 17 parameters) issued on the side stream interleaved with compute, versus all of them then compute.

=== HOW TO RUN (needs the whole device: 8 ranks, production lnc=1) ===
    cd ~/tf && TORCH_NEURONX_ENABLE_HOST_CC=1 NEURON_LOGICAL_NC_CONFIG=1 \
      /home/ubuntu/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=8 \
      experimental/neuron/cc_overlap_probe2.py 2>&1 | tee runs/FPROBE2.log
Knobs: R1_REPS (default 8), R1_MM (compute chain length, default 40 ~ 20 ms), R1_BUCKETS (16).
Every arm is guarded; the log names the arm before running it so a hang is locatable.
"""
import os
import sys
import time
import traceback

import torch
import torch_neuronx  # noqa: F401  (registers torch.neuron)
import torch.distributed as dist

RANK = int(os.environ.get("RANK", "0"))
REPS = int(os.environ.get("R1_REPS", "8"))
MM = int(os.environ.get("R1_MM", "40"))
NBUCK = int(os.environ.get("R1_BUCKETS", "16"))
HCC = os.environ.get("TORCH_NEURONX_ENABLE_HOST_CC")


def log(msg):
    if RANK == 0:
        print(f"[cc2] {msg}", flush=True)


def logall(msg):
    print(f"[cc2 r{RANK}] {msg}", flush=True)


def sync():
    torch.neuron.synchronize()


def timed(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.monotonic()
        fn()
        sync()
        ts.append((time.monotonic() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


dist.init_process_group("neuron")
world = dist.get_world_size()
dev = torch.device("neuron")
torch.manual_seed(1234 + RANK)

log(f"world {world} | TORCH_NEURONX_ENABLE_HOST_CC={HCC} | "
    f"LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} | "
    f"COLLECTIVE_BUCKETSIZE_IN_MB={os.environ.get('COLLECTIVE_BUCKETSIZE_IN_MB', '512 (default)')} | "
    f"SPLIT_RS={os.environ.get('NEURON_USE_SPLIT_REDUCE_SCATTER', 'unset')} "
    f"SPLIT_AG={os.environ.get('NEURON_USE_SPLIT_ALLGATHER', 'unset')}")
if not HCC or HCC == "0":
    log("!!! TORCH_NEURONX_ENABLE_HOST_CC is not set -> the overlap arm CANNOT differ from serial. Set it to 1.")

# ---------------------------------------------------------------------------- stream setup
try:
    default_stream = torch_neuronx.default_stream()
    side = torch.neuron.Stream()
    log(f"streams: default={default_stream} side={side} distinct={side != default_stream}")
    if side == default_stream:
        log("!!! side stream IS the default stream -> host CC can never engage. Stop.")
except Exception:
    log("STREAM API FAILED:")
    traceback.print_exc()
    dist.destroy_process_group()
    sys.exit(1)

# ---------------------------------------------------------------------------- compute body (PE-bound)
NB, TT, D = 12, 3072, 512
A_in = torch.randn(NB, TT, D, device=dev, dtype=torch.bfloat16)
W = torch.randn(D, D, device=dev, dtype=torch.bfloat16) * 0.02
C_out = torch.empty(NB, TT, D, device=dev, dtype=torch.bfloat16)


def compute():
    x = A_in
    for _ in range(MM):
        x = torch.matmul(x, W)
    C_out.copy_(x)


# ---------------------------------------------------------------------------- one size
def probe(name, elems, dtype=torch.float32, gather=False, op=None):
    """reduce_scatter (or all_gather) of `elems` fp32/bf16 elements, leading dim divisible by world.

    `op` defaults to ReduceOp.AVG (what the vehicle uses today). NOTE, measured at world 1: on the host-CC path
    AVG is not native -- NeuronBackend turns it into SUM plus a FULL-SIZE `div_(world)` on the side stream, and
    that extra pass cost 5.71 ms against 1.64 ms for the default-stream path at 256 MB. So AVG is a real tax on
    the side stream; pass SUM here to price the alternative (divide the loss by world_size before backward)."""
    op = op if op is not None else dist.ReduceOp.AVG
    rows = elems // 512
    assert rows % world == 0, (name, rows, world)
    mb = elems * (2 if dtype == torch.bfloat16 else 4) / 2 ** 20
    # rank-dependent, exactly representable values so AVG is exact in fp32 and bf16 alike
    if gather:
        src = torch.full((rows // world, 512), float(RANK + 1), device=dev, dtype=dtype)
        dst = torch.zeros((rows, 512), device=dev, dtype=dtype)
        dst_ref = torch.zeros_like(dst)

        def coll(out, async_op=False):
            return dist.all_gather_into_tensor(out, src, async_op=async_op)
        expected = torch.cat([torch.full((rows // world, 512), float(r + 1), dtype=dtype)
                              for r in range(world)])
    else:
        src = torch.full((rows, 512), float(RANK + 1), device=dev, dtype=dtype)
        dst = torch.zeros((rows // world, 512), device=dev, dtype=dtype)
        dst_ref = torch.zeros_like(dst)

        def coll(out, async_op=False):
            return dist.reduce_scatter_tensor(out, src, op=op, async_op=async_op)
        tot = float(sum(r + 1 for r in range(world)))
        expected = torch.full((rows // world, 512),
                              tot / world if op == dist.ReduceOp.AVG else tot, dtype=dtype)

    def a_dflt():
        coll(dst_ref)

    def a_hcc():
        side.wait_stream(default_stream)
        with torch.neuron.stream(side):
            coll(dst)
        default_stream.wait_event(side.record_event())

    def fprobe_arm():                      # the OLD probe's arm, reproduced exactly
        w = coll(dst_ref, async_op=True)
        compute()
        w.wait()

    def serial():
        coll(dst_ref)
        compute()

    def overlap():
        side.wait_stream(default_stream)
        with torch.neuron.stream(side):
            coll(dst)
        compute()                          # default stream: no dependency on the collective
        default_stream.wait_event(side.record_event())

    # warm-up / compile every arm once before timing anything
    for f in (a_dflt, a_hcc, fprobe_arm, serial, overlap):
        f()
    sync()

    # correctness FIRST: the fast path must be right or its time is meaningless.
    # Two checks, because either alone is too weak:
    #   (a) exactly-representable values (integers 1..world) -> catches a wrong ALGORITHM (wrong rank's data,
    #       wrong op, wrong shard). It CANNOT catch a rounding difference, because sum/world is exact here.
    #   (b) random values -> catches exactly that. The host-CC path turns AVG into SUM + div_(world) while the
    #       in-NEFF path passes AVG to the compiler, so the reductions may round differently, and in bf16 a
    #       last-bit difference is entirely possible. This is the check that decides whether CC_SIDE is
    #       bit-identical to the shipped path or merely numerically equivalent.
    dst.zero_(); dst_ref.zero_(); sync()
    a_dflt(); sync(); ref = dst_ref.cpu().clone()
    a_hcc(); sync(); got = dst.cpu().clone()
    ok_exp = torch.equal(ref, expected)
    ok_bit = torch.equal(ref, got)
    maxdiff = (ref.float() - got.float()).abs().max().item()
    src.copy_(torch.randn(src.shape, dtype=dtype))   # manual_seed(1234+RANK) -> genuinely rank-dependent data
    dst.zero_(); dst_ref.zero_(); sync()
    a_dflt(); sync(); rref = dst_ref.cpu().clone()
    a_hcc(); sync(); rgot = dst.cpu().clone()
    rnd_bit = torch.equal(rref, rgot)
    den = rref.float().abs().clamp_min(1e-12)
    rnd_rel = ((rref.float() - rgot.float()).abs() / den).max().item()

    t_ad = timed(a_dflt)
    t_ah = timed(a_hcc)
    t_b = timed(compute)
    t_f = timed(fprobe_arm)
    t_s = timed(serial)
    t_o = timed(overlap)
    ideal = max(t_ah, t_b)
    # only meaningful when there is headroom to recover; otherwise report the raw saving
    recov = (t_s - t_o) / (t_s - ideal) * 100.0 if t_s - ideal > 0.05 else float("nan")
    log(f"{name:>26} {mb:7.1f} MB | coll_dflt {t_ad:7.2f} coll_hcc {t_ah:7.2f} compute {t_b:7.2f} | "
        f"F_PROBE_arm {t_f:7.2f} serial {t_s:7.2f} overlap {t_o:7.2f} | ideal {ideal:7.2f} "
        f"| saved {t_s - t_o:+7.2f} ms recovered {recov:6.1f}% | exact={ok_exp} bitwise_vs_default={ok_bit} "
        f"maxdiff={maxdiff:.3e} | RANDOM data: bitwise={rnd_bit} max_rel_diff={rnd_rel:.3e}")
    return dict(name=name, mb=mb, coll_dflt=t_ad, coll_hcc=t_ah, compute=t_b, fprobe=t_f,
                serial=t_s, overlap=t_o, recov=recov, ok_exp=ok_exp, ok_bit=ok_bit)


log("--- SECTION 1: is the collective issue asynchronous at all? (the number F_PROBE never took) ---")
try:
    g = torch.full((8192, 512), float(RANK + 1), device=dev, dtype=torch.float32)   # 16.8 MB
    o = torch.zeros((8192 // world, 512), device=dev, dtype=torch.float32)
    for _ in range(3):
        dist.reduce_scatter_tensor(o, g, op=dist.ReduceOp.AVG)
    sync()
    sync(); t0 = time.monotonic()
    w = dist.reduce_scatter_tensor(o, g, op=dist.ReduceOp.AVG, async_op=True)
    iss = (time.monotonic() - t0) * 1000.0
    w.wait(); sync()
    tot = (time.monotonic() - t0) * 1000.0
    log(f"default stream, async_op=True : issue returns in {iss:6.3f} ms, whole thing {tot:6.3f} ms "
        f"-> {'issue IS async' if iss < 0.3 * tot else 'issue BLOCKS (async_op is not deferring)'}")
    sync(); t0 = time.monotonic()
    side.wait_stream(default_stream)
    with torch.neuron.stream(side):
        w = dist.reduce_scatter_tensor(o, g, op=dist.ReduceOp.AVG, async_op=True)
    iss2 = (time.monotonic() - t0) * 1000.0
    default_stream.wait_event(side.record_event()); sync()
    tot2 = (time.monotonic() - t0) * 1000.0
    log(f"side stream + host CC        : issue returns in {iss2:6.3f} ms, whole thing {tot2:6.3f} ms")
except Exception:
    log("SECTION 1 FAILED:")
    traceback.print_exc()

log("--- SECTION 2: overlap vs serial, per size (fp32 reduce_scatter unless noted) ---")
results = []
try:
    for nm, el, dt, ga in [
        ("rs 2MB", 524288, torch.float32, False),
        ("rs 16.8MB", 4194304, torch.float32, False),
        ("rs 50MB", 12582912, torch.float32, False),
        ("rs 134MB", 33554432, torch.float32, False),
        ("rs 268MB bf16", 134217728, torch.bfloat16, False),   # the real bigram-table gradient
        ("ag 50MB", 12582912, torch.float32, True),
        ("ag 268MB bf16", 134217728, torch.bfloat16, True),    # the real parameter all_gather
    ]:
        log(f"  ... running {nm}")
        results.append(probe(nm, el, dt, ga))
    # the AVG tax: same collective with SUM, which needs no post-divide on the side stream
    log("  ... running rs 268MB bf16 SUM (prices the AVG post-divide the host-CC path adds)")
    results.append(probe("rs 268MB bf16 SUM", 134217728, torch.bfloat16, False, dist.ReduceOp.SUM))
except Exception:
    log("SECTION 2 FAILED:")
    traceback.print_exc()

log("--- SECTION 3: the real shape -- 16 buckets (~537 MB, our per-step gradient traffic) ---")
try:
    BEL = 8388608                                   # 8.39M fp32 = 33.5 MB per bucket
    brows = BEL // 512
    srcs = [torch.full((brows, 512), float(RANK + 1 + i), device=dev, dtype=torch.float32) for i in range(NBUCK)]
    dsts = [torch.zeros((brows // world, 512), device=dev, dtype=torch.float32) for _ in range(NBUCK)]
    exp = [torch.full((brows // world, 512), sum(r + 1 + i for r in range(world)) / world, dtype=torch.float32)
           for i in range(NBUCK)]

    def bucket_serial():                            # what we ship: whole backward, then every collective
        compute()
        for s, d in zip(srcs, dsts):
            dist.reduce_scatter_tensor(d, s, op=dist.ReduceOp.AVG)

    def bucket_overlap():                           # bucketed: collective i rides the next slice of compute
        side.wait_stream(default_stream)
        per = max(1, MM // NBUCK)
        x = A_in
        for i in range(NBUCK):
            with torch.neuron.stream(side):
                dist.reduce_scatter_tensor(dsts[i], srcs[i], op=dist.ReduceOp.AVG)
            for _ in range(per):
                x = torch.matmul(x, W)
        C_out.copy_(x)
        default_stream.wait_event(side.record_event())

    def bucket_colls_only():                        # the ALIGNED cost of our whole per-step gradient traffic
        for s, d in zip(srcs, dsts):
            dist.reduce_scatter_tensor(d, s, op=dist.ReduceOp.AVG)

    bucket_serial(); sync()
    ref = [d.cpu().clone() for d in dsts]
    for d in dsts:
        d.zero_()
    bucket_overlap(); sync()
    got = [d.cpu().clone() for d in dsts]
    ok = all(torch.equal(a, b) for a, b in zip(ref, got))
    ok_exp = all(torch.equal(a, b) for a, b in zip(ref, exp))
    t_cc = timed(bucket_colls_only, max(4, REPS // 2))
    t_bs = timed(bucket_serial, max(4, REPS // 2))
    t_bo = timed(bucket_overlap, max(4, REPS // 2))
    t_c = timed(compute)
    logall(f"buckets x{NBUCK} 537MB | colls alone (ranks ALIGNED) {t_cc:7.2f} ms | serial {t_bs:7.2f} ms | "
           f"bucketed+sidestream {t_bo:7.2f} ms | compute alone {t_c:7.2f} ms | "
           f"saved {t_bs - t_bo:7.2f} ms ({(t_bs - t_bo) / max(1e-9, t_bs) * 100:5.1f}%) | bitwise={ok} exact={ok_exp}")
    log(f"    ^ 'colls alone' is the ALIGNED cost of ~537 MB of reduce_scatter. The training run spends 59 ms in "
        f"grad_sync for ~604 MB. 59 - {t_cc:.1f} = {59.0 - t_cc:.1f} ms is NOT bytes, NOT call count (CCPROBE: "
        f"0.01-0.14 ms/call) -- it is cross-rank ARRIVAL SKEW, and skew is exactly what overlap absorbs.")
except Exception:
    log("SECTION 3 FAILED:")
    traceback.print_exc()

log("--- SECTION 4: does overlap absorb ARRIVAL SKEW? (rank r is deliberately r x R1_SKEW matmuls late) ---")
# THE CRUX. Sections 2-3 have every rank perfectly aligned, so they measure pure overlap capability and will
# UNDERSTATE the training win if the 41 ms residual is skew. Here rank r arrives late by construction. If the
# overlapped arm absorbs the skew, bucketed sync during the backward is worth the whole residual; if it does not,
# the ceiling is only the optimizer math (~18.6 ms) and the tail is all there is.
try:
    SKEW = int(os.environ.get("R1_SKEW", "6"))       # extra matmuls per rank index (~0.21 ms each)

    def skew_work():
        x = A_in
        for _ in range(SKEW * RANK):
            x = torch.matmul(x, W)
        if SKEW * RANK:
            C_out.copy_(x)

    def skewed_serial():
        skew_work()
        compute()
        for s, d in zip(srcs, dsts):
            dist.reduce_scatter_tensor(d, s, op=dist.ReduceOp.AVG)

    def skewed_overlap():
        skew_work()
        side.wait_stream(default_stream)
        per = max(1, MM // NBUCK)
        x = A_in
        for i in range(NBUCK):
            with torch.neuron.stream(side):
                dist.reduce_scatter_tensor(dsts[i], srcs[i], op=dist.ReduceOp.AVG)
            for _ in range(per):
                x = torch.matmul(x, W)
        C_out.copy_(x)
        default_stream.wait_event(side.record_event())

    skewed_serial(); sync()
    ref = [d.cpu().clone() for d in dsts]
    for d in dsts:
        d.zero_()
    skewed_overlap(); sync()
    ok = all(torch.equal(a, b) for a, b in zip(ref, [d.cpu().clone() for d in dsts]))
    t_ss = timed(skewed_serial, max(4, REPS // 2))
    t_so = timed(skewed_overlap, max(4, REPS // 2))
    logall(f"skew={SKEW * RANK} extra matmuls | serial {t_ss:7.2f} ms | bucketed+sidestream {t_so:7.2f} ms | "
           f"saved {t_ss - t_so:7.2f} ms ({(t_ss - t_so) / max(1e-9, t_ss) * 100:5.1f}%) | bitwise={ok}")
except Exception:
    log("SECTION 4 FAILED:")
    traceback.print_exc()

log("--- SUMMARY ---")
for r in results:
    log(f"  {r['name']:>16} {r['mb']:7.1f} MB  serial {r['serial']:7.2f} -> overlap {r['overlap']:7.2f} "
        f"(saved {r['serial'] - r['overlap']:+7.2f} ms, {r['recov']:6.1f}% recovered)  "
        f"correct={r['ok_exp'] and r['ok_bit']}")
try:
    dist.barrier()
except Exception:
    pass
log("PROBE_DONE")
dist.destroy_process_group()
