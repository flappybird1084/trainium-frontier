#!/usr/bin/env python3
"""Device experiment (ON the box, single core, ~4 min): do this runtime's NeuronStreams give REAL device
concurrency, and is eager dispatch asynchronous at all?

WHY THIS EXISTS. Ledger `F_PROBE` (2026-09-16) concluded "collectives are SERIALIZED against compute on this
runtime" from `experimental/neuron/cc_overlap_probe.py`. Its own log contains the refutation of its method:

    reduce_scatter 50MB : async coll + compute 4.10 ms | wait-then-compute 4.12 ms
    reduce_scatter 537MB: async coll + compute 12.30 ms | wait-then-compute 12.13 ms
    all_gather     50MB : async coll + compute 4.09 ms | wait-then-compute 4.03 ms

`wait-then-compute` is the DELIBERATELY SERIAL control. It is identical to the "overlapped" case at every size
(and at 537 MB the "overlapped" case is SLOWER). A probe whose overlap arm equals its serial control did not
create an overlap opportunity; it measured the API, not the hardware.

The mechanism, read out of the runtime source on the box
(`~/workspace/torch_neuron_eager/torch_neuronx/csrc/distributed/NeuronBackend.cpp`):

    bool NeuronBackend::UseHostCC() const {
      static const bool hostcc_enabled = getenv("TORCH_NEURONX_ENABLE_HOST_CC") ...;
      return (current != default_stream) && hostcc_enabled;      // <-- BOTH conditions
    }
    ...
    if (UseHostCC()) SubmitHostCCCollective(...);   // NRT host-collectives API, dedicated collectives queue
    else             SubmitInNeffCollective(...);   // a NEFF on the CURRENT stream == the compute queue

So a collective can only leave the compute queue when it is issued on a NON-DEFAULT stream with
TORCH_NEURONX_ENABLE_HOST_CC=1. F_PROBE did neither, so its serialization was structural, not hardware.
`async_op=True` alone cannot help: the work is on the same stream either way.

THIS FILE tests the PRECONDITIONS, which need only one core and no ranks:
  1. Is eager dispatch asynchronous?  (issue-return time vs execution time.) If dispatch blocks, nothing can
     ever overlap and the whole idea dies here.
  2. Does `torch.neuron.Stream()` give a stream distinct from the default stream, and how many distinct ones?
     (Source says `constexpr int kStreamPoolSize = 1`, static_asserted equal to `nrt::kCollectivesQueues` -- so
     the expectation is exactly ONE side stream, i.e. one collectives queue. Confirm it.)
  3. Does COMPUTE on the side stream overlap COMPUTE on the default stream? This is the sibling question to
     nki_coschedule_test.py's 19%: if two streams overlap compute, the concurrency axis is wide open for
     everybody. If they do not, streams are useful ONLY for collectives (which run on a different queue) and
     nobody should try to parallelise compute with them.

Body A is a PE-bound matmul chain. Body B is a Vector-bound elementwise chain. Independent inputs.

Usage on the box (gated single-core runner):
    bash trn2_job.sh STREAM1 "TORCH_NEURONX_ENABLE_HOST_CC=1 python nki/nki_stream_probe.py"
Knobs: R1_MM (matmul chain length, default 20), R1_EW (elementwise chain length, default 40), R1_REPS (20).
"""
import os, sys, time, traceback

sys.path.insert(0, os.path.expanduser("~/tf"))
import torch, torch_neuronx  # noqa: F401  (import registers torch.neuron)

dev = torch.device("neuron")
torch.manual_seed(0)

MM = int(os.environ.get("R1_MM", "20"))
EW = int(os.environ.get("R1_EW", "40"))
REPS = int(os.environ.get("R1_REPS", "20"))
NB, TT, D = 12, 3072, 512          # A: (12, 3072, 512) @ (512, 512), MM times   -- PE
NR, ND = 16384, 512                # B: (16384, 512) fp32 elementwise, EW times  -- Vector


def log(m):
    print(f"[stream] {m}", flush=True)


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
    return ts[len(ts) // 2], ts[0], ts[-1]


log(f"torch {torch.__version__}  HOST_CC={os.environ.get('TORCH_NEURONX_ENABLE_HOST_CC')}  "
    f"VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES')} LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG')}")

A_in = torch.randn(NB, TT, D, device=dev, dtype=torch.bfloat16)
W = torch.randn(D, D, device=dev, dtype=torch.bfloat16) * 0.02
B_in = torch.randn(NR, ND, device=dev, dtype=torch.float32)
B_out = torch.empty_like(B_in)


def body_a():
    x = A_in
    for _ in range(MM):
        x = torch.matmul(x, W)
    return x


def body_b():
    t = B_in
    for _ in range(EW):
        t = t * 1.0001 + 0.0001
    B_out.copy_(t)
    return t


# ---------------------------------------------------------------- 0. warm up / compile every shape once
for _ in range(3):
    body_a(); body_b()
sync()
log("warm")

# ---------------------------------------------------------------- 1. is eager dispatch asynchronous?
# Issue body A WITHOUT syncing afterwards and time only the issue. If dispatch is async this is much smaller
# than the execution time; if it blocks, the two are equal and no overlap of any kind is reachable.
sync()
t0 = time.monotonic(); body_a(); issue_a = (time.monotonic() - t0) * 1000.0
sync(); full_a = (time.monotonic() - t0) * 1000.0
sync()
t0 = time.monotonic(); body_b(); issue_b = (time.monotonic() - t0) * 1000.0
sync(); full_b = (time.monotonic() - t0) * 1000.0
log(f"1. dispatch: A issue {issue_a:7.2f} ms of {full_a:7.2f} ms total | B issue {issue_b:7.2f} of {full_b:7.2f} "
    f"-> {'ASYNC' if issue_a < 0.5 * full_a else 'SYNCHRONOUS (nothing can overlap; stop here)'}")

# ---------------------------------------------------------------- 2. stream identity
try:
    default_stream = torch_neuronx.default_stream()
    s1 = torch.neuron.Stream()
    s2 = torch.neuron.Stream()
    cur = torch_neuronx.current_stream()
    log(f"2. streams: default={default_stream} current={cur} s1={s1} s2={s2}")
    log(f"   s1 != default: {s1 != default_stream} | s1 == s2 (pool size 1?): {s1 == s2}")
except Exception:
    log("2. STREAM API FAILED:")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------- 3. alone / serial / two-stream
a, alo, ahi = timed(body_a)
b, blo, bhi = timed(body_b)


def serial():
    body_a(); body_b()


ser, slo, shi = timed(serial)


def two_stream():
    s1.wait_stream(default_stream)
    with torch.neuron.stream(s1):
        body_b()
    body_a()                       # default stream
    ev = s1.record_event()
    default_stream.wait_event(ev)


ts, tlo, thi = timed(two_stream)


def two_stream_swapped():
    s1.wait_stream(default_stream)
    with torch.neuron.stream(s1):
        body_a()
    body_b()
    ev = s1.record_event()
    default_stream.wait_event(ev)


tsw, wlo, whi = timed(two_stream_swapped)

log(f"3. A alone          {a:7.2f} ms  [{alo:.2f}-{ahi:.2f}]   ({MM} matmuls {NB}x{TT}x{D} @ {D}x{D}, PE)")
log(f"   B alone          {b:7.2f} ms  [{blo:.2f}-{bhi:.2f}]   ({EW} elementwise {NR}x{ND} fp32, Vector)")
log(f"   serial A;B       {ser:7.2f} ms  [{slo:.2f}-{shi:.2f}]")
log(f"   B on s1 | A dflt {ts:7.2f} ms  [{tlo:.2f}-{thi:.2f}]")
log(f"   A on s1 | B dflt {tsw:7.2f} ms  [{wlo:.2f}-{whi:.2f}]")
ideal = max(a, b)
best = min(ts, tsw)
recov = (ser - best) / max(1e-9, ser - ideal) * 100.0
log(f"   serial {ser:.2f} | ideal max(A,B) {ideal:.2f} | best two-stream {best:.2f} "
    f"-> recovers {recov:.1f}% of ideal compute-compute overlap")
log(f"   VERDICT: {'STREAMS OVERLAP COMPUTE' if best < 0.9 * ser else 'streams do NOT overlap compute'}")

# correctness: the two-stream arm must produce the same B_out as the serial arm
sync()
body_b(); ref = B_out.clone()
sync()
two_stream(); sync()
same = torch.equal(ref.cpu(), B_out.cpu())
log(f"   B_out bitwise identical serial vs two-stream: {same}")
log("PROBE_DONE")
