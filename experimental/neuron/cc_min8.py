#!/usr/bin/env python3
"""MINIMAL 8-rank host-CC ladder (2026-09-28). Why does the side-stream collective path die at 8 ranks?

BACKGROUND. `cc_overlap_probe2.py` was verified end-to-end at world 1, every correctness check bitwise, and
then died at world 8 inside SECTION 1 -- twice, once with `NeuronWatchdog: Work timed out: op=reduce_scatter,
seq=2` and once silently. This file replaces it for the diagnosis.

WHAT THE VENDOR'S OWN TEST SAYS (tests/distributed/collective_ops/test_multistream_collectives_with_host_cc.py,
read on the box). Three facts, and they reshaped this file:

  (a) It sets ONLY `TORCH_NEURONX_ENABLE_HOST_CC=1`. `TORCH_NEURONX_ENABLE_ASYNC_NRT` appears nowhere in the
      test or anywhere in the torch_neuron_eager sources. So host CC needs no second framework flag, and this
      ladder does NOT require one. (That variable is documented by AWS and may still be read by the runtime
      library rather than the framework, so there is an optional arm for it -- see launch B -- but it is not
      the hypothesis and nothing here depends on it.)
  (b) `reduce_scatter` IS covered there, alongside all_reduce, all_gather, broadcast, gather, scatter, reduce,
      all_to_all and the coalesced variants. So "the host-CC path cannot carry reduce_scatter" is unlikely.
  (c) Its world size defaults to TWO. The vendor may simply never have exercised this at 8.

  (d) AND THE ONE REAL DIFFERENCE IN OUR IDIOM -- the reason for this rewrite. Entry is identical; the JOIN is
      not. The vendor blocks the HOST:
              stream.wait_stream(default_stream)
              with torch.neuron.stream(stream): <collective>
              stream.synchronize()                         # host blocks
      We create a DEVICE-SIDE dependency instead:
              default_stream.wait_event(side.record_event())
      Ours is the form you actually want for overlap, but it is the part the vendor never exercises, and a
      broken or unsupported event path is exactly what hangs at 8 ranks and passes at 1. So this ladder runs
      the vendor's join FIRST (S6-S8) and ours SECOND (S9-S11). If S6-S8 pass and S9-S11 hang, the blocker is
      OUR join and it is fixable by us. If S6 hangs, host CC itself is broken at 8 ranks.

  AND THE MEASUREMENT SURVIVES EITHER WAY. The primary overlap arm (S12) is built only from calls the vendor
  exercises -- issue on the side stream, run compute on the default stream, THEN block the host -- so we get
  the overlap number even if the event join turns out to be unusable.

WORLD 1 PROVED ALMOST NOTHING HERE, which is why this exists. At world 1 a reduce_scatter and an all_gather are
local copies with input and output the SAME SHAPE; no data crosses ranks, no transport runs, and a join that
does not actually join still passes every bitwise check. Every correctness check below can only pass if data
genuinely crossed ranks (the gather arms check rank world-1's block, not rank 0's).

EVERY RANK PRINTS. `cc_overlap_probe2.py` logged only from rank 0, so a crash on rank != 0 presents exactly as
"rank 0 froze in a collective" or "it exited with no traceback" -- which is both recorded failure modes. Here
every rank prints to stdout and to its own progress file, fsynced, so even a SIGSEGV leaves the last stage
reached on disk.

A HANG IS BOUNDED. Launch with NEURON_RT_EXEC_TIMEOUT=90. The default is 600 s on trn, which is what cost the
first attempt nine minutes.

HOW TO RUN -- launch A, the vendor's own configuration (whole device, 8 ranks, production lnc=1):

    cd ~/tf && mkdir -p runs && \
      TORCH_NEURONX_ENABLE_HOST_CC=1 \
      NEURON_RT_EXEC_TIMEOUT=90 NEURON_LOGICAL_NC_CONFIG=1 R1_OUT=runs \
      /home/ubuntu/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=8 \
      experimental/neuron/cc_min8.py 2>&1 | tee runs/CCMIN8.log

Conditional follow-ups, each only if the one before it leaves a question open:
  B  optional async-runtime arm -- same as A plus TORCH_NEURONX_ENABLE_ASYNC_NRT=1. Only interesting if A
     dies early, or if A works and we want to know whether that variable changes the overlap number.
  C  control, host CC unset -> if the ladder completes then, the FLAG is the blocker, not the side stream.
  D  two ranks, same as A:  --nproc_per_node=2   (only needed if S13 never ran)

READING THE RESULT -- the last stage that prints OK on all 8 ranks is the verdict:
    dies at S2-S4  (ordinary path)   -> host CC breaks the collectives we already ship. Run control C.
    dies at S6     (vendor join, smallest collective)
                                     -> host CC carries nothing at 8 ranks. Axis closed, and it is the
                                        runtime, not us.
    dies at S7     (vendor join, reduce_scatter)
                                     -> that collective specifically. S8 still tells us whether the 28 ms
                                        parameter all_gather can go on the side stream anyway.
    dies at S9-S11 (OUR event join, after the vendor's join passed)
                                     -> THE JOIN IS THE BUG AND IT IS OURS. Host CC works at 8 ranks; the
                                        device-side event dependency does not. Fixable by us -- and S12 still
                                        produces the overlap number using the host-side join.
    reaches S12                      -> we have the overlap number. S12 runs once per join that passed, so a
                                        broken join costs a column, not the measurement. `coll_dflt` against
                                        each `coll_hcc` shows whether the side path is slower per byte -- it
                                        was by ~2.5x at world 1, and overlap has to beat that.
    S13 (sub-world 2 and 4 inside the same launch) localises a world-size dependence without a second slot --
        the vendor defaults to world 2, so this is where their coverage ends and ours begins.

Knobs: R1_REPS (7), R1_MM (40 matmuls ~ 20 ms of compute), R1_STAGE_MAX (14), R1_OUT (progress dir).
"""
import os
import sys
import time
import traceback

import torch
import torch_neuronx  # noqa: F401  (registers torch.neuron)
import torch.distributed as dist

RANK = int(os.environ.get("RANK", "0"))
REPS = int(os.environ.get("R1_REPS", "7"))
MM = int(os.environ.get("R1_MM", "40"))
STAGE_MAX = int(os.environ.get("R1_STAGE_MAX", "14"))
OUTDIR = os.environ.get("R1_OUT", "/tmp")

os.makedirs(OUTDIR, exist_ok=True)
_pf = open(os.path.join(OUTDIR, f"cc_min8_r{RANK}.txt"), "w", buffering=1)


def say(msg):
    """Print from EVERY rank, flushed and fsynced, so a silent crash still leaves the last stage on disk."""
    line = f"[min8 r{RANK}] {msg}"
    print(line, flush=True)
    _pf.write(line + "\n")
    _pf.flush()
    os.fsync(_pf.fileno())


def sync():
    torch.neuron.synchronize()


def stage(n, name):
    """Name the stage before it runs, report OK/FAIL after, never let one failure kill the rest."""
    def deco(fn):
        if n > STAGE_MAX:
            return None
        say(f"S{n} {name}: START")
        t0 = time.monotonic()
        try:
            out = fn()
        except Exception as exc:                                    # noqa: BLE001
            say(f"S{n} {name}: FAIL {type(exc).__name__}: {str(exc)[:300]}")
            traceback.print_exc()
            _pf.write(traceback.format_exc() + "\n")
            _pf.flush()
            return None
        say(f"S{n} {name}: OK {(time.monotonic() - t0) * 1000:.1f} ms"
            + (f" | {out}" if isinstance(out, str) else ""))
        return out
    return deco


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


# ---------------------------------------------------------------------------- S0 init
dist.init_process_group("neuron")
world = dist.get_world_size()
dev = torch.device("neuron")
say(f"S0 init: world={world} rank={RANK} torch={torch.__version__} "
    f"HOST_CC={os.environ.get('TORCH_NEURONX_ENABLE_HOST_CC')} "
    f"ASYNC_NRT={os.environ.get('TORCH_NEURONX_ENABLE_ASYNC_NRT', 'unset (not required)')} "
    f"EXEC_TIMEOUT={os.environ.get('NEURON_RT_EXEC_TIMEOUT')} "
    f"LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} "
    f"CC_STREAM_SPLIT={os.environ.get('NEURON_RT_CC_STREAM_SPLIT', 'unset (3:1 default)')}")

# Two shapes only, so the in-NEFF path compiles as few NEFFs as possible on a cold cache.
SMALL = 512      # (512, 512) fp32 = 1 MB                     -- the smoke size
BIG = 8192       # (8192, 512) fp32 = 16.8 MB                 -- a2a_probe's size: rs 0.595 ms, ag 1.99 ms
assert SMALL % 8 == 0 and BIG % 8 == 0


def mk(rows, val=None):
    return torch.full((rows, 512), float(RANK + 1) if val is None else val, device=dev, dtype=torch.float32)


sm_full = mk(SMALL)
big_full = mk(BIG)
big_shard = torch.zeros((BIG // world, 512), device=dev, dtype=torch.float32)
big_src = mk(BIG // world)
big_gathered = torch.zeros((BIG, 512), device=dev, dtype=torch.float32)

TOT = float(sum(r + 1 for r in range(world)))       # exact in fp32 for world <= 8


def check(t, want, tag):
    got = t.cpu()
    ok = bool(torch.all(got == want).item())
    return f"{tag} correct={ok} first={got.flatten()[0].item():.4f} want={want:.4f}"


@stage(1, "barrier")
def _s1():
    dist.barrier()


@stage(2, "default-stream all_reduce 1MB")
def _s2():
    a = sm_full.clone()
    dist.all_reduce(a, op=dist.ReduceOp.SUM)
    sync()
    return check(a[:1, :1], TOT, "all_reduce")


@stage(3, "default-stream reduce_scatter 16.8MB  <- where probe2 died")
def _s3():
    for _ in range(3):
        dist.reduce_scatter_tensor(big_shard, big_full, op=dist.ReduceOp.AVG)
    sync()
    return check(big_shard[:1, :1], TOT / world, "reduce_scatter")


@stage(4, "default-stream all_gather 16.8MB")
def _s4():
    dist.all_gather_into_tensor(big_gathered, big_src)
    sync()
    # the LAST block belongs to rank world-1: it can only be right if data really crossed ranks. Checking the
    # first block would pass vacuously on rank 0 -- which is why the world-1 verification proved so little.
    return check(big_gathered[-1:, :1], float(world), f"all_gather[rank{world - 1} block]")


side = None
default_stream = None
vendor_ok = False       # set by S6-S8: the host-blocking join works
event_ok = False        # set by S9-S11: our device-side event join works


@stage(5, "create side stream (no collective)")
def _s5():
    global side, default_stream
    default_stream = torch_neuronx.default_stream()
    side = torch.neuron.Stream()
    sync()
    return (f"default={default_stream} side={side} distinct={side != default_stream} "
            f"has_synchronize={hasattr(side, 'synchronize')} has_record_event={hasattr(side, 'record_event')}")


def have_side():
    return side is not None and side != default_stream


# --- the two joins, isolated ------------------------------------------------------------------------------
def on_side_vendor(fn):
    """THE VENDOR'S EXACT IDIOM: enter the side stream, then block the HOST until it drains."""
    side.wait_stream(default_stream)
    with torch.neuron.stream(side):
        fn()
    side.synchronize()


def on_side_event(fn):
    """OUR IDIOM: same entry, but a DEVICE-SIDE dependency instead of a host block. This is the part the
    vendor never exercises and the prime suspect for the 8-rank hang."""
    side.wait_stream(default_stream)
    with torch.neuron.stream(side):
        fn()
    default_stream.wait_event(side.record_event())


# --- S6-S8: vendor join, smallest first -------------------------------------------------------------------
@stage(6, "VENDOR JOIN (stream.synchronize) side all_reduce 1MB  <- first host-CC collective")
def _s6():
    global vendor_ok
    if not have_side():
        return "skipped: no distinct side stream"
    a = sm_full.clone()
    on_side_vendor(lambda: dist.all_reduce(a, op=dist.ReduceOp.SUM))
    sync()
    r = check(a[:1, :1], TOT, "all_reduce")
    vendor_ok = "correct=True" in r
    return r


@stage(7, "VENDOR JOIN side reduce_scatter 16.8MB")
def _s7():
    if not (have_side() and vendor_ok):
        return "skipped"
    big_shard.zero_()
    sync()
    on_side_vendor(lambda: dist.reduce_scatter_tensor(big_shard, big_full, op=dist.ReduceOp.AVG))
    sync()
    return check(big_shard[:1, :1], TOT / world, "reduce_scatter AVG")


@stage(8, "VENDOR JOIN side all_gather 16.8MB  <- the 28 ms parameter all_gather, the fallback prize")
def _s8():
    if not (have_side() and vendor_ok):
        return "skipped"
    big_gathered.zero_()
    sync()
    on_side_vendor(lambda: dist.all_gather_into_tensor(big_gathered, big_src))
    sync()
    return check(big_gathered[-1:, :1], float(world), f"all_gather[rank{world - 1} block]")


# --- S9-S11: our event join, same three collectives --------------------------------------------------------
@stage(9, "EVENT JOIN (record_event/wait_event) side all_reduce 1MB  <- OUR idiom, the prime suspect")
def _s9():
    global event_ok
    if not have_side():
        return "skipped"
    a = sm_full.clone()
    on_side_event(lambda: dist.all_reduce(a, op=dist.ReduceOp.SUM))
    sync()
    r = check(a[:1, :1], TOT, "all_reduce")
    event_ok = "correct=True" in r
    return r


@stage(10, "EVENT JOIN side reduce_scatter 16.8MB")
def _s10():
    if not (have_side() and event_ok):
        return "skipped"
    big_shard.zero_()
    sync()
    on_side_event(lambda: dist.reduce_scatter_tensor(big_shard, big_full, op=dist.ReduceOp.AVG))
    sync()
    return check(big_shard[:1, :1], TOT / world, "reduce_scatter AVG")


@stage(11, "EVENT JOIN side all_gather 16.8MB")
def _s11():
    if not (have_side() and event_ok):
        return "skipped"
    big_gathered.zero_()
    sync()
    on_side_event(lambda: dist.all_gather_into_tensor(big_gathered, big_src))
    sync()
    return check(big_gathered[-1:, :1], float(world), f"all_gather[rank{world - 1} block]")


# ---------------------------------------------------------------------------- S12 the measurement
NB, TT, D = 12, 3072, 512
A_in = torch.randn(NB, TT, D, device=dev, dtype=torch.bfloat16)
W = torch.randn(D, D, device=dev, dtype=torch.bfloat16) * 0.02
C_out = torch.empty(NB, TT, D, device=dev, dtype=torch.bfloat16)


def compute():
    x = A_in
    for _ in range(MM):
        x = torch.matmul(x, W)
    C_out.copy_(x)


@stage(12, "MEASURE overlap vs serial (reduce_scatter and all_gather, 16.8MB)")
def _s12():
    if not (have_side() and (vendor_ok or event_ok)):
        return "skipped: the side stream never carried a correct collective"
    lines = []
    for tag, coll in (
        ("reduce_scatter 16.8MB",
         lambda: dist.reduce_scatter_tensor(big_shard, big_full, op=dist.ReduceOp.AVG)),
        ("all_gather     16.8MB",
         lambda: dist.all_gather_into_tensor(big_gathered, big_src)),
    ):
        def serial():
            coll()
            compute()

        def mk_overlap(join):
            """The overlap construction: issue the collective on the side stream, run compute on the default
            stream with no dependency on it, and only THEN join. With the vendor's host-side join this is
            still a real overlap, because the host block happens after compute has been issued -- which is
            why the measurement survives even if our event join turns out to be unusable."""
            def f():
                side.wait_stream(default_stream)
                with torch.neuron.stream(side):
                    coll()
                compute()
                join()
            return f

        variants = []
        if vendor_ok:
            variants.append(("vendor", lambda: on_side_vendor(coll),
                             mk_overlap(lambda: side.synchronize())))
        if event_ok:
            variants.append(("event", lambda: on_side_event(coll),
                             mk_overlap(lambda: default_stream.wait_event(side.record_event()))))

        for f in [coll, compute, serial] + [x for _, a, o in variants for x in (a, o)]:
            f()                                             # warm / compile every arm before timing
        sync()
        t_cd = timed(coll)
        t_cp = timed(compute)
        t_se = timed(serial)
        parts = [f"coll_dflt {t_cd:7.2f} compute {t_cp:7.2f} | serial {t_se:7.2f}"]
        for nm, alone, ov in variants:
            t_a = timed(alone)
            t_o = timed(ov)
            ideal = max(t_a, t_cp)
            rec = (t_se - t_o) / (t_se - ideal) * 100.0 if (t_se - ideal) > 0.05 else float("nan")
            parts.append(f"| [{nm} join] coll_hcc {t_a:7.2f} overlap {t_o:7.2f} ideal {ideal:7.2f} "
                         f"saved {t_se - t_o:+7.2f} ms recovered {rec:6.1f}%")
        lines.append(f"{tag} | " + " ".join(parts))
        say("  " + lines[-1])
    return " ;; ".join(lines)


# ---------------------------------------------------------------------------- S13 world-size localisation
@stage(13, "SUB-WORLD 2 and 4 on the side stream (the vendor's coverage stops at 2)")
def _s13():
    if not have_side():
        return "skipped"
    out = []
    for gsz in (2, 4):
        if world % gsz:
            continue
        groups = [dist.new_group(list(range(b, b + gsz))) for b in range(0, world, gsz)]
        mine = groups[RANK // gsz]
        want = float(sum((RANK // gsz) * gsz + i + 1 for i in range(gsz)))
        for jn, joiner in (("vendor", on_side_vendor), ("event", on_side_event)):
            a = sm_full.clone()
            try:
                joiner(lambda: dist.all_reduce(a, op=dist.ReduceOp.SUM, group=mine))
                sync()
                out.append(f"world{gsz} {jn}-join all_reduce " + check(a[:1, :1], want, ""))
            except Exception as exc:                                   # noqa: BLE001
                out.append(f"world{gsz} {jn}-join all_reduce FAILED {type(exc).__name__}: {str(exc)[:160]}")
            say("  " + out[-1])
    return " ;; ".join(out)


@stage(14, "final barrier")
def _s14():
    dist.barrier()


say(f"MIN8_DONE vendor_join_ok={vendor_ok} event_join_ok={event_ok}")
try:
    dist.destroy_process_group()
except Exception:                                                  # noqa: BLE001
    pass
sys.stdout.flush()
