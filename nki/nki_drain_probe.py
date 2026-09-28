#!/usr/bin/env python3
"""Device probe (single core): what does pinning a LIVE kernel's PSUM->SBUF drains to a chosen engine do to its wall time?

WHY. On a NON-SHIPPING attention backward, three variants differing in nothing but `engine=` on 8 PSUM drains measured
    SCALAR 1.836 / 1.825 ms   VECTOR 2.225 / 2.213 ms   ->  -17.5%, replicated to +-0.5%, outputs bit-identical
and DEFAULT (compiler picks) moved 12% between the two runs while both pinned variants held to 0.6%. Our LIVE kernels
contain 28 PSUM drain sites in the LAYER=op path and pass `engine=` on NONE of them. The vendor pins all four of its
big ones to Scalar. This file converts that borrowed number into a per-kernel number on the kernels we actually ship.

The sign is EXPECTED TO FLIP between forward and backward: the forward graph is Scalar-dominated (~61% busy) and the
backward Vector-dominated, so this is never a blanket change. mlp_bwd is the strongest candidate - it issues ZERO
`activation` instructions, so its Scalar engine is idle for the whole kernel.

HOW, and why there is no copied kernel body in this file. The probe READS THE VEHICLE'S OWN SOURCE TEXT, extracts each
target kernel verbatim, and emits three renamed clones per kernel whose only difference is a literal `engine=` kwarg
appended to the tensor_copy calls WHOSE SOURCE IS A PSUM TILE. Nothing else is touched, so a variant cannot differ from
the shipped kernel in anything but the engine argument, and the file cannot go stale when the vehicle changes.

THREE TRAPS THIS DESIGN AVOIDS, each of which killed an earlier harness:
  1. The NKI tracer resolves module globals and kernel locals, NOT closure cells and NOT Python branches on a kernel
     argument. `def k(..., eng: int = 0)` + `if eng == 1:` inside the body fails EVERY variant including eng=0 with
     "error: unsupported expression"; a closure over `eng` fails with "error: unbound variable 'eng'". Reproduced in
     the local simulator. So the engine must be a LITERAL IN THE SOURCE, which is what the text patch produces.
  2. A monkeypatched nisa.* helper is compiled as part of the kernel, so it may not use **kwargs, keyword expansion,
     or getattr. No monkeypatching here at all.
  3. A blanket shim would also pin the SBUF->SBUF tensor_copy calls, which are not drains. pre_attn_bwd_ko has 4 PSUM
     drains and 10 SBUF copies; pinning all 14 measures something else. This patches the 4.

WHAT IS NOT MOVABLE, so it is not attempted (confirmed in the local simulator against NKI's own validator):
  - GpSimd cannot take a drain at all: "tensor_copy src must be in [sbuf], got psum".
  - tensor_tensor can never go to Scalar: "tensor_tensor engine must be one of [vector, gpsimd, unknown]". That closes
    mlp_bwd's weight-gradient accumulator (tensor_tensor with a PSUM operand) as a Scalar candidate, permanently.
  So the movable set really is tensor_copy drains, Vector <-> Scalar only.

HOW TO READ IT.
  SCALAR < VECTOR on a BACKWARD kernel -> pin that kernel's drains to Scalar.
  SCALAR > VECTOR on a FORWARD kernel  -> expected and fine; the right pin differs per kernel. That asymmetry IS the
        result, and it is why a blanket change would be wrong.
  all three within ~1%                 -> drains are not on that kernel's critical path. Say so and close it.
Every variant must be BIT-IDENTICAL to DEFAULT (max|diff| 0.000e+00). The engine choice cannot change arithmetic, so
any nonzero difference is a bug in this file, not a result.
DEFAULT is reported from both timing passes precisely because the compiler's own choice is the unstable one.

Usage on the box (gated, single core):
    cd ~/tf && bash trn2_job.sh DRAIN /home/ubuntu/workspace/native_venv/bin/python nki/nki_drain_probe.py
Env:
    R1_KERNELS  comma list, default "mlp_bwd,post_attn_fwd_h,pre_attn_bwd_ko,head_bwd,mlp_fwd"
                (3 NEFFs each, so trim this to split the run across two launches)
    R1_ITERS    timed iterations per pass, default 20
    R1_VEHICLE  path to the vehicle source, default ../submission_raw/train3_canon.py beside this file
    R1_EMIT_ONLY=1  generate and print the patch plan, compile nothing (works off-box)
"""
import os, re, sys, ast, time, textwrap, traceback

# Both caches off: a stale NEFF would silently void the whole run. Must be set before torch_neuronx is imported.
os.environ.setdefault("NKI_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("TORCH_NEURONX_NEFF_DISABLE_CACHE", "1")
os.environ.setdefault("NKI_STRICT", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/tf"))
sys.path.insert(0, HERE)

VEHICLE = os.environ.get("R1_VEHICLE", os.path.join(HERE, os.pardir, "submission_raw", "train3_canon.py"))
KERNELS = [k for k in os.environ.get(
    "R1_KERNELS", "mlp_bwd,post_attn_fwd_h,pre_attn_bwd_ko,head_bwd,mlp_fwd").split(",") if k.strip()]
ITERS = int(os.environ.get("R1_ITERS", "20"))
EMIT_ONLY = os.environ.get("R1_EMIT_ONLY", "0") != "0"

# production shape: one layer, one rank, one micro-step
B, TLEN, D, NH, MLP_RATIO, V = 16, 1024, 512, 4, 6, 8192
T, H, HD = B * TLEN, D * MLP_RATIO, D // NH

VARIANTS = [("default", None), ("scalar", "nisa.scalar_engine"), ("vector", "nisa.vector_engine")]

# Calls per optimizer step at the shipping launch (8 ranks, lnc=1, DEPTH=12, one micro-step per step because
# TOTAL_BATCH_SIZE == world x device_batch x SEQ_LEN). This is what turns a per-call ms into a step-level ms.
# mlp_fwd is 0 ON PURPOSE: under LAYER_IMPL="op" (the default) the forward MLP is INLINED INTO post_attn_fwd_h and
# the standalone mlp_fwd kernel is wrapped but never called. It is measured here only as the closest analog of the
# non-shipping kernel the -17.5% came from.
CALLS_PER_STEP = {"mlp_bwd": 12, "post_attn_fwd_h": 12, "pre_attn_bwd_ko": 12, "pre_attn_fwd_ko": 12,
                  "head_bwd": 1, "head_fwd": 1, "mlp_fwd": 0}
STEP_SECONDS, BPB_PER_EFOLD, BUDGET_SECONDS = 0.402, 0.035, 1800.0


# --------------------------------------------------------------------------- extract + patch the vehicle's own source
def _kernel_source(lines, name):
    """The verbatim `def <name>(...)` block from the vehicle, dedented to column 0. The vehicle's kernels live inside
    `if NKI_AVAILABLE:` so they are indented; dedent is what makes them re-emittable at module level."""
    for i, line in enumerate(lines):
        m = re.match(r"(\s*)def %s\(" % re.escape(name), line)
        if not m:
            continue
        ind = len(m.group(1))
        j = i + 1
        while j < len(lines):
            s = lines[j]
            if s.strip() and (len(s) - len(s.lstrip())) <= ind:
                break
            j += 1
        return textwrap.dedent("\n".join(lines[i:j])).rstrip(), i + 1
    raise SystemExit(f"kernel {name!r} not found in {VEHICLE}")


def _buffer_map(body):
    """var -> 'psum' | 'sbuf' | 'shared_hbm', resolving the three forms the vehicle uses plus aliases
    (`x1 = x1_tiles[tb]`). Run to a fixpoint so an alias declared before its target still resolves."""
    buf, alias = {}, {}
    for line in body.split("\n"):
        m = re.search(r"(\w+)\s*=\s*nl\.ndarray\(.*buffer=nl\.(\w+)", line)
        if m:
            buf[m.group(1)] = m.group(2)
            continue
        m = re.search(r"(\w+)\.append\(\s*nl\.ndarray\(.*buffer=nl\.(\w+)", line)
        if m:
            buf[m.group(1)] = m.group(2)
            continue
        m = re.search(r"(\w+)\s*=\s*\[\s*nl\.ndarray\(.*buffer=nl\.(\w+)", line)
        if m:
            buf[m.group(1)] = m.group(2)
            continue
        m = re.match(r"\s*(\w+)\s*=\s*(\w+)\s*(\[|$)", line)
        if m and m.group(1) != m.group(2):
            alias.setdefault(m.group(1), m.group(2))
    for _ in range(6):
        for a, tgt in alias.items():
            if a not in buf and tgt in buf:
                buf[a] = buf[tgt]
    return buf


def _close_paren(line, open_idx):
    depth = 0
    for k in range(open_idx, len(line)):
        c = line[k]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError("unbalanced tensor_copy call (multi-line?): " + line.strip())


def drain_sites(body):
    """Indices of body lines holding a nisa.tensor_copy whose src is a confirmed PSUM tile, plus the skipped ones."""
    buf = _buffer_map(body)
    hits, skipped = [], []
    for i, line in enumerate(body.split("\n")):
        if "nisa.tensor_copy(" not in line:
            continue
        m = re.search(r"src=([A-Za-z_]\w*)", line)
        src = m.group(1) if m else "?"
        where = buf.get(src, "UNKNOWN")
        if where == "psum":
            hits.append((i, src))
        else:
            skipped.append((i, src, where))
    return hits, skipped


def patch(body, newname, engine_expr, sites):
    """Rename and append `, engine=<expr>` to exactly the PSUM-src tensor_copy calls. Nothing else changes."""
    lines = body.split("\n")
    lines[0] = re.sub(r"^def \w+\(", "def %s(" % newname, lines[0], count=1)
    if engine_expr is not None:
        for i, _src in sites:
            line = lines[i]
            o = line.index("nisa.tensor_copy(") + len("nisa.tensor_copy")
            c = _close_paren(line, o)
            lines[i] = line[:c] + ", engine=" + engine_expr + line[c:]
    return "@nki.jit\n" + "\n".join(lines)


def build_module():
    src = open(VEHICLE).read()
    lines = src.split("\n")
    consts = {}
    for cname, default in (("_NKI_P", "128"), ("_FMAX", "512"), ("_HVC", "512")):
        m = re.search(r"^\s*%s\s*=\s*(\d+)" % cname, src, re.M)
        consts[cname] = m.group(1) if m else default
    rope = re.search(r"^ROPE_FRAC\s*=.*$", src, re.M)
    header = [
        "# AUTO-GENERATED by nki_drain_probe.py - verbatim clones of the vehicle's kernels, one literal kwarg apart.",
        "import os", "import nki", "import nki.isa as nisa", "import nki.language as nl", "",
    ] + ["%s = %s" % (k, v) for k, v in consts.items()] + [
        rope.group(0) if rope else 'ROPE_FRAC = float(os.environ.get("NEURON_COMPETITION_R1_ROPE_FRAC", "0.5"))',
        "",
    ]
    # post_attn_fwd_h names _fq8 under `if fq:`, and the vehicle never defines it (fq is 0 on every live call).
    # The specializer prunes the dead branch before name resolution, so leaving it undefined - exactly as the vehicle
    # does - specializes cleanly, while a stub whose body contains `raise` does NOT ("NKI does not support 'raise'").
    if re.search(r"^\s*def _fq8\(", src, re.M):
        fq_body, _ = _kernel_source(lines, "_fq8")
        header += [fq_body, "", ""]
    else:
        header += ["# _fq8 is intentionally undefined here, as it is in the vehicle (only reachable under fq=1).",
                   "", ""]
    parts, plan = list(header), {}
    for name in KERNELS:
        body, ln = _kernel_source(lines, name)
        sites, skipped = drain_sites(body)
        plan[name] = (ln, sites, skipped)
        for tag, eng in VARIANTS:
            parts.append(patch(body, "%s__%s" % (name, tag), eng, sites))
            parts.append("")
            parts.append("")
    if "head_bwd" in KERNELS and "head_fwd" not in KERNELS:
        # one extra, UNPATCHED clone: head_bwd recomputes a softmax from lse, so it needs a real lse/rstd/h or dW
        # comes back nan and the bit-exactness check is meaningless. Not timed.
        body, _ln = _kernel_source(lines, "head_fwd")
        parts.append(patch(body, "head_fwd__prime", None, []))
        parts.append("")
        parts.append("")
    return "\n".join(parts), plan


# --------------------------------------------------------------------------- inputs at the production shape
def make_inputs(torch, dev, name, cache):
    bf, f32 = torch.bfloat16, torch.float32
    g = lambda *s, sc=1.0, dt=bf: (torch.randn(*s) * sc).to(dt).to(dev)
    if name == "mlp_fwd":
        return (g(T, D, sc=0.5), g(H, D, sc=0.03), g(D, H, sc=0.02)), {}
    if name == "mlp_bwd":
        # pre_in is relu(pre) as the LIVE forward saves it, not pre: non-negative, so feed |.|
        return (g(T, D, sc=0.5), g(T, D, sc=0.1), (torch.randn(H, T) * 0.3).abs().to(bf).to(dev),
                g(H, D, sc=0.03), g(D, H, sc=0.02)), {}
    if name == "post_attn_fwd_h":
        return ((g(B * NH, TLEN, HD, sc=0.5), g(D, D, sc=0.04), g(T, D, sc=0.5), g(H, D, sc=0.03), g(D, H, sc=0.02)),
                dict(eps=float(torch.finfo(bf).eps), n_head=NH, t_len=TLEN))
    if name == "pre_attn_fwd_ko":
        pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        fr = torch.arange(HD // 2, dtype=torch.float32).unsqueeze(0)
        ang = (pos * (10000.0 ** (-fr / (HD // 2)))).to(dev)
        dstart = (torch.arange(T) % TLEN == 0).float().reshape(T, 1).to(dev)
        return ((g(T, D, sc=0.5), g(T, D, sc=0.5), g(T, D, sc=0.5), g(D, D, sc=0.04), g(D, D, sc=0.04),
                 g(D, D, sc=0.04), ang.cos().contiguous(), ang.sin().contiguous(), cache["scal"], dstart),
                dict(n_head=NH, eps=float(torch.finfo(bf).eps), t_len=TLEN, q_scale=0.1, hq=HD // 4))
    if name == "pre_attn_bwd_ko":
        pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        fr = torch.arange(HD // 2, dtype=torch.float32).unsqueeze(0)
        ang = (pos * (10000.0 ** (-fr / (HD // 2)))).to(dev)
        dstart = (torch.arange(T) % TLEN == 0).float().reshape(T, 1).to(dev)
        return ((g(B * NH, TLEN, HD, sc=0.1), g(B * NH, TLEN, HD, sc=0.1), g(B * NH, TLEN, HD, sc=0.1),
                 g(NH, T, HD, sc=0.5), g(NH, T, HD, sc=0.5), g(T, D, sc=0.5), g(T, D, sc=0.1),
                 ang.cos().contiguous(), ang.sin().contiguous(), dstart,
                 g(D, D, sc=0.04), g(D, D, sc=0.04), g(D, D, sc=0.04)),
                dict(n_head=NH, eps=float(torch.finfo(bf).eps), scale=0.1, hq=HD // 4))
    if name == "head_fwd":
        return ((g(T, D, sc=0.5), g(D, V, sc=0.03), cache["tgt"], cache["scal"]),
                dict(eps=float(torch.finfo(bf).eps)))
    if name == "head_bwd":
        # h/rstd/lse come from a real head_fwd so the recomputed softmax is in range and dW is not nan.
        h, rstd, lse = cache["head_h"], cache["head_rstd"], cache["head_lse"]
        wT = cache["head_wT"]
        return ((h, rstd, lse, (torch.randn(T, 1) * 0.01).to(f32).to(dev), wT.t().contiguous(), wT,
                 cache["tgt"], cache["scal"]), {})
    raise SystemExit("no input recipe for kernel %r - add one" % name)


def main():
    text, plan = build_module()
    print(f"vehicle : {os.path.realpath(VEHICLE)}")
    print(f"kernels : {', '.join(KERNELS)}   variants: default/scalar/vector   iters: {ITERS}")
    print(f"shape   : T={T} D={D} H={H} NH={NH} HD={HD} V={V} (b={B} t={TLEN})\n")
    print("PATCH PLAN (only tensor_copy calls with a confirmed PSUM source are pinned)")
    for name, (ln, sites, skipped) in plan.items():
        print(f"  {name:18s} vehicle line {ln:5d}   PSUM drains pinned: {len(sites):2d}   "
              f"non-PSUM copies left alone: {len(skipped)}")
        if not sites:
            print(f"      *** {name} has ZERO patchable drains in this vehicle. Point R1_VEHICLE at the UNMODIFIED "
                  f"vehicle (train3_canon.py): once the drains are routed through a helper such as _e2_drain(), "
                  f"there is no nisa.tensor_copy left for this probe to patch and every variant is identical.",
                  flush=True)
        unknown = [s for s in skipped if s[2] == "UNKNOWN"]
        if unknown:
            print(f"      NOTE unresolved src buffers (left unpinned, conservative): "
                  f"{', '.join(sorted({u[1] for u in unknown}))}")
    gen = os.path.join(HERE, "_drain_gen.py")
    open(gen, "w").write(text)
    print(f"\ngenerated {gen}  ({text.count(chr(10))} lines)\n", flush=True)
    if EMIT_ONLY:
        return

    import torch
    import torch_neuronx
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores
    import _drain_gen

    dev = torch.device("neuron")
    grid = (int(get_logical_neuron_cores()),)
    torch.manual_seed(0)
    cache = {
        "scal": torch.tensor([[11.0, 11.0, 1.0, 1.0]], dtype=torch.float32).repeat(128, 1).contiguous().to(dev),
        "tgt": torch.randint(0, V, (T, 1), dtype=torch.int32).to(dev),
    }

    def timed(fn, n=ITERS):
        fn()
        torch_neuronx.synchronize()
        t0 = time.monotonic()
        for _ in range(n):
            fn()
        torch_neuronx.synchronize()
        return (time.monotonic() - t0) / n * 1000.0

    if "head_bwd" in KERNELS:
        # one extra compile, to give head_bwd a realistic lse/rstd instead of a softmax that overflows
        print("priming head_bwd inputs with a real head_fwd ...", flush=True)
        try:
            prime = getattr(_drain_gen, "head_fwd__prime", None) or getattr(_drain_gen, "head_fwd__default")
            wT = (torch.randn(D, V) * 0.03).bfloat16().to(dev)
            hin = (torch.randn(T, D) * 0.5).bfloat16().to(dev)
            _loss, lse, h, rstd = wrap_nki(prime)[grid](hin, wT, cache["tgt"], cache["scal"],
                                                        eps=float(torch.finfo(torch.bfloat16).eps))
            torch_neuronx.synchronize()
            cache.update(head_h=h.contiguous(), head_rstd=rstd.float().contiguous(),
                         head_lse=lse.float().contiguous(), head_wT=wT)
            print(f"    ok: lse mean {lse.float().mean().item():.3f}  rstd mean {rstd.float().mean().item():.4f}\n",
                  flush=True)
        except Exception as e:
            print(f"    head_fwd priming FAILED ({type(e).__name__}: {str(e)[:160]}); head_bwd will be skipped\n",
                  flush=True)
            traceback.print_exc(limit=3)
            KERNELS.remove("head_bwd")

    print("=" * 104, flush=True)
    for name in KERNELS:
        print(f"\n{name}   ({len(plan[name][1])} PSUM drains pinned)", flush=True)
        try:
            args, kwargs = make_inputs(torch, dev, name, cache)
        except Exception as e:
            print(f"    input build FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        calls, ref = {}, None
        for tag, _eng in VARIANTS:
            fn = getattr(_drain_gen, f"{name}__{tag}")
            k = wrap_nki(fn)
            calls[tag] = lambda kk=k: kk[grid](*args, **kwargs)
            try:
                out = calls[tag]()
                torch_neuronx.synchronize()
                outs = out if isinstance(out, (tuple, list)) else (out,)
                flat = [o.float().cpu() for o in outs]
                if ref is None:
                    ref, note = flat, "reference"
                else:
                    md = max((a - b).abs().max().item() for a, b in zip(flat, ref))
                    note = f"max|diff| vs DEFAULT {md:.3e}" + ("" if md == 0.0 else "   <-- NOT BIT-EXACT, BUG")
                print(f"    {tag:8s} compiled+ran   {note}", flush=True)
            except Exception as e:
                print(f"    {tag:8s} FAILED: {type(e).__name__}: {str(e)[:220]}", flush=True)
                traceback.print_exc(limit=3)
                calls.pop(tag, None)
        # two timing passes in ONE process: cross-run repeats of the same kernel have moved 16%, within-process
        # consecutive comparisons repeat to 0.5%, so only the in-process pair is worth anything.
        ms = {tag: [] for tag in calls}
        for _p in range(2):
            for tag in calls:
                ms[tag].append(timed(calls[tag]))
        if "default" in ms:
            base = min(ms["default"])
            print(f"    {'':8s} {'pass1':>9s} {'pass2':>9s} {'best':>9s}   vs DEFAULT", flush=True)
            for tag in ("default", "scalar", "vector"):
                if tag not in ms:
                    continue
                a, b = ms[tag]
                best = min(a, b)
                rel = "" if tag == "default" else f"   {best / base - 1:+6.1%}"
                spread = abs(a - b) / max(a, b)
                print(f"    {tag:8s} {a:8.3f}ms {b:8.3f}ms {best:8.3f}ms{rel}   (pass spread {spread:.1%})",
                      flush=True)
            got = {t: min(ms[t]) for t in ms}
            if "scalar" in got and "vector" in got:
                win = min(got, key=got.get)
                gain = (base - got[win]) / base
                print(f"    --> best placement is {win.upper()}, {gain:+.1%} vs the compiler's own choice; "
                      f"SCALAR/VECTOR ratio {got['scalar'] / got['vector']:.3f}", flush=True)
                n = CALLS_PER_STEP.get(name, 0)
                per_step = (base - got[win]) * n
                if n == 0:
                    print("    --> NOT CALLED at the shipping launch (LAYER_IMPL=op inlines the forward MLP into "
                          "post_attn_fwd_h), so this number is a reference point, not a saving.", flush=True)
                elif abs(gain) < 0.01:
                    print(f"    --> under 1% on a LIVE kernel: the drains are not on this kernel's critical path. "
                          f"Close it.", flush=True)
                else:
                    frac = per_step / 1000.0 / STEP_SECONDS
                    print(f"    --> {n} calls/step -> {per_step:+.2f} ms/step ({frac:+.2%} of the step), worth about "
                          f"{-BPB_PER_EFOLD * frac:+.5f} bpb at -{BPB_PER_EFOLD} per e-fold of steps.", flush=True)

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
