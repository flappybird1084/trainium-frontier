#!/usr/bin/env python3
"""review_check.py - static review-safety checker for a competition submission.

Answers one question: "would a portal reviewer object to this file?"  It exists because
`prepare.py check_submission()` answers a different, much narrower one, and the two things that
actually got us rejected/broken are invisible to it:

  * a hard 262,144-byte limit on the submitted file (four of our versions silently exceeded it), and
  * drift that makes the SCORED FORWARD depend on something other than the input tokens and the
    checkpoint - an environment variable, or a value written on only one of the train/eval paths.

Usage:
    python experimental/review_check.py CANDIDATE.py [--ref LAST_ACCEPTED.py] [--verbose]
    python experimental/review_check.py --backtest            # every submission_v*.py, one line each

Severities: FAIL (a reviewer can point at it), WARN (defensible but asks to be explained),
INFO (recorded so it is not rediscovered; not a problem).
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

BYTE_LIMIT = 262_144

# The last file the portal ACCEPTED.  Diffing against this, not against the last file we built,
# is the whole lesson of the v78 rejection - both causes were 26 versions of invisible drift.
DEFAULT_REF = Path(__file__).resolve().parent.parent / "submission_raw" / "submission_v52_clean_compile_8way_both.py"

FORBIDDEN_IMPORT_ROOTS = {"socket", "subprocess", "requests", "urllib", "http", "ftplib"}
FORBIDDEN_NAMES = {"eval", "exec", "__import__", "compile"}
FORBIDDEN_CALLS = {("os", "system"), ("os", "popen"), ("os", "execv"), ("os", "spawnl")}

# Predicates that let a forward tell training from evaluation.  The keyword list is a backstop;
# the structural one-path-write rule below is the part meant to catch what the list misses.
MODE_ATTRS = {"training"}
# class -> predicates.  "training-flag", "tracing" and "process" are what the review notes deny
# outright; "grad-mode" is a weaker cousin that still makes train and eval take different branches.
MODE_CALL_CLASS = {
    "is_compiling": "tracing", "is_tracing": "tracing", "is_scripting": "tracing",
    "is_dynamo_compiling": "tracing", "currently_compiling": "tracing",
    "is_grad_enabled": "grad-mode", "is_inference_mode_enabled": "grad-mode", "is_inference": "grad-mode",
    "getpid": "process", "get_ipython": "process", "stack": None,
}
HARD_MODE_CLASSES = {"training-flag", "tracing", "process"}

# What the portal actually did with each version (findings/playbook.md, handoff-2026-09-06b.md,
# review-rejection-2026-09-28.md).  Everything else was built but never uploaded.
PORTAL = {
    7: "SCORED", 12: "REJECTED", 32: "REJECTED", 33: "SCORED", 52: "SCORED",
    73: "oversize", 74: "oversize", 75: "oversize", 76: "oversize", 78: "REJECTED",
    79: "candidate",
}

NUMWORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


# ---------------------------------------------------------------------------------------------
# findings

SEV_ORDER = {"FAIL": 0, "WARN": 1, "INFO": 2}


@dataclass
class Finding:
    sev: str
    check: str
    msg: str
    line: int = 0
    key: str = ""          # line-free identity, for matching against the accepted reference
    carried: bool = False  # the accepted reference has the same finding -> not new drift


@dataclass
class Report:
    path: Path
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, sev, check, msg, line=0, key=None):
        raw = key if key is not None else f"{check}:{msg}"
        # keys must be line-free: the same defect at a different line is still the same defect
        norm = "".join(c for c in raw if not c.isdigit()).replace("L@", "@")
        self.findings.append(Finding(sev, check, msg, line, norm))

    def count(self, sev, include_carried=False):
        return sum(1 for f in self.findings if f.sev == sev and (include_carried or not f.carried))

    @property
    def verdict(self):
        if self.count("FAIL"):
            return "REJECT-RISK"
        if self.count("WARN"):
            return "EXPLAIN"
        if self.count("WARN", include_carried=True) or self.count("FAIL", include_carried=True):
            return "CLEAN (= accepted ref)"
        return "CLEAN"


# ---------------------------------------------------------------------------------------------
# AST index


def qual_of(stack):
    return ".".join(stack)


class Index:
    """Functions, classes, call edges, attribute reads/writes, keyed by qualified name."""

    def __init__(self, tree: ast.AST):
        self.tree = tree
        self.funcs: dict[str, ast.AST] = {}          # qualname -> FunctionDef
        self.func_class: dict[str, str] = {}         # qualname -> enclosing class (or "")
        self.classes: dict[str, ast.ClassDef] = {}
        self.class_bases: dict[str, list[str]] = {}
        self.by_simple: dict[str, list[str]] = {}    # simple name -> [qualnames]
        self.module_assigns: dict[str, ast.AST] = {}  # module-level NAME -> value node
        self._walk(tree, [], "")

    def _walk(self, node, stack, cls):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = qual_of(stack + [child.name])
                self.funcs[q] = child
                self.func_class[q] = cls
                self.by_simple.setdefault(child.name, []).append(q)
                self._walk(child, stack + [child.name], cls)
            elif isinstance(child, ast.ClassDef):
                q = qual_of(stack + [child.name])
                self.classes[q] = child
                self.class_bases[q] = [_dotted(b) for b in child.bases]
                self._walk(child, stack + [child.name], q)
            else:
                if not stack and isinstance(child, (ast.Assign, ast.AnnAssign)):
                    tgts = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for t in tgts:
                        if isinstance(t, ast.Name):
                            self.module_assigns[t.id] = child.value
                self._walk(child, stack, cls)

    # -- call graph ---------------------------------------------------------------------------

    def callees(self, qual):
        """Over-approximate: resolve by simple name, plus class instantiation -> __init__."""
        node = self.funcs.get(qual)
        if node is None:
            return set()
        out = set()
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name is None:
                continue
            out.update(self.by_simple.get(name, []))
            for cq in self.classes:
                if cq.rsplit(".", 1)[-1] == name:
                    out.update(q for q in self.by_simple.get("__init__", []) if q.startswith(cq + "."))
        return out - {qual}

    def closure(self, seeds):
        seen, stack = set(), list(seeds)
        while stack:
            q = stack.pop()
            if q in seen or q not in self.funcs:
                continue
            seen.add(q)
            stack.extend(self.callees(q))
        return seen

    def is_module_class(self, cq):
        return any(b.endswith(("nn.Module", "Module", "autograd.Function", "Function")) for b in self.class_bases.get(cq, []))


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    return ""


def enclosing_map(idx: Index):
    """node id -> qualname of the innermost function containing it ('' = module level)."""
    out = {}

    def rec(node, qual):
        for child in ast.iter_child_nodes(node):
            q = qual
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = (qual + "." if qual else "") + child.name
                # nested-in-class needs the class qual; recompute from idx
                for cand in idx.by_simple.get(child.name, []):
                    if idx.funcs[cand] is child:
                        q = cand
                        break
            elif isinstance(child, ast.ClassDef):
                q = qual
                for cand, cnode in idx.classes.items():
                    if cnode is child:
                        q = ""
                        break
            out[id(child)] = q if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else q
            rec(child, q)

    rec(idx.tree, "")
    return out


# ---------------------------------------------------------------------------------------------
# scopes: what runs in the forward, on the eval path only, on the train path only


def build_scopes(idx: Index):
    forward_seeds = set()
    for q in idx.funcs:
        simple = q.rsplit(".", 1)[-1]
        cls = idx.func_class.get(q, "")
        if simple in ("forward", "head_logits", "fused_linear_cross_entropy", "backward", "setup_context") and cls:
            forward_seeds.add(q)
    F = idx.closure(forward_seeds)
    E = idx.closure([q for q in idx.funcs if q == "load_for_eval"])
    T = idx.closure([q for q in idx.funcs if q in ("main", "train")])
    inits = {q for q in idx.funcs if q.endswith(".__init__")}
    common = (E & T) | F | inits | {""}          # "" = module level
    eval_only = E - T - F - inits
    train_only = T - E - F - inits
    return F, E, T, common, eval_only, train_only


# ---------------------------------------------------------------------------------------------
# checks


def check_size(rep: Report, data: bytes):
    n = len(data)
    rep.stats["bytes"] = n
    margin = BYTE_LIMIT - n
    if n > BYTE_LIMIT:
        rep.add("FAIL", "size", f"{n} bytes, {-margin} OVER the {BYTE_LIMIT} portal limit")
    elif margin < 8192:
        rep.add("WARN", "size", f"{n} bytes, only {margin} under the {BYTE_LIMIT} limit")


def code_only_text(src: str) -> str:
    """Source with comments blanked, via tokenize (a regex on '#' corrupts string literals)."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return src
    return "\n".join(out)


def env_reads(idx: Index, enc):
    """[(var, line, enclosing_qual, [assigned names])]"""
    out = []
    for n in ast.walk(idx.tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("get", "getenv"):
            base = _dotted(n.func.value)
            if base not in ("os.environ", "environ", "os"):
                continue
            if n.func.attr == "get" and base == "os":
                continue
            if not n.args or not isinstance(n.args[0], ast.Constant) or not isinstance(n.args[0].value, str):
                continue
            out.append((n.args[0].value, n.lineno, enc.get(id(n), ""), n))
        elif isinstance(n, ast.Subscript) and _dotted(n.value) == "os.environ":
            if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                out.append((n.slice.value, n.lineno, enc.get(id(n), ""), n))
    return out


def assigned_names_for(tree, call_node):
    """Module/local names that receive the value of call_node (one hop through float()/int()/etc)."""
    names = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            val = n.value
            if val is None:
                continue
            if _contains(val, call_node):
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tgts:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
    return names


def _contains(root, target):
    for n in ast.walk(root):
        if n is target:
            return True
    return False


def names_used_in(idx: Index, quals, extra_class_bodies=True):
    used = set()
    for q in quals:
        node = idx.funcs.get(q)
        if node is None:
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                used.add(n.id)
    if extra_class_bodies:
        for cq, cnode in idx.classes.items():
            for n in ast.walk(cnode):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    used.add(n.id)
    return used


def check_env(rep: Report, idx: Index, enc, F, ref_vars):
    reads = env_reads(idx, enc)
    varset = sorted({v for v, _, _, _ in reads})
    rep.stats["env_vars"] = varset
    rep.stats["env_count"] = len(varset)

    # names that can reach the scored forward / the model's construction
    arch_quals = set(F) | {q for q in idx.funcs if q.rsplit(".", 1)[-1] in ("build_config", "init_weights", "_precompute_rotary_embeddings")}
    arch_names = names_used_in(idx, arch_quals, extra_class_bodies=True)

    classed = {}
    for var, line, where, node in reads:
        tainted = assigned_names_for(idx.tree, node)
        reaches_forward = bool(set(tainted) & arch_names) or where in F
        cls = "FORWARD" if reaches_forward else "ops/train"
        classed.setdefault(var, (cls, line, where, tainted))
        if classed[var][0] != "FORWARD" and cls == "FORWARD":
            classed[var] = (cls, line, where, tainted)

    def short(v):
        return v.replace("NEURON_COMPETITION_R1_", "")

    arch_vars = sorted(v for v in classed if classed[v][0] == "FORWARD")
    ops_new = sorted(v for v in classed if classed[v][0] != "FORWARD"
                     and ref_vars is not None and v not in ref_vars)
    rep.stats["env_arch"] = arch_vars
    rep.stats["env_new"] = sorted(v for v in varset if ref_vars is not None and v not in ref_vars)

    if arch_vars:
        rep.add("FAIL", "env",
                f"{len(arch_vars)} environment variable(s) reach the SCORED MODEL (architecture/numerics), so the "
                f"scored architecture can be changed without changing the file:\n        "
                + ", ".join(short(v) for v in arch_vars),
                classed[arch_vars[0]][1], key="env:arch")
    if ops_new:
        rep.add("WARN", "env",
                f"{len(ops_new)} operational env var(s) not in the accepted reference (each one is a knob a "
                f"reviewer can ask about, and it moves any 'reads exactly N' claim):\n        "
                + ", ".join(short(v) for v in ops_new),
                classed[ops_new[0]][1], key="env:new")
    if ref_vars is not None:
        gone = sorted(set(ref_vars) - set(varset))
        if gone:
            rep.add("INFO", "env", f"the reference reads {len(gone)} var(s) this file does not: "
                                   + ", ".join(short(v) for v in gone))


def check_forbidden(rep: Report, idx: Index):
    for n in ast.walk(idx.tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS:
                    rep.add("FAIL", "forbidden", f"import {a.name} (L{n.lineno})", n.lineno)
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] in FORBIDDEN_IMPORT_ROOTS:
                rep.add("FAIL", "forbidden", f"from {n.module} import ... (L{n.lineno})", n.lineno)
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_NAMES:
                rep.add("FAIL", "forbidden", f"{f.id}() (L{n.lineno})", n.lineno)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in FORBIDDEN_CALLS:
                rep.add("FAIL", "forbidden", f"{f.value.id}.{f.attr}() (L{n.lineno})", n.lineno)


def check_harness(rep: Report, idx: Index, src: str):
    prepare_names = set()
    for n in ast.walk(idx.tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "") == "prepare":
            prepare_names.update(a.asname or a.name for a in n.names)
    # nodes inside an `except ImportError:` shim are compatibility fallbacks, not substitutions
    in_shim = set()
    for n in ast.walk(idx.tree):
        if isinstance(n, ast.ExceptHandler) and "ImportError" in (_dotted(n.type) if n.type else ""):
            for d in ast.walk(n):
                in_shim.add(id(d))

    # the v12 rejection message: "unsafe checkpoint loading ... with a fallback mechanism"
    loads = [n for n in ast.walk(idx.tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "load"
             and _dotted(n.func.value) == "torch"]
    for n in loads:
        if not any(k.arg == "weights_only" and getattr(k.value, "value", None) is True for k in n.keywords):
            rep.add("FAIL", "harness", f"torch.load without weights_only=True (L{n.lineno})", n.lineno)
    if len(loads) > 1:
        rep.add("WARN", "harness",
                f"{len(loads)} torch.load call sites (L{', L'.join(str(n.lineno) for n in loads)}) - a second one "
                f"reads as a deserialization fallback, which is literally the v12 rejection text")
    # monkey-patching prepare
    for n in ast.walk(idx.tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Attribute) and _dotted(t.value) in ("prepare",):
                    rep.add("FAIL", "harness", f"assigns prepare.{t.attr} (L{n.lineno})", n.lineno)
                if isinstance(t, ast.Name) and t.id in prepare_names:
                    shim = id(n) in in_shim
                    rep.add("WARN" if shim else "FAIL", "harness",
                            f"rebinds the imported harness symbol {t.id} (L{n.lineno})"
                            + (" - inside an ImportError shim, so it only bites on an old prepare.py" if shim else ""),
                            n.lineno)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr":
            if n.args and _dotted(n.args[0]) == "prepare":
                rep.add("FAIL", "harness", f"setattr on prepare (L{n.lineno})", n.lineno)
    # reimplementing the scorer
    for q, node in idx.funcs.items():
        simple = q.rsplit(".", 1)[-1]
        if simple in ("evaluate_bpb", "make_dataloader", "causality_check", "record_step_boundary") and "." not in q:
            shim = id(node) in in_shim
            rep.add("WARN" if shim else "FAIL", "harness",
                    f"defines its own {simple}() (L{node.lineno})"
                    + (" - inside an ImportError shim" if shim else " - a reimplementation of the trusted harness"),
                    node.lineno)
    # record_step_boundary must be called
    called = any(
        isinstance(n, ast.Call)
        and ((isinstance(n.func, ast.Name) and n.func.id == "record_step_boundary")
             or (isinstance(n.func, ast.Attribute) and n.func.attr == "record_step_boundary"))
        for n in ast.walk(idx.tree)
    )
    if not called:
        rep.add("FAIL", "harness", "never calls record_step_boundary() - the run is unscorable")
    # unknown-flag death: the runner may pass --compile-only
    if "--compile-only" not in code_only_text(src):
        rep.add("WARN", "harness", "no --compile-only flag; an unknown flag makes argparse exit(2) at launch")
    # eval hook
    if "load_for_eval" not in idx.funcs:
        rep.add("FAIL", "harness", "no load_for_eval() - the scorer has no entry point")


def check_mode_predicates(rep: Report, idx: Index, enc, F):
    hits = []  # (class, text, line, where)
    for n in ast.walk(idx.tree):
        where = enc.get(id(n), "")
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and n.attr in MODE_ATTRS:
            if _dotted(n.value) in ("self", "model", "module", "m", "block", "mod"):
                hits.append(("training-flag", f"{_dotted(n.value)}.{n.attr}", n.lineno, where))
        elif isinstance(n, ast.Call):
            f = n.func
            attr = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            cls = MODE_CALL_CLASS.get(attr)
            if cls:
                hits.append((cls, f"{_dotted(f)}()", n.lineno, where))
        elif isinstance(n, ast.Compare) and where != "":
            txt = _dotted(n.left)
            if txt.endswith("__name__") or txt.endswith("argv"):
                hits.append(("process", f"{txt} == ...", n.lineno, where))

    rep.stats["mode_hits"] = {}
    for cls, what, line, where in hits:
        in_fwd = where in F
        rep.stats["mode_hits"].setdefault(cls, []).append((what, line, where, in_fwd))
        if in_fwd and cls in HARD_MODE_CLASSES:
            sev = "FAIL"
        elif in_fwd:
            sev = "WARN"
        else:
            sev = "INFO"
        rep.add(sev, "mode",
                f"[{cls}] {what} at L{line} in {where or '<module>'}"
                f"{' - REACHABLE FROM THE SCORED FORWARD' if in_fwd else ' (outside the scored forward)'}",
                line, key=f"mode:{cls}:{what}:{where}")


def attr_writes(idx: Index, enc):
    """attr name -> {qualname of writer}"""
    w: dict[str, set] = {}
    for n in ast.walk(idx.tree):
        tgts = []
        if isinstance(n, ast.Assign):
            tgts = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            tgts = [n.target]
        for t in tgts:
            if isinstance(t, ast.Attribute):
                w.setdefault(t.attr, set()).add(enc.get(id(n), ""))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr":
            if len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
                w.setdefault(n.args[1].value, set()).add(enc.get(id(n), ""))
    return w


def global_writes(idx: Index, enc):
    g: dict[str, set] = {}
    for name in idx.module_assigns:
        g.setdefault(name, set()).add("")
    for q, node in idx.funcs.items():
        declared = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Global):
                declared.update(n.names)
        if not declared:
            continue
        for n in ast.walk(node):
            tgts = n.targets if isinstance(n, ast.Assign) else ([n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else [])
            for t in tgts:
                if isinstance(t, ast.Name) and t.id in declared:
                    g.setdefault(t.id, set()).add(enc.get(id(n), q))
    return g


def forward_attr_reads(idx: Index, F):
    """attr name -> [(line, qual)] for attribute loads inside forward-reachable code."""
    reads: dict[str, list] = {}
    for q in F:
        node = idx.funcs.get(q)
        if node is None:
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
                reads.setdefault(n.attr, []).append((n.lineno, q))
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr":
                if len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
                    reads.setdefault(n.args[1].value, []).append((n.lineno, q))
                    if len(n.args) >= 3:
                        reads.setdefault("__getattr_default__" + n.args[1].value, []).append((n.lineno, q))
    return reads


def forward_global_reads(idx: Index, F):
    reads: dict[str, list] = {}
    for q in F:
        node = idx.funcs.get(q)
        if node is None:
            continue
        local = {a.arg for a in getattr(node, "args", ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], defaults=[], kw_defaults=[])).args} if hasattr(node, "args") else set()
        assigned = {t.id for n in ast.walk(node) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id in local or n.id in assigned:
                    continue
                reads.setdefault(n.id, []).append((n.lineno, q))
    return reads


def check_one_path_writes(rep: Report, idx: Index, enc, F, common, eval_only, train_only):
    """THE general rule: a value read in the scored forward that is written on only one path."""
    aw = attr_writes(idx, enc)
    gw = global_writes(idx, enc)
    ar = forward_attr_reads(idx, F)
    gr = forward_global_reads(idx, F)

    def classify(writers):
        return (writers & common, writers & eval_only, writers & train_only)

    def emit(kind, name, sites, writers):
        w_c, w_e, w_t = classify(writers)
        line = sites[0][0]
        loc = f"L{line} in {sites[0][1]}"
        if not writers:
            return  # nothing in this file assigns it; handled by the getattr-default rule
        if not w_c and w_e and not w_t:
            rep.add("FAIL", "one-path",
                    f"{kind} `{name}` is read by the scored forward ({loc}) but written ONLY on the eval path "
                    f"({', '.join(sorted(w_e))}) - training and evaluation compute different values", line)
        elif not w_c and w_t and not w_e:
            rep.add("FAIL", "one-path",
                    f"{kind} `{name}` is read by the scored forward ({loc}) but written ONLY on the training path "
                    f"({', '.join(sorted(w_t))}) - evaluation reads an unset value", line)
        elif w_e and not w_t:
            rep.add("WARN", "one-path",
                    f"{kind} `{name}` is read by the scored forward ({loc}) and re-written on the eval path "
                    f"({', '.join(sorted(w_e))}) with no matching training write", line)
        elif w_t and not w_e:
            rep.add("INFO", "one-path",
                    f"{kind} `{name}` has a common default plus a training-path write "
                    f"({', '.join(sorted(w_t))}); disclose it in the notes", line)

    for name, sites in ar.items():
        if name.startswith("__getattr_default__"):
            real = name[len("__getattr_default__"):]
            if real not in aw:
                line = sites[0][0]
                rep.add("FAIL", "one-path",
                        f"attribute `{real}` is read with a getattr default in the scored forward "
                        f"(L{line} in {sites[0][1]}) and is assigned NOWHERE in the file - it can only "
                        f"be injected from outside, which is the v78 shape exactly", line)
            continue
        if name in aw:
            emit("attribute", name, sites, aw[name])

    for name, sites in gr.items():
        if name in gw and len(gw[name]) > 1:
            emit("global", name, sites, gw[name])


def check_eval_hook(rep: Report, idx: Index, enc):
    node = idx.funcs.get("load_for_eval")
    if node is None:
        return
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Attribute) and t.attr in ("forward", "head_logits", "__class__"):
                    rep.add("FAIL", "eval-hook", f"load_for_eval replaces .{t.attr} (L{n.lineno})", n.lineno)
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in ("train",) and _dotted(f.value) in ("model", "m"):
                rep.add("WARN", "eval-hook", f"load_for_eval calls .train() (L{n.lineno})", n.lineno)


# ---------------------------------------------------------------------------------------------
# "Notes for review" claim verification


def notes_bullets(tree):
    doc = ast.get_docstring(tree) or ""
    if "Notes for review" not in doc:
        return []
    tail = doc.split("Notes for review", 1)[1]
    bullets, cur = [], None
    for raw in tail.splitlines():
        line = raw.strip()
        if line.startswith("* "):
            if cur:
                bullets.append(" ".join(cur))
            cur = [line[2:]]
        elif cur is not None:
            if not line:
                bullets.append(" ".join(cur))
                cur = None
            else:
                cur.append(line)
    if cur:
        bullets.append(" ".join(cur))
    return bullets


def count_calls(idx: Index, name):
    n = 0
    for node in ast.walk(idx.tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == name) or (isinstance(f, ast.Attribute) and f.attr == name):
                n += 1
    return n


def _references(idx: Index, name):
    """Uses of `name` outside its own definition: called, passed to wrap_nki(), or torch.ops.<ns>.name(...)."""
    defs = {id(idx.funcs[q]) for q in idx.funcs if q.rsplit(".", 1)[-1] == name}
    n = 0
    for node in ast.walk(idx.tree):
        if id(node) in defs:
            continue
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            n += 1
        elif isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.ctx, ast.Load):
            n += 1
    return n


def module_const(idx: Index, name):
    v = idx.module_assigns.get(name)
    if v is None:
        return None
    try:
        return ast.literal_eval(v)
    except Exception:
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("float", "int") and v.args:
            try:
                return ast.literal_eval(v.args[0])
            except Exception:
                return None
        return None


def number_in(text, *after):
    """First integer (digit or word) appearing in `text` after any of the marker phrases."""
    low = text.lower()
    for marker in after:
        i = low.find(marker)
        if i < 0:
            continue
        seg = low[i:i + 80]
        for w in seg.replace(",", " ").replace("`", " ").split():
            if w.isdigit():
                return int(w)
            if w in NUMWORDS:
                return NUMWORDS[w]
    return None


def check_notes(rep: Report, idx: Index, src: str, F):
    bullets = notes_bullets(idx.tree)
    rep.stats["claims"] = len(bullets)
    if not bullets:
        rep.add("INFO", "notes", "no 'Notes for review' block - nothing to contradict, and nothing to argue the case")
        return

    n_fail_before = rep.count("FAIL")
    results = []

    for b in bullets:
        low = b.lower()
        verdicts = []

        # -- one definition of the logits ----------------------------------------------------
        if "head_logits" in b and ("single definition" in low or "returns exactly that" in low):
            defs = [q for q in idx.funcs if q.rsplit(".", 1)[-1] == "head_logits"]
            if len(defs) != 1:
                verdicts.append(("FAIL", f"{len(defs)} definitions of head_logits, claim says one"))
            else:
                verdicts.append(("OK", "one head_logits definition"))
            fwd = idx.funcs.get("GPT.forward")
            if fwd is not None:
                ret_hl = any(
                    isinstance(n, ast.Return) and n.value is not None
                    and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                            and c.func.attr == "head_logits" for c in ast.walk(n.value))
                    for n in ast.walk(fwd)
                )
                verdicts.append(("OK", "forward returns head_logits") if ret_hl
                                else ("FAIL", "GPT.forward never returns head_logits(...)"))

        if "call site" in low:
            claimed = number_in(b, "there are", "call site")
            actual = count_calls(idx, "head_logits")
            if claimed is not None:
                verdicts.append(("OK", f"head_logits call sites = {actual}") if actual == claimed
                                else ("FAIL", f"claims {claimed} head_logits call sites, file has {actual}"))

        if "no scale" in low or "temperature" in low or "no offset" in low:
            hl = idx.funcs.get([q for q in idx.funcs if q.rsplit(".", 1)[-1] == "head_logits"][0]) \
                if any(q.rsplit(".", 1)[-1] == "head_logits" for q in idx.funcs) else None
            bad = []
            if hl is not None:
                for n in ast.walk(hl):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr":
                        bad.append(n.lineno)
            verdicts.append(("FAIL", f"head_logits uses getattr at L{bad[0]} - an optional, one-path value") if bad
                            else ("OK", "head_logits reads no optional attribute"))

        # -- fused loss == CE(head_logits) ---------------------------------------------------
        if "cross_entropy(head_logits" in b or "check_fused_loss" in b:
            for fn in ("check_fused_loss", "check_fused_head"):
                if fn in b:
                    if fn not in idx.funcs:
                        verdicts.append(("FAIL", f"{fn}() is claimed but not defined"))
                    elif count_calls(idx, fn) < 1:
                        verdicts.append(("FAIL", f"{fn}() is defined but never called"))
                    else:
                        verdicts.append(("OK", f"{fn}() defined and called"))
            if "switches training onto" in low or "disables the fused path" in low:
                v = _check_fail_direction(idx)
                verdicts.append(v)

        # -- no mode / env inspection --------------------------------------------------------
        if "no code path inspects" in low or "training/eval mode" in low:
            hits = rep.stats.get("mode_hits", {})
            hard = [(c, w, ln, wh) for c, lst in hits.items() if c in HARD_MODE_CLASSES
                    for w, ln, wh, inf in lst if inf]
            soft = [(c, w, ln, wh) for c, lst in hits.items() if c not in HARD_MODE_CLASSES
                    for w, ln, wh, inf in lst if inf]
            if hard:
                verdicts.append(("FAIL", f"{len(hard)} mode/tracing predicate(s) in the scored forward: "
                                         + ", ".join(f"{w}@L{ln}" for _, w, ln, _ in hard[:3])))
            else:
                verdicts.append(("OK", "no training-flag/tracing/process predicate in the scored forward"))
            if soft:
                verdicts.append(("SOFT", f"the claim denies inspecting training/eval mode; the forward still calls "
                                         + ", ".join(f"{w}@L{ln} in {wh}" for _, w, ln, wh in soft[:3])
                                         + " (grad-mode, not mode detection - but it is the nearest thing to a contradiction)"))
        if "environment variable" in low:
            claimed = number_in(b, "reads exactly", "reads ")
            actual = rep.stats.get("env_count", 0)
            if claimed is not None:
                verdicts.append(("OK", f"env var count = {actual}") if actual == claimed
                                else ("FAIL", f"claims {claimed} environment variables, file reads {actual}"))
            if rep.stats.get("env_arch"):
                verdicts.append(("FAIL", f"{len(rep.stats['env_arch'])} env var(s) reach the scored model; "
                                         f"the claim says all are operational"))
        if "is a literal" in low and "constant" in low:
            env_backed = set()
            for var, line, where, node in env_reads(idx, enclosing_map(idx)):
                env_backed.update(assigned_names_for(idx.tree, node))
            arch = names_used_in(idx, F, extra_class_bodies=True)
            leaked = sorted(env_backed & arch)
            verdicts.append(("OK", "every constant in the forward is a literal") if not leaked
                            else ("FAIL", f"env-backed constants used in the forward: {', '.join(leaked)}"))

        # -- inert / zero-weight terms -------------------------------------------------------
        if "fixed at zero" in low or "weights fixed at zero" in low:
            names = _idents_in_claim(b, idx)
            bad = [f"{n}={module_const(idx, n)!r}" for n in names if module_const(idx, n) not in (0, 0.0)]
            verdicts.append(("OK", f"{len(names)} named term(s) are 0") if not bad
                            else ("FAIL", f"claimed-zero constants that are NOT zero: {', '.join(bad)}"))
            live, ops = _guard_audit(idx, names)
            if live:
                verdicts.append(("FAIL", "a guard is TRUE at the declared value, so the 'inert' code runs: "
                                         + ", ".join(live)))
            claimed_op = "> 0" if "> 0" in b or "`> 0`" in b else None
            odd = sorted({o for o in ops if claimed_op and o != claimed_op})
            if odd:
                verdicts.append(("SOFT", f"the claim says the guard is `{claimed_op}`; the file actually uses "
                                         + ", ".join(f"`{o}`" for o in odd)
                                         + " at some sites (behaviour is still inert, the sentence is not accurate)"))

        # -- live kernel chain ---------------------------------------------------------------
        if "live forward is" in low or "live backward is" in low:
            named = [w.strip("`,.") for w in b.replace("->", " ").split() if w.strip("`,.").islower() and "_" in w.strip("`,.")]
            named = [n for n in dict.fromkeys(named) if any(q.rsplit(".", 1)[-1] == n for q in idx.funcs)]
            # NKI kernels are handed to wrap_nki()/torch.ops, not called by name - count REFERENCES.
            missing = [n for n in named if _references(idx, n) == 0]
            verdicts.append(("OK", f"{len(named)} named live kernel(s) are wired up") if not missing
                            else ("FAIL", f"named as live but referenced nowhere: {', '.join(missing)}"))

        # -- checkpoint / EMA ----------------------------------------------------------------
        if "lerp_" in b:
            verdicts.append(("OK", "lerp_ present") if "lerp_" in code_only_text(src)
                            else ("FAIL", "the claimed `lerp_` weight average is not in the code"))

        if not verdicts:
            verdicts.append(("?", "no mechanical check for this claim"))
        results.append((b, verdicts))

    for b, verdicts in results:
        short = (b[:96] + "...") if len(b) > 99 else b
        for sev, why in verdicts:
            if sev == "FAIL":
                rep.add("FAIL", "notes", f"CLAIM BROKEN: {why}\n        claim: \"{short}\"", key=f"notes:{why}")
            elif sev == "SOFT":
                rep.add("WARN", "notes", f"claim is strained: {why}\n        claim: \"{short}\"", key=f"notes:{why}")
            elif sev == "?":
                rep.add("INFO", "notes", f"unchecked claim (read it yourself): \"{short}\"")
    ok = sum(1 for _, vs in results for s, _ in vs if s == "OK")
    rep.stats["claims_ok"] = ok
    rep.stats["claims_broken"] = rep.count("FAIL", include_carried=True) - n_fail_before


def _idents_in_claim(text, idx: Index):
    """Identifiers named in a claim that exist as module-level constants (handles CANON_*_W globs)."""
    out = []
    for raw in text.replace(",", " ").replace("`", " ").split():
        tok = raw.strip(".;:()")
        if not tok or not tok[0].isupper():
            continue
        if "*" in tok:
            pre, _, suf = tok.partition("*")
            out.extend(k for k in idx.module_assigns if k.startswith(pre) and k.endswith(suf))
        elif tok in idx.module_assigns:
            out.append(tok)
    return list(dict.fromkeys(out))


_CMP_TXT = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}
_CMP_FN = {ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b, ast.Lt: lambda a, b: a < b,
           ast.LtE: lambda a, b: a <= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


def _guard_audit(idx: Index, names):
    """For constants a claim says are inert: does any guard evaluate TRUE at the declared value,
    and what operator/threshold do the guards actually use?"""
    live, ops = [], set()
    for name in names:
        val = module_const(idx, name)
        for n in ast.walk(idx.tree):
            if not (isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == name):
                continue
            for op, comp in zip(n.ops, n.comparators):
                if type(op) not in _CMP_TXT or not isinstance(comp, ast.Constant):
                    continue
                thr = comp.value
                if isinstance(thr, float) and thr == int(thr):
                    thr = int(thr)          # `> 0.0` and `> 0` are the same sentence
                ops.add(f"{_CMP_TXT[type(op)]} {thr}")
                if val is not None:
                    try:
                        if _CMP_FN[type(op)](val, comp.value):
                            live.append(f"{name} {_CMP_TXT[type(op)]} {comp.value} at L{n.lineno} (value {val!r})")
                    except TypeError:
                        pass
    return live, ops


def _check_fail_direction(idx: Index):
    """The v78 defect: a try/finally that re-enables the path under test whatever happened."""
    for fn in ("check_fused_loss", "check_fused_head"):
        node = idx.funcs.get(fn)
        if node is None:
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Try) and n.finalbody:
                for f in n.finalbody:
                    for a in ast.walk(f):
                        if isinstance(a, ast.Assign) and isinstance(a.value, ast.Constant) and a.value.value is True:
                            for t in a.targets:
                                if isinstance(t, ast.Attribute) and "ok" in t.attr.lower():
                                    return ("FAIL", f"{fn} sets {t.attr}=True in a finally: any exception RE-ENABLES "
                                                    f"the path under test (L{a.lineno})")
        # every handler must be able to disable
        handlers = [n for n in ast.walk(node) if isinstance(n, ast.ExceptHandler)]
        for h in handlers:
            sets_true = [a for a in ast.walk(h) if isinstance(a, ast.Assign)
                         and isinstance(a.value, ast.Constant) and a.value.value is True
                         and any(isinstance(t, ast.Attribute) and "ok" in t.attr.lower() for t in a.targets)]
            if sets_true:
                return ("WARN", f"{fn} sets the ok-flag True inside an except handler (L{sets_true[0].lineno}) - "
                                f"check the fail direction by hand")
    return ("OK", "the fused-path check fails safe")


# ---------------------------------------------------------------------------------------------
# driver


def _analyze_one(path: Path, ref_vars):
    data = path.read_bytes()
    src = data.decode("utf-8")
    tree = ast.parse(src, filename=str(path))
    idx = Index(tree)
    enc = enclosing_map(idx)
    F, E, T, common, eval_only, train_only = build_scopes(idx)

    rep = Report(path)
    rep.stats["forward_funcs"] = len(F)
    rep.stats["eval_only_funcs"] = len(eval_only)
    rep.stats["train_only_funcs"] = len(train_only)

    check_size(rep, data)
    check_env(rep, idx, enc, F, ref_vars)
    check_forbidden(rep, idx)
    check_harness(rep, idx, src)
    check_mode_predicates(rep, idx, enc, F)
    check_one_path_writes(rep, idx, enc, F, common, eval_only, train_only)
    check_eval_hook(rep, idx, enc)
    check_notes(rep, idx, src, F)
    return rep, idx, enc


_REF_CACHE: dict = {}


def analyze(path: Path, ref: Path | None):
    ref_vars = None
    ref_rep = None
    if ref is not None and ref.exists():
        key = str(ref.resolve())
        if key not in _REF_CACHE:
            ridx = Index(ast.parse(ref.read_text(encoding="utf-8"), filename=str(ref)))
            rvars = {v for v, _, _, _ in env_reads(ridx, enclosing_map(ridx))}
            _REF_CACHE[key] = (rvars, _analyze_one(ref, None)[0])
        ref_vars, ref_rep = _REF_CACHE[key]

    rep, _, _ = _analyze_one(path, ref_vars)
    if ref_vars is not None:
        rep.stats["ref"] = ref.name
        rep.stats["ref_env"] = sorted(ref_vars)
    if ref_rep is not None:
        # Anything the ACCEPTED file also has is not new drift.  This is the whole lesson of the
        # v78 rejection: diff against the last file the portal accepted, not the last one we built.
        ref_keys = {f.key for f in ref_rep.findings if f.sev in ("FAIL", "WARN")}
        carried = 0
        for f in rep.findings:
            if f.sev in ("FAIL", "WARN") and f.key in ref_keys:
                f.carried = True
                carried += 1
        rep.stats["carried"] = carried
    return rep


def print_report(rep: Report, verbose=False):
    s = rep.stats
    carried_note = f" / {s['carried']} carried from {s.get('ref', 'ref')}" if s.get("carried") else ""
    print(f"=== {rep.path.name}  ->  {rep.verdict}"
          f"   [{rep.count('FAIL')} FAIL / {rep.count('WARN')} WARN{carried_note}]")
    margin = BYTE_LIMIT - s.get("bytes", 0)
    print(f"    {s.get('bytes',0):,} bytes ({margin:+,} vs the {BYTE_LIMIT:,} limit)"
          f" | {s.get('env_count',0)} env vars"
          f"{' (accepted ref: ' + str(len(s.get('ref_env', []))) + ')' if 'ref_env' in s else ''}"
          f", {len(s.get('env_arch', []))} reach the model"
          f" | {s.get('claims',0)} review claims, {s.get('claims_ok',0)} checks verified"
          f" | scored-forward fns {s.get('forward_funcs',0)},"
          f" eval-only {s.get('eval_only_funcs',0)}, train-only {s.get('train_only_funcs',0)}")
    order = sorted(rep.findings, key=lambda f: (f.carried, SEV_ORDER[f.sev], f.check, f.line))
    for f in order:
        if f.carried and not verbose:
            continue
        if f.sev == "INFO" and not verbose:
            continue
        print(f"    [{f.sev}{'/carried' if f.carried else ''}] {f.check}: {f.msg}")
    if not verbose:
        hidden = sum(1 for f in rep.findings if f.sev == "INFO" or f.carried)
        if hidden:
            print(f"    ({hidden} carried/INFO line(s) hidden; -v to see them)")


def main():
    ap = argparse.ArgumentParser(description="review-safety checker for a competition submission")
    ap.add_argument("candidate", nargs="?", help="the file about to be uploaded")
    ap.add_argument("--ref", help="the last file the portal ACCEPTED (diff env reads against it)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--backtest", action="store_true", help="one line per submission_v*.py")
    ap.add_argument("--labels", help="path=LABEL,... for the backtest table")
    args = ap.parse_args()

    if args.backtest:
        root = Path(__file__).resolve().parent.parent
        files = sorted(
            (root / "submission_raw").glob("submission_v*.py"),
            key=lambda p: int("".join(c for c in p.name.split("_")[1][1:] if c.isdigit())),
        )
        ref = root / "submission_raw" / "submission_v52_clean_compile_8way_both.py"
        print(f"{'file':<24} {'portal':<12} {'bytes':>8} {'env':>4} {'FAIL':>5} {'new':>4}  top finding")
        for f in files:
            vnum = int("".join(c for c in f.name.split("_")[1][1:] if c.isdigit()))
            label = PORTAL.get(vnum, "-")
            try:
                abs_rep = analyze(f, None)
                rep = analyze(f, ref)
            except SyntaxError as exc:
                print(f"{f.name.split('_')[1]:<24} {label:<12} {'?':>8} {'?':>4} {'ERR':>5} {'':>4}  {exc}")
                continue
            fails = [x for x in abs_rep.findings if x.sev == "FAIL"]
            top = fails[0].msg.splitlines()[0][:96] if fails else ""
            if not top:
                warns = [x for x in abs_rep.findings if x.sev == "WARN"]
                top = "(warn) " + warns[0].msg.splitlines()[0][:88] if warns else "clean"
            print(f"{f.name.split('_')[1]:<24} {label:<12} {abs_rep.stats['bytes']:>8,} "
                  f"{abs_rep.stats['env_count']:>4} {abs_rep.count('FAIL', include_carried=True):>5} "
                  f"{rep.count('FAIL'):>4}  {top}")
        print("\n  FAIL = absolute findings; new = findings the ACCEPTED v52 does not also have.")
        return 0

    if not args.candidate:
        ap.error("give a candidate file or --backtest")
    ref = Path(args.ref) if args.ref else DEFAULT_REF
    if not ref.exists():
        print(f"(no reference file at {ref}; env reads are not diffed)")
        ref = None
    rep = analyze(Path(args.candidate), ref)
    print_report(rep, args.verbose)
    return 1 if rep.count("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
