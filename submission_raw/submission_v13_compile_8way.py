print("SUBMISSION_BOOT: module import starting", flush=True)  # first executable line:
# discriminates "our code never ran" from "crashed later" in harness stdout logs

"""
train3_submission.py - v9-dev (2026-09-06): v7 + permanently stacked matrix weights and block scalars
(one Muon tensor and one all_reduce per weight shape; one AdamW tensor per scalar kind) to cut the
optimizer's per-step launch count on Trn2. Checkpoint layout changes (stacks.*, block_scalars.*).

v7 (2026-09-06): v6 + minimum-novelty portal build after sub-39b23de5 (v6, eager)
died with the same `no_run_meta`: packing off (PACK=1 = the baseline loader shape), per-worker rows capped
at 8 (single-process launch safety), for the 8-way LNC=1 launch that has scored twice for this team.

v6 (2026-09-06): v5 + portal hardening after sub-9ea276d1 died `no_run_meta`:
compile default OFF (eager, the organizers' known-good path); a compile failure at torch.compile() or at the
first step falls back to eager instead of crashing; the runner-contract import is guarded for an older
harness prepare.py. Validated on our Trn2: eager 8-way 1.1743 (T1), compile 1.1519 (T2), compile+LNC=2
1.1400 (T3/T4).

v5 (2026-09-05): v4 + first-real-Trn2 fixes: `x0 = x.clone()` (dynamo alias
guard under torch.compile backend=neuron) and tensor-valued per-step LR/momentum (a python-float scalar
bakes into the NEFF as a constant, so every distinct LR value forced a ~4 s recompile - the whole warmup
and the entire 60% warmdown ran ~6x slow on Trn2 S1d).

v4 (2026-09-03): v3 + the upstream step-boundary contract.
Upstream c907da4 changed scoring: the runner watches an append-only step log
(prepare.record_step_boundary, called after every synchronized optimizer step) and
excludes launch through the 5th boundary (cap 900 s) as the startup/compilation
allowance. check_submission now REJECTS files that never call record_step_boundary.
This port mirrors the new baseline exactly: budget clock restart at boundary 5 with
excess back-dating, predictive stop gated behind 5 completed steps riding the health
collective, --compile-only mode, full-budget MAX_TRAIN_SECONDS (was budget-60).
The old self-managed pre-clock warmup is superseded and removed.

v3 (2026-09-02): fixed-time champion, minimal-risk rebuild.
v2 portal failure (no_run_meta, no log) forced: lazy CUDA-only flex import (a broken
native import can kill a process past try/except), boot banners, baseline-shaped rest.

v1 post-mortem: the fixed-step 215M-param stack scored worse than baseline on the
portal (too few steps in the 30-min budget). v2 is the config that won the overnight
fixed-wall-time campaign on a 3090 (val_bpb 1.1427 per 720 s, T10): small and fast -
depth 10 x width 256, all-attention, batch 32768 with scaled LRs, organizer packing.

Functions like the shipped train.py (Neuron/CUDA/CPU devices, torchrun DDP, official
30-minute budget timer, load_for_eval contract) but with the architecture and schedule
found by the loss-per-step experiment loop (dev val_bpb 1.2657 @ 420x65k steps vs
1.3988 baseline-equivalent; see findings/train2-experiments.md):

  - LR warmup (steps) + linear warmdown over the last 60% of the TIME budget to 2%
  - rope base 200k, asymmetric learned logit softcap (init 11)
  - triple value-embedding tables (per-depth-third, gated per layer)
  - hashed (prev,cur) bigram table 2^17 gated into the input embedding
  - U-net skip connections (encoder half pushes, decoder half pops)
  - doc-masked attention + 4 learnable attention registers per layer
    (FlexAttention on CUDA; identical math via SDPA + dense mask elsewhere, incl. Neuron)

Launch (same shape as baseline; the portal saves the upload as train.py):
  NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py
"""

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Proxy for process launch, captured before the heavy imports below. Used to
# align the self-managed budget clock with the runner's startup allowance.
_PROCESS_STARTED = time.monotonic()

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    EVAL_TIMEOUT_SECONDS,
    TOKENIZER_VOCAB_SIZE,
    TRAIN_TIME_BUDGET_SECONDS,
    causality_check,
    ensure_tokenizer,
    evaluate_bpb,
    get_dist_info,
    make_dataloader,
    print0,
)
try:  # c907da4 runner contract; a harness on an older prepare.py lacks these -> pre-contract behavior
    from prepare import TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS, TRAIN_STARTUP_STEPS_EXCLUDED, record_step_boundary
except ImportError:  # pragma: no cover
    TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS = 900.0
    TRAIN_STARTUP_STEPS_EXCLUDED = 5

    def record_step_boundary(step: int) -> None:
        return None

print("SUBMISSION_BOOT: all imports done", flush=True)

# FlexAttention is a CUDA-only fast path. Import LAZILY and only when running on
# CUDA: a broken native import can kill the process in ways try/except cannot catch,
# so on Neuron/CPU this module is never touched at all.
_FLEX = None


def _get_flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention import flex_attention as _fa
        _FLEX = (_fa.create_block_mask, _fa.flex_attention)
    return _FLEX


# -----------------------------------------------------------------------------
# Hyperparameters (champion stack)

SEQ_LEN = 1024
# Dev-only sweep knobs (the portal sets none of these; every default == the champion). Trn2 campaign 2026-09-05.
_ENV = os.environ.get
DEPTH = int(_ENV("NEURON_COMPETITION_R1_DEPTH", "10"))
N_EMBD = int(_ENV("NEURON_COMPETITION_R1_N_EMBD", "512"))   # fixed-time optimum on 3090 (T5/T11); Trn2 re-tunes
N_HEAD = int(_ENV("NEURON_COMPETITION_R1_N_HEAD", "4"))
HEAD_DIM = N_EMBD // N_HEAD
MLP_RATIO = int(_ENV("NEURON_COMPETITION_R1_MLP_RATIO", "4"))

PACK = int(_ENV("NEURON_COMPETITION_R1_PACK", "1"))   # v7: 1 = the baseline loader shape (rows of SEQ_LEN).
                                # 4 (T10 packing trick, worth only -0.0013) is a post-v1 novelty; dropped for
                                # the portal until a run of ours scores there.
DEVICE_BATCH_SIZE = int(os.environ.get("NEURON_COMPETITION_R1_DEVICE_BATCH", "0")) or None
                                # post-reshape rows per micro-step (raw loader rows = this // PACK). None ->
                                # resolved in main() from world size so ONE file is optimal under both
                                # sanctioned launches: 8 workers (LNC=1) -> 4, 4 workers (LNC=2) -> 8; either way
                                # world tokens/micro = 32,768 = TOTAL_BATCH (grad_accum 1). Env overrides.
TOTAL_BATCH_SIZE = int(_ENV("NEURON_COMPETITION_R1_TOTAL_BATCH", "32768"))  # fixed-time optimum w/ sqrt-scaled LRs (T6b)
_BATCH_LR = (TOTAL_BATCH_SIZE / 32768) ** 0.5   # T6b sqrt-batch LR rule relative to the 32768 champion (1.0 by default)
NUM_STEPS = 100000              # never binds; the time budget does
# The official runner charges wall time only after the first five completed
# optimizer steps (the startup/compilation allowance, capped); the budget clock
# restarts at that boundary, so the full budget is usable.
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 5.0
# Covers the launch -> module-import gap this process cannot observe when
# checking its startup time against the allowance cap.
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

ROPE_BASE = 200_000
LOGIT_SOFTCAP = 11.0
BOS_ID = 8188                   # fixed tokenizer's <|reserved_0|>
NUM_SINKS = int(_ENV("NEURON_COMPETITION_R1_SINKS", "4"))   # attention registers per layer (128 = one flex/NKI block)
BIGRAM_SIZE = 2 ** 17
TRIGRAM_SIZE = 2 ** 17
QK_DIM = int(_ENV("NEURON_COMPETITION_R1_QK_DIM", "0"))              # dev knob; 0 = head_dim
ATTN_IMPL = _ENV("NEURON_COMPETITION_R1_ATTN", "explicit")           # dev knob: explicit | sdpa (bool mask) | nki (vendor flash kernels; needs SINK=logit)
MUON_IMPL = _ENV("NEURON_COMPETITION_R1_MUON", "fused")
DOCMASK = _ENV("NEURON_COMPETITION_R1_DOCMASK", "mask")               # mask: boolean doc mask | score: doc id encoded in 6 extra
                                                                       # q/k dims -> exact -1024*(doc distance)^2 penalty; the
                                                                       # attention mask becomes STATIC (causal only): the Neuron
                                                                       # backend can then lower flex_attention to an inlined NKI kernel
ROPE_IMPL = _ENV("NEURON_COMPETITION_R1_ROPE", "torch")               # dev knob: torch | nki (fused RoPE fwd/bwd kernel, eager Neuron only)
ADAMW_IMPL = _ENV("NEURON_COMPETITION_R1_ADAMW", "nki")             # dev knob: torch | nki (fused AdamW kernel for 2-D fp32 params)
NKI_ADAMW_INPLACE = _ENV("NEURON_COMPETITION_R1_ADAMW_INPLACE", "1") != "0"   # set once the device test shows in-place writes land
SINK_MODE = _ENV("NEURON_COMPETITION_R1_SINK", "kv")                   # kv: 4 learned k/v registers (champion) | logit: one learned
                                                                       # denominator-only sink logit per (layer, head) = what the NKI
                                                                       # flash kernels support (3090 E23: values were worth ~0.003)
PROFILE = _ENV("NEURON_COMPETITION_R1_PROFILE", "0") != "0"           # dev knob: per-phase step timing, steps 10-40
CE_IMPL = _ENV("NEURON_COMPETITION_R1_CE", "manual")                    # torch | manual (log_softmax+gather: no graph break on Neuron)
USE_TRIGRAM = _ENV("NEURON_COMPETITION_R1_TRIGRAM", "0") != "0"   # dev knob; L1 on the 4080: -0.0042 alone

# Dev knobs (Trn2 warm screens, 2026-09-06): multiplicative LR scales and schedule shape; defaults = the champion.
_F = lambda name, default: float(_ENV(name, default))
EMBEDDING_LR = 0.30 * 0.71 * _BATCH_LR * _F("NEURON_COMPETITION_R1_EMBED_LR_SCALE", "2.0")   # sqrt-batch LR comp. (T6b)
UNEMBEDDING_LR = 0.006 * 0.71 * _BATCH_LR * _F("NEURON_COMPETITION_R1_HEAD_LR_SCALE", "1.0")
MATRIX_LR = 0.025 * 0.71 * _BATCH_LR * _F("NEURON_COMPETITION_R1_MATRIX_LR_SCALE", "1.0")
SCALAR_LR = 0.20 * 0.71 * _BATCH_LR * _F("NEURON_COMPETITION_R1_SCALAR_LR_SCALE", "1.0")
WEIGHT_DECAY = _F("NEURON_COMPETITION_R1_WD", "0.10")
WARMUP_STEPS = int(_ENV("NEURON_COMPETITION_R1_WARMUP", "20"))
WARMDOWN_TIME_FRAC = _F("NEURON_COMPETITION_R1_WARMDOWN", "0.8")   # last 60% of the time budget decays linearly...
FINAL_LR_FRAC = _F("NEURON_COMPETITION_R1_FINAL_LR", "0.02")        # ...to 2% of peak
# Architecture backlog knobs (findings/arch-backlog-speedrun.md, 2026-09-06); every default = the champion (off).
WINDOW = int(_ENV("NEURON_COMPETITION_R1_WINDOW", "0"))                 # long-short attention: odd layers see only the last WINDOW keys (0 = off)
WINDOW_WARMUP = _F("NEURON_COMPETITION_R1_WINDOW_WARMUP", "0")           # fraction of the time budget over which ALL windows grow from 128 to their target
MTP_WEIGHT = _F("NEURON_COMPETITION_R1_MTP", "0")                        # multi-token prediction aux loss weight (t+2 head), 0 = off
VALUE_RESID = _ENV("NEURON_COMPETITION_R1_VALUE_RESID", "0") != "0"      # blend layer-0 V into every layer's V (learned per-layer lambda, zero-init)
BACKOUT = _ENV("NEURON_COMPETITION_R1_BACKOUT", "0") != "0"              # x_final -= lambda * x_mid (learned, zero-init)
MERGED_QKV = _ENV("NEURON_COMPETITION_R1_MERGED_QKV", "0") != "0"        # one (3*n_embd, n_embd) projection instead of q/k/v
PLE = _ENV("NEURON_COMPETITION_R1_PLE", "0") != "0"                      # per-layer token embeddings added to each block input               # dev knob: torch | nki (fused Polar Express kernel, Trn2)

FLEX_KERNEL_OPTS = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 32,
                    "BLOCK_M2": 32, "BLOCK_N2": 32, "num_stages": 1}


# -----------------------------------------------------------------------------
# Device / distributed setup (as in the shipped train.py)

def neuron_available() -> bool:
    try:
        torch.device("neuron")
        return True
    except Exception:
        return False


def detect_device_type(requested: str = "") -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if neuron_available():
        return "neuron"
    return "cpu"


def init_runtime(device_type: str):
    assert device_type in {"cpu", "cuda", "neuron"}
    torch.manual_seed(42)
    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    elif ddp and device_type == "neuron":
        dist.init_process_group("neuron")
        device = torch.device("neuron")
    elif ddp:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return ddp, rank, local_rank, world_size, device


def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "neuron" and hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def compute_dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "neuron":
        return torch.bfloat16
    return torch.float32


COMPUTE_DTYPE: torch.dtype = torch.bfloat16


# -----------------------------------------------------------------------------
# Model

@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    use_trigram: bool = False   # saved in the checkpoint so load_for_eval rebuilds the same model
    qk_dim: int = 0             # 0 -> head_dim (champion). L2 on the 4080: qk 64 / v 128 was -0.0075 via +27% steps
    sink_mode: str = "kv"       # "kv" (registers) | "logit" (per-head sink logit; NKI-compatible). Saved in the checkpoint.
    window: int = 0             # long-short attention window for odd layers (0 = all global)
    mtp: bool = False           # multi-token prediction aux head (training only; eval uses the main head)
    value_resid: bool = False
    backout: bool = False
    merged_qkv: bool = False
    ple: bool = False
    docmask: str = "mask"       # "mask" | "score" (see DOCMASK)


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (bf16 input, f32 weight)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


class StackedLinear(nn.Module):
    """A bias-free Linear whose weight is slice `idx` of a (n, out, in) Parameter shared by every layer of
    the same shape. The stack is registered once on GPT (`stacks`), so Muon updates one tensor per shape
    group and the gradient all_reduce is one collective per group: on Trn2 the eager optimizer's cost is
    launch count (~180 stacking copies + ~120 per-parameter collectives per step before this)."""

    def __init__(self, registry, out_features: int, in_features: int):
        super().__init__()
        self.key, self.idx = registry.register(out_features, in_features)
        object.__setattr__(self, "_stack", None)   # bound later; NOT a registered parameter of this module

    def bind(self, stack: nn.Parameter) -> None:
        object.__setattr__(self, "_stack", stack)

    @property
    def weight(self) -> torch.Tensor:
        return self._stack[self.idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if STACK_GRAD_MODE == "function" and torch.is_grad_enabled() and self._stack.requires_grad:
            return _StackSliceLinear.apply(x, self._stack, self.idx)
        return F.linear(x, self._stack[self.idx].to(dtype=x.dtype), None)


STACK_GRAD_MODE = "function"   # eager: accumulate weight grads straight into stack.grad[idx]; compile: "view"


class _StackSliceLinear(torch.autograd.Function):
    """Eager-mode linear on slice `idx` of a stacked weight whose backward writes the weight gradient directly
    into `stack.grad[idx]`. Plain autograd through `stack[idx]` allocates a full (n, out, in) zero tensor per
    layer per step (select backward), which compile fuses away but eager pays for (Trn2 PR4: +0.08 s/step)."""

    @staticmethod
    def forward(ctx, x, stack, idx):
        w = stack[idx].to(dtype=x.dtype)
        ctx.save_for_backward(x, w)
        ctx.stack, ctx.idx = stack, idx
        return F.linear(x, w, None)

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        grad_x = grad_out @ w
        gw = grad_out.reshape(-1, grad_out.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
        stack = ctx.stack
        if stack.grad is None:
            stack.grad = torch.zeros_like(stack)
        stack.grad[ctx.idx].add_(gw.to(dtype=stack.grad.dtype))
        return grad_x, None, None


class StackRegistry:
    """Assigns (shape key, slice index) to StackedLinears in construction order; builds the stacks."""

    def __init__(self):
        self.counts: dict[tuple[int, int], int] = {}
        self.members: list[StackedLinear] = []

    def register(self, out_features: int, in_features: int):
        shape = (out_features, in_features)
        idx = self.counts.get(shape, 0)
        self.counts[shape] = idx + 1
        return f"w{out_features}x{in_features}", idx

    def build(self) -> nn.ParameterDict:
        return nn.ParameterDict({f"w{o}x{i}": nn.Parameter(torch.empty(n, o, i)) for (o, i), n in self.counts.items()})


_NKI_ROPE = None


def build_nki_rope():
    """Fused split-half RoPE (dev copy: nki_rope.py): one launch per q/k instead of ~9 eager ops; backward = same rotation
    with -sin. Eager Neuron only (a kernel inside a compiled graph partitions the NEFF, S_NKI3 2026-09-06)."""
    global _NKI_ROPE
    if _NKI_ROPE is not None:
        return _NKI_ROPE
    global nki, nisa, nl
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    @nki.jit
    def rope_kernel(x_in: nl.ndarray, cos_in: nl.ndarray, sin_in: nl.ndarray, n_head: int, sign: float = 1.0) -> nl.ndarray:
        rows, hd = x_in.shape
        t, d2 = cos_in.shape
        d = hd // n_head
        out = nl.ndarray((rows, hd), dtype=x_in.dtype, buffer=nl.shared_hbm)
        tiles = rows // _NKI_P
        tpt = t // _NKI_P                         # tiles per sequence
        nprog = nl.num_programs(0)
        per = tiles // nprog
        for ii in nl.affine_range(per):
            i = nl.program_id(0) * per + ii
            ti = i % tpt                     # position tile within the sequence (rows are b-major, t-minor)
            x = nl.ndarray((_NKI_P, hd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=x_in[i * _NKI_P:(i + 1) * _NKI_P, :])
            cs = nl.ndarray((_NKI_P, d2), dtype=nl.float32, buffer=nl.sbuf)
            sn = nl.ndarray((_NKI_P, d2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs, src=cos_in[ti * _NKI_P:(ti + 1) * _NKI_P, :])
            nisa.dma_copy(dst=sn, src=sin_in[ti * _NKI_P:(ti + 1) * _NKI_P, :])
            if sign < 0:
                nisa.tensor_scalar(dst=sn, data=sn, op0=nl.multiply, operand0=-1.0)
            o = nl.ndarray((_NKI_P, hd), dtype=nl.float32, buffer=nl.sbuf)
            t1 = nl.ndarray((_NKI_P, d2), dtype=nl.float32, buffer=nl.sbuf)
            t2 = nl.ndarray((_NKI_P, d2), dtype=nl.float32, buffer=nl.sbuf)
            for h in range(n_head):
                a = h * d
                x1 = x[:, a:a + d2]
                x2 = x[:, a + d2:a + d]
                nisa.tensor_tensor(dst=t1, data1=x1, data2=cs, op=nl.multiply)          # x1*cos
                nisa.tensor_tensor(dst=t2, data1=x2, data2=sn, op=nl.multiply)          # x2*sin
                nisa.tensor_tensor(dst=o[:, a:a + d2], data1=t1, data2=t2, op=nl.add)    # y1
                nisa.tensor_tensor(dst=t1, data1=x2, data2=cs, op=nl.multiply)          # x2*cos
                nisa.tensor_tensor(dst=t2, data1=x1, data2=sn, op=nl.multiply)          # x1*sin
                nisa.tensor_tensor(dst=o[:, a + d2:a + d], data1=t1, data2=t2, op=nl.subtract)   # y2 = x2*cos - x1*sin
            nisa.dma_copy(dst=out[i * _NKI_P:(i + 1) * _NKI_P, :], src=o)
        return out

    _NKI_ROPE = (wrap_nki(rope_kernel), (int(get_logical_neuron_cores()),))
    return _NKI_ROPE


class _NKIRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, cos2d, sin2d, n_head):
        kern, grid = build_nki_rope()
        ctx.save_for_backward(cos2d, sin2d)
        ctx.n_head = n_head
        return kern[grid](x2d.contiguous(), cos2d, sin2d, n_head=n_head, sign=1.0)

    @staticmethod
    def backward(ctx, dy):
        kern, grid = build_nki_rope()
        cos2d, sin2d = ctx.saved_tensors
        return kern[grid](dy.contiguous(), cos2d, sin2d, n_head=ctx.n_head, sign=-1.0), None, None, None


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


# -----------------------------------------------------------------------------
# NKI flash attention (Trn2): vendor kernels nkilib attention_cte (fwd) / attention_bwd (bwd) with sequence-packing
# bounds (= our document mask, causal applied on top) and a per-row denominator sink logit. Exposed as two
# torch.library custom ops built with torch_neuronx's `nki_op` (dynamo sees an opaque op -> the model stays ONE graph;
# a plain autograd.Function around `wrap_nki` graph-breaks per layer and hits the recompile limit, S_NKI 2026-09-06)
# with autograd wired by `torch.library.register_autograd`. Imports are guarded and lazy: off-Neuron the explicit path
# in CausalSelfAttention implements the identical math (eval-public on CPU/CUDA).
_NKI_OPS = None


def build_nki_ops():
    """Define (once) tf_nki::doc_attn_fwd / doc_attn_bwd. Must run BEFORE torch.compile traces the model."""
    global _NKI_OPS
    if _NKI_OPS is not None:
        return _NKI_OPS
    from torch_neuronx.nki_hop import wrap_nki, nki_op
    from torch_neuronx.utils import get_logical_neuron_cores
    from nkilib.core.attention.attention_cte import attention_cte
    from nkilib.core.attention.attention_bwd import attention_bwd
    grid = (int(get_logical_neuron_cores()),)
    fwd = wrap_nki(attention_cte)
    bwd = wrap_nki(attention_bwd)

    @nki_op("tf_nki::doc_attn_fwd", mutates_args={})
    def doc_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink: torch.Tensor,
                     start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # q (pre-scaled by 1/sqrt(d)), k, v: (b, h, t, d) bf16; sink (h,) fp32; start/end (b, t) int32
        b, h, t, d = q.shape
        qf = q.reshape(b * h, t, d).contiguous()
        kf = k.reshape(b * h, t, d).contiguous()
        vf = v.reshape(b * h, t, d).contiguous()
        sink_rows = sink.float().view(1, h).expand(b, h).reshape(b * h, 1).contiguous()
        bmin = start.unsqueeze(1).expand(b, h, t).reshape(b * h, t, 1).contiguous()
        bmax = end.unsqueeze(1).expand(b, h, t).reshape(b * h, t, 1).contiguous()
        # the kernel's sequence-packing (range_select) path requires scale == 1.0 (hence the pre-scaled q)
        out, neg_max, recip = fwd[grid](qf, kf, vf, tp_q=True, tp_k=True, scale=1.0, causal_mask=True,
                                        cache_softmax=True, bound_min=bmin, bound_max=bmax, sink=sink_rows)
        return out.reshape(b, h, t, d), neg_max, recip

    @nki_op("tf_nki::doc_attn_bwd", mutates_args={})
    def doc_attn_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, dy: torch.Tensor,
                     neg_max: torch.Tensor, recip: torch.Tensor, sink: torch.Tensor, start: torch.Tensor,
                     end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, t, d = q.shape
        tr = lambda x: x.transpose(-2, -1).contiguous()                     # (b, h, d, t) layouts for the bwd
        lse = (-(neg_max.float() + torch.log(recip.float()))).reshape(b, h, *neg_max.shape[-2:]).contiguous()
        res = bwd[grid](tr(q), tr(k), tr(v), tr(out), tr(dy.to(q.dtype)), lse,
                        sinks_ref=sink.float().view(1, h).expand(b, h).contiguous(),
                        bound_min=start.float().contiguous(), bound_max=end.float().contiguous(),
                        use_causal_mask=True, mixed_precision=True, softmax_scale=1.0)
        dq, dk, dv = (r.transpose(-2, -1).contiguous() for r in res[:3])
        dsink = res[3].float().sum(dim=0)
        return dq, dk, dv, dsink

    def setup_context(ctx, inputs, output):
        q, k, v, sink, start, end = inputs
        out, neg_max, recip = output
        ctx.save_for_backward(q, k, v, sink, start, end, out, neg_max, recip)

    def backward(ctx, dout, _dneg_max, _drecip):
        q, k, v, sink, start, end, out, neg_max, recip = ctx.saved_tensors
        dq, dk, dv, dsink = torch.ops.tf_nki.doc_attn_bwd(q, k, v, out, dout, neg_max, recip, sink, start, end)
        return dq, dk, dv, dsink.to(sink.dtype), None, None

    torch.library.register_autograd("tf_nki::doc_attn_fwd", backward, setup_context=setup_context)
    _NKI_OPS = (torch.ops.tf_nki.doc_attn_fwd, torch.ops.tf_nki.doc_attn_bwd)
    return _NKI_OPS


def nki_doc_attention(q, k, v, sink, start, end, scale: float):
    """softmax_j([q.k_j * scale for j in doc(i), j <= i] + [sink]) @ v; q, k, v (b, h, t, d) bf16, sink (h,) fp32."""
    fwd_op = build_nki_ops()[0]
    q_scaled = (q * scale).to(q.dtype)
    return fwd_op(q_scaled, k, v, sink, start, end)[0]


DOC_PENALTY = 1024.0   # power of two: every product below stays exact in bf16 (digits <= 7, squares <= 49)


def doc_score_dims(docs: torch.Tensor, dtype):
    """Encode doc ids (b, t) as q-side / k-side 6-vectors so that q_ext . k_ext = -DOC_PENALTY * ((da)^2 + (db)^2) with
    d = 8a + b (a, b in 0..7): zero for the same document, <= -1024 otherwise. Ids wrap mod 64 (a row has far fewer docs)."""
    d = (docs % 64)
    a = (d // 8).to(torch.float32)
    b = (d % 8).to(torch.float32)
    one = torch.ones_like(a)
    qx = torch.stack([a * a, -2.0 * a, one, b * b, -2.0 * b, one], dim=-1) * (-DOC_PENALTY)
    kx = torch.stack([one, a, a * a, one, b, b * b], dim=-1)
    return qx.to(dtype), kx.to(dtype)          # (b, t, 6) each


def doc_bounds(idx: torch.Tensor):
    """Per-position [doc start, doc end) as int32 (b, t): the KV range each query may attend to (the kernel clamps
    the end to the causal limit). Same semantics as the explicit mask: doc = cumsum(idx == BOS); tokens before the
    first BOS form doc 0 starting at 0. Built with scatter/gather only (no cummax) so it lowers on every backend."""
    b, t = idx.shape
    is_bos = idx == BOS_ID
    pos = torch.arange(t, device=idx.device, dtype=torch.int32).unsqueeze(0).expand(b, t)
    docs = is_bos.to(torch.int32).cumsum(dim=1)                       # doc id per position, 0..n
    start_tab = torch.zeros(b, t + 2, dtype=torch.int32, device=idx.device)
    start_tab = start_tab.scatter(1, (docs * is_bos.to(torch.int32)).long(), pos * is_bos.to(torch.int32))
    start = start_tab.gather(1, docs.long())                          # start of the doc containing each position
    nxt = start_tab.gather(1, (docs + 1).long())                      # start of the next doc (0 = none)
    end = torch.where(nxt > 0, nxt, torch.full_like(nxt, t))
    return start, end


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, registry: "StackRegistry"):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        self.qk_dim = config.qk_dim or self.head_dim   # narrow QK: q/k at qk_dim, v stays head_dim
        self.merged = config.merged_qkv
        if self.merged:
            self.c_qkv = StackedLinear(registry, 2 * self.n_head * self.qk_dim + config.n_embd, config.n_embd)
        else:
            self.c_q = StackedLinear(registry, self.n_head * self.qk_dim, config.n_embd)
            self.c_k = StackedLinear(registry, self.n_head * self.qk_dim, config.n_embd)
            self.c_v = StackedLinear(registry, config.n_embd, config.n_embd)
        self.c_proj = StackedLinear(registry, config.n_embd, config.n_embd)

    def forward(self, x, ve, cos, sin, block_mask=None, dense_mask=None, k_sink=None, v_sink=None,
                sink=None, bounds=None, v0=None, vr_lambda=None, doc_dims=None):
        b, t, c = x.shape
        if self.merged:
            qkv = self.c_qkv(x)
            nq = self.n_head * self.qk_dim
            q, k, vv = qkv[..., :nq], qkv[..., nq:2 * nq], qkv[..., 2 * nq:]
        else:
            q, k, vv = self.c_q(x), self.c_k(x), self.c_v(x)
        q = q.view(b, t, self.n_head, self.qk_dim)
        k = k.view(b, t, self.n_head, self.qk_dim)
        v = (vv + ve).view(b, t, self.n_head, self.head_dim)
        if vr_lambda is not None:                    # value residual: v += lambda * (v0 - v), v0 = layer-0 value
            if v0 is None:
                v0 = v
            else:
                v = v + vr_lambda * (v0 - v)
        if (ROPE_IMPL == "nki" and x.device.type == "neuron" and not torch.compiler.is_compiling()
                and t % 128 == 0 and (b * t) % 256 == 0):
            cos2d = cos.reshape(t, -1).float().contiguous()
            sin2d = sin.reshape(t, -1).float().contiguous()
            q = _NKIRope.apply(q.reshape(b * t, -1), cos2d, sin2d, self.n_head).view(b, t, self.n_head, self.qk_dim)
            k = _NKIRope.apply(k.reshape(b * t, -1), cos2d, sin2d, self.n_head).view(b, t, self.n_head, self.qk_dim)
            q = norm(q)
            k = norm(k)
        else:
            q = norm(apply_rotary_emb(q, cos, sin))
            k = norm(apply_rotary_emb(k, cos, sin))
        if doc_dims is not None:                      # DOCMASK=score: the LAST 6 q/k dims carry the document penalty
            qx, kx = doc_dims                          # (head_dim stays <= 128 for the kernels; stacks keep their shapes)
            q = torch.cat([q[..., :-6], qx.unsqueeze(2).expand(-1, -1, self.n_head, -1)], dim=-1)
            k = torch.cat([k[..., :-6], kx.unsqueeze(2).expand(-1, -1, self.n_head, -1)], dim=-1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if sink is not None:
            # sink-logit mode: no registers; one learned logit per head in the softmax denominator only
            scale = 1.0 / math.sqrt(self.qk_dim)
            if bounds is not None and q.device.type == "neuron":
                y = nki_doc_attention(q, k, v, sink, bounds[0], bounds[1], scale)
            else:
                scores = (q.float() @ k.float().transpose(-2, -1)) * scale
                scores = scores.masked_fill(~dense_mask, float("-inf"))
                s_col = sink.float().view(1, self.n_head, 1, 1).expand(b, -1, t, 1)
                p = F.softmax(torch.cat([scores, s_col], dim=-1), dim=-1)[..., :t]
                y = (p @ v.float()).to(v.dtype)
            y = y.transpose(1, 2).contiguous().view(b, t, c)
            return self.c_proj(y), v0
        # registers: NUM_SINKS always-visible kv slots (learned k, learned v, zero-init)
        sk = k_sink.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1).to(k.dtype)
        if doc_dims is not None:
            sk = torch.cat([sk[..., :-6], torch.zeros(*sk.shape[:-1], 6, dtype=sk.dtype, device=sk.device)], dim=-1)
        sv = v_sink.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1).to(v.dtype)
        k = torch.cat([sk, k], dim=2)
        v = torch.cat([sv, v], dim=2)
        if block_mask is not None:
            _, flex_attention = _get_flex()
            if q.is_cuda:
                y = flex_attention(q, k, v, block_mask=block_mask, kernel_options=FLEX_KERNEL_OPTS)
            else:   # Neuron: the backend's flex pass lowers this to an inlined NKI kernel (static block mask)
                y = flex_attention(q, k, v, block_mask=block_mask)
        elif ATTN_IMPL == "sdpa":
            # fused path: bool mask (True = attend), scale = 1/sqrt(qk_dim) from q's last dim
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)
        else:
            # Portable path (Neuron/CPU): explicit attention with only baseline-proven ops.
            scores = (q.float() @ k.float().transpose(-2, -1)) * (1.0 / math.sqrt(self.qk_dim))
            scores = scores.masked_fill(~dense_mask, float("-inf"))
            y = (F.softmax(scores, dim=-1) @ v.float()).to(v.dtype)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y), v0


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, registry: "StackRegistry"):
        super().__init__()
        self.c_fc = StackedLinear(registry, MLP_RATIO * config.n_embd, config.n_embd)
        self.c_proj = StackedLinear(registry, config.n_embd, MLP_RATIO * config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(relu_squared(self.c_fc(x)))


class Block(nn.Module):
    """Per-block scalars (resid/x0/ve lambdas, k/v registers) live in GPT-level stacks (`block_scalars`) and
    arrive as slices: one AdamW tensor per kind instead of five per block (~50 tiny tensors before)."""

    def __init__(self, config: GPTConfig, registry: "StackRegistry"):
        super().__init__()
        self.attn = CausalSelfAttention(config, registry)
        self.mlp = MLP(config, registry)

    def forward(self, x, x0, ve, cos, sin, block_mask, dense_mask, resid_lambda, x0_lambda, ve_lambda, k_sink, v_sink,
                sink=None, bounds=None, v0=None, vr_lambda=None, doc_dims=None):
        x = resid_lambda * x + x0_lambda * x0
        a, v0 = self.attn(norm(x), ve_lambda * ve, cos, sin, block_mask, dense_mask, k_sink, v_sink, sink, bounds, v0, vr_lambda,
                          doc_dims)
        x = x + a
        x = x + self.mlp(norm(x))
        return x, v0


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "vte": nn.Embedding(padded_vocab_size, config.n_embd),
            "vte_mid": nn.Embedding(padded_vocab_size, config.n_embd),
            "vte2": nn.Embedding(padded_vocab_size, config.n_embd),
            "bte": nn.Embedding(BIGRAM_SIZE, config.n_embd),
        })
        registry = StackRegistry()
        self.transformer["h"] = nn.ModuleList([Block(config, registry) for _ in range(config.n_layer)])
        if config.mtp:
            self.mtp_proj = StackedLinear(registry, config.n_embd, config.n_embd)   # joins the (n_embd x n_embd) stack
        self.stacks = registry.build()                      # one (n, out, in) Parameter per weight shape
        for m in self.modules():
            if isinstance(m, StackedLinear):
                m.bind(self.stacks[m.key])
        L, S, H = config.n_layer, NUM_SINKS, config.n_head
        qk = config.qk_dim or config.n_embd // config.n_head
        self.block_scalars = nn.ParameterDict({
            "resid": nn.Parameter(torch.ones(L, 1)),
            "x0": nn.Parameter(torch.zeros(L, 1)),
            "ve": nn.Parameter(torch.full((L, 1), 0.5)),
        })
        if config.value_resid:
            self.block_scalars["vr"] = nn.Parameter(torch.zeros(L, 1))
        if config.ple:
            self.block_scalars["ple"] = nn.Parameter(torch.full((L, 1), 0.5))
            self.transformer["ple"] = nn.ModuleList([nn.Embedding(padded_vocab_size, config.n_embd) for _ in range(L)])
        if config.backout:
            self.backout_lambda = nn.Parameter(torch.zeros(1))
        # per-step attention windows (device tensors so the compiled graph is invariant to their values)
        self.register_buffer("win_local", torch.tensor(config.window if config.window else config.sequence_len, dtype=torch.int32), persistent=False)
        self.register_buffer("win_global", torch.tensor(config.sequence_len, dtype=torch.int32), persistent=False)
        if config.sink_mode == "logit":
            # init log(NUM_SINKS): the same denominator mass as NUM_SINKS zero-logit registers at step 0
            self.block_scalars["sink"] = nn.Parameter(torch.full((L, H), math.log(NUM_SINKS)))
        else:
            self.block_scalars["k_sink"] = nn.Parameter(torch.zeros(L, S, H, qk))
            self.block_scalars["v_sink"] = nn.Parameter(torch.zeros(L, S, H, config.n_embd // config.n_head))
        if config.use_trigram:
            self.transformer["tte"] = nn.Embedding(TRIGRAM_SIZE, config.n_embd)
            self.trigram_lambda = nn.Parameter(torch.full((1,), 0.2))
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.skip_lambdas = nn.Parameter(torch.ones(config.n_layer // 2))
        self.bigram_lambda = nn.Parameter(torch.full((1,), 0.2))
        self.softcap_pos = nn.Parameter(torch.full((1,), LOGIT_SOFTCAP))
        self.softcap_neg = nn.Parameter(torch.full((1,), LOGIT_SOFTCAP))
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.qk_dim or config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.vte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.vte_mid.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.vte2.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.bte.weight, mean=0.0, std=0.8)
        if self.config.use_trigram:
            torch.nn.init.normal_(self.transformer.tte.weight, mean=0.0, std=0.8)
            self.trigram_lambda.fill_(0.2)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        for block in self.transformer.h:   # .weight is a slice view of the shared stack; in-place under no_grad
            if self.config.merged_qkv:
                torch.nn.init.uniform_(block.attn.c_qkv.weight, -s, s)
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.block_scalars["resid"].fill_(1.0)
        self.block_scalars["x0"].fill_(0.05)
        self.block_scalars["ve"].fill_(0.5)
        if "vr" in self.block_scalars:
            self.block_scalars["vr"].zero_()
        if "ple" in self.block_scalars:
            self.block_scalars["ple"].fill_(0.5)
            for e in self.transformer.ple:
                torch.nn.init.normal_(e.weight, mean=0.0, std=0.8)
        if self.config.backout:
            self.backout_lambda.zero_()
        if self.config.mtp:
            torch.nn.init.zeros_(self.mtp_proj.weight)
        for key in ("k_sink", "v_sink"):
            if key in self.block_scalars:
                self.block_scalars[key].zero_()
        if "sink" in self.block_scalars:
            self.block_scalars["sink"].fill_(math.log(NUM_SINKS))
        self.skip_lambdas.fill_(1.0)
        self.bigram_lambda.fill_(0.2)
        self.softcap_pos.fill_(LOGIT_SOFTCAP)
        self.softcap_neg.fill_(LOGIT_SOFTCAP)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.qk_dim or self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = ROPE_BASE):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (freqs / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def _static_block_mask(self, device, t: int):
        """Static (data-independent) block mask: NUM_SINKS register prefix always visible + causal; built once."""
        cache = getattr(self, "_bm_cache", None)
        if cache is None or cache[0] != (str(device), t):
            create_block_mask, _ = _get_flex()

            def causal_sink(b, h, q_idx, kv_idx):
                return (kv_idx < NUM_SINKS) | (q_idx + NUM_SINKS >= kv_idx)

            bm = create_block_mask(causal_sink, None, None, t, t + NUM_SINKS, device="cpu", BLOCK_SIZE=128)
            self._bm_cache = ((str(device), t), bm.to(device))
        return self._bm_cache[1]

    @torch.no_grad()
    def set_windows(self, frac: float) -> None:
        """Window warmup: every window grows linearly (in 128-token buckets) from 128 to its target over the warmup."""
        def grow(target: int) -> int:
            w = 128 + (target - 128) * frac
            return int(min(target, max(128, 128 * round(w / 128))))
        tgt_l = self.config.window if self.config.window else self.config.sequence_len
        # host->device copies (a python-int fill_ bakes the value into a NEFF -> one recompile per distinct value)
        self.win_local.copy_(torch.tensor(grow(tgt_l), dtype=torch.int32))
        self.win_global.copy_(torch.tensor(grow(self.config.sequence_len), dtype=torch.int32))

    def setup_optimizer(self):
        scalar_params = [self.skip_lambdas, self.bigram_lambda, self.softcap_pos, self.softcap_neg]
        if self.config.backout:
            scalar_params.append(self.backout_lambda)
        if self.config.use_trigram:
            scalar_params.append(self.trigram_lambda)
        scalar_params.extend(self.block_scalars[k] for k in ("resid", "x0", "ve", "k_sink", "v_sink", "sink", "vr", "ple") if k in self.block_scalars)
        embedding_params = (list(self.transformer.wte.parameters())
                            + list(self.transformer.vte.parameters())
                            + list(self.transformer.vte_mid.parameters())
                            + list(self.transformer.vte2.parameters())
                            + list(self.transformer.bte.parameters()))
        if self.config.use_trigram:
            embedding_params += list(self.transformer.tte.parameters())
        if self.config.ple:
            embedding_params += list(self.transformer.ple.parameters())
        lm_head_params = list(self.lm_head.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for key in sorted(self.stacks.keys()):   # one permanently stacked (n, out, in) Parameter per shape
            param_groups.append(dict(kind="muon", params=[self.stacks[key]], lr=MATRIX_LR, momentum=0.95, beta2=0.90, ns_steps=5, weight_decay=WEIGHT_DECAY))
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean"):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        if targets is not None:
            # Neuron dynamo backend (torch.compile backend="neuron"): user inputs saved for backward unchanged
            # (idx by the embedding gather, targets by cross-entropy) become pass-through outputs and collide in
            # its input-output alias table ("Output 2 already aliases input 123, cannot also alias input 0", Trn2
            # S1f 2026-09-05). Cloning makes the saved tensors intermediates. Tiny int copies; math-neutral.
            idx = idx.clone()
            targets = targets.clone()
        cos, sin = self.cos[:, :t], self.sin[:, :t]

        # hashed bigram table: int32 multiplicative hash (Neuron-friendly; wraps mod 2^32)
        prev = torch.cat([torch.full_like(idx[:, :1], BOS_ID), idx[:, :-1]], dim=1)
        pair = prev.to(torch.int32) * 8192 + idx.to(torch.int32)
        bh = (((pair * -1640531527) >> 13) & (BIGRAM_SIZE - 1)).long()

        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = x + self.bigram_lambda * self.transformer.bte(bh).to(COMPUTE_DTYPE)
        if self.config.use_trigram:
            # hashed (prev2, prev, cur) trigram table, own gate; same int32 mixing, different shift
            prev2 = torch.cat([torch.full_like(idx[:, :2], BOS_ID), idx[:, :-2]], dim=1)
            tri = (prev2.to(torch.int32) * 8192 + prev.to(torch.int32)) * 8192 + idx.to(torch.int32)
            th = (((tri * -1640531527) >> 15) & (TRIGRAM_SIZE - 1)).long()
            x = x + self.trigram_lambda * self.transformer.tte(th).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x.clone()  # NOT `x0 = x`: under torch.compile(backend="neuron") graph breaks make x/x0 subgraph
        #                 inputs, and dynamo rejects an output that aliases two inputs (Trn2 S1c, 2026-09-05)
        ve1 = self.transformer.vte(idx).to(COMPUTE_DTYPE)
        ve_mid = self.transformer.vte_mid(idx).to(COMPUTE_DTYPE)
        ve2 = self.transformer.vte2(idx).to(COMPUTE_DTYPE)
        t1 = self.config.n_layer // 3
        t2 = self.config.n_layer - t1

        # doc-masked attention + registers; same math on every backend
        docs = (idx == BOS_ID).cumsum(dim=1)
        score_doc = self.config.docmask == "score"
        doc_dims = doc_score_dims(docs, COMPUTE_DTYPE) if score_doc else None
        if self.config.docmask == "mask6":            # test reference: boolean mask with the same 6 dims zeroed
            z = torch.zeros(*docs.shape, 6, dtype=COMPUTE_DTYPE, device=idx.device)
            doc_dims = (z, z)
        block_mask = None
        dense_mask = None
        bounds = None
        bounds_l = None
        dense_mask_local = None
        logit_sink = self.config.sink_mode == "logit"
        if ATTN_IMPL == "flex" and idx.device.type == "neuron" and score_doc and not logit_sink:
            block_mask = self._static_block_mask(idx.device, t)      # backend lowers flex_attention to an inlined NKI kernel
        elif logit_sink and ATTN_IMPL == "nki" and idx.device.type == "neuron" and _NKI_OPS is not None:
            bounds = doc_bounds(idx)                                  # NKI kernels take the doc mask as bounds
            pos_i = torch.arange(t, device=idx.device, dtype=torch.int32).unsqueeze(0)
            bounds_g = (torch.maximum(bounds[0], pos_i - self.win_global + 1), bounds[1])
            bounds_l = (torch.maximum(bounds[0], pos_i - self.win_local + 1), bounds[1]) if self.config.window else bounds_g
            bounds = bounds_g
        elif idx.is_cuda and not logit_sink:
            create_block_mask, _ = _get_flex()
            docs_kv = torch.cat([docs[:, :1].expand(-1, NUM_SINKS), docs], dim=1)

            def doc_causal_sink(b, h, q_idx, kv_idx):
                real = (q_idx + NUM_SINKS >= kv_idx) & (docs[b, q_idx] == docs_kv[b, kv_idx])
                return (kv_idx < NUM_SINKS) | real

            def causal_sink(b, h, q_idx, kv_idx):     # STATIC mask (DOCMASK=score carries the document term in the scores)
                return (kv_idx < NUM_SINKS) | (q_idx + NUM_SINKS >= kv_idx)

            if score_doc:
                doc_causal_sink = causal_sink

            block_mask = create_block_mask(doc_causal_sink, idx.size(0), None, t, t + NUM_SINKS, device=idx.device)
        else:
            pos = torch.arange(t, device=idx.device)
            causal = pos.view(1, t, 1) >= pos.view(1, 1, t)
            same_doc = docs.unsqueeze(2) == docs.unsqueeze(1)
            real = (causal & same_doc) if not score_doc else causal.expand(idx.size(0), t, t)   # (b, t, t)
            dist = pos.view(1, t, 1) - pos.view(1, 1, t)          # query - key distance (>= 0 where causal)
            real_g = real & (dist < self.win_global)              # window warmup shrinks the global window early on
            real_l = real & (dist < self.win_local) if self.config.window else real_g
            def _dense(r):
                if logit_sink:
                    return r.unsqueeze(1)                          # (b, 1, t, t); the sink column is appended in attn
                sinks = torch.ones(idx.size(0), t, NUM_SINKS, dtype=torch.bool, device=idx.device)
                return torch.cat([sinks, r], dim=2).unsqueeze(1)   # (b, 1, t, t+S)
            dense_mask = _dense(real_g)
            dense_mask_local = _dense(real_l) if self.config.window else dense_mask

        n_enc = self.config.n_layer // 2
        n_dec_start = self.config.n_layer - n_enc
        skips: list[torch.Tensor] = []
        bs = self.block_scalars
        v0 = None
        x_mid = None
        for i, block in enumerate(self.transformer.h):
            if i >= n_dec_start:
                x = x + self.skip_lambdas[i - n_dec_start] * skips.pop()
            if self.config.ple:
                x = x + bs["ple"][i] * self.transformer.ple[i](idx).to(COMPUTE_DTYPE)
            ve_i = ve1 if i < t1 else (ve_mid if i < t2 else ve2)
            local = bool(self.config.window) and (i % 2 == 1)
            dm = dense_mask_local if (local and dense_mask is not None) else dense_mask
            bd = (bounds_l if local else bounds) if bounds is not None else None
            vr = bs["vr"][i] if self.config.value_resid else None
            if logit_sink:
                x, v0 = block(x, x0, ve_i, cos, sin, block_mask, dm,
                              bs["resid"][i], bs["x0"][i], bs["ve"][i], None, None, bs["sink"][i], bd, v0, vr, doc_dims)
            else:
                x, v0 = block(x, x0, ve_i, cos, sin, block_mask, dm,
                              bs["resid"][i], bs["x0"][i], bs["ve"][i], bs["k_sink"][i], bs["v_sink"][i], None, None, v0, vr,
                              doc_dims)
            if i < n_enc:
                skips.append(x)
            if i == n_enc - 1:
                x_mid = x
        if self.config.backout:
            x = x - self.backout_lambda * x_mid
        x_pre = x
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        sp = self.softcap_pos.clamp_min(1.0).float()
        sn = self.softcap_neg.clamp_min(1.0).float()
        logits = torch.where(logits >= 0, sp * torch.tanh(logits / sp), sn * torch.tanh(logits / sn))
        if targets is None:
            return logits
        if CE_IMPL == "manual":
            # Neuron: F.cross_entropy lowers to a custom autograd Function that splits the compiled graph
            # (profiler 2026-09-06: separate region + eager ne + host copies). Plain ops fuse into one graph.
            flat_t = targets.reshape(-1)
            valid = flat_t != -1
            logp = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
            nll = -logp.gather(1, flat_t.clamp_min(0).unsqueeze(1)).squeeze(1) * valid.to(logp.dtype)
            if loss_reduction == "none":
                return nll.view_as(targets)
            loss = nll.sum() / valid.sum().clamp_min(1).to(nll.dtype)
            if self.config.mtp and MTP_WEIGHT > 0 and self.training:
                # multi-token prediction: predict token t+2 from a zero-init projected copy of the final hidden state
                h2 = norm(x_pre + self.mtp_proj(norm(x_pre)))
                logits2 = self.lm_head(h2)[..., :self.config.vocab_size].float()
                logits2 = torch.where(logits2 >= 0, sp * torch.tanh(logits2 / sp), sn * torch.tanh(logits2 / sn))
                t2 = torch.cat([targets[:, 1:], torch.full_like(targets[:, :1], -1)], dim=1).reshape(-1)
                valid2 = t2 != -1
                logp2 = F.log_softmax(logits2.reshape(-1, logits2.size(-1)), dim=-1)
                nll2 = -logp2.gather(1, t2.clamp_min(0).unsqueeze(1)).squeeze(1) * valid2.to(logp2.dtype)
                loss = loss + MTP_WEIGHT * (nll2.sum() / valid2.sum().clamp_min(1).to(nll2.dtype))
            return loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction == "none":
            return loss.view_as(targets)
        return loss


def build_config() -> GPTConfig:
    return GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=DEPTH,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        use_trigram=USE_TRIGRAM,
        qk_dim=QK_DIM,
        sink_mode=SINK_MODE,
        window=WINDOW,
        mtp=MTP_WEIGHT > 0,
        value_resid=VALUE_RESID,
        backout=BACKOUT,
        merged_qkv=MERGED_QKV,
        ple=PLE,
        docmask=DOCMASK,
    )


# -----------------------------------------------------------------------------
# Optimizer (identical to the shipped train.py)

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


# -----------------------------------------------------------------------------
# NKI fused Polar Express (Muon orthogonalization) for the stacked matrix parameters. One kernel launch per stack
# replaces ~40 eager launches; X and X^T stay SBUF-resident (bf16 operands, fp32 PSUM accumulation) across the 5
# iterations. Tall (m >= k) matrices only; wide stacks are orthogonalized as their transpose (polar factor commutes
# with transposition). Kernel source: nki_muon.py (dev copy); guarded import - off-Neuron the torch loop runs.
_NKI_MUON = None


def build_nki_muon():
    global _NKI_MUON
    if _NKI_MUON is not None:
        return _NKI_MUON
    # the NKI tracer resolves names through the kernel's module globals, not the builder's closure
    global nki, nisa, nl
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores
    _NKI_P, _FMAX = 128, 512

    @nki.jit
    def polar_express_tall(x_in: nl.ndarray, ns_steps: int = 5) -> nl.ndarray:
        n, m, k = x_in.shape
        mt, kt, mc = m // _NKI_P, k // _NKI_P, m // _FMAX
        out = nl.ndarray((n, m, k), dtype=x_in.dtype, buffer=nl.shared_hbm)
        nprog = nl.num_programs(0)
        per = n // nprog
        for ii in nl.affine_range(per):
            i = nl.program_id(0) * per + ii
            X = []
            for _j in range(mt):
                X.append(nl.ndarray((_NKI_P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
            Xt = []
            for _j in range(kt):
                Xt.append(nl.ndarray((_NKI_P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
            for p in range(mt):
                nisa.dma_copy(dst=X[p], src=x_in[i, p * _NKI_P:(p + 1) * _NKI_P, :])
            sq = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.sbuf)
            rowsum = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            part = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(rowsum, 0.0)
            for p in range(mt):
                nisa.tensor_tensor(dst=sq, data1=X[p], data2=X[p], op=nl.multiply)
                nisa.tensor_reduce(dst=part, op=nl.add, data=sq, axis=(1,))
                nisa.tensor_tensor(dst=rowsum, data1=rowsum, data2=part, op=nl.add)
            tot_ps = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.psum)
            ones32 = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(ones32, 1.0)
            nisa.nc_matmul(dst=tot_ps, stationary=ones32, moving=rowsum)       # all-partition sum, broadcast
            scale = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=scale, op=nl.sqrt, data=tot_ps)
            nisa.tensor_scalar(dst=scale, data=scale, op0=nl.multiply, operand0=1.01, op1=nl.add, operand1=1e-6)
            nisa.reciprocal(dst=scale, data=scale)
            for p in range(mt):
                nisa.tensor_scalar(dst=X[p], data=X[p], op0=nl.multiply, operand0=scale)
            for p in range(mt):
                for c in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=X[p][:, c * _NKI_P:(c + 1) * _NKI_P])
                    nisa.tensor_copy(dst=Xt[c][:, p * _NKI_P:(p + 1) * _NKI_P], src=tps)
            for it in range(ns_steps):
                a = POLAR_EXPRESS_COEFFS[it][0]
                b = POLAR_EXPRESS_COEFFS[it][1]
                c = POLAR_EXPRESS_COEFFS[it][2]
                A = []
                for _j in range(kt):
                    A.append(nl.ndarray((_NKI_P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for r in range(kt):
                    aps = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.psum)
                    for p in range(mt):
                        nisa.nc_matmul(dst=aps, stationary=X[p][:, r * _NKI_P:(r + 1) * _NKI_P], moving=X[p], accumulate=(p > 0))
                    nisa.tensor_copy(dst=A[r], src=aps)
                B = []
                for _j in range(kt):
                    B.append(nl.ndarray((_NKI_P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for r in range(kt):
                    a2ps = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.psum)
                    for q in range(kt):
                        nisa.nc_matmul(dst=a2ps, stationary=A[q][:, r * _NKI_P:(r + 1) * _NKI_P], moving=A[q], accumulate=(q > 0))
                    bA = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=bA, data=A[r], op0=nl.multiply, operand0=b)
                    nisa.scalar_tensor_tensor(dst=B[r], data=a2ps, op0=nl.multiply, operand0=c, op1=nl.add, operand1=bA)
                Xn = []
                for _j in range(mt):
                    Xn.append(nl.ndarray((_NKI_P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for p in range(mt):
                    xps = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.psum)
                    for r in range(kt):
                        nisa.nc_matmul(dst=xps, stationary=Xt[r][:, p * _NKI_P:(p + 1) * _NKI_P], moving=B[r], accumulate=(r > 0))
                    nisa.scalar_tensor_tensor(dst=Xn[p], data=X[p], op0=nl.multiply, operand0=a, op1=nl.add, operand1=xps)
                Xtn = []
                for _j in range(kt):
                    Xtn.append(nl.ndarray((_NKI_P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
                for r in range(kt):
                    for ch in range(mc):
                        tps2 = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                        for q in range(kt):
                            nisa.nc_matmul(dst=tps2, stationary=B[q][:, r * _NKI_P:(r + 1) * _NKI_P],
                                           moving=Xt[q][:, ch * _FMAX:(ch + 1) * _FMAX], accumulate=(q > 0))
                        nisa.scalar_tensor_tensor(dst=Xtn[r][:, ch * _FMAX:(ch + 1) * _FMAX], data=Xt[r][:, ch * _FMAX:(ch + 1) * _FMAX],
                                                  op0=nl.multiply, operand0=a, op1=nl.add, operand1=tps2)
                X, Xt = Xn, Xtn
            for p in range(mt):
                nisa.dma_copy(dst=out[i, p * _NKI_P:(p + 1) * _NKI_P, :], src=X[p])
        return out

    kern = wrap_nki(polar_express_tall)
    grid = (int(get_logical_neuron_cores()),)

    def orthogonalize(x: torch.Tensor, ns_steps: int) -> torch.Tensor:
        # x: (n, out, in) bf16 momentum-blended gradient; includes the Frobenius prescale (done in-kernel)
        if x.shape[-2] >= x.shape[-1]:
            return kern[grid](x.contiguous(), ns_steps=ns_steps)
        return kern[grid](x.mT.contiguous(), ns_steps=ns_steps).mT

    _NKI_MUON = orthogonalize
    return _NKI_MUON


nki = None
nisa = None
nl = None   # bound by the builders (module globals for the tracer)
_NKI_ADAMW = None
_NKI_MUON_FUSED = None
_FMAX = 512
_NKI_P = 128


def build_nki_adamw():
    """Fused AdamW (dev copy: nki_adamw.py) for 2-D fp32 params with rows % 128 == 0: one launch per tensor instead of ~10."""
    global _NKI_ADAMW
    if _NKI_ADAMW is not None:
        return _NKI_ADAMW
    # the NKI tracer resolves names through the kernel's module globals, not the builder's closure
    global nki, nisa, nl
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores
    
    @nki.jit
    def adamw_step(p_in: nl.ndarray, g_in: nl.ndarray, m_in: nl.ndarray, v_in: nl.ndarray, scal: nl.ndarray,
                   beta1: float = 0.8, beta2: float = 0.995, eps: float = 1e-10):
        R, C = p_in.shape
        rt = R // _NKI_P
        out = nl.ndarray((R, C), dtype=p_in.dtype, buffer=nl.shared_hbm)
        out_m = nl.ndarray((R, C), dtype=p_in.dtype, buffer=nl.shared_hbm)
        out_v = nl.ndarray((R, C), dtype=p_in.dtype, buffer=nl.shared_hbm)
        # the 4 per-step scalars, pre-broadcast by the caller to [_NKI_P, 4]
        sc = nl.ndarray((_NKI_P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        nprog = nl.num_programs(0)
        per = rt // nprog
        for ii in nl.affine_range(per):
            i = nl.program_id(0) * per + ii
            rows = slice(i * _NKI_P, (i + 1) * _NKI_P)
            p = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
            g = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
            m = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
            v = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=p, src=p_in[rows, :])
            nisa.dma_copy(dst=g, src=g_in[rows, :])
            nisa.dma_copy(dst=m, src=m_in[rows, :])
            nisa.dma_copy(dst=v, src=v_in[rows, :])
            # p *= 1 - lr*wd   (scal[3] = lr*wd per partition)
            one_minus = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=one_minus, data=sc[:, 3:4], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            nisa.tensor_scalar(dst=p, data=p, op0=nl.multiply, operand0=one_minus)
            # m = b1*m + (1-b1)*g
            tmp = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tmp, data=g, op0=nl.multiply, operand0=1.0 - beta1)
            nisa.scalar_tensor_tensor(dst=m, data=m, op0=nl.multiply, operand0=beta1, op1=nl.add, operand1=tmp)
            # v = b2*v + (1-b2)*g*g
            nisa.tensor_tensor(dst=tmp, data1=g, data2=g, op=nl.multiply)
            nisa.tensor_scalar(dst=tmp, data=tmp, op0=nl.multiply, operand0=1.0 - beta2)
            nisa.scalar_tensor_tensor(dst=v, data=v, op0=nl.multiply, operand0=beta2, op1=nl.add, operand1=tmp)
            # denom = sqrt(v) * (1/sqrt(bias2)) + eps ; upd = m * (1/bias1) / denom ; p -= lr * upd
            den = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=den, op=nl.sqrt, data=v)
            nisa.tensor_scalar(dst=den, data=den, op0=nl.multiply, operand0=sc[:, 2:3], op1=nl.add, operand1=eps)
            nisa.reciprocal(dst=den, data=den)
            nisa.scalar_tensor_tensor(dst=tmp, data=m, op0=nl.multiply, operand0=sc[:, 1:2], op1=nl.multiply, operand1=den)
            nisa.scalar_tensor_tensor(dst=p, data=tmp, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.subtract, operand1=p, reverse1=True)
            nisa.dma_copy(dst=out[rows, :], src=p)
            nisa.dma_copy(dst=out_m[rows, :], src=m)
            nisa.dma_copy(dst=out_v[rows, :], src=v)
            nisa.dma_copy(dst=p_in[rows, :], src=p)     # in-place too (whether the torch wrapper honors it is tested on device)
            nisa.dma_copy(dst=m_in[rows, :], src=m)
            nisa.dma_copy(dst=v_in[rows, :], src=v)
        return out, out_m, out_v

    _NKI_ADAMW = (wrap_nki(adamw_step), (int(get_logical_neuron_cores()),))
    return _NKI_ADAMW


def build_nki_muon_fused():
    """Fully fused Muon step (dev copy: nki_muon.py muon_step_stack): momentum + Polar Express + second-moment scaling +
    masked weight decay update, one launch per stack, in place on w / momentum / second (outputs also returned)."""
    global _NKI_MUON_FUSED
    if _NKI_MUON_FUSED is not None:
        return _NKI_MUON_FUSED
    # the NKI tracer resolves names through the kernel's module globals, not the builder's closure
    global nki, nisa, nl
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores
    
    @nki.jit
    def muon_step_stack(w_in: nl.ndarray, g_in: nl.ndarray, mom_in: nl.ndarray, sec_in: nl.ndarray, scal: nl.ndarray,
                        ns_steps: int = 5, beta2: float = 0.9):
        n, o, i_ = w_in.shape
        tall = o >= i_
        m = o if tall else i_          # tall-frame rows
        k = i_ if tall else o          # tall-frame cols (== 512)
        mt, kt, mc = m // _NKI_P, k // _NKI_P, m // _FMAX
        w_out = nl.ndarray((n, o, i_), dtype=nl.float32, buffer=nl.shared_hbm)
        mom_out = nl.ndarray((n, o, i_), dtype=nl.float32, buffer=nl.shared_hbm)
        sec_out = nl.ndarray(sec_in.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        sc = nl.ndarray((_NKI_P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        ones32 = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(ones32, 1.0)
        nprog = nl.num_programs(0)
        per = n // nprog
        for ii in nl.affine_range(per):
            i = nl.program_id(0) * per + ii
            # ---- natural-layout tiles: o // 128 tiles of [128, i_]; momentum update + nesterov blend, cast to bf16
            ot = o // _NKI_P
            Gn = []
            for _j in range(ot):
                Gn.append(nl.ndarray((_NKI_P, i_), dtype=nl.bfloat16, buffer=nl.sbuf))
            for p in range(ot):
                g = nl.ndarray((_NKI_P, i_), dtype=nl.float32, buffer=nl.sbuf)
                mo = nl.ndarray((_NKI_P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=g, src=g_in[i, p * _NKI_P:(p + 1) * _NKI_P, :])
                nisa.dma_copy(dst=mo, src=mom_in[i, p * _NKI_P:(p + 1) * _NKI_P, :])
                # mo = mo + (1-mom)(g - mo) = mom*mo + (1-mom)*g ;  gb = g + mom*(mo - g) = (1-mom)*g + mom*mo
                t1 = nl.ndarray((_NKI_P, i_), dtype=nl.float32, buffer=nl.sbuf)
                one_m = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=one_m, data=sc[:, 0:1], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
                nisa.tensor_scalar(dst=t1, data=g, op0=nl.multiply, operand0=one_m)                         # (1-mom) g
                nisa.scalar_tensor_tensor(dst=mo, data=mo, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=t1)  # new momentum
                nisa.dma_copy(dst=mom_out[i, p * _NKI_P:(p + 1) * _NKI_P, :], src=mo)
                nisa.dma_copy(dst=mom_in[i, p * _NKI_P:(p + 1) * _NKI_P, :], src=mo)
                nisa.scalar_tensor_tensor(dst=Gn[p], data=mo, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=t1)  # blend, bf16
            # ---- Frobenius prescale over the natural tiles
            sq = nl.ndarray((_NKI_P, i_), dtype=nl.float32, buffer=nl.sbuf)
            rowsum = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            part = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(rowsum, 0.0)
            for p in range(ot):
                nisa.tensor_tensor(dst=sq, data1=Gn[p], data2=Gn[p], op=nl.multiply)
                nisa.tensor_reduce(dst=part, op=nl.add, data=sq, axis=(1,))
                nisa.tensor_tensor(dst=rowsum, data1=rowsum, data2=part, op=nl.add)
            tot_ps = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=tot_ps, stationary=ones32, moving=rowsum)
            scale = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=scale, op=nl.sqrt, data=tot_ps)
            nisa.tensor_scalar(dst=scale, data=scale, op0=nl.multiply, operand0=1.01, op1=nl.add, operand1=1e-6)
            nisa.reciprocal(dst=scale, data=scale)
            for p in range(ot):
                nisa.tensor_scalar(dst=Gn[p], data=Gn[p], op0=nl.multiply, operand0=scale)
            # ---- tall frame: X (mt tiles of [128, k]) and Xt (kt tiles of [128, m]); one of them is Gn, the other transposed
            X = []
            Xt = []
            if tall:
                X = Gn
                for _j in range(kt):
                    Xt.append(nl.ndarray((_NKI_P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
                for p in range(mt):
                    for c in range(kt):
                        tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=tps, data=X[p][:, c * _NKI_P:(c + 1) * _NKI_P])
                        nisa.tensor_copy(dst=Xt[c][:, p * _NKI_P:(p + 1) * _NKI_P], src=tps)
            else:
                Xt = Gn
                for _j in range(mt):
                    X.append(nl.ndarray((_NKI_P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for c in range(kt):
                    for p in range(mt):
                        tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=tps, data=Xt[c][:, p * _NKI_P:(p + 1) * _NKI_P])
                        nisa.tensor_copy(dst=X[p][:, c * _NKI_P:(c + 1) * _NKI_P], src=tps)
            for it in range(ns_steps):
                a = POLAR_EXPRESS_COEFFS[it][0]
                b = POLAR_EXPRESS_COEFFS[it][1]
                c = POLAR_EXPRESS_COEFFS[it][2]
                A = []
                for _j in range(kt):
                    A.append(nl.ndarray((_NKI_P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for r in range(kt):
                    aps = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.psum)
                    for p in range(mt):
                        nisa.nc_matmul(dst=aps, stationary=X[p][:, r * _NKI_P:(r + 1) * _NKI_P], moving=X[p], accumulate=(p > 0))
                    nisa.tensor_copy(dst=A[r], src=aps)
                B = []
                for _j in range(kt):
                    B.append(nl.ndarray((_NKI_P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for r in range(kt):
                    a2ps = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.psum)
                    for q in range(kt):
                        nisa.nc_matmul(dst=a2ps, stationary=A[q][:, r * _NKI_P:(r + 1) * _NKI_P], moving=A[q], accumulate=(q > 0))
                    bA = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=bA, data=A[r], op0=nl.multiply, operand0=b)
                    nisa.scalar_tensor_tensor(dst=B[r], data=a2ps, op0=nl.multiply, operand0=c, op1=nl.add, operand1=bA)
                Xn = []
                for _j in range(mt):
                    Xn.append(nl.ndarray((_NKI_P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for p in range(mt):
                    xps = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.psum)
                    for r in range(kt):
                        nisa.nc_matmul(dst=xps, stationary=Xt[r][:, p * _NKI_P:(p + 1) * _NKI_P], moving=B[r], accumulate=(r > 0))
                    nisa.scalar_tensor_tensor(dst=Xn[p], data=X[p], op0=nl.multiply, operand0=a, op1=nl.add, operand1=xps)
                Xtn = []
                for _j in range(kt):
                    Xtn.append(nl.ndarray((_NKI_P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
                for r in range(kt):
                    for ch in range(mc):
                        tps2 = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                        for q in range(kt):
                            nisa.nc_matmul(dst=tps2, stationary=B[q][:, r * _NKI_P:(r + 1) * _NKI_P],
                                           moving=Xt[q][:, ch * _FMAX:(ch + 1) * _FMAX], accumulate=(q > 0))
                        nisa.scalar_tensor_tensor(dst=Xtn[r][:, ch * _FMAX:(ch + 1) * _FMAX], data=Xt[r][:, ch * _FMAX:(ch + 1) * _FMAX],
                                                  op0=nl.multiply, operand0=a, op1=nl.add, operand1=tps2)
                X, Xt = Xn, Xtn
            # ---- second-moment scaling. Reduction axis = the tall-frame free axis (k): v_mean per tall-frame row.
            vmean = []                       # mt tiles of [128, 1] fp32
            for _j in range(mt):
                vmean.append(nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf))
            vsum = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(vsum, 0.0)
            sqk = nl.ndarray((_NKI_P, k), dtype=nl.float32, buffer=nl.sbuf)
            for p in range(mt):
                nisa.tensor_tensor(dst=sqk, data1=X[p], data2=X[p], op=nl.multiply)
                nisa.tensor_reduce(dst=vmean[p], op=nl.add, data=sqk, axis=(1,))
                nisa.tensor_scalar(dst=vmean[p], data=vmean[p], op0=nl.multiply, operand0=1.0 / k)
                nisa.tensor_tensor(dst=vsum, data1=vsum, data2=vmean[p], op=nl.add)
            vn_ps = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=vn_ps, stationary=ones32, moving=vsum)                # sum of v_mean over all rows
            vnorm = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=vnorm, data=vn_ps, op0=nl.multiply, operand0=float(k))
            nisa.activation(dst=vnorm, op=nl.sqrt, data=vnorm)                      # v_norm
            # second (per tall-frame row) lerp, step_size = rsqrt(max(second, 1e-10)), scaled = v_mean*k*step_size^2
            step = []
            ssum = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(ssum, 0.0)
            for p in range(mt):
                sec = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                if tall:
                    nisa.dma_copy(dst=sec, src=sec_in[i, p * _NKI_P:(p + 1) * _NKI_P, :])
                else:
                    nisa.dma_copy(dst=sec, src=sec_in[i, :, p * _NKI_P:(p + 1) * _NKI_P].ap(pattern=[[1, _NKI_P], [1, 1]], offset=0))
                t2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=t2, data=vmean[p], op0=nl.multiply, operand0=1.0 - beta2)
                nisa.scalar_tensor_tensor(dst=sec, data=sec, op0=nl.multiply, operand0=beta2, op1=nl.add, operand1=t2)
                if tall:
                    nisa.dma_copy(dst=sec_out[i, p * _NKI_P:(p + 1) * _NKI_P, :], src=sec)
                    nisa.dma_copy(dst=sec_in[i, p * _NKI_P:(p + 1) * _NKI_P, :], src=sec)
                else:
                    nisa.dma_copy(dst=sec_out[i, :, p * _NKI_P:(p + 1) * _NKI_P].ap(pattern=[[1, _NKI_P], [1, 1]], offset=0), src=sec)
                    nisa.dma_copy(dst=sec_in[i, :, p * _NKI_P:(p + 1) * _NKI_P].ap(pattern=[[1, _NKI_P], [1, 1]], offset=0), src=sec)
                st = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=st, data=sec, op0=nl.maximum, operand0=1e-10)
                nisa.activation(dst=st, op=nl.rsqrt, data=st)
                step.append(st)
                sc2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sc2, data1=st, data2=st, op=nl.multiply)
                nisa.scalar_tensor_tensor(dst=sc2, data=vmean[p], op0=nl.multiply, operand0=float(k), op1=nl.multiply, operand1=sc2)
                nisa.tensor_tensor(dst=ssum, data1=ssum, data2=sc2, op=nl.add)
            sn_ps = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=sn_ps, stationary=ones32, moving=ssum)
            ratio = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=ratio, op=nl.sqrt, data=sn_ps)                       # v_norm_new
            nisa.tensor_scalar(dst=ratio, data=ratio, op0=nl.maximum, operand0=1e-10)
            nisa.reciprocal(dst=ratio, data=ratio)
            nisa.tensor_tensor(dst=ratio, data1=ratio, data2=vnorm, op=nl.multiply)  # v_norm / v_norm_new (all partitions equal)
            # per tall-frame row factor f[r] = step_size[r] * ratio ; the update runs in the NATURAL frame
            # tall: natural rows == tall rows -> per-partition scalar; wide: natural rows are k, factor varies along the free axis
            f_rows = []
            for p in range(mt):
                fr = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=fr, data1=step[p], data2=ratio, op=nl.multiply)
                f_rows.append(fr)
            if not tall:
                # broadcast tile F[128, m]: F[:, r] = f[r]  via K=1 matmul: ones[1, 128]^T @ f_row[1, m]
                f_row = nl.ndarray((1, m), dtype=nl.float32, buffer=nl.sbuf)
                for p in range(mt):
                    tp = nl.ndarray((1, _NKI_P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=tp, data=f_rows[p])
                    nisa.tensor_copy(dst=f_row[:, p * _NKI_P:(p + 1) * _NKI_P], src=tp)
                ones_row = nl.ndarray((1, _NKI_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(ones_row, 1.0)
                Fb = nl.ndarray((_NKI_P, m), dtype=nl.float32, buffer=nl.sbuf)
                for ch in range(mc):
                    fps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=fps, stationary=ones_row, moving=f_row[:, ch * _FMAX:(ch + 1) * _FMAX])
                    nisa.tensor_copy(dst=Fb[:, ch * _FMAX:(ch + 1) * _FMAX], src=fps)
            # ---- update in the natural frame: g = orth * f ; mask = (g*w >= 0) ; w -= lr*g + lr*wd*w*mask
            Gnat = X if tall else Xt
            for p in range(ot):
                w = nl.ndarray((_NKI_P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=w, src=w_in[i, p * _NKI_P:(p + 1) * _NKI_P, :])
                gg = nl.ndarray((_NKI_P, i_), dtype=nl.float32, buffer=nl.sbuf)
                if tall:
                    nisa.tensor_scalar(dst=gg, data=Gnat[p], op0=nl.multiply, operand0=f_rows[p])
                else:
                    nisa.tensor_tensor(dst=gg, data1=Gnat[p], data2=Fb, op=nl.multiply)
                prod = nl.ndarray((_NKI_P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=prod, data1=gg, data2=w, op=nl.multiply)
                mask = nl.ndarray((_NKI_P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mask, data=prod, op0=nl.greater_equal, operand0=0.0)     # 1.0 where g*w >= 0
                wdm = nl.ndarray((_NKI_P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=wdm, data1=w, data2=mask, op=nl.multiply)
                nisa.tensor_scalar(dst=wdm, data=wdm, op0=nl.multiply, operand0=sc[:, 2:3])    # lr*wd*w*mask
                nisa.scalar_tensor_tensor(dst=gg, data=gg, op0=nl.multiply, operand0=sc[:, 1:2], op1=nl.add, operand1=wdm)  # lr*g + ...
                nisa.tensor_tensor(dst=w, data1=w, data2=gg, op=nl.subtract)
                nisa.dma_copy(dst=w_out[i, p * _NKI_P:(p + 1) * _NKI_P, :], src=w)
                nisa.dma_copy(dst=w_in[i, p * _NKI_P:(p + 1) * _NKI_P, :], src=w)
        return w_out, mom_out, sec_out

    _NKI_MUON_FUSED = (wrap_nki(muon_step_stack), (int(get_logical_neuron_cores()),))
    return _NKI_MUON_FUSED


class MuonAdamW(torch.optim.Optimizer):
    tensor_step = False   # True only under the compiled-optimizer dev knob (bias corrections from a device counter)

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    def _adamw_state(self, p):
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["step_t"] = torch.zeros((), dtype=torch.float32, device=p.device)
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        return state

    def _muon_state(self, group):
        p0 = group["params"][0]            # the permanently stacked (n, out, in) Parameter
        state = self.state[p0]
        n, out_f, in_f = p0.shape
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p0)
        if "second_momentum_buffer" not in state:
            state_shape = (n, out_f, 1) if out_f >= in_f else (n, 1, in_f)
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        return state

    @torch.no_grad()
    def init_state(self):
        """Pre-allocate all optimizer state so a compiled step() never traces the lazy-init branches."""
        for group in self.param_groups:
            if group["kind"] == "adamw":
                for p in group["params"]:
                    self._adamw_state(p)
            elif group["kind"] == "muon" and group["params"]:
                self._muon_state(group)

    @torch.no_grad()
    def refresh_scalars(self):
        """Host->device refresh of the per-step scalars (lr per group, momentum per Muon group) with ONE
        copy per step: every scalar is a view into a single device buffer. Each separate H2D copy is a
        device synchronization on Neuron (profiler 2026-09-06: 14 copies ~ 280 ms/step of waiting)."""
        groups = [g for g in self.param_groups if g["params"]]
        if not groups:
            return
        dev = groups[0]["params"][0].device
        vals = []
        for g in groups:
            vals.append(float(g["lr"]))
            if g["kind"] == "muon":
                vals.append(float(g["momentum"]))
        buf = getattr(self, "_scalar_buf", None)
        if buf is None or buf.numel() != len(vals) or buf.device != dev:
            buf = torch.zeros(len(vals), dtype=torch.float32, device=dev)
            self._scalar_buf = buf
            k = 0
            for g in groups:
                g["lr_t"] = buf[k]; k += 1
                if g["kind"] == "muon":
                    g["momentum_t"] = buf[k]; k += 1
        buf.copy_(torch.tensor(vals, dtype=torch.float32))

    def _group_scalar(self, group, key):
        # Read ONLY the on-device scalar refreshed by refresh_scalars(). The python floats in the group dict
        # change every step; touching them inside a compiled step() makes dynamo guard on their values and
        # recompile each step (S3b 2026-09-05: 12 s/step). Eager fallback creates it from the float once.
        t = group.get(key)
        if t is None:  # eager fallback when refresh_scalars() was not called
            src = "lr" if key == "lr_t" else "momentum"
            t = self._dev_scalar(group, key, group[src], group["params"][0].device)
        return t

    @staticmethod
    def _dev_scalar(group, key, value, device):
        # Per-step scalars (lr, momentum) live on-device as 0-d tensors refreshed by a host->device copy,
        # never by a scalar-valued kernel: on Neuron a python float bakes into the NEFF as a constant, so
        # every new LR value forced a ~4 s recompile (whole warmup + the 60% warmdown; Trn2 S1d 2026-09-05).
        t = group.get(key)
        if t is None or t.device != device:
            t = torch.zeros((), dtype=torch.float32, device=device)
            group[key] = t
        t.copy_(torch.tensor(float(value), dtype=torch.float32))
        return t

    @torch.no_grad()
    def _adamw_step(self, group):
        global ADAMW_IMPL
        beta1, beta2 = group["betas"]
        lr_t = self._group_scalar(group, "lr_t")
        scal = None
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self._adamw_state(p)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            if (ADAMW_IMPL == "nki" and p.device.type == "neuron" and p.dim() == 2 and p.dtype == torch.float32
                    and p.shape[0] % 256 == 0 and not self.tensor_step):
                try:
                    kern, grid = build_nki_adamw()
                    if scal is None:   # per-group per-step scalars, one H2D copy (values never bake into the kernel)
                        b1c, b2c = 1 - beta1 ** (state["step"] + 1), 1 - beta2 ** (state["step"] + 1)
                        lr = float(group["lr"])
                        scal = torch.tensor([lr, 1.0 / b1c, 1.0 / (b2c ** 0.5), lr * group["weight_decay"]],
                                            dtype=torch.float32).expand(128, 4).contiguous().to(p.device)
                    outs = kern[grid](p, grad.contiguous(), exp_avg, exp_avg_sq, scal, beta1=beta1, beta2=beta2, eps=group["eps"])
                    if not NKI_ADAMW_INPLACE:
                        p.copy_(outs[0]); exp_avg.copy_(outs[1]); exp_avg_sq.copy_(outs[2])
                    state["step"] += 1
                    continue
                except Exception as exc:   # kernel unavailable on this build -> torch path for good (never a crash)
                    ADAMW_IMPL = "torch"
                    print0(f"NKI AdamW kernel disabled ({type(exc).__name__}: {str(exc)[:120]}); torch path")
            p.mul_(1 - lr_t * group["weight_decay"])
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            if self.tensor_step:
                state["step_t"] += 1
                bias1 = 1 - beta1 ** state["step_t"]
                bias2 = 1 - beta2 ** state["step_t"]
            else:
                state["step"] += 1
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
            update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group["eps"])
            p.sub_(lr_t * update)

    @torch.no_grad()
    def _muon_step(self, group):
        global MUON_IMPL
        params = group["params"]
        if not params or params[0].grad is None:
            return
        w = params[0]                       # (n, out, in) stack: updated in place, no per-layer copies
        shape = w.shape[1:]
        state = self._muon_state(group)
        g = w.grad
        momentum_buffer = state["momentum_buffer"]
        second = state["second_momentum_buffer"]
        if (MUON_IMPL == "fused" and w.device.type == "neuron" and min(shape) == _FMAX and max(shape) % _FMAX == 0
                and w.shape[0] % 2 == 0):
            try:
                kern, grid = build_nki_muon_fused()
                lr_eff = float(group["lr"]) * (max(1.0, shape[-2] / shape[-1]) ** 0.5)
                scal = torch.tensor([float(group["momentum"]), lr_eff, lr_eff * group["weight_decay"], 0.0],
                                    dtype=torch.float32).expand(128, 4).contiguous().to(w.device)
                outs = kern[grid](w, g.contiguous(), momentum_buffer, second, scal, ns_steps=group["ns_steps"], beta2=group["beta2"])
                if not NKI_ADAMW_INPLACE:
                    w.copy_(outs[0]); momentum_buffer.copy_(outs[1]); second.copy_(outs[2])
                return
            except Exception as exc:   # kernel unavailable on this build -> torch path for good (never a crash)
                MUON_IMPL = "torch"
                print0(f"NKI Muon kernel disabled ({type(exc).__name__}: {str(exc)[:120]}); torch path")

        mom_t = self._group_scalar(group, "momentum_t").to(momentum_buffer.dtype)
        momentum_buffer.lerp_(g, 1 - mom_t)
        g = g.lerp_(momentum_buffer, mom_t)
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        if MUON_IMPL == "nki" and x.device.type == "neuron" and x.dtype == torch.bfloat16 and x.shape[-1] % 512 == 0 \
                and x.shape[-2] % 512 == 0:
            try:
                x = build_nki_muon()(x, group["ns_steps"])
            except Exception as exc:
                MUON_IMPL = "torch"
                print0(f"NKI NS kernel disabled ({type(exc).__name__}); torch path")
                x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
                for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                    xtx = x.mT @ x
                    x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        elif x.shape[-2] > x.shape[-1]:
            x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xtx = x.mT @ x
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
            x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xxt = x @ x.mT
                x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
        g = x

        red_dim = -1 if shape[-2] >= shape[-1] else -2
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        second.lerp_(v_mean, 1 - group["beta2"])
        step_size = second.clamp_min(1e-10).rsqrt()
        scaled = (v_mean * red_dim_size) * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

        lr = self._group_scalar(group, "lr_t") * (max(1.0, shape[-2] / shape[-1]) ** 0.5)
        wd = group["weight_decay"]
        mask = (g * w) >= 0
        w.sub_(lr * g + lr * wd * w * mask)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._adamw_step(group)
            elif group["kind"] == "muon":
                self._muon_step(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")


def sync_gradients(model: nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    # Per-parameter all_reduce (T2-validated). A bucketed variant (flatten -> one all_reduce -> copy back)
    # was a wash on Trn2 (S2 2026-09-05: 0.56 vs 0.53 s/step) - the 208 MB flatten/copy-back costs what the
    # ~122 small collectives cost - so it was reverted.
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


# -----------------------------------------------------------------------------
# Checkpoint / scoring hook

def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model.config),
        "meta": meta,
    }
    torch.save(payload, path)


def torch_load_weights(path: str | Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)


def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    payload = torch_load_weights(checkpoint_path, map_location="cpu")
    config = GPTConfig(**payload["model_config"])
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    if ATTN_IMPL == "nki" and config.sink_mode == "logit" and device.type == "neuron":
        try:
            build_nki_ops()
        except Exception as exc:  # the scorer's forward then takes the explicit path (identical math)
            print0(f"NKI attention unavailable for eval ({type(exc).__name__}); using the explicit path")
    return model


# -----------------------------------------------------------------------------
# Schedules: warmup by steps, warmdown by elapsed TIME (Trn2 step count unknown).
# Elapsed is broadcast from rank 0 each step so all ranks apply identical LRs.

def lr_multiplier(step: int, elapsed_frac: float) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    warmdown_start = 1.0 - WARMDOWN_TIME_FRAC
    if elapsed_frac <= warmdown_start:
        return 1.0
    frac = min(1.0, (elapsed_frac - warmdown_start) / WARMDOWN_TIME_FRAC)
    return 1.0 - (1.0 - FINAL_LR_FRAC) * frac


def muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def synced_elapsed_frac(budget_started: float, max_seconds: float, device: torch.device) -> float:
    elapsed = (time.monotonic() - budget_started) / max(1.0, max_seconds)
    if dist.is_available() and dist.is_initialized():
        te = torch.tensor([elapsed], dtype=torch.float32, device=device)
        dist.broadcast(te, src=0)
        elapsed = float(te.item())
    return elapsed


# -----------------------------------------------------------------------------
# Training

def main() -> None:
    parser = argparse.ArgumentParser(description="Neuron Competition R1 submission (train2_best port)")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_NUM_STEPS", NUM_STEPS)))
    parser.add_argument("--max-train-seconds", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS", MAX_TRAIN_SECONDS)))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("NEURON_COMPETITION_R1_OUT_DIR", "out"))
    parser.add_argument("--eval-public", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_EVAL_PUBLIC", "1") != "0")
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_EVAL_TOKENS", str(4 * 524288))))
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_COMPILE", "1") != "0")
                        # default EAGER (v6, 2026-09-06): both portal submissions that died with
                        # `no_run_meta` (v2, v5) defaulted compile ON; the organizers' baseline is eager.
                        # torch.compile(backend="neuron") needed two backend-bug workarounds on our build and
                        # the portal runs a newer build. Compile stays available (env) and falls back to eager.
    parser.add_argument("--compile-only", action="store_true",
                        help="run one synthetic optimizer step to warm the compile cache, then exit")
    parser.add_argument("--compile-opt", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_COMPILE_OPT", "0") != "0",
                        help="dev knob: also torch.compile the optimizer step (Trn2 per-step overhead)")
    args = parser.parse_args()

    print(f"SUBMISSION_BOOT: main() entered (rank env={os.environ.get('RANK', 'none')})", flush=True)
    torch.set_float32_matmul_precision("high")  # CUDA TF32; harmless elsewhere. Without it
    # the fp32 pieces of the graph run ~20x slower on GPU dev boxes (found 2026-09-02).
    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE, DEVICE_BATCH_SIZE
    COMPUTE_DTYPE = compute_dtype_for(device)
    if DEVICE_BATCH_SIZE is None:
        # 32 // world_size rows per worker, capped at 8 so a single-process launch (nproc=1 is a
        # sanctioned shape) does not put 32 rows on one core; grad accumulation covers the rest.
        DEVICE_BATCH_SIZE = max(PACK, min(8, (TOTAL_BATCH_SIZE // SEQ_LEN) // world_size))
    assert DEVICE_BATCH_SIZE % PACK == 0, f"DEVICE_BATCH_SIZE={DEVICE_BATCH_SIZE} must be a multiple of PACK={PACK}"

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = build_config()
    print0(f"model_config={json.dumps(asdict(config))}")
    model = GPT(config)
    model.init_weights()
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"num_params={num_params:,}")

    if ATTN_IMPL == "nki" and SINK_MODE == "logit" and device.type == "neuron":
        try:
            build_nki_ops()   # custom ops must exist before torch.compile traces the model
        except Exception as exc:  # unknown kernel API on the scoring box -> explicit path (same math), never a crash
            print0(f"NKI attention unavailable ({type(exc).__name__}: {str(exc)[:120]}); using the explicit path")
    orig_model = model
    global STACK_GRAD_MODE
    STACK_GRAD_MODE = "view" if args.compile else "function"   # compile fuses the view path; eager needs the Function
    compile_backend = os.environ.get("NEURON_COMPETITION_R1_COMPILE_BACKEND",
                                     "neuron" if device_type == "neuron" else "inductor")
    if args.compile:
        try:
            model = torch.compile(model, backend=compile_backend, dynamic=False)
            print0(f"torch.compile enabled (backend={compile_backend})")
        except Exception as exc:  # unknown backend etc. -> eager, never a crash
            print0(f"torch.compile unavailable ({type(exc).__name__}: {str(exc)[:160]}); running eager")
            model = orig_model

    world_tokens_per_micro = DEVICE_BATCH_SIZE * SEQ_LEN * world_size
    if TOTAL_BATCH_SIZE % world_tokens_per_micro != 0:
        raise ValueError(f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={world_tokens_per_micro}")
    grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_micro
    print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps}")

    if args.compile_only:
        synthetic_x = torch.zeros((DEVICE_BATCH_SIZE, SEQ_LEN), dtype=torch.int32).to(device)
        synthetic_y = torch.zeros((DEVICE_BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(device)
        optimizer = model.setup_optimizer()
        model.zero_grad(set_to_none=True)
        try:
            for _ in range(grad_accum_steps):
                loss = model(synthetic_x, synthetic_y)
                (loss / grad_accum_steps).backward()
        except Exception as exc:
            if model is orig_model:
                raise
            print0(f"compile-only: compiled path failed ({type(exc).__name__}); warming eager instead")
            model = orig_model
            model.zero_grad(set_to_none=True)
            for _ in range(grad_accum_steps):
                loss = model(synthetic_x, synthetic_y)
                (loss / grad_accum_steps).backward()
        sync_gradients(model)
        optimizer.refresh_scalars()
        optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    # organizer-sanctioned packing: fetch PACK-times-long rows, reshape to SEQ_LEN chunks
    _raw_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE // PACK, SEQ_LEN * PACK, "train", device)

    def _packed():
        for _x, _y, _st in _raw_loader:
            yield _x.reshape(-1, SEQ_LEN), _y.reshape(-1, SEQ_LEN), _st

    train_loader = _packed()
    x, y, loader_state = next(train_loader)
    optimizer = model.setup_optimizer()
    optimizer.init_state()
    if args.compile_opt:
        # dev knob: the optimizer is ~300 eager ops/step on Trn2 (Muon Polar-Express loops, AdamW per-param);
        # compiling step() turns it into a few graphs. Scalars are refreshed eagerly before each call.
        optimizer.tensor_step = True
        opt_step = torch.compile(optimizer.step, backend=compile_backend, dynamic=False)
        print0(f"torch.compile enabled for optimizer.step (backend={compile_backend})")
    else:
        opt_step = optimizer.step
    loop_started = time.monotonic()
    budget_started = loop_started

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    prof = {"fwd_bwd": 0.0, "grad_sync": 0.0, "optimizer": 0.0, "total": 0.0, "n": 0}
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0

    while step < args.num_steps:
        synchronize(device)
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        for _ in range(grad_accum_steps):
            try:
                loss = model(x, y)
                last_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            except Exception as exc:
                if step != 0 or model is orig_model:
                    raise
                # The compiled path died on its first graph (backend bug on this build): degrade to
                # eager instead of crashing the submission. Nothing has been stepped yet.
                print0(f"torch.compile path failed at step 0 ({type(exc).__name__}: {str(exc)[:200]}); "
                       "falling back to eager")
                model = orig_model
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                loss = model(x, y)
                last_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            x, y, loader_state = next(train_loader)
        if PROFILE and 10 <= step < 40:
            synchronize(device); t_fb = time.monotonic()

        sync_gradients(model)
        if PROFILE and 10 <= step < 40:
            synchronize(device); t_gs = time.monotonic()

        elapsed_frac = synced_elapsed_frac(budget_started, args.max_train_seconds, device)
        lrm = lr_multiplier(step, elapsed_frac)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step)
                group["weight_decay"] = WEIGHT_DECAY
        optimizer.refresh_scalars()
        if WINDOW_WARMUP > 0:
            orig_model.set_windows(min(1.0, elapsed_frac / WINDOW_WARMUP))
        opt_step()

        synchronize(device)
        record_step_boundary(step)
        dt = time.monotonic() - t0
        if PROFILE and 10 <= step < 40:
            prof["fwd_bwd"] += t_fb - t0; prof["grad_sync"] += t_gs - t_fb
            prof["optimizer"] += time.monotonic() - t_gs; prof["total"] += dt; prof["n"] += 1
            if step == 39:
                n = prof["n"]
                print0("PROFILE (avg s/step over 30 steps): " + " ".join(f"{k}={v/n:.3f}" for k, v in prof.items() if k != "n")
                       + f" other={(prof['total']-prof['fwd_bwd']-prof['grad_sync']-prof['optimizer'])/n:.3f}")
        recent_step_times.append(dt)
        train_loss = float(last_loss.item())
        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
            # The runner excludes launch through this boundary (capped), so
            # restart the budget clock here to use the full charged budget.
            # If startup exceeded the cap, the excess is already charged:
            # back-date the clock by that excess, with a margin for the
            # launch -> import gap this process cannot observe.
            now = time.monotonic()
            observed_startup = now - _PROCESS_STARTED
            startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            # The predictive stop below uses max(last 20 step times); drop the startup steps (step 0 carries
            # the whole cold compile, 500-650 s at 512-wide) or a short budget stops right here (S_H8, 2026-09-06).
            # Keep this step's own time so the window is never empty (S_BASE crash, 2026-09-06).
            recent_step_times[:] = [dt]
            excess = max(
                0.0,
                observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
            )
            budget_started = now - excess
        elapsed_after_step = time.monotonic() - budget_started
        # Stop decision rides the same collective as the health check so every
        # rank breaks on the same step (a local wall-clock check can desync
        # ranks). Time-based stops are gated behind the excluded-step count:
        # the run must always report at least TRAIN_STARTUP_STEPS_EXCLUDED
        # boundaries or it is not scorable.
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        stop_after_step = (
            step + 1 >= args.num_steps
            or (
                reached_min_steps
                and (
                    elapsed_after_step >= args.max_train_seconds
                    or elapsed_after_step
                    + max(recent_step_times[-20:])
                    + CHECKPOINT_RESERVE_SECONDS
                    >= args.max_train_seconds
                )
            )
        )
        control = torch.tensor(
            [
                1 if (math.isnan(train_loss) or train_loss > 100) else 0,
                1 if stop_after_step else 0,
            ],
            device=device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        if int(control[0].item()) != 0:
            raise RuntimeError(f"bad loss detected: {train_loss}")

        total_train_tokens += TOTAL_BATCH_SIZE
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        if rank == 0 and (step < 5 or step % 10 == 0):
            remaining_time = max(0.0, args.max_train_seconds - (time.monotonic() - budget_started))
            print(
                f"step {step:05d} | loss {debiased:.4f} | lrm {lrm:.3f} | dt {dt:.2f}s | "
                f"tok/s {TOTAL_BATCH_SIZE / max(dt, 1e-6):,.0f} | remaining_time {remaining_time:.0f}s | "
                f"epoch {loader_state['epoch']}",
                flush=True,
            )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1
        if int(control[1].item()) != 0:
            break

    checkpoint_path = Path(args.out_dir) / "final.pt"
    budget_elapsed_before_save = time.monotonic() - budget_started
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, {
            "step": step,
            "total_train_tokens": total_train_tokens,
            "budget_elapsed_before_save": budget_elapsed_before_save,
            "world_size": world_size,
            "compiled": args.compile,
        })
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started

    public_eval = None
    causal = None
    if args.eval_public:
        causal = causality_check(orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device)
        public_eval = evaluate_bpb(
            orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device,
            split="public_val", max_tokens=args.eval_tokens, timeout_seconds=EVAL_TIMEOUT_SECONDS,
        )

    if rank == 0:
        print("---")
        print(f"checkpoint_path:    {checkpoint_path}")
        # The budget clock restarts at the excluded-startup boundary, so
        # post-restart elapsed IS the charged estimate (conservative: the
        # runner also excludes pre-loop startup).
        print(f"training_seconds:  {budget_elapsed_before_save:.1f}")
        print(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}")
        print(f"startup_allowance_est: {startup_allowance_est:.1f}")
        print(f"charged_seconds_est:   {budget_elapsed_after_save:.1f}")
        print(f"over_budget:       {budget_elapsed_after_save > TRAIN_TIME_BUDGET_SECONDS}")
        print(f"total_seconds:     {time.monotonic() - loop_started:.1f}")
        print(f"num_steps:         {step}")
        print(f"total_tokens_M:    {total_train_tokens / 1e6:.1f}")
        print(f"world_size:        {world_size}")
        print(f"num_params_M:      {num_params / 1e6:.1f}")
        if causal is not None:
            print(f"causality_passed:  {causal['passed']}")
            print(f"causality_max_abs: {causal['max_abs_diff']:.6f}")
        if public_eval is not None:
            print(f"public_val_bpb:    {public_eval['val_bpb']:.6f}")
            print(f"eval_seconds:      {public_eval['eval_seconds']:.1f}")

    cleanup_runtime()


if __name__ == "__main__":
    main()
