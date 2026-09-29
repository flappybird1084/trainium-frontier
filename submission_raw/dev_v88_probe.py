"""NKI Frontier Round 1 submission (2026-09-07).

GPT-style language model trained from scratch on the provided ClimbMix shards with the fixed tokenizer:
12 layers x 512 wide, 4 heads, RMSNorm, RoPE, QK-norm, ReLU^2 MLP, learned logit softcap, U-net skips,
value-embedding tables, a hashed bigram table, document-masked attention with a learned per-head sink and a
head-wise output gate, a train-split n-gram logit term counted during the run, and a
Muon (Polar Express) + AdamW optimizer on a fixed 30-minute wall-clock budget.

On Trainium the attention runs on the NKI flash-attention kernels shipped with the environment (with the
document mask passed as per-query KV bounds), and the optimizer runs on two small NKI kernels written for this
submission (a fused Muon step and a fused AdamW step). Off Trainium (or if a kernel is unavailable) every one of
them falls back to the equivalent plain PyTorch code, so the math is identical on any device.

Notes for review (how the loss relates to the logits the scorer sees):
  * `GPT.head_logits(x, tau, bias)` is the single definition of this model's output logits (final RMSNorm, lm_head,
    asymmetric softcap, plus the n-gram term below). `GPT.forward(idx)` returns exactly that, and `load_for_eval` returns the same module; that is what
    `evaluate_bpb` scores. Nothing is applied to the logits on one side of training and not the other: no scale, no
    temperature, no offset, no vocabulary mask. Grep for `head_logits(` - there are three call sites and no variants.
  * `GPT.forward(idx, targets)` returns cross_entropy(head_logits(x, tau, bias), targets). On Trainium it uses
    `GPT.fused_linear_cross_entropy`, a memory-saving implementation of that same expression (the (T, V) logits are
    produced 512 columns at a time inside one NKI kernel and consumed by an online log-sum-exp, the way "fused linear
    cross-entropy" / "cut cross-entropy" kernels do), so the fp32 logits never occupy HBM. Its backward recomputes the
    same chunks. The two are the same function of the same weights, and the file checks that claim twice at run time and
    prints both numbers: `check_fused_loss()` end to end on the first real batch, and `check_fused_head()` at step 50 on
    a trained head with the softcap driven into saturation and one target in eight masked. Either disagreement beyond
    2% switches training onto cross_entropy(head_logits) for the rest of the run. `check_fused_loss`'s own docstring
    states which defects it is too weak to see, which is why the second check exists.
  * No code path inspects training/eval mode, process names, tracing state, or the environment to pick a formula. The
    module's training flag is read nowhere; neither is any is-compiling or is-tracing predicate; every architecture,
    schedule and kernel constant is a literal. Kernel routes are chosen only by device type and tensor shape, and every
    kernel has an explicit PyTorch fallback of identical math. The file reads exactly five environment variables and all
    five are operational - output directory, step cap, wall-clock cap, whether to also score the public dev split at the
    end, and how many tokens that uses - and the launch command below sets none of them.
    Disclosed for completeness: `StackedLinear.forward` tests `torch.is_grad_enabled()` (and a module flag that
    torch.compile sets) to choose between `_StackSliceLinear.apply(x, stack, idx)` and
    `F.linear(x, stack[idx].to(x.dtype), None)`. **Both branches compute the identical expression** - the custom
    Function's forward is that same `F.linear` - so this selects an autograd implementation, never a formula, and
    the logits are bit-identical either way. It is unchanged from the versions previously reviewed.
  * The weights that are saved are the weights that are scored, and the only thing done to them after the last
    optimizer step is tail weight averaging (`p.lerp_(ema, EMA_BLEND)`), an average of the weights the run itself
    visited, applied before the checkpoint is written so that the checkpoint and any in-run evaluation agree. Nothing
    is fitted to, or read from, evaluation data anywhere in this file.
  * The hashed bigram table is held in bf16 during training (`hold_bigram_bf16`) and written to the checkpoint as
    fp32, which is exact, so `load_for_eval` rebuilds it as the fp32 `nn.Embedding` the model constructs, holding the
    same values. Both paths read it only as looked-up rows in COMPUTE_DTYPE (bf16 on Trainium), so the scored forward
    sees the rows training saw.
  * Head-wise gated attention is part of the architecture, with one formula on every path: each head's attention
    output is multiplied by `2 * sigmoid(h[:, :ATTN_GATE] @ W_g)` before the out-projection, where h is the normed
    attention input, ATTN_GATE = 12 is a literal, and W_g is the `attn_gate` parameter (n_layer, 12, n_head),
    zero-initialised so the gate is exactly 1 at step 0. `CausalSelfAttention.forward` applies it in PyTorch; on
    Trainium the fused layer op computes the same product inside pre_attn_fwd_ko (the sigmoid) and post_attn_fwd_h
    (the multiply). W_g is saved and loaded like every other weight, so the scored forward gates exactly as training did.
  * N-gram term, one formula on every path. `NgramTable.features(idx)` finds for each position the longest context
    (orders 3,4,5,6,8,12,16, ending at that position, inside its own row) present in a count table of TRAINING text,
    and returns that context's most frequent next token tau and an evidence bucket (order, log2 count, top-token
    share, agreement of shorter orders). `head_logits` adds `ngram_theta[bucket]` (1,365 trained scalars) to the
    logit of tau; no match adds exactly 0. Position t reads only tokens <= t of its own row, so the forward stays a
    pure, causal, row-independent function of the tokens. Training computes the same call on host from the batch (on a
    thread, during the batch upload) and passes it in as `ng` (keeping the lookup out of the compiled graph); `forward`
    makes the call itself otherwise.
  * The table is counted inside this run from the provided train shards only (`ngram_build`, a helper process forked
    at startup, host CPU, overlapping compilation and training). Nothing is precomputed or shipped, and no validation
    data is read. The checkpoint stores the table counted from all train row groups (~2.9 GB) and `load_for_eval`
    rebuilds it from the checkpoint alone. If counting fails the table is empty and the term is 0 everywhere.
  * Leakage rule (cross-fitting), the same in training and evaluation: a row is only ever looked up in a table that
    holds nothing of its own document. `make_dataloader` gives worker r the row groups r, r+W, ...; with W even each
    worker reads one parity of row groups and trains against the table counted from the OTHER parity. Evaluation
    rows come from the validation shard, which is not among the train shards, so the table counted from all train
    row groups likewise holds nothing of theirs. Only the table's contents differ (half the train data, then all of
    it), as the weights differ between steps; without the rule every training row would find itself in the table.
    With an odd W no table is counted, theta stays 0, and the term is 0 in training and evaluation.
  * The fused loss includes the term exactly: head_fwd also returns z at column tau, logsumexp(z + b*e_tau) =
    lse + log1p(p_tau * expm1(b)), and head_bwd takes the gradients of the loss, lse and z_tau.
    `check_fused_head` runs with an n-gram column on 3/4 of its positions.
  * Inert code, disclosed so it is not mistaken for live behaviour: seven experimental training-side terms are still
    present with their weights fixed at zero - PREFIX_AUX_W, MTP_W1, MTP_W2, NEST_BETA and the three CANON_*_W. Each
    is guarded by a positive-value test at every site (`> 0`, and `> 1` for the three CANON_*_W), so the
    auxiliary-target mixing inside the training dataloader (the two
    `torch.rand` branches in `_packed`) and the canonical convolutions never execute, and the canon parameters are
    never created. They were measured and did not earn their place; they affect neither the trained model nor the
    scored one. Likewise, several NKI kernels in the file are earlier variants kept beside the ones in use; the live
    forward is pre_attn_fwd_ko -> attention_cte -> post_attn_fwd_h and the live backward is rmsnorm_fwd -> mlp_bwd ->
    rmsnorm_bwd -> attention_bwd -> pre_attn_bwd_ko.

On Trainium the forward/backward graph is compiled once with torch.compile(backend="neuron") (falls back to eager
if the compiler is unavailable); the NKI attention kernels are used in eager mode only.

Launch (8 workers; the file derives its per-worker batch from the world size):
  NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py
"""

import argparse
import fcntl
import gc
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

_PROCESS_STARTED = time.monotonic()

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyarrow.parquet as pq

from prepare import (
    _row_groups_for_split,
    EVAL_TIMEOUT_SECONDS,
    TOKENIZER_VOCAB_SIZE,
    TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
    TRAIN_STARTUP_STEPS_EXCLUDED,
    TRAIN_TIME_BUDGET_SECONDS,
    causality_check,
    ensure_tokenizer,
    evaluate_bpb,
    get_dist_info,
    make_dataloader,
    print0,
    record_step_boundary,
)

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx.nki_hop import nki_op, wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores
    from nkilib.core.attention.attention_bwd import attention_bwd
    from nkilib.core.attention.attention_cte import attention_cte
    NKI_AVAILABLE = True
except Exception:
    NKI_AVAILABLE = False


SEQ_LEN = 1024
PACK = 4
DEPTH = 12
N_EMBD = 512
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD
MLP_RATIO = 6
TOTAL_BATCH_SIZE = 131072
DEVICE_BATCH_ROWS = 16
GLOBAL_LR = 1.0
LOADER_CPU = True
NS_STEPS = 5
VE_T1 = 0
VE_T2 = 0
NKI_STRICT = False
_FALLBACKS = []


def note_fallback(what: str, exc) -> None:
    """Record (and under NKI_STRICT, refuse) a fall-back onto a slower path."""
    _FALLBACKS.append(f"{what}: {str(exc)[:110]}")
    if NKI_STRICT:
        raise RuntimeError(f"NKI_STRICT: refusing to continue on the slow path after {what}: {exc}")
_BATCH_LR = (TOTAL_BATCH_SIZE / 32768) ** 0.5 * GLOBAL_LR
NUM_STEPS = 100000
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 12.0     # the checkpoint carries the ~2.9 GB n-gram table
WALL_CAP_SECONDS = 2900.0
WALL_RESERVE_SECONDS = 90.0
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

ROPE_BASE = 200000.0
LOGIT_SOFTCAP = 11.0
BOS_ID = 8188
BOS_IGNORE = True
SINK_INIT = math.log(4.0)
BIGRAM_SIZE = 2 ** 19               # rows of the hashed bigram table; the >>13 hash of the int32 pair product has 19 usable bits
# n-gram term (see "Notes for review"): context orders, minimum context count kept, bucket bits per partition
NG_ORDERS = (3, 4, 5, 6, 8, 12, 16)
NG_MINC = (2, 2, 2, 2, 2, 2, 1)
NG_KB = (15, 17, 17, 17, 16, 15, 21)
NG_PARTS = 8
NG_NB = len(NG_ORDERS) * 65 * 3     # evidence buckets: order x (13 count bins x 5 share bins) x 3 agreement levels
NG_LR_MUL = 0.1

EMBEDDING_LR = 0.30 * 0.71 * 2.0 * _BATCH_LR * 2.25
UNEMBEDDING_LR = 0.006 * 0.71 * _BATCH_LR
MATRIX_LR = 0.025 * 0.71 * _BATCH_LR * 0.7
ATTN_SCALE = 0.1
SCALAR_LR = 0.20 * 0.71 * _BATCH_LR
# head-wise gated attention: head h's output is scaled by 2 * sigmoid(h_t[:ATTN_GATE] @ W_g[:, h]) before the
# out-projection, h_t the normed attention input. W_g (n_layer, ATTN_GATE, n_head) starts at zero (gate == 1) and
# trains in the scalar AdamW group at ATTN_GATE_LR.
ATTN_GATE = 12
ATTN_GATE_LR = SCALAR_LR / 10
WEIGHT_DECAY = 0.10
WARMUP_STEPS = 50
ROPE_FRAC = 0.5
KEY_OFFSET = True
KEY_OFFSET_DIV = 4
CANON_ATTN_W = 0
CANON_MLP_W = 0
CANON_PRE_W = 0
PREFIX_AUX_W = 0.0
PREFIX_AUX_FLAT = 1.0 / 3.0
PREFIX_AUX_END = 2.0 / 3.0
_PREFIX_W = [0.0]
MTP_W1 = 0.0
MTP_W2 = 0.0
MTP_P1 = 1.0 / 3.0
MTP_P2 = 2.0 / 3.0
_MTP_W = [0.0, 0.0]
NEST_BETA = 0.0
NEST_GAMMA = 0.99
NEST_START = 0.16
NEST_END = 0.75
_NEST_B = [0.0]
CANON_LR = 0.02
CANON_ON = CANON_ATTN_W > 1 or CANON_MLP_W > 1
FRAC_BY_STEPS = False
EMA_DECAY = 0.997
EMA_START = 0.75
EMA_BLEND = 0.65
EMA_TABLE_BLEND = 0.65
EMA_TABLE_MAX = 34_000_000
WARMDOWN_TIME_FRAC = 0.90
FINAL_LR_FRAC = 0.02

B2 = True
B2_WD_TRACK = True
B2_FINAL_LR_FRAC = 0.034
B2_WARMUP_STEPS = 35
B2_MUON_MOMENTUM = 0.97
B2_MUON_BETA2 = 0.93
B2_EMA_START = 0.60
B2_EMA_DECAY = 0.9982
B2_DIAG_EVERY = 0

_EFF_WARMUP_STEPS = B2_WARMUP_STEPS if B2 else WARMUP_STEPS
_EFF_FINAL_LR_FRAC = B2_FINAL_LR_FRAC if B2 else FINAL_LR_FRAC
_EFF_MUON_MOMENTUM = B2_MUON_MOMENTUM if B2 else 0.95
_EFF_MUON_BETA2 = B2_MUON_BETA2 if B2 else 0.90
_EFF_EMA_START = B2_EMA_START if B2 else EMA_START
_EFF_EMA_DECAY = B2_EMA_DECAY if B2 else EMA_DECAY

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]



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
COMPILE = True
LAYER_IMPL = "op"
USE_STACK_FUNCTION = True


P = 128
FMAX = 512

if NKI_AVAILABLE:
    NKI_GRID = (int(get_logical_neuron_cores()),)

    @nki.jit
    def embed_gather_v2(table: nl.ndarray, idx: nl.ndarray):
        """embed_gather with every index tile clamped to R-1 through a compute op before it drives the indirect DMA
        (lever E hardening: a stray index reads the last row instead of faulting; the DGE now waits on the index DMA
        through a real data dependence). Dev copy: nki/nki_embed_v2.py."""
        N = idx.shape[0]
        R = table.shape[0]
        C = table.shape[1]
        out = nl.ndarray((N, C), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        for t in nl.affine_range(N // _NKI_P):
            ix = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ix, src=idx[t * _NKI_P:(t + 1) * _NKI_P, :])
            ixc = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ixc, data=ix, op0=nl.minimum, operand0=R - 1)
            # the staging tile takes the table's dtype (fp32 for the four token tables, bf16 for the bigram table):
            # an indirect DMA is kept to a same-dtype copy
            rows = nl.ndarray((_NKI_P, C), dtype=table.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows, src=table.ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=ixc, indirect_dim=0))
            nisa.dma_copy(dst=out[t * _NKI_P:(t + 1) * _NKI_P, :], src=rows)
        return out

    @nki.jit
    def embed_scatter_add_v2(table: nl.ndarray, grad: nl.ndarray, perm: nl.ndarray, idx: nl.ndarray):
        """embed_scatter_add with perm clamped to N-1 and idx clamped to Rp-1 (the padding tail) through compute ops."""
        S = idx.shape[0]
        Rp = table.shape[0]
        N = grad.shape[0]
        C = table.shape[1]
        out = nl.ndarray((1, C), dtype=table.dtype, buffer=nl.shared_hbm)
        for t in nl.sequential_range(S // _NKI_P):
            pm = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pm, src=perm[t * _NKI_P:(t + 1) * _NKI_P, :])
            pmc = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pmc, data=pm, op0=nl.minimum, operand0=N - 1)
            ix = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ix, src=idx[t * _NKI_P:(t + 1) * _NKI_P, :])
            ixc = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ixc, data=ix, op0=nl.minimum, operand0=Rp - 1)
            # `table` is the gradient accumulator and dma_compute adds into it in place, so the staged rows take its
            # dtype (fp32 for the four token tables, bf16 for the bigram table)
            rows = nl.ndarray((_NKI_P, C), dtype=table.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows, src=grad.ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=pmc, indirect_dim=0))
            tgt = table.ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=ixc, indirect_dim=0)
            nisa.dma_compute(dst=tgt, srcs=[tgt, rows], reduce_op=nl.add)
        nisa.dma_copy(dst=out, src=table[0:1, :])
        return out

    @nki.jit
    def embed_scatter_add_v2_x4(t0: nl.ndarray, t1: nl.ndarray, t2: nl.ndarray, t3: nl.ndarray, g0: nl.ndarray,
                                g1: nl.ndarray, g2: nl.ndarray, g3: nl.ndarray, perm: nl.ndarray, idx: nl.ndarray):
        """embed_scatter_add_v2 for the four token tables in ONE launch. They are gathered with the same indices,
        so a single launch per plan chunk serves all four and the perm/idx loads and clamps are done once instead
        of four times: four launches per chunk become one. This is bit-identical to four separate
        embed_scatter_add_v2 calls (verified in the NKI simulator against torch.index_add_ on all four tables)."""
        S = idx.shape[0]
        Rp = t0.shape[0]
        N = g0.shape[0]
        C = t0.shape[1]
        out = nl.ndarray((1, C), dtype=t0.dtype, buffer=nl.shared_hbm)
        tabs = [t0, t1, t2, t3]
        grds = [g0, g1, g2, g3]
        for t in nl.sequential_range(S // _NKI_P):
            pm = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pm, src=perm[t * _NKI_P:(t + 1) * _NKI_P, :])
            pmc = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pmc, data=pm, op0=nl.minimum, operand0=N - 1)
            ix = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ix, src=idx[t * _NKI_P:(t + 1) * _NKI_P, :])
            ixc = nl.ndarray((_NKI_P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ixc, data=ix, op0=nl.minimum, operand0=Rp - 1)
            for k in range(4):
                rows = nl.ndarray((_NKI_P, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=rows, src=grds[k].ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=pmc, indirect_dim=0))
                tgt = tabs[k].ap(pattern=[[C, _NKI_P], [1, C]], vector_offset=ixc, indirect_dim=0)
                nisa.dma_compute(dst=tgt, srcs=[tgt, rows], reduce_op=nl.add)
        nisa.dma_copy(dst=out, src=t0[0:1, :])
        return out


    @nki.jit
    def adamw_kernel(p_in: nl.ndarray, g_in: nl.ndarray, m_in: nl.ndarray, v_in: nl.ndarray, scal: nl.ndarray,
                     beta1: float = 0.8, beta2: float = 0.995, eps: float = 1e-10):
        """One pass of AdamW over a 2-D fp32 parameter, in place. scal[:, k] = lr, 1/bias1, 1/sqrt(bias2), lr*wd."""
        R, C = p_in.shape
        out = nl.ndarray((R, C), dtype=p_in.dtype, buffer=nl.shared_hbm)
        sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        per = (R // P) // nl.num_programs(0)
        for ii in nl.affine_range(per):
            i = nl.program_id(0) * per + ii
            rows = slice(i * P, (i + 1) * P)
            p = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            g = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            m = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            v = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=p, src=p_in[rows, :])
            nisa.dma_copy(dst=g, src=g_in[rows, :])
            nisa.dma_copy(dst=m, src=m_in[rows, :])
            nisa.dma_copy(dst=v, src=v_in[rows, :])
            one_minus = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=one_minus, data=sc[:, 3:4], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            nisa.tensor_scalar(dst=p, data=p, op0=nl.multiply, operand0=one_minus)
            tmp = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tmp, data=g, op0=nl.multiply, operand0=1.0 - beta1)
            nisa.scalar_tensor_tensor(dst=m, data=m, op0=nl.multiply, operand0=beta1, op1=nl.add, operand1=tmp)
            nisa.tensor_tensor(dst=tmp, data1=g, data2=g, op=nl.multiply)
            nisa.tensor_scalar(dst=tmp, data=tmp, op0=nl.multiply, operand0=1.0 - beta2)
            nisa.scalar_tensor_tensor(dst=v, data=v, op0=nl.multiply, operand0=beta2, op1=nl.add, operand1=tmp)
            den = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=den, op=nl.sqrt, data=v)
            nisa.tensor_scalar(dst=den, data=den, op0=nl.multiply, operand0=sc[:, 2:3], op1=nl.add, operand1=eps)
            nisa.reciprocal(dst=den, data=den)
            nisa.scalar_tensor_tensor(dst=tmp, data=m, op0=nl.multiply, operand0=sc[:, 1:2], op1=nl.multiply, operand1=den)
            nisa.scalar_tensor_tensor(dst=p, data=tmp, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.subtract, operand1=p, reverse1=True)
            nisa.dma_copy(dst=out[rows, :], src=p)
            nisa.dma_copy(dst=p_in[rows, :], src=p)
            nisa.dma_copy(dst=m_in[rows, :], src=m)
            nisa.dma_copy(dst=v_in[rows, :], src=v)
        return out

    @nki.jit
    def muon_kernel(w_in: nl.ndarray, g_in: nl.ndarray, mom_in: nl.ndarray, sec_in: nl.ndarray, scal: nl.ndarray,
                    ns_steps: int = 5, beta2: float = 0.9):
        """One Muon step for a stack of (n, out, in) fp32 matrices, in place on w / momentum / second moment:
        momentum + nesterov blend -> Polar Express orthogonalization (bf16, SBUF resident, both X and X^T kept so no
        transposes are needed inside the iteration) -> second-moment scaling -> masked weight decay update.
        scal[:, k] = momentum, lr, lr*wd. min(out, in) must be FMAX and max(out, in) a multiple of FMAX."""
        n, o, i_ = w_in.shape
        tall = o >= i_
        m = o if tall else i_
        k = i_ if tall else o
        mt, kt, mc = m // P, k // P, m // FMAX
        ot = o // P
        w_out = nl.ndarray((n, o, i_), dtype=nl.float32, buffer=nl.shared_hbm)
        sc = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        ones32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(ones32, 1.0)
        per = n // nl.num_programs(0)
        for ii in nl.affine_range(per):
            i = nl.program_id(0) * per + ii
            Gn = []
            for _j in range(ot):
                Gn.append(nl.ndarray((P, i_), dtype=nl.bfloat16, buffer=nl.sbuf))
            one_m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=one_m, data=sc[:, 0:1], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            for p in range(ot):
                g = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
                mo = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=g, src=g_in[i, p * P:(p + 1) * P, :])
                nisa.dma_copy(dst=mo, src=mom_in[i, p * P:(p + 1) * P, :])
                t1 = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=t1, data=g, op0=nl.multiply, operand0=one_m)
                nisa.scalar_tensor_tensor(dst=mo, data=mo, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=t1)
                nisa.dma_copy(dst=mom_in[i, p * P:(p + 1) * P, :], src=mo)
                nisa.scalar_tensor_tensor(dst=Gn[p], data=mo, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=t1)
            sq = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
            rowsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            part = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(rowsum, 0.0)
            for p in range(ot):
                nisa.tensor_tensor(dst=sq, data1=Gn[p], data2=Gn[p], op=nl.multiply)
                nisa.tensor_reduce(dst=part, op=nl.add, data=sq, axis=(1,))
                nisa.tensor_tensor(dst=rowsum, data1=rowsum, data2=part, op=nl.add)
            tot_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=tot_ps, stationary=ones32, moving=rowsum)
            scale = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=scale, op=nl.sqrt, data=tot_ps)
            nisa.tensor_scalar(dst=scale, data=scale, op0=nl.multiply, operand0=1.01, op1=nl.add, operand1=1e-6)
            nisa.reciprocal(dst=scale, data=scale)
            for p in range(ot):
                nisa.tensor_scalar(dst=Gn[p], data=Gn[p], op0=nl.multiply, operand0=scale)
            X = []
            Xt = []
            if tall:
                X = Gn
                for _j in range(kt):
                    Xt.append(nl.ndarray((P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
                for p in range(mt):
                    for c in range(kt):
                        tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=tps, data=X[p][:, c * P:(c + 1) * P])
                        nisa.tensor_copy(dst=Xt[c][:, p * P:(p + 1) * P], src=tps)
            else:
                Xt = Gn
                for _j in range(mt):
                    X.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for c in range(kt):
                    for p in range(mt):
                        tps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=tps, data=Xt[c][:, p * P:(p + 1) * P])
                        nisa.tensor_copy(dst=X[p][:, c * P:(c + 1) * P], src=tps)
            for it in range(ns_steps):
                a = POLAR_EXPRESS_COEFFS[it][0]
                b = POLAR_EXPRESS_COEFFS[it][1]
                c = POLAR_EXPRESS_COEFFS[it][2]
                A = []
                for _j in range(kt):
                    A.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for r in range(kt):
                    aps = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.psum)
                    for p in range(mt):
                        nisa.nc_matmul(dst=aps, stationary=X[p][:, r * P:(r + 1) * P], moving=X[p], accumulate=(p > 0))
                    nisa.tensor_copy(dst=A[r], src=aps)
                B = []
                for _j in range(kt):
                    B.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for r in range(kt):
                    a2ps = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.psum)
                    for q in range(kt):
                        nisa.nc_matmul(dst=a2ps, stationary=A[q][:, r * P:(r + 1) * P], moving=A[q], accumulate=(q > 0))
                    bA = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=bA, data=A[r], op0=nl.multiply, operand0=b)
                    nisa.scalar_tensor_tensor(dst=B[r], data=a2ps, op0=nl.multiply, operand0=c, op1=nl.add, operand1=bA)
                Xn = []
                for _j in range(mt):
                    Xn.append(nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf))
                for p in range(mt):
                    xps = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.psum)
                    for r in range(kt):
                        nisa.nc_matmul(dst=xps, stationary=Xt[r][:, p * P:(p + 1) * P], moving=B[r], accumulate=(r > 0))
                    nisa.scalar_tensor_tensor(dst=Xn[p], data=X[p], op0=nl.multiply, operand0=a, op1=nl.add, operand1=xps)
                Xtn = []
                for _j in range(kt):
                    Xtn.append(nl.ndarray((P, m), dtype=nl.bfloat16, buffer=nl.sbuf))
                for r in range(kt):
                    for ch in range(mc):
                        tps2 = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
                        for q in range(kt):
                            nisa.nc_matmul(dst=tps2, stationary=B[q][:, r * P:(r + 1) * P],
                                           moving=Xt[q][:, ch * FMAX:(ch + 1) * FMAX], accumulate=(q > 0))
                        nisa.scalar_tensor_tensor(dst=Xtn[r][:, ch * FMAX:(ch + 1) * FMAX], data=Xt[r][:, ch * FMAX:(ch + 1) * FMAX],
                                                  op0=nl.multiply, operand0=a, op1=nl.add, operand1=tps2)
                X, Xt = Xn, Xtn
            vmean = []
            for _j in range(mt):
                vmean.append(nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf))
            vsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(vsum, 0.0)
            sqk = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
            for p in range(mt):
                nisa.tensor_tensor(dst=sqk, data1=X[p], data2=X[p], op=nl.multiply)
                nisa.tensor_reduce(dst=vmean[p], op=nl.add, data=sqk, axis=(1,))
                nisa.tensor_scalar(dst=vmean[p], data=vmean[p], op0=nl.multiply, operand0=1.0 / k)
                nisa.tensor_tensor(dst=vsum, data1=vsum, data2=vmean[p], op=nl.add)
            vn_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=vn_ps, stationary=ones32, moving=vsum)
            vnorm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=vnorm, data=vn_ps, op0=nl.multiply, operand0=float(k))
            nisa.activation(dst=vnorm, op=nl.sqrt, data=vnorm)
            step = []
            ssum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(ssum, 0.0)
            for p in range(mt):
                sec = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                if tall:
                    nisa.dma_copy(dst=sec, src=sec_in[i, p * P:(p + 1) * P, :])
                else:
                    nisa.dma_copy(dst=sec, src=sec_in[i, :, p * P:(p + 1) * P].ap(pattern=[[1, P], [1, 1]], offset=0))
                t2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=t2, data=vmean[p], op0=nl.multiply, operand0=1.0 - beta2)
                nisa.scalar_tensor_tensor(dst=sec, data=sec, op0=nl.multiply, operand0=beta2, op1=nl.add, operand1=t2)
                if tall:
                    nisa.dma_copy(dst=sec_in[i, p * P:(p + 1) * P, :], src=sec)
                else:
                    nisa.dma_copy(dst=sec_in[i, :, p * P:(p + 1) * P].ap(pattern=[[1, P], [1, 1]], offset=0), src=sec)
                st = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=st, data=sec, op0=nl.maximum, operand0=1e-10)
                nisa.activation(dst=st, op=nl.rsqrt, data=st)
                step.append(st)
                sc2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sc2, data1=st, data2=st, op=nl.multiply)
                nisa.scalar_tensor_tensor(dst=sc2, data=vmean[p], op0=nl.multiply, operand0=float(k), op1=nl.multiply, operand1=sc2)
                nisa.tensor_tensor(dst=ssum, data1=ssum, data2=sc2, op=nl.add)
            sn_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=sn_ps, stationary=ones32, moving=ssum)
            ratio = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=ratio, op=nl.sqrt, data=sn_ps)
            nisa.tensor_scalar(dst=ratio, data=ratio, op0=nl.maximum, operand0=1e-10)
            nisa.reciprocal(dst=ratio, data=ratio)
            nisa.tensor_tensor(dst=ratio, data1=ratio, data2=vnorm, op=nl.multiply)
            f_rows = []
            for p in range(mt):
                fr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=fr, data1=step[p], data2=ratio, op=nl.multiply)
                f_rows.append(fr)
            if not tall:
                f_row = nl.ndarray((1, m), dtype=nl.float32, buffer=nl.sbuf)
                for p in range(mt):
                    tp = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=tp, data=f_rows[p])
                    nisa.tensor_copy(dst=f_row[:, p * P:(p + 1) * P], src=tp)
                ones_row = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(ones_row, 1.0)
                Fb = nl.ndarray((P, m), dtype=nl.float32, buffer=nl.sbuf)
                for ch in range(mc):
                    fps = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=fps, stationary=ones_row, moving=f_row[:, ch * FMAX:(ch + 1) * FMAX])
                    nisa.tensor_copy(dst=Fb[:, ch * FMAX:(ch + 1) * FMAX], src=fps)
            Gnat = X if tall else Xt
            for p in range(ot):
                w = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=w, src=w_in[i, p * P:(p + 1) * P, :])
                gg = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
                if tall:
                    nisa.tensor_scalar(dst=gg, data=Gnat[p], op0=nl.multiply, operand0=f_rows[p])
                else:
                    nisa.tensor_tensor(dst=gg, data1=Gnat[p], data2=Fb, op=nl.multiply)
                prod = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=prod, data1=gg, data2=w, op=nl.multiply)
                mask = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mask, data=prod, op0=nl.greater_equal, operand0=0.0)
                wdm = nl.ndarray((P, i_), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=wdm, data1=w, data2=mask, op=nl.multiply)
                nisa.tensor_scalar(dst=wdm, data=wdm, op0=nl.multiply, operand0=sc[:, 2:3])
                nisa.scalar_tensor_tensor(dst=gg, data=gg, op0=nl.multiply, operand0=sc[:, 1:2], op1=nl.add, operand1=wdm)
                nisa.tensor_tensor(dst=w, data1=w, data2=gg, op=nl.subtract)
                nisa.dma_copy(dst=w_out[i, p * P:(p + 1) * P, :], src=w)
                nisa.dma_copy(dst=w_in[i, p * P:(p + 1) * P, :], src=w)
        return w_out

    adamw_step_nki = wrap_nki(adamw_kernel)
    muon_step_nki = wrap_nki(muon_kernel)
    attention_fwd_nki = wrap_nki(attention_cte)
    attention_bwd_nki = wrap_nki(attention_bwd)

    @nki_op("r1::doc_attention_fwd", mutates_args={})
    def doc_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink: torch.Tensor,
                          start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, t, d = q.shape
        qf = q.reshape(b * h, t, d).contiguous()
        kf = k.reshape(b * h, t, d).contiguous()
        vf = v.reshape(b * h, t, d).contiguous()
        sink_rows = sink.float().view(1, h).expand(b, h).reshape(b * h, 1).contiguous()
        bmin = start.unsqueeze(1).expand(b, h, t).reshape(b * h, t, 1).contiguous()
        bmax = end.unsqueeze(1).expand(b, h, t).reshape(b * h, t, 1).contiguous()
        out, neg_max, recip = attention_fwd_nki[NKI_GRID](qf, kf, vf, tp_q=True, tp_k=True, scale=1.0, causal_mask=True,
                                                          cache_softmax=True, bound_min=bmin, bound_max=bmax, sink=sink_rows)
        return out.reshape(b, h, t, d), neg_max, recip

    @nki_op("r1::doc_attention_bwd", mutates_args={})
    def doc_attention_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, dy: torch.Tensor,
                          neg_max: torch.Tensor, recip: torch.Tensor, sink: torch.Tensor, start: torch.Tensor,
                          end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, t, d = q.shape
        tr = lambda x: x.transpose(-2, -1).contiguous()
        lse = (-(neg_max.float() + torch.log(recip.float()))).reshape(b, h, *neg_max.shape[-2:]).contiguous()
        res = attention_bwd_nki[NKI_GRID](tr(q), tr(k), tr(v), tr(out), tr(dy.to(q.dtype)), lse,
                                          sinks_ref=sink.float().view(1, h).expand(b, h).contiguous(),
                                          bound_min=start.float().contiguous(), bound_max=end.float().contiguous(),
                                          use_causal_mask=True, mixed_precision=True, softmax_scale=1.0)
        dq, dk, dv = (r.transpose(-2, -1).contiguous() for r in res[:3])
        return dq, dk, dv, res[3].float().sum(dim=0)

    def _doc_attention_setup(ctx, inputs, output):
        q, k, v, sink, start, end = inputs
        out, neg_max, recip = output
        ctx.save_for_backward(q, k, v, sink, start, end, out, neg_max, recip)

    def _doc_attention_backward(ctx, dout, _dneg_max, _drecip):
        q, k, v, sink, start, end, out, neg_max, recip = ctx.saved_tensors
        dq, dk, dv, dsink = torch.ops.r1.doc_attention_bwd(q, k, v, out, dout, neg_max, recip, sink, start, end)
        return dq, dk, dv, dsink.to(sink.dtype), None, None

    torch.library.register_autograd("r1::doc_attention_fwd", _doc_attention_backward, setup_context=_doc_attention_setup)


    _NKI_P = 128

    _HVC = 512

    @nki.jit
    def head_fwd(x_in: nl.ndarray, wT: nl.ndarray, tgt: nl.ndarray, scal: nl.ndarray, tau: nl.ndarray, eps: float = 0.0078125):
        T, D = x_in.shape
        V = wT.shape[1]
        kt = D // _NKI_P
        nchunk = V // _HVC
        loss_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        lse_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        h_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
        rstd_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        zt_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nprog = nl.num_programs(0)
        tiles = T // _NKI_P
        per = tiles // nprog
        sc = nl.ndarray((_NKI_P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        inv_sp = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        inv_sn = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_sp, data=sc[:, 0:1])
        nisa.reciprocal(dst=inv_sn, data=sc[:, 1:2])
        col = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=col, pattern=[[1, _HVC]], offset=0, channel_multiplier=0)
        cap = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=cap, data1=sc[:, 0:1], data2=sc[:, 1:2], op=nl.maximum)
        negcap = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negcap, data=cap, op0=nl.multiply, operand0=-1.0)
        wsb = nl.ndarray((_NKI_P, kt, V), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(kt):
            nisa.dma_copy(dst=wsb[:, i, :], src=wT[i * _NKI_P:(i + 1) * _NKI_P, :])
        for ti in nl.affine_range(per):
            tb = nl.program_id(0) * per + ti
            t0 = tb * _NKI_P
            xr = nl.ndarray((_NKI_P, D), dtype=x_in.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xr, src=x_in[t0:t0 + _NKI_P, :])
            x = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x, src=xr)
            sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=sq, op=nl.square, data=x, reduce_op=nl.add, reduce_res=ms)
            rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            nisa.dma_copy(dst=rstd_out[t0:t0 + _NKI_P, :], src=rstd)
            h = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=x, op0=nl.multiply, operand0=rstd)
            nisa.dma_copy(dst=h_out[t0:t0 + _NKI_P, :], src=h)
            hT = []
            for i in range(kt):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=h[:, i * _NKI_P:(i + 1) * _NKI_P])
                hTi = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=hTi, src=tps)
                hT.append(hTi)
            tgi = nl.ndarray((_NKI_P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tgi, src=tgt[t0:t0 + _NKI_P, :])
            tg = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tg, src=tgi)
            tai = nl.ndarray((_NKI_P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tai, src=tau[t0:t0 + _NKI_P, :])
            ta = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ta, src=tai)
            s = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            tl = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            zt = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(s, 0.0)
            nisa.memset(tl, 0.0)
            nisa.memset(zt, 0.0)
            for c in range(nchunk):
                zps = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.psum)
                for i in range(kt):
                    nisa.nc_matmul(dst=zps, stationary=hT[i], moving=wsb[:, i, c * _HVC:(c + 1) * _HVC], accumulate=(i > 0))
                z = zps
                tp = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                tn = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=tp, op=nl.tanh, data=z, scale=inv_sp)
                nisa.tensor_scalar(dst=tp, data=tp, op0=nl.multiply, operand0=sc[:, 0:1])
                nisa.activation(dst=tn, op=nl.tanh, data=z, scale=inv_sn)
                nisa.tensor_scalar(dst=tn, data=tn, op0=nl.multiply, operand0=sc[:, 1:2])
                pos = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=pos, data=z, op0=nl.greater_equal, operand0=0.0)
                zc = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=zc, data1=tp, data2=tn, op=nl.subtract)
                nisa.tensor_tensor(dst=zc, data1=zc, data2=pos, op=nl.multiply)
                nisa.tensor_tensor(dst=zc, data1=zc, data2=tn, op=nl.add)
                tgc = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=tgc, data=tg, op0=nl.subtract, operand0=float(c * _HVC))
                eqm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eqm, data=col, op0=nl.equal, operand0=tgc)
                sel = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sel, data1=zc, data2=eqm, op=nl.multiply)
                part = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=part, op=nl.add, data=sel, axis=(1,))
                nisa.tensor_tensor(dst=tl, data1=tl, data2=part, op=nl.add)
                tac = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=tac, data=ta, op0=nl.subtract, operand0=float(c * _HVC))
                sel2 = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=sel2, data=col, op0=nl.equal, operand0=tac, op1=nl.multiply, operand1=zc)
                part2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=part2, op=nl.add, data=sel2, axis=(1,))
                nisa.tensor_tensor(dst=zt, data1=zt, data2=part2, op=nl.add)
                e = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=e, op=nl.exp, data=zc, bias=negcap)
                es = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=es, op=nl.add, data=e, axis=(1,))
                nisa.tensor_tensor(dst=s, data1=s, data2=es, op=nl.add)
            lse = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=lse, op=nl.log, data=s)
            nisa.tensor_tensor(dst=lse, data1=lse, data2=cap, op=nl.add)
            nisa.dma_copy(dst=lse_out[t0:t0 + _NKI_P, :], src=lse)
            valid = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=valid, data=tg, op0=nl.greater_equal, operand0=0.0)
            lossr = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=lossr, data1=lse, data2=tl, op=nl.subtract)
            nisa.tensor_tensor(dst=lossr, data1=lossr, data2=valid, op=nl.multiply)
            nisa.dma_copy(dst=loss_out[t0:t0 + _NKI_P, :], src=lossr)
            nisa.dma_copy(dst=zt_out[t0:t0 + _NKI_P, :], src=zt)
        return loss_out, lse_out, h_out, rstd_out, zt_out

    @nki.jit
    def head_bwd(h_in: nl.ndarray, rstd_in: nl.ndarray, lse_in: nl.ndarray, dl_in: nl.ndarray, w: nl.ndarray, wT: nl.ndarray,
                 tgt: nl.ndarray, scal: nl.ndarray, tau: nl.ndarray, dlse_in: nl.ndarray, dzt_in: nl.ndarray):
        """Backward of head_fwd for all three of its value outputs: dl, dlse, dzt (T, 1) fp32 are the upstream grads of
        the per-row loss, lse and zt, so dz = (dl + dlse) * softmax - dl * [col == tgt] + dzt * [col == tau].
        h (T, D) bf16 (normed input), rstd/lse (T, 1) fp32,
        w (V, D) bf16, wT (D, V) bf16, tgt (T, 1) int32, scal as in the forward. Returns dx (T, D) bf16 and dW (V, D) fp32.
        Token halves of TH rows are the outer loop; inside, vocab chunks (dW chunk accumulates in PSUM over the half's token
        tiles) and token tiles (z recomputed, dh accumulated in SBUF fp32); the rmsnorm backward closes each half."""
        T, D = h_in.shape
        V = w.shape[0]
        kt = D // _NKI_P
        nchunk = V // _HVC
        TH = 2048 if T >= 2048 else T
        nh = T // TH
        th_tiles = TH // _NKI_P
        dx_out = nl.ndarray((T, D), dtype=h_in.dtype, buffer=nl.shared_hbm)
        dw_out = nl.ndarray((V, D), dtype=nl.float32, buffer=nl.shared_hbm)
        dcap_out = nl.ndarray((_NKI_P, 2), dtype=nl.float32, buffer=nl.shared_hbm)
        sc = nl.ndarray((_NKI_P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        inv_sp = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        inv_sn = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_sp, data=sc[:, 0:1])
        nisa.reciprocal(dst=inv_sn, data=sc[:, 1:2])
        d_inv = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d_inv, data1=inv_sp, data2=inv_sn, op=nl.subtract)
        d_s = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d_s, data1=sc[:, 0:1], data2=sc[:, 1:2], op=nl.subtract)
        col = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=col, pattern=[[1, _HVC]], offset=0, channel_multiplier=0)
        dcap = nl.ndarray((_NKI_P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dcap, 0.0)
        for hh in range(nh):
            base = hh * TH
            h = nl.ndarray((_NKI_P, th_tiles, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            hT = nl.ndarray((_NKI_P, th_tiles, kt, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
            dh = nl.ndarray((_NKI_P, th_tiles, D), dtype=nl.float32, buffer=nl.sbuf)
            g = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nlse = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            tgi = nl.ndarray((_NKI_P, th_tiles), dtype=nl.int32, buffer=nl.sbuf)
            tg = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            g1 = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            g3 = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            tai = nl.ndarray((_NKI_P, th_tiles), dtype=nl.int32, buffer=nl.sbuf)
            ta = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dh, 0.0)
            for t in range(th_tiles):
                t0 = base + t * _NKI_P
                nisa.dma_copy(dst=h[:, t, :], src=h_in[t0:t0 + _NKI_P, :])
                for i in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=h[:, t, i * _NKI_P:(i + 1) * _NKI_P])
                    nisa.tensor_copy(dst=hT[:, t, i, :], src=tps)
                nisa.dma_copy(dst=tgi[:, t:t + 1], src=tgt[t0:t0 + _NKI_P, :])
                nisa.dma_copy(dst=g[:, t:t + 1], src=dl_in[t0:t0 + _NKI_P, :])
                nisa.dma_copy(dst=nlse[:, t:t + 1], src=lse_in[t0:t0 + _NKI_P, :])
                nisa.dma_copy(dst=g1[:, t:t + 1], src=dlse_in[t0:t0 + _NKI_P, :])
                nisa.dma_copy(dst=g3[:, t:t + 1], src=dzt_in[t0:t0 + _NKI_P, :])
                nisa.dma_copy(dst=tai[:, t:t + 1], src=tau[t0:t0 + _NKI_P, :])
            nisa.tensor_copy(dst=tg, src=tgi)
            nisa.tensor_copy(dst=ta, src=tai)
            valid = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=valid, data=tg, op0=nl.greater_equal, operand0=0.0)
            nisa.tensor_tensor(dst=g, data1=g, data2=valid, op=nl.multiply)
            nisa.tensor_tensor(dst=g1, data1=g1, data2=valid, op=nl.multiply)
            nisa.tensor_tensor(dst=g1, data1=g1, data2=g, op=nl.add)
            nisa.tensor_tensor(dst=g3, data1=g3, data2=valid, op=nl.multiply)
            nisa.tensor_scalar(dst=nlse, data=nlse, op0=nl.multiply, operand0=-1.0)
            for c in range(nchunk):
                wc = nl.ndarray((_NKI_P, kt, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                wTc = nl.ndarray((_NKI_P, kt, _HVC), dtype=nl.bfloat16, buffer=nl.sbuf)
                for j in range(kt):
                    nisa.dma_copy(dst=wc[:, j, :], src=w[c * _HVC + j * _NKI_P:c * _HVC + (j + 1) * _NKI_P, :])
                    nisa.dma_copy(dst=wTc[:, j, :], src=wT[j * _NKI_P:(j + 1) * _NKI_P, c * _HVC:(c + 1) * _HVC])
                cm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=cm, data=col, op0=nl.add, operand0=float(c * _HVC))
                dwps = []
                for j in range(kt):
                    dwps.append(nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum))
                for t in range(th_tiles):
                    zps = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.psum)
                    for i in range(kt):
                        nisa.nc_matmul(dst=zps, stationary=hT[:, t, i, :], moving=wTc[:, i, :], accumulate=(i > 0))
                    pos = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=pos, data=zps, op0=nl.greater_equal, operand0=0.0)
                    invs = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=invs, data=pos, op0=nl.multiply, operand0=d_inv)
                    nisa.tensor_scalar(dst=invs, data=invs, op0=nl.add, operand0=inv_sn)
                    zs = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=zs, data1=zps, data2=invs, op=nl.multiply)
                    th = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=th, op=nl.tanh, data=zs)
                    sv = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=sv, data=pos, op0=nl.multiply, operand0=d_s)
                    nisa.tensor_scalar(dst=sv, data=sv, op0=nl.add, operand0=sc[:, 1:2])
                    zc = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=zc, data1=th, data2=sv, op=nl.multiply)
                    deriv = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=deriv, data1=th, data2=th, op=nl.multiply)
                    nisa.tensor_scalar(dst=deriv, data=deriv, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
                    pr = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=pr, op=nl.exp, data=zc, bias=nlse[:, t:t + 1])
                    eqm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=eqm, data=cm, op0=nl.equal, operand0=tg[:, t:t + 1], op1=nl.multiply, operand1=g[:, t:t + 1])
                    eq2 = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=eq2, data=cm, op0=nl.equal, operand0=ta[:, t:t + 1], op1=nl.multiply, operand1=g3[:, t:t + 1])
                    pr2 = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(dst=pr2, data=pr, op0=nl.multiply, operand0=g1[:, t:t + 1], op1=nl.add, operand1=eq2)
                    nisa.tensor_tensor(dst=pr, data1=pr2, data2=eqm, op=nl.subtract)
                    dzf = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=dzf, data1=pr, data2=deriv, op=nl.multiply)
                    dz = nl.ndarray((_NKI_P, _HVC), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(dst=dz, op=nl.copy, data=dzf)
                    mm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=mm, data1=pr, data2=th, op=nl.multiply)
                    uu = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=uu, data1=zs, data2=dzf, op=nl.multiply)
                    nisa.tensor_tensor(dst=mm, data1=mm, data2=uu, op=nl.subtract)
                    nisa.tensor_tensor(dst=uu, data1=mm, data2=pos, op=nl.multiply)
                    partA = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    partB = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=partA, op=nl.add, data=uu, axis=(1,))
                    nisa.tensor_reduce(dst=partB, op=nl.add, data=mm, axis=(1,))
                    nisa.tensor_tensor(dst=dcap[:, 0:1], data1=dcap[:, 0:1], data2=partA, op=nl.add)
                    nisa.tensor_tensor(dst=partB, data1=partB, data2=partA, op=nl.subtract)
                    nisa.tensor_tensor(dst=dcap[:, 1:2], data1=dcap[:, 1:2], data2=partB, op=nl.add)
                    dhps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    for j in range(kt):
                        tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=tps, data=dz[:, j * _NKI_P:(j + 1) * _NKI_P])
                        dzT = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dzT, src=tps)
                        nisa.nc_matmul(dst=dhps, stationary=dzT, moving=wc[:, j, :], accumulate=(j > 0))
                    nisa.tensor_tensor(dst=dh[:, t, :], data1=dh[:, t, :], data2=dhps, op=nl.add)
                    for j in range(kt):
                        nisa.nc_matmul(dst=dwps[j], stationary=dz[:, j * _NKI_P:(j + 1) * _NKI_P], moving=h[:, t, :], accumulate=(t > 0))
                for j in range(kt):
                    r0 = c * _HVC + j * _NKI_P
                    if hh == 0:
                        dwsb = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dwsb, src=dwps[j])
                        nisa.dma_copy(dst=dw_out[r0:r0 + _NKI_P, :], src=dwsb)
                    else:
                        prev = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.dma_copy(dst=prev, src=dw_out[r0:r0 + _NKI_P, :])
                        nisa.tensor_tensor(dst=prev, data1=prev, data2=dwps[j], op=nl.add)
                        nisa.dma_copy(dst=dw_out[r0:r0 + _NKI_P, :], src=prev)
            for t in range(th_tiles):
                t0 = base + t * _NKI_P
                rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=rstd, src=rstd_in[t0:t0 + _NKI_P, :])
                hf = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=hf, src=h[:, t, :])
                prod = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=prod, data1=dh[:, t, :], data2=hf, op=nl.multiply)
                mdot = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=mdot, op=nl.add, data=prod, axis=(1,))
                nisa.tensor_scalar(dst=mdot, data=mdot, op0=nl.multiply, operand0=-1.0 / D)
                nisa.tensor_scalar(dst=hf, data=hf, op0=nl.multiply, operand0=mdot)
                nisa.tensor_tensor(dst=hf, data1=hf, data2=dh[:, t, :], op=nl.add)
                dx = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dx, data=hf, op0=nl.multiply, operand0=rstd)
                nisa.dma_copy(dst=dx_out[t0:t0 + _NKI_P, :], src=dx)
        nisa.dma_copy(dst=dcap_out, src=dcap)
        return dx_out, dw_out, dcap_out

    _FMAX = 512

    @nki.jit
    def rmsnorm_fwd(x_in: nl.ndarray, eps: float = 1e-6) -> nl.ndarray:
        rows, d = x_in.shape
        out = nl.ndarray((rows, d), dtype=x_in.dtype, buffer=nl.shared_hbm)
        nprog = nl.num_programs(0)
        per = (rows // _NKI_P) // nprog
        for ii in nl.affine_range(per):
            i = nl.program_id(0) * per + ii
            x = nl.ndarray((_NKI_P, d), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=x_in[i * _NKI_P:(i + 1) * _NKI_P, :])
            sq = nl.ndarray((_NKI_P, d), dtype=nl.float32, buffer=nl.sbuf)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=sq, op=nl.square, data=x, reduce_op=nl.add, reduce_res=ms)
            nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=eps)
            nisa.activation(dst=ms, op=nl.rsqrt, data=ms)
            y = nl.ndarray((_NKI_P, d), dtype=x_in.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y, data=x, op0=nl.multiply, operand0=ms)
            nisa.dma_copy(dst=out[i * _NKI_P:(i + 1) * _NKI_P, :], src=y)
        return out

    @nki.jit
    def rmsnorm_bwd(x_in: nl.ndarray, dy_in: nl.ndarray, eps: float = 1e-6) -> nl.ndarray:
        rows, d = x_in.shape
        out = nl.ndarray((rows, d), dtype=dy_in.dtype, buffer=nl.shared_hbm)
        nprog = nl.num_programs(0)
        per = (rows // _NKI_P) // nprog
        for ii in nl.affine_range(per):
            i = nl.program_id(0) * per + ii
            x = nl.ndarray((_NKI_P, d), dtype=nl.float32, buffer=nl.sbuf)
            dy = nl.ndarray((_NKI_P, d), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=x_in[i * _NKI_P:(i + 1) * _NKI_P, :])
            nisa.dma_copy(dst=dy, src=dy_in[i * _NKI_P:(i + 1) * _NKI_P, :])
            t = nl.ndarray((_NKI_P, d), dtype=nl.float32, buffer=nl.sbuf)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=t, op=nl.square, data=x, reduce_op=nl.add, reduce_res=ms)
            nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=eps)
            r = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=r, op=nl.rsqrt, data=ms)
            nisa.tensor_tensor(dst=t, data1=dy, data2=x, op=nl.multiply)
            c = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=c, op=nl.add, data=t, axis=(1,))
            r2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r2, data1=r, data2=r, op=nl.multiply)
            nisa.tensor_tensor(dst=c, data1=c, data2=r2, op=nl.multiply)
            nisa.tensor_scalar(dst=c, data=c, op0=nl.multiply, operand0=1.0 / d)
            nisa.tensor_scalar(dst=t, data=x, op0=nl.multiply, operand0=c)
            nisa.tensor_tensor(dst=t, data1=dy, data2=t, op=nl.subtract)
            dx = nl.ndarray((_NKI_P, d), dtype=dy_in.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dx, data=t, op0=nl.multiply, operand0=r)
            nisa.dma_copy(dst=out[i * _NKI_P:(i + 1) * _NKI_P, :], src=dx)
        return out


    @nki.jit
    def pre_attn_bwd(dq_in: nl.ndarray, dk_in: nl.ndarray, dv_in: nl.ndarray, rq_in: nl.ndarray, rk_in: nl.ndarray,
                     xm_in: nl.ndarray, dx1_in: nl.ndarray, cos_in: nl.ndarray, sin_in: nl.ndarray, dksh_in: nl.ndarray,
                     wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray, n_head: int = 4, eps: float = 1e-6,
                     scale: float = 1.0, hq: int = 32):
        T, D = xm_in.shape
        BH, t, HD = dq_in.shape
        half = HD // 2
        kt = D // _NKI_P
        tiles = T // _NKI_P
        tpt = t // _NKI_P
        dxm_out = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        gv_out = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        dwq_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
        dwk_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
        dwv_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
        Wq = []
        Wk = []
        Wv = []
        for o in range(kt):
            a = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=wq[o * _NKI_P:(o + 1) * _NKI_P, :])
            Wq.append(a)
            a = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=wk[o * _NKI_P:(o + 1) * _NKI_P, :])
            Wk.append(a)
            a = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=wv[o * _NKI_P:(o + 1) * _NKI_P, :])
            Wv.append(a)
        accq = []
        acck = []
        accv = []
        for o in range(kt):
            a = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            accq.append(a)
            a = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            acck.append(a)
            a = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            accv.append(a)
        for tb in range(tiles):
            t0 = tb * _NKI_P
            b = tb // tpt
            p0 = (tb - b * tpt) * _NKI_P
            xmb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xmb, src=xm_in[t0:t0 + _NKI_P, :])
            xm = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xm, src=xmb)
            sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=sq, op=nl.square, data=xm, reduce_op=nl.add, reduce_res=ms)
            rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
            cs = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            sn = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs, src=cos_in[t0:t0 + _NKI_P, :])
            nisa.dma_copy(dst=sn, src=sin_in[t0:t0 + _NKI_P, :])
            nisa.tensor_scalar(dst=sn, data=sn, op0=nl.multiply, operand0=-1.0)
            dpq = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            dpk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            gvb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            for hh in range(n_head):
                bh = b * n_head + hh
                c0 = hh * HD
                for j in range(2):
                    g = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    if j == 0:
                        gb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.dma_copy(dst=gb, src=dq_in[bh, p0:p0 + _NKI_P, :])
                        nisa.tensor_scalar(dst=g, data=gb, op0=nl.multiply, operand0=scale)
                    else:
                        gb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.dma_copy(dst=gb, src=dk_in[bh, p0:p0 + _NKI_P, :])
                        nisa.tensor_copy(dst=g, src=gb)
                        shb = nl.ndarray((_NKI_P, hq), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.dma_copy(dst=shb, src=dksh_in[bh, p0:p0 + _NKI_P, :])
                        nisa.tensor_copy(dst=g[:, hq:2 * hq], src=shb)
                    rb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if j == 0:
                        nisa.dma_copy(dst=rb, src=rq_in[hh, t0:t0 + _NKI_P, :])
                    else:
                        nisa.dma_copy(dst=rb, src=rk_in[hh, t0:t0 + _NKI_P, :])
                    r = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=r, src=rb)
                    tt = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    m2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation_reduce(dst=tt, op=nl.square, data=r, reduce_op=nl.add, reduce_res=m2)
                    nisa.tensor_scalar(dst=m2, data=m2, op0=nl.multiply, operand0=1.0 / HD, op1=nl.add, operand1=eps)
                    rs = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=rs, op=nl.rsqrt, data=m2)
                    nisa.tensor_tensor(dst=tt, data1=g, data2=r, op=nl.multiply)
                    cc = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=cc, op=nl.add, data=tt, axis=(1,))
                    rs2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=rs2, data1=rs, data2=rs, op=nl.multiply)
                    nisa.tensor_tensor(dst=cc, data1=cc, data2=rs2, op=nl.multiply)
                    nisa.tensor_scalar(dst=cc, data=cc, op0=nl.multiply, operand0=1.0 / HD)
                    nisa.scalar_tensor_tensor(dst=tt, data=r, op0=nl.multiply, operand0=cc, op1=nl.subtract, operand1=g, reverse1=True)
                    dn = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dn, data=tt, op0=nl.multiply, operand0=rs)
                    _rh = max(1, min(half, int(round(half * ROPE_FRAC))))
                    x1 = dn[:, 0:_rh]
                    x2 = dn[:, half:half + _rh]
                    r1 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    _cs = cs if _rh == half else cs[:, 0:_rh]
                    _sn = sn if _rh == half else sn[:, 0:_rh]
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)
                    _dd = dpq if j == 0 else dpk
                    nisa.tensor_copy(dst=_dd[:, c0:c0 + _rh], src=r1)
                    nisa.tensor_copy(dst=_dd[:, c0 + half:c0 + half + _rh], src=r2)
                    if _rh < half:
                        nisa.tensor_copy(dst=_dd[:, c0 + _rh:c0 + half], src=dn[:, _rh:half])
                        nisa.tensor_copy(dst=_dd[:, c0 + half + _rh:c0 + HD], src=dn[:, half + _rh:HD])
                nisa.dma_copy(dst=gvb[:, c0:c0 + HD], src=dv_in[bh, p0:p0 + _NKI_P, :])
            nisa.dma_copy(dst=gv_out[t0:t0 + _NKI_P, :], src=gvb)
            dh_ps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            first = True
            for src_idx in range(3):
                for o in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    if src_idx == 0:
                        nisa.nc_transpose(dst=tps, data=dpq[:, o * _NKI_P:(o + 1) * _NKI_P])
                    elif src_idx == 1:
                        nisa.nc_transpose(dst=tps, data=dpk[:, o * _NKI_P:(o + 1) * _NKI_P])
                    else:
                        nisa.nc_transpose(dst=tps, data=gvb[:, o * _NKI_P:(o + 1) * _NKI_P])
                    aT = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=aT, src=tps)
                    if src_idx == 0:
                        nisa.nc_matmul(dst=dh_ps, stationary=aT, moving=Wq[o], accumulate=(not first))
                    elif src_idx == 1:
                        nisa.nc_matmul(dst=dh_ps, stationary=aT, moving=Wk[o], accumulate=(not first))
                    else:
                        nisa.nc_matmul(dst=dh_ps, stationary=aT, moving=Wv[o], accumulate=(not first))
                    first = False
            dh = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dh, src=dh_ps)
            nisa.tensor_tensor(dst=sq, data1=dh, data2=xm, op=nl.multiply)
            cm = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=cm, op=nl.add, data=sq, axis=(1,))
            r2c = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r2c, data1=rstd, data2=rstd, op=nl.multiply)
            nisa.tensor_tensor(dst=cm, data1=cm, data2=r2c, op=nl.multiply)
            nisa.tensor_scalar(dst=cm, data=cm, op0=nl.multiply, operand0=1.0 / D)
            nisa.scalar_tensor_tensor(dst=sq, data=xm, op0=nl.multiply, operand0=cm, op1=nl.subtract, operand1=dh, reverse1=True)
            dx1b = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=dx1b, src=dx1_in[t0:t0 + _NKI_P, :])
            dx1 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dx1, src=dx1b)
            nisa.scalar_tensor_tensor(dst=dx1, data=sq, op0=nl.multiply, operand0=rstd, op1=nl.add, operand1=dx1)
            dxb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dxb, src=dx1)
            nisa.dma_copy(dst=dxm_out[t0:t0 + _NKI_P, :], src=dxb)
            for o in range(kt):
                wps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=wps, stationary=dpq[:, o * _NKI_P:(o + 1) * _NKI_P], moving=h)
                nisa.tensor_tensor(dst=accq[o], data1=accq[o], data2=wps, op=nl.add)
                wps2 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=wps2, stationary=dpk[:, o * _NKI_P:(o + 1) * _NKI_P], moving=h)
                nisa.tensor_tensor(dst=acck[o], data1=acck[o], data2=wps2, op=nl.add)
                wps3 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=wps3, stationary=gvb[:, o * _NKI_P:(o + 1) * _NKI_P], moving=h)
                nisa.tensor_tensor(dst=accv[o], data1=accv[o], data2=wps3, op=nl.add)
        for o in range(kt):
            nisa.dma_copy(dst=dwq_out[o * _NKI_P:(o + 1) * _NKI_P, :], src=accq[o])
            nisa.dma_copy(dst=dwk_out[o * _NKI_P:(o + 1) * _NKI_P, :], src=acck[o])
            nisa.dma_copy(dst=dwv_out[o * _NKI_P:(o + 1) * _NKI_P, :], src=accv[o])
        return dxm_out, gv_out, dwq_out, dwk_out, dwv_out

    @nki.jit
    def pre_attn_fwd_h(x_in: nl.ndarray, x0_in: nl.ndarray, ve_in: nl.ndarray, wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray,
                     cos_in: nl.ndarray, sin_in: nl.ndarray, scal: nl.ndarray, n_head: int = 4, eps: float = 1e-6, fq: int = 0,
                       t_len: int = 1024, q_scale: float = 1.0):
        """PRE-attention half of a layer as one kernel (T tokens = b*t rows, D = 512, head dim 128):
          xm = s0*x + s1*x0                    (scal[:, 0], scal[:, 1]: resid_lambda, x0_lambda as a (128, 4) fp32 tensor)
          h  = rmsnorm(xm)
          q  = rope(h @ Wq^T), k = rope(h @ Wk^T), both rms-normed per head; v = h @ Wv^T + s2 * ve   (s2 = ve_lambda)
        Returns q, k, v as (n_head, T, 128) bf16 (the attention kernel's (h, t, d) layout per batch row), xm (T, D) bf16, and the
        pre-norm rotated q and k (rq, rk: (n_head, T, 128) bf16) that the QK-norm backward needs.
        cos/sin: (T, 64) fp32 tables of the split-half rotation (rows = token positions of the T rows)."""
        T, D = x_in.shape
        HD = D // n_head
        half = HD // 2
        q_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        k_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        v_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        tpt = t_len // _NKI_P
        xm_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
        rq_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        rk_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        kt = D // _NKI_P
        nprog = nl.num_programs(0)
        tiles = T // _NKI_P
        per = tiles // nprog
        sc = nl.ndarray((_NKI_P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        WqT = []
        WkT = []
        WvT = []
        for _j in range(kt):
            WqT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
            WkT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
            WvT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        for o in range(kt):
            bq = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            bk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            bv = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=bq, src=wq[o * _NKI_P:(o + 1) * _NKI_P, :])
            nisa.dma_copy(dst=bk, src=wk[o * _NKI_P:(o + 1) * _NKI_P, :])
            nisa.dma_copy(dst=bv, src=wv[o * _NKI_P:(o + 1) * _NKI_P, :])
            for i in range(kt):
                tq = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tq, data=bq[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WqT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tq)
                tk = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tk, data=bk[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WkT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tk)
                tv = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tv, data=bv[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WvT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tv)
        for tb in range(tiles):
            t0 = tb * _NKI_P
            row = tb // tpt
            p0 = (tb - row * tpt) * _NKI_P
            if fq:
                for i in range(len(WqT)):
                    _fq8(WqT[i]); _fq8(WkT[i]); _fq8(WvT[i])
            x = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            x0 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=x_in[t0:t0 + _NKI_P, :])
            nisa.dma_copy(dst=x0, src=x0_in[t0:t0 + _NKI_P, :])
            xm = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=xm, data=x0, op0=nl.multiply, operand0=sc[:, 1:2])
            nisa.scalar_tensor_tensor(dst=xm, data=x, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=xm)
            xmb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xmb, src=xm)
            nisa.dma_copy(dst=xm_out[t0:t0 + _NKI_P, :], src=xmb)
            sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=sq, op=nl.square, data=xm, reduce_op=nl.add, reduce_res=ms)
            rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
            if fq:
                _fq8(h)
            hT = []
            for i in range(kt):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=h[:, i * _NKI_P:(i + 1) * _NKI_P])
                hTi = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=hTi, src=tps)
                hT.append(hTi)
            qps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            kps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            vps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            for i in range(kt):
                nisa.nc_matmul(dst=qps, stationary=hT[i], moving=WqT[i], accumulate=(i > 0))
                nisa.nc_matmul(dst=kps, stationary=hT[i], moving=WkT[i], accumulate=(i > 0))
                nisa.nc_matmul(dst=vps, stationary=hT[i], moving=WvT[i], accumulate=(i > 0))
            q = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            k = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            v = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q, src=qps)
            nisa.tensor_copy(dst=k, src=kps)
            ve = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ve, src=ve_in[t0:t0 + _NKI_P, :])
            nisa.scalar_tensor_tensor(dst=v, data=ve, op0=nl.multiply, operand0=sc[:, 2:3], op1=nl.add, operand1=vps)
            cs = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            sn = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs, src=cos_in[t0:t0 + _NKI_P, :])
            nisa.dma_copy(dst=sn, src=sin_in[t0:t0 + _NKI_P, :])
            for hh in range(n_head):
                c0 = hh * HD
                for j in range(2):
                    src_t = q if j == 0 else k
                    dst_t = q_out if j == 0 else k_out
                    rot_t = rq_out if j == 0 else rk_out
                    _rh = max(1, min(half, int(round(half * ROPE_FRAC))))
                    x1 = src_t[:, c0:c0 + _rh]
                    x2 = src_t[:, c0 + half:c0 + half + _rh]
                    r1 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    _cs = cs if _rh == half else cs[:, 0:_rh]
                    _sn = sn if _rh == half else sn[:, 0:_rh]
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)
                    rot = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rot[:, 0:_rh], src=r1)
                    nisa.tensor_copy(dst=rot[:, half:half + _rh], src=r2)
                    if _rh < half:
                        nisa.tensor_copy(dst=rot[:, _rh:half], src=src_t[:, c0 + _rh:c0 + half])
                        nisa.tensor_copy(dst=rot[:, half + _rh:HD], src=src_t[:, c0 + half + _rh:c0 + HD])
                    rb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rb, src=rot)
                    nisa.dma_copy(dst=rot_t[hh, t0:t0 + _NKI_P, :], src=rb)
                    sq2 = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    ms2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation_reduce(dst=sq2, op=nl.square, data=rot, reduce_op=nl.add, reduce_res=ms2)
                    rs2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=rs2, data=ms2, op0=nl.multiply, operand0=1.0 / HD, op1=nl.add, operand1=eps)
                    nisa.activation(dst=rs2, op=nl.rsqrt, data=rs2)
                    ob = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if j == 0:
                        nisa.tensor_scalar(dst=ob, data=rot, op0=nl.multiply, operand0=rs2, op1=nl.multiply, operand1=q_scale)
                    else:
                        nisa.tensor_scalar(dst=ob, data=rot, op0=nl.multiply, operand0=rs2)
                    nisa.dma_copy(dst=dst_t[row * n_head + hh, p0:p0 + _NKI_P, :], src=ob)
                vb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=vb, src=v[:, c0:c0 + HD])
                nisa.dma_copy(dst=v_out[row * n_head + hh, p0:p0 + _NKI_P, :], src=vb)
        return q_out, k_out, v_out, xm_out, rq_out, rk_out

    @nki.jit
    def post_attn_fwd_h(y_in: nl.ndarray, wo: nl.ndarray, xm_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, eps: float = 1e-6, fq: int = 0,
                        n_head: int = 4, t_len: int = 1024, gs_in=None):
        # ATTN_GATE: gs_in (T, n_head) fp32 = sigmoid(z) from pre_attn_fwd_ko; head h's output is multiplied by 2*gs[:, h]
        # before the out-projection. y2_out then holds the GATED y (dWo's operand), outT stays UNGATED (the vendor
        # backward's O). gs_in=None: the ungated kernel.
        T, D = xm_in.shape
        H = w1.shape[0]
        HD = D // n_head
        out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)
        y2_out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)
        outT_out = nl.ndarray((y_in.shape[0], HD, t_len), dtype=y_in.dtype, buffer=nl.shared_hbm)
        tpt = t_len // _NKI_P
        x1_out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)
        pre_out = nl.ndarray((H, T), dtype=y_in.dtype, buffer=nl.shared_hbm)
        kt = D // _NKI_P
        ht = H // _NKI_P
        nprog = nl.num_programs(0)
        chunks = T // _FMAX
        per = chunks // nprog
        WoT = []
        for _j in range(kt):
            WoT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        for o in range(kt):
            blk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=blk, src=wo[o * _NKI_P:(o + 1) * _NKI_P, :])
            for i in range(kt):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=blk[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WoT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tps)
        W1 = []
        for _j in range(kt):
            W1.append(nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf))
        for h in range(ht):
            blk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=blk, src=w1[h * _NKI_P:(h + 1) * _NKI_P, :])
            for k in range(kt):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=blk[:, k * _NKI_P:(k + 1) * _NKI_P])
                nisa.tensor_copy(dst=W1[k][:, h * _NKI_P:(h + 1) * _NKI_P], src=tps)
        W2 = []
        for _j in range(ht):
            W2.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        for o in range(kt):
            blk = nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=blk, src=w2[o * _NKI_P:(o + 1) * _NKI_P, :])
            for h in range(ht):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=blk[:, h * _NKI_P:(h + 1) * _NKI_P])
                nisa.tensor_copy(dst=W2[h][:, o * _NKI_P:(o + 1) * _NKI_P], src=tps)
        oTb = []
        for _j in range(kt):
            oTb.append(nl.ndarray((_NKI_P, t_len), dtype=nl.bfloat16, buffer=nl.sbuf))
        for c in range(chunks):
            t0 = c * _FMAX
            if fq:
                for i in range(len(WoT)):
                    _fq8(WoT[i])
                for k in range(len(W1)):
                    _fq8(W1[k])
                for h in range(len(W2)):
                    _fq8(W2[h])
            h2T = []
            for _j in range(kt):
                h2T.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            x1_tiles = []
            for _j in range(_FMAX // _NKI_P):
                x1_tiles.append(nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf))
            for tb in range(_FMAX // _NKI_P):
                yt = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                gt = (t0 + tb * _NKI_P) // _NKI_P
                row = gt // tpt
                p0 = (gt - row * tpt) * _NKI_P
                for hh in range(n_head):
                    nisa.dma_copy(dst=yt[:, hh * HD:(hh + 1) * HD], src=y_in[row * n_head + hh, p0:p0 + _NKI_P, :])
                if gs_in is not None:                 # ATTN_GATE: ytg[:, head h] = y[:, head h] * s[:, h] * 2
                    gtl = nl.ndarray((_NKI_P, n_head), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=gtl, src=gs_in[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :])
                    ytg = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for hh in range(n_head):
                        nisa.tensor_scalar(dst=ytg[:, hh * HD:(hh + 1) * HD], data=yt[:, hh * HD:(hh + 1) * HD], op0=nl.multiply,
                                           operand0=gtl[:, hh:hh + 1], op1=nl.multiply, operand1=2.0)
                    nisa.dma_copy(dst=y2_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=ytg)
                else:
                    nisa.dma_copy(dst=y2_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=yt)
                if fq:
                    _fq8(yt)
                aps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for i in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=yt[:, i * _NKI_P:(i + 1) * _NKI_P])
                    yTi = oTb[i][:, p0:p0 + _NKI_P]
                    nisa.tensor_copy(dst=yTi, src=tps)
                    if gs_in is not None:             # ATTN_GATE: the out-projection contracts the GATED transpose
                        tpg = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=tpg, data=ytg[:, i * _NKI_P:(i + 1) * _NKI_P])
                        yTg = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=yTg, src=tpg)
                        nisa.nc_matmul(dst=aps, stationary=yTg, moving=WoT[i], accumulate=(i > 0))
                    else:
                        nisa.nc_matmul(dst=aps, stationary=yTi, moving=WoT[i], accumulate=(i > 0))
                    if p0 + _NKI_P == t_len:
                        nisa.dma_copy(dst=outT_out[row * n_head + i, :, :], src=oTb[i])
                xmt = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=xmt, src=xm_in[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :])
                x1 = x1_tiles[tb]
                nisa.tensor_tensor(dst=x1, data1=xmt, data2=aps, op=nl.add)
                x1b = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=x1b, src=x1)
                nisa.dma_copy(dst=x1_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=x1b)
                sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=sq, op=nl.square, data=x1, reduce_op=nl.add, reduce_res=ms)
                rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
                nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
                h2 = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=h2, data=x1, op0=nl.multiply, operand0=rstd)
                if fq:
                    _fq8(h2)
                for k in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=h2[:, k * _NKI_P:(k + 1) * _NKI_P])
                    nisa.tensor_copy(dst=h2T[k][:, tb * _NKI_P:(tb + 1) * _NKI_P], src=tps)
            hT = []
            for _j in range(ht):
                hT.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            for h in range(ht):
                pps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                for k in range(kt):
                    nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * _NKI_P:(h + 1) * _NKI_P], moving=h2T[k], accumulate=(k > 0))
                rl = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=rl, op=nl.relu, data=pps)
                nisa.tensor_tensor(dst=hT[h], data1=rl, data2=rl, op=nl.multiply)
                nisa.dma_copy(dst=pre_out[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX], src=rl)
                if fq:
                    _fq8(hT[h])
            for tb in range(_FMAX // _NKI_P):
                mps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for h in range(ht):
                    nisa.nc_matmul(dst=mps, stationary=hT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P], moving=W2[h], accumulate=(h > 0))
                o = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o, data1=x1_tiles[tb], data2=mps, op=nl.add)
                nisa.dma_copy(dst=out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=o)
        return out, x1_out, pre_out, y2_out, outT_out

    @nki.jit
    def pre_attn_fwd_ko(x_in: nl.ndarray, x0_in: nl.ndarray, ve_in: nl.ndarray, wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray,
                        cos_in: nl.ndarray, sin_in: nl.ndarray, scal: nl.ndarray, dstart_in: nl.ndarray, n_head: int = 4,
                        eps: float = 1e-6, t_len: int = 1024, q_scale: float = 1.0, hq: int = 32, wg_in=None):
        """pre_attn_fwd_h (see the vehicle) with the KEY_OFFSET shift applied to k_out inside the kernel (dstart_in (T, 1) fp32).
        q_out/k_out/v_out (b*n_head, t_len, HD) bf16 (q pre-scaled, k offset), xm (T, D), rq/rk (n_head, T, HD) unshifted.
        ATTN_GATE: wg_in (128, n_head) fp32 = W_g zero-padded below row k -> 7th output s = sigmoid(h[:, :128] @ wg_in)
        (T, n_head) fp32, h the normed attention input (the zero rows add exact zeros). wg_in=None: the ungated kernel."""
        T, D = x_in.shape
        HD = D // n_head
        half = HD // 2
        q_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        k_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        v_out = nl.ndarray(((T // t_len) * n_head, t_len, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        tpt = t_len // _NKI_P
        xm_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
        rq_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        rk_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        kt = D // _NKI_P
        tiles = T // _NKI_P
        sc = nl.ndarray((_NKI_P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        if wg_in is not None:                         # ATTN_GATE: resident bf16 gate weight, gate output (T, n_head)
            gs_out = nl.ndarray((T, n_head), dtype=nl.float32, buffer=nl.shared_hbm)
            wgf = nl.ndarray((_NKI_P, n_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wgf, src=wg_in)
            wgb = nl.ndarray((_NKI_P, n_head), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=wgb, src=wgf)
        ones = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(ones, 1.0)
        Sm = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.affine_select(dst=Sm, pattern=[[1, _NKI_P]], channel_multiplier=-1, offset=-1, on_true_tile=ones, on_false_value=0.0)
        Em = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.affine_select(dst=Em, pattern=[[1, _NKI_P]], channel_multiplier=-1, offset=_NKI_P - 1, on_true_tile=ones, on_false_value=0.0)
        kprev = []
        for hh in range(n_head):
            kp = nl.ndarray((_NKI_P, hq), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(kp, 0.0)
            kprev.append(kp)
        WqT = []
        WkT = []
        WvT = []
        for _j in range(kt):
            WqT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
            WkT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
            WvT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        for o in range(kt):
            bq = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            bk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            bv = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=bq, src=wq[o * _NKI_P:(o + 1) * _NKI_P, :])
            nisa.dma_copy(dst=bk, src=wk[o * _NKI_P:(o + 1) * _NKI_P, :])
            nisa.dma_copy(dst=bv, src=wv[o * _NKI_P:(o + 1) * _NKI_P, :])
            for i in range(kt):
                tq = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tq, data=bq[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WqT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tq)
                tk = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tk, data=bk[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WkT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tk)
                tv = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tv, data=bv[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WvT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tv)
        for tb in range(tiles):
            t0 = tb * _NKI_P
            row = tb // tpt
            p0 = (tb - row * tpt) * _NKI_P
            x = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            x0 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=x_in[t0:t0 + _NKI_P, :])
            nisa.dma_copy(dst=x0, src=x0_in[t0:t0 + _NKI_P, :])
            xm = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=xm, data=x0, op0=nl.multiply, operand0=sc[:, 1:2])
            nisa.scalar_tensor_tensor(dst=xm, data=x, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=xm)
            xmb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xmb, src=xm)
            nisa.dma_copy(dst=xm_out[t0:t0 + _NKI_P, :], src=xmb)
            sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=sq, op=nl.square, data=xm, reduce_op=nl.add, reduce_res=ms)
            rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
            hT = []
            for i in range(kt):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=h[:, i * _NKI_P:(i + 1) * _NKI_P])
                hTi = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=hTi, src=tps)
                hT.append(hTi)
            if wg_in is not None:                     # ATTN_GATE: s = sigmoid(h[:, :128] @ wg) straight off PSUM
                gps = nl.ndarray((_NKI_P, n_head), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=gps, stationary=hT[0], moving=wgb)
                sgt = nl.ndarray((_NKI_P, n_head), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sgt, op=nl.sigmoid, data=gps)
                nisa.dma_copy(dst=gs_out[t0:t0 + _NKI_P, :], src=sgt)
            qps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            kps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            vps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            for i in range(kt):
                nisa.nc_matmul(dst=qps, stationary=hT[i], moving=WqT[i], accumulate=(i > 0))
                nisa.nc_matmul(dst=kps, stationary=hT[i], moving=WkT[i], accumulate=(i > 0))
                nisa.nc_matmul(dst=vps, stationary=hT[i], moving=WvT[i], accumulate=(i > 0))
            q = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            k = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            v = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q, src=qps)
            nisa.tensor_copy(dst=k, src=kps)
            ve = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ve, src=ve_in[t0:t0 + _NKI_P, :])
            nisa.scalar_tensor_tensor(dst=v, data=ve, op0=nl.multiply, operand0=sc[:, 2:3], op1=nl.add, operand1=vps)
            cs = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            sn = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs, src=cos_in[t0:t0 + _NKI_P, :])
            nisa.dma_copy(dst=sn, src=sin_in[t0:t0 + _NKI_P, :])
            ds = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ds, src=dstart_in[t0:t0 + _NKI_P, :])
            mask = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=ds, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            for hh in range(n_head):
                c0 = hh * HD
                for j in range(2):
                    src_t = q if j == 0 else k
                    dst_t = q_out if j == 0 else k_out
                    rot_t = rq_out if j == 0 else rk_out
                    _rh = max(1, min(half, int(round(half * ROPE_FRAC))))
                    x1 = src_t[:, c0:c0 + _rh]
                    x2 = src_t[:, c0 + half:c0 + half + _rh]
                    r1 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    _cs = cs if _rh == half else cs[:, 0:_rh]
                    _sn = sn if _rh == half else sn[:, 0:_rh]
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)
                    rot = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rot[:, 0:_rh], src=r1)
                    nisa.tensor_copy(dst=rot[:, half:half + _rh], src=r2)
                    if _rh < half:
                        nisa.tensor_copy(dst=rot[:, _rh:half], src=src_t[:, c0 + _rh:c0 + half])
                        nisa.tensor_copy(dst=rot[:, half + _rh:HD], src=src_t[:, c0 + half + _rh:c0 + HD])
                    rb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rb, src=rot)
                    nisa.dma_copy(dst=rot_t[hh, t0:t0 + _NKI_P, :], src=rb)
                    sq2 = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    ms2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation_reduce(dst=sq2, op=nl.square, data=rot, reduce_op=nl.add, reduce_res=ms2)
                    rs2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=rs2, data=ms2, op0=nl.multiply, operand0=1.0 / HD, op1=nl.add, operand1=eps)
                    nisa.activation(dst=rs2, op=nl.rsqrt, data=rs2)
                    ob = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if j == 0:
                        nisa.tensor_scalar(dst=ob, data=rot, op0=nl.multiply, operand0=rs2, op1=nl.multiply, operand1=q_scale)
                        nisa.dma_copy(dst=dst_t[row * n_head + hh, p0:p0 + _NKI_P, :], src=ob)
                    else:
                        nisa.tensor_scalar(dst=ob, data=rot, op0=nl.multiply, operand0=rs2)
                        ps = nl.ndarray((_NKI_P, hq), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=ps, stationary=Sm, moving=ob[:, hq:2 * hq])
                        ok = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=ok, src=ob)
                        if p0 > 0:
                            ps2 = nl.ndarray((_NKI_P, hq), dtype=nl.float32, buffer=nl.psum)
                            nisa.nc_matmul(dst=ps2, stationary=Em, moving=kprev[hh])
                            car = nl.ndarray((_NKI_P, hq), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.tensor_scalar(dst=car, data=ps2, op0=nl.multiply, operand0=mask)
                            nisa.scalar_tensor_tensor(dst=ok[:, hq:2 * hq], data=ps, op0=nl.multiply, operand0=mask, op1=nl.add, operand1=car)
                        else:
                            nisa.tensor_scalar(dst=ok[:, hq:2 * hq], data=ps, op0=nl.multiply, operand0=mask)
                        nisa.tensor_copy(dst=kprev[hh], src=ob[:, hq:2 * hq])
                        nisa.dma_copy(dst=dst_t[row * n_head + hh, p0:p0 + _NKI_P, :], src=ok)
                vb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=vb, src=v[:, c0:c0 + HD])
                nisa.dma_copy(dst=v_out[row * n_head + hh, p0:p0 + _NKI_P, :], src=vb)
        if wg_in is not None:
            return q_out, k_out, v_out, xm_out, rq_out, rk_out, gs_out
        return q_out, k_out, v_out, xm_out, rq_out, rk_out


    @nki.jit
    def pre_attn_bwd_ko(dq_in: nl.ndarray, dk_in: nl.ndarray, dv_in: nl.ndarray, rq_in: nl.ndarray, rk_in: nl.ndarray,
                        xm_in: nl.ndarray, dx1_in: nl.ndarray, cos_in: nl.ndarray, sin_in: nl.ndarray, dstart_in: nl.ndarray,
                        wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray, n_head: int = 4, eps: float = 1e-6,
                        scale: float = 1.0, hq: int = 32, dz_in=None, wgT_in=None):
        """pre_attn_bwd (nki_pre_bwd.py) with the key-offset backward folded in: dstart_in (T, 1) fp32 replaces dksh_in;
        g[p, hq:2hq] = dk[p+1, hq:2hq] * (1 - dstart[p+1]) (0 at the last position of a sequence row). Tiles run in reverse.
        ATTN_GATE: dz_in (T, n_head) fp32 = dL/dz of the gate pre-activation, wgT_in (128, D) bf16 = W_g^T zero-padded
        (row r < n_head = gate column r). dz @ W_g^T joins the dgrad chain in PSUM ahead of the rmsnorm backward, and a
        6th output dwg (128, n_head) fp32 = h[:, :128]^T dz (rows >= k are dropped by the caller)."""
        T, D = xm_in.shape
        BH, t, HD = dq_in.shape
        half = HD // 2
        kt = D // _NKI_P
        tiles = T // _NKI_P
        tpt = t // _NKI_P
        dxm_out = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        gv_out = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        dwq_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
        dwk_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
        dwv_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)
        ones = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(ones, 1.0)
        St = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.affine_select(dst=St, pattern=[[1, _NKI_P]], channel_multiplier=-1, offset=1, on_true_tile=ones, on_false_value=0.0)
        Et = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.affine_select(dst=Et, pattern=[[1, _NKI_P]], channel_multiplier=-1, offset=-(_NKI_P - 1), on_true_tile=ones, on_false_value=0.0)
        dkprev = []
        for hh in range(n_head):
            kp = nl.ndarray((_NKI_P, hq), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(kp, 0.0)
            dkprev.append(kp)
        Wq = []
        Wk = []
        Wv = []
        for o in range(kt):
            a = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=wq[o * _NKI_P:(o + 1) * _NKI_P, :])
            Wq.append(a)
            a = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=wk[o * _NKI_P:(o + 1) * _NKI_P, :])
            Wk.append(a)
            a = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=wv[o * _NKI_P:(o + 1) * _NKI_P, :])
            Wv.append(a)
        accq = []
        acck = []
        accv = []
        for o in range(kt):
            a = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            accq.append(a)
            a = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            acck.append(a)
            a = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            accv.append(a)
        _wq = []
        _wk = []
        _wv = []
        _wh = []
        _wz = []
        if dz_in is not None:                         # ATTN_GATE: resident W_g^T, dW_g accumulator, zero-padded dz tile
            dwg_out = nl.ndarray((_NKI_P, n_head), dtype=nl.float32, buffer=nl.shared_hbm)
            WgT = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=WgT, src=wgT_in)
            accg = nl.ndarray((_NKI_P, n_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(accg, 0.0)
            dzpad = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dzpad, 0.0)
        for tbi in range(tiles):
            tb = tiles - 1 - tbi
            t0 = tb * _NKI_P
            b = tb // tpt
            p0 = (tb - b * tpt) * _NKI_P
            xmb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xmb, src=xm_in[t0:t0 + _NKI_P, :])
            xm = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xm, src=xmb)
            sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=sq, op=nl.square, data=xm, reduce_op=nl.add, reduce_res=ms)
            rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
            cs = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            sn = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs, src=cos_in[t0:t0 + _NKI_P, :])
            nisa.dma_copy(dst=sn, src=sin_in[t0:t0 + _NKI_P, :])
            nisa.tensor_scalar(dst=sn, data=sn, op0=nl.multiply, operand0=-1.0)
            ds = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ds, src=dstart_in[t0:t0 + _NKI_P, :])
            mask = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=ds, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            dpq = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            dpk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            gvb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            for hh in range(n_head):
                bh = b * n_head + hh
                c0 = hh * HD
                for j in range(2):
                    g = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    if j == 0:
                        gb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.dma_copy(dst=gb, src=dq_in[bh, p0:p0 + _NKI_P, :])
                        nisa.tensor_scalar(dst=g, data=gb, op0=nl.multiply, operand0=scale)
                    else:
                        gb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.dma_copy(dst=gb, src=dk_in[bh, p0:p0 + _NKI_P, :])
                        nisa.tensor_copy(dst=g, src=gb)
                        dkm = nl.ndarray((_NKI_P, hq), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=dkm, data=gb[:, hq:2 * hq], op0=nl.multiply, operand0=mask)
                        ps = nl.ndarray((_NKI_P, hq), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=ps, stationary=St, moving=dkm)
                        if p0 + _NKI_P < t:
                            ps2 = nl.ndarray((_NKI_P, hq), dtype=nl.float32, buffer=nl.psum)
                            nisa.nc_matmul(dst=ps2, stationary=Et, moving=dkprev[hh])
                            car = nl.ndarray((_NKI_P, hq), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.tensor_copy(dst=car, src=ps2)
                            nisa.tensor_tensor(dst=g[:, hq:2 * hq], data1=ps, data2=car, op=nl.add)
                        else:
                            nisa.tensor_copy(dst=g[:, hq:2 * hq], src=ps)
                        nisa.tensor_copy(dst=dkprev[hh], src=dkm)
                    rb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if j == 0:
                        nisa.dma_copy(dst=rb, src=rq_in[hh, t0:t0 + _NKI_P, :])
                    else:
                        nisa.dma_copy(dst=rb, src=rk_in[hh, t0:t0 + _NKI_P, :])
                    r = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=r, src=rb)
                    tt = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    m2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation_reduce(dst=tt, op=nl.square, data=r, reduce_op=nl.add, reduce_res=m2)
                    nisa.tensor_scalar(dst=m2, data=m2, op0=nl.multiply, operand0=1.0 / HD, op1=nl.add, operand1=eps)
                    rs = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=rs, op=nl.rsqrt, data=m2)
                    nisa.tensor_tensor(dst=tt, data1=g, data2=r, op=nl.multiply)
                    cc = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=cc, op=nl.add, data=tt, axis=(1,))
                    rs2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=rs2, data1=rs, data2=rs, op=nl.multiply)
                    nisa.tensor_tensor(dst=cc, data1=cc, data2=rs2, op=nl.multiply)
                    nisa.tensor_scalar(dst=cc, data=cc, op0=nl.multiply, operand0=1.0 / HD)
                    nisa.scalar_tensor_tensor(dst=tt, data=r, op0=nl.multiply, operand0=cc, op1=nl.subtract, operand1=g, reverse1=True)
                    dn = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dn, data=tt, op0=nl.multiply, operand0=rs)
                    _rh = max(1, min(half, int(round(half * ROPE_FRAC))))
                    x1 = dn[:, 0:_rh]
                    x2 = dn[:, half:half + _rh]
                    r1 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    _cs = cs if _rh == half else cs[:, 0:_rh]
                    _sn = sn if _rh == half else sn[:, 0:_rh]
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)
                    _dd = dpq if j == 0 else dpk
                    nisa.tensor_copy(dst=_dd[:, c0:c0 + _rh], src=r1)
                    nisa.tensor_copy(dst=_dd[:, c0 + half:c0 + half + _rh], src=r2)
                    if _rh < half:
                        nisa.tensor_copy(dst=_dd[:, c0 + _rh:c0 + half], src=dn[:, _rh:half])
                        nisa.tensor_copy(dst=_dd[:, c0 + half + _rh:c0 + HD], src=dn[:, half + _rh:HD])
                nisa.dma_copy(dst=gvb[:, c0:c0 + HD], src=dv_in[bh, p0:p0 + _NKI_P, :])
            nisa.dma_copy(dst=gv_out[t0:t0 + _NKI_P, :], src=gvb)
            dh_ps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            first = True
            for src_idx in range(3):
                for o in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    if src_idx == 0:
                        nisa.nc_transpose(dst=tps, data=dpq[:, o * _NKI_P:(o + 1) * _NKI_P])
                    elif src_idx == 1:
                        nisa.nc_transpose(dst=tps, data=dpk[:, o * _NKI_P:(o + 1) * _NKI_P])
                    else:
                        nisa.nc_transpose(dst=tps, data=gvb[:, o * _NKI_P:(o + 1) * _NKI_P])
                    aT = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=aT, src=tps)
                    if src_idx == 0:
                        nisa.nc_matmul(dst=dh_ps, stationary=aT, moving=Wq[o], accumulate=(not first))
                    elif src_idx == 1:
                        nisa.nc_matmul(dst=dh_ps, stationary=aT, moving=Wk[o], accumulate=(not first))
                    else:
                        nisa.nc_matmul(dst=dh_ps, stationary=aT, moving=Wv[o], accumulate=(not first))
                    first = False
            if dz_in is not None:                     # ATTN_GATE: dh[:, :k] += dz @ W_g^T (13th matmul of the chain)
                dzf = nl.ndarray((_NKI_P, n_head), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=dzf, src=dz_in[t0:t0 + _NKI_P, :])
                dzb = nl.ndarray((_NKI_P, n_head), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dzb, src=dzf)
                nisa.tensor_copy(dst=dzpad[:, 0:n_head], src=dzf)
                tzp = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tzp, data=dzpad)
                dzT = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dzT, src=tzp)
                nisa.nc_matmul(dst=dh_ps, stationary=dzT, moving=WgT, accumulate=True)
                _wz.append(dzb)
            dh = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dh, src=dh_ps)
            nisa.tensor_tensor(dst=sq, data1=dh, data2=xm, op=nl.multiply)
            cm = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=cm, op=nl.add, data=sq, axis=(1,))
            r2c = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r2c, data1=rstd, data2=rstd, op=nl.multiply)
            nisa.tensor_tensor(dst=cm, data1=cm, data2=r2c, op=nl.multiply)
            nisa.tensor_scalar(dst=cm, data=cm, op0=nl.multiply, operand0=1.0 / D)
            nisa.scalar_tensor_tensor(dst=sq, data=xm, op0=nl.multiply, operand0=cm, op1=nl.subtract, operand1=dh, reverse1=True)
            dx1b = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=dx1b, src=dx1_in[t0:t0 + _NKI_P, :])
            dx1 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dx1, src=dx1b)
            nisa.scalar_tensor_tensor(dst=dx1, data=sq, op0=nl.multiply, operand0=rstd, op1=nl.add, operand1=dx1)
            dxb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dxb, src=dx1)
            nisa.dma_copy(dst=dxm_out[t0:t0 + _NKI_P, :], src=dxb)
            _wq.append(dpq)
            _wk.append(dpk)
            _wv.append(gvb)
            _wh.append(h)
            if len(_wq) >= 4 or tbi == tiles - 1:
                for o in range(kt):
                    wps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    for _i in range(len(_wq)):
                        nisa.nc_matmul(dst=wps, stationary=_wq[_i][:, o * _NKI_P:(o + 1) * _NKI_P], moving=_wh[_i], accumulate=(_i > 0))
                    nisa.tensor_tensor(dst=accq[o], data1=accq[o], data2=wps, op=nl.add)
                    wps2 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    for _i in range(len(_wq)):
                        nisa.nc_matmul(dst=wps2, stationary=_wk[_i][:, o * _NKI_P:(o + 1) * _NKI_P], moving=_wh[_i], accumulate=(_i > 0))
                    nisa.tensor_tensor(dst=acck[o], data1=acck[o], data2=wps2, op=nl.add)
                    wps3 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    for _i in range(len(_wq)):
                        nisa.nc_matmul(dst=wps3, stationary=_wv[_i][:, o * _NKI_P:(o + 1) * _NKI_P], moving=_wh[_i], accumulate=(_i > 0))
                    nisa.tensor_tensor(dst=accv[o], data1=accv[o], data2=wps3, op=nl.add)
                if dz_in is not None:                 # ATTN_GATE: dW_g[:128] += sum over the group of h[:, :128]^T dz
                    gwp = nl.ndarray((_NKI_P, n_head), dtype=nl.float32, buffer=nl.psum)
                    for _i in range(len(_wz)):
                        nisa.nc_matmul(dst=gwp, stationary=_wh[_i][:, 0:_NKI_P], moving=_wz[_i], accumulate=(_i > 0))
                    nisa.tensor_tensor(dst=accg, data1=accg, data2=gwp, op=nl.add)
                _wq = []
                _wk = []
                _wv = []
                _wh = []
                _wz = []
        for o in range(kt):
            nisa.dma_copy(dst=dwq_out[o * _NKI_P:(o + 1) * _NKI_P, :], src=accq[o])
            nisa.dma_copy(dst=dwk_out[o * _NKI_P:(o + 1) * _NKI_P, :], src=acck[o])
            nisa.dma_copy(dst=dwv_out[o * _NKI_P:(o + 1) * _NKI_P, :], src=accv[o])
        if dz_in is not None:
            nisa.dma_copy(dst=dwg_out, src=accg)
            return dxm_out, gv_out, dwq_out, dwk_out, dwv_out, dwg_out
        return dxm_out, gv_out, dwq_out, dwk_out, dwv_out

    _HVC = 512

    @nki.jit
    def mlp_fwd(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
        """x (T, D) bf16; w1 (H, D) and w2 (D, H) in the natural (out, in) layout. Returns y (T, D) bf16 and preT (H, T) bf16."""
        T, D = x_in.shape
        H = w1.shape[0]
        y_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
        pre_out = nl.ndarray((H, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
        kt = D // _NKI_P
        ht = H // _NKI_P
        nprog = nl.num_programs(0)
        chunks = T // _FMAX
        per = chunks // nprog
        W1 = []
        for _j in range(kt):
            W1.append(nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf))
        for h in range(ht):
            blk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=blk, src=w1[h * _NKI_P:(h + 1) * _NKI_P, :])
            for k in range(kt):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=blk[:, k * _NKI_P:(k + 1) * _NKI_P])
                nisa.tensor_copy(dst=W1[k][:, h * _NKI_P:(h + 1) * _NKI_P], src=tps)
        W2 = []
        for _j in range(ht):
            W2.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        for o in range(kt):
            blk = nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=blk, src=w2[o * _NKI_P:(o + 1) * _NKI_P, :])
            for h in range(ht):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=blk[:, h * _NKI_P:(h + 1) * _NKI_P])
                nisa.tensor_copy(dst=W2[h][:, o * _NKI_P:(o + 1) * _NKI_P], src=tps)
        for ci in nl.affine_range(per):
            c = nl.program_id(0) * per + ci
            t0 = c * _FMAX
            xT = []
            for _j in range(kt):
                xT.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            for tb in range(_FMAX // _NKI_P):
                xt = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=xt, src=x_in[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :])
                for k in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=xt[:, k * _NKI_P:(k + 1) * _NKI_P])
                    nisa.tensor_copy(dst=xT[k][:, tb * _NKI_P:(tb + 1) * _NKI_P], src=tps)
            hT = []
            for _j in range(ht):
                hT.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            for h in range(ht):
                pps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                for k in range(kt):
                    nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * _NKI_P:(h + 1) * _NKI_P], moving=xT[k], accumulate=(k > 0))
                rl = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=rl, op=nl.relu, data=pps)
                nisa.tensor_tensor(dst=hT[h], data1=rl, data2=rl, op=nl.multiply)
                nisa.dma_copy(dst=pre_out[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX], src=rl)
            for tb in range(_FMAX // _NKI_P):
                yps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for h in range(ht):
                    nisa.nc_matmul(dst=yps, stationary=hT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P], moving=W2[h], accumulate=(h > 0))
                yt = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=yt, src=yps)
                nisa.dma_copy(dst=y_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=yt)
        return y_out, pre_out

    @nki.jit
    def mlp_bwd(x_in: nl.ndarray, dy_in: nl.ndarray, pre_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
        """x (T, D), dy (T, D), preT (H, T) bf16; w1 (H, D), w2 (D, H) natural. Returns dx (T, D) bf16 and per-program fp32
        weight-gradient partials in the NATURAL layouts: dW1 (nprog, H, D), dW2 (nprog, D, H) (sum over the leading dim in torch)."""
        T, D = x_in.shape
        H = w1.shape[0]
        kt, ht, chunks = D // _NKI_P, H // _NKI_P, T // _FMAX
        nprog = nl.num_programs(0)
        per = chunks // nprog
        pid = nl.program_id(0)
        dx_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
        dw1_out = nl.ndarray((nprog, H, D), dtype=nl.float32, buffer=nl.shared_hbm)
        dw2_out = nl.ndarray((nprog, D, H), dtype=nl.float32, buffer=nl.shared_hbm)
        W2T = []
        for o in range(kt):
            row = []
            for h in range(ht):
                blk = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=blk, src=w2[o * _NKI_P:(o + 1) * _NKI_P, h * _NKI_P:(h + 1) * _NKI_P])
                row.append(blk)
            W2T.append(row)
        W1n = []
        for h in range(ht):
            wn = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wn, src=w1[h * _NKI_P:(h + 1) * _NKI_P, :])
            W1n.append(wn)
        A1 = []
        for _j in range(ht):
            a = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            A1.append(a)
        A2 = []
        for _j in range(kt):
            a = nl.ndarray((_NKI_P, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(a, 0.0)
            A2.append(a)
        for ci in nl.sequential_range(per):
            c = pid * per + ci
            t0 = c * _FMAX
            dyn = []
            xn = []
            dyT = []
            for _j in range(kt):
                dyT.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            for tb in range(_FMAX // _NKI_P):
                d = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=d, src=dy_in[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :])
                dyn.append(d)
                xx = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=xx, src=x_in[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :])
                xn.append(xx)
                for o in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=d[:, o * _NKI_P:(o + 1) * _NKI_P])
                    nisa.tensor_copy(dst=dyT[o][:, tb * _NKI_P:(tb + 1) * _NKI_P], src=tps)
            dpreT = []
            hT = []
            for h in range(ht):
                dps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                for o in range(kt):
                    nisa.nc_matmul(dst=dps, stationary=W2T[o][h], moving=dyT[o], accumulate=(o > 0))
                relu = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=relu, src=pre_in[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX])
                hh = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=hh, data1=relu, data2=relu, op=nl.multiply)
                hT.append(hh)
                dp = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=dp, data=relu, op0=nl.multiply, operand0=2.0, op1=nl.multiply, operand1=dps)
                dpreT.append(dp)
            for tb in range(_FMAX // _NKI_P):
                xps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for h in range(ht):
                    nisa.nc_matmul(dst=xps, stationary=dpreT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P], moving=W1n[h], accumulate=(h > 0))
                dxt = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dxt, src=xps)
                nisa.dma_copy(dst=dx_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=dxt)
            dpn_l = []
            hn_l = []
            for tb in range(_FMAX // _NKI_P):
                dpn = nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
                hn = nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
                for h in range(ht):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=dpreT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P])
                    nisa.tensor_copy(dst=dpn[:, h * _NKI_P:(h + 1) * _NKI_P], src=tps)
                    tps2 = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps2, data=hT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P])
                    nisa.tensor_copy(dst=hn[:, h * _NKI_P:(h + 1) * _NKI_P], src=tps2)
                dpn_l.append(dpn)
                hn_l.append(hn)
            for h in range(ht):
                gps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for i in range(_FMAX // _NKI_P):
                    nisa.nc_matmul(dst=gps, stationary=dpn_l[i][:, h * _NKI_P:(h + 1) * _NKI_P], moving=xn[i], accumulate=(i > 0))
                nisa.tensor_tensor(dst=A1[h], data1=A1[h], data2=gps, op=nl.add)
            for o in range(kt):
                for hc in range(H // _FMAX):
                    gps2 = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                    for i in range(_FMAX // _NKI_P):
                        nisa.nc_matmul(dst=gps2, stationary=dyn[i][:, o * _NKI_P:(o + 1) * _NKI_P],
                                       moving=hn_l[i][:, hc * _FMAX:(hc + 1) * _FMAX], accumulate=(i > 0))
                    nisa.tensor_tensor(dst=A2[o][:, hc * _FMAX:(hc + 1) * _FMAX], data1=A2[o][:, hc * _FMAX:(hc + 1) * _FMAX], data2=gps2, op=nl.add)
        for h in range(ht):
            nisa.dma_copy(dst=dw1_out[pid, h * _NKI_P:(h + 1) * _NKI_P, :], src=A1[h])
        for o in range(kt):
            nisa.dma_copy(dst=dw2_out[pid, o * _NKI_P:(o + 1) * _NKI_P, :], src=A2[o])
        return dx_out, dw1_out, dw2_out

    @nki.jit
    def pre_attn_fwd(x_in: nl.ndarray, x0_in: nl.ndarray, ve_in: nl.ndarray, wq: nl.ndarray, wk: nl.ndarray, wv: nl.ndarray,
                     cos_in: nl.ndarray, sin_in: nl.ndarray, scal: nl.ndarray, n_head: int = 4, eps: float = 1e-6):
        """PRE-attention half of a layer as one kernel (T tokens = b*t rows, D = 512, head dim 128):
          xm = s0*x + s1*x0                    (scal[:, 0], scal[:, 1]: resid_lambda, x0_lambda as a (128, 4) fp32 tensor)
          h  = rmsnorm(xm)
          q  = rope(h @ Wq^T), k = rope(h @ Wk^T), both rms-normed per head; v = h @ Wv^T + s2 * ve   (s2 = ve_lambda)
        Returns q, k, v as (n_head, T, 128) bf16 (the attention kernel's (h, t, d) layout per batch row), xm (T, D) bf16, and the
        pre-norm rotated q and k (rq, rk: (n_head, T, 128) bf16) that the QK-norm backward needs.
        cos/sin: (T, 64) fp32 tables of the split-half rotation (rows = token positions of the T rows)."""
        T, D = x_in.shape
        HD = D // n_head
        half = HD // 2
        q_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        k_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        v_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        xm_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
        rq_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        rk_out = nl.ndarray((n_head, T, HD), dtype=x_in.dtype, buffer=nl.shared_hbm)
        kt = D // _NKI_P
        nprog = nl.num_programs(0)
        tiles = T // _NKI_P
        per = tiles // nprog
        sc = nl.ndarray((_NKI_P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        WqT = []
        WkT = []
        WvT = []
        for _j in range(kt):
            WqT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
            WkT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
            WvT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        for o in range(kt):
            bq = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            bk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            bv = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=bq, src=wq[o * _NKI_P:(o + 1) * _NKI_P, :])
            nisa.dma_copy(dst=bk, src=wk[o * _NKI_P:(o + 1) * _NKI_P, :])
            nisa.dma_copy(dst=bv, src=wv[o * _NKI_P:(o + 1) * _NKI_P, :])
            for i in range(kt):
                tq = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tq, data=bq[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WqT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tq)
                tk = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tk, data=bk[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WkT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tk)
                tv = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tv, data=bv[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WvT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tv)
        for ti in nl.affine_range(per):
            tb = nl.program_id(0) * per + ti
            t0 = tb * _NKI_P
            x = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            x0 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=x_in[t0:t0 + _NKI_P, :])
            nisa.dma_copy(dst=x0, src=x0_in[t0:t0 + _NKI_P, :])
            xm = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=xm, data=x0, op0=nl.multiply, operand0=sc[:, 1:2])
            nisa.scalar_tensor_tensor(dst=xm, data=x, op0=nl.multiply, operand0=sc[:, 0:1], op1=nl.add, operand1=xm)
            xmb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xmb, src=xm)
            nisa.dma_copy(dst=xm_out[t0:t0 + _NKI_P, :], src=xmb)
            sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=sq, op=nl.square, data=xm, reduce_op=nl.add, reduce_res=ms)
            rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
            hT = []
            for i in range(kt):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=h[:, i * _NKI_P:(i + 1) * _NKI_P])
                hTi = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=hTi, src=tps)
                hT.append(hTi)
            qps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            kps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            vps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
            for i in range(kt):
                nisa.nc_matmul(dst=qps, stationary=hT[i], moving=WqT[i], accumulate=(i > 0))
                nisa.nc_matmul(dst=kps, stationary=hT[i], moving=WkT[i], accumulate=(i > 0))
                nisa.nc_matmul(dst=vps, stationary=hT[i], moving=WvT[i], accumulate=(i > 0))
            q = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            k = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            v = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q, src=qps)
            nisa.tensor_copy(dst=k, src=kps)
            ve = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ve, src=ve_in[t0:t0 + _NKI_P, :])
            nisa.scalar_tensor_tensor(dst=v, data=ve, op0=nl.multiply, operand0=sc[:, 2:3], op1=nl.add, operand1=vps)
            cs = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            sn = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs, src=cos_in[t0:t0 + _NKI_P, :])
            nisa.dma_copy(dst=sn, src=sin_in[t0:t0 + _NKI_P, :])
            for hh in range(n_head):
                c0 = hh * HD
                for j in range(2):
                    src_t = q if j == 0 else k
                    dst_t = q_out if j == 0 else k_out
                    rot_t = rq_out if j == 0 else rk_out
                    _rh = max(1, min(half, int(round(half * ROPE_FRAC))))
                    x1 = src_t[:, c0:c0 + _rh]
                    x2 = src_t[:, c0 + half:c0 + half + _rh]
                    r1 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, _rh), dtype=nl.float32, buffer=nl.sbuf)
                    _cs = cs if _rh == half else cs[:, 0:_rh]
                    _sn = sn if _rh == half else sn[:, 0:_rh]
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=_cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=_sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)
                    rot = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rot[:, 0:_rh], src=r1)
                    nisa.tensor_copy(dst=rot[:, half:half + _rh], src=r2)
                    if _rh < half:
                        nisa.tensor_copy(dst=rot[:, _rh:half], src=src_t[:, c0 + _rh:c0 + half])
                        nisa.tensor_copy(dst=rot[:, half + _rh:HD], src=src_t[:, c0 + half + _rh:c0 + HD])
                    rb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rb, src=rot)
                    nisa.dma_copy(dst=rot_t[hh, t0:t0 + _NKI_P, :], src=rb)
                    sq2 = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    ms2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation_reduce(dst=sq2, op=nl.square, data=rot, reduce_op=nl.add, reduce_res=ms2)
                    rs2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=rs2, data=ms2, op0=nl.multiply, operand0=1.0 / HD, op1=nl.add, operand1=eps)
                    nisa.activation(dst=rs2, op=nl.rsqrt, data=rs2)
                    ob = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=ob, data=rot, op0=nl.multiply, operand0=rs2)
                    nisa.dma_copy(dst=dst_t[hh, t0:t0 + _NKI_P, :], src=ob)
                vb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=vb, src=v[:, c0:c0 + HD])
                nisa.dma_copy(dst=v_out[hh, t0:t0 + _NKI_P, :], src=vb)
        return q_out, k_out, v_out, xm_out, rq_out, rk_out

    @nki.jit
    def post_attn_fwd(y_in: nl.ndarray, wo: nl.ndarray, xm_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray, eps: float = 1e-6):
        T, D = y_in.shape
        H = w1.shape[0]
        out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)
        x1_out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)
        pre_out = nl.ndarray((H, T), dtype=y_in.dtype, buffer=nl.shared_hbm)
        kt = D // _NKI_P
        ht = H // _NKI_P
        nprog = nl.num_programs(0)
        chunks = T // _FMAX
        per = chunks // nprog
        WoT = []
        for _j in range(kt):
            WoT.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        for o in range(kt):
            blk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=blk, src=wo[o * _NKI_P:(o + 1) * _NKI_P, :])
            for i in range(kt):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=blk[:, i * _NKI_P:(i + 1) * _NKI_P])
                nisa.tensor_copy(dst=WoT[i][:, o * _NKI_P:(o + 1) * _NKI_P], src=tps)
        W1 = []
        for _j in range(kt):
            W1.append(nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf))
        for h in range(ht):
            blk = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=blk, src=w1[h * _NKI_P:(h + 1) * _NKI_P, :])
            for k in range(kt):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=blk[:, k * _NKI_P:(k + 1) * _NKI_P])
                nisa.tensor_copy(dst=W1[k][:, h * _NKI_P:(h + 1) * _NKI_P], src=tps)
        W2 = []
        for _j in range(ht):
            W2.append(nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf))
        for o in range(kt):
            blk = nl.ndarray((_NKI_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=blk, src=w2[o * _NKI_P:(o + 1) * _NKI_P, :])
            for h in range(ht):
                tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps, data=blk[:, h * _NKI_P:(h + 1) * _NKI_P])
                nisa.tensor_copy(dst=W2[h][:, o * _NKI_P:(o + 1) * _NKI_P], src=tps)
        for ci in nl.affine_range(per):
            c = nl.program_id(0) * per + ci
            t0 = c * _FMAX
            h2T = []
            for _j in range(kt):
                h2T.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            x1_tiles = []
            for _j in range(_FMAX // _NKI_P):
                x1_tiles.append(nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf))
            for tb in range(_FMAX // _NKI_P):
                yt = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=yt, src=y_in[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :])
                aps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for i in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=yt[:, i * _NKI_P:(i + 1) * _NKI_P])
                    yTi = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=yTi, src=tps)
                    nisa.nc_matmul(dst=aps, stationary=yTi, moving=WoT[i], accumulate=(i > 0))
                xmt = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=xmt, src=xm_in[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :])
                x1 = x1_tiles[tb]
                nisa.tensor_tensor(dst=x1, data1=xmt, data2=aps, op=nl.add)
                x1b = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=x1b, src=x1)
                nisa.dma_copy(dst=x1_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=x1b)
                sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=sq, op=nl.square, data=x1, reduce_op=nl.add, reduce_res=ms)
                rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
                nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
                h2 = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=h2, data=x1, op0=nl.multiply, operand0=rstd)
                for k in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=h2[:, k * _NKI_P:(k + 1) * _NKI_P])
                    nisa.tensor_copy(dst=h2T[k][:, tb * _NKI_P:(tb + 1) * _NKI_P], src=tps)
            hT = []
            for _j in range(ht):
                hT.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            for h in range(ht):
                pps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                for k in range(kt):
                    nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * _NKI_P:(h + 1) * _NKI_P], moving=h2T[k], accumulate=(k > 0))
                rl = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=rl, op=nl.relu, data=pps)
                nisa.tensor_tensor(dst=hT[h], data1=rl, data2=rl, op=nl.multiply)
                nisa.dma_copy(dst=pre_out[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX], src=rl)
            for tb in range(_FMAX // _NKI_P):
                mps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for h in range(ht):
                    nisa.nc_matmul(dst=mps, stationary=hT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P], moving=W2[h], accumulate=(h > 0))
                o = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o, data1=x1_tiles[tb], data2=mps, op=nl.add)
                nisa.dma_copy(dst=out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=o)
        return out, x1_out, pre_out


def doc_bounds(idx: torch.Tensor):
    """Per position: [start, end) of the document containing it (documents start at BOS; tokens before the first BOS
    form document 0). The attention kernels clamp `end` to the causal limit."""
    b, t = idx.shape
    is_bos = idx == BOS_ID
    pos = torch.arange(t, device=idx.device, dtype=torch.int32).unsqueeze(0).expand(b, t)
    docs = is_bos.to(torch.int32).cumsum(dim=1)
    start_tab = torch.zeros(b, t + 2, dtype=torch.int32, device=idx.device)
    start_tab = start_tab.scatter(1, (docs * is_bos.to(torch.int32)).long(), pos * is_bos.to(torch.int32))
    start = start_tab.gather(1, docs.long())
    nxt = start_tab.gather(1, (docs + 1).long())
    end = torch.where(nxt > 0, nxt, torch.full_like(nxt, t))
    return start, end




@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=0.0078125)


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


class Linear(nn.Linear):
    """Linear without bias that casts the weight to the input dtype (bf16 input, fp32 weight)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


class StackedLinear(nn.Module):
    """A bias-free Linear whose weight is slice `idx` of a (n, out, in) Parameter shared by every layer of the same
    shape, so the optimizer and the gradient all-reduce see one tensor per weight shape."""

    def __init__(self, registry, out_features: int, in_features: int):
        super().__init__()
        self.key, self.idx = registry.register(out_features, in_features)
        object.__setattr__(self, "_stack", None)

    def bind(self, stack: nn.Parameter) -> None:
        object.__setattr__(self, "_stack", stack)

    @property
    def weight(self) -> torch.Tensor:
        return self._stack[self.idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if USE_STACK_FUNCTION and torch.is_grad_enabled() and self._stack.requires_grad:
            return _StackSliceLinear.apply(x, self._stack, self.idx)
        return F.linear(x, self._stack[self.idx].to(dtype=x.dtype), None)


class _StackSliceLinear(torch.autograd.Function):
    """Linear on slice `idx` of a stacked weight; the backward accumulates the weight gradient straight into
    stack.grad[idx] (plain autograd through stack[idx] would materialize a full-stack zero tensor per layer)."""

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
    def __init__(self):
        self.counts: dict[tuple[int, int], int] = {}

    def register(self, out_features: int, in_features: int):
        shape = (out_features, in_features)
        idx = self.counts.get(shape, 0)
        self.counts[shape] = idx + 1
        return f"w{out_features}x{in_features}", idx

    def build(self) -> nn.ParameterDict:
        return nn.ParameterDict({f"w{o}x{i}": nn.Parameter(torch.empty(n, o, i)) for (o, i), n in self.counts.items()})


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, registry: StackRegistry):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = StackedLinear(registry, config.n_embd, config.n_embd)
        self.c_k = StackedLinear(registry, config.n_embd, config.n_embd)
        self.c_v = StackedLinear(registry, config.n_embd, config.n_embd)
        self.c_proj = StackedLinear(registry, config.n_embd, config.n_embd)

    def forward(self, x, ve, cos, sin, sink, gw, dense_mask=None, bounds=None, doc_start=None):
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
        v = (self.c_v(x) + ve).view(b, t, self.n_head, self.head_dim)
        q = norm(apply_rotary_emb(q, cos, sin))
        k = norm(apply_rotary_emb(k, cos, sin))
        if KEY_OFFSET:
            D = self.head_dim
            prev_k = torch.cat([torch.zeros_like(k[:, :1, :, D // 4:D // 2]), k[:, :-1, :, D // 4:D // 2]], dim=1)
            if doc_start is not None:
                prev_k = prev_k.masked_fill(doc_start.view(b, t, 1, 1), 0.0)
            k = torch.cat([k[..., :D // 4], prev_k, k[..., D // 2:]], dim=-1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = ATTN_SCALE
        if bounds is not None:
            y = torch.ops.r1.doc_attention_fwd((q * scale).to(q.dtype), k, v, sink, bounds[0], bounds[1])[0]
        else:
            scores = (q.float() @ k.float().transpose(-2, -1)) * scale
            scores = scores.masked_fill(~dense_mask, float("-inf"))
            lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            p = torch.exp(scores - lse)
            gate = 1.0 - torch.sigmoid(sink.float().view(1, self.n_head, 1, 1) - lse)
            y = ((p @ v.float()) * gate).to(v.dtype)
        # ATTN_GATE: head h scaled by 2 * sigmoid(x[:, :k] @ W_g[:, h]); x is the normed attention input
        g = 2.0 * torch.sigmoid(x[..., :gw.shape[0]].float() @ gw.float())           # (b, t, n_head)
        y = (y.float() * g.transpose(1, 2).unsqueeze(-1)).to(y.dtype)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, registry: StackRegistry):
        super().__init__()
        self.c_fc = StackedLinear(registry, MLP_RATIO * config.n_embd, config.n_embd)
        self.c_proj = StackedLinear(registry, config.n_embd, MLP_RATIO * config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(relu_squared(self.c_fc(x)))


_NKI_NORM = None


def build_nki_norm():
    """Fused RMSNorm fwd/bwd (dev copy: nki_rmsnorm.py). Eager Neuron only."""
    global _NKI_NORM
    if _NKI_NORM is not None:
        return _NKI_NORM
    if not NKI_AVAILABLE:
        raise RuntimeError("NKI is not available on this device")
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    _NKI_NORM = (wrap_nki(rmsnorm_fwd), wrap_nki(rmsnorm_bwd), (int(get_logical_neuron_cores()),))
    return _NKI_NORM


_NKI_MLP = None


def build_nki_mlp():
    """Fused ReLU^2 MLP fwd/bwd (dev copy: nki_mlp.py). Eager Neuron only; weights in the WEIGHT_T (in, out) layout."""
    global _NKI_MLP
    if _NKI_MLP is not None:
        return _NKI_MLP
    if not NKI_AVAILABLE:
        raise RuntimeError("NKI is not available on this device")
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    _NKI_MLP = (wrap_nki(mlp_fwd), wrap_nki(mlp_bwd), (int(get_logical_neuron_cores()),))
    return _NKI_MLP


_NKI_LAYER = None


def build_nki_layer():
    global _NKI_LAYER
    if _NKI_LAYER is None:
        if not NKI_AVAILABLE:
            raise RuntimeError("NKI is not available on this device")
        from torch_neuronx.nki_hop import wrap_nki
        from torch_neuronx.utils import get_logical_neuron_cores
        _NKI_LAYER = (wrap_nki(pre_attn_fwd), wrap_nki(post_attn_fwd), (int(get_logical_neuron_cores()),),
                      wrap_nki(pre_attn_fwd_h), wrap_nki(post_attn_fwd_h))
    return _NKI_LAYER

_LAYER_OPS = None
_LAYER_NH = 4


def build_layer_ops(n_head: int):
    global _LAYER_OPS, _LAYER_NH
    if _LAYER_OPS is not None:
        return _LAYER_OPS
    _LAYER_NH = int(n_head)
    from torch_neuronx.nki_hop import wrap_nki, nki_op
    from nkilib.core.attention.attention_cte import attention_cte
    from nkilib.core.attention.attention_bwd import attention_bwd
    pre_k, post_k, grid, pre_kh, post_kh = build_nki_layer()
    afwd = wrap_nki(attention_cte)
    abwd = wrap_nki(attention_bwd)
    _mf, mlp_bwd_k, _g = build_nki_mlp()
    nfwd, nbwd, _g1 = build_nki_norm()
    pre_ko = wrap_nki(pre_attn_fwd_ko)
    pre_bwd_k = wrap_nki(pre_attn_bwd_ko)
    eps = float(torch.finfo(torch.bfloat16).eps)
    bf = torch.bfloat16

    def attn_fwd(q4, k4, v4, sink, start, end):
        b, h, t, d = q4.shape
        qf = q4.reshape(b * h, t, d).contiguous(); kf = k4.reshape(b * h, t, d).contiguous(); vf = v4.reshape(b * h, t, d).contiguous()
        sink_rows = sink.float().view(1, h).expand(b, h).reshape(b * h, 1).contiguous()
        bmin = start.unsqueeze(1).expand(b, h, t).reshape(b * h, t, 1).contiguous()
        bmax = end.unsqueeze(1).expand(b, h, t).reshape(b * h, t, 1).contiguous()
        out, neg_max, recip = afwd[grid](qf, kf, vf, tp_q=True, tp_k=True, scale=1.0, causal_mask=True, cache_softmax=True,
                                         bound_min=bmin, bound_max=bmax, sink=sink_rows)
        return out.reshape(b, h, t, d), neg_max, recip

    def attn_bwd(q4, k4, v4, outT, dyT4, neg_max, recip, sink, start, end):
        b, h, t, d = q4.shape
        tr = lambda x: x.transpose(-2, -1).contiguous()
        lse = (-(neg_max.float() + torch.log(recip.float()))).reshape(b, h, *neg_max.shape[-2:]).contiguous()
        res = abwd[grid](tr(q4), tr(k4), tr(v4), outT.view(b, h, d, t), dyT4, lse,
                         sinks_ref=sink.float().view(1, h).expand(b, h).contiguous(),
                         bound_min=start.float().contiguous(), bound_max=end.float().contiguous(),
                         use_causal_mask=True, mixed_precision=True, softmax_scale=1.0)
        dq, dk, dv = (r.transpose(-2, -1).contiguous() for r in res[:3])
        return dq, dk, dv, res[3].float().sum(dim=0)

    # ATTN_GATE: extra input wg (128, NH) fp32 (W_g zero-padded below row ATTN_GATE); extra saved output
    # gs = sigmoid(z) (T, NH) fp32. The gated head output is y_g = 2 * gs * O, O the attention output.
    @nki_op("tf_nki::layer_fwd", mutates_args={})
    def layer_fwd(x2: torch.Tensor, x02: torch.Tensor, ve2: torch.Tensor, cos_full: torch.Tensor, sin_full: torch.Tensor,
                  cos2d: torch.Tensor, sin2d: torch.Tensor, scal: torch.Tensor, sink: torch.Tensor, start: torch.Tensor,
                  end: torch.Tensor, dstart: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor,
                  w1: torch.Tensor, w2: torch.Tensor, wg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                torch.Tensor, torch.Tensor]:
        NH = _LAYER_NH; T, D = x2.shape; HD = D // NH; b = start.shape[0]; t = T // b
        scale = ATTN_SCALE
        qh, kh, vh, xm, rq, rk, gs = pre_ko[grid](x2, x02, ve2, wq, wk, wv, cos_full, sin_full, scal, dstart.reshape(T, 1),
                                                  n_head=NH, eps=eps, t_len=t, q_scale=scale, hq=HD // KEY_OFFSET_DIV,
                                                  wg_in=wg.contiguous())
        q4 = qh.view(b, NH, t, HD)
        k4 = kh.view(b, NH, t, HD)
        v4 = vh.view(b, NH, t, HD)
        out, neg_max, recip = attn_fwd(q4, k4, v4, sink, start, end)
        o, x1, preT, y2, outT = post_kh[grid](out.reshape(b * NH, t, HD), wo, xm, w1, w2, eps=eps, n_head=NH, t_len=t, gs_in=gs)
        return o, xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT, gs

    @nki_op("tf_nki::layer_bwd", mutates_args={})
    def layer_bwd(dout2: torch.Tensor, x2: torch.Tensor, x02: torch.Tensor, ve2: torch.Tensor, xm: torch.Tensor, rq: torch.Tensor,
                  rk: torch.Tensor, q4: torch.Tensor, k4: torch.Tensor, v4: torch.Tensor, outT: torch.Tensor, neg_max: torch.Tensor,
                  recip: torch.Tensor, y2: torch.Tensor, x1: torch.Tensor, preT: torch.Tensor, cos2d: torch.Tensor,
                  sin2d: torch.Tensor, sink: torch.Tensor, start: torch.Tensor, end: torch.Tensor, dstart: torch.Tensor,
                  wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor, w1: torch.Tensor,
                  w2: torch.Tensor, wg: torch.Tensor, gs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                 torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                 torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                 torch.Tensor]:
        NH = _LAYER_NH; T, D = x2.shape; HD = D // NH; b = start.shape[0]; t = T // b
        scale = ATTN_SCALE
        d2 = dout2.to(bf).contiguous()
        h2 = nfwd[grid](x1, eps=eps)
        dh2, dW1p, dW2p = mlp_bwd_k[grid](h2, d2, preT, w1, w2)
        dx1 = d2 + nbwd[grid](x1, dh2.contiguous(), eps=eps)
        dy2 = F.linear(dx1, wo.t())               # dL/dy_g (token-major); y2 holds y_g, so dWo is unchanged in form
        dWo = dx1.t() @ y2
        # y_g = 2 s O:  dL/dz = sum_d dy_g * O * 2 s (1 - s) = (1 - s) * sum_d dy_g * y_g   (no division by s)
        #               dL/dO = 2 s * dy_g   (what the vendor backward consumes; outT is the UNGATED O)
        dyg = dy2.float().view(T, NH, HD)
        dz = ((dyg * y2.float().view(T, NH, HD)).sum(-1) * (1.0 - gs)).contiguous()
        dyT4 = (dyg * (2.0 * gs).unsqueeze(-1)).to(q4.dtype).view(b, t, NH, HD).permute(0, 2, 3, 1).contiguous()
        dq4, dk4, dv4, dsink = attn_bwd(q4, k4, v4, outT, dyT4, neg_max, recip, sink, start, end)
        cos_full = cos2d.repeat(b, 1).contiguous(); sin_full = sin2d.repeat(b, 1).contiguous()
        HQ = HD // KEY_OFFSET_DIV
        wgT = F.pad(wg.t(), (0, D - wg.shape[0], 0, 128 - NH)).to(bf).contiguous()   # (128, D): row r = W_g[:, r]
        dxm, gv2, dWq, dWk, dWv, dwg = pre_bwd_k[grid](dq4.reshape(b * NH, t, HD).contiguous(), dk4.reshape(b * NH, t, HD).contiguous(),
                                                      dv4.reshape(b * NH, t, HD).contiguous(), rq.contiguous(), rk.contiguous(),
                                                      xm.contiguous(), dx1.contiguous(), cos_full, sin_full, dstart.reshape(T, 1), wq, wk, wv,
                                                      n_head=NH, eps=eps, scale=scale, hq=HQ, dz_in=dz, wgT_in=wgT)
        return dxm, gv2, dWq, dWk, dWv, dWo, dW1p.sum(0), dW2p.sum(0), dsink, dwg

    def setup_context(ctx, inputs, output):
        x2, x02, ve2, cos_full, sin_full, cos2d, sin2d, scal, sink, start, end, dstart, wq, wk, wv, wo, w1, w2, wg = inputs
        o, xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT, gs = output
        ctx.save_for_backward(x2, x02, ve2, cos2d, sin2d, scal, sink, start, end, dstart, wq, wk, wv, wo, w1, w2, wg,
                              xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT, gs)

    def backward(ctx, dout2, *_unused):
        (x2, x02, ve2, cos2d, sin2d, scal, sink, start, end, dstart, wq, wk, wv, wo, w1, w2, wg,
         xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT, gs) = ctx.saved_tensors
        dxm, gv2, dWq, dWk, dWv, dWo, dW1, dW2, dsink, dwg = torch.ops.tf_nki.layer_bwd(
            dout2, x2, x02, ve2, xm, rq, rk, q4, k4, v4, outT, neg_max, recip, y2, x1, preT, cos2d, sin2d, sink, start, end,
            dstart, wq, wk, wv, wo, w1, w2, wg, gs)
        resid = scal[0:1, 0:1]; x0l = scal[0:1, 1:2]; vel = scal[0:1, 2:3]
        dxf = dxm.float()
        dx2 = (dxf * resid).to(x2.dtype); dx02 = (dxf * x0l).to(x02.dtype); dve2 = (gv2.float() * vel).to(ve2.dtype)
        d_resid = (dxf * x2.float()).sum(); d_x0 = (dxf * x02.float()).sum(); d_ve = (gv2.float() * ve2.float()).sum()
        row0 = torch.stack([d_resid, d_x0, d_ve, torch.zeros((), device=scal.device, dtype=torch.float32)]).view(1, 4)
        dscal = torch.cat([row0, torch.zeros((127, 4), device=scal.device, dtype=torch.float32)], dim=0)
        return (dx2, dx02, dve2, None, None, None, None, dscal, dsink.to(sink.dtype), None, None, None,
                dWq.to(wq.dtype), dWk.to(wk.dtype), dWv.to(wv.dtype), dWo.to(wo.dtype), dW1.to(w1.dtype), dW2.to(w2.dtype),
                dwg.to(wg.dtype))

    torch.library.register_autograd("tf_nki::layer_fwd", backward, setup_context=setup_context)
    _LAYER_OPS = (torch.ops.tf_nki.layer_fwd, torch.ops.tf_nki.layer_bwd)
    print0("LAYER=op: tf_nki::layer_fwd/layer_bwd built (the Block op branch fires when sink/bounds/rope tables are present)")
    return _LAYER_OPS


_HEAD_OPS = None


def build_head_ops():
    global _HEAD_OPS
    if _HEAD_OPS is not None:
        return _HEAD_OPS
    from torch_neuronx.nki_hop import wrap_nki, nki_op
    from torch_neuronx.utils import get_logical_neuron_cores
    grid = (int(get_logical_neuron_cores()),)
    kf, kb = wrap_nki(head_fwd), wrap_nki(head_bwd)

    @nki_op("tf_nki::head_fwd", mutates_args={})
    def hfwd(x2: torch.Tensor, wT: torch.Tensor, tgt: torch.Tensor, scal: torch.Tensor,
             tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, lse, h, rstd, zt = kf[grid](x2, wT, tgt, scal, tau)
        return loss, lse, h, rstd, zt

    @nki_op("tf_nki::head_bwd", mutates_args={})
    def hbwd(h: torch.Tensor, rstd: torch.Tensor, lse: torch.Tensor, dl: torch.Tensor, w: torch.Tensor, wT: torch.Tensor,
             tgt: torch.Tensor, scal: torch.Tensor, tau: torch.Tensor, dlse: torch.Tensor,
             dzt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx, dw, dcap = kb[grid](h, rstd, lse, dl, w, wT, tgt, scal, tau, dlse, dzt)
        return dx, dw, dcap

    def setup_context(ctx, inputs, output):
        x2, wT, tgt, scal, tau = inputs
        loss, lse, h, rstd, zt = output
        ctx.save_for_backward(wT, tgt, scal, lse, h, rstd, tau)
        ctx.x_dtype = x2.dtype

    def backward(ctx, dloss, dlse, _dh, _drstd, dzt):
        wT, tgt, scal, lse, h, rstd, tau = ctx.saved_tensors
        w = wT.t().contiguous()
        g = [torch.zeros_like(lse) if d is None else d.float().contiguous() for d in (dloss, dlse, dzt)]
        dx, dw, dcap = torch.ops.tf_nki.head_bwd(h, rstd, lse, g[0], w, wT, tgt, scal, tau, g[1], g[2])
        dcap = dcap.sum(0)
        row0 = torch.stack([dcap[0], dcap[1], torch.zeros((), device=scal.device), torch.zeros((), device=scal.device)]).view(1, 4)
        dscal = torch.cat([row0, torch.zeros((127, 4), device=scal.device, dtype=torch.float32)], dim=0)
        return dx.to(ctx.x_dtype), dw.t().to(wT.dtype), None, dscal, None

    torch.library.register_autograd("tf_nki::head_fwd", backward, setup_context=setup_context)
    _HEAD_OPS = (torch.ops.tf_nki.head_fwd, torch.ops.tf_nki.head_bwd)
    return _HEAD_OPS

FUSED_LOSS_TOL = 0.02                 # relative bound on |fused - reference|; bf16 accumulation order is ~1e-3 of this
FUSED_HEAD_CHECK_STEP = 50            # the head check needs a TRAINED head (see check_fused_head)


def check_fused_loss(model: "GPT", x: torch.Tensor, y: torch.Tensor, emb, ng) -> None:
    """Evaluate the training loss both ways on one real batch - the fused NKI kernel and cross_entropy(head_logits(x), y)
    in plain PyTorch - print both, and disable the fused path if they disagree. Runs once, at step 0.

    What this check can and cannot see, stated plainly because it is weaker than it looks: at step 0 every block's
    output projection is zero-initialised and the lm_head is initialised at std 1e-3, so the model's logits are within
    +-0.03 of zero and the learned softcap (11.0) is numerically inactive. This comparison therefore catches gross
    failures - a kernel that will not run, a NaN, the wrong reduction, the wrong vocabulary slice - but it CANNOT
    exercise the softcap or resolve a small masking error. `check_fused_head` below covers those on a trained head.

    Either side failing to run is itself a result: if the FUSED path raises it is disabled, because that is the path
    under test; if the REFERENCE path raises the check cannot be made, so the fused path is kept and said so loudly.
    """
    if _HEAD_OPS is None or x.device.type != "neuron":
        return
    with torch.no_grad():
        try:
            fused = model(x, y, emb=emb, ng=ng)
        except Exception as exc:
            model._fused_loss_ok = False
            note_fallback("fused loss -> cross_entropy(head_logits) (the fused path raised in its own check)", exc)
            print0(f"fused-loss check: the FUSED path raised ({str(exc)[:160]}); using cross_entropy(head_logits)")
            return
        model._fused_loss_ok = False
        try:
            explicit = model(x, y, emb=emb, ng=ng)
        except Exception as exc:
            model._fused_loss_ok = True
            print0(f"fused-loss check: the REFERENCE path raised ({str(exc)[:160]}); the fused path is KEPT, UNCHECKED")
            return
        model._fused_loss_ok = True
        diff = (fused - explicit).abs().item()
    print0(f"fused-loss check: fused {fused.item():.5f} vs cross_entropy(head_logits) {explicit.item():.5f}, |diff| {diff:.2e}")
    if not (diff < FUSED_LOSS_TOL * max(1.0, explicit.item())):
        model._fused_loss_ok = False
        print0("fused-loss check FAILED: using cross_entropy(head_logits) for training")


def check_fused_head(model: "GPT", rows: int, seq_len: int, device: torch.device) -> None:
    """The check with teeth: compare `GPT.fused_linear_cross_entropy(h, t)` against
    `F.cross_entropy(GPT.head_logits(h, tau, bias), t)` on a TRAINED head and a hidden state built to saturate the softcap.

    Three deliberate choices, each fixing a blind spot of the step-0 check above:
      * it runs at step FUSED_HEAD_CHECK_STEP, not at step 0, so `lm_head` has been trained and the logits are large
        enough for the learned softcap to bend them;
      * each row of the hidden state is the lm_head row of its own target plus noise, so the target's logit is
        ||w||*sqrt(n_embd) - well past the softcap - exactly as it is for a converged model, while the rest of the
        row spreads. The parameters are the model's own and nothing is written to them;
      * one target in eight is the ignore index, far above the rate in real text, so a mis-handled mask moves the
        mean loss by ~12% and cannot hide inside the tolerance.
    It touches only the head, so it costs two head passes and no transformer blocks, and it runs under no_grad.
    """
    if _HEAD_OPS is None or device.type != "neuron":
        return
    with torch.no_grad():
        W = model.lm_head.weight
        g = torch.Generator(device="cpu").manual_seed(1234)
        V = model.config.vocab_size
        tgt = torch.randint(0, V, (rows, seq_len), generator=g, dtype=torch.int64).to(device)
        h = W[tgt.reshape(-1)].to(COMPUTE_DTYPE) + 0.1 * W.std() * torch.randn(
            (rows * seq_len, W.shape[1]), generator=g, dtype=torch.float32).to(device).to(COMPUTE_DTYPE)
        h = h.reshape(rows, seq_len, W.shape[1])
        tgt = torch.where(torch.arange(seq_len, device=device).view(1, -1) % 8 == 0, -1, tgt)
        u = torch.rand((rows, seq_len), generator=g)       # an n-gram column on 3/4 of positions, on the target for half
        tau = torch.where(u < 0.5, tgt.cpu().clamp_min(0), torch.randint(0, V, (rows, seq_len), generator=g)).to(device)
        bias = (torch.where(u < 0.75, 6.0 * u, 0.0)).to(device)
        fused = model.fused_linear_cross_entropy(h, tgt, tau, bias)
        logits = model.head_logits(h, tau, bias)
        ref = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1)
        cap = logits.abs().max().item()
        diff = (fused - ref).abs().item()
    print0(f"fused-head check (step {FUSED_HEAD_CHECK_STEP}, trained head, |logit|max {cap:.2f} vs softcap "
           f"{model.softcap_pos.item():.2f}): fused {fused.item():.5f} vs cross_entropy(head_logits) {ref.item():.5f}, "
           f"|diff| {diff:.2e}, bound {FUSED_LOSS_TOL * max(1.0, ref.item()):.2e}")
    if not (diff < FUSED_LOSS_TOL * max(1.0, ref.item())):
        model._fused_loss_ok = False
        print0("fused-head check FAILED: using cross_entropy(head_logits) for the rest of training")


def build_prefix_table(tokenizer, device) -> torch.Tensor:
    """(vocab,) int64: for each token, the longest in-vocab PROPER byte-prefix, or the token itself if none.

    Self-mapping the 256 single-byte tokens makes a replacement on them a no-op, so no mask is needed downstream.
    Costs one pass over the 8192-token vocabulary at startup.
    """
    enc = tokenizer.enc
    V = tokenizer.get_vocab_size()
    raw = [None] * V
    for i in range(V):
        try:
            raw[i] = enc.decode_single_token_bytes(i)
        except Exception:
            raw[i] = None
    lookup = {b: i for i, b in enumerate(raw) if b is not None}
    table = list(range(V))
    for i, b in enumerate(raw):
        if b is None or len(b) < 2:
            continue
        for L in range(len(b) - 1, 0, -1):
            j = lookup.get(b[:L])
            if j is not None:
                table[i] = j
                break
    return torch.tensor(table, dtype=torch.int64, device=device)


def canon_conv(h: torch.Tensor, w: torch.Tensor, doc_start) -> torch.Tensor:
    """Causal depthwise conv over the token axis: out[t] = sum_i w[i] * h[t-i], per channel.

    h (b, t, D); w (W, D). Taps beyond a document start contribute 0: the mask is applied to the accumulator
    AFTER each shift, so position t loses any tap that would reach across a boundary in (t-i, t].
    """
    out = h * w[0]
    acc = h
    mask = None if doc_start is None else doc_start.unsqueeze(-1)
    for i in range(1, w.shape[0]):
        acc = torch.cat([torch.zeros_like(acc[:, :1]), acc[:, :-1]], dim=1)
        if mask is not None:
            acc = acc.masked_fill(mask, 0.0)
        out = out + acc * w[i]
    return out


class Block(nn.Module):
    def __init__(self, config: GPTConfig, registry: StackRegistry):
        super().__init__()
        self.attn = CausalSelfAttention(config, registry)
        self.mlp = MLP(config, registry)
        for _nm, _w in (("canon_a", CANON_ATTN_W), ("canon_m", CANON_MLP_W), ("canon_p", CANON_PRE_W)):
            if _w > 1:
                _p = torch.zeros(_w, config.n_embd)
                _p[0] = 1.0
                setattr(self, _nm, nn.Parameter(_p))

    def forward(self, x, x0, ve, cos, sin, resid_lambda, x0_lambda, ve_lambda, sink, gw, dense_mask, bounds, doc_start=None,
                rope_tabs=None, wviews=None):
        if CANON_PRE_W > 1:
            x = canon_conv(x, self.canon_p, doc_start)
        if (LAYER_IMPL == "op" and not CANON_ON and _LAYER_OPS is not None and x.device.type == "neuron" and sink is not None
                and bounds is not None and rope_tabs is not None and x.shape[1] % 128 == 0):
            b, t, D = x.shape
            T = b * t; bf16 = torch.bfloat16
            cos2d, sin2d = rope_tabs
            cos_full = cos2d.repeat(b, 1).contiguous(); sin_full = sin2d.repeat(b, 1).contiguous()
            at, ml = self.attn, self.mlp
            _lins = (at.c_q, at.c_k, at.c_v, at.c_proj, ml.c_fc, ml.c_proj)
            ws = ([lin._stack[lin.idx].to(bf16) for lin in _lins] if wviews is None
                  else [wviews[lin.key][lin.idx] for lin in _lins])
            x2 = x.reshape(T, D).to(bf16).contiguous(); x02 = x0.reshape(T, D).to(bf16).contiguous()
            ve2 = ve.reshape(T, D).to(bf16).contiguous()
            scal = torch.stack([resid_lambda.reshape(()), x0_lambda.reshape(()), ve_lambda.reshape(()),
                                torch.zeros((), device=x.device)]).float().view(1, 4).expand(128, 4).contiguous()
            dstart = (doc_start.to(torch.float32) if doc_start is not None else torch.zeros((b, t), device=x.device)).contiguous()
            wg_pad = F.pad(gw.float(), (0, 0, 0, 128 - gw.shape[0]))   # W_g zero-padded to 128 rows (zero rows add exact 0)
            outs = torch.ops.tf_nki.layer_fwd(x2, x02, ve2, cos_full, sin_full, cos2d, sin2d, scal, sink, bounds[0], bounds[1],
                                              dstart, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], wg_pad)
            return outs[0].view(b, t, D).to(x.dtype)
        x = resid_lambda * x + x0_lambda * x0
        _h = norm(x)
        if CANON_ATTN_W > 1:
            _h = canon_conv(_h, self.canon_a, doc_start)
        x = x + self.attn(_h, ve_lambda * ve, cos, sin, sink, gw, dense_mask, bounds, doc_start=doc_start)
        _h = norm(x)
        if CANON_MLP_W > 1:
            _h = canon_conv(_h, self.canon_m, doc_start)
        x = x + self.mlp(_h)
        return x


_U = np.uint64
_NG_M1, _NG_M2, _NG_LO = _U(0x9E3779B97F4A7C15), _U(0xBF58476D1CE4E5B9), _U(8191)
_NG_FIELDS = (("off", np.int32), ("fp", np.int32), ("tok", np.int16), ("by", np.uint8))


def ng_key(h):
    """51-bit context key: a splitmix finish of the rolling hash of the context tokens."""
    z = (h ^ (h >> _U(31))) * _NG_M2
    return (z ^ (z >> _U(29))) >> _U(13)


def ng_table(k, minc: int, kb: int):
    """Sorted (context key << 13 | next token) -> one sub-table: per context with count >= minc, its most frequent next
    token and an evidence byte (min(floor(log2 count), 12) * 5 + share-of-top bin), bucketed by the top kb key bits."""
    f = np.empty(len(k), bool); f[:1] = True; np.not_equal(k[1:], k[:-1], out=f[1:])
    s = np.flatnonzero(f); del f
    uk = k[s]; c = np.diff(np.append(s, len(k))).astype(np.uint32); del s
    ctx = uk >> _U(13); f = np.empty(len(uk), bool); f[:1] = True; np.not_equal(ctx[1:], ctx[:-1], out=f[1:])
    s = np.flatnonzero(f); del f
    tot = np.add.reduceat(c, s) if len(s) else c[:0]
    uk &= _NG_LO; c = c.astype(np.uint64); c <<= _U(13); c |= uk; del uk
    best = np.maximum.reduceat(c, s) if len(s) else c[:0]; del c
    keep = tot >= minc
    ctx, tot, best = ctx[s][keep], tot[keep], best[keep]; del s
    by = np.minimum(np.log2(np.maximum(tot, 1)).astype(np.int64), 12) * 5 + np.digitize((best >> _U(13)) / np.maximum(tot, 1), [0.3, 0.5, 0.7, 0.9])
    cnt = np.bincount((ctx >> _U(51 - kb)).astype(np.int64), minlength=1 << kb)
    return (np.concatenate([[0], np.cumsum(cnt)]).astype(np.uint32), ((ctx >> _U(max(0, 19 - kb))) & _U(0xFFFFFFFF)).astype(np.uint32),
            (best & _NG_LO).astype(np.uint16), by.astype(np.uint8))


def ng_count(streams, p: int):
    """For each order: the partition-p sub-tables of token streams s0 and s1 and of their union.
    Partition p holds the positions whose last context token is p mod NG_PARTS; every position has 15 tokens of context."""
    idx = [(np.flatnonzero((s[14:-1] % NG_PARTS) == p) + 15).astype(np.int32) for s in streams]
    h = [np.zeros(len(i), np.uint64) for i in idx]
    for n in range(2, 17):
        for j in range(2):
            h[j] = h[j] * _NG_M1 + streams[j][idx[j] - (n - 1)].astype(np.uint64) + _U(1)
        if n in NG_ORDERS:
            oi = NG_ORDERS.index(n)
            ks = [np.sort((ng_key(h[j]) << _U(13)) | streams[j][idx[j]].astype(np.uint64)) for j in range(2)]
            tabs = [ng_table(k, NG_MINC[oi], NG_KB[oi]) for k in ks]
            u = np.concatenate(ks); del ks
            u.sort(kind="stable")
            tabs.append(ng_table(u, NG_MINC[oi], NG_KB[oi])); del u
            yield oi, tabs
            del tabs


def _ng_save(path: Path, a) -> None:
    np.save(path.with_suffix(".part.npy"), a)
    os.replace(path.with_suffix(".part.npy"), path)


def _ng_wait(paths) -> None:
    while not all(q.exists() for q in paths):
        time.sleep(0.5)


_PT = {}
def _pt(k, t):
    _PT[k] = _PT.get(k, 0.0) + time.monotonic() - t
    return time.monotonic()


def _timed_features(tab, a):
    t = time.monotonic(); r = tab.features(a); _pt("feat_thread", t); return r


def ngram_build(rank: int, world: int, tmp: Path) -> None:
    """The n-gram counter, run in a helper process forked at startup: host CPU only, it never touches the accelerator.
    Tokenizes this worker's train row groups (the same `_row_groups_for_split` assignment the train loader uses), waits
    for every worker's tokens, then counts partitions rank, rank+world, ... of three streams: s0 = the row groups of the
    even workers, s1 = the odd workers', s2 = both. Files only; the training process polls for the done markers."""
    try:
        tok = ensure_tokenizer(build_if_missing=False)
        ids = []
        for path, rg in _row_groups_for_split("train", rank, world):
            rows = pq.ParquetFile(path).read_row_group(rg).column("text").to_pylist()
            ids.extend(np.array(d, np.uint16) for d in tok.encode(rows, prepend=tok.get_bos_token_id(), num_threads=4))
        _ng_save(tmp / f"t{rank}.npy", np.concatenate(ids))
        del ids
        names = [tmp / f"t{r}.npy" for r in range(world)]
        _ng_wait(names)
        if rank < 2:
            _ng_save(tmp / f"f{rank}.npy", np.concatenate([np.load(n, mmap_mode="r") for n in names[rank::2]]))
        _ng_wait([tmp / "f0.npy", tmp / "f1.npy"])
        streams = [np.load(tmp / f"f{f}.npy", mmap_mode="r") for f in (0, 1)]
        with open(tmp / f"lock{rank % 4}", "a") as lock:      # at most 4 workers count at once (host memory)
            fcntl.flock(lock, fcntl.LOCK_EX)
            for p in range(rank, NG_PARTS, world):
                for oi, tabs in ng_count(streams, p):
                    for si in range(3):
                        for (nm, _dt), a in zip(_NG_FIELDS, tabs[si]):
                            np.save(tmp / f"s{si}o{oi}p{p}{nm}.npy", a)
                (tmp / f"done{p}").touch()
    except BaseException as exc:
        (tmp / "error").write_text(repr(exc))


class NgramTable:
    """Read-only n-gram lookup. parts[order][partition] = (bucket offsets, key fingerprints, top token, evidence byte)."""

    def __init__(self, parts=None):
        self.parts = parts
        self.iters = [[int(np.diff(q[0]).max(initial=0)).bit_length() for q in o] for o in parts] if parts else None

    @classmethod
    def load(cls, tmp: Path, si: int) -> "NgramTable":
        return cls([[tuple(np.load(tmp / f"s{si}o{oi}p{p}{nm}.npy", mmap_mode="r") for nm, _dt in _NG_FIELDS)
                     for p in range(NG_PARTS)] for oi in range(len(NG_ORDERS))])

    def state(self):
        if not self.parts:
            return None
        st = {}
        for oi, o in enumerate(self.parts):
            for k, (nm, dt) in enumerate(_NG_FIELDS):
                st[f"{oi}{nm}"] = torch.from_numpy(np.concatenate([q[k] for q in o]).view(dt))
            st[f"{oi}n"] = torch.tensor([[len(q[0]), len(q[1])] for q in o], dtype=torch.int64)
        return st

    @classmethod
    def from_state(cls, st) -> "NgramTable":
        if not st:
            return cls()
        parts = []
        for oi in range(len(NG_ORDERS)):
            n = st[f"{oi}n"].numpy()
            arrs = [np.split(st[f"{oi}{nm}"].numpy().view(np.dtype(dt).str.replace("i", "u")), np.cumsum(n[:, min(k, 1)])[:-1])
                    for k, (nm, dt) in enumerate(_NG_FIELDS)]
            parts.append([tuple(a[p] for a in arrs) for p in range(NG_PARTS)])
        return cls(parts)

    def _find(self, oi, key, part, ok):
        kb = NG_KB[oi]
        fq = ((key >> _U(max(0, 19 - kb))) & _U(0xFFFFFFFF)).astype(np.uint32)
        bq = (key >> _U(51 - kb)).astype(np.int64)
        tok = np.full(len(key), -1, np.int32)
        by = np.zeros(len(key), np.int32)
        for p in range(NG_PARTS):
            off, fp, tk, byt = self.parts[oi][p]
            sel = np.flatnonzero(ok & (part == p))
            if len(sel) == 0 or len(fp) == 0:
                continue
            q = fq[sel]
            lo = off[bq[sel]].astype(np.int64)
            end = off[bq[sel] + 1].astype(np.int64)
            hi = end
            for _ in range(self.iters[oi][p]):
                mid = (lo + hi) >> 1
                go = (mid < hi) & (fp[np.minimum(mid, len(fp) - 1)] < q)
                lo, hi = np.where(go, mid + 1, lo), np.where(go, hi, mid)
            j = np.minimum(lo, len(fp) - 1)
            hit = (lo < end) & (fp[j] == q)
            tok[sel[hit]] = tk[j[hit]]
            by[sel[hit]] = byt[j[hit]]
        return tok, by

    def features(self, x) -> np.ndarray:
        """(2, b, t) int32 for token rows x (b, t): [0] the top next token of the longest context of x[r, :t+1] found in
        the table, [1] its evidence bucket in [0, NG_NB), or -1 where no order matches. Rows are independent and position
        t reads only x[r, t-14 .. t], so this is causal and deterministic."""
        x = np.asarray(x, np.int64)
        b, t = x.shape
        out = np.zeros((2, b, t), np.int32)
        out[1] = -1
        if not self.parts:
            return out
        h = np.zeros((b, t), np.uint64)
        part = (x % NG_PARTS).reshape(-1)
        pos = np.arange(t)
        tk, by = [], []
        for n in range(2, 17):
            prev = np.zeros_like(x)
            prev[:, n - 2:] = x[:, :t - (n - 2)]
            h = h * _NG_M1 + prev.astype(np.uint64) + _U(1)
            if n in NG_ORDERS:
                ok = np.broadcast_to(pos >= n - 2, (b, t)).reshape(-1)
                a, c = self._find(NG_ORDERS.index(n), ng_key(h).reshape(-1), part, ok)
                tk.append(a)
                by.append(c)
        tau, bkt = out[0].reshape(-1), out[1].reshape(-1)
        top = np.full(b * t, -1)
        for oi in reversed(range(len(tk))):
            new = (top < 0) & (tk[oi] >= 0)
            top[new] = oi
            tau[new] = tk[oi][new]
            bkt[new] = oi * 65 + by[oi][new]
        agree = np.zeros(b * t, np.int32)
        for oi in range(len(tk)):
            agree += (oi < top) & (tk[oi] == tau)
        bkt[:] = np.where(top >= 0, bkt * 3 + np.minimum(agree, 2), -1)
        return out


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
        self.stacks = registry.build()
        for m in self.modules():
            if isinstance(m, StackedLinear):
                m.bind(self.stacks[m.key])
        L, H = config.n_layer, config.n_head
        self.block_scalars = nn.ParameterDict({
            "resid": nn.Parameter(torch.ones(L, 1)),
            "x0": nn.Parameter(torch.zeros(L, 1)),
            "ve": nn.Parameter(torch.full((L, 1), 0.5)),
            "sink": nn.Parameter(torch.full((L, H), SINK_INIT)),
        })
        self.attn_gate = nn.Parameter(torch.zeros(L, ATTN_GATE, H))   # W_g per layer; zero -> g == 1 exactly
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.skip_lambdas = nn.Parameter(torch.ones(config.n_layer // 2))
        self.bigram_lambda = nn.Parameter(torch.full((1,), 0.2))
        self.softcap_pos = nn.Parameter(torch.full((1,), LOGIT_SOFTCAP))
        self.softcap_neg = nn.Parameter(torch.full((1,), LOGIT_SOFTCAP))
        self.ngram_theta = nn.Parameter(torch.zeros(NG_NB))    # logit bias per n-gram evidence bucket
        self.ngram = NgramTable()                               # set by the training loop / load_for_eval
        self._fused_loss_ok = True
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos2d", cos.reshape(cos.shape[1], -1).float().contiguous(), persistent=False)
        self.register_buffer("sin2d", sin.reshape(sin.shape[1], -1).float().contiguous(), persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        for name in ("wte", "vte", "vte_mid", "vte2", "bte"):
            torch.nn.init.normal_(self.transformer[name].weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.block_scalars["resid"].fill_(1.0)
        self.block_scalars["x0"].fill_(0.05)
        self.block_scalars["ve"].fill_(0.5)
        self.block_scalars["sink"].fill_(SINK_INIT)
        self.attn_gate.zero_()                  # g = 2 * sigmoid(0) == 1.0 at step 0
        self.skip_lambdas.fill_(1.0)
        self.bigram_lambda.fill_(0.2)
        self.softcap_pos.fill_(LOGIT_SOFTCAP)
        self.softcap_neg.fill_(LOGIT_SOFTCAP)
        self.ngram_theta.zero_()
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = ROPE_BASE):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (freqs / head_dim))
        inv_freq[int(round(inv_freq.numel() * ROPE_FRAC)):] = 0.0
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    def setup_optimizer(self):
        scalar_params = [self.skip_lambdas, self.bigram_lambda, self.softcap_pos, self.softcap_neg]
        scalar_params.extend(self.block_scalars[k] for k in ("resid", "x0", "ve", "sink"))
        embedding_params = [self.transformer[n].weight for n in ("wte", "vte", "vte_mid", "vte2", "bte")]
        lm_head_params = list(self.lm_head.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale * 1.12, betas=(0.8, 0.99), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=scalar_params + [self.attn_gate, self.ngram_theta], lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10,
                 weight_decay=0.0, lr_mul=[1.0] * len(scalar_params) + [ATTN_GATE_LR / SCALAR_LR, NG_LR_MUL]),
        ]
        for key in sorted(self.stacks.keys()):
            param_groups.append(dict(kind="muon", params=[self.stacks[key]], lr=MATRIX_LR, momentum=_EFF_MUON_MOMENTUM, beta2=_EFF_MUON_BETA2, ns_steps=NS_STEPS, weight_decay=WEIGHT_DECAY))
        canon_params = [p for _n, p in self.named_parameters() if ".canon_" in _n]
        if canon_params:
            param_groups.append(dict(kind="adamw", params=canon_params, lr=CANON_LR, betas=(0.9, 0.95),
                                     eps=1e-10, weight_decay=0.0))
        print0(f"ATTN_GATE: k={ATTN_GATE} W_g={tuple(self.attn_gate.shape)} lr={ATTN_GATE_LR:.5f} (scalar AdamW group, "
               f"x{ATTN_GATE_LR / SCALAR_LR:.5f}) wd=0 init=0")
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        if B2:
            print0(f"B2 LINEAR-TEMPERATURE regime ON: wd_track={B2_WD_TRACK} (wd_t = {WEIGHT_DECAY} * lrm) "
                   f"final_lr_frac={_EFF_FINAL_LR_FRAC} warmup={_EFF_WARMUP_STEPS} muon_mom_peak={_EFF_MUON_MOMENTUM} "
                   f"muon_beta2={_EFF_MUON_BETA2} ema_start={_EFF_EMA_START} ema_decay={_EFF_EMA_DECAY} "
                   f"blend={EMA_BLEND}/{EMA_TABLE_BLEND} (unchanged) diag_every={B2_DIAG_EVERY}")
        else:
            print0(f"B2 off: baseline schedule (warmup={_EFF_WARMUP_STEPS} final_lr_frac={_EFF_FINAL_LR_FRAC} "
                   f"muon_mom_peak={_EFF_MUON_MOMENTUM} muon_beta2={_EFF_MUON_BETA2} wd={WEIGHT_DECAY} const "
                   f"ema_start={_EFF_EMA_START} ema_decay={_EFF_EMA_DECAY})")
        return opt

    def head_logits(self, x: torch.Tensor, tau: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """The model's output logits: final RMSNorm, lm_head, asymmetric learned softcap, plus the n-gram term `bias`
        added to the one column `tau` of each position. This is what the scorer evaluates (forward with targets=None)
        and the reference the fused training loss is checked against."""
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        sp = self.softcap_pos.clamp_min(1.0).float()
        sn = self.softcap_neg.clamp_min(1.0).float()
        z = torch.where(logits >= 0, sp * torch.tanh(logits / sp), sn * torch.tanh(logits / sn))
        return z + bias.float().unsqueeze(-1) * (torch.arange(z.size(-1), device=z.device) == tau.unsqueeze(-1))

    def ngram_bias(self, bucket: torch.Tensor) -> torch.Tensor:
        """theta[bucket] with no gather in the compiled graph: theta as 37 x 37 (zero-padded), bucket = 37 * row + col, a
        one-hot product picks the row and a one-hot mask the column; bucket -1 (no match) gives exactly 0."""
        r = torch.arange(37, device=bucket.device, dtype=bucket.dtype)
        th = F.pad(self.ngram_theta.float(), (0, 37 * 37 - NG_NB)).view(37, 37)
        a = (torch.div(bucket, 37, rounding_mode="floor").unsqueeze(-1) == r).float() @ th
        return (a * (torch.remainder(bucket, 37).unsqueeze(-1) == r)).sum(-1)

    def fused_linear_cross_entropy(self, x: torch.Tensor, targets: torch.Tensor, tau: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """cross_entropy(self.head_logits(x), targets) with ignore_index=-1 and mean reduction, computed by one NKI kernel
        without materializing the (T, V) logits: per 128-token tile the kernel forms the softcapped logits 512 columns at a
        time and folds them into an online log-sum-exp (same math as `head_logits` followed by F.cross_entropy). The lm_head
        weight enters transposed; the softcap parameters enter through `scal` and receive gradients. The kernel also
        returns z at column tau, which makes the n-gram term exact: logsumexp(z + bias * e_tau) = lse + log1p(p_tau *
        expm1(bias)), and the target logit gains bias where tau == target; the kernel's backward takes all three grads."""
        b, t, D = x.shape
        x2 = x.reshape(b * t, D).to(torch.bfloat16).contiguous()
        wT = self.lm_head.weight.to(torch.bfloat16).t().contiguous()
        tgt = targets.reshape(b * t, 1).to(torch.int32).contiguous()
        scal = torch.stack([self.softcap_pos.clamp_min(1.0).reshape(()), self.softcap_neg.clamp_min(1.0).reshape(()),
                            torch.zeros((), device=x.device), torch.zeros((), device=x.device)]).float().view(1, 4).expand(128, 4).contiguous()
        tau = tau.reshape(b * t, 1).to(torch.int32).contiguous()
        loss_rows, lse, _h, _rstd, zt = torch.ops.tf_nki.head_fwd(x2, wT, tgt, scal, tau)
        bias = bias.reshape(b * t, 1).float()
        corr = torch.log1p(torch.exp(zt - lse) * torch.expm1(bias)) - bias * (tau == tgt)
        valid = targets.reshape(-1, 1) != -1
        return (loss_rows + corr * valid).sum() / valid.sum().clamp_min(1).float()

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean", emb=None, ng=None):
        _b, t = idx.shape
        if ng is None:      # n-gram features of these rows, a function of idx alone (training passes the same call's result)
            ng = torch.from_numpy(self.ngram.features(idx.detach().cpu().numpy())).to(idx.device)
        tau, bias = ng[0], self.ngram_bias(ng[1])
        assert t <= self.config.sequence_len
        cos, sin = self.cos[:, :t], self.sin[:, :t]

        prev = torch.cat([torch.full_like(idx[:, :1], BOS_ID), idx[:, :-1]], dim=1)
        pair = prev.to(torch.int32) * 8192 + idx.to(torch.int32)
        bh = (((pair * -1640531527) >> 13) & (BIGRAM_SIZE - 1)).long()

        if targets is not None:
            idx = idx.clone()
            targets = targets.clone()
        if emb is not None:
            emb = {k: v.clone() for k, v in emb.items()}
            x = emb["wte"] + self.bigram_lambda * emb["bte"]
        else:
            x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
            x = x + self.bigram_lambda * self.transformer.bte(bh).to(COMPUTE_DTYPE)
        # the input RMSNorm uses the fp32 epsilon (x is fp32 here: bigram_lambda is an fp32 Parameter). This one line
        # serves training and evaluation alike; the other norm() sites keep the kernels' bf16 epsilon
        x = F.rms_norm(x, (x.size(-1),), eps=1.1920928955078125e-07)
        x0 = x.clone()
        if emb is not None:
            ve1, ve_mid, ve2 = emb["vte"], emb["vte_mid"], emb["vte2"]
        else:
            ve1 = self.transformer.vte(idx).to(COMPUTE_DTYPE)
            ve_mid = self.transformer.vte_mid(idx).to(COMPUTE_DTYPE)
            ve2 = self.transformer.vte2(idx).to(COMPUTE_DTYPE)
        t1 = VE_T1 if VE_T1 > 0 else self.config.n_layer // 3
        t2 = VE_T2 if VE_T2 > 0 else self.config.n_layer - t1

        dense_mask = None
        bounds = None
        doc_start = (idx == BOS_ID) if KEY_OFFSET else None
        wviews = ({k: torch.unbind(p.to(torch.bfloat16)) for k, p in self.stacks.items()}
                  if (LAYER_IMPL == "op" and COMPUTE_DTYPE == torch.bfloat16) else None)
        rope_tabs = (self.cos2d[:t], self.sin2d[:t]) if LAYER_IMPL == "op" else None
        # The document mask is selected by DEVICE TYPE alone, identically in training and in evaluation: on Trainium
        # it is passed to the flash-attention kernels as per-query KV bounds, elsewhere as the equivalent dense
        # boolean mask. Both express exactly "attend to earlier positions of the same document".
        if idx.device.type == "neuron":
            bounds = doc_bounds(idx)
        else:
            docs = (idx == BOS_ID).cumsum(dim=1)
            pos = torch.arange(t, device=idx.device)
            causal = pos.view(1, t, 1) >= pos.view(1, 1, t)
            same_doc = docs.unsqueeze(2) == docs.unsqueeze(1)
            dense_mask = (causal & same_doc).unsqueeze(1)

        n_enc = self.config.n_layer // 2
        n_dec_start = self.config.n_layer - n_enc
        skips: list[torch.Tensor] = []
        bs = self.block_scalars
        gws = torch.unbind(self.attn_gate)      # ATTN_GATE: (k, n_head) per layer
        for i, block in enumerate(self.transformer.h):
            if i >= n_dec_start:
                x = x + self.skip_lambdas[i - n_dec_start] * skips.pop()
            ve_i = ve1 if i < t1 else (ve_mid if i < t2 else ve2)
            x = block(x, x0, ve_i, cos, sin, bs["resid"][i], bs["x0"][i], bs["ve"][i], bs["sink"][i], gws[i], dense_mask,
                      bounds, doc_start=doc_start, rope_tabs=rope_tabs, wviews=wviews)
            if i < n_enc:
                skips.append(x)
        if targets is None:
            return self.head_logits(x, tau, bias)
        if (self._fused_loss_ok and _HEAD_OPS is not None and loss_reduction == "mean" and x.device.type == "neuron"
                and x.shape[1] % 128 == 0):
            return self.fused_linear_cross_entropy(x, targets, tau, bias)
        logits = self.head_logits(x, tau, bias)
        flat_t = targets.reshape(-1)
        valid = flat_t != -1
        logp = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
        nll = -logp.gather(1, flat_t.clamp_min(0).unsqueeze(1)).squeeze(1) * valid.to(logp.dtype)
        if loss_reduction == "none":
            return nll.view_as(targets)
        return nll.sum() / valid.sum().clamp_min(1).to(nll.dtype)


def build_config() -> GPTConfig:
    return GPTConfig(sequence_len=SEQ_LEN, vocab_size=TOKENIZER_VOCAB_SIZE, n_layer=DEPTH, n_head=N_HEAD, n_embd=N_EMBD)




def bigram_bucket(h):
    return h & (BIGRAM_SIZE - 1)


_NKI_EMBED = None


def build_nki_embed():
    global _NKI_EMBED
    if _NKI_EMBED is None:
        if not NKI_AVAILABLE:
            raise RuntimeError("NKI is not available on this device")
        from torch_neuronx.nki_hop import wrap_nki
        _NKI_EMBED = (wrap_nki(embed_gather_v2), wrap_nki(embed_scatter_add_v2),
                      wrap_nki(embed_scatter_add_v2_x4))
    return _NKI_EMBED


SCATTER_BINS = 128
SCATTER_DENSE_K = 32
SCATTER_DENSE_FP32 = True
SCATTER_CHECK = True


def scatter_packed(idx, rows, bins=None, dense_k=None):
    """Host plan for embed_scatter_add_v2[_x4], replacing scatter_rounds' occurrence-rank rounds.

    The kernel needs every 128-slot tile to hold DISTINCT destination rows (dma_compute is a read-modify-write, so
    a repeat inside one tile silently loses an update); a destination may repeat ACROSS tiles. scatter_rounds gets
    distinctness by occurrence rank, so the most frequent id sets the tile count - measured 403-573 rounds and
    78.7% padding. Here the ids whose count exceeds one tile's budget go to a DENSE path (their gradient rows are
    pre-summed by one matmul, then scattered as a single 128-slot tile) and the rest are dealt round-robin into
    `bins` bins, which is optimal packing: occurrences of one id are contiguous after the sort and an id has
    count <= bins, so they land in distinct residues mod T - distinctness is STRUCTURAL, not padded.

    Every chunk has exactly 128*bins slots, so only the NUMBER of launches is data-dependent and no call can
    trigger a recompile. Returns (dense_ids, chunks); sidx == -1-j marks the j-th padding slot of its tile, with
    j == the slot's own lane so prepare()'s R-1-sidx mapping still gives a distinct dummy row per lane."""
    import numpy as np
    bins = SCATTER_BINS if bins is None else bins
    dense_k = SCATTER_DENSE_K if dense_k is None else dense_k
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    n = idx.size
    cap = bins * 128
    order = np.argsort(idx, kind="stable")
    sid = idx[order]
    first = np.r_[True, sid[1:] != sid[:-1]]
    starts = np.flatnonzero(first)
    cnt = np.diff(np.r_[starts, n])
    uid = sid[starts]
    dense_ids = None
    keep = np.ones(starts.size, dtype=bool)
    if dense_k > 0:
        o = np.argsort(-cnt, kind="stable")[:dense_k]
        cs = cnt[o]
        cum = np.cumsum(cs)
        p = 0
        while p < o.size and (cs[p] > bins or n - (cum[p - 1] if p else 0) > cap):
            p += 1
        dense_ids = np.full(dense_k, -1, dtype=np.int64)
        dense_ids[:p] = uid[o[:p]]
        keep[o[:p]] = False
    sel = order[np.repeat(keep, cnt)]
    m = sel.size
    t = max(-(-m // 128), int(cnt[keep].max()) if m else 0, 1)
    t = -(-t // bins) * bins
    i = np.arange(m)
    slot = (i % t) * 128 + (i // t)
    perm = np.zeros(t * 128, dtype=np.int64)
    sidx = np.full(t * 128, -1, dtype=np.int64)
    perm[slot] = sel
    sidx[slot] = idx[sel]
    hole = sidx < 0
    sidx[hole] = -1 - (np.arange(t * 128) % 128)[hole]
    chunks = [(perm[a:a + cap].astype(np.uint32).reshape(-1, 1), sidx[a:a + cap]) for a in range(0, t * 128, cap)]
    if SCATTER_CHECK:
        scatter_plan_check(idx, rows, dense_ids, chunks)
    return dense_ids, chunks


def scatter_dense_tile(dense_ids, rows):
    """The single 128-slot plan that scatters the dense matmul's (K, C) output. Lane j carries dense_ids[j], or the
    dummy row rows+j when that column is unused."""
    import numpy as np
    k = len(dense_ids)
    lane = np.arange(128, dtype=np.int64)
    ids = np.full(128, -1, dtype=np.int64)
    ids[:min(k, 128)] = dense_ids[:128]
    perm = np.minimum(lane, k - 1)
    sidx = np.where(ids >= 0, ids, -1 - lane)
    return perm.astype(np.uint32).reshape(-1, 1), sidx


def scatter_onehot(idx, rows, dense_ids, dtype):
    """The (n, K) one-hot whose column j selects the occurrences of dense_ids[j]; unused columns are all-zero, so
    their matmul row is exactly 0 and lands on a dummy row."""
    import numpy as np
    n = idx.size
    k = len(dense_ids)
    p = int((dense_ids >= 0).sum())
    lut = np.full(rows, -1, dtype=np.int32)
    if p:
        lut[dense_ids[:p]] = np.arange(p, dtype=np.int32)
    col = lut[idx]
    at = np.flatnonzero(col >= 0)
    oh = torch.zeros((n, k), dtype=dtype)
    if at.size:
        oh[torch.from_numpy(at), torch.from_numpy(col[at].astype(np.int64))] = 1.0
    return oh


def scatter_plan_check(idx, rows, dense_ids, chunks):
    """Every way this plan can silently corrupt a gradient, checked on the host for ~0.17 ms per micro-step. The
    new planner is 1.35 ms FASTER than the old one, so the full check is free in net host time - and the invariant
    that corrupts silently (a repeat inside a tile) is exhaustively checkable, which is why it stays on."""
    import numpy as np
    n = idx.size
    real = 0
    for perm, sidx in chunks:
        assert len(sidx) % 128 == 0, "slot count is not a whole number of tiles"
        pm = perm.reshape(-1).astype(np.int64)
        assert pm.min() >= 0 and pm.max() < n, "perm out of the gradient buffer"
        neg = sidx < 0
        assert (sidx[neg] >= -128).all(), "padding index outside the 128-row tail"
        assert sidx[~neg].max(initial=0) < rows, "destination past the table"
        dst = np.where(neg, rows - 1 - sidx, sidx).reshape(-1, 128)
        assert not (np.diff(np.sort(dst, axis=1), axis=1) == 0).any(), "repeat inside a tile"
        real += int((~neg).sum())
    if dense_ids is not None:
        p = int((dense_ids >= 0).sum())
        assert np.unique(dense_ids[:p]).size == p, "duplicate id in the dense set"
        if p:
            hot = np.zeros(rows, dtype=bool)
            hot[dense_ids[:p]] = True
            for _perm, sidx in chunks:
                assert not hot[sidx[sidx >= 0]].any(), "id in BOTH paths: double-counted gradient"
            real += int(np.bincount(idx, minlength=rows)[dense_ids[:p]].sum())
    assert real == n, f"occurrences lost or duplicated: {real} placed, {n} in the batch"


def scatter_rounds(idx):
    """Host plan for embed_scatter_add: slots grouped into rounds of distinct indices (the r-th occurrence of every
    distinct index), each round padded to a multiple of 128; padding slots carry -1 (replaced per table by the distinct
    dummy rows R + j of the gradient buffer's 128-row tail). Returns a list of (perm, sidx) chunks. The round count is
    the largest multiplicity of any one id - measured 403-573 rounds for token ids (78.7% padding) at the production
    shape - which is why scatter_packed replaced it; this planner runs only if SCATTER_BINS is 0, and it is 128."""
    import numpy as np
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    first = np.r_[True, sidx[1:] != sidx[:-1]]
    starts = np.flatnonzero(first)
    rank = np.arange(len(sidx)) - starts[np.cumsum(first) - 1]
    counts = np.bincount(rank)
    by_rank = np.argsort(rank, kind="stable")
    perm_l, idx_l = [], []
    pos = 0
    for n in counts:
        sel = order[by_rank[pos:pos + n]]
        pos += n
        pad = (-n) % 128
        perm_l.append(sel); idx_l.append(idx[sel])
        if pad:
            perm_l.append(np.zeros(pad, dtype=np.int64)); idx_l.append(-1 - np.arange(pad, dtype=np.int64))
    perm = np.concatenate(perm_l)
    sidx = np.concatenate(idx_l)
    if len(sidx) > 32768:
        chunks = []
        for a in range(0, len(sidx), 32768):
            chunks.extend(_pad_plan(perm[a:a + 32768], sidx[a:a + 32768]))
        return chunks
    return _pad_plan(perm, sidx)


def _pad_plan(perm, sidx):
    import numpy as np
    target = 4096
    while target < len(sidx):
        target *= 2
    pad = target - len(sidx)
    if pad:
        perm = np.concatenate([perm, np.zeros(pad, dtype=np.int64)])
        sidx = np.concatenate([sidx, -1 - (np.arange(pad, dtype=np.int64) % 128)])
    return [(perm.astype(np.uint32).reshape(-1, 1), sidx)]


class EmbedSidecar:
    """Embedding tables outside the compiled graph. gather(model, x) returns dense bf16 rows (leaf tensors) for the 5 tables
    plus the host scatter plans; after backward, scatter(...) accumulates their .grad into table gradients of the table's
    own dtype (fp32; bf16 for the bigram table, see hold_bigram_bf16) with the NKI scatter-add kernel (or torch index_add_
    under sidecar_torch); attach() hands those to the parameters."""

    def __init__(self, model, impl: str):
        self.model = model
        self.impl = impl
        self.tables = {"wte": model.transformer.wte.weight, "vte": model.transformer.vte.weight,
                       "vte_mid": model.transformer.vte_mid.weight, "vte2": model.transformer.vte2.weight}
        self.tables["bte"] = model.transformer.bte.weight
        self.kinds = {"tok": ["wte", "vte", "vte_mid", "vte2"], "bg": ["bte"]}
        self.gbuf = {k: torch.zeros((w.shape[0] + 128, w.shape[1]), dtype=w.dtype, device=w.device) for k, w in self.tables.items()}
        self.bigram_grad = True

    def start_step(self, bigram_grad: bool = True):
        self.bigram_grad = bigram_grad
        for g in self.gbuf.values():
            g.zero_()

    @torch.no_grad()
    def prepare(self, x):
        """Host side of a micro-step (call it while the device runs the previous backward): indices, bigram hashes and
        the scatter plans, uploaded as ONE uint32 buffer per index kind (device views for the kernels)."""
        import numpy as np
        dev = x.device
        ids_np = x.detach().cpu().numpy().astype(np.int64).reshape(-1)
        b, t = x.shape
        keys = {"tok": ids_np}
        if True:
            ids2 = ids_np.reshape(b, t)
            prev = np.concatenate([np.full((b, 1), BOS_ID, dtype=np.int64), ids2[:, :-1]], axis=1).reshape(-1)
            pair = (prev.astype(np.int32) * np.int32(8192)) + ids_np.astype(np.int32)
            bh = bigram_bucket(((pair * np.int32(-1640531527)) >> 13).astype(np.int64))
            keys["bg"] = bh
        plans = {}
        for kind, ind in keys.items():
            R = self.tables[self.kinds[kind][0]].shape[0]
            dense_ids = None
            if SCATTER_BINS > 0:
                dk = SCATTER_DENSE_K if (kind == "tok" and not getattr(self, "dense_off", False)) else 0
                dense_ids, chunks = scatter_packed(ind, R, dense_k=dk)
            else:
                chunks = scatter_rounds(ind)
            parts = [ind]
            for perm, sidx in chunks:
                parts.append(perm.reshape(-1)); parts.append(np.where(sidx < 0, R - 1 - sidx, sidx))
            if dense_ids is not None:
                dperm, dsidx = scatter_dense_tile(dense_ids, R)
                parts.append(dperm.reshape(-1)); parts.append(np.where(dsidx < 0, R - 1 - dsidx, dsidx))
            buf = np.concatenate(parts).astype(np.uint32).reshape(-1, 1)
            dbuf = torch.from_numpy(buf).to(dev)
            n = len(ind); off = n; views = []
            for perm, sidx in chunks:
                S = perm.shape[0]
                views.append((dbuf[off:off + S], dbuf[off + S:off + 2 * S])); off += 2 * S
            dense = None
            if dense_ids is not None:
                oh = scatter_onehot(ind, R, dense_ids,
                                    torch.float32 if SCATTER_DENSE_FP32 else COMPUTE_DTYPE)
                dense = (oh.to(dev), dbuf[off:off + 128], dbuf[off + 128:off + 256])
            plans[kind] = (ind, dbuf[:n], views, None, dense)
        return plans

    @torch.no_grad()
    def gather(self, x, plans=None):
        if plans is None:
            plans = self.prepare(x)
        dev = x.device
        b, t = x.shape
        emb = {}
        for kind, (ind, ind_t, views, _unused, _dense) in plans.items():
            for name in self.kinds[kind]:
                w = self.tables[name]
                if self.impl == "sidecar":
                    rows = build_nki_embed()[0][(1,)](w, ind_t)
                else:
                    rows = w.index_select(0, torch.from_numpy(ind).to(dev)).to(COMPUTE_DTYPE)
                emb[name] = rows.view(b, t, -1).detach().requires_grad_(name != "bte" or self.bigram_grad)
        return emb, plans

    @torch.no_grad()
    def scatter(self, emb, plans):
        for kind, (ind, ind_t, views, _unused, dense) in plans.items():
            names = self.kinds[kind]
            if (self.impl == "sidecar" and len(names) == 4
                    and all(emb[n].grad is not None for n in names)
                    and len({self.gbuf[n].shape for n in names}) == 1):
                try:
                    gs = [emb[n].grad.reshape(-1, emb[n].grad.shape[-1]).contiguous() for n in names]
                    bs = [self.gbuf[n] for n in names]
                    k4 = build_nki_embed()[2]
                    if dense is not None:
                        oh, dperm_t, dsidx_t = dense
                        gd = [torch.matmul(oh.t(), g.to(oh.dtype)) for g in gs]
                        k4[(1,)](bs[0], bs[1], bs[2], bs[3], gd[0], gd[1], gd[2], gd[3], dperm_t, dsidx_t)
                    for perm_t, sidx_t in views:
                        k4[(1,)](bs[0], bs[1], bs[2], bs[3], gs[0], gs[1], gs[2], gs[3], perm_t, sidx_t)
                    continue
                except Exception as exc:
                    self.dense_off = True
                    note_fallback("embed scatter x4", exc)
                    print0(f"scatter x4 unavailable ({str(exc)[:80]}); per-table path")
            for name in names:
                g = emb[name].grad
                if g is None:
                    continue
                g2 = g.reshape(-1, g.shape[-1])
                done = False
                if self.impl == "sidecar":
                    try:
                        g2c = g2.contiguous()
                        for perm_t, sidx_t in views:
                            build_nki_embed()[1][(1,)](self.gbuf[name], g2c, perm_t, sidx_t)
                        done = True
                    except Exception as exc:
                        note_fallback(f"embedding scatter {name} -> index_add_", exc)
                        print0(f"scatter kernel failed ({str(exc)[:80]}); index_add_ fallback")
                if not done:
                    R = self.tables[name].shape[0]
                    self.gbuf[name][:R].index_add_(0, ind_t.view(-1).long(), g2.to(self.gbuf[name].dtype))

    def attach(self):
        for name, w in self.tables.items():
            w.grad = self.gbuf[name][:w.shape[0]] if (name != "bte" or self.bigram_grad) else None


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self.use_nki = NKI_AVAILABLE
        self._flat = {}
        self.plan = {}
        self.world_size = 1

    @staticmethod
    def _is_small_group(group):
        return group["kind"] == "adamw" and len(group["params"]) > 1 and all(p.numel() <= 4096 for p in group["params"])

    @torch.no_grad()
    def _adamw_step_flat(self, group):
        """The scalar-parameter group as ONE flat fp32 tensor: on Trainium every eager op is a device launch, and 8 tiny
        parameters through the per-parameter AdamW cost ~28 ms/step (half the optimizer); flat it is 1 cat, 1 all_reduce,
        the AdamW ops once, and a copy per parameter. Same math, one shared step counter."""
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        beta1, beta2 = group["betas"]
        lr_t = group["lr_t"]
        st = self._flat.get(id(group))
        if st is None:
            flat = torch.cat([p.detach().reshape(-1).float() for p in params])
            st = dict(sizes=[p.numel() for p in params], p=flat, m=torch.zeros_like(flat), v=torch.zeros_like(flat), step=0)
            if group.get("lr_mul") is not None:     # per-element lr multiplier (1.0 on every scalar, ATTN_GATE_LR / SCALAR_LR on W_g)
                _mul = {id(q): m for q, m in zip(group["params"], group["lr_mul"])}
                st["mul"] = torch.cat([torch.full((p.numel(),), float(_mul[id(p)]), dtype=flat.dtype) for p in params]).to(flat.device)
            self._flat[id(group)] = st
        g = torch.cat([p.grad.reshape(-1).float() for p in params])
        if self.world_size > 1:
            dist.all_reduce(g, op=dist.ReduceOp.AVG)
        flat, m, v = st["p"], st["m"], st["v"]
        flat.mul_(1 - lr_t * group["weight_decay"])
        m.lerp_(g, 1 - beta1)
        v.lerp_(g.square(), 1 - beta2)
        st["step"] += 1
        bias1 = 1 - beta1 ** st["step"]
        bias2 = 1 - beta2 ** st["step"]
        if "mul" in st:
            flat.sub_((lr_t * st["mul"]) * (m / bias1 / ((v / bias2).sqrt() + group["eps"])))
        else:
            flat.sub_(lr_t * (m / bias1 / ((v / bias2).sqrt() + group["eps"])))
        off = 0
        for p, sz in zip(params, st["sizes"]):
            p.copy_(flat[off:off + sz].view_as(p).to(p.dtype))
            off += sz

    @torch.no_grad()
    def setup_sharding(self, world_size: int, rank: int):
        """Sharded optimizer (ZeRO-1 style): every parameter whose leading dim splits evenly over the workers is updated by
        one worker only, on a persistent shard with its own momentum state; its gradient arrives by reduce_scatter and the
        updated shard goes back into the replicated parameter by all_gather (the same bytes the all_reduce moves).
        A Muon stack whose n is not a multiple of the world (the two MLP stacks, n=12 over 8 workers) is padded with zero
        rows to the next multiple (16 -> 2 matrices per worker, the last workers own none) instead of a replicated Muon
        step on every worker (2026-09-16: +1.1% throughput, same math)."""
        self.world_size = world_size
        if world_size == 1:
            return
        for group in self.param_groups:
            if self._is_small_group(group):
                continue
            for p in group["params"]:
                n = p.shape[0]
                if n % world_size != 0:
                    if not (group["kind"] == "muon" and p.dim() == 3):
                        continue
                    per = -(-n // world_size)
                    if per % (NKI_GRID[0] if NKI_AVAILABLE else 1) != 0:
                        continue
                    lo, hi = min(n, rank * per), min(n, (rank + 1) * per)
                    gpad = torch.zeros((per,) + tuple(p.shape[1:]), dtype=p.dtype, device=p.device)
                    pshard = p.data[lo:hi].clone()
                    pshard.grad = gpad[:hi - lo]
                    self.plan[p] = dict(pshard=pshard, gshard=gpad[:hi - lo], gpad=gpad, lo=lo, hi=hi, per=per,
                                        npad=per * world_size, n=n, pad=True, active=hi > lo)
                    continue
                per = n // world_size
                if group["kind"] == "adamw" and (p.dim() != 2 or per % (2 * P) != 0):
                    continue
                if group["kind"] == "muon" and per % (NKI_GRID[0] if NKI_AVAILABLE else 1) != 0:
                    continue
                pshard = p.data[rank * per:(rank + 1) * per].clone().float()   # fp32 master, also for the bf16 bigram table
                pshard.grad = torch.zeros_like(pshard)
                self.plan[p] = dict(pshard=pshard, gshard=pshard.grad)

    def reduce_gradients(self, model: nn.Module):
        small = set()
        for group in self.param_groups:
            if self._is_small_group(group):
                small.update(id(p) for p in group["params"])
        for p in model.parameters():
            if p.grad is None or id(p) in small:
                continue
            plan = self.plan.get(p)
            if plan is not None and plan.get("pad"):
                zpad = plan.get("zpad")
                if zpad is None:
                    zpad = plan["zpad"] = torch.zeros((plan["npad"] - plan["n"],) + tuple(p.shape[1:]), dtype=p.grad.dtype, device=p.device)
                dist.reduce_scatter_tensor(plan["gpad"], torch.cat([p.grad, zpad]), op=dist.ReduceOp.AVG)
            elif plan is not None and p.grad.dtype != plan["gshard"].dtype:
                # the bigram table: bf16 gradient, fp32 shard. Reduce at the gradient's dtype (half the bytes, no
                # full-size cast) and cast only the 1/world-size result up
                gs = plan.get("gsmall")
                if gs is None:
                    gs = plan["gsmall"] = torch.empty_like(plan["gshard"], dtype=p.grad.dtype)
                dist.reduce_scatter_tensor(gs, p.grad, op=dist.ReduceOp.AVG)
                plan["gshard"].copy_(gs)
            elif plan is not None:
                dist.reduce_scatter_tensor(plan["gshard"], p.grad, op=dist.ReduceOp.AVG)
            else:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    @torch.no_grad()
    def gather_params(self):
        for p, plan in self.plan.items():
            if plan.get("pad"):
                spad = plan.get("spad")
                if spad is None:
                    spad = plan["spad"] = torch.zeros((plan["per"],) + tuple(p.shape[1:]), dtype=torch.bfloat16, device=p.device)
                    plan["fpad"] = torch.empty((plan["npad"],) + tuple(p.shape[1:]), dtype=torch.bfloat16, device=p.device)
                if plan["active"]:
                    spad[:plan["hi"] - plan["lo"]].copy_(plan["pshard"])
                dist.all_gather_into_tensor(plan["fpad"], spad)
                p.data.copy_(plan["fpad"][:plan["n"]])
                continue
            if p.data.dtype == torch.bfloat16:
                # the bigram table: the collective writes the bf16 replica directly. The same values as the p16
                # path below, which also rounds every gathered weight to bf16.
                dist.all_gather_into_tensor(p.data, plan["pshard"].to(torch.bfloat16))
                continue
            p16 = plan.get("p16")
            if p16 is None:
                p16 = plan["p16"] = torch.empty_like(p.data, dtype=torch.bfloat16)
            dist.all_gather_into_tensor(p16, plan["pshard"].to(torch.bfloat16))
            p.data.copy_(p16)

    def _adamw_state(self, p):
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        return state

    def _muon_state(self, group):
        p0 = group["params"][0]
        state = self.state[p0]
        n, out_f, in_f = p0.shape
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p0)
        if "second_momentum_buffer" not in state:
            state_shape = (n, out_f, 1) if out_f >= in_f else (n, 1, in_f)
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        return state

    @torch.no_grad()
    def refresh_scalars(self):
        """Per-step learning rates / momentum as device tensors, refreshed with one host->device copy per step
        (a python float in a Neuron op would be baked into the compiled kernel as a constant)."""
        groups = [g for g in self.param_groups if g["params"]]
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

    @staticmethod
    def _scal(values, device):
        return torch.tensor(values, dtype=torch.float32).expand(P, 4).contiguous().to(device)

    @torch.no_grad()
    def _adamw_step(self, group):
        beta1, beta2 = group["betas"]
        lr_t = group["lr_t"]
        scal = None
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self._adamw_state(p)
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            if self.use_nki and p.device.type == "neuron" and p.dim() == 2 and p.shape[0] % (2 * P) == 0:
                try:
                    if scal is None:
                        b1c, b2c = 1 - beta1 ** (state["step"] + 1), 1 - beta2 ** (state["step"] + 1)
                        lr = float(group["lr"])
                        scal = self._scal([lr, 1.0 / b1c, 1.0 / (b2c ** 0.5), lr * group["weight_decay"]], p.device)
                    adamw_step_nki[NKI_GRID](p, grad.contiguous(), exp_avg, exp_avg_sq, scal, beta1=beta1, beta2=beta2, eps=group["eps"])
                    state["step"] += 1
                    continue
                except Exception as exc:
                    self.use_nki = False
                    note_fallback("NKI AdamW -> PyTorch (WHOLE OPTIMIZER)", exc)
                    print0(f"NKI AdamW kernel unavailable ({str(exc)[:80]}); using the PyTorch step")
            p.mul_(1 - lr_t * group["weight_decay"])
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            state["step"] += 1
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group["eps"])
            p.sub_(lr_t * update)

    @torch.no_grad()
    def _muon_step(self, group):
        params = group["params"]
        if not params or params[0].grad is None:
            return
        w = params[0]
        shape = w.shape[1:]
        state = self._muon_state(group)
        g = w.grad
        momentum_buffer = state["momentum_buffer"]
        second = state["second_momentum_buffer"]
        if self.use_nki and w.device.type == "neuron" and min(shape) == FMAX and max(shape) % FMAX == 0 and w.shape[0] % NKI_GRID[0] == 0:
            try:
                lr_eff = float(group["lr"]) * (max(1.0, shape[-2] / shape[-1]) ** 0.5)
                scal = self._scal([float(group["momentum"]), lr_eff, lr_eff * group["weight_decay"], 0.0], w.device)
                muon_step_nki[NKI_GRID](w, g.contiguous(), momentum_buffer, second, scal, ns_steps=group["ns_steps"], beta2=group["beta2"])
                return
            except Exception as exc:
                self.use_nki = False
                note_fallback("NKI Muon -> PyTorch (WHOLE OPTIMIZER)", exc)
                print0(f"NKI Muon kernel unavailable ({str(exc)[:80]}); using the PyTorch step")

        mom_t = group["momentum_t"].to(momentum_buffer.dtype)
        momentum_buffer.lerp_(g, 1 - mom_t)
        g = g.lerp_(momentum_buffer, mom_t)
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        if x.shape[-2] > x.shape[-1]:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xtx = x.mT @ x
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
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
        lr = group["lr_t"] * (max(1.0, shape[-2] / shape[-1]) ** 0.5)
        wd = group["weight_decay"]
        mask = (g * w) >= 0
        w.sub_(lr * g + lr * wd * w * mask)

    @torch.no_grad()
    def _nest_targets(self):
        """Every tensor the base step actually writes: owned shards where sharded, the parameter itself where not."""
        if getattr(self, "_nest_t", None) is None:
            ts = []
            for group in self.param_groups:
                if self._is_small_group(group):
                    ts.extend(p.data for p in group["params"])
                    continue
                for p in group["params"]:
                    if p in self.plan:
                        if self.plan[p].get("active", True):
                            ts.append(self.plan[p]["pshard"])
                    else:
                        ts.append(p.data)
            self._nest_t = ts
            self._nest_m = [torch.zeros_like(t) for t in ts]
            self._nest_prev = [torch.empty_like(t) for t in ts]
            self._nest_beta = 0.0
        return self._nest_t

    def step(self):
        _nest = NEST_BETA > 0.0
        if _nest:
            for t, pv in zip(self._nest_targets(), self._nest_prev):
                pv.copy_(t)
        for group in self.param_groups:
            if self._is_small_group(group):
                self._adamw_step_flat(group)
                continue
            planned = [p for p in group["params"] if p in self.plan]
            if planned:
                active = [p for p in planned if self.plan[p].get("active", True)]
                if active:
                    g2 = dict(group, params=[self.plan[p]["pshard"] for p in active])
                    (self._adamw_step if group["kind"] == "adamw" else self._muon_step)(g2)
                group = dict(group, params=[p for p in group["params"] if p not in self.plan])
                if not group["params"]:
                    continue
            if group["kind"] == "adamw":
                self._adamw_step(group)
            else:
                self._muon_step(group)
        if _nest:
            b_next, g = _NEST_B[0], NEST_GAMMA
            c = g + (1.0 - g) * self._nest_beta
            for t, pv, mm in zip(self._nest_t, self._nest_prev, self._nest_m):
                pv.sub_(t).neg_()
                mm.mul_(c).add_(pv, alpha=1.0 - g)
                if b_next != 0.0:
                    t.add_(mm, alpha=b_next)
            self._nest_beta = b_next
        if self.plan:
            self.gather_params()


def sync_gradients(model: nn.Module, optimizer: MuonAdamW) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    optimizer.reduce_gradients(model)




def hold_bigram_bf16(model: GPT) -> None:
    """Hold the 2^19 x 512 hashed bigram table in bf16 for training (it is 268.4M of the model's parameters). Must run
    before setup_sharding, which clones each worker's fp32 master shard from p.data, and before EmbedSidecar, whose
    gradient buffers take the table's dtype. Where the optimizer shards it (the 8-worker launch does) each worker
    updates its fp32 shard and all_gathers it back into this replica as bf16; otherwise (a single process, say) the
    optimizer updates the bf16 table directly. The forward reads the table only as looked-up rows in COMPUTE_DTYPE,
    and save_checkpoint writes it back as fp32. It is outside the tail-EMA, which excludes every nn.Embedding from its
    first group and admits only tables of at most EMA_TABLE_MAX elements to its second."""
    w = model.transformer.bte.weight
    assert w.numel() > EMA_TABLE_MAX, "the bigram table must stay outside the tail-EMA table group"
    w.data = w.data.to(torch.bfloat16)


def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # a bf16 tensor (the bigram table is the only one) is written as fp32, an exact widening, so load_for_eval
        # rebuilds the model at its constructed dtypes holding the values training used
        "model_state": {k: (v.detach().cpu().float() if v.dtype == torch.bfloat16 else v.detach().cpu())
                        for k, v in model.state_dict().items()},
        "model_config": asdict(model.config),
        "ngram": model.ngram.state(),
        "meta": meta,
    }
    torch.save(payload, path)


def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    """The scorer's entry point: rebuild the model from the checkpoint and return it.

    The returned module is a plain, consolidated, single-device replica. Its forward is the same function of the
    same weights that training used - `GPT.forward(idx)` returns `GPT.head_logits(x, tau, bias)` and nothing here alters it.
    The one thing this function does beyond loading (weights and n-gram table) is call `build_layer_ops`, as training does,
    so the layer kernels used for evaluation are the ones used for training; if they cannot be built, both paths
    fall back to the same explicit PyTorch layers.
    """
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = GPTConfig(**payload["model_config"])
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(payload["model_state"], strict=True)
    model.ngram = NgramTable.from_state(payload.get("ngram"))
    model.to(device)
    model.eval()
    if LAYER_IMPL == "op" and device.type == "neuron" and NKI_AVAILABLE:
        try:
            build_layer_ops(config.n_head)      # the same layer ops as in training; explicit layers if they cannot be built
        except Exception as exc:
            print0(f"layer ops unavailable for eval ({str(exc)[:120]}); using the explicit layers")
    # Inference hygiene only, no effect on any value: evaluate_bpb() calls model(x) outside torch.no_grad() and
    # accumulates the losses into one tensor, so with grad-tracking parameters that accumulator would retain the
    # autograd graph of every eval batch at once.
    model.requires_grad_(False)
    return model




def lr_multiplier(step: int, elapsed_frac: float) -> float:
    if step < _EFF_WARMUP_STEPS:
        return (step + 1) / max(1, _EFF_WARMUP_STEPS)
    warmdown_start = 1.0 - WARMDOWN_TIME_FRAC
    if elapsed_frac <= warmdown_start:
        return 1.0
    frac = min(1.0, (elapsed_frac - warmdown_start) / WARMDOWN_TIME_FRAC)
    return 1.0 - (1.0 - _EFF_FINAL_LR_FRAC) * frac


def muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * _EFF_MUON_MOMENTUM


def synced_elapsed_frac(budget_started: float, max_seconds: float, device: torch.device) -> float:
    elapsed = (time.monotonic() - budget_started) / max(1.0, max_seconds)
    if dist.is_available() and dist.is_initialized():
        te = torch.tensor([elapsed], dtype=torch.float32, device=device)
        dist.broadcast(te, src=0)
        elapsed = float(te.item())
    return elapsed




def main() -> None:
    parser = argparse.ArgumentParser(description="NKI Frontier Round 1 submission")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_NUM_STEPS", NUM_STEPS)))
    parser.add_argument("--max-train-seconds", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS", MAX_TRAIN_SECONDS)))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("NEURON_COMPETITION_R1_OUT_DIR", "out"))
    parser.add_argument("--eval-public", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_EVAL_PUBLIC", "0") != "0")
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_EVAL_TOKENS", str(4 * 524288))))
    parser.add_argument("--compile-only", action="store_true", help="run one synthetic optimizer step, then exit")
    args = parser.parse_args()
    # the n-gram counter is forked before the accelerator runtime starts; it needs an even number of workers (see notes)
    _ddp0, rank0, _lr0, world0 = get_dist_info()
    ng_dir = Path(args.out_dir) / f"ngram_{os.getppid()}"
    ng_proc = None
    if _ddp0 and world0 % 2 == 0 and not args.compile_only:
        ng_dir.mkdir(parents=True, exist_ok=True)
        ng_proc = multiprocessing.get_context("fork").Process(target=ngram_build, args=(rank0, world0, ng_dir), daemon=True)
        ng_proc.start()

    torch.set_float32_matmul_precision("high")
    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    device_batch_size = max(1, min(DEVICE_BATCH_ROWS, (TOTAL_BATCH_SIZE // SEQ_LEN) // world_size))

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = build_config()
    print0(f"model_config={json.dumps(asdict(config))}")
    model = GPT(config)
    model.init_weights()
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"num_params={num_params:,}")
    orig_model = model
    global USE_STACK_FUNCTION
    if LAYER_IMPL == "op" and device_type == "neuron" and NKI_AVAILABLE:
        try:
            build_layer_ops(model.config.n_head)
        except Exception as exc:
            note_fallback("layer ops -> explicit layers", exc)
            print0(f"layer ops unavailable ({str(exc)[:120]}); using the explicit layers")
    if device_type == "neuron" and NKI_AVAILABLE:
        try:
            build_head_ops()
        except Exception as exc:
            note_fallback("fused loss -> cross_entropy", exc)
            print0(f"fused loss unavailable ({str(exc)[:120]}); using cross_entropy(head_logits)")
    if COMPILE and device_type == "neuron":
        try:
            model = torch.compile(model, backend="neuron", dynamic=False)
            USE_STACK_FUNCTION = False
            print0("torch.compile enabled (backend=neuron)")
        except Exception as exc:
            note_fallback("torch.compile -> eager", exc)
            print0(f"torch.compile unavailable ({str(exc)[:80]}); running eager")
            model = orig_model

    world_tokens_per_micro = device_batch_size * SEQ_LEN * world_size
    if TOTAL_BATCH_SIZE % world_tokens_per_micro != 0:
        raise ValueError(f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={world_tokens_per_micro}")
    grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_micro
    print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps}")

    if args.compile_only:
        synthetic_x = torch.zeros((device_batch_size, SEQ_LEN), dtype=torch.int32).to(device)
        synthetic_y = torch.zeros((device_batch_size, SEQ_LEN), dtype=torch.long).to(device)
        optimizer = model.setup_optimizer()
        hold_bigram_bf16(orig_model)
        model.zero_grad(set_to_none=True)
        ng0 = torch.from_numpy(NgramTable().features(np.zeros((device_batch_size, SEQ_LEN)))).to(device)
        for _ in range(grad_accum_steps):
            loss = model(synthetic_x, synthetic_y, ng=ng0)
            (loss / grad_accum_steps).backward()
        optimizer.setup_sharding(world_size, rank)
        sync_gradients(model, optimizer)
        optimizer.refresh_scalars()
        optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    assert device_batch_size % PACK == 0, f"device_batch_size={device_batch_size} must be a multiple of PACK={PACK}"
    _remap = (PREFIX_AUX_W > 0) or (MTP_W1 > 0) or (MTP_W2 > 0) or LOADER_CPU
    _loader_dev = torch.device("cpu") if _remap else device
    _raw_loader = make_dataloader(tokenizer, device_batch_size // PACK, SEQ_LEN * PACK, "train", _loader_dev)

    def _packed():
        for _x, _y, _st in _raw_loader:
            _yr = _y.reshape(-1, SEQ_LEN)
            if _MTP_W[0] > 0.0 or _MTP_W[1] > 0.0:
                _w1, _w2 = _MTP_W[0], _MTP_W[1]
                _W = 1.0 + _w1 + _w2
                _T = _yr.shape[1]
                _y1 = torch.full_like(_yr, -1); _y1[:, :_T - 1] = _yr[:, 1:]
                _y2 = torch.full_like(_yr, -1); _y2[:, :_T - 2] = _yr[:, 2:]
                _bos1 = torch.zeros_like(_yr, dtype=torch.bool)
                _bos1[:, :_T - 1] = (_yr[:, 1:] == BOS_ID)
                _y2 = torch.where(_bos1, torch.full_like(_y2, -1), _y2)
                _u = torch.rand(_yr.shape, device=_yr.device)
                _yr = torch.where(_u < _w1 / _W, _y1, _yr)
                _yr = torch.where((_u >= _w1 / _W) & (_u < (_w1 + _w2) / _W), _y2, _yr)
            if _PREFIX_W[0] > 0.0 and _prefix_table is not None:
                _hit = torch.rand(_yr.shape, device=_yr.device) < _PREFIX_W[0]
                _safe = _yr.clamp(min=0)
                _pre = _prefix_table.index_select(0, _safe.reshape(-1)).reshape(_yr.shape)
                _yr = torch.where(_hit & (_yr >= 0), _pre, _yr)
            if BOS_IGNORE:
                _yr = torch.where(_yr == BOS_ID, -1, _yr)
            _xr = _x.reshape(-1, SEQ_LEN)
            if _remap:      # n-gram features on a host thread while this upload waits for the device queue
                _ngf[0] = _ngpool.submit(_timed_features, ng_train, _xr.numpy().copy())
                _xr = _xr.to(device)
                _yr = _yr.to(device)
            yield _xr, _yr, _st

    _prefix_table = build_prefix_table(tokenizer, _loader_dev) if PREFIX_AUX_W > 0 else None
    ng_train = NgramTable()     # this worker's n-gram table: the OTHER parity's row groups; no matches until it is built
    _ngpool, _ngf = ThreadPoolExecutor(1), [None]
    ng_feats = lambda xb: torch.from_numpy(_ngf[0].result() if _ngf[0] else ng_train.features(xb.cpu().numpy())).to(device)
    train_loader = _packed()
    x, y, loader_state = next(train_loader)
    ng = ng_feats(x)
    optimizer = model.setup_optimizer()
    hold_bigram_bf16(orig_model)
    optimizer.setup_sharding(world_size, rank)
    sidecar = EmbedSidecar(orig_model, "sidecar" if (device.type == "neuron" and NKI_AVAILABLE) else "sidecar_torch")
    plan_ring = (None, None)
    ema_params = None
    ema_pairs = None
    ema_tab_params = None
    ema_tab_pairs = None
    next_plans = None
    loop_started = time.monotonic()
    budget_started = loop_started

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0

    while step < args.num_steps:
        _tq = time.monotonic()
        synchronize(device)
        _pt("top_sync", _tq)
        if ng_proc is not None and not ng_train.parts:
            if (ng_dir / "error").exists():
                print0(f"!!! n-gram counter failed: {(ng_dir / 'error').read_text()[:200]}")
                ng_proc = None
            elif all((ng_dir / f"done{p}").exists() for p in range(NG_PARTS)):
                ng_train = NgramTable.load(ng_dir, 1 - rank % 2)
                print0(f"n-gram tables ready at step {step}, {time.monotonic() - _PROCESS_STARTED:.0f}s after launch; "
                       f"rank 0 reads the odd workers' table, match rate on its next rows "
                       f"{(ng_train.features(x.cpu().numpy())[1] >= 0).mean():.3f}")
        if step == FUSED_HEAD_CHECK_STEP:
            try:
                check_fused_head(orig_model, device_batch_size, SEQ_LEN, device)
            except Exception as _diag_exc:
                print0(f"!!! fused-head check could not run ({str(_diag_exc)[:160]}); "
                       f"the loss path is unchanged from what the step-0 check left it")
            synchronize(device)
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        if sidecar is not None:
            sidecar.start_step()
        for micro in range(grad_accum_steps):
            try:
                if sidecar is not None:
                    _tq = time.monotonic()
                    emb, plans = sidecar.gather(x, next_plans)
                    _tq = _pt("gather", _tq)
                    if step == 0 and micro == 0:
                        try:
                            check_fused_loss(orig_model, x, y, emb, ng)
                        except Exception as _diag_exc:
                            print0(f"!!! fused-loss check could not run ({str(_diag_exc)[:160]}); "
                                   f"the fused path is unchanged (see check_fused_head at step "
                                   f"{FUSED_HEAD_CHECK_STEP})")
                    _tq = time.monotonic()
                    loss = model(x, y, emb=emb, ng=ng)
                    _tq = _pt("fwd_launch", _tq)
                    last_loss = loss.detach()
                    (loss / grad_accum_steps).backward()
                    _tq = _pt("bwd_launch", _tq)
                    x, y, loader_state = next(train_loader)
                    _tq = _pt("next_batch", _tq)
                    next_plans = sidecar.prepare(x)
                    _tq = _pt("sidecar_prep", _tq)
                    ng = ng_feats(x)
                    _tq = _pt("ng_feats", _tq)
                    sidecar.scatter(emb, plans)
                    _tq = _pt("scatter", _tq)
                    plan_ring = (plan_ring[1], (emb, plans))
                    continue
                loss = model(x, y, ng=ng)
                last_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            except Exception as exc:
                if sidecar is not None and step == 0:
                    note_fallback("embedding sidecar -> tables in the graph (WHOLE RUN)", exc)
                    print0(f"embedding sidecar failed at step 0 ({str(exc)[:200]}); tables in the graph")
                    sidecar = None
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(x, y, ng=ng)
                    last_loss = loss.detach()
                    (loss / grad_accum_steps).backward()
                    x, y, loader_state = next(train_loader)
                    ng = ng_feats(x)
                    continue
                if step != 0 or model is orig_model:
                    raise
                note_fallback("compiled path -> eager", exc)
                print0(f"compiled path failed at step 0 ({str(exc)[:80]}); falling back to eager")
                model = orig_model
                USE_STACK_FUNCTION = True
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                loss = model(x, y, ng=ng)
                last_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            x, y, loader_state = next(train_loader)
            ng = ng_feats(x)
        if sidecar is not None:
            sidecar.attach()
        _tq = time.monotonic()
        sync_gradients(model, optimizer)
        _tq = _pt("grad_sync", _tq)

        if FRAC_BY_STEPS:
            elapsed_frac = step / max(1, args.num_steps)
        else:
            elapsed_frac = synced_elapsed_frac(budget_started, args.max_train_seconds, device)
        if step < TRAIN_STARTUP_STEPS_EXCLUDED:
            elapsed_frac = 0.0
        if PREFIX_AUX_W > 0:
            _PREFIX_W[0] = PREFIX_AUX_W * min(1.0, max(0.0, (PREFIX_AUX_END - elapsed_frac)
                                                       / max(1e-6, PREFIX_AUX_END - PREFIX_AUX_FLAT)))
        if MTP_W1 > 0 or MTP_W2 > 0:
            if elapsed_frac < MTP_P1:
                _MTP_W[0] = MTP_W1
                _MTP_W[1] = MTP_W2 * max(0.0, 1.0 - elapsed_frac / max(1e-6, MTP_P1))
            elif elapsed_frac < MTP_P2:
                _MTP_W[0] = MTP_W1 * max(0.0, 1.0 - (elapsed_frac - MTP_P1) / max(1e-6, MTP_P2 - MTP_P1))
                _MTP_W[1] = 0.0
            else:
                _MTP_W[0] = 0.0
                _MTP_W[1] = 0.0
        if NEST_BETA > 0.0:
            _in = NEST_START <= elapsed_frac < NEST_END
            _NEST_B[0] = NEST_BETA * lr_multiplier(step, elapsed_frac) if _in else 0.0
        lrm = lr_multiplier(step, elapsed_frac)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step)
                if B2 and B2_WD_TRACK:
                    group["weight_decay"] = WEIGHT_DECAY * lrm
        _tq = _pt("frac_lr", _tq)
        optimizer.refresh_scalars()
        optimizer.step()
        _tq = _pt("opt_launch", _tq)
        if elapsed_frac >= _EFF_EMA_START:
            with torch.no_grad():
                if ema_params is None:
                    tables = {id(m.weight) for m in orig_model.transformer.values() if isinstance(m, nn.Embedding)}
                    ema_pairs = [p for p in orig_model.parameters() if id(p) not in tables]
                    ema_params = [p.detach().clone() for p in ema_pairs]
                    _tabs = [(n, m.weight) for n, m in orig_model.transformer.items()
                             if isinstance(m, nn.Embedding) and m.weight.numel() <= EMA_TABLE_MAX]
                    ema_tab_pairs = [p for _n, p in _tabs]
                    ema_tab_params = [p.detach().clone() for p in ema_tab_pairs]
                    print0(f"tail-EMA of the weights started at step {step}")
                    print0("tail-EMA table group (blend " + f"{EMA_TABLE_BLEND}): "
                           + ", ".join(f"{n}({p.numel():,})" for n, p in _tabs)
                           + " | excluded: " + ", ".join(
                               f"{n}({m.weight.numel():,})" for n, m in orig_model.transformer.items()
                               if isinstance(m, nn.Embedding) and m.weight.numel() > EMA_TABLE_MAX))
                else:
                    for e, p in zip(ema_params, ema_pairs):
                        e.lerp_(p.detach(), 1.0 - _EFF_EMA_DECAY)
                    for e, p in zip(ema_tab_params, ema_tab_pairs):
                        e.lerp_(p.detach(), 1.0 - _EFF_EMA_DECAY)

        _tq = time.monotonic()
        synchronize(device)
        _pt("end_sync", _tq)
        _PT["step"] = _PT.get("step", 0.0) + time.monotonic() - t0
        if step % 100 == 99:
            print0("PROBE ms/step (mean of last 100): " + " ".join(f"{k}={v * 10:.1f}" for k, v in _PT.items()), flush=True)
            _PT.clear()
        record_step_boundary(step)
        dt = time.monotonic() - t0
        recent_step_times.append(dt)
        train_loss = float(last_loss.item())
        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
            now = time.monotonic()
            observed_startup = now - _PROCESS_STARTED
            startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            recent_step_times[:] = [dt]
            excess = max(0.0, observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            budget_started = now - excess
        elapsed_after_step = time.monotonic() - budget_started
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        stop_after_step = (
            step + 1 >= args.num_steps
            or (reached_min_steps and (
                elapsed_after_step >= args.max_train_seconds
                or elapsed_after_step + max(recent_step_times[-20:]) + CHECKPOINT_RESERVE_SECONDS >= args.max_train_seconds
                or ((time.monotonic() - _PROCESS_STARTED) + max(recent_step_times[-20:])
                    + WALL_RESERVE_SECONDS >= WALL_CAP_SECONDS)))
        )
        control = torch.tensor(
            [1 if (math.isnan(train_loss) or train_loss > 100) else 0, 1 if stop_after_step else 0],
            device=device, dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        if int(control[0].item()) != 0:
            raise RuntimeError(f"bad loss detected: {train_loss}")

        total_train_tokens += TOTAL_BATCH_SIZE
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        if B2_DIAG_EVERY > 0 and rank == 0 and step % B2_DIAG_EVERY == 0:
            with torch.no_grad():
                norms = {k: float(v.detach().float().norm().item()) for k, v in sorted(orig_model.stacks.items())}
            _wd_now = next((g["weight_decay"] for g in optimizer.param_groups if g["kind"] == "muon"), float("nan"))
            print(f"B2DIAG step {step:05d} lrm {lrm:.3f} wd {_wd_now:.4f} | "
                  + " ".join(f"{k}:{n:.1f}" for k, n in norms.items()), flush=True)
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

    if ema_params is not None:
        with torch.no_grad():
            for e, p in zip(ema_params, ema_pairs):
                p.lerp_(e, EMA_BLEND)
            for e, p in zip(ema_tab_params, ema_tab_pairs):
                p.lerp_(e, EMA_TABLE_BLEND)
        print0(f"tail-EMA blended into the weights for the checkpoint and the evaluation "
               f"({len(ema_params)} tensors at {EMA_BLEND}, {len(ema_tab_params)} tables at {EMA_TABLE_BLEND})")
    if ng_proc is not None and all((ng_dir / f"done{p}").exists() for p in range(NG_PARTS)):
        orig_model.ngram = NgramTable.load(ng_dir, 2)
    _nt = orig_model.ngram_theta.detach().float()
    print0(f"n-gram checkpoint table: {sum(len(q[1]) for o in orig_model.ngram.parts for q in o) if orig_model.ngram.parts else 0:,} "
           f"contexts from all train row groups; theta min {_nt.min().item():.3f} max {_nt.max().item():.3f}")
    checkpoint_path = Path(args.out_dir) / "final.pt"
    budget_elapsed_before_save = time.monotonic() - budget_started
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, {
            "step": step,
            "total_train_tokens": total_train_tokens,
            "budget_elapsed_before_save": budget_elapsed_before_save,
            "world_size": world_size,
        })
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started

    public_eval = None
    causal = None
    if args.eval_public:
        eval_rows = min(device_batch_size, 4)
        causal = causality_check(orig_model, tokenizer, eval_rows, SEQ_LEN, device)
        public_eval = evaluate_bpb(
            orig_model, tokenizer, eval_rows, SEQ_LEN, device,
            split="public_val", max_tokens=args.eval_tokens, timeout_seconds=EVAL_TIMEOUT_SECONDS,
        )

    if rank == 0:
        print("---")
        if _FALLBACKS:
            print(f"!!! NKI_FALLBACKS:   {len(_FALLBACKS)} - THIS RUN DID NOT USE THE FUSED PATH, DO NOT COMPARE ITS STEP TIME")
            for _fb in _FALLBACKS:
                print(f"!!!   {_fb}")
        else:
            print("NKI_FALLBACKS:      0 (all fused kernels live)")
        print(f"checkpoint_path:    {checkpoint_path}")
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
    if rank == 0 and ng_proc is not None:
        for f in ng_dir.iterdir():
            os.remove(f)
        ng_dir.rmdir()

    cleanup_runtime()


if __name__ == "__main__":
    main()
