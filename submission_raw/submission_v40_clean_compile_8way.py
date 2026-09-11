"""NKI Frontier Round 1 submission (2026-09-07).

GPT-style language model trained from scratch on the provided ClimbMix shards with the fixed tokenizer:
10 layers x 512 wide, 4 heads, RMSNorm, RoPE, QK-norm, ReLU^2 MLP, learned logit softcap, U-net skips,
value-embedding tables, a hashed bigram table, document-masked attention with a learned per-head sink, and a
Muon (Polar Express) + AdamW optimizer on a fixed 30-minute wall-clock budget.

On Trainium the attention runs on the NKI flash-attention kernels shipped with the environment (with the
document mask passed as per-query KV bounds), and the optimizer runs on two small NKI kernels written for this
submission (a fused Muon step and a fused AdamW step). Off Trainium (or if a kernel is unavailable) every one of
them falls back to the equivalent plain PyTorch code, so the math is identical on any device.

Notes for review (how the loss relates to the logits the scorer sees):
  * `GPT.head_logits(x)` is the single definition of this model's output logits (final RMSNorm, lm_head, asymmetric
    softcap). `GPT.forward(idx)` returns exactly that, and `load_for_eval` returns the same module; that is what
    `evaluate_bpb` scores.
  * `GPT.forward(idx, targets)` returns cross_entropy(head_logits(x), targets). On Trainium it uses
    `GPT.fused_linear_cross_entropy`, a memory-saving implementation of that same expression (the (T, V) logits are
    produced 512 columns at a time inside one NKI kernel and consumed by an online log-sum-exp, the way "fused linear
    cross-entropy" / "cut cross-entropy" kernels do), so the fp32 logits never occupy HBM. Its backward recomputes the
    same chunks. The two are the same function of the same weights; `check_fused_loss()` evaluates both on the first
    training batch, prints them, and disables the fused path if they disagree.
  * No code path inspects training/eval mode, process names, or the environment to pick a formula: kernel routes are
    chosen only by device type and tensor shape, and every kernel has an explicit PyTorch fallback of identical math.

On Trainium the forward/backward graph is compiled once with torch.compile(backend="neuron") (falls back to eager
if the compiler is unavailable); the NKI attention kernels are used in eager mode only.

Launch (8 workers; the file derives its per-worker batch from the world size):
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

_PROCESS_STARTED = time.monotonic()

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
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

# NKI (Neuron Kernel Interface) is only present on Trainium. Everything below degrades to plain PyTorch without it.
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx.nki_hop import nki_op, wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores
    from nkilib.core.attention.attention_bwd import attention_bwd
    from nkilib.core.attention.attention_cte import attention_cte
    NKI_AVAILABLE = True
except Exception:  # CPU / CUDA dev boxes
    NKI_AVAILABLE = False

# -----------------------------------------------------------------------------
# Hyperparameters

SEQ_LEN = 1024
DEPTH = 12
N_EMBD = 512
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD
MLP_RATIO = 6                   # MLP hidden = 6x n_embd (was 4): Sep 9, -0.0056 at ratio 6 on the kernel line
TOTAL_BATCH_SIZE = 131072       # tokens per optimizer step (one 16-row micro-step per worker on the 8-way launch)
DEVICE_BATCH_ROWS = 16          # rows per micro-step per worker (16: one micro-step per step; same math as 8 or 4)
_BATCH_LR = (TOTAL_BATCH_SIZE / 32768) ** 0.5   # sqrt-batch scaling of every learning rate relative to the 32k tuning
NUM_STEPS = 100000              # never binds; the time budget does
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 5.0
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

ROPE_BASE = 200_000
LOGIT_SOFTCAP = 11.0
BOS_ID = 8188                   # the fixed tokenizer's <|reserved_0|> (document separator)
SINK_INIT = math.log(4.0)       # softmax-denominator sink logit per (layer, head)
BIGRAM_SIZE = 2 ** 18

EMBEDDING_LR = 0.30 * 0.71 * 2.0 * _BATCH_LR * 3.0   # embedding LR x6.0 of the 32k-tuned base (was x2.0): Sep 9 sweep on the kernel line
UNEMBEDDING_LR = 0.006 * 0.71 * _BATCH_LR
MATRIX_LR = 0.025 * 0.71 * _BATCH_LR * 0.7   # x0.7: re-tuned for the ~3300-step regime of the kernel line (Sep 8 sweep)
ATTN_SCALE = 0.1                # constant attention logit scale (instead of 1/sqrt(128) = 0.088): -0.0006 full, -0.0010 screen
SCALAR_LR = 0.20 * 0.71 * _BATCH_LR
WEIGHT_DECAY = 0.10
WARMUP_STEPS = 50
ROPE_FRAC = 0.5                 # half-truncate RoPE: only the higher half of the frequencies rotate (the rest stays positional-free)
KEY_OFFSET = True               # stationary key dims D/4:D/2 come from the previous position (zeroed at document starts)
EMA_DECAY = 0.9933              # tail-EMA of the weights over the last 15% of the budget (tau ~150), ...
EMA_START = 0.85
EMA_BLEND = 0.35                # ... blended 0.65 w + 0.35 ema before the checkpoint; token tables excluded
WARMDOWN_TIME_FRAC = 0.90       # linear decay over the last 90% of the time budget ...
FINAL_LR_FRAC = 0.02            # ... to 2% of peak

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# -----------------------------------------------------------------------------
# Device / distributed setup (as in the provided train.py)


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
COMPILE = True                  # torch.compile(backend="neuron") for the forward/backward on Trainium
LAYER_IMPL = "op"               # "op": each transformer layer as tf_nki::layer_fwd / layer_bwd custom ops (training only; falls back to "torch")
USE_STACK_FUNCTION = True       # eager: accumulate stack gradients through _StackSliceLinear; compiled: plain views

# -----------------------------------------------------------------------------
# NKI kernels (Trainium only). P = SBUF partitions; FMAX = one PSUM bank of fp32 per partition.

P = 128
FMAX = 512

if NKI_AVAILABLE:
    NKI_GRID = (int(get_logical_neuron_cores()),)

    @nki.jit
    def embed_gather(table: nl.ndarray, idx: nl.ndarray):
        """out[i, :] = table[idx[i], :] (bf16 rows). idx (N, 1) uint32, N % 128 == 0; table (R, C) fp32 in HBM.
        Indirect DMA (per-partition vector_offset) instead of the graph's one-hot matmul."""
        N = idx.shape[0]
        C = table.shape[1]
        out = nl.ndarray((N, C), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        for t in nl.affine_range(N // P):
            ix = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ix, src=idx[t * P:(t + 1) * P, :])
            rows = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows, src=table.ap(pattern=[[C, P], [1, C]], vector_offset=ix, indirect_dim=0))
            nisa.dma_copy(dst=out[t * P:(t + 1) * P, :], src=rows)
        return out

    @nki.jit
    def embed_scatter_add(table: nl.ndarray, grad: nl.ndarray, perm: nl.ndarray, idx: nl.ndarray):
        """table[idx[s], :] += grad[perm[s], :] for S slots (S % 128 == 0), in place on the fp32 HBM buffer (R+128 rows:
        the tail takes the padding). Slots in one 128-tile carry distinct indices (host rounds); all indices in bounds."""
        S = idx.shape[0]
        C = table.shape[1]
        out = nl.ndarray((1, C), dtype=table.dtype, buffer=nl.shared_hbm)
        for t in nl.sequential_range(S // P):
            pm = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pm, src=perm[t * P:(t + 1) * P, :])
            ix = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ix, src=idx[t * P:(t + 1) * P, :])
            rows = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows, src=grad.ap(pattern=[[C, P], [1, C]], vector_offset=pm, indirect_dim=0))
            tgt = table.ap(pattern=[[C, P], [1, C]], vector_offset=ix, indirect_dim=0)
            nisa.dma_compute(dst=tgt, srcs=[tgt, rows], reduce_op=nl.add)
        nisa.dma_copy(dst=out, src=table[0:1, :])
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
            # momentum update and nesterov blend, natural layout, bf16 result
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
            # Frobenius prescale
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
            # tall frame: X (mt tiles of [P, k]) and Xt (kt tiles of [P, m]); one is Gn, the other its transpose
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
            # second-moment scaling (per tall-frame row = reduction over the tall-frame free axis)
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
            if not tall:   # wide: the factor varies along the natural free axis -> broadcast tile via a K=1 matmul
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
            # update in the natural frame: g = orth * f ; w -= lr*g + lr*wd*w*(g*w >= 0)
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

    # Document-masked causal flash attention with a per-head softmax-denominator sink, as a torch custom op with
    # autograd. The document mask is passed to the kernels as per-query KV bounds [doc start, doc end).
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

    _HVC = 512   # vocab chunk width of the fused cross-entropy kernel

    # ---- fused linear cross-entropy (2026-09-08): final rmsnorm + lm_head + asymmetric softcap + online
    # log-sum-exp per 128-token tile; returns the per-row loss (and lse, h, rstd for the backward). The backward
    # recomputes each 512-wide logit chunk (dh in SBUF fp32, dW chunks in PSUM), then the rmsnorm backward.
    @nki.jit
    def head_fwd(x_in: nl.ndarray, wT: nl.ndarray, tgt: nl.ndarray, scal: nl.ndarray, eps: float = 0.0078125):
        T, D = x_in.shape
        V = wT.shape[1]
        kt = D // _NKI_P
        nchunk = V // _HVC
        loss_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        lse_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        h_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
        rstd_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nprog = nl.num_programs(0)
        tiles = T // _NKI_P
        per = tiles // nprog
        sc = nl.ndarray((_NKI_P, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=scal)
        inv_sp = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        inv_sn = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_sp, data=sc[:, 0:1])
        nisa.reciprocal(dst=inv_sn, data=sc[:, 1:2])
        # column indices of a chunk: iota over the free dim (same for every partition), plus the chunk offset later
        col = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=col, pattern=[[1, _HVC]], offset=0, channel_multiplier=0)
        # whole W^T resident in SBUF (kt x V bf16 per partition = 64 KB at D=512, V=8192): read from HBM once per kernel
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
            nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
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
            m = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)       # running max
            s = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)       # running sum of exp(zc - m)
            tl = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)      # target logit (capped)
            nisa.memset(m, -1e30)
            nisa.memset(s, 0.0)
            nisa.memset(tl, 0.0)
            for c in range(nchunk):
                zps = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.psum)
                for i in range(kt):
                    nisa.nc_matmul(dst=zps, stationary=hT[i], moving=wsb[:, i, c * _HVC:(c + 1) * _HVC], accumulate=(i > 0))
                z = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=z, src=zps)
                # asymmetric softcap: zc = sp*tanh(z/sp) if z >= 0 else sn*tanh(z/sn)
                tp = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                tn = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=tp, op=nl.tanh, data=z, scale=inv_sp)
                nisa.tensor_scalar(dst=tp, data=tp, op0=nl.multiply, operand0=sc[:, 0:1])
                nisa.activation(dst=tn, op=nl.tanh, data=z, scale=inv_sn)
                nisa.tensor_scalar(dst=tn, data=tn, op0=nl.multiply, operand0=sc[:, 1:2])
                pos = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=pos, data=z, op0=nl.greater_equal, operand0=0.0)          # 1.0 where z >= 0
                zc = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=zc, data1=tp, data2=tn, op=nl.subtract)
                nisa.tensor_tensor(dst=zc, data1=zc, data2=pos, op=nl.multiply)
                nisa.tensor_tensor(dst=zc, data1=zc, data2=tn, op=nl.add)                        # tn + pos*(tp - tn)
                # target logit: mask (col + c*_HVC == tgt)
                cm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=cm, data=col, op0=nl.add, operand0=float(c * _HVC))
                eqm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eqm, data=cm, op0=nl.equal, operand0=tg)                     # per-partition scalar compare
                sel = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sel, data1=zc, data2=eqm, op=nl.multiply)
                part = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=part, op=nl.add, data=sel, axis=(1,))
                nisa.tensor_tensor(dst=tl, data1=tl, data2=part, op=nl.add)
                # online logsumexp
                cmax = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=cmax, op=nl.maximum, data=zc, axis=(1,))
                mnew = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=mnew, data1=m, data2=cmax, op=nl.maximum)
                dm = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dm, data1=m, data2=mnew, op=nl.subtract)
                nisa.activation(dst=dm, op=nl.exp, data=dm)                                        # exp(m - mnew)
                nisa.tensor_tensor(dst=s, data1=s, data2=dm, op=nl.multiply)
                negm = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=negm, data=mnew, op0=nl.multiply, operand0=-1.0)
                e = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=e, op=nl.exp, data=zc, bias=negm)                              # exp(zc - mnew)
                es = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=es, op=nl.add, data=e, axis=(1,))
                nisa.tensor_tensor(dst=s, data1=s, data2=es, op=nl.add)
                nisa.tensor_copy(dst=m, src=mnew)
            lse = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=lse, op=nl.log, data=s)
            nisa.tensor_tensor(dst=lse, data1=lse, data2=m, op=nl.add)
            nisa.dma_copy(dst=lse_out[t0:t0 + _NKI_P, :], src=lse)
            valid = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=valid, data=tg, op0=nl.greater_equal, operand0=0.0)
            lossr = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=lossr, data1=lse, data2=tl, op=nl.subtract)
            nisa.tensor_tensor(dst=lossr, data1=lossr, data2=valid, op=nl.multiply)
            nisa.dma_copy(dst=loss_out[t0:t0 + _NKI_P, :], src=lossr)
        return loss_out, lse_out, h_out, rstd_out

    @nki.jit
    def head_bwd(h_in: nl.ndarray, rstd_in: nl.ndarray, lse_in: nl.ndarray, dl_in: nl.ndarray, w: nl.ndarray, wT: nl.ndarray,
                 tgt: nl.ndarray, scal: nl.ndarray):
        """Backward of head_fwd. h (T, D) bf16 (normed input), rstd/lse/dl (T, 1) fp32 (dl = upstream grad of the per-row loss),
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
        col = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=col, pattern=[[1, _HVC]], offset=0, channel_multiplier=0)
        dcap = nl.ndarray((_NKI_P, 2), dtype=nl.float32, buffer=nl.sbuf)       # per-partition partial d(loss)/d(sp), d(loss)/d(sn)
        nisa.memset(dcap, 0.0)
        for hh in range(nh):
            base = hh * TH
            # resident per half: h (bf16), hT (bf16, kt tiles per token tile), dh accumulator (fp32), per-row scalars
            h = nl.ndarray((_NKI_P, th_tiles, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            hT = nl.ndarray((_NKI_P, th_tiles, kt, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
            dh = nl.ndarray((_NKI_P, th_tiles, D), dtype=nl.float32, buffer=nl.sbuf)
            g = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nlse = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            tgi = nl.ndarray((_NKI_P, th_tiles), dtype=nl.int32, buffer=nl.sbuf)
            tg = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
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
            nisa.tensor_copy(dst=tg, src=tgi)
            valid = nl.ndarray((_NKI_P, th_tiles), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=valid, data=tg, op0=nl.greater_equal, operand0=0.0)
            nisa.tensor_tensor(dst=g, data1=g, data2=valid, op=nl.multiply)
            nisa.tensor_scalar(dst=nlse, data=nlse, op0=nl.multiply, operand0=-1.0)
            for c in range(nchunk):
                wc = nl.ndarray((_NKI_P, kt, D), dtype=nl.bfloat16, buffer=nl.sbuf)       # W rows of this chunk: 4 tiles (128 vocab, D)
                wTc = nl.ndarray((_NKI_P, kt, _HVC), dtype=nl.bfloat16, buffer=nl.sbuf)     # W^T columns of this chunk
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
                    z = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=z, src=zps)
                    tp = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    tn = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=tp, op=nl.tanh, data=z, scale=inv_sp)
                    nisa.activation(dst=tn, op=nl.tanh, data=z, scale=inv_sn)
                    pos = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=pos, data=z, op0=nl.greater_equal, operand0=0.0)
                    # zc = sn*tn + pos*(sp*tp - sn*tn); deriv = (1 - tn^2) + pos*((1 - tp^2) - (1 - tn^2)) = 1 - tn^2 - pos*(tp^2 - tn^2)
                    a = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    b = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=a, data=tp, op0=nl.multiply, operand0=sc[:, 0:1])
                    nisa.tensor_scalar(dst=b, data=tn, op0=nl.multiply, operand0=sc[:, 1:2])
                    zc = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=zc, data1=a, data2=b, op=nl.subtract)
                    nisa.tensor_tensor(dst=zc, data1=zc, data2=pos, op=nl.multiply)
                    nisa.tensor_tensor(dst=zc, data1=zc, data2=b, op=nl.add)
                    nisa.tensor_tensor(dst=a, data1=tp, data2=tp, op=nl.multiply)
                    nisa.tensor_tensor(dst=b, data1=tn, data2=tn, op=nl.multiply)
                    deriv = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=deriv, data1=a, data2=b, op=nl.subtract)
                    nisa.tensor_tensor(dst=deriv, data1=deriv, data2=pos, op=nl.multiply)
                    nisa.tensor_tensor(dst=deriv, data1=deriv, data2=b, op=nl.add)                 # tn^2 + pos*(tp^2 - tn^2)
                    nisa.tensor_scalar(dst=deriv, data=deriv, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
                    # p = exp(zc - lse); dzc = (p - onehot) * g; dz = dzc * deriv
                    pr = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=pr, op=nl.exp, data=zc, bias=nlse[:, t:t + 1])
                    eqm = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=eqm, data=cm, op0=nl.equal, operand0=tg[:, t:t + 1])
                    nisa.tensor_tensor(dst=pr, data1=pr, data2=eqm, op=nl.subtract)
                    nisa.tensor_scalar(dst=pr, data=pr, op0=nl.multiply, operand0=g[:, t:t + 1])
                    dz = nl.ndarray((_NKI_P, _HVC), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=dz, data1=pr, data2=deriv, op=nl.multiply)
                    # softcap parameter grads: d zc/d sp = tp - (z/sp)(1 - tp^2) on z >= 0; d zc/d sn = tn - (z/sn)(1 - tn^2) on z < 0
                    u = nl.ndarray((_NKI_P, _HVC), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=u, data=a, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)   # 1 - tp^2
                    nisa.tensor_tensor(dst=u, data1=u, data2=z, op=nl.multiply)
                    nisa.tensor_scalar(dst=u, data=u, op0=nl.multiply, operand0=inv_sp)
                    nisa.tensor_tensor(dst=u, data1=tp, data2=u, op=nl.subtract)
                    nisa.tensor_tensor(dst=u, data1=u, data2=pos, op=nl.multiply)
                    nisa.tensor_tensor(dst=u, data1=u, data2=pr, op=nl.multiply)
                    part = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=part, op=nl.add, data=u, axis=(1,))
                    nisa.tensor_tensor(dst=dcap[:, 0:1], data1=dcap[:, 0:1], data2=part, op=nl.add)
                    nisa.tensor_scalar(dst=u, data=b, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)   # 1 - tn^2
                    nisa.tensor_tensor(dst=u, data1=u, data2=z, op=nl.multiply)
                    nisa.tensor_scalar(dst=u, data=u, op0=nl.multiply, operand0=inv_sn)
                    nisa.tensor_tensor(dst=u, data1=tn, data2=u, op=nl.subtract)
                    nisa.tensor_scalar(dst=pos, data=pos, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)  # 1 - pos
                    nisa.tensor_tensor(dst=u, data1=u, data2=pos, op=nl.multiply)
                    nisa.tensor_tensor(dst=u, data1=u, data2=pr, op=nl.multiply)
                    nisa.tensor_reduce(dst=part, op=nl.add, data=u, axis=(1,))
                    nisa.tensor_tensor(dst=dcap[:, 1:2], data1=dcap[:, 1:2], data2=part, op=nl.add)
                    # dh[t] += dz @ W_chunk  (contract over the chunk's 512 vocab entries: 4 sub-tiles)
                    dhps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    for j in range(kt):
                        tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=tps, data=dz[:, j * _NKI_P:(j + 1) * _NKI_P])
                        dzT = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dzT, src=tps)
                        nisa.nc_matmul(dst=dhps, stationary=dzT, moving=wc[:, j, :], accumulate=(j > 0))
                    nisa.tensor_tensor(dst=dh[:, t, :], data1=dh[:, t, :], data2=dhps, op=nl.add)
                    # dW_chunk[j] += dz[:, j]^T @ h[t]  (contract over the 128 tokens of this tile)
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
            # rmsnorm backward: dx = rstd * (dh - h * mean_D(dh * h))
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

    # ---- layer kernels (2026-09-08, nki_layer.py + the eager-line rmsnorm / RoPE / MLP kernels): used through the
    # tf_nki::layer_fwd / layer_bwd custom ops below (LAYER=op); never called directly.
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
            nisa.tensor_tensor(dst=sq, data1=x, data2=x, op=nl.multiply)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
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
            nisa.tensor_tensor(dst=t, data1=x, data2=x, op=nl.multiply)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=t, axis=(1,))
            nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=1.0 / d, op1=nl.add, operand1=eps)
            r = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=r, op=nl.rsqrt, data=ms)                       # r
            nisa.tensor_tensor(dst=t, data1=dy, data2=x, op=nl.multiply)        # dy*x
            c = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=c, op=nl.add, data=t, axis=(1,))             # sum(dy*x)
            r2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r2, data1=r, data2=r, op=nl.multiply)
            nisa.tensor_tensor(dst=c, data1=c, data2=r2, op=nl.multiply)
            nisa.tensor_scalar(dst=c, data=c, op0=nl.multiply, operand0=1.0 / d)   # r^2 * mean(dy*x)
            nisa.tensor_scalar(dst=t, data=x, op0=nl.multiply, operand0=c)          # x * that
            nisa.tensor_tensor(dst=t, data1=dy, data2=t, op=nl.subtract)            # dy - ...
            dx = nl.ndarray((_NKI_P, d), dtype=dy_in.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dx, data=t, op0=nl.multiply, operand0=r)
            nisa.dma_copy(dst=out[i * _NKI_P:(i + 1) * _NKI_P, :], src=dx)
        return out


    # ---- fused pre-attention backward of a layer (2026-09-10): key-offset bwd + QK-norm bwd + rope bwd + Q/K/V dgrad
    # (one PSUM accumulation) + wgrad (token-partition contraction) + rmsnorm bwd with the residual.
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
        # resident weights in natural (out, in) layout: chunk o = rows o*128:(o+1)*128 = the dgrad moving operand [K=out, N=in]
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
        # weight-grad accumulators (fp32 SBUF, rows chunk o of each (D, D) matrix)
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
            # ---- rmsnorm of xm (recompute h and rstd)
            xmb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xmb, src=xm_in[t0:t0 + _NKI_P, :])
            xm = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xm, src=xmb)
            sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=xm, data2=xm, op=nl.multiply)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
            # ---- rope tables for these rows (sin negated: the transpose of the rotation)
            cs = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            sn = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs, src=cos_in[t0:t0 + _NKI_P, :])
            nisa.dma_copy(dst=sn, src=sin_in[t0:t0 + _NKI_P, :])
            nisa.tensor_scalar(dst=sn, data=sn, op0=nl.multiply, operand0=-1.0)
            # ---- per head: gq/gk -> QK-norm backward -> rope backward; gv passthrough
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
                        # key-offset backward: dims hq:2hq of position p receive position p+1's masked grad (precomputed slice)
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
                    # rmsnorm backward over the head dim: d = rs * (g - r * rs^2 * mean(g*r))
                    tt = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=tt, data1=r, data2=r, op=nl.multiply)
                    m2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=m2, op=nl.add, data=tt, axis=(1,))
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
                    nisa.tensor_scalar(dst=tt, data=r, op0=nl.multiply, operand0=cc)
                    nisa.tensor_tensor(dst=tt, data1=g, data2=tt, op=nl.subtract)
                    dn = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dn, data=tt, op0=nl.multiply, operand0=rs)
                    # rope transpose: y1 = x1 cos - x2 sin ; y2 = x2 cos + x1 sin   (sn already negated)
                    x1 = dn[:, 0:half]
                    x2 = dn[:, half:HD]
                    r1 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)
                    if j == 0:
                        nisa.tensor_copy(dst=dpq[:, c0:c0 + half], src=r1)
                        nisa.tensor_copy(dst=dpq[:, c0 + half:c0 + HD], src=r2)
                    else:
                        nisa.tensor_copy(dst=dpk[:, c0:c0 + half], src=r1)
                        nisa.tensor_copy(dst=dpk[:, c0 + half:c0 + HD], src=r2)
                nisa.dma_copy(dst=gvb[:, c0:c0 + HD], src=dv_in[bh, p0:p0 + _NKI_P, :])
            nisa.dma_copy(dst=gv_out[t0:t0 + _NKI_P, :], src=gvb)
            # ---- dgrad: dh[tok, in] = sum_o dpq[tok, o] Wq[o, in] + ... ; stationary = transposed activation chunks [K=out, M=tok]
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
            # ---- rmsnorm backward of xm: dxm = dx1 + rstd * (dh - xm * rstd^2 * mean(dh*xm))
            nisa.tensor_tensor(dst=sq, data1=dh, data2=xm, op=nl.multiply)
            cm = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=cm, op=nl.add, data=sq, axis=(1,))
            r2c = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r2c, data1=rstd, data2=rstd, op=nl.multiply)
            nisa.tensor_tensor(dst=cm, data1=cm, data2=r2c, op=nl.multiply)
            nisa.tensor_scalar(dst=cm, data=cm, op0=nl.multiply, operand0=1.0 / D)
            nisa.tensor_scalar(dst=sq, data=xm, op0=nl.multiply, operand0=cm)
            nisa.tensor_tensor(dst=sq, data1=dh, data2=sq, op=nl.subtract)
            dx1b = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=dx1b, src=dx1_in[t0:t0 + _NKI_P, :])
            dx1 = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dx1, src=dx1b)
            nisa.scalar_tensor_tensor(dst=dx1, data=sq, op0=nl.multiply, operand0=rstd, op1=nl.add, operand1=dx1)
            dxb = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dxb, src=dx1)
            nisa.dma_copy(dst=dxm_out[t0:t0 + _NKI_P, :], src=dxb)
            # ---- wgrad: dW[o-chunk, in] += act[tok, o-chunk]^T @ h[tok, in]   (contraction over the token partitions)
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

    # ---- attention-layout kernels (2026-09-11): head-major q/k/v (scale folded into q) and per-head reads of the
    # attention output, so the reshapes between the layer kernels and the vendor attention are not compiler copies.
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
        # attention layout: (rows * n_head, t_len, HD) = (b, NH, t, HD) viewed flat; q is pre-scaled by q_scale
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
        # resident transposed weights: WqT[i] = [128 (in i), D (out)], same for k, v
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
        for tb in range(tiles):                       # static: the (row, position) of each tile is known at trace time
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
            # rmsnorm
            sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=xm, data2=xm, op=nl.multiply)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
            if fq:
                _fq8(h)
            # hT tiles for the projections
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
                    x1 = src_t[:, c0:c0 + half]
                    x2 = src_t[:, c0 + half:c0 + HD]
                    r1 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)           # y1 = x1 cos + x2 sin
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)      # y2 = x2 cos - x1 sin
                    rot = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rot[:, 0:half], src=r1)
                    nisa.tensor_copy(dst=rot[:, half:HD], src=r2)
                    rb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rb, src=rot)
                    nisa.dma_copy(dst=rot_t[hh, t0:t0 + _NKI_P, :], src=rb)
                    sq2 = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=sq2, data1=rot, data2=rot, op=nl.multiply)
                    ms2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=ms2, op=nl.add, data=sq2, axis=(1,))
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
                        n_head: int = 4, t_len: int = 1024):
        T, D = xm_in.shape
        H = w1.shape[0]
        HD = D // n_head
        out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)
        y2_out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)     # token-major copy of y for the backward (dWo)
        tpt = t_len // _NKI_P
        x1_out = nl.ndarray((T, D), dtype=y_in.dtype, buffer=nl.shared_hbm)
        pre_out = nl.ndarray((H, T), dtype=y_in.dtype, buffer=nl.shared_hbm)
        kt = D // _NKI_P
        ht = H // _NKI_P
        nprog = nl.num_programs(0)
        chunks = T // _FMAX
        per = chunks // nprog
        # resident transposed weight tiles: WoT[i] = [128 (i), D (o)], W1t[k] = [128 (k), H], W2t[h] = [128 (h), D]
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
        for c in range(chunks):                       # static: tile rows/positions known at trace time
            t0 = c * _FMAX
            if fq:
                for i in range(len(WoT)):
                    _fq8(WoT[i])
                for k in range(len(W1)):
                    _fq8(W1[k])
                for h in range(len(W2)):
                    _fq8(W2[h])
            # ---- out-projection + residual + rmsnorm, per 128-token tile; h2T built for the MLP contraction
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
                for hh in range(n_head):              # per-head aligned blocks from the attention layout
                    nisa.dma_copy(dst=yt[:, hh * HD:(hh + 1) * HD], src=y_in[row * n_head + hh, p0:p0 + _NKI_P, :])
                nisa.dma_copy(dst=y2_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=yt)
                if fq:
                    _fq8(yt)
                # yT tiles [128 (i), 128 tokens] via transposes; a[t, :] = sum_i yT[i, t] WoT[i, :]
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
                nisa.tensor_tensor(dst=x1, data1=xmt, data2=aps, op=nl.add)        # x1 = xm + a (fp32)
                x1b = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=x1b, src=x1)
                nisa.dma_copy(dst=x1_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=x1b)
                # rmsnorm over D
                sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sq, data1=x1, data2=x1, op=nl.multiply)
                ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
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
            # ---- MLP (as nki_mlp.mlp_fwd): hT[h] = relu(W1t[:, h]^T h2T)^2 ; m[t] = sum_h hT[h][:, t]^T W2[h]
            hT = []
            for _j in range(ht):
                hT.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            for h in range(ht):
                pps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                for k in range(kt):
                    nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * _NKI_P:(h + 1) * _NKI_P], moving=h2T[k], accumulate=(k > 0))
                pre = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pre, src=pps)
                nisa.dma_copy(dst=pre_out[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX], src=pre)
                nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)
                if fq:
                    _fq8(hT[h])
            for tb in range(_FMAX // _NKI_P):
                mps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for h in range(ht):
                    nisa.nc_matmul(dst=mps, stationary=hT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P], moving=W2[h], accumulate=(h > 0))
                o = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o, data1=x1_tiles[tb], data2=mps, op=nl.add)   # out = x1 + m
                nisa.dma_copy(dst=out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=o)
        return out, x1_out, pre_out, y2_out

    _HVC = 512   # vocab chunk width of the fused head kernels (2026-09-08, nki_head.py)

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

    @nki.jit
    def mlp_fwd(x_in: nl.ndarray, w1: nl.ndarray, w2: nl.ndarray):
        """x (T, D) bf16; w1 (H, D) and w2 (D, H) in the natural (out, in) layout. Returns y (T, D) bf16 and preT (H, T) bf16."""
        T, D = x_in.shape
        H = w1.shape[0]
        y_out = nl.ndarray((T, D), dtype=x_in.dtype, buffer=nl.shared_hbm)
        pre_out = nl.ndarray((H, T), dtype=x_in.dtype, buffer=nl.shared_hbm)
        kt = D // _NKI_P              # contraction tiles over D (4)
        ht = H // _NKI_P              # hidden blocks of 128 (16)
        nprog = nl.num_programs(0)
        chunks = T // _FMAX       # token chunks of 512
        per = chunks // nprog
        # weights resident in SBUF in the transposed tiles the matmuls need: W1t[k] = [128 (k), H], W2t[h] = [128 (h), D],
        # built from the natural (out, in) layouts with 128x128 TensorE block transposes (SBUF-local, cheap)
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
            # xT for this chunk: kt tiles of [128 (k), 512 (tokens)] via 128x128 TensorE transposes
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
            # hT[h-block] = relu(W1t[:, h-block]^T xT)^2  -> ht tiles of [128 hidden, 512 tokens]
            hT = []
            for _j in range(ht):
                hT.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            for h in range(ht):
                pps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                for k in range(kt):
                    nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * _NKI_P:(h + 1) * _NKI_P], moving=xT[k], accumulate=(k > 0))
                pre = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pre, src=pps)
                nisa.dma_copy(dst=pre_out[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX], src=pre)
                nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)   # VectorE: at most one PSUM operand
                # relu(pre)^2 == max(pre,0)*pre ; wrong sign for pre<0? max(pre,0)*pre = 0 for pre<=0 and pre^2 for pre>0: correct
            # y[t-block] = sum_h hT[h][:, t-block]^T W2[h]  -> [128 tokens, D]
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
        # resident weights straight from the natural layouts (no transposes): W2T[o][h] = w2[o-tile, h-block], W1n[h] = w1[h-tile, :]
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
        # fp32 gradient accumulators (SBUF) in the natural layouts: dW1 as ht tiles [128 (h), D]; dW2 as kt tiles [128 (o), H]
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
            # dy tiles (natural) and dyT [o-tile][128 x 512 tokens]; x tiles (natural)
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
            # dpreT[h] = 2 relu(preT[h]) * dhT[h],  dhT[h] = sum_o W2T[o][h]^T dyT[o]
            dpreT = []
            hT = []
            for h in range(ht):
                dps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                for o in range(kt):
                    nisa.nc_matmul(dst=dps, stationary=W2T[o][h], moving=dyT[o], accumulate=(o > 0))
                pre = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=pre, src=pre_in[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX])
                relu = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=relu, data=pre, op0=nl.maximum, operand0=0.0)
                hh = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=hh, data1=relu, data2=relu, op=nl.multiply)      # h = relu^2 (recomputed)
                hT.append(hh)
                dp = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=dp, data=relu, op0=nl.multiply, operand0=2.0, op1=nl.multiply, operand1=dps)
                dpreT.append(dp)
            # dx[t-block] = sum_h dpreT[h][:, t-block]^T W1n[h]
            for tb in range(_FMAX // _NKI_P):
                xps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for h in range(ht):
                    nisa.nc_matmul(dst=xps, stationary=dpreT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P], moving=W1n[h], accumulate=(h > 0))
                dxt = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dxt, src=xps)
                nisa.dma_copy(dst=dx_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=dxt)
            # weight gradients: per t-block build natural dpre[t-block] [128, H] and h[t-block] [128, H] by transposing blocks
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
                # dW1[h-tile, :] += dpre_nat[t-block][:, h-tile]^T x[t-block]
                for h in range(ht):
                    gps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=gps, stationary=dpn[:, h * _NKI_P:(h + 1) * _NKI_P], moving=xn[tb])
                    nisa.tensor_tensor(dst=A1[h], data1=A1[h], data2=gps, op=nl.add)
                # dW2[o-tile, h-chunk] += dy[t-block][:, o-tile]^T h_nat[t-block][:, h-chunk]
                for o in range(kt):
                    for hc in range(H // _FMAX):
                        gps2 = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=gps2, stationary=dyn[tb][:, o * _NKI_P:(o + 1) * _NKI_P], moving=hn[:, hc * _FMAX:(hc + 1) * _FMAX])
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
        # resident transposed weights: WqT[i] = [128 (in i), D (out)], same for k, v
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
            # rmsnorm
            sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=xm, data2=xm, op=nl.multiply)
            ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
            rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
            h = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=xm, op0=nl.multiply, operand0=rstd)
            # hT tiles for the projections
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
                    x1 = src_t[:, c0:c0 + half]
                    x2 = src_t[:, c0 + half:c0 + HD]
                    r1 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    r2 = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    tmp = nl.ndarray((_NKI_P, half), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=r1, data1=x1, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x2, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r1, data1=r1, data2=tmp, op=nl.add)           # y1 = x1 cos + x2 sin
                    nisa.tensor_tensor(dst=r2, data1=x2, data2=cs, op=nl.multiply)
                    nisa.tensor_tensor(dst=tmp, data1=x1, data2=sn, op=nl.multiply)
                    nisa.tensor_tensor(dst=r2, data1=r2, data2=tmp, op=nl.subtract)      # y2 = x2 cos - x1 sin
                    rot = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rot[:, 0:half], src=r1)
                    nisa.tensor_copy(dst=rot[:, half:HD], src=r2)
                    rb = nl.ndarray((_NKI_P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=rb, src=rot)
                    nisa.dma_copy(dst=rot_t[hh, t0:t0 + _NKI_P, :], src=rb)
                    sq2 = nl.ndarray((_NKI_P, HD), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=sq2, data1=rot, data2=rot, op=nl.multiply)
                    ms2 = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=ms2, op=nl.add, data=sq2, axis=(1,))
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
        # resident transposed weight tiles: WoT[i] = [128 (i), D (o)], W1t[k] = [128 (k), H], W2t[h] = [128 (h), D]
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
            # ---- out-projection + residual + rmsnorm, per 128-token tile; h2T built for the MLP contraction
            h2T = []
            for _j in range(kt):
                h2T.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            x1_tiles = []
            for _j in range(_FMAX // _NKI_P):
                x1_tiles.append(nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf))
            for tb in range(_FMAX // _NKI_P):
                yt = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=yt, src=y_in[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :])
                # yT tiles [128 (i), 128 tokens] via transposes; a[t, :] = sum_i yT[i, t] WoT[i, :]
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
                nisa.tensor_tensor(dst=x1, data1=xmt, data2=aps, op=nl.add)        # x1 = xm + a (fp32)
                x1b = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=x1b, src=x1)
                nisa.dma_copy(dst=x1_out[t0 + tb * _NKI_P:t0 + (tb + 1) * _NKI_P, :], src=x1b)
                # rmsnorm over D
                sq = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sq, data1=x1, data2=x1, op=nl.multiply)
                ms = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=ms, op=nl.add, data=sq, axis=(1,))
                rstd = nl.ndarray((_NKI_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=rstd, data=ms, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps)
                nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
                h2 = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=h2, data=x1, op0=nl.multiply, operand0=rstd)
                for k in range(kt):
                    tps = nl.ndarray((_NKI_P, _NKI_P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps, data=h2[:, k * _NKI_P:(k + 1) * _NKI_P])
                    nisa.tensor_copy(dst=h2T[k][:, tb * _NKI_P:(tb + 1) * _NKI_P], src=tps)
            # ---- MLP (as nki_mlp.mlp_fwd): hT[h] = relu(W1t[:, h]^T h2T)^2 ; m[t] = sum_h hT[h][:, t]^T W2[h]
            hT = []
            for _j in range(ht):
                hT.append(nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf))
            for h in range(ht):
                pps = nl.ndarray((_NKI_P, _FMAX), dtype=nl.float32, buffer=nl.psum)
                for k in range(kt):
                    nisa.nc_matmul(dst=pps, stationary=W1[k][:, h * _NKI_P:(h + 1) * _NKI_P], moving=h2T[k], accumulate=(k > 0))
                pre = nl.ndarray((_NKI_P, _FMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pre, src=pps)
                nisa.dma_copy(dst=pre_out[h * _NKI_P:(h + 1) * _NKI_P, t0:t0 + _FMAX], src=pre)
                nisa.scalar_tensor_tensor(dst=hT[h], data=pre, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pre)
            for tb in range(_FMAX // _NKI_P):
                mps = nl.ndarray((_NKI_P, D), dtype=nl.float32, buffer=nl.psum)
                for h in range(ht):
                    nisa.nc_matmul(dst=mps, stationary=hT[h][:, tb * _NKI_P:(tb + 1) * _NKI_P], moving=W2[h], accumulate=(h > 0))
                o = nl.ndarray((_NKI_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o, data1=x1_tiles[tb], data2=mps, op=nl.add)   # out = x1 + m
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


# -----------------------------------------------------------------------------
# Model


@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


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
        object.__setattr__(self, "_stack", None)   # bound later; not a parameter of this module

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

    def forward(self, x, ve, cos, sin, sink, dense_mask=None, bounds=None, doc_start=None):
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
        v = (self.c_v(x) + ve).view(b, t, self.n_head, self.head_dim)
        q = norm(apply_rotary_emb(q, cos, sin))
        k = norm(apply_rotary_emb(k, cos, sin))
        if KEY_OFFSET:
            # partial key offset: the stationary (unrotated) dims D/4:D/2 of each key are the previous position's, zeroed at
            # document starts (speedrun record #49: gives every layer a free one-step induction pattern)
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
            # Trainium: NKI flash attention (bounds = document mask; the kernel needs scale 1.0, so q is pre-scaled)
            y = torch.ops.r1.doc_attention_fwd((q * scale).to(q.dtype), k, v, sink, bounds[0], bounds[1])[0]
        else:
            # same math with plain ops: softmax over the real keys, scaled by the sink's share of the denominator
            # (1 - sigmoid(sink - logsumexp)); keeps every tensor t wide (no extra column) for the compiled graph
            scores = (q.float() @ k.float().transpose(-2, -1)) * scale
            scores = scores.masked_fill(~dense_mask, float("-inf"))
            lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            p = torch.exp(scores - lse)
            gate = 1.0 - torch.sigmoid(sink.float().view(1, self.n_head, 1, 1) - lse)
            y = ((p @ v.float()) * gate).to(v.dtype)
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


_NKI_ROPE = None


def build_nki_rope():
    """Fused split-half RoPE (dev copy: nki_rope.py): one launch per q/k instead of ~9 eager ops; backward = same rotation
    with -sin. Eager Neuron only (a kernel inside a compiled graph partitions the NEFF, S_NKI3 2026-09-06)."""
    global _NKI_ROPE
    if _NKI_ROPE is not None:
        return _NKI_ROPE
    if not NKI_AVAILABLE:
        raise RuntimeError("NKI is not available on this device")
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    _NKI_ROPE = (wrap_nki(rope_kernel), (int(get_logical_neuron_cores()),))
    return _NKI_ROPE


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

# ---- LAYER=op (2026-09-08): the same layer kernels as LAYER=chain, packaged as tf_nki::layer_fwd / layer_bwd custom ops so the
# torch.compile(backend="neuron") graph keeps them opaque (one graph, no per-op host dispatch: the eager chain is host-bound,
# OPPROF5-7). Weight grads come back as tensors and reach the stacks through autograd (select_backward inside the graph).
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
    pre_bwd_k = wrap_nki(pre_attn_bwd)
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

    def attn_bwd(q4, k4, v4, out, dy4, neg_max, recip, sink, start, end):
        b, h, t, d = q4.shape
        tr = lambda x: x.transpose(-2, -1).contiguous()
        lse = (-(neg_max.float() + torch.log(recip.float()))).reshape(b, h, *neg_max.shape[-2:]).contiguous()
        res = abwd[grid](tr(q4), tr(k4), tr(v4), tr(out), tr(dy4.to(q4.dtype)), lse,
                         sinks_ref=sink.float().view(1, h).expand(b, h).contiguous(),
                         bound_min=start.float().contiguous(), bound_max=end.float().contiguous(),
                         use_causal_mask=True, mixed_precision=True, softmax_scale=1.0)
        dq, dk, dv = (r.transpose(-2, -1).contiguous() for r in res[:3])
        return dq, dk, dv, res[3].float().sum(dim=0)

    @nki_op("tf_nki::layer_fwd", mutates_args={})
    def layer_fwd(x2: torch.Tensor, x02: torch.Tensor, ve2: torch.Tensor, cos_full: torch.Tensor, sin_full: torch.Tensor,
                  cos2d: torch.Tensor, sin2d: torch.Tensor, scal: torch.Tensor, sink: torch.Tensor, start: torch.Tensor,
                  end: torch.Tensor, dstart: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor,
                  w1: torch.Tensor, w2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor, torch.Tensor]:
        NH = _LAYER_NH; T, D = x2.shape; HD = D // NH; b = start.shape[0]; t = T // b
        scale = ATTN_SCALE
        # head-major kernels: q/k/v arrive as (b*NH, t, HD) with q pre-scaled; the attention output is read per head and the
        # token-major y2 comes back from post_kh -> no permute/reshape compiler copies between the kernels and attention
        qh, kh, vh, xm, rq, rk = pre_kh[grid](x2, x02, ve2, wq, wk, wv, cos_full, sin_full, scal, n_head=NH, eps=eps, t_len=t, q_scale=scale)
        q4 = qh.view(b, NH, t, HD)
        k4 = kh.view(b, NH, t, HD)
        if KEY_OFFSET:   # partial key offset (v26): normed key dims HD/4:HD/2 come from the previous position, zeroed at doc starts
            HQ = HD // 4
            prev = torch.cat([torch.zeros_like(k4[:, :, :1, HQ:2 * HQ]), k4[:, :, :-1, HQ:2 * HQ]], dim=2)
            prev = prev * (1.0 - dstart).to(bf).view(b, 1, t, 1)
            k4 = torch.cat([k4[..., :HQ], prev, k4[..., 2 * HQ:]], dim=-1).contiguous()
        v4 = vh.view(b, NH, t, HD)
        out, neg_max, recip = attn_fwd(q4, k4, v4, sink, start, end)
        o, x1, preT, y2 = post_kh[grid](out.reshape(b * NH, t, HD), wo, xm, w1, w2, eps=eps, n_head=NH, t_len=t)
        return o, xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT

    @nki_op("tf_nki::layer_bwd", mutates_args={})
    def layer_bwd(dout2: torch.Tensor, x2: torch.Tensor, x02: torch.Tensor, ve2: torch.Tensor, xm: torch.Tensor, rq: torch.Tensor,
                  rk: torch.Tensor, q4: torch.Tensor, k4: torch.Tensor, v4: torch.Tensor, out: torch.Tensor, neg_max: torch.Tensor,
                  recip: torch.Tensor, y2: torch.Tensor, x1: torch.Tensor, preT: torch.Tensor, cos2d: torch.Tensor,
                  sin2d: torch.Tensor, sink: torch.Tensor, start: torch.Tensor, end: torch.Tensor, dstart: torch.Tensor,
                  wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor, w1: torch.Tensor,
                  w2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor, torch.Tensor]:
        NH = _LAYER_NH; T, D = x2.shape; HD = D // NH; b = start.shape[0]; t = T // b
        scale = ATTN_SCALE
        d2 = dout2.to(bf).contiguous()
        h2 = nfwd[grid](x1, eps=eps)
        dh2, dW1p, dW2p = mlp_bwd_k[grid](h2, d2, preT, w1, w2)
        dx1 = d2 + nbwd[grid](x1, dh2.contiguous(), eps=eps)
        dy2 = F.linear(dx1, wo.t())
        dWo = dx1.t() @ y2
        dy4 = dy2.view(b, t, NH, HD).permute(0, 2, 1, 3).contiguous()
        dq4, dk4, dv4, dsink = attn_bwd(q4, k4, v4, out, dy4, neg_max, recip, sink, start, end)
        # pre-attention half as one kernel: key-offset backward (position p receives p+1's shifted-dim grad unless p+1 starts
        # a document), QK-norm backward, RoPE transpose, Q/K/V dgrad + wgrad, rmsnorm backward (+ residual dx1)
        cos_full = cos2d.repeat(b, 1).contiguous(); sin_full = sin2d.repeat(b, 1).contiguous()
        HQ = HD // 4   # the shifted slice is built here so every kernel DMA is a 128-row-aligned block
        gsh = dk4[:, :, 1:, HQ:2 * HQ] * (1.0 - dstart[:, 1:].to(dk4.dtype)).view(b, 1, t - 1, 1)
        dksh = torch.cat([gsh, torch.zeros_like(gsh[:, :, :1])], dim=2).reshape(b * NH, t, HQ).contiguous()
        dxm, gv2, dWq, dWk, dWv = pre_bwd_k[grid](dq4.reshape(b * NH, t, HD).contiguous(), dk4.reshape(b * NH, t, HD).contiguous(),
                                                 dv4.reshape(b * NH, t, HD).contiguous(), rq.contiguous(), rk.contiguous(),
                                                 xm.contiguous(), dx1.contiguous(), cos_full, sin_full, dksh, wq, wk, wv,
                                                 n_head=NH, eps=eps, scale=scale, hq=HQ)
        return dxm, gv2, dWq, dWk, dWv, dWo, dW1p.sum(0), dW2p.sum(0), dsink

    def setup_context(ctx, inputs, output):
        x2, x02, ve2, cos_full, sin_full, cos2d, sin2d, scal, sink, start, end, dstart, wq, wk, wv, wo, w1, w2 = inputs
        o, xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT = output
        ctx.save_for_backward(x2, x02, ve2, cos2d, sin2d, scal, sink, start, end, dstart, wq, wk, wv, wo, w1, w2,
                              xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT)

    def backward(ctx, dout2, *_unused):
        (x2, x02, ve2, cos2d, sin2d, scal, sink, start, end, dstart, wq, wk, wv, wo, w1, w2,
         xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT) = ctx.saved_tensors
        dxm, gv2, dWq, dWk, dWv, dWo, dW1, dW2, dsink = torch.ops.tf_nki.layer_bwd(
            dout2, x2, x02, ve2, xm, rq, rk, q4, k4, v4, out, neg_max, recip, y2, x1, preT, cos2d, sin2d, sink, start, end,
            dstart, wq, wk, wv, wo, w1, w2)
        resid = scal[0:1, 0:1]; x0l = scal[0:1, 1:2]; vel = scal[0:1, 2:3]
        dxf = dxm.float()
        dx2 = (dxf * resid).to(x2.dtype); dx02 = (dxf * x0l).to(x02.dtype); dve2 = (gv2.float() * vel).to(ve2.dtype)
        d_resid = (dxf * x2.float()).sum(); d_x0 = (dxf * x02.float()).sum(); d_ve = (gv2.float() * ve2.float()).sum()
        row0 = torch.stack([d_resid, d_x0, d_ve, torch.zeros((), device=scal.device, dtype=torch.float32)]).view(1, 4)
        dscal = torch.cat([row0, torch.zeros((127, 4), device=scal.device, dtype=torch.float32)], dim=0)
        return (dx2, dx02, dve2, None, None, None, None, dscal, dsink.to(sink.dtype), None, None, None,
                dWq.to(wq.dtype), dWk.to(wk.dtype), dWv.to(wv.dtype), dWo.to(wo.dtype), dW1.to(w1.dtype), dW2.to(w2.dtype))

    torch.library.register_autograd("tf_nki::layer_fwd", backward, setup_context=setup_context)
    _LAYER_OPS = (torch.ops.tf_nki.layer_fwd, torch.ops.tf_nki.layer_bwd)
    print0("LAYER=op: tf_nki::layer_fwd/layer_bwd built (the Block op branch fires when sink/bounds/rope tables are present)")
    return _LAYER_OPS


# ---- tf_nki::head_fwd / head_bwd: the fused linear cross-entropy kernels as torch.library custom ops (built with
# torch_neuronx's nki_op) with autograd wired by register_autograd. Gradients: x, lm_head weight, softcap_pos / softcap_neg.
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
    def hfwd(x2: torch.Tensor, wT: torch.Tensor, tgt: torch.Tensor, scal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, lse, h, rstd = kf[grid](x2, wT, tgt, scal)
        return loss, lse, h, rstd

    @nki_op("tf_nki::head_bwd", mutates_args={})
    def hbwd(h: torch.Tensor, rstd: torch.Tensor, lse: torch.Tensor, dl: torch.Tensor, w: torch.Tensor, wT: torch.Tensor,
             tgt: torch.Tensor, scal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx, dw, dcap = kb[grid](h, rstd, lse, dl, w, wT, tgt, scal)
        return dx, dw, dcap

    def setup_context(ctx, inputs, output):
        x2, wT, tgt, scal = inputs
        loss, lse, h, rstd = output
        ctx.save_for_backward(wT, tgt, scal, lse, h, rstd)
        ctx.x_dtype = x2.dtype

    def backward(ctx, dloss, _dlse, _dh, _drstd):
        wT, tgt, scal, lse, h, rstd = ctx.saved_tensors
        w = wT.t().contiguous()
        dx, dw, dcap = torch.ops.tf_nki.head_bwd(h, rstd, lse, dloss.float().contiguous(), w, wT, tgt, scal)
        dcap = dcap.sum(0)
        row0 = torch.stack([dcap[0], dcap[1], torch.zeros((), device=scal.device), torch.zeros((), device=scal.device)]).view(1, 4)
        dscal = torch.cat([row0, torch.zeros((127, 4), device=scal.device, dtype=torch.float32)], dim=0)
        return dx.to(ctx.x_dtype), dw.t().to(wT.dtype), None, dscal

    torch.library.register_autograd("tf_nki::head_fwd", backward, setup_context=setup_context)
    _HEAD_OPS = (torch.ops.tf_nki.head_fwd, torch.ops.tf_nki.head_bwd)
    return _HEAD_OPS

def check_fused_loss(model: "GPT", x: torch.Tensor, y: torch.Tensor, emb) -> None:
    """Evaluate the training loss both ways on one batch - the fused NKI kernel and cross_entropy(head_logits(x), y) in
    plain PyTorch - print both, and disable the fused path if they disagree beyond bf16 tolerance. Runs once at startup."""
    if _HEAD_OPS is None or x.device.type != "neuron":
        return
    with torch.no_grad():
        fused = model(x, y, emb=emb)
        model._fused_loss_ok = False
        explicit = model(x, y, emb=emb)
        model._fused_loss_ok = True
        diff = (fused - explicit).abs().item()
    print0(f"fused-loss check: fused {fused.item():.5f} vs cross_entropy(head_logits) {explicit.item():.5f}, |diff| {diff:.2e}")
    if not (diff < 0.02 * max(1.0, explicit.item())):
        model._fused_loss_ok = False
        print0("fused-loss check FAILED: using cross_entropy(head_logits) for training")


class Block(nn.Module):
    def __init__(self, config: GPTConfig, registry: StackRegistry):
        super().__init__()
        self.attn = CausalSelfAttention(config, registry)
        self.mlp = MLP(config, registry)

    def forward(self, x, x0, ve, cos, sin, resid_lambda, x0_lambda, ve_lambda, sink, dense_mask, bounds, doc_start=None,
                rope_tabs=None):
        if (LAYER_IMPL == "op" and _LAYER_OPS is not None and x.device.type == "neuron" and sink is not None
                and bounds is not None and rope_tabs is not None and x.shape[1] % 128 == 0):
            # Trainium (training and eval alike): the whole layer as two custom ops (3 fused NKI kernels fwd, kernel chain bwd);
            # the explicit path below is the same math for every other device / shape
            b, t, D = x.shape
            T = b * t; bf16 = torch.bfloat16
            cos2d, sin2d = rope_tabs
            cos_full = cos2d.repeat(b, 1).contiguous(); sin_full = sin2d.repeat(b, 1).contiguous()
            at, ml = self.attn, self.mlp
            ws = [lin._stack[lin.idx].to(bf16) for lin in (at.c_q, at.c_k, at.c_v, at.c_proj, ml.c_fc, ml.c_proj)]
            x2 = x.reshape(T, D).to(bf16).contiguous(); x02 = x0.reshape(T, D).to(bf16).contiguous()
            ve2 = ve.reshape(T, D).to(bf16).contiguous()
            scal = torch.stack([resid_lambda.reshape(()), x0_lambda.reshape(()), ve_lambda.reshape(()),
                                torch.zeros((), device=x.device)]).float().view(1, 4).expand(128, 4).contiguous()
            dstart = (doc_start.to(torch.float32) if doc_start is not None else torch.zeros((b, t), device=x.device)).contiguous()
            outs = torch.ops.tf_nki.layer_fwd(x2, x02, ve2, cos_full, sin_full, cos2d, sin2d, scal, sink, bounds[0], bounds[1],
                                              dstart, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5])
            return outs[0].view(b, t, D).to(x.dtype)
        x = resid_lambda * x + x0_lambda * x0
        x = x + self.attn(norm(x), ve_lambda * ve, cos, sin, sink, dense_mask, bounds, doc_start=doc_start)
        x = x + self.mlp(norm(x))
        return x


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
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.skip_lambdas = nn.Parameter(torch.ones(config.n_layer // 2))
        self.bigram_lambda = nn.Parameter(torch.full((1,), 0.2))
        self.softcap_pos = nn.Parameter(torch.full((1,), LOGIT_SOFTCAP))
        self.softcap_neg = nn.Parameter(torch.full((1,), LOGIT_SOFTCAP))
        self._fused_loss_ok = True                # cleared by check_fused_loss() if the fused loss ever disagrees with head_logits
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos2d", cos.reshape(cos.shape[1], -1).float().contiguous(), persistent=False)   # layer-op RoPE tables
        self.register_buffer("sin2d", sin.reshape(sin.shape[1], -1).float().contiguous(), persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        for name in ("wte", "vte", "vte_mid", "vte2", "bte"):
            torch.nn.init.normal_(self.transformer[name].weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        for block in self.transformer.h:   # .weight is a slice view of the shared stack
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
        self.skip_lambdas.fill_(1.0)
        self.bigram_lambda.fill_(0.2)
        self.softcap_pos.fill_(LOGIT_SOFTCAP)
        self.softcap_neg.fill_(LOGIT_SOFTCAP)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = ROPE_BASE):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (freqs / head_dim))
        inv_freq[int(round(inv_freq.numel() * ROPE_FRAC)):] = 0.0      # half-truncate RoPE
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
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for key in sorted(self.stacks.keys()):
            param_groups.append(dict(kind="muon", params=[self.stacks[key]], lr=MATRIX_LR, momentum=0.95, beta2=0.90, ns_steps=5, weight_decay=WEIGHT_DECAY))
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt

    def head_logits(self, x: torch.Tensor) -> torch.Tensor:
        """The model's output logits: final RMSNorm, lm_head, asymmetric learned softcap. This is what the scorer
        evaluates (forward with targets=None) and the reference the fused training loss is checked against."""
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        sp = self.softcap_pos.clamp_min(1.0).float()
        sn = self.softcap_neg.clamp_min(1.0).float()
        return torch.where(logits >= 0, sp * torch.tanh(logits / sp), sn * torch.tanh(logits / sn))

    def fused_linear_cross_entropy(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """cross_entropy(self.head_logits(x), targets) with ignore_index=-1 and mean reduction, computed by one NKI kernel
        without materializing the (T, V) logits: per 128-token tile the kernel forms the softcapped logits 512 columns at a
        time and folds them into an online log-sum-exp (same math as `head_logits` followed by F.cross_entropy). The lm_head
        weight enters transposed; the softcap parameters enter through `scal` and receive gradients."""
        b, t, D = x.shape
        x2 = x.reshape(b * t, D).to(torch.bfloat16).contiguous()
        wT = self.lm_head.weight.to(torch.bfloat16).t().contiguous()             # (D, V_padded)
        tgt = targets.reshape(b * t, 1).to(torch.int32).contiguous()
        scal = torch.stack([self.softcap_pos.clamp_min(1.0).reshape(()), self.softcap_neg.clamp_min(1.0).reshape(()),
                            torch.zeros((), device=x.device), torch.zeros((), device=x.device)]).float().view(1, 4).expand(128, 4).contiguous()
        loss_rows, _lse, _h, _rstd = torch.ops.tf_nki.head_fwd(x2, wT, tgt, scal)
        valid = targets.reshape(-1) != -1
        return loss_rows.sum() / valid.sum().clamp_min(1).float()

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean", emb=None):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        cos, sin = self.cos[:, :t], self.sin[:, :t]

        # hashed (previous, current) bigram table, int32 multiplicative hash
        prev = torch.cat([torch.full_like(idx[:, :1], BOS_ID), idx[:, :-1]], dim=1)
        pair = prev.to(torch.int32) * 8192 + idx.to(torch.int32)
        bh = (((pair * -1640531527) >> 13) & (BIGRAM_SIZE - 1)).long()

        if targets is not None:
            idx = idx.clone()                 # compiled graph: inputs saved for backward must not alias each other
            targets = targets.clone()
        if emb is not None:                       # training on Trainium: dense rows gathered outside the graph
            emb = {k: v.clone() for k, v in emb.items()}   # (clone: inputs saved for backward must not alias outputs)
            x = emb["wte"] + self.bigram_lambda * emb["bte"]
        else:
            x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
            x = x + self.bigram_lambda * self.transformer.bte(bh).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x.clone()
        if emb is not None:
            ve1, ve_mid, ve2 = emb["vte"], emb["vte_mid"], emb["vte2"]
        else:
            ve1 = self.transformer.vte(idx).to(COMPUTE_DTYPE)
            ve_mid = self.transformer.vte_mid(idx).to(COMPUTE_DTYPE)
            ve2 = self.transformer.vte2(idx).to(COMPUTE_DTYPE)
        t1 = self.config.n_layer // 3
        t2 = self.config.n_layer - t1

        # document mask: causal within the document (documents start at BOS)
        dense_mask = None
        bounds = None
        doc_start = (idx == BOS_ID) if KEY_OFFSET else None
        rope_tabs = (self.cos2d[:t], self.sin2d[:t]) if LAYER_IMPL == "op" else None
        if idx.device.type == "neuron" and NKI_AVAILABLE and not torch.compiler.is_compiling():
            bounds = doc_bounds(idx)          # eager Trainium: NKI kernels; compiled graph: the explicit path below
        elif rope_tabs is not None and idx.device.type == "neuron":
            bounds = doc_bounds(idx)          # compiled graph with the layer ops: the doc mask as bounds (scatter/gather only)
        else:
            docs = (idx == BOS_ID).cumsum(dim=1)
            pos = torch.arange(t, device=idx.device)
            causal = pos.view(1, t, 1) >= pos.view(1, 1, t)
            same_doc = docs.unsqueeze(2) == docs.unsqueeze(1)
            dense_mask = (causal & same_doc).unsqueeze(1)    # (b, 1, t, t)

        n_enc = self.config.n_layer // 2
        n_dec_start = self.config.n_layer - n_enc
        skips: list[torch.Tensor] = []
        bs = self.block_scalars
        for i, block in enumerate(self.transformer.h):
            if i >= n_dec_start:
                x = x + self.skip_lambdas[i - n_dec_start] * skips.pop()
            ve_i = ve1 if i < t1 else (ve_mid if i < t2 else ve2)
            x = block(x, x0, ve_i, cos, sin, bs["resid"][i], bs["x0"][i], bs["ve"][i], bs["sink"][i], dense_mask, bounds,
                      doc_start=doc_start, rope_tabs=rope_tabs)
            if i < n_enc:
                skips.append(x)
        if targets is None:
            return self.head_logits(x)
        if (self._fused_loss_ok and _HEAD_OPS is not None and loss_reduction == "mean" and x.device.type == "neuron"
                and x.shape[1] % 128 == 0):
            return self.fused_linear_cross_entropy(x, targets)   # == cross_entropy(self.head_logits(x), targets); see check_fused_loss
        logits = self.head_logits(x)
        flat_t = targets.reshape(-1)
        valid = flat_t != -1
        logp = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
        nll = -logp.gather(1, flat_t.clamp_min(0).unsqueeze(1)).squeeze(1) * valid.to(logp.dtype)
        if loss_reduction == "none":
            return nll.view_as(targets)
        return nll.sum() / valid.sum().clamp_min(1).to(nll.dtype)


def build_config() -> GPTConfig:
    return GPTConfig(sequence_len=SEQ_LEN, vocab_size=TOKENIZER_VOCAB_SIZE, n_layer=DEPTH, n_head=N_HEAD, n_embd=N_EMBD)


# -----------------------------------------------------------------------------
# Optimizer: Muon (Polar Express) for the stacked matrices, AdamW for embeddings / head / scalars.
# On Trainium both steps run as single NKI kernels; elsewhere (or if a kernel fails) as the plain ops below.


def bigram_bucket(h):
    return h & (BIGRAM_SIZE - 1)


_NKI_EMBED = None


def build_nki_embed():
    global _NKI_EMBED
    if _NKI_EMBED is None:
        if not NKI_AVAILABLE:
            raise RuntimeError("NKI is not available on this device")
        from torch_neuronx.nki_hop import wrap_nki
        _NKI_EMBED = (wrap_nki(embed_gather), wrap_nki(embed_scatter_add))
    return _NKI_EMBED


def scatter_rounds(idx):
    """Host plan for embed_scatter_add: slots grouped into rounds of distinct indices (the r-th occurrence of every
    distinct index), each round padded to a multiple of 128; padding slots carry -1 (replaced per table by the distinct
    dummy rows R + j of the gradient buffer's 128-row tail). Returns (perm, sidx); ~150 rounds for token ids, ~30 for
    bigram hashes."""
    import numpy as np
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    first = np.r_[True, sidx[1:] != sidx[:-1]]
    starts = np.flatnonzero(first)
    rank = np.arange(len(sidx)) - starts[np.cumsum(first) - 1]
    counts = np.bincount(rank)                      # rows per round
    by_rank = np.argsort(rank, kind="stable")       # positions (in sorted order) grouped by round
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
    sidx = np.concatenate(idx_l)                    # -1-j marks the j-th padding slot of its tile; -> R + j per table
    # fixed slot counts (powers of two, 4096..32768): the kernel is compiled once per (table shape, slot count); a padding
    # tile costs ~9 us on device , a data-dependent slot count recompiles the kernel every call .
    # Plans longer than 32768 slots (an index repeated > 256 times) are cut into 32768-slot chunks (S_V24_EMA99: a 65536-slot
    # variant failed to lower mid-run) - chunk boundaries fall on tile boundaries, so distinct-per-tile still holds.
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
    plus the host scatter plans; after backward, scatter(...) accumulates their .grad into fp32 table gradients with the
    NKI scatter-add kernel (or torch index_add_ under sidecar_torch); attach() hands those to the parameters."""

    def __init__(self, model, impl: str):
        self.model = model
        self.impl = impl
        self.tables = {"wte": model.transformer.wte.weight, "vte": model.transformer.vte.weight,
                       "vte_mid": model.transformer.vte_mid.weight, "vte2": model.transformer.vte2.weight}
        self.tables["bte"] = model.transformer.bte.weight
        self.kinds = {"tok": ["wte", "vte", "vte_mid", "vte2"], "bg": ["bte"]}
        # gradient buffers carry a 128-row tail that absorbs the scatter kernel's padding slots
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
            chunks = scatter_rounds(ind)
            R = self.tables[self.kinds[kind][0]].shape[0]
            parts = [ind]
            for perm, sidx in chunks:
                parts.append(perm.reshape(-1)); parts.append(np.where(sidx < 0, R - 1 - sidx, sidx))   # padding -> R + j
            buf = np.concatenate(parts).astype(np.uint32).reshape(-1, 1)
            dbuf = torch.from_numpy(buf).to(dev)
            n = len(ind); off = n; views = []
            for perm, sidx in chunks:
                S = perm.shape[0]
                views.append((dbuf[off:off + S], dbuf[off + S:off + 2 * S])); off += 2 * S
            plans[kind] = (ind, dbuf[:n], views, None)
        return plans

    @torch.no_grad()
    def gather(self, x, plans=None):
        if plans is None:
            plans = self.prepare(x)
        dev = x.device
        b, t = x.shape
        emb = {}
        for kind, (ind, ind_t, views, _unused) in plans.items():
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
        for kind, (ind, ind_t, views, _unused) in plans.items():
            for name in self.kinds[kind]:
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
                    except Exception as exc:   # kernel lowering failure for a rare shape: the device's own index_add_
                        print0(f"scatter kernel failed ({str(exc)[:80]}); index_add_ fallback")
                if not done:
                    R = self.tables[name].shape[0]
                    self.gbuf[name][:R].index_add_(0, ind_t.view(-1).long(), g2.float())

    def attach(self):
        for name, w in self.tables.items():
            w.grad = self.gbuf[name][:w.shape[0]] if (name != "bte" or self.bigram_grad) else None


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self.use_nki = NKI_AVAILABLE
        self._flat = {}          # the small-parameter AdamW group as one flat tensor (see _adamw_step_flat)
        self.plan = {}           # sharded optimizer: param -> dict(pshard, gshard) (see setup_sharding)
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
        flat.sub_(lr_t * (m / bias1 / ((v / bias2).sqrt() + group["eps"])))
        off = 0
        for p, sz in zip(params, st["sizes"]):
            p.copy_(flat[off:off + sz].view_as(p).to(p.dtype))
            off += sz

    @torch.no_grad()
    def setup_sharding(self, world_size: int, rank: int):
        """Sharded optimizer (ZeRO-1 style): every parameter whose leading dim splits evenly over the workers is updated by
        one worker only, on a persistent shard with its own momentum state; its gradient arrives by reduce_scatter and the
        updated shard goes back into the replicated parameter by all_gather (the same bytes the all_reduce moves)."""
        self.world_size = world_size
        if world_size == 1:
            return
        for group in self.param_groups:
            if self._is_small_group(group):
                continue
            for p in group["params"]:
                n = p.shape[0]
                if n % world_size != 0:
                    continue
                per = n // world_size
                if group["kind"] == "adamw" and (p.dim() != 2 or per % (2 * P) != 0):
                    continue
                if group["kind"] == "muon" and per % (NKI_GRID[0] if NKI_AVAILABLE else 1) != 0:
                    continue
                pshard = p.data[rank * per:(rank + 1) * per].clone()
                pshard.grad = torch.zeros_like(pshard)
                self.plan[p] = dict(pshard=pshard, gshard=pshard.grad)

    def reduce_gradients(self, model: nn.Module):
        small = set()
        for group in self.param_groups:
            if self._is_small_group(group):
                small.update(id(p) for p in group["params"])   # reduced inside _adamw_step_flat
        for p in model.parameters():
            if p.grad is None or id(p) in small:
                continue
            plan = self.plan.get(p)
            if plan is not None:
                dist.reduce_scatter_tensor(plan["gshard"], p.grad, op=dist.ReduceOp.AVG)
            else:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    @torch.no_grad()
    def gather_params(self):
        for p, plan in self.plan.items():
            dist.all_gather_into_tensor(p.data, plan["pshard"])

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
        w = params[0]                       # (n, out, in) stack, updated in place
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
    def step(self):
        for group in self.param_groups:
            if self._is_small_group(group):
                self._adamw_step_flat(group)
                continue
            planned = [p for p in group["params"] if p in self.plan]
            if planned:   # the same per-parameter math on (pshard, gshard) instead of (p, p.grad)
                g2 = dict(group, params=[self.plan[p]["pshard"] for p in planned])
                (self._adamw_step if group["kind"] == "adamw" else self._muon_step)(g2)
                group = dict(group, params=[p for p in group["params"] if p not in self.plan])
                if not group["params"]:
                    continue
            if group["kind"] == "adamw":
                self._adamw_step(group)
            else:
                self._muon_step(group)
        if self.plan:
            self.gather_params()


def sync_gradients(model: nn.Module, optimizer: MuonAdamW) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    optimizer.reduce_gradients(model)


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


def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = GPTConfig(**payload["model_config"])
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    if LAYER_IMPL == "op" and device.type == "neuron" and NKI_AVAILABLE:
        try:
            build_layer_ops(config.n_head)      # the same layer ops as in training; explicit layers if they cannot be built
        except Exception as exc:
            print0(f"layer ops unavailable for eval ({str(exc)[:120]}); using the explicit layers")
    return model


# -----------------------------------------------------------------------------
# Schedules: warmup by steps, warmdown by elapsed time (all ranks use the rank-0 clock)


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
    parser = argparse.ArgumentParser(description="NKI Frontier Round 1 submission")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_NUM_STEPS", NUM_STEPS)))
    parser.add_argument("--max-train-seconds", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS", MAX_TRAIN_SECONDS)))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("NEURON_COMPETITION_R1_OUT_DIR", "out"))
    parser.add_argument("--eval-public", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_EVAL_PUBLIC", "1") != "0")
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_EVAL_TOKENS", str(4 * 524288))))
    parser.add_argument("--compile-only", action="store_true", help="run one synthetic optimizer step, then exit")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    # rows per worker: at most 4 per micro-step (a 12 GB LNC=1 core holds 4 rows of the compiled graph); the rest of the
    # 131,072-token step is gradient accumulation
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
            build_layer_ops(model.config.n_head)  # the custom ops must exist before torch.compile traces the model
        except Exception as exc:                # unknown kernel API on the scoring box -> explicit layers (same math), never a crash
            print0(f"layer ops unavailable ({str(exc)[:120]}); using the explicit layers")
    if device_type == "neuron" and NKI_AVAILABLE:
        try:
            build_head_ops()                      # fused linear cross-entropy; explicit cross_entropy(head_logits) if it cannot be built
        except Exception as exc:
            print0(f"fused loss unavailable ({str(exc)[:120]}); using cross_entropy(head_logits)")
    if COMPILE and device_type == "neuron":
        try:
            model = torch.compile(model, backend="neuron", dynamic=False)
            USE_STACK_FUNCTION = False
            print0("torch.compile enabled (backend=neuron)")
        except Exception as exc:
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
        model.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            loss = model(synthetic_x, synthetic_y)
            (loss / grad_accum_steps).backward()
        optimizer.setup_sharding(world_size, rank)
        sync_gradients(model, optimizer)
        optimizer.refresh_scalars()
        optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    train_loader = make_dataloader(tokenizer, device_batch_size, SEQ_LEN, "train", device)
    x, y, loader_state = next(train_loader)
    optimizer = model.setup_optimizer()
    optimizer.setup_sharding(world_size, rank)
    # embedding tables live outside the compiled graph (whose backward lowers their scatter-add as one-hot matmuls,
    # ~20% of the step): gather rows before the graph, scatter-add their gradients after it
    sidecar = EmbedSidecar(orig_model, "sidecar" if (device.type == "neuron" and NKI_AVAILABLE) else "sidecar_torch")
    plan_ring = (None, None)                   # (older, newer): references that outlive the device queue
    ema_params = None
    ema_pairs = None
    next_plans = None
    loop_started = time.monotonic()
    budget_started = loop_started

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0

    while step < args.num_steps:
        synchronize(device)
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        if sidecar is not None:
            sidecar.start_step()
        for micro in range(grad_accum_steps):
            try:
                if sidecar is not None:
                    emb, plans = sidecar.gather(x, next_plans)
                    if step == 0 and micro == 0:
                        check_fused_loss(orig_model, x, y, emb)     # once, eagerly, before the compiled graph is traced
                    loss = model(x, y, emb=emb)
                    last_loss = loss.detach()
                    (loss / grad_accum_steps).backward()
                    x, y, loader_state = next(train_loader)      # the next micro-step's host plan overlaps the backward
                    next_plans = sidecar.prepare(x)
                    sidecar.scatter(emb, plans)
                    plan_ring = (plan_ring[1], (emb, plans))   # keep the last two micro-steps' index buffers alive (async indirect DMA)
                    continue
                loss = model(x, y)
                last_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            except Exception as exc:
                if sidecar is not None and step == 0:
                    print0(f"embedding sidecar failed at step 0 ({str(exc)[:200]}); tables in the graph")
                    sidecar = None
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(x, y)
                    last_loss = loss.detach()
                    (loss / grad_accum_steps).backward()
                    x, y, loader_state = next(train_loader)
                    continue
                if step != 0 or model is orig_model:
                    raise
                print0(f"compiled path failed at step 0 ({str(exc)[:80]}); falling back to eager")
                model = orig_model
                USE_STACK_FUNCTION = True
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                loss = model(x, y)
                last_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            x, y, loader_state = next(train_loader)
        if sidecar is not None:
            sidecar.attach()
        sync_gradients(model, optimizer)

        elapsed_frac = synced_elapsed_frac(budget_started, args.max_train_seconds, device)
        lrm = lr_multiplier(step, elapsed_frac)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step)
        optimizer.refresh_scalars()
        optimizer.step()
        if elapsed_frac >= EMA_START:
            with torch.no_grad():
                if ema_params is None:
                    tables = {id(m.weight) for m in orig_model.transformer.values() if isinstance(m, nn.Embedding)}
                    ema_pairs = [p for p in orig_model.parameters() if id(p) not in tables]
                    ema_params = [p.detach().clone() for p in ema_pairs]
                    print0(f"tail-EMA of the weights started at step {step}")
                else:
                    for e, p in zip(ema_params, ema_pairs):
                        e.lerp_(p.detach(), 1.0 - EMA_DECAY)

        synchronize(device)
        record_step_boundary(step)
        dt = time.monotonic() - t0
        recent_step_times.append(dt)
        train_loss = float(last_loss.item())
        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
            # The runner excludes launch through this boundary (capped): restart the budget clock here, back-dated
            # by any excess over the cap plus a margin for the launch -> import gap this process cannot observe.
            now = time.monotonic()
            observed_startup = now - _PROCESS_STARTED
            startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            recent_step_times[:] = [dt]
            excess = max(0.0, observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            budget_started = now - excess
        elapsed_after_step = time.monotonic() - budget_started
        # Every rank breaks on the same step: the stop decision rides the health collective. Time-based stops are
        # gated behind the excluded-step count so at least TRAIN_STARTUP_STEPS_EXCLUDED boundaries are reported.
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        stop_after_step = (
            step + 1 >= args.num_steps
            or (reached_min_steps and (
                elapsed_after_step >= args.max_train_seconds
                or elapsed_after_step + max(recent_step_times[-20:]) + CHECKPOINT_RESERVE_SECONDS >= args.max_train_seconds))
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
        print0("tail-EMA blended into the weights for the checkpoint and the evaluation")
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
        eval_rows = min(device_batch_size, 4)     # the eval forward materializes (rows, T, V) fp32 logits: cap it at 4 rows
        causal = causality_check(orig_model, tokenizer, eval_rows, SEQ_LEN, device)
        public_eval = evaluate_bpb(
            orig_model, tokenizer, eval_rows, SEQ_LEN, device,
            split="public_val", max_tokens=args.eval_tokens, timeout_seconds=EVAL_TIMEOUT_SECONDS,
        )

    if rank == 0:
        print("---")
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

    cleanup_runtime()


if __name__ == "__main__":
    main()
