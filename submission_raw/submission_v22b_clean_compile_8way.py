"""NKI Frontier Round 1 submission (2026-09-07).

GPT-style language model trained from scratch on the provided ClimbMix shards with the fixed tokenizer:
10 layers x 512 wide, 4 heads, RMSNorm, RoPE, QK-norm, ReLU^2 MLP, learned logit softcap, U-net skips,
value-embedding tables, a hashed bigram table, document-masked attention with a learned per-head sink, and a
Muon (Polar Express) + AdamW optimizer on a fixed 30-minute wall-clock budget.

On Trainium the attention runs on the NKI flash-attention kernels shipped with the environment (with the
document mask passed as per-query KV bounds), and the optimizer runs on two small NKI kernels written for this
submission (a fused Muon step and a fused AdamW step). Off Trainium (or if a kernel is unavailable) every one of
them falls back to the equivalent plain PyTorch code, so the math is identical on any device.

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
DEPTH = 10
N_EMBD = 512
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD
MLP_RATIO = 4
TOTAL_BATCH_SIZE = 131072       # tokens per optimizer step (four 4-row micro-steps per worker on the 8-way launch)
_BATCH_LR = (TOTAL_BATCH_SIZE / 32768) ** 0.5   # sqrt-batch scaling of every learning rate relative to the 32k tuning
NUM_STEPS = 100000              # never binds; the time budget does
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 5.0
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

ROPE_BASE = 200_000
LOGIT_SOFTCAP = 11.0
BOS_ID = 8188                   # the fixed tokenizer's <|reserved_0|> (document separator)
SINK_INIT = math.log(4.0)       # softmax-denominator sink logit per (layer, head)
BIGRAM_SIZE = 2 ** 17

EMBEDDING_LR = 0.30 * 0.71 * 2.0 * _BATCH_LR
UNEMBEDDING_LR = 0.006 * 0.71 * _BATCH_LR
MATRIX_LR = 0.025 * 0.71 * _BATCH_LR
SCALAR_LR = 0.20 * 0.71 * _BATCH_LR
WEIGHT_DECAY = 0.10
WARMUP_STEPS = 100
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
USE_STACK_FUNCTION = True       # eager: accumulate stack gradients through _StackSliceLinear; compiled: plain views

# -----------------------------------------------------------------------------
# NKI kernels (Trainium only). P = SBUF partitions; FMAX = one PSUM bank of fp32 per partition.

P = 128
FMAX = 512

if NKI_AVAILABLE:
    NKI_GRID = (int(get_logical_neuron_cores()),)

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

    def forward(self, x, ve, cos, sin, sink, dense_mask=None, bounds=None):
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
        v = (self.c_v(x) + ve).view(b, t, self.n_head, self.head_dim)
        q = norm(apply_rotary_emb(q, cos, sin)).transpose(1, 2)
        k = norm(apply_rotary_emb(k, cos, sin)).transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
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


class Block(nn.Module):
    def __init__(self, config: GPTConfig, registry: StackRegistry):
        super().__init__()
        self.attn = CausalSelfAttention(config, registry)
        self.mlp = MLP(config, registry)

    def forward(self, x, x0, ve, cos, sin, resid_lambda, x0_lambda, ve_lambda, sink, dense_mask, bounds):
        x = resid_lambda * x + x0_lambda * x0
        x = x + self.attn(norm(x), ve_lambda * ve, cos, sin, sink, dense_mask, bounds)
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
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean"):
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
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = x + self.bigram_lambda * self.transformer.bte(bh).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x.clone()
        ve1 = self.transformer.vte(idx).to(COMPUTE_DTYPE)
        ve_mid = self.transformer.vte_mid(idx).to(COMPUTE_DTYPE)
        ve2 = self.transformer.vte2(idx).to(COMPUTE_DTYPE)
        t1 = self.config.n_layer // 3
        t2 = self.config.n_layer - t1

        # document mask: causal within the document (documents start at BOS)
        dense_mask = None
        bounds = None
        if idx.device.type == "neuron" and NKI_AVAILABLE and not torch.compiler.is_compiling():
            bounds = doc_bounds(idx)          # eager Trainium: NKI kernels; compiled graph: the explicit path below
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
            x = block(x, x0, ve_i, cos, sin, bs["resid"][i], bs["x0"][i], bs["ve"][i], bs["sink"][i], dense_mask, bounds)
            if i < n_enc:
                skips.append(x)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        sp = self.softcap_pos.clamp_min(1.0).float()
        sn = self.softcap_neg.clamp_min(1.0).float()
        logits = torch.where(logits >= 0, sp * torch.tanh(logits / sp), sn * torch.tanh(logits / sn))
        if targets is None:
            return logits
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
                    print0(f"NKI AdamW kernel unavailable ({type(exc).__name__}); using the PyTorch step")
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
                print0(f"NKI Muon kernel unavailable ({type(exc).__name__}); using the PyTorch step")

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
    device_batch_size = max(1, min(4, (TOTAL_BATCH_SIZE // SEQ_LEN) // world_size))

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
    if COMPILE and device_type == "neuron":
        try:
            model = torch.compile(model, backend="neuron", dynamic=False)
            USE_STACK_FUNCTION = False
            print0("torch.compile enabled (backend=neuron)")
        except Exception as exc:
            print0(f"torch.compile unavailable ({type(exc).__name__}); running eager")
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
        for _ in range(grad_accum_steps):
            try:
                loss = model(x, y)
                last_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            except Exception as exc:
                if step != 0 or model is orig_model:
                    raise
                print0(f"compiled path failed at step 0 ({type(exc).__name__}); falling back to eager")
                model = orig_model
                USE_STACK_FUNCTION = True
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                loss = model(x, y)
                last_loss = loss.detach()
                (loss / grad_accum_steps).backward()
            x, y, loader_state = next(train_loader)
        sync_gradients(model, optimizer)

        elapsed_frac = synced_elapsed_frac(budget_started, args.max_train_seconds, device)
        lrm = lr_multiplier(step, elapsed_frac)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step)
        optimizer.refresh_scalars()
        optimizer.step()

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
        causal = causality_check(orig_model, tokenizer, device_batch_size, SEQ_LEN, device)
        public_eval = evaluate_bpb(
            orig_model, tokenizer, device_batch_size, SEQ_LEN, device,
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
