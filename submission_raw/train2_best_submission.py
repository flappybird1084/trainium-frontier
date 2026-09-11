"""
train2_best_submission.py — v2 (2026-09-02): submission-ready FIXED-TIME champion.

v1 post-mortem: the fixed-step 215M-param stack scored worse than baseline on the
portal (too few steps in the 30-min budget). v2 is the config that won the overnight
fixed-wall-time campaign on a 3090 (val_bpb 1.1427 per 720 s, T10): small and fast —
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

Launch (same shape as baseline):
  NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train2_best_submission.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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

try:  # CUDA dev boxes only; Neuron/CPU use the SDPA dense-mask path below
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _HAS_FLEX = True
except Exception:
    _HAS_FLEX = False


# -----------------------------------------------------------------------------
# Hyperparameters (champion stack)

SEQ_LEN = 1024
DEPTH = 10
N_EMBD = 256                    # fixed-time optimum (T5/T11: beats 384/512/640/768 per wall-clock)
N_HEAD = 2
HEAD_DIM = N_EMBD // N_HEAD
MLP_RATIO = 4

PACK = 4                        # organizer-sanctioned packing: fetch 4x rows, reshape (less truncation)
DEVICE_BATCH_SIZE = int(os.environ.get("NEURON_COMPETITION_R1_DEVICE_BATCH", "4"))
                                # post-reshape rows per micro-step (raw loader rows = this // PACK);
                                # 4 fits TOTAL_BATCH at 8 Trn2 workers; dev boxes may override via env
TOTAL_BATCH_SIZE = 32768        # fixed-time optimum with sqrt-scaled LRs (T6b)
NUM_STEPS = 100000              # never binds; the time budget does
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS - 60

ROPE_BASE = 200_000
LOGIT_SOFTCAP = 11.0
BOS_ID = 8188                   # fixed tokenizer's <|reserved_0|>
NUM_SINKS = 4                   # attention registers per layer
BIGRAM_SIZE = 2 ** 17

EMBEDDING_LR = 0.30 * 0.71     # sqrt-batch LR compensation for TOTAL_BATCH 32768 (T6b)
UNEMBEDDING_LR = 0.006 * 0.71
MATRIX_LR = 0.025 * 0.71
SCALAR_LR = 0.20 * 0.71
WEIGHT_DECAY = 0.10
WARMUP_STEPS = 20
WARMDOWN_TIME_FRAC = 0.60       # last 60% of the time budget decays linearly...
FINAL_LR_FRAC = 0.02            # ...to 2% of peak

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


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (bf16 input, f32 weight)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, ve, cos, sin, block_mask=None, dense_mask=None, k_sink=None, v_sink=None):
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
        v = (self.c_v(x) + ve).view(b, t, self.n_head, self.head_dim)
        q = norm(apply_rotary_emb(q, cos, sin))
        k = norm(apply_rotary_emb(k, cos, sin))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # registers: NUM_SINKS always-visible kv slots (learned k, learned v, zero-init)
        sk = k_sink.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1).to(k.dtype)
        sv = v_sink.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1).to(v.dtype)
        k = torch.cat([sk, k], dim=2)
        v = torch.cat([sv, v], dim=2)
        if block_mask is not None:
            y = flex_attention(q, k, v, block_mask=block_mask, kernel_options=FLEX_KERNEL_OPTS)
        else:
            # Portable path (Neuron/CPU): explicit attention with only baseline-proven ops.
            scores = (q.float() @ k.float().transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            scores = scores.masked_fill(~dense_mask, float("-inf"))
            y = (F.softmax(scores, dim=-1) @ v.float()).to(v.dtype)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, MLP_RATIO * config.n_embd, bias=False)
        self.c_proj = Linear(MLP_RATIO * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(relu_squared(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.resid_lambda = nn.Parameter(torch.ones(1))
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        self.ve_lambda = nn.Parameter(torch.full((1,), 0.5))
        self.k_sink = nn.Parameter(torch.zeros(NUM_SINKS, config.n_head, config.n_embd // config.n_head))
        self.v_sink = nn.Parameter(torch.zeros(NUM_SINKS, config.n_head, config.n_embd // config.n_head))

    def forward(self, x, x0, ve, cos, sin, block_mask=None, dense_mask=None):
        x = self.resid_lambda * x + self.x0_lambda * x0
        x = x + self.attn(norm(x), self.ve_lambda * ve, cos, sin, block_mask, dense_mask,
                          self.k_sink, self.v_sink)
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
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
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
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.vte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.vte_mid.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.vte2.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.bte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            block.resid_lambda.fill_(1.0)
            block.x0_lambda.fill_(0.05)
            block.ve_lambda.fill_(0.5)
            block.k_sink.zero_()
            block.v_sink.zero_()
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

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def setup_optimizer(self):
        scalar_params = [self.skip_lambdas, self.bigram_lambda, self.softcap_pos, self.softcap_neg]
        matrix_params = []
        for block in self.transformer.h:
            scalar_params.extend([block.resid_lambda, block.x0_lambda, block.ve_lambda,
                                  block.k_sink, block.v_sink])
            matrix_params.extend(block.attn.parameters())
            matrix_params.extend(block.mlp.parameters())
        embedding_params = (list(self.transformer.wte.parameters())
                            + list(self.transformer.vte.parameters())
                            + list(self.transformer.vte_mid.parameters())
                            + list(self.transformer.vte2.parameters())
                            + list(self.transformer.bte.parameters()))
        lm_head_params = list(self.lm_head.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=params, lr=MATRIX_LR, momentum=0.95, beta2=0.90, ns_steps=5, weight_decay=WEIGHT_DECAY))
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean"):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        cos, sin = self.cos[:, :t], self.sin[:, :t]

        # hashed bigram table: int32 multiplicative hash (Neuron-friendly; wraps mod 2^32)
        prev = torch.cat([torch.full_like(idx[:, :1], BOS_ID), idx[:, :-1]], dim=1)
        pair = prev.to(torch.int32) * 8192 + idx.to(torch.int32)
        bh = (((pair * -1640531527) >> 13) & (BIGRAM_SIZE - 1)).long()

        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = x + self.bigram_lambda * self.transformer.bte(bh).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        ve1 = self.transformer.vte(idx).to(COMPUTE_DTYPE)
        ve_mid = self.transformer.vte_mid(idx).to(COMPUTE_DTYPE)
        ve2 = self.transformer.vte2(idx).to(COMPUTE_DTYPE)
        t1 = self.config.n_layer // 3
        t2 = self.config.n_layer - t1

        # doc-masked attention + registers; same math on every backend
        docs = (idx == BOS_ID).cumsum(dim=1)
        block_mask = None
        dense_mask = None
        if _HAS_FLEX and idx.is_cuda:
            docs_kv = torch.cat([docs[:, :1].expand(-1, NUM_SINKS), docs], dim=1)

            def doc_causal_sink(b, h, q_idx, kv_idx):
                real = (q_idx + NUM_SINKS >= kv_idx) & (docs[b, q_idx] == docs_kv[b, kv_idx])
                return (kv_idx < NUM_SINKS) | real

            block_mask = create_block_mask(doc_causal_sink, idx.size(0), None, t, t + NUM_SINKS, device=idx.device)
        else:
            pos = torch.arange(t, device=idx.device)
            causal = pos.view(1, t, 1) >= pos.view(1, 1, t)
            same_doc = docs.unsqueeze(2) == docs.unsqueeze(1)
            real = causal & same_doc                              # (b, t, t)
            sinks = torch.ones(idx.size(0), t, NUM_SINKS, dtype=torch.bool, device=idx.device)
            dense_mask = torch.cat([sinks, real], dim=2).unsqueeze(1)  # (b, 1, t, t+S)

        n_enc = self.config.n_layer // 2
        n_dec_start = self.config.n_layer - n_enc
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.transformer.h):
            if i >= n_dec_start:
                x = x + self.skip_lambdas[i - n_dec_start] * skips.pop()
            ve_i = ve1 if i < t1 else (ve_mid if i < t2 else ve2)
            x = block(x, x0, ve_i, cos, sin, block_mask, dense_mask)
            if i < n_enc:
                skips.append(x)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        sp = self.softcap_pos.clamp_min(1.0).float()
        sn = self.softcap_neg.clamp_min(1.0).float()
        logits = torch.where(logits >= 0, sp * torch.tanh(logits / sp), sn * torch.tanh(logits / sn))
        if targets is None:
            return logits
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


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def _adamw_step(self, group):
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            p.mul_(1 - group["lr"] * group["weight_decay"])
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group["eps"])
            p.add_(update, alpha=-group["lr"])

    @torch.no_grad()
    def _muon_step(self, group):
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        shape = p0.shape
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "second_momentum_buffer" not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        if "stacked_grads" not in state:
            state["stacked_grads"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "stacked_params" not in state:
            state["stacked_params"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)

        g = state["stacked_grads"]
        w = state["stacked_params"]
        for i, p in enumerate(params):
            if p.grad is None:
                g[i].zero_()
            else:
                g[i].copy_(p.grad)
            w[i].copy_(p)
        momentum_buffer = state["momentum_buffer"]
        second = state["second_momentum_buffer"]

        momentum_buffer.lerp_(g, 1 - group["momentum"])
        g = g.lerp_(momentum_buffer, group["momentum"])
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

        lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        wd = group["weight_decay"]
        mask = (g * w) >= 0
        w.sub_(lr * g + lr * wd * w * mask)
        torch._foreach_copy_(params, list(w.unbind(0)))

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
    args = parser.parse_args()

    process_started = time.monotonic()
    torch.set_float32_matmul_precision("high")  # CUDA TF32; harmless elsewhere. Without it
    # the fp32 pieces of the graph run ~20x slower on GPU dev boxes (found 2026-09-02).
    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)

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
    if args.compile:
        compile_backend = "neuron" if device_type == "neuron" else "inductor"
        model = torch.compile(model, backend=compile_backend, dynamic=False)
        print0(f"torch.compile enabled (backend={compile_backend})")

    world_tokens_per_micro = DEVICE_BATCH_SIZE * SEQ_LEN * world_size
    if TOTAL_BATCH_SIZE % world_tokens_per_micro != 0:
        raise ValueError(f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={world_tokens_per_micro}")
    grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_micro
    print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps}")

    # organizer-sanctioned packing: fetch PACK-times-long rows, reshape to SEQ_LEN chunks
    _raw_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE // PACK, SEQ_LEN * PACK, "train", device)

    def _packed():
        for _x, _y, _st in _raw_loader:
            yield _x.reshape(-1, SEQ_LEN), _y.reshape(-1, SEQ_LEN), _st

    train_loader = _packed()
    x, y, loader_state = next(train_loader)
    optimizer = model.setup_optimizer()

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0

    # Compile/warm-up BEFORE starting the budget clock. Officially sanctioned:
    # "compilation is excluded from the 30-minute budget; run warm-up steps to
    # compile, then start the clock" (organizer ruling). No optimizer updates here.
    for _ in range(3):
        wloss = model(x, y)
        (wloss / grad_accum_steps).backward()
    optimizer.zero_grad(set_to_none=True)
    x, y, loader_state = next(train_loader)
    synchronize(device)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    warmup_seconds = time.monotonic() - process_started
    print0(f"compile_warmup_seconds={warmup_seconds:.1f}; budget clock starts now")
    budget_started = time.monotonic()

    while step < args.num_steps:
        elapsed = time.monotonic() - budget_started
        if elapsed >= args.max_train_seconds:
            break

        synchronize(device)
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        for _ in range(grad_accum_steps):
            loss = model(x, y)
            last_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, loader_state = next(train_loader)

        sync_gradients(model)

        elapsed_frac = synced_elapsed_frac(budget_started, args.max_train_seconds, device)
        lrm = lr_multiplier(step, elapsed_frac)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step)
                group["weight_decay"] = WEIGHT_DECAY
        optimizer.step()

        synchronize(device)
        dt = time.monotonic() - t0
        train_loss = float(last_loss.item())
        bad = torch.tensor(1 if (math.isnan(train_loss) or train_loss > 100) else 0, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if int(bad.item()) != 0:
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
        print(f"training_seconds:  {budget_elapsed_before_save:.1f}")
        print(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}")
        print(f"over_budget:       {budget_elapsed_after_save > TRAIN_TIME_BUDGET_SECONDS}")
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
