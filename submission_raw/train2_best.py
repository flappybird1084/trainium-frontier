"""
train2_best.py — FROZEN best config under the FIXED-TIME protocol (720 s train, steps float).

Champion as of 2026-09-02 ~17:30 PT: val_bpb 1.118476 @ 6517 steps / 720 s on idle RTX 4080
SUPER (L2). Do NOT compare to 3090-era T-numbers (T10 1.1427): faster card, more steps.
Stack = T10 + narrow QK (QK_DIM 64 vs v head_dim 128; -0.0075, +27% steps at equal quality).
Trigram table won alone (L1, -0.0042) but does NOT stack with narrow QK (L4r3 combo: wash,
-0.0002) — champion stays trigram-free. History: train2_results.tsv,
findings/train2-experiments.md (L-series section).
"""
"""
train2.py — loss-per-step experimentation script (development only, NOT a submission).

Goal: the lowest val_bpb in a FIXED number of optimizer steps, with one run
fitting in ~15 minutes wall clock on a single GPU. Derived from train.py.

The budget here is steps, not seconds, so throughput tricks that do not change
the math (torch.compile, TF32, larger device batch) are free — but remember the
real competition trades loss-per-step against seconds-per-step on Trn2, so any
win found here must eventually be re-timed on the chip.

Differences vs the train.py baseline (each one is a loss-per-step lever):
  - LR schedule: warmup + linear warmdown to FINAL_LR_FRAC (baseline: flat
    after warmup). This is the largest known win over the baseline schedule.
  - ROPE_BASE = 2e6 (baseline 1e5) and LOGIT_SOFTCAP = 11 (baseline 15),
    taken from the tuned autoresearch config in this repo.
  - Single process only: no torch.distributed, no NKI, no grad-sync — keeps
    the file short so experiments are one-line edits to the knobs below.

Run:    .venv/bin/python train2.py            (defaults sized for ~15 min)
Score:  .venv/bin/python prepare.py eval-public --train-py train2.py \
            --checkpoint out_train2/final.pt
"""


import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from prepare import (
    TOKENIZER_VOCAB_SIZE,
    causality_check,
    ensure_tokenizer,
    evaluate_bpb,
    make_dataloader,
)

try:  # research-only dependency (pod); a submission needs a self-contained port
    from fla.layers import GatedDeltaNet, KimiDeltaAttention
    _HAS_FLA = True
except Exception:
    _HAS_FLA = False


# -----------------------------------------------------------------------------
# Knobs. Edit these between runs; keep TOTAL_BATCH_SIZE fixed across experiments
# so "loss in N steps" comparisons stay apples-to-apples.

SEQ_LEN = 1024
DEPTH = 10
ASPECT_RATIO = 25               # T5: n_embd 256, 2 heads — hunting the turn
MLP_RATIO = 4                   # frontier: 768->1.2590, 640->1.2143, 512->1.1919, 384->1.1767@828
HEAD_DIM = 128
QK_DIM = 64                     # L2: narrow QK — q/k head dim halved, v stays HEAD_DIM (flex E != Ev)

DEVICE_BATCH_SIZE = 32          # per-micro-step rows (fits at width 256)
TOTAL_BATCH_SIZE = 32768        # T6: half batch -> ~2x optimizer steps under fixed time
NUM_STEPS = 100000              # never binds: FIXED-TIME protocol (2026-09-02 pivot)
MAX_TRAIN_SECONDS = 720         # THE experiment budget: train exactly this long, steps float.
                                # Mirrors the official fixed-time scoring (fixed-step wins
                                # did not transfer: submission scored worse than baseline).

ROPE_BASE = 200_000             # E17b probe; 500k -> 1.2906 (E17), 2M -> 1.2945 (E14)
LOGIT_SOFTCAP = 11.0            # baseline: 15.0
BOS_ID = 8188                   # fixed tokenizer's <|reserved_0|> BOS token
DOC_MASK = True                 # attention may not cross packed-document boundaries
# sm86 (RTX 3090) has ~99KB smem/SM; flex_attention's default Triton blocks need ~128KB.
FLEX_KERNEL_OPTS = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32, "num_stages": 1}
BIGRAM_SIZE = 2 ** 17           # hashed (prev, cur) token-pair table entries; 2^18 measured worse (E13)
NUM_SINKS = 4                   # E23b: sink slots -> registers (learnable zero-init values)
DELTANET_LAYERS = ()            # T9: all-attention variant — the fla-free SUBMITTABLE config

EMBEDDING_LR = 0.30 * 0.71     # T6b: sqrt-batch LR compensation for TOTAL_BATCH 32768
UNEMBEDDING_LR = 0.006 * 0.71
MATRIX_LR = 0.025 * 0.71
SCALAR_LR = 0.20 * 0.71
WEIGHT_DECAY = 0.10
WARMUP_STEPS = 20
WARMDOWN_FRAC = 0.60            # last 60% of steps decay linearly...
FINAL_LR_FRAC = 0.02            # ...down to 2% of peak LR


# -----------------------------------------------------------------------------
# Device

def detect_device_type(requested: str = "") -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


COMPUTE_DTYPE: torch.dtype = torch.bfloat16


# -----------------------------------------------------------------------------
# Model (identical skeleton to train.py so findings transfer)

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
        self.c_q = Linear(config.n_embd, config.n_head * QK_DIM, bias=False)
        self.c_k = Linear(config.n_embd, config.n_head * QK_DIM, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, ve: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None, k_sink=None, v_sink=None) -> torch.Tensor:
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, QK_DIM)
        k = self.c_k(x).view(b, t, self.n_head, QK_DIM)
        v = (self.c_v(x) + ve).view(b, t, self.n_head, self.head_dim)
        q = norm(apply_rotary_emb(q, cos, sin))
        k = norm(apply_rotary_emb(k, cos, sin))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if block_mask is not None:
            sk = k_sink.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1).to(k.dtype)
            sv = v_sink.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1).to(v.dtype)
            k = torch.cat([sk, k], dim=2)
            v = torch.cat([sv, v], dim=2)
            y = flex_attention(q, k, v, block_mask=block_mask, kernel_options=FLEX_KERNEL_OPTS)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


@torch.compiler.disable
def doc_cu_seqlens(idx: torch.Tensor) -> torch.Tensor:
    """Flat cut points at every doc start (BOS) AND every row start.
    Row starts are mandatory: they guarantee logits stay independent of other
    rows in the batch (eval rows are not BOS-aligned)."""
    b, t = idx.shape
    bos_pos = (idx == BOS_ID).view(-1).nonzero().flatten()
    row_starts = torch.arange(0, b * t + 1, t, device=idx.device)
    cu = torch.unique(torch.cat([bos_pos, row_starts]))
    # drop non-row cut points that would create segments shorter than 8 tokens
    # (tiny varlen segments crash fla's chunked kernel); row starts always kept
    is_row = (cu % t == 0)
    gap_prev = torch.diff(cu, prepend=cu.new_full((1,), -100))
    next_row = (cu // t + 1) * t
    keep = is_row | ((gap_prev >= 8) & (next_row - cu >= 8))
    return cu[keep].to(torch.int32)


class DeltaNetAdapter(nn.Module):
    """fla GatedDeltaNet behind the CausalSelfAttention interface (ve/rope/sinks unused).
    E25c: recurrent state resets at doc/row boundaries via fla varlen (cu_seqlens)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert _HAS_FLA, "flash-linear-attention not installed"
        # E25d: KDA (vector-gated delta rule, Kimi K3) in place of scalar-gated GatedDeltaNet
        # head_dim 64 x 12 heads: KDA's kernel needs >101KB smem at head_dim 128 on sm86
        self.dn = KimiDeltaAttention(hidden_size=config.n_embd, num_heads=config.n_embd // 64,
                                     head_dim=64, expand_v=1, mode="chunk")

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, ve: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None, k_sink=None, v_sink=None, doc_cu=None) -> torch.Tensor:
        out = self.dn(x)   # no doc-reset: E25c measured cross-doc state as beneficial
        return out[0] if isinstance(out, tuple) else out


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, MLP_RATIO * config.n_embd, bias=False)
        self.c_proj = Linear(MLP_RATIO * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(relu_squared(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_id: int = -1):
        super().__init__()
        self.attn = DeltaNetAdapter(config) if layer_id in DELTANET_LAYERS else CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.resid_lambda = nn.Parameter(torch.ones(1))
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        # value embedding gate: v = c_v(x) + ve_lambda * vte(idx)
        self.ve_lambda = nn.Parameter(torch.full((1,), 0.5))
        # attention registers: learned per-head sink keys AND values, all zero-init ->
        # starts as pure off-by-one sinks, can become global memory tokens.
        self.k_sink = nn.Parameter(torch.zeros(NUM_SINKS, config.n_head, QK_DIM))
        self.v_sink = nn.Parameter(torch.zeros(NUM_SINKS, config.n_head, config.n_embd // config.n_head))

    def forward(self, x: torch.Tensor, x0: torch.Tensor, ve: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None, doc_cu=None) -> torch.Tensor:
        x = self.resid_lambda * x + self.x0_lambda * x0
        if isinstance(self.attn, DeltaNetAdapter):
            x = x + self.attn(norm(x), self.ve_lambda * ve, cos, sin, block_mask, self.k_sink, self.v_sink, doc_cu)
        else:
            x = x + self.attn(norm(x), self.ve_lambda * ve, cos, sin, block_mask, self.k_sink, self.v_sink)
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
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.bigram_lambda = nn.Parameter(torch.full((1,), 0.2))
        # asymmetric logit softcap: independent learned pos/neg caps, init symmetric
        self.softcap_pos = nn.Parameter(torch.full((1,), LOGIT_SOFTCAP))
        self.softcap_neg = nn.Parameter(torch.full((1,), LOGIT_SOFTCAP))
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # U-net skips: encoder half pushes activations, decoder half pops them
        self.skip_lambdas = nn.Parameter(torch.ones(config.n_layer // 2))
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, QK_DIM)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.vte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.vte_mid.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.vte2.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.transformer.bte.weight, mean=0.0, std=0.8)
        self.bigram_lambda.fill_(0.2)
        self.softcap_pos.fill_(LOGIT_SOFTCAP)
        self.softcap_neg.fill_(LOGIT_SOFTCAP)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        for block in self.transformer.h:
            if isinstance(block.attn, CausalSelfAttention):
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
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, QK_DIM)
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
        deltanet_params: list = []
        matrix_params = []
        for block in self.transformer.h:
            scalar_params.extend([block.resid_lambda, block.x0_lambda, block.ve_lambda, block.k_sink, block.v_sink])
            if isinstance(block.attn, CausalSelfAttention):
                matrix_params.extend(block.attn.parameters())
            else:
                deltanet_params.extend(block.attn.parameters())
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
            dict(kind="adamw", params=deltanet_params, lr=0.01, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
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
        # hashed bigram embedding: capacity indexed by (prev, cur) token pair.
        # prev of position 0 is BOS (rows are BOS-aligned). Causal by construction.
        prev = torch.cat([torch.full_like(idx[:, :1], BOS_ID), idx[:, :-1]], dim=1)
        pair = prev.to(torch.int64) * 8192 + idx.to(torch.int64)
        bh = ((pair * -7046029254386353131) >> 32) & (BIGRAM_SIZE - 1)  # 0x9E3779B97F4A7C15 as signed i64
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = x + self.bigram_lambda * self.transformer.bte(bh).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        ve1 = self.transformer.vte(idx).to(COMPUTE_DTYPE)
        ve_mid = self.transformer.vte_mid(idx).to(COMPUTE_DTYPE)
        ve2 = self.transformer.vte2(idx).to(COMPUTE_DTYPE)
        doc_cu = doc_cu_seqlens(idx) if DELTANET_LAYERS else None
        t1 = self.config.n_layer // 3
        t2 = self.config.n_layer - t1
        block_mask = None
        if DOC_MASK and idx.is_cuda:
            # doc id via causal cumsum of BOS markers; mask = causal AND same-doc.
            # Built inside forward so the fixed evaluate_bpb (model(x)) gets it too.
            docs = (idx == BOS_ID).cumsum(dim=1)
            docs_kv = torch.cat([docs[:, :1].expand(-1, NUM_SINKS), docs], dim=1)  # first slots = sinks

            def doc_causal_sink(b, h, q_idx, kv_idx):
                real = (q_idx + NUM_SINKS >= kv_idx) & (docs[b, q_idx] == docs_kv[b, kv_idx])
                return (kv_idx < NUM_SINKS) | real

            block_mask = create_block_mask(doc_causal_sink, idx.size(0), None, t, t + NUM_SINKS, device=idx.device)
        n_enc = self.config.n_layer // 2
        n_dec_start = self.config.n_layer - n_enc  # odd depth: middle layer is neither
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.transformer.h):
            if i >= n_dec_start:
                x = x + self.skip_lambdas[i - n_dec_start] * skips.pop()
            ve_i = ve1 if i < t1 else (ve_mid if i < t2 else ve2)
            x = block(x, x0, ve_i, cos, sin, block_mask, doc_cu)
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


def build_config(depth: int = DEPTH) -> GPTConfig:
    base_dim = depth * ASPECT_RATIO
    n_embd = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    return GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=depth,
        n_head=n_head,
        n_embd=n_embd,
    )


# -----------------------------------------------------------------------------
# Optimizer (identical to train.py)

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
# Schedules

def lr_multiplier(step: int, elapsed_frac: float) -> float:
    """Warmup by steps, warmdown by elapsed TIME fraction (fixed-time protocol)."""
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    warmdown_start = 1.0 - WARMDOWN_FRAC
    if elapsed_frac <= warmdown_start:
        return 1.0
    frac = min(1.0, (elapsed_frac - warmdown_start) / WARMDOWN_FRAC)
    return 1.0 - (1.0 - FINAL_LR_FRAC) * frac


def muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


# -----------------------------------------------------------------------------
# Training

def main() -> None:
    parser = argparse.ArgumentParser(description="loss-per-step experimentation run")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--max-train-seconds", type=int, default=MAX_TRAIN_SECONDS)
    parser.add_argument("--device-batch-size", type=int, default=DEVICE_BATCH_SIZE)
    parser.add_argument("--out-dir", type=str, default="out_train2")
    parser.add_argument("--eval-tokens", type=int, default=4 * 524288)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                        help="torch.compile the model (default: on for cuda, off for cpu)")
    args = parser.parse_args()

    budget_started = time.monotonic()
    torch.manual_seed(42)
    device = torch.device(detect_device_type(args.device_type))
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"device={device} dtype={COMPUTE_DTYPE}")
    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = build_config()
    print(f"model_config={json.dumps(asdict(config))}")
    model = GPT(config)
    model.init_weights()
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params={num_params:,}")

    orig_model = model
    use_compile = args.compile if args.compile is not None else device.type == "cuda"
    if use_compile:
        model = torch.compile(model, dynamic=False)
        print("torch.compile enabled")

    device_batch = args.device_batch_size
    micro_tokens = device_batch * SEQ_LEN
    if TOTAL_BATCH_SIZE % micro_tokens != 0:
        raise ValueError(f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={micro_tokens}")
    grad_accum_steps = TOTAL_BATCH_SIZE // micro_tokens
    print(f"micro_tokens={micro_tokens:,} grad_accum_steps={grad_accum_steps}")

    # T10: organizer-sanctioned packing trick — fetch 8x-long rows and reshape, so only
    # docs longer than 8192 tokens lose their tail (vs 1024 before). Alignment preserved.
    PACK = 8
    _raw_loader = make_dataloader(tokenizer, device_batch // PACK, SEQ_LEN * PACK, "train", device)

    def _packed():
        for _x, _y, _st in _raw_loader:
            yield _x.reshape(-1, SEQ_LEN), _y.reshape(-1, SEQ_LEN), _st

    train_loader = _packed()
    x, y, loader_state = next(train_loader)
    optimizer = model.setup_optimizer()

    smooth_loss = 0.0
    step = 0
    truncated = False

    while step < args.num_steps:
        if time.monotonic() - budget_started >= args.max_train_seconds:
            truncated = True
            break

        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        for _ in range(grad_accum_steps):
            loss = model(x, y)
            last_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, loader_state = next(train_loader)

        lrm = lr_multiplier(step, (time.monotonic() - budget_started) / max(1.0, args.max_train_seconds))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step)
                group["weight_decay"] = WEIGHT_DECAY
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.monotonic() - t0
        train_loss = float(last_loss.item())
        if math.isnan(train_loss) or train_loss > 100:
            raise RuntimeError(f"bad loss detected: {train_loss}")

        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        if step < 5 or step % 10 == 0:
            remaining_time = max(0.0, args.max_train_seconds - (time.monotonic() - budget_started))
            print(
                f"step {step:05d}/{args.num_steps} | loss {debiased:.4f} | lrm {lrm:.3f} | "
                f"dt {dt:.2f}s | tok/s {TOTAL_BATCH_SIZE / max(dt, 1e-6):,.0f} | "
                f"remaining_time {remaining_time:.0f}s | epoch {loader_state['epoch']}",
                flush=True,
            )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1

    train_seconds = time.monotonic() - budget_started
    checkpoint_path = Path(args.out_dir) / "final.pt"
    save_checkpoint(checkpoint_path, orig_model, {
        "step": step,
        "num_steps_target": args.num_steps,
        "truncated": truncated,
        "train_seconds": train_seconds,
    })

    causal = causality_check(orig_model, tokenizer, device_batch, SEQ_LEN, device)
    result = evaluate_bpb(
        orig_model, tokenizer, device_batch, SEQ_LEN, device,
        split="public_val", max_tokens=args.eval_tokens,
    )

    print("---")
    # fixed-time protocol: stopping on the time budget is the NORMAL end condition
    print(f"val_bpb: {result['val_bpb']:.6f}")
    print(f"budget_s: {args.max_train_seconds}")
    print(f"steps:   {step}")
    print(f"params:  {num_params:,}")
    print(f"train_s: {train_seconds:.1f}")
    print(f"total_s: {time.monotonic() - budget_started:.1f}")
    print(f"causality_passed: {causal['passed']} (max_abs {causal['max_abs_diff']:.6f})")
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
