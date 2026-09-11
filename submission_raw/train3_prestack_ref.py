print("SUBMISSION_BOOT: module import starting", flush=True)  # first executable line:
# discriminates "our code never ran" from "crashed later" in harness stdout logs

"""
train3_submission.py - v7 (2026-09-06): v6 + minimum-novelty portal build after sub-39b23de5 (v6, eager)
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
N_EMBD = int(_ENV("NEURON_COMPETITION_R1_N_EMBD", "256"))   # fixed-time optimum on 3090 (T5/T11); Trn2 re-tunes
N_HEAD = int(_ENV("NEURON_COMPETITION_R1_N_HEAD", "2"))
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
NUM_SINKS = 4                   # attention registers per layer
BIGRAM_SIZE = 2 ** 17
TRIGRAM_SIZE = 2 ** 17
QK_DIM = int(_ENV("NEURON_COMPETITION_R1_QK_DIM", "0"))              # dev knob; 0 = head_dim
ATTN_IMPL = _ENV("NEURON_COMPETITION_R1_ATTN", "explicit")           # dev knob: explicit | sdpa (bool mask; NKI-able)
PROFILE = _ENV("NEURON_COMPETITION_R1_PROFILE", "0") != "0"           # dev knob: per-phase step timing, steps 10-40
USE_TRIGRAM = _ENV("NEURON_COMPETITION_R1_TRIGRAM", "0") != "0"   # dev knob; L1 on the 4080: -0.0042 alone

EMBEDDING_LR = 0.30 * 0.71 * _BATCH_LR     # sqrt-batch LR compensation for TOTAL_BATCH 32768 (T6b)
UNEMBEDDING_LR = 0.006 * 0.71 * _BATCH_LR
MATRIX_LR = 0.025 * 0.71 * _BATCH_LR
SCALAR_LR = 0.20 * 0.71 * _BATCH_LR
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
    use_trigram: bool = False   # saved in the checkpoint so load_for_eval rebuilds the same model
    qk_dim: int = 0             # 0 -> head_dim (champion). L2 on the 4080: qk 64 / v 128 was -0.0075 via +27% steps


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
        self.qk_dim = config.qk_dim or self.head_dim   # narrow QK: q/k at qk_dim, v stays head_dim
        self.c_q = Linear(config.n_embd, self.n_head * self.qk_dim, bias=False)
        self.c_k = Linear(config.n_embd, self.n_head * self.qk_dim, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, ve, cos, sin, block_mask=None, dense_mask=None, k_sink=None, v_sink=None):
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.qk_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.qk_dim)
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
            _, flex_attention = _get_flex()
            y = flex_attention(q, k, v, block_mask=block_mask, kernel_options=FLEX_KERNEL_OPTS)
        elif ATTN_IMPL == "sdpa":
            # fused path: bool mask (True = attend), scale = 1/sqrt(qk_dim) from q's last dim
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)
        else:
            # Portable path (Neuron/CPU): explicit attention with only baseline-proven ops.
            scores = (q.float() @ k.float().transpose(-2, -1)) * (1.0 / math.sqrt(self.qk_dim))
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
        self.k_sink = nn.Parameter(torch.zeros(NUM_SINKS, config.n_head, config.qk_dim or config.n_embd // config.n_head))
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

    def setup_optimizer(self):
        scalar_params = [self.skip_lambdas, self.bigram_lambda, self.softcap_pos, self.softcap_neg]
        if self.config.use_trigram:
            scalar_params.append(self.trigram_lambda)
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
        if self.config.use_trigram:
            embedding_params += list(self.transformer.tte.parameters())
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
        block_mask = None
        dense_mask = None
        if idx.is_cuda:
            create_block_mask, _ = _get_flex()
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
        use_trigram=USE_TRIGRAM,
        qk_dim=QK_DIM,
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
        params = group["params"]
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
        """Host->device refresh of the per-step scalars (lr, momentum), done eagerly right before step() so a
        compiled step() sees them as ordinary tensor inputs: no scalar-valued ops or H2D copies in the graph."""
        for group in self.param_groups:
            if not group["params"]:
                continue
            dev = group["params"][0].device
            self._dev_scalar(group, "lr_t", group["lr"], dev)
            if group["kind"] == "muon":
                self._dev_scalar(group, "momentum_t", group["momentum"], dev)

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
        beta1, beta2 = group["betas"]
        lr_t = self._group_scalar(group, "lr_t")
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self._adamw_state(p)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
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
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        shape = p0.shape
        state = self._muon_state(group)

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

        mom_t = self._group_scalar(group, "momentum_t").to(momentum_buffer.dtype)
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

        lr = self._group_scalar(group, "lr_t") * (max(1.0, shape[-2] / shape[-1]) ** 0.5)
        wd = group["weight_decay"]
        mask = (g * w) >= 0
        w.sub_(lr * g + lr * wd * w * mask)
        for i, p in enumerate(params):  # not torch._foreach_copy_: the neuron Torch-MLIR lowering cannot
            p.copy_(w[i])               # legalize the foreach op inside a compiled optimizer step (Trn2 S3)

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
                        default=os.environ.get("NEURON_COMPETITION_R1_COMPILE", "0") != "0")
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

    orig_model = model
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
