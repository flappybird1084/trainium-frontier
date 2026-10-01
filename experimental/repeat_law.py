"""Measure the data-repetition penalty on OUR corpus at small scale (local RTX 4080).

Fixed total training tokens and fixed optimizer steps in every arm; only the size of the
UNIQUE token pool changes, so the arms differ solely in how many times each token is seen.
That isolates the repetition term and lets us fit Muennighoff's R_D* on ClimbMix + the
competition tokenizer instead of importing 15.387 from a C4 fit.

Never touches the val shard: validation rows come from held-out DOCUMENTS of train shard 0.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare  # noqa: E402

DEV = torch.device("cuda")
SEQ = 512
os.makedirs(os.environ.get("REPEAT_LAW_CACHE", os.path.expanduser("~/.cache/repeat_law")), exist_ok=True)
CACHE = os.environ.get("REPEAT_LAW_CACHE", os.path.expanduser("~/.cache/repeat_law"))


# ---------------------------------------------------------------- data
def build_tokens(n_tokens_target: int, skip_docs: int, tag: str) -> np.ndarray:
    """Tokenize documents from train shards into one flat int32 stream (BOS-prefixed)."""
    path = f"{CACHE}/toks_{tag}_{n_tokens_target}.npy"
    if os.path.exists(path):
        return np.load(path)
    tok = prepare.ensure_tokenizer()
    bos = tok.get_bos_token_id()
    paths = prepare.list_shard_paths(prepare.PUBLIC_TRAIN_SHARDS)
    import pyarrow.parquet as pq
    out, total, seen = [], 0, 0
    for p in paths:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            rows = pf.read_row_group(rg).column("text").to_pylist()
            if seen + len(rows) <= skip_docs:
                seen += len(rows)
                continue
            if seen < skip_docs:
                rows = rows[skip_docs - seen:]
                seen = skip_docs
            enc = tok.encode(rows, prepend=bos, num_threads=8)
            for e in enc:
                out.append(np.asarray(e, dtype=np.int32))
                total += len(e)
            if total >= n_tokens_target:
                break
        if total >= n_tokens_target:
            break
    arr = np.concatenate(out)[:n_tokens_target]
    assert arr.shape[0] == n_tokens_target, f"only got {arr.shape[0]} of {n_tokens_target}"
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------- model
def rms(x, eps=1.0 / 128):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class Block(nn.Module):
    def __init__(self, d, h, ratio):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, ratio * d, bias=False)
        self.proj = nn.Linear(ratio * d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(rms(x)).split(D, dim=-1)
        q = rms(q.view(B, T, self.h, self.hd)).transpose(1, 2)
        k = rms(k.view(B, T, self.h, self.hd)).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        x = x + self.proj(F.relu(self.fc(rms(x))).square())
        return x


def rope(t, cos, sin):
    d = t.shape[-1] // 2
    a, b = t[..., :d], t[..., d:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


class GPT(nn.Module):
    def __init__(self, V, d, L, h, ratio):
        super().__init__()
        self.wte = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, h, ratio) for _ in range(L)])
        self.head = nn.Linear(d, V, bias=False)
        hd = d // h
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(SEQ).float()
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos()[None, None], persistent=False)
        self.register_buffer("sin", f.sin()[None, None], persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return 15.0 * torch.tanh(self.head(rms(x)) / 15.0)


# ---------------------------------------------------------------- train / eval
def run(pool_tokens, total_tokens, wd, d, L, h, ratio, bs, lr, seed, val_x, val_y, tb, label):
    torch.manual_seed(seed)
    V = prepare.TOKENIZER_VOCAB_SIZE
    model = GPT(V, d, L, h, ratio).to(DEV).to(torch.bfloat16)
    for m in model.modules():  # keep norms/params in fp32 master via AdamW on bf16 is bad -> use fp32 weights
        pass
    model = model.float()
    n_dense = sum(p.numel() for n, p in model.named_parameters() if "wte" not in n and "head" not in n)
    n_all = sum(p.numel() for p in model.parameters())

    tokens_per_step = bs * SEQ
    steps = total_tokens // tokens_per_step
    rows = pool_tokens.shape[0] // (SEQ + 1)
    pool = torch.from_numpy(pool_tokens[: rows * (SEQ + 1)].reshape(rows, SEQ + 1).astype(np.int64))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd, fused=True)
    warm, wdown = int(0.02 * steps), int(0.4 * steps)

    g = torch.Generator().manual_seed(seed)
    order, cur = torch.randperm(rows, generator=g), 0
    t0 = time.time()
    for s in range(steps):
        if cur + bs > rows:
            order, cur = torch.randperm(rows, generator=g), 0
        sel = order[cur:cur + bs]; cur += bs
        batch = pool[sel].to(DEV, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        f = s / steps
        m = min(1.0, (s + 1) / max(warm, 1))
        if f > 1 - wdown / steps:
            m *= max(0.02, (1 - f) / (wdown / steps))
        for pg in opt.param_groups:
            pg["lr"] = lr * m
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

    # exact bpb on held-out documents
    model.eval()
    nats = 0.0; nbytes = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, val_x.shape[0], 32):
            xb = val_x[i:i + 32].to(DEV); yb = val_y[i:i + 32].to(DEV)
            lg = model(xb).float()
            l = F.cross_entropy(lg.view(-1, V), yb.reshape(-1), reduction="none")
            yf = yb.reshape(-1)
            nb = tb[yf]
            mask = nb > 0
            nats += (l * mask).sum().item()
            nbytes += int((nb).sum().item())
    bpb = nats / (math.log(2) * nbytes)
    epochs = total_tokens / pool_tokens.shape[0]
    rec = dict(label=label, pool=int(pool_tokens.shape[0]), total=int(total_tokens), epochs=round(epochs, 3),
               wd=wd, d=d, L=L, ratio=ratio, bs=bs, lr=lr, steps=int(steps), seed=seed,
               n_dense=int(n_dense), n_all=int(n_all), bpb=round(bpb, 6), secs=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True)
    del model, opt, pool
    torch.cuda.empty_cache()
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=64_000_000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=str, default="1,4,8,16,32,64")
    ap.add_argument("--out", type=str, default=f"{CACHE}/repeat_law.jsonl")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    tb = prepare.get_token_bytes(DEV).long()

    train_all = build_tokens(a.total + 2_000_000, skip_docs=20_000, tag="train")
    val_all = build_tokens(1_600_000, skip_docs=0, tag="val")
    vr = val_all.shape[0] // (SEQ + 1)
    vv = torch.from_numpy(val_all[: vr * (SEQ + 1)].reshape(vr, SEQ + 1).astype(np.int64))
    val_x, val_y = vv[:, :-1], vv[:, 1:]
    print(f"train pool {train_all.shape[0]:,} tok | val {vr} rows / {vr*SEQ:,} tok", flush=True)

    with open(a.out, "a") as fh:
        for e in [float(z) for z in a.epochs.split(",")]:
            pool_n = int(a.total / e)
            rec = run(train_all[:pool_n], a.total, a.wd, a.d, a.L, a.heads, a.ratio,
                      a.bs, a.lr, a.seed, val_x, val_y, tb, f"E{e:g}_wd{a.wd}_d{a.d}L{a.L}_s{a.seed}")
            fh.write(json.dumps(rec) + "\n"); fh.flush()
