"""Tier-1 capability probes for the Trn2 native-torch beta (single process, eager).

Each probe is independently try/excepted and timed; output is one PASS/FAIL line
per probe so the log doubles as a capability matrix. CPU reference checks catch
silently-wrong math, not just crashes.
"""
import math
import time

import torch
import torch.nn.functional as F

DEV = None
RESULTS = []


def probe(name):
    def deco(fn):
        def run():
            t0 = time.monotonic()
            try:
                fn()
                dt = time.monotonic() - t0
                print(f"PASS  {name}  ({dt:.2f}s)", flush=True)
                RESULTS.append((name, "PASS", dt))
            except Exception as e:
                dt = time.monotonic() - t0
                msg = str(e).replace("\n", " | ")[:200]
                print(f"FAIL  {name}  ({dt:.2f}s): {type(e).__name__}: {msg}", flush=True)
                RESULTS.append((name, "FAIL", dt))
        return run
    return deco


def close(a, b, tol=2e-2):
    d = (a.float().cpu() - b.float().cpu()).abs().max().item()
    assert d < tol, f"max abs diff {d} > {tol}"


@probe("device_create")
def p_device():
    global DEV
    DEV = torch.device("neuron")
    x = torch.ones(4, 4, device=DEV)
    assert float(x.sum().cpu()) == 16.0


@probe("matmul_fp32_correctness")
def p_matmul32():
    a = torch.randn(128, 256)
    b = torch.randn(256, 128)
    close(a.to(DEV) @ b.to(DEV), a @ b, tol=1e-2)


@probe("matmul_bf16")
def p_matmul16():
    a = torch.randn(128, 256, dtype=torch.bfloat16)
    b = torch.randn(256, 128, dtype=torch.bfloat16)
    close(a.to(DEV) @ b.to(DEV), a.float() @ b.float(), tol=0.5)


@probe("embedding_int32_index")
def p_embed():
    w = torch.randn(1000, 64)
    idx = torch.randint(0, 1000, (8, 128), dtype=torch.int32)
    e = torch.nn.Embedding(1000, 64)
    with torch.no_grad():
        e.weight.copy_(w)
    en = torch.nn.Embedding(1000, 64).to(DEV)
    with torch.no_grad():
        en.weight.copy_(w.to(DEV))
    close(en(idx.to(DEV)), e(idx))


@probe("bigram_hash_int32_shift_and")
def p_hash():
    idx = torch.randint(0, 8192, (4, 64), dtype=torch.int32)
    prev = torch.cat([torch.full_like(idx[:, :1], 8188), idx[:, :-1]], dim=1)
    pair = prev * 8192 + idx
    h_cpu = ((pair * -1640531527) >> 13) & (2 ** 17 - 1)
    pn = pair.to(DEV)
    h_dev = ((pn * -1640531527) >> 13) & (2 ** 17 - 1)
    assert (h_dev.cpu() == h_cpu).all(), "hash mismatch on device"


@probe("cumsum_bool_docmask")
def p_cumsum():
    idx = torch.randint(0, 8192, (4, 256), dtype=torch.int32)
    docs_cpu = (idx == 8188).cumsum(dim=1)
    docs_dev = (idx.to(DEV) == 8188).cumsum(dim=1)
    assert (docs_dev.cpu() == docs_cpu).all()


@probe("masked_fill_softmax_attention")
def p_dense_attn():
    q = torch.randn(2, 2, 64, 32)
    k = torch.randn(2, 2, 64, 32)
    v = torch.randn(2, 2, 64, 32)
    pos = torch.arange(64)
    mask = (pos.view(1, 64, 1) >= pos.view(1, 1, 64)).unsqueeze(1)

    def attn(q, k, v, m):
        s = (q @ k.transpose(-2, -1)) / math.sqrt(32)
        s = s.masked_fill(~m, float("-inf"))
        return F.softmax(s, dim=-1) @ v

    close(attn(q.to(DEV), k.to(DEV), v.to(DEV), mask.to(DEV)), attn(q, k, v, mask), tol=3e-2)


@probe("sdpa_causal")
def p_sdpa():
    q = torch.randn(2, 2, 64, 32)
    k = torch.randn(2, 2, 64, 32)
    v = torch.randn(2, 2, 64, 32)
    close(
        F.scaled_dot_product_attention(q.to(DEV), k.to(DEV), v.to(DEV), is_causal=True),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        tol=3e-2,
    )


@probe("rms_norm")
def p_rms():
    x = torch.randn(4, 64, 256)
    close(F.rms_norm(x.to(DEV), (256,)), F.rms_norm(x, (256,)), tol=1e-2)


@probe("tanh_where_clamp_softcap")
def p_softcap():
    x = torch.randn(4, 64, 100) * 20
    sp = torch.tensor([11.0])
    def cap(x, sp):
        return torch.where(x >= 0, sp * torch.tanh(x / sp), sp * torch.tanh(x / sp))
    close(cap(x.to(DEV), sp.to(DEV)), cap(x, sp), tol=1e-2)


@probe("cross_entropy_ignore_index")
def p_ce():
    lg = torch.randn(256, 100)
    tg = torch.randint(0, 100, (256,))
    tg[::7] = -1
    close(
        F.cross_entropy(lg.to(DEV), tg.to(DEV), ignore_index=-1),
        F.cross_entropy(lg, tg, ignore_index=-1),
        tol=1e-2,
    )


@probe("cat_transpose_view_contiguous")
def p_shapes():
    x = torch.randn(2, 64, 4, 32)
    def f(x):
        y = torch.cat([x, x], dim=1).transpose(1, 2)
        return y.contiguous().view(2, 4, -1)
    close(f(x.to(DEV)), f(x))


@probe("miniblock_fwd_bwd_adamw")
def p_train_step():
    torch.manual_seed(0)
    class Mini(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = torch.nn.Embedding(512, 64)
            self.q = torch.nn.Linear(64, 64, bias=False)
            self.p = torch.nn.Linear(64, 512, bias=False)
        def forward(self, idx):
            x = F.rms_norm(self.e(idx), (64,))
            x = x + F.relu(self.q(x)).square()
            return self.p(x)
    m = Mini().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    idx = torch.randint(0, 512, (4, 64), dtype=torch.int32, device=DEV)
    tgt = torch.randint(0, 512, (4, 64), device=DEV)
    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(idx).view(-1, 512), tgt.view(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


@probe("bf16_weight_cast_linear")
def p_cast_linear():
    w = torch.randn(64, 64)
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    yc = F.linear(x.float(), w).to(torch.bfloat16)
    yn = F.linear(x.to(DEV), w.to(DEV).to(dtype=torch.bfloat16))
    close(yn, yc, tol=0.5)


@probe("stacked_muon_ops_norm_lerp_mT")
def p_muon_ops():
    g = torch.randn(10, 64, 64)
    m = torch.zeros_like(g)
    def f(g, m):
        m = m.lerp(g, 0.05)
        x = g.lerp(m, 0.95)
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        return x @ x.mT
    close(f(g.to(DEV), m.to(DEV)), f(g, m), tol=1e-2)


@probe("torch_compile_backend_neuron")
def p_compile():
    def f(x):
        return F.relu(x).square().sum()
    cf = torch.compile(f, backend="neuron", dynamic=False)
    x = torch.randn(32, 32, device=DEV)
    close(cf(x), f(x.cpu()), tol=1e-2)


@probe("int64_tensor_roundtrip")
def p_int64():
    x = torch.arange(100, dtype=torch.int64)
    y = (x.to(DEV) * 3 + 1).cpu()
    assert (y == x * 3 + 1).all()


if __name__ == "__main__":
    t0 = time.monotonic()
    for fn in [p_device, p_matmul32, p_matmul16, p_embed, p_hash, p_cumsum,
               p_dense_attn, p_sdpa, p_rms, p_softcap, p_ce, p_shapes,
               p_train_step, p_cast_linear, p_muon_ops, p_compile, p_int64]:
        fn()
    npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print(f"TIER1_SUMMARY: {npass}/{len(RESULTS)} passed in {time.monotonic() - t0:.1f}s", flush=True)
