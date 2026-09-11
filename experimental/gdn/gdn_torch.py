"""Chunked gated delta rule (Gated DeltaNet, Yang et al. 2024) in plain torch ops, for the compiled Neuron graph.
State S (d_k x d_v) per head: S_t = exp(g_t) S_{t-1} + beta_t k_t (v_t - k_t^T S_{t-1})^T ; o_t = S_t^T q_t  (fla's recurrent form).
Chunked form: per chunk of C tokens the intra-chunk solve is the WY representation with the unit-lower-triangular inverse
computed EXACTLY by a nilpotent product (I + A)^{-1} = prod_k (I + (-A)^(2^k)); chunks are carried by a short recurrence."""
import torch


def gdn_recurrent(q, k, v, g, beta):
    """Reference. q, k, v: (B, T, d); g: (B, T) log-decay (<= 0); beta: (B, T). Returns o (B, T, d)."""
    B, T, d = q.shape
    S = q.new_zeros(B, d, d)
    outs = []
    for t in range(T):
        S = S * torch.exp(g[:, t]).view(B, 1, 1)
        kt = k[:, t]; vt = v[:, t]
        v_new = vt - torch.einsum("bk,bkv->bv", kt, S)
        S = S + beta[:, t].view(B, 1, 1) * kt.unsqueeze(-1) * v_new.unsqueeze(1)
        outs.append(torch.einsum("bk,bkv->bv", q[:, t], S))
    return torch.stack(outs, dim=1)


def _unit_lower_inverse(A):
    """(I + A)^{-1} for strictly lower-triangular A (..., C, C): exact nilpotent product."""
    C = A.shape[-1]
    I = torch.eye(C, dtype=A.dtype, device=A.device)
    X = -A                       # (I + A)^{-1} = sum_n (-A)^n = prod_k (I + X^(2^k))
    R = I + X
    n = 1
    while n < C:
        X = X @ X
        R = R @ (I + X)
        n *= 2
    return R


def gdn_chunked(q, k, v, g, beta, C=64):
    """q, k, v: (B, T, d) (q, k already normalized); g: (B, T) log-decay; beta: (B, T). T % C == 0. Returns o (B, T, d)."""
    B, T, d = q.shape
    n = T // C
    q = q.view(B, n, C, d); k = k.view(B, n, C, d); v = v.view(B, n, C, d)
    g = g.view(B, n, C); beta = beta.view(B, n, C)
    gc = torch.cumsum(g, dim=-1)                                   # gamma_i = sum_{j<=i} g_j within the chunk
    idx = torch.arange(C, device=q.device)
    causal = (idx.view(C, 1) >= idx.view(1, C))                    # j <= i
    strict = (idx.view(C, 1) > idx.view(1, C))                     # j < i
    dexp = gc.unsqueeze(-1) - gc.unsqueeze(-2)                     # gamma_i - gamma_j (> 0 above the diagonal: mask BEFORE exp,
    decay = torch.exp(torch.where(causal, dexp, torch.zeros_like(dexp)))   # or 0 * inf poisons the backward)
    kb = k * beta.unsqueeze(-1)
    A = (kb @ k.transpose(-1, -2)) * decay
    A = torch.where(strict, A, torch.zeros_like(A))
    Tm = _unit_lower_inverse(A)                                    # (I + A)^{-1}
    w = Tm @ (kb * torch.exp(gc).unsqueeze(-1))                     # (B, n, C, d)
    u = Tm @ (v * beta.unsqueeze(-1))
    qk = (q @ k.transpose(-1, -2)) * decay
    qk = torch.where(causal, qk, torch.zeros_like(qk))
    qg = q * torch.exp(gc).unsqueeze(-1)
    kg = k * torch.exp(gc[..., -1:] - gc).unsqueeze(-1)              # k_j * exp(gamma_C - gamma_j)
    S = q.new_zeros(B, d, d)
    outs = []
    for c in range(n):
        v_new = u[:, c] - w[:, c] @ S
        o = qg[:, c] @ S + qk[:, c] @ v_new
        S = torch.exp(gc[:, c, -1]).view(B, 1, 1) * S + kg[:, c].transpose(-1, -2) @ v_new
        outs.append(o)
    return torch.stack(outs, dim=1).view(B, T, d)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d, C = 3, 256, 32, 64
    q = torch.nn.functional.normalize(torch.randn(B, T, d, dtype=torch.float64), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, d, dtype=torch.float64), dim=-1)
    v = torch.randn(B, T, d, dtype=torch.float64)
    g = -torch.nn.functional.softplus(torch.randn(B, T, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(B, T, dtype=torch.float64))
    ref = gdn_recurrent(q, k, v, g, beta)
    out = gdn_chunked(q, k, v, g, beta, C)
    print("max |chunked - recurrent| =", (out - ref).abs().max().item(), "(ref max", ref.abs().max().item(), ")")
