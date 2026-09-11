"""Device test (one core): a transformer-layer BACKWARD as a chain of existing kernels + a few torch matmuls, at the training
shape (4x1024 tokens, D=512, 4 heads x 128, MLP 2048), timed over 10 layers vs the compiled graph's ~82 ms backward per
micro-step. Pieces: mlp_bwd + rmsnorm_bwd + out-proj grads (2 matmuls) | vendor attention bwd (autograd) | rope bwd x2 +
qk-norm bwd x2 + Q/K/V projection grads (6 matmuls) + rmsnorm bwd + residual. usage: python nki_layer_bwd_chain_test.py"""
import importlib.util, os, time
import torch
os.environ.setdefault("NEURON_COMPETITION_R1_SINK", "logit"); os.environ.setdefault("NEURON_COMPETITION_R1_ATTN", "nki")
spec = importlib.util.spec_from_file_location("veh", "submission_raw/train3_submission.py")
veh = importlib.util.module_from_spec(spec); spec.loader.exec_module(veh)
dev = torch.device("neuron"); torch.manual_seed(0)
B, Tt, D, NH, HD, H = 4, 1024, 512, 4, 128, 2048
T = B * Tt
bf = lambda *s: torch.randn(*s).bfloat16().to(dev)
x1, h2, y2, dout = bf(T, D), bf(T, D), bf(T, D), (torch.randn(T, D) * 0.1).bfloat16().to(dev)
preT = bf(H, T); wo, wq, wk, wv = (bf(D, D) * 0.03 for _ in range(4)); w1 = bf(H, D) * 0.04; w2 = bf(D, H) * 0.02
xm = bf(T, D)
mlp_fwd_k, mlp_bwd_k, grid = veh.build_nki_mlp()
norm_fwd_k, norm_bwd_k, _ = veh.build_nki_norm()
rope_k, _ = veh.build_nki_rope()
pos = (torch.arange(Tt)).float(); inv = 1.0 / (10000 ** (torch.arange(0, HD, 2).float() / HD))
ang = torch.outer(pos, inv); cos2d, sin2d = ang.cos().contiguous().to(dev), ang.sin().contiguous().to(dev)
sink = torch.zeros(NH).to(dev)
idx = torch.randint(1, 8192, (B, Tt), dtype=torch.int32); idx[:, 0] = veh.BOS_ID; idx = idx.to(dev)
start, end = veh.doc_bounds(idx)
q4, k4, v4 = (bf(B, NH, Tt, HD).requires_grad_(True) for _ in range(3))
rq, rk, hh = bf(T, D), bf(T, D), bf(T, D)                                  # saved rotated q/k and normed input (stand-ins, hoisted)
scale = 1.0 / (HD ** 0.5)

def post_bwd(dout):
    # out = x1 + mlp(norm(x1)) ; x1 = xm + y2 @ Wo^T
    dx, dw1, dw2 = mlp_bwd_k[grid](h2, dout, preT, w1, w2)                  # grad wrt h2 and the MLP weights
    dx1 = dout + norm_bwd_k[grid](x1, dx)                                    # residual + norm backward
    dy2 = torch.nn.functional.linear(dx1, wo.t())                            # dy = dx1 @ Wo
    dwo = torch.matmul(dx1.t(), y2)                                          # dWo = dx1^T y2
    return dy2, dx1, dwo, dw1, dw2

def attn_bwd(dy2):
    dy4 = dy2.view(B, Tt, NH, HD).permute(0, 2, 1, 3).contiguous()
    y = veh.nki_doc_attention(q4, k4, v4, sink, start, end, scale)
    gq, gk, gv = torch.autograd.grad(y, (q4, k4, v4), dy4)
    return gq, gk, gv

def pre_bwd(gq, gk, gv, dxm_in):
    # q = norm(rope(h Wq^T)) per head, k likewise, v = h Wv^T + ve ; h = norm(xm)
    gq2 = gq.permute(0, 2, 1, 3).reshape(T, D); gk2 = gk.permute(0, 2, 1, 3).reshape(T, D); gv2 = gv.permute(0, 2, 1, 3).reshape(T, D)
    dq = norm_bwd_k[grid](rq.view(T * NH, HD), gq2.reshape(T * NH, HD).contiguous()).view(T, D)
    dk = norm_bwd_k[grid](rk.view(T * NH, HD), gk2.reshape(T * NH, HD).contiguous()).view(T, D)
    dq = rope_k[grid](dq.contiguous(), cos2d, sin2d, n_head=NH, sign=-1.0)
    dk = rope_k[grid](dk.contiguous(), cos2d, sin2d, n_head=NH, sign=-1.0)
    dh = torch.nn.functional.linear(dq, wq.t()) + torch.nn.functional.linear(dk, wk.t()) + torch.nn.functional.linear(gv2, wv.t())
    dwq = torch.matmul(dq.t(), hh); dwk = torch.matmul(dk.t(), hh); dwv = torch.matmul(gv2.t(), hh)
    dxm = dxm_in + norm_bwd_k[grid](xm, dh.contiguous())
    return dxm, dwq, dwk, dwv

def layer_bwd(dout):
    dy2, dx1, dwo, dw1, dw2 = post_bwd(dout)
    gq, gk, gv = attn_bwd(dy2)
    dxm, dwq, dwk, dwv = pre_bwd(gq, gk, gv, dx1)
    return dxm

d = layer_bwd(dout); torch.neuron.synchronize(); print("layer backward ok", tuple(d.shape), flush=True)
t0 = time.monotonic()
for _ in range(10): d = layer_bwd(dout)
torch.neuron.synchronize(); dt = (time.monotonic() - t0) / 10 * 1000
print(f"CHAIN layer backward: {dt:.2f} ms per layer -> {dt * 10:.1f} ms per 10-layer micro-step (compiled bwd ~82 ms)", flush=True)
for name, fn in (("post_bwd", lambda: post_bwd(dout)), ("attn_bwd", lambda: attn_bwd(dout)), ("pre_bwd", lambda: pre_bwd(q4.detach(), k4.detach(), v4.detach(), dout))):
    fn(); torch.neuron.synchronize(); t0 = time.monotonic()
    for _ in range(10): fn()
    torch.neuron.synchronize(); print(f"  {name}: {(time.monotonic() - t0) / 10 * 1000:.2f} ms", flush=True)
print("done")
