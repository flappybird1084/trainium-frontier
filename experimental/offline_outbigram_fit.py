"""Redundancy test for the output-side bigram bias on a TRAINED checkpoint.

Freeze the v72 model; learn only B (V, V), zero-init, on TRAIN tokens by gradient, with
logits = softcap(z + B[x_t]) (bias before the cap, as the proposed kernel would do). The loss is convex in B.
Score on public_val with prepare.evaluate_bpb at intervals. Gain ~0 => the trained model already expresses the
bigram it could get from B (redundant). A measurement only: nothing here ships.
"""
import argparse, importlib.util, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/rianbutala/projects/trainium-frontier")
import prepare  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/rianbutala/projects/trainium-frontier/ckpt/v72_final.pt")
ap.add_argument("--vehicle", default="/home/rianbutala/projects/trainium-frontier/submission_raw/submission_v72_clean_dyfuse.py")
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--opt", default="adam")
ap.add_argument("--c0", type=float, default=4.0)
ap.add_argument("--decay", type=int, default=0)
ap.add_argument("--rows", type=int, default=8)
ap.add_argument("--train-tokens", type=int, default=60_000_000)
ap.add_argument("--evals", default="2000000,8000000,20000000,40000000,60000000")
ap.add_argument("--eval-tokens", type=int, default=1048576)
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("veh", a.vehicle)
veh = importlib.util.module_from_spec(spec); spec.loader.exec_module(veh)
dev = torch.device("cuda")
tok = prepare.ensure_tokenizer()
V = tok.get_vocab_size()
m = veh.load_for_eval(a.ckpt, dev)
for p in m.parameters():
    p.requires_grad_(False)
m.head_logits = lambda x: m.lm_head(veh.norm(x))[..., :V].float()     # pre-softcap z
sp = m.softcap_pos.clamp_min(1.0).float(); sn = m.softcap_neg.clamp_min(1.0).float()
cap = lambda z: torch.where(z >= 0, sp * torch.tanh(z / sp), sn * torch.tanh(z / sn))
B = torch.zeros(V, V, device=dev, requires_grad=True)
opt = torch.optim.Adam([B], lr=a.lr, betas=(0.9, 0.99), eps=1e-10)


class Wrap(nn.Module):
    def __init__(self, use_b):
        super().__init__(); self.use_b = use_b
    def forward(self, idx, *aa, **kk):
        z = m(idx)
        return cap(z + B.detach()[idx]) if self.use_b else cap(z)


def score(use_b):
    with torch.no_grad():
        return prepare.evaluate_bpb(Wrap(use_b), tok, 4, veh.SEQ_LEN, dev, split="public_val",
                                    max_tokens=a.eval_tokens)["val_bpb"]


ref = score(False)
print(f"BASE val_bpb {ref:.6f}  sp={sp.item():.3f} sn={sn.item():.3f}", flush=True)
evals = [int(s) for s in a.evals.split(",")]
loader = prepare.make_dataloader(tok, a.rows, veh.SEQ_LEN, "train", dev, tokenizer_threads=4)
seen, acc_b, acc_0, n_acc, t0 = 0, 0.0, 0.0, 0, time.time()
for x, y, _ in loader:
    with torch.no_grad():
        z = m(x)
    l = cap(z + B[x])
    loss = F.cross_entropy(l.view(-1, V), y.reshape(-1), ignore_index=-1)
    if a.opt == "adam":
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    else:
        # row-normalised SGD: B[v] -= lr * mean over this batch's occurrences of v of dL/dz' (per-token, unaveraged)
        B.grad = None; loss.backward()
        with torch.no_grad():
            cnt = torch.bincount(x.reshape(-1), minlength=V).float().clamp_min(a.c0)
            B.sub_(a.lr * (max(0.0, 1.0 - seen / a.train_tokens) if a.decay else 1.0) * B.grad * (y.numel() / cnt).unsqueeze(1))
    with torch.no_grad():
        acc_0 += F.cross_entropy(cap(z).view(-1, V), y.reshape(-1), ignore_index=-1).item()
    acc_b += loss.item(); n_acc += 1
    seen += x.numel()
    if evals and seen >= evals[0]:
        evals.pop(0)
        v = score(True)
        print(f"tokens {seen/1e6:6.1f}M  train CE base {acc_0/n_acc:.5f} withB {acc_b/n_acc:.5f} (d {(acc_b-acc_0)/n_acc:+.5f})"
              f"  |B| rms {B.detach().pow(2).mean().sqrt().item():.4f} max {B.detach().abs().max().item():.3f}"
              f"  val_bpb {v:.6f}  delta {v-ref:+.6f}  {time.time()-t0:.0f}s", flush=True)
        acc_b = acc_0 = 0.0; n_acc = 0
    if seen >= a.train_tokens or not evals:
        break
