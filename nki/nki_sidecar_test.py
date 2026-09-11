"""Device test of the vehicle's EmbedSidecar on one real micro-batch (one core): gather rows vs the tables, scatter-add vs
torch index_add_, host index ranges vs table sizes. usage: NEURON_COMPETITION_R1_BIGRAM=18 python nki_sidecar_test.py"""
import importlib.util, os, sys
import numpy as np, torch
os.environ.setdefault("NEURON_COMPETITION_R1_EMBED", "sidecar")
os.environ.setdefault("NEURON_COMPETITION_R1_DEPTH", "2")
spec = importlib.util.spec_from_file_location("veh", "submission_raw/train3_submission.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = torch.device("neuron")
torch.manual_seed(0)
model = m.GPT(m.build_config()); model.init_weights(); model.to(dev)
tok = m.ensure_tokenizer(build_if_missing=False)
dl = m.make_dataloader(tok, 4, m.SEQ_LEN, "train", dev)
x, y, _ = next(dl)
sc = m.EmbedSidecar(model, "sidecar")
print("tables:", {k: tuple(w.shape) for k, w in sc.tables.items()}, "gbuf:", {k: tuple(g.shape) for k, g in sc.gbuf.items()})
plans = sc.prepare(x)
for kind, (ind, ind_t, perm_t, sidx_t) in plans.items():
    R = sc.tables[sc.kinds[kind][0]].shape[0]
    print(f"kind {kind}: n={len(ind)} idx max={int(ind.max())} R={R} perm max={int(perm_t.max().item())} sidx max={int(sidx_t.max().item())} slots={sidx_t.shape[0]}", flush=True)
sc.start_step()
emb, plans = sc.gather(x, plans)
torch.neuron.synchronize()
ids = x.reshape(-1)
for name in ("wte", "bte"):
    ref = sc.tables[name][plans["bg" if name == "bte" else "tok"][0]].to(m.COMPUTE_DTYPE) if name == "bte" else sc.tables[name][ids].to(m.COMPUTE_DTYPE)
    got = emb[name].reshape(-1, emb[name].shape[-1])
    print(f"gather {name}: max err {(got.float().cpu() - ref.float().cpu()).abs().max().item():.4f}", flush=True)
for name, e in emb.items():
    if e.requires_grad:
        e.grad = torch.randn_like(e)
sc.scatter(emb, plans)
torch.neuron.synchronize()
for kind, names in sc.kinds.items():
    ind = plans[kind][0]
    for name in names:
        ref = torch.zeros_like(sc.gbuf[name].cpu()); ref[:sc.tables[name].shape[0]].index_add_(0, torch.from_numpy(ind), emb[name].grad.reshape(-1, emb[name].shape[-1]).float().cpu())
        got = sc.gbuf[name].cpu()
        print(f"scatter {name}: max err {(got[:sc.tables[name].shape[0]] - ref[:sc.tables[name].shape[0]]).abs().max().item():.4f}", flush=True)
print("done")
