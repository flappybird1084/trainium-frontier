# The backward is VectorE-bound: what deleting VectorE work is worth (2026-09-23)

Follow-on to `step-budget-2026-09-22b.md`. That session closed the *fixed-overhead* axis (grad accumulation, LR
scaling, split schedules, bigram confidence — all lost, see the R_* rows in `trn2_results.tsv`). This one attacks
the 80% of the step that is `fwd_bwd`.

## Where the step actually goes (P_SPLIT, PROFILE=2, v46 config)

```
PROFILE  fwd_bwd=0.425  grad_sync=0.045  optimizer=0.052  total=0.523
PROFILE2 (synchronized ms per micro-step): gather=3.0  fwd=126.4  bwd=252.1  host=19.6  scatter=21.3
```
Sum 422 ms ~ fwd_bwd 425 ms, so the earlier "123 ms unaccounted" is resolved: the two compiled graphs are
**378 ms of the 425** (89%), and **the backward alone is 252 ms = 48% of the whole step**. The sidecar
(gather 3.0 + scatter 21.3) and host (19.6, overlapped in the real run) are the rest.

The engine profile says the backward is **VectorE-bound** (Vector 82% busy, Scalar 45%, Tensor 51%). Independent
confirmation: `R_RATIO7` (MLP_RATIO 6 -> 7, i.e. +17% hidden width) cost **+22% step time** — super-linear, so it
is the (H, T) hidden tensor, not the matmuls, that sets the step.

## The rule that came out of it: DELETE VectorE work, do not RELOCATE it

Measured with paired 300 s PROFILE screens (fwd_bwd / total, baseline 0.416 / 0.522):

| change | what it does | fwd_bwd | total | verdict |
|---|---|---|---|---|
| `MLP_WG_GROUP=4` | MLP weight-grad matmuls accumulate over 4 token blocks in PSUM -> one VectorE add per group instead of per block (deletes 3.62e9 element-passes) | 0.394 | **0.494 (-5.6%)** | **WIN** |
| `MLP_STORE_H=2` | forward saves h instead of pre; backward recovers relu = sqrt(h) on ScalarE (relocates ~1.8e9 off VectorE) | 0.418 | 0.519 (-0.6%) | **worthless** |
| both | | 0.397 | 0.494 | does not stack |
| `MLP_WG_GROUP=2` | half-strength grouping | 0.404 | 0.503 (-3.8%) | confirms the mechanism scales |
| `+ NORM_FUSE + ROPE_SKIP` | two further exact deletions (below) | 0.388 | **0.489 (-6.3%)** | small further win |

**Why relocation failed:** ScalarE `activation` is not a free lunch. `nl.sqrt` is a transcendental and appears to
run well below VectorE's bf16 rate, and per the playbook **VectorE is 2-4x faster on bf16 SBUF operands while
ScalarE is not**. So moving bf16 elementwise work to ScalarE buys nothing even when VectorE is the constraint.
NORM_FUSE (which relocates a square onto ScalarE via `activation_reduce`) plus ROPE_SKIP together only bought
-0.7%, consistent with the same effect. **Corollary (untested): fp32 relocations should pay, because VectorE has
no dtype advantage there — the head kernels' logits tensor is fp32 throughout (~2.1e9 of candidate passes).**

## The changes, and how each was verified

All are mathematically exact; every one was checked in the NKI simulator against a torch reference *before* going
near the device. Knobs live in `submission_raw/train3_s2.py` / `_s3.py` / `_s4.py`.

1. **`MLP_WG_GROUP`** (the win). `mlp_bwd` accumulated `dW1`/`dW2` with a `tensor_tensor` add per (token block,
   hidden tile); the PE can sum the four blocks in PSUM with `accumulate=(i>0)` and add once. Costs ~6 MB SBUF for
   the four buffered `dpn`/`hn` tiles (G=2 needs half and gets -3.8%, so fall back if SBUF is tight).
   Simulator: `dx`/`dW1`/`dW2` relative errors **identical** to baseline (4.01e-3 / 2.98e-3 / 3.42e-3).
2. **`ROPE_SKIP`**. `ROPE_FRAC=0.5` zeroes the upper half of `inv_freq`, so `cos == 1` and `sin == 0` on those
   columns exactly — the rotation there is the identity, yet the kernel did all 64 columns. Now it does the active
   32 and copies the passthrough spans. Simulator: **bit-identical** (max abs diff 0.000e+00), and the
   `cos==1 / sin==0` claim verified directly. Applied to the forward and the backward transpose.
3. **`NORM_FUSE`**. `tensor_tensor(sq, x, x, multiply)` + `tensor_reduce(ms, add, sq)` -> one
   `nisa.activation_reduce(dst=sq, op=nl.square, data=x, reduce_op=nl.add, reduce_res=ms)` at 17 norm sites.
   Simulator: squared tensor exact, row-sum differs by 3e-5 on a sum of ~257 (1.2e-7 relative, accumulation order).
4. **`MLP_STORE_H=3` (store relu)** — the salvage of the failed idea. Store `relu(pre)` rather than `pre` or `h`:
   the backward then gets relu for free and computes `h = relu*relu` with one VectorE pass, deleting the `max`
   pass with **no new ScalarE op** (which is what killed level 2). Simulator: saved tensor exactly `relu(preT)`,
   gradients identical.
5. **`STT_FUSE`**. The rmsnorm-backward tail `x*scalar` then `tensor - that` is one
   `scalar_tensor_tensor(..., reverse1=True)`. `reverse1` semantics verified in the simulator against
   `g - r*cc` (0.000e+00) — worth checking, since a reversed operand order here would be silent.
6. **`PREBWD_WG`**. The same PSUM grouping as (1), applied to the Q/K/V weight-gradient accumulation in
   `pre_attn_bwd_ko` (12 VectorE adds per token tile). Buffered via a small list so the reverse-order `dkprev`
   carry is untouched.

## Operational notes

- **`nl.sqrt`, `nl.square`, `nl.relu`, `nl.copy` all exist**; `nisa.activation_reduce` and
  `scalar_tensor_tensor(reverse0/reverse1)` exist on this box. `nl.sqrt` is already used 16x elsewhere in the repo.
- **A NKI kernel can be traced from a queued-but-unlaunched vehicle file**, so never edit a vehicle a queue still
  has a pending line for — hence the `train3_s2/_s3/_s4` chain rather than editing one file.
- `pkill -f 'trn2_queue...'` over ssh **self-matches the ssh command string** and kills your own command before it
  runs (lost one queue file this way). Always use a bracketed pattern: `pkill -f 'trn2_[q]ueue.sh queue_r4'`.
  Killing the queue runner does **not** kill a run it already launched (`trn2_run.sh` setsids it first).
- `G_WG` (the first gate of the -5.6% win) died at step ~35 with
  `NRT Execution error ... operation=nki_kernel_global` on rank 1 — the known intermittent sidecar/DGE fault, not
  the change (the identical config ran 600 clean steps in `H_WG`). Budget for a re-run when gating.
- Comm is **latency-bound, not bandwidth-bound**: `GATHER_BF16` was worth ~2 ms/step and `GRAD_BF16` was a wash
  (both already in the ledger). That also rules out the sparse-bigram-gradient idea, which only reduces bytes.

## Second round: measured increments, two tracer violations, and the scatter

| screen | stack | fwd_bwd | total | vs baseline 0.522 |
|---|---|---|---|---|
| `H_ALL` | WG_GROUP + NORM_FUSE + ROPE_SKIP | 0.388 | 0.489 | -6.3% |
| `H_S4b` | + STORE_H=3 (store relu) + STT_FUSE + PREBWD_WG | 0.383 | **0.485** | **-7.1%** |

So the second three changes are worth only **-0.8%** together, against the ~-3.8% I projected from element counts.
**Element-pass counting predicts the FIRST deletion well and later ones badly** — once the largest VectorE item is
gone the wall stops tracking VectorE, so each further deletion converts at a lower rate. Treat element counts as a
ranking device, never as a forecast.

### Two device-tracer violations, both silent in the simulator (now memory nki-tracer-rules #14 / #15)
- `for _i, _e in enumerate(buf):` -> **"error: expecting simple variable"**. The tracer wants simple loop targets:
  no tuple unpacking, no `enumerate`. Use `for _i in range(len(buf)):` over parallel lists. This construct was also
  in the generated v47, so the screen crash prevented a wasted 40-minute gate — screen before every gate.
- `nisa.activation(dst=u, data=u, ...)` (in place) -> **"error: argument 'x_in' not supplied"**, pointing
  misleadingly at the kernel's own `def` line. `tensor_scalar` tolerates in-place; `activation` does not. This
  killed the head-kernel fp32 relocation reserve, which is dropped (it was speculative).

### The embedding scatter: 21.3 ms/step, and it is launch-bound
`PROFILE2` puts `scatter` at 21.3 ms of the 425 ms `fwd_bwd` (4% of the step). `scatter_rounds`' own docstring says
it emits **~150 rounds for token ids**, and `EmbedSidecar.scatter` calls the kernel once per round *per table* —
**~150 x 4 token tables = ~600 launches per micro-step**, which at ~30 us/launch is essentially the whole 21.3 ms.
The four token tables (`wte`, `vte`, `vte_mid`, `vte2`) are gathered with the **same index plan**, so one launch can
serve all four and the perm/idx loads and clamps happen once instead of four times: `embed_scatter_add_v2_x4`
(knob `SCAT4`, `train3_s6.py`). Verified **bit-exact** on all four tables against both the per-table kernel and
`torch.index_add_`. ~600 launches -> ~150.

### The clean file is genuinely faster than the vehicle
v46: clean 0.976163 vs vehicle-at-v46-config 0.977482 (-0.0013). v47 shows the same: 0.44 s/step clean vs 0.47 s
for the identical changes in the vehicle. The knob branches and dead kernels in the vehicle are not free, so
**always gate the clean file before believing a vehicle number is the ceiling**. v47 cold compile 433 s (v46: 510 s),
well inside the 900 s excluded allowance.
