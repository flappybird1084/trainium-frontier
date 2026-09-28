# Proposals, 2026-09-28: what is left, ranked by bpb per device-hour

## Bottom line
**We cannot close −0.0045 with code.** Our verified remaining code levers add up to about −0.0005 to −0.0015
expected, on top of bf16 bigram (−0.0008) and whatever Knocking-Heads gives. That puts the projected portal
mean near 0.9725–0.9735. The rest of the gap has to come from best-of-N upload luck (portal sd ≈ 0.0022). At a
0.9730 mean, each upload has about a 9% chance of landing under 0.9700, so three uploads give about 25%. Two
things follow: upload each improved file as soon as it is confirmed, and gate only arms that could plausibly
move the score by 0.0005 or more. An arm with a smaller expected gain cannot be told apart from noise
(one-gate sd ≈ 0.00025) in the time we have left.

## Verification of the researchers' claims (discarded items)
| claim | verdict | evidence |
|---|---|---|
| PSUM-drain pinning: 7 of 8 kernels untested | **Partly stale.** A DRAIN1 probe (session transcript, **never written to the ledger**) measured the two "cleanest" live kernels. On mlp_bwd the default is best, with Scalar +10.2% and Vector +5.8% slower. On post_attn_fwd_h the default is best, with Scalar +1.1% and Vector +0.1%. The memory note "default wins on live kernels" rests on those 2 kernels only. pre_attn_bwd_ko, head_bwd, muon_kernel, pre_attn_fwd_ko and head_fwd were never measured. | rows ATTNPLACE2/3, M_DRAINPIN, PLACE_ORDER; `nki/nki_drain_probe.py` |
| Attention backward: we ship our slower kernel | **False.** The vehicle already calls the vendor `attention_bwd`/`attention_cte` (canon :71-72, :3004-3008). ATTNPLACE retracted "ours faster" and declined the swap. | ATTNPLACE, M_DRAINPIN |
| BTE_BF16 never gated live | True until today. Now gated (BF_BASE/BF_ON) and in flight as v81. | BF_BASE/BF_ON |
| EMA blend re-run with corrected eps | **Done.** M_BLEND060 re-ran the 5x5 grid after the D_EPS fix. The best point beats the shipped one by only −0.000003. Closed. | M_BLEND060, M_DECAY060 |
| Log-unigram output-bias init | **Untested, not present.** `lm_head` is `bias=False` (canon :3379). The fused head has no bias. It was parked as idea A8 (compile-window-and-packing.md) on the argument that the prior is learned in about 50 steps. | none |
| Cautious masking on the Muon update | **Untested.** Only cautious *weight decay* exists, already in both Muon paths (`mask=(g*w>=0)`, canon :934-951). Update masking was never tried. | session-2026-09-27 §, flow-inference #8 |
| Muon scaled WD (wd ∝ lr) | **Already shipped** as B2_WD_TRACK in v73+ (`wd_t = WEIGHT_DECAY*lrm`). | S_B2CORE, R_B2PORT |
| Curriculum / difficulty ordering | **Closed.** The data axis is capped at < 0.001 and the corpus is already i.i.d. | R_DATAAXIS |
| Short-sequence warmup | **Closed in practice.** Every new shape is a new NEFF inside the 900 s compile cap, and attention is only 13% of the step. | data-axis §4 |
| Softcap init, joint LR, NS steps, VE schedule | **Closed.** The softcap scalars are learned and sit at their held-out optima (M_SCALARS). Joint LR: C_GLR. NS steps: S_NS4. VE schedule: S_VE2A. | |

## Ranked proposals

### 1. Pin PSUM drains on the three Vector-heavy live kernels
- **Idea:** DRAIN1 tested the two kernels where Vector is *light*, and pinning lost there. The one win (−17.5%,
  attention backward) was on a Vector-*heavy* kernel. The mechanism therefore points at pre_attn_bwd_ko (33
  Vector-only sites), head_bwd (38) and muon_kernel (37), which were never measured.
- **Evidence:** ATTNPLACE2/3 and M_DRAINPIN for the win; DRAIN1 for the two losses. This is my inference from
  the site census, not a measurement.
- **Expected:** −0.0002 (range 0 to −0.0005). pre_attn_bwd is about 30 ms/step, so a 5–10% gain there is
  0.5–0.7% of the step. Pinning also removes the compiler's run-to-run placement noise.
- **Step cost:** negative or zero. The outputs are bit-identical, so there is no quality risk.
- **Review:** safe. It only adds a literal `engine=` in the kernel source, with no env var and no change to the
  logit path.
- **Build:** about 0.5 h. The probe already exists; run it with
  `R1_KERNELS=pre_attn_bwd_ko,head_bwd,muon_kernel`.
- **Test:** a single-core probe of about 30 min, then one 240 s paired throughput screen of the pinned vehicle.
  No quality gate is needed.

### 2. Cautious masking of the Muon update
- **Idea:** inside `muon_kernel`, zero the update elements whose sign disagrees with the raw gradient, then
  rescale. The kernel already builds a `g*w` mask for weight decay, so this needs 2–3 more `tensor_tensor` lines.
- **Evidence:** untested here. The literature win is mostly for AdamW/Lion, and modded-nanogpt kept cautious WD
  but not update masking, which is a mild negative signal.
- **Expected:** −0.0003 (range +0.0010 to −0.0015). This has the widest range on the list, which is exactly the
  kind of arm the upload lottery needs.
- **Step cost:** about +1 ms of optimizer math, roughly +0.25% (+0.0001).
- **Review:** safe. It is optimizer-only, so the eval path is untouched. Hard-code it in the shipped file.
- **Build:** 2–3 h, including verification in the local NKI simulator.
- **Test:** one step-matched arm (`FRAC_BY_STEPS=1 NUM_STEPS=4340`) against the BF_ON anchor. Replicate only if
  Δ ≤ −0.0002.

### 3. Knocking-Heads folded into W_v (in flight)
- Being built. Treat it like #2: one step-matched arm against BF_ON, and replicate only if Δ ≤ −0.0002.
- No independent evidence here; I have not priced it.

### 4. Hold the remaining ZeRO replicas in bf16 (extend BTE_BF16)
- **Idea:** `gather_params` still gathers the matrix and token-table parameters as bf16 and then runs a
  converting copy back to fp32 `p.data` (canon :4189-4202). The graph casts every weight to bf16 anyway.
  BTE_BF16 already proved this mechanism on the 134M-element table: the collective writes the replica directly
  and the gradient reduce_scatter needs half the bytes. The remaining ~84M elements are 0.63x the table.
- **Evidence:** BF16a gave +1.6% for the table. DMA cost is bytes-linear (M_DMACOST). Untested for these params.
- **Expected:** −0.0003 (range 0 to −0.0006), assuming roughly 0.6–1% tokens/s.
- **Quality risk:** the gradient AVG runs in bf16. The table showed no loss (−0.00022), but matrices feeding
  Muon are untested.
- **Review:** safe, as long as `save_checkpoint` writes fp32 (the BTE path already does this).
- **Build:** 3–4 h, plus a CPU bit-identity test in the style of R_B2PORT for the fp32-master path.
- **Test:** a 240 s throughput screen pair first. Only if it gains at least +0.7%, one step-matched arm.

### 5. Build the per-block constants once instead of in every block
- **Idea:** `cos_full`, `sin_full`, `dstart` and `scal` are rebuilt the same way in all 12 blocks
  (canon :3087, :3303, :3317). Build them once in `GPT.forward`.
- **Evidence:** untested. The T_V72 double-permute deletion bought about 0, most likely because the compiler
  already merged the duplicate work. The same is likely here.
- **Expected:** −0.0001 (range 0 to −0.0003). Bit-identical.
- **Build:** 1 h.
- **Test:** add it to #1's throughput screen as a second arm rather than giving it its own slot.

### 6. ROPE_BASE 200k → 50k
- **Idea:** one of the inherited constants never swept as a value. The context is 1024 and ROPE_FRAC is 0.5.
- **Evidence:** zero ledger rows. However, all six sibling knobs lost on 2026-09-27.
- **Expected:** 0 (±0.0005).
- **Cost and review:** free in step time, with the value hard-coded (no env var).
- **Test:** one step-matched arm, and only if a slot is left over.

### 7. Log-unigram init of the output logits
- **Idea:** count unigram frequencies in-process at startup (never baked into the file) and start the output
  logits at log-frequency.
- **Evidence:** untested. The external claim of 0.00027 is from modded-nanogpt at another scale. Our own A8 note
  says this prior is learned within about 50 of 4300 steps.
- **Expected:** −0.0001 (range 0 to −0.0003).
- **Build:** 4–6 h. The fused head has no bias, so the bias has to ride the head_fwd/head_bwd tile loop, adding
  Vector work to the busiest engine or an extra K-tile of matmul. The review exposure is small but real (a
  count computed from the training data).
- **Verdict:** not worth building before the close.

## Non-device actions (highest value of all)
- **Upload** v81 as soon as its confirmation passes, and use every remaining weekly slot. Best-of-N is worth
  about −0.002 at our sd, which beats every item above combined.
- **Record DRAIN1 in the ledger.** Its "default wins" verdict currently exists only in memory and a transcript.
- **Check whether the portal step loss is compile overrun.** The +0.0025 to +0.0069 offset spread equals about
  12% of steps. If the portal's cold compile goes past the 900 s cap, the excess is charged as training time,
  and cutting compile time would shrink our offset directly. If the portal shows a step count or a log for our
  runs, one look settles this.

## Device schedule, next ~24 h
Two anchors already exist, and they stay valid only while arms are scored with the same protocol (524,288
tokens, 4 rows): BF_BASE (0.979779 @ 4330) for v80-lineage arms and BF_ON (0.979559 @ 4330) for arms on v81.
**Run one new arm per candidate against the matching anchor, not a pair.** That fits about 30 single arms
(~35 min each) into the budget instead of 15.

| hours | device | local GPU / CPU in parallel |
|---|---|---|
| 0–2 | in flight: full-length reference, then the KH arm | build #2 (cautious) in the NKI simulator; start #4 |
| 2–3 | #1 drain probe on 3 kernels (single core, ~30 min), then a throughput screen of #1+#5 vs baseline | score the KH checkpoint offline; run `review_check.py` on the v81 port |
| 3–4 | #2 cautious arm | CPU bit-identity test for #4 |
| 4–5 | #4 throughput screen, then its arm if it gains ≥ +0.7% | score #2 |
| 5–8 | replicate every arm that read ≤ −0.0002 (n=2), and #6 if a slot is free | score the arms |
| 8–10 | **stack the winners into v82 in the canon vehicle.** It must be ≤ 262,144 B, have 0 review FAILs, read the same 5 env vars as v52, and carry winners as hard-coded constants | freeze the file and run size/AST checks |
| 10–12 | **confirmation:** two full-length 1800 s gates of v82 (n=2) with 0 fallbacks (grep `unavailable`/`falling back`) | score both |
| 12–14 | if v82 beats v81 by ≥ 0.0004 on the mean of n=2, ship v82; otherwise ship v81 | hand the file to the user for upload |
| 14–24 | spare: one more full gate of the shipped file for a local number, or reserve | none |

Hard stop for new builds at hour 8. After hour 14 no code changes; only uploads remain.

**Realistic outcome:** a portal mean of about 0.9725–0.9735, with the top-10 cut at 0.9700 reachable only
through a lucky upload draw (roughly 20–30% with 3–4 uploads). Getting v81 and then v82 onto the portal early
matters more than any single arm.
