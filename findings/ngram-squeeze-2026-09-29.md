# N-gram logit squeeze beyond v91/v92: offline measurement (2026-09-29)

**Bottom line.** The best change that needs no kernel work and fits in the byte budget (805 B, of the 828 B available)
gains **-0.00064 on the v91 table set** and **-0.00091 on the v92 table set**. A version that ignores the byte limit
reaches -0.00066 / -0.00101. Changing the kernel to expose the max logit and the entropy would add only about -0.00015
to -0.0002 more. Per-bucket temperature gains nothing (about -0.00003). The -0.0015 target was **not found**. With v92,
the local gain over v88b comes to about -0.0030, which gives a portal estimate of about 0.9645 (0.9641 with K=2), not 0.9637.

## Method

- The pipeline is the previous study's (`n2/`): V88b weights with the n-gram term off, and per-order stats counted
  from the train tables. θ is fitted by L-BFGS on the **2M fit (train) tokens only** and scored once on the 2M val
  tokens. It reproduces v91 = 0.9595857 exactly at the old solver tolerance.
- All candidates were fitted with a tighter L-BFGS tolerance (so that weak features actually move). At that tolerance
  the unchanged v91 base scores **0.9596324** on val and v92 scores **0.9589440**. **Gains below are measured against
  these same-tolerance bases.** Solver noise is about ±0.00005.
- Model-state features (max log-prob, runner-up, entropy, and logsumexp at scales 0.70..1.30) came from one extra
  V88b forward pass, stored as `n2/q_ms_{v,f}.npy`.
- Scripts: `n2/q_lib.py` (general fitter), `q_feat.py`, `q_run1..10.py`, `q_sel.py`, `q_port*.py`, `q_patch*.py`,
  `q_smoke.py`. Results are in `n2/q_res.jsonl`.

## Ranked results (gain in val bpb against the same-tolerance base)

| # | Candidate | v91 set | v92 set | Code bytes | Kernel change | Extra scalars |
|---|---|---|---|---|---|---|
| A* | Uncapped full spec: record code (streak 0-3 × row log-odds bin) + agreement code + per-column p-power `a` | -0.00066 | **-0.00101** | 1,176 | no | 3×1,393 (+1,365 rows of `a`) |
| **A** | **Spec below (805 B)**: record code [prev-tau-right × row log-odds bin] (8) + agreement code [share-weighted agree bin × disagree count] (12) + per-column `a` | **-0.00064** | **-0.00091** | **805** | **no** | θ 2×1,365 → 3×1,385 |
| B | Agreement code + per-column `a` only | -0.00052 | -0.00071 | 544 | no | 3×1,377 |
| C | p-power only: t2 → t2 + a·log p (per-column `a`; one global scalar is as good) | -0.00026 | -0.00029 | ~250 | no | 1,365 (or 1) |
| – | A + max-logit gap (o × gap bin + slope) | -0.00081 | -0.00113 | +kernel | **yes** (head_fwd must output the running max) | +56 |
| – | A + gap + entropy + max-lp | -0.00086 | – | +kernel | **yes** (max + Σp·z accumulator) | +70 |
| – | Continuous-feature version of A (exact shares, log count; needs a new table byte) | -0.00078 | – | table change | no (table format) | ~100 |
| K2 | Second token of longest match, as a 2nd biased token (other agent's lever) | -0.00048* | – | – | **yes (K=2)** | +910 |
| K2' | Next-shorter order's distinct top token as the 2nd token | -0.00015* | – | – | yes (K=2) | +910 |
| – | Order selection (use a shorter order when its fit-row precision is higher by ≥0.2) | -0.00014 | – | ~300 | no | 0 |
| – | Temperature on all logits at matched positions (per order, or order × count bin) | ≈-0.00003 (from gradient and curvature at s=1) | – | – | **yes** | 7-91 |
| – | Position in row or in document | +0.00005 | – | – | no | – |

\* The K2 rows were fitted by the previous study's `pmix3` at the loose tolerance, so they are measured against
0.9595857, not the tighter base.

**Single features on the v91 set** (each added alone, as per-order slopes unless noted):

| Feature | Gain |
|---|---|
| p-power (t2 slope on log p_tau) | -0.00032 (in t1 only: -0.00008; in both t1 and t2: -0.00029) |
| Share-weighted agreement of shorter orders | -0.00019 |
| In-row streak of correct taus | -0.00015 |
| Disagreeing shorter orders | -0.00010 |
| Previous tau right | -0.00011 |
| Row log-odds | -0.00008 |
| log count, continuous | -0.00007 |
| Share, continuous | -0.00006 |
| Uncapped agreement count | -0.00006 |
| Runner-up share (c2/tot) | +0.00002 |
| Kernel: max-logit gap | -0.00039 |
| Kernel: entropy | -0.00034 |
| Kernel: max log-prob | -0.00029 |
| Kernel: top-1 minus top-2 gap | -0.00018 |
| Kernel: "tau is the argmax" | -0.00006 |

## Answers to the five questions

1. **Per-order p-mix and richer mix features.** Per-(order, count-bin) mixing is already in the bucket. The one
   functional-form gain is making the mix weight a power of p: b = t1 + log(1 + e^t2 · p^(a-1)), about -0.0003. The
   continuous count and share features are each worth under 0.0001, and they would need a new table byte. The number
   of distinct continuations is not in the table; the runner-up share, its proxy, is worth nothing.
2. **Blending two orders.** Adding a same-token combined score (share-weighted agreement plus a disagreement count, as
   a 12-cell code) is worth -0.0002 alone and about -0.00026 inside A. The K=2 second token is -0.00048 for
   second-of-longest and only -0.00015 for the next order's distinct top. The other agent should use second-of-longest.
3. **Model-state features at the head.** The logit gap to the model's own top-1 is the best single feature
   (-0.00039). On top of A, though, it adds only -0.00015 (v91) or -0.00013 (v92), and it needs head_fwd to output
   the running max. It is not worth a kernel change now.
4. **Temperature at matched positions.** The model is temperature-calibrated at matched positions. The loss gradient
   at s=1 is about 1e-3 against a curvature of about 0.3-0.5, which gives ≤0.00003 total. It is not worth doing.
5. **Other ideas.** The in-row n-gram track record (was tau right at t-1, and binned log-odds over the row so far)
   gives -0.0001 to -0.00015, and it is in A. A confidence gate is subsumed by the p-power term and the gap feature.
   Order selection by precision gives -0.00014. Position features give nothing.

## Spec for option A (port to `submission_raw/submission_v91_ng12.py`; +805 bytes, kernel unchanged)

The exact diff is in `ngram_squeeze_v91.diff` next to this file. `ngram_squeeze_patch.py` produces the patched copy
from the original, and it is the file I measured and smoke-tested (`n2/q_v91_squeeze4.py`). Changes:

1. **Constant:** add `NG_K = NG_NB + 20` after `NG_NB`.
2. **`ng_logit(bias, lpt)`:** use `bias[..., 0] + torch.log1p(torch.exp(bias[..., 1] + (bias[..., 2] - 1) * lpt))`,
   with bias = (t1, t2, a). When a = 0 this is exactly v91.
3. **`NgramTable.features`:**
   - Return `(4, b, t)`, with `out[1:] = -1`.
   - In the final agreement loop, with `a = (oi < top) & (tk[oi] == tau)`:
     - accumulate `ag += a * (by[oi] % 5 * .2 + .2)`, where the share-bin proxy is .2..1.0;
     - accumulate `dis += (oi < top) & (tk[oi] >= 0) & ~a`.
   - Set `out[3].flat = where(on, NG_NB + 8 + digitize(ag, (.5, 1.5, 2.5)) * 3 + min(dis, 2), -1)`.
   - Record code:
     - `pc[:, 1:] = on[:, :-1] & (out[0][:, :-1] == x[:, 1:])`, so x[t] shows whether the previous position's tau was right;
     - `c = pc.cumsum(1)`;
     - `out[2] = where(on, NG_NB + pc*4 + digitize(log((c+.5)/(pm.cumsum(1)-c+.5)), (-1,0,1)), -1)`, where pm = `on`
       shifted right by one.
   - `pc` **must be an int array**: bool + bool silently ORs in numpy (I caught this in the port check).
   - The docstring now says position t reads x[r, :t+1], which is still causal and row-independent.
4. **θ:** `nn.Parameter(torch.zeros(3, NG_K))`, initialised with `zero_()[1, :NG_NB] = -2.0`. The code columns must
   start at 0, not -2, so step 0 is exactly v91.
5. **`ngram_bias(ng)`:** takes `ng[1:]`, meaning the bucket and two codes.
   `b = sum(i.unsqueeze(-1) == arange(NG_K) for i in ng).float() @ θ.T` (the same one-hot matmul, now with 3 hot
   entries), then `b - 1e4 * (ng[0] < 0)[..., None] * tensor([0., 1, 0])`. If the Neuron tracer dislikes the
   in-graph constant, use a `torch.stack` of the three columns instead (about +20 B).
6. **Fused path:** `bias.reshape(b * t, 3)`. In `forward`, use `self.ngram_bias(ng[1:])`.
7. **`check_fused_head`:** its synthetic bias gets a third column `u - .5`, which exercises `a`.
8. **Review notes:** the n-gram paragraph now describes the two codes and b = t1 + log(1 + e^t2 p_tau^(a-1)), with
   3 × 1,385 scalars. These notes are included in the 805 B.

**Checks already done:**
- The features tail, taken verbatim from the patched file and run on tk/by rebuilt from stats, reproduces the offline
  bucket and codes 100% on val and fit.
- The 0.9589902 (v91) and 0.9580308 (v92) numbers are fitted on exactly those ported codes.
- CPU smoke test of the patched module:
  - with θ columns ≥ NG_NB at 0 and a = 0, logits are bit-identical to v91;
  - no-match rows are unchanged;
  - all three θ rows receive gradient.

**Not done:** a device run, or a fused-kernel check on Trainium (the NKI kernel itself is untouched).

**Budget warning:** A leaves 23 B. With v92's constants plus the K=2 port, A will not fit unless something is cut.
Fallback options, in order:
- B (agreement only, 544 B): -0.00052 / -0.00071.
- C (p-power only, ~250 B): -0.00026 / -0.00029.
