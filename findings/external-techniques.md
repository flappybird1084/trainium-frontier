# Techniques mined from external submissions

## Calibration: "Parameter Golf: What Really Works?" (arXiv 2607.01517, fetched 2026-09-02)
Controlled re-ablation of the parameter-golf leaderboard: **real** = GLU-family activations, RoPE, careful LR scheduling,
data ordering/shuffling. **Noise under control** = most optimizer modifications, init schemes, SWA-family averaging, many
community "speedrun" tricks. Consequence for our fixed-time queue: SwiGLU + packing/data-order rank up; MIN_LR /
Muon-momentum / init probes rank down. XSA (researched separately) = "exclusive self-attention", a parameter-budget
reallocation for their 16MB cap — not time-relevant for us.

## parameter-golf PR #2135 "Calib32 Token-Only N-gram + AsymLogit Stack" (noted 2026-09-01)
Source: github.com/codemath3000/parameter-golf, branch `submission/pr2130-calib32` (a different competition: bpb under a
16MB model-size cap, 10-min budget; their rules explicitly allow eval-time online adaptation — ours do not). Three items,
all since built on Trn2: **AsymLogit softcap** (two learned scalars `softcap_pos`/`softcap_neg`,
`where(l>=0, sp*tanh(l/sp), sn*tanh(l/sn))` on the logits; init to the current softcap -> transparent at step 0; in our
stack); **SmearGate** (PR #1667, modded-nanogpt lineage: `x_t += lam * sigmoid(W x_t[:gate_window]) * x_{t-1}` on the
embedding lane, W zero-init and lam=0 -> transparent; closed +0.0032); **within-row in-context n-gram boost** (boost the
logit of the token that followed the most recent match of the current context within the same row's strict prefix, learned
scalar; stateless per-row -> legal; closed, +0.0008/+0.0012).

## DeepSeek Engram (researched 2026-09-01; sources: github.com/deepseek-ai/Engram `engram_demo_v1.py`, arXiv 2601.07372 "Conditional Memory via Scalable Lookup")
Official demo mechanics, verified from code (not memory):
- **Compressed tokenizer**: vocab ids are remapped so tokens with identical *normalized* text (NFKC/NFD, strip accents,
  lowercase, whitespace collapse) share one id; n-grams hash over this compressed space.
- **Multi-head hashing**: per order n∈{2,3}, hash = XOR over the n compressed tokens of (token_k × odd random
  multiplier_k); each of 8 heads takes `hash % (distinct prime near table size)` into its own slice of one big embedding
  (head dim 64 = 512/8). Distinct primes + multiple heads average out collision noise — likely the fix for our E13
  regression (single-hash big table).
- **Context-aware gating**: gate = sigmoid(signed-sqrt(rms(key_proj(engram_emb)) · rms(hidden) / √d)) — the hidden state
  decides per-position whether to trust the retrieved memory. Value = gate × value_proj(emb).
- **ShortConv**: depthwise causal conv (kernel 4, dilation = max n-gram order) over the value stream, output =
  value + conv(value).
- **Injection**: additive into the residual stream at an early layer and a middle layer (their [1, 15] of 30), before
  attention. Per-layer independent multipliers → different hashes per layer.
- Demo config is V3-scale (~646k rows/head); our port would scale to prime≈65536 per head, 8 heads, orders {2,3}, layers
  (1, 5) of 10 → ~67M table params, replacing the E12 input bigram (whose order-2 signal it subsumes). **Not yet tried.**

## parameter-golf PR #1953 (openai/parameter-golf, noted 2026-09-01)
- **Learnable per-head QK gain** (their `QK_GAIN_INIT` lever): per-head learned scalar multiplying the normed query —
  restores learnable attention temperature under QK-norm. Init 1.0 → transparent at step 0; their data says init slightly
  above the neutral value helped, so sweep init if the learnable version wins. Queued as E20.
- **BOS-fixed SmearGate**: the naive smear (`x_t += gate·x_{t-1}`) leaks across packed-document boundaries; mask the
  smear where position t starts a new doc. (Folded into the smear experiment, which closed.)
- **NOT portable — do not implement for the Trainium competition:**
  - Their headline **online n-gram tilt**: builds n-gram statistics from the *validation stream during scoring* and
    adjusts per-token NLL post-hoc (`ptl - beta*is_hit + log Z`). Illegal here on three independent grounds:
    `evaluate_bpb()` is fixed (no loss-side hooks); the `load_for_eval` contract requires a stateless forward (no
    cross-batch state); the rules ban "reading, mirroring, or inferring the validation data used for scoring."
  - **Test-time training (LoRA TTT in any form)** — `load_for_eval` requires a frozen, stateless forward.
  - **EVAL_SEQ_LEN extension** — the official scorer controls eval row length and `evaluate_bpb` is fixed.
  - **LQER/GPTQ/AWQ quantization chain** — serves their 16MB artifact cap; no analog here.
  - **"Progressive context growth"** (their PR #2014) is a *training-time* context schedule and could be legal here, but
    it changes tokens-per-step accounting — parked unless the arch ladder stalls.

---
A filtered, prioritized backlog built from these notes plus the modded-nanogpt record history lives in
`arch-backlog-speedrun.md` (compiled 2026-09-06); the newer 2026-09-08 sweep is `strange-wins-research.md`.
