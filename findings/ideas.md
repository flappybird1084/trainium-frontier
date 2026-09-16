# Ideas — master catalog (consolidated 2026-09-16)

Single home for idea/backlog/research notes. **Supersedes and replaces** `arch-backlog-speedrun.md`,
`external-techniques.md`, `strange-wins-research.md`, `wild-ideas-2026-09-15.md`, and `ideas-2026-09-16.md`
(all folded in here; git keeps their history). Current lever *plans/status* (D/E/F, the OOB flake) live in
`levers-def-2026-09-16.md`; the data-axis corrections + self-distillation rationale in `repeat-regime.md`; the
full experiment ledger in `trn2-experiments.md` / `trn2_results.tsv` and the exhausted list in `plan-097.md`.

Nothing in the "untried" sections has been gated. Verdicts with numbers are measured.

---

## 0. How to read this

**Baseline:** v44 = 0.979132 local (~0.9857 portal), 0.52-0.53 s/step, ~3400 steps, depth 12 / width 512 / 4
heads. Leaders ~0.960 local. **The gap is a compute gap** — `repeat-regime.md` shows the corpus is not a wall
(~2.8x more tokens still converts), so step-time converts to bpb and is the highest-leverage axis; loss-side
architecture gives 0.002-0.008 apiece and stacks sublinearly. Both external streams found the frontier of this
exact benchmark is now systems/kernel work.

**The cost filter (pick against it):**
- Adding FLOPs on tensors we already materialize is ~free (256->512-wide cost ~0% step).
- Adding graph nodes / kernel launches is expensive — that is the actual bottleneck (launch/DMA-bound; on the
  compiled kernel line, custom-op Python bodies run eagerly at runtime, so extra elementwise ops are launch-bound).
- Adding new shapes is worst: a cold NEFF compile is 500-650 s (cap 900 s).

**Measurement reality (this binds harder than idea supply):**
- 1800 s same-config noise band **~0.0004** (paired, same seed+data order). Exchange rate **1% step ~= 0.0003 bpb**.
- **Every cheap screen has mispredicted the 1800 s gate in the same direction:** G_REINJ local -0.0019 -> box
  +0.0008; multi-hop local -0.0083 -> box +0.0074; v41 screen -0.004 -> full +0.009; 600 s over-favors high-LR.
  So sort ideas by **what it costs to kill them**, not by expected gain. A timing/numerics/statistics falsifier
  beats a gate. Verdicts below ~0.0008 need n~3 (1.5 h each).
- Judge only by **eval bpb** — training loss lies under any train/eval asymmetry (hops, EMA, aux losses).
- Eval-contract + NKI-tracer rules constrain what is even buildable — see `nki-kernel-playbook-2.md` and the
  memory `nki-tracer-rules` (strided DMAs on graph tensors NaN; a graph-produced gather index NaNs; module-level
  kernels only; never overwrite the vehicle mid-run).

---

## 1. Current top 5 (the live forward plan, Sep-16 research)

Ranked by leverage. **Live status / queue state for these five is tracked in `ideas-2026-09-16.md`** (kept as a
separate, actively-updated focus doc — as of last read: #5 TOP implemented + gated G_TOP05; #2/#3 in flight as
plan-097 levers F/D; #1 prototype delegated to the CPU simulator; #4 parked). D/E/F plans in
`levers-def-2026-09-16.md`.

1. **Intra-block NKI engine pipelining (step — flagship).** Software-pipeline tiles inside one fused block kernel
   (`affine_range` + manual SBUF) so a tile's matmul on TensorE overlaps the previous tile's softmax/norm/RoPE on
   Vector/Scalar — the 15% MFU / 30-60% engine occupancy ceiling. **Precedent: Anthropic's own NKI FlashAttention
   24->8.5 us (3x)**, bottlenecked by the same small-op launch latency. *New vs. closed:* our PARRES_T null
   ("nesting kernels gives no overlap") nested whole kernel *bodies*; this is tile-level, not falsified by it. AWS
   perf book **advises against monolithic megakernels** on Trn2 → target intra-kernel pipelining. Upside 20-30%
   step ~= 0.006-0.009 bpb-equiv. **De-risk free first on the Neuron 2.29 CPU simulator** before box-hours.
2. **Optimizer-in-backward overlap (step — lever F).** Apply the cheap AdamW/embed/scalar updates the moment each
   bucket's grad is ready, on Vector/Scalar during backward matmuls; park Muon's matmul-heavy orthogonalization in
   TensorE idle windows (it competes with backward for TensorE). Bounded to the ~10% optimizer cost. Fights ZeRO-1
   sharding. ~0.0015-0.0025 bpb-equiv.
3. **Cheap K/V-reuse multi-hop attention (bpb via step — lever D).** Reuse K/V in the extra hop, recompute only
   Q+attention+out-proj; cut the hop from ~15% toward ~10% step (full hop lost: gate 0.9926 vs 0.9852; fixed-step
   was -0.0083). Needs the hop under ~8% to net -0.002..-0.003. Fused op exists. Use EMA_BLEND <=0.1 with hops.
4. **Cached-logit self-distillation on the repeat pass (bpb).** Train pass 2 against a blend of hard labels and
   our own cached top-k logits from pass 1 (k=1 already helps, ~50% blend, skip the soft target on the lowest-
   entropy ~15% of tokens). ~0 extra forward cost; reuses the embedding-sidecar plumbing. *Honest flag:* replicated
   wins are mostly diversity/pass@k not single-pass loss, and we are undertrained (regularizers help least here).
   The "only survivor" of `repeat-regime.md`. **PARKED on cost (2026-09-16):** the fused NKI head never
   materializes logits, so both the pass-1 top-k cache and the pass-2 soft loss need the logits path (+~10% step),
   and a per-pass graph switch would recompile mid-run — so it must be ONE graph with the soft loss always present
   = +10-12% step = +0.003-0.0035 bpb to recover before any gain, vs a mostly-diversity literature gain in an
   undertrained regime. Build only if the lottery ticket is wanted anyway.
5. **Token-Order-Prediction (TOP) aux head (bpb, eval-pure).** Extra linear head trained with a ranking loss over
   how soon each token reappears; dropped at eval. Reported to beat NTP and MTP at a fraction of MTP's cost.
   *Honest flag:* our MTP lost +0.040 and small models benefit less from aux heads — real risk it hurts; but it is
   a genuinely different objective and cheap to gate.

---

## 2. Untried backlog (deduped). Sorted by cost-to-kill.

### 2a. Falsifiable WITHOUT a training run — do these first
- **Joint (not just batched) Muon orthogonalization.** The Muon kernel already batches the whole (n,out,in) stack
  in one launch (so "12 launches -> 1" is done). *Joint* orthogonalization of the concatenated matrix is a
  different optimizer (decorrelates gradient directions across depth) → needs a gate, not free.
- **Stiefel retraction on W_O.** Every ~100 steps run a couple extra Newton-Schulz iters on W_O itself. **Kill by
  logging** the Frobenius distance of W_O from the nearest orthogonal matrix in an existing run; if already tiny,
  Muon's update-orthogonalization holds it, nothing to retract. ~0% amortized.
- **Mixture-of-softmaxes plumbing.** K=2-4 branches sharing lm_head, combined in prob space, returned as log(p).
  Numerics **verified OK** (fp32 log_softmax(log p)-log p = 9.5e-7; must return fp32; watch bf16 sum(p)~1). Head
  itself K=2 = +3-8% step; not built. (Also 4.4 below.)
- **DFA / forward-gradient learning (step-time via backward collapse).** Backward is 144 ms vs forward 60 ms. DFA
  = true grad only at the head, projected through frozen random matrices to give each layer a synthetic local
  grad; forward-gradient = K=2-4 duplicate forwards, no reverse pass. **Free dev-path falsifier:** run on CPU/CUDA
  a few hundred steps; if the loss curve plateaus far above backprop, kill before any NKI work. DFA's failure mode
  is depth; 12 layers is in range.
- **Pre-transposed backward** — CLOSED by profile read: the 182k transposes are per-step *activation* transposes,
  not reusable weight transposes → weight pre-transpose saves <1%. (kept here so it is not re-proposed.)

### 2b. Near-free bpb tweaks (one line / one op; each needs a gate)
- **PPM / Witten-Bell closed-form mixing** *(archaeology top pick).* Offline (train data only) per-bigram-slot
  escape weight `w = distinct_cont/(distinct_cont+total)`, gathered alongside the bigram row and mixed into
  pre-softmax logits `+= w[slot]*(row @ Wproj)`. **Exact and correct from step 0 — zero bootstrap**, which is what
  the undertrained regime rewards; bpb is literally a codelength metric (PAQ/CMIX lineage). *Not* bigram
  re-injection (that asked the net to *learn* the trust weight; closed +0.0008/+0.0012). Falsifier: does it beat a
  single learned global scalar at equal step time? ~1-2%.
- **Prefix / backward token-prediction aux loss** (arch #7 / strange #1). Aux loss on the SAME logits predicting
  the token whose bytes are the longest proper prefix (or predicting earlier tokens). PR #337: -15 steps/-0.8 s at
  124M, speedrun-verified; stacks with MTP so our MTP failure does not rule it out. Needs a 2nd target gather in
  the fused head kernel. ~free.
- **Length-factored softmax** (4.3). Factor P(token)=P(byte-length bucket)*P(token|bucket); byte length is exactly
  what the metric weights by and nothing currently predicts it. Both softmaxes off the same hidden vector, no
  serial dep. Falsifier: log 4-way bucket accuracy in first 200 steps; >95% already => dead. ~2-3%.
- **Exact-marginal byte CE** (4.2). Marginalize the 8192-way dist to next byte (fixed [8192x256] matmul on probs),
  add `-log2(marginal[true_byte])`. Removes the spurious penalty when two tokenizations share a first byte.
  Falsifier: fraction of mass split across same-first-byte tokens in a trained ckpt; near 0 => dead. ~2%.
- **SOM vocab reorder + logit conv** (4.5). Offline SOM permutation of the 8192 vocab (cached like the tokenizer),
  baked into lm_head row order at init (zero runtime), then a small learned depthwise 1D conv across the reordered
  logits pre-softmax — one update corrects a neighborhood of confusable tokens. Falsifier: swap SOM for a random
  permutation; identical => gain is the conv's params, not topology. <1%.
- **Fast-weight token modulation** (3.3). `y_extra = (x . v_shared) * fast_u[token_id]` — multiplicative per-token
  capacity, no gating net to bootstrap. ~1%.
- **Exact-bpb loss reweighting** (1.1) — CLOSED on arithmetic: every token's nats count once, byte total is
  constant, so uniform CE already IS the metric. (kept so it is not re-proposed; devalues the byte-aware family's
  *reweighting* framing but not 4.2/4.3's marginalization framing.)

### 2c. Attention / architecture (eval-arch, gate each)
- **Knocking-Heads attention** (strange #14). One shared d_h x d_h matrix after per-head V, **diagonal-init**
  (random init destroys it); algebraically absorbable into W_v (fold `W_v_eff = W_v @ T_v` per opt step → ~0.0006%
  vs 1.8%/layer in activation space). Paper: -0.010/-0.020 at 100B. Muon orthogonalizes per-matrix grads → effect
  could amplify or vanish. One run to settle.
- **Trimmed MUDD / DenseFormer depth-mixing** (arch #12/#15). Rank-64 bottleneck emits per-token mixing weights
  over {x0, h_{L-4}, h_{L-2}} on the last two layers, W2+bias zero-init (no-op at step 0). Paper -65 steps
  (-3.0%), gated follow-on -4.3% more. **Slow-gate group mandatory** (dedicated lr-0.02; E28a burned exactly here).
- **Retie embedding<->lm_head for the first ~2/3, then untie** (strange #16). Retying alone -55 steps (~-2.6%),
  n=9, p=0.0002; zero forward cost, negative optimizer/comm (one fewer matrix to shard for 2/3 of the run). Our
  lm_head init std 0.001 vs wte 0.8 — test plain retying with a retuned init first.
- **SwiGLU / GLU-family re-screen at 512** (arch #21). Calibration paper flags it among the few *real* effects;
  lost under fixed-time at width 256. Worth exactly one re-screen at 512, then close the axis permanently.
- **Modern Hopfield global memory bank** (5.1). At 1-2 layers, M~128 learned key/value slots read by one softmax
  (`softmax(h@K^T/sqrt d)@V`), zero-scale-init — content-addressed (vs every ID-addressed table we have), legal
  (single-pass read, no energy settling). ~1-2%/layer, ~131k params. **Do NOT build the hash-the-hidden-state
  sibling** — a graph-produced gather index NaNs (rule #11); use the matmul form.
- **Pi-Sigma cross-term MLP branch** (5.2). `extra = (Wa x)*(Wb x)` in parallel with relu(W1 x)^2, Wb zero-init —
  ReLU^2 can't express a cross-term between two different directions (GLU is a restricted special case). ~3-5%,
  tunable by rank. Falsifier: Wb norm collapsing mid-run => model rejecting it.
- **Reservoir / echo-state multi-timescale memory** (5.3). Fixed log-spaced exponential decays (4/16/64/256
  tokens) over the value stream as a parallel cumsum (diagonal transition = no recurrence loop); train only a small
  readout. Matches "no time to learn slow dynamics." ~2-3%.
- **Ghost-basis / Hadamard-spectral parameterization** (5.4). `W = H diag(v) H` with fixed Hadamard H → 512
  learned params with an exactly-diagonal loss surface (AdamW on v = per-eigenmode Newton). As dense matmuls with
  the constant, NOT the O(d log d) butterfly (9 serial stages, fatal on a latency-bound machine). ~1-3%.
- **Paired-head attention** (arch #11, pure layout) and **cross-sparse attention** (arch #19, maps onto the same
  `bounds` mechanism as sliding-window). Untried, cheap.

### 2d. The 134M-param table — restructure the READ, never the CONTENT
**HDC binding closed catastrophically (+0.098):** the learned table *content* is load-bearing; a frozen algebraic
composite adds only +0.0007 over no bigram at all. So any idea that cuts per-row *learned* capacity carries a heavy
negative prior (this killed rank-1 memory 1.4 and downgrades trigram-permutation). Only *addressing/read* ideas
survive:
- **Multi-read addressing** (3.2) — BIGRAM_SPLIT=2 (independent hashes concatenated) was a WASH, which closes only
  *linear* decorrelation. Still open: **(a) Kanerva SDM sum** (K=4 hashes into one shared table, read=sum);
  **(b) median-of-K** (nonlinear collision *suppressor*, drops the bad colliding row — the variant most distinct
  from the closed split); **(c) bit-flip neighborhood** (read h, h^1, h^2 with learned per-offset scalars so a
  novel bigram borrows partially-trained neighbors). ~2-8% (extra gathers cheap, scatter-adds are the cost).
- **Key-embedding table** (arch #23) — the ve trick on keys, same family as our most consistent wins. Untried.
- **Sparse / low-rank value embeddings** (arch #13) — we carry 3 dense 8192x512 ve tables; the speedrun found dense
  ve overparameterized; sparsify or restrict to a layer subset and respend the params.

### 2e. Schedule / LR constants (partially screened; interactions untested)
- **Split cooldown fractions** (strange #18) — aux-Adam 0.4 / matrix 0.8 (Track-3 -4.5%, n=10). Our cooldown is a
  single 0.90. We screened FINAL_LR 0.0 (wash) / 0.10 (worse) at 0.90 but these interact.
- **Muon momentum warmup** 0.85->0.95 over ~300 steps then down over the last ~50 (ours fixed 0.95).
- **Per-group LR: attention-vs-MLP split + upper-layer QK damping** (strange #19). The muP gap concentrates in
  embedding LR early (helps more at short horizons); "Learning Less Is More" multiplies upper-half W_Q/W_K LR by
  0.25 for the first ~3-12% (PPL 26.80->26.30); Prime Intellect landed attn-LR 0.62x vs MLP 1.0x. We swept
  EMBED_LR (4.5 optimum) — the **untested part is the attention-vs-MLP split and upper-QK damping**.
- **Drop global gradient clipping** (strange #21) — absent in both speedrun codebases at 124M-560M with QK-norm +
  Muon; removes an all-reduce under ZeRO-1. Low risk, cost-negative. (Counter-evidence ZClip is 1B/50B w/o QK-norm.)
- **Learnable per-head QK gain** (arch #17 / E20) — restores attention temperature under QK-norm; init 1.0
  transparent, slow-gate group mandatory. *Caution:* the learnable version at 124M "pushed every head back to ~1.0
  within 200 steps," so the **constant ATTN_SCALE=0.1 already in v44 is the safer form.** Adam-alternate-steps is
  CLOSED (worse — halving the x4.5-LR embed updates hurts).

### 2f. Needs-adaptation / high-effort (discount — none has landed in the speedrun in 2 yr)
- **Variable-width "bowtie" transformer** (strange #22). Vary d per layer to a bottleneck at 0.75L; at 200M loss
  3.452->3.430 while cutting FLOPs 3.2%. Kernels for many shapes + a NEFF per width — expensive on our stack.
- **Primer MDHA / static short conv (Canon-B)** (strange #11). Width-3/4 causal depthwise conv on Q/K/V post-proj,
  identity-init; -2.4% PPL for +0.07% params but 5.8-12% GPU wall-clock = an NKI kernel problem. Mask at BOS.
- **DeepSeek Engram** (compressed tokenizer + multi-head hashing w/ distinct primes + context gating + ShortConv,
  injected at 2 layers). Would replace the input bigram (~67M table params). *Caution:* our bigram-hash family —
  every resize/re-hash/decorrelation — lost; Engram is the higher-effort multi-gram version.

---

## 3. CLOSED — do not re-pitch (measured, with numbers)

Full ledger in `trn2-experiments.md`; exhausted list in `plan-097.md`. Consolidated here so ideas above are not
re-litigated.

**Step-time:** compiler flags (9 configs, all 0.52-0.53 s; the "corrupt eval" was the cold-start EMA bug, retracted
— see `ema-cold-start-bug.md`); fp8 (x1.0-1.04 HBM-bound, x1.3-1.4 PE-bound MLP → ~6 ms, step isn't matmul-bound;
won't build in torch.compile); width 640 (+28% step, kernels cap free-dim 512); comm flattening (worse), async
reduce-scatter (no gain), bf16 grad collectives (+0.0006), compiled optimizer (+3%); own flash-attn (strided-DMA
loads NaN in-graph, rule #11; per-tile-load +4%; glue-hoist = parity); nested vendor attention in another kernel
(bit-exact, NO overlap — scheduler serializes bodies); Local-SGD/stale-sync (OPT_SHARD=0 is +8%; SYNC_EVERY 4/8 no
throughput gain); pre-transposed backward (<1%); MLP recompute / LOADER_DEV=cpu / static DMA unroll / packed
layer-op outputs (wash or -2%); tinygrad/fusion search (no Neuron backend; _HVC=256 fails tracer).

**Data / packing:** stream packer 100% coverage +0.0017 (integrity, not coverage, is the mechanism); PACK=8 wash;
PACK_MODE=bfd +0.0025; PACK_ROLL +0.0006; PACK_MAXDOC=2048 +0.0033, 4097 wash; second-pass repack +0.0003 —
**best-fit PACK=4 is the optimum of everything tried.** EPOCH2=rotate +0.0176 (memorization).

**Regularization (thesis falsified — undertrained not overfit at 1.4 epochs):** WD 0.2 +0.0045, WD 0.3 +0.0109;
MIR 0.05 +0.0050, 0.10 +0.0096 (also +0.014 fixed-step); parallel residuals +0.005 at equal steps. Do not
re-litigate WD / input masking below ~3 epochs.

**Attention / arch:** multi-hop (K/V recomputed) full gate +0.0074; prehead denoise +0.006; MTP +0.040; per-layer
embeddings +0.016; XSA torch glue 2x step; value-residual +0.0010; backout +0.0013; smear gate +0.0032; smear +
drop-attn0 +0.0032; merged QKV +0.0005 eager / +0.010 compile; trigram +0.0035/+0.0050/wash; narrow QK (64) breaks
128-wide kernels (+0.020); SDPA +0.0011; context 2048 +0.0039; window-256-on-odd-layers -0.0017 (free, only backlog
winner) but +0.0007 on compile line, not adopted; window warmup adds nothing; depth 14 = OOB flake (2 deaths).
DeltaNet/GDN not viable (device NaN, eager 19 s/step); flow/rectified-diffusion parked (wash/hurts); MoD +0.0267;
latent scratchpad +0.0022; stack recurrence +0.0011; output-side logit flows +0.0025/+0.0045.

**Bigram table (68% of params — all resizes/re-hashes lost):** 2^19 worse (+0.0009, OOMs LNC=1); sign trick wash;
prime/non-power-of-2 modulus worse; SPLIT=2 wash; positional keys +0.0182; FiLM combine +0.0098; **HDC binding
+0.098**; bigram re-injection +0.0008/+0.0012; rank-1 memory (dead by HDC).

**Scalars/schedule (on PACK=4 base):** HEAD_LR 0.7 +0.0011 / 1.4 +0.0014; SCALAR_LR 1.4 +0.0055; WARMUP 100
+0.0021; ATTN_SCALE 0.08 +0.0025 / 0.125 +0.0001 (0.1 optimal); EMA 0.99 +0.0003 / **0.997 -0.0006 (v44)** / 0.9985
equal; WARMDOWN 0.7/0.8, MATRIX_LR 0.55/0.85, EMA blend 0.5, EMA start 0.75 all flat; EMBED_LR 3.5 +0.0039 / **4.5
optimal** / 6 +0.0011 / 8 +0.0008; EMA-Nesterov step-corrected wash; cautious-WD extension to AdamW groups
negligible; canonical/BPE-impossible masking = 2.2% density, eval-only -0.0007 (one band) + changes the scored
forward → parked; exact-bpb reweighting killed on arithmetic.

**Optimizer family (ruled out by regime — Muon is best at our ~1.4-epoch data ratio, "Fantastic Pretraining
Optimizers"):** nGPT (1.29x, +9% runtime); DynamicTanh (worse at T/P~8.4, NaN); Hyperball/MuonH; PSGD-Kron;
Shampoo; AdEMAMix/Schedule-Free/AdaMuon/Dion (need >1500 steps to warm up); SOAP/Kron only overtake Muon at >=8x
Chinchilla; NorMuon/Gluon marginal; differential/sigmoid/softpick attention; hyper-connections; MLA/GQA/MoE; label
smoothing (raises CE=raises bpb); z-loss (softcap present); mimetic V-O init worse; SkyLadder window warmup (gains
muted at seq 1024 + our "window warmup adds nothing"); Differential/Forgetting Transformer (lose on wallclock w/o a
fused kernel).

**Illegal under our rules (never pitch):** online n-gram tilt from the val stream; test-time training / LoRA-TTT;
eval-time online adaptation / EVAL_SEQ_LEN extension / reading val; quantization chain (serves a 16 MB cap, no
analog); baking a training-data-derived table into the source file (a reviewer calls it a pre-trained weight —
compute it in-process each run).

---

## 4. Methodological warnings (worth more than several items above)

1. **Optimizer/schedule rankings flip mid-training** because of LR decay (arXiv 2509.02046). A compressed-schedule
   5-min proxy mis-ranks every schedule/optimizer item and specifically PENALIZES tail-EMA (pays only in the tail).
2. **Single-LR ablations measure "did this break my tuned LR," not "is this better."** Most likely false negatives
   among our failures: value-residual and per-channel gates (both change effective update scale). The speedrun's
   per-channel gate reads only the first 12 residual dims in the *Adam* group; widening it measured worse — a
   gate-input-width failure, not a gating failure.
3. **The only public benchmark in our regime (modded-nanogpt) has never had a famous architecture paper land.**
   What lands is near-zero-FLOP, shape-preserving, zero-init-transparent tweaks. Track 3 (fixed arch, minimize
   steps, n=4-30 seeds, 0.0045 loss ~ 100 steps) is the most in-distribution evidence source.

---

## 5. Sources
- modded-nanogpt record history: https://github.com/KellerJordan/modded-nanogpt · Track 3 / WR analysis
  https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg · worklog https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/
  · WR#82 https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html
- Parameter Golf: https://openai.com/index/what-parameter-golf-taught-us/ · arXiv 2607.01517
- NKI step-time: perf book https://pierredemalliard.com/neuron-performance-book/part5/ch16-advanced-nki.html · perf
  guide https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/nki_perf_guide.html · Neuron 2.29 + CPU sim
  https://aws.amazon.com/about-aws/whats-new/2026/04/announcing-neuron-2-29/ · fused self-attn
  https://awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/general/nki/tutorials/fused-self-attn.html
- Optimizer-in-backward: https://github.com/pytorch/pytorch/issues/67570
- Distilled Pretraining https://arxiv.org/html/2509.01649v1 · Born-Again saturation https://arxiv.org/pdf/2407.04600
  · Fantastic Pretraining Optimizers https://arxiv.org/html/2509.02046v2
- TOP https://arxiv.org/html/2508.19228v2 · SkyLadder https://arxiv.org/html/2503.15450v1 · Rho-1
  https://arxiv.org/html/2404.07965v3 · value embeddings https://snimu.github.io/2025/10/07/modded-nanogpt-value-embeddings.html
- Idea-specific: tail-EMA arXiv 2602.02522 · cautious WD 2510.12402 · EMA-Nesterov 2605.25395 · half-RoPE + key
  offset PR#169 / 2502.14837 · XSA 2603.09078 · Knocking-Heads 2510.23052 · MUDD 2502.12170 · Primer 2109.08668 /
  2606.03825 · Hopfield (Ramsauer 2020) · variable-width 2606.18246 · DeepSeek Engram
  https://github.com/deepseek-ai/Engram · Learning-Less-Is-More 2605.10504 · muP embedding-LR 2605.21486
