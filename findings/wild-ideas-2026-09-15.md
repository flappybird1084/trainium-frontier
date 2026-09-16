# Wild ideas - 5-agent generation pass (2026-09-15)

52 raw ideas from 5 research agents (objective/tokenization, associative memory, hardware-shaped, parameterization,
architecture archaeology), deduped to 18. **None of these are results. Nothing here has been tested.**

Baseline for everything below: **v43 = 0.979941 local / 0.9859 portal**, 0.52 s/step, ~3400 steps.
Exchange rate 1% step ~ 0.0003 bpb. Same-config gate spread ~0.0004.

## The organizing principle

The binding constraint is not idea supply, it is **measurement**. Every cheap screen we have (local fixed-step,
600 s box) has now mispredicted the 1800 s gate in the same direction: G_REINJ2 local -0.0019 -> box +0.0008;
G_REINJ4 local -0.002 -> box +0.0012; multi-hop local -0.0083 -> box +0.0074; v41 screen -0.004 -> full +0.009.

So the ideas are sorted by **what it costs to kill them**, not by expected gain. An idea falsifiable by a timing
run, a numerics check, or a statistics pass is worth several that need a gate.

---

## TIER 0 - Falsifiable WITHOUT a training run

These can be killed (or promoted) by measurement that is not a gate. Do these first regardless of merit ranking.

### 0.1 Depth-joint / batched Muon
*(converged on independently by the parameterization and hardware agents)*
Stack all same-shaped weight matrices (all 12 `W_Q`, all 12 `W_down`, ...) into one `(12, d, d)` tensor and run
the existing Polar-Express Newton-Schulz kernel **once** instead of 12 times.
- **Two rationales:** (a) the device is launch-bound, so 12 small kernel launches cost more than one batched call
  at 12x the FLOPs (PE is only 31-34% occupied); (b) joint orthogonalization decorrelates gradient directions
  *across depth*, an implicit anti-redundancy constraint nothing currently enforces.
- **Cost:** plausibly **negative**. Numerically exact - zero convergence risk.
- **Kill without a gate:** read `nki_muon.py` first. If it already batches, drop. If it loops per-matrix, implement
  and compare step time with output forced numerically identical. No loss run needed to decide.

### 0.2 Stiefel retraction on W_O
Every ~100 steps, run a couple of extra Newton-Schulz iterations on the attention output weight *itself* (not its
gradient), snapping it back onto the orthogonal manifold instead of letting it drift over 3400 steps.
- **Cost:** ~0% amortized; reuses an already-compiled kernel; gated on the host between launches, so no per-step op.
- **Kill without a gate:** log the Frobenius distance of `W_O` from the nearest orthogonal matrix every 100 steps
  in an existing run. If it is already tiny, Muon's update-orthogonalization is holding it and there is nothing to
  retract. Pure logging, no code path change.

### 0.3 Stationary-everything backward (pre-transpose on idle engines)
Backward carries 182k PE transposes (36 ms) + 221k stationary weight loads (25 ms) = 61 ms, ~11.7% of step. During
forward, PE is only 31% busy and DMA has headroom - precompute and cache the transposes the backward will need.
- **Kill without a gate:** we already own `neff_instr_agg.py` / `trn2_profile_one.sh`. Read the existing NEFF
  instruction profile: if the 182k transposes are mostly *reused weight* transposes, little is left; if they are
  fresh per-step *activation* transposes, it is live. One profile read, no run.

### 0.4 Mixture-of-softmaxes - the plumbing check
K=2-4 parallel context->logit branches sharing `lm_head`, combined in probability space, returned as `log(p)`.
- **The trap, and the free test:** because `p` sums to exactly 1, `log_softmax(log p) = log p`, so returning
  `log(p)` as "logits" scores correctly under the fixed `evaluate_bpb()` without touching `prepare.py`.
  **Verify this numerically in fp32 before writing any training code**, and watch bf16 drift in `sum(p) ~ 1`.
- If the numerics check fails, the whole idea is dead for correctness reasons, independent of quality. ~5 minutes.

### 0.5 BPE-impossible-token masking - the density check
= `strange-wins-research.md` #2 (canonical token masking), already on plan-097 lever C but never built.
Static prev-token -> forbidden-next-token mask from tokenizer merge determinism, added as a `-inf` logit bias.
- **Kill without a gate:** compute mask density offline first. If under ~5% of (prev,next) pairs are forbidden,
  there is not enough structure to move bpb. Pure offline analysis.
- **NOTE:** MIR's failure does NOT bear on this. MIR corrupted *inputs*; this removes provably-impossible
  *outputs* from the softmax. Different mechanism entirely.

---

## TIER 1 - Near-free, one-line or one-op (each still needs a gate, but costs ~0% step time)

### 1.1 Exact-bpb loss reweighting
Weight each token's CE by its byte length, in log2, so the training objective **is** bits-per-byte rather than a
proxy. Currently a 7-byte token and a 1-byte token contribute equally to the loss but 7:1 to the score.
- **Cost:** ~0% (a per-token scalar multiply on an already-computed loss vector). Forward/eval path untouched.
- **This is the single cheapest idea in the whole set.** One loss-function line.
- **Falsifier:** if val_bpb does not move outside ~0.0004, uniform-weight CE was already a good proxy - and that
  also devalues every other byte-aware idea below, so run it FIRST as a probe of the whole family.

### 1.2 Positional bigram keys
Fold a coarse position bucket into the existing hash: `bh = (hash(prev,cur) ^ (pos%64 * SALT)) & (SIZE-1)`.
Same table, same gather, same op count - different constants.
- **Cost:** ~0%.
- **Risk to check first:** this spreads each surface bigram across up to 64x more buckets, diluting capacity at a
  table size we already know is at its ceiling (2^19 lost). Check bucket-load stats before trusting a negative.

### 1.3 FiLM-modulation read
Stop *adding* the bigram row. Split it 256/256 into (scale, shift) and modulate: `x = wte(idx)*(1+scale) + shift`.
Same gather, same scatter, different combine op - turns the memory from additive storage into computation.
- **Cost:** ~0%.
- **Falsifier:** if worse, additive superposition was load-bearing (multiplication is unforgiving of near-zero
  untrained rows) - which is itself a useful fact about the table.

### 1.4 Rank-1 memory (frozen direction + learned scalar gain) - DOWNGRADED, see 3.1
**S_HDC (+0.098) is strong evidence against this.** HDC froze per-row content entirely; rank-1 leaves one learned
scalar. Same axis, milder dose. Do not spend a gate here until something else re-opens the "less learned per-row
capacity" direction.
Stop learning 512 dims per bigram row. Freeze a random direction per row; learn only a scalar gain.
- **Why:** with 2^18 rows and 446M tokens, most rare rows get 0-3 gradient updates all run. Fitting 512 dims from
  one noisy gradient is hopeless; fitting one scalar is tractable. This is the project's own "immediate gradient
  flow wins" rule applied at the *per-row* level.
- **Cost: negative (~-2%)** - gradient scatter traffic collapses from (2^18, 512) to (2^18, 1).
- **Falsifier:** if materially worse at equal steps, per-row direction was load-bearing.

---

## TIER 2 - Systems levers (big numbers, moderate implementation)

### 2.1 Local-SGD / stale gradient sync  <- biggest clean number in the set
Sync every N=4-8 steps instead of every step. Each core trains independently between syncs.
- **Why it is NOT the closed comm-overlap experiment:** that tried to *hide* the per-step comm concurrently and
  washed. This reduces how *often* the comm happens at all.
- **The number:** reduce-scatter 55 ms + optimizer all-gather 50 ms = **105 ms/step, ~20% of step**, entirely
  outside the compiled graph. At N=4 that is ~13-15% of step; at N=8, ~18%.
- **Cost:** a step counter gating the existing `all_reduce`. Small code, big lever.
- **Falsifier:** one gate at N=4. If matched-step val_bpb degrades more than the exchange rate justifies,
  staleness is too expensive over only 3400 steps.

### 2.2 Direct Feedback Alignment
Compute the true gradient only at the head; project it through frozen random matrices to give every other layer a
synthetic local gradient. No chain rule crosses layers.
- **Why here:** backward is 144 ms vs forward 60 ms. The 182k transposes and 221k weight loads exist *because*
  gradients thread backward layer-by-layer. Under DFA all 12 layers' pseudo-gradients are independent and issue as
  one batched op. Best case backward collapses toward forward's cost: ~8-12% of step.
- **Falsifier - free:** run it on the existing CPU/CUDA dev path for a few hundred steps at any width and compare
  loss-curve shape to backprop. If it plateaus far above baseline, kill before any NKI work. DFA's known failure
  mode is depth; 12 layers is within the range it has been shown trainable.

### 2.3 Forward-gradient learning
Forward-mode differentiation with local per-layer tangents, K=2-4 duplicate forwards on otherwise-idle engines,
no reverse pass at all.
- **Optimistic:** 2 forwards (~120 ms) undercut the 144 ms backward *and* eliminate the transpose + weight-reload
  lines entirely. **Pessimistic:** variance forces K=4+ and it is a wash or worse.
- **Falsifier - free:** global (not even local) forward-gradient on the dev path for a few hundred steps. If train
  loss does not clearly beat random init, variance is fatal at this scale.

---

## TIER 3 - The 134M-parameter question (restructure the table, do not resize it)

The table is 68% of the model. Every closed experiment resized it (2^19 lost), re-hashed it (prime worse), split
it (wash), or decorrelated it (sign trick wash). **None changed its addressing or read mechanism.**

### 3.1 HDC binding - CLOSED, FAILED CATASTROPHICALLY (S_HDC, 2026-09-15 20:11)
**Result: 1.150833 vs ~1.053 at 600 s = +0.098.** Also NOT cost-negative as predicted: 0.54 s/step, +4%.
The learned table's *content* is load-bearing, not just its addressing - frozen algebraic binding cannot
substitute for 134M learned parameters. **This is the strongest single fact to come out of this whole pass, and it
propagates:** any idea that reduces per-row LEARNED capacity now carries a heavy negative prior, including 1.4
(rank-1 memory) and the trigram-permutation extension below. Restructure the table's *read*, never its *content*.
Replace the learned hashed-bigram table with vector-symbolic binding: a frozen random +-1 sign vector per token
(8192x512, never trained), and the bigram feature is `wte(prev) (Hadamard) signbook[cur]`.
- **Every one of the 8192^2 bigrams gets a distinct composite vector with graceful degradation instead of hard
  collisions** - using ~4M frozen params instead of 134M learned ones, and freeing ~130M for a model diagnosed as
  undertrained.
- **Cost: ~-2 to +4%** - it *drops* the `bte` gather AND scatter-add entirely; `wte(prev)` is a static shift of an
  existing gather, not a new one.
- **Extension (archaeology agent):** add fixed channel permutations to bind trigrams algebraically -
  `e_t * perm1(e_{t-1}) * perm2(e_{t-2})` - which generalizes continuously to n-grams never seen, unusually
  valuable at 1.4 epochs.
- **Falsifier:** bind-only with no learned gain; if it does not beat "no bigram signal at all" within a few hundred
  steps, the composite is not separable by the head.

### 3.2 Multi-read addressing - three variants, genuinely different combine rules
**Prior to respect: BIGRAM_SPLIT=2 (independent hashes, concatenated) was tested and was a WASH.** That closes
*linear* decorrelation. It does not close these:
- **(a) Kanerva SDM sum:** K=4 independent hashes into ONE shared table, read = sum. Soft generalization to
  near-duplicate contexts; zero new parameters.
- **(b) Median-of-K:** same K gathers, combined by elementwise **median** - a nonlinear collision-*suppressor*
  (drops the one bad colliding row) rather than an averager that mixes its noise in. This is the variant most
  clearly distinct from the closed split test.
- **(c) Bit-flip neighborhood:** read `h`, `h^1`, `h^2` with learned per-offset scalars, so a novel bigram borrows
  partially-trained statistics from address neighbors instead of returning an untouched random row.
- **Cost:** ~2-8% depending on variant (extra gathers are parallel and cheap; extra scatter-adds are the real cost).
- **Falsifier for (b):** compare median-combine against plain-sum-of-K *with the same K gathers*. If sum wins, the
  nonlinearity is not earning its scatter cost. For (c): if `w1`/`w2` train to ~0, neighbors carry nothing.

### 3.3 Fast-weight token modulation
Per-token rank-1 weight delta without materializing per-token matrices:
`y_extra = (x . v_shared) * fast_u[token_id]` - one dot product, one gather, one broadcast multiply.
- Multiplicative per-token capacity rather than additive, with no gating network to bootstrap. Structurally
  different from the ve/bigram tables already present.
- **Cost:** ~1%.

---

## TIER 4 - Output side and the compression metric

### 4.1 PPM / Witten-Bell closed-form mixing  <- archaeology agent's top pick
Precompute offline (training data only) a per-bigram-slot escape weight `w = distinct_continuations /
(distinct_continuations + total_count)` from classical PPM, gather it alongside the bigram row, and mix into
**pre-softmax logits**: `logits += w[slot] * (bigram_row @ W_proj)`.
- **Why it is not bigram re-injection (closed, G_REINJ +0.0008/+0.0012):** re-injection asked the *network to
  learn* how much to trust the table at each depth. This weight is **exact and correct from step 0** - zero
  bootstrap, which is precisely the property the undertrained regime rewards.
- **bpb is literally a codelength metric.** The PAQ/CMIX lineage optimized exactly this and their core move -
  mixing predictors by a confidence-derived non-learned weight - has never been tried here.
- **Cost:** ~1-2% (one scalar gather + one small projection).
- **Falsifier:** does the closed-form weight beat a single learned global scalar at equal step time? If a plain
  scalar matches it, the classical framing buys nothing. One-line ablation.

### 4.2 Exact-marginal byte cross-entropy
Marginalize the model's own 8192-way distribution down to the next *byte* (fixed token->first-byte matrix, one
[8192x256] matmul on already-computed probabilities), and add `-log2(marginal[true_byte])` as an auxiliary term.
- **Distinct from 1.1's reweighting:** when the model is uncertain between two tokenizations that start with the
  same byte, token CE punishes it even though that costs **zero bits at the byte level**. Marginalizing removes a
  spurious penalty rather than re-weighting a real one.
- **Cost:** ~2%. Forward returns normal logits; eval path byte-identical.
- **Falsifier:** measure what fraction of probability mass is split across same-first-byte tokens in a trained
  checkpoint. Near zero -> identical to standard CE -> dead.

### 4.3 Length-factored softmax
Factor `P(token) = P(byte-length bucket) x P(token | bucket)`, buckets from the fixed tokenizer byte lengths.
- Byte length is *exactly* what the metric weights by, and nothing in the current loss ever asks the model to
  predict it. A coarse 4-way high-base-rate classification is the easiest possible target to get immediate
  gradient on, and it sharpens the within-bucket call.
- **Cost:** ~2-3%. Both softmaxes run off the same hidden vector in parallel - no serial dependency.
- **Falsifier:** log 4-way bucket accuracy in the first 200 steps. If already >95%, the signal carries nothing.
- *Family variants not separately listed: byte-trie hierarchical, merge-rank two-tier, frequency-class two-level.
  Same shape, different factoring key. Byte-length is the one that targets the metric, so test it first.*

### 4.4 Mixture-of-softmaxes head
K=2-4 branches sharing `lm_head`, mixed in probability space. See 0.4 for the mandatory numerics check.
- **Why the historical objection does not apply:** MoS never became standard because at 50k-256k vocab, multiplying
  the most expensive op by K is real cost. Our vocab is 8192 and the head is ~1-10% of step - the entire reason it
  died is absent.
- **Variant (objective agent):** gate the mixture on a *free* signal (position parity, or the predicted byte-length
  bucket from 4.3) instead of a learned gate network - removes the bootstrap.
- **Cost:** ~3-8% for K=2.

### 4.5 SOM vocab reordering + logit smoothing
Offline (training-data-only, cached like the tokenizer artifact), compute a co-occurrence/SOM permutation of the
8192 vocab so similar tokens are adjacent; bake it into `lm_head` row order at init (**zero runtime cost**); then
apply a small learned depthwise 1D conv across the reordered logit vector before softmax.
- One gradient update then corrects a whole *neighborhood* of confusable tokens instead of 8192 independent logits
  - free generalization in an undertrained regime.
- **Why the historical objection does not apply:** SOMs died because competitive learning does not compose with
  backprop. Here the SOM runs once, offline, outside the timed graph; everything inside the graph is a plain conv.
- **Cost:** <1%.
- **Falsifier:** swap the SOM permutation for a fixed *random* one. If identical, any gain is from the conv's
  params, not the topology.

---

## TIER 5 - Structural, high variance

### 5.1 Modern Hopfield global memory bank
At 1-2 layers, add M~128 learned key/value slots (independent of sequence position) read by one small softmax:
`softmax(h @ K_mem^T / sqrt(d)) @ V_mem`, zero-scale-init.
- **Content-addressed** (keyed on hidden state), unlike every table in the model, which is **ID-addressed**
  (token-indexed). Genuinely complementary, not redundant.
- **Why legal:** classical Hopfield needed iterative energy-descent settling at retrieval - illegal under the
  one-pure-forward eval contract. The modern continuous version (Ramsauer 2020) converges in a *single* softmax
  read, which is just attention over a small fixed key set.
- **Cost:** ~1-2% per included layer; ~131k new params.
- **Note:** a sibling proposal - hashing the *hidden state* to index a table - is **pre-falsified by NKI rule #11**
  (a gather whose index is produced inside the graph NaNs on device; this already burned a device run on own-
  attention v4). The Hopfield version avoids it entirely by using a matmul instead of a gather. Do not build the
  gather version.

### 5.2 Pi-Sigma cross-term branch
Add `extra = (Wa x) * (Wb x)` (two low-rank projections, elementwise product) in parallel with the existing
`relu(W1 x)^2`, with `Wb` zero-init.
- **ReLU^2 can only square a channel against itself - it can never express a cross term between two different
  learned directions.** GLU-family gating (flagged as a *real* effect in our own external-techniques calibration)
  is a restricted special case of exactly this.
- The historical instability objection has an existence proof inside our own model: ReLU^2 is itself a product
  unit and trains fine under RMSNorm/QK-norm.
- **Cost:** ~3-5%, directly tunable via rank `r`.
- **Falsifier:** if `Wb`'s norm collapses toward zero mid-run, the model is rejecting the cross term.

### 5.3 Reservoir / echo-state multi-timescale memory
Fixed, untrained, log-spaced exponential decays (4/16/64/256 tokens) over the value stream, computed as a plain
cumulative sum; only a small readout projection is trained.
- **Why the hardware objection does not apply:** ESNs need serial step-by-step simulation - fatal here. But a
  *diagonal* fixed transition is a closed-form cumsum, fully parallel, no recurrence loop.
- "Only the readout is trained" is a direct match for 3400 steps with no time to learn slow dynamics - useful from
  the first gradient step.
- **Cost:** ~2-3%.

### 5.4 Ghost-basis norm / Hadamard-spectral parameterization
Frozen random orthogonal `R` sandwiching the RMSNorm gain (`normalize(x) @ R @ diag(g) @ R^T`) so the same `d`
learned numbers express an arbitrary symmetric operator in a hidden basis; or `W = H diag(v) H` with fixed
Hadamard `H`, collapsing a 262k-param matrix to 512 learned params whose loss surface is **exactly diagonal**
(AdamW on `v` becomes per-eigenmode Newton descent - about as fast as 3400-step convergence can get).
- **Cost:** ~1-3% (as *dense* matmuls with the constant - NOT the O(d log d) butterfly, which needs 9 serial
  stages and would be catastrophic on a latency-bound machine).
- **Falsifier:** apply at one site only, matched steps, freed params unused to control for the parameter-count
  confound.

---

## Explicitly rejected from the 52

- **Flat fan-out trunk** (all 12 layers read x0, one wide matmul): a strictly *larger* version of the parallel-
  residual bet that already lost +0.005 at equal steps (PARRES_T). The hardware agent flagged this itself.
- **Jacobi-unrolled depth** and **weight-tied depth loop**: same family; weight-tying is a 12x capacity cut in a
  regime we know is capacity-starved.
- **Pipeline-parallel layer split**: arithmetically dead - 8 stage-boundary comm events against a 60 ms/step
  launch-gap proxy exceeds the depth-shortening saving.
- **Content-addressable hidden-state gather**: pre-falsified by NKI rule #11 (see 5.1).
- **Head-side bigram echo** (second table injected pre-head): bigram re-injection at 2 and 4 block inputs both lost
  (+0.0008, +0.0012); pre-head is the same bet at the extreme.
- **Per-rank asymmetric backward truncation** (8 different NEFFs, each rank backprops a different layer subset):
  interesting, but it almost certainly breaks the submission's rank-symmetry assumptions. Check the harness before
  spending any thought on it.
- **Quaternion/hypercomplex projections** (proposed twice, two different sites): historical verdict is "unclear
  gains vs matched-parameter real nets," and neither agent could name a reason that changes here.

## Suggested order

1. **Tier 0 in full** - none of it needs a gate, and 0.1/0.3 might be free step time.
2. **1.1 (exact-bpb reweighting)** - one line, and it is a probe of the entire byte-aware family (4.2/4.3).
3. **2.1 (Local-SGD)** - biggest clean number, small code.
4. **4.1 (PPM closed-form mixing)** - the zero-bootstrap property is exactly what this regime rewards, and it is
   the cleanest non-overlap with the closed re-injection result.
5. **3.1 (HDC binding)** - the only idea that would restructure 68% of the parameters, and it is cost-negative.
6. Free dev-path falsifiers for **2.2 / 2.3** whenever the box is busy - they cost no box time at all.
