# The excluded startup window and packing (2026-09-16)

> **Status 2026-09-16 (after triage; scoreboard rows in `flow-scoreboard-2026-09-13.md`):**
> - **Theme A is bounded by measurement:** the whole per-micro-step host path (loader `next` + `EmbedSidecar.prepare`)
>   is **7.3 ms** vs 159 ms of device fwd+bwd (PROFILE2, S_V26_HEAD_P2) and it already overlaps the backward; the 87-100%
>   host CPU is the runtime's busy-wait, not work. A1/A2/A3 can recover at most ~1.4% of the step (~0.0004) even at zero
>   overlap - not worth the loader rewrite. Re-measured at the PACK=4 config by `H_PROF2` (queued). A5: full-run logs show
>   no dt spike at the EMA start (no charged mid-run recompiles). A4 buys nothing for the scored run (the 900 s cap is
>   not binding at ~500 s). A8 (n-gram init): the unigram/bigram prior is learned in the first ~50 of 3400 steps, so the
>   ceiling is ~1% of training; parked. Note also `findings/ema-cold-start-bug.md`: the step-0 clock inside the window
>   was silently corrupting short screens until today.
> - **Theme B implemented as vehicle knobs (default off) and gated:** `PACK_MODE=bfd` (B1 + B5: best-fit-decreasing over
>   4096-document rounds, open rows carried across rounds, over-length docs split into whole rows, overflow closes keep
>   the remainder -> 100% token coverage, ~94% of documents placed whole, no truncation loss; local smoke clean) =
>   `P_BFD`; `PACK_ROLL=1.0` (B6 at f=1.0: every packed 4097-row rolled by a random offset so all four sub-rows start
>   mid-document; the provided rows are BOS-aligned so the wrap is a clean document start) = `P_ROLL`. B8 is moot (the
>   provided document stream is already rank-strided, each rank packs its own documents). B7 (second-pass repack) and
>   B4 (relatedness) wait for the P_BFD verdict.

17 ideas on two linked axes: **what to do with the free pre-clock time**, and **how to pack better**. Packing is the
only axis that has produced a large win since v40 (PACK=4, -0.0045), and the startup window is the only free compute
in the run. Nothing here is tested.

Baseline: **v43 = 0.979941 local / 0.9859 portal**, 0.52 s/step, ~3400 steps.

## CORRECTION: the window is NOT unlimited

The submission contract (upstream `c907da4`, `playbook.md` s"The submission contract") is precise:

> Wall time from launch through the **5th** reported boundary is the excluded startup/compilation allowance,
> **capped at 900 s**. The budget clock restarts at boundary 5, **back-dated by any over-cap excess** + 30 s margin.

So it is not "however long it wants". It is 900 s, and **every second past 900 is charged back against the 30
minutes of training**. Current usage: cold compile ~500 s (T53 measured 499 s at step 0). **Unused free time:
~400 s.** That is the actual budget for everything in Theme A.

Two further constraints that kill the obvious tricks:
- **We upload ONE file.** No shipped NEFF cache, no shipped preprocessed data, no side artifacts. Everything must be
  computed in-process, every run. (`neuron_parallel_compile` / `NEURON_COMPILE_CACHE_URL` are dev-box tools only -
  useful for OUR iteration speed, irrelevant to the scored run.)
- **`record_step_boundary(step)` is called exactly once per completed optimizer step.** There is no way to run
  uncounted optimizer steps inside the window. Do not try; `check_submission` and the boundary count would catch it,
  and we have two review rejections already.

### The one adjacent trick, and why to skip it
The first 4 steps are inside the allowance, so they could in principle use an enormous batch. Skip it: 4 optimizer
updates are 4 updates regardless of batch size, worth ~nothing against 3400 steps, and it is exactly the shape of
thing that draws review attention for no gain.

---

# THEME A - Spend the ~400 s of free host time

The measured justification (2026-09-16, live `neuron-monitor`, 60 s during a v43-line gate):
**host CPU 87-100%**, NC utilization 1.9-4.5%, device memory 86.63 GB, host 32.4 GB.
The host is saturated. This is the same signature that drove the LNC=2 decision at v5 ("CPU ~95% with 8 dispatching
workers - the host, not the chip, was the bottleneck"). Anything moved off the steady-state host path is a direct
hit on the measured bottleneck.

### A1. Stage the entire corpus in device memory
315M tokens at vocab 8192 fits in **uint16 = ~630 MB**. Device already holds 86.63 GB. Load the whole training set
into HBM during startup; the training loop then slices a device tensor and never touches the host for data.
- Kills per-step host batch assembly, H2D copies, and a large share of the ~5,000 small DMA copy-ins per step that
  PR7 flagged as launch/DMA-bound.
- **Not the same as the closed `LOADER_DEV=cpu` experiment** (wash), which moved work *onto* the host. This removes
  it from the loop entirely and pays for it in excluded time.
- **Cost:** ~1-2 s of window. **Risk:** low. This is the single highest-value item in the file.

### A2. Precompute every gather index and scatter plan
The data order is deterministic (`manual_seed(42)`, fixed packing), so **every micro-step's inputs are knowable at
startup**. Precompute and stage:
- bigram hash indices for the whole corpus (315M x int32 ~ 1.26 GB, or recompute on-device from the staged tokens),
- `EmbedSidecar` gather indices (currently host-computed per step - this is *why* NKI rule #11 exists: indices must
  be host-supplied graph inputs),
- the **scatter plans** (v23/v25 capped/chunked slot plans), which depend only on which tokens appear in each batch.
- **Cost:** tens of seconds of window. Removes a per-step host stage that currently runs 3400 times.

### A3. Overlap data prep with compilation
The compiler is also host-bound, but the box has many cores. Run the Theme-A/B preprocessing **concurrently** with
the ~500 s compile instead of after it. Uses the same 900 s twice.
- **Falsifier:** check whether `neuronx-cc` already saturates all cores during compile. If it does, there is no
  headroom and prep must run serially in the remaining ~400 s.

### A4. Shrink compile to buy more window
Every second cut from the ~500 s compile is a second available for A1/A2/A6/A9. Worth auditing how many distinct
graph shapes get compiled (train graph, eval graph, any warmup variant) and whether any can be eliminated or shared.
- Also guarantees headroom: mid-run recompiles cost ~60 s+ and are charged at full price.

### A5. Guarantee zero mid-run recompilation
Force every shape the run will ever use (including the eval path and the final EMA/checkpoint path) to be traced and
compiled **before boundary 5**. A shape change later in the run is charged against training time, not the allowance.
- **Cost:** ~0 if the shapes are compiled anyway; pure insurance otherwise.

### A6. Precompute tokenizer-derived tables (free enablers)
All of these are pure functions of the provided tokenizer, cost seconds, and each unlocks an idea from
`wild-ideas-2026-09-15.md` that is otherwise priced with its preprocessing:
byte-length vector (enables exact-bpb reweighting), token->first-byte map (byte-marginal CE), BPE-impossible
prev->next mask (canonical token masking), byte-trie / merge-rank structures, frequency-class assignment.

### A7. Precompute corpus statistics
A counting pass over 315M tokens on 8 host cores is plausible inside the window (subsample if not). Yields:
Witten-Bell escape weights per bigram slot (enables the PPM idea, `wild-ideas` 4.1), unigram/bigram frequencies,
per-slot occupancy counts. **This is legal** - it is the provided training data, processed in-process, which is the
same thing `prepare.py` itself does.

### A8. Data-dependent initialization from n-gram statistics  <- possibly the biggest idea here
Initialize the `lm_head` bias (and optionally the bigram table rows) to **empirical log-frequencies** computed in
A7, so the model starts at roughly the n-gram solution instead of at uniform.
- **Why it could matter a lot:** at 3400 steps we are undertrained, and a large share of early steps is spent
  learning the unigram/bigram prior that a counting pass gives exactly and instantly. This is the purest possible
  instance of "zero bootstrap, immediate signal" - the property that has consistently separated winners from losers
  in this project (it is also why PPM mixing looks good and why re-injection lost).
- Unigram-bias init is completely standard practice and unambiguous. Bigram-table init from counts is the more
  aggressive version of the same move.
- **REVIEW LINE:** computing this in-process at runtime is fine. **Baking the numbers into the source file is not** -
  a reviewer will reasonably call an embedded table derived from training data a pre-trained weight, and we have no
  appeal. Compute it every run, even though that costs window.

### A9. Solve the packing properly instead of greedily
Greedy online packing is used because it must be cheap. In a 400 s window it does not have to be - run a real
optimizer (see B3) over the length histogram once, at startup, and keep the result for the whole run.

---

# THEME B - Packing

What is measured, and what it rules out:

| variant | coverage | rows starting mid-doc | val_bpb |
|---|---|---|---|
| PACK=1 (provided) | 72% (18% truncated, 10% never placed: docs > 1025 never fit) | 0% | 0.985741 |
| PACK=2 | - | - | 0.981913 |
| **PACK=4 (v42/v43)** | **87%** | **75%** | **0.981197** |
| PACK=8 | - | - | 0.980043 (wash, +0.0001) |
| A_STREAM | 100% | 99.8% | 0.987395 (**LOST +0.0017**) |

**Coverage is not the mechanism** (stream had the most and lost). **Eval-distribution matching is not the
mechanism** (stream matched eval's 99.9% mid-doc and lost). What is left is **document integrity**: whole documents,
long documents placed whole. The packing curve in *row length* is flat past 4. So the remaining value is in a
**better packing algorithm at the same row length**, not a longer row.

### B1. Best-fit-DECREASING instead of online best-fit
`prepare.py:466` is a *"BOS-aligned best-fit packed"* loader. Classical best-fit/first-fit **decreasing** sorts items
by descending size before placing - that sort is what buys the guarantee (FFD/BFD have absolute ratio 1.5, and FFD is
asymptotically 11/9 OPT + 6/9). An online best-fit without the sort has no such bound.
- We already own our packer (custom packing in train.py is organizer-sanctioned, `competition-rulings.md` #1).
- **Cost:** one sort in the startup window. Pure host-side.
- **Expected:** several points of whole-doc coverage at the same row length, on the axis that produced -0.0045.

### B2. SPFHP (shortest-pack-first histogram packing)
From [Krell et al., *Efficient Sequence Packing without Cross-contamination*](https://arxiv.org/abs/2107.02027):
build a histogram of sequence lengths, sort long->short, worst-fit into the largest remaining packs. Explicitly
designed for this exact problem, "processes millions of sequences in seconds", so it fits the window trivially.

### B3. NNLSHP (non-negative least squares histogram packing)
Same paper's stronger variant: formulate packing as a non-negative least squares problem over enumerated
length-combination strategies, weighting short-sequence residuals down to avoid long-sequence leftovers. Slower than
SPFHP but still far inside 400 s, and closer to optimal - exactly the trade the free window is for. **This is the
"solve it properly because compile time is free" idea in its purest form.**

### B4. Relatedness-aware packing  <- the untested half of a known result
[Analysing the Impact of Sequence Composition on LM Pre-Training](https://arxiv.org/pdf/2402.13991) finds that
**increasing the relatedness of documents within a sequence improves language-modelling ability**, and that
intra-document causal masking achieves the lowest perplexity. See also
[Structured Packing in LLM Training](https://arxiv.org/pdf/2312.17296).
- **We already have the masking half** (BOS-cumsum doc ids, causal AND same-doc, plumbed through the vendor
  attention's `bound_min`/`bound_max`). We have never tried the **relatedness half**.
- Cheap in-window proxy for relatedness: shared rare-token overlap, or a hashed bag-of-bigrams similarity, or simply
  keeping same-source/adjacent-shard documents together rather than shuffling globally.
- **Why it is plausible here specifically:** PACK=4's whole win came from changing document composition of rows.
  This is the next variable along the same axis, and it is the one the literature says matters.

### B5. Dedicated whole-document rows for long documents
At PACK=1, documents > 1025 tokens **never fit and were never trained on** (10% of corpus). PACK=4 raised the
threshold to 4097 but the same cliff exists above it. Give over-length documents their own rows (truncate at a
paragraph/sentence boundary rather than arbitrarily, or split into consecutive whole rows) so long-document content
is trained on at all, without fragmenting the short documents that stream packing destroyed.

### B6. Tune the mid-doc-start fraction explicitly
PACK=1 is 0% mid-doc-start and loses; stream is 99.8% and loses; PACK=4 is 75% and wins. **The fraction was never
tuned - it fell out of the row length.** Decouple it: pack whole documents, then deliberately offset a controlled
fraction *f* of rows to start mid-document. Sweep *f* in {0.5, 0.75, 0.9}. Zero step cost, pure loader change, and it
tests the one variable that correlates with every result in the table above.

### B7. Second-pass repacking
We run ~1.42 epochs, so ~42% of the run is a second pass that currently replays **the same packing**. Re-pack the
second pass with a different grouping so the model sees new *document combinations* without new data.
- **Prior to respect:** `F_V24_EP2` (rotated order) came back +0.0176 - but the ledger note suspects a bug ("the
  `shard_ids` path shards rows differently across ranks -> duplicated data"), so the hypothesis was never actually
  tested. Implement cleanly: same tokens, same count, different packing, no cross-rank duplication.

### B8. Per-rank packing diversity
8 workers currently derive rows from one packing arrangement. Give each rank a different arrangement so the global
batch sees more document combinations per step. Zero step cost; purely how the startup-computed plan is sliced.
- **Watch for exactly the F_V24_EP2 failure mode:** verify no row is duplicated across ranks.

---

## Suggested order

1. **A1 + A2** (corpus and indices on device) - biggest measured-bottleneck hit, low risk, and it makes everything
   else in Theme A cheaper by freeing host time.
2. **B1** (best-fit-*decreasing*) - one sort, on the only axis that has paid since v40.
3. **B6** (mid-doc-start fraction sweep) - zero cost, tests the variable every packing result so far correlates with.
4. **A8** (n-gram init) - highest ceiling in the file, but respect the review line: compute it, never bake it.
5. **B4** (relatedness packing) - the literature's result, and the untested half of something we half-implement.
6. **B2/B3** (SPFHP/NNLSHP) if B1 shows the packer is the binding constraint.

**Before any of it:** instrument where the 87-100% host CPU actually goes (loader? sidecar index build? scatter
plan? collectives?). If it is data-path work, A1/A2 are large. If it is collective/dispatch overhead, they are small
and Local-SGD (`wild-ideas` 2.1) is the better lever. That instrumentation is a host-side `PROFILE=2`-style split we
already have machinery for, and it does not need the device free.

Sources: [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027) ·
[Impact of Sequence Composition on LM Pre-Training](https://arxiv.org/pdf/2402.13991) ·
[Structured Packing in LLM Training](https://arxiv.org/pdf/2312.17296) ·
[Dataset Decomposition](https://arxiv.org/pdf/2405.13226) ·
[Neuron Persistent Cache](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/general/arch/neuron-features/neuron-caching.html) ·
[neuron_parallel_compile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html)
