# Where the non-compute 22% of the step goes (2026-09-24)

Written while chasing 0.970 local. The forward/backward work of `findings/vector-deletions-2026-09-23.md` is
separate; this file is about the **other 107 ms** of the step, which had never been accounted for byte by byte.

## Measured phase split of v52 (vehicle, 300 s, PROFILE=2 + OPT_PROF=1)

```
fwd_bwd 0.375   grad_sync 0.060   optimizer 0.047   total 0.482
optimizer split: muon 0.0049  adamw 0.0050  adamw_flat 0.0075  gather 0.0276
per micro-step:  embed-gather 3.2  fwd 122.6  bwd 217.2  host 16.8  embed-scatter 12.9   (sums to 372.7 ~ fwd_bwd)
```

So **`grad_sync` + parameter `gather` = 88 ms = 18% of the step**, and it is pure data movement.

## The sharded parameter set, exactly

The run prints `OPT_SHARD=zero: 9 params (218,103,808 elements)`. With `N_EMBD=512`, `DEPTH=12`, `N_HEAD=4`,
`MLP_RATIO=6`, `BIGRAM=18` those 9 params are, and the element count **matches to the digit**:

| param | shape | elements | share |
|---|---|---|---|
| bigram table `bte` | (262144, 512) | 134.22 M | **61.5%** |
| MLP `W1` (padded 12 -> 16) | (12, 512, 3072) | 25.17 M | 11.5% |
| MLP `W2` (padded 12 -> 16) | (12, 3072, 512) | 25.17 M | 11.5% |
| attention stack | (48, 512, 512) | 12.58 M | 5.8% |
| `wte`, `vte`, `vte_mid`, `vte2` | (8192, 512) x4 | 16.78 M | 7.7% |
| head | (8192, 512) | 4.19 M | 1.9% |
| | | **218.10 M** | |

`W1`/`W2` are padded because 12 does not divide 8 (`OPT_SHARD_PAD`, muon, dim 3); everything else splits evenly.

## The two findings

### 1. RETRACTED - the gradient collective is NOT byte-bound

**I wrote the opposite here first and it was wrong. Measured and refuted the same afternoon** (`Q_REF` /
`Q_GBF16` / `Q_GBFPAD`, three paired 300 s screens):

| arm | gradient bytes | grad_sync | steps |
|---|---|---|---|
| fp32 (reference) | 872 MB | 0.059 | 646 |
| bf16 for the 77% that is not padded | 536 MB | **0.059** | 648 |
| bf16 for 100% of it | 436 MB | **0.058** | 643 |

**Halving the bytes of every one of the 218.1 M sharded elements moved the phase by 0.000.** The reasoning that
produced the wrong answer was: divide a phase by its byte count, call the quotient a bandwidth ("~13 GB/s"), and
then treat that number as if it were a property of the link. It is not - it is just 59 ms divided by 763 MB, and
it says nothing about what the 59 ms *is*. `CCPROBE`'s 85 MB all_reduce in 2.34 ms (~64 GB/s) was the contrary
evidence already in the ledger, and I should have reconciled the two before building on either.

This kills, for the **gradient** collective: `GRAD_BF16` (re-confirming `S_V26_OPHEAD_GB16` from v26, which I
should have grepped for first), the bf16-gradient half of `BTE_BF16`, and the sparse / `all_to_all` bigram
gradient exchange described at the bottom of this file - removing 688 MB of wire would buy approximately nothing.

**Bytes do still pay on the parameter `all_gather`** (`GATHER_BF16`, +0.6%), so the two collectives are not
alike; do not generalise either result to the other. `F_V24_BGP` (prime-sized table -> 537 MB `all_reduce` ->
0.78 to 1.44 s/step) is also still real, but it removed *sharding*, which changes far more than byte count.

### 2. The bf16 gather only half-worked, and the reason says what to do next

`gather` went 30.6 -> 27.6 ms when the parameter all-gather became bf16. Halving the bytes of a bandwidth-bound
collective should have halved the phase. It did not, because the bf16 path **adds** a full-size converting copy:

```python
dist.all_gather_into_tensor(p16, plan["pshard"].to(torch.bfloat16))   # N bf16 - half the bytes
p.data.copy_(p16)                                                     # N elements bf16 -> fp32: 1.3 GB of traffic
```

Taking fp32-gather = 30.6 ms as pure collective, the bf16 collective is ~15.3 ms, so **the copy costs ~12 ms**.
The fix is to make the sharded *replica* `p.data` bf16 so the collective writes into it directly; the fp32 master
lives in `plan["pshard"]` and is untouched. With `GATHER_BF16` already shipped, `p.data` **already holds
bf16-rounded values stored in fp32**, so this is bit-identical to what the graph sees.

Constraint found while scoping it: autograd requires `p.grad.dtype == p.dtype`, so a bf16 `p.data` forces bf16
gradients for the **in-graph** params, and the fused Muon/AdamW NKI kernels require an fp32 `(pshard, gshard)`
pair (`p.dtype == torch.float32` is an explicit guard) behind a `try/except` that would **silently fall back to
the torch path**. The five sidecar embedding tables have their `.grad` attached manually, so they can go bf16
without touching autograd or the optimizer - and they are 151 M of the 218 M elements.

## The sparse-bigram idea: correctly described, and DEAD anyway

Each rank processes 16384 tokens, so it touches **at most 16384 of the bigram table's 262144 rows**:
93.75% of that 536 MB reduce-scatter is provably zeros, and a fixed-shape `all_to_all` of only the touched rows
would move ~67 MB per rank instead of 469 MB. That reasoning is sound and the sparsity is real.

**It is still not worth building**, because the three arms above show the phase does not respond to byte count at
all. Do not revive it on the strength of the sparsity argument alone - the thing to establish first is what the
59 ms *is* (see below), and only then whether the candidate removes any of that.

The `handoff-2026-09-24.md` line that "merging collectives is dead and therefore the sparse-bigram idea is dead"
was self-contradictory as written (it ended "reduce bytes, not call counts"). The correct statement is that
**for the gradient reduce-scatter, neither call count nor byte count is the cost.**

## What the 59 ms actually is: the eager launch floor

The model that survives all of the above, and independently explains `adamw_flat` taking **8.4 ms to update 93
numbers**, is that **an eager device op costs ~0.3 ms regardless of its size**. Op counts per step:
`gather_params` 27 (9 casts + 9 all-gathers + 9 converting copies), `reduce_gradients` ~13, `_adamw_step_flat`
~23. At ~0.3 ms that is ~19 ms of pure launch overhead in a 107 ms phase, and it predicts the bf16 wash exactly:
bf16 halves bytes but changes no op count.

So the lever in this phase is **fewer device ops**, not fewer bytes. Ranked, with the blocker for each:
1. `gather_params` 27 -> ~9: needs the fused optimizer kernels to emit a bf16 shard (deletes 9 casts) and bf16
   *replicas* (deletes 9 copies). The replicas are blocked by autograd requiring `p.grad.dtype == p.dtype` for
   in-graph params, and by the tail EMA - `ema_params` is cloned from `p`, so a bf16 replica silently makes
   `e.lerp_(p, 0.003)` a no-op. Fix the EMA by cloning `.float()` explicitly.
2. One flat gradient buffer so `FLAT_COMM` needs no `cat`: 18 collectives -> 2. `FLAT_COMM` lost before purely
   because its `cat` moves 1.7 GB; pre-placing every sharded gradient as a view of one flat buffer removes the
   `cat` **and** both padded-stack `cat`s - but it requires `zero_grad(set_to_none=False)` so autograd
   accumulates in place instead of allocating fresh gradients each step.
3. `_adamw_step_flat` ~23 -> ~14 (`FLAT_OPT2`, built): params become views of the flat tensor so the 8
   write-back copies vanish, and the update expression collapses. Verified bit-identical in float64.

## What was wrong in the older notes

`findings/handoff-2026-09-24.md` says merging collectives is dead because per-call overhead is 0.01-0.14 ms, and
then lists "the sparse-bigram-gradient idea, which only reduces bytes" as killed by the same measurement. That is
backwards: the same note ends "Reduce bytes, not call counts", and the sparse-bigram idea is *precisely* a byte
reduction - the largest one available. Only the **call-merging** conclusion survives.
