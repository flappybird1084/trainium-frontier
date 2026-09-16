# Gap: we have no off-device compile path (2026-09-08)

Prompted by a Devpost post from another team (Peter Van Geldorp, ~2026-09-07) reporting a compile hang on the beta Neuron
runtime. **We do not have their bug.** What is worth taking is one diagnostic technique they used, plus the operational
capability it unlocks.

## The technique
They dumped the exact HLO from the failing path and compiled it **standalone with `neuronx-cc`**, off the runtime: 1.066 s,
a valid 26,427-byte NEFF — which isolated the fault to the runtime's process-launch path rather than the compiler,
without needing the device to cooperate.

## What we have today
We invoke `neuronx-cc` exactly once in the whole toolchain, and only as a process name to grep: `trn2_status.sh:4`
(`pgrep -c neuronx-cc`). `TORCH_NEURONX_HLO_CACHE_DIR` appears only as prose in `handoff-2026-09-06.md` (rule 3), never
in a script. Of ~40 box tools, none compiles anything off-device: everything we know about compilation, we learned by
running on the device and watching. Diagnostics from their post we already have: process-state inspection at hang time
(`pgrep neuronx-cc` / `wchan` — the silent-startup tiebreaker) and cache-key awareness (the compile cache is
source-keyed / content-addressed). `NEURON_COMPILE_SERVICE` toggling is the only other novel item and it is specific to
their hang.

## What the gap has cost us (from `trn2_results.tsv`)
Three device runs and three queue slots went to establishing that two compiler flags do not change the graph: **T17** (an
`-O2` "test" actually served from the persistent cache — loss bit-identical through step 450, killed), **T17b** (real
`-O2`, step-0 dt 568 s, fresh 46 MB NEFF, identical step time and loss to `-O1`), **T18b**
(`--model-type=transformer`, step-0 dt 534 s, fresh 74 MB cache, identical). All three would have been offline-answerable
by diffing NEFFs. Same class, also offline-answerable: **T15** (`NCC_EOOM002` at compile, peak HBM 28.56 GB > 24 GB/core,
full-run slot burned), **S_H8** (cold compile made step-0 dt 631 s > the 600 s screen budget; predictive stop fired at
step 5), **F_V24_BG19** (device OOM at step 0, 2^19 bigram). Ten rows in the results log mention cold compile at all.

## The bigger prize: cache pre-warm (UNVERIFIED)
The NEFF cache is content-addressed — `PROFV24` reads `/tmp/neff_cache/54/01/<hash>.neff` directly. **If** a
standalone-compiled NEFF drops into that cache and is picked up by the runtime, cold compiles stop eating the 900 s
startup allowance and stop killing short screens. Unverified (cache keying may include runtime metadata a standalone
compile does not reproduce), but cheap to test and the stakes are real: cold compile at 512-wide is 500-650 s; **T13 at
640-wide hit 893 s against the 900 s cap** — which, not its loss, is what ruled 640 out on the portal; and every new shape
pays it again (`S_V24_T2048` was a fresh shape). If pre-warm works, the startup allowance stops being a constraint on
model shape, re-opening width and depth as free axes on the compile line.

## Proposed tool
`hlo_compile.sh <config-env...>`: dump the HLO for a given vehicle config, compile it standalone with `neuronx-cc`,
report **compile seconds, NEFF size, and the peak-memory estimate**. Optionally place the result in the NEFF cache and
verify the next device run gets a cache hit. Converts three failure classes (compiler-flag experiments, compile-time OOM,
cold-compile-overruns-the-screen) from device-run experiments into offline checks that cost no queue slots and no device
time.

Status: not built. Proposed 2026-09-08.
