# What neuronx-cc produces: the NEFF (inspected 2026-09-15, neuronx-cc 2.26, largest cached training NEFF, 139 MB)

Container: 1024-byte binary header (u64s: 2, 1024, payload_len, 2) then a gzip stream of a tar (918 MB unpacked).
Top level: `neff.json`, `info.json` (tool_version, kelp_version, num_tpb), `hlo_stats.json`, `kelf-0.json`
(manifest, version 0.5, target `cayman` = Trn2; one graph `sg00` -> `sg00/def.json`).
`sg00/def.json` (version 0.6): `runtime_event_count`, `runtime_semaphore_count`, and a (json, bin) pair per engine.

The payload is literally one instruction stream per engine plus DMA programs:
| file | size | what |
|---|---|---|
| `PE0.bin` | 34 MB | TensorEngine (matmul) instructions |
| `DVE0.bin` | 24 MB | VectorEngine instructions |
| `Activation0.bin` | 14 MB | ScalarEngine instructions |
| `Pool0.bin` | 405 MB | GpSimd/Pool engine instructions (12x the PE stream: the graph is dominated by non-matmul work, matching the latency-bound profile) |
| `SP0.bin` + `SP0.json` | 5 + 11 MB | sync/scheduler program; the JSON is the DMA descriptor list (from-tensor, sizes, steps, to SBUF offset, queue name, semaphore id) |
| `tensor_map.json` | 0.3 MB | HLO tensor -> HBM allocation |
| `*_bkt.bin` / `*.json` | small | activation-function lookup tables (exp/log, rsqrt buckets) |
| `value_*_CRSM.npy` | small | compile-time constants |
| `debug_info_backend_*.dbg`, `debug_info_asm_*.dbg` | 460 MB | protobuf: maps every emitted instruction back to its source op (self-described "Debug info for one engine's assembly instructions") |

The instruction encoding is an undocumented binary ISA (`neuronxcc/isa_tpb/sunda`, `nki/isa/neuron_isa.so`); there is no
public assembler or disassembler, and `neuronx-cc compile --help` exposes no emit/dump flag (only --verbose). The
runtime (`libnrt`, python `nrtpy.NrtModel`) loads any NEFF, and torch_neuronx's cache is keyed by HLO hash, so a hand-made
NEFF *could* be dropped in - but nothing exists to author one.

Verdict: writing a NEFF by hand is not a 19-day project (no ISA spec, no assembler, semaphore/DMA-queue scheduling
done by the backend). The sanctioned ISA-level entry point is NKI `nisa` (one engine instruction per call, our choice of
engine and tiles; the backend still does semaphores, DMA descriptors, and instruction encoding). "Our own NEFF" in
practice = one big nisa megakernel per layer/step; measured ceiling for that is the 5-10% latency slack (~0.002-0.003).
