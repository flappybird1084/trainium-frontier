#!/usr/bin/env python3
"""Generate train3_s21.py = train3_s20.py + a TOK_THREADS knob on the dataloader's tokenizer thread pool.

## The hypothesis this tests

Two of the eight ranks are consistently ~18 ms slower than the other six, reproducibly, and not because of
their data (`Q_SKEW`, `Q_OMP1`, and again on v55 in `P_SKEW`: six ranks at 0.353-0.356 s, r1 and r7 at
0.371-0.373). The ledger closed this as "structurally slow core, cannot be fixed from train.py". That closure
was reached by elimination **without CPU contention in the candidate set**: the one test cited
(`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`) does not touch the tokenizer's own thread pool.

`prepare.make_dataloader` takes `tokenizer_threads: int = 4` (`prepare.py:462`, used at `:487` as
`tokenizer.encode(..., num_threads=)`) and every call site here uses the default. With 8 ranks that is
**32 tokenizer threads on a 12-vCPU box**, and refills happen *during* training in unsynchronised per-rank
bursts. A host-side probe on the real corpus measured the loader's `next()` at 3.6 ms mean but **40.9 ms max**,
i.e. exactly the bursty tail that would make two unlucky ranks pay every step.

Note the duty-cycle argument does NOT rule this out (I briefly thought it did): 18.3 ms of host work in a
424 ms step is only a 4% duty cycle *if the work is single-threaded*. Four tokenizer threads per rank during a
refill burst is a different regime, and 32 runnable threads on 12 vCPUs is oversubscribed by 2.7x.

`TOK_THREADS=1` is a one-line change per call site and **modifies nothing in the fixed harness** — it passes a
public keyword argument. If the r1/r7 offsets and the ~42 ms barrier wait do not move, the CPU-contention
hypothesis dies for one 300 s screen and the straggler axis is closed for the right reason at last.

Run from the repo root:  .venv/bin/python submission_raw/make_s21.py
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "submission_raw/train3_s20.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "submission_raw/train3_s21.py")

s = SRC.read_text()
applied = {}


def sub(name, old, new, expect=1):
    global s
    n = s.count(old)
    assert n == expect, f"{name}: expected {expect}, found {n}"
    s = s.replace(old, new)
    applied[name] = n


# ---------------------------------------------------------------------------- the knob
sub("knob", '''EMA_TABLES = os.environ.get("NEURON_COMPETITION_R1_EMA_TABLES", "0") != "0"''',
    '''# TOK_THREADS: the per-rank tokenizer thread pool in prepare.make_dataloader (its own default is 4). With 8 ranks
# that is 32 threads on a 12-vCPU box; the loader's refill tail measured 40.9 ms against a 3.6 ms mean on the real
# corpus. Candidate cause of the reproducible r1/r7 +18 ms straggle. Passing the keyword modifies nothing in prepare.py.
TOK_THREADS = int(os.environ.get("NEURON_COMPETITION_R1_TOK_THREADS", "4"))
EMA_TABLES = os.environ.get("NEURON_COMPETITION_R1_EMA_TABLES", "0") != "0"''')

# ---------------------------------------------------------------------------- the three loader call sites
sub("loader maxdoc", """        _raw_loader = make_dataloader(_MaxDocTokenizer(tokenizer), DEVICE_BATCH_SIZE // PACK, SEQ_LEN * PACK, "train", loader_dev)""",
    """        _raw_loader = make_dataloader(_MaxDocTokenizer(tokenizer), DEVICE_BATCH_SIZE // PACK, SEQ_LEN * PACK, "train", loader_dev,
                                     tokenizer_threads=TOK_THREADS)""")

sub("loader plain", """        _raw_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE // PACK, SEQ_LEN * PACK, "train", loader_dev)""",
    """        _raw_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE // PACK, SEQ_LEN * PACK, "train", loader_dev,
                                     tokenizer_threads=TOK_THREADS)""")

sub("loader rotate", """        src = iter(make_dataloader(tokenizer, DEVICE_BATCH_SIZE // PACK, SEQ_LEN * PACK, "train", loader_dev, shard_ids=ids))""",
    """        src = iter(make_dataloader(tokenizer, DEVICE_BATCH_SIZE // PACK, SEQ_LEN * PACK, "train", loader_dev, shard_ids=ids,
                                   tokenizer_threads=TOK_THREADS))""")

# ---------------------------------------------------------------------------- prove which value is live
sub("startup print", '''    print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps}")''',
    '''    print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps}")
    print0(f"TOK_THREADS={TOK_THREADS} per rank ({TOK_THREADS * world_size} tokenizer threads over {os.cpu_count()} vCPUs)")''')

OUT.write_text(s)
import ast
ast.parse(s)
print(f"wrote {OUT}  ({len(s.splitlines())} lines, was {len(SRC.read_text().splitlines())})")
for k, v in applied.items():
    print(f"   {k:16s} x{v}")
