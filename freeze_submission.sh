#!/usr/bin/env bash
# usage: bash freeze_submission.sh <version> KEY=VALUE ...   e.g. bash freeze_submission.sh v12 SINK=logit ATTN=nki MUON=fused ADAMW=nki ADAMW_INPLACE=1
# Copies submission_raw/train3_submission.py to submission_raw/submission_<version>_eager_8way.py with the given
# NEURON_COMPETITION_R1_<KEY> defaults baked in, then runs prepare.py check as train.py with the 8-way launch string.
set -eu
V=$1; shift
OUT=submission_raw/submission_${V}_eager_8way.py
cp submission_raw/train3_submission.py "$OUT"
for kv in "$@"; do
  K=${kv%%=*}; VAL=${kv#*=}
  python3 - "$OUT" "$K" "$VAL" <<'PY'
import re, sys
p, k, v = sys.argv[1:]
s = open(p).read()
pat = re.compile(r'((?:_ENV|_F|os\.environ\.get)\("NEURON_COMPETITION_R1_%s", ")([^"]*)("\))' % re.escape(k))
s2, n = pat.subn(lambda m: m.group(1) + v + m.group(3), s)
assert n == 1, (k, n)
open(p, "w").write(s2)
PY
done
python3 -m py_compile "$OUT"
cp "$OUT" /tmp/train.py && PYTHONPATH=. ./.venv/bin/python prepare.py check --train-py /tmp/train.py --launch-command "NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py"
sha256sum "$OUT" | cut -c1-16; grep -nE '(_ENV|_F)\("NEURON_COMPETITION_R1_(SINK|ATTN|MUON|ADAMW|ADAMW_INPLACE|N_EMBD|N_HEAD|EMBED_LR_SCALE|WARMDOWN)"' "$OUT" | cut -c1-100
