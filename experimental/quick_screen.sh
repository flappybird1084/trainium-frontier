#!/usr/bin/env bash
# Usage: experimental/quick_screen.sh <submission.py> [steps=500] [eval_tokens=524288]
# Local-GPU screen: the submission file itself, n-gram tables counted in-run (before step 0), a fixed number of
# 32,768-token steps on its own step-based LR/EMA schedule, then prepare.evaluate_bpb on the public val shard.
# ~12-13 min wall on a free RTX 4080 SUPER at 500 steps; ~10 GB GPU, ~5 GB RAM. Prints public_val_bpb.
# Env: QS_SEED (init seed, default 42 = the file's), QS_BATCH (tokens/step, default 32768), QS_ROWS (micro rows, default 4).
set -euo pipefail
f=$1; N=${2:-500}; E=${3:-524288}
root=$(cd "$(dirname "$0")/.." && pwd)
out=/tmp/quick_screen/$(basename "$f" .py)_N${N}_B${QS_BATCH:-32768}_s${QS_SEED:-42}; mkdir -p "$out"
cd "$root"
start=$(date +%s)
PYTHONPATH="$root" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$root/.venv/bin/python" experimental/quick_screen.py "$f" "$out" --num-steps "$N" --eval-public --eval-tokens "$E" \
  > "$out/log.txt" 2>&1 || echo "RUN FAILED (see $out/log.txt)"
rm -rf "$out/final.pt" "$out"/ngram_*
grep -E "QS n-gram|num_steps:|public_val_bpb|Traceback|Error" "$out/log.txt" || true
echo "nan_steps: $(grep -E '^step' "$out/log.txt" | grep -ci nan)  first: $(grep -E '^step' "$out/log.txt" | grep -i nan | head -1 | cut -c1-40)"
echo "wall_seconds: $(( $(date +%s) - start ))   log: $out/log.txt"
