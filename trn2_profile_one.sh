#!/usr/bin/env bash
# usage: bash prof_one.sh <tag> <neff>  (on the box; capture + summary + parquet export with a timeout + pyarrow aggregation)
set -u
TAG=$1; NEFF=$2; cd ~/tf; mkdir -p runs/prof_$TAG
source /home/ubuntu/workspace/native_venv/bin/activate
[ -f runs/prof_$TAG/profile_exec_2.ntff ] || timeout 900 neuron-profile capture -n "$NEFF" -s runs/prof_$TAG/profile.ntff --profile-nth-exec=2 2>&1 | tail -2
NTFF=$(ls runs/prof_$TAG/*.ntff | head -1)
timeout 600 neuron-explorer view -n "$NEFF" -s "$NTFF" --output-format=summary-text > runs/prof_$TAG/summary.txt 2>&1
rm -rf runs/prof_$TAG/parquet; mkdir -p runs/prof_$TAG/parquet
timeout 300 neuron-explorer view -n "$NEFF" -s "$NTFF" --output-format=parquet --output-file=runs/prof_$TAG/parquet/prof > /dev/null 2>&1
ls runs/prof_$TAG/parquet/prof | wc -l
grep -aE "^    (total_time|tensor_engine_active_time_percent|scalar_engine_active_time_percent|vector_engine_active_time_percent|gpsimd_engine_active_time_percent|dma_active_time_percent|mfu_estimated|sync_engine_active_time_percent)" runs/prof_$TAG/summary.txt
python neff_instr_agg.py runs/prof_$TAG/parquet 22 2>&1 | grep -av "^\s*$"
echo PROFDONE
