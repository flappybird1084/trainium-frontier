#!/usr/bin/env bash
# Dev tool (runs ON the box via trn2_job.sh, one core). usage: bash trn2_profile_ops.sh <tag> <neff-path>
# Captures a profile of the NEFF (synthetic inputs, 2nd exec), writes summary-text + parquet, then aggregates per op
# (neff_ops_agg.py): where the compiled graph's time / FLOPs go, by framework op.
set -u
TAG=$1; NEFF=$2; cd ~/tf; mkdir -p runs/prof_$TAG
if [ ! -f runs/prof_$TAG/profile.ntff ]; then
  neuron-profile capture -n "$NEFF" -s runs/prof_$TAG/profile.ntff --profile-nth-exec=2 2>&1 | tail -3
fi
ls -la runs/prof_$TAG/*.ntff || { echo "capture failed"; exit 1; }
NTFF=$(ls runs/prof_$TAG/*.ntff | head -1)
neuron-explorer view -n "$NEFF" -s "$NTFF" --output-format=summary-text > runs/prof_$TAG/summary.txt 2>&1; tail -2 runs/prof_$TAG/summary.txt
rm -rf runs/prof_$TAG/parquet; mkdir -p runs/prof_$TAG/parquet
neuron-explorer view -n "$NEFF" -s "$NTFF" --output-format=parquet --output-file=runs/prof_$TAG/parquet/prof 2>&1 | tail -2
ls -la runs/prof_$TAG runs/prof_$TAG/parquet | head -20
python neff_ops_agg.py runs/prof_$TAG/parquet
