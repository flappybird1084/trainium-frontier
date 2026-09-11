#!/usr/bin/env bash
# Dev tool (runs ON the box via trn2_job.sh). usage: bash trn2_profile_neff.sh <tag> <neff-path> [capture|view|both]
# capture: executes the NEFF under neuron-profile (synthetic inputs) -> runs/prof_<tag>/profile_exec_2.ntff (~1 GB).
# view: summary-text to runs/prof_<tag>/summary.txt (console format) and a JSON dump for offline aggregation.
set -u
TAG=$1; NEFF=$2; MODE=${3:-both}; cd ~/tf; mkdir -p runs/prof_$TAG
if [ "$MODE" != view ]; then
  neuron-profile capture -n "$NEFF" -s runs/prof_$TAG/profile.ntff --profile-nth-exec=2 2>&1 | tail -3
fi
NTFF=$(ls runs/prof_$TAG/*.ntff | head -1)
if [ "$MODE" != capture ]; then
  neuron-profile view -n "$NEFF" -s "$NTFF" --output-format summary-text > runs/prof_$TAG/summary.txt 2>&1
  neuron-profile view -n "$NEFF" -s "$NTFF" --output-format json --output-file runs/prof_$TAG/profile.json 2>&1 | tail -2
  ls -la runs/prof_$TAG; head -c 6000 runs/prof_$TAG/summary.txt
fi
