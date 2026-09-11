#!/usr/bin/env bash
# Dev tool (runs ON the Trn2 box from ~/tf). usage: [LNC=1|2] [NPROC=8|4] [FILE=path.py] bash trn2_run.sh <ID> <MAX_TRAIN_SECONDS|full> [ENV=VAL ...]
# Launches submission_raw/train3_submission.py (PYTHONPATH=~/tf so it imports prepare.py) detached (setsid nohup), logging to runs/<ID>.log. Never run from inside a monitor loop.
set -u
ID=$1; SECS=$2; shift 2
LNC=${LNC:-1}; NPROC=${NPROC:-8}; FILE=${FILE:-submission_raw/train3_submission.py}
cd ~/tf && mkdir -p runs
# HARD LAUNCH GATE (memory trn2-launch-gating): no trainer procs AND no Neuron core owners, else refuse.
P=$(pgrep -c -f "native_venv/bin/[p]ython"); O=$(neuron-ls 2>/dev/null | grep -cE "\| +[0-9]{3,} +\|")   # PID column rows = live core owners
if [ "$P" != 0 ] || [ "$O" != 0 ]; then echo "GATE BLOCKED: procs=$P owners=$O (refusing to launch $ID)"; pgrep -af "native_venv/bin/[p]ython" | cut -c1-140; exit 1; fi
# DISK GUARD (2026-09-08: 9,921 stale /tmp/nki_* kernel build dirs = 40 GB filled the disk and killed a compile): clear
# builds older than 2 h before every launch and refuse below 15 GB free.
find /tmp -maxdepth 1 -name "nki_*" -type d -mmin +120 -exec rm -rf {} + 2>/dev/null
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "${FREE_GB:-0}" -lt 15 ]; then echo "GATE BLOCKED: only ${FREE_GB} GB free on / (refusing to launch $ID)"; exit 1; fi
rm -rf "out_$ID"
ENVS="NEURON_LOGICAL_NC_CONFIG=$LNC NEURON_COMPETITION_R1_OUT_DIR=out_$ID"
[ "$SECS" != "full" ] && ENVS="$ENVS NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=$SECS"
date -u +%FT%TZ > "runs/$ID.start"; echo "nproc=$NPROC file=$FILE $ENVS $*" > "runs/$ID.env"
setsid nohup env PYTHONPATH=$HOME/tf PATH=$HOME/workspace/native_venv/bin:/opt/aws/neuron/bin:$PATH $ENVS "$@" \
  $HOME/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=$NPROC $FILE \
  > "runs/$ID.log" 2>&1 < /dev/null &
sleep 4; echo "$ID launched: $(pgrep -f "$(basename $FILE | sed 's/[.]py$/[.]py/')" | wc -l) procs at $(date -u +%H:%M:%S) | nproc=$NPROC $ENVS $*"
