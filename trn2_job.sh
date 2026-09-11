#!/usr/bin/env bash
# Dev tool (runs ON the Trn2 box from ~/tf). usage: [LNC=2] [CORES=<logical core idx>] bash trn2_job.sh <ID> <command...>
# Gated single-process device job (NKI tests, profiler captures). Gate: never while a training run (torchrun) holds the
# device; other jobs may run concurrently on different logical cores (pinned with NEURON_RT_VISIBLE_CORES, tracked by
# runs/.core_<n>.lock). Without CORES= the job takes the lowest free logical core (LNC=2: 4 logical cores = 4 jobs).
set -u
ID=$1; shift
cd ~/tf && mkdir -p runs
if pgrep -f "[t]orchrun --standalone" >/dev/null || pgrep -f "[s]ubmission_raw/train3_submission.py" >/dev/null || pgrep -f "[s]ubmission_v[0-9]*_.*\.py" >/dev/null; then
  echo "GATE BLOCKED: a training run holds the device (refusing $ID)"; exit 1; fi
NCORES=$(( 8 / ${LNC:-2} ))
for f in runs/.core_*.lock; do [ -f "$f" ] && ! kill -0 "$(cat "$f" 2>/dev/null)" 2>/dev/null && rm -f "$f"; done   # stale locks
if [ -z "${CORES:-}" ]; then
  for c in $(seq 0 $((NCORES-1))); do [ ! -f runs/.core_$c.lock ] && { CORES=$c; break; }; done
  [ -z "${CORES:-}" ] && { echo "GATE BLOCKED: all $NCORES logical cores busy with jobs (refusing $ID)"; exit 1; }
fi
[ -f runs/.core_$CORES.lock ] && { echo "GATE BLOCKED: core $CORES busy (refusing $ID)"; exit 1; }
date -u +%FT%TZ > "runs/$ID.start"; echo "job: CORES=$CORES $*" > "runs/$ID.env"
# PATH-based lookup of `python` failed under `env PATH=... bash -c` on this box; activating the venv works.
setsid nohup env PYTHONPATH=$HOME/tf NEURON_LOGICAL_NC_CONFIG=${LNC:-2} NEURON_RT_VISIBLE_CORES=$CORES \
  bash -c "echo \$\$ > runs/.core_$CORES.lock; source $HOME/workspace/native_venv/bin/activate && $*; rm -f runs/.core_$CORES.lock" \
  > "runs/$ID.log" 2>&1 < /dev/null &
sleep 2; echo "$ID launched at $(date -u +%H:%M:%S) on logical core $CORES: $*"
