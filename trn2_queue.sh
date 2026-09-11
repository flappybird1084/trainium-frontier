#!/usr/bin/env bash
# Dev tool (runs ON the Trn2 box from ~/tf). usage: setsid nohup bash trn2_queue.sh <queue-file> > runs/queue.log 2>&1 &
# Each non-empty, non-# line of <queue-file> is "<ID> <SECS|full> [LNC=..] [NPROC=..] [FILE=..] [ENV=VAL ...]".
# For each line: wait until the device is free (no venv python procs, no neuron-ls owners), launch it through the
# gated trn2_run.sh, then wait for that run to exit. Never assumes the device is free; never skips the gate.
set -u
Q=$1
free() { [ "$(pgrep -c -f 'native_venv/bin/[p]ython')" = 0 ] && [ "$(neuron-ls 2>/dev/null | grep -cE '\| +[0-9]{3,} +\|')" = 0 ]; }
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  set -- $line
  if [ "$1" = "WAIT" ]; then  # wait for all parallel jobs (PJOB) to finish: their core lock files exist from launch to exit
    sleep 5; while ls runs/.core_*.lock >/dev/null 2>&1 || [ "$(pgrep -c -f 'native_venv/bin/[p]ython')" != 0 ]; do sleep 10; done
    echo "$(date -u +%T) QUEUE: jobs done"; continue
  fi
  if [ "$1" = "PJOB" ]; then  # "PJOB <ID> <command...>": launch a job on the next free logical core WITHOUT waiting
    shift; ID=$1; shift
    w=0; until ! pgrep -f "[t]orchrun --standalone" >/dev/null; do sleep 15; w=$((w+15)); [ $w -ge 7200 ] && { echo "$(date -u +%T) QUEUE: device busy for 2h, giving up before $ID"; exit 1; }; done
    for _ in $(seq 1 240); do LNC=${LNC:-2} bash trn2_job.sh "$ID" "$@" && break; sleep 15; done
    echo "$(date -u +%T) QUEUE: launched parallel job $ID"; sleep 5; continue
  fi
  if [ "$1" = "JOB" ]; then   # "JOB <ID> <command...>": a gated single-process device job (trn2_job.sh), waited for
    shift; ID=$1; shift
    w=0; until free; do sleep 15; w=$((w+15)); [ $w -ge 7200 ] && { echo "$(date -u +%T) QUEUE: device busy for 2h, giving up before $ID"; exit 1; }; done
    sleep 5; echo "$(date -u +%T) QUEUE: launching job $ID"
    LNC=${LNC:-2} bash trn2_job.sh "$ID" "$@" || { echo "$(date -u +%T) QUEUE: job $ID refused"; exit 1; }
    sleep 20; while [ "$(pgrep -c -f 'native_venv/bin/[p]ython')" != 0 ]; do sleep 10; done
    echo "$(date -u +%T) QUEUE: job $ID finished: $(grep -cE 'Traceback|Error' runs/$ID.log) errors"; continue
  fi
  ID=$1; SECS=$2; shift 2
  LNC=1; NPROC=8; FILE=submission_raw/train3_submission.py; ENVS=""
  for kv in "$@"; do case "$kv" in LNC=*) LNC=${kv#LNC=};; NPROC=*) NPROC=${kv#NPROC=};; FILE=*) FILE=${kv#FILE=};; *) ENVS="$ENVS $kv";; esac; done
  w=0; until free; do sleep 15; w=$((w+15)); [ $w -ge 7200 ] && { echo "$(date -u +%T) QUEUE: device busy for 2h, giving up before $ID"; exit 1; }; done
  sleep 5; free || { echo "$(date -u +%T) QUEUE: device became busy again before $ID"; exit 1; }
  echo "$(date -u +%T) QUEUE: launching $ID"
  LNC=$LNC NPROC=$NPROC FILE=$FILE bash trn2_run.sh "$ID" "$SECS" $ENVS || { echo "$(date -u +%T) QUEUE: launch of $ID refused"; exit 1; }
  sleep 30
  while [ "$(pgrep -c -f 'native_venv/bin/[p]ython')" != 0 ]; do sleep 15; done
  echo "$(date -u +%T) QUEUE: $ID finished: $(grep -E 'public_val_bpb' runs/$ID.log | tail -1) $(grep -cE 'Traceback|RuntimeError' runs/$ID.log) errors"
done < "$Q"
echo "$(date -u +%T) QUEUE: done"
