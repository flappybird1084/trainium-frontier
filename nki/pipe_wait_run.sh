#!/bin/bash
# Gated waiter for the tile-pipelining MLP bench (single process, one core). Launches ONLY after three consecutive 60 s
# polls with zero torchrun procs, zero neuron-ls core owners and no trn2_queue.sh runner alive (the third gate covers the
# gap between queued jobs). Logs: ~/tf/runs/pipe_wait.log (polls) and ~/tf/runs/pipe_bench.log (bench output).
cd ~/tf || exit 1
source ~/workspace/native_venv/bin/activate
export PYTHONPATH=~/tf
mkdir -p ~/tf/runs
WLOG=~/tf/runs/pipe_wait.log
LOG=~/tf/runs/pipe_bench.log
clean=0
while true; do
  a=$(pgrep -f "torchrun --standalone" | wc -l)
  b=$(neuron-ls 2>/dev/null | grep -c native_venv)
  c=$(pgrep -f "trn2_queue.sh" | wc -l)
  if [ "$a" = 0 ] && [ "$b" = 0 ] && [ "$c" = 0 ]; then clean=$((clean + 1)); else clean=0; fi
  echo "$(date -u +%H:%M:%S) torchrun=$a owners=$b queue=$c clean=$clean" >> "$WLOG"
  [ "$clean" -ge 3 ] && break
  sleep 60
done
echo "=== launching bench $(date -u)" >> "$WLOG"
timeout 2400 python nki/nki_pipe_mlp_bench.py 16384 20 > "$LOG" 2>&1
echo "=== bench exit $? $(date -u)" >> "$WLOG"
