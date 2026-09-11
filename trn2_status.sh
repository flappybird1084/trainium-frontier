#!/usr/bin/env bash
# Dev tool (runs ON the box). usage: bash trn2_status.sh <ID>  -> one-line status + flags __DONE__/__ERR__
cd ~/tf; L="runs/$1.log"
echo "procs=$(pgrep -c -f 'native_venv/bin/[p]ython') cc=$(pgrep -c neuronx-cc 2>/dev/null || echo 0) steps=$(grep -cE "^step " $L 2>/dev/null)"; true
grep -E '^step ' "$L" 2>/dev/null | tail -2 | cut -c1-110
grep -q 'public_val_bpb' "$L" 2>/dev/null && { echo "__DONE__"; grep -E "num_steps|training_seconds|startup_allowance|total_tokens|public_val_bpb|causality_passed" "$L"; }
grep -qE 'Traceback|RuntimeError|ValueError|BackendCompilerFailed' "$L" 2>/dev/null && echo "__ERR__"
