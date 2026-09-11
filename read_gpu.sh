#!/usr/bin/env bash
# Rank the local-GPU fixed-step screens by val_bpb. usage: bash read_gpu.sh
for f in runs/gpu/*.log; do [ -f "$f" ] || continue; s=$(grep -a "public_val_bpb" "$f" | grep -oE "[0-9]+\.[0-9]+" | tail -1); st=$(grep -ac "^step " "$f"); e=$(grep -acE "bad loss|Traceback|Error" "$f"); printf "%-14s steps=%-5s %s %s\n" "$(basename "$f" .log)" "${st:-0}" "${s:-running}" "$([ "${e:-0}" -gt 0 ] && echo ERR)"; done | sort -k3 -n
