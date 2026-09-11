#!/usr/bin/env bash
# Rank the overnight sweep by public_val_bpb. usage: bash read_sweep.sh   (reads the box logs)
ssh -i trn2.pem -o StrictHostKeyChecking=no ubuntu@54.20.68.225 'cd ~/tf; for f in runs/S_H*.log runs/S_C*.log runs/S_REF*.log; do [ -f "$f" ] || continue; s=$(grep -a "public_val_bpb" "$f" | tail -1 | grep -oE "[0-9]+\.[0-9]+"); e=$(grep -acE "bad loss|Traceback|Failed to allocate|out-of-bound" "$f"); printf "%-22s %s %s\n" "$(basename "$f" .log)" "${s:-CRASH}" "$([ "${e:-0}" -gt 0 ] && echo ERR)"; done | sort -k2 -n'
