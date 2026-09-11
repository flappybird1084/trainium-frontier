#!/bin/bash
# Probe driver for the Trn2 native beta. Waits for any competition run to release
# the device, then Tier 1 (single-proc eager), then Tier 2 (dist topologies),
# each bounded by timeout so hangs become labeled results (exit 124), not stalls.
V=$HOME/workspace/native_venv/bin
cd $HOME/frontier || exit 1

waited=0
while pgrep -f "train\.py|train3_submission" >/dev/null; do
  if [ $waited -ge 1200 ]; then
    echo "DRIVER: waited 20 min; killing lingering run to free the device"
    pkill -f "train\.py|train3_submission"; sleep 10; break
  fi
  sleep 20; waited=$((waited+20))
done

echo "===== TIER1 single-process eager probes ====="
timeout 1200 $V/python -u neuron_probes.py 2>&1 | grep -vE "Warning|warn|registered|dispatch|kernel:|UserWarning|torch.tensor"
echo "tier1_exit=$?"

echo "===== TIER2 distributed probes ====="
for cfg in "1:2" "1:4" "1:8" "2:4"; do
  lnc="${cfg%%:*}"; n="${cfg##*:}"
  echo "--- dist probe LNC=$lnc nproc=$n ---"
  rm -f /tmp/neuron_compile_service* /tmp/neuron_hlo*
  NEURON_LOGICAL_NC_CONFIG=$lnc timeout 420 $V/torchrun --standalone --nproc_per_node=$n dist_probe.py 2>&1 | grep -E "\[r|SUCCESS|Error|Traceback|exitcode" | tail -25
  ec=$?
  echo "dist_probe LNC=$lnc nproc=$n exit=$ec $([ $ec -eq 124 ] && echo '(TIMEOUT=HANG)')"
done
echo "ALL_PROBES_DONE"
