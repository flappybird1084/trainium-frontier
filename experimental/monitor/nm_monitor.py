#!/usr/bin/env python3
"""Compact 2s utilization lines from `stdbuf -oL neuron-monitor | python3 nm_monitor.py`.
Prints: time | NC avg util | per-core util | device/host mem | completed execs | CPU% | RAM. Dev tool, not a submission file."""
import sys, json, time
GB = 1024**3; PERIOD = 2.0
def ram_fallback():
    try:
        m = {}
        for line in open('/proc/meminfo'):
            k, v = line.split(':'); m[k] = int(v.split()[0]) * 1024
        return m['MemTotal'] - m['MemAvailable'], m['MemTotal']
    except Exception: return None, None
last = 0.0
for raw in sys.stdin:
    raw = raw.strip()
    if not raw: continue
    try: d = json.loads(raw)
    except Exception: continue
    now = time.time()
    if now - last < PERIOD - 0.1: continue
    last = now
    cores, dev, host, done = {}, 0, 0, 0
    for rt in d.get('neuron_runtime_data', []) or []:
        rep = rt.get('report', {}) or {}
        for idx, c in ((rep.get('neuroncore_counters', {}) or {}).get('neuroncores_in_use', {}) or {}).items():
            cores.setdefault(int(idx), []).append(float(c.get('neuroncore_utilization', 0)))
        mu = (rep.get('memory_used', {}) or {}).get('neuron_runtime_used_bytes', {}) or {}
        dev += mu.get('neuron_device', 0) or 0; host += mu.get('host', 0) or 0
        done += ((rep.get('execution_stats', {}) or {}).get('execution_summary', {}) or {}).get('completed', 0) or 0
    sysd = d.get('system_data', {}) or {}
    idle = (sysd.get('vcpu_usage', {}) or {}).get('average_usage', {}) or {}
    cpu = f"{100 - float(idle.get('idle', 0)):5.1f}%" if idle else "  n/a"
    mi = sysd.get('memory_info', {}) or {}
    ru, rt_ = mi.get('memory_used_bytes'), mi.get('memory_total_bytes')
    if not rt_: ru, rt_ = ram_fallback()
    ram = f"{ru/GB:5.1f}/{rt_/GB:.0f}G" if rt_ else "n/a"
    if cores:
        util = ' '.join(f"c{i}:{sum(v)/len(v):5.1f}" for i, v in sorted(cores.items()))
        avg = sum(sum(v)/len(v) for v in cores.values())/len(cores)
    else: util, avg = "idle (no runtime)", 0.0
    print(f"{time.strftime('%H:%M:%S')} | NC avg {avg:5.1f}% | {util} | dev {dev/GB:5.2f}G host {host/GB:4.1f}G | execs {done} | CPU {cpu} | RAM {ram}", flush=True)
