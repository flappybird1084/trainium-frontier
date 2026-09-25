import numpy as np, pyarrow.parquet as pq
t = pq.read_table("/home/ubuntu/tf/runs/prof_V52INSTR/parquet/prof/Instruction.parquet",
                  columns=["start_ts", "end_ts", "engine", "opcode"])
eng = np.array(t["engine"].to_pylist())
s = np.array(t["start_ts"].to_pylist(), dtype=np.int64)
e = np.array(t["end_ts"].to_pylist(), dtype=np.int64)
ops = np.array(t["opcode"].to_pylist())
t0, t1 = s.min(), e.max()
span = (t1 - t0) / 1e6
print(f"span {span:.1f} ms over {len(s)} instructions, {len(set(eng))} engines\n")

names = sorted(set(eng))
# merge each engine's intervals into disjoint busy intervals
merged = {}
for n in names:
    m = eng == n
    a, b = np.sort(s[m]), None
    idx = np.argsort(s[m]); a = s[m][idx]; b = e[m][idx]
    out_a, out_b = [], []
    ca, cb = a[0], b[0]
    for i in range(1, len(a)):
        if a[i] <= cb:
            cb = max(cb, b[i])
        else:
            out_a.append(ca); out_b.append(cb); ca, cb = a[i], b[i]
    out_a.append(ca); out_b.append(cb)
    merged[n] = (np.array(out_a), np.array(out_b))
    busy = (np.array(out_b) - np.array(out_a)).sum() / 1e6
    print(f"  {n:8s} busy {busy:7.1f} ms  ({busy/span*100:4.1f}% of span)")

# sweep line: how many engines busy at each moment, and who is alone
evs = []
for n in names:
    a, b = merged[n]
    for x in a: evs.append((x, 1, n))
    for x in b: evs.append((x, -1, n))
evs.sort()
active, prev, alone, nbusy_time = set(), None, {n: 0.0 for n in names}, {}
for x, d, n in evs:
    if prev is not None and x > prev:
        dt = (x - prev) / 1e6
        k = len(active)
        nbusy_time[k] = nbusy_time.get(k, 0.0) + dt
        if k == 1:
            alone[next(iter(active))] += dt
    if d == 1: active.add(n)
    else: active.discard(n)
    prev = x

print("\nconcurrency (how many engines busy simultaneously):")
for k in sorted(nbusy_time):
    print(f"  {k} engine(s) busy: {nbusy_time[k]:7.1f} ms  ({nbusy_time[k]/span*100:4.1f}%)")
print("\nEXCLUSIVE time - the ONLY engine working, i.e. where deleting its work shortens the wall:")
for n in sorted(alone, key=lambda z: -alone[z]):
    print(f"  {n:8s} {alone[n]:7.1f} ms  ({alone[n]/span*100:4.1f}% of span)")
idle = nbusy_time.get(0, 0.0)
print(f"\n  NOTHING busy (pure gaps/latency): {idle:.1f} ms ({idle/span*100:.1f}%)")
