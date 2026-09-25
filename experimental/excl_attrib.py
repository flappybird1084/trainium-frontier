import numpy as np, pyarrow.parquet as pq
t = pq.read_table("/home/ubuntu/tf/runs/prof_V52INSTR/parquet/prof/Instruction.parquet",
                  columns=["start_ts","end_ts","engine","opcode","nki_source_location"])
eng = np.array(t["engine"].to_pylist()); s = np.array(t["start_ts"].to_pylist(), dtype=np.int64)
e = np.array(t["end_ts"].to_pylist(), dtype=np.int64); ops = np.array(t["opcode"].to_pylist())
loc = np.array([str(x) for x in t["nki_source_location"].to_pylist()])
names = sorted(set(eng))

def merge(a, b):
    i = np.argsort(a); a, b = a[i], b[i]
    oa, ob = [], []; ca, cb = a[0], b[0]
    for k in range(1, len(a)):
        if a[k] <= cb: cb = max(cb, b[k])
        else: oa.append(ca); ob.append(cb); ca, cb = a[k], b[k]
    oa.append(ca); ob.append(cb); return np.array(oa), np.array(ob)

merged = {n: merge(s[eng==n], e[eng==n]) for n in names}
# sweep to find, per engine, the intervals where it is the ONLY engine busy
evs = []
for n in names:
    a, b = merged[n]
    evs += [(x, 1, n) for x in a] + [(x, -1, n) for x in b]
evs.sort()
excl = {n: [] for n in names}
active, prev = set(), None
for x, d, n in evs:
    if prev is not None and x > prev and len(active) == 1:
        excl[next(iter(active))].append((prev, x))
    active.add(n) if d == 1 else active.discard(n)
    prev = x

def cover_fn(iv):
    if not iv: return None
    a = np.array([p[0] for p in iv]); b = np.array([p[1] for p in iv])
    pref = np.concatenate([[0], np.cumsum(b - a)])
    return a, b, pref

def covered(x, cf):                      # exclusive time strictly before x
    a, b, pref = cf
    j = np.searchsorted(a, x, side="right") - 1
    base = pref[np.maximum(j, 0)] * (j >= 0)
    part = np.where(j >= 0, np.clip(x - a[np.maximum(j,0)], 0, b[np.maximum(j,0)] - a[np.maximum(j,0)]), 0)
    return base + part

print("EXCLUSIVE-time attribution: where the wall actually shortens if you delete the work\n")
for n in ("Vector", "Tensor", "Scalar"):
    cf = cover_fn(excl[n])
    if cf is None: continue
    m = eng == n
    ov = covered(e[m], cf) - covered(s[m], cf)          # each instruction's overlap with exclusive windows
    tot = ov.sum() / 1e6
    print(f"== {n}: {tot:.1f} ms exclusive, over {m.sum()} instructions")
    key = np.array([f"{a.split('/')[-1]}" for a in loc[m]])
    agg = {}
    for k, v, o in zip(key, ops[m], ov):
        agg[(k, v)] = agg.get((k, v), 0.0) + o
    for (k, v), o in sorted(agg.items(), key=lambda z: -z[1])[:8]:
        print(f"   {o/1e6:6.2f} ms  {v:22s} {k[:58]}")
    print()
