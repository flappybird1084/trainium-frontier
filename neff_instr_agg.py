"""Aggregate a neuron-explorer Instruction.parquet by engine and by every low-cardinality string column (framework op, hlo name,
opcode ...): summed instruction time, count. pyarrow only (the box venv has no pandas).
usage: python neff_instr_agg.py <prof dir with parquet/Instruction.parquet> [top]"""
import sys, os, glob
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
d = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
f = glob.glob(os.path.join(d, "**", "Instruction.parquet"), recursive=True)[0]
t = pq.read_table(f)
print(f"{f}: {t.num_rows} rows; columns: {t.column_names}")
tot = pc.sum(t["duration_ns"]).as_py()
print(f"total instruction time {tot/1e6:.1f} ms (sum over engines, overlapping)")
for col in t.column_names:
    if col in ("duration_ns", "start_ts", "end_ts", "timeline_track", "pcore_idx"): continue
    typ = t.schema.field(col).type
    if not (pa.types.is_string(typ) or pa.types.is_large_string(typ) or pa.types.is_integer(typ)): continue
    n = pc.count_distinct(t[col]).as_py()
    if n < 2 or n > 5000: continue
    g = t.group_by(col).aggregate([("duration_ns", "sum"), ("duration_ns", "count")]).sort_by([("duration_ns_sum", "descending")])
    print(f"\n== by {col} ({n} distinct)")
    for i in range(min(top, g.num_rows)):
        print(f"  {g['duration_ns_sum'][i].as_py()/1e6:8.2f} ms {g['duration_ns_count'][i].as_py():8d}  {str(g[col][i].as_py())[:110]}")
