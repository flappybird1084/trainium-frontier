"""Aggregate a neuron-explorer parquet profile by op: time and FLOPs per framework op / instruction type.
usage: python neff_ops_agg.py <parquet dir or file>  (schema-agnostic: prints columns, then groups by every name-like column)"""
import sys, glob, os
import pyarrow.parquet as pq
import pyarrow as pa

path = sys.argv[1]
files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)) if os.path.isdir(path) else [path]
print(f"{len(files)} parquet files")
tables = []
for f in files:
    t = pq.read_table(f)
    print(f"  {os.path.basename(f)}: {t.num_rows} rows, cols={t.column_names[:40]}")
    tables.append((f, t))
import pandas as pd
for f, t in tables:
    df = t.to_pandas()
    dur_col = next((c for c in df.columns if c.lower() in ("duration", "duration_ns", "dur", "time_ns", "duration_us")), None)
    if dur_col is None:
        dur_col = next((c for c in df.columns if "dur" in c.lower()), None)
    if dur_col is None:
        continue
    print(f"\n=== {os.path.basename(f)}: total {dur_col} = {df[dur_col].sum():,.0f}")
    name_cols = [c for c in df.columns if any(k in c.lower() for k in ("op", "name", "type", "engine", "instr", "layer", "framework"))]
    flop_col = next((c for c in df.columns if "flop" in c.lower()), None)
    for c in name_cols[:8]:
        if df[c].nunique() > 5000 or df[c].nunique() < 2:
            continue
        g = df.groupby(c)[dur_col].agg(["sum", "count"]).sort_values("sum", ascending=False)
        g["pct"] = 100 * g["sum"] / df[dur_col].sum()
        if flop_col:
            g["flops"] = df.groupby(c)[flop_col].sum()
        print(f"\n--- by {c} (top 25 of {len(g)}):")
        print(g.head(25).to_string(max_colwidth=70))
