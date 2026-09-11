"""Dev tool (runs ON the box via trn2_job.sh, one core; a python process so the queue runner waits for it).
usage: python trn2_profile_ops.py <tag> <neff-path>
Captures a profile of the NEFF if runs/prof_<tag>/*.ntff is missing, exports summary-text + parquet with neuron-explorer,
then aggregates time / FLOPs per op (neff_ops_agg.py)."""
import glob, os, subprocess, sys

tag, neff = sys.argv[1], sys.argv[2]
d = f"runs/prof_{tag}"
os.makedirs(d, exist_ok=True)
ntff = glob.glob(f"{d}/*.ntff")
if not ntff:
    r = subprocess.run(["neuron-profile", "capture", "-n", neff, "-s", f"{d}/profile.ntff", "--profile-nth-exec=2"],
                       capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-800:])
    ntff = glob.glob(f"{d}/*.ntff")
    if not ntff:
        print("capture failed"); sys.exit(1)
ntff = ntff[0]
with open(f"{d}/summary.txt", "w") as f:
    subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", ntff, "--output-format=summary-text"], stdout=f, stderr=subprocess.STDOUT)
print(open(f"{d}/summary.txt").read()[-1500:])
pq = f"{d}/parquet"
subprocess.run(["rm", "-rf", pq]); os.makedirs(pq)
r = subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", ntff, "--output-format=parquet", f"--output-file={pq}/prof"],
                   capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
subprocess.run(["bash", "-c", f"ls -la {pq} {pq}/prof 2>/dev/null | head -30"])
subprocess.run([sys.executable, "neff_ops_agg.py", pq])
