"""Dev tool (box, one core, via trn2_job.sh). usage: python trn2_profile_sum.py <tag> <neff>
Capture (if missing) + summary-text of a NEFF; waits for the explorer's background worker to exit so the queue's
job wait covers it (neuron-explorer view returns immediately and finishes in a detached process)."""
import glob, os, subprocess, sys, time
tag, neff = sys.argv[1], sys.argv[2]
d = f"runs/prof_{tag}"; os.makedirs(d, exist_ok=True)
if not glob.glob(f"{d}/*.ntff"):
    r = subprocess.run(["neuron-profile", "capture", "-n", neff, "-s", f"{d}/profile.ntff", "--profile-nth-exec=2"], capture_output=True, text=True)
    print(r.stdout[-600:], r.stderr[-600:])
ntff = (glob.glob(f"{d}/*.ntff") or [None])[0]
if not ntff:
    print("capture failed"); sys.exit(1)
with open(f"{d}/summary.txt", "w") as f:
    subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", ntff, "--output-format=summary-text"], stdout=f, stderr=subprocess.STDOUT)
for _ in range(240):   # wait for the detached worker
    if subprocess.run(["pgrep", "-f", "[n]euron-explorer view"], capture_output=True).returncode != 0:
        break
    time.sleep(5)
txt = open(f"{d}/summary.txt").read()
keys = ("total_time", "dma_active_time_percent", "tensor_engine", "pe_active", "hbm_read_bytes", "hbm_write_bytes", "sbuf_read_bytes",
        "mfu_estimated", "mfu_max", "hardware_flops", "transpose_flops", "mm_arithmetic")
for line in txt.splitlines():
    if any(k in line for k in keys):
        print("SUM", line.strip()[:140])
print("done")
