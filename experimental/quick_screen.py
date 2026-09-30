"""Local CUDA screen of a submission file, unmodified. One GPU process that plays rank 0 of a 2-worker launch:
it trains on the even row groups (prepare's loader reads RANK/WORLD_SIZE from env) and reads the odd workers' n-gram
table, exactly the device's cross-fitting parity. Both halves of the table are counted BEFORE training starts (the
device has them by step ~1), and the schedule runs by step count (FRAC_BY_STEPS), so a GPU shared with another job
changes the wall time, not the result. Patched: init_runtime (single device, no process group), micro-batch rows,
TOTAL_BATCH_SIZE (LRs stay the file's own), FRAC_BY_STEPS. Usage: see quick_screen.sh."""
import importlib.util, multiprocessing, os, sys, time
from pathlib import Path

path, out_dir = sys.argv[1], sys.argv[2]
os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="2")
spec = importlib.util.spec_from_file_location("screened", path)
m = importlib.util.module_from_spec(spec)
sys.modules["screened"] = m
spec.loader.exec_module(m)


def init_runtime(device_type):
    import torch
    torch.manual_seed(int(os.environ.get("QS_SEED", "42")))   # the file seeds 42
    return False, 0, 0, 1, torch.device("cuda", 0)


m.init_runtime = init_runtime
m.DEVICE_BATCH_ROWS = int(os.environ.get("QS_ROWS", "4"))       # micro-batch rows (grad accumulation absorbs it)
m.TOTAL_BATCH_SIZE = int(os.environ.get("QS_BATCH", "32768"))   # tokens per optimizer step (device: 131072)
m.FRAC_BY_STEPS = True

ng_dir = Path(out_dir) / f"ngram_{os.getppid()}"
ng_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()
ctx = multiprocessing.get_context("fork")
builders = [ctx.Process(target=m.ngram_build, args=(r, 2, ng_dir), daemon=True) for r in (0, 1)]
for b in builders:
    b.start()
for b in builders:
    b.join()
if (ng_dir / "error").exists():
    raise SystemExit(f"n-gram build failed: {(ng_dir / 'error').read_text()}")
print(f"QS n-gram tables built in {time.time() - t0:.0f}s", flush=True)
m.ngram_build = lambda *a: None      # main() still forks its helper (so it loads the tables); it has nothing to do
sys.argv = [path, "--device-type", "cuda", "--out-dir", out_dir, "--max-train-seconds", "1000000"] + sys.argv[3:]
m.main()
