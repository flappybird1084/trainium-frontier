import asyncio
from runpod_flash import Endpoint, GpuType

BLACKWELL = GpuType.NVIDIA_RTX_PRO_6000_BLACKWELL_SERVER_EDITION
DEPS = ["torch", "numpy", "tiktoken", "pyarrow", "requests"]


@Endpoint(name="tf_train", gpu=BLACKWELL, dependencies=DEPS, idle_timeout=30, execution_timeout_ms=3600000)
async def train(files: dict, shards: int = 2, steps: int = 400, base_env: dict = None, configs: list = None):
    import base64, os, subprocess, re, zipfile, io, json, time
    os.chdir("/app")
    # write shipped code
    for name, b64 in files.items():
        data = base64.b64decode(b64)
        path = os.path.join("/app", name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    # tokenizer artifact
    os.makedirs("/app/artifacts", exist_ok=True)
    if "artifacts/tokenizer_8192.zip" in files:
        pass  # already written above
    env0 = dict(os.environ); env0["PYTHONPATH"] = "/app"
    # data
    t0 = time.time()
    setup = subprocess.run(["python", "prepare.py", "setup", "--num-train-shards", str(shards)],
                           cwd="/app", env=env0, capture_output=True, text=True, timeout=1800)
    setup_log = (setup.stdout + setup.stderr)[-500:]
    results = {"_setup_seconds": round(time.time() - t0, 1), "_setup_tail": setup_log, "_gpu": None}
    import torch
    results["_gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda"
    for cfg in (configs or []):
        env = dict(env0)
        for k, v in (base_env or {}).items():
            env["NEURON_COMPETITION_R1_" + k] = str(v)
        for k, v in cfg.get("env", {}).items():
            env["NEURON_COMPETITION_R1_" + k] = str(v)
        env["NEURON_COMPETITION_R1_NUM_STEPS"] = str(steps)
        env["NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS"] = "36000"
        env["NEURON_COMPETITION_R1_COMPILE"] = "0"
        env["NEURON_COMPETITION_R1_OUT_DIR"] = "/app/out_" + cfg["name"]
        env["NEURON_COMPETITION_R1_EVAL_PUBLIC"] = "1"
        t1 = time.time()
        r = subprocess.run(["python", "submission_raw/train3_submission.py"], cwd="/app", env=env,
                           capture_output=True, text=True, timeout=3000)
        out = r.stdout + r.stderr
        m = re.search(r"public_val_bpb:\s*([0-9.]+)", out)
        laststep = re.findall(r"^step (\d+) \| loss ([0-9.]+)", out, re.M)
        results[cfg["name"]] = {"val_bpb": float(m.group(1)) if m else None,
                                "steps": int(laststep[-1][0]) if laststep else 0,
                                "last_loss": float(laststep[-1][1]) if laststep else None,
                                "secs": round(time.time() - t1, 1),
                                "err_tail": None if m else out[-400:]}
        subprocess.run(["rm", "-rf", env["NEURON_COMPETITION_R1_OUT_DIR"]])
    return results
