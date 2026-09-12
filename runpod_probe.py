import asyncio
from runpod_flash import Endpoint, GpuType

@Endpoint(name="tf_probe", gpu=GpuType.NVIDIA_RTX_PRO_6000_BLACKWELL_SERVER_EDITION,
          dependencies=["torch", "numpy", "tiktoken", "pyarrow", "requests"],
          idle_timeout=30, execution_timeout_ms=300000)
async def probe():
    import torch, os, shutil, subprocess, sys
    info = {}
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA"
    info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
    info["torch"] = torch.__version__
    try:
        with open("/proc/meminfo") as f:
            info["ram_gb"] = round(int([l for l in f if l.startswith("MemTotal")][0].split()[1]) / 1e6, 1)
    except Exception as e:
        info["ram_gb"] = str(e)
    total, used, free = shutil.disk_usage("/")
    info["disk_free_gb"] = round(free / 1e9, 1)
    info["cwd"] = os.getcwd()
    for mod in ("tiktoken", "pyarrow", "requests", "numpy"):
        try:
            __import__(mod); info[mod] = "ok"
        except Exception as e:
            info[mod] = f"MISSING {e}"
    info["internet"] = subprocess.run(["python", "-c", "import requests; print(requests.get('https://huggingface.co', timeout=10).status_code)"],
                                      capture_output=True, text=True).stdout.strip()
    return info

r = asyncio.run(probe())
print("PROBE RESULT:", r)
