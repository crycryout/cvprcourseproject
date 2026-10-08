#!/usr/bin/env python3
"""Read-only probes plus an optional JSON report; never install or download."""
import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command(args, timeout=20):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()[:1000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": type(exc).__name__}


def collect(cuda_smoke=False, device=0):
    packages = {}
    for name in ("torch", "torchvision", "transformers", "pycocotools", "numpy", "pandas", "matplotlib", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "protocol_version": 2,
        "cuda_smoke_requested": cuda_smoke,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "environment_probe_not_inference_benchmark",
        "python": platform.python_version(), "system": platform.system(),
        "architecture": platform.machine(), "packages": packages,
        "working_disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / 1e9, 2),
    }
    if shutil.which("nvidia-smi"):
        report["gpus"] = command([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ])
    else:
        report["gpus"] = {"available": False, "reason": "nvidia-smi not found"}
    probe = r'''
import json
try:
    import torch
    result = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    if result["cuda_available"] and RUN_SMOKE:
        torch.cuda.set_device(DEVICE_INDEX)
        result["visible_devices"] = torch.cuda.device_count()
        x = torch.ones((16, 16), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        result["cuda_smoke_ok"] = bool(torch.all(y == 16).item())
        result["bf16_supported"] = torch.cuda.is_bf16_supported()
        result["device_name"] = torch.cuda.get_device_name(DEVICE_INDEX)
    print(json.dumps(result))
except Exception as exc:
    print(json.dumps({"cuda_available": False, "error_type": type(exc).__name__}))
'''
    probe = "RUN_SMOKE = " + repr(cuda_smoke) + "\nDEVICE_INDEX = " + str(device) + "\n" + probe
    run = command([sys.executable, "-c", probe], timeout=45)
    try:
        report["torch_probe"] = json.loads(run.get("stdout", ""))
    except json.JSONDecodeError:
        report["torch_probe"] = {"cuda_available": False, "probe_failed": True,
                                 "returncode": run.get("returncode")}
    report["cuda_smoke_passed"] = bool(report["torch_probe"].get("cuda_smoke_ok", False))
    report["ready_for_experiments"] = False
    report["next_gate"] = "M0 v2: verify DETR/processor, COCO split, quality and isolated environment"
    report["note"] = "CUDA smoke alone does not verify runtime readiness or an idle GPU."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cuda-smoke", action="store_true", help="Opt in to a small GPU computation after checking the device is available")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index for explicit smoke test")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.require_cuda and not args.cuda_smoke:
        parser.error("--require-cuda requires explicit --cuda-smoke")
    report = collect(cuda_smoke=args.cuda_smoke, device=args.device)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 2 if args.require_cuda and not report["cuda_smoke_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
