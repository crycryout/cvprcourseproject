#!/usr/bin/env python3
"""Resume a measured stage after 60 seconds without another GPU job.

Retry only explicit GPU-interference failures. Other failures remain actionable
errors rather than being hidden in an endless retry loop.
"""
import argparse
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path
from cvpr_project.runs import other_gpu_pids, resource_totals, stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify-model", "pilot", "compile-baseline", "calibrate", "evaluate-quality", "run-matrix", "validate-slots", "profile", "collect-nsight"])
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", os.environ.get("CVPR_GPU", "0"))
    if "," in selected or selected.startswith("MIG-"):
        raise ValueError("This protocol requires exactly one full physical H800")
    selected = subprocess.check_output(["nvidia-smi", "--id", selected, "--query-gpu=uuid",
                                        "--format=csv,noheader"], text=True).strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    def full_mode_available():
        mode = subprocess.check_output(["nvidia-smi", "--id", selected, "--query-gpu=mig.mode.current",
                                        "--format=csv,noheader"], text=True).strip()
        return mode == "Disabled"
    command = ([sys.executable, "-u", "scripts/collect_nsight.py"] if args.stage == "collect-nsight" else
               [sys.executable, "-u", "scripts/validate_slots.py"] if args.stage == "validate-slots"
               else [sys.executable, "-u", "-m", "cvpr_project", args.stage]) + extra
    Path("artifacts").mkdir(exist_ok=True)
    attempt = 0
    while True:
        idle_since = None
        print(f"{stamp()} waiting for a full H800 and an idle host before {args.stage}", flush=True)
        while idle_since is None or time.monotonic() - idle_since < args.idle_seconds:
            if not full_mode_available() or other_gpu_pids():
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            time.sleep(1)
        cost = resource_totals()
        if cost["gpu_hours"] >= 30 or cost["project_storage_gb"] >= 30:
            raise RuntimeError(f"Resource cap reached: {cost}")
        attempt += 1
        path = Path("artifacts") / f"{args.stage}_guarded_attempt_{attempt}_{time.time_ns()}.log"
        print(f"{stamp()} starting {args.stage}; log={path}", flush=True)
        with path.open("w") as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            print(f"{stamp()} {args.stage} completed", flush=True)
            return
        log = path.read_text()
        if not any(message in log for message in ["Shared GPU activity detected during timing",
                                                   "Formal timing blocked by"]):
            print(log[-6000:], flush=True)
            sys.exit(result.returncode)
        if args.stage == "profile" and "--output" in extra:
            profile = Path(extra[extra.index("--output") + 1])
            if profile.exists():
                archive = Path("artifacts/profile_attempts") / f"{profile.name}_{time.time_ns()}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(profile, archive)
        print(f"{stamp()} interference attempt preserved; waiting to resume", flush=True)


if __name__ == "__main__":
    main()
