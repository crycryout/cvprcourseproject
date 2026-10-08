#!/usr/bin/env python3
"""Resume a measured stage after 60 seconds without another GPU job.

Retry only explicit GPU-interference failures. Other failures remain actionable
errors rather than being hidden in an endless retry loop.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from cvpr_project.runs import other_gpu_pids, resource_totals, stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate", "evaluate-quality", "run-matrix"])
    parser.add_argument("--idle-seconds", type=int, default=60)
    args = parser.parse_args()
    Path("artifacts").mkdir(exist_ok=True)
    attempt = 0
    while True:
        idle_since = None
        print(f"{stamp()} waiting for an idle GPU before {args.stage}", flush=True)
        while idle_since is None or time.monotonic() - idle_since < args.idle_seconds:
            if other_gpu_pids():
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
            result = subprocess.run([sys.executable, "-u", "-m", "cvpr_project", args.stage],
                                    stdout=f, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            print(f"{stamp()} {args.stage} completed", flush=True)
            return
        log = path.read_text()
        if not any(message in log for message in ["Shared GPU activity detected during timing",
                                                   "Formal timing blocked by"]):
            print(log[-6000:], flush=True)
            sys.exit(result.returncode)
        print(f"{stamp()} interference attempt preserved; waiting to resume", flush=True)


if __name__ == "__main__":
    main()
