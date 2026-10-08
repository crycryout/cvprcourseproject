#!/usr/bin/env python3
"""Collect actual Nsight evidence separately from the formal serving matrix."""
import argparse
import csv
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path
from cvpr_project.matrix import load_frozen
from cvpr_project.runs import assert_no_other_gpu_jobs, read_json, run_id, sha256, stamp, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/nsight_v2")
    args = parser.parse_args()
    frozen = load_frozen("artifacts/frozen_v2.json")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "summary.json"
    state = read_json(state_path) if state_path.exists() else {
        "protocol_version": 2, "profiler_not_formal_timing": True,
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "input_subset": "calibration", "commands": [], "status": "running",
    }
    if state["frozen_config_sha256"] != frozen["frozen_config_sha256"]:
        raise ValueError("Nsight evidence belongs to a different freeze; preserve it separately")
    if state.get("status") == "completed":
        return
    nsys = shutil.which("nsys")
    ncu = shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
    if not nsys or not Path(ncu).exists():
        raise FileNotFoundError("Install Nsight tools separately; this script does not modify system packages")

    def execute(stage, argv, timeout=300):
        identifier = run_id(stage)
        log = root / f"{identifier}.log"
        started = stamp()
        with log.open("w") as output:
            try:
                result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        item = {"stage": stage, "argv": argv, "started_utc": started, "ended_utc": stamp(),
                "returncode": code, "raw_log": log.name, "raw_log_sha256": sha256(log)}
        state["commands"].append(item)
        write_json(state_path, state)
        output = log.read_text(errors="replace")
        if code:
            state.update(status="failed", failed_stage=stage, last_error=output[-3000:])
            write_json(state_path, state)
            if any(s in output for s in ["Shared GPU activity detected during timing", "Formal timing blocked by"]):
                raise RuntimeError("Shared GPU activity detected during timing; preserve Nsight attempt and retry when idle")
            raise RuntimeError(f"{stage} failed with exit {code}; inspect {log}")
        return output

    # Unique outputs preserve reports from interrupted captures without replacing them.
    if not state.get("systems"):
        assert_no_other_gpu_jobs()
        prefix = root / run_id("P0-systems")
        metadata = prefix.with_suffix(".json")
        execute("nsys-capture", [nsys, "profile", "--sample=none", "--cpuctxsw=none",
            "--trace=cuda,nvtx,osrt", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "--force-overwrite=false", "--output", str(prefix), sys.executable,
            "scripts/profile_native.py", "--backend", "graph", "--mode", "serving",
            "--rate-multiplier", "1.1", "--output", str(metadata)])
        report = prefix.with_suffix(".nsys-rep")
        if not report.exists() or not metadata.exists():
            raise RuntimeError("Nsight capture returned success without its actual report and request metadata")
        statistics = {}
        for name in ["cuda_gpu_kern_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum", "nvtx_sum"]:
            output = execute(f"nsys-{name}", [nsys, "stats", "--report", name, "--format", "csv",
                                               "--output", "-", str(report)])
            parsed = list(csv.reader(io.StringIO(output)))
            header_index = next((i for i, row in enumerate(parsed) if "Name" in row), None)
            if header_index is None:
                raise ValueError(f"No CSV table in actual Nsight report {name}")
            header = parsed[header_index]
            statistics[name] = [dict(zip(header, row)) for row in parsed[header_index + 1:]
                                if len(row) == len(header)][:40]
        if not statistics["cuda_gpu_kern_sum"]:
            raise ValueError("Nsight report contains no actual CUDA kernels")
        state["systems"] = {"report": report.name, "report_sha256": sha256(report),
                            "metadata": read_json(metadata), "statistics": statistics}
        state["selected_hotspot_kernel"] = statistics["cuda_gpu_kern_sum"][0]["Name"]
        write_json(state_path, state)
    if not state.get("compute"):
        assert_no_other_gpu_jobs()
        prefix = root / run_id("P0-hotspot")
        kernel = state["selected_hotspot_kernel"]
        metadata = prefix.with_suffix(".json")
        execute("ncu-capture", [ncu, "--clock-control", "none", "--profile-from-start", "off",
            "--kernel-name-base", "demangled", "--kernel-name", "regex:" + re.escape(kernel),
            "--launch-count", "1", "--set", "basic", "--export", str(prefix),
            sys.executable, "scripts/profile_native.py", "--backend", "graph", "--mode", "forward",
            "--bucket", "1", "--output", str(metadata)])
        report = prefix.with_suffix(".ncu-rep")
        if not report.exists() or not metadata.exists():
            raise RuntimeError("Nsight Compute did not produce a report and actual native-input metadata")
        output = execute("ncu-import", [ncu, "--import", str(report), "--csv", "--page", "raw"])
        parsed = list(csv.reader(io.StringIO(output)))
        index = next((i for i, row in enumerate(parsed) if "Metric Name" in row), None)
        if index is None:
            raise ValueError("Nsight Compute import has no real metric rows")
        metrics = []
        for row in parsed[index + 1:]:
            if len(row) != len(parsed[index]):
                continue
            item = dict(zip(parsed[index], row))
            metrics.append({key: item[key] for key in ["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]})
        if not metrics:
            raise ValueError("Nsight Compute captured zero matching kernels; repair the filter")
        state["compute"] = {"report": report.name, "report_sha256": sha256(report),
                            "metadata": read_json(metadata), "metrics": metrics}
    state.update(status="completed", ended_utc=stamp())
    state.pop("last_error", None)
    state.pop("failed_stage", None)
    write_json(state_path, state)
    print("Actual Nsight Systems timeline and one measured hotspot kernel captured", flush=True)


if __name__ == "__main__":
    main()
