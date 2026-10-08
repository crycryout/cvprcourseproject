#!/usr/bin/env python3
"""Verify published small evidence without COCO, model weights, or a GPU."""
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from cvpr_project.runs import object_hash, read_json, sha256


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-8), (actual, expected)


def main():
    root = Path("results")
    delivery = read_json(root / "delivery.json")
    frozen = read_json(root / "frozen_v2.json")
    identity = frozen.pop("frozen_config_sha256")
    assert object_hash(frozen) == identity == delivery["frozen_config_sha256"]
    with (root / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    expected = 264 + delivery["compile_baseline_expected"]
    assert len(rows) == expected == delivery["completed_cases"]
    assert len({r["case_id"] for r in rows}) == len(rows)
    counts = Counter(r["policy"] for r in rows)
    hardware = read_json(root / "environment_timing_v2.json")["cuda_device"]
    assert counts["A0"] == 12
    for policy in ["E0", "E1", "G0", "P0", "R0", "F0", "D0"]:
        assert counts[policy] == 36, (policy, counts[policy])
    assert counts["C0"] == delivery["compile_baseline_expected"]
    trace_groups = defaultdict(set)
    for row in rows:
        public = read_json(root / "evidence" / f"{row['run_id']}.json")
        manifest, metric = public["manifest"], public["metrics"]
        assert manifest["frozen_config_sha256"] == identity
        assert manifest["case_id"] == row["case_id"]
        assert manifest["status"] in {"completed", "completed_with_failures"}
        assert manifest["precision"] == frozen["calibration"]["precision"]
        assert manifest["cuda_device_info"] == hardware
        assert not metric["loadgen_limited"] and not metric["shared_gpu_jobs_observed"]
        assert all(s["other_gpu_process_count"] == 0 for s in metric["gpu_activity_samples"])
        assert metric["offered"] == sum(metric[k] for k in ["completed", "rejected", "failed", "unfinished"])
        assert metric["completed"] == metric["ontime"] + metric["late_completed"]
        close(metric["slo_success"], metric["ontime"] / metric["offered"])
        close(metric["goodput_rps"], metric["ontime"] / 60)
        close(metric["completion_ratio"], metric["completed"] / metric["offered"])
        for key in ["offered", "ontime", "goodput_rps", "slo_success", "completion_ratio"]:
            close(row[key], metric[key])
        assert len(manifest["request_events_sha256"]) == 64
        trace_groups[(row["arrival_type"], row["rate_multiplier"], row["trace_seed"])].add(manifest["trace_sha256"])
    assert len(trace_groups) == 36 and all(len(hashes) == 1 for hashes in trace_groups.values())
    with (root / "quality.csv").open() as f:
        quality = list(csv.DictReader(f))
    assert len(quality) == (12 if counts["C0"] else 8)
    for q in quality:
        assert int(q["evaluated_images"]) == 4000
        assert q["precision"] == frozen["calibration"]["precision"]
        assert q["quality_gate_passed"] == "True" and abs(float(q["AP_change_pp"])) <= .1
        assert len(q["predictions_sha256"]) == 64
    for name in ["buffer_pool.py", "data.py", "executor.py", "graph_pool.py", "metrics.py", "model.py",
                 "scheduler.py", "serve.py", "trace.py", "quality.py", "matrix.py"]:
        path = Path("src/cvpr_project") / name
        assert sha256(path) == frozen["source"]["source_files_sha256"][str(path)], path
    assert delivery["resource_totals"]["gpu_hours"] < 30
    assert delivery["resource_totals"]["project_storage_gb"] < 30
    assert (root / "report.pdf").read_bytes().startswith(b"%PDF-")
    for policy in ["E1", "P0"] + (["C0"] if counts["C0"] else []):
        for suffix, rate in [("_low", .3), ("", 1.1)]:
            profile = read_json(root / f"profile_{policy}{suffix}.json")
            assert profile["frozen_config_sha256"] == identity and profile["rate_multiplier"] == rate
            assert profile["hardware"] == hardware and profile["profiler_not_formal_timing"]
    assert all((root / "figures" / f"{name}.png").exists() for name in
               ["latency_p95", "latency_p99", "goodput", "deadline_ablation", "execution_ablation", "detection_examples"])
    print(f"Verified {len(rows)} serving cases, {len(quality)} complete 4000-image quality evaluations, shared traces, accounting, and frozen runtime.")


if __name__ == "__main__":
    main()
