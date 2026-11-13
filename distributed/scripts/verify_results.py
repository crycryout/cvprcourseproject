#!/usr/bin/env python3
"""CPU-only delivery audit. No JPEGs, checkpoints, GPUs or endpoints required."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from batchserve.common import digest, file_hash, source_hashes
from batchserve.benchmark import make_trace


def verify(root):
    root = Path(root)
    cfg = json.loads((root / "frozen.json").read_text())
    expected = cfg["freeze_sha256"]
    copy = dict(cfg)
    copy.pop("freeze_sha256")
    assert digest(copy) == expected, "Frozen config hash mismatch"
    assert cfg["source_sha256"] == source_hashes(ROOT), "Source differs from freeze"
    assert all(w["source_sha256"] == cfg["source_sha256"] for w in cfg["workers"]), "Workers loaded another source snapshot"
    provenance = json.loads((ROOT / "configs/upstream.json").read_text())
    for path, sha in provenance["files_sha256"].items():
        assert file_hash(ROOT / "src" / path) == sha, f"Upstream modified: {path}"
    split = json.loads((ROOT / "configs/data_split.json").read_text())
    assert len(split["calibration"]) == 1000 and len(split["held_out"]) == 4000
    assert not set(split["calibration"]) & set(split["held_out"])
    assert digest(split["calibration"]) == cfg["calibration_subset_sha256"]
    runs = [json.loads(p.read_text()) for p in sorted((root / "runs").glob("*.json"))]
    expected_cases = {(p, k, float(r), s) for p in cfg["policies"] for k in cfg["arrival_types"]
                      for r in cfg["rates"] for s in cfg["seeds"]}
    actual = {(r["policy"], r["arrival_type"], r["offered_rate"], r["seed"]) for r in runs}
    assert len(runs) == len(actual) == len(expected_cases) and actual == expected_cases, "Incomplete/duplicate matrix"
    traces = {}
    for r in runs:
        assert r["freeze_sha256"] == expected and r["protocol_version"] == 3
        assert r["status"] in {"completed", "completed_with_failures"}
        assert r["valid_loadgen"], "Generator-limited result included"
        assert r["arrival_mode"] == "independent_process_nonblocking_atomic_pipe", "Cooperative generator epoch included"
        assert sum(r["status_counts"].values()) == r["offered"], "All-offered count mismatch"
        assert 0 <= r["on_time"] <= r["completed"] <= r["offered"]
        assert abs(r["goodput_rps"] - r["on_time"] / cfg["measurement_s"]) < 1e-9
        assert abs(r["slo_success"] - r["on_time"] / r["offered"]) < 1e-9
        key = (r["arrival_type"], r["offered_rate"], r["seed"])
        if key not in traces:
            trace = make_trace(*key[:2], key[2], split["held_out"], cfg["deadlines_s"], cfg["warmup_s"], cfg["measurement_s"])
            traces[key] = digest([asdict(x) for x in trace])
            assert sum(x.measurement for x in trace) == r["offered"]
        assert r["trace_sha256"] == traces[key], "Unpaired trace"
    quality = json.loads((root / "quality.json").read_text())
    accepted_quality_freezes = {expected}
    if (root / "quality_compatibility.json").exists():
        from retain_quality import verify_compatibility
        origin = json.loads((root / "quality_origin_frozen.json").read_text())
        certificate = json.loads((root / "quality_compatibility.json").read_text())
        regenerated = verify_compatibility(cfg, origin, (root / "quality_origin_benchmark.py").read_text())
        assert regenerated == certificate, "Retained quality compatibility proof failed"
        accepted_quality_freezes.add(origin["freeze_sha256"])
        sanity = json.loads((root / "stress_rebuild.json").read_text())
        assert len(sanity["results"]) == 2 and all(r["status"] == "passed" and r["requests"] >= 64 for r in sanity["results"])
    assert len(quality) == 10
    for j in range(2):
        q = [r for r in quality if r["worker"] == j]
        assert {(r["backend"], r["bucket"]) for r in q} == {("eager", 1)} | {("graph", b) for b in cfg["buckets"]}
        ref = next(r["metrics_percent"]["AP"] for r in q if r["backend"] == "eager")
        for r in q:
            assert r["evaluated_images"] == 4000 and r["freeze_sha256"] in accepted_quality_freezes
            assert abs(r["metrics_percent"]["AP"] - ref) <= .1
    stress = json.loads((root / "stress.json").read_text())
    assert len(stress["results"]) == 2
    assert all(r["status"] == "passed" and r["requests"] >= 1000 and r["concurrent_slot_batches_verified"] == 2 for r in stress["results"])
    controls = [json.loads(p.read_text()) for p in (root / "controls").glob("*.json")]
    assert len(controls) == 48
    assert {(r["topology"], r["arrival_type"], r["offered_rate"], r["seed"]) for r in controls} == {
        (f"worker{j}_only", k, r, s) for j in range(2) for k in cfg["arrival_types"] for r in cfg["rates"] for s in cfg["seeds"]}
    assert all(r["freeze_sha256"] == expected and r["valid_loadgen"] for r in controls)
    assert all(r["arrival_mode"] == "independent_process_nonblocking_atomic_pipe" for r in controls)
    lag_limit_ms = max(1.0, min(cfg["deadlines_s"]) * 1000 * .05)
    for r in runs + controls:
        key = (r["arrival_type"], r["offered_rate"], r["seed"])
        assert r["trace_sha256"] == traces[key], "Control trace differs from two-host trace"
        assert r["generator_lag_p99_ms"] <= lag_limit_ms, "Producer lag exceeds frozen validity rule"
        assert sum(r["status_counts"].values()) == r["offered"]
        assert r["status_counts"].get("completed", 0) == r["completed"]
        assert 0 <= r["on_time"] <= r["completed"] <= r["offered"]
        assert abs(r["goodput_rps"] - r["on_time"] / cfg["measurement_s"]) < 1e-9
        assert abs(r["slo_success"] - r["on_time"] / r["offered"]) < 1e-9
        assert abs(r["completion_ratio"] - r["completed"] / r["offered"]) < 1e-9
    resources = json.loads((root / "resources.json").read_text())
    assert all(r.get("status") != "active_at_export" and r.get("stopped_unix_s") for r in resources["leases"]), "Workers still active at export"
    assert abs(sum(r["aggregate_gpu_hours"] for r in resources["leases"]) - resources["new_aggregate_gpu_hours"]) < 1e-9
    assert resources["new_aggregate_gpu_hours"] <= resources["extension_cap_gpu_hours"] == 16
    assert resources["new_aggregate_gpu_hours"] + resources["historical_cvpr_gpu_hours"] <= 30
    if (root / "storage.json").exists():
        storage = json.loads((root / "storage.json").read_text())
        assert sum(x["allocated_bytes"] for x in storage["components"]) == storage["total_allocated_bytes"]
        assert storage["total_allocated_bytes"] < storage["limit_bytes"] == 30_000_000_000
    for path, sha in json.loads((root / "checksums.json").read_text()).items():
        assert file_hash(root / path) == sha, f"Evidence checksum mismatch: {path}"
    print(f"Verified {len(runs)} paired two-host runs, {len(controls)} single-host controls, {len(quality)} full 4000-image AP evaluations, source freeze and two-worker numerical stress.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(ROOT / "results"))
    verify(p.parse_args().results)
