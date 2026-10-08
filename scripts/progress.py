#!/usr/bin/env python3
"""Read-only progress from hashes and manifests, with no GPU computation."""
from collections import Counter
from pathlib import Path
from cvpr_project.runs import read_json, resource_totals, sha256


def main():
    names = ["executor.py", "buffer_pool.py", "graph_pool.py", "serve.py", "scheduler.py", "model.py"]
    hashes = {f"src/cvpr_project/{n}": sha256(Path("src/cvpr_project") / n) for n in names}
    completed_calibration = []
    for path in Path("artifacts/calibration_runs").glob("*/*/manifest.json"):
        m = read_json(path)
        if m["status"] == "completed" and all(m["source"]["source_files_sha256"].get(p) == h for p, h in hashes.items()):
            completed_calibration.append(m)
    print(f"Calibration: {len(completed_calibration)}/252 current-runtime configurations; "
          f"{dict(Counter(m['policy'] for m in completed_calibration))}")
    frozen_path = Path("artifacts/frozen_v2.json")
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        expected = 300 if frozen["compile"]["status"] == "passed" else 264
        completed, failed = [], []
        for path in Path("artifacts/runs_v2").glob("*/manifest.json"):
            m = read_json(path)
            if m["frozen_config_sha256"] == frozen["frozen_config_sha256"]:
                if m["status"] in {"completed", "completed_with_failures"}:
                    completed.append(m["case_id"])
                elif m["status"] in {"failed", "interrupted"}:
                    failed.append(m["run_id"])
        print(f"Serving: {len(set(completed))}/{expected} distinct completed cases; {len(failed)} preserved failed attempts")
        quality_paths = list(Path("artifacts/quality_held_out").glob("*/metrics.json"))
        valid = [read_json(p) for p in quality_paths if read_json(p).get("frozen_config_sha256") == frozen["frozen_config_sha256"]]
        print(f"Held-out quality: {len(valid)}/{'12' if expected == 300 else '8'} complete 4000-image backend/bucket evaluations")
    active = Path("artifacts/active_gpu_job.json")
    print(f"Active GPU lease: {read_json(active)['stage']}" if active.exists() else "Active GPU lease: none")
    cost = resource_totals()
    print(f"Budget: {cost['gpu_hours']:.3f}/30 GPU-hours including {cost['initial_unmetered_setup_reserve_hours']:.2f} reserve; "
          f"{cost['project_storage_gb']:.2f}/30 GB")


if __name__ == "__main__":
    main()
