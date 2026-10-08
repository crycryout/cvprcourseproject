"""Resumable randomized policy order; measured run manifests never invented."""
from pathlib import Path
import gc
import json
import random
import time
import torch
from .data import Corpus
from .executor import Executor
from .model import Detector
from .quality import evaluate_backend
from .runs import (read_json, write_json, run_id, provenance, sha256, object_hash, stamp,
                   assert_no_other_gpu_jobs, gpu_snapshot, resource_totals)
from .serve import run_serving
from .trace import make_trace, trace_hash


def load_frozen(path):
    frozen = read_json(path)
    h = frozen.pop("frozen_config_sha256")
    if frozen.get("protocol_version") != 2 or frozen.get("status") != "frozen" or object_hash(frozen) != h:
        raise ValueError("Invalid/tampered freeze")
    frozen["frozen_config_sha256"] = h
    return frozen


def verify_evaluation_inputs(frozen, weights, data_root):
    """Reject code/data/weights drift before accepting or producing evidence."""
    assert_no_other_gpu_jobs()
    source = frozen["source"]["source_files_sha256"]
    for name in ["buffer_pool.py", "data.py", "executor.py", "graph_pool.py", "metrics.py",
                 "model.py", "scheduler.py", "serve.py", "trace.py", "quality.py", "matrix.py"]:
        path = Path("src/cvpr_project") / name
        if sha256(path) != source[str(path)]:
            raise ValueError(f"Frozen evaluation code changed: {path}")
    for name, expected in frozen["model"]["files_sha256"].items():
        if sha256(Path(weights) / name) != expected:
            raise ValueError(f"Frozen model changed: {name}")
    data = frozen["data"]
    local = read_json("artifacts/data_v2.json")
    if object_hash(local) != object_hash(data):
        raise ValueError("Frozen data manifest changed")
    annotation = Path(data_root) / "annotations/instances_val2017.json"
    if sha256(annotation) != data["annotation_sha256"]:
        raise ValueError("Frozen COCO annotations changed")
    images = {x["id"]: x for x in read_json(annotation)["images"]}
    hashes = {str(i): sha256(Path(data_root) / "val2017" / images[i]["file_name"])
              for i in sorted(data["calibration"] + data["held_out"])}
    if object_hash(hashes) != data["image_bytes_sha256"]:
        raise ValueError("Frozen COCO image bytes changed")


def evaluate_quality(frozen_path, weights, data_root, device):
    frozen = load_frozen(frozen_path)
    verify_evaluation_inputs(frozen, weights, data_root)
    detector = Detector(weights, frozen["calibration"]["precision"], device)
    corpus = Corpus(data_root, subset="held_out")
    output = Path("artifacts/quality_held_out")
    summaries = []
    for backend in ["eager", "graph"] + (["compile"] if frozen["compile"]["status"] == "passed" else []):
        executor = Executor(detector, buckets=frozen["calibration"]["common_buckets"], backend=backend)
        for b in executor.buckets:
            target = output / f"{backend}_b{b}"
            if (target / "metrics.json").exists():
                q = read_json(target / "metrics.json")
                if (q.get("frozen_config_sha256") != frozen["frozen_config_sha256"] or
                    q.get("backend") != backend or q.get("bucket") != b or
                    q.get("precision") != frozen["calibration"]["precision"] or
                    q.get("evaluated_images") != len(corpus.ids) or
                    sha256(target / "predictions.json") != q.get("predictions_sha256")):
                    raise ValueError(f"Conflicting or incomplete quality evidence: {target}")
            else:
                q, _ = evaluate_backend(executor, corpus, bucket=b, output=target, label=f"held-out-{backend}")
                q["frozen_config_sha256"] = frozen["frozen_config_sha256"]
                write_json(target / "metrics.json", q)
            summaries.append(q)
        del executor
        gc.collect()
        torch.cuda.empty_cache()
    reference = next(q["metrics_percent"]["AP"] for q in summaries if q["backend"] == "eager" and q["bucket"] == 1)
    for q in summaries:
        q["AP_change_from_same_precision_eager_b1_pp"] = q["metrics_percent"]["AP"] - reference
        q["quality_gate_passed"] = abs(q["AP_change_from_same_precision_eager_b1_pp"]) <= .1
    result = {"protocol_version": 2, "frozen_config_sha256": frozen["frozen_config_sha256"],
              "reference_AP": reference, "rows": summaries,
              "all_quality_gates_passed": all(q["quality_gate_passed"] for q in summaries)}
    write_json("artifacts/quality_v2.json", result)
    if not result["all_quality_gates_passed"]:
        raise RuntimeError("Held-out AP equality gate failed; retain evidence and investigate correctness")
    return result


def run_matrix(matrix_path, frozen_path, weights, data_root, device, output="artifacts/runs_v2", limit=None):
    frozen = load_frozen(frozen_path)
    verify_evaluation_inputs(frozen, weights, data_root)
    calibration = frozen["calibration"]
    quality = read_json("artifacts/quality_v2.json")
    if not quality["all_quality_gates_passed"] or quality.get("frozen_config_sha256") != frozen["frozen_config_sha256"]:
        raise ValueError("Verified held-out quality is required before formal serving matrix")
    corpus = Corpus(data_root, subset="held_out")
    detector = Detector(weights, calibration["precision"], device)
    graph = Executor(detector, buckets=calibration["common_buckets"], backend="graph")
    eager = Executor(detector, buckets=calibration["common_buckets"])
    pageable = Executor(detector, buckets=(1,), slots_per_bucket=1, pinned=False)
    compile_executor = Executor(detector, buckets=calibration["common_buckets"], backend="compile") if frozen["compile"]["status"] == "passed" else None
    matrix = read_json(matrix_path)
    if matrix.get("config_sha256") != frozen["config_sha256"]:
        raise ValueError("Matrix was generated from a different protocol configuration")
    cases = list(matrix["cases"])
    if compile_executor:
        cases += [{**x, "policy": "C0", "case_id": x["case_id"].replace("F0", "C0"), "stage": "compile_baseline"}
                  for x in cases if x["policy"] == "F0"]
    # Group by shared trace, then permute policy order within every seed/trace/load.
    grouped = {}
    for case in cases:
        key = (case["arrival_type"], case["rate_multiplier"], case["trace_seed"])
        grouped.setdefault(key, []).append(case)
    ordered = []
    rng = random.Random(20261008)
    keys = list(grouped)
    rng.shuffle(keys)
    for key in keys:
        group = grouped[key]
        rng.shuffle(group)
        ordered.extend(group)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for index, case in enumerate(ordered):
        successful = []
        for path in root.glob(f"{case['case_id']}*/manifest.json"):
            m = read_json(path)
            if m["status"] in {"completed", "completed_with_failures"} and m["frozen_config_sha256"] == frozen["frozen_config_sha256"]:
                successful.append(path)
        if len(successful) > 1:
            raise ValueError(f"Duplicate completed case: {case['case_id']}")
        if successful:
            old = read_json(successful[0])
            if sha256(successful[0].parent / "request_events.csv") != old["request_events_sha256"]:
                raise ValueError(f"Completed request evidence changed: {case['case_id']}")
            continue
        cost = resource_totals()
        if cost["gpu_hours"] >= 30 or cost["project_storage_gb"] >= 30:
            print(f"Resource cap: {cost}; stopping new runs", flush=True)
            break
        assert_no_other_gpu_jobs()
        policy = case["policy"]
        settings = calibration["policies"].get(policy, calibration["policies"]["F0"])
        executor = pageable if policy == "E0" else eager if policy in {"E1", "R0"} else compile_executor if policy == "C0" else graph
        rate = case["rate_multiplier"] * calibration["lambda_ref_rps"]
        work = frozen["config"]["workload"]
        trace = make_trace(case["arrival_type"], rate, case["trace_seed"], corpus.ids, calibration["s_base_s"],
                           work["warmup_seconds"], work["measurement_seconds"])
        rid = run_id(case["case_id"])
        target = root / rid
        target.mkdir()
        manifest = {"protocol_version": 2, **case, "run_id": rid, "status": "running",
                    "policy": policy, "settings": settings, "offered_rate_rps": rate,
                    "trace_sha256": trace_hash(trace), "frozen_config_sha256": frozen["frozen_config_sha256"],
                    "model_revision": frozen["model"]["revision"], "model_files_sha256": frozen["model"]["files_sha256"],
                    "data_split_sha256": frozen["data"]["split_sha256"],
                    "source": provenance(), "precision": calibration["precision"], "backend": executor.backend,
                    "cuda_device_info": detector.hardware,
                    "common_buckets": calibration["common_buckets"], "cpu_workers": calibration["cpu_workers"],
                    "started_utc": stamp(), "gpu_before": gpu_snapshot(), "timing_scope": frozen["output_scope"],
                    "synthetic_arrivals": True, "actual_image_bytes": "COCO2017_validation_held_out_subset",
                    "evidence": {"metrics": "metrics.json", "requests": "request_events.csv"}}
        write_json(target / "manifest.json", manifest)
        write_json(target / "frozen_config.json", frozen)
        start = time.monotonic()
        cpu_start = time.process_time()
        print(f"matrix {index+1}/{len(ordered)} {case['case_id']}", flush=True)
        try:
            metrics, request_rows = run_serving(executor, corpus, trace, policy, settings,
                                    {int(k): v for k, v in calibration["service_ns"].items()}, calibration["s_base_s"],
                                    warmup_s=work["warmup_seconds"], measurement_s=work["measurement_seconds"],
                                    drain_s=work["drain_cap_seconds"], workers=calibration["cpu_workers"], output=target,
                                    progress=lambda done, total: print(f"  completed {done}/{total}", flush=True))
            cpu_seconds = time.process_time() - cpu_start
            cpu_wall_seconds = time.monotonic() - start
            batches = {}
            for row in request_rows:
                if row["measurement"] and row["status"] == "completed":
                    batches[(row["bucket"], row["slot_id"], row["dispatch_ns"])] = row["valid_count"]
            distribution = {}
            for (bucket, _, _), occupancy in batches.items():
                key = f"b{bucket}_n{occupancy}"
                distribution[key] = distribution.get(key, 0) + 1
            metrics.update(main_process_cpu_seconds=cpu_seconds,
                           main_process_cpu_core_equivalents=cpu_seconds / cpu_wall_seconds,
                           cpu_measurement_scope="main_process_warmup_measurement_drain_and_request_export_excludes_loadgen",
                           completed_measurement_batch_distribution=distribution,
                           completed_measurement_batches=len(batches))
            write_json(target / "metrics.json", metrics)
            if metrics["loadgen_limited"]:
                manifest.update(status="failed", failure="load_generator_lag_exceeds_protocol_gate")
            else:
                manifest["status"] = "completed_with_failures" if metrics["failed"] else "completed"
            manifest["request_events_sha256"] = sha256(target / "request_events.csv")
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            failure_type=type(exc).__name__, failure_message=str(exc))
            raise
        finally:
            manifest.update(ended_utc=stamp(), elapsed_seconds=time.monotonic() - start,
                            actual_gpu_hours=(time.monotonic() - start) / 3600,
                            cumulative_resources=resource_totals(), gpu_after=gpu_snapshot())
            write_json(target / "manifest.json", manifest)
        completed += 1
        if manifest["status"] == "failed":
            raise RuntimeError(f"Run failed protocol gate: {rid}")
        if limit and completed >= limit:
            break
    return {"new_runs": completed, "planned_cases": len(ordered)}
