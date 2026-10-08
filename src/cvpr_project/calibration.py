"""Equal-budget baseline tuning on calibration images, then immutable freeze."""
from pathlib import Path
import gc
import itertools
import time
import numpy as np
import torch
from .benchmark import pilot, service_table, capacity
from .data import Corpus
from .executor import Executor
from .model import Detector
from .quality import evaluate_backend, stress_outputs, visual_examples
from .runs import (read_json, write_json, provenance, object_hash, sha256, sanitize_public,
                   stamp, run_id, gpu_snapshot, assert_no_other_gpu_jobs)
from .serve import run_serving
from .trace import make_trace


def release(*objects):
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def verify_model(weights, data_root, device):
    corpus = Corpus(data_root, subset="calibration")
    fp32 = Detector(weights, "fp32", device)
    contract = fp32.validate_contract(corpus)
    write_json("artifacts/environment_v2.json", {"protocol_version": 2, "cuda_device": fp32.hardware,
                                                "source": provenance()})
    write_json("artifacts/input_contract_v2.json", contract)
    eager = Executor(fp32, buckets=(1,), slots_per_bucket=1)
    raw_reference = []
    from .scheduler import Ready
    for i in corpus.ids[:20]:
        slot = eager.free_slot(1)
        eager.submit(slot, [Ready(0, i, 0, 0, 0, fp32.preprocess(corpus.image(i)))])
        slot.events["done"].synchronize()
        raw_reference.append((slot.host_logits.clone().float(), slot.host_boxes.clone().float()))
        eager.consume(slot, corpus)
    q32, predictions = evaluate_backend(eager, corpus, output="artifacts/quality_calibration_fp32", label="calibration-fp32")
    visual_examples(corpus, predictions)
    del eager, fp32
    gc.collect()
    torch.cuda.empty_cache()
    bf16 = Detector(weights, "bf16", device)
    eager = Executor(bf16, buckets=(1,), slots_per_bucket=1)
    raw_errors = []
    for i, (logits32, boxes32) in zip(corpus.ids[:20], raw_reference):
        slot = eager.free_slot(1)
        eager.submit(slot, [Ready(0, i, 0, 0, 0, bf16.preprocess(corpus.image(i)))])
        slot.events["done"].synchronize()
        raw_errors.append({"image_id": i, "logits_max_abs_error": float((slot.host_logits.float()-logits32).abs().max()),
                           "boxes_max_abs_error": float((slot.host_boxes.float()-boxes32).abs().max())})
        eager.consume(slot, corpus)
    q16, _ = evaluate_backend(eager, corpus, output="artifacts/quality_calibration_bf16", label="calibration-bf16")
    drop = q32["metrics_percent"]["AP"] - q16["metrics_percent"]["AP"]
    selected = "bf16" if drop <= .3 else "fp32"
    bucket_quality = []
    fallback_reason = None
    if selected == "bf16":
        batch_executor = Executor(bf16, buckets=(2, 4, 8), slots_per_bucket=1)
        for b in [2, 4, 8]:
            q, _ = evaluate_backend(batch_executor, corpus, bucket=b,
                                    output=f"artifacts/quality_calibration_bf16_b{b}")
            delta = q["metrics_percent"]["AP"] - q16["metrics_percent"]["AP"]
            bucket_quality.append({"precision": "bf16", "bucket": b, "AP": q["metrics_percent"]["AP"],
                                   "AP_change_from_batch1_pp": delta})
            if abs(delta) > .1:
                selected = "fp32"
                fallback_reason = f"Calibration BF16 bucket {b} changes AP by {delta:.6f} pp (>0.1 pp)"
                break
        del batch_executor
    if selected == "fp32":
        del eager, bf16
        gc.collect()
        torch.cuda.empty_cache()
        fp32 = Detector(weights, "fp32", device)
        batch_executor = Executor(fp32, buckets=(2, 4, 8), slots_per_bucket=1)
        for b in [2, 4, 8]:
            q, _ = evaluate_backend(batch_executor, corpus, bucket=b,
                                    output=f"artifacts/quality_calibration_fp32_b{b}")
            delta = q["metrics_percent"]["AP"] - q32["metrics_percent"]["AP"]
            bucket_quality.append({"precision": "fp32", "bucket": b, "AP": q["metrics_percent"]["AP"],
                                   "AP_change_from_batch1_pp": delta})
            if abs(delta) > .1:
                raise RuntimeError(f"FP32 calibration batch {b} violates AP equality; inspect model math")
    selection = {"protocol_version": 2, "fp32_AP": q32["metrics_percent"]["AP"],
                 "bf16_AP": q16["metrics_percent"]["AP"], "bf16_drop_pp": drop,
                 "maximum_bf16_drop_pp": .3, "precision": selected,
                 "bucket_quality_calibration": bucket_quality, "prefreeze_fallback_reason": fallback_reason,
                 "bf16_vs_fp32_raw_errors": raw_errors,
                 "bf16_logits_atol": max(x["logits_max_abs_error"] for x in raw_errors),
                 "bf16_boxes_atol": max(x["boxes_max_abs_error"] for x in raw_errors),
                 "selection_subset": "calibration", "held_out_used": False,
                 "tf32": False, "attention_implementation": "eager", "source": provenance()}
    write_json("artifacts/precision_v2.json", selection)
    return selection


def correctness_and_pilot(weights, data_root, device, workers=4):
    assert_no_other_gpu_jobs()
    precision = read_json("artifacts/precision_v2.json")["precision"]
    detector = Detector(weights, precision, device)
    corpus = Corpus(data_root, subset="calibration")
    write_json("artifacts/environment_timing_v2.json", {"protocol_version": 2, "cuda_device": detector.hardware,
                                                       "source": provenance()})
    from concurrent.futures import ThreadPoolExecutor
    worker_trials = []
    for count in [1, 2, 4, 8]:
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=count) as pool:
            list(pool.map(lambda i: detector.preprocess(corpus.image(i)), corpus.ids[:100]))
        worker_trials.append({"workers": count, "elapsed_seconds": time.monotonic() - start})
    selected_workers = min(worker_trials, key=lambda r: (r["elapsed_seconds"], r["workers"]))["workers"]
    write_json("artifacts/cpu_workers_v2.json", {"protocol_version": 2, "selection_subset": "calibration",
               "images_per_candidate": 100, "trials": worker_trials, "selected_workers": selected_workers})
    e0 = Executor(detector, buckets=(1,), slots_per_bucket=1, pinned=False)
    result0 = pilot(e0, corpus, requests=1000, time_cap_s=300)
    write_json("artifacts/pilot_E0_v2.json", result0)
    del e0
    e1 = Executor(detector, buckets=(1,), slots_per_bucket=1)
    result1 = pilot(e1, corpus, requests=1000, time_cap_s=300)
    write_json("artifacts/pilot_E1_v2.json", result1)
    del e1
    graph = Executor(detector, backend="graph")
    eager = Executor(detector)
    # Same-precision graph capture should preserve operations. Use the protocol
    # starting tolerances without widening them after a failure.
    stress = stress_outputs(eager, graph, corpus, requests=1000, atol=1e-5, rtol=1e-4)
    write_json("artifacts/stress_v2.json", stress)
    if not stress["memory_stable_after_warmup"]:
        raise RuntimeError("GPU allocations continue growing after correctness warmup")
    table = service_table(graph, corpus, samples=100)
    write_json("artifacts/service_table_v2.json", table)
    write_json("artifacts/graph_pool_v2.json", graph.memory())
    return {"precision": precision, "s_base_s": result1["e2e_p95_ms"] / 1000,
            "e1_capacity_rps": result1["serial_rps"], "service_ns": table["service_ns"]}


def _score(records):
    return (sum(r["ontime"] for r in records),
            -float(np.mean([r["latency_p95_ms"] for r in records if r["latency_p95_ms"] is not None])))


def calibrate(weights, data_root, device, workers=4, resume=True):
    assert_no_other_gpu_jobs()
    workers = read_json("artifacts/cpu_workers_v2.json")["selected_workers"]
    base = {"precision": read_json("artifacts/precision_v2.json")["precision"],
            "s_base_s": read_json("artifacts/pilot_E1_v2.json")["e2e_p95_ms"] / 1000,
            "e1_capacity_rps": read_json("artifacts/pilot_E1_v2.json")["serial_rps"],
            "service_ns": read_json("artifacts/service_table_v2.json")["service_ns"]}
    detector = Detector(weights, base["precision"], device)
    corpus = Corpus(data_root, subset="calibration")
    graph = Executor(detector, backend="graph")
    eager = Executor(detector, backend="eager")
    compile_path = Path("artifacts/compile_v2.json")
    compile_record = read_json(compile_path) if compile_path.exists() else {"status": "not_attempted"}
    compiled = Executor(detector, backend="compile") if compile_record["status"] == "passed" and compile_record["precision"] == base["precision"] else None
    service = {int(k): v for k, v in base["service_ns"].items()}
    candidates = [{"max_batch": b, "wait_ms": w} for b, w in itertools.product(graph.buckets, [0, 1, 2])]
    selected = {}
    evidence = []
    # A shared Poisson seed and four loads are predeclared for both baselines.
    # Every candidate gets the protocol's 10 s warmup / 20 s measurement.
    calibration_seed = 17
    def evaluate(policy, executor, settings, rates, phase):
        records = []
        for load_index, rate in enumerate(rates):
            case = f"{phase}_{policy}_b{settings['max_batch']}_w{settings['wait_ms']}_l{load_index}"
            trace = make_trace("poisson", rate, calibration_seed, corpus.ids, base["s_base_s"], 10, 20)
            from .trace import trace_hash
            identity = object_hash({"policy": policy, "settings": settings, "rate_rps": rate,
                                   "trace_sha256": trace_hash(trace), "precision": base["precision"],
                                   "workers": workers, "service_ns": service,
                                   "runtime_sha256": {name: sha256(Path("src/cvpr_project") / name) for name in
                                       ["executor.py", "buffer_pool.py", "graph_pool.py", "serve.py", "scheduler.py", "model.py"]}})
            case_root = Path("artifacts/calibration_runs") / case
            previous = []
            if resume:
                for p in sorted(case_root.glob("*/manifest.json")):
                    old = read_json(p)
                    if old["status"] == "completed" and old["calibration_config_sha256"] == identity:
                        previous.append(p.parent)
            if len(previous) > 1:
                raise ValueError(f"Duplicate completed calibration candidate {case}")
            if previous:
                output = previous[0]
                record = read_json(output / "metrics.json")
            else:
                assert_no_other_gpu_jobs()
                rid = run_id(case)
                output = case_root / rid
                manifest = {"protocol_version": 2, "run_id": rid, "status": "running", "policy": policy,
                            "phase": phase, "settings": settings, "rate_rps": rate, "trace_seed": calibration_seed,
                            "trace_sha256": trace_hash(trace), "calibration_config_sha256": identity,
                            "model": read_json("artifacts/model_v2.json"), "precision": base["precision"],
                            "split_sha256": corpus.split["split_sha256"], "subset": "calibration",
                            "cpu_workers": workers, "cuda_device_info": detector.hardware,
                            "source": provenance(), "started_utc": stamp(), "gpu_before": gpu_snapshot()}
                write_json(output / "manifest.json", manifest)
                started = time.monotonic()
                print(f"calibration {case} rate={rate:.2f}", flush=True)
                try:
                    record, _ = run_serving(executor, corpus, trace, policy, settings, service, base["s_base_s"],
                                            warmup_s=10, measurement_s=20, drain_s=30, workers=workers, output=output)
                    manifest["status"] = "failed" if record["loadgen_limited"] else "completed"
                except BaseException as exc:
                    manifest.update(status="failed", error_type=type(exc).__name__, error_message=str(exc))
                    raise
                finally:
                    manifest.update(elapsed_seconds=time.monotonic() - started, ended_utc=stamp(), gpu_after=gpu_snapshot())
                    write_json(output / "manifest.json", manifest)
                write_json(output / "settings.json", {"phase": phase, "policy": policy, "settings": settings,
                                                      "rate_rps": rate, "trace_seed": calibration_seed})
            if record["loadgen_limited"]:
                raise RuntimeError(f"Calibration load generator exceeds lag gate in {case}")
            records.append(record)
            evidence.append({"case": case, "run_id": output.name, "policy": policy, "settings": settings, "rate_rps": rate,
                             **{k: record[k] for k in ["offered", "ontime", "slo_success", "latency_p95_ms", "generator_lag_p99_ms"]}})
        return records
    initial_rates = [base["e1_capacity_rps"] * x for x in [.5, 1, 2, 4]]
    for policy, executor in [("R0", eager), ("F0", graph)]:
        scored = []
        for candidate in candidates:
            result = evaluate(policy, executor, candidate, initial_rates, "initial")
            scored.append((*_score(result), -candidate["max_batch"], -candidate["wait_ms"], candidate))
        selected[policy] = max(scored, key=lambda x: x[:-1])[-1]
    capacity_identity = object_hash({"initial_F0": selected["F0"], "base": base, "workers": workers,
        "initial_evidence": [e["run_id"] for e in evidence if e["policy"] == "F0"],
        "runtime": {name: sha256(Path("src/cvpr_project") / name) for name in
                    ["executor.py", "buffer_pool.py", "graph_pool.py", "serve.py", "scheduler.py", "model.py"]}})
    capacity_path = Path("artifacts/capacity_v2.json")
    if capacity_path.exists():
        cap = read_json(capacity_path)
        if cap.get("capacity_config_sha256") != capacity_identity:
            raise ValueError("Existing capacity belongs to another calibration; preserve it in a new evidence directory")
    else:
        cap = capacity(graph, corpus, selected["F0"]["max_batch"], seconds=90, workers=workers)
        cap["capacity_config_sha256"] = capacity_identity
        write_json(capacity_path, cap)
    rates = [cap["lambda_ref_rps"] * x for x in [.3, .6, .9, 1.1]]
    for policy, executor in [("R0", eager), ("F0", graph)]:
        scored = []
        for candidate in candidates:
            result = evaluate(policy, executor, candidate, rates, "review")
            scored.append((*_score(result), -candidate["max_batch"], -candidate["wait_ms"], candidate))
        selected[policy] = max(scored, key=lambda x: x[:-1])[-1]
    if compiled is not None:
        scored = []
        for candidate in candidates:
            result = evaluate("C0", compiled, candidate, rates, "review")
            scored.append((*_score(result), -candidate["max_batch"], -candidate["wait_ms"], candidate))
        selected["C0"] = max(scored, key=lambda x: x[:-1])[-1]
    scored = []
    for tau in [0, 1, 2]:
        candidate = {"max_batch": max(graph.buckets), "wait_ms": tau}
        result = evaluate("D0", graph, candidate, rates, "review")
        scored.append((*_score(result), -tau, candidate))
    selected["D0"] = max(scored, key=lambda x: x[:-1])[-1]
    selected["A0"] = selected["F0"].copy()
    for policy in ["E0", "E1", "G0", "P0"]:
        selected[policy] = {"max_batch": 1, "wait_ms": 0}
    result = {"protocol_version": 2, "status": "calibrated", **base,
              "lambda_ref_rps": cap["lambda_ref_rps"], "rate_multipliers": [.3, .6, .9, 1.1],
              "rates_rps": rates, "policies": selected, "common_buckets": list(graph.buckets),
              "cpu_workers": workers, "calibration_trace_type": "poisson", "calibration_trace_seed": calibration_seed,
              "search_candidates": candidates, "initial_rate_multipliers": [.5, 1, 2, 4],
              "search_warmup_seconds": 10, "search_measurement_seconds": 20, "evidence": evidence,
              "lambda_ref_frozen_after_initial_F0_selection": True,
              "selection_objective": "sum_ontime_then_lower_mean_p95_then_smaller_batch_then_wait"}
    write_json("artifacts/calibration_v2.json", result)
    return result


def freeze(config_path="configs/project.json"):
    config = read_json(config_path)
    if config["protocol_version"] != 2:
        raise ValueError("Only protocol v2 is supported")
    calibration = read_json("artifacts/calibration_v2.json")
    model = read_json("artifacts/model_v2.json")
    data = read_json("artifacts/data_v2.json")
    result = {"protocol_version": 2, "status": "frozen", "config": config, "config_sha256": sha256(config_path),
              "calibration": calibration, "model": model, "data": data, "source": provenance(),
              "output_scope": "RAM_encoded_image_bytes_to_CPU_detections",
              "frozen_before_held_out": True, "tf32": False, "attention_implementation": "eager"}
    compile_path = Path("artifacts/compile_v2.json")
    result["compile"] = sanitize_public(read_json(compile_path)) if compile_path.exists() else {"status": "not_attempted"}
    if result["compile"].get("status") == "passed" and result["compile"].get("precision") != calibration["precision"]:
        raise ValueError("Compile compatibility evidence belongs to another precision")
    result["frozen_config_sha256"] = object_hash(result)
    path = Path("artifacts/frozen_v2.json")
    if path.exists() and read_json(path)["frozen_config_sha256"] != result["frozen_config_sha256"]:
        raise ValueError("Existing freeze differs; use a new evidence directory, never overwrite a freeze")
    write_json(path, result)
    return result
