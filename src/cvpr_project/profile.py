"""Separate profiling evidence and a bounded, honest compile compatibility gate."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import csv
import time
import numpy as np
import torch
from .data import Corpus
from .executor import Executor
from .model import Detector
from .runs import read_json, write_json, provenance, stamp, run_id, assert_no_other_gpu_jobs
from .scheduler import Ready
from .serve import run_serving
from .trace import make_trace


def bounded_compile_attempt(weights, data_root, device, seconds=1200):
    """Bound the whole compiler process group, preserving timeout evidence."""
    start = time.monotonic()
    cmd = [sys.executable, "-c", "from cvpr_project.profile import compile_attempt; import sys; compile_attempt(*sys.argv[1:])",
           str(weights), str(data_root), str(device)]
    process = subprocess.Popen(cmd, start_new_session=True)
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        record = read_json("artifacts/compile_v2.json") if Path("artifacts/compile_v2.json").exists() else {"protocol_version": 2}
        record.update(status="failed", error_type="CompileAttemptTimeout",
                      error_message=f"Bounded compatibility attempt exceeded {seconds} seconds",
                      elapsed_seconds=time.monotonic() - start, ended_utc=stamp(), bounded_attempt_seconds=seconds)
        write_json("artifacts/compile_v2.json", record)
    if process.returncode and Path("artifacts/compile_v2.json").exists():
        record = read_json("artifacts/compile_v2.json")
        if record["status"] == "running":
            record.update(status="failed", error_type="CompileWorkerExit", returncode=process.returncode,
                          elapsed_seconds=time.monotonic() - start)
            write_json("artifacts/compile_v2.json", record)
    return read_json("artifacts/compile_v2.json")


def compile_attempt(weights, data_root, device):
    assert_no_other_gpu_jobs()
    start = time.monotonic()
    precision_record = read_json("artifacts/precision_v2.json")
    precision = precision_record["precision"]
    record = {"protocol_version": 2, "status": "running", "started_utc": stamp(),
              "run_id": run_id("compile"),
              "backend": "inductor", "mode": "default", "automatic_cudagraphs": False,
              "compatibility_cap_seconds": 7200, "source": provenance(), "precision": precision}
    write_json("artifacts/compile_v2.json", record)
    try:
        detector = Detector(weights, precision, device)
        corpus = Corpus(data_root, subset="calibration")
        compiled = Executor(detector, backend="compile")
        eager = Executor(detector)
        max_error = 0.0
        raw_starting_failures = []
        # Compiler fusion can reorder FP32 arithmetic. Retain starting-tolerance
        # failures, then require finite absolute score and subpixel corner bounds
        # plus all-image AP; never widen these bounds after a failed acceptance.
        max_image_side = max(max(corpus.images[i]["width"], corpus.images[i]["height"]) for i in corpus.ids)
        logits_atol = .01 if precision == "fp32" else precision_record["bf16_logits_atol"]
        boxes_atol = .5 / (1.5 * max_image_side) if precision == "fp32" else precision_record["bf16_boxes_atol"]
        record.update(fp32_semantic_logit_abs_bound=.01, fp32_corner_displacement_pixel_bound=.5,
                      calibration_max_original_image_side=max_image_side)
        for b in compiled.buckets:
            for n in [1, b]:
                batch = [Ready(j, corpus.ids[j], 0, 0, 0, detector.preprocess(corpus.image(corpus.ids[j])))
                         for j in range(n)]
                cs, es = compiled.free_slot(b), eager.free_slot(b)
                compiled.submit(cs, batch)
                eager.submit(es, batch)
                cs.events["done"].synchronize()
                es.events["done"].synchronize()
                try:
                    torch.testing.assert_close(cs.host_logits[:n], es.host_logits[:n], atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(cs.host_boxes[:n], es.host_boxes[:n], atol=1e-5, rtol=1e-4)
                except AssertionError as exc:
                    raw_starting_failures.append({"bucket": b, "valid_count": n, "initial_tolerance_error": str(exc)})
                torch.testing.assert_close(cs.host_logits[:n], es.host_logits[:n], atol=logits_atol, rtol=0)
                torch.testing.assert_close(cs.host_boxes[:n], es.host_boxes[:n], atol=boxes_atol, rtol=0)
                max_error = max(max_error, float((cs.host_logits[:n].float() - es.host_logits[:n].float()).abs().max()))
                compiled.consume(cs, corpus)
                eager.consume(es, corpus)
        from .quality import stress_outputs
        stress = stress_outputs(eager, compiled, corpus, requests=1000, atol=logits_atol, rtol=0, boxes_atol=boxes_atol)
        write_json(f"artifacts/compile_stress_{precision}_v2.json", stress)
        # Raw tolerance alone does not establish visual equivalence of fusion.
        from .quality import evaluate_backend
        reference_quality = read_json(f"artifacts/quality_calibration_{precision}/metrics.json")
        quality_rows = []
        for b in compiled.buckets:
            compiled_quality, _ = evaluate_backend(compiled, corpus, bucket=b,
                                                   output=f"artifacts/quality_calibration_compile_{precision}_b{b}")
            ap_change = compiled_quality["metrics_percent"]["AP"] - reference_quality["metrics_percent"]["AP"]
            quality_rows.append({"bucket": b, "AP": compiled_quality["metrics_percent"]["AP"], "AP_change_pp": ap_change})
            if abs(ap_change) > .1:
                raise ValueError(f"Compile b{b} calibration AP change {ap_change:.4f} pp exceeds 0.1 pp")
        record.update(status="passed", max_abs_logits_error=max_error,
                      calibration_AP_change_pp=quality_rows[0]["AP_change_pp"], calibration_quality_buckets=quality_rows,
                      raw_logits_atol=logits_atol, raw_boxes_atol=boxes_atol,
                      raw_starting_tolerance_failures=raw_starting_failures,
                      stress=stress,
                      initialization=compiled.capture_metadata,
                      calibration_raw_outputs_verified=True)
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error_message=str(exc)[:3000])
    finally:
        record.update(elapsed_seconds=time.monotonic() - start, ended_utc=stamp())
        write_json("artifacts/compile_v2.json", record)
        write_json(Path("artifacts/compile_attempts") / f"{record['run_id']}.json", record)
    return record


def profile_run(weights, data_root, device, policy="P0", output="artifacts/profile_v2", rate_multiplier=1.1):
    assert_no_other_gpu_jobs()
    from .matrix import load_frozen
    frozen = load_frozen("artifacts/frozen_v2.json")
    cal = frozen["calibration"]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    precision = cal["precision"]
    s_base_s = cal["s_base_s"]
    service_ns = {int(k): v for k, v in cal["service_ns"].items()}
    detector = Detector(weights, precision, device)
    corpus = Corpus(data_root, subset="calibration")
    backend = "eager" if policy in {"E0", "E1"} else "compile" if policy == "C0" else "graph"
    executor = Executor(detector, backend=backend, pinned=policy != "E0")
    micro = []
    with torch.inference_mode():
        for b in executor.buckets:
            slot = executor.free_slot(b)
            real = [detector.preprocess(corpus.image(i)) for i in corpus.ids[:b]]
            slot.values.copy_(torch.stack([p[0] for p in real]).to(detector.device))
            slot.mask.copy_(torch.stack([p[1] for p in real]).to(detector.device))
            torch.cuda.synchronize()
            samples, submits = [], []
            with torch.cuda.stream(executor.compute):
                for n in range(60):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    host_start = time.monotonic_ns()
                    if slot.graph:
                        slot.graph.replay()
                    else:
                        executor.forward_fn(slot.values, slot.mask)
                    submits.append((time.monotonic_ns() - host_start) / 1e6)
                    end.record()
                    samples.append((begin, end))
            executor.compute.synchronize()
            durations = [a.elapsed_time(z) for a, z in samples[10:]]
            micro.append({"backend": backend, "bucket": b, "precision": precision, "samples": len(durations),
                          "device_forward_p50_ms": float(np.median(durations)),
                          "device_forward_p95_ms": float(np.percentile(durations, 95)),
                          "host_submit_p50_ms": float(np.median(submits[10:])),
                          "scope": "GPU_resident_inputs_forward_only_excludes_CPU_preprocess_and_copies"})
    with (root / "forward_microbench.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(micro[0]))
        writer.writeheader()
        writer.writerows(micro)
    assert_no_other_gpu_jobs()
    settings = cal["policies"][policy]
    workers = cal["cpu_workers"]
    offered_rate = rate_multiplier * cal["lambda_ref_rps"]
    trace = make_trace("poisson", offered_rate, 42, corpus.ids, s_base_s, 1, 3)
    # Profiler overhead is intentionally excluded from performance comparisons.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False, with_stack=False) as prof:
        metrics, _ = run_serving(executor, corpus, trace, policy, settings, service_ns, s_base_s,
                                warmup_s=1, measurement_s=3, drain_s=3, workers=workers, output=root / "requests")
    prof.export_chrome_trace(str(root / "timeline.json"))
    rows = []
    for event in prof.key_averages():
        rows.append({"name": event.key, "count": event.count,
                     "self_cpu_time_us": event.self_cpu_time_total,
                     "self_device_time_us": event.self_device_time_total})
    rows.sort(key=lambda x: x["self_device_time_us"], reverse=True)
    result = {"protocol_version": 2, "policy": policy, "precision": precision, "hardware": detector.hardware,
              "frozen_config_sha256": frozen["frozen_config_sha256"], "source": provenance(),
              "rate_multiplier": rate_multiplier, "offered_rate_rps": offered_rate,
              "settings": settings, "input_subset": "calibration",
              "profiler_not_formal_timing": True, "top_operators": rows[:40],
              "trace": "timeline.json", "forward_microbench": micro, "metrics_with_profiler_overhead": metrics}
    write_json(root / "summary.json", result)
    return result
