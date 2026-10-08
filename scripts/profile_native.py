#!/usr/bin/env python3
"""Separate Nsight capture on calibration inputs; never formal timing evidence."""
import argparse
from pathlib import Path
import torch
from cvpr_project.data import Corpus
from cvpr_project.executor import Executor
from cvpr_project.model import Detector
from cvpr_project.matrix import load_frozen
from cvpr_project.hardware import require_device_scope
from cvpr_project.runs import assert_no_other_gpu_jobs, gpu_lease, provenance, read_json, write_json
from cvpr_project.scheduler import Ready
from cvpr_project.serve import run_serving
from cvpr_project.trace import make_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["eager", "graph", "compile"], default="graph")
    parser.add_argument("--mode", choices=["serving", "forward"], default="serving")
    parser.add_argument("--bucket", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--rate-multiplier", type=float, choices=[.3, 1.1], default=1.1)
    parser.add_argument("--output", default="artifacts/native_profile.json")
    args = parser.parse_args()
    with gpu_lease(f"native-profile-{args.mode}-{args.backend}"):
        assert_no_other_gpu_jobs()
        frozen = load_frozen("artifacts/frozen_v2.json")
        cal = frozen["calibration"]
        detector = Detector(precision=cal["precision"])
        require_device_scope(detector.hardware, cal["cuda_device_info"])
        corpus = Corpus(subset="calibration")
        executor = Executor(detector, backend=args.backend)
        batch = [Ready(i, image_id, 0, 0, 0, detector.preprocess(corpus.image(image_id)))
                 for i, image_id in enumerate(corpus.ids[:args.bucket])]
        for _ in range(5):
            slot = executor.free_slot(args.bucket)
            executor.submit(slot, batch)
            slot.events["done"].synchronize()
            executor.consume(slot, corpus)
        assert_no_other_gpu_jobs()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            if args.mode == "forward":
                with torch.inference_mode(), torch.cuda.stream(executor.compute):
                    torch.cuda.nvtx.range_push("resident_forward_only")
                    for _ in range(10):
                        if slot.graph:
                            slot.graph.replay()
                        else:
                            executor.forward_fn(slot.values, slot.mask)
                    torch.cuda.nvtx.range_pop()
                executor.compute.synchronize()
                metrics = None
            else:
                policy = "E1" if args.backend == "eager" else "P0" if args.backend == "graph" else "C0"
                trace = make_trace("poisson", args.rate_multiplier * cal["lambda_ref_rps"], 42,
                                   corpus.ids, cal["s_base_s"], 1, 3)
                metrics, _ = run_serving(executor, corpus, trace, policy, cal["policies"][policy],
                    {int(k): v for k, v in cal["service_ns"].items()}, cal["s_base_s"],
                    warmup_s=1, measurement_s=3, drain_s=3, workers=cal["cpu_workers"])
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        write_json(Path(args.output), {"protocol_version": 2, "profiler_not_formal_timing": True,
            "backend": args.backend, "mode": args.mode, "bucket": args.bucket,
            "frozen_config_sha256": frozen["frozen_config_sha256"], "rate_multiplier": args.rate_multiplier,
            "input_subset": "calibration", "precision": cal["precision"], "source": provenance(),
            "hardware": detector.hardware, "metrics_with_profiler_overhead": metrics})


if __name__ == "__main__":
    main()
