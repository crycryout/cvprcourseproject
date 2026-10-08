"""Executable evidence gates; no hypothetical commands."""
import argparse
from pathlib import Path
import json
import os
import time
from .runs import gpu_lease, read_json, write_json, provenance, stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = ["prepare-data", "prepare-model", "verify-model", "pilot", "calibrate", "freeze",
                "evaluate-quality", "run-matrix", "analyze", "profile", "compile-baseline"]
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/project.json")
        p.add_argument("--data-root", default="data/coco")
        p.add_argument("--weights", default="checkpoints/detr")
        p.add_argument("--device", default="cuda:0", help="Index inside CUDA_VISIBLE_DEVICES")
        p.add_argument("--frozen", default="artifacts/frozen_v2.json")
        p.add_argument("--workers", type=int, default=4)
        if name == "run-matrix":
            p.add_argument("--matrix", default="artifacts/serving_matrix_v2.json")
            p.add_argument("--output", default="artifacts/runs_v2")
            p.add_argument("--limit", type=int)
        if name == "analyze":
            p.add_argument("--runs", default="artifacts/runs_v2")
            p.add_argument("--output", default="results")
        if name == "profile":
            p.add_argument("--policy", choices=["E0", "E1", "G0", "P0", "F0", "D0", "C0"], default="P0")
            p.add_argument("--output", default="artifacts/profile_v2")
        if name == "compile-baseline":
            p.add_argument("--compile-backend", default="inductor", choices=["inductor"])
    args = parser.parse_args()
    if args.command == "prepare-data":
        from .data import prepare
        result = prepare(args.data_root)
        print(json.dumps({"prepared": True, "calibration_images": len(result["calibration"]),
                          "held_out_images": len(result["held_out"]), "split_sha256": result["split_sha256"]}))
        return
    if args.command == "prepare-model":
        from .model import prepare_weights
        print(json.dumps(prepare_weights(args.weights)))
        return
    if args.command == "freeze":
        from .calibration import freeze
        result = freeze(args.config)
        print(json.dumps({"status": result["status"], "frozen_config_sha256": result["frozen_config_sha256"]}))
        return
    if args.command == "analyze":
        from .analysis import analyze
        analyze(args.runs, args.output, args.frozen)
        return
    with gpu_lease(args.command):
        if args.command == "verify-model":
            from .calibration import verify_model
            result = verify_model(args.weights, args.data_root, args.device)
        elif args.command == "pilot":
            from .calibration import correctness_and_pilot
            result = correctness_and_pilot(args.weights, args.data_root, args.device, args.workers)
        elif args.command == "calibrate":
            from .calibration import calibrate
            result = calibrate(args.weights, args.data_root, args.device, args.workers)
        elif args.command == "evaluate-quality":
            from .matrix import evaluate_quality
            result = evaluate_quality(args.frozen, args.weights, args.data_root, args.device)
        elif args.command == "run-matrix":
            from .matrix import run_matrix
            result = run_matrix(args.matrix, args.frozen, args.weights, args.data_root, args.device, args.output, args.limit)
        elif args.command == "profile":
            from .profile import profile_run
            result = profile_run(args.weights, args.data_root, args.device, args.policy, args.output)
        elif args.command == "compile-baseline":
            from .profile import bounded_compile_attempt
            result = bounded_compile_attempt(args.weights, args.data_root, args.device)
        print(json.dumps(result, default=str)[:6000])
