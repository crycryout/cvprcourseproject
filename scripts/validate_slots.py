#!/usr/bin/env python3
"""Revalidate both slots, including reversed box outputs, before freezing."""
from pathlib import Path
import shutil
from cvpr_project.data import Corpus
from cvpr_project.executor import Executor
from cvpr_project.model import Detector
from cvpr_project.quality import stress_outputs
from cvpr_project.runs import assert_no_other_gpu_jobs, gpu_lease, provenance, read_json, stamp, write_json
from cvpr_project.hardware import require_device_scope


def main():
    cal = read_json("artifacts/calibration_v2.json")
    compiled_record = read_json("artifacts/compile_v2.json")
    expected = 252 if compiled_record["status"] == "passed" else 204
    assert cal["status"] == "calibrated" and len(cal["evidence"]) == expected
    with gpu_lease("post-calibration-slot-validation"):
        assert_no_other_gpu_jobs()
        detector = Detector(precision=cal["precision"])
        require_device_scope(detector.hardware, cal["cuda_device_info"])
        corpus = Corpus(subset="calibration")
        eager = Executor(detector)
        graph = Executor(detector, backend="graph")
        result = stress_outputs(eager, graph, corpus, requests=1000)
        result.update(source=provenance(), hardware=detector.hardware, validated_utc=stamp())
        initial = Path("artifacts/stress_v2.json")
        archived = Path("artifacts/stress_before_reverse_box_check_v2.json")
        if not archived.exists():
            shutil.copy2(initial, archived)
        write_json(initial, result)
        if compiled_record["status"] == "passed":
            compiled = Executor(detector, backend="compile")
            result = stress_outputs(eager, compiled, corpus, requests=1000,
                                   atol=compiled_record["raw_logits_atol"], rtol=0,
                                   boxes_atol=compiled_record["raw_boxes_atol"])
            result.update(source=provenance(), hardware=detector.hardware, validated_utc=stamp())
            write_json("artifacts/compile_slot_validation_v2.json", result)
            compiled_record["post_calibration_slot_revalidation"] = result
            write_json("artifacts/compile_v2.json", compiled_record)
        print("Both slots and reverse-order logits/boxes passed calibration stress", flush=True)


if __name__ == "__main__":
    main()
