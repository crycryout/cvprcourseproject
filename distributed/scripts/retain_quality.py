#!/usr/bin/env python3
"""Retain unchanged-executor AP with explicit origin, never relabel measurements."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from batchserve.common import digest, file_hash, write_json


def contract(source, names=("Images", "connect_workers")):
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    return {name: hashlib.sha256(ast.get_source_segment(source, nodes[name]).encode()).hexdigest() for name in names}


def verify_compatibility(current, origin, old_benchmark):
    old = dict(origin)
    old_hash = old.pop("freeze_sha256")
    assert digest(old) == old_hash
    old_files, new_files = origin["source_sha256"], current["source_sha256"]
    changed = "src/batchserve/benchmark.py"
    assert set(new_files) - set(old_files) == {"src/batchserve/arrivals.py"}
    unchanged = {k: v for k, v in old_files.items() if k != changed}
    assert all(new_files[k] == v for k, v in unchanged.items()), "Inference, transport or quality source changed"
    assert hashlib.sha256(old_benchmark.encode()).hexdigest() == old_files[changed]
    functions = contract(old_benchmark)
    assert functions == contract((ROOT / changed).read_text()), "Quality data helpers changed"
    for key in ["model_revision", "split_sha256", "buckets"]:
        assert current[key] == origin[key]
    for a, b in zip(current["workers"], origin["workers"]):
        for key in ["hardware", "weights_sha256", "buckets", "cpu_workers"]:
            assert a[key] == b[key], f"Quality platform changed: {key}"
    return {"protocol_version": 3, "status": "compatible_unchanged_inference", "origin_freeze_sha256": old_hash,
            "accepted_timing_freeze_sha256": current["freeze_sha256"],
            "unchanged_source_files": unchanged, "benchmark_quality_helpers_sha256": functions,
            "changed_files": [changed, "src/batchserve/arrivals.py"],
            "reason": "Only the open-loop timing harness changed. Worker, detector, executor, codecs, quality evaluation and image packing remain identical.",
            "measurement_freeze_ids_are_preserved": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--origin", required=True)
    p.add_argument("--output", default="artifacts/distributed_v3")
    args = p.parse_args()
    old, new = Path(args.origin), Path(args.output)
    origin = json.loads((old / "frozen.json").read_text())
    current = json.loads((new / "frozen.json").read_text())
    source = (old / "quality_origin_benchmark.py").read_text()
    certificate = verify_compatibility(current, origin, source)
    if (new / "quality_compatibility.json").exists():
        assert json.loads((new / "quality_compatibility.json").read_text()) == certificate
        print("Unchanged quality retention already recorded; original measurements preserved.")
        return
    sanity = json.loads((new / "stress.json").read_text())
    assert len(sanity["results"]) == 2
    assert all(r["status"] == "passed" and r["requests"] >= 64 and r["concurrent_slot_batches_verified"] == 2 for r in sanity["results"])
    write_json(new / "stress_rebuild.json", sanity)
    quality = json.loads((old / "quality_summary.json").read_text())
    assert len(quality) == 10 and all(q["freeze_sha256"] == origin["freeze_sha256"] for q in quality)
    write_json(new / "quality_summary.json", quality)
    write_json(new / "stress.json", json.loads((old / "stress.json").read_text()))
    write_json(new / "quality_origin_frozen.json", origin)
    (new / "quality_origin_benchmark.py").write_text(source)
    write_json(new / "quality_compatibility.json", certificate)
    # Preserve paths to raw evidence without duplicating a large prediction corpus.
    quality_link = new / "quality"
    if not quality_link.exists():
        quality_link.symlink_to((old / "quality").resolve(), target_is_directory=True)
    print("Retained 10 genuine all-image AP evaluations and numerical evidence with unchanged origin freeze IDs and verified inference-source compatibility.")


if __name__ == "__main__":
    main()
