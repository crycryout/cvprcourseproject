#!/usr/bin/env python3
"""Validate the research design and emit planned validation runs, never results."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trace_schedule(config, r, p):
    model, merge = config["model"], config["merging"]
    n = model["initial_image_tokens"] + model["special_tokens"]
    trace = []
    for block in range(model["num_blocks"]):
        amount = 0 if block < merge["warmup_blocks"] else r
        n_a = (n + 1) // 2 - 1  # even full-sequence partition, excluding CLS
        n_b = n // 2
        require(0 <= amount <= n_a, f"block {block}: insufficient source tokens")
        require(amount == 0 or n_b >= 1, f"block {block}: no destination")
        k_a = min(math.ceil(p * n_a), n_a - amount) if amount else 0
        k_b = min(math.ceil(p * n_b), n_b - 1) if amount else 0
        require(k_a >= 0 and k_b >= 0, "negative protection count")
        require(n - amount - 1 >= merge["min_image_tokens"], "too few image tokens")
        trace.append({"block_1based": block + 1, "input_tokens_with_cls": n,
                      "r": amount, "protected_a": k_a, "protected_b": k_b,
                      "output_tokens_with_cls": n - amount})
        n -= amount
    return trace


def build_matrix(config):
    model, data, merge = config["model"], config["dataset"], config["merging"]
    require(model["special_tokens"] == 1 and not model["distilled"], "requires only one CLS token")
    require(model["input_size"] % model["patch_size"] == 0, "non-integral patch grid")
    require((model["input_size"] // model["patch_size"]) ** 2 == model["initial_image_tokens"],
            "image-token count disagrees with patch grid")
    require(data["train_samples"] + data["validation_samples"] == 50000, "invalid train/val split")
    require(data["test_samples"] == 10000 and data["label_mode"] == "fine", "invalid CIFAR-100 test/labels")
    require(data["validation_per_class"] * model["num_classes"] == data["validation_samples"],
            "validation class count mismatch")
    require(merge["selection_split"] == "validation" and data["test_requires_frozen_protocol"],
            "test leakage in selection settings")
    require(0 <= merge["warmup_blocks"] < model["num_blocks"], "invalid warmup blocks")
    require(merge["selection_training_seed"] in config["training"]["seeds"], "selection seed missing")
    require(all(isinstance(r, int) and r > 0 for r in merge["r_candidates"]), "r must be positive integer")
    require(len(set(merge["r_candidates"])) == len(merge["r_candidates"]), "duplicate r values")
    for p in merge["protection_candidates"] + [merge["random_development_p"]]:
        require(0 < p < 1, "protection ratio must lie in (0,1)")
    runs = []

    def add(method, r, p, purpose):
        trace = trace_schedule(config, r, p)
        runs.append({
            "config_id": f"{method}_r{r}_p{p:.2f}", "status": "planned",
            "method": method, "r": r, "p": p,
            "training_seed": merge["selection_training_seed"], "split": "validation",
            "backend": "explicit_eager", "purpose": purpose,
            "layer_trace": trace, "final_image_tokens": trace[-1]["output_tokens_with_cls"] - 1,
        })

    add("dense", 0, 0.0, "quality_reference")
    for r in merge["r_candidates"]:
        add("tome", r, 0.0, "compression_reference")
        for p in merge["protection_candidates"]:
            add("ap_tome", r, p, "validation_selection")
        add("random_protect", r, merge["random_development_p"], "development_only_not_final_ablation")
    return {"schema_version": 1, "status": "plan_not_results", "runs": runs,
            "final_ablation_rule": "Use selected AP p for Random at each r after freezing; do not reuse development p blindly."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    result = build_matrix(json.loads(raw))
    result["config_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Validated {len(result['runs'])} planned configurations; wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
