"""Derive public small evidence and figures exclusively from completed real runs."""
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .runs import read_json, write_json, sha256, sanitize_public, resource_totals
from .matrix import load_frozen
from .intervals import overlapping_duration


def ablation_figures(frame, output):
    """Plot execution controls and EDF/service-feasibility controls separately."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for column, kind in enumerate(["periodic", "poisson", "burst"]):
        subset = frame[(frame.arrival_type == kind) & frame.policy.isin(["E0", "E1", "G0", "P0"])]
        for row, metric in enumerate(["latency_p95_ms", "goodput_rps"]):
            ax = axes[row, column]
            for policy, group in subset.groupby("policy"):
                stats = group.groupby("rate_multiplier")[metric].agg(["mean", "min", "max"])
                ax.plot(stats.index, stats["mean"], "o-", label=policy)
                ax.fill_between(stats.index, stats["min"], stats["max"], alpha=.12)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(kind)
            else:
                ax.set_xlabel("Offered rate / common F0 capacity")
    axes[0, 0].set_ylabel("Completed p95 latency (ms)")
    axes[1, 0].set_ylabel("On-time results per second")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(output / "execution_ablation.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    labels = ["F0", "A0", "D0", "C0"]
    for ax, kind in zip(axes, ["poisson", "burst"]):
        subset = frame[(frame.arrival_type == kind) & (frame.rate_multiplier >= .9)]
        width = .18
        for index, rho in enumerate([.9, 1.1]):
            group = subset[subset.rate_multiplier == rho].groupby("policy").slo_success.agg(["mean", "min", "max"])
            x = np.arange(len(labels)) + (index - .5) * width
            mean = [group.loc[p, "mean"] if p in group.index else np.nan for p in labels]
            low = [group.loc[p, "mean"] - group.loc[p, "min"] if p in group.index else 0 for p in labels]
            high = [group.loc[p, "max"] - group.loc[p, "mean"] if p in group.index else 0 for p in labels]
            ax.bar(x, mean, width, label=f"load {rho}", yerr=[low, high], capsize=3)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_title(f"Synthetic {kind}")
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(0, 1)
        ax.legend()
    axes[0].set_ylabel("SLO success / all offered requests")
    fig.tight_layout()
    fig.savefig(output / "deadline_ablation.png", dpi=160)
    plt.close(fig)


def detection_figure(root, output):
    attribution_path = root / "detections/attribution.json"
    if not attribution_path.exists():
        return
    examples = read_json(attribution_path)["examples"]
    if not examples:
        return
    fig, axes = plt.subplots(1, len(examples), figsize=(12, 3.6), squeeze=False)
    for ax, example in zip(axes[0], examples):
        ax.imshow(plt.imread(root / f"detections/coco_{example['image_id']}_detections.jpg"))
        ax.set_title(f"COCO image {example['image_id']}\nscore >= 0.7", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "detection_examples.png", dpi=160)
    plt.close(fig)


def timeline_figure(trace_path, output):
    trace = read_json(trace_path)
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and e.get("dur", 0) > 0 and
              e.get("cat") in {"kernel", "gpu_memcpy"}]
    if not events:
        return {"status": "no_gpu_trace_events", "overlap_verified": False}
    streams = sorted({str(e.get("args", {}).get("stream", e.get("tid", "unknown"))) for e in events})
    kernels = [e for e in events if e.get("cat") == "kernel"]
    copies = [e for e in events if e.get("cat") == "gpu_memcpy"]
    copy_us = sum(c["dur"] for c in copies)
    overlap_us = overlapping_duration([(k["ts"], k["ts"] + k["dur"]) for k in kernels],
                                      [(c["ts"], c["ts"] + c["dur"]) for c in copies])
    first = min(e["ts"] for e in events)
    # Show 100 ms from the beginning of measured GPU activity.
    window_ms = 100
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for e in events:
        start_ms = (e["ts"] - first) / 1000
        if start_ms > window_ms:
            continue
        stream = str(e.get("args", {}).get("stream", e.get("tid", "unknown")))
        color = "#2878b5" if e.get("cat") == "kernel" else "#e07b39"
        ax.broken_barh([(start_ms, e["dur"] / 1000)], (streams.index(stream) - .35, .7), facecolors=color)
    ax.set_yticks(range(len(streams)), [f"CUDA stream {s}" for s in streams])
    ax.set_xlim(0, window_ms)
    ax.set_xlabel("Time within separate profile (ms)")
    ax.set_title("Actual CUDA timeline: blue = kernels; orange = memory copies")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {"status": "measured", "gpu_events": len(events), "memcpy_events": len(copies),
            "copy_duration_us": copy_us, "copy_kernel_overlap_us": overlap_us,
            "fraction_copy_time_overlapping_kernel": overlap_us / copy_us if copy_us else 0,
            "overlap_verified": overlap_us > 0, "clock": "profiler_correlated_CUPTI_trace_clock"}


def analyze(runs, output, frozen_path):
    root, output = Path(runs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    frozen = load_frozen(frozen_path)
    records, failures, batch_rows = [], [], []
    seen = set()
    for path in sorted(root.glob("*/manifest.json")):
        manifest = read_json(path)
        if manifest["frozen_config_sha256"] != frozen["frozen_config_sha256"]:
            continue
        if manifest["status"] not in {"completed", "completed_with_failures"}:
            failures.append({"run_id": manifest["run_id"], "status": manifest["status"],
                             "failure": manifest.get("failure", manifest.get("failure_type"))})
            continue
        metrics = read_json(path.parent / "metrics.json")
        if metrics["loadgen_limited"]:
            raise ValueError("A loadgen-limited run cannot enter measured summary")
        key = manifest["case_id"]
        if key in seen:
            raise ValueError(f"Duplicate successful case {key}; select a predeclared run, not the fastest")
        seen.add(key)
        row = {"run_id": manifest["run_id"], "case_id": key, "policy": manifest["policy"],
               "arrival_type": manifest["arrival_type"], "rate_multiplier": manifest["rate_multiplier"],
               "trace_seed": manifest["trace_seed"], "offered_rate_rps": manifest["offered_rate_rps"],
               "backend": manifest["backend"], "precision": manifest["precision"],
               **{k: v for k, v in metrics.items() if k not in
                  {"memory", "gpu_activity_samples", "completed_measurement_batch_distribution"}}}
        row.update(gpu_peak_allocated_gib=metrics["memory"]["gpu_peak_allocated_bytes"] / 2**30,
                   gpu_reserved_gib=metrics["memory"]["gpu_reserved_bytes"] / 2**30,
                   pinned_host_mib=metrics["memory"]["pinned_host_bytes"] / 2**20)
        batch_rows.extend({"run_id": manifest["run_id"], "policy": manifest["policy"],
                           "arrival_type": manifest["arrival_type"], "rate_multiplier": manifest["rate_multiplier"],
                           "trace_seed": manifest["trace_seed"], "bucket_occupancy": bucket, "completed_batches": count}
                          for bucket, count in metrics["completed_measurement_batch_distribution"].items())
        records.append(row)
        # Small manifests retain hashes and public provenance; full request logs stay local.
        public_manifest = {k: v for k, v in manifest.items() if k not in {"cumulative_resources", "failure_message"}}
        write_json(evidence / f"{manifest['run_id']}.json", {"manifest": public_manifest, "metrics": metrics})
    if not records:
        raise ValueError("No completed real runs; refuse to manufacture figures")
    frame = pd.DataFrame(records)
    frame.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(output / "batch_distribution.csv", index=False)
    ablation_figures(frame, figures)
    detection_figure(output, figures)
    metrics_to_plot = [("latency_p95_ms", "Completed-request p95 latency (ms)", "latency_p95"),
                       ("latency_p99_ms", "Completed-request p99 latency (ms)", "latency_p99"),
                       ("goodput_rps", "On-time results per second (all-offered denominator)", "goodput"),
                       ("slo_success", "SLO success fraction (all offered requests)", "slo_success"),
                       ("completion_ratio", "Completion fraction (including drain)", "completion_ratio")]
    for metric, ylabel, name in metrics_to_plot:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
        for ax, kind in zip(axes, ["periodic", "poisson", "burst"]):
            subset = frame[frame.arrival_type == kind]
            for policy, group in subset.groupby("policy"):
                ag = group.groupby("rate_multiplier")[metric].agg(["mean", "min", "max"])
                ax.plot(ag.index, ag["mean"], marker="o", label=policy)
                ax.fill_between(ag.index, ag["min"], ag["max"], alpha=.12)
            ax.set_title(f"Synthetic {kind} arrivals")
            ax.set_xlabel("Offered rate / common F0 capacity")
            ax.grid(alpha=.2)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=8)
        fig.tight_layout(rect=[0, 0, 1, .9])
        fig.savefig(figures / f"{name}.png", dpi=160)
        plt.close(fig)
    # Paired seed differences; do not treat individual requests as independent runs.
    pairs = []
    for (kind, rho, seed), group in frame.groupby(["arrival_type", "rate_multiplier", "trace_seed"]):
        rows = {r["policy"]: r for r in group.to_dict("records")}
        for baseline in ["F0", "A0", "C0"]:
            if "D0" in rows and baseline in rows:
                d, b = rows["D0"], rows[baseline]
                pairs.append({"arrival_type": kind, "rate_multiplier": rho, "trace_seed": seed,
                              "baseline": baseline, "D0_run_id": d["run_id"], "baseline_run_id": b["run_id"],
                              "goodput_difference_rps": d["goodput_rps"] - b["goodput_rps"],
                              "slo_success_difference": d["slo_success"] - b["slo_success"],
                              "latency_p95_difference_ms": d["latency_p95_ms"] - b["latency_p95_ms"]})
    pd.DataFrame(pairs).to_csv(output / "paired_comparisons.csv", index=False)
    quality = read_json("artifacts/quality_v2.json")
    qrows = [{"backend": q["backend"], "bucket": q["bucket"], "precision": q["precision"],
              "evaluated_images": q["evaluated_images"], **q["metrics_percent"],
              "AP_change_pp": q["AP_change_from_same_precision_eager_b1_pp"],
              "quality_gate_passed": q["quality_gate_passed"], "predictions_sha256": q.get("predictions_sha256")}
             for q in quality["rows"]]
    pd.DataFrame(qrows).to_csv(output / "quality.csv", index=False)
    profiles = []
    microbench = []
    for trace_path in Path("artifacts").glob("profile*/timeline.json"):
        summary_path = trace_path.parent / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        if summary.get("frozen_config_sha256") != frozen["frozen_config_sha256"]:
            continue
        write_json(output / f"{trace_path.parent.name}.json", sanitize_public(summary))
        profiles.append({"profile": trace_path.parent.name, **timeline_figure(trace_path, figures / f"{trace_path.parent.name}_timeline.png")})
        micro_path = trace_path.parent / "forward_microbench.csv"
        if micro_path.exists():
            microbench.extend({"profile": trace_path.parent.name, **row}
                              for row in pd.read_csv(micro_path).to_dict("records"))
    pd.DataFrame(microbench).to_csv(output / "forward_microbench.csv", index=False)
    write_json(output / "profile_overlap.json", {"protocol_version": 2, "profiles": profiles})
    for name in ["precision_v2.json", "stress_v2.json", "graph_pool_v2.json", "environment_v2.json",
                 "environment_timing_v2.json", "compile_v2.json", "cpu_workers_v2.json", "input_contract_v2.json",
                 "service_table_v2.json", "capacity_v2.json", "preflight_delivery_v2.json"]:
        p = Path("artifacts") / name
        if p.exists():
            write_json(output / name, sanitize_public(read_json(p)))
    attempts = []
    for p in sorted(Path("artifacts/compile_attempts").glob("*.json")):
        record = sanitize_public(read_json(p))
        write_json(evidence / f"compile_attempt_{p.name}", record)
        attempts.append({"file": f"compile_attempt_{p.name}", "status": record.get("status"),
                         "precision": record.get("precision"), "elapsed_seconds": record.get("elapsed_seconds")})
    initial_failure = Path("artifacts/compile-initial-mode-options-failure.json")
    if initial_failure.exists():
        write_json(evidence / initial_failure.name, sanitize_public(read_json(initial_failure)))
    selected_calibration = {r["run_id"] for r in frozen["calibration"]["evidence"]}
    calibration_attempts = []
    for p in sorted(Path("artifacts/calibration_runs").glob("*/*/manifest.json")):
        m = read_json(p)
        calibration_attempts.append({"run_id": m["run_id"], "status": m["status"],
            "selected_for_frozen_calibration": m["run_id"] in selected_calibration,
            "elapsed_seconds": m.get("elapsed_seconds"), "config_sha256": m["calibration_config_sha256"],
            "error_type": m.get("error_type")})
    write_json(output / "compatibility_attempts.json", {"protocol_version": 2, "compile_attempts": attempts,
                                                        "calibration_attempts": calibration_attempts})
    ledger = Path("artifacts/cost_ledger.jsonl")
    if ledger.exists():
        entries = [json.loads(line) for line in ledger.read_text().splitlines()]
        write_json(output / "resource_ledger.json", {"protocol_version": 2,
                   "entries": [{k: v for k, v in item.items() if k not in {"pid", "start_unix"}} for item in entries]})
    # Pilot rows are bounded real evidence; retain their table and cost summaries.
    pilots = []
    for policy in ["E0", "E1"]:
        p = read_json(f"artifacts/pilot_{policy}_v2.json")
        pilots.append({"policy": policy, **{k: v for k, v in p.items() if k != "rows"}})
        pd.DataFrame(p["rows"]).to_csv(output / f"pilot_{policy}.csv", index=False)
    write_json(output / "pilot_summary.json", {"protocol_version": 2, "rows": pilots})
    write_json(output / "frozen_v2.json", frozen)
    write_json(output / "delivery.json", {"protocol_version": 2, "completed_cases": len(records),
                "expected_main_plus_ablation": 264, "compile_baseline_expected": 36 if frozen["compile"]["status"] == "passed" else 0,
                "failures": failures, "quality_gate_passed": quality["all_quality_gates_passed"],
                "resource_totals": resource_totals(), "frozen_config_sha256": frozen["frozen_config_sha256"]})
    print(f"Analysis generated from {len(records)} real runs", flush=True)
    from .report import write_report
    write_report(output)
    return frame
