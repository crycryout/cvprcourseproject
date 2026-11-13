#!/usr/bin/env python3
"""Audit raw events, export small evidence, and generate figures plus English PDFs."""
import argparse
from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from batchserve.common import write_json, file_hash, digest


def read(path):
    return json.loads(Path(path).read_text())


def audit_case(folder, cfg):
    m = read(folder / "metrics.json")
    raw = folder / "requests.jsonl"
    if file_hash(raw) != m["request_events_sha256"]:
        raise ValueError(f"Raw events changed: {folder.name}")
    rows = [json.loads(s) for s in raw.read_text().splitlines()]
    if len(rows) != len({r["request_id"] for r in rows}):
        raise ValueError("Duplicate request ID")
    for row in rows:
        if row["status"] == "completed":
            event = row["worker_event"]
            if (event["request_id"], event["image_id"]) != (row["request_id"], row["image_id"]):
                raise ValueError("Completed request/output identity mismatch in raw log")
    # Deliberately do not call the runtime summarizer: recompute the contract
    # directly from raw records, including zero-completion and rejection cases.
    import numpy as np
    from collections import Counter
    cohort = [r for r in rows if r["measurement"]]
    done = [r for r in cohort if r["status"] == "completed"]
    timely = sum(r["complete_s"] <= r["deadline"] for r in done)
    latencies = np.array([(r["complete_s"] - r["arrival"]) * 1000 for r in done])
    batch_sizes = {(r["worker"], r["worker_event"]["batch_id"]): r["worker_event"]["valid_count"] for r in done}
    actual = {"offered": len(cohort), "completed": len(done), "on_time": timely,
              "status_counts": dict(Counter(r["status"] for r in cohort)),
              "goodput_rps": timely / cfg["measurement_s"],
              "slo_success": timely / len(cohort) if cohort else 0,
              "completion_ratio": len(done) / len(cohort) if cohort else 0,
              "completed_batches": len(batch_sizes),
              "mean_batch_occupancy": sum(batch_sizes.values()) / len(batch_sizes) if batch_sizes else 0,
              "worker_completed": dict(Counter(str(r["worker"]) for r in done)),
              "jpeg_bytes_sent": sum(r.get("jpeg_bytes_sent", 0) for r in cohort),
              "generator_lag_p99_ms": float(np.quantile([r["generator_lag_ms"] for r in cohort], .99)) if cohort else None}
    actual.update({f"latency_p{q}_ms": float(np.quantile(latencies, q / 100)) if len(latencies) else None
                   for q in [50, 95, 99]})
    if "ingress_pickup_lag_p99_ms" in m:
        actual["ingress_pickup_lag_p99_ms"] = float(np.quantile([r["ingress_pickup_lag_ms"] for r in cohort], .99)) if cohort else None
    for key, value in actual.items():
        if isinstance(value, float):
            if not math.isclose(value, m[key], abs_tol=1e-8, rel_tol=1e-10):
                raise ValueError(f"Metric differs from raw events: {folder.name}/{key}")
        elif value != m[key]:
            raise ValueError(f"Metric differs from raw events: {folder.name}/{key}")
    return m, rows


def export(inp, out):
    import numpy as np
    import pandas as pd
    cfg = read(inp / "frozen.json")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "frozen.json", cfg)
    for src, dest in [("quality_summary.json", "quality.json"), ("stress.json", "stress.json"),
                      ("curves.json", "service_curves.json"), ("capacity.json", "capacity.json")]:
        write_json(out / dest, read(inp / src))
    if (ROOT / "artifacts/environment.json").exists():
        write_json(out / "environment.json", read(ROOT / "artifacts/environment.json"))
    if (ROOT / "artifacts/attempts.json").exists():
        write_json(out / "attempts.json", read(ROOT / "artifacts/attempts.json"))
    for name in ["quality_compatibility.json", "quality_origin_frozen.json", "quality_origin_benchmark.py", "stress_rebuild.json"]:
        if (inp / name).exists():
            shutil.copyfile(inp / name, out / name)
    metrics, stages, controls = [], [], []
    for group, container in [("runs", metrics), ("controls", controls)]:
        (out / group).mkdir(exist_ok=True)
        for p in sorted((inp / group).glob("*/metrics.json")):
            m, rows = audit_case(p.parent, cfg)
            write_json(out / group / (p.parent.name + ".json"), m)
            container.append(m)
            # Batches share GPU spans. Do not count each request as a GPU sample.
            completed = [r for r in rows if r["measurement"] and r["status"] == "completed"]
            batches = {(r["worker"], r["worker_event"]["batch_id"]): r["worker_event"] for r in completed}
            record = {k: m[k] for k in ["policy", "arrival_type", "offered_rate", "seed"]}
            record["topology"] = m.get("topology", "two_hosts")
            for key in ["cpu_wait_ms", "preprocess_ms", "ready_wait_ms", "worker_total_ms"]:
                record[key + "_median"] = float(np.median([r["worker_event"][key] for r in completed])) if completed else None
            for key in ["forward_ms", "h2d_ms", "d2h_ms", "pack_ms", "postprocess_ms"]:
                record[key + "_batch_median"] = float(np.median([b[key] for b in batches.values()])) if batches else None
            record["rpc_elapsed_ms_median"] = float(np.median([r["rpc_elapsed_ms"] for r in completed])) if completed else None
            record["driver_queue_ms_median"] = float(np.median([(r["dispatch_s"] - r["arrival"]) * 1000 for r in completed])) if completed else None
            record["single_request_batch_fraction"] = float(np.mean([b["valid_count"] == 1 for b in batches.values()])) if batches else None
            record["expired_at_driver_dispatch_fraction"] = float(np.mean([r["dispatch_s"] > r["deadline"] for r in completed])) if completed else None
            record["audited_request_count"] = len(rows)
            stages.append(record)
    pd.DataFrame(metrics).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(controls).to_csv(out / "controls.csv", index=False)
    pd.DataFrame(stages).to_csv(out / "stages.csv", index=False)
    quality = read(out / "quality.json")
    pd.DataFrame([{k: r[k] for k in ["worker", "backend", "bucket", "evaluated_images", "reference_AP", "AP_delta_pp"]}
                  | r["metrics_percent"] for r in quality]).to_csv(out / "quality.csv", index=False)
    leases = ROOT / "artifacts/worker_leases.jsonl"
    ledger = [json.loads(s) for s in leases.read_text().splitlines()] if leases.exists() else []
    active_state = ROOT / "artifacts/workers.local.json"
    if active_state.exists():
        import time
        state = read(active_state)
        if not state.get("stopped_unix_s"):
            ledger.append({"lease_id": state["lease_id"], "status": "active_at_export", "started_unix_s": state["started_unix_s"],
                           "aggregate_gpu_hours": 2 * (time.time() - state["started_unix_s"]) / 3600})
    write_json(out / "resources.json", {"protocol_version": 3, "leases": ledger,
        "new_aggregate_gpu_hours": sum(r["aggregate_gpu_hours"] for r in ledger),
        "historical_cvpr_gpu_hours": 12.473090, "extension_cap_gpu_hours": 16,
        "scope": "Both worker lifetimes including idle time, capture, failed attempts and shutdown"})
    write_json(out / "audit.json", {"protocol_version": 3, "status": "passed", "two_host_cases": len(metrics),
        "single_host_controls": len(controls), "request_logs_recomputed": len(stages),
        "raw_requests_audited": sum(r["audited_request_count"] for r in stages),
        "metrics": "All-offered accounting, completions, on-time, quantiles, bytes and unique batch occupancy recomputed from raw logs",
        "raw_data_public": False})
    return cfg, pd.DataFrame(metrics), pd.DataFrame(controls), pd.DataFrame(stages)


def figures(out, cfg, df, controls, stages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {"RR": "#64748b", "WRR": "#0891b2", "SECT": "#d97706", "GLOBAL_GPU": "#7c3aed", "GLOBAL_E2E": "#dc2626"}
    root = out / "figures"
    root.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.dpi": 180})
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [(.02, .79, .23, .17, "Independent emitter\nTimestamped pipe", "#fef9c3"),
             (.02, .32, .23, .38, "Driver process\nRAM JPEG bytes\nIngress and RPC\nDecoded results", "#e0f2fe"),
             (.35, .32, .25, .38, "Ingress scheduler\nRoute single requests\nor form global batches\nCausal work reservations", "#ede9fe"),
             (.76, .59, .22, .32, "H800 worker\n8 preprocessing threads\nDETR Graph executor\nLocal TCP", "#dcfce7"),
             (.76, .09, .22, .32, "RTX 5090 worker\n8 preprocessing threads\nDETR Graph executor\nSecond physical host", "#ffedd5")]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", facecolor=color, edgecolor="#64748b", linewidth=1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((.135, .79), (.135, .70), arrowstyle="->", mutation_scale=11, color="#475569", linewidth=1.2))
    for a, z in [((.25, .51), (.35, .51)), ((.60, .61), (.76, .74)), ((.60, .41), (.76, .25))]:
        ax.add_patch(FancyArrowPatch(a, z, arrowstyle="<->", mutation_scale=11, color="#475569", linewidth=1.2))
    ax.text(.665, .82, "Loopback TCP", ha="center", fontsize=7)
    ax.text(.665, .14, "TCP through\nSSH forward", ha="center", fontsize=7)
    ax.text(.02, .035, "E2E: scheduled arrival to matching detections at the driver; one monotonic clock", fontsize=8, color="#475569")
    fig.tight_layout(pad=.3)
    fig.savefig(root / "architecture.png", bbox_inches="tight")
    fig.savefig(root / "architecture.svg", bbox_inches="tight")
    plt.close(fig)
    for metric, label, name in [("goodput_rps", "On-time goodput (requests/s)", "goodput"),
                                ("latency_p99_ms", "Completed-request p99 (ms)", "p99"),
                                ("mean_batch_occupancy", "Mean actual requests per batch", "batching"),
                                ("slo_success", "On-time / all offered requests", "slo")]:
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1), sharey=True)
        for ax, kind in zip(axes, cfg["arrival_types"]):
            for policy in cfg["policies"]:
                data = df[(df.policy == policy) & (df.arrival_type == kind)]
                g = data.groupby("offered_rate")[metric].agg(["mean", "min", "max"]).sort_index()
                x = g.index.to_numpy(dtype=float)
                ax.plot(x, g["mean"], "o-", ms=3, lw=1.5, color=colors[policy], label=policy)
                ax.fill_between(x, g["min"].to_numpy(dtype=float), g["max"].to_numpy(dtype=float), color=colors[policy], alpha=.12)
            ax.set_title(kind.capitalize())
            ax.set_xlabel("Offered requests/s")
            ax.grid(alpha=.18)
        axes[0].set_ylabel(label)
        if name == "p99" and df[metric].max() > 1000:
            axes[0].set_yscale("log")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1), sharey=True)
    for ax, kind in zip(axes, cfg["arrival_types"]):
        for topology, frame, label in [("two", df[df.policy == "RR"], "Two hosts, RR"),
                                        ("worker0_only", controls[controls.topology == "worker0_only"], "H800 only"),
                                        ("worker1_only", controls[controls.topology == "worker1_only"], "RTX 5090 only")]:
            g = frame[frame.arrival_type == kind].groupby("offered_rate").goodput_rps.agg(["mean", "min", "max"]).sort_index()
            x = g.index.to_numpy(dtype=float)
            ax.plot(x, g["mean"], "o-", ms=3, label=label)
            ax.fill_between(x, g["min"].to_numpy(dtype=float), g["max"].to_numpy(dtype=float), alpha=.1)
        ax.set_title(kind.capitalize())
        ax.set_xlabel("Same offered requests/s")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("On-time goodput (requests/s)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "scaleout.png", bbox_inches="tight")
    plt.close(fig)


def table_markdown(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def report_text(out, cfg, df, controls, stages):
    import numpy as np
    means = df.groupby("policy").goodput_rps.mean()
    baseline = max(["RR", "WRR", "SECT"], key=lambda p: means[p])
    new = "GLOBAL_E2E"
    change = (means[new] / means[baseline] - 1) * 100 if means[baseline] else 0
    result = "did not improve" if change <= 0 else "improved"
    quality = read(out / "quality.json")
    worst_ap = max(abs(r["AP_delta_pp"]) for r in quality)
    ap_spread = max(r["metrics_percent"]["AP"] for r in quality) - min(r["metrics_percent"]["AP"] for r in quality)
    success = df.groupby("policy").slo_success.mean()
    batch = df.groupby("policy").mean_batch_occupancy.mean()
    rows = [[p, f"{means[p]:.3f}", f"{success[p] * 100:.2f}", f"{batch[p]:.3f}"] for p in cfg["policies"]]
    qrows = [[str(j), backend, str(b), f"{r['metrics_percent']['AP']:.6f}", f"{r['AP_delta_pp']:+.6f}"]
             for j in range(2) for r in quality if r["worker"] == j for backend, b in [(r["backend"], r["bucket"])]]
    resources = read(out / "resources.json")
    stage_rows = []
    dispatch_rows = []
    for p in cfg["policies"]:
        g = stages[(stages.policy == p) & (stages.topology == "two_hosts")]
        stage_rows.append([p] + [f"{g[k].mean():.2f}" for k in ["driver_queue_ms_median", "cpu_wait_ms_median", "preprocess_ms_median", "forward_ms_batch_median", "rpc_elapsed_ms_median"]])
        dispatch_rows.append([p, f"{100 * g.single_request_batch_fraction.mean():.2f}",
                              f"{100 * g.expired_at_driver_dispatch_fraction.mean():.2f}"])
    paired = df.pivot(index=["arrival_type", "offered_rate", "seed"], columns="policy", values="goodput_rps")
    estimator_difference = paired.GLOBAL_E2E - paired.GLOBAL_GPU
    estimator_wins = int((estimator_difference > 1e-9).sum())
    estimator_losses = int((estimator_difference < -1e-9).sum())
    estimator_ties = len(estimator_difference) - estimator_wins - estimator_losses
    scale_rows = []
    for rate in cfg["rates"]:
        two = df[(df.policy == "RR") & (df.offered_rate == rate)].goodput_rps.mean()
        singles = [controls[(controls.topology == f"worker{j}_only") & (controls.offered_rate == rate)].goodput_rps.mean() for j in range(2)]
        best_single = max(singles)
        scale_rows.append([f"{rate:.3f}", f"{two:.3f}", *(f"{v:.3f}" for v in singles),
                           f"{(two / best_single - 1) * 100:+.1f}" if best_single else "undefined"])
    local = "; ".join(f"worker {j}: max batch {x['max_batch']}, timeout {x['wait_ms']} ms" for j, x in enumerate(cfg["local_settings"]))
    compatibility_note = ""
    if (out / "quality_compatibility.json").exists():
        certificate = read(out / "quality_compatibility.json")
        compatibility_note = ("The final timing harness uses an independent timestamped arrival process after a cooperative-event-loop "
            "generator exceeded its lag limit. All timing cases were recalibrated and rerun. The ten genuine AP evaluations "
            "and 1024-request numerical checks are retained from the preceding campaign, whose original measurement freeze "
            f"IDs ({certificate['origin_freeze_sha256'][:12]}) remain unchanged. A published compatibility certificate verifies "
            "identical inference, transport, image-packing and quality code, weights, split and per-worker hardware/software. "
            "Offline AP does not use the arrival generator. A further 64-request-per-worker numerical and concurrent-slot sanity "
            "check passed in the rebuilt worker processes. The origin configuration and benchmark source are included for a "
            "self-contained CPU-only provenance audit.")
    env = read(out / "environment.json") if (out / "environment.json").exists() else {}
    cpus = "; ".join(f"{name}: {x['cpu'].get('model name', 'unknown CPU')}, {x['logical_cpus']} logical CPUs, Python {x['python']}, NVIDIA driver {','.join(x.get('nvidia_driver_versions', []))}" for name, x in env.items() if isinstance(x, dict) and "cpu" in x)
    s = f"""# Where Should We Batch?
## Coordinating request routing and batching for heterogeneous object detection serving

### Abstract
We implement and evaluate a real two-machine DETR inference service that compares
three request-routing policies with worker-local batching against two global
batching policies. The latter differ only in whether dispatch decisions use GPU
forward profiles or end-to-end RPC service profiles. Across {len(df)} paired
two-host configurations and {len(controls)} single-host controls, global E2E
batching {result} mean deadline goodput relative to the strongest measured routing
baseline ({baseline}): {means[new]:.3f} versus {means[baseline]:.3f} requests/s
({change:+.1f}%). Ten independent 4000-image evaluations establish a maximum
absolute within-worker AP change of {worst_ap:.6f} percentage points. The study
uses real COCO JPEGs and synthetic open-loop arrivals, preserves all offered
requests in the SLO denominator, and releases source, frozen configurations and
auditable small evidence. Results apply to this workload and two-host platform.

### 1. Problem and research questions
In a replicated inference service, routing an image immediately to a worker can
reduce that worker's queueing delay but can fragment opportunities to form larger
GPU batches. Global batching pools arrivals, but its waiting and dispatch decisions
depend on an estimate of remaining work. A GPU forward estimate omits CPU decode,
image transformation, packing and communication; a conservative E2E estimate can
also reduce the batch sizes considered feasible under short deadlines. We ask:
RQ1, how do global and worker-local batching compare at the same offered load?
RQ2, what changes when the global scheduler uses an E2E rather than GPU-only
service curve? RQ3, how does scale-out compare with each single worker at the same
arrival trace? Adding a worker is measured, not assumed to improve every metric.

The completed CVPR course project supplies the immutable detector and executor.
Its deadline heuristic had lower batch occupancy and goodput than its tuned FIFO
baseline. That observation motivated this experiment; it is not evidence that
distributed routing itself causes degradation. This extension introduces real
network transport, heterogeneous workers, global scheduling and independent
two-host measurements, while retaining separate protocol and result identities.

### 2. Related work and contribution boundary
Nexus [1] studies batching-aware allocation and execution for DNN video analytics.
Clockwork [2] exploits predictable inference execution for request-level SLOs.
Symphony [3] is the closest mechanism: it studies deferred batch scheduling,
load-proportional GPU consolidation, scalable coordination and network latency.
Consequently, global batching, EDF, deferred dispatch and service profiling are
prior work. Our contribution is a controlled implementation and empirical extension
using real DETR image processing, heterogeneous workers, identical executors and
an all-offered E2E latency contract. The global policies are simplified bounded
heuristics, not full reproductions of Symphony, Nexus or Clockwork. We do not
claim algorithmic priority or superiority over their complete implementations.

### 3. System design
![Real request path and physical-worker boundaries.](figures/architecture.png)

The driver and ingress scheduler run in one process on the H800 host. The driver
caches compressed JPEG bytes in RAM and sends actual bytes over persistent framed
TCP. The H800 worker is reached over loopback. The remote RTX 5090 worker is
reached through an authenticated SSH TCP forward; encryption and physical network
cost remain in the measured request path. Workers bind loopback and return the
complete threshold-zero detections, not acknowledgments or cached outputs.

An independent CPU process emits scheduled arrival notifications through a
nonblocking atomic pipe. Its pre-write monotonic timestamps measure producer lag;
the router separately records notification pickup delay. A full pipe causes
producer lag rather than hiding backpressure. Router JSON decoding cannot block
the producer's clock. E2E still begins at planned arrival and charges producer
lag, pickup delay and all subsequent work. The validity threshold applies to
producer emission lag; router delay remains part of the measured system.

Every request is identified by request ID, image ID and JPEG SHA-256. The receiver
checks the image bytes and dimensions. Worker CPU pools decode and transform each
arrival independently. The frozen CVPR executor owns fixed host/device buffers,
two slots per batch bucket and one GPU compute stream with explicit copy events.
One batch may execute while one is prefetched. A slot is released only after D2H
and CPU postprocessing have consumed its output. Graphs are local to each process
and GPU, never serialized or shared across machines. The original H800-only guard
is unchanged; a derived adapter generalizes device initialization and inherits the
same preprocessing, forward and output decoding methods.

Global batch membership is fixed before worker preprocessing, so the group waits
for every selected image to become ready. Local batching draws from already-ready
preprocessing jobs. Comparing these architectures includes a different readiness
barrier and cannot isolate batch fragmentation alone. The GPU-only/E2E ablation
within the global architecture retains identical readiness semantics. The global
two-outstanding-batch cap can also limit CPU admission at small batch sizes,
whereas local input queues can keep preprocessing jobs populated. Both retain
the same two-slot GPU execution limit and eight preprocessing threads.

RR sends individual arrivals alternately. WRR uses smooth weights proportional to
the best calibrated requests/s of each worker. SECT chooses the worker with the
lowest virtual predicted finish, updating reservations synchronously at dispatch.
These three policies use worker-local FIFO ready-queue batching with identical
calibration search budgets. GLOBAL_GPU and GLOBAL_E2E maintain an EDF global
queue and enumerate worker/bucket candidates for EDF prefixes. Feasible candidates
meet the earliest selected deadline under the chosen service table; the policy
maximizes requests per estimated service time, with deterministic ties. If no
candidate is feasible it serves the earliest request at batch one. It never drops
expired requests. The common waiting cap is absolute from the oldest arrival,
so new arrivals cannot reset it. At most two batches are outstanding per worker.
These are empirical heuristics, without a deadline guarantee.

### 4. Experimental method
Hardware is one full NVIDIA H800 PCIe (114 SMs, 85.02 GB device memory) and one
NVIDIA GeForce RTX 5090 (170 SMs, 33.66 GB). Both use PyTorch 2.14.0+cu130,
Transformers 4.46.3 and cuDNN 92400. {cpus}.
Eight CPU preprocessing threads per worker are fixed. Hardware differences include
the host CPUs and transport, so results are about heterogeneous whole workers,
not an isolated causal comparison of the two GPU chips. No clocks, MIG settings,
drivers or unrelated workloads are modified. An isolation monitor stops our own
workers if foreign GPU computation appears.

Weights are the frozen facebook/detr-resnet-50 detector [5] revision
70120ba84d68ca1211e007c4fb61d0cd5424be54. Inference uses FP32 with TF32 off,
shortest-edge 480 / longest-edge 640 resizing, 640x640 padding and pixel masks.
COCO [6] official validation is partitioned with PCG64 seed 20261008 into 1000
calibration and 4000 evaluation images. This is a project subset split, not an
official test split. The previous project already evaluated these images; this
extension is not a claim to have an entirely unseen external dataset. New runtime
parameters are chosen using calibration images and frozen before new evaluation.

Service curves use 40 post-warmup batch samples per worker/bucket and their p95.
Local batching searches max batch in 1/2/4/8 and timeout in 0/1/2 ms, with equal
candidate and trace budgets on each worker. After measuring saturated capacity,
all local candidates are rechecked once at 0.5/0.85/1.3 of each worker's observed
steady-window completion rate. The common load reference is retained after this
recheck rather than iterated. The selected settings are {local}.
Global wait caps 0/2/5/10 ms are evaluated with equal budgets for both estimators;
the common selected cap is {cfg['global_wait_ms']} ms. A separate backlogged WRR
calibration establishes lambda_ref={cfg['lambda_ref_rps']:.3f} requests/s from
actual completions during a 15-second steady window after five seconds warmup.
Common offered rates are {', '.join(f'{r:.3f}' for r in cfg['rates'])} requests/s,
or 0.2/0.5/0.85/1.3 times that reference. These labels do not imply equal utilization
for all methods or both single-worker controls.

Each policy runs Poisson and one-second burst arrivals with paired seeds
17/42/2026 at each load: 120 two-host cases. Bursts place Poisson arrivals in the
first 200 ms at five times the mean rate. Single-worker controls contribute 48
additional cases. Every case has {cfg['warmup_s']} seconds warmup,
{cfg['measurement_s']} seconds of measurement arrivals and up to {cfg['drain_s']}
seconds drain. SLO classes are {', '.join(f'{d * 1000:.2f}' for d in cfg['deadlines_s'])}
ms with probabilities 50/30/20 percent, independent of image content. They are
calibration-normalized research SLOs. Policy order is deterministically randomized.

E2E is matching detections decoded at the driver minus scheduled arrival on the
same monotonic clock. It includes generator lag, scheduler wait, JPEG transmission,
CPU queues/preprocessing, packing/copies, forward, postprocessing and return data.
Worker durations use a separate clock and are never subtracted from driver times.
Goodput is on-time completions from the measurement-arrival cohort divided by
measurement seconds. SLO success uses every offered cohort request, including
rejects, errors, late completions and unfinished drain. Completed percentiles are
reported with completion ratio. A common 512-request pending cap rejects newest
arrivals. No policy selectively drops expired images. All valid runs pass the
predeclared generator-lag threshold. Formal timing runs have no profiler attached.

### 5. Results
The following values are arithmetic means over 24 common trace configurations
per policy. They summarize this grid, not a universal workload distribution.
The strongest routing baseline is identified descriptively from the same table;
all policies and paired per-seed results remain published.

{table_markdown(['Policy', 'Mean goodput (r/s)', 'Mean SLO success (%)', 'Mean batch occupancy'], rows)}

![Goodput; bands show the min/max of three trace seeds, not confidence intervals.](figures/goodput.png)

![Completed-request p99; completion ratios remain in summary.csv.](figures/p99.png)

![Actual batch occupancy from unique completed batches.](figures/batching.png)

The E2E global variant {result} the strongest routing baseline by this aggregate
metric ({change:+.1f}%). The global estimator ablation changes only its service
table while retaining algorithm, waiting cap, executor, traffic and workers.
Differences in observed occupancy and queue stages help explain what each policy
actually executed. They do not identify every possible causal mechanism, and a
negative result is retained without retuning on the evaluation set. Per-load
curves matter because an across-grid mean can hide different operating regimes.

For RQ2, GLOBAL_E2E minus GLOBAL_GPU averages
{estimator_difference.mean():+.3f} requests/s across the 24 paired cases.
GLOBAL_E2E wins {estimator_wins}, loses {estimator_losses}, and ties {estimator_ties}
cases (absolute goodput difference within 1e-9 counts as a tie). This is a
descriptive service-table ablation; the three trace seeds do not establish
statistical significance or a generally superior estimator.

The next table reports means of per-run fractions. The first column of fractions
uses unique completed batches; the second uses completed requests. Neither is an
all-offered success measure, and both should be read alongside completion ratios.

{table_markdown(['Policy', 'Single-request batches (%)', 'Already expired at driver dispatch (%)'], dispatch_rows)}

The implementation falls back to batch one when its EDF head has no predicted
feasible placement. Expired heads can therefore limit batching, while the global
two-RPC cap also bounds how many images can enter CPU preprocessing. The observed
fractions are diagnostics, not a controlled ablation of either mechanism.
Testing alternative admission, expiration and fallback rules requires a new
calibration/freeze and is left to follow-up work.

![Adding a worker: identical offered rates, compared with each real single host.](figures/scaleout.png)

The scale-out figure compares RR on two hosts with the same local batching on
each single host. It separates the question of adding capacity from selecting a
new global policy. This small cluster cannot establish large-fleet scalability.

For RQ3, this table averages the six paired traces (two arrival types and three
seeds) at each offered rate. The final column compares two-host RR with the
better single host at that rate. The reference-rate labels are common loads,
not equal per-topology utilization.

{table_markdown(['Offered r/s', 'Two-host RR goodput', 'H800 goodput', 'RTX 5090 goodput', 'Change vs best single (%)'], scale_rows)}

The following stage values are means of per-run medians. CPU and driver statistics
sample requests; GPU statistics deduplicate batches. RPC elapsed and GPU spans
overlap other stages, so their sum is not an E2E reconstruction.

{table_markdown(['Policy', 'Driver queue ms', 'CPU queue ms', 'Preprocess ms', 'GPU forward ms/batch', 'RPC elapsed ms'], stage_rows)}

### 6. Vision quality and numerical correctness
Worker 0 is the H800 and worker 1 is the RTX 5090.

Before evaluation, each worker passes at least 1024 calibration request comparisons
between same-worker eager and Graph logits/boxes at atol 1e-5 and rtol 1e-4.
Tests cover all buckets, partial occupancy, reordered image identities and two
simultaneously outstanding independent graph slots. Request IDs and image hashes
are checked throughout real RPC replay. Ten independent offline quality runs
evaluate all 4000 images: eager b1 and Graph b1/b2/b4/b8 on each worker. AP includes
all images regardless of online completion or deadline success. No NMS or
visualization threshold changes are introduced.

{table_markdown(['Worker', 'Backend', 'Bucket', 'AP (%)', 'Delta vs same-worker eager (pp)'], qrows)}

Maximum absolute within-worker AP change is {worst_ap:.6f} pp, against the fixed
0.1 pp acceptance threshold. Different hardware may produce small floating-point
differences; this table verifies deployment semantics rather than claiming a new
detector or higher recognition accuracy.
The AP spread across all workers/backends/buckets is {ap_spread:.6f} pp.

{compatibility_note}

### 7. Reproducibility, resources and limitations
The release includes source provenance, exact package versions, source/config
freeze, 120 per-run two-host metrics, 48 controls, quality tables, request-log hashes
and a CPU-only validator. Before export, every request log is independently
recomputed for offered counts, on-time completions, quantiles and batch occupancy.
The raw JPEGs, weights, full request logs, private endpoints and credentials stay
outside Git. Published hashes expose changes but cannot independently establish
unpublished measurements; reproduction regenerates the raw evidence. Resume
rejects mismatched freezes and retains failed attempts rather than overwriting.

New aggregate GPU reservation cost at export is
{resources['new_aggregate_gpu_hours']:.4f} hours, including idle worker lifetimes,
capture and failed starts. The prior CVPR project used 12.473090 hours; its results
are not mixed with this experiment. The automated extension cap is 16 hours.
Resources are measured reservation time, not utilization-equivalent GPU-hours.

Limitations are a single fixed model/input shape, two physical workers, synthetic
arrival processes, calibration-normalized deadlines, a shared client/router process,
the asymmetric loopback versus SSH-forwarded transport, and three trace seeds.
Host CPUs are shared laboratory resources, rather than exclusively reserved
cores. Randomized policy order and seed ranges do not eliminate CPU interference.
Runs that exceed the predeclared producer-lag bound are preserved outside the
valid matrix and repeated with the same frozen configuration; selection does not
use latency or goodput. The published attempts record identifies such exclusions.
The policies do not implement router high availability or retry-based worker
recovery. Clock synchronization is not needed for E2E, but absolute cross-host
stage ordering is not inferred. No performance claims are made about production
traffic, large clusters, unmeasured Triton/TensorRT engines or complete prior systems.
Three seeds support paired descriptive comparisons, not high-confidence significance
claims. A runtime-source and configuration freeze provides a clear boundary for follow-up work.

### References
[1] Haichen Shen et al. Nexus: A GPU Cluster Engine for Accelerating DNN-Based
Video Analysis. SOSP 2019. https://yuchenjin.github.io/papers/sosp19-Nexus.pdf

[2] Arpan Gujarati et al. Serving DNNs like Clockwork: Performance Predictability
from the Bottom Up. OSDI 2020.
https://www.usenix.org/conference/osdi20/presentation/gujarati

[3] Lequn Chen et al. Symphony: Optimized DNN Model Serving using Deferred Batch
Scheduling. arXiv:2308.07470, revised 2024. https://arxiv.org/abs/2308.07470

[4] Original CVPR course implementation and measured v2 report.
https://github.com/crycryout/cvprcourseproject

[5] Nicolas Carion et al. End-to-End Object Detection with Transformers.
ECCV 2020. https://arxiv.org/abs/2005.12872

[6] Tsung-Yi Lin et al. Microsoft COCO: Common Objects in Context.
ECCV 2014. https://arxiv.org/abs/1405.0312

AI assistance: source, tests, orchestration and report generation were assisted
by Codex. The author reviews the work and handles course submission.
"""
    (out / "report.md").write_text(s)
    write_json(out / "headline.json", {"strongest_routing_baseline": baseline,
        "mean_goodput_rps": means.to_dict(), "GLOBAL_E2E_change_percent": change,
        "maximum_abs_AP_delta_pp": worst_ap, "cross_worker_AP_spread_pp": ap_spread,
        "two_host_runs": len(df), "controls": len(controls)})
    return s


def make_pdf(out, text):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
    import re
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="CaptionSmall", fontSize=8, leading=10, alignment=TA_CENTER, spaceAfter=9))
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=(595, 842), leftMargin=45, rightMargin=45,
                            topMargin=42, bottomMargin=42, title="Where Should We Batch?", author="CISC8003 Research Project")
    story, paragraph = [], []
    def clean(s):
        s = html.escape(s)
        return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", s)
    def flush():
        if paragraph:
            story.append(Paragraph(clean(" ".join(paragraph)), styles["BodySmall"]))
            paragraph.clear()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= {"-", ":"} for c in cells):
                    rows.append([Paragraph(clean(c), styles["CaptionSmall"]) for c in cells])
                i += 1
            table = Table(rows, colWidths=[505 / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign="CENTER")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("LINEBELOW", (0, 0), (-1, 0), .5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story.extend([table, Spacer(1, 8)])
            continue
        if line.startswith("!["):
            flush()
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            image = Image(str(out / m.group(2)))
            ratio = image.imageHeight / image.imageWidth
            image.drawWidth, image.drawHeight = 505, 505 * ratio
            story.append(KeepTogether([image, Paragraph(clean(m.group(1)), styles["CaptionSmall"])]))
        elif line.startswith("#"):
            flush()
            n = len(line) - len(line.lstrip("#"))
            style = styles["Title"] if n == 1 else styles["Heading2"] if n == 2 else styles["Heading3"]
            story.append(Paragraph(clean(line.lstrip("# ")), style))
        elif not line:
            flush()
        else:
            paragraph.append(line)
        i += 1
    flush()
    def page(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(45, 23, "CISC8003 | Distributed DETR serving | Protocol 3")
        canvas.drawRightString(550, 23, str(document.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=page, onLaterPages=page)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="artifacts/distributed_v3")
    p.add_argument("--output", default="results")
    args = p.parse_args()
    inp, out = Path(args.input), Path(args.output)
    cfg, df, controls, stages = export(inp, out)
    figures(out, cfg, df, controls, stages)
    text = report_text(out, cfg, df, controls, stages)
    make_pdf(out, text)
    hashes = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "checksums.json"}
    write_json(out / "checksums.json", hashes)
    print(json.dumps(read(out / "headline.json")), flush=True)


if __name__ == "__main__":
    main()
