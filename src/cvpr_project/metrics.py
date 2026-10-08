"""Measurement-cohort accounting including all SLO failures and drain."""
import numpy as np


def censor_completions_at_cap(rows, cap_ns):
    """A result materialized just after the drain cap remains unfinished at cap."""
    for row in rows:
        if row["status"] == "completed" and row["complete_ns"] > cap_ns:
            row["status"] = "unfinished"
            row["completion_observed_after_drain_cap"] = True


def summarize(rows, measurement_s, observed_s, min_deadline_s):
    cohort = [r for r in rows if r["measurement"]]
    completed = [r for r in cohort if r["status"] == "completed"]
    ontime = [r for r in completed if r["complete_ns"] <= r["deadline_ns"] and r.get("identity_ok", False)]
    latency = [(r["complete_ns"] - r["arrival_ns"]) / 1e6 for r in completed]
    lag = [(r["enqueue_ns"] - r["arrival_ns"]) / 1e6 for r in cohort if r.get("enqueue_ns") is not None]
    percentile = lambda xs, q: float(np.percentile(xs, q)) if xs else None
    statuses = {s: sum(r["status"] == s for r in cohort) for s in ["completed", "rejected", "failed", "unfinished"]}
    total = len(cohort)
    return {"protocol_version": 2, "offered": total, **statuses, "ontime": len(ontime),
            "late_completed": len(completed) - len(ontime),
            "completion_ratio": len(completed) / total if total else 0,
            "slo_success": len(ontime) / total if total else 0,
            "goodput_rps": len(ontime) / measurement_s,
            "cohort_throughput_rps_including_drain": len(completed) / observed_s,
            "latency_p50_ms": percentile(latency, 50), "latency_p95_ms": percentile(latency, 95),
            "latency_p99_ms": percentile(latency, 99), "generator_lag_p99_ms": percentile(lag, 99),
            "loadgen_limited": bool(lag and percentile(lag, 99) > max(1, 0.05 * min_deadline_s * 1000)),
            "measurement_seconds": measurement_s, "observed_seconds_including_drain": observed_s}
