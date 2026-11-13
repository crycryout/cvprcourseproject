#!/usr/bin/env python3
"""Generate an English presentation from the same verified measured evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap


def main():
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    root = Path(args.results)
    cfg = json.loads((root / "frozen.json").read_text())
    headline = json.loads((root / "headline.json").read_text())
    c = canvas.Canvas(str(root / "slides.pdf"), pagesize=(960, 540))
    c.setTitle("Where Should We Batch? Distributed DETR Serving")
    number = 0
    def slide(title, points=(), figure=None):
        nonlocal number
        number += 1
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.rect(0, 0, 960, 540, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 28)
        c.drawString(44, 478, title)
        c.setFillColor(colors.HexColor("#0891b2"))
        c.rect(44, 459, 872, 4, fill=1, stroke=0)
        if figure:
            from PIL import Image
            with Image.open(root / "figures" / figure) as im:
                ratio = im.height / im.width
            width = 850
            height = min(width * ratio, 330)
            c.drawImage(str(root / "figures" / figure), 55, 112, width=width, height=height, preserveAspectRatio=True, anchor="c")
            y = 84
            size = 14
        else:
            y, size = 412, 21
        for point in points:
            c.setFillColor(colors.HexColor("#1e293b"))
            c.setFont("Helvetica", size)
            lines = textwrap.wrap(point, width=105 if figure else 78)
            for k, line in enumerate(lines):
                c.drawString(52 if k == 0 else 68, y, ("- " if k == 0 else "") + line)
                y -= size * 1.4
            y -= 17 if not figure else 3
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor("#64748b"))
        c.drawString(44, 25, "CISC8003 | Real two-host inference | Protocol 3")
        c.drawRightString(916, 25, str(number))
        c.showPage()
    slide("Where Should We Batch?", ["Coordinating request routing and batching for heterogeneous object detection serving",
        "Reuse a verified DETR runtime; investigate distributed queues, communication and worker heterogeneity.",
        "Real H800 + RTX 5090 inference, paired synthetic arrivals, independent detection quality."])
    slide("Question: route first or batch first?", ["Immediate routing can reduce queue delay but split opportunities to batch requests.",
        "Global pooling can form larger batches, while its waiting and service estimates can also hurt deadlines.",
        "RQ1: global vs local batching. RQ2: GPU-only vs E2E estimates. RQ3: one worker vs two."])
    slide("Architecture and measured boundary", ["Actual JPEGs and full detections traverse persistent TCP; the remote leg uses an SSH forward.",
        "E2E uses scheduled arrival and decoded detections on one driver clock."], "architecture.png")
    slide("Five policies, one detector and executor", ["RR / WRR / SECT: individual request routing followed by tuned worker-local FIFO batching.",
        "GLOBAL_GPU / GLOBAL_E2E: global EDF queue, bounded waiting, worker/bucket selection.",
        "The global variants differ only in the service table. They never drop expired images.",
        "The original executor retains fixed addresses, event dependencies and two independent graph slots."])
    slide("Controlled calibration and evaluation", [f"Reference capacity: {cfg['lambda_ref_rps']:.2f} requests/s; shared wait cap: {cfg['global_wait_ms']} ms.",
        "Tune on 1000 calibration images; freeze source, settings, rates and deadlines before new evaluation.",
        "120 two-host cases: 5 policies x 2 arrival patterns x 4 loads x 3 paired seeds.",
        "48 matched single-worker controls; every request remains in the success denominator."])
    slide("Deadline goodput", ["Means and seed min/max. Bands are ranges, not confidence intervals."], "goodput.png")
    slide("Tail latency and completion", ["Completed p99 must be read with SLO success and completion ratios in the published CSV."], "p99.png")
    slide("What the batcher actually executed", ["Occupancy counts real requests per unique completed batch, rather than configured maximum size."], "batching.png")
    slide("Does adding a worker help?", ["Compare identical offered traces with each real single host; no claim of large-cluster scalability."], "scaleout.png")
    slide("Detection correctness", ["Each worker passed 1024 eager/Graph request comparisons, all buckets and partial batches.",
        "Concurrent independent slots were checked against same-worker eager outputs.",
        "10 offline AP runs: eager b1 plus Graph b1/b2/b4/b8 on each GPU, all 4000 images.",
        f"Largest within-worker AP change: {headline['maximum_abs_AP_delta_pp']:.6f} percentage points."])
    means = headline["mean_goodput_rps"]
    baseline = headline["strongest_routing_baseline"]
    slide("Measured conclusion and its limits", [f"GLOBAL_E2E mean goodput: {means['GLOBAL_E2E']:.3f} r/s; {baseline}: {means[baseline]:.3f} r/s ({headline['GLOBAL_E2E_change_percent']:+.1f}%).",
        "Expired-head batch-one fallback and the two-RPC CPU admission cap are plausible limits; no causal ablation isolates them.",
        "One model, two shared hosts, synthetic arrivals and asymmetric transport limit generalization.",
        "This frozen heuristic loses; Nexus, Clockwork and Symphony remain prior mechanisms, not reproduced baselines."])
    slide("Reproduce and inspect", ["Read results/report.pdf and docs/PROTOCOL.md; inspect source/config hashes and raw-log audit counts.",
        "Run python scripts/verify_results.py without GPUs or image data.",
        "For real reproduction: start task-owned workers, then bash scripts/reproduce.sh.",
        "Course submission and explanation remain with the author; implementation was assisted by Codex."])
    c.save()
    checksums = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob("*")) if p.is_file() and p.name != "checksums.json"}
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True, indent=2) + "\n")
    print(f"Generated {number} English slides: {root / 'slides.pdf'}")


if __name__ == "__main__":
    main()
