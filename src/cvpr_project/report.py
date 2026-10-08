"""English paper generated from verified evidence; eight explicit paper sections."""
from pathlib import Path
import html
import re
import pandas as pd
from .runs import read_json


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def write_report(results="results", destination="docs/REPORT.md"):
    root = Path(results)
    delivery = read_json(root / "delivery.json")
    frozen = read_json(root / "frozen_v2.json")
    cal = frozen["calibration"]
    quality = pd.read_csv(root / "quality.csv")
    runs = pd.read_csv(root / "summary.csv")
    precision = read_json(root / "precision_v2.json")
    stress = read_json(root / "stress_v2.json")
    pool = read_json(root / "graph_pool_v2.json")
    pool_table = table(["Initialization/storage item", "Measured value", "Scope"], [
        ["Capture plus capture warmup", f"{pool['capture']['initialization_seconds']:.3f} s", "All 8 graph slots; excludes model loading"],
        ["Pinned host slot storage", f"{pool['pinned_host_bytes']/2**20:.2f} MiB", "Graph executor only"],
        ["GPU allocated", f"{pool['gpu_allocated_bytes']/2**30:.3f} GiB", "Calibration process; graph and eager resident"],
        ["GPU peak allocated", f"{pool['gpu_peak_allocated_bytes']/2**30:.3f} GiB", "Calibration process, including earlier serial work"],
        ["GPU reserved", f"{pool['gpu_reserved_bytes']/2**30:.3f} GiB", "Calibration process caching allocator"],
    ])
    pilots = read_json(root / "pilot_summary.json")["rows"]
    overlap = read_json(root / "profile_overlap.json")["profiles"]
    hardware_path = root / "environment_timing_v2.json"
    hardware = read_json(hardware_path if hardware_path.exists() else root / "environment_v2.json")["cuda_device"]
    environment = read_json(root / "preflight_delivery_v2.json")
    host = environment["host"]
    host_text = (f"The host has {'; '.join(host['cpu_models'])}, {host['physical_cores']} physical cores, "
                 f"{host['logical_processors']} logical processors and {host['ram_total_bytes']/1e9:.2f} GB RAM. "
                 f"The NVIDIA driver is {', '.join(environment['gpu_driver_versions'])}. "
                 "This environment probe records specifications, not a benchmark.")
    grouped = runs.groupby(["policy", "arrival_type", "rate_multiplier"])[
        ["goodput_rps", "slo_success", "latency_p95_ms", "latency_p99_ms", "completion_ratio"]].mean()
    comparisons = pd.read_csv(root / "paired_comparisons.csv")
    fpaired = comparisons[comparisons.baseline == "F0"]
    mean_diff = fpaired.goodput_difference_rps.mean()
    cpaired = comparisons[comparisons.baseline == "C0"]
    compile_difference = (f"Mean paired D0 minus C0 goodput is {cpaired.goodput_difference_rps.mean():+.3f} requests/s. "
                          "This valid optimized backend remains in the comparison regardless of which method wins."
                          if len(cpaired) else "No valid measured C0 backend is available.")
    ap_ref = quality[(quality.backend == "eager") & (quality.bucket == 1)].iloc[0].AP
    max_ap_drift = quality.AP_change_pp.abs().max()
    metered = delivery["resource_totals"]["metered_gpu_hours"]
    reserve = delivery["resource_totals"]["initial_unmetered_setup_reserve_hours"]
    qtable = table(["Backend", "Bucket", "AP", "AP50", "AP75", "AP small", "AP medium", "AP large"],
                   [[r.backend, int(r.bucket), *[f"{getattr(r, k):.3f}" for k in
                     ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]]] for r in quality.itertuples()])
    stable = table(["Policy", "Max batch", "Wait (ms)"], [[p, s["max_batch"], s["wait_ms"]] for p, s in cal["policies"].items()])
    stage = table(["Policy", "Preprocess", "Pack", "H2D", "Forward", "D2H", "Postprocess", "E2E p95"],
                  [[p["policy"], *[f"{p['stage_median_ms'][k]:.3f}" for k in
                    ["preprocess_ms", "pack_ms", "h2d_ms", "forward_ms", "d2h_ms", "postprocess_ms"]],
                    f"{p['e2e_p95_ms']:.3f}"] for p in pilots])
    cpu_table = table(["Policy", "Mean CPU cores", "Max allocated GiB", "Mean dummy fraction"],
                     [[p, f"{g.main_process_cpu_core_equivalents.mean():.2f}",
                       f"{g.gpu_peak_allocated_gib.max():.2f}", f"{g.dummy_lane_fraction.mean():.3f}"]
                      for p, g in runs.groupby("policy")])
    service = table(["Bucket", "Dispatch-to-CPU-result p95 (ms)"],
                    [[b, f"{int(ns)/1e6:.3f}"] for b, ns in cal["service_ns"].items()])
    high = runs[(runs.rate_multiplier == 1.1) & (runs.arrival_type != "periodic") &
                runs.policy.isin(["F0", "A0", "D0", "C0"])]
    high_table = table(["Policy", "Arrival", "Load", "Goodput (r/s)", "SLO success", "p95 (ms)", "Completion"],
                      [[p, k, f"{rho:.1f}", f"{g.goodput_rps:.3f}", f"{g.slo_success:.3f}",
                        f"{g.latency_p95_ms:.3f}", f"{g.completion_ratio:.3f}"] for (p, k, rho), g in
                       high.groupby(["policy", "arrival_type", "rate_multiplier"]).mean(numeric_only=True).iterrows()])
    compile_record = frozen["compile"]
    compile_text = ("The Inductor baseline passed calibration correctness and was included in quality and serving comparisons. "
                    "It used default compilation options with automatic CUDA Graphs explicitly disabled; initialization and compilation costs are recorded. Compiler fusion did not meet every initial raw FP32 tolerance. Acceptance additionally required absolute logit error at most 0.01 and a normalized box bound implying at most 0.5-pixel corner displacement, frozen before held-out quality inspection; all initial failures and final raw errors are retained."
                    if compile_record["status"] == "passed" else
                    f"The bounded Inductor attempt ended with status `{compile_record['status']}`. "
                    f"The recorded failure was {compile_record.get('error_type', 'unknown')}: "
                    f"{compile_record.get('error_message', 'see compile_v2.json')[:500]}. "
                    "This limits comparisons to the valid measured eager/graph backends; no TensorRT or Triton performance claim is made.")
    sections = []
    sections.append(("Abstract and research question", f"""We study pretrained DETR object detection serving on one {hardware['name']}. The workload starts from cached compressed COCO image bytes and ends when CPU detection dictionaries are available. We keep model weights, masks, resolution, precision, and decoding common while varying graph execution, buffering, pipeline overlap, and batching. The evaluation contains {delivery['completed_cases']} completed real-image serving cases with synthetic open-loop arrivals, plus independent detection quality on all 4,000 held-out validation images. Same-precision eager batch-one AP is {ap_ref:.3f}; the largest observed backend/bucket AP change is {max_ap_drift:.3f} percentage points. Across paired main settings, D0 minus tuned F0 mean goodput is {mean_diff:+.3f} requests/s. This aggregate is descriptive, not a significance test or a guarantee across loads.

Interactive detection combines CPU image work, host submission, device execution, copying, and queueing. A shorter GPU forward alone does not establish shorter request latency. Likewise, reporting only completed-request latency can hide overload rejection. We ask three questions: which stages constrain this detector on this hardware; what graph replay and explicit buffer ownership change; and whether a measured-service-time batching heuristic improves deadline satisfaction over a tuned fixed timer.

The contribution is an implementation and controlled evaluation for a computer vision course. CUDA Graphs, pinned memory, asynchronous streams, EDF, and deadline feasibility already exist. We do not claim a new detection architecture or a new general serving algorithm. The visual boundary is concrete: real RGB images enter, COCO categories and original-coordinate boxes leave, and the quality constraint is tested independently of which online requests finish on time.

There is no model training, camera ingestion, HTTP/RPC service, tracking, or production traffic. Rates and deadlines are normalized research choices. The primary comparison is against the strongest valid backend represented in the evidence, with F0/A0/D0 isolating execution from queue ordering and service-time feasibility."""))
    sections.append(("Background and related work", """DETR uses a convolutional backbone and transformer encoder-decoder with a fixed set of object queries and classification/box heads. We use the authors' existing COCO checkpoint rather than train a model. Our 480/640 resize and 640-square padding are deployment settings, so the resulting subset AP is not an attempted reproduction of the original paper's higher-resolution full-validation result. [Carion et al., ECCV 2020](https://arxiv.org/abs/2005.12872); [author implementation](https://github.com/facebookresearch/detr).

Clockwork explicitly controls execution and uses model/batch execution profiles and executor availability to predict feasible completion. It maintains batch queues and schedules through latest-start strategies; deadline feasibility and efficient batch selection are therefore established ideas. Our one-model local harness uses a conservative dispatch-to-result percentile and never rejects expired requests selectively. We do not reproduce Clockwork's distributed multi-model controller. [Gujarati et al., OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/gujarati).

Clipper separates framework execution from serving and adapts batch size to a latency target through additive increase and multiplicative decrease. It also supports prediction caching and model composition. Here we disable result caching, fix one detector, and give FIFO timers a finite calibration search. [Crankshaw et al., NSDI 2017](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw).

Triton documents dynamic batching, preferred batch sizes, and maximum queue delay. Our F0 is a local explicitly implemented timer baseline; it is not a full Triton deployment. [NVIDIA dynamic batching documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html).

PyTorch's CUDA semantics require stream ordering and fixed addresses for graph replay. Pinned storage permits asynchronous copies but does not itself prove overlap. We warm up before capture, retain tensor ownership until the final copy event and CPU consumption, and inspect a separate correlated profile. [PyTorch 2.14 CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html). The report gives measured scope and negative results rather than extrapolating to another framework, detector, GPU, or Grace/NVLink-C2C platform."""))
    sections.append(("Execution and scheduling design", f"""A request identifies an image and carries a scheduled arrival and absolute deadline. A separate load-generation process enqueues arrivals without waiting for service; CPU workers decode JPEG bytes and apply the common processor. Ready requests feed a scheduler. Packing fills a fixed host slot, H2D copies populate its device inputs, one compute stream executes eager or captured forward, and D2H copies populate its host logits/boxes. CPU decoding materializes detection dictionaries before releasing the slot.

Every bucket has two independent slots and private graph pools. Events enforce H2D before forward and forward before D2H. Initialization on the caller stream is ordered before side-stream access. At most one executing batch and one locked next batch exist globally. A newly arriving earlier deadline cannot overwrite either submitted batch. Dummy lanes repeat a valid image/mask and never produce real request results. The optimized serving loop queries events without a per-stage device-wide synchronization.

D0 orders ready requests by deadline, arrival, and ID. For each EDF prefix of length n it selects the smallest common bucket b that fits and predicts F(n)=now+G+R_b. G retains the full past service estimate until the GPU forward-start event is observed; only then is host elapsed time subtracted. Queued time therefore cannot exhaust an unstarted batch's estimate. No future completion is inspected. R_b is a calibration dispatch-to-CPU-result p95 including pack, copies, forward, and postprocess. Feasible candidates satisfy the earliest selected deadline; D0 maximizes n/R_b, then prefers an earlier finish and smaller bucket. With no feasible prefix it serves the earliest deadline immediately rather than dropping a difficult request.

Waiting is bounded by the oldest ready timestamp plus frozen tau. The implementation waits for a larger bucket only when its conservative predicted completion still meets the earliest deadline. Arrivals re-trigger the decision without resetting the absolute timer. This is a heuristic: model error, CPU variability, and locked prefetch can still cause misses.

{service}

E0 uses pageable synchronous serial copying; E1 uses pinned buffers with a serial pipeline; G0 adds graph replay; P0 adds two-slot pipelining. R0 and F0 use tuned FIFO timers with eager and graph execution respectively. A0 shares F0's timer and executor but changes the ready order to EDF. D0 changes batch feasibility selection. Executor pools remain resident across randomized cases; capture time and scoped storage evidence are reported separately."""))
    sections.append(("Vision correctness and quality", f"""The immutable model revision is `{frozen['model']['revision']}`. Model configuration, processor configuration, and weights have SHA-256 identities. COCO image IDs are numerically sorted and permuted by PCG64 seed 20261008; the first 1,000 form calibration and the remaining 4,000 form held-out evaluation. These are project subsets of official validation, not an official test split. The split and annotation bytes are hashed.

The official processor resizes with shortest edge 480 and longest edge 640 while preserving aspect ratio. It normalizes and pads on the right and bottom to 640x640. The corresponding pixel mask distinguishes real pixels from padding. Tests compare valid pixels and padding against the unpadded official processor path. Box decoding uses the official postprocessor and original image (height,width), including an analytically known box test. COCO's non-contiguous category IDs are checked against checkpoint labels rather than shifted by one.

Calibration eager FP32 AP is {precision['fp32_AP']:.3f}; BF16 AP is {precision['bf16_AP']:.3f}, a signed drop of {precision['bf16_drop_pp']:.3f} points. The initial 0.3-point drop rule is followed by a pre-freeze bucket stability check; the resulting choice is {cal['precision'].upper()} for every policy. The recorded fallback reason is {precision.get('prefreeze_fallback_reason') or 'no fallback needed'}. This decision uses calibration only. TF32 is disabled and attention implementation is common eager attention. Graph consistency compares {stress['requests_compared']} real-image requests drawn repeatedly from 32 calibration images, covering all common buckets, partial occupancy, alternating IDs, reversed independent slots, and delayed CPU consumption. Maximum same-precision logits/box errors are {stress['max_abs_logits_error']:.6g}/{stress['max_abs_boxes_error']:.6g}; raw tolerances are retained in the stress record.

Offline COCOeval uses every held-out image, threshold zero, up to 100 queries, and no added NMS. Runtime optimization must remain within 0.1 AP point of same-precision eager batch one. Scheduler-only policies can share executor AP because request identity and executor correctness are separately validated.

{qtable}

AP values above are percentages, with COCO IoU 0.50:0.95 for AP. Visualization uses threshold 0.7 only; the illustrated detections and attribution metadata are in `results/detections`. Online misses/rejections never select the images used in this table.

![Actual eager FP32 DETR detections on calibration images; sources and licenses are recorded in detections/attribution.json and detections/README.md.](../results/figures/detection_examples.png)"""))
    sections.append(("Experimental method and calibration", f"""The timing device is {hardware['name']} with {hardware['multiprocessors']} multiprocessors and {hardware['total_memory_bytes']/1e9:.2f} GB visible device memory. The isolated environment uses PyTorch {hardware['torch']}, CUDA build {hardware['cuda_build']}, and pinned Transformers 4.46.3. Initial compatibility smoke used the other card's MIG 2g.20gb partition; it is excluded from full-device timing claims. Formal timing checks other GPU activity instead of terminating unrelated work or changing clocks, drivers, or MIG configuration.

{host_text}

The main input scope is RAM-cached compressed image bytes to CPU detection dictionaries. Every request decodes and transforms again; model outputs and preprocessed inputs are not cached for serving. The worker-pool limit is chosen using calibration images and then fixed at {cal['cpu_workers']}. Serial E0/E1/G0 controls permit only one request at a time, including CPU preprocessing; concurrent policies can use the full worker pool. Thus G0/P0 compares the combined CPU-GPU pipeline, not GPU-copy overlap alone. The SLO reference is E1 serial E2E p95, S_base={cal['s_base_s']*1000:.3f} ms. Deadline classes are {2*cal['s_base_s']*1000:.3f}, {4*cal['s_base_s']*1000:.3f}, and {8*cal['s_base_s']*1000:.3f} ms with proportions 50/30/20 percent, independent of image content.

FIFO candidates search max_batch in the common buckets and wait in 0/1/2 ms. R0 and F0 share calibration traces and search budget, first at 0.5/1/2/4 times E1 serial throughput and then once at the final common rates. D0 searches tau in 0/1/2 ms. The selected initial F0 backlog run measures common lambda_ref={cal['lambda_ref_rps']:.3f} requests/s over 90 seconds after warmup; the reference is not redefined for each method. Freeze precedes held-out inspection.

{stable}

Periodic, Poisson, and one-second bursts are synthetic arrivals over real images; burst activity is concentrated in 200 ms at five times the average rate. Main seeds are 17/42/2026, and methods share identical trace hashes. Rates are 0.3/0.6/0.9/1.1 times lambda_ref. A run warms for 10 seconds, measures arrivals over 60 seconds, and drains for at most 30 seconds. All offered measurement arrivals remain in the denominator, including newest-overflow rejection, failure, late completion, and unfinished drain. The pending cap is 512. Generator lag p99 must not exceed max(1 ms, five percent of the smallest deadline).

Latency is CPU_result_ready minus scheduled arrival. We report completed percentiles with completion ratio; goodput is on-time measurement-cohort results divided by 60 seconds. Cohort throughput includes drain observation time, while steady-window completion rate is separate. Three seeds are run-level repeats; individual requests are not independent replicates. Shading shows the seed min/max, not a confidence interval."""))
    sections.append(("Serving results and controls", f"""The public per-seed table `results/summary.csv` is derived from {delivery['completed_cases']} completed manifests. The protocol requires 252 main cases and 12 A0 controls; a valid compile backend adds 36 cases. Failed attempts remain indexed in `delivery.json` and are charged to resource accounting. The following table gives seed means at load 1.1 for Poisson and burst controls. Completed-request p95 must be read together with completion and SLO success.

{high_table}

Across paired F0/D0 settings, mean D0 goodput minus F0 is {mean_diff:+.3f} requests/s. Per-trace paired differences are retained in `paired_comparisons.csv`; that single aggregate does not imply universal superiority. A0 changes queue order while retaining F0's timer, so F0/A0 isolates EDF and A0/D0 isolates service-time feasibility and bucket selection. Small differences below five percent are descriptive unless additional predeclared repeats affect a conclusion. We do not infer statistical significance from three seeds.

{compile_text}

{compile_difference}

![Completed-request p95 latency](../results/figures/latency_p95.png)

![SLO goodput](../results/figures/goodput.png)

![EDF and feasibility controls; bars are seed means and whiskers are seed ranges.](../results/figures/deadline_ablation.png)

The latency axes use a logarithmic scale so overload tails and shorter latencies remain visible together. Method colors are consistent across arrival types. The p99, SLO-success, completion-ratio curves and execution ablation are exported alongside these figures. Raw request stage records stay local with hashes in small public manifests. Formal timing excludes profiler overhead. A detector can retain offline AP while missing online deadlines; these are different constraints and both are reported."""))
    profile_rows = table(["Profile", "Memcpy events", "Copy time overlapping kernels", "Overlap observed"],
                         [[p["profile"], p.get("memcpy_events", 0),
                           f"{100*p.get('fraction_copy_time_overlapping_kernel',0):.2f}%", p.get("overlap_verified", False)] for p in overlap])
    sections.append(("Stage evidence, costs, and interpretation", f"""Serial pilots process real calibration images, preserve every measured row, and exclude the first twenty cold requests only when constructing the SLO reference. The following medians are milliseconds; E2E p95 is measured independently. Host and CUDA clocks differ, and overlapping stage sums are not an E2E reconstruction.

{stage}

![E0, E1, G0 and P0 execution controls with shared traces; shaded regions show seed ranges.](../results/figures/execution_ablation.png)

`submit_host_ms` includes Python packing/submission and can overlap GPU execution. A long host submission is not, by itself, measured idle launch gap. H2D/D2H event durations are device measurements. CPU JPEG/transform and postprocessing remain part of the serving scope. The profiler shows kernel and memcpy intervals in its correlated clock; only actual interval intersection establishes copy/compute overlap.

{profile_rows}

Separate E1, P0 and valid C0 profiles replay calibration images at 0.3 and 1.1 times the frozen common capacity, using each policy's frozen settings. Their profiled metrics include instrumentation overhead and never replace formal timing. Profiles ending in `_low` denote 0.3 load; the others denote 1.1 load. GPU-resident forward microbenchmarks use 50 timed samples after 10 warmups per bucket and exclude JPEG processing, transfer, queueing, and CPU postprocessing; the measured values are exported in `forward_microbench.csv`.

![Separate P0 correlated CUDA timeline; blue denotes kernels and orange denotes memory copies.](../results/figures/profile_P0_timeline.png)

Graph initialization includes capture warmup, every slot's capture, and private pools. The calibration snapshot below retains graph and eager executors simultaneously; GPU allocator counters are process-wide, while pinned bytes sum this graph executor's slots. They are not an incremental graph memory measurement. The peak also retains earlier serial work in that calibration process.

{pool_table}

Per-run memory includes all resident executor pools because the matrix reuses one process across randomized policies. These numbers must not be interpreted as an isolated E0-versus-F0 memory comparison. Main-process CPU cores are CPU seconds divided by wall seconds over warmup, measurement, drain and request export; the separate load generator is excluded. Completed-batch occupancies are in `batch_distribution.csv`.

{cpu_table}

The execution ledger records {metered:.3f} metered GPU reservation hours, including failed work and compilation/capture inside instrumented leases. An additional {reserve:.3f}-hour conservative reserve covers initial unmetered smoke calls; it is a budget bound, not a fabricated measurement. Dataset archives, model weights, and full traces remain outside public Git. Project storage is checked against 30 GB, and automated GPU cost is capped at 30 hours.

Potential losses are meaningful results: graph replay may reduce submission without improving the CPU-limited end-to-end path; batching may increase throughput while delaying small queues; EDF may redistribute lateness without helping total goodput; a conservative service table may sacrifice batch efficiency. The curves and stage evidence support only the observed workload and frozen settings."""))
    sections.append(("Limitations, reproducibility, and conclusion", f"""This is one pretrained detector, one resolution contract, one precision selected on calibration, and one H800 timing device. Fixed padding, synthetic arrival processes, image replay, and a local non-network harness limit external validity. The image pool repeats in deterministic trace order; each repetition still executes the full detector. Uniform deadline mixtures are a research tool rather than an observed application's requirements. We evaluate a held-out part of COCO validation and make no claim about an official test submission.

The service predictor is a p95 heuristic rather than a scheduling guarantee. CPU variability and irrevocable prefetched work can invalidate predicted feasibility. The result set contains three independent traffic seeds per main scenario and does not establish high-confidence significance for a small effect. A separate profiler perturbs execution and is used only for mechanism evidence. No throughput or latency claim is made for a production server, a full Triton stack, an unmeasured TensorRT backend, or GH200/NVLink-C2C.

Reproduction starts with `configs/requirements.lock.txt` in a new environment, one available full H800, official COCO archives, and the immutable model revision. Run `scripts/reproduce.sh` after activating that environment. Every CLI implements its help interface. The runner reconstructs graphs in a fresh process because a CUDA Graph object is not a portable checkpoint; it resumes only verified completed cases and preserves failed manifests. A conflicting freeze is rejected instead of overwritten.

The frozen identity is `{frozen['frozen_config_sha256']}`. Model files, annotation bytes, image-byte set, split IDs, trace contents, source version/diff, and request CSVs have hashes. Public tables link to run IDs; local evidence includes stage records and full traces. Existing evidence is not regenerated by selecting the fastest rerun. Dataset bytes, weights, credentials, private hostnames, and personal paths are excluded from the public deliverable.

The project establishes a reproducible connection between detection correctness and serving performance. Same-precision eager batch-one AP is {ap_ref:.3f}; maximum observed backend/bucket drift is {max_ap_drift:.3f} points. The paired F0/D0 mean goodput difference is {mean_diff:+.3f} requests/s across the measured main settings, with individual scenarios shown in the figures. Improvements and regressions are both retained. These findings support the implementation's measured scope and do not constitute a novelty claim for its underlying mechanisms.

Code implementation, debugging assistance, and report assembly used Codex. The user must review numerical claims and comply with the actual course's AI policy; no approval from an instructor is asserted and no coursework is submitted automatically."""))
    title = "Deadline-Aware Object Detection Serving on H800 with CUDA Graphs and CPU-GPU Pipelining"
    markdown = "# " + title + "\n\nCISC8005 course project — measured protocol v2\n\n" + "\n\n".join(
        f"## {i}. {heading}\n\n{body}" for i, (heading, body) in enumerate(sections, 1)) + "\n"
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(markdown)
    _pdf(sections, title, root / "report.pdf", root)
    return destination


def _pdf(sections, title, destination, root):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperBody", fontName="Helvetica", fontSize=9, leading=12,
                             spaceAfter=7, splitLongWords=True))
    styles.add(ParagraphStyle(name="PaperTitle", fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=12))
    styles.add(ParagraphStyle(name="PaperCaption", fontName="Helvetica-Oblique", fontSize=8, leading=10, spaceAfter=7))
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=42, leftMargin=42,
                           topMargin=42, bottomMargin=42, title=title, author="CISC8005 project")
    story = []
    def rich(text):
        text = html.escape(text)
        text = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#225588">\1</link>', text)
        text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
        return text
    for i, (heading, body) in enumerate(sections):
        if i:
            story.append(PageBreak())
        if not i:
            story.append(Paragraph(title, styles["PaperTitle"]))
            story.append(Paragraph("CISC8005 — measured protocol v2", styles["Normal"]))
            story.append(Spacer(1, 12))
        story.append(Paragraph(f"{i+1}. {heading}", styles["Heading2"]))
        for block in body.split("\n\n"):
            if block.startswith("|"):
                rows = [[x.strip() for x in line.strip().strip("|").split("|")] for line in block.splitlines()
                        if not set(line.replace("|", "").strip()) <= {"-", " "}]
                if not rows:
                    continue
                content = [[Paragraph(html.escape(x), styles["PaperBody"]) for x in row] for row in rows]
                t = Table(content, colWidths=[doc.width / len(rows[0])] * len(rows[0]), repeatRows=1)
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf5")),
                                      ("GRID", (0, 0), (-1, -1), .3, colors.lightgrey),
                                      ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                      ("TOPPADDING", (0, 0), (-1, -1), 3),
                                      ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
                story.extend([t, Spacer(1, 7)])
            elif block.startswith("!["):
                m = re.match(r"!\[([^]]+)\]\(([^)]+)\)", block)
                if m:
                    file = root / "figures" / Path(m[2]).name
                    if file.exists():
                        image = Image(str(file))
                        scale = doc.width / image.imageWidth
                        image.drawWidth, image.drawHeight = doc.width, image.imageHeight * scale
                        story.extend([image, Paragraph(m[1], styles["PaperCaption"])])
            else:
                story.append(Paragraph(rich(block), styles["PaperBody"]))
    def page_number(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(A4[0] - 42, 24, str(document.page))
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
