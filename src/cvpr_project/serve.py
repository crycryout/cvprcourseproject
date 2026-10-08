"""Open-loop child-process load generation, CPU workers, bounded GPU pipeline."""
from concurrent.futures import ThreadPoolExecutor
import csv
import multiprocessing as mp
from pathlib import Path
import queue
import time
import threading
import sys
import torch
from .metrics import summarize
from .scheduler import Scheduler, Ready, remaining_service_ns
from .runs import write_json, other_gpu_pids
from .trace import trace_hash


def _load_generator(trace, channel, semaphore, start_event, generator_ready, epoch):
    generator_ready.set()
    start_event.wait()
    origin = epoch.value
    for arrival in trace:
        scheduled = origin + round(arrival.arrival_s * 1e9)
        # Sleep releases the CPU; separate process protects arrival scheduling.
        while (remaining := scheduled - time.monotonic_ns()) > 0:
            if remaining > 300000:
                time.sleep((remaining - 150000) / 1e9)
        enqueue = time.monotonic_ns()
        accepted = semaphore.acquire(block=False)
        channel.put((arrival.request_id, enqueue, accepted))
    channel.put((None, time.monotonic_ns(), True))


def preprocess_request(detector, corpus, image_id):
    start = time.monotonic_ns()
    torch.cuda.nvtx.range_push("decode_resize_normalize")
    image = corpus.image(image_id)
    decoded = time.monotonic_ns()
    payload = detector.preprocess(image)
    torch.cuda.nvtx.range_pop()
    return payload, start, decoded, time.monotonic_ns()


def run_serving(executor, corpus, trace, policy, settings, service_ns, s_base_s,
                warmup_s=10, measurement_s=60, drain_s=30, workers=4, max_pending=512,
                output=None, progress=None):
    if any(s.state != "FREE" for s in executor.slots):
        raise RuntimeError("All slots must be free before replay")
    torch.cuda.reset_peak_memory_stats(executor.detector.device)
    serial = policy in {"E0", "E1", "G0"}
    batch1 = policy in {"E0", "E1", "G0", "P0"}
    scheduler = Scheduler(policy, buckets=executor.buckets,
                          max_batch=1 if batch1 else settings["max_batch"],
                          wait_ms=0 if batch1 else settings["wait_ms"], service_ns=service_ns)
    ctx = mp.get_context("spawn")
    channel = ctx.Queue()
    capacity = ctx.BoundedSemaphore(max_pending)
    start_event = ctx.Event()
    generator_ready = ctx.Event()
    epoch = ctx.Value("q", 0)
    generator = ctx.Process(target=_load_generator, args=(trace, channel, capacity, start_event, generator_ready, epoch))
    generator.start()
    if not generator_ready.wait(timeout=60):
        generator.terminate()
        generator.join()
        raise RuntimeError("Load-generator startup timed out")
    futures, ready, inflight = {}, [], []
    rows = [{"request_id": a.request_id, "image_id": a.image_id, "measurement": a.measurement,
             "arrival_ns": None, "deadline_ns": None, "enqueue_ns": None,
             "status": "unfinished", "identity_ok": False} for a in trace]
    generator_done = False
    activity_samples = []
    shared_activity = threading.Event()
    monitor_stop = threading.Event()
    def monitor():
        while not monitor_stop.is_set():
            try:
                count = len(other_gpu_pids())
            except Exception:
                count = -1
            activity_samples.append({"host_ns": time.monotonic_ns(), "other_gpu_process_count": count})
            if count != 0:
                shared_activity.set()
            monitor_stop.wait(5)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    peak_pending = 0
    last_progress = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        epoch.value = time.monotonic_ns() + 200000000
        for r, a in zip(rows, trace):
            r["arrival_ns"] = epoch.value + round(a.arrival_s * 1e9)
            r["deadline_ns"] = r["arrival_ns"] + round(a.relative_deadline_s * 1e9)
        start_event.set()
        stop_ns = epoch.value + round((warmup_s + measurement_s + drain_s) * 1e9)
        try:
            while True:
                now = time.monotonic_ns()
                if shared_activity.is_set():
                    raise RuntimeError("Shared GPU activity detected during timing; preserve this failed attempt and retry when idle")
                if now >= stop_ns:
                    break
                # Consume completed copies and CPU results before considering reuse.
                for slot in list(inflight):
                    done = executor.consume(slot, corpus)
                    if done is None:
                        continue
                    inflight.remove(slot)
                    for request, detections in zip(done["requests"], done["results"]):
                        r = rows[request.request_id]
                        r.update({k: v for k, v in done.items() if k not in {"requests", "results"}})
                        r.update(status="completed", detection_count=len(detections),
                                 identity_ok=all(d["image_id"] == request.image_id for d in detections))
                        capacity.release()
                # A serial baseline includes CPU preprocessing without overlap.
                can_accept = not serial or (not futures and not ready and not inflight)
                if can_accept:
                    for _ in range(1 if serial else max_pending):
                        try:
                            request_id, enqueue, accepted = channel.get_nowait()
                        except queue.Empty:
                            break
                        if request_id is None:
                            generator_done = True
                            break
                        r = rows[request_id]
                        r["enqueue_ns"] = enqueue
                        if not accepted:
                            r["status"] = "rejected"
                            continue
                        futures[request_id] = pool.submit(preprocess_request, executor.detector, corpus, r["image_id"])
                for rid, future in list(futures.items()):
                    if not future.done():
                        continue
                    del futures[rid]
                    r = rows[rid]
                    try:
                        payload, begin, decoded, end = future.result()
                        r.update(preprocess_start_ns=begin, decode_end_ns=decoded, ready_ns=end,
                                 decode_ms=(decoded - begin) / 1e6, preprocess_ms=(end - begin) / 1e6)
                        ready.append(Ready(rid, r["image_id"], r["arrival_ns"], r["deadline_ns"], end, payload))
                    except Exception as exc:
                        r.update(status="failed", error_type=type(exc).__name__)
                        capacity.release()
                peak_pending = max(peak_pending, len(futures) + len(ready) + sum(len(s.requests) for s in inflight))
                limit = 1 if serial else 2
                if ready and len(inflight) < limit:
                    # Host dispatch may precede GPU start by an entire queued
                    # batch. Only an observed CUDA start permits elapsed-time
                    # subtraction; unstarted work retains its full estimate.
                    gpu_remaining = 0
                    if policy == "D0":
                        for s in inflight:
                            if s.forward_observed_start_ns is None and s.events["forward_start"].query():
                                s.forward_observed_start_ns = time.monotonic_ns()
                            gpu_remaining += remaining_service_ns(service_ns[s.bucket],
                                s.forward_observed_start_ns, time.monotonic_ns())
                    decision = scheduler.decide(ready, now, gpu_remaining)
                    if decision.request_ids:
                        slot = executor.free_slot(decision.bucket)
                        if slot is not None:
                            lookup = {r.request_id: r for r in ready}
                            batch = [lookup[rid] for rid in decision.request_ids]
                            chosen = set(decision.request_ids)
                            ready = [r for r in ready if r.request_id not in chosen]
                            executor.submit(slot, batch)
                            inflight.append(slot)
                if generator_done and not futures and not ready and not inflight:
                    break
                if not generator.is_alive() and not generator_done and generator.exitcode:
                    raise RuntimeError("Open-loop load generator failed")
                if progress and time.monotonic() - last_progress >= 30:
                    progress(sum(r["status"] == "completed" for r in rows), len(rows))
                    last_progress = time.monotonic()
                # CUDA events are queried without a device-wide synchronization.
                time.sleep(0.00005)
        finally:
            ended_ns = time.monotonic_ns()
            monitor_stop.set()
            monitor_thread.join(timeout=12)
            if sys.exc_info()[0] is not None and output:
                partial_path = Path(output)
                partial_path.mkdir(parents=True, exist_ok=True)
                fields = sorted({key for row in rows for key in row})
                with (partial_path / "request_events.partial.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                write_json(partial_path / "gpu_activity_partial.json", activity_samples)
            generator.join(timeout=2)
            if generator.is_alive():
                generator.terminate()
                generator.join()
            # This cleanup lies outside the drain observation window. No late cleanup
            # result is retroactively counted as a completion.
            executor.drain()
            for slot in inflight:
                slot.transition("D2H", "CPU_CONSUMING")
                slot.device_outputs = None
                slot.release()
    # Loadgen sends only metadata; recover metadata still buffered at the drain cap.
    while True:
        try:
            rid, enqueue, accepted = channel.get_nowait()
        except queue.Empty:
            break
        if rid is not None:
            rows[rid]["enqueue_ns"] = enqueue
            if not accepted:
                rows[rid]["status"] = "rejected"
    channel.close()
    channel.join_thread()
    observed = max(measurement_s, (ended_ns - epoch.value) / 1e9 - warmup_s)
    metrics = summarize(rows, measurement_s, observed, min(a.relative_deadline_s for a in trace))
    metrics.update(policy=policy, trace_sha256=trace_hash(trace), peak_pending=peak_pending,
                   shared_gpu_jobs_observed=shared_activity.is_set(), gpu_activity_samples=activity_samples,
                   steady_window_completion_rps=sum(r["status"] == "completed" and
                       epoch.value + warmup_s * 1e9 <= r["complete_ns"] <
                       epoch.value + (warmup_s + measurement_s) * 1e9 for r in rows) / measurement_s,
                   memory=executor.memory(),
                   dummy_lane_fraction=(sum(r.get("bucket", 1) / max(1, r.get("valid_count", 1)) - 1
                                             for r in rows if r["status"] == "completed") /
                                         max(1, sum(r.get("bucket", 1) / max(1, r.get("valid_count", 1))
                                                    for r in rows if r["status"] == "completed"))))
    if len({r["request_id"] for r in rows}) != len(trace) or len(rows) != len(trace):
        raise RuntimeError("Request identity accounting failed")
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row})
        with (output / "request_events.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        write_json(output / "metrics.json", metrics)
    return metrics, rows
