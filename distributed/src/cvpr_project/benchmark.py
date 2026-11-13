"""Real-image serial pilot, conservative bucket service calibration, capacity."""
import time
import numpy as np
import torch
from .scheduler import Ready
from .serve import preprocess_request
from .runs import assert_no_other_gpu_jobs


def pilot(executor, corpus, requests=1000, time_cap_s=600):
    start = time.monotonic()
    rows = []
    for index in range(requests):
        if time.monotonic() - start >= time_cap_s:
            break
        image_id = corpus.ids[index % len(corpus.ids)]
        arrival = time.monotonic_ns()
        payload, begin, decoded, ready = preprocess_request(executor.detector, corpus, image_id)
        request = Ready(index, image_id, arrival, arrival + 10**12, ready, payload)
        slot = executor.free_slot(1)
        executor.submit(slot, [request])
        slot.events["done"].synchronize()
        done = executor.consume(slot, corpus)
        rows.append({"request_id": index, "image_id": image_id,
                     "e2e_ms": (done["complete_ns"] - arrival) / 1e6,
                     "preprocess_ms": (ready - begin) / 1e6, "decode_ms": (decoded - begin) / 1e6,
                     **{k: done[k] for k in ["h2d_ms", "forward_ms", "d2h_ms", "pack_ms", "submit_host_ms", "postprocess_ms"]}})
        if index % 100 == 0:
            print(f"pilot {executor.backend} pinned={executor.pinned}: {index}/{requests}", flush=True)
    # Exclude first 20 cold requests from the SLO reference; preserve raw rows.
    measured = rows[20:]
    if not measured:
        raise RuntimeError("Insufficient pilot samples")
    return {"protocol_version": 2, "requests": len(rows), "cold_requests_excluded": 20,
            "precision": executor.detector.precision, "hardware": executor.detector.hardware,
            "elapsed_seconds": time.monotonic() - start,
            "serial_rps": 1000 / float(np.mean([r["e2e_ms"] for r in measured])),
            "stage_median_ms": {k: float(np.median([r[k] for r in measured])) for k in measured[0] if k.endswith("_ms")},
            "e2e_p95_ms": float(np.percentile([r["e2e_ms"] for r in measured], 95)), "rows": rows}


def service_table(executor, corpus, samples=100):
    values = {}
    all_rows = []
    for b in executor.buckets:
        for n in range(samples + 10):
            ids = [corpus.ids[(n * b + j) % len(corpus.ids)] for j in range(b)]
            payloads = [executor.detector.preprocess(corpus.image(i)) for i in ids]
            requests = [Ready(n * b + j, i, 0, 0, 0, p) for j, (i, p) in enumerate(zip(ids, payloads))]
            slot = executor.free_slot(b)
            executor.submit(slot, requests)
            slot.events["done"].synchronize()
            done = executor.consume(slot, corpus)
            if n >= 10:
                all_rows.append({"bucket": b, "dispatch_to_result_ns": done["complete_ns"] - done["dispatch_ns"],
                                 **{k: done[k] for k in ["h2d_ms", "forward_ms", "d2h_ms", "pack_ms", "submit_host_ms", "postprocess_ms"]}})
        values[b] = round(np.percentile([r["dispatch_to_result_ns"] for r in all_rows if r["bucket"] == b], 95))
        print(f"service b{b}: p95 {values[b]/1e6:.3f} ms", flush=True)
    return {"protocol_version": 2, "service_ns": values, "scope": "dispatch_pack_to_CPU_detections_p95",
            "precision": executor.detector.precision, "backend": executor.backend,
            "rows": all_rows, "samples_per_bucket": samples}


def capacity(executor, corpus, bucket, seconds=90, workers=4):
    # Capacity includes real CPU decode/transform, two slots, H2D and CPU results.
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    start, completed, sequence = None, 0, 0
    active = []
    sequence_lock = Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        def next_batch():
            nonlocal sequence
            with sequence_lock:
                first = sequence
                ids = [corpus.ids[(first + j) % len(corpus.ids)] for j in range(bucket)]
                sequence += bucket
            requests = [Ready(first + j, i, 0, 0, 0, executor.detector.preprocess(corpus.image(i))) for j, i in enumerate(ids)]
            return requests
        pending = [pool.submit(next_batch) for _ in range(max(4, workers * 2))]
        warmup_start = time.monotonic()
        last_activity_check = warmup_start
        while start is None or time.monotonic() - start < seconds:
            if time.monotonic() - last_activity_check >= 5:
                assert_no_other_gpu_jobs()
                last_activity_check = time.monotonic()
            for slot in list(active):
                done = executor.consume(slot, corpus)
                if done:
                    active.remove(slot)
                    if start is not None:
                        completed += done["valid_count"]
            if start is None and time.monotonic() - warmup_start >= 10:
                # Counting completion times over the 90 s steady window is deliberate.
                start = time.monotonic()
            if len(active) < 2 and pending[0].done():
                slot = executor.free_slot(bucket)
                if slot is not None:
                    batch = pending.pop(0).result()
                    executor.submit(slot, batch)
                    active.append(slot)
                    pending.append(pool.submit(next_batch))
            time.sleep(.00005)
        elapsed = time.monotonic() - start
    executor.drain()
    for slot in active:
        executor.consume(slot, corpus)
    return {"protocol_version": 2, "duration_seconds": elapsed, "completed_in_window": completed,
            "lambda_ref_rps": completed / elapsed, "bucket": bucket, "backend": executor.backend,
            "scope": "backlogged_RAM_encoded_bytes_to_CPU_results", "workers": workers}
