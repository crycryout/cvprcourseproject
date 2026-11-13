"""Real JPEG open-loop replay and exact all-offered cohort accounting."""
import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

from .common import Connection, digest, write_json, file_hash
from .policy import Item, POLICIES, VirtualWork, smooth_weighted_choice, global_decision


class Images:
    def __init__(self, root="data/coco", split="artifacts/data_v2.json", subset="calibration"):
        self.root = Path(root)
        self.split = json.loads(Path(split).read_text())
        self.ids = self.split[subset]
        annotation = json.loads((self.root / "annotations/instances_val2017.json").read_text())
        self.metadata = {x["id"]: x for x in annotation["images"]}
        self.annotation_path = self.root / "annotations/instances_val2017.json"
        self.encoded = {i: (self.root / "val2017" / self.metadata[i]["file_name"]).read_bytes() for i in self.ids}
        self.hashes = {i: hashlib.sha256(b).hexdigest() for i, b in self.encoded.items()}

    def pack(self, items):
        metas, chunks = [], []
        for item in items:
            i, rid = item.image_id, item.request_id
            image = self.metadata[i]
            blob = self.encoded[i]
            metas.append({"request_id": rid, "image_id": i,
                          "height": image["height"], "width": image["width"],
                          "jpeg_bytes": len(blob), "jpeg_sha256": self.hashes[i]})
            chunks.append(blob)
        return metas, b"".join(chunks)


def make_trace(kind, rate, seed, ids, deadlines_s, warmup_s, measurement_s):
    rng = np.random.Generator(np.random.PCG64(seed))
    end, times = warmup_s + measurement_s, []
    if kind == "periodic":
        times = np.arange(0, end, 1 / rate).tolist()
    elif kind == "poisson":
        t = 0
        while (t := t + rng.exponential(1 / rate)) < end:
            times.append(t)
    elif kind == "burst":
        for period in range(math.ceil(end)):
            t = float(period)
            while (t := t + rng.exponential(1 / (5 * rate))) < min(period + 0.2, end):
                times.append(t)
    else:
        raise ValueError(kind)
    image_order = np.random.Generator(np.random.PCG64(seed + 100000)).permutation(ids).tolist()
    slo = np.random.Generator(np.random.PCG64(seed + 200000)).choice(deadlines_s, len(times), p=[.5, .3, .2])
    return [Item(i, int(image_order[i % len(image_order)]), float(t), float(t + d), t >= warmup_s)
            for i, (t, d) in enumerate(zip(times, slo))]


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def summarize(events, measurement_s):
    cohort = [r for r in events if r["measurement"]]
    completed = [r for r in cohort if r["status"] == "completed"]
    on_time = [r for r in completed if r["complete_s"] <= r["deadline"]]
    latency = [(r["complete_s"] - r["arrival"]) * 1000 for r in completed]
    statuses = Counter(r["status"] for r in cohort)
    batch_rows = {(r["worker"], r["worker_event"]["batch_id"]): r["worker_event"]
                  for r in completed}
    return {"offered": len(cohort), "completed": len(completed), "on_time": len(on_time),
            "status_counts": dict(statuses),
            "goodput_rps": len(on_time) / measurement_s,
            "slo_success": len(on_time) / len(cohort) if cohort else 0,
            "completion_ratio": len(completed) / len(cohort) if cohort else 0,
            "latency_p50_ms": percentile(latency, 50), "latency_p95_ms": percentile(latency, 95),
            "latency_p99_ms": percentile(latency, 99),
            "generator_lag_p99_ms": percentile([r["generator_lag_ms"] for r in cohort], 99),
            "mean_batch_occupancy": float(np.mean([r["valid_count"] for r in batch_rows.values()])) if batch_rows else 0,
            "completed_batches": len(batch_rows),
            "worker_completed": dict(Counter(r["worker"] for r in completed)),
            "jpeg_bytes_sent": sum(r.get("jpeg_bytes_sent", 0) for r in cohort)}


async def connect_workers(hosts):
    return [await Connection.open(h.get("host", "127.0.0.1"), h["port"]) for h in hosts]


async def run_case(connections, images, profiles, cfg, policy, kind, rate, seed,
                   output=None, progress=False):
    if policy not in POLICIES:
        raise ValueError(policy)
    is_global = policy.startswith("GLOBAL")
    for j, connection in enumerate(connections):
        settings = cfg["local_settings"][j]
        await connection.request({"kind": "configure", "mode": "global" if is_global else "local",
                                  "max_batch": settings["max_batch"], "wait_ms": settings["wait_ms"]})
    trace = make_trace(kind, rate, seed, images.ids, cfg["deadlines_s"], cfg["warmup_s"], cfg["measurement_s"])
    trace_identity = digest([asdict(r) for r in trace])
    start = time.monotonic() + .1
    events = {r.request_id: dict(asdict(r), status="pending", complete_s=None, generator_lag_ms=0) for r in trace}
    active, global_queue, tasks = set(), [], set()
    wake = asyncio.Event()
    batch_counts, routed = [0] * len(connections), [0] * len(connections)
    virtual = [VirtualWork() for _ in connections]
    reservation_counter = 0
    weights = [max(int(b) / p["e2e_s"][b] for b in p["e2e_s"]) for p in profiles]
    credits = [0.] * len(connections)
    all_generated, rr_count = False, 0

    async def send(j, items, bucket=None, reservation=None):
        sent = time.monotonic() - start
        metas, payload = images.pack(items)
        for r, m in zip(items, metas):
            events[r.request_id].update(worker=j, dispatch_s=sent, jpeg_bytes_sent=m["jpeg_bytes"])
        try:
            request = {"kind": "infer", "requests": metas}
            if bucket is not None:
                request["bucket"] = bucket
            response, _ = await connections[j].request(request, payload, timeout=120)
            received = time.monotonic() - start
            results = response["results"]
            if len(results) != len(items):
                raise ValueError("Wrong result count")
            for r, row in zip(items, results):
                if (row["request_id"], row["image_id"], row["jpeg_sha256"]) != (r.request_id, r.image_id, images.hashes[r.image_id]):
                    raise ValueError("Request/output identity violation")
                if any(p["image_id"] != r.image_id for p in row["detections"]):
                    raise ValueError("Detection image identity mismatch")
                row = dict(row)
                predictions = row.pop("detections")
                event = events[r.request_id]
                # Drain-censored requests remain unfinished even if cleanup later succeeds.
                if event["status"] != "unfinished":
                    event.update(status="completed", complete_s=received,
                                 detection_count=len(predictions), worker_event=row,
                                 rpc_elapsed_ms=(received - sent) * 1000)
        except Exception as exc:
            for r in items:
                if events[r.request_id]["status"] != "unfinished":
                    events[r.request_id].update(status="error", error=type(exc).__name__ + ": " + str(exc))
        finally:
            for r in items:
                active.discard(r.request_id)
            batch_counts[j] -= 1
            virtual[j].complete(reservation, time.monotonic() - start)
            wake.set()

    def dispatch(j, items, bucket=None):
        nonlocal reservation_counter
        now = time.monotonic() - start
        batch_counts[j] += 1
        routed[j] += len(items)
        # Virtual-work reservation incorporates every dispatch before any new decision.
        if is_global:
            key = "gpu_s" if policy == "GLOBAL_GPU" else "e2e_s"
            service = profiles[j][key][str(bucket)]
        else:
            settings = cfg["local_settings"][j]
            b = str(settings["max_batch"])
            service = profiles[j]["e2e_s"][b] / settings["max_batch"]
        reservation_counter += 1
        virtual[j].reserve(reservation_counter, service, now)
        task = asyncio.create_task(send(j, items, bucket, reservation_counter))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def generate():
        nonlocal all_generated, rr_count
        for r in trace:
            target = start + r.arrival
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            now = time.monotonic() - start
            events[r.request_id]["generator_lag_ms"] = max(0., (now - r.arrival) * 1000)
            if len(active) >= cfg.get("max_pending", 512):
                events[r.request_id]["status"] = "rejected"
                continue
            active.add(r.request_id)
            if is_global:
                global_queue.append(r)
                wake.set()
            else:
                if policy == "RR":
                    j = rr_count % len(connections)
                    rr_count += 1
                elif policy == "WRR":
                    j = smooth_weighted_choice(weights, credits)
                else:
                    j = min(range(len(connections)), key=lambda k: (max(now, virtual[k].until) + profiles[k]["e2e_s"]["1"], k))
                dispatch(j, [r])
        all_generated = True
        wake.set()

    generator = asyncio.create_task(generate())
    cutoff = start + cfg["warmup_s"] + cfg["measurement_s"] + cfg.get("drain_s", 30)
    while not (all_generated and not active):
        now = time.monotonic() - start
        if time.monotonic() >= cutoff:
            break
        decision = None
        if is_global and global_queue:
            available = [(j, virtual[j].until) for j in range(len(connections))
                         if batch_counts[j] < cfg.get("max_outstanding_batches", 2)]
            decision = global_decision(global_queue, now, available, profiles,
                "gpu_s" if policy == "GLOBAL_GPU" else "e2e_s", cfg["buckets"], cfg["global_wait_ms"] / 1000)
            if decision and "items" in decision:
                selected = {r.request_id for r in decision["items"]}
                global_queue[:] = [r for r in global_queue if r.request_id not in selected]
                dispatch(decision["worker"], decision["items"], decision["bucket"])
                continue
        wake.clear()
        wait = min(.01, max(.0001, decision["wake"] - now)) if decision and "wake" in decision else .01
        try:
            await asyncio.wait_for(wake.wait(), wait)
        except asyncio.TimeoutError:
            pass
    await generator
    for rid in list(active):
        if events[rid]["status"] == "pending":
            events[rid]["status"] = "unfinished"
    # This is cleanup, outside the defined drain and never promotes censored results.
    if tasks:
        await asyncio.wait_for(asyncio.gather(*list(tasks)), 120)
    for r in events.values():
        if r["status"] == "pending":
            r["status"] = "unfinished"
    rows = list(events.values())
    metrics = summarize(rows, cfg["measurement_s"])
    lag_limit = max(1., min(cfg["deadlines_s"]) * 1000 * .05)
    metrics.update(protocol_version=3, policy=policy, arrival_type=kind, offered_rate=rate, seed=seed,
                   trace_sha256=trace_identity, image_subset_sha256=digest(images.ids),
                   warmup_s=cfg["warmup_s"], measurement_s=cfg["measurement_s"],
                   loadgen_lag_limit_ms=lag_limit,
                   valid_loadgen=(metrics["generator_lag_p99_ms"] or 0) <= lag_limit,
                   elapsed_s=time.monotonic() - start, routed_counts=routed,
                   steady_window_completions_rps=sum(r["status"] == "completed" and
                       cfg["warmup_s"] <= r["complete_s"] < cfg["warmup_s"] + cfg["measurement_s"]
                       for r in rows) / cfg["measurement_s"],
                   status="completed_with_failures" if metrics["status_counts"].get("error") else "completed")
    if output:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "requests.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        metrics["request_events_sha256"] = file_hash(root / "requests.jsonl")
        write_json(root / "metrics.json", metrics)
    if progress:
        print(json.dumps({k: metrics[k] for k in ["policy", "arrival_type", "offered_rate", "seed", "goodput_rps", "slo_success", "mean_batch_occupancy", "valid_loadgen"]}), flush=True)
    return metrics, rows
