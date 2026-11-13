"""Calibration, source freeze, paired real experiments, and offline AP over RPC."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
import numpy as np

from .common import digest, file_hash, source_hashes, write_json
from .benchmark import Images, connect_workers, run_case
from .policy import Item, POLICIES


def read(path):
    return json.loads(Path(path).read_text())


async def rpc_batch(connection, images, ids, bucket, base=0, include_raw=False):
    items = [Item(base + k, i, 0, 10, True) for k, i in enumerate(ids)]
    metas, payload = images.pack(items)
    if include_raw:
        for meta in metas:
            meta["include_raw"] = True
    begin = time.monotonic()
    response, _ = await connection.request({"kind": "infer", "bucket": bucket, "requests": metas}, payload)
    elapsed = time.monotonic() - begin
    rows = response["results"]
    if [(r["request_id"], r["image_id"]) for r in rows] != [(i.request_id, i.image_id) for i in items]:
        raise ValueError("RPC request/output mismatch")
    if [r["jpeg_sha256"] for r in rows] != [images.hashes[i] for i in ids]:
        raise ValueError("RPC image bytes mismatch")
    return elapsed, rows


async def get_info(connections):
    return [(await c.request({"kind": "info"}))[0]["info"] for c in connections]


async def curves(connections, images, root, repeats=40):
    profiles, raw = [], []
    for j, c in enumerate(connections):
        await c.request({"kind": "configure", "mode": "global"})
        p = {"e2e_s": {}, "gpu_s": {}}
        for b in [1, 2, 4, 8]:
            times, gpu = [], []
            for k in range(repeats + 5):
                ids = [images.ids[(k * b + z) % len(images.ids)] for z in range(b)]
                elapsed, rows = await rpc_batch(c, images, ids, b, k * 8)
                if k >= 5:
                    times.append(elapsed)
                    gpu.append(rows[0]["forward_ms"] / 1000)
                    raw.append({"worker": j, "bucket": b, "elapsed_s": elapsed,
                                "worker": j, "sample": k - 5, "events": rows})
            p["e2e_s"][str(b)] = float(np.percentile(times, 95))
            p["gpu_s"][str(b)] = float(np.percentile(gpu, 95))
            print(json.dumps({"curve_worker": j, "bucket": b, "e2e_p95_ms": p["e2e_s"][str(b)] * 1000,
                              "gpu_p95_ms": p["gpu_s"][str(b)] * 1000}), flush=True)
        profiles.append(p)
    write_json(root / "curve_samples.json", raw)
    write_json(root / "curves.json", profiles)
    return profiles


async def stress(connections, images, root, requests=1024):
    """Every bucket/occupancy, different image order, same-worker eager reference."""
    evidence = []
    for j, c in enumerate(connections):
        count, max_logits, max_boxes = 0, 0., 0.
        while count < requests:
            for b in [1, 2, 4, 8]:
                n = min(1 + (count // 7) % b, requests - count)
                if n <= 0:
                    continue
                ids = [images.ids[(count * 7 + z) % 64] for z in range(n)]
                outputs = []
                for backend in ["eager", "graph"]:
                    await c.request({"kind": "configure", "mode": "global", "backend": backend})
                    _, rows = await rpc_batch(c, images, ids, b, count, include_raw=True)
                    outputs.append(rows)
                for eager, graph in zip(*outputs):
                    for key in ["raw_logits", "raw_boxes"]:
                        a, z = np.asarray(eager[key]), np.asarray(graph[key])
                        np.testing.assert_allclose(z, a, atol=1e-5, rtol=1e-4)
                        error = float(np.max(np.abs(z - a)))
                        if key == "raw_logits":
                            max_logits = max(max_logits, error)
                        else:
                            max_boxes = max(max_boxes, error)
                count += n
        evidence.append({"worker": j, "requests": count, "buckets": [1, 2, 4, 8],
                         "partial_batches": True, "max_logits_error": max_logits,
                         "max_boxes_error": max_boxes, "atol": 1e-5, "rtol": 1e-4, "status": "passed"})
        # Two RPC batches occupy independent slots while their outputs travel.
        for b in [1, 2, 4, 8]:
            groups = [images.ids[z:z + b] for z in [0, 16]]
            await c.request({"kind": "configure", "mode": "global", "backend": "eager"})
            reference = [await rpc_batch(c, images, ids, b, 2000 + k * 8, True)
                         for k, ids in enumerate(groups)]
            await c.request({"kind": "configure", "mode": "global", "backend": "graph"})
            actual = await asyncio.gather(*(rpc_batch(c, images, ids, b, 2000 + k * 8, True)
                                            for k, ids in enumerate(groups)))
            for (_, ref), (_, got) in zip(reference, actual):
                for a, z in zip(ref, got):
                    for key in ["raw_logits", "raw_boxes"]:
                        np.testing.assert_allclose(z[key], a[key], atol=1e-5, rtol=1e-4)
        evidence[-1]["concurrent_slot_batches_verified"] = 2
        print(json.dumps(evidence[-1]), flush=True)
    write_json(root / "stress.json", {"protocol_version": 3, "results": evidence})
    return evidence


def base_config(profiles, measurement_s=6, warmup_s=1):
    sbase = max(p["e2e_s"]["1"] for p in profiles)
    return {"protocol_version": 3, "buckets": [1, 2, 4, 8],
            "deadlines_s": [2 * sbase, 4 * sbase, 8 * sbase],
            "warmup_s": warmup_s, "measurement_s": measurement_s, "drain_s": 30,
            "max_pending": 512, "max_outstanding_batches": 2,
            "global_wait_ms": 2,
            "local_settings": [{"max_batch": 4, "wait_ms": 0} for _ in profiles]}


async def calibrate(connections, images, root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen.json").exists():
        raise ValueError("Existing freeze is immutable; use a fresh output root")
    profiles = await curves(connections, images, root)
    cfg = base_config(profiles)
    searches = []
    # Same twelve candidates, two offered rates and seed budget for each worker.
    for j, c in enumerate(connections):
        ref = max(int(b) / t for b, t in profiles[j]["e2e_s"].items())
        candidates = [(b, w) for b in [1, 2, 4, 8] for w in [0, 1, 2]]
        random.Random(611 + j).shuffle(candidates)
        scores = []
        for b, wait in candidates:
            solo_cfg = dict(cfg, local_settings=[{"max_batch": b, "wait_ms": wait}])
            runs = []
            for load in [.6, 1.5]:
                m, _ = await run_case([c], images, [profiles[j]], solo_cfg, "RR", "poisson", ref * load, 113,
                    root / f"local_w{j}_b{b}_wait{wait}_load{load}")
                runs.append(m)
            score = sum(r["goodput_rps"] for r in runs)
            scores.append((score, -sum(r["latency_p95_ms"] or 1e9 for r in runs), -b, -wait, b, wait))
            searches.append({"stage": "local", "worker": j, "max_batch": b, "wait_ms": wait, "runs": runs})
            print(json.dumps({"calibration_worker": j, "b": b, "wait_ms": wait, "score": score}), flush=True)
        _, _, _, _, b, wait = max(scores)
        cfg["local_settings"][j] = {"max_batch": b, "wait_ms": wait}
    ref = sum(max(int(b) / t for b, t in p["e2e_s"].items()) for p in profiles)
    capacity_cfg = dict(cfg, warmup_s=5, measurement_s=15)
    capacity, capacity_rows = await run_case(connections, images, profiles, capacity_cfg, "WRR", "poisson", ref * 4, 331,
                                root / "saturated_capacity", progress=True)
    if capacity["status_counts"].get("error") or not capacity["valid_loadgen"]:
        raise RuntimeError("Capacity calibration failed integrity/loadgen checks")
    ref = capacity["steady_window_completions_rps"]
    if ref <= 0:
        raise RuntimeError("No measured cluster capacity")
    write_json(root / "capacity.json", capacity)
    # A once-only final-rate recheck prevents serial-RPC curve estimates from
    # leaving local baselines tuned only at loads below the actual pipeline rate.
    node_ref = [sum(r["status"] == "completed" and r.get("worker") == j and
                   capacity_cfg["warmup_s"] <= r["complete_s"] < capacity_cfg["warmup_s"] + capacity_cfg["measurement_s"]
                   for r in capacity_rows) / capacity_cfg["measurement_s"] for j in range(len(connections))]
    for j, c in enumerate(connections):
        scores = []
        candidates = [(b, w) for b in [1, 2, 4, 8] for w in [0, 1, 2]]
        random.Random(711 + j).shuffle(candidates)
        for b, wait in candidates:
            solo_cfg = dict(cfg, local_settings=[{"max_batch": b, "wait_ms": wait}])
            runs = []
            for load in [.5, .85, 1.3]:
                m, _ = await run_case([c], images, [profiles[j]], solo_cfg, "RR", "poisson", node_ref[j] * load, 227,
                                     root / f"recheck_w{j}_b{b}_wait{wait}_load{load}")
                runs.append(m)
            score = sum(r["goodput_rps"] for r in runs)
            scores.append((score, -sum(r["latency_p95_ms"] or 1e9 for r in runs), -b, -wait, b, wait))
            searches.append({"stage": "final_rate_recheck", "worker": j, "max_batch": b,
                             "wait_ms": wait, "node_reference_rps": node_ref[j], "runs": runs})
            print(json.dumps({"recheck_worker": j, "b": b, "wait_ms": wait, "score": score}), flush=True)
        _, _, _, _, b, wait = max(scores)
        cfg["local_settings"][j] = {"max_batch": b, "wait_ms": wait}
    # Shared waiting cap selected with equal evaluation of both global estimators.
    scores = []
    for wait in [0, 2, 5, 10]:
        trial_cfg = dict(cfg, global_wait_ms=wait)
        runs = []
        for policy in ["GLOBAL_GPU", "GLOBAL_E2E"]:
            for kind in ["poisson", "burst"]:
                m, _ = await run_case(connections, images, profiles, trial_cfg, policy, kind, ref * .7, 113,
                                     root / f"global_wait{wait}_{policy}_{kind}")
                runs.append(m)
        score = sum(r["goodput_rps"] for r in runs)
        scores.append((score, -wait, wait))
        searches.append({"stage": "global", "wait_ms": wait, "runs": runs})
        print(json.dumps({"global_wait_ms": wait, "calibration_score": score}), flush=True)
    cfg["global_wait_ms"] = max(scores)[2]
    cfg.update(warmup_s=5, measurement_s=30, rates=[ref * x for x in [.2, .5, .85, 1.3]],
               rate_multipliers=[.2, .5, .85, 1.3], lambda_ref_rps=ref,
               lambda_ref_scope="WRR saturated steady-window completions before once-only final-rate local recheck",
               final_recheck_node_reference_rps=node_ref,
               arrival_types=["poisson", "burst"], seeds=[17, 42, 2026], policies=list(POLICIES))
    write_json(root / "calibration_search.json", searches)
    info = await get_info(connections)
    cfg.update(profiles=profiles, workers=info, source_sha256=source_hashes(),
               calibration_subset_sha256=digest(images.ids),
               split_sha256=images.split["split_sha256"],
               model_revision="70120ba84d68ca1211e007c4fb61d0cd5424be54",
               created_unix_s=time.time(), transport="persistent TCP; remote via authenticated SSH forward")
    cfg["freeze_sha256"] = digest(cfg)
    write_json(root / "frozen.json", cfg, exclusive=True)
    print(json.dumps({"frozen": str(root / "frozen.json"), "sha256": cfg["freeze_sha256"],
                      "local_settings": cfg["local_settings"], "global_wait_ms": cfg["global_wait_ms"],
                      "deadlines_s": cfg["deadlines_s"], "rates": cfg["rates"]}), flush=True)


def verify_freeze(frozen):
    copy = dict(frozen)
    recorded = copy.pop("freeze_sha256")
    if digest(copy) != recorded or source_hashes() != frozen["source_sha256"]:
        raise ValueError("Freeze/source hashes differ; preserve evidence and recalibrate")


async def verify_workers(connections, cfg):
    for actual, expected in zip(await get_info(connections), cfg["workers"]):
        for key in ["hardware", "weights_sha256", "source_sha256", "cpu_workers", "buckets"]:
            if actual[key] != expected[key]:
                raise ValueError(f"Worker changed after calibration: {key}")


async def matrix(connections, images, root, cfg):
    verify_freeze(cfg)
    await verify_workers(connections, cfg)
    cases = [(p, k, x, s) for k in cfg["arrival_types"] for x in cfg["rates"]
             for s in cfg["seeds"] for p in cfg["policies"]]
    random.Random(20261113).shuffle(cases)
    write_json(root / "matrix_plan.json", {"protocol_version": 3, "freeze_sha256": cfg["freeze_sha256"],
                                         "cases": cases})
    for index, (p, k, rate, seed) in enumerate(cases):
        name = f"{p}_{k}_{rate:.6f}_s{seed}"
        target = root / "runs" / name
        if (target / "metrics.json").exists():
            record = read(target / "metrics.json")
            if record.get("freeze_sha256") != cfg["freeze_sha256"]:
                raise ValueError("Existing case has another freeze")
            continue
        if target.exists():
            raise ValueError(f"Partial case retained at {target}; inspect before resuming")
        print(f"case {index + 1}/{len(cases)} {name}", flush=True)
        target.mkdir(parents=True)
        write_json(target / "manifest.json", {"status": "running", "freeze_sha256": cfg["freeze_sha256"],
                                             "started_unix_s": time.time(), "case": [p, k, rate, seed]})
        try:
            m, _ = await run_case(connections, images, cfg["profiles"], cfg, p, k, rate, seed, target, progress=True)
            m["freeze_sha256"] = cfg["freeze_sha256"]
            write_json(target / "metrics.json", m)
            write_json(target / "manifest.json", {"status": m["status"], "freeze_sha256": cfg["freeze_sha256"],
                                                 "finished_unix_s": time.time()})
            if not m["valid_loadgen"]:
                raise RuntimeError("Load generator lag exceeds frozen tolerance; preserve and diagnose")
        except BaseException as exc:
            write_json(target / "failure.json", {"error": str(exc), "type": type(exc).__name__, "unix_s": time.time()})
            raise


async def controls(connections, images, root, cfg):
    verify_freeze(cfg)
    await verify_workers(connections, cfg)
    for j, c in enumerate(connections):
        solo = dict(cfg, local_settings=[cfg["local_settings"][j]])
        cases = [(k, r, s) for k in cfg["arrival_types"] for r in cfg["rates"] for s in cfg["seeds"]]
        random.Random(617 + j).shuffle(cases)
        for k, rate, seed in cases:
            target = root / "controls" / f"worker{j}_{k}_{rate:.6f}_s{seed}"
            if (target / "metrics.json").exists():
                if read(target / "metrics.json").get("freeze_sha256") != cfg["freeze_sha256"]:
                    raise ValueError("Control belongs to another freeze")
                continue
            print(f"single-worker control {j} {k} {rate:.3f} seed{seed}", flush=True)
            m, _ = await run_case([c], images, [cfg["profiles"][j]], solo, "RR", k, rate, seed, target, progress=True)
            m.update(freeze_sha256=cfg["freeze_sha256"], topology=f"worker{j}_only")
            write_json(target / "metrics.json", m)
            if not m["valid_loadgen"]:
                raise RuntimeError("Control loadgen-limited; preserve and diagnose")


async def quality(connections, images, root, cfg):
    from cvpr_project.quality import coco_evaluate
    verify_freeze(cfg)
    await verify_workers(connections, cfg)
    corpus = SimpleNamespace(annotation_path=images.annotation_path, ids=images.ids)
    all_metrics = []
    for j, c in enumerate(connections):
        reference = None
        for backend, b in [("eager", 1)] + [("graph", z) for z in cfg["buckets"]]:
            dest = root / "quality" / f"worker{j}_{backend}_b{b}"
            if (dest / "metrics.json").exists():
                m = read(dest / "metrics.json")
                if m.get("freeze_sha256") != cfg["freeze_sha256"]:
                    raise ValueError("Quality output has another freeze")
            else:
                await c.request({"kind": "configure", "mode": "global", "backend": backend})
                predictions, stage = [], []
                start = time.monotonic()
                for begin in range(0, len(images.ids), b):
                    _, rows = await rpc_batch(c, images, images.ids[begin:begin + b], b, begin)
                    predictions.extend(p for r in rows for p in r["detections"])
                    stage.extend({k: r[k] for k in ["request_id", "image_id", "bucket", "valid_count", "forward_ms", "preprocess_ms"]} for r in rows)
                    if begin % (b * 100) == 0:
                        print(f"AP worker{j} {backend}/b{b}: {begin}/{len(images.ids)}", flush=True)
                m = coco_evaluate(corpus, predictions)
                dest.mkdir(parents=True, exist_ok=True)
                write_json(dest / "predictions.json", predictions)
                write_json(dest / "stages.json", stage)
                m.update(protocol_version=3, worker=j, backend=backend, bucket=b,
                         elapsed_s=time.monotonic() - start, freeze_sha256=cfg["freeze_sha256"],
                         predictions_sha256=file_hash(dest / "predictions.json"),
                         stages_sha256=file_hash(dest / "stages.json"))
                write_json(dest / "metrics.json", m)
            if backend == "eager":
                reference = m["metrics_percent"]["AP"]
            m["reference_AP"] = reference
            m["AP_delta_pp"] = m["metrics_percent"]["AP"] - reference
            if abs(m["AP_delta_pp"]) > .1:
                raise ValueError("Independent all-image AP failed the 0.1 pp quality gate")
            all_metrics.append(m)
            write_json(root / "quality_summary.json", all_metrics)
            print(json.dumps({"quality_worker": j, "backend": backend, "bucket": b,
                              "AP": m["metrics_percent"]["AP"], "delta_pp": m["AP_delta_pp"]}), flush=True)


async def main_async(args):
    hosts = read(args.hosts)
    connections = await connect_workers(hosts)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "info":
            info = await get_info(connections)
            write_json(root / "workers.json", info)
            print(json.dumps([{k: row[k] for k in ["hardware", "backend", "outstanding", "initialization_seconds"]} for row in info]), flush=True)
            return
        images = Images(subset="held_out" if args.command in {"matrix", "quality", "controls"} else "calibration")
        if args.command == "stress":
            await stress(connections, images, root, args.requests)
        elif args.command == "calibrate":
            await calibrate(connections, images, root)
        elif args.command == "pilot":
            profiles = await curves(connections, images, root, repeats=12)
            cfg = base_config(profiles, measurement_s=5, warmup_s=1)
            ref = sum(max(int(b) / t for b, t in p["e2e_s"].items()) for p in profiles)
            results = []
            for p in POLICIES:
                m, _ = await run_case(connections, images, profiles, cfg, p, "poisson", ref * .5, 113,
                                     root / p, progress=True)
                results.append(m)
            write_json(root / "pilot.json", {"config": cfg, "profiles": profiles, "runs": results})
        else:
            cfg = read(args.frozen or root / "frozen.json")
            if args.command == "matrix":
                await matrix(connections, images, root, cfg)
            elif args.command == "quality":
                await quality(connections, images, root, cfg)
            elif args.command == "controls":
                await controls(connections, images, root, cfg)
    finally:
        for c in connections:
            await c.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["info", "pilot", "stress", "calibrate", "matrix", "quality", "controls"])
    p.add_argument("--hosts", default="configs/hosts.local.json")
    p.add_argument("--output", default="artifacts/distributed_v3")
    p.add_argument("--frozen")
    p.add_argument("--requests", type=int, default=1024)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
