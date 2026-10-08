"""Independent all-image COCO AP; timing failures never select AP images."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import contextlib
import io
import time
import re
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .runs import write_json, read_json, sha256
from .scheduler import Ready

AP_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def coco_evaluate(corpus, predictions):
    with contextlib.redirect_stdout(io.StringIO()) as text:
        coco = COCO(str(corpus.annotation_path))
        dt = coco.loadRes(predictions)
        evaluator = COCOeval(coco, dt, "bbox")
        evaluator.params.imgIds = sorted(corpus.ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"metrics_percent": dict(zip(AP_NAMES, (evaluator.stats * 100).tolist())),
            "cocoeval_stdout": text.getvalue(), "evaluated_images": len(corpus.ids)}


def evaluate_backend(executor, corpus, bucket=1, output=None, workers=4, label=None):
    started = time.monotonic()
    predictions = []
    # Keep preprocessed tensors bounded; encoded bytes are the persistent corpus.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for begin in range(0, len(corpus.ids), bucket * 8):
            image_ids = corpus.ids[begin:begin + bucket * 8]
            inputs = list(pool.map(lambda i: executor.detector.preprocess(corpus.image(i)), image_ids))
            for offset in range(0, len(inputs), bucket):
                ids = image_ids[offset:offset + bucket]
                batch = [Ready(begin + offset + j, i, 0, 0, 0, payload)
                         for j, (i, payload) in enumerate(zip(ids, inputs[offset:offset + bucket]))]
                slot = executor.free_slot(bucket)
                executor.submit(slot, batch)
                slot.events["done"].synchronize()
                result = executor.consume(slot, corpus)
                predictions.extend(p for rows in result["results"] for p in rows)
            if begin // (bucket * 8) % 20 == 0:
                print(f"quality {label or executor.backend}/b{bucket}: {min(begin + bucket * 8, len(corpus.ids))}/{len(corpus.ids)}", flush=True)
    metrics = coco_evaluate(corpus, predictions)
    metrics.update(protocol_version=2, backend=executor.backend, bucket=bucket,
                   precision=executor.detector.precision, label=label,
                   elapsed_seconds=time.monotonic() - started,
                   annotation_sha256=corpus.split["annotation_sha256"], split_sha256=corpus.split["split_sha256"],
                   scope="independent_offline_all_subset_images", max_detections=100, score_threshold=0)
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "metrics.json", metrics)
        # Predictions are local evidence, excluded from the public repository.
        write_json(output / "predictions.json", predictions)
        metrics["predictions_sha256"] = sha256(output / "predictions.json")
        write_json(output / "metrics.json", metrics)
    return metrics, predictions


def stress_outputs(eager, graph, corpus, requests=1000, atol=1e-5, rtol=1e-4, boxes_atol=None):
    """Alternating images, every occupancy, delayed CPU ownership, raw outputs."""
    if eager.detector.precision != graph.detector.precision:
        raise ValueError("Stress comparisons must use the same precision")
    inputs = {i: graph.detector.preprocess(corpus.image(i)) for i in corpus.ids[:32]}
    boxes_atol = atol if boxes_atol is None else boxes_atol
    count = 0
    max_logits_error, max_boxes_error = 0.0, 0.0
    memory_before = torch.cuda.memory_allocated()
    memory_samples = []
    while count < requests:
        for bucket in graph.buckets:
            n = 1 + ((count // 7) % bucket)
            n = min(n, requests - count)
            if n <= 0:
                break
            batch = [Ready(count + j, corpus.ids[(count * 7 + j) % 32], 0, 0, 0,
                           inputs[corpus.ids[(count * 7 + j) % 32]]) for j in range(n)]
            gs = graph.free_slot(bucket)
            es = eager.free_slot(bucket)
            graph.submit(gs, batch)
            eager.submit(es, batch)
            gs.events["done"].synchronize()
            es.events["done"].synchronize()
            # A second graph uses another slot while the first host output is held.
            second = graph.free_slot(bucket)
            if second is gs:
                raise AssertionError("In-flight slot returned by free_slot")
            held_logits = gs.host_logits[:n].clone()
            held_boxes = gs.host_boxes[:n].clone()
            if second is not None:
                reverse = list(reversed(batch))
                graph.submit(second, reverse)
                second.events["done"].synchronize()
                time.sleep(0.001)
                torch.testing.assert_close(gs.host_logits[:n], held_logits, atol=0, rtol=0)
                torch.testing.assert_close(gs.host_boxes[:n], held_boxes, atol=0, rtol=0)
                torch.testing.assert_close(second.host_logits[:n], held_logits.flip(0), atol=atol, rtol=rtol)
                graph.consume(second, corpus)
            max_logits_error = max(max_logits_error, float((held_logits.float() - es.host_logits[:n].float()).abs().max()))
            max_boxes_error = max(max_boxes_error, float((held_boxes.float() - es.host_boxes[:n].float()).abs().max()))
            torch.testing.assert_close(held_logits, es.host_logits[:n], atol=atol, rtol=rtol)
            torch.testing.assert_close(held_boxes, es.host_boxes[:n], atol=boxes_atol, rtol=rtol)
            result = graph.consume(gs, corpus)
            eager.consume(es, corpus)
            assert [r.request_id for r in result["requests"]] == [r.request_id for r in batch]
            assert len(result["results"]) == n
            assert all(all(p["image_id"] == r.image_id for p in rows)
                       for r, rows in zip(batch, result["results"]))
            count += n
            if count % 100 < n:
                memory_samples.append({"requests": count, "allocated_bytes": torch.cuda.memory_allocated()})
                print(f"stress {count}/{requests}", flush=True)
    tail = [x["allocated_bytes"] for x in memory_samples if x["requests"] >= 100]
    growth = max(tail) - min(tail) if tail else 0
    return {"protocol_version": 2, "status": "passed", "requests_compared": count,
            "same_precision": eager.detector.precision, "atol": atol, "boxes_atol": boxes_atol, "rtol": rtol,
            "max_abs_logits_error": max_logits_error, "max_abs_boxes_error": max_boxes_error,
            "every_bucket": list(graph.buckets), "partial_batches": True,
            "delayed_cpu_consumption": True, "separate_slot_reverse_order": True,
            "allocated_memory_before": memory_before, "allocated_memory_after": torch.cuda.memory_allocated(),
            "memory_samples": memory_samples, "allocated_growth_after_initial_100_requests_bytes": growth,
            "memory_stable_after_warmup": growth <= 1024 * 1024}


def visual_examples(corpus, predictions, output="results/detections"):
    from PIL import ImageDraw
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    by_image = {}
    for p in predictions:
        by_image.setdefault(p["image_id"], []).append(p)
    selected = []
    # Prefer COCO sources labelled "no known copyright restrictions" or US government.
    eligible = [i for i in corpus.ids if corpus.images[i].get("license") in {7, 8}]
    if len(eligible) < 3:
        eligible = [i for i in corpus.ids if corpus.images[i].get("license") in {1, 2, 4, 5, 7, 8}]
    annotation = read_json(corpus.annotation_path)
    licenses = {x["id"]: x for x in annotation["licenses"]}
    choices = [lambda x: x["width"] > x["height"], lambda x: x["height"] > x["width"],
               lambda x: sum(p["category_id"] == 1 and p["score"] >= .7 and
                             p["bbox"][2] * p["bbox"][3] < .08 * x["width"] * x["height"]
                             for p in by_image.get(x["id"], [])) >= 4]
    for predicate in choices:
        candidate = next((i for i in eligible if i not in selected and predicate(corpus.images[i]) and
                          any(p["score"] >= .7 for p in by_image.get(i, []))), None)
        if candidate is not None:
            selected.append(candidate)
    attribution = []
    for image_id in selected:
        image = corpus.image(image_id)
        draw = ImageDraw.Draw(image)
        for p in by_image[image_id]:
            if p["score"] < .7:
                continue
            x, y, w, h = p["bbox"]
            draw.rectangle([x, y, x + max(0, w), y + max(0, h)], outline="red", width=3)
            draw.text((max(0, x), max(0, y)), f"{corpus.categories[p['category_id']]} {p['score']:.2f}", fill="red")
        image.save(root / f"coco_{image_id}_detections.jpg", quality=90)
        meta = corpus.images[image_id]
        photo_id = re.search(r"/(\d+)_", meta.get("flickr_url", ""))
        attribution.append({"image_id": image_id, "source": meta.get("flickr_url", meta.get("coco_url")),
                            "coco_url": meta.get("coco_url"), "coco_license_id": meta.get("license"),
                            "license": licenses.get(meta.get("license")),
                            "source_photo_page": f"https://www.flickr.com/photo.gne?id={photo_id.group(1)}" if photo_id else None,
                            "author": "Not supplied by COCO metadata; original photo page retained for attribution",
                            "modification": "Overlaid pretrained DETR category, confidence and bounding boxes",
                            "visualization_threshold": .7, "evaluation_threshold": 0})
    write_json(root / "attribution.json", {"protocol_version": 2, "examples": attribution})
