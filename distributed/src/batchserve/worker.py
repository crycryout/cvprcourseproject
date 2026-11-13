"""Loopback TCP worker with per-request preprocessing and the frozen executor."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import traceback

from .common import read_frame, send_frame, source_hashes, file_hash, write_json


@dataclass
class Work:
    meta: dict
    jpeg: bytes
    received_ns: int
    future: Future
    request: object = None
    preprocess_start_ns: int = 0
    ready_ns: int = 0


class Engine:
    def __init__(self, weights, workers=8, buckets=(1, 2, 4, 8), backend="graph"):
        self.weights, self.workers, self.buckets, self.backend = weights, workers, buckets, backend
        self.source_identity = source_hashes()
        self.lock = threading.Condition()
        self.incoming, self.local_ready, self.groups = deque(), deque(), deque()
        self.outstanding, self.stopping, self.failure = 0, False, None
        self.mode, self.max_batch, self.wait_ns = "global", max(buckets), 0
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        if not self.ready.wait(180):
            raise TimeoutError("Worker model/capture initialization exceeded 180 seconds")
        if self.failure:
            raise RuntimeError(self.failure)

    def preprocess(self, work):
        from cvpr_project.scheduler import Ready
        work.preprocess_start_ns = time.monotonic_ns()
        with __import__("PIL.Image", fromlist=["Image"]).open(io.BytesIO(work.jpeg)) as source:
            image = source.convert("RGB")
            if [image.height, image.width] != [work.meta["height"], work.meta["width"]]:
                raise ValueError("JPEG dimensions differ from metadata")
            payload = self.detector.preprocess(image)
        work.jpeg = b""
        work.ready_ns = time.monotonic_ns()
        work.request = Ready(work.meta["request_id"], work.meta["image_id"], 0, 0,
                             work.ready_ns, payload)
        return work

    def submit(self, metas, payload, bucket=None):
        if not metas or len(metas) > 8:
            raise ValueError("Expected one to eight requests")
        if len({m["request_id"] for m in metas}) != len(metas):
            raise ValueError("Duplicate request_id within message")
        if sum(m["jpeg_bytes"] for m in metas) != len(payload):
            raise ValueError("JPEG length contract failed")
        if self.mode == "local" and len(metas) != 1:
            raise ValueError("Local batching receives individual requests")
        if self.mode == "global" and (bucket not in self.buckets or len(metas) > bucket):
            raise ValueError("Invalid explicit bucket")
        now = time.monotonic_ns()
        works, offset = [], 0
        with self.lock:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.outstanding + len(metas) > 512:
                raise RuntimeError("Worker pending bound exceeded")
            for meta in metas:
                length = meta["jpeg_bytes"]
                if not isinstance(length, int) or not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("Invalid JPEG size")
                w = Work(meta, payload[offset:offset + length], now, Future())
                offset += length
                works.append(w)
                self.images[meta["image_id"]] = {"height": meta["height"], "width": meta["width"]}
                self.incoming.append(w)
            if self.mode == "global":
                self.groups.append((works, bucket))
            self.outstanding += len(works)
            self.lock.notify_all()
        return works

    def configure(self, mode, max_batch=8, wait_ms=0, execution_backend="graph"):
        with self.lock:
            if self.outstanding:
                raise ValueError("Cannot reconfigure active worker")
            if mode not in {"global", "local"} or max_batch not in self.buckets or not 0 <= wait_ms <= 1000:
                raise ValueError("Invalid worker configuration")
            if execution_backend not in self.executors:
                raise ValueError("Unavailable execution backend")
            self.executor = self.executors[execution_backend]
            self.mode, self.max_batch, self.wait_ns = mode, max_batch, round(wait_ms * 1e6)
            self.local_ready.clear()
            self.groups.clear()
        return self.info()

    def info(self):
        return {"protocol_version": 3, "hardware": self.detector.hardware,
                "backend": self.executor.backend, "buckets": list(self.buckets),
                "cpu_workers": self.workers, "outstanding": self.outstanding,
                "mode": self.mode, "max_batch": self.max_batch, "wait_ms": self.wait_ns / 1e6,
                "initialization_seconds": self.initialization_seconds,
                "memory": self.executor.memory(),
                "weights_sha256": {p.name: file_hash(p) for p in sorted(Path(self.weights).glob("*")) if p.is_file()},
                "source_sha256": self.source_identity}

    def run(self):
        try:
            self._run()
        except BaseException as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.ready.set()
            for w in getattr(self, "all_work", {}).values():
                if not w.future.done():
                    w.future.set_exception(RuntimeError(self.failure))

    def _run(self):
        from .detector import DistributedDetector
        from cvpr_project.executor import Executor
        started = time.monotonic()
        self.detector = DistributedDetector(self.weights)
        self.executor = Executor(self.detector, self.buckets, slots_per_bucket=2, backend=self.backend)
        self.executors = {self.backend: self.executor}
        if self.backend == "graph":
            self.executors["eager"] = Executor(self.detector, self.buckets, slots_per_bucket=2, backend="eager")
        self.images, self.all_work = {}, {}
        self.categories = {k: v for k, v in self.detector.id2label.items() if v != "N/A"}
        self.corpus = SimpleNamespace(images=self.images, categories=self.categories)
        self.initialization_seconds = time.monotonic() - started
        pool = ThreadPoolExecutor(max_workers=self.workers)
        preprocessing, inflight = {}, []
        self.ready.set()
        while not self.stopping or self.outstanding:
            with self.lock:
                while self.incoming:
                    w = self.incoming.popleft()
                    self.all_work[id(w)] = w
                    preprocessing[pool.submit(self.preprocess, w)] = w
            for f in list(preprocessing):
                if not f.done():
                    continue
                w = preprocessing.pop(f)
                try:
                    f.result()
                    if self.mode == "local":
                        self.local_ready.append(w)
                except BaseException as exc:
                    w.future.set_exception(exc)
                    self.outstanding -= 1
                    self.all_work.pop(id(w), None)
            for slot, works in list(inflight):
                if not slot.events["done"].query():
                    continue
                raw = [(slot.host_logits[i].tolist(), slot.host_boxes[i].tolist())
                       for i in range(len(works))] if any(w.meta.get("include_raw") for w in works) else None
                result = self.executor.consume(slot, self.corpus)
                if result is None:
                    continue
                inflight.remove((slot, works))
                for index, (w, rows) in enumerate(zip(works, result["results"])):
                    if w.future.done():
                        raise RuntimeError("Request completed twice")
                    response = {"request_id": w.meta["request_id"], "image_id": w.meta["image_id"],
                        "detections": rows, "jpeg_sha256": w.meta["jpeg_sha256"],
                        "bucket": result["bucket"], "valid_count": result["valid_count"],
                        "batch_id": result["dispatch_ns"],
                        "worker_total_ms": (result["complete_ns"] - w.received_ns) / 1e6,
                        "cpu_wait_ms": (w.preprocess_start_ns - w.received_ns) / 1e6,
                        "preprocess_ms": (w.ready_ns - w.preprocess_start_ns) / 1e6,
                        "ready_wait_ms": (result["dispatch_ns"] - w.ready_ns) / 1e6,
                        **{k: result[k] for k in ["h2d_ms", "forward_ms", "d2h_ms", "pack_ms", "postprocess_ms"]}}
                    if raw is not None and w.meta.get("include_raw"):
                        response["raw_logits"], response["raw_boxes"] = raw[index]
                    with self.lock:
                        self.outstanding -= 1
                        self.all_work.pop(id(w), None)
                    w.future.set_result(response)
            selected, bucket = None, None
            if len(inflight) < 2:
                if self.mode == "global" and self.groups:
                    works, b = self.groups[0]
                    if all(w.request is not None or w.future.done() for w in works):
                        if any(w.future.done() for w in works):
                            self.groups.popleft()
                            for w in works:
                                if not w.future.done():
                                    w.future.set_exception(RuntimeError("Global batch preprocessing failed"))
                                    self.outstanding -= 1
                                    self.all_work.pop(id(w), None)
                        elif self.executor.free_slot(b) is not None:
                            self.groups.popleft()
                            selected, bucket = works, b
                elif self.mode == "local" and self.local_ready:
                    n = min(len(self.local_ready), self.max_batch)
                    oldest = self.local_ready[0].ready_ns
                    if n == self.max_batch or time.monotonic_ns() >= oldest + self.wait_ns:
                        b = next(z for z in self.buckets if z >= n)
                        if self.executor.free_slot(b) is not None:
                            selected, bucket = [self.local_ready.popleft() for _ in range(n)], b
            if selected:
                slot = self.executor.free_slot(bucket)
                self.executor.submit(slot, [w.request for w in selected])
                inflight.append((slot, selected))
            else:
                # Releases the GIL, bounds CPU polling, and permits prep/RPC progress.
                with self.lock:
                    self.lock.wait(timeout=0.0002)
        pool.shutdown(wait=True)
        self.executor.drain()


async def serve(engine, host, port):
    async def client(reader, writer):
        lock, tasks = asyncio.Lock(), set()

        async def respond(header, payload):
            mid = header["message_id"]
            try:
                kind = header.get("kind", "infer")
                if kind == "info":
                    response = {"info": engine.info()}
                elif kind == "configure":
                    response = {"info": engine.configure(header["mode"], header.get("max_batch", 8), header.get("wait_ms", 0), header.get("backend", "graph"))}
                elif kind == "infer":
                    metas = header["requests"]
                    offset = 0
                    for m in metas:
                        length = m["jpeg_bytes"]
                        if hashlib.sha256(payload[offset:offset + length]).hexdigest() != m["jpeg_sha256"]:
                            raise ValueError("JPEG digest mismatch")
                        offset += length
                    work = engine.submit(metas, payload, header.get("bucket"))
                    results = await asyncio.gather(*(asyncio.wrap_future(w.future) for w in work))
                    response = {"results": results}
                else:
                    raise ValueError("Unsupported request kind")
                async with lock:
                    await send_frame(writer, dict(response, message_id=mid))
            except Exception as exc:
                async with lock:
                    await send_frame(writer, {"message_id": mid, "error": f"{type(exc).__name__}: {exc}"})

        try:
            while True:
                header, payload = await read_frame(reader)
                task = asyncio.create_task(respond(header, payload))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()

    server = await asyncio.start_server(client, host, port, limit=128 * 1024)
    print(json.dumps({"status": "ready", "port": port, "hardware": engine.detector.hardware}), flush=True)
    async with server:
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--weights", default="checkpoints/detr")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=["graph", "eager"], default="graph")
    args = parser.parse_args()
    engine = Engine(args.weights, args.workers, backend=args.backend)
    asyncio.run(serve(engine, args.host, args.port))


if __name__ == "__main__":
    main()
