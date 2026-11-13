"""Fixed pools plus H2D -> compute -> D2H event dependencies."""
import time
import torch
from .buffer_pool import Slot
from .graph_pool import capture_slots


class Executor:
    def __init__(self, detector, buckets=(1, 2, 4, 8), slots_per_bucket=2, backend="eager", pinned=True):
        self.detector = detector
        self.buckets = tuple(buckets)
        self.backend = backend
        self.pinned = pinned
        self.compute = torch.cuda.Stream(device=detector.device)
        self.h2d = torch.cuda.Stream(device=detector.device)
        self.d2h = torch.cuda.Stream(device=detector.device)
        self.slots = [Slot(b, s, detector.dtype, detector.device, pinned=pinned)
                      for b in buckets for s in range(slots_per_bucket)]
        # Allocation/zero-fill happens on the caller's stream. Establish ownership
        # before a side stream reads or overwrites those tensors.
        initialization_stream = torch.cuda.current_stream(detector.device)
        self.compute.wait_stream(initialization_stream)
        self.h2d.wait_stream(initialization_stream)
        self.d2h.wait_stream(initialization_stream)
        self.capture_metadata = {"capture": "none", "initialization_seconds": 0}
        self.forward_fn = detector.forward
        if backend == "graph":
            self.capture_metadata = capture_slots(detector, self.slots, self.compute)
        elif backend == "compile":
            # Explicitly disable automatic graphs: our slot implementation owns addresses.
            self.forward_fn = torch.compile(detector.forward, backend="inductor",
                                            options={"triton.cudagraphs": False}, fullgraph=False)
            start = time.monotonic()
            with torch.inference_mode(), torch.cuda.stream(self.compute):
                for b in buckets:
                    slot = self.free_slot(b)
                    self.forward_fn(slot.values, slot.mask)
            self.compute.synchronize()
            self.capture_metadata = {"capture": "none", "compile_mode": "default",
                                     "automatic_cudagraphs": False,
                                     "initialization_seconds": time.monotonic() - start}

    def free_slot(self, bucket):
        return next((s for s in self.slots if s.bucket == bucket and s.state == "FREE"), None)

    @torch.inference_mode()
    def submit(self, slot, requests, dispatch_ns=None):
        if not 0 < len(requests) <= slot.bucket:
            raise ValueError("Invalid batch occupancy")
        slot.transition("FREE", "FILLING")
        slot.requests = list(requests)
        slot.dispatch_ns = dispatch_ns or time.monotonic_ns()
        slot.forward_observed_start_ns = None
        torch.cuda.nvtx.range_push(f"pack_b{slot.bucket}_s{slot.slot_id}")
        for i, request in enumerate(requests):
            values, mask = request.payload
            slot.host_values[i].copy_(values)
            slot.host_mask[i].copy_(mask)
        # Dummy lanes repeat a valid image/mask, never an all-masked NaN input.
        for i in range(len(requests), slot.bucket):
            slot.host_values[i].copy_(slot.host_values[0])
            slot.host_mask[i].copy_(slot.host_mask[0])
        torch.cuda.nvtx.range_pop()
        slot.pack_end_ns = time.monotonic_ns()
        slot.transition("FILLING", "H2D")
        with torch.cuda.stream(self.h2d):
            torch.cuda.nvtx.range_push(f"h2d_b{slot.bucket}_s{slot.slot_id}")
            slot.events["h2d_start"].record()
            slot.values.copy_(slot.host_values, non_blocking=self.pinned)
            slot.mask.copy_(slot.host_mask, non_blocking=self.pinned)
            slot.events["h2d_end"].record()
            torch.cuda.nvtx.range_pop()
        slot.transition("H2D", "READY_GPU")
        with torch.cuda.stream(self.compute):
            self.compute.wait_event(slot.events["h2d_end"])
            slot.transition("READY_GPU", "COMPUTING")
            torch.cuda.nvtx.range_push(f"{self.backend}_forward_b{slot.bucket}_s{slot.slot_id}")
            slot.events["forward_start"].record()
            if slot.graph is not None:
                slot.graph.replay()
                logits, boxes = slot.graph_outputs
            else:
                logits, boxes = self.forward_fn(slot.values, slot.mask)
            slot.events["forward_end"].record()
            torch.cuda.nvtx.range_pop()
        with torch.cuda.stream(self.d2h):
            self.d2h.wait_event(slot.events["forward_end"])
            slot.transition("COMPUTING", "D2H")
            torch.cuda.nvtx.range_push(f"d2h_b{slot.bucket}_s{slot.slot_id}")
            slot.events["d2h_start"].record()
            slot.host_logits.copy_(logits, non_blocking=self.pinned)
            slot.host_boxes.copy_(boxes, non_blocking=self.pinned)
            slot.events["done"].record()
            torch.cuda.nvtx.range_pop()
        # Keep eager tensors alive until copies complete, even across streams.
        slot.device_outputs = (logits, boxes)
        slot.submit_end_ns = time.monotonic_ns()
        return slot

    def consume(self, slot, corpus):
        if not slot.events["done"].query():
            return None
        slot.transition("D2H", "CPU_CONSUMING")
        d2h_observed_ns = time.monotonic_ns()
        n = len(slot.requests)
        torch.cuda.nvtx.range_push(f"postprocess_b{slot.bucket}_s{slot.slot_id}")
        results = self.detector.postprocess(
            slot.host_logits[:n], slot.host_boxes[:n],
            [(corpus.images[r.image_id]["height"], corpus.images[r.image_id]["width"]) for r in slot.requests],
            [r.image_id for r in slot.requests], corpus.categories)
        torch.cuda.nvtx.range_pop()
        complete_ns = time.monotonic_ns()
        e = slot.events
        durations = {"h2d_ms": e["h2d_start"].elapsed_time(e["h2d_end"]),
                     "forward_ms": e["forward_start"].elapsed_time(e["forward_end"]),
                     "d2h_ms": e["d2h_start"].elapsed_time(e["done"]),
                     "pack_ms": (slot.pack_end_ns - slot.dispatch_ns) / 1e6,
                     "submit_host_ms": (slot.submit_end_ns - slot.dispatch_ns) / 1e6,
                     "postprocess_ms": (complete_ns - d2h_observed_ns) / 1e6}
        value = {"requests": list(slot.requests), "results": results, "complete_ns": complete_ns,
                 "d2h_observed_ns": d2h_observed_ns, "dispatch_ns": slot.dispatch_ns,
                 "pack_end_ns": slot.pack_end_ns, "submit_end_ns": slot.submit_end_ns,
                 "bucket": slot.bucket, "slot_id": slot.slot_id, "valid_count": n, **durations}
        # The host outputs and request payloads are reusable only after decoding.
        slot.device_outputs = None
        slot.release()
        return value

    def drain(self):
        self.h2d.synchronize()
        self.compute.synchronize()
        self.d2h.synchronize()

    def memory(self):
        return {"pinned_host_bytes": sum(s.host_bytes for s in self.slots if s.pinned),
                "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                "gpu_reserved_bytes": torch.cuda.memory_reserved(),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "slots_per_bucket": len(self.slots) // len(self.buckets),
                "compute_streams": 1, "capture": self.capture_metadata}
