"""Full-forward capture at immutable per-slot addresses, one compute stream."""
import time
import torch


def capture_slots(detector, slots, compute_stream):
    started = time.monotonic()
    for slot in slots:
        if slot.state != "FREE":
            raise RuntimeError("Capture requires a free slot")
        with torch.inference_mode(), torch.cuda.stream(compute_stream):
            for _ in range(3):
                detector.forward(slot.values, slot.mask)
        compute_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, stream=compute_stream):
            outputs = detector.forward(slot.values, slot.mask)
        slot.graph = graph
        slot.graph_outputs = outputs
    compute_stream.synchronize()
    return {"protocol_version": 2, "capture": "full_DETR_forward_with_pixel_mask",
            "slots": len(slots), "private_graph_pools": True,
            "initialization_seconds": time.monotonic() - started}
