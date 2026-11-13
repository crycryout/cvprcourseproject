"""General-GPU adapter retaining the exact v2 preprocessing/decoding methods."""
import hashlib
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from cvpr_project.model import Detector


class DistributedDetector(Detector):
    def __init__(self, weights="checkpoints/detr", precision="fp32", device="cuda:0"):
        if precision != "fp32":
            raise ValueError("The distributed experiment freezes FP32")
        self.precision, self.dtype = precision, torch.float32
        self.device = torch.device(device)
        p = torch.cuda.get_device_properties(self.device)
        if "MIG" in p.name:
            raise ValueError("Full physical GPU required; do not alter MIG automatically")
        self.hardware = {"name": p.name, "total_memory_bytes": p.total_memory,
                         "multiprocessors": p.multi_processor_count,
                         "compute_capability": [p.major, p.minor],
                         "torch": torch.__version__, "cuda_build": torch.version.cuda,
                         "cudnn_version": torch.backends.cudnn.version(),
                         "device_uuid_sha256": hashlib.sha256(str(p.uuid).encode()).hexdigest()}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(1)
        self.processor = DetrImageProcessor.from_pretrained(weights, local_files_only=True)
        self.processor.size = {"shortest_edge": 480, "longest_edge": 640}
        self.processor.pad_size = {"height": 640, "width": 640}
        self.model = DetrForObjectDetection.from_pretrained(
            weights, local_files_only=True, attn_implementation="eager").eval()
        self.model.requires_grad_(False)
        self.model.to(device=self.device, dtype=self.dtype)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
