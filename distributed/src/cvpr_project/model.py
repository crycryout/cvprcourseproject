"""Pinned pretrained model and common pixel-mask/coordinate contract."""
from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from .data import download
from .runs import read_json, sha256, write_json
from .hardware import require_device_scope

MODEL_ID = "facebook/detr-resnet-50"
PINNED_REVISION = "70120ba84d68ca1211e007c4fb61d0cd5424be54"


def prepare_weights(root="checkpoints/detr"):
    root = Path(root)
    manifest = Path("artifacts/model_v2.json")
    if manifest.exists():
        record = read_json(manifest)
        if all((root / f).exists() and sha256(root / f) == h for f, h in record["files_sha256"].items()):
            return record
        raise ValueError("Model files differ from existing manifest; preserve and inspect before replacing")
    url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{PINNED_REVISION}"
    with urllib.request.urlopen(url, timeout=30) as response:
        revision = json.load(response)["sha"]
    if revision != PINNED_REVISION:
        raise ValueError("The verified model revision does not match the immutable project revision")
    hashes = {}
    for name in ["config.json", "preprocessor_config.json", "pytorch_model.bin"]:
        print(f"Downloading pretrained {name} at {revision}", flush=True)
        download(f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}", root / name)
        hashes[name] = sha256(root / name)
    record = {"protocol_version": 2, "model_id": MODEL_ID, "revision": revision,
              "files_sha256": hashes, "weights_frozen": True, "training": False}
    write_json(manifest, record)
    return record


class Detector:
    def __init__(self, weights="checkpoints/detr", precision="fp32", device="cuda:0"):
        self.precision = precision
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.device = torch.device(device)
        properties = torch.cuda.get_device_properties(self.device)
        self.hardware = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                         "multiprocessors": properties.multi_processor_count,
                         "compute_capability": [properties.major, properties.minor],
                         "torch": torch.__version__, "cuda_build": torch.version.cuda,
                         "cudnn_version": torch.backends.cudnn.version(),
                         "device_uuid_sha256": hashlib.sha256(str(properties.uuid).encode()).hexdigest()}
        require_device_scope(self.hardware)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(1)
        self.processor = DetrImageProcessor.from_pretrained(weights, local_files_only=True)
        self.processor.size = {"shortest_edge": 480, "longest_edge": 640}
        self.processor.pad_size = {"height": 640, "width": 640}
        self.model = DetrForObjectDetection.from_pretrained(weights, local_files_only=True,
                                                          attn_implementation="eager").eval()
        self.model.requires_grad_(False)
        self.model.to(device=self.device, dtype=self.dtype)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def preprocess(self, image):
        # Uses version-pinned official API; padding is not real image content.
        batch = self.processor(images=image, return_tensors="pt",
                               size={"shortest_edge": 480, "longest_edge": 640},
                               do_pad=True, pad_size={"height": 640, "width": 640})
        values, mask = batch["pixel_values"][0], batch["pixel_mask"][0]
        if tuple(values.shape) != (3, 640, 640) or not bool(mask.any()):
            raise ValueError("Invalid fixed-size image/mask")
        return values.to(self.dtype), mask

    def forward(self, values, mask):
        out = self.model(pixel_values=values, pixel_mask=mask, return_dict=True)
        return out.logits, out.pred_boxes

    def postprocess(self, logits, boxes, sizes, image_ids, categories):
        from types import SimpleNamespace
        outputs = SimpleNamespace(logits=logits.float(), pred_boxes=boxes.float())
        decoded = self.processor.post_process_object_detection(outputs, threshold=0,
                                                               target_sizes=torch.tensor(sizes))
        results = []
        for image_id, prediction in zip(image_ids, decoded):
            rows = []
            for score, label, box in zip(prediction["scores"].tolist(), prediction["labels"].tolist(),
                                         prediction["boxes"].tolist()):
                if label not in categories:
                    continue
                x0, y0, x1, y1 = box
                rows.append({"image_id": int(image_id), "category_id": int(label),
                             "score": float(score), "bbox": [x0, y0, x1 - x0, y1 - y0]})
            results.append(rows)
        return results

    def validate_contract(self, corpus):
        for cid, name in corpus.categories.items():
            if self.id2label.get(cid) != name:
                raise ValueError(f"COCO category mapping mismatch at {cid}")
        checked = []
        for image_id in corpus.ids[:20]:
            image = corpus.image(image_id)
            official = self.processor(images=image, return_tensors="pt",
                                      size={"shortest_edge": 480, "longest_edge": 640}, do_pad=False)
            values, mask = self.preprocess(image)
            h, w = official["pixel_values"].shape[-2:]
            torch.testing.assert_close(values[:, :h, :w].float(),
                                       official["pixel_values"][0].to(self.dtype).float(), rtol=0, atol=0)
            expected_mask = torch.zeros((640, 640), dtype=mask.dtype)
            expected_mask[:h, :w] = 1
            torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
            if bool(values[:, h:, :].any()) or bool(values[:, :, w:].any()):
                raise ValueError("Padding must be zero after normalization")
            checked.append({"image_id": image_id, "original_hw": [image.height, image.width],
                            "resized_hw": [h, w], "valid_pixels": int(mask.sum())})
        # An analytically known box checks original-size decoding independently.
        logits = torch.full((1, 1, 92), -100.0)
        logits[0, 0, 1] = 100
        boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
        row = self.postprocess(logits, boxes, [(100, 200)], [1], corpus.categories)[0][0]
        np.testing.assert_allclose(row["bbox"], [50, 25, 100, 50])
        return {"protocol_version": 2, "mapping_verified": True, "padding_verified": True,
                "coordinate_decode_verified": True, "samples": checked}
