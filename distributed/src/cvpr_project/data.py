"""Official COCO download, immutable split, and encoded-byte corpus."""
from pathlib import Path
import io
import json
import urllib.request
import zipfile
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from .runs import object_hash, sha256, write_json

SOURCES = {
    "val2017.zip": "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(url, target):
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    partial.replace(target)


def download_ranges(url, target, concurrency=24, chunk_bytes=4 * 1024 * 1024):
    """Resume verified byte ranges from the official S3 object; validate ZIP later."""
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith("bytes 0-0/"):
            raise ValueError("Official source does not support verified byte ranges")
        length = int(content_range.split("/")[-1])
        etag = response.headers.get("ETag")
    parts = target.with_suffix(target.suffix + ".parts")
    parts.mkdir(exist_ok=True)
    count = (length + chunk_bytes - 1) // chunk_bytes
    def fetch(index):
        begin = index * chunk_bytes
        end = min(length, begin + chunk_bytes) - 1
        p = parts / f"{index:05}.part"
        if p.exists() and p.stat().st_size == end - begin + 1:
            return
        expected = f"bytes {begin}-{end}/{length}"
        for attempt in range(4):
            try:
                headers = {"Range": f"bytes={begin}-{end}", "If-Match": etag}
                # Query distinguishes ranges for intermediary caches.
                request = urllib.request.Request(url + f"?cvpr_range={index}", headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Invalid range response")
                    payload = response.read()
                if len(payload) != end - begin + 1:
                    raise ValueError("Truncated range")
                p.write_bytes(payload)
                print(f"COCO {target.name}: part {index+1}/{count}", flush=True)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(fetch, range(count)))
    partial = target.with_suffix(target.suffix + ".assembled")
    with partial.open("wb") as out:
        for index in range(count):
            with (parts / f"{index:05}.part").open("rb") as f:
                while block := f.read(1024 * 1024):
                    out.write(block)
    if partial.stat().st_size != length:
        raise ValueError("Assembled object length mismatch")
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError("COCO ZIP CRC verification failed")
    partial.replace(target)
    write_json(target.with_suffix(".download.json"), {"source": url, "etag": etag,
               "bytes": length, "sha256": sha256(target), "range_validation": True, "zip_crc_verified": True})


def prepare(root=Path("data/coco")):
    root = Path(root)
    images = root / "val2017"
    annotations = root / "annotations/instances_val2017.json"
    for filename, url in SOURCES.items():
        wanted = images.is_dir() if filename == "val2017.zip" else annotations.is_file()
        if not wanted:
            archive = root / filename
            print(f"Downloading official COCO {filename}", flush=True)
            download_ranges(url, archive)
            with zipfile.ZipFile(archive) as z:
                members = z.namelist() if filename == "val2017.zip" else ["annotations/instances_val2017.json"]
                for name in members:
                    if ".." in Path(name).parts or Path(name).is_absolute():
                        raise ValueError("Unsafe archive member")
                z.extractall(root, members=members)
    annotation = json.loads(annotations.read_text())
    ids = np.array(sorted(x["id"] for x in annotation["images"]), dtype=np.int64)
    if len(ids) != 5000:
        raise ValueError("Expected official 5000-image COCO2017 validation set")
    ids = np.random.Generator(np.random.PCG64(20261008)).permutation(ids).tolist()
    index = {x["id"]: x for x in annotation["images"]}
    missing = [i for i in ids if not (images / index[i]["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} COCO images")
    split = {"protocol_version": 2, "split_seed": 20261008, "rng": "numpy.Generator(PCG64)",
             "calibration": ids[:1000], "held_out": ids[1000:],
             "annotation_sha256": sha256(annotations),
             "image_bytes_sha256": object_hash({str(i): sha256(images / index[i]["file_name"]) for i in sorted(ids)}),
             "sources": SOURCES}
    split["split_sha256"] = object_hash({"calibration": split["calibration"], "held_out": split["held_out"]})
    write_json("artifacts/data_v2.json", split)
    return split


class Corpus:
    def __init__(self, root="data/coco", split_path="artifacts/data_v2.json", subset="calibration"):
        from .runs import read_json
        self.root = Path(root)
        self.split = read_json(split_path)
        self.ids = self.split[subset]
        self.annotation_path = self.root / "annotations/instances_val2017.json"
        annotation = json.loads(self.annotation_path.read_text())
        self.images = {x["id"]: x for x in annotation["images"]}
        self.categories = {x["id"]: x["name"] for x in annotation["categories"]}
        self.encoded = {i: (self.root / "val2017" / self.images[i]["file_name"]).read_bytes() for i in self.ids}

    def image(self, image_id):
        return Image.open(io.BytesIO(self.encoded[image_id])).convert("RGB")
