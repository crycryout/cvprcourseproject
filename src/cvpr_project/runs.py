"""Atomic evidence, source provenance, and cumulative resource accounting."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
import uuid
import re

ARTIFACTS = Path("artifacts")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def sanitize_public(value):
    if isinstance(value, dict):
        return {k: sanitize_public(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_public(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"/(?:home|tmp|root)/[^\s\"']+", "<local-path>", value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def run_id(prefix):
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
    packages = {}
    for name in ["torch", "torchvision", "transformers", "numpy", "pillow", "pycocotools"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"git_sha": git("rev-parse", "HEAD").decode().strip(),
            "dirty_diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
            "source_files_sha256": {str(p): sha256(p) for p in sorted(Path("src").rglob("*.py"))},
            "packages": packages}


def gpu_snapshot():
    fields = "index,name,memory.used,utilization.gpu,temperature.gpu,power.draw"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10)
    return {"fields": fields.split(","), "rows": result.stdout.strip().splitlines()}


def other_gpu_pids():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    def own_process(pid):
        visited = set()
        while pid > 1 and pid not in visited:
            if pid == os.getpid():
                return True
            visited.add(pid)
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
                pid = int(stat[stat.rfind(")") + 2:].split()[1])
            except (OSError, ValueError, IndexError):
                return False
        return False
    others = [int(x) for x in result.stdout.splitlines() if x.strip().isdigit() and not own_process(int(x))]
    if others:
        # Local diagnostics distinguish an actual foreign job from a stale
        # driver PID or thread identifier without changing the exclusion rule.
        processes = []
        for pid in others:
            detail = {"pid": pid}
            try:
                status = Path(f"/proc/{pid}/status").read_text()
                detail.update({key: next((line.split(":", 1)[1].strip() for line in status.splitlines()
                                         if line.startswith(key + ":")), None) for key in ["Name", "Tgid", "PPid"]})
                detail["cwd"] = str(Path(f"/proc/{pid}/cwd").resolve())
            except OSError:
                detail["process_no_longer_present"] = True
            processes.append(detail)
        try:
            write_json(ARTIFACTS / "gpu_interference_checks" / f"probe-{time.time_ns()}.json",
                       {"observer_pid": os.getpid(), "host_ns": time.monotonic_ns(), "processes": processes})
        except OSError:
            pass
    return others


def assert_no_other_gpu_jobs():
    others = other_gpu_pids()
    if others:
        raise RuntimeError(f"Formal timing blocked by {len(others)} other GPU process(es); retry when idle.")


def resource_totals():
    ledger = ARTIFACTS / "cost_ledger.jsonl"
    entries = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    # Count unfinished GPU leases conservatively through the present time.
    active = ARTIFACTS / "active_gpu_job.json"
    seconds = sum(e["elapsed_seconds"] for e in entries)
    metered_seconds = sum(e["elapsed_seconds"] for e in entries if e.get("measured", True))
    if active.exists():
        active_seconds = max(0, time.time() - read_json(active)["start_unix"])
        seconds += active_seconds
        metered_seconds += active_seconds
    storage = sum(p.stat().st_size for p in Path(".").rglob("*") if p.is_file() and ".git" not in p.parts)
    return {"gpu_hours": seconds / 3600, "metered_gpu_hours": metered_seconds / 3600,
            "initial_unmetered_setup_reserve_hours": (seconds - metered_seconds) / 3600,
            "project_storage_gb": storage / 1e9}


@contextmanager
def gpu_lease(stage, cap_hours=30, cap_gb=30):
    ARTIFACTS.mkdir(exist_ok=True)
    active = ARTIFACTS / "active_gpu_job.json"
    if active.exists():
        record = read_json(active)
        try:
            os.kill(record["pid"], 0)
        except ProcessLookupError:
            # Preserve a interrupted lease rather than erase its cost.
            with (ARTIFACTS / "cost_ledger.jsonl").open("a") as f:
                f.write(json.dumps({**record, "status": "interrupted", "elapsed_seconds":
                                   max(0, time.time() - record["start_unix"])}) + "\n")
            active.unlink()
        else:
            raise RuntimeError("Another project GPU lease is active; inspect it before resuming.")
    cost = resource_totals()
    if cost["gpu_hours"] >= cap_hours or cost["project_storage_gb"] >= cap_gb:
        raise RuntimeError(f"Resource cap reached: {cost}")
    entry = {"protocol_version": 2, "stage": stage, "pid": os.getpid(), "start_unix": time.time(),
             "started_utc": stamp()}
    write_json(active, entry)
    status = "completed"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        entry.update(status=status, elapsed_seconds=time.time() - entry["start_unix"], ended_utc=stamp())
        with (ARTIFACTS / "cost_ledger.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        active.unlink(missing_ok=True)
