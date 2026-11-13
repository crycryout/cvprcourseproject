#!/usr/bin/env python3
"""Start/inspect/stop only project-owned workers; record full GPU reservation time."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from batchserve.common import Connection, write_json

STATE = ROOT / "artifacts/workers.local.json"
SESSION_OPTIONS = ["-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "ControlPersist=no"]


def command(args, remote=None):
    if remote:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", remote, shlex.join(args)]
    return subprocess.check_output(args, text=True, timeout=15).strip()


def probe(remote=None):
    apps = command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], remote)
    gpus = command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], remote)
    return {"gpus": gpus, "apps": apps}


def alive(pid):
    try:
        os.kill(pid, 0)
        status = Path(f"/proc/{pid}/stat").read_text()
        if status.rsplit(")", 1)[1].strip().startswith("Z"):
            return False
        return True
    except (ProcessLookupError, FileNotFoundError):
        return False


def stop(state, reason="requested project cleanup"):
    with (ROOT / "artifacts/worker_stop.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if STATE.exists():
            recorded = json.loads(STATE.read_text())
            if recorded["lease_id"] == state["lease_id"] and recorded.get("stopped_unix_s"):
                state.update(recorded)
                return
        _stop(state, reason)


def _stop(state, reason):
    if state.get("stopped_unix_s"):
        return
    # Inspect /proc before signaling: PID reuse must not target another task.
    for item in state["processes"]:
        pid = item["pid"]
        path = Path(f"/proc/{pid}/cmdline")
        if path.exists():
            try:
                cmd = path.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            except FileNotFoundError:
                continue
            if not cmd.strip():
                continue  # Already-exited zombie has no command line.
            if "batchserve.worker" not in cmd and ("ssh" not in cmd or "18732" not in cmd):
                raise RuntimeError(f"PID ownership mismatch; refused to signal {pid}")
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    # Explicitly terminate the remote process only after verifying its command.
    remote_pid = state.get("remote_worker_pid")
    if remote_pid:
        script = ("import os,signal,pathlib; p=" + str(remote_pid) + "; "
                  "f=pathlib.Path('/proc')/str(p)/'cmdline'; "
                  "c=f.read_bytes() if f.exists() else b''; "
                  "os.kill(p,signal.SIGTERM) if b'batchserve.worker' in c else None")
        try:
            command(["python3", "-c", script], state["remote_alias"])
        except (subprocess.SubprocessError, OSError):
            pass
    state["stopped_unix_s"] = time.time()
    state["stop_reason"] = reason
    state["aggregate_gpu_hours"] = sum((state["stopped_unix_s"] - x["started_unix_s"]) / 3600
                                        for x in state["processes"] if x["kind"] == "worker")
    write_json(STATE, state)
    ledger = ROOT / "artifacts/worker_leases.jsonl"
    with ledger.open("a") as f:
        f.write(json.dumps({k: state[k] for k in ["lease_id", "started_unix_s", "stopped_unix_s", "stop_reason", "aggregate_gpu_hours"]}) + "\n")


async def readiness(state):
    for _ in range(180):
        try:
            infos = []
            for port in [18731, 18732]:
                c = await Connection.open("127.0.0.1", port)
                response, _ = await c.request({"kind": "info"}, timeout=5)
                infos.append(response["info"])
                await c.close()
            return infos
        except (OSError, asyncio.TimeoutError, ConnectionError):
            if not all(alive(p["pid"]) for p in state["processes"]):
                raise RuntimeError("Worker/forward exited during startup; inspect task logs")
            await asyncio.sleep(1)
    raise RuntimeError("Worker readiness exceeded 180 seconds")


def start(args):
    if STATE.exists():
        old = json.loads(STATE.read_text())
        if not old.get("stopped_unix_s"):
            raise RuntimeError("Existing lease active; inspect or stop it first")
    preflight = {"local": probe(), "remote": probe(args.remote_alias)}
    if any(p["apps"] for p in preflight.values()):
        raise RuntimeError("A host has active GPU work; no worker started")
    uuid = command(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"])
    remote_uuid = command(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], args.remote_alias)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    logs = ROOT / "artifacts" / ("lease_" + stamp)
    logs.mkdir(parents=True, exist_ok=False)
    state = {"lease_id": stamp, "started_unix_s": time.time(), "remote_alias": args.remote_alias,
             "remote_root": args.remote_root, "preflight": preflight, "processes": []}
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES=uuid,
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (logs / "H800.log").open("w") as log:
        proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m", "batchserve.worker", "--port", "18731"],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["processes"].append({"pid": proc.pid, "kind": "worker", "started_unix_s": time.time(), "label": "H800"})
    remote_cmd = "cd " + shlex.quote(args.remote_root) + " && exec " + shlex.join([
        "env", "PYTHONPATH=src", "CUDA_VISIBLE_DEVICES=" + remote_uuid,
        "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        ".venv/bin/python", "-m", "batchserve.worker", "--port", "18732"])
    with (logs / "RTX5090.log").open("w") as log:
        proc = subprocess.Popen(["ssh", *SESSION_OPTIONS, args.remote_alias, remote_cmd],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["processes"].append({"pid": proc.pid, "kind": "worker", "started_unix_s": time.time(), "label": "RTX5090"})
    with (logs / "forward.log").open("w") as log:
        proc = subprocess.Popen(["ssh", "-N", *SESSION_OPTIONS, "-o", "ExitOnForwardFailure=yes",
                                "-o", "ServerAliveInterval=15", "-L", "18732:127.0.0.1:18732", args.remote_alias],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["processes"].append({"pid": proc.pid, "kind": "forward", "started_unix_s": time.time()})
    write_json(STATE, state)
    try:
        info = asyncio.run(readiness(state))
        remote_pids = command(["pgrep", "-f", "^.venv/bin/python -m batchserve.worker --port 18732$"], args.remote_alias).splitlines()
        if len(remote_pids) != 1:
            raise RuntimeError("Remote worker PID is ambiguous")
        state["remote_worker_pid"] = int(remote_pids[0])
        state["local_worker_pid"] = state["processes"][0]["pid"]
        write_json(STATE, state)
        write_json(ROOT / "configs/hosts.local.json", [{"label": "H800", "host": "127.0.0.1", "port": 18731},
                                                      {"label": "RTX5090", "host": "127.0.0.1", "port": 18732}])
        write_json(logs / "worker_info.json", info)
        state["watch_pid"] = os.getpid()
        write_json(STATE, state)
        print(json.dumps({"status": "ready", "lease": stamp, "workers": [i["hardware"]["name"] for i in info]}), flush=True)
        # Remain the foreground owner. Execution environments may reap background
        # descendants after a launcher exits, even when workers use new sessions.
        watch(state["lease_id"])
    except BaseException:
        stop(state, "startup failure")
        raise


def watch(expected_lease=None):
    if expected_lease is None:
        expected_lease = json.loads(STATE.read_text())["lease_id"]
    while True:
        state = json.loads(STATE.read_text())
        if state["lease_id"] != expected_lease:
            return
        if state.get("stopped_unix_s"):
            return
        reason = None
        try:
            for remote, permitted in [(None, state["local_worker_pid"]), (state["remote_alias"], state["remote_worker_pid"])]:
                result = probe(remote)
                pids = [int(line.split(",")[0]) for line in result["apps"].splitlines() if line.strip()]
                if any(pid != permitted for pid in pids):
                    reason = "foreign GPU process detected; only project workers stopped"
            if not all(alive(p["pid"]) for p in state["processes"]):
                reason = "project process exited"
            previous = 0.
            ledger = ROOT / "artifacts/worker_leases.jsonl"
            if ledger.exists():
                previous = sum(json.loads(line)["aggregate_gpu_hours"] for line in ledger.read_text().splitlines())
            if previous + 2 * (time.time() - state["started_unix_s"]) / 3600 >= 16:
                reason = "16 aggregate GPU-hour extension cap reached"
        except Exception as exc:
            reason = "GPU isolation monitor failed: " + type(exc).__name__
        if reason:
            stop(state, reason)
            print(reason, flush=True)
            return
        time.sleep(5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "stop", "status", "watch"])
    p.add_argument("--remote-alias")
    p.add_argument("--remote-root", default="CISC8003Project")
    args = p.parse_args()
    if args.action == "start":
        if not args.remote_alias:
            p.error("start requires --remote-alias")
        start(args)
    elif args.action == "watch":
        watch()
    elif args.action == "stop":
        state = json.loads(STATE.read_text())
        stop(state)
        print(json.dumps({"status": "stopped", "aggregate_gpu_hours": state.get("aggregate_gpu_hours")}))
    else:
        state = json.loads(STATE.read_text())
        print(json.dumps({"lease": state["lease_id"], "stopped": bool(state.get("stopped_unix_s")),
                          "elapsed_aggregate_gpu_hours": state.get("aggregate_gpu_hours", 2 * (time.time() - state["started_unix_s"]) / 3600)}))


if __name__ == "__main__":
    main()
