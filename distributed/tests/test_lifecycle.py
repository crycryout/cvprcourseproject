"""Regression checks for task-owned process and lease boundaries, without GPUs."""
import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
from batchserve.common import read_frame
from batchserve.arrivals import emitted_arrivals


def manager():
    p = Path(__file__).resolve().parents[1] / "scripts/manage_workers.py"
    spec = importlib.util.spec_from_file_location("project_manager_test", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_old_monitor_never_touches_a_new_lease(tmp_path, monkeypatch):
    m = manager()
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"lease_id": "new-lease"}))
    monkeypatch.setattr(m, "STATE", state)
    def forbidden(*args):
        raise AssertionError("An old monitor inspected a new lease")
    monkeypatch.setattr(m, "probe", forbidden)
    m.watch("old-lease")


def test_dead_child_is_not_a_live_worker():
    m = manager()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        assert m.alive(p.pid)
    finally:
        p.terminate()
        p.wait(timeout=5)
    assert not m.alive(p.pid)


def test_invalid_frame_size_is_rejected_before_body_allocation():
    async def check():
        reader = asyncio.StreamReader()
        reader.feed_data(b"\x00\x00\x00\x00")
        reader.feed_eof()
        with pytest.raises(ValueError, match="header length"):
            await read_frame(reader)
    asyncio.run(check())


def test_producer_emits_while_router_event_loop_is_blocked():
    async def check():
        source = emitted_arrivals([0, .002, .004, .006], time.monotonic() + .1)
        first = await source.__anext__()
        time.sleep(.06)  # Models synchronous JSON work in the router process.
        pickup_ns = time.monotonic_ns()
        rest = [row async for row in source]
        assert [first[0]] + [r[0] for r in rest] == [0, 1, 2, 3]
        assert all(r[1] < pickup_ns for r in rest)
    asyncio.run(check())
