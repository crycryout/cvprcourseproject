import asyncio
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest

from batchserve.common import send_frame, read_frame, Connection, digest
from batchserve.policy import Item, VirtualWork, global_decision, smooth_weighted_choice
from batchserve.benchmark import summarize, make_trace, run_case


def test_binary_framing_and_out_of_order_rpc():
    async def exercise():
        async def server(reader, writer):
            a, ap = await read_frame(reader)
            b, bp = await read_frame(reader)
            await send_frame(writer, dict(b, echoed=True), bp)
            await send_frame(writer, dict(a, echoed=True), ap)
        listener = await asyncio.start_server(server, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]
        c = await Connection.open("127.0.0.1", port)
        a, b = await asyncio.gather(c.request({"test": 1}, b"\x00jpeg\xff"), c.request({"test": 2}, b"other"))
        assert a[1] == b"\x00jpeg\xff" and b[1] == b"other"
        await c.close()
        listener.close()
        await listener.wait_closed()
    asyncio.run(exercise())


def test_weighted_routing_preserves_capacity_ratio():
    weights, credits, counts = [1., 3.], [0., 0.], [0, 0]
    for _ in range(400):
        counts[smooth_weighted_choice(weights, credits)] += 1
    assert counts == [100, 300]


def test_completed_conservative_work_does_not_create_phantom_backlog():
    work = VirtualWork()
    work.reserve(0, .1, 0.)
    work.reserve(1, .1, .001)
    now = .02
    # Every completion occurs earlier than the conservative service estimate.
    # Always retain one active reservation, reproducing the former drift trigger.
    for token in range(500):
        work.complete(token, now)
        work.reserve(token + 2, .1, now)
        assert len(work.pending) == 2
        assert work.until <= now + .2 + 1e-10
        now += .02
    work.complete(500, now)
    work.complete(501, now)
    assert work.until == now and not work.pending


def test_deadline_prefix_and_no_future_or_expired_drop():
    p = [{"e2e_s": {"1": .01, "2": .015, "4": .02, "8": .03}}]
    q = [Item(1, 1, 0, .001, True), Item(2, 2, 0, .2, True)]
    d = global_decision(q, .02, [(0, .02)], p, "e2e_s", [1, 2, 4, 8], 0)
    assert [i.request_id for i in d["items"]] == [1]
    assert len(q) == 2
    assert global_decision(q, .02, [], p, "e2e_s", [1, 2, 4, 8], 0) is None


def test_absolute_wait_timer_does_not_reset():
    p = [{"e2e_s": {"1": .001, "2": .002, "4": .003, "8": .004}}]
    q = [Item(0, 1, 0, .2, True)]
    d = global_decision(q, .004, [(0, .004)], p, "e2e_s", [1, 2, 4, 8], .01)
    assert d == {"wake": .01}
    q.append(Item(1, 2, .009, .209, True))
    d = global_decision(q, .011, [(0, .011)], p, "e2e_s", [1, 2, 4, 8], .01)
    assert "items" in d


def test_trace_is_paired_and_has_unsaturated_arrivals():
    a = make_trace("poisson", 100, 17, [1, 2, 3], [.1, .2, .4], 1, 2)
    b = make_trace("poisson", 100, 17, [1, 2, 3], [.1, .2, .4], 1, 2)
    assert a == b
    assert len({r.request_id for r in a}) == len(a)
    assert all(r.measurement == (r.arrival >= 1) for r in a)
    assert a != make_trace("poisson", 100, 42, [1, 2, 3], [.1, .2, .4], 1, 2)


def test_all_offered_denominator_and_unique_batches():
    def event(rid, status, complete=None, measurement=True):
        return dict(request_id=rid, status=status, complete_s=complete,
                    measurement=measurement, arrival=0, deadline=.1,
                    generator_lag_ms=0, worker=0,
                    worker_event={"batch_id": 2, "valid_count": 2})
    m = summarize([event(0, "completed", .05), event(1, "completed", .2),
                   event(2, "rejected"), event(3, "error"), event(4, "unfinished"),
                   event(5, "completed", .01, False)], 2)
    assert m["offered"] == 5 and m["completed"] == 2 and m["on_time"] == 1
    assert m["slo_success"] == .2 and m["goodput_rps"] == .5
    assert m["completed_batches"] == 1 and m["completion_ratio"] == .4


def test_loopback_harness_accounts_transport_error():
    class FakeImages:
        ids = [1, 2]
        hashes = {1: "hash", 2: "hash"}
        def pack(self, items):
            return [{"jpeg_bytes": 1} for _ in items], b"x" * len(items)
    class FakeConnection:
        async def request(self, header, payload=b"", timeout=120):
            if header["kind"] == "configure":
                return {}, b""
            raise ConnectionError("injected fixture")
    async def exercise():
        cfg = dict(local_settings=[{"max_batch": 2, "wait_ms": 0}],
                   deadlines_s=[.1, .2, .4], warmup_s=0, measurement_s=.05,
                   drain_s=.1, max_pending=512)
        profiles = [{"e2e_s": {"1": .001, "2": .002, "4": .003, "8": .005}}]
        m, rows = await run_case([FakeConnection()], FakeImages(), profiles, cfg,
                                "RR", "periodic", 100, 17)
        assert m["offered"] == 5 and m["status_counts"] == {"error": 5}
        assert m["slo_success"] == 0 and m["completion_ratio"] == 0
    asyncio.run(exercise())
