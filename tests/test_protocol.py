import pytest
from cvpr_project.trace import make_trace, trace_hash
from cvpr_project.scheduler import Ready, Scheduler, remaining_service_ns
from cvpr_project.metrics import summarize, censor_completions_at_cap
from cvpr_project.intervals import overlapping_duration


def req(rid, deadline, ready=0):
    return Ready(rid, 100 + rid, rid, deadline, ready)


def test_open_loop_deterministic_and_policy_independent():
    for kind in ["periodic", "poisson", "burst"]:
        a = make_trace(kind, 100, 42, list(range(50)), .01)
        b = make_trace(kind, 100, 42, list(range(50)), .01)
        assert trace_hash(a) == trace_hash(b)
        assert len({r.request_id for r in a}) == len(a)
        assert all(x.arrival_s < y.arrival_s for x, y in zip(a, a[1:]))
        assert all(r.relative_deadline_s in [.02, .04, .08] for r in a)


def test_edf_and_impossible_deadline_are_served():
    s = Scheduler("D0", service_ns={1: 100, 2: 150, 4: 250, 8: 400})
    assert s.decide([req(0, 1000), req(1, 300)], 0).request_ids == (1, 0)
    assert s.decide([req(0, 1), req(1, 2)], 1000).request_ids == (0,)


def test_fixed_wait_cannot_reset_timer():
    s = Scheduler("F0", wait_ms=2)
    assert s.decide([req(0, 10000000)], 0).wakeup_ns == 2000000
    d = s.decide([req(0, 10000000), req(1, 10000000, ready=1500000)], 2000000)
    assert d.wakeup_ns is None and d.request_ids == (0, 1)


def test_deadline_wait_is_bounded_by_service_and_oldest_ready():
    s = Scheduler("D0", wait_ms=2, service_ns={1: 1000000, 2: 2000000, 4: 3000000, 8: 4000000})
    d = s.decide([req(0, 2500000)], 0)
    assert d.wakeup_ns == 500000
    assert s.decide([req(0, 2500000)], 500000).request_ids == (0,)


def test_queued_gpu_work_does_not_expire_before_its_cuda_start():
    # Dispatch was 20 ms ago, but this 10 ms batch is still behind another
    # batch. A dispatch-age estimate would incorrectly claim zero remaining.
    assert remaining_service_ns(10000000, None, 20000000) == 10000000
    assert remaining_service_ns(10000000, 20000000, 23000000) == 7000000
    assert remaining_service_ns(10000000, 20000000, 40000000) == 0
    s = Scheduler("D0", service_ns={1: 1000000, 2: 1500000, 4: 3000000, 8: 6000000})
    d = s.decide([req(0, 25000000), req(1, 25000000)], 20000000,
                 remaining_service_ns(10000000, None, 20000000))
    assert d.request_ids == (0,)  # infeasible jobs are served, not dropped


def test_all_offered_requests_are_slo_denominator():
    rows = [dict(measurement=True, status="completed", arrival_ns=0, deadline_ns=10,
                 enqueue_ns=0, complete_ns=5, identity_ok=True),
            dict(measurement=True, status="completed", arrival_ns=0, deadline_ns=10,
                 enqueue_ns=0, complete_ns=20, identity_ok=True)]
    rows += [dict(measurement=True, status=s, arrival_ns=0, enqueue_ns=0) for s in
             ["rejected", "failed", "unfinished"]]
    m = summarize(rows, 60, 90, .01)
    assert m["offered"] == 5 and m["slo_success"] == .2
    assert m["goodput_rps"] == 1 / 60 and m["completion_ratio"] == .4
    assert m["cohort_throughput_rps_including_drain"] == 2 / 90


def test_profile_overlap_uses_union_and_separate_copy_durations():
    # Concurrent/nested kernels must not make a copy's overlap exceed its length.
    kernels = [(8, 10), (1, 5), (2, 4), (4, 7)]
    copies = [(0, 3), (6, 9), (10, 12)]
    assert overlapping_duration(kernels, copies) == 4
    assert overlapping_duration([], copies) == 0
    assert overlapping_duration(kernels, [(3, 4), (3, 4)]) == 2


def test_cpu_result_after_drain_cap_is_censored_even_if_gpu_copy_finished():
    rows = [dict(measurement=True, status="completed", arrival_ns=0, deadline_ns=100,
                 enqueue_ns=0, complete_ns=t, identity_ok=True) for t in [89, 90, 91]]
    censor_completions_at_cap(rows, 90)
    m = summarize(rows, 60, 90, .01)
    assert [r["status"] for r in rows] == ["completed", "completed", "unfinished"]
    assert m["offered"] == 3 and m["ontime"] == 2 and m["unfinished"] == 1
    assert m["slo_success"] == 2 / 3 and m["completion_ratio"] == 2 / 3
