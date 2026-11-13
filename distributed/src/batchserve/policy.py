"""Causal, pure scheduling decisions; times are driver-clock seconds."""
from dataclasses import dataclass
import math

POLICIES = ("RR", "WRR", "SECT", "GLOBAL_GPU", "GLOBAL_E2E")


@dataclass
class Item:
    request_id: int
    image_id: int
    arrival: float
    deadline: float
    measurement: bool


class VirtualWork:
    """Release completed reservations, so conservative estimates cannot accumulate."""
    def __init__(self):
        self.pending = {}
        self.until = 0.

    def reserve(self, token, service, now):
        if token in self.pending or service <= 0:
            raise ValueError("Invalid or duplicate work reservation")
        self.pending[token] = service
        self.until = max(now, self.until) + service

    def complete(self, token, now):
        if token not in self.pending:
            raise ValueError("Completed work was not reserved")
        del self.pending[token]
        # Full service of the still-outstanding work is a conservative bound.
        # In particular it never includes already-completed requests.
        self.until = now + sum(self.pending.values())


def smooth_weighted_choice(weights, credits):
    total = sum(weights)
    for j, weight in enumerate(weights):
        credits[j] += weight
    j = max(range(len(weights)), key=lambda k: (credits[k], -k))
    credits[j] -= total
    return j


def global_decision(queue, now, available, profiles, kind, buckets, wait_s):
    """No future input: choose best feasible EDF prefix, bound absolute waiting.

    A worker can hold at most two batches (one running and one prefetched).
    profiles[j][kind][b] is a past-calibrated batch service p95 in seconds.
    available entries are (worker index, predicted ready time).
    """
    if not queue or not available:
        return None
    ordered = sorted(queue, key=lambda r: (r.deadline, r.arrival, r.request_id))
    choices = []
    nmax = min(len(ordered), max(buckets))
    for j, ready_at in available:
        for n in range(1, nmax + 1):
            b = next(b for b in buckets if b >= n)
            service = profiles[j][kind][str(b)]
            finish = max(now, ready_at) + service
            if finish <= ordered[0].deadline:
                choices.append((n / service, -finish, -b, -j, n, b, j))
    if choices:
        _, _, _, _, n, b, j = max(choices)
    else:
        # All methods continue serving late requests, never cherry-pick easy work.
        j, _ = min(available, key=lambda p: max(now, p[1]) + profiles[p[0]][kind]["1"])
        n, b = 1, 1
    # A bounded FIFO-age timer cannot be reset by subsequent arrivals.
    wait_end = min(r.arrival for r in queue) + wait_s
    if n < max(buckets) and now < wait_end:
        next_bucket = next((z for z in buckets if z > n), None)
        if next_bucket:
            future_ok = [ordered[0].deadline - profiles[k][kind][str(next_bucket)]
                         for k, ready_at in available
                         if max(now, ready_at) + profiles[k][kind][str(next_bucket)] <= ordered[0].deadline]
            if future_ok:
                wake = min(wait_end, max(future_ok))
                if wake > now:
                    return {"wake": wake}
    return {"worker": j, "bucket": b, "items": ordered[:n]}
