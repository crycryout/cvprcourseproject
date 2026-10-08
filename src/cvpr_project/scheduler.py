"""Pure FIFO/EDF scheduling without future arrival/completion information."""
from dataclasses import dataclass


def remaining_service_ns(service_ns, observed_start_ns, now_ns):
    """A queued batch retains its full estimate until GPU start is observed.

    First host observation is later than the real CUDA event, so subtracting
    from it is conservative and uses no future completion information.
    """
    if observed_start_ns is None:
        return service_ns
    return max(0, service_ns - max(0, now_ns - observed_start_ns))


@dataclass
class Ready:
    request_id: int
    image_id: int
    arrival_ns: int
    deadline_ns: int
    ready_ns: int
    payload: object = None


@dataclass(frozen=True)
class Decision:
    request_ids: tuple
    bucket: int
    wakeup_ns: int | None = None


class Scheduler:
    def __init__(self, policy, buckets=(1, 2, 4, 8), max_batch=8, wait_ms=0, service_ns=None):
        self.policy = policy
        self.buckets = tuple(b for b in buckets if b <= max_batch)
        self.max_batch = max(self.buckets)
        self.wait_ns = round(wait_ms * 1e6)
        self.service_ns = {int(k): int(v) for k, v in (service_ns or {}).items()}
        if policy == "D0" and not all(b in self.service_ns for b in self.buckets):
            raise ValueError("D0 requires a calibrated service table for every common bucket")

    def decide(self, queue, now_ns, gpu_remaining_ns=0):
        if not queue:
            return None
        edf = self.policy in {"D0", "A0"}
        ordered = sorted(queue, key=(lambda x: (x.deadline_ns, x.arrival_ns, x.request_id)) if edf
                         else (lambda x: (x.ready_ns, x.arrival_ns, x.request_id)))
        nmax = min(len(ordered), self.max_batch)
        bucket_for = lambda n: next(b for b in self.buckets if b >= n)
        # Absolute timer based on oldest ready request, never reset on arrival.
        wait_end = min(x.ready_ns for x in queue) + self.wait_ns
        if self.policy == "D0":
            feasible = []
            for n in range(1, nmax + 1):
                b = bucket_for(n)
                finish = now_ns + gpu_remaining_ns + self.service_ns[b]
                if finish <= ordered[0].deadline_ns:
                    feasible.append((n / self.service_ns[b], -finish, -b, n, b))
            if not feasible:
                return Decision((ordered[0].request_id,), 1)
            _, _, _, n, b = max(feasible)
            bigger = next((z for z in self.buckets if z > b), None)
            if bigger and len(ordered) < bigger and now_ns < wait_end:
                wake = min(wait_end, ordered[0].deadline_ns - gpu_remaining_ns - self.service_ns[bigger])
                if wake > now_ns:
                    return Decision((), b, wake)
            return Decision(tuple(x.request_id for x in ordered[:n]), b)
        if nmax < self.max_batch and now_ns < wait_end:
            return Decision((), bucket_for(nmax), wait_end)
        return Decision(tuple(x.request_id for x in ordered[:nmax]), bucket_for(nmax))
