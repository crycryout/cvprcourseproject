"""Deterministic synthetic open-loop arrivals; image data remain real COCO."""
from dataclasses import dataclass, asdict
import numpy as np
from .runs import object_hash


@dataclass(frozen=True)
class Arrival:
    request_id: int
    image_id: int
    arrival_s: float
    relative_deadline_s: float
    measurement: bool


def make_trace(kind, rate, seed, image_ids, s_base_s, warmup_s=10, measurement_s=60):
    if rate <= 0 or s_base_s <= 0 or not image_ids:
        raise ValueError("Positive rate, SLO reference, and nonempty corpus are required")
    rng = np.random.Generator(np.random.PCG64(seed))
    end = warmup_s + measurement_s
    times = []
    if kind == "periodic":
        times = np.arange(0, end, 1 / rate).tolist()
    elif kind == "poisson":
        t = 0.0
        while (t := t + rng.exponential(1 / rate)) < end:
            times.append(t)
    elif kind == "burst":
        for period in range(int(np.ceil(end))):
            t = float(period)
            while (t := t + rng.exponential(1 / (5 * rate))) < min(period + 0.2, end):
                times.append(t)
    else:
        raise ValueError(f"Unknown arrival type: {kind}")
    # Independently sample deadline classes; a separate RNG protects arrivals.
    order = np.random.Generator(np.random.PCG64(seed + 100000)).permutation(image_ids).tolist()
    deadlines = np.random.Generator(np.random.PCG64(seed + 200000)).choice(
        [2, 4, 8], len(times), p=[0.5, 0.3, 0.2])
    return [Arrival(i, int(order[i % len(order)]), float(t), float(d * s_base_s), t >= warmup_s)
            for i, (t, d) in enumerate(zip(times, deadlines))]


def trace_hash(trace):
    return object_hash([asdict(x) for x in trace])
