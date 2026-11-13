"""Exact interval unions for correlated profiler clocks, independent of CUDA."""
from bisect import bisect_right


def overlapping_duration(intervals, queries):
    """Sum each query's overlap with the union, never count two kernels twice."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    ends = [end for _, end in merged]
    result = 0.0
    for start, end in queries:
        index = bisect_right(ends, start)
        while index < len(merged) and merged[index][0] < end:
            a, b = merged[index]
            result += max(0, min(b, end) - max(a, start))
            index += 1
    return result
