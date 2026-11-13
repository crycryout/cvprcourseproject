"""An independent Linux process emits scheduled arrivals through a bounded pipe.

The router can be busy decoding results without delaying the emitter's clock.
Nonblocking atomic writes account for a full pipe as generator lag, rather than
hiding producer backpressure. Notification pickup delay remains part of E2E.
"""
import asyncio
import multiprocessing
import os
import select
import struct
import time

FRAME = struct.Struct("!QQ")


def _emit(write_fd, read_fd, offsets_ns, start_ns):
    os.close(read_fd)
    os.set_blocking(write_fd, False)
    if hasattr(os, "sched_getaffinity"):
        available = os.sched_getaffinity(0)
        if available:
            os.sched_setaffinity(0, {max(available)})
    try:
        for rid, offset in enumerate(offsets_ns):
            target = start_ns + offset
            delay = (target - time.monotonic_ns()) / 1e9
            if delay > 0:
                time.sleep(delay)
            while True:
                stamp = time.monotonic_ns()
                try:
                    written = os.write(write_fd, FRAME.pack(rid, stamp))
                    if written != FRAME.size:
                        raise RuntimeError("Arrival notification was not atomic")
                    break
                except BlockingIOError:
                    select.select([], [write_fd], [], .01)
                except BrokenPipeError:
                    return
    finally:
        os.close(write_fd)


async def emitted_arrivals(offsets_s, start_s):
    """Yield (request index, actual producer emission ns), never future events."""
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("The measured harness requires Linux/POSIX fork")
    read_fd, write_fd = os.pipe()
    child = multiprocessing.get_context("fork").Process(
        target=_emit, args=(write_fd, read_fd, [round(x * 1e9) for x in offsets_s], round(start_s * 1e9)),
        daemon=True)
    child.start()
    os.close(write_fd)
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: protocol, os.fdopen(read_fd, "rb", buffering=0))
    try:
        for expected in range(len(offsets_s)):
            rid, emitted_ns = FRAME.unpack(await reader.readexactly(FRAME.size))
            if rid != expected:
                raise RuntimeError("Arrival notification identity/order violated")
            yield rid, emitted_ns
    finally:
        transport.close()
        child.join(timeout=.2)
        if child.is_alive():
            child.terminate()
            child.join(timeout=1)
