"""Small immutable evidence files and bounded, non-pickle TCP framing."""
import asyncio
import hashlib
import json
import struct
from pathlib import Path

MAX_HEADER = 16 * 1024 * 1024
MAX_PAYLOAD = 64 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def source_hashes(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)): file_hash(p) for p in sorted((root / "src").rglob("*.py"))}


async def send_frame(writer, header, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Payload too large")
    header = dict(header, payload_bytes=len(payload))
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_HEADER:
        raise ValueError("Header too large")
    writer.write(struct.pack("!I", len(encoded)))
    writer.write(encoded)
    if payload:
        writer.write(payload)
    await writer.drain()


async def read_frame(reader):
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    if not 0 < size <= MAX_HEADER:
        raise ValueError("Invalid header length")
    header = json.loads(await reader.readexactly(size))
    payload_size = header.pop("payload_bytes", 0)
    if not isinstance(payload_size, int) or not 0 <= payload_size <= MAX_PAYLOAD:
        raise ValueError("Invalid payload length")
    return header, await reader.readexactly(payload_size)


class Connection:
    """One persistent connection; responses can complete out of request order."""
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.pending, self.counter = {}, 0
        self.lock = asyncio.Lock()
        self.task = asyncio.create_task(self.receive())

    @classmethod
    async def open(cls, host, port):
        return cls(*(await asyncio.open_connection(host, port)))

    async def receive(self):
        try:
            while True:
                header, payload = await read_frame(self.reader)
                future = self.pending.pop(header["message_id"], None)
                if future and not future.done():
                    if header.get("error"):
                        future.set_exception(RuntimeError(header["error"]))
                    else:
                        future.set_result((header, payload))
        except (Exception, asyncio.CancelledError) as exc:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ConnectionError(type(exc).__name__))
            self.pending.clear()

    async def request(self, header, payload=b"", timeout=120):
        async with self.lock:
            self.counter += 1
            mid = self.counter
            future = asyncio.get_running_loop().create_future()
            self.pending[mid] = future
            try:
                await send_frame(self.writer, dict(header, message_id=mid), payload)
            except BaseException:
                self.pending.pop(mid, None)
                future.cancel()
                raise
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self.pending.pop(mid, None)

    async def close(self):
        self.writer.close()
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        try:
            await self.writer.wait_closed()
        except ConnectionError:
            pass
