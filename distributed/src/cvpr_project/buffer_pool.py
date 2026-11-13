"""Each slot owns its captured input/output addresses until CPU consumption."""
from dataclasses import dataclass, field
import torch


@dataclass
class Slot:
    bucket: int
    slot_id: int
    dtype: object
    device: object
    pinned: bool = True
    state: str = "FREE"
    graph: object = None
    graph_outputs: object = None
    requests: list = field(default_factory=list)
    dispatch_ns: int = 0
    forward_observed_start_ns: int | None = None

    def __post_init__(self):
        b = self.bucket
        self.host_values = torch.empty((b, 3, 640, 640), dtype=self.dtype, pin_memory=self.pinned)
        self.host_mask = torch.empty((b, 640, 640), dtype=torch.int64, pin_memory=self.pinned)
        self.values = torch.zeros_like(self.host_values, device=self.device)
        self.mask = torch.ones_like(self.host_mask, device=self.device)
        self.host_logits = torch.empty((b, 100, 92), dtype=self.dtype, pin_memory=self.pinned)
        self.host_boxes = torch.empty((b, 100, 4), dtype=self.dtype, pin_memory=self.pinned)
        self.events = {name: torch.cuda.Event(enable_timing=True) for name in
                       ["h2d_start", "h2d_end", "forward_start", "forward_end", "d2h_start", "done"]}

    def transition(self, expected, next_state):
        if self.state != expected:
            raise RuntimeError(f"Slot {self.bucket}/{self.slot_id}: expected {expected}, got {self.state}")
        self.state = next_state

    def release(self):
        self.transition("CPU_CONSUMING", "FREE")
        self.requests = []

    @property
    def host_bytes(self):
        return sum(x.numel() * x.element_size() for x in
                   [self.host_values, self.host_mask, self.host_logits, self.host_boxes])
