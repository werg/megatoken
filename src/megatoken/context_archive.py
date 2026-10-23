"""Bounded host-side source copies for explicit context rereads.

These are original available inputs and completed text outputs, never targets or
copies of every model cache. Rereads are new causal observations, not restoration
of an old model state. Durable source collections remain independently usable.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import torch


class ContextArchive:
    def __init__(self, *, max_bytes=64 << 20, max_events=1024):
        if (
            type(max_bytes) is not int
            or max_bytes < 0
            or type(max_events) is not int
            or max_events < 0
        ):
            raise ValueError("Context archive limits must be nonnegative integers")
        self.max_bytes, self.max_events = max_bytes, max_events
        self.events = OrderedDict()
        self.payload_bytes = 0

    def put(self, event):
        size = (
            len(event.payload)
            if isinstance(event.payload, bytes)
            else event.payload.numel() * event.payload.element_size()
        )
        if event.valid_mask is not None:
            size += event.valid_mask.numel() * event.valid_mask.element_size()
        if size > self.max_bytes or not self.max_events:
            return False
        if event.event_id in self.events:
            return True
        while self.events and (
            self.payload_bytes + size > self.max_bytes or len(self.events) >= self.max_events
        ):
            _, (_, old_size) = self.events.popitem(last=False)
            self.payload_bytes -= old_size
        if isinstance(event.payload, torch.Tensor):
            event = replace(
                event,
                payload=event.payload.detach().to("cpu").clone(),
                valid_mask=None
                if event.valid_mask is None
                else event.valid_mask.detach().to("cpu").clone(),
            )
        self.events[event.event_id] = (event, size)
        self.payload_bytes += size
        return True

    def get(self, event_id):
        if event_id not in self.events:
            return None
        self.events.move_to_end(event_id)
        return self.events[event_id][0]
