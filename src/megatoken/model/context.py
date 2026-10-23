"""Public track-local context policy, independent of model parameter ownership.

Local age counts events on that track. Foreign age counts global committed
events, so unrelated traffic cannot evict local history or multiply foreign reads.
Windows control direct reads, not information already summarized in a state.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

# Provisional operating defaults, shared by new models and the data curriculum.
DEFAULT_LOCAL_EVENTS = 256
DEFAULT_CROSS_EVENTS = 16


@dataclass(frozen=True)
class ContextPolicy:
    local_events: int = DEFAULT_LOCAL_EVENTS
    cross_events: int = DEFAULT_CROSS_EVENTS
    # Overrides also allow retirement: (track, 0, 0) stops retaining its local KV.
    windows: tuple[tuple[str, int, int], ...] = ()
    pins: tuple[tuple[str, str], ...] = ()  # (reader track, retained event address)
    max_pins: int = 8
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "windows", tuple(tuple(row) for row in self.windows))
        object.__setattr__(self, "pins", tuple(tuple(row) for row in self.pins))
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unknown context policy version")
        for value in (self.local_events, self.cross_events, self.max_pins):
            if type(value) is not int or value < 0:
                raise ValueError("Context windows and pin limits must be nonnegative integers")
        if len({row[0] for row in self.windows}) != len(self.windows):
            raise ValueError("Context windows require distinct track addresses")
        for address, local, cross in self.windows:
            if not isinstance(address, str) or not address:
                raise ValueError("Context override needs a track address")
            if any(type(value) is not int or value < 0 for value in (local, cross)):
                raise ValueError("Context windows must be nonnegative integers")
        if len(set(self.pins)) != len(self.pins) or any(
            not isinstance(reader, str) or not reader or not isinstance(event, str) or not event
            for reader, event in self.pins
        ):
            raise ValueError("Context pins require distinct reader/event addresses")
        for reader, _ in self.pins:
            if sum(track == reader for track, _ in self.pins) > self.max_pins:
                raise ValueError("Context pin allowance exhausted")

    def window(self, track):
        return next(
            ((local, cross) for address, local, cross in self.windows if address == track),
            (self.local_events, self.cross_events),
        )

    def configure(self, track, *, local_events, cross_events, pinned_events=()):
        return replace(
            self,
            windows=tuple(row for row in self.windows if row[0] != track)
            + ((track, local_events, cross_events),),
            pins=tuple(row for row in self.pins if row[0] != track)
            + tuple((track, event) for event in pinned_events),
        )

    def visible(self, reader, tracks, positions, event_ids, seen):
        local, cross = self.window(reader)
        own = [i for i, track in enumerate(tracks) if track == reader]
        own = set(own[-local:]) if local else set()
        pins = {event for track, event in self.pins if track == reader}
        return tuple(
            i in own or (track != reader and position >= seen - cross) or event in pins
            for i, (track, position, event) in enumerate(
                zip(tracks, positions, event_ids, strict=True)
            )
        )

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value):
        row = json.loads(value)
        if not isinstance(row, dict):
            raise ValueError("Context policy must be an object")
        row["windows"] = tuple(tuple(item) for item in row.get("windows", ()))
        row["pins"] = tuple(tuple(item) for item in row.get("pins", ()))
        return cls(**row)


DEFAULT_CONTEXT_POLICY = ContextPolicy().to_json()
