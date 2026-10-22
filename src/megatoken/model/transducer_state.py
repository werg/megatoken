"""Safe tensor-only snapshots of the causal v10 multitrack history.

Reconstruct KBContext from the current model rather than deserializing legacy
point/pair memory objects from a pickled checkpoint.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .multitrack import MultiTrackCore, MultiTrackState
from .state import KVHistory

CAUSAL_STATE_IDENTITY = "v10-native-causal-session"


def snapshot_causal_state(state: MultiTrackState) -> dict[str, Any]:
    """Save all native history and frontier tensors for weights-only restoration."""
    if state.kb is None or state.kb.groups:
        raise ValueError("A v10 causal snapshot cannot contain v9 point/pair memory")
    def cpu(value):
        return None if value is None else value.detach().cpu().clone()

    return {
        "identity": CAUSAL_STATE_IDENTITY,
        "raw": cpu(state.raw),
        "final": cpu(state.final),
        "history_keys": cpu(state.history.keys),
        "history_values": cpu(state.history.values),
        "history_mask": cpu(state.history.mask),
        "history_tracks": cpu(state.history_tracks),
        "meridian_sizes": list(state.meridian_sizes),
        "meridians_seen": cpu(state.meridians_seen),
        "events_seen": cpu(state.events_seen),
        "frontier_tracks": list(state.frontier_tracks),
        "frontier_modalities": list(state.frontier_modalities),
        "track_registry": list(state.track_registry),
        "frontier_roles": cpu(state.frontier_roles),
        "model_version": state.model_version,
        "routing_version": state.routing_version,
        "geometry": list(state.geometry),
    }


def restore_causal_state(tracks: MultiTrackCore, data: dict[str, Any]) -> MultiTrackState:
    """Reject incompatible native models and reconstruct a validated state."""
    if data.get("identity") != CAUSAL_STATE_IDENTITY:
        raise ValueError("Only v10 native causal histories can be restored")
    core = tracks.core
    if (
        data["model_version"] != core.model_version
        or data["routing_version"] != core.config.routing_version
        or tuple(data["geometry"]) != core.config.state_geometry
    ):
        raise ValueError("Saved causal history belongs to another model or routing geometry")
    device = core.initial_facets.device
    base = tracks.initial_state(data["raw"].shape[0], device=device)
    state = replace(
        base,
        raw=data["raw"].to(device),
        final=None if data["final"] is None else data["final"].to(device),
        history=KVHistory(
            data["history_keys"].to(device),
            data["history_values"].to(device),
            None if data["history_mask"] is None else data["history_mask"].to(device),
        ),
        history_tracks=data["history_tracks"].to(device),
        meridian_sizes=tuple(map(int, data["meridian_sizes"])),
        meridians_seen=data["meridians_seen"].to(device),
        events_seen=data["events_seen"].to(device),
        frontier_tracks=tuple(data["frontier_tracks"]),
        frontier_modalities=tuple(data["frontier_modalities"]),
        track_registry=tuple(data["track_registry"]),
        frontier_roles=data["frontier_roles"].to(device),
    )
    tracks._validate(state)
    return state
