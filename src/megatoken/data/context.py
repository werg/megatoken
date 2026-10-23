"""Replay public system context announcements without reading hidden labels."""

from __future__ import annotations

import json


def context_announcement(payload, role):
    if role != "system":
        return None
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(value, dict) or "context_window" not in value:
        return None
    window = value["context_window"]
    if not isinstance(window, dict) or set(window) != {
        "local_events",
        "cross_events",
        "pinned_events",
    }:
        raise ValueError("Public context announcement has invalid fields")
    if not isinstance(window["pinned_events"], list):
        raise ValueError("Public context pins must be an array")
    return window


def apply_context_announcement(multi, state, track, payload, role):
    window = context_announcement(payload, role)
    if window is None:
        return state
    if any(
        type(window[k]) is not int
        or not 0 <= window[k] <= multi.generator.config.max_history_chunks
        for k in ("local_events", "cross_events")
    ):
        raise ValueError("Public context request exceeds the session event budget")
    policy = state.context_policy.configure(track, **window)
    return multi.tracks.set_context_policy(state, policy)


def validate_context_announcements(episode):
    """Reject future or same-group context pins before preparation or training."""
    from megatoken.model.context import ContextPolicy

    records = {record.record_id: record for record in episode.records}
    known, policy = set(), ContextPolicy()
    native = json.loads(episode.supervision.annotations_json).get("native_action_tapes", ())
    cursor = 0
    for group in episode.meridians:
        for entry in native:
            if (
                "record_ids" in entry
                and entry["record_cursor"] <= cursor
                and all(
                    address is None or address in known for address in entry.get("record_ids", ())
                )
            ):
                known.update(entry["tape"]["event_ids"])
        for event in group.events:
            record = records[event.record_id]
            window = context_announcement(record.payload, record.role)
            if window is not None:
                policy = policy.configure(event.track_id, **window)
                if any(address not in known for _, address in policy.pins):
                    raise ValueError("Public context pins require evidence from an earlier group")
        known.update(event.record_id for event in group.events)
        cursor += len(group.events)
