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
