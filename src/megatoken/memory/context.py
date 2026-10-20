"""Admission/replay helpers for external KB context, not native selected events."""
from __future__ import annotations

from dataclasses import replace

from .semantic import AdmittedGroup


def admit_kb(state, group: AdmittedGroup):
    if state.kb is None:
        raise ValueError("Enable the native memory layout before admitting KB context")
    if group.anchor != int(state.meridians_seen.min()):
        raise ValueError("Direct KB admission anchors at the NEXT committed native meridian")
    updated = state.kb.admit(group)
    return state if updated is state.kb else replace(state, kb=updated)


def evict_kb(state, group_id: str):
    if state.kb is None:
        return state
    updated = state.kb.evict(group_id)
    return state if updated is state.kb else replace(state, kb=updated)
