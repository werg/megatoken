"""Validate and replay native raw-facet events alongside physical records.

A native event is a complete jointly selected group, including same-track peers.

Physical record cursors count transcript records, not model iterations. Invalid
decodes and valid empty draws still recur but do not consume a physical record.
This index keeps them causal without feeding a scored target's own tape forward.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING
from weakref import ReferenceType, ref

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.data.schema import canonical_json
from megatoken.generation.structure import strict_json_loads
from megatoken.model import CoreState, GroupActionTape

if TYPE_CHECKING:
    from megatoken.data.schema import Episode
    from megatoken.generation.policy import FacetedGenerator


def _validate_tape(tape: GroupActionTape, model: FacetedGenerator) -> None:
    if tape.codec_fingerprint != model.codec.fingerprint:
        raise ValueError("native tape codec differs from the model codec")
    if tape.facet_shape != (len(tape.tracks), model.config.n_facets, model.config.facet_dim):
        raise ValueError("native tape facet geometry differs from the model")
    for index, modality in enumerate(tape.modalities):
        if type(tape.roles[index]) is not int or tape.roles[index] not in ROLE_TO_ID.values():
            raise ValueError("native tape has an unsupported role")
        if type(tape.valid[index]) is not bool or type(tape.terminal[index]) is not bool:
            raise ValueError("native tape validity and terminal status must be booleans")
        if modality == "text":
            tokens = tape.emitted_ids[index]
            if any(type(token) is not int for token in tokens):
                raise ValueError("native tape emitted IDs must be integers")
            if model.codec.decode(tokens) != tape.payloads[index]:
                raise ValueError("native tape emitted IDs do not reproduce its exact bytes")
            if tape.valid[index] and len(tokens) > model.config.max_capacity:
                raise ValueError(
                    "native output exceeds model capacity; re-collect instead of splitting it"
                )
        elif tape.render_status == "rendered" and tape.valid[index]:
            receipt = strict_json_loads(tape.payloads[index])
            if (
                receipt.get("kind") != "media-sha256-v1"
                or receipt.get("modality") != modality
                or receipt.get("realizer") != tape.pack_fingerprints[index]
            ):
                raise ValueError("native media receipt differs from its selected decoder")


@dataclass(frozen=True)
class NativeHistory:
    """Chronological complete selections plus exact per-event physical bindings.

    A group may produce several text/media records, or no physical output. Its
    complete selection is replayed once; member views are only supervision.
    Explicit record_ids prevent missing, failed or cancelled peers from being
    inferred from text counts. Every group explicitly binds all selected members,
    including members that have no physical output.
    """

    record_count: int
    _events: Mapping[int, tuple[GroupActionTape, ...]]
    _generated_records: frozenset[int]
    _members: Mapping[int, object]
    _groups: Mapping[tuple[int, ...], GroupActionTape]
    _credits: Mapping[int, tuple]

    @classmethod
    def from_episode(cls, episode: Episode, model: FacetedGenerator) -> NativeHistory:
        from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord
        from megatoken.model.multitrack import GroupActionMember

        annotations = strict_json_loads(episode.supervision.annotations_json)
        if not isinstance(annotations, dict):
            raise ValueError("Native history annotations must be a JSON object")
        entries = annotations.get("native_action_tapes", [])
        if not isinstance(entries, list):
            raise ValueError("native_action_tapes must be an ordered list")
        record_count = len(episode.records)
        positions = {r.record_id: i for i, r in enumerate(episode.records)}
        events, members, groups, credits = {}, {}, {}, {}
        minimum_cursor = 0
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or "record_cursor" not in entry:
                raise ValueError(
                    "Native action metadata lacks record_cursor; re-collect its chronology"
                )
            cursor = entry["record_cursor"]
            if type(cursor) is not int or not 0 <= cursor <= record_count:
                raise ValueError(f"Native history entry {index} has an out-of-range record_cursor")
            if cursor < minimum_cursor:
                raise ValueError(
                    f"Native history entry {index} reorders or repeats a consumed record cursor"
                )
            try:
                tape = GroupActionTape.from_json(canonical_json(entry["tape"]))
                _validate_tape(tape, model)
                if entry.get("outcome_credit") is not None:
                    from megatoken.training.outcome import OutcomeCredit

                    credit = OutcomeCredit(**entry["outcome_credit"])
                    if tape.policy_context is None or len(tape.policy_context) != tape.meridian_id:
                        raise ValueError(
                            "Verified outcome lacks complete committed serving context"
                        )
                    credits.setdefault(cursor, []).append((tape, credit))
                if "record_ids" not in entry:
                    raise ValueError(
                        "Native selections require explicit record_ids for every member; re-collect"
                    )
                bindings = entry["record_ids"]
                if not isinstance(bindings, (list, tuple)) or len(bindings) != len(tape.tracks):
                    raise ValueError("Native record bindings must cover every selected member")
                bound = []
                for event, record_id in enumerate(bindings):
                    if record_id is None:
                        continue
                    if tape.render_status != "rendered" or not tape.valid[event]:
                        raise ValueError(
                            "An unrendered or invalid event cannot own a physical record"
                        )
                    position = positions[record_id]
                    if position in members or position != cursor + len(bound):
                        raise ValueError(
                            "Native group records must be distinct and contiguous in event order"
                        )
                    record = episode.records[position]
                    if (
                        record.role != "assistant"
                        or tape.roles[event] != ROLE_TO_ID[record.role]
                        or record.part_index != 0
                        or record.part_count != 1
                    ):
                        raise ValueError("native output must own one whole assistant record")
                    if tape.modalities[event] == "text":
                        if record.codec_fingerprint != model.codec.fingerprint:
                            raise ValueError("native physical record codec differs from its tape")
                        if (
                            record.payload != tape.payloads[event]
                            or record.emitted_ids != tape.emitted_ids[event]
                        ):
                            raise ValueError(
                                "native physical record differs from its tape bytes or IDs"
                            )
                    else:
                        if record.media_type != MEDIA_RECORD_TYPE:
                            raise ValueError(
                                "Native media requires its checked physical descriptor"
                            )
                        media = MediaRecord.from_bytes(record.payload)
                        receipt = strict_json_loads(tape.payloads[event])
                        asset = media.rendered
                        if (
                            asset is None
                            or asset.sha256 != receipt["sha256"]
                            or list(asset.shape) != receipt["shape"]
                            or asset.extractor != receipt["realizer"]
                            or media.modality != tape.modalities[event]
                        ):
                            raise ValueError(
                                "Collected media does not match the actual decoded action"
                            )
                    members[position] = (
                        tape if len(tape.tracks) == 1 else GroupActionMember(tape, event)
                    )
                    bound.append(position)
                if bound:
                    groups[tuple(bound)] = tape
                events.setdefault(cursor, []).append(tape)
                minimum_cursor = cursor + len(bound)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Native history entry {index}: {exc}") from exc
        return cls(
            record_count,
            MappingProxyType({c: tuple(t) for c, t in events.items()}),
            frozenset(members),
            MappingProxyType(members),
            MappingProxyType(groups),
            MappingProxyType({c: tuple(t) for c, t in credits.items()}),
        )

    @classmethod
    def of(cls, episode: Episode, model: FacetedGenerator) -> NativeHistory:
        key = (id(episode), id(model))
        entry = _HISTORIES.get(key)
        if entry is not None and entry[0] is episode and entry[1]() is model:
            _HISTORIES.move_to_end(key)
            return entry[2]
        history = cls.from_episode(episode, model)
        _HISTORIES[key] = (episode, ref(model), history)
        while len(_HISTORIES) > _HISTORY_CACHE:
            _HISTORIES.popitem(last=False)
        return history

    @property
    def has_events(self) -> bool:
        return bool(self._events)

    def _check_cursor(self, cursor: int) -> None:
        if type(cursor) is not int or not 0 <= cursor <= self.record_count:
            raise ValueError("physical record cursor is outside the native history")

    def has_generated_record(self, cursor: int) -> bool:
        self._check_cursor(cursor)
        return cursor in self._members

    def paired_tape(self, cursor: int):
        self._check_cursor(cursor)
        return self._members.get(cursor)

    def group_for_records(self, cursors) -> GroupActionTape | None:
        return self._groups.get(tuple(cursors))

    def policy_entries(self, start, end):
        return tuple(
            pair
            for cursor, entries in self._credits.items()
            if start <= cursor < end or start == end == cursor == self.record_count
            for pair in entries
        )

    def events_for_record(
        self, cursor: int, *, include_record: bool
    ) -> tuple[GroupActionTape, ...]:
        self._check_cursor(cursor)
        events = self._events.get(cursor, ())
        if not include_record and cursor in self._members and events:
            member = self._members[cursor]
            own = getattr(member, "group", member)
            return tuple(tape for tape in events if tape is not own)
        return events


_HISTORY_CACHE = 16
_HISTORIES: OrderedDict[
    tuple[int, int], tuple[Episode, ReferenceType[FacetedGenerator], NativeHistory]
] = OrderedDict()


def replay_raw_tape(
    model: FacetedGenerator,
    state: CoreState,
    tape: GroupActionTape,
) -> CoreState:
    """Recur through a validated fixed event without decoding or scoring it.

    NativeHistory validates stored tapes once. This transition needs no schema,
    re-encoding, random draw, or discarded prediction stack. Central/history
    gradients remain live unless the caller establishes an explicit no_grad or
    detach boundary. Prospective control belongs to the predictor; once the raw
    action is known it cannot alter the contextual transition or its KV write.
    Tape-owned controls are retained for policy-density replay, which is a
    separate operation from this fixed-action state reconstruction.
    """
    if state.batch_size != 1:
        raise ValueError("Native raw history replay requires one independent episode")
    if tape.codec_fingerprint != model.codec.fingerprint or tape.facet_shape != (
        1,
        model.config.n_facets,
        model.config.facet_dim,
    ):
        raise ValueError("Native tape codec or facet geometry differs from this model")
    roles = torch.tensor([tape.roles[0]], device=state.device, dtype=torch.long)
    return model.core.advance_raw(state, tape.single_raw(state.device), role_ids=roles)
