"""Validate and replay native raw-facet events alongside physical records.

Physical record cursors count transcript records, not model iterations. Invalid
decodes and valid empty draws still recur but do not consume a physical record.
This index keeps them causal without feeding a scored target's own tape forward.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.data.schema import canonical_json
from megatoken.generation.structure import ActionTape, strict_json_loads
from megatoken.model import CoreState
from megatoken.model.core import EFFORTS

if TYPE_CHECKING:
    from megatoken.data.schema import Episode
    from megatoken.generation.policy import FacetedGenerator


def _validate_tape(tape: ActionTape, model: FacetedGenerator) -> None:
    if tape.codec_fingerprint != model.codec.fingerprint:
        raise ValueError("native tape codec differs from the model codec")
    if tape.facet_shape != (model.config.n_facets, model.config.facet_dim):
        raise ValueError("native tape facet geometry differs from the model")
    if type(tape.effort) is not int or tape.effort not in EFFORTS:
        raise ValueError("native tape has an unsupported effort")
    if type(tape.role_id) is not int or not 0 <= tape.role_id < len(ROLE_TO_ID):
        raise ValueError("native tape has an unsupported role")
    if type(tape.valid) is not bool or type(tape.terminal) is not bool:
        raise ValueError("native tape validity and terminal status must be booleans")
    # Generated tapes store exact FP32 values. Reject a finite JSON float that
    # would overflow or change when installed as the authoritative action tensor.
    raw = torch.from_numpy(tape.facet_array().copy())
    if not bool(torch.isfinite(raw).all()) or (
        not isinstance(tape.facets, bytes) and raw.tolist() != list(tape.facets)
    ):
        raise ValueError("native tape coordinates are not finite exact FP32 values")
    if any(type(token) is not int for token in tape.emitted_ids):
        raise ValueError("native tape emitted IDs must be integers")
    if model.codec.decode(tape.emitted_ids) != tape.payload:
        raise ValueError("native tape emitted IDs do not reproduce its exact bytes")
    if tape.valid and len(tape.emitted_ids) > model.config.max_capacity:
        raise ValueError("native output exceeds model capacity; re-collect instead of splitting it")


@dataclass(frozen=True)
class NativeHistory:
    """An immutable chronological index over an episode's original native tapes.

    At a cursor, zero-record events precede an optional valid nonempty event that
    owns the corresponding assistant record. A trailing cursor equal to the
    number of physical records may contain zero-record events only.
    """

    record_count: int
    _events: Mapping[int, tuple[ActionTape, ...]]
    _generated_records: frozenset[int]

    @classmethod
    def from_episode(cls, episode: Episode, model: FacetedGenerator) -> NativeHistory:
        annotations = strict_json_loads(episode.supervision.annotations_json)
        if not isinstance(annotations, dict):
            raise ValueError("Native history annotations must be a JSON object")
        entries = annotations.get("native_action_tapes", [])
        if not isinstance(entries, list):
            raise ValueError("native_action_tapes must be an ordered list")
        record_count = len(episode.records)
        events: dict[int, list[ActionTape]] = {}
        generated: set[int] = set()
        minimum_cursor = 0
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Native history entry {index} must be an object")
            if "record_cursor" not in entry:
                raise ValueError(
                    "Native action metadata lacks record_cursor; re-collect the native "
                    "trajectory because its zero-record event chronology cannot be inferred"
                )
            cursor = entry["record_cursor"]
            if type(cursor) is not int or not 0 <= cursor <= record_count:
                raise ValueError(f"Native history entry {index} has an out-of-range record_cursor")
            if cursor < minimum_cursor:
                raise ValueError(
                    f"Native history entry {index} reorders or repeats a consumed record cursor"
                )
            try:
                tape = ActionTape.from_json(canonical_json(entry["tape"]))
                _validate_tape(tape, model)
                owns_record = tape.valid and bool(tape.payload)
                if owns_record:
                    if cursor == record_count:
                        raise ValueError("valid nonempty native output has no physical record")
                    record = episode.records[cursor]
                    if (
                        record.role != "assistant"
                        or tape.role_id != ROLE_TO_ID[record.role]
                        or record.part_index != 0
                        or record.part_count != 1
                    ):
                        raise ValueError("native output must own one whole assistant record")
                    if record.codec_fingerprint != model.codec.fingerprint:
                        raise ValueError("native physical record codec differs from its tape")
                    if record.payload != tape.payload or record.emitted_ids != tape.emitted_ids:
                        raise ValueError(
                            "native physical record differs from its tape bytes or IDs"
                        )
                    generated.add(cursor)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Native history entry {index}: {exc}") from exc
            events.setdefault(cursor, []).append(tape)
            # A nonempty valid draw consumes a transcript record. A later draw
            # at the same cursor would otherwise silently be treated as earlier.
            minimum_cursor = cursor + int(owns_record)
        return cls(
            record_count,
            MappingProxyType({cursor: tuple(tapes) for cursor, tapes in events.items()}),
            frozenset(generated),
        )

    @property
    def has_events(self) -> bool:
        return bool(self._events)

    def _check_cursor(self, cursor: int) -> None:
        if type(cursor) is not int or not 0 <= cursor <= self.record_count:
            raise ValueError("physical record cursor is outside the native history")

    def has_generated_record(self, cursor: int) -> bool:
        self._check_cursor(cursor)
        return cursor in self._generated_records

    def paired_tape(self, cursor: int) -> ActionTape | None:
        """Return only the genuine raw action that owns this exact target record.

        This is a supervised label, never a prefix event. Invalid/empty draws
        and teacher replacements have no byte-to-facet correspondence to borrow.
        ``from_episode`` has already validated complete bytes, original emitted
        IDs, role, codec, and the one-record boundary for an eligible pair.
        """
        self._check_cursor(cursor)
        return self._events[cursor][-1] if cursor in self._generated_records else None

    def events_for_record(self, cursor: int, *, include_record: bool) -> tuple[ActionTape, ...]:
        """Select events immediately preceding this record in their original order.

        Prefix reconstruction includes the record's own tape and skips encoding
        that generated record. For a scored target, include_record=False returns
        only preceding zero-record draws, excluding that target's future facets.
        """
        self._check_cursor(cursor)
        events = self._events.get(cursor, ())
        if not include_record and cursor in self._generated_records:
            return events[:-1]
        return events


def replay_raw_tape(
    model: FacetedGenerator,
    state: CoreState,
    tape: ActionTape,
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
        model.config.n_facets,
        model.config.facet_dim,
    ):
        raise ValueError("Native tape codec or facet geometry differs from this model")
    raw = torch.as_tensor(
        tape.facet_array().copy(), device=state.device, dtype=torch.float32
    ).reshape(1, *tape.facet_shape)
    roles = torch.tensor([tape.role_id], device=state.device, dtype=torch.long)
    return model.core.advance_raw(state, raw, role_ids=roles)
