"""Immutable, byte-exact causal records shared by preparation and training.

Metadata is deliberately not rendered into model input. In particular, an
``Episode.supervision`` object is never a physical record. Byte spans refer to
decoded source-field bytes, not character offsets in a JSON transport file.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Protocol

from megatoken.contracts import ROLE_TO_ID

from .chunking import ChunkingConfig, segment_bytes
from .media_assets import MEDIA_RECORD_TYPE

SCHEMA_VERSION = 3
Role = Literal["system", "user", "assistant", "tool", "document", "environment"]
Provenance = Literal["natural", "teacher", "student", "executed", "simulated", "unknown"]
Outcome = Literal["success", "failure", "unknown", "incomplete"]


class Codec(Protocol):
    """The raw output codec; importing the data package needs no ML runtime."""

    fingerprint: str

    def encode(self, payload: bytes) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> bytes: ...


def _json_default(value: Any) -> str:
    # Columnar sources deliver timestamps as datetimes; the ISO form is their canonical text.
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_id(namespace: str, *parts: str | bytes) -> str:
    digest = hashlib.sha256(namespace.encode("utf-8"))
    for part in parts:
        encoded = part.encode("utf-8") if isinstance(part, str) else part
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ByteSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid half-open byte span {self}")

    def contains(self, start: int, end: int) -> bool:
        return self.start <= start and end <= self.end


@dataclass(frozen=True, slots=True)
class SourceSpan:
    uri: str
    field: str
    start: int
    end: int
    # Canonical wrappers have generated origins; they must not claim to be raw.
    transform: str = "identity"

    def __post_init__(self) -> None:
        ByteSpan(self.start, self.end)


@dataclass(frozen=True, slots=True)
class Lineage:
    source_id: str
    revision: str
    family_id: str
    ancestor_ids: tuple[str, ...] = ()
    repository: str | None = None
    task_id: str | None = None
    parent_episode_id: str | None = None
    source_uri: str = ""
    created_at: str | None = None
    source_row: int | None = None
    raw_sha256: str = ""
    frozen_split: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.revision or not self.family_id:
            raise ValueError("Source, pinned revision, and ancestor family are required")
        if not isinstance(self.ancestor_ids, tuple):
            raise TypeError("Lineage ancestor_ids must be an immutable tuple")
        if self.frozen_split not in {None, "train", "validation", "test", "quarantine"}:
            raise ValueError("Invalid inherited frozen family split")

    @property
    def family_keys(self) -> tuple[str, ...]:
        keys = [self.family_id, *self.ancestor_ids]
        if self.repository:
            repo = self.repository.removesuffix(".git").strip("/").casefold()
            # Sources spell the same repository "o/r", "github.com/o/r" or a full URL.
            repo = repo.removeprefix("https://").removeprefix("http://").removeprefix("github.com/")
            keys.append(f"repository:{repo}")
        if self.task_id:
            # Task IDs shared by released collections must share this namespace.
            keys.append(f"task:{self.task_id}")
        return tuple(sorted(set(keys)))


@dataclass(frozen=True, slots=True)
class Supervision:
    outcome: Outcome = "unknown"
    verification: str = "none"
    reward: float | None = None
    recovery_turns: tuple[int, ...] = ()
    benchmark_tags: tuple[str, ...] = ()
    annotations_json: str = "{}"

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "failure", "unknown", "incomplete"}:
            raise ValueError(f"Unknown outcome: {self.outcome}")
        if self.outcome == "success" and self.verification == "none":
            raise ValueError("A success label needs an explicit verification source")
        if self.reward is not None and not math.isfinite(self.reward):
            raise ValueError("Reward must be finite")
        annotations = json.loads(self.annotations_json)
        if not isinstance(annotations, dict):
            raise ValueError("Supervision annotations must encode an object")
        if "native_action_tapes" in annotations:
            from megatoken.model.multitrack import GroupActionTape

            try:
                for entry in annotations["native_action_tapes"]:
                    GroupActionTape.from_json(canonical_json(entry["tape"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Native action metadata contains an malformed tape") from error

    @property
    def verified_success(self) -> bool:
        return self.outcome == "success" and self.verification in {
            "external_tests",
            "local_assertions",
            "publisher_tests",
        }


@dataclass(frozen=True, slots=True)
class LogicalAction:
    """One complete tool invocation, regardless of its physical chunk count."""

    action_id: str
    call_id: str
    tool_name: str
    arguments_json: str
    record_ids: tuple[str, ...]
    serialized_sha256: str
    execution_id: str | None = None
    schema_id: str | None = None

    def __post_init__(self) -> None:
        if not self.action_id or not self.call_id or not self.tool_name or not self.record_ids:
            raise ValueError("Actions require identities, a tool, and their physical records")
        if not isinstance(json.loads(self.arguments_json), dict):
            raise ValueError("Tool arguments must be a complete JSON object")


@dataclass(frozen=True, slots=True)
class MediaSpan:
    """Content-time alignment, independent of causal commit order."""

    clock_id: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.clock_id or not 0 <= self.start_ms <= self.end_ms:
            raise ValueError("Media span needs an ordered interval on a named clock")


@dataclass(frozen=True, slots=True)
class TrackSpec:
    """One session-visible stream; track numbers are not spatial coordinates."""

    track_id: str
    role: Role
    default_modality: str = "text"
    description: str = ""
    settings_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.track_id or self.role not in ROLE_TO_ID:
            raise ValueError("Track spec needs an identity and a supported role")
        if not isinstance(self.description, str) or not isinstance(
            json.loads(self.settings_json), dict
        ):
            raise ValueError("Track defaults must contain a description and JSON settings object")


@dataclass(frozen=True, slots=True)
class TrackEvent:
    """One physical event assigned to a semantic meridian."""

    record_id: str
    track_id: str
    modality: str
    meridian_id: int
    role: Role
    alignment_id: str | None = None
    media_span: MediaSpan | None = None
    asset_sha256: str | None = None
    request: str = ""
    request_revision: int = 0
    instruction: str = ""
    settings_json: str = "{}"
    reference_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.record_id or not self.track_id or self.meridian_id < 0:
            raise ValueError("Track event needs ID, track and nonnegative meridian")
        if self.modality not in {"text", "audio", "image", "video", "compressed_text"}:
            raise ValueError("Unknown event modality")
        if self.role not in ROLE_TO_ID:
            raise ValueError("Unknown track event role")
        if (
            not isinstance(self.request, str)
            or not isinstance(self.request_revision, int)
            or self.request_revision < 0
        ):
            raise ValueError("Event request and revision must be well-formed")
        if self.asset_sha256 is not None and (
            len(self.asset_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.asset_sha256)
        ):
            raise ValueError("Media asset digest must be a lowercase SHA256")
        if (
            not isinstance(self.instruction, str)
            or type(self.request_revision) is not int
            or self.request_revision < 0
            or not isinstance(json.loads(self.settings_json), dict)
        ):
            raise ValueError("Event request requires a revision, instruction and settings object")
        if (
            not isinstance(self.reference_event_ids, tuple)
            or any(not isinstance(ref, str) or not ref for ref in self.reference_event_ids)
            or len(set(self.reference_event_ids)) != len(self.reference_event_ids)
        ):
            raise ValueError("Event references must be distinct immutable event identities")


@dataclass(frozen=True, slots=True)
class Meridian:
    """Exactly one causal group; all selected events can attend to each other."""

    meridian_id: int
    events: tuple[TrackEvent, ...]

    def __post_init__(self) -> None:
        if self.meridian_id < 0 or not self.events:
            raise ValueError("Meridian must contain at least one event")
        if any(event.meridian_id != self.meridian_id for event in self.events):
            raise ValueError("Meridian members must share one causal commit index")
        if len({event.track_id for event in self.events}) != len(self.events):
            raise ValueError("A meridian has at most one event per track")
        if len({event.record_id for event in self.events}) != len(self.events):
            raise ValueError("Meridian members must be distinct")


@dataclass(frozen=True, slots=True)
class PhysicalRecord:
    record_id: str
    role: Role
    payload: bytes
    turn_index: int = 0
    logical_id: str = ""
    part_index: int = 0
    part_count: int = 1
    policy_mask: bool = False
    provenance: Provenance = "unknown"
    source_span: SourceSpan | None = None
    # None means the whole record is eligible; () means no lexical positions.
    free_spans: tuple[ByteSpan, ...] | None = None
    emitted_ids: tuple[int, ...] | None = None
    # Source-codec cache is separate from IDs observed during actual generation.
    # Tokenizing the logical source before splitting keeps U invariant to chunk size.
    cached_ids: tuple[int, ...] | None = None
    emitted_free_mask: tuple[bool, ...] | None = None
    codec_fingerprint: str | None = None
    action_id: str | None = None
    result_of: str | None = None
    execution_id: str | None = None
    media_type: str = "text/plain; charset=utf-8"
    artifact_version: str = ""
    base_artifact_version: str = ""
    edit_continuation: str = ""
    chunk_boundary: str | None = None
    chunking_fingerprint: str | None = None
    forced_chunk_split: bool = False
    chunk_context: tuple[str, ...] = ()
    chunk_languages: tuple[str, ...] = ()
    chunk_diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ROLE_TO_ID:
            raise ValueError(f"Unknown record role: {self.role}")
        if self.provenance not in {
            "natural",
            "teacher",
            "student",
            "executed",
            "simulated",
            "unknown",
        }:
            raise ValueError(f"Unknown provenance: {self.provenance}")
        if not self.record_id or not isinstance(self.payload, bytes):
            raise TypeError("Records need an identity and immutable payload bytes")
        if self.edit_continuation not in ("", "recorded", "student"):
            raise ValueError("Edit continuations must identify actual recorded/student source")
        if self.base_artifact_version and not self.artifact_version:
            raise ValueError("An edited record needs its new artifact version")
        if self.turn_index < 0 or not 0 <= self.part_index < self.part_count:
            raise ValueError("Invalid turn/physical part index")
        if (self.chunk_boundary is None) != (self.chunking_fingerprint is None):
            raise ValueError("Semantic records require both boundary and chunking identity")
        for name in ("chunk_context", "chunk_languages", "chunk_diagnostics"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, str) for item in values):
                raise TypeError(f"{name} must be an immutable tuple of strings")
        if self.policy_mask and self.role not in {"assistant", "document"}:
            raise ValueError("Only assistant/document emission can carry policy loss")
        if self.action_id and (self.role != "assistant" or self.result_of):
            raise ValueError("A tool action must be assistant-owned and cannot also be its result")
        if self.result_of and self.role not in {"tool", "environment"}:
            raise ValueError("A result must be an environmental record")
        # A feature descriptor's span locates the original source bytes it encodes.
        if (
            self.source_span
            and self.media_type != MEDIA_RECORD_TYPE
            and self.source_span.end - self.source_span.start != len(self.payload)
        ):
            raise ValueError("Source span length differs from physical byte length")
        if self.free_spans is not None:
            if not isinstance(self.free_spans, tuple):
                raise TypeError("free_spans must be immutable")
            previous = 0
            for span in self.free_spans:
                if span.end > len(self.payload) or span.start < previous:
                    raise ValueError(
                        "Scored spans must be ordered, disjoint, and within the record"
                    )
                previous = span.end
        if self.emitted_ids is not None:
            if not isinstance(self.emitted_ids, tuple) or not self.codec_fingerprint:
                raise ValueError("Emitted IDs must be immutable and name the exact codec")
            if any(not isinstance(token, int) or token < 0 for token in self.emitted_ids):
                raise ValueError("Invalid emitted token IDs")
        if self.cached_ids is not None:
            if not isinstance(self.cached_ids, tuple) or not self.codec_fingerprint:
                raise ValueError("Cached IDs must be immutable and name their codec")
            if any(not isinstance(token, int) or token < 0 for token in self.cached_ids):
                raise ValueError("Invalid cached token IDs")
        if self.emitted_free_mask is not None:
            if self.emitted_ids is None or len(self.emitted_free_mask) != len(self.emitted_ids):
                raise ValueError("Emitted free-position mask must match the emitted IDs")

    @property
    def text(self) -> str:
        """Strict decoding; binary input is never silently replaced with U+FFFD."""
        return self.payload.decode("utf-8")


@dataclass(frozen=True, slots=True)
class Dependency:
    writer_id: str
    reader_id: str
    kind: str = "evidence"


@dataclass(frozen=True, slots=True)
class EpisodeRelation:
    """Out-of-band supervision, never model-visible evidence."""

    kind: str
    left: str
    right: str
    weight: float = 1.0

    def __post_init__(self):
        kinds = {
            "same_spoken_content",
            "translation_of",
            "describes",
            "synchronized_with",
            "first_frame_of",
            "last_frame_of",
            "edit_of",
            "preserves",
        }
        if (
            self.kind not in kinds
            or not self.left
            or not self.right
            or self.left == self.right
            or not math.isfinite(self.weight)
            or not 0 < self.weight <= 10
        ):
            raise ValueError("Relation requires a bounded pair of distinct artifacts")


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: str
    records: tuple[PhysicalRecord, ...]
    lineage: Lineage
    stream: str = "chat"
    actions: tuple[LogicalAction, ...] = ()
    supervision: Supervision = Supervision()
    dependencies: tuple[Dependency, ...] = ()
    tools_json: str = "[]"
    control_seed: bytes = b""
    # Reference controls are external, available evidence, never the future target.
    control_reference: bytes = b""
    reference_origin: str | None = None
    tracks: tuple[TrackSpec, ...] = ()
    meridians: tuple[Meridian, ...] = ()
    omni_relations: tuple[EpisodeRelation, ...] = ()
    # Hashing every payload is linear in the episode; it is computed once, on first use.
    _content_sha256: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.records or not isinstance(self.records, tuple):
            raise ValueError("An episode needs an identity and nonempty immutable records")
        if self.control_reference and self.reference_origin not in {
            "external_evidence",
            "earlier_history",
        }:
            raise ValueError("Reference controls require a causal origin")
        if not isinstance(json.loads(self.tools_json), list):
            raise ValueError("Tool schemas must be a JSON list")
        ids: dict[str, int] = {}
        completed: set[str] = set()
        logical_seen: set[str] = set()
        action_map = {action.action_id: action for action in self.actions}
        if len(action_map) != len(self.actions) or len({a.call_id for a in self.actions}) != len(
            self.actions
        ):
            raise ValueError("Duplicate action/call identity")
        result_logical: dict[str, str] = {}
        previous_turn = -1
        previous_logical = ""
        for index, record in enumerate(self.records):
            if record.record_id in ids or record.turn_index < previous_turn:
                raise ValueError("Duplicate record or reordered turn")
            ids[record.record_id] = index
            previous_turn = record.turn_index
            logical = record.logical_id or record.record_id
            if logical != previous_logical:
                if logical in logical_seen or record.part_index != 0:
                    raise ValueError("Logical record parts must be contiguous and start at zero")
                if (
                    index
                    and self.records[index - 1].part_index + 1 != self.records[index - 1].part_count
                ):
                    raise ValueError("Incomplete preceding logical record")
                logical_seen.add(logical)
            else:
                previous = self.records[index - 1]
                if (
                    record.part_index != previous.part_index + 1
                    or record.part_count != previous.part_count
                    or record.role != previous.role
                    or record.action_id != previous.action_id
                    or record.result_of != previous.result_of
                    or record.execution_id != previous.execution_id
                ):
                    raise ValueError("Physical parts disagree about their logical record")
            previous_logical = logical
            if record.action_id:
                action = action_map.get(record.action_id)
                if action is None or record.record_id not in action.record_ids:
                    raise ValueError("Physical action has no matching complete logical action")
                if record.part_index + 1 == record.part_count:
                    completed.add(record.action_id)
            if record.result_of:
                if record.result_of not in completed:
                    raise ValueError("An observation precedes its complete action")
                action = action_map[record.result_of]
                if action.execution_id != record.execution_id:
                    raise ValueError("Result execution identity does not match its action")
                if record.provenance == "executed" and not record.execution_id:
                    raise ValueError("Executed observations require an execution identity")
                if (
                    record.result_of in result_logical
                    and result_logical[record.result_of] != logical
                ):
                    raise ValueError("A logical action must execute exactly once")
                result_logical[record.result_of] = logical
        logical_ids = {record.logical_id for record in self.records}
        if any(
            relation.left not in logical_ids or relation.right not in logical_ids
            for relation in self.omni_relations
        ):
            raise ValueError("Omnimodal relation references a missing source artifact")
        if self.records[-1].part_index + 1 != self.records[-1].part_count:
            raise ValueError("Incomplete final logical record")
        for action in self.actions:
            try:
                positions = [ids[record_id] for record_id in action.record_ids]
            except KeyError as exc:
                raise ValueError("Logical action refers to an absent physical record") from exc
            if positions != list(range(positions[0], positions[0] + len(positions))):
                raise ValueError("Logical action records are not contiguous/in order")
            physical = tuple(self.records[pos] for pos in positions)
            if any(record.action_id != action.action_id for record in physical):
                raise ValueError("Logical action and physical action identities differ")
            if (
                digest_bytes(b"".join(record.payload for record in physical))
                != action.serialized_sha256
            ):
                raise ValueError("Logical action digest differs from actual emitted bytes")
        for dependency in self.dependencies:
            if dependency.writer_id not in ids or dependency.reader_id not in ids:
                raise ValueError("Dependency refers to an absent physical record")
            if ids[dependency.writer_id] >= ids[dependency.reader_id]:
                raise ValueError("Future evidence cannot be a writer for an earlier reader")
            if not self.records[ids[dependency.reader_id]].policy_mask:
                raise ValueError("A declared dependent prediction must carry policy loss")

        if self.tracks or self.meridians:
            names = {track.track_id for track in self.tracks}
            if len(names) != len(self.tracks):
                raise ValueError("Track specifications must have distinct identities")
            event_ids: set[str] = set()
            preceding = -1
            previous_record = -1
            prior_events: set[str] = set()
            track_revisions: dict[str, int] = {}
            for meridian in self.meridians:
                if meridian.meridian_id <= preceding:
                    raise ValueError("Causal meridian numbers must increase strictly")
                preceding = meridian.meridian_id
                for event in meridian.events:
                    if any(ref not in prior_events for ref in event.reference_event_ids):
                        raise ValueError(
                            "Event request references missing or future meridian media"
                        )
                    if event.request_revision < track_revisions.get(event.track_id, 0):
                        raise ValueError("Track request revisions must not decrease")
                    track_revisions[event.track_id] = event.request_revision
                    if event.track_id not in names:
                        raise ValueError("Meridian refers to a track absent from the session")
                    if event.record_id not in ids or event.record_id in event_ids:
                        raise ValueError("Meridian records must exist and occur exactly once")
                    event_ids.add(event.record_id)
                    record_index = ids[event.record_id]
                    if record_index <= previous_record:
                        raise ValueError("Meridian order must follow physical record order")
                    previous_record = record_index
                    if self.records[record_index].role != event.role:
                        raise ValueError("Event role differs from its physical record")
                prior_events.update(event.record_id for event in meridian.events)
            if self.meridians and event_ids != set(ids):
                raise ValueError("Every physical record requires a meridian event")

    @property
    def content_sha256(self) -> str:
        if not self._content_sha256:
            digest = stable_id(
                "episode-content-v3",
                *(
                    value
                    for group in self.iter_logical_records()
                    for value in (group[0].role, b"".join(record.payload for record in group))
                ),
            )
            object.__setattr__(self, "_content_sha256", digest)
        return self._content_sha256

    def iter_logical_records(self) -> Iterable[tuple[PhysicalRecord, ...]]:
        """Return original logical events independent of physical chunk limits."""
        start = 0
        for index, record in enumerate(self.records):
            if record.part_index + 1 == record.part_count:
                yield self.records[start : index + 1]
                start = index + 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema_version"] = SCHEMA_VERSION
        del result["_content_sha256"]
        for record, value in zip(self.records, result["records"], strict=True):
            value["payload"] = base64.b64encode(record.payload).decode("ascii")
        result["control_seed"] = base64.b64encode(self.control_seed).decode("ascii")
        result["control_reference"] = base64.b64encode(self.control_reference).decode("ascii")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Episode:
        obj = dict(value)
        if obj.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("Unsupported episode schema version")
        lineage = dict(obj["lineage"])
        lineage["ancestor_ids"] = tuple(lineage.get("ancestor_ids", ()))
        obj["lineage"] = Lineage(**lineage)
        supervision = dict(obj.get("supervision", {}))
        for key in ("recovery_turns", "benchmark_tags"):
            supervision[key] = tuple(supervision.get(key, ()))
        obj["supervision"] = Supervision(**supervision)
        records = []
        for raw in obj["records"]:
            record = dict(raw)
            record["payload"] = base64.b64decode(record["payload"], validate=True)
            if record.get("source_span") is not None:
                record["source_span"] = SourceSpan(**record["source_span"])
            if record.get("free_spans") is not None:
                record["free_spans"] = tuple(ByteSpan(**span) for span in record["free_spans"])
            for key in (
                "emitted_ids",
                "cached_ids",
                "emitted_free_mask",
                "chunk_context",
                "chunk_languages",
                "chunk_diagnostics",
            ):
                if record.get(key) is not None:
                    record[key] = tuple(record[key])
            records.append(PhysicalRecord(**record))
        obj["records"] = tuple(records)
        obj["actions"] = tuple(
            LogicalAction(**{**action, "record_ids": tuple(action["record_ids"])})
            for action in obj.get("actions", ())
        )
        obj["dependencies"] = tuple(Dependency(**dep) for dep in obj.get("dependencies", ()))
        obj["omni_relations"] = tuple(
            EpisodeRelation(**relation) for relation in obj.get("omni_relations", ())
        )
        obj["tracks"] = tuple(TrackSpec(**spec) for spec in obj.get("tracks", ()))
        obj["meridians"] = tuple(
            Meridian(
                meridian["meridian_id"],
                tuple(
                    TrackEvent(
                        **{
                            **event,
                            "media_span": (
                                MediaSpan(**event["media_span"])
                                if event.get("media_span") is not None
                                else None
                            ),
                            "reference_event_ids": tuple(event.get("reference_event_ids", ())),
                        }
                    )
                    for event in meridian["events"]
                ),
            )
            for meridian in obj.get("meridians", ())
        )
        for key in ("control_seed", "control_reference"):
            obj[key] = base64.b64decode(obj.get(key, ""), validate=True)
        return cls(**obj)


@dataclass(frozen=True, slots=True)
class EncodedRecord:
    record: PhysicalRecord
    token_ids: tuple[int, ...]
    free_mask: tuple[bool, ...]
    token_spans: tuple[ByteSpan, ...]
    codec_fingerprint: str
    transcript_tokens: int

    @property
    def payload(self) -> bytes:
        return self.record.payload

    @property
    def P(self) -> int:
        return sum(self.free_mask)

    @property
    def tokens(self) -> int:
        return self.transcript_tokens

    @property
    def positions(self) -> int:
        """Actual likelihood positions; a noncanonical sample may differ from D."""
        return len(self.token_ids)


def encode_record(record: PhysicalRecord, codec: Codec) -> EncodedRecord:
    """Keep emitted IDs; mask fixed/copied bytes conservatively at boundaries."""
    # A media reference is a typed asset descriptor, never lexical supervision.
    # It counts toward the data meters by its real input size: the number of
    # teacher feature vectors the model reads for it.
    from .media_assets import MEDIA_RECORD_TYPE, MediaRecord

    if record.media_type == MEDIA_RECORD_TYPE:
        media = MediaRecord.from_bytes(record.payload)
        if (
            record.emitted_ids is not None
            or record.cached_ids is not None
            or (record.emitted_free_mask is not None or record.free_spans is not None)
        ):
            raise ValueError("Media references cannot carry lexical token labels")
        return EncodedRecord(record, (), (), (), codec.fingerprint, media.source.shape[0])
    if record.media_type.split("/", 1)[0] in {"audio", "image", "video"}:
        raise ValueError(
            "Raw media payloads must be converted to canonical media asset references "
            "before text-token preparation"
        )
    if record.emitted_ids is not None or record.cached_ids is not None:
        if record.codec_fingerprint != codec.fingerprint:
            raise ValueError("Recorded emission uses a different codec fingerprint")
        ids = record.emitted_ids if record.emitted_ids is not None else record.cached_ids
        assert ids is not None
    else:
        ids = tuple(codec.encode(record.payload))
    if codec.decode(ids) != record.payload:
        raise ValueError("Codec/emitted IDs do not reproduce the original bytes")
    if record.cached_ids is not None:
        # Preparation encodes the original logical field once, then partitions
        # its reference IDs. These chunks keep U independent of chunk size.
        if codec.decode(record.cached_ids) != record.payload:
            raise ValueError("Reference meter IDs do not reproduce the original bytes")
        transcript_tokens = len(record.cached_ids)
    elif record.emitted_ids is not None:
        reference_ids = tuple(codec.encode(record.payload))
        if codec.decode(reference_ids) != record.payload:
            raise ValueError("Reference tokenization changed the emitted bytes")
        transcript_tokens = len(reference_ids)
    else:
        transcript_tokens = len(ids)
    spans = []
    offset = 0
    for token in ids:
        piece = codec.decode([token])
        if not piece or record.payload[offset : offset + len(piece)] != piece:
            raise ValueError("The raw meter requires nonempty, compositional token bytes")
        spans.append(ByteSpan(offset, offset + len(piece)))
        offset += len(piece)
    if offset != len(record.payload):
        raise ValueError("Token byte spans do not cover the physical record")
    allowed_spans = record.free_spans
    mask = tuple(
        record.policy_mask
        and (
            allowed_spans is None
            or any(span.contains(token.start, token.end) for span in allowed_spans)
        )
        and (record.emitted_free_mask is None or record.emitted_free_mask[index])
        for index, token in enumerate(spans)
    )
    return EncodedRecord(record, ids, mask, tuple(spans), codec.fingerprint, transcript_tokens)


def encode_episode(episode: Episode, codec: Codec) -> tuple[EncodedRecord, ...]:
    return tuple(encode_record(record, codec) for record in episode.records)


def split_payload(payload: bytes, max_bytes: int) -> Iterable[tuple[int, bytes]]:
    """Prefer semantic line/word breaks; retain every byte including whitespace."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    offset = 0
    if not payload:
        yield 0, b""
        return
    while offset < len(payload):
        end = min(len(payload), offset + max_bytes)
        if end < len(payload):
            candidate = payload.rfind(b"\n", offset + max_bytes // 2, end)
            if candidate < 0:
                candidate = payload.rfind(b" ", offset + max_bytes // 2, end)
            if candidate >= 0:
                end = candidate + 1
            # Avoid breaking UTF-8 when possible; binary is still lossless.
            safe = end
            while safe > offset and payload[safe] & 0xC0 == 0x80:
                safe -= 1
            if safe > offset:
                end = safe
        yield offset, payload[offset:end]
        offset = end


def make_records(
    payload: bytes,
    *,
    logical_id: str,
    role: Role,
    turn_index: int = 0,
    policy_mask: bool = False,
    provenance: Provenance = "unknown",
    max_bytes: int = 16384,
    source_uri: str = "",
    source_field: str = "",
    transform: str = "identity",
    action_id: str | None = None,
    result_of: str | None = None,
    execution_id: str | None = None,
    free_spans: tuple[ByteSpan, ...] | None = None,
    codec: Codec | None = None,
    max_tokens: int | None = None,
    chunking: ChunkingConfig | None = None,
    content_kind: str = "auto",
    language: str | None = None,
    filename: str | None = None,
    mime_type: str | None = None,
    locale: str | None = None,
    chat_template: str | None = None,
) -> tuple[PhysicalRecord, ...]:
    """Split one authoritative role-owned logical field without reframing it.

    Serialized conversations must first pass through a source adapter so roles,
    turn masks, source offsets, and canonical encoding units stay aligned.
    """
    if chat_template is not None and chat_template != "none":
        raise ValueError(
            "make_records owns one role-framed logical field; use the serialized_chat "
            "adapter to frame a conversation before creating records"
        )
    if max_tokens is not None and (codec is None or max_tokens <= 0):
        raise ValueError("A positive token chunk limit requires an explicit raw codec")
    token_chunks: list[tuple[int, ...]] = []
    semantic_chunks = None
    chunking_fingerprint = None
    if codec is None:
        if chunking is not None:
            raise ValueError("Semantic chunking requires the explicit reference codec")
        pieces = list(split_payload(payload, max_bytes))
    else:
        policy = chunking or ChunkingConfig()
        policy = policy.with_hard_limit(max_tokens or policy.hard_max_tokens, max_bytes=max_bytes)
        semantic_chunks = segment_bytes(
            payload,
            codec,
            policy,
            # A native call is a JSON object; an action written as prose around a call is not.
            content_kind="json"
            if action_id is not None and payload.lstrip().startswith(b"{")
            else content_kind,
            language=language,
            filename=filename,
            mime_type=mime_type,
            locale=locale,
            chat_template="none",
        )
        pieces = [(item.byte_start, item.payload) for item in semantic_chunks]
        token_chunks = [item.token_ids for item in semantic_chunks]
        chunking_fingerprint = policy.identity(codec)["fingerprint"]
    result = []
    for index, (offset, chunk) in enumerate(pieces):
        local_spans = (
            None
            if free_spans is None
            else tuple(
                ByteSpan(max(0, span.start - offset), min(len(chunk), span.end - offset))
                for span in free_spans
                if span.start < offset + len(chunk) and span.end > offset
            )
        )
        result.append(
            PhysicalRecord(
                record_id=stable_id("physical-v1", logical_id, str(index), chunk),
                role=role,
                payload=chunk,
                turn_index=turn_index,
                logical_id=logical_id,
                part_index=index,
                part_count=len(pieces),
                policy_mask=policy_mask,
                provenance=provenance,
                source_span=SourceSpan(
                    source_uri, source_field, offset, offset + len(chunk), transform
                ),
                free_spans=local_spans,
                action_id=action_id,
                result_of=result_of,
                execution_id=execution_id,
                cached_ids=token_chunks[index] if codec is not None else None,
                codec_fingerprint=codec.fingerprint if codec is not None else None,
                chunk_boundary=semantic_chunks[index].boundary if semantic_chunks else None,
                chunking_fingerprint=chunking_fingerprint,
                forced_chunk_split=semantic_chunks[index].forced if semantic_chunks else False,
                chunk_context=semantic_chunks[index].context if semantic_chunks else (),
                chunk_languages=semantic_chunks[index].languages if semantic_chunks else (),
                chunk_diagnostics=semantic_chunks[index].diagnostics if semantic_chunks else (),
            )
        )
    return tuple(result)


def apply_recovery_policy(episode: Episode) -> Episode:
    """Failures provide context/outcome learning; imitate only annotated repairs."""
    if episode.supervision.outcome == "success":
        return episode
    turns = frozenset(episode.supervision.recovery_turns)
    records = tuple(
        replace(record, policy_mask=record.policy_mask and record.turn_index in turns)
        for record in episode.records
    )
    kept_ids = {record.record_id for record in records if record.policy_mask}
    return replace(
        episode,
        records=records,
        dependencies=tuple(dep for dep in episode.dependencies if dep.reader_id in kept_ids),
    )


def sequential_meridians(episode: Episode) -> Episode:
    """Group an imported linear episode for meridian training.

    Every physical record is its own meridian, in record order, on one track
    per role. A canonical feature descriptor becomes a feature event of its
    kind (text descriptors are ``compressed_text``); everything else is text.
    """
    from .media_assets import MEDIA_RECORD_TYPE, MediaRecord

    if episode.tracks or episode.meridians:
        raise ValueError("Episode already declares its own causal groups")
    roles: list[str] = []
    meridians = []
    for index, record in enumerate(episode.records):
        if record.role not in roles:
            roles.append(record.role)
        modality, asset = "text", None
        if record.media_type == MEDIA_RECORD_TYPE:
            descriptor = MediaRecord.from_bytes(record.payload)
            modality = "compressed_text" if descriptor.modality == "text" else descriptor.modality
            asset = descriptor.source.sha256
        event = TrackEvent(
            record.record_id, record.role, modality, index, record.role, asset_sha256=asset
        )
        meridians.append(Meridian(index, (event,)))
    return replace(
        episode,
        tracks=tuple(TrackSpec(role, role) for role in roles),
        meridians=tuple(meridians),
    )
