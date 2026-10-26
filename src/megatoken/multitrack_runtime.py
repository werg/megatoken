"""The one serving runtime: grouped observations, text and media generation.

Every way of running a model goes through it: the ``generate`` and ``serve``
commands, the JSON-lines service, and the frozen student that collects rollouts for
training. An ordinary conversation shares a task track across message roles.

A session owns its committed multitrack state and its transducer agenda: every
committed ordinary meridian may make a bank lookup due, served before the
next one; a completed user message or assistant turn, or an arriving tool,
document or environment result, always does. Ready tool results are observed in later
meridians. Speculative branches are copy-on-write and can be rejected
without disturbing the committed session.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
from pathlib import Path
from threading import BoundedSemaphore, RLock

import torch

from megatoken.context_archive import ContextArchive
from megatoken.contracts import MAIN_TRACK, MAIN_TRACK_PURPOSE, ROLE_TO_ID
from megatoken.data.chunking import ChunkingConfig, segment_bytes
from megatoken.data.knowledge import FilePreparer
from megatoken.generation.structure import ActionSpec
from megatoken.lifecycle import TrackRegistry
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import (
    GroupGeneration,
    MultiTrackGenerator,
    MultiTrackState,
    ObservedEvent,
    SelectedGroup,
)
from megatoken.transducer_service import (
    AddedFiles,
    TransducerInferenceEngine,
    TransducerSession,
)


def _commit_owner(method):
    @wraps(method)
    def serialized(self, session, *args, **kwargs):
        with session.lock:
            return method(self, session, *args, **kwargs)

    return serialized


@dataclass(frozen=True)
class PendingGroup:
    base_revision: int
    result: GroupGeneration
    output_bytes: int = 0
    selected_at_ns: int = 0
    specs: tuple[ActionSpec | MediaOutputSpec, ...] = ()
    visible_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadyWork:
    event_id: str
    track: str
    spec: ActionSpec | MediaOutputSpec
    control: bytes = b""
    prerequisites: tuple[str, ...] = ()
    task_revision: int = 0
    priority: int = 0
    modality_options: tuple[ActionSpec | MediaOutputSpec, ...] = ()


@dataclass(frozen=True)
class LifecycleRecordedTape:
    """An explicitly recorded decision, deliberately not a sampled model tape."""

    meridian_id: int
    tracks: tuple[str, ...]
    event_ids: tuple[str, ...]
    payloads: tuple[bytes, ...]
    valid: tuple[bool, ...] = (True,)
    modalities: tuple[str, ...] = ("text",)
    terminal: tuple[bool, ...] = (True,)
    emitted_ids: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class LifecycleTeacherAction:
    tape: LifecycleRecordedTape
    policy_digest: str
    decision: int


@dataclass(frozen=True)
class ControlledFixtureAction:
    """Controlled engineering input with neither model nor teacher authority."""

    tape: LifecycleRecordedTape
    fixture_digest: str
    decision: int


@dataclass
class RenderWork:
    selection: SelectedGroup
    future: Future
    status: str = "rendering"
    result: GroupGeneration | None = None
    error: str | None = None
    delivered_events: set[str] = field(default_factory=set)
    delivery_progress: dict[str, int] = field(default_factory=dict)
    delivery_log: list[dict] = field(default_factory=list)
    stop_published: bool = False
    completion_committed: bool = False
    outcome_published: bool = False


@dataclass
class TrackSession:
    """A single shared-history context, with no private per-agent transformers."""

    memory: TransducerSession
    revision: int = 0
    pending: PendingGroup | None = None
    events: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)
    # Every text payload committed so far, oldest first: what a tool call may copy from.
    history: list[bytes] = field(default_factory=list)

    tracks: TrackRegistry = field(default_factory=TrackRegistry)
    lock: RLock = field(default_factory=RLock, repr=False)
    completed_events: set[str] = field(default_factory=set)
    occupied_events: set[str] = field(default_factory=set)
    tool_routes: dict[str, tuple[str, str]] = field(default_factory=dict)
    tool_outcomes: dict[str, str] = field(default_factory=dict)
    tool_ready: dict[str, bytes] = field(default_factory=dict)
    tool_errors: dict[str, str] = field(default_factory=dict)
    tool_publication: dict[str, int] = field(default_factory=dict)
    selected_groups: dict[int, GroupGeneration] = field(default_factory=dict)
    teacher_lifecycle_actions: dict[str, LifecycleTeacherAction] = field(default_factory=dict)
    controlled_fixture_actions: dict[str, ControlledFixtureAction] = field(default_factory=dict)
    rendering: dict[int, RenderWork] = field(default_factory=dict)
    queue: list[ReadyWork] = field(default_factory=list)
    work_sequence: int = 0
    scheduling_steps: int = 0
    scheduling_log: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    reserved_output_bytes: dict[int, int] = field(default_factory=dict)
    context_archive: ContextArchive = field(default_factory=ContextArchive)
    execution_programs: dict[str, dict] = field(default_factory=dict)
    execution_inputs: dict[str, dict] = field(default_factory=dict)
    execution_publication: dict[str, int] = field(default_factory=dict)
    execution_source_events: dict[str, tuple[str, ...]] = field(default_factory=dict)
    execution_receipts: list[dict] = field(default_factory=list)
    execution_delivered: set[str] = field(default_factory=set)
    event_arrived_at_ns: dict[str, int] = field(default_factory=dict)
    selection_at_ns: dict[int, int] = field(default_factory=dict)
    selection_visible_event_ids: dict[int, frozenset[str]] = field(default_factory=dict)

    @property
    def state(self) -> MultiTrackState:
        return self.memory.state


class MultiTrackRuntime:
    """Reference scheduler: one ready meridian at a time, no global track barrier."""

    def __init__(
        self,
        generator: MultiTrackGenerator,
        *,
        chunking: ChunkingConfig | None = None,
        collections: Mapping[str, str | Path] | None = None,
        preparer: Callable[[], FilePreparer] | None = None,
        memory_root: str | Path | None = None,
        output_specs: Mapping[str, ActionSpec | MediaOutputSpec] | None = None,
        record_policy_context: bool = False,
        max_event_output_bytes: int = 64 << 20,
        max_session_output_bytes: int = 256 << 20,
        context_archive_bytes: int = 64 << 20,
        context_archive_events: int = 1024,
    ) -> None:
        """``chunking`` is the served bundle's physical chunk policy (default: the model's)."""
        self.generator = generator
        ContextArchive(max_bytes=context_archive_bytes, max_events=context_archive_events)
        self.context_archive_bytes, self.context_archive_events = (
            context_archive_bytes,
            context_archive_events,
        )
        if not 1 <= max_event_output_bytes <= max_session_output_bytes:
            raise ValueError("Output byte budgets must be positive and ordered")
        self.max_event_output_bytes = max_event_output_bytes
        self.max_session_output_bytes = max_session_output_bytes
        generator.tracks.record_policy_context = record_policy_context
        self.render_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="media-render")
        self.max_rendering = 4
        self.render_capacity = BoundedSemaphore(8)
        self.max_retained_outputs = 64
        model = generator.generator
        self.max_piece_bytes = max(
            len(model.codec.token_bytes(i)) for i in model.codec.allowed_ids()
        )
        self.output_specs = dict(
            output_specs or {"text": ActionSpec(max_tokens=min(512, model.config.max_capacity))}
        )
        for modality, spec in self.output_specs.items():
            if modality != ("text" if isinstance(spec, ActionSpec) else spec.modality):
                raise ValueError("Output capability and decoder specification disagree")
            if isinstance(spec, MediaOutputSpec):
                from megatoken.model.media import validate_public_media_spec

                validate_public_media_spec(spec)
                model.media_realizers.resident(modality)
        self.chunking = (chunking or ChunkingConfig()).with_hard_limit(model.config.max_capacity)
        self.memory = TransducerInferenceEngine(
            generator, collections=collections, preparer=preparer, memory_root=memory_root
        )

    def _output_capacity(self, session, tracks, specs=None, modality_options=None):
        """Reserve physical-output memory from public geometry before selection."""
        max_piece = self.max_piece_bytes

        def bound(spec):
            if isinstance(spec, MediaOutputSpec):
                return 4 * (
                    spec.sample_count
                    if spec.modality == "audio"
                    else spec.frames * 3 * spec.height * spec.width
                )
            copy = max((len(source.payload) for source in spec.copy_sources), default=0)
            fixed = max(
                (
                    sum(
                        len((field.fixed_json or "").encode()) * 6
                        + len(field.name.encode()) * 6
                        + 8
                        for field in schema.fields
                    )
                    for schema in spec.schemas
                ),
                default=0,
            )
            return fixed + spec.max_tokens * max(max_piece, copy, 1) * 6

        plans = (
            tuple(specs)
            if specs is not None
            else (ActionSpec(max_tokens=self.generator.generator.config.max_capacity),)
            * len(tracks)
        )
        options = modality_options or tuple((spec,) for spec in plans)
        sizes = [max(bound(spec) for spec in choices) for choices in options]
        if len(sizes) != len(tracks) or any(size > self.max_event_output_bytes for size in sizes):
            raise ValueError("Event output exceeds its byte budget; request shorter media chunks")
        pending = 0 if session.pending is None else session.pending.output_bytes
        if (
            sum(session.reserved_output_bytes.values()) + pending + sum(sizes)
            > self.max_session_output_bytes
        ):
            raise ValueError(
                "Retained output byte budget exhausted; release completed outputs before selecting more"
            )
        return sum(sizes)

    def open(
        self,
        metadata: dict[str, object] | None = None,
        *,
        bases: Mapping[str, float] | None = None,
        trained: float = 1.0,
        effort: int = 6,
        collections: Sequence[str] | None = None,
        max_sources: int | None = None,
        threshold: float | None = None,
        overlay: str | None = None,
        scope: str | None = None,
        write_policy=None,
        memory_writes: bool = True,
    ) -> TrackSession:
        """Commit the metadata prefix and the first lookup, before any answer.

        Training shows the model the fields ``task``, ``modality`` and
        ``version``, each present or absent at random, so pass any of them
        (``metadata_payload`` lists the rest, among them an optional ``request``)
        or none; unknown fields stay null. A question needs no field: the first
        user message is itself a lookup boundary.
        Supplying ``scope`` automatically mounts and writes that scope's private memory;
        configure ``memory_root`` when constructing the runtime. ``memory_writes=False``
        disables automatic memory. Surprise writing remains off by default.
        ``bases`` mounts the engine's open knowledge bases by name with blend weights
        (``memory.open_base`` / ``create_base``); ``trained`` weights the trained base.
        A session searches the mounted ``collections`` it names (default: all);
        the items it retrieves are in ``session.memory.sources.log``, each with
        the original file it came from.
        """
        options = {} if max_sources is None else {"max_sources": max_sources}
        session = TrackSession(
            self.memory.new_session(
                metadata or {},
                bases=bases,
                trained=trained,
                effort=effort,
                collections=collections,
                threshold=threshold,
                overlay=overlay,
                scope=scope,
                write_policy=write_policy,
                memory_writes=memory_writes,
                **options,
            ),
            tracks=TrackRegistry(max_active=self.generator.generator.config.max_tracks),
            context_archive=ContextArchive(
                max_bytes=self.context_archive_bytes, max_events=self.context_archive_events
            ),
        )
        session.memory.state = self.generator.tracks.describe_track(
            session.state, MAIN_TRACK, MAIN_TRACK_PURPOSE
        )
        return session

    @_commit_owner
    def configure_context(
        self,
        session: TrackSession,
        *,
        track: str,
        local_events: int,
        cross_events: int,
        pinned_events: Sequence[str] = (),
    ):
        """Record a bounded public context change after validating its references.

        Expanding a window grants access to still resident context. It cannot
        fabricate data already evicted; those sources need an explicit new read.
        """
        state = session.state
        if track not in state.track_registry and track not in session.tracks.tracks:
            raise ValueError("Context policy requires a declared track")
        if any(
            type(n) is not int or not 0 <= n <= self.generator.generator.config.max_history_chunks
            for n in (local_events, cross_events)
        ):
            raise ValueError("Context request exceeds the session event budget")
        values = dict(
            local_events=local_events, cross_events=cross_events, pinned_events=tuple(pinned_events)
        )
        policy = state.context_policy.configure(track, **values)
        updated = self.generator.tracks.set_context_policy(state, policy)
        # Public announcement and cache read use the newly accepted policy.
        # The same typed announcement is consumed by ordinary training replay.
        payload = json.dumps({"context_window": values}, sort_keys=True).encode()
        event = ObservedEvent(
            track,
            "text",
            payload,
            ROLE_TO_ID["system"],
            event_id=f"context:{session.revision}:{session.work_sequence}",
        )
        session.memory.state = updated
        try:
            self.observe(session, (event,), effort=1)
        except Exception:
            session.memory.state = state
            raise
        return self.context_status(session, track=track)

    @_commit_owner
    def context_status(
        self, session: TrackSession, *, track: str | None = None, include_sources=False
    ):
        state = session.state
        cache_bytes = sum(
            t.numel() * t.element_size() for t in (state.history.keys, state.history.values)
        )
        states = [state, *(group.state for group in session.selected_groups.values())]
        if session.pending is not None:
            states.append(session.pending.result.state)
        for work in session.rendering.values():
            states.extend((work.selection.before, work.selection.state))
            if work.result is not None:
                states.append(work.result.state)
        storage = {}
        for retained in states:
            for tensor in (retained.history.keys, retained.history.values):
                owner = tensor.untyped_storage()
                storage[(str(tensor.device), owner.data_ptr())] = owner.nbytes()
        result = {
            "policy": json.loads(state.context_policy.to_json()),
            "resident_events": len(state.history_event_ids),
            "cache_bytes": cache_bytes,
            "cache_storage_bytes_including_snapshots": sum(storage.values()),
            "event_budget": self.generator.generator.config.max_history_chunks,
            "archived_events": len(session.context_archive.events),
            "host_payload_bytes": session.context_archive.payload_bytes,
            "last_cache_evictions": [
                {
                    "event_id": event,
                    "reason": reason,
                    "retrievable": event in session.context_archive.events,
                }
                for event, reason in state.context_dropped
            ],
        }
        if track is not None:
            mask = self.generator.tracks.context_visibility(state, (track,))[0]
            result["visible_events"] = [
                event
                for event, visible in zip(state.history_event_ids, mask, strict=True)
                if visible
            ]
        if include_sources:
            purposes = {address: purpose for address, purpose, _ in state.track_descriptions}
            result["recent_sources"] = [
                {
                    "event_id": event_id,
                    "purpose": purposes.get(event.track, "Available session evidence"),
                    "modality": event.modality,
                }
                for event_id, (event, _) in list(session.context_archive.events.items())[-8:]
            ]
        return result

    @_commit_owner
    def fetch_context(self, session: TrackSession, *, track: str, source_event: str, effort=1):
        """Reobserve an admitted source after an explicit request on its consumer track."""
        if track not in session.state.track_registry and track not in session.tracks.tracks:
            raise ValueError("Context retrieval requires a declared track")
        event = session.context_archive.get(source_event)
        origin = "session_archive"
        if event is None and session.memory.sources is not None:
            from megatoken.model.sources import source_event as read_source

            found = [
                entry for entry in session.memory.sources.log if entry.event_id == source_event
            ]
            if len(found) == 1:
                event = read_source(session.memory.sources, found[0], session.state.device)
                origin = "mounted_source"
        if event is None:
            raise ValueError(
                "Requested context is unavailable; request its original source or clarification"
            )
        payload = event.payload
        if isinstance(payload, torch.Tensor):
            payload = payload.to(session.state.device)
        event_id = f"context-read:{session.revision}:{session.work_sequence}"
        self.observe(
            session,
            (
                replace(
                    event,
                    track=track,
                    event_id=event_id,
                    payload=payload,
                    role_id=ROLE_TO_ID["document"],
                    valid_mask=None
                    if event.valid_mask is None
                    else event.valid_mask.to(session.state.device),
                ),
            ),
            effort=effort,
        )
        return {"event_id": event_id, "source_event_id": source_event, "origin": origin}

    def add_files(
        self,
        paths: Sequence[str | Path],
        *,
        collection: str | None = None,
        base: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AddedFiles:
        """Add files to a mounted collection and/or an open base (see ``TransducerInferenceEngine.add_files``)."""
        return self.memory.add_files(paths, collection=collection, base=base, metadata=metadata)

    @_commit_owner
    def observe(
        self, session: TrackSession, events: Sequence[ObservedEvent], *, effort: int = 6
    ) -> MultiTrackState:
        """Only already available inputs may become shared causal observations.

        Each text event is one physical chunk, kept whole; ``observe_text`` cuts a
        longer message into chunks.
        """
        arrived_at_ns = time.monotonic_ns()
        ready = tuple(events)
        if not ready:
            raise ValueError("A committed meridian must contain one or more events")
        ready = tuple(
            replace(event, event_id=event.event_id or f"observed:{session.revision}:{index}")
            for index, event in enumerate(ready)
        )
        ids = [e.event_id for e in ready]
        if len(set(ids)) != len(ids) or set(ids) & session.occupied_events:
            raise ValueError("Observed events need new distinct stable event addresses")
        codec = self.generator.generator.codec
        for event in ready:
            if event.modality == "text" and (
                len(event.payload) > self.chunking.max_bytes
                or len(codec.encode(event.payload)) > self.chunking.hard_max_tokens
            ):
                raise ValueError(
                    "Text event exceeds the served chunk policy; observe_text cuts it into chunks"
                )
        from megatoken.data.context import apply_context_announcement

        previous = session.state
        updated = previous
        for event in ready:
            if event.modality == "text" and event.role_id == ROLE_TO_ID["system"]:
                updated = apply_context_announcement(
                    self.generator, updated, event.track, event.payload, "system"
                )
        session.memory.state = updated
        try:
            self.memory.observe(session.memory, ready, effort=effort)
        except Exception:
            session.memory.state = previous
            raise
        session.revision += 1
        session.pending = None
        session.events.append((session.revision, tuple(x.track for x in ready)))
        session.history.extend(x.payload for x in ready if x.modality == "text")
        session.completed_events.update(ids)
        session.occupied_events.update(ids)
        for event in ready:
            session.context_archive.put(event)
            session.event_arrived_at_ns[event.event_id] = arrived_at_ns
        self._prune_event_arrival_times(session)
        return session.state

    @staticmethod
    def _prune_event_arrival_times(session: TrackSession) -> None:
        retained = set(session.state.history_event_ids) | set(session.state.frontier_event_ids)
        for visible in session.selection_visible_event_ids.values():
            retained.update(visible)
        for event_id in session.event_arrived_at_ns.keys() - retained:
            session.event_arrived_at_ns.pop(event_id, None)

    def observe_text(
        self,
        session: TrackSession,
        track: str,
        payload: bytes,
        *,
        role: str = "user",
        effort: int = 6,
        content_kind: str = "auto",
        language: str | None = None,
        filename: str | None = None,
        mime_type: str | None = None,
        locale: str | None = None,
    ) -> int:
        """Observe one message of ``role`` on ``track``, cut into chunks by the served policy.

        The chunks are consecutive meridians and only the last ends the message, so a
        finished message is what a lookup boundary sees. The payload is one authoritative
        event: chat delimiters inside it are content. Returns the number of chunks. An
        error part-way leaves the chunks before it observed.
        """
        if role not in ROLE_TO_ID:
            raise ValueError(f"Unknown physical-record role {role!r}")
        chunks = segment_bytes(
            payload,
            self.generator.generator.codec,
            self.chunking,
            content_kind=content_kind,
            language=language,
            filename=filename,
            mime_type=mime_type,
            locale=locale,
            chat_template="none",
        )
        for index, chunk in enumerate(chunks, 1):
            event = ObservedEvent(
                track, "text", chunk.payload, ROLE_TO_ID[role], terminal=index == len(chunks)
            )
            self.observe(session, (event,), effort=effort)
        return len(chunks)

    @_commit_owner
    def speculate(
        self,
        session: TrackSession,
        *,
        tracks: Sequence[str],
        event_ids: Sequence[str] | None = None,
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        modality_options: Sequence[Sequence[ActionSpec | MediaOutputSpec]] | None = None,
        controls: Sequence[bytes] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> PendingGroup:
        """Produce a candidate without committing raw state, KV or output."""
        output_bytes = self._output_capacity(session, tracks, specs, modality_options)
        if (
            len(session.selected_groups.keys() | session.rendering.keys())
            >= self.max_retained_outputs
        ):
            raise ValueError(
                "Retained output capacity exhausted; release delivered output snapshots"
            )
        for address in tracks:
            if address in session.tracks.tracks and not session.tracks.selectable(
                address, session.completed_events
            ):
                raise ValueError("Track is closed, waiting or has unfinished prerequisites")
        for address in set(tracks):
            if address in session.tracks.tracks and session.tracks.tracks[
                address
            ].budget < tracks.count(address):
                raise ValueError("Track work budget exhausted before selection")
        visible_event_ids = tuple(
            dict.fromkeys((*session.state.history_event_ids, *session.state.frontier_event_ids))
        )
        selected = self.generator.select_group(
            session.state,
            tracks=tracks,
            event_ids=event_ids,
            specs=specs,
            modality_options=modality_options,
            controls=controls,
            effort=effort,
            generator=generator,
        )
        selected_at_ns = time.monotonic_ns()
        result = self.generator.render_selected(selected)
        branch = PendingGroup(
            session.revision,
            result,
            output_bytes,
            selected_at_ns,
            selected.specs,
            visible_event_ids,
        )
        session.pending = branch
        return branch

    @_commit_owner
    def accept(self, session: TrackSession, branch: PendingGroup) -> GroupGeneration:
        """Atomic group acceptance; stale or foreign branches are never merged."""
        if session.pending is not branch or branch.base_revision != session.revision:
            raise ValueError("Speculation is stale or no longer the active candidate")
        result = branch.result
        if result.tape.meridian_id != int(session.state.meridians_seen[0]):
            raise ValueError("Speculative branch has the wrong causal history")
        if set(result.tape.event_ids) & session.occupied_events:
            raise ValueError("Selected events reuse previously committed event addresses")
        for address in set(result.tape.tracks):
            if address in session.tracks.tracks and session.tracks.tracks[
                address
            ].budget < result.tape.tracks.count(address):
                raise ValueError("Track work budget exhausted before acceptance")
        # The lookup this meridian makes due thinks as hard as the meridian did.
        self.memory.generated(
            session.memory, result.state, result.tape.terminal, effort=result.tape.effort
        )
        session.tracks.charge(result.tape.tracks)
        session.revision += 1
        session.pending = None
        session.events.append((session.revision, result.tape.tracks))
        session.selected_groups[session.revision] = result
        session.reserved_output_bytes[session.revision] = branch.output_bytes
        session.selection_at_ns[session.revision] = branch.selected_at_ns or time.monotonic_ns()
        session.selection_visible_event_ids[session.revision] = frozenset(branch.visible_event_ids)
        session.completed_events.update(result.tape.event_ids)
        session.occupied_events.update(result.tape.event_ids)
        session.history.extend(
            payload
            for payload, modality in zip(result.tape.payloads, result.tape.modalities, strict=True)
            if modality == "text"
        )
        self._archive_outputs(session, result)
        return result

    @staticmethod
    def _archive_outputs(session, group):
        for track, event_id, modality, payload, valid in zip(
            group.tape.tracks,
            group.tape.event_ids,
            group.tape.modalities,
            group.tape.payloads,
            group.tape.valid,
            strict=True,
        ):
            if modality == "text" and valid and payload:
                session.context_archive.put(
                    ObservedEvent(
                        track,
                        "text",
                        payload,
                        ROLE_TO_ID["assistant"],
                        event_id=event_id,
                    )
                )

    def generate(
        self,
        session: TrackSession,
        *,
        tracks: Sequence[str],
        event_ids: Sequence[str] | None = None,
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        modality_options: Sequence[Sequence[ActionSpec | MediaOutputSpec]] | None = None,
        controls: Sequence[bytes] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> GroupGeneration:
        return self.accept(
            session,
            self.speculate(
                session,
                tracks=tracks,
                event_ids=event_ids,
                specs=specs,
                modality_options=modality_options,
                controls=controls,
                effort=effort,
                generator=generator,
            ),
        )

    @_commit_owner
    def cancel(self, session: TrackSession) -> None:
        session.pending = None

    @_commit_owner
    def enqueue(self, session: TrackSession, work: ReadyWork):
        if len(session.queue) >= 64:
            raise ValueError("Ready-work queue exhausted; drain ready work before admission")
        if (
            not work.event_id
            or work.event_id in session.occupied_events
            or any(w.event_id == work.event_id for w in session.queue)
        ):
            raise ValueError("Queued work requires a new stable event address")
        if work.track not in session.tracks.tracks:
            raise ValueError("Queued work requires a declared track purpose")
        session.queue.append(work)
        session.scheduling_log.append(("queued", (work.event_id,)))

    @_commit_owner
    def schedule_ready(
        self,
        session: TrackSession,
        *,
        max_events: int = 4,
        effort: int = 6,
        before_select: Callable[[Sequence[ReadyWork]], None] | None = None,
    ):
        if not 1 <= max_events <= self.generator.generator.config.max_history_chunks:
            raise ValueError("Ready group exceeds the retained-history event capacity")
        candidates = []
        for index, work in enumerate(session.queue):
            track = session.tracks.tracks[work.track]
            if work.task_revision != track.revision or track.status in {"completed", "cancelled"}:
                session.scheduling_log.append(("stale", (work.event_id,)))
                continue
            candidates.append((index, work))
        # Drop stale requests explicitly, preserving a deterministic outcome record.
        session.queue[:] = [work for _, work in candidates]
        ready = [
            (i, w)
            for i, w in candidates
            if set(w.prerequisites) <= session.completed_events
            and session.tracks.selectable(w.track, session.completed_events)
        ]
        # Periodically serve the oldest ready work; priority protects urgent
        # user responses while sustained lower-priority work cannot starve.
        ordered = sorted(
            ready,
            key=lambda pair: (
                0 if session.scheduling_steps % 8 == 7 else pair[1].priority,
                pair[0],
            ),
        )
        chosen = []
        for _, work in ordered:
            # Producers and consumers cannot be selected as simultaneous peers.
            if set(work.prerequisites) & {w.event_id for w in chosen}:
                continue
            count = sum(w.track == work.track for w in chosen)
            if session.tracks.tracks[work.track].budget > count:
                chosen.append(work)
            if len(chosen) == max_events:
                break
        if not chosen:
            return None
        if before_select is not None:
            before_select(tuple(w for _, w in chosen))
        revision = self.start_generation(
            session,
            tracks=tuple(w.track for w in chosen),
            event_ids=tuple(w.event_id for w in chosen),
            specs=tuple(w.spec for w in chosen),
            controls=tuple(w.control for w in chosen),
            modality_options=tuple(w.modality_options or (w.spec,) for w in chosen),
            effort=effort,
        )
        ids = {w.event_id for w in chosen}
        session.queue[:] = [w for w in session.queue if w.event_id not in ids]
        session.scheduling_steps += 1
        session.scheduling_log.append(("selected", tuple(w.event_id for w in chosen)))
        return revision

    @_commit_owner
    def start_generation(self, session: TrackSession, **kwargs) -> int:
        """Commit selected content before starting bounded asynchronous rendering."""
        output_bytes = self._output_capacity(
            session, kwargs["tracks"], kwargs.get("specs"), kwargs.get("modality_options")
        )
        if (
            len(session.selected_groups.keys() | session.rendering.keys())
            >= self.max_retained_outputs
        ):
            raise ValueError(
                "Retained output capacity exhausted; release delivered output snapshots"
            )
        if sum(w.status == "rendering" for w in session.rendering.values()) >= self.max_rendering:
            raise ValueError("Rendering capacity exhausted; queue selection until work drains")
        tracks = tuple(kwargs["tracks"])
        for address in tracks:
            if address in session.tracks.tracks and not session.tracks.selectable(
                address, session.completed_events
            ):
                raise ValueError("Track is closed, waiting or has unfinished prerequisites")
        # Check budget before spending work on selection.
        for address in set(tracks):
            if address in session.tracks.tracks and session.tracks.tracks[
                address
            ].budget < tracks.count(address):
                raise ValueError("Track work budget exhausted before selection")
        if not self.render_capacity.acquire(blocking=False):
            raise ValueError("Global rendering queue exhausted; wait for active decoders")
        try:
            visible_event_ids = frozenset(
                (*session.state.history_event_ids, *session.state.frontier_event_ids)
            )
            selected = self.generator.select_group(session.state, **kwargs)
            selected_at_ns = time.monotonic_ns()
            if set(selected.tape.event_ids) & session.occupied_events:
                raise ValueError("Selected events reuse previously committed event addresses")
            lookup = self.memory.generated(
                session.memory, selected.state, selected.tape.terminal, effort=selected.tape.effort
            )
            session.tracks.charge(tracks)
            session.revision += 1
            session.pending = None
            revision = session.revision
            session.selection_at_ns[revision] = selected_at_ns
            session.selection_visible_event_ids[revision] = visible_event_ids
            session.reserved_output_bytes[revision] = output_bytes
            session.events.append((revision, selected.tape.tracks))
            session.occupied_events.update(selected.tape.event_ids)
            # A renderer never receives mutable live session state.
            try:
                future = self.render_executor.submit(self.generator.render_selected, selected)
            except RuntimeError as exc:
                future = Future()
                future.set_exception(exc)
            future.add_done_callback(lambda _: self.render_capacity.release())
            session.rendering[revision] = RenderWork(
                selected, future, completion_committed=lookup is not None
            )
            return revision
        except BaseException:
            self.render_capacity.release()
            raise

    @_commit_owner
    def poll_generation(self, session: TrackSession, revision: int) -> RenderWork:
        work = session.rendering[revision]
        if work.status == "rendering" and work.future.done():
            try:
                work.result = work.future.result()
            except Exception as exc:
                work.status = "failed"
                work.error = f"{type(exc).__name__}: {exc}"
            else:
                if not work.completion_committed:
                    self.memory.generated_completion(
                        session.memory,
                        work.result.tape.roles,
                        work.result.tape.terminal,
                        effort=work.result.tape.effort,
                    )
                    work.completion_committed = True
                work.status = "rendered"
                session.selected_groups[revision] = work.result
                session.completed_events.update(work.result.tape.event_ids)
                session.history.extend(
                    p
                    for p, m in zip(
                        work.result.tape.payloads, work.result.tape.modalities, strict=True
                    )
                    if m == "text"
                )
                self._archive_outputs(session, work.result)
        if work.status == "failed" and not work.outcome_published:
            self.observe_text(
                session,
                work.selection.tape.tracks[0],
                json.dumps(
                    {"rendering_failed": list(work.selection.tape.event_ids), "error": work.error}
                ).encode(),
                role="environment",
                effort=work.selection.tape.effort,
            )
            work.outcome_published = True
        return work

    @_commit_owner
    def stop_delivery(self, session: TrackSession, revision: int) -> RenderWork:
        work = self.poll_generation(session, revision)
        work.future.cancel()  # running decoder computation may not be cancellable
        work.status = "cancelled"
        if not work.stop_published:
            self.observe_text(
                session,
                work.selection.tape.tracks[0],
                json.dumps(
                    {
                        "delivery_stopped": list(work.selection.tape.event_ids),
                        "delivered_extents": work.delivery_progress,
                    },
                    sort_keys=True,
                ).encode(),
                role="environment",
                effort=work.selection.tape.effort,
            )
            work.stop_published = True
        return work

    @_commit_owner
    def delivered(
        self,
        session: TrackSession,
        revision: int,
        event_ids: Sequence[str],
        *,
        extents: Mapping[str, int] | None = None,
    ):
        work = self.poll_generation(session, revision)
        if work.status not in {"rendered", "delivered"}:
            raise ValueError("Only completed, uncancelled rendering can be delivered")
        if not set(event_ids) <= set(work.selection.tape.event_ids):
            raise ValueError("Delivery must name original selected event addresses")
        if extents is not None and set(extents) != set(event_ids):
            raise ValueError("Partial delivery must specify an extent for each acknowledged event")
        updates = []
        for event_id in event_ids:
            index = work.selection.tape.event_ids.index(event_id)
            spec = work.selection.specs[index]
            if isinstance(spec, MediaOutputSpec):
                bound, unit = (
                    (spec.sample_count, "samples")
                    if spec.modality == "audio"
                    else (spec.frames, "frames")
                )
            else:
                bound, unit = len(work.result.tape.payloads[index]), "bytes"
            extent = bound if extents is None else extents[event_id]
            if (
                type(extent) is not int
                or not work.result.tape.valid[index]
                or not work.delivery_progress.get(event_id, 0) < extent <= bound
            ):
                raise ValueError("Delivery extent must advance within the actual rendered event")
            updates.append((event_id, extent, bound, unit))
        for event_id, extent, bound, unit in updates:
            work.delivery_progress[event_id] = extent
            work.delivery_log.append(
                {
                    "event_id": event_id,
                    "delivered": extent,
                    "extent": bound,
                    "unit": unit,
                    "available_at_ns": time.monotonic_ns(),
                }
            )
            if extent == bound:
                work.delivered_events.add(event_id)
        if work.delivered_events == set(work.selection.tape.event_ids):
            work.status = "delivered"

    @_commit_owner
    def translation_timing_report(
        self,
        session: TrackSession,
        revision: int,
        source_event_ids: Sequence[str],
        output_event_id: str,
    ) -> dict:
        """Measure elapsed source-ingest-to-selection/delivery clocks; never score quality."""
        if not source_event_ids or len(set(source_event_ids)) != len(source_event_ids):
            raise ValueError("Timing report requires distinct source event IDs")
        if any(event_id not in session.event_arrived_at_ns for event_id in source_event_ids):
            raise ValueError("Timing report references an unobserved source event")
        group = session.selected_groups.get(revision)
        work = session.rendering.get(revision)
        tape = group.tape if group is not None else (None if work is None else work.selection.tape)
        if tape is None or output_event_id not in tape.event_ids:
            raise ValueError(
                "Timing report output is absent, stale, or belongs to another selection"
            )
        selected_at = session.selection_at_ns.get(revision)
        if selected_at is None:
            raise ValueError("Timing report selection clock is unavailable")
        source_arrival = max(session.event_arrived_at_ns[event_id] for event_id in source_event_ids)
        visible = session.selection_visible_event_ids.get(revision, frozenset())
        if not set(source_event_ids) <= visible:
            raise ValueError("Timing report source evidence was not visible to the selected output")
        if source_arrival > selected_at:
            raise ValueError("Selected output predates its declared source evidence")
        delivered_at = None
        delivered_extent = None
        if work is not None:
            acknowledgements = [
                row
                for row in work.delivery_log
                if row["event_id"] == output_event_id and row["delivered"] > 0
            ]
            if acknowledgements:
                latest = max(acknowledgements, key=lambda row: row["available_at_ns"])
                delivered_at = latest["available_at_ns"]
                delivered_extent = latest["delivered"]
        return {
            "clock": "monotonic_elapsed_v1",
            "source_event_ids": list(source_event_ids),
            "output_event_id": output_event_id,
            "selection_revision": revision,
            "source_arrived_at_ns": source_arrival,
            "selected_at_ns": selected_at,
            "delivery_acknowledged_at_ns": delivered_at,
            "delivery_acknowledged_extent": delivered_extent,
            "source_to_selection_seconds": (selected_at - source_arrival) / 1e9,
            "source_to_delivery_ack_seconds": None
            if delivered_at is None
            else (delivered_at - source_arrival) / 1e9,
            "delivery_ack_is_client_ack_not_audio_playback": True,
        }

    @_commit_owner
    def release_generation(self, session: TrackSession, revision: int):
        """Release physical payloads after the caller archived the returned tape."""
        if revision in session.rendering:
            work = self.poll_generation(session, revision)
            no_undelivered_content = work.result is not None and not any(
                valid and payload and event_id not in work.delivered_events
                for event_id, valid, payload in zip(
                    work.result.tape.event_ids,
                    work.result.tape.valid,
                    work.result.tape.payloads,
                    strict=True,
                )
            )
            if (
                work.status not in {"delivered", "cancelled", "failed"}
                and not (work.status == "rendered" and no_undelivered_content)
            ) or not work.future.done():
                raise ValueError("Finish or stop delivery before releasing rendering memory")
            del session.rendering[revision]
        session.selected_groups.pop(revision, None)
        session.reserved_output_bytes.pop(revision, None)
        session.selection_at_ns.pop(revision, None)
        session.selection_visible_event_ids.pop(revision, None)
        self._prune_event_arrival_times(session)

    def close(self):
        self.render_executor.shutdown(wait=True, cancel_futures=True)
        self.memory.close()

    @_commit_owner
    def track_action(
        self, session: TrackSession, operation: str, *, address: str | None = None, **values
    ):
        """Apply a validated lifecycle action, publishing its meaning before work."""
        for key in ("capabilities", "prerequisites"):
            if key in values:
                values[key] = tuple(values[key])
        registry = session.tracks
        snapshot = (
            dict(registry.tracks),
            list(registry.transitions),
            dict(registry.admissions),
            dict(registry.dependencies),
            dict(registry.completions),
        )
        try:
            if operation == "create":
                if address is not None and address != MAIN_TRACK:
                    raise ValueError("Only the main conversation may adopt its existing address")
                while (
                    f"{registry.address_prefix}:{registry.sequence}" in session.state.track_registry
                ):
                    registry.sequence += 1
                ingress_key = values.pop("ingress_key", None)
                if ingress_key is not None:
                    description = registry.admit_execution(ingress_key, **values)
                else:
                    description = registry.create(
                        **values, **({"address": address} if address is not None else {})
                    )
            else:
                if address is None:
                    raise ValueError("Lifecycle action needs its returned track address")
                description = registry.change(address, operation, **values)
            # Publish only semantic fields. The address attaches the event to
            # its task through attention equality, never instruction bytes.
            public = {
                k: v
                for k, v in asdict(description).items()
                if k not in {"address", "parent", "prerequisites"}
            }
            window = None
            if description.status in {"completed", "cancelled"}:
                window = {
                    "local_events": 0,
                    "cross_events": 0,
                    "pinned_events": [],
                }
            elif operation == "resume":
                config = self.generator.generator.config
                window = {
                    "local_events": min(config.context_local_events, config.max_history_chunks),
                    "cross_events": min(config.context_cross_events, config.max_history_chunks),
                    "pinned_events": [],
                }
            payload = json.dumps(public, sort_keys=True).encode()
            self.observe_text(session, description.address, payload, role="system", effort=1)
            session.memory.state = self.generator.tracks.describe_track(
                session.state, description.address, description.purpose, description.revision
            )
            if window is not None:
                self.configure_context(session, track=description.address, **window)
        except Exception:
            # Published chunks remain causal facts. An allocated address is
            # consumed even if the rest of its announcement could not publish.
            (
                registry.tracks,
                registry.transitions,
                registry.admissions,
                registry.dependencies,
                registry.completions,
            ) = snapshot
            raise
        return description

    @_commit_owner
    def register_execution_context(self, session: TrackSession, *, program: dict, invocation: dict):
        """Register immutable host context without executing code or exposing future state."""
        if (
            not isinstance(program, dict)
            or not isinstance(invocation, dict)
            or program.get("schema") != "semantic-execution/4"
            or program.get("kind") != "program"
            or program.get("mode") != "observed"
            or invocation.get("schema") != "semantic-execution/4"
            or invocation.get("kind") != "invocation"
            or invocation.get("mode") != "observed"
            or not isinstance(program.get("source"), str)
            or hashlib.sha256(program["source"].encode()).hexdigest() != program.get("digest")
            or invocation.get("sourceDigest") != program["digest"]
            or invocation.get("programId") != program.get("programId")
        ):
            raise ValueError("Execution context requires exact observed program and invocation")

        def canonical(item):
            return json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)

        program = json.loads(canonical(program))
        invocation = json.loads(canonical(invocation))
        program_id = program["programId"]
        input_ref = hashlib.sha256(canonical(invocation["inputGraph"]).encode()).hexdigest()
        key = invocation["invocationId"] + ":" + input_ref
        if (
            program_id in session.execution_programs
            and session.execution_programs[program_id] != program
        ):
            raise ValueError("Published execution program is immutable")
        if key in session.execution_inputs and session.execution_inputs[key] != invocation:
            raise ValueError("Published execution invocation is immutable")
        session.execution_programs[program_id] = program
        session.execution_inputs[key] = invocation
        return input_ref

    @_commit_owner
    def bootstrap_execution(self, session: TrackSession, address: str):
        """Publish code, then invocation, before making a reserved child runnable."""
        track = session.tracks.tracks[address]
        binding = track.execution
        if binding is None:
            raise ValueError("Execution bootstrap requires a typed binding")
        if track.bootstrap_ready:
            return track
        try:
            program = session.execution_programs[binding.program_id]
            invocation = session.execution_inputs[binding.invocation_id + ":" + binding.input_ref]
            if (
                program["digest"] != binding.source_digest
                or invocation["realmId"] != binding.realm_id
                or invocation["entry"] != binding.entry
                or invocation["completionPolicy"] != binding.completion_mode
                or not set(binding.capability_grants) <= set(program["capabilities"])
            ):
                raise ValueError("Execution bootstrap context differs from its typed binding")
            progress = session.execution_publication.get(address, 0)
            for index, event in enumerate((program, invocation)):
                if index < progress:
                    continue
                before = set(session.occupied_events)
                self.observe_text(
                    session,
                    address,
                    json.dumps(event, sort_keys=True, separators=(",", ":")).encode(),
                    role="user",
                    effort=1,
                )
                if index == 0:
                    session.execution_source_events[address] = tuple(
                        identity
                        for identity in session.context_archive.events
                        if identity not in before
                    )
                session.execution_publication[address] = index + 1
            return session.tracks.activate_execution(
                address, source_digest=binding.source_digest, input_ref=binding.input_ref
            )
        except Exception as error:
            if session.tracks.tracks[address].status not in {"completed", "cancelled"}:
                session.tracks.change(address, "cancel")
            receipt = {
                "schema": "semantic-bootstrap/4",
                "track": address,
                "invocation": binding.invocation_id,
                "status": "failed",
                "error": str(error),
            }
            session.execution_receipts.append(receipt)
            raise ValueError("Execution bootstrap failed; reservation cancelled") from error

    @_commit_owner
    def complete_execution(self, session: TrackSession, address: str, *, kind: str, value):
        """Deliver child completion independently of its creation tool result."""
        receipt = session.tracks.complete_execution(address, kind=kind, value=value)
        key = receipt["dependency"]["id"]
        if key in session.execution_delivered:
            return receipt
        target = receipt["target"]
        if target is not None:
            owner = session.tracks.tracks.get(target)
            if owner is None:
                raise ValueError("Completion destination has no runtime owner")
            if owner.status not in {"cancelled", "completed"}:
                self.observe_text(
                    session,
                    target,
                    json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(),
                    role="environment",
                    effort=1,
                )
        session.execution_receipts.append(receipt)
        session.execution_delivered.add(key)
        session.completed_events.add(key)
        session.tracks.wake_ready_executions(delivered=session.completed_events)
        return receipt

    @_commit_owner
    def wake_execution(self, session: TrackSession, address: str, *, work: ReadyWork | None = None):
        track = session.tracks.wake_execution(address, delivered=session.completed_events)
        if work is not None:
            if work.track != address or work.task_revision != track.revision:
                raise ValueError("Wake output belongs to another track or stale task revision")
            self.enqueue(session, work)
        return track

    def _publish_tool_result(
        self, session: TrackSession, action_id: str, payload: bytes, *, effort: int = 6
    ):
        address, _ = session.tool_routes[action_id]
        chunks = segment_bytes(
            payload, self.generator.generator.codec, self.chunking, chat_template="none"
        )
        beginning = session.tool_publication.get(action_id, 0)
        for index in range(beginning, len(chunks)):
            self.observe(
                session,
                (
                    ObservedEvent(
                        address,
                        "text",
                        chunks[index].payload,
                        ROLE_TO_ID["tool"],
                        terminal=index + 1 == len(chunks),
                        event_id=f"result:{action_id}:part:{index}",
                    ),
                ),
                effort=effort,
            )
            session.tool_publication[action_id] = index + 1
        session.completed_events.add(f"result:{action_id}")
        session.occupied_events.add(f"result:{action_id}")

    @_commit_owner
    def commit_lifecycle_teacher(
        self,
        session: TrackSession,
        *,
        track: str,
        payload: bytes,
        policy_digest: str,
        decision: int,
    ) -> LifecycleTeacherAction:
        """Collect an explicit trusted coordinator policy through the real executor.

        This is a data-collection authority: it records observed teacher
        continuation and cannot manufacture a model action tape or dispatch a
        guest semantic request. The collector retains the policy and independent
        child admission outside predictor inputs.
        """
        import jsonschema

        from megatoken.lifecycle import lifecycle_tools
        from megatoken.tool_schema import parse_tool_action

        if (
            not isinstance(policy_digest, str)
            or len(policy_digest) != 64
            or any(char not in "0123456789abcdef" for char in policy_digest)
            or type(decision) is not int
            or decision < 0
        ):
            raise ValueError("Teacher orchestration requires a pinned policy and decision index")
        name, arguments, _, _ = parse_tool_action(payload)
        definitions = {
            tool["name"]: tool["parameters"] for tool in lifecycle_tools(tuple(self.output_specs))
        }
        if name not in definitions:
            raise ValueError("Coordinator teacher authority is restricted to lifecycle tools")
        jsonschema.validate(arguments, definitions[name])
        identity = f"teacher:{policy_digest}:{decision}"
        previous = session.teacher_lifecycle_actions.get(identity)
        if previous is not None:
            if previous.tape.payloads != (payload,) or previous.tape.tracks != (track,):
                raise ValueError("Teacher retry rewrote its committed decision")
            return previous
        if not session.tracks.selectable(track, session.completed_events):
            raise ValueError("Coordinator teacher cannot select closed or blocked work")
        if session.tracks.tracks[track].budget < 1:
            raise ValueError("Coordinator teacher exhausted its work budget")
        revision = session.revision
        chunks = segment_bytes(
            payload, self.generator.generator.codec, self.chunking, chat_template="none"
        )
        for index, chunk in enumerate(chunks):
            self.observe(
                session,
                (
                    ObservedEvent(
                        track,
                        "text",
                        chunk.payload,
                        ROLE_TO_ID["assistant"],
                        terminal=index + 1 == len(chunks),
                        event_id=f"{identity}:part:{index}",
                    ),
                ),
                effort=1,
            )
        session.tracks.charge((track,))
        action = LifecycleTeacherAction(
            LifecycleRecordedTape(revision, (track,), (identity,), (payload,)),
            policy_digest,
            decision,
        )
        session.teacher_lifecycle_actions[identity] = action
        return action

    @_commit_owner
    def commit_fixture_action(self, session, *, track, payload, fixture_digest, decision, family):
        """Commit an explicit controlled selection at the engineering input boundary.

        This produces no model-selected or source-teacher provenance. The public
        semantic serving entry point gates access behind a separate fixture flag.
        Invalid fixture bytes are retained just like invalid model selections.
        """
        if (
            not isinstance(payload, bytes)
            or len(payload) > 65536
            or not isinstance(fixture_digest, str)
            or len(fixture_digest) != 64
            or any(char not in "0123456789abcdef" for char in fixture_digest)
            or type(decision) is not int
            or decision < 0
            or family not in {"lifecycle", "semantic"}
        ):
            raise ValueError(
                "Controlled lifecycle input needs exact bounded bytes and fixture identity"
            )
        identity = f"controlled:{family}:{track}:{fixture_digest}:{decision}"
        previous = session.controlled_fixture_actions.get(identity)
        if previous is not None:
            if previous.tape.payloads != (payload,) or previous.tape.tracks != (track,):
                raise ValueError("Controlled retry rewrote its committed decision")
            return previous
        if not session.tracks.selectable(track, session.completed_events):
            raise ValueError("Controlled selection cannot select closed or blocked work")
        revision = session.revision
        chunks = segment_bytes(
            payload, self.generator.generator.codec, self.chunking, chat_template="none"
        )
        for index, chunk in enumerate(chunks):
            self.observe(
                session,
                (
                    ObservedEvent(
                        track,
                        "text",
                        chunk.payload,
                        ROLE_TO_ID["assistant"],
                        terminal=index + 1 == len(chunks),
                        event_id=f"{identity}:part:{index}",
                    ),
                ),
                effort=1,
            )
        session.tracks.charge((track,))
        action = ControlledFixtureAction(
            LifecycleRecordedTape(
                revision,
                (track,),
                (identity,),
                (payload,),
                emitted_ids=(tuple(self.generator.generator.codec.encode(payload)),),
            ),
            fixture_digest,
            decision,
        )
        session.controlled_fixture_actions[identity] = action
        return action

    @_commit_owner
    def execute_lifecycle(
        self,
        session: TrackSession,
        group: GroupGeneration | LifecycleTeacherAction | ControlledFixtureAction,
    ):
        """Execute selected, validated control messages through the same policy as tools."""
        import jsonschema

        from megatoken.lifecycle import lifecycle_tools
        from megatoken.tool_schema import parse_tool_action

        if not any(
            value is group
            for registry in (
                session.selected_groups,
                session.teacher_lifecycle_actions,
                session.controlled_fixture_actions,
            )
            for value in registry.values()
        ):
            raise ValueError("Lifecycle execution requires committed selected actions")
        tools = {
            tool["name"]: tool["parameters"] for tool in lifecycle_tools(tuple(self.output_specs))
        }
        outcomes = []
        bootstraps = []
        for index, payload in enumerate(group.tape.payloads):
            if not group.tape.valid[index] or group.tape.modalities[index] != "text":
                continue
            try:
                name, arguments, _, _ = parse_tool_action(payload)
            except (ValueError, KeyError, TypeError):
                continue
            if name not in tools:
                continue
            try:
                jsonschema.validate(arguments, tools[name])
            except jsonschema.ValidationError:
                continue  # Malformed like an unparseable action: skipped, later actions still run.
            action_id = f"{group.tape.meridian_id}:{group.tape.event_ids[index]}"
            if action_id in session.tool_outcomes:
                continue
            address = group.tape.tracks[index]
            # Keep attachment to the requesting task even when it closes itself.
            session.tool_routes[action_id] = (address, group.tape.event_ids[index])
            try:
                if name == "track_join":
                    if arguments["address"] != address:
                        raise ValueError("A selected join must suspend its own parent track")
                    description = session.tracks.wait_for_execution(
                        address,
                        arguments["completion_dependency"],
                        delivered=session.completed_events,
                    )
                    outcome = {"action_id": action_id, "track": asdict(description)}
                elif name == "track_context_status":
                    outcome = self.context_status(
                        session, track=arguments["address"], include_sources=True
                    )
                elif name == "track_fetch":
                    target = arguments.pop("address")
                    outcome = self.fetch_context(session, track=target, **arguments)
                elif name == "track_context":
                    target = arguments.pop("address")
                    # A selected worker action can only hand off evidence it
                    # could itself read. Host configuration is separately trusted.
                    visible = self.context_status(session, track=address)["visible_events"]
                    if any(event not in visible for event in arguments["pinned_events"]):
                        raise ValueError("Context handoff requires evidence visible to the sender")
                    outcome = self.configure_context(session, track=target, **arguments)
                elif name == "track_output":
                    target = session.tracks.tracks[arguments["address"]]
                    if not session.tracks.selectable(target.address, session.completed_events):
                        raise ValueError("Output work requires an active ready track")
                    kinds = (
                        tuple(self.output_specs)
                        if arguments["modality"] == "auto"
                        else (arguments["modality"],)
                    )
                    options = tuple(self.output_specs[k] for k in kinds if k in target.capabilities)
                    if not options:
                        raise ValueError("No requested output capability is allowed for this track")
                    event_id = f"work:{session.work_sequence}"
                    session.work_sequence += 1
                    self.enqueue(
                        session,
                        ReadyWork(
                            event_id,
                            target.address,
                            options[0],
                            arguments["request"].encode(),
                            task_revision=target.revision,
                            modality_options=options,
                        ),
                    )
                    outcome = {
                        "action_id": action_id,
                        "event_id": event_id,
                        "track": target.address,
                        "status": "queued",
                    }
                else:
                    if (
                        name == "track_create"
                        and arguments.get("parent") is None
                        and address in session.tracks.tracks
                    ):
                        # Work started by a track descends from it: it cannot grant a child
                        # what it does not itself hold by leaving the parent out.
                        arguments = {**arguments, "parent": address}
                        if arguments.get("budget", 128) > session.tracks.tracks[address].budget:
                            raise ValueError(
                                "Child work cannot exceed its parent's remaining work budget"
                            )
                    description = self.track_action(
                        session,
                        name.removeprefix("track_"),
                        **arguments,
                        **(
                            {"ingress_key": action_id}
                            if name == "track_create" and arguments.get("execution")
                            else {}
                        ),
                    )
                    outcome = {"action_id": action_id, "track": asdict(description)}
                    if description.execution is not None and name in {
                        "track_create",
                        "track_assign",
                    }:
                        outcome["completion_dependency"] = session.tracks.completion_dependency(
                            description.address
                        )
                        bootstraps.append(description.address)
                status = "completed"
            except (ValueError, KeyError) as exc:
                outcome = {"action_id": action_id, "error": str(exc)}
                status = "failed"
            session.tool_ready[action_id] = json.dumps(outcome, sort_keys=True).encode()
            session.tool_outcomes[action_id] = status
            outcomes.append(outcome)
        # Outcomes are observed after the complete selected group, never as a
        # hidden input to one of the peer predictions.
        for action_id, payload in tuple(session.tool_ready.items()):
            self._publish_tool_result(session, action_id, payload)
            del session.tool_ready[action_id]
        for address in bootstraps:
            try:
                self.bootstrap_execution(session, address)
            except ValueError:
                # The failure receipt and cancelled reservation remain visible;
                # creation acknowledgement is never rewritten as completion.
                parent = session.tracks.tracks[address].parent
                if parent is not None:
                    self.observe_text(
                        session,
                        parent,
                        json.dumps(session.execution_receipts[-1], sort_keys=True).encode(),
                        role="environment",
                        effort=1,
                    )
        return outcomes

    @_commit_owner
    def prepare_source(self, session: TrackSession, *, track: str, event_id: str, runner, prepare):
        """Run preparation without owning model state; publish readiness on collection.

        ``prepare`` must finish atomic collection publication and return its
        immutable receipt. Only collect_tools integrates that receipt into the
        shared causal history, so ongoing rendering never blocks ingestion.
        """
        if not event_id or event_id in session.occupied_events:
            raise ValueError("Source preparation needs a fresh event identity")
        action_id = f"source:{event_id}"
        if action_id in session.tool_routes:
            raise ValueError("Source preparation already admitted")

        def work():
            receipt = prepare()
            if (
                not isinstance(receipt, dict)
                or not receipt.get("item_id")
                or type(receipt.get("revision")) is not int
                or receipt["revision"] < 1
                or type(receipt.get("available_at_ns")) is not int
            ):
                raise ValueError(
                    "Source preparation did not return a committed publication receipt"
                )
            return json.dumps({"source_ready": receipt}, sort_keys=True).encode()

        runner.submit(action_id, work)
        session.tool_routes[action_id] = (track, event_id)
        session.occupied_events.add(event_id)
        return action_id

    @_commit_owner
    def dispatch_tools(
        self, session: TrackSession, group: GroupGeneration, runner, adapters, *, schemas=None
    ):
        """Dispatch only accepted, typed model calls through configured adapters.

        Adapter functions explicitly declare the allowed external capability.
        The immutable selected payload is the call's complete input.
        """
        if not any(value is group for value in session.selected_groups.values()):
            raise ValueError("External execution requires a committed selected group")
        from megatoken.tool_schema import parse_tool_action

        for index, payload in enumerate(group.tape.payloads):
            if not group.tape.valid[index] or not payload or group.tape.modalities[index] != "text":
                continue
            try:
                name, arguments, _, _ = parse_tool_action(payload)
            except (ValueError, KeyError, TypeError):
                continue
            if name.startswith("track_"):
                continue
            if name not in adapters:
                raise ValueError("Generated tool capability is not installed")
            if schemas is not None:
                import jsonschema

                jsonschema.validate(arguments, schemas[name])
            address = group.tape.tracks[index]
            if (
                address in session.tracks.tracks
                and name not in session.tracks.tracks[address].capabilities
            ):
                raise ValueError("Generated tool exceeds the track's allowed capabilities")
            action_id = f"{group.tape.meridian_id}:{group.tape.event_ids[index]}"
            if action_id in session.tool_routes:
                continue
            function = adapters[name]
            if hasattr(function, "execute_selected"):
                runner.submit(
                    action_id,
                    lambda fn=function, identity=action_id, raw=payload: fn.execute_selected(
                        action_id=identity, payload=raw
                    ),
                )
            else:
                runner.submit(action_id, lambda fn=function, args=arguments: fn(args))
            session.tool_routes[action_id] = (address, group.tape.event_ids[index])

    @_commit_owner
    def dispatch_semantic_requests(
        self,
        session: TrackSession,
        group: GroupGeneration,
        runner,
        adapters,
        *,
        verify_source,
        fragments=None,
        allow_controlled_fixture=False,
    ) -> tuple[str, ...]:
        """Dispatch derived, fully committed semantic tool actions without
        re-encoding selected witness bytes as model-generated tool JSON.

        The caller supplies an independent selected-state/source validator.
        That validator must check the invocation, source callsite, grants and
        all committed argument dependencies. This method never authors a new
        group, teacher tool payload, or predicted return.
        """
        from megatoken.data.semantic_request import _host_request
        from megatoken.data.witness_v4 import WitnessV4Error, parse_record

        committed = dict(session.selected_groups)
        if allow_controlled_fixture:
            committed.update(
                {
                    value.tape.meridian_id: value
                    for value in session.controlled_fixture_actions.values()
                }
            )
        if not any(value is group for value in committed.values()):
            raise ValueError("Semantic requests require a committed selected group")
        if not callable(verify_source):
            raise ValueError("Semantic bridge requires an independent source/ownership validator")
        payloads = group.tape.payloads
        if fragments is not None:
            parts = tuple(fragments)
            revisions = []
            if not parts or parts[-1] is not group or len(parts) > 64:
                raise ValueError("Logical semantic action needs its final committed fragment")
            for index, part in enumerate(parts):
                revision = next((key for key, value in committed.items() if value is part), None)
                if (
                    revision is None
                    or len(part.tape.payloads) != 1
                    or part.tape.tracks != group.tape.tracks
                    or part.tape.modalities != ("text",)
                    or part.tape.valid != (True,)
                    or part.tape.terminal != (index == len(parts) - 1,)
                ):
                    raise ValueError("Incomplete, foreign or uncommitted semantic fragment")
                revisions.append(revision)
            if any(
                right != left + 1 for left, right in zip(revisions, revisions[1:], strict=False)
            ):
                raise ValueError("A logical selection cannot incorporate interleaved observations")
            payloads = (b"".join(part.tape.payloads[0] for part in parts),)
            if len(payloads[0]) > 65536:
                raise ValueError("Logical semantic request exceeds its declared byte budget")
        dispatched: list[str] = []
        for index, raw in enumerate(payloads):
            if not isinstance(raw, bytes) or not raw.startswith(b"request "):
                continue
            if not group.tape.valid[index] or group.tape.modalities[index] != "text":
                raise ValueError("Incomplete or invalid semantic requests cannot dispatch")
            try:
                kind, fields = parse_record(raw.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, WitnessV4Error) as error:
                raise ValueError("Malformed committed semantic request") from error
            if kind != "request":
                raise ValueError("Selected action must be a request witness")
            operation = fields["operation"]
            address = group.tape.tracks[index]
            # The source validator operates on the already committed selection
            # and explicit track, and MUST establish source/realm ownership.
            verified = verify_source(address, fields)
            if verified is not True and not (
                isinstance(verified, dict) and verified.get("verified") is True
            ):
                raise ValueError("Selected request failed source, capability or realm checks")
            _host_request(fields, heap=verified.get("heap") if isinstance(verified, dict) else None)
            registry = getattr(session, "tracks", None)
            if registry is not None:
                owner = registry.tracks.get(address)
                if owner is None or owner.status not in {"active", "waiting"}:
                    raise ValueError("Selected request has no live runtime owner")
                if owner.execution and operation not in owner.execution.capability_grants:
                    raise ValueError("Selected request exceeds its runtime-established grant")
            adapter = adapters.get(operation)
            if adapter is None or not callable(getattr(adapter, "execute_selected", None)):
                raise ValueError(
                    "Native semantic adapter must accept validated original witness bytes"
                )
            action_id = f"{group.tape.meridian_id}:{group.tape.event_ids[index]}"
            if action_id in session.tool_routes:
                continue  # Already submitted; never repeat a native write.
            # Store original routing before transport submission. A submitter
            # that might have started work cannot make a retry duplicate it.
            session.tool_routes[action_id] = (address, group.tape.event_ids[index])
            runner.submit(
                action_id,
                lambda fn=adapter, identity=action_id, payload=raw: fn.execute_selected(
                    action_id=identity, payload=payload
                ),
            )
            dispatched.append(action_id)
        return tuple(dispatched)

    @_commit_owner
    def collect_tools(self, session: TrackSession, runner, *, effort: int = 6):
        results, failures = runner.collect_ready()
        for action_id in (*results, *failures):
            if action_id not in session.tool_routes:
                raise ValueError("Tool result has no original selected action")
            if action_id in session.tool_outcomes:
                continue
            if action_id in failures:
                session.tool_errors[action_id] = str(failures[action_id])
                session.tool_ready[action_id] = json.dumps(
                    {"error": str(failures[action_id])}
                ).encode()
            else:
                session.tool_ready[action_id] = results[action_id]
        # Retain ready outcomes if publication fails; collection never loses an
        # executed effect merely because ingestion ran out of capacity.
        for action_id, payload in tuple(session.tool_ready.items()):
            self._publish_tool_result(session, action_id, payload, effort=effort)
            session.tool_outcomes[action_id] = (
                "failed" if action_id in session.tool_errors else "completed"
            )
            del session.tool_ready[action_id]
        return results, failures


class ConcurrentToolRunner:
    """IO-only parallel calls; results enter KV only through observe()."""

    def __init__(
        self, max_workers: int = 4, max_pending: int = 64, call_timeout: float | None = 120.0
    ) -> None:
        if not 1 <= max_workers <= 32:
            raise ValueError("Tool runner requires 1 to 32 IO workers")
        if max_pending < max_workers:
            raise ValueError("Tool queue must accommodate its active workers")
        if call_timeout is not None and not call_timeout > 0:
            raise ValueError("A tool call timeout must be positive")
        self.max_pending = max_pending
        self.call_timeout = call_timeout
        self.seen: set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending: dict[str, Future[bytes]] = {}
        self.started: dict[str, float] = {}

    def submit(self, action_id: str, function: Callable[[], bytes]) -> None:
        if not action_id or action_id in self.seen:
            raise ValueError("Parallel action requires a new stable action ID")
        if len(self.pending) >= self.max_pending:
            raise ValueError(
                "Tool request queue exhausted; retry admission after collecting results"
            )
        self.pending[action_id] = self.executor.submit(function)
        self.started[action_id] = time.monotonic()
        self.seen.add(action_id)

    def collect_ready(self) -> tuple[dict[str, bytes], dict[str, BaseException]]:
        """Finished calls, removed from the pending set: their bytes, and the errors that ended others.

        One failed call must not hide the results of the others that finished with it.
        """
        results: dict[str, bytes] = {}
        failures: dict[str, BaseException] = {}
        now = time.monotonic()
        for action_id, future in tuple(self.pending.items()):
            if not future.done():
                if self.call_timeout is not None and now - self.started[action_id] > (
                    self.call_timeout
                ):
                    # A thread cannot be interrupted: stop waiting for it (an unstarted
                    # call is cancelled) and report the call as failed, so the model is
                    # told instead of waiting forever. A late result is discarded.
                    future.cancel()
                    del self.pending[action_id]
                    del self.started[action_id]
                    failures[action_id] = TimeoutError(
                        f"Tool call exceeded its {self.call_timeout:g} s limit"
                    )
                continue
            del self.pending[action_id]
            del self.started[action_id]
            error = (
                CancelledError("Tool computation cancelled before execution")
                if future.cancelled()
                else future.exception()
            )
            if error is None and not isinstance(future.result(), bytes):
                error = TypeError("Executed tool result must return raw bytes")
            if error is None:
                results[action_id] = future.result()
            else:
                failures[action_id] = error
        return results, failures

    def close(self) -> None:
        # Do not wait for a hung tool: closing a session must always return.
        self.executor.shutdown(wait=False, cancel_futures=True)
