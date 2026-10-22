"""The one serving runtime: grouped observations, text and media generation.

Every way of running a model goes through it: the ``generate`` and ``serve``
commands, the JSON-lines service, and the frozen student that collects rollouts for
training. A text conversation is a session with one track per role.

A session owns its committed multitrack state and its transducer agenda: every
committed ordinary meridian may make a bank lookup due, served before the
next one; a completed user message or assistant turn, or an arriving tool,
document or environment result, always does. Ready tool results are observed in later
meridians. Speculative branches are copy-on-write and can be rejected
without disturbing the committed session.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.data.chunking import ChunkingConfig, segment_bytes
from megatoken.data.knowledge import FilePreparer
from megatoken.generation.structure import ActionSpec
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import (
    GroupGeneration,
    MultiTrackGenerator,
    MultiTrackState,
    ObservedEvent,
)
from megatoken.transducer_service import (
    AddedFiles,
    TransducerInferenceEngine,
    TransducerSession,
)


@dataclass(frozen=True)
class PendingGroup:
    base_revision: int
    result: GroupGeneration


@dataclass
class TrackSession:
    """A single shared-history context, with no private per-agent transformers."""

    memory: TransducerSession
    revision: int = 0
    pending: PendingGroup | None = None
    events: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)
    # Every text payload committed so far, oldest first: what a tool call may copy from.
    history: list[bytes] = field(default_factory=list)

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
    ) -> None:
        """``chunking`` is the served bundle's physical chunk policy (default: the model's)."""
        self.generator = generator
        model = generator.generator
        self.chunking = (chunking or ChunkingConfig()).with_hard_limit(model.config.max_capacity)
        self.memory = TransducerInferenceEngine(
            generator, collections=collections, preparer=preparer
        )

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
    ) -> TrackSession:
        """Commit the metadata prefix and the first lookup, before any answer.

        Training shows the model the fields ``task``, ``modality`` and
        ``version``, each present or absent at random, so pass any of them
        (``metadata_payload`` lists the rest, among them an optional ``request``)
        or none; unknown fields stay null. A question needs no field: the first
        user message is itself a lookup boundary.
        ``bases`` mounts the engine's open knowledge bases by name with blend weights
        (``memory.open_base`` / ``create_base``); ``trained`` weights the trained base.
        A session searches the mounted ``collections`` it names (default: all);
        the items it retrieves are in ``session.memory.sources.log``, each with
        the original file it came from.
        """
        options = {} if max_sources is None else {"max_sources": max_sources}
        return TrackSession(
            self.memory.new_session(
                metadata or {},
                bases=bases,
                trained=trained,
                effort=effort,
                collections=collections,
                threshold=threshold,
                **options,
            )
        )

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

    def observe(
        self, session: TrackSession, events: Sequence[ObservedEvent], *, effort: int = 6
    ) -> MultiTrackState:
        """Only already available inputs may become shared causal observations.

        Each text event is one physical chunk, kept whole; ``observe_text`` cuts a
        longer message into chunks.
        """
        ready = tuple(events)
        if not ready:
            raise ValueError("A committed meridian must contain one or more events")
        codec = self.generator.generator.codec
        for event in ready:
            if event.modality == "text" and (
                len(event.payload) > self.chunking.max_bytes
                or len(codec.encode(event.payload)) > self.chunking.hard_max_tokens
            ):
                raise ValueError(
                    "Text event exceeds the served chunk policy; observe_text cuts it into chunks"
                )
        self.memory.observe(session.memory, ready, effort=effort)
        session.revision += 1
        session.pending = None
        session.events.append((session.revision, tuple(x.track for x in ready)))
        session.history.extend(x.payload for x in ready if x.modality == "text")
        return session.state

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

    def speculate(
        self,
        session: TrackSession,
        *,
        tracks: Sequence[str],
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        controls: Sequence[bytes] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> PendingGroup:
        """Produce a candidate without committing raw state, KV or output."""
        result = self.generator.generate_group(
            session.state,
            tracks=tracks,
            specs=specs,
            controls=controls,
            effort=effort,
            generator=generator,
        )
        branch = PendingGroup(session.revision, result)
        session.pending = branch
        return branch

    def accept(self, session: TrackSession, branch: PendingGroup) -> GroupGeneration:
        """Atomic group acceptance; stale or foreign branches are never merged."""
        if session.pending is not branch or branch.base_revision != session.revision:
            raise ValueError("Speculation is stale or no longer the active candidate")
        result = branch.result
        if result.tape.meridian_id != int(session.state.meridians_seen[0]):
            raise ValueError("Speculative branch has the wrong causal history")
        # The lookup this meridian makes due thinks as hard as the meridian did.
        self.memory.generated(
            session.memory, result.state, result.tape.terminal, effort=result.tape.effort
        )
        session.revision += 1
        session.pending = None
        session.events.append((session.revision, result.tape.tracks))
        session.history.extend(
            payload
            for payload, modality in zip(result.tape.payloads, result.tape.modalities, strict=True)
            if modality == "text"
        )
        return result

    def generate(
        self,
        session: TrackSession,
        *,
        tracks: Sequence[str],
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        controls: Sequence[bytes] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> GroupGeneration:
        return self.accept(
            session,
            self.speculate(
                session,
                tracks=tracks,
                specs=specs,
                controls=controls,
                effort=effort,
                generator=generator,
            ),
        )

    def cancel(self, session: TrackSession) -> None:
        session.pending = None


class ConcurrentToolRunner:
    """IO-only parallel calls; results enter KV only through observe()."""

    def __init__(self, max_workers: int = 4) -> None:
        if not 1 <= max_workers <= 32:
            raise ValueError("Tool runner requires 1 to 32 IO workers")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending: dict[str, Future[bytes]] = {}

    def submit(self, action_id: str, function: Callable[[], bytes]) -> None:
        if not action_id or action_id in self.pending:
            raise ValueError("Parallel action requires a new stable action ID")
        self.pending[action_id] = self.executor.submit(function)

    def collect_ready(self) -> tuple[dict[str, bytes], dict[str, BaseException]]:
        """Finished calls, removed from the pending set: their bytes, and the errors that ended others.

        One failed call must not hide the results of the others that finished with it.
        """
        results: dict[str, bytes] = {}
        failures: dict[str, BaseException] = {}
        for action_id, future in tuple(self.pending.items()):
            if not future.done():
                continue
            del self.pending[action_id]
            error = future.exception()
            if error is None and not isinstance(future.result(), bytes):
                error = TypeError("Executed tool result must return raw bytes")
            if error is None:
                results[action_id] = future.result()
            else:
                failures[action_id] = error
        return results, failures

    def close(self) -> None:
        self.executor.shutdown(wait=True)
