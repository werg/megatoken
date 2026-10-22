"""One canonical causal runtime for grouped observations and media generation.

A session owns its committed multitrack state and its transducer agenda: every
committed ordinary meridian may make a bank lookup due, served before the
next one; a completed assistant turn or an arriving tool, document or
environment result always does. Ready tool results are observed in later
meridians. Speculative branches are copy-on-write and can be rejected
without disturbing the committed session.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from megatoken.data.knowledge import FilePreparer
from megatoken.generation.structure import ActionSpec
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import (
    GroupGeneration,
    MultiTrackGenerator,
    MultiTrackState,
    ObservedEvent,
)
from megatoken.transducer_service import TransducerInferenceEngine, TransducerSession


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

    @property
    def state(self) -> MultiTrackState:
        return self.memory.state


class MultiTrackRuntime:
    """Reference scheduler: one ready meridian at a time, no global track barrier."""

    def __init__(
        self,
        generator: MultiTrackGenerator,
        *,
        preparer: Callable[[], FilePreparer] | None = None,
    ) -> None:
        self.generator = generator
        self.memory = TransducerInferenceEngine(generator, preparer=preparer)

    def open(
        self,
        metadata: dict[str, object] | None = None,
        *,
        bases: Mapping[str, float] | None = None,
        trained: float = 1.0,
        effort: int = 6,
    ) -> TrackSession:
        """Commit the metadata prefix and the first lookup, before any answer.

        Training shows the model the fields ``task``, ``modality`` and
        ``version``, each present or absent at random, so pass any of them
        (``metadata_payload`` lists the rest) or none; unknown fields stay null.
        ``bases`` mounts the engine's open knowledge bases by name with blend weights
        (``memory.open_base`` / ``create_base``); ``trained`` weights the trained base.
        """
        return TrackSession(
            self.memory.new_session(metadata or {}, bases=bases, trained=trained, effort=effort)
        )

    def observe(
        self, session: TrackSession, events: Sequence[ObservedEvent], *, effort: int = 6
    ) -> MultiTrackState:
        """Only already available inputs may become shared causal observations."""
        ready = tuple(events)
        if not ready:
            raise ValueError("A committed meridian must contain one or more events")
        self.memory.observe(session.memory, ready, effort=effort)
        session.revision += 1
        session.pending = None
        session.events.append((session.revision, tuple(x.track for x in ready)))
        return session.state

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
