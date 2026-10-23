"""Frozen local student actor for executed student-state correction collection."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from megatoken.contracts import CONVERSATION_PROFILE_VERSION, MAIN_TRACK, ROLE_TO_ID

from .collection import TeacherCall, TeacherEvent, TeacherRequest, TeacherResponse
from .schema import canonical_json

# Match ordinary preparation: roles share the continuing conversation's context.
# Explicit grouped outputs retain their own worker addresses.
STUDENT_TRACK = MAIN_TRACK


def _group_event_ids(
    request: TeacherRequest, outputs: tuple[dict[str, Any], ...]
) -> tuple[str, ...]:
    """Give each grouped draw fresh addresses while preserving its requested track."""
    namespace = hashlib.sha256(f"{request.session_id}:{request.request_id}".encode()).hexdigest()[
        :20
    ]
    return tuple(
        f"{namespace}:{output.get('event_id', f'event-{index}')}"
        for index, output in enumerate(outputs)
    )


class ModelTeacher:
    """Adapt the serving runtime into a frozen student, rolling out exactly as served.

    One active task session is retained. A rollout therefore runs the behaviour a deployed
    model has and training replays: the bank lookups and adapter mixture at the same
    boundaries, and any collections or bases mounted through ``open_options``. Every draw
    advances the session exactly once, including invalid decodes; only later actual
    environment records are observed on the next request. The group tape of the draw (one
    single-track meridian) and its emitted IDs travel into the collection journal.
    Validation gates execution, never state advance. Loading a served bundle is deliberate:
    its checksum identifies the frozen student and its codec, independent of training-only
    checkpoint state.
    """

    def __init__(
        self,
        bundle: Any,
        *,
        device: str = "auto",
        effort: int = 6,
        seed: int = 0,
        collections: Mapping[str, str | Path] | None = None,
        open_options: Mapping[str, Any] | None = None,
        media_preparer: Any = None,
        packs: str | Path | None = None,
    ):
        """``bundle`` is a served bundle directory (or an already loaded ``ServedBundle``).

        ``collections`` are mounted for the student's life; ``open_options`` are what every
        session opens with (``metadata``, ``collections``, ``bases``, ``threshold``, ...,
        see ``MultiTrackRuntime.open``).
        """
        from megatoken.model.multitrack import MultiTrackGenerator
        from megatoken.multitrack_runtime import MultiTrackRuntime
        from megatoken.served import ServedBundle

        if isinstance(bundle, str | Path):
            bundle = ServedBundle(bundle, packs=packs, device=device)
        self.runtime = MultiTrackRuntime(
            MultiTrackGenerator(bundle.model),
            chunking=bundle.chunking,
            collections=collections,
            record_policy_context=True,
        )
        self.media_preparer = media_preparer
        self._capacity = bundle.model.config.max_capacity
        self.device = bundle.model.core.initial_facets.device
        self.codec = bundle.codec
        self.snapshot = "sha256:" + bundle.manifest["model_sha256"]
        self.model = f"frozen-student:conversation-{CONVERSATION_PROFILE_VERSION}:" + self.snapshot
        self.effort, self.seed = effort, seed
        self.open_options = dict(open_options or {})
        self._session_id: str | None = None
        self._session: Any = None
        self._record_ids: tuple[str, ...] = ()
        self._pending: tuple[tuple[bytes, tuple[int, ...] | None], ...] | None = None
        self._pending_revision: int | None = None
        self._pending_event_ids: tuple[str, ...] = ()
        self._pending_tracks: tuple[str, ...] = ()
        self._action_tracks: dict[str, str] = {}

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        import torch

        from megatoken.controls import build_control_payload
        from megatoken.generation.structure import ActionSpec
        from megatoken.model.multitrack import ObservedEvent
        from megatoken.tool_schema import compile_tool_spec, parse_tool_action, recent_copy_sources

        if request.codec_fingerprint != self.codec.fingerprint:
            raise ValueError("Frozen student and collection meter codecs differ")
        if not request.session_id:
            raise ValueError("Local student inference requires the collection session identity")
        if request.session_id != self._session_id:
            self._session_id = request.session_id
            self._session = self.runtime.open(effort=self.effort, **self.open_options)
            self._record_ids, self._pending = (), None
            self._pending_revision, self._pending_event_ids = None, ()
            self._pending_tracks, self._action_tracks = (), {}
        observed_ids = tuple(record.record_id for record in request.records)
        consumed = len(self._record_ids)
        if observed_ids[:consumed] != self._record_ids:
            raise ValueError("Student history was reordered, replaced, or shortened")
        if self._pending is not None:
            for index, (payload, emitted_ids) in enumerate(self._pending):
                if consumed >= len(request.records):
                    raise ValueError("The student's executed action is missing from actual history")
                record = request.records[consumed]
                if (
                    record.role != "assistant"
                    or record.part_count != 1
                    or record.payload != payload
                    or record.emitted_ids != emitted_ids
                ):
                    raise ValueError("A sampled student action was changed before execution")
                if record.action_id:
                    self._action_tracks[record.action_id] = (
                        self._pending_tracks[index] if self._pending_tracks else MAIN_TRACK
                    )
                consumed += 1
            # The selected bytes are now in immutable collection records. The
            # runtime can discard its retained rendering/tape snapshot.
            if self._pending_revision is not None:
                self.runtime.delivered(
                    self._session, self._pending_revision, self._pending_event_ids
                )
                self.runtime.release_generation(self._session, self._pending_revision)
            else:
                # Synchronous generations retain selected groups by revision.
                for revision in tuple(getattr(self._session, "selected_groups", ())):
                    self.runtime.release_generation(self._session, revision)
            self._pending_revision, self._pending_event_ids = None, ()
            self._pending = None
            self._pending_tracks = ()
        for record in request.records[consumed:]:
            from .media_assets import MEDIA_RECORD_TYPE, MediaRecord

            track = self._action_tracks.get(record.result_of, MAIN_TRACK)

            if record.media_type == MEDIA_RECORD_TYPE:
                if self.media_preparer is None:
                    raise ValueError(
                        "Media observations require the checked collection asset store"
                    )
                reference = MediaRecord.from_bytes(record.payload)
                source = torch.from_numpy(self.media_preparer.store.load(reference.source)).to(
                    self.device
                )[None]
                modality = reference.modality if reference.modality != "text" else "compressed_text"
                event = ObservedEvent(
                    track,
                    modality,
                    source,
                    ROLE_TO_ID[record.role],
                    extractor=reference.source.extractor,
                    terminal=record.part_index + 1 == record.part_count,
                    event_id=record.record_id,
                )
            else:
                event = ObservedEvent(
                    track,
                    "text",
                    record.payload,
                    ROLE_TO_ID[record.role],
                    terminal=record.part_index + 1 == record.part_count,
                    event_id=record.record_id,
                )
            self.runtime.observe(self._session, (event,), effort=self.effort)
            if record.result_of and record.part_index + 1 == record.part_count:
                self._action_tracks.pop(record.result_of, None)
        self._record_ids = observed_ids
        if request.outputs:
            return self._respond_group(request)
        maximum = min(request.max_output_tokens, self._capacity)
        if json.loads(request.tools_json):
            spec = replace(
                compile_tool_spec(request.tools_json, max_tokens=maximum),
                allow_text=True,
                copy_sources=recent_copy_sources(list(request.records)),
            )
        else:
            spec = ActionSpec(max_tokens=maximum)
        draw_seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{request.request_id}".encode()).digest()[:8], "big"
        ) % (2**63)
        group = self.runtime.generate(
            self._session,
            tracks=(STUDENT_TRACK,),
            specs=(spec,),
            controls=(
                build_control_payload(seed=request.control_seed, tools_json=request.tools_json),
            ),
            effort=self.effort,
            generator=torch.Generator(device=self.device).manual_seed(draw_seed),
        )
        tape = group.tape
        payload, emitted_ids, valid = tape.payloads[0], tape.emitted_ids[0], tape.valid[0]
        tape_json = tape.to_json()
        self._pending = ((payload, emitted_ids),) if valid and payload else None
        self._pending_tracks = (MAIN_TRACK,) if self._pending is not None else ()
        # The exact tape has been serialized into the response; selected groups
        # otherwise remain resident for the life of a long collection session.
        revisions = tuple(getattr(self._session, "selected_groups", ()))
        if revisions:
            self.runtime.release_generation(self._session, revisions[-1])
        if not valid:
            return TeacherResponse(native_tape_json=tape_json, finish_reason="invalid")
        try:
            name, arguments, call_id, _ = parse_tool_action(payload)
        except ValueError:
            return TeacherResponse(
                content=payload,
                emitted_ids=emitted_ids,
                native_tape_json=tape_json,
            )
        return TeacherResponse(
            calls=(
                TeacherCall(
                    call_id or "student-" + request.request_id[:16],
                    name,
                    canonical_json(arguments),
                    wire_payload_base64=base64.b64encode(payload).decode(),
                    emitted_ids=emitted_ids,
                ),
            ),
            finish_reason="tool_calls",
            native_tape_json=tape_json,
        )

    def _respond_group(self, request: TeacherRequest) -> TeacherResponse:
        import torch

        from megatoken.controls import build_control_payload
        from megatoken.generation.structure import ActionSpec
        from megatoken.multitrack_service import _output
        from megatoken.tool_schema import parse_tool_action

        outputs = tuple(request.outputs)
        specs, controls, options = [], [], []
        for output in outputs:
            spec, control = _output(output, self._capacity, self._session.history)
            choices = tuple(
                _output({**output, **option}, self._capacity, self._session.history)[0]
                for option in output.get("options", ())
            ) or (spec,)
            if (
                any(not isinstance(choice, ActionSpec) for choice in choices)
                and self.media_preparer is None
            ):
                raise ValueError(
                    "Grouped media collection requires a real teacher/codec preparer before selection"
                )
            specs.append(spec)
            controls.append(
                build_control_payload(seed=request.control_seed, tools_json=request.tools_json)
                + control
            )
            options.append(choices)
        seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{request.request_id}".encode()).digest()[:8], "big"
        ) % (2**63)
        revision = self.runtime.start_generation(
            self._session,
            tracks=tuple(o["track"] for o in outputs),
            event_ids=_group_event_ids(request, outputs),
            specs=tuple(specs),
            modality_options=tuple(options),
            controls=tuple(controls),
            effort=self.effort,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        work = self._session.rendering[revision]
        try:
            work.future.result()
        except Exception:
            pass  # The runtime retains the selection and records rendering failure.
        work = self.runtime.poll_generation(self._session, revision)
        if work.result is None:
            self._pending = None
            self.runtime.release_generation(self._session, revision)
            return TeacherResponse(
                native_tape_json=work.selection.tape.to_json(),
                finish_reason="render_failed",
                events=tuple(
                    TeacherEvent(i, track, modality, b"")
                    for i, (track, modality) in enumerate(
                        zip(work.selection.tape.tracks, work.selection.tape.modalities, strict=True)
                    )
                ),
                outcome_json=canonical_json({"rendering": work.status, "error": work.error}),
            )
        group = work.result
        tape = group.tape
        events, calls, content, pending = [], [], [], []
        for index, (track, modality, payload, tokens, valid) in enumerate(
            zip(
                tape.tracks,
                tape.modalities,
                tape.payloads,
                tape.emitted_ids,
                tape.valid,
                strict=True,
            )
        ):
            call_id, physical = None, payload if valid else b""
            if modality != "text" and valid:
                physical = self.media_preparer.rendered(
                    group.media_outputs[index], work.selection.specs[index]
                ).to_bytes()
                emitted = None
            else:
                emitted = tokens
                if valid and payload:
                    try:
                        name, arguments, wire_id, _ = parse_tool_action(payload)
                    except ValueError:
                        content.append(payload)
                    else:
                        call_id = wire_id or f"student-{request.request_id[:12]}-{index}"
                        calls.append(
                            TeacherCall(
                                call_id,
                                name,
                                canonical_json(arguments),
                                wire_payload_base64=base64.b64encode(payload).decode(),
                                emitted_ids=tokens,
                            )
                        )
            events.append(TeacherEvent(index, track, modality, physical, emitted, call_id))
            if valid and physical:
                pending.append((physical, emitted))
        self._pending = tuple(pending) or None
        self._pending_tracks = tuple(event.track for event in events if event.payload)
        self._pending_revision = revision if pending else None
        self._pending_event_ids = tuple(
            event_id for event_id, valid in zip(tape.event_ids, tape.valid, strict=True) if valid
        )
        if not pending:
            self.runtime.delivered(self._session, revision, self._pending_event_ids)
            self.runtime.release_generation(self._session, revision)
            self._pending_revision, self._pending_event_ids = None, ()
        return TeacherResponse(
            content=b"".join(content),
            calls=tuple(calls),
            events=tuple(events),
            native_tape_json=tape.to_json(),
            finish_reason="tool_calls" if calls else "stop",
            outcome_json=canonical_json({"rendering": "rendered", "selection_revision": revision}),
        )
