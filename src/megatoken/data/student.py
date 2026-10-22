"""Frozen local student actor for executed student-state correction collection."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from megatoken.contracts import ROLE_TO_ID

from .collection import TeacherCall, TeacherRequest, TeacherResponse
from .schema import canonical_json

# The student speaks on one track; what it reads arrives on the track of each record's role,
# exactly as ``sequential_meridians`` groups a collected episode for training.
STUDENT_TRACK = "assistant"


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
            bundle = ServedBundle(bundle, device=device)
        self.runtime = MultiTrackRuntime(
            MultiTrackGenerator(bundle.model), chunking=bundle.chunking, collections=collections
        )
        self._capacity = bundle.model.config.max_capacity
        self.device = bundle.model.core.initial_facets.device
        self.codec = bundle.codec
        self.snapshot = "sha256:" + bundle.manifest["model_sha256"]
        self.model = "frozen-student:" + self.snapshot
        self.effort, self.seed = effort, seed
        self.open_options = dict(open_options or {})
        self._session_id: str | None = None
        self._session: Any = None
        self._record_ids: tuple[str, ...] = ()
        self._pending: tuple[bytes, tuple[int, ...]] | None = None

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
        observed_ids = tuple(record.record_id for record in request.records)
        consumed = len(self._record_ids)
        if observed_ids[:consumed] != self._record_ids:
            raise ValueError("Student history was reordered, replaced, or shortened")
        if self._pending is not None:
            if consumed >= len(request.records):
                raise ValueError("The student's executed action is missing from actual history")
            record = request.records[consumed]
            payload, emitted_ids = self._pending
            if (
                record.role != "assistant"
                or record.part_count != 1
                or record.payload != payload
                or record.emitted_ids != emitted_ids
            ):
                raise ValueError("A sampled student action was changed before execution")
            # The draw already advanced the session using the raw facets;
            # observing their decoded bytes would add a second, different event.
            consumed += 1
        for record in request.records[consumed:]:
            event = ObservedEvent(
                record.role,
                "text",
                record.payload,
                ROLE_TO_ID[record.role],
                terminal=record.part_index + 1 == record.part_count,
            )
            self.runtime.observe(self._session, (event,), effort=self.effort)
        self._record_ids = observed_ids
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
        self._pending = (payload, emitted_ids) if valid and payload else None
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
