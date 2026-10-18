"""Frozen local student actor for executed student-state correction collection."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .collection import TeacherCall, TeacherRequest, TeacherResponse
from .schema import canonical_json, encode_record


class ModelTeacher:
    """Adapt the shared served policy without rebuilding or approximating its state.

    One active task session is retained. Every native facet draw advances state
    exactly once, including invalid decodes; only later actual environment records
    are ingested on the next request. The complete tape and emitted IDs travel
    into the collection journal. Validation gates execution, never state advance.
    Loading a served bundle is deliberate: its checksum identifies the frozen
    student and its codec, independent of training-only checkpoint state.
    """

    def __init__(
        self,
        bundle: str | Path | None = None,
        *,
        device: str = "auto",
        effort: int = 6,
        seed: int = 0,
        _engine: Any | None = None,
    ):
        if _engine is None:
            if bundle is None:
                raise ValueError("A frozen served student bundle is required")
            from megatoken.inference import InferenceEngine

            _engine = InferenceEngine(bundle, device=device)
        self.engine = _engine
        self.codec = _engine.codec
        self.snapshot = "sha256:" + _engine.manifest["model_sha256"]
        self.model = "frozen-student:" + self.snapshot
        self.effort, self.seed = effort, seed
        self._session_id: str | None = None
        self._session: Any = None
        self._record_ids: tuple[str, ...] = ()
        self._pending: tuple[bytes, tuple[int, ...]] | None = None

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        from megatoken.generation.structure import ActionSpec
        from megatoken.tool_schema import compile_tool_spec, parse_tool_action, recent_copy_sources

        if request.codec_fingerprint != self.codec.fingerprint:
            raise ValueError("Frozen student and collection meter codecs differ")
        if not request.session_id:
            raise ValueError("Local student inference requires the collection session identity")
        if request.session_id != self._session_id:
            self._session_id = request.session_id
            self._session = self.engine.new_session()
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
            # engine.generate already advanced state using the raw facets;
            # observing their decoded bytes would add a second, different event.
            consumed += 1
        for record in request.records[consumed:]:
            encoded = encode_record(record, self.codec)
            self.engine.observe_physical(
                self._session, record.payload, role=record.role, token_ids=encoded.token_ids
            )
        self._record_ids = observed_ids
        maximum = min(request.max_output_tokens, self.engine.config.max_capacity)
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
        result = self.engine.generate(
            self._session,
            spec=spec,
            effort=self.effort,
            seed=draw_seed,
            control_seed=request.control_seed,
            tools_json=request.tools_json,
        )
        tape_json = result.tape.to_json()
        self._pending = (
            (result.payload, tuple(result.emitted_ids))
            if result.tape.valid and result.payload
            else None
        )
        if not result.tape.valid:
            return TeacherResponse(native_tape_json=tape_json, finish_reason="invalid")
        try:
            name, arguments, call_id, _ = parse_tool_action(result.payload)
        except ValueError:
            return TeacherResponse(
                content=result.payload,
                emitted_ids=tuple(result.emitted_ids),
                native_tape_json=tape_json,
            )
        return TeacherResponse(
            calls=(
                TeacherCall(
                    call_id or "student-" + request.request_id[:16],
                    name,
                    canonical_json(arguments),
                    wire_payload_base64=base64.b64encode(result.payload).decode(),
                    emitted_ids=tuple(result.emitted_ids),
                ),
            ),
            finish_reason="tool_calls",
            native_tape_json=tape_json,
        )
