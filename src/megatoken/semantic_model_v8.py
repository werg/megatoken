"""Actual native v8 generation with transactional public acknowledgement.

Only the public committed prefix and its independently reconstructed next
context reach the model. A generated candidate lives on a private copy-on-write
runtime branch until the caller commits the exact bytes through its semantic
controller. Rejection preserves the preceding native model continuation.
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field, fields, replace
from threading import Lock, RLock

from megatoken.controls import (
    build_request_control,
    canonical_semantic_observation,
    semantic_execution_metadata_v8,
    semantic_selection_instruction_v8,
)
from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.prefix_v8 import (
    validate_v8_committed_prefix,
    validate_v8_prediction_prefix,
)
from megatoken.data.schema import make_records

ADAPTER = "native-semantic-v8/1"


class NativeSelectionV8Error(ValueError):
    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def _fork_session(session):
    """Fork only the bounded, writeless session profile used by this adapter."""
    from megatoken.multitrack_runtime import TrackSession

    memory = session.memory
    if (
        memory.writes is not None
        or memory.sources is not None
        or session.rendering
        or session.pending
    ):
        raise NativeSelectionV8Error("V8 speculation requires an idle writeless native session")
    branch_memory = replace(
        memory,
        agenda=memory.agenda.copy(),
        scheduler=copy.copy(memory.scheduler),
        lock=Lock(),
    )
    values = {}
    for declaration in fields(TrackSession):
        name = declaration.name
        if name in {"memory", "lock", "tracks", "context_archive"}:
            continue
        value = getattr(session, name)
        values[name] = copy.copy(value) if isinstance(value, (dict, list, set)) else value
    archive = copy.copy(session.context_archive)
    archive.events = session.context_archive.events.copy()
    return TrackSession(
        memory=branch_memory,
        tracks=type(session.tracks).restore(session.tracks.checkpoint()),
        context_archive=archive,
        **values,
    )


@dataclass
class _Session:
    runtime: object
    public: dict | None = None
    contexts: list = field(default_factory=list)
    pending: dict | None = None
    completed: set = field(default_factory=set)


class NativeSemanticInferenceV8:
    """One native continuation per public v8 session, without teacher repair."""

    def __init__(
        self,
        runtime,
        *,
        model_identity,
        effort=1,
        max_tokens=512,
        max_fragments=64,
        max_sessions=16,
        max_attempts=10000,
    ):
        if not re.fullmatch("[0-9a-f]{64}", model_identity or ""):
            raise ValueError("V8 serving requires an immutable loaded model identity")
        if not 1 <= effort <= 6 or not 1 <= max_tokens <= 8192:
            raise ValueError("Invalid v8 effort or per-fragment token budget")
        if (
            not 1 <= max_fragments <= 64
            or not 1 <= max_sessions <= 256
            or not 1 <= max_attempts <= 100000
        ):
            raise ValueError("Invalid v8 serving capacity")
        self.runtime = runtime
        self.codec = runtime.generator.generator.codec
        self.model_identity = model_identity
        self.effort = effort
        self.max_tokens = min(max_tokens, runtime.generator.generator.config.max_capacity)
        self.max_fragments = max_fragments
        self.max_sessions = max_sessions
        self.max_attempts = max_attempts
        self.sessions = {}
        self.attempts = []
        self.openings = []
        self.initialization = None
        self.lock = RLock()

    def initialize(self, *, seed, checkpointIdentity, tokenizerIdentity):
        import random

        import numpy as np
        import torch

        with self.lock:
            self.initialization = None
            if self.sessions:
                raise NativeSelectionV8Error("Close active v8 sessions before reinitialization")
            if type(seed) is not int or seed not in {17, 29, 43}:
                raise NativeSelectionV8Error("V8 evaluation requires a declared seed")
            if (
                checkpointIdentity != self.model_identity
                or tokenizerIdentity != self.codec.fingerprint
            ):
                raise NativeSelectionV8Error("Loaded v8 checkpoint or tokenizer identity differs")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = self.runtime.generator.generator
            self.initialization = {
                "schema": "semantic-inference-identity/8",
                "adapter": ADAPTER,
                "seed": seed,
                "checkpointIdentity": self.model_identity,
                "tokenizerIdentity": self.codec.fingerprint,
                "nativeAdmissionClaim": False,
                "generationSettings": {
                    "device": str(next(model.parameters()).device)
                    if hasattr(model, "parameters")
                    else "test-runtime",
                    "maxTokens": self.max_tokens,
                    "maxFragments": self.max_fragments,
                    "effort": self.effort,
                },
            }
            return dict(self.initialization)

    def _opened(self, identity):
        from megatoken.model.transducer import metadata_payload

        current = self.sessions.get(identity)
        if current is not None:
            return current
        if len(self.sessions) >= self.max_sessions:
            raise NativeSelectionV8Error("V8 session capacity exhausted")
        started = time.perf_counter()
        metadata = semantic_execution_metadata_v8()
        current = _Session(
            self.runtime.open(
                metadata=metadata,
                effort=self.effort,
                memory_writes=False,
                collections=[],
            )
        )
        self.sessions[identity] = current
        payload = metadata_payload(metadata)
        self.openings.append(
            {
                "session": identity,
                "metadataTokens": len(self.codec.encode(payload)),
                "metadataBytes": len(payload),
                "openSeconds": time.perf_counter() - started,
            }
        )
        return current

    def _observe(self, session, payload, role, logical_id):
        """Use the exact physical splitter used by v8 training preparation."""
        from megatoken.contracts import MAIN_TRACK, ROLE_TO_ID
        from megatoken.model.multitrack import ObservedEvent

        chunks = make_records(
            payload,
            logical_id=logical_id,
            role=role,
            codec=self.codec,
            max_tokens=self.runtime.generator.generator.config.max_capacity,
            max_bytes=16384,
            content_kind="text",
        )
        for chunk in chunks:
            self.runtime.observe(
                session,
                (
                    ObservedEvent(
                        MAIN_TRACK,
                        "text",
                        chunk.payload,
                        ROLE_TO_ID[role],
                        terminal=chunk.part_index + 1 == chunk.part_count,
                        event_id=chunk.record_id,
                    ),
                ),
                effort=self.effort,
            )
        return {
            "physicalRecords": len(chunks),
            "tokens": sum(len(chunk.cached_ids) for chunk in chunks),
            "bytes": len(payload),
            "events": [chunk.record_id for chunk in chunks],
        }

    def select(self, *, session, public, contexts, context):
        from megatoken.contracts import MAIN_TRACK
        from megatoken.generation.structure import ActionSpec

        with self.lock:
            if self.initialization is None:
                raise NativeSelectionV8Error("Initialize and attest the native v8 model first")
            checked = validate_v8_prediction_prefix(public, contexts, context)
            if session != checked["session"]:
                raise NativeSelectionV8Error("V8 selection session differs from public authority")
            if len(self.attempts) >= self.max_attempts:
                raise NativeSelectionV8Error("V8 attempt accounting capacity exhausted")
            work = self._opened(session)
            if context["invocationId"] in work.completed:
                raise NativeSelectionV8Error("Completed v8 invocation cannot select another record")
            if work.pending is not None:
                raise NativeSelectionV8Error(
                    "A v8 candidate requires commit or rejection before another selection"
                )
            old = work.public["timeline"] if work.public is not None else []
            rows = public["timeline"]
            if rows[: len(old)] != old or contexts != work.contexts:
                raise NativeSelectionV8Error("V8 selection rewrites acknowledged native history")
            incoming = rows[len(old) :]
            if any(row["mode"] != "observed" for row in incoming):
                raise NativeSelectionV8Error(
                    "V8 model history cannot import unacknowledged generated records"
                )
            branch = _fork_session(work.runtime)
            attempt = {
                "adapter": ADAPTER,
                "selectionId": f"v8:{len(self.attempts)}",
                "session": session,
                "invocationId": context["invocationId"],
                "checkpointIdentity": self.model_identity,
                "tokenizerIdentity": self.codec.fingerprint,
                "codecFingerprint": self.codec.fingerprint,
                "prefixDigest": context["prefixDigest"],
                "contextDigest": checked["contextDigest"],
                "publicDigest": checked["publicDigest"],
                "origin": "actual-model-selected",
                "modelSelected": True,
                "status": "submitted",
                "fragments": [],
                "payloadHex": None,
                "generatedTokens": 0,
                "observedTokens": 0,
                "contextTokens": 0,
                "observedBytes": 0,
                "contextBytes": 0,
                "observedPhysicalRecords": 0,
                "contextPhysicalRecords": 0,
                "instructionTokens": 0,
                "instructionBytes": 0,
                "generationSeconds": 0.0,
                "preparationSeconds": 0.0,
            }
            self.attempts.append(attempt)
            started = time.perf_counter()
            generation_started = None
            try:
                for row in incoming:
                    measured = self._observe(
                        branch,
                        canonical_semantic_observation(row),
                        "user" if row["kind"] in {"program", "invocation"} else "environment",
                        f"semantic:v8:observation:{row['sequence']}",
                    )
                    attempt["observedTokens"] += measured["tokens"]
                    attempt["observedBytes"] += measured["bytes"]
                    attempt["observedPhysicalRecords"] += measured["physicalRecords"]
                measured = self._observe(
                    branch,
                    canonical_semantic_observation(context),
                    "system",
                    f"semantic:v8:context:{len(rows) + 1}",
                )
                attempt["contextTokens"] = measured["tokens"]
                attempt["contextBytes"] = measured["bytes"]
                attempt["contextPhysicalRecords"] = measured["physicalRecords"]
                required_events = set(measured["events"])
                visibility = self.runtime.context_status(branch, track=MAIN_TRACK)
                if not required_events <= set(visibility["visible_events"]):
                    raise NativeSelectionV8Error(
                        "The complete v8 current context does not fit native visible history"
                    )
                attempt["contextBefore"] = visibility
                attempt["preparationSeconds"] = time.perf_counter() - started
                generation_started = time.perf_counter()
                for index in range(self.max_fragments):
                    if index:
                        visibility = self.runtime.context_status(branch, track=MAIN_TRACK)
                        if not required_events <= set(visibility["visible_events"]):
                            raise NativeSelectionV8Error(
                                "The complete v8 context and earlier record fragments "
                                "are no longer visible"
                            )
                    control = build_request_control(
                        instruction=semantic_selection_instruction_v8(continuation=index > 0)
                    )
                    attempt["instructionTokens"] += len(self.codec.encode(control))
                    attempt["instructionBytes"] += len(control)
                    generated = self.runtime.generate(
                        branch,
                        tracks=(MAIN_TRACK,),
                        event_ids=(f"{attempt['selectionId']}:fragment:{index}",),
                        specs=(
                            ActionSpec(
                                text_kind="utf8",
                                min_tokens=1,
                                max_tokens=self.max_tokens,
                                require_terminal=None,
                            ),
                        ),
                        controls=(control,),
                        effort=self.effort,
                        continuation_mode="encoded_text",
                    )
                    tape = generated.tape
                    payload = tape.payloads[0] if len(tape.payloads) == 1 else None
                    ids = tape.emitted_ids[0] if len(tape.emitted_ids) == 1 else ()
                    fragment = {
                        "selectedEventId": tape.event_ids[0] if len(tape.event_ids) == 1 else None,
                        "meridian": tape.meridian_id,
                        "modelTrack": MAIN_TRACK,
                        "payloadHex": payload.hex() if isinstance(payload, bytes) else None,
                        "emittedTokenIds": list(ids),
                        "tokenIds": list(ids),
                        "terminal": tape.terminal[0] if len(tape.terminal) == 1 else None,
                    }
                    attempt["fragments"].append(fragment)
                    attempt["generatedTokens"] += len(ids)
                    attempt["generationSeconds"] = time.perf_counter() - generation_started
                    if (
                        len(tape.payloads) != 1
                        or tape.valid != (True,)
                        or tape.modalities != ("text",)
                        or tape.tracks != (MAIN_TRACK,)
                        or not isinstance(payload, bytes)
                        or not ids
                        or any(type(value) is not int or value < 0 for value in ids)
                        or self.codec.decode(ids) != payload
                    ):
                        raise NativeSelectionV8Error(
                            "Native model emitted an invalid v8 text fragment"
                        )
                    required_events.update(tape.event_ids)
                    original = b"".join(
                        bytes.fromhex(part["payloadHex"]) for part in attempt["fragments"]
                    )
                    attempt["payloadHex"] = original.hex()
                    if len(original) > 65536:
                        raise NativeSelectionV8Error(
                            "Native v8 logical record exceeds its byte budget"
                        )
                    self.runtime.release_generation(branch, branch.revision)
                    if tape.terminal[0]:
                        break
                else:
                    raise NativeSelectionV8Error(
                        "Native v8 record exhausted its fragment budget without termination"
                    )
                record = original.decode("utf-8", errors="strict")
                attempt["contextAfter"] = self.runtime.context_status(branch, track=MAIN_TRACK)
                attempt["status"] = "awaiting-commit"
                attempt["totalSeconds"] = time.perf_counter() - started
                work.pending = {
                    "branch": branch,
                    "checked": checked,
                    "attempt": attempt,
                    "record": record,
                }
                return {
                    "schema": "semantic-model-output/8",
                    "record": record,
                    "selection": copy.deepcopy(attempt),
                }
            except Exception as error:
                if generation_started is not None:
                    attempt["generationSeconds"] = time.perf_counter() - generation_started
                attempt["status"] = "rejected"
                attempt["failure"] = str(error)
                attempt["totalSeconds"] = time.perf_counter() - started
                raise NativeSelectionV8Error(str(error), evidence=copy.deepcopy(attempt)) from error

    def commit(self, *, session, public, contexts, invocationId, record, selection):
        with self.lock:
            work = self.sessions.get(session)
            pending = work.pending if work is not None else None
            if pending is None:
                raise NativeSelectionV8Error("No native v8 candidate awaits acknowledgement")
            attempt, checked = pending["attempt"], pending["checked"]
            if (
                not isinstance(selection, dict)
                or selection.get("selectionId") != attempt["selectionId"]
                or record != pending["record"]
                or invocationId != attempt["invocationId"]
                or any(
                    selection.get(key) != attempt[key]
                    for key in (
                        "checkpointIdentity",
                        "tokenizerIdentity",
                        "prefixDigest",
                        "contextDigest",
                        "payloadHex",
                    )
                )
            ):
                raise NativeSelectionV8Error(
                    "V8 acknowledgement differs from the original native selection"
                )
            previous = checked["public"]["timeline"]
            rows = public.get("timeline", [])
            appended = rows[len(previous) :]
            if (
                rows[: len(previous)] != previous
                or not appended
                or appended[0].get("mode") != "generated"
                or appended[0].get("fields", {}).get("line") != record
                or appended[0].get("fields", {}).get("invocationId") != invocationId
                or appended[0].get("fields", {}).get("session") != session
                or any(row.get("mode") != "observed" for row in appended[1:])
                or contexts != checked["contexts"] + [checked["context"]]
            ):
                raise NativeSelectionV8Error(
                    "V8 acknowledgement must append exactly its checked candidate"
                )
            validate_v8_committed_prefix(public, contexts)
            # The branch already contains the generated bytes. Later host rows
            # are ingested only by the next selection, once independently checked.
            work.runtime = pending["branch"]
            self.runtime.memory.sessions[work.runtime.memory.session_id] = work.runtime.memory
            work.public = copy.deepcopy({**public, "timeline": rows[: len(previous) + 1]})
            work.contexts = copy.deepcopy(contexts)
            work.pending = None
            attempt["status"] = "committed"
            return {
                "schema": "semantic-native-commit/8",
                "selectionId": attempt["selectionId"],
                "publicDigest": stable_algorithm_hash(public),
                "committed": True,
            }

    def reject(self, *, session, selectionId, reason):
        with self.lock:
            work = self.sessions.get(session)
            if (
                work is None
                or work.pending is None
                or work.pending["attempt"]["selectionId"] != selectionId
            ):
                raise NativeSelectionV8Error("No matching v8 candidate can be rejected")
            attempt = work.pending["attempt"]
            attempt.update(status="semantic-rejected", failure=str(reason))
            work.pending = None
            return {
                "schema": "semantic-native-rejection/8",
                "selectionId": selectionId,
                "rejected": True,
            }

    def complete(self, *, session, public, contexts, invocationId, completion=None):
        with self.lock:
            work = self.sessions.get(session)
            if work is None or work.pending is not None or work.public is None:
                raise NativeSelectionV8Error("V8 completion needs acknowledged native selections")
            previous = work.public["timeline"]
            rows = public.get("timeline", [])
            if (
                rows[: len(previous)] != previous
                or contexts != work.contexts
                or any(row.get("mode") != "observed" for row in rows[len(previous) :])
            ):
                raise NativeSelectionV8Error("V8 completion rewrites acknowledged model history")
            validate_v8_committed_prefix(public, contexts)
            generated = [
                row
                for row in rows
                if row["mode"] == "generated" and row["fields"].get("invocationId") == invocationId
            ]
            if not generated:
                raise NativeSelectionV8Error("V8 completion has no generated terminal record")
            terminal = generated[-1]
            if terminal["kind"] == "semantic":
                from megatoken.data.detailed_v7 import parse_record

                ended = parse_record(terminal["fields"]["line"])[0] == "end"
            elif terminal["kind"] == "protocol":
                from megatoken.data.protocol_v8 import decode_protocol_record

                ended = decode_protocol_record(terminal["fields"]["line"])["kind"] == "end"
            else:
                ended = False
            if not ended:
                raise NativeSelectionV8Error("V8 completion precedes its checked terminal record")
            work.completed.add(invocationId)
            # A callback may end while its caller is suspended or a retained
            # handler can still run again. Whole-session closure is explicit.
            return {
                "schema": "semantic-native-completion/8",
                "session": session,
                "invocationId": invocationId,
                "completed": True,
                "sessionRetained": True,
                "publicDigest": stable_algorithm_hash(public),
            }

    def cancel(self, *, session, reason, **unused):
        with self.lock:
            work = self.sessions.pop(session, None)
            if work is not None:
                if work.pending is not None:
                    work.pending["attempt"].update(status="cancelled", failure=str(reason))
                self.runtime.memory.close_session(work.runtime.memory)
            return {
                "schema": "semantic-native-close/8",
                "session": session,
                "closed": True,
                "reason": str(reason),
            }

    def metrics(self):
        with self.lock:
            attempts = copy.deepcopy(self.attempts)
            return {
                "schema": "semantic-native-serving-metrics/8",
                "adapter": ADAPTER,
                "initialization": copy.deepcopy(self.initialization),
                "attempts": attempts,
                "predictionAttempts": len(attempts),
                "committed": sum(row["status"] == "committed" for row in attempts),
                "failedAttempts": sum(
                    row["status"] in {"rejected", "semantic-rejected", "cancelled"}
                    for row in attempts
                ),
                "generatedTokens": sum(row["generatedTokens"] for row in attempts),
                "observedTokens": sum(row["observedTokens"] for row in attempts),
                "contextTokens": sum(row["contextTokens"] for row in attempts),
                "instructionTokens": sum(row["instructionTokens"] for row in attempts),
                "observedBytes": sum(row["observedBytes"] for row in attempts),
                "contextBytes": sum(row["contextBytes"] for row in attempts),
                "instructionBytes": sum(row["instructionBytes"] for row in attempts),
                "sessionOpenings": copy.deepcopy(self.openings),
                "metadataTokens": sum(row["metadataTokens"] for row in self.openings),
                "metadataBytes": sum(row["metadataBytes"] for row in self.openings),
                "generatedPhysicalRecords": sum(len(row["fragments"]) for row in attempts),
                "activeSessions": len(self.sessions),
                "optimizerSteps": 0,
                "learnedSuccess": None,
                "scope": "Actual native runtime input framing, original emitted tokens, wall times and cache snapshots; semantic acceptance requires explicit acknowledgement",
            }
