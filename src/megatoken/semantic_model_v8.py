"""Actual native v8 generation with transactional public acknowledgement.

Only the public committed prefix and its independently reconstructed next
context reach the model. A generated candidate lives on a private copy-on-write
runtime branch until the caller commits the exact bytes through its semantic
controller. Rejection preserves the preceding native model continuation.
"""

from __future__ import annotations

import copy
import math
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
from megatoken.data.prefix_v8 import (
    validate_v8_committed_prefix,
    validate_v8_prediction_prefix,
)
from megatoken.data.schema import make_records
from megatoken.data.semantic_json import stable_algorithm_hash, stable_algorithm_json
from megatoken.semantic_checkpoint_v8 import (
    capture_random_state,
    continuation_fingerprint,
    restore_random_state,
    runtime_identity,
)

ADAPTER = "native-semantic-v8/1"


def _json_nonnegative_integer(value):
    # JSON's 1 and 1.0 are the same ECMAScript number; bool is a separate type.
    return type(value) in (int, float) and 0 <= value <= 2**53 - 1 and value == int(value)


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
    # Acknowledgement may include host rows that the next selection must ingest.
    ingested_through: int = 0
    contexts: list = field(default_factory=list)
    pending: dict | None = None
    completed: set = field(default_factory=set)
    opening_random: dict | None = None
    replay: list = field(default_factory=list)


class NativeSemanticInferenceV8:
    """One native continuation per public v8 session, without teacher repair."""

    def __init__(
        self,
        runtime,
        *,
        model_identity,
        training_identity=None,
        effort=1,
        max_tokens=512,
        max_fragments=64,
        max_sessions=16,
        max_attempts=10000,
    ):
        if not re.fullmatch("[0-9a-f]{64}", model_identity or ""):
            raise ValueError("V8 serving requires an immutable loaded model identity")
        if training_identity is not None and (
            not isinstance(training_identity, str)
            or not re.fullmatch("[0-9a-f]{64}", training_identity)
        ):
            raise ValueError("V8 serving requires a valid training-lineage identity")
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
        # Prepared V8 records may use a stricter physical budget than the model's
        # position capacity. The retained bundle policy owns their boundaries.
        from megatoken.data.chunking import ChunkingConfig

        self.chunking = getattr(runtime, "chunking", None) or ChunkingConfig().with_hard_limit(
            runtime.generator.generator.config.max_capacity
        )
        self.model_identity = model_identity
        self.training_identity = training_identity
        self.effort = effort
        self.max_tokens = min(
            max_tokens,
            runtime.generator.generator.config.max_capacity,
            self.chunking.hard_max_tokens,
        )
        self.max_fragments = max_fragments
        self.max_sessions = max_sessions
        self.max_attempts = max_attempts
        self.sessions = {}
        self.attempts = []
        self.openings = []
        # Closed IDs remain tombstoned for the lifetime of the adapter. A
        # durable checkpoint keeps their final acknowledged cut so restoration
        # can verify it without making a closed session selectable again.
        self.closed_sessions = {}
        self.initialization = None
        self.lock = RLock()

    def initialize(self, *, seed, checkpointIdentity, tokenizerIdentity, trainingIdentity=None):
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
                or trainingIdentity != self.training_identity
            ):
                raise NativeSelectionV8Error(
                    "Loaded v8 checkpoint, tokenizer or training-lineage identity differs"
                )
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
                "trainingIdentity": self.training_identity,
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

        if identity in self.closed_sessions:
            raise NativeSelectionV8Error("Closed v8 session cannot be reopened")
        current = self.sessions.get(identity)
        if current is not None:
            return current
        if len(self.sessions) >= self.max_sessions:
            raise NativeSelectionV8Error("V8 session capacity exhausted")
        started = time.perf_counter()
        opening_random = capture_random_state()
        metadata = semantic_execution_metadata_v8()
        current = _Session(
            self.runtime.open(
                metadata=metadata,
                effort=self.effort,
                memory_writes=False,
                collections=[],
            ),
            opening_random=opening_random,
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
            chunking=self.chunking,
            max_tokens=self.chunking.hard_max_tokens,
            max_bytes=self.chunking.max_bytes,
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
            if session in self.closed_sessions:
                raise NativeSelectionV8Error("Closed v8 session cannot select another record")
            if len(self.attempts) >= self.max_attempts:
                raise NativeSelectionV8Error("V8 attempt accounting capacity exhausted")
            work = self.sessions.get(session)
            if work is not None and context["invocationId"] in work.completed:
                raise NativeSelectionV8Error("Completed v8 invocation cannot select another record")
            if work is not None and work.pending is not None:
                raise NativeSelectionV8Error(
                    "A v8 candidate requires commit or rejection before another selection"
                )
            old = work.public["timeline"] if work is not None and work.public is not None else []
            rows = public["timeline"]
            if stable_algorithm_json(rows[: len(old)]) != stable_algorithm_json(
                old
            ) or stable_algorithm_json(contexts) != stable_algorithm_json(
                work.contexts if work is not None else []
            ):
                raise NativeSelectionV8Error("V8 selection rewrites acknowledged native history")
            incoming = rows[work.ingested_through if work is not None else 0 :]
            if any(row["mode"] != "observed" for row in incoming):
                raise NativeSelectionV8Error(
                    "V8 model history cannot import unacknowledged generated records"
                )
            if work is None:
                work = self._opened(session)
            random_before = capture_random_state()
            branch = _fork_session(work.runtime)
            attempt = {
                "adapter": ADAPTER,
                "selectionId": f"v8:{len(self.attempts)}",
                "session": session,
                "invocationId": context["invocationId"],
                "checkpointIdentity": self.model_identity,
                "tokenizerIdentity": self.codec.fingerprint,
                "trainingIdentity": self.training_identity,
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
                    "randomBefore": random_before,
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
                or stable_algorithm_hash(selection) != stable_algorithm_hash(attempt)
                or record != pending["record"]
                or invocationId != attempt["invocationId"]
            ):
                raise NativeSelectionV8Error(
                    "V8 acknowledgement differs from the original native selection"
                )
            previous = checked["public"]["timeline"]
            rows = public.get("timeline", [])
            appended = rows[len(previous) :]
            if (
                stable_algorithm_json(rows[: len(previous)]) != stable_algorithm_json(previous)
                or not appended
                or appended[0].get("mode") != "generated"
                or appended[0].get("fields", {}).get("line") != record
                or appended[0].get("fields", {}).get("invocationId") != invocationId
                or appended[0].get("fields", {}).get("session") != session
                or any(row.get("mode") != "observed" for row in appended[1:])
                or stable_algorithm_json(contexts)
                != stable_algorithm_json(checked["contexts"] + [checked["context"]])
            ):
                raise NativeSelectionV8Error(
                    "V8 acknowledgement must append exactly its checked candidate"
                )
            validate_v8_committed_prefix(public, contexts)
            # The branch already contains the generated bytes. Later host rows
            # are ingested only by the next selection, once independently checked.
            work.runtime = pending["branch"]
            self.runtime.memory.sessions[work.runtime.memory.session_id] = work.runtime.memory
            work.public = copy.deepcopy(public)
            work.ingested_through = len(previous) + 1
            work.contexts = copy.deepcopy(contexts)
            work.replay.append(
                {
                    "attempt": int(attempt["selectionId"].removeprefix("v8:")),
                    "contextIndex": len(contexts) - 1,
                    "acknowledgedThrough": len(rows),
                    "randomBefore": pending["randomBefore"],
                }
            )
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
                stable_algorithm_json(rows[: len(previous)]) != stable_algorithm_json(previous)
                or stable_algorithm_json(contexts) != stable_algorithm_json(work.contexts)
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
                from megatoken.data.detailed_v8 import parse_record

                ended = parse_record(terminal["fields"]["line"])[0] == "end"
            elif terminal["kind"] == "protocol":
                from megatoken.data.protocol_v8 import decode_protocol_record

                ended = decode_protocol_record(terminal["fields"]["line"])["kind"] == "end"
            else:
                ended = False
            if not ended:
                raise NativeSelectionV8Error("V8 completion precedes its checked terminal record")
            work.public = copy.deepcopy(public)
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

    def _checkpoint_settings(self):
        return {
            "effort": self.effort,
            "maxTokens": self.max_tokens,
            "maxFragments": self.max_fragments,
            "maxSessions": self.max_sessions,
            "maxAttempts": self.max_attempts,
            "runtimeIdentity": runtime_identity(self.runtime),
        }

    def _session_checkpoint(self, identity, work):
        if work.public is not None:
            checked = validate_v8_committed_prefix(work.public, work.contexts)
            if checked["session"] != identity:
                raise NativeSelectionV8Error("Checkpoint public session differs")
        return {
            "public": copy.deepcopy(work.public),
            "contexts": copy.deepcopy(work.contexts),
            "ingestedThrough": work.ingested_through,
            "completed": sorted(work.completed),
            "openingRandom": copy.deepcopy(work.opening_random),
            "replay": copy.deepcopy(work.replay),
            "continuationDigest": continuation_fingerprint(work.runtime),
            "unacknowledgedSelection": work.pending["attempt"]["selectionId"]
            if work.pending is not None
            else None,
        }

    def checkpoint(self):
        """Save the whole adapter at its acknowledged boundaries.

        RNG and attempt limits belong to the adapter, so saving only one of its
        sessions could reset another session's random stream or consumed budget.
        Pending candidates stay in attempt evidence; their branches are omitted.
        Host effects are represented by checked public observations only.
        """
        from megatoken.data.semantic_contract_v8 import CONTRACT_DIGEST

        with self.lock:
            if self.initialization is None:
                raise NativeSelectionV8Error("Initialize the V8 adapter before checkpoint")
            if len(self.closed_sessions) > self.max_attempts:
                raise NativeSelectionV8Error("V8 closed-session history capacity exhausted")
            sessions = {
                identity: self._session_checkpoint(identity, work)
                for identity, work in self.sessions.items()
            }
            packet = {
                "schema": "semantic-model-checkpoint/8",
                "equivalence": "verified-model-reexecution",
                "contractDigest": CONTRACT_DIGEST,
                "initialization": copy.deepcopy(self.initialization),
                "settings": self._checkpoint_settings(),
                "randomState": capture_random_state(),
                "attempts": copy.deepcopy(self.attempts),
                "openings": copy.deepcopy(self.openings),
                "sessions": sessions,
                "closedSessions": copy.deepcopy(self.closed_sessions),
            }
            packet["digest"] = stable_algorithm_hash(packet)
            return packet

    def restore(self, *, checkpoint):
        """Reconstruct accepted model continuations without issuing host work.

        Every committed candidate is generated again under its original RNG
        state, compared byte/token-for-token, and independently acknowledged.
        The full resulting cache/retrieval/track continuation must match. No
        saved executable objects or tensor caches are loaded from input.
        """
        from megatoken.data.semantic_contract_v8 import CONTRACT_DIGEST

        with self.lock:
            if self.sessions or self.attempts or self.runtime.memory.sessions:
                raise NativeSelectionV8Error("Restore requires a fresh empty V8 runtime")
            packet = copy.deepcopy(checkpoint)
            expected_keys = {
                "schema",
                "equivalence",
                "contractDigest",
                "initialization",
                "settings",
                "randomState",
                "attempts",
                "openings",
                "sessions",
                "closedSessions",
                "digest",
            }
            if type(packet) is not dict or set(packet) != expected_keys:
                raise NativeSelectionV8Error("Invalid V8 model checkpoint fields")
            digest = packet.pop("digest")
            if (
                packet["schema"] != "semantic-model-checkpoint/8"
                or packet["equivalence"] != "verified-model-reexecution"
                or packet["contractDigest"] != CONTRACT_DIGEST
                or digest != stable_algorithm_hash(packet)
                or stable_algorithm_json(packet["settings"])
                != stable_algorithm_json(self._checkpoint_settings())
                or type(packet["attempts"]) is not list
                or len(packet["attempts"]) > self.max_attempts
                or type(packet["openings"]) is not list
                or type(packet["sessions"]) is not dict
                or len(packet["sessions"]) > self.max_sessions
                or type(packet["closedSessions"]) is not dict
                or len(packet["closedSessions"]) > self.max_attempts
            ):
                raise NativeSelectionV8Error(
                    "V8 model checkpoint identity, settings or budget differs"
                )
            identity = packet["initialization"]
            if (
                type(identity) is not dict
                or identity.get("checkpointIdentity") != self.model_identity
                or identity.get("tokenizerIdentity") != self.codec.fingerprint
                or identity.get("trainingIdentity") != self.training_identity
            ):
                raise NativeSelectionV8Error(
                    "Restored v8 checkpoint, tokenizer or training-lineage identity differs"
                )
            attempts = packet["attempts"]
            session_ids = set(packet["sessions"])
            closed_ids = set(packet["closedSessions"])
            if session_ids & closed_ids:
                raise NativeSelectionV8Error("V8 checkpoint session is both active and closed")
            if any(type(session) is not str or not session for session in session_ids | closed_ids):
                raise NativeSelectionV8Error("V8 checkpoint contains an invalid session identity")
            for _session, saved in packet["closedSessions"].items():
                if saved is not None and type(saved) is not dict:
                    raise NativeSelectionV8Error("V8 closed-session checkpoint structure differs")
            for index, attempt in enumerate(attempts):
                if (
                    type(attempt) is not dict
                    or attempt.get("selectionId") != f"v8:{index}"
                    or type(attempt.get("session")) is not str
                    or attempt.get("session") not in session_ids | closed_ids
                    or attempt.get("checkpointIdentity") != self.model_identity
                    or attempt.get("tokenizerIdentity") != self.codec.fingerprint
                    or attempt.get("trainingIdentity") != self.training_identity
                    or attempt.get("status")
                    not in {
                        "committed",
                        "rejected",
                        "semantic-rejected",
                        "cancelled",
                        "awaiting-commit",
                        "abandoned-on-restore",
                    }
                ):
                    raise NativeSelectionV8Error("V8 checkpoint attempt accounting differs")
                if (
                    attempt["session"] in closed_ids
                    and packet["closedSessions"][attempt["session"]] is None
                ):
                    raise NativeSelectionV8Error("V8 attempt has no retained closed-session cut")
                if attempt["session"] in closed_ids and attempt["status"] == "awaiting-commit":
                    raise NativeSelectionV8Error("Closed v8 session retains a pending attempt")
            opened_by_attempt = []
            for attempt in attempts:
                if attempt["session"] not in opened_by_attempt:
                    opened_by_attempt.append(attempt["session"])
            opening_sessions = [
                row.get("session") if type(row) is dict else None for row in packet["openings"]
            ]
            if opening_sessions != opened_by_attempt:
                raise NativeSelectionV8Error("V8 session opening order differs from attempts")
            if any(
                type(row) is not dict
                or set(row) != {"session", "metadataTokens", "metadataBytes", "openSeconds"}
                or type(row["session"]) is not str
                or not _json_nonnegative_integer(row["metadataTokens"])
                or not _json_nonnegative_integer(row["metadataBytes"])
                or type(row["openSeconds"]) not in (int, float)
                or not math.isfinite(row["openSeconds"])
                or row["openSeconds"] < 0
                for row in packet["openings"]
            ):
                raise NativeSelectionV8Error("V8 session opening evidence differs")
            saved_sessions = copy.deepcopy(packet["sessions"])
            for session, saved in packet["closedSessions"].items():
                if saved is not None:
                    saved_sessions[session] = saved
            if set(opening_sessions) != set(saved_sessions):
                raise NativeSelectionV8Error("V8 session opening inventory differs")
            before_random = capture_random_state()
            staged = NativeSemanticInferenceV8(
                self.runtime,
                model_identity=self.model_identity,
                training_identity=self.training_identity,
                effort=self.effort,
                max_tokens=self.max_tokens,
                max_fragments=self.max_fragments,
                max_sessions=self.max_sessions,
                max_attempts=self.max_attempts,
            )
            restored = False
            replay_count = 0
            try:
                identity = packet["initialization"]
                if type(identity) is not dict:
                    raise NativeSelectionV8Error("V8 checkpoint initialization is missing")
                loaded = staged.initialize(
                    seed=identity["seed"],
                    checkpointIdentity=identity["checkpointIdentity"],
                    tokenizerIdentity=identity["tokenizerIdentity"],
                    trainingIdentity=identity.get("trainingIdentity"),
                )
                if stable_algorithm_json(loaded) != stable_algorithm_json(identity):
                    raise NativeSelectionV8Error(
                        "Restored model device or generation settings differ"
                    )
                for session in opening_sessions:
                    saved = saved_sessions[session]
                    if (
                        type(session) is not str
                        or not session
                        or type(saved) is not dict
                        or set(saved)
                        != {
                            "public",
                            "contexts",
                            "ingestedThrough",
                            "completed",
                            "openingRandom",
                            "replay",
                            "continuationDigest",
                            "unacknowledgedSelection",
                        }
                        or type(saved["contexts"]) is not list
                        or type(saved["replay"]) is not list
                        or type(saved["completed"]) is not list
                        or any(type(value) is not str or not value for value in saved["completed"])
                        or len(set(saved["completed"])) != len(saved["completed"])
                        or type(saved["ingestedThrough"]) is not int
                        or len(saved["replay"]) != len(saved["contexts"])
                    ):
                        raise NativeSelectionV8Error("V8 checkpoint session structure differs")
                    if saved["public"] is not None:
                        checked = validate_v8_committed_prefix(saved["public"], saved["contexts"])
                        if checked["session"] != session:
                            raise NativeSelectionV8Error("Restored public session differs")
                    elif saved["contexts"] or saved["completed"] or saved["ingestedThrough"]:
                        raise NativeSelectionV8Error("Empty V8 session contains accepted work")
                    restore_random_state(saved["openingRandom"])
                    staged._opened(session)
                    expected = [
                        index
                        for index, attempt in enumerate(attempts)
                        if attempt["session"] == session and attempt["status"] == "committed"
                    ]
                    actual = []
                    for context_index, entry in enumerate(saved["replay"]):
                        if (
                            type(entry) is not dict
                            or set(entry)
                            != {"attempt", "contextIndex", "acknowledgedThrough", "randomBefore"}
                            or type(entry["attempt"]) is not int
                            or not 0 <= entry["attempt"] < len(attempts)
                            or type(entry["contextIndex"]) is not int
                            or entry["contextIndex"] != context_index
                            or type(entry["acknowledgedThrough"]) is not int
                        ):
                            raise NativeSelectionV8Error(
                                "Invalid V8 checkpoint reconstruction entry"
                            )
                        actual.append(entry["attempt"])
                    if actual != expected:
                        raise NativeSelectionV8Error(
                            "Checkpoint omits or imports acknowledged selections"
                        )
                    pending = [
                        attempt["selectionId"]
                        for attempt in attempts
                        if attempt["session"] == session and attempt["status"] == "awaiting-commit"
                    ]
                    if pending != (
                        []
                        if saved["unacknowledgedSelection"] is None
                        else [saved["unacknowledgedSelection"]]
                    ):
                        raise NativeSelectionV8Error(
                            "Checkpoint pending selection accounting differs"
                        )
                    if session in closed_ids and saved["unacknowledgedSelection"] is not None:
                        raise NativeSelectionV8Error("Closed v8 session has unacknowledged work")
                    work = staged.sessions[session]
                    for index, entry in zip(actual, saved["replay"], strict=True):
                        context_index = entry["contextIndex"]
                        context = saved["contexts"][context_index]
                        rows = saved["public"]["timeline"]
                        through = context["visibleThrough"]
                        acknowledged_through = entry["acknowledgedThrough"]
                        if not through < acknowledged_through <= len(rows):
                            raise NativeSelectionV8Error("Invalid V8 acknowledged-history cursor")
                        public = {**saved["public"], "timeline": rows[:through]}
                        contexts = saved["contexts"][:context_index]
                        staged.attempts = copy.deepcopy(attempts[:index])
                        restore_random_state(entry["randomBefore"])
                        generated = staged.select(
                            session=session, public=public, contexts=contexts, context=context
                        )
                        original, actual_selection = attempts[index], generated["selection"]
                        # A committed ledger entry is the evidence for the exact
                        # selection that was acknowledged. Comparing just its
                        # payload and identity leaves rehashed edits to accounting
                        # (token counts, context visibility, event IDs, etc.)
                        # undetected even though those fields are deterministic.
                        # Wall-clock measurements and the acknowledgement lifecycle
                        # label are expected to differ when replaying in a new process.
                        replay_volatile = {
                            "status",
                            "generationSeconds",
                            "preparationSeconds",
                            "totalSeconds",
                        }
                        actual_evidence = {
                            key: value
                            for key, value in actual_selection.items()
                            if key not in replay_volatile
                        }
                        original_evidence = {
                            key: value
                            for key, value in original.items()
                            if key not in replay_volatile
                        }
                        if stable_algorithm_json(actual_evidence) != stable_algorithm_json(
                            original_evidence
                        ):
                            differing = next(
                                (
                                    key
                                    for key in sorted(set(actual_evidence) | set(original_evidence))
                                    if stable_algorithm_json(actual_evidence.get(key))
                                    != stable_algorithm_json(original_evidence.get(key))
                                ),
                                "selection evidence",
                            )
                            raise NativeSelectionV8Error(
                                "Restored native model selection differs: " + differing
                            )
                        staged.commit(
                            session=session,
                            public={**saved["public"], "timeline": rows[:acknowledged_through]},
                            contexts=saved["contexts"][: context_index + 1],
                            invocationId=context["invocationId"],
                            record=generated["record"],
                            selection=actual_selection,
                        )
                        replay_count += 1
                    for invocation in saved["completed"]:
                        staged.complete(
                            session=session,
                            public=saved["public"],
                            contexts=saved["contexts"],
                            invocationId=invocation,
                        )
                    if (
                        work.ingested_through != saved["ingestedThrough"]
                        or stable_algorithm_json(work.contexts)
                        != stable_algorithm_json(saved["contexts"])
                        or stable_algorithm_json(sorted(work.completed))
                        != stable_algorithm_json(saved["completed"])
                        or stable_algorithm_json(work.opening_random)
                        != stable_algorithm_json(saved["openingRandom"])
                        or stable_algorithm_json(work.replay)
                        != stable_algorithm_json(saved["replay"])
                        or continuation_fingerprint(work.runtime) != saved["continuationDigest"]
                    ):
                        raise NativeSelectionV8Error(
                            "Restored V8 cache, continuation or observation cursor differs"
                        )
                    if stable_algorithm_json(work.public) != stable_algorithm_json(saved["public"]):
                        raise NativeSelectionV8Error(
                            "Restored V8 acknowledged public history differs"
                        )
                    if session in closed_ids:
                        staged.cancel(session=session, reason="restore closed session")
                        if stable_algorithm_json(
                            staged.closed_sessions[session]
                        ) != stable_algorithm_json(packet["closedSessions"][session]):
                            raise NativeSelectionV8Error("Restored closed-session evidence differs")
                if len(packet["openings"]) != len(staged.openings):
                    raise NativeSelectionV8Error("V8 session opening accounting differs")
                opening_volatile = {"openSeconds"}
                actual_openings = [
                    {key: value for key, value in row.items() if key not in opening_volatile}
                    for row in staged.openings
                ]
                saved_openings = [
                    {key: value for key, value in row.items() if key not in opening_volatile}
                    for row in packet["openings"]
                    if type(row) is dict
                ]
                if stable_algorithm_json(actual_openings) != stable_algorithm_json(saved_openings):
                    raise NativeSelectionV8Error("V8 session opening evidence differs")
                for session in closed_ids:
                    saved = packet["closedSessions"][session]
                    if saved is None:
                        staged.closed_sessions[session] = None
                restore_random_state(packet["randomState"])
                restored_attempts = copy.deepcopy(attempts)
                for attempt in restored_attempts:
                    if attempt["status"] == "awaiting-commit":
                        attempt.update(
                            status="abandoned-on-restore",
                            failure="Unacknowledged candidate excluded from restored history",
                        )
                self.sessions = staged.sessions
                self.closed_sessions = staged.closed_sessions
                self.attempts = restored_attempts
                self.openings = copy.deepcopy(packet["openings"])
                self.initialization = copy.deepcopy(staged.initialization)
                restored = True
                return {
                    "schema": "semantic-model-restore/8",
                    "checkpointDigest": digest,
                    "restoredSessions": len(self.sessions),
                    "reexecutedSelections": replay_count,
                    "consumedAttempts": len(self.attempts),
                    "hostDispatches": 0,
                    "equivalence": "verified-model-reexecution",
                    "restored": True,
                }
            except Exception as error:
                raise NativeSelectionV8Error(
                    "V8 model restoration failed: " + str(error)
                ) from error
            finally:
                if not restored:
                    for session in list(staged.sessions):
                        staged.cancel(session=session, reason="failed checkpoint reconstruction")
                    # Opening a runtime may allocate its memory session before
                    # a later initialization step fails. The runtime was empty
                    # on entry, so every remaining allocation belongs to this
                    # failed reconstruction and must also be released.
                    for memory in list(self.runtime.memory.sessions.values()):
                        self.runtime.memory.close_session(memory)
                    restore_random_state(before_random)

    def cancel(self, *, session, reason, **unused):
        with self.lock:
            if type(session) is not str or not session:
                raise NativeSelectionV8Error("V8 close requires a session identity")
            work = self.sessions.get(session)
            if work is not None:
                if (
                    len(self.closed_sessions) >= self.max_attempts
                    and session not in self.closed_sessions
                ):
                    raise NativeSelectionV8Error("V8 closed-session history capacity exhausted")
                failure = str(reason)
                saved = self._session_checkpoint(session, work)
                # The closed cut excludes a speculative candidate. Build and
                # validate the durable summary before consuming its outcome.
                saved["unacknowledgedSelection"] = None
                if work.pending is not None:
                    work.pending["attempt"].update(status="cancelled", failure=failure)
                    work.pending = None
                self.sessions.pop(session)
                self.closed_sessions[session] = saved
                self.runtime.memory.close_session(work.runtime.memory)
            else:
                if (
                    len(self.closed_sessions) >= self.max_attempts
                    and session not in self.closed_sessions
                ):
                    raise NativeSelectionV8Error("V8 closed-session history capacity exhausted")
                self.closed_sessions.setdefault(session, None)
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
                    row["status"]
                    in {"rejected", "semantic-rejected", "cancelled", "abandoned-on-restore"}
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
                "closedSessions": len(self.closed_sessions),
                "optimizerSteps": 0,
                "learnedSuccess": None,
                "scope": "Actual native runtime input framing, original emitted tokens, wall times and cache snapshots; semantic acceptance requires explicit acknowledgement",
            }
