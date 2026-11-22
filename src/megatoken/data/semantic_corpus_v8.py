"""Fresh source/native admission and ordinary prepared shards for V8 traces."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from megatoken.data.native_v8 import prepare_native_v8_training, validate_v8_artifact
from megatoken.data.semantic_contract_v8 import (
    CONTRACT_DIGEST,
    FIXTURE_DIGEST,
    INTEROPERABILITY_DIGEST,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.data.semantic_orchestration_training_v8 import (
    validate_orchestration_records_v8,
    validate_orchestration_transcript_v8,
)
from megatoken.data.source_identity_v8 import source_evidence_binding_v8
from megatoken.generation.codec import LING_TOKENIZER_SHA256
from megatoken.io import write_json

from .chunking import ChunkingConfig
from .dedup import DedupConfig, DedupStore
from .detailed_v8 import parse_record
from .storage import export_prepared


def _verify_binding(document, registry):
    side = document.get("sidecar", {})
    if not isinstance(side, dict):
        raise ValueError("V8 artifact sidecar is malformed")
    identity = side.get("splitIdentity") if isinstance(side, dict) else None
    if (
        not isinstance(identity, dict)
        or identity.get("schema") != "semantic-region-split-identity/1"
    ):
        raise ValueError("V8 artifact has no immutable source-family split identity")
    if identity.get("registry") != registry:
        raise ValueError("V8 artifact split identity differs from the complete frozen registry")
    digest = stable_algorithm_hash(
        {key: value for key, value in registry.items() if key != "digest"}
    )
    if digest != registry.get("digest") or identity.get("registryDigest") != digest:
        raise ValueError("V8 family registry digest is invalid")
    member = next(
        (row for row in registry["sources"] if row["id"] == identity.get("sourceId")), None
    )
    component = next(
        (row for row in registry["components"] if identity.get("sourceId") in row["sourceIds"]),
        None,
    )
    if member is None or component is None:
        raise ValueError("V8 source is absent from the frozen registry")
    if (
        identity.get("sourceDigest") != member.get("sourceDigest")
        or identity.get("coreDigest") != member.get("coreDigest")
        or identity.get("componentId") != component.get("id")
        or identity.get("members") != component.get("members")
        or identity.get("split") != component.get("split")
        or side.get("split") != component.get("split")
        or side.get("sourceComponent") != component.get("id")
        or side.get("familyRegistryDigest") != digest
    ):
        raise ValueError("V8 frozen split assignment or exact source membership changed")
    binding = side.get("datasetBinding")
    expected = {
        "schema": "semantic-region-dataset-binding/1",
        "traceFormat": "semantic-trace/8",
        "episodeDigest": stable_algorithm_hash(document["public"]),
        "contextsDigest": stable_algorithm_hash(document["contexts"]),
        **source_evidence_binding_v8(side),
        "registryDigest": digest,
        "componentId": component["id"],
        "split": component["split"],
        "sourceId": member["id"],
        "family": member["family"],
        "sourceDigest": member["sourceDigest"],
        "coreDigest": member["coreDigest"],
    }
    if "orchestrationTranscript" in side:
        transcript = side["orchestrationTranscript"]
        if not isinstance(transcript, dict):
            raise ValueError("V8 orchestration transcript declaration is malformed")
        expected["orchestrationTranscriptDigest"] = stable_algorithm_hash(transcript)
    if binding != expected:
        raise ValueError("V8 binding does not cover original source, public records and contexts")
    return identity, member, component, binding


def _native_admission(
    document,
    *,
    codec,
    tokenizer_evidence,
    consumer_revision,
    source_receipt,
    binding,
    max_positions,
    max_history,
):
    result = validate_v8_artifact(document)
    episode, training = prepare_native_v8_training(
        document, codec=codec, max_positions=max_positions, max_history=max_history
    )
    checks = {
        **result["checks"],
        "masks": "passed" if training["maskValidation"] else "failed",
        "sourceBinding": "passed",
        "tokenizer": "passed",
    }
    orchestration = None
    document_sidecar = document.get("sidecar", {})
    has_orchestration = isinstance(document_sidecar, dict) and (
        "orchestrationTranscript" in document_sidecar
    )
    if has_orchestration:
        orchestration = validate_orchestration_transcript_v8(document)
        if not isinstance(source_receipt, dict):
            raise ValueError("Fresh V8 source receipt is malformed")
        if (
            result["checks"].get("orchestration") != "passed"
            or result.get("orchestrationTranscriptDigest") != orchestration["transcriptDigest"]
            or training.get("orchestrationRecords") != orchestration["recordCount"]
            or training.get("orchestrationTranscriptDigest") != orchestration["transcriptDigest"]
            or binding.get("orchestrationTranscriptDigest") != orchestration["transcriptDigest"]
            or source_receipt.get("orchestrationValidated") is not True
            or source_receipt.get("orchestrationTranscriptDigest")
            != orchestration["transcriptDigest"]
        ):
            raise ValueError("Fresh V8 orchestration transcript binding did not pass")
        checks["orchestration"] = "passed"
    elif any(
        key in source_receipt for key in ("orchestrationValidated", "orchestrationTranscriptDigest")
    ) or any(key in binding for key in ("orchestrationTranscriptDigest",)):
        raise ValueError("V8 artifact without a transcript cannot claim orchestration admission")
    if any(
        checks.get(key) != "passed"
        for key in (
            "replay",
            "contexts",
            "continuation",
            "masks",
            "sourceBinding",
            "tokenizer",
            *(("orchestration",) if orchestration is not None else ()),
        )
    ):
        raise ValueError("Fresh V8 native checks did not all pass")
    if (
        not isinstance(tokenizer_evidence, dict)
        or tokenizer_evidence.get("schema") != "semantic-tokenizer-admission/1"
        or tokenizer_evidence.get("status") != "passed"
        or tokenizer_evidence.get("identity") != codec.fingerprint
        or tokenizer_evidence.get("sha256") != LING_TOKENIZER_SHA256
        or tokenizer_evidence.get("verification") != "pinned-tokenizer-bytes"
    ):
        raise ValueError("V8 admission requires a verified pinned tokenizer and codec identity")
    receipt = {
        "schema": "semantic-native-admission/8",
        "traceFormat": "semantic-trace/8",
        "contextFormat": "semantic-context/8",
        "representation": "mixed",
        "componentGate": "native-consumer-only",
        "publicDigest": stable_algorithm_hash(document["public"]),
        "contextsDigest": stable_algorithm_hash(document["contexts"]),
        "sourceArtifactDigest": binding["sourceArtifactDigest"],
        **(
            {"sourceIdentityDigest": binding["sourceIdentityDigest"]}
            if "sourceIdentityDigest" in binding
            else {}
        ),
        "sourceVerification": source_receipt,
        "sourceProfile": document.get("sidecar", {}).get("profile"),
        "protocolProfile": document.get("sidecar", {}).get("protocolProfile"),
        "protocolExtensions": document.get("sidecar", {}).get("protocolExtensions", []),
        "datasetBindingDigest": stable_algorithm_hash(binding),
        "consumerRevision": consumer_revision,
        "contractDigest": CONTRACT_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "interoperabilityDigest": INTEROPERABILITY_DIGEST,
        "tokenizer": {
            "status": "passed",
            "identity": codec.fingerprint,
            "sha256": LING_TOKENIZER_SHA256,
        },
        "checks": checks,
        "counts": {
            "records": len(document["public"]["timeline"]),
            "contexts": training["contexts"],
            "generatedRecords": training["generatedRecords"],
            "algorithmRecords": result["counts"].get("algorithmRecords", 0),
            "regionRecords": result["counts"].get("regionRecords", 0),
            "detailedRecords": result["counts"].get("detailedRecords", 0),
            "protocolRecords": result["counts"].get("protocolRecords", 0),
            "generatedTokens": training["generatedTokens"],
            "observedTokens": training["observedTokens"],
        },
        "framing": {
            "kind": training["framing"],
            "eosTokenId": training["eosTokenId"],
            "terminalTokens": training["terminalTokens"],
            "physicalRecords": training["physicalRecords"],
        },
        "nativeTrainingDigest": training["nativeTrainingDigest"],
        "optimizerSteps": 0,
        "learningClaim": None,
        "admitted": True,
    }
    if orchestration is not None:
        receipt["orchestrationTranscriptDigest"] = orchestration["transcriptDigest"]
        receipt["counts"]["orchestrationRecords"] = orchestration["recordCount"]
    elif "orchestrationTranscriptDigest" in result:
        raise ValueError("Native V8 receipt cannot add detached orchestration evidence")
    return episode, training, receipt


def semantic_behaviors_v8(episode):
    """Derive stage coverage only from source-bound V8 public records."""
    annotations = json.loads(episode.supervision.annotations_json)
    semantic = annotations.get("semantic_execution", {})
    if semantic.get("schema") != "semantic-execution/8":
        raise ValueError("V8 behavior census requires a V8 prepared episode")
    artifact = semantic.get("publicArtifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("public"), dict):
        raise ValueError("V8 behavior census requires retained public records")
    rows = artifact["public"].get("timeline", [])
    if not isinstance(rows, list):
        raise ValueError("V8 behavior census requires a public timeline")

    program_rows = defaultdict(list)
    invocation_rows = defaultdict(list)
    generated = []
    generated_rows = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("fields"), dict):
            raise ValueError("V8 behavior census requires typed public rows")
        kind, fields = row.get("kind"), row["fields"]
        if kind == "program":
            program_rows[fields.get("programId")].append((row, fields))
        elif kind == "invocation":
            invocation_rows[fields.get("invocationId")].append((row, fields))
        if row.get("mode") == "generated":
            generated_rows.append(row)
        if kind == "semantic" and row.get("mode") == "generated":
            line = fields.get("line")
            # A V8 timeline can carry compact region, algorithm, or protocol
            # records as well as detailed records. Only detailed records use
            # this parser; other canonical families remain valid evidence but
            # do not carry the await/request wire kinds counted below.
            if not isinstance(line, str) or not line.startswith("detail "):
                continue
            try:
                record_kind, record_fields = parse_record(line, role="generated")
            except (TypeError, ValueError):
                # Invalid source-admitted artifacts are rejected by admission;
                # a noncanonical line cannot establish behavior coverage here.
                continue
            generated.append((row, record_kind, record_fields))

    # Duplicate source or invocation identities are invalid for admitted V8
    # data. Ignore them here as an additional fail-closed guard when this
    # helper is called independently of prepared-corpus validation.
    programs = {
        identity: rows_for_program[0]
        for identity, rows_for_program in program_rows.items()
        if isinstance(identity, str) and len(rows_for_program) == 1
    }
    invocations = {
        identity: invocation_rows_for_id[0]
        for identity, invocation_rows_for_id in invocation_rows.items()
        if isinstance(identity, str) and len(invocation_rows_for_id) == 1
    }

    # Admission checks the complete source, context and native replay. The
    # census still binds each behavior to the exact published source and call
    # identity so that a detached row cannot establish coverage on its own.
    def sequence(row):
        value = row.get("sequence")
        return value if type(value) is int and value > 0 else None

    def track(fields):
        value = fields.get("track")
        return value if isinstance(value, str) and value else None

    def session(fields):
        value = fields.get("session")
        return value if isinstance(value, str) and value else None

    source_bound_programs = {}
    for identity, (row, fields) in programs.items():
        if (
            row.get("mode") == "observed"
            and sequence(row) is not None
            and session(fields) is not None
            and isinstance(fields.get("source"), str)
            and isinstance(fields.get("digest"), str)
            and stable_algorithm_hash(fields["source"]) == fields["digest"]
        ):
            source_bound_programs[identity] = (row, fields)
    bound_invocations = {}
    for identity, (row, fields) in invocations.items():
        program_pair = source_bound_programs.get(fields.get("programId"))
        if program_pair is None:
            continue
        program_row, program = program_pair
        if (
            row.get("mode") == "observed"
            and sequence(row) is not None
            and sequence(program_row) < sequence(row)
            and session(row["fields"]) is not None
            and session(row["fields"]) == session(program)
            and track(row["fields"]) is not None
            and fields.get("sourceDigest") == program.get("digest")
        ):
            bound_invocations[identity] = (row, fields)

    behaviors = {"semantic_v8"}
    if programs and bound_invocations:
        behaviors.add("semantic_code_invocation")

    # Repeated invocation means separate root calls to the same immutable
    # program. Callback activations and child calls have their own coverage
    # labels and cannot inflate this count.
    root_calls = defaultdict(set)
    for identity, (row, fields) in bound_invocations.items():
        if fields.get("activation") == "call" and fields.get("parentInvocationId") is None:
            root_calls[
                (
                    session(row["fields"]),
                    fields.get("programId"),
                    fields.get("sourceDigest"),
                )
            ].add(identity)
    if any(len(identities) > 1 for identities in root_calls.values()):
        behaviors.add("semantic_repeated_invocation")

    # `parentInvocationId` describes the JavaScript call tree, not task
    # orchestration. A child behavior is counted only from a source-qualified
    # transcript whose exact lifecycle actions and receipts became physical
    # training rows and are bound by both producer and native admission.
    sidecar = artifact.get("sidecar")
    if isinstance(sidecar, dict) and "orchestrationTranscript" in sidecar:
        try:
            orchestration = validate_orchestration_transcript_v8(artifact)
        except (TypeError, ValueError):
            orchestration = None
        binding = semantic.get("datasetBinding")
        source_receipt = semantic.get("sourceAdmission")
        native_receipt = semantic.get("nativeAdmission")
        native_checks = native_receipt.get("checks") if isinstance(native_receipt, dict) else None
        native_counts = native_receipt.get("counts") if isinstance(native_receipt, dict) else None
        if not isinstance(native_checks, dict):
            native_checks = {}
        if not isinstance(native_counts, dict):
            native_counts = {}
        if (
            orchestration is not None
            and isinstance(binding, dict)
            and binding.get("orchestrationTranscriptDigest") == orchestration["transcriptDigest"]
            and isinstance(source_receipt, dict)
            and source_receipt.get("schema") == "semantic-native-source-recheck/1"
            and source_receipt.get("sourceArtifactDigest") == binding.get("sourceArtifactDigest")
            and source_receipt.get("publicDigest") == binding.get("episodeDigest")
            and source_receipt.get("contextDigest") == binding.get("contextsDigest")
            and source_receipt.get("familyBindingValidated") is True
            and source_receipt.get("sourceValidated") is True
            and source_receipt.get("projectionValidated") is True
            and source_receipt.get("replayValidated") is True
            and source_receipt.get("datasetBindingDigest") == stable_algorithm_hash(binding)
            and source_receipt.get("orchestrationValidated") is True
            and source_receipt.get("orchestrationTranscriptDigest")
            == orchestration["transcriptDigest"]
            and isinstance(native_receipt, dict)
            and native_receipt.get("admitted") is True
            and native_receipt.get("publicDigest") == binding.get("episodeDigest")
            and native_receipt.get("contextsDigest") == binding.get("contextsDigest")
            and native_receipt.get("sourceArtifactDigest") == binding.get("sourceArtifactDigest")
            and native_receipt.get("nativeTrainingDigest") == semantic.get("nativeTrainingDigest")
            and isinstance(semantic.get("nativeTrainingDigest"), str)
            and re.fullmatch(r"[0-9a-f]{64}", semantic["nativeTrainingDigest"]) is not None
            and native_receipt.get("orchestrationTranscriptDigest")
            == orchestration["transcriptDigest"]
            and native_receipt.get("checks", {}).get("orchestration") == "passed"
            and native_counts.get("orchestrationRecords") == orchestration["recordCount"]
            and native_receipt.get("datasetBindingDigest") == stable_algorithm_hash(binding)
            and native_receipt.get("sourceVerification") == source_receipt
            and all(
                native_checks.get(name) == "passed"
                for name in (
                    "replay",
                    "contexts",
                    "continuation",
                    "masks",
                    "sourceBinding",
                    "tokenizer",
                    "orchestration",
                )
            )
            and validate_orchestration_records_v8(episode, orchestration)
        ):
            behaviors.add("semantic_child")

    # V8's generated operations are typed detail records inside `semantic`
    # rows. Match await/resume by their continuation and invocation identities
    # rather than looking for top-level envelope kinds that V8 never emits.
    awaits = {}
    for row, kind, fields in generated:
        invocation_id = row["fields"].get("invocationId")
        bound = bound_invocations.get(invocation_id)
        if bound is None:
            continue
        invocation_row, _ = bound
        if (
            row.get("mode") != "generated"
            or session(row["fields"]) != session(invocation_row["fields"])
            or track(row["fields"]) != track(invocation_row["fields"])
            or fields.get("invocation") != invocation_id
            or sequence(row) is None
            or sequence(row) <= sequence(invocation_row)
        ):
            continue
        if kind == "await":
            awaits[(track(row["fields"]), invocation_id, fields.get("continuation"))] = row
        elif kind == "resume":
            key = (track(row["fields"]), invocation_id, fields.get("continuation"))
            prior = awaits.get(key)
            if prior is not None and sequence(prior) < sequence(row):
                behaviors.add("semantic_async")

    # Tool feedback needs the generated request, its exact observed result,
    # then more generated work on the same invocation and track.
    for request_row, kind, fields in generated:
        if kind != "request":
            continue
        effect_id = fields.get("effect")
        invocation_id = fields.get("invocation")
        bound = bound_invocations.get(invocation_id)
        if bound is None:
            continue
        invocation_row, _ = bound
        request_track = track(request_row["fields"])
        request_sequence = sequence(request_row)
        if (
            not isinstance(effect_id, str)
            or not isinstance(invocation_id, str)
            or request_row["fields"].get("invocationId") != invocation_id
            or request_row.get("mode") != "generated"
            or session(request_row["fields"]) != session(invocation_row["fields"])
            or request_track != track(invocation_row["fields"])
            or request_sequence is None
            or request_sequence <= sequence(invocation_row)
        ):
            continue
        result_rows = [
            row
            for row in rows
            if row.get("kind") == "result"
            and row.get("mode") == "observed"
            and session(row.get("fields", {})) == session(invocation_row["fields"])
            and track(row.get("fields", {})) == request_track
            and sequence(row) is not None
            and sequence(row) > request_sequence
            and row.get("fields", {}).get("effectId") == effect_id
            and row.get("fields", {}).get("invocationId") == invocation_id
        ]
        if any(
            generated_row.get("mode") == "generated"
            and session(generated_row.get("fields", {})) == session(invocation_row["fields"])
            and track(generated_row.get("fields", {})) == request_track
            and sequence(generated_row) is not None
            and sequence(generated_row) > sequence(result_row)
            and generated_row.get("fields", {}).get("invocationId") == invocation_id
            for result_row in result_rows
            for generated_row in generated_rows
        ):
            behaviors.add("semantic_tool_feedback")
            break

    if any(fields.get("activation") == "callback" for _, fields in bound_invocations.values()):
        behaviors.add("semantic_callback")
    if any(
        fields.get("activation") == "module-init"
        and (program_pair := source_bound_programs.get(fields.get("programId"))) is not None
        and program_pair[1].get("module") is True
        for _, fields in bound_invocations.values()
    ):
        behaviors.add("semantic_initialization")
    if any(
        row.get("mode") == "observed"
        and row["kind"] == "cancel"
        and row["fields"].get("invocationId") in bound_invocations
        and session(row["fields"])
        == session(bound_invocations[row["fields"]["invocationId"]][0]["fields"])
        for row in rows
    ):
        behaviors.add("semantic_cancel")

    def matching_observed_result(row, invocation_id, effect_id, status):
        fields = row.get("fields", {})
        if (
            row.get("kind") != "result"
            or row.get("mode") != "observed"
            or fields.get("invocationId") != invocation_id
            or fields.get("effectId") != effect_id
            or fields.get("status") != status
        ):
            return False
        try:
            kind, observed = parse_record(fields.get("line"), role="observed")
        except (TypeError, ValueError):
            return False
        return (
            kind == "observe"
            and observed.get("effect") == effect_id
            and observed.get("status") == status
            and observed.get("result") == fields.get("result")
        )

    def linked_reconciliation(row):
        fields = row["fields"]
        invocation_id = fields.get("invocationId")
        effect_id = fields.get("effectId")
        bound = bound_invocations.get(invocation_id)
        if bound is None or not isinstance(effect_id, str) or not effect_id:
            return False
        invocation_row, _ = bound
        reconciliation_sequence = sequence(row)
        invocation_sequence = sequence(invocation_row)
        invocation_track = track(invocation_row["fields"])
        invocation_session = session(invocation_row["fields"])
        disposition = fields.get("disposition")
        if (
            row.get("mode") != "observed"
            or reconciliation_sequence is None
            or invocation_sequence is None
            or reconciliation_sequence <= invocation_sequence
            or track(fields) is None
            or track(fields) != invocation_track
            or session(fields) is None
            or session(fields) != invocation_session
            or not isinstance(fields.get("evidenceId"), str)
            or not fields["evidenceId"]
            or disposition not in {"not-dispatched", "fulfilled", "rejected"}
        ):
            return False

        requests = [
            request_row
            for request_row, kind, request in generated
            if kind == "request"
            and request.get("invocation") == invocation_id
            and request.get("effect") == effect_id
            and request_row.get("mode") == "generated"
            and request_row.get("fields", {}).get("invocationId") == invocation_id
            and session(request_row.get("fields", {})) == invocation_session
            and track(request_row.get("fields", {})) == invocation_track
            and sequence(request_row) is not None
            and invocation_sequence < sequence(request_row) < reconciliation_sequence
        ]
        if len(requests) != 1:
            return False
        request_sequence = sequence(requests[0])
        dispatches = [
            candidate
            for candidate in rows
            if candidate.get("kind") == "dispatch"
            and candidate.get("mode") == "observed"
            and candidate.get("fields", {}).get("invocationId") == invocation_id
            and candidate.get("fields", {}).get("effectId") == effect_id
            and candidate.get("fields", {}).get("key") == f"{invocation_id}/{effect_id}"
            and type(candidate.get("fields", {}).get("attempt")) is int
            and candidate["fields"]["attempt"] > 0
            and session(candidate.get("fields", {})) == invocation_session
            and track(candidate.get("fields", {})) == invocation_track
            and sequence(candidate) is not None
            and request_sequence < sequence(candidate) < reconciliation_sequence
        ]
        dispatches.sort(key=sequence)
        if (
            not dispatches
            or dispatches[0]["fields"]["attempt"] != 1
            or any(
                right["fields"]["attempt"] != left["fields"]["attempt"] + 1
                for left, right in zip(dispatches, dispatches[1:], strict=False)
            )
        ):
            return False
        dispatch = dispatches[-1]
        if any(
            result.get("kind") == "result"
            and result.get("fields", {}).get("invocationId") == invocation_id
            and result.get("fields", {}).get("effectId") == effect_id
            and sequence(result) is not None
            and sequence(dispatch) < sequence(result) < reconciliation_sequence
            for result in rows
        ):
            return False

        if disposition in {"fulfilled", "rejected"}:
            return any(
                sequence(result) == reconciliation_sequence + 1
                and session(result.get("fields", {})) == invocation_session
                and track(result.get("fields", {})) == invocation_track
                and matching_observed_result(result, invocation_id, effect_id, disposition)
                for result in rows
            )
        if disposition == "not-dispatched":
            retries = [
                candidate
                for candidate in rows
                if candidate.get("kind") == "dispatch"
                and candidate.get("mode") == "observed"
                and candidate.get("fields", {}).get("invocationId") == invocation_id
                and candidate.get("fields", {}).get("effectId") == effect_id
                and candidate.get("fields", {}).get("key") == f"{invocation_id}/{effect_id}"
                and candidate.get("fields", {}).get("attempt") == dispatch["fields"]["attempt"] + 1
                and session(candidate.get("fields", {})) == invocation_session
                and track(candidate.get("fields", {})) == invocation_track
                and sequence(candidate) is not None
                and sequence(candidate) > reconciliation_sequence
            ]
            if len(retries) != 1:
                return False
            retry_sequence = sequence(retries[0])
            return any(
                sequence(result) is not None
                and sequence(result) > retry_sequence
                and session(result.get("fields", {})) == invocation_session
                and track(result.get("fields", {})) == invocation_track
                and result.get("fields", {}).get("status") in {"fulfilled", "rejected"}
                and matching_observed_result(
                    result,
                    invocation_id,
                    effect_id,
                    result.get("fields", {}).get("status"),
                )
                for result in rows
            )
        return False

    if any(row.get("kind") == "reconcile" and linked_reconciliation(row) for row in rows):
        # A bare callback-reattach row is not proof that the prior handler was
        # lost: the public V8 timeline records the reattach receipt but no
        # authoritative lost-state transition. Count recovery only when a
        # reconciled source effect is linked to its exact request and outcome.
        behaviors.add("semantic_recovery")

    # A checked reread is an observed eviction/read pair for one already
    # invoked source on the same track. Both digest and exact source bytes must
    # match the immutable program publication. Unpaired, cross-track, or
    # substituted reads are not coverage evidence.
    invoked_tracks = defaultdict(list)
    for row, fields in bound_invocations.values():
        invoked_tracks[
            (
                session(row["fields"]),
                track(row["fields"]),
                fields.get("programId"),
                fields.get("sourceDigest"),
            )
        ].append(sequence(row))
    evicted = {}
    for row in rows:
        kind, fields = row["kind"], row["fields"]
        if kind not in {"source-evict", "source-read"}:
            continue
        program_pair = source_bound_programs.get(fields.get("programId"))
        if program_pair is None:
            continue
        program_row, program = program_pair
        row_sequence = sequence(row)
        key = (
            fields.get("session"),
            track(fields),
            fields.get("programId"),
            fields.get("digest"),
        )
        if (
            row.get("mode") != "observed"
            or row_sequence is None
            or row_sequence <= sequence(program_row)
            or fields.get("digest") != program.get("digest")
            or stable_algorithm_hash(program.get("source")) != program.get("digest")
            or not any(
                invoked_sequence < row_sequence for invoked_sequence in invoked_tracks.get(key, ())
            )
        ):
            continue
        if kind == "source-evict":
            evicted[key] = row_sequence
        elif (
            key in evicted
            and evicted[key] < row_sequence
            and fields.get("source") == program.get("source")
        ):
            behaviors.add("semantic_source_reread")
            del evicted[key]

    return behaviors


def prepare_semantic_corpus_v8(
    candidates,
    output,
    *,
    codec,
    producer_revision,
    consumer_revision,
    source_verifier,
    tokenizer_evidence,
    max_positions=8192,
    max_history=1024,
):
    """Prepare only candidates with fresh source correspondence and native receipts.

    Each candidate is a bound V8 artifact. ``source_verifier`` must freshly replay
    source correspondence for that artifact and return its producer receipt.
    """
    if not callable(source_verifier) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
        for value in (producer_revision, consumer_revision)
    ):
        raise ValueError(
            "V8 preparation requires fresh source replay and exact repository revisions"
        )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() or (root / "semantic-v8-dedup.sqlite").exists():
        raise ValueError("V8 preparation requires a new output directory")
    counts, refusals = Counter(), []
    registry = None
    with DedupStore(
        root / "semantic-v8-dedup.sqlite",
        DedupConfig(
            split_seed="semantic-region-family-components/1", max_descendants_per_parent=20000
        ),
    ) as store:
        for index, candidate in enumerate(candidates):
            counts["attempted"] += 1
            document = (
                candidate.get("exported", candidate) if isinstance(candidate, dict) else candidate
            )
            try:
                if (
                    not isinstance(document, dict)
                    or document.get("public", {}).get("schema") != "semantic-execution/8"
                ):
                    raise ValueError("Only exact semantic-execution/8 artifacts can be prepared")
                identity = document.get("sidecar", {}).get("splitIdentity", {})
                candidate_registry = identity.get("registry")
                if registry is None:
                    registry = candidate_registry
                elif registry != candidate_registry:
                    raise ValueError("All V8 candidates must use one complete family registry")
                identity, member, component, binding = _verify_binding(document, registry)
                source = source_verifier(document)
                if not isinstance(source, dict):
                    raise ValueError("Fresh V8 producer source receipt is malformed")
                expected_source = {
                    "schema": "semantic-native-source-recheck/1",
                    "producerRevision": producer_revision,
                    "producerTreeClean": True,
                    "sourceArtifactDigest": binding["sourceArtifactDigest"],
                    "publicDigest": binding["episodeDigest"],
                    "contextDigest": binding["contextsDigest"],
                    "sourceValidated": True,
                    "projectionValidated": True,
                    "replayValidated": True,
                    "familyBindingValidated": True,
                    "registryDigest": registry["digest"],
                    "datasetBindingDigest": stable_algorithm_hash(binding),
                }
                if "sourceIdentityDigest" in binding:
                    expected_source.update(
                        sourceIdentityDigest=binding["sourceIdentityDigest"], identityValidated=True
                    )
                document_sidecar = document.get("sidecar", {})
                has_orchestration = isinstance(document_sidecar, dict) and (
                    "orchestrationTranscript" in document_sidecar
                )
                if has_orchestration:
                    expected_source.update(
                        orchestrationValidated=True,
                        orchestrationTranscriptDigest=binding["orchestrationTranscriptDigest"],
                    )
                elif any(
                    key in source
                    for key in ("orchestrationValidated", "orchestrationTranscriptDigest")
                ):
                    raise ValueError(
                        "Source receipt cannot claim orchestration validation without a transcript"
                    )
                if any(source.get(key) != value for key, value in expected_source.items()):
                    raise ValueError(
                        "Fresh V8 producer receipt does not bind the exact source/public/context bytes"
                    )
                episode, training, native = _native_admission(
                    document,
                    codec=codec,
                    consumer_revision=consumer_revision,
                    tokenizer_evidence=tokenizer_evidence,
                    source_receipt=source,
                    binding=binding,
                    max_positions=max_positions,
                    max_history=max_history,
                )
                semantic = json.loads(episode.supervision.annotations_json)
                semantic["semantic_geometry"] = {
                    "schema": "semantic-native-geometry/8",
                    "maxHistory": max_history,
                    "maxPositions": max_positions,
                }
                semantic["semantic_execution"].update(
                    {
                        "continuation": "teacher_conditioned",
                        "admission": "admitted",
                        "targetEligibility": "source_admitted",
                        "modelExecutionOutcome": "unmeasured",
                        "sourceId": member["id"],
                        "componentId": component["id"],
                        "members": component["members"],
                        "split": component["split"],
                        "registryDigest": registry["digest"],
                        "datasetBinding": binding,
                        "sourceAdmission": source,
                        "nativeAdmission": native,
                        "tokenizerEvidence": tokenizer_evidence,
                        "nativeTrainingDigest": training["nativeTrainingDigest"],
                        "publicArtifact": {
                            "public": document["public"],
                            "contexts": document["contexts"],
                            **({"sidecar": document["sidecar"]} if has_orchestration else {}),
                        },
                    }
                )
                # Eligibility is a property of the target evidence; model success
                # remains unmeasured until a separate learned-execution study.
                records = tuple(
                    replace(record, provenance="teacher") if record.policy_mask else record
                    for record in episode.records
                )
                episode = replace(
                    episode,
                    records=records,
                    lineage=replace(
                        episode.lineage,
                        family_id=component["id"],
                        ancestor_ids=tuple(
                            "semantic-family:" + name for name in component["members"]
                        ),
                        frozen_split=component["split"],
                    ),
                    supervision=replace(
                        episode.supervision,
                        annotations_json=json.dumps(semantic, separators=(",", ":")),
                    ),
                )
                store.add(episode, codec)
                if training.get("orchestrationRecords") is not None:
                    counts["orchestrationRecords"] += training["orchestrationRecords"]
                counts["admitted"] += 1
            except (ValueError, KeyError, TypeError) as error:
                counts["refused"] += 1
                refusals.append({"candidate": index, "reason": str(error)})
        if counts["admitted"] == 0:
            write_json(
                root / "semantic-v8-report.json",
                {
                    "schema": "semantic-corpus-report/8",
                    "counts": dict(counts),
                    "refusals": refusals,
                    "optimizerSteps": 0,
                },
            )
            raise ValueError("No source/native-admitted V8 examples; refusal report retained")
        store.freeze()
        for row in store.accepted_rows():
            if row["split"] != json.loads(row["episode_json"])["lineage"]["frozen_split"]:
                raise ValueError("Deduplication changed an immutable V8 family split")
        pins = {
            "schema": "semantic-preparation/8",
            "representation": "mixed",
            "producerRevision": producer_revision,
            "consumerRevision": consumer_revision,
            "contractDigest": CONTRACT_DIGEST,
            "fixtureDigest": FIXTURE_DIGEST,
            "tokenizerIdentity": codec.fingerprint,
            "tokenizerSha256": LING_TOKENIZER_SHA256,
            "registryDigest": registry["digest"],
            "maxPositions": max_positions,
            "maxHistory": max_history,
            "optimizerSteps": 0,
            "learningClaim": None,
        }
        report = {**pins, "counts": dict(counts), "refusals": refusals}
        manifest = export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=[pins],
            pipeline_fingerprint=stable_algorithm_hash(pins),
            chunking=ChunkingConfig(
                target_tokens=min(512, max_positions),
                soft_max_tokens=min(768, max_positions),
                hard_max_tokens=max_positions,
            ),
            sidecars={"semantic-v8-report.json": report, "semantic-v8-families.json": registry},
        )
    return manifest, report
