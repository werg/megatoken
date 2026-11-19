"""Independent v7 execution admission over the actual native record path."""

from __future__ import annotations

import json
import re
from dataclasses import asdict

from megatoken.controls import canonical_semantic_observation
from megatoken.generation.codec import LING_TOKENIZER_SHA256
from megatoken.model.context import ContextPolicy, retained_context_events

from .algorithm_v7 import stable_algorithm_hash
from .context import context_announcement, validate_context_announcements
from .host_operations import CONTRACT_DIGEST
from .schema import encode_record


def adapt_semantic_v7(document, **options):
    public = document.get("public", {})
    if (
        public.get("schema") != "semantic-execution/7"
        or public.get("traceFormat") != "semantic-trace/7"
        or public.get("contextFormat") != "semantic-context/7"
    ):
        raise ValueError("Only newly generated semantic execution/7 artifacts are accepted")
    if public.get("representation") == "algorithm":
        from .algorithm_harness_v7 import adapt_algorithm_harness_v7

        return adapt_algorithm_harness_v7(document, **options)
    if public.get("representation") == "detailed":
        from .detailed_harness_v7 import adapt_detailed_harness_v7

        return adapt_detailed_harness_v7(document, **options)
    raise ValueError("An explicit algorithm or detailed v7 representation is required")


def _json(payload):
    if not payload.startswith(b"{"):
        return None
    value = json.loads(payload)
    return value if isinstance(value, dict) else None


def native_training_digest_v7(episode):
    """Bind exact native bytes, grouping, controls, ownership and physical cuts."""
    body = episode.to_dict()
    body.pop("lineage", None)
    annotations = json.loads(episode.supervision.annotations_json)
    body["supervision"] = {
        "modelAnnotations": {
            key: annotations[key] for key in ("session_metadata", "semantic_geometry")
        }
    }
    for record in body["records"]:
        record.pop("provenance", None)
    return stable_algorithm_hash(json.loads(json.dumps(body)))


def validate_native_visibility_v7(episode):
    """Apply serving's exact event retention and visibility before each fragment.

    A full detailed context supersedes its earlier prefix. Algorithm references
    additionally require a currently visible complete definition/input copy;
    an importer cache is never treated as model-visible text.
    """
    annotations = json.loads(episode.supervision.annotations_json)
    semantic = annotations.get("semantic_execution", {})
    geometry = annotations.get("semantic_geometry", {})
    if semantic.get("schema") != "semantic-execution/7" or semantic.get("representation") not in {
        "algorithm",
        "detailed",
    }:
        raise ValueError("Native v7 visibility requires a v7 episode annotation")
    if (
        set(geometry) != {"schema", "maxHistory", "maxPositions"}
        or geometry["schema"] != "semantic-native-geometry/7"
        or any(
            type(geometry[key]) is not int or geometry[key] < 1
            for key in ("maxHistory", "maxPositions")
        )
    ):
        raise ValueError("Native v7 visibility requires its exact physical geometry")
    if any(len(group.events) != 1 for group in episode.meridians):
        raise ValueError("V7 physical fragments require ordered atomic meridians")
    validate_context_announcements(episode)
    events = {event.record_id: event for group in episode.meridians for event in group.events}
    tracks, positions, identities = [], [], []
    policy, seen = ContextPolicy(), 0
    definitions, inputs, latest = {}, {}, {}
    contexts = generated = announcements = 0

    def append(part, track):
        nonlocal tracks, positions, identities, seen
        kept, _, _ = retained_context_events(
            policy=policy,
            tracks=tracks,
            positions=positions,
            event_ids=identities,
            meridian_sizes=(1,) * len(identities),
            seen=seen,
            new_tracks=(track,),
            budget=geometry["maxHistory"],
        )
        tracks = [tracks[index] for index in kept] + [track]
        positions = [positions[index] for index in kept] + [seen]
        identities = [identities[index] for index in kept] + [part.record_id]
        seen += 1

    for parts in episode.iter_logical_records():
        payload = b"".join(part.payload for part in parts)
        item = _json(payload) if not parts[0].policy_mask else None
        track = events[parts[0].record_id].track_id
        if any(events[part.record_id].track_id != track for part in parts):
            raise ValueError("V7 fragments changed their owning track")
        ids = {part.record_id for part in parts}
        window = context_announcement(payload, parts[0].role)
        if window is not None:
            if len(parts) != 1 or parts[0].policy_mask:
                raise ValueError("V7 context policy must remain one unscored observation")
            policy = policy.configure(track, **window)
            announcements += 1
        if item and item.get("schema") == "semantic-context/7":
            if parts[0].role != "system" or any(part.policy_mask for part in parts):
                raise ValueError("V7 current public context acquired generation authority")
            if item.get("representation") != semantic["representation"]:
                raise ValueError("Native context representation differs from its target")
            required = [ids]
            if semantic["representation"] == "algorithm":
                if "program" in item:
                    definitions.setdefault(item["definition"], []).append(ids)
                if "inputGraph" in item:
                    inputs.setdefault(item["input"], []).append(ids)
                if item["definition"] not in definitions or item["input"] not in inputs:
                    raise ValueError("Referenced v7 source/input was never publicly observed")
                required.extend((definitions[item["definition"]], inputs[item["input"]]))
            latest[track] = required
            contexts += 1
        if item and semantic["representation"] == "algorithm" and item.get("kind") == "program":
            program = item.get("program", {})
            definitions.setdefault(program.get("definitionDigest"), []).append(ids)
        if item and semantic["representation"] == "algorithm" and item.get("kind") == "input":
            inputs.setdefault(stable_algorithm_hash(item.get("graph")), []).append(ids)
        previous_fragments = set()
        for part in parts:
            if part.policy_mask:
                if track not in latest:
                    raise ValueError(
                        "Every v7 selection needs its own earlier exact public context"
                    )
                allowed = policy.visible(track, tracks, positions, identities, seen)
                available = {
                    identity
                    for identity, visible in zip(identities, allowed, strict=True)
                    if visible
                }
                requirements = latest[track]
                if not requirements[0] <= available or not previous_fragments <= available:
                    raise ValueError(
                        "V7 selection cannot see all current context or earlier fragments"
                    )
                if any(
                    not any(copy <= available for copy in copies) for copies in requirements[1:]
                ):
                    raise ValueError(
                        "V7 referenced immutable definition/input was evicted from actual model visibility"
                    )
                previous_fragments.add(part.record_id)
            append(part, track)
        if parts[0].policy_mask:
            generated += 1
            latest.pop(track)
    if not generated or generated != contexts:
        raise ValueError("V7 generated labels and current public contexts are not one-to-one")
    return {
        "contexts": contexts,
        "generatedRecords": generated,
        "contextAnnouncements": announcements,
    }


def _validate_exact_transport(document, episode):
    contexts, logical = [], []
    for parts in episode.iter_logical_records():
        payload = b"".join(part.payload for part in parts)
        item = _json(payload) if not parts[0].policy_mask else None
        if item and item.get("schema") == "semantic-context/7":
            contexts.append(payload)
            continue
        if parts[0].logical_id.startswith(("semantic:context:", "semantic:lifecycle:")):
            continue
        logical.append(parts)
    if contexts != [canonical_semantic_observation(context) for context in document["contexts"]]:
        raise ValueError("Native model contexts differ from exact current public contexts")
    timeline = document["public"]["timeline"]
    if len(logical) != len(timeline):
        raise ValueError("V7 physical framing changed logical execution boundaries")
    algorithm = document["public"]["representation"] == "algorithm"
    for row, parts in zip(timeline, logical, strict=True):
        generated = row["kind"] == "semantic"
        raw_line = generated or algorithm and row["kind"] == "observation"
        expected = row["line"].encode("utf-8") if raw_line else canonical_semantic_observation(row)
        if b"".join(part.payload for part in parts) != expected or any(
            part.policy_mask != generated for part in parts
        ):
            raise ValueError("V7 transport rewrote selected/observed bytes or their loss authority")
    positions = {record.record_id: index for index, record in enumerate(episode.records)}
    for edge in episode.dependencies:
        if positions[edge.writer_id] >= positions[edge.reader_id]:
            raise ValueError("Future data became visible to an earlier v7 prediction")
    for action in episode.actions:
        last = max(positions[name] for name in action.record_ids)
        if any(
            positions[record.record_id] <= last
            for record in episode.records
            if record.result_of == action.action_id
        ):
            raise ValueError("A v7 request can see its own uncommitted result")


def _validate_continuations(document):
    if document["public"]["representation"] == "algorithm":
        from .algorithm_context_v7 import restore_algorithm_continuation_v7
        from .algorithm_harness_v7 import AlgorithmReplayV7

        timeline = document["public"]["timeline"]
        program = timeline[0]["program"]
        replay = AlgorithmReplayV7(
            program,
            timeline[1]["graph"],
            capabilities=list(dict.fromkeys(row["operation"] for row in program["boundaries"])),
            invocation_id=document["contexts"][0]["invocationId"],
        )
        checked = 0
        for row in timeline[2:]:
            if row["kind"] == "semantic":
                replay.commit(row["line"])
            elif row["kind"] == "observation":
                replay.observe(row["line"])
            else:
                continue
            restored = restore_algorithm_continuation_v7(replay.checkpoint())
            if stable_algorithm_hash(restored.live_state()) != stable_algorithm_hash(
                replay.live_state()
            ):
                raise ValueError("V7 continuation changed independently replayed state")
            checked += 1
        return checked
    from .detailed_context_v7 import validate_detailed_continuation_v7

    contexts = iter(document["contexts"])
    timeline = document["public"]["timeline"]
    for index, target in enumerate(timeline):
        if target["kind"] == "semantic":
            validate_detailed_continuation_v7(
                next(contexts), prior_timeline=timeline[:index], target=target
            )
    return len(document["contexts"])


def native_admission_v7(
    document, *, codec, consumer_revision, max_positions=8192, max_history=1024
):
    from .semantic_contract_v7 import FIXTURE_DIGEST, SCHEMA_DIGEST

    if (
        not isinstance(consumer_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", consumer_revision) is None
    ):
        raise ValueError("V7 native admission requires an exact consumer Git commit")
    if codec.fingerprint != "7f52faca2fe82273a889023defd2f1a6fd6080a21096ec77e97f4798a0a370b2":
        raise ValueError("V7 production admission requires the pinned production tokenizer")
    episode, imported = adapt_semantic_v7(
        document,
        codec=codec,
        max_positions=max_positions,
        max_history=max_history,
        allow_local=True,
        admission_verifier=lambda _: True,
    )
    encoded = tuple(encode_record(record, codec) for record in episode.records)
    if any(record.P for record in encoded if not record.record.policy_mask):
        raise ValueError("Observed v7 source, context or effects acquired generation loss")
    if any(record.tokens > max_positions for record in encoded):
        raise ValueError("V7 physical record exceeded its exact tokenizer-position budget")
    visibility = validate_native_visibility_v7(episode)
    _validate_exact_transport(document, episode)
    continuations = _validate_continuations(document)
    import_receipt = json.loads(json.dumps(asdict(imported)))
    # The native gate does not depend on whether the source gate was previously
    # admitted. Recollection over identical public bytes must be identical.
    import_receipt.pop("admission", None)
    return {
        "schema": "semantic-native-admission/7",
        "componentGate": "native-consumer-only",
        "traceFormat": "semantic-trace/7",
        "representation": document["public"]["representation"],
        "episodeDigest": stable_algorithm_hash(document["public"]),
        "contextsDigest": stable_algorithm_hash(document["contexts"]),
        "nativeTrainingDigest": native_training_digest_v7(episode),
        "datasetBindingDigest": stable_algorithm_hash(document["sidecar"]["datasetBinding"])
        if "datasetBinding" in document["sidecar"]
        else None,
        "consumerRevision": consumer_revision,
        "schemaDigest": SCHEMA_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "contractDigest": CONTRACT_DIGEST,
        "tokenizerIdentity": codec.fingerprint,
        "tokenizerSha256": LING_TOKENIZER_SHA256,
        "generatedTokens": sum(record.tokens for record in encoded if record.record.policy_mask),
        "observedTokens": sum(record.tokens for record in encoded if not record.record.policy_mask),
        "physicalRecords": len(episode.records),
        "actions": len(episode.actions),
        "lossMaskValidated": True,
        "visibilityValidated": True,
        "fragmentationValidated": True,
        "contextValidated": True,
        "continuationValidated": True,
        "admitted": True,
        "optimizerSteps": 0,
        "visibility": visibility,
        "continuations": continuations,
        "importReceipt": import_receipt,
    }
