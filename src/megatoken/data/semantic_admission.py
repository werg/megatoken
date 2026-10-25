"""Non-optimizing native admission checks for complete v4 interaction episodes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec

from .host_operations import CONTRACT_DIGEST, canonical_json
from .schema import encode_record
from .semantic_contract import FIXTURE_DIGEST, SCHEMA_DIGEST
from .semantic_harness_v4 import adapt_selected_harness_v4


def validate_native_visibility(episode):
    """Run the production context visibility and bounded resident-history rules."""
    from megatoken.model.context import ContextPolicy, retained_context_events

    from .context import context_announcement, validate_context_announcements
    from .witness_v4 import parse_record

    validate_context_announcements(episode)
    geometry = json.loads(episode.supervision.annotations_json).get("semantic_geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("schema") != "semantic-native-geometry/4"
        or type(geometry.get("maxHistory")) is not int
        or geometry["maxHistory"] < 1
    ):
        raise ValueError("Native visibility requires its declared physical history geometry")
    if any(len(group.events) != 1 for group in episode.meridians):
        raise ValueError("Semantic transport requires atomic ordered physical meridians")
    budget = geometry["maxHistory"]
    policy = ContextPolicy()
    events = {event.record_id: event for group in episode.meridians for event in group.events}
    tracks, positions, identities = [], [], []
    programs, invocations, active, rereads, realm_history, snapshots = {}, {}, {}, {}, {}, {}
    coordinators, seen, transport = set(), 0, 0

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
            budget=budget,
        )
        tracks = [tracks[index] for index in kept] + [track]
        positions = [positions[index] for index in kept] + [seen]
        identities = [identities[index] for index in kept] + [part.record_id]
        seen += 1

    for parts in episode.iter_logical_records():
        payload = b"".join(part.payload for part in parts)
        track = events[parts[0].record_id].track_id
        owner = active.get(track)
        context = parts[0].logical_id.startswith("semantic:context:")
        state_refresh = False
        lifecycle = False
        if context:
            if len(parts) != 1 or parts[0].policy_mask:
                raise ValueError("Native context receipt must be one atomic observed event")
            window = context_announcement(payload, parts[0].role)
            if window is None:
                raise ValueError("Native source visibility receipt lost its context policy")
            policy = policy.configure(track, **window)
            transport += 1
        elif parts[0].policy_mask:
            if payload.startswith(b"{"):
                action = json.loads(payload)
                if action.get("schema") != "semantic-outcome/4":
                    if not action.get("name", "").startswith("track_") or track not in coordinators:
                        raise ValueError(
                            "Lifecycle teaching requires prior observed coordinator policy"
                        )
                    lifecycle = True
            else:
                _, fields = parse_record(payload.decode())
                if fields.get("invocation") in invocations:
                    owner = fields["invocation"]
                    active[track] = owner
        else:
            item = json.loads(payload)
            if item.get("schema") == "semantic-orchestration-input/4":
                coordinators.add(track)
            if item.get("kind") == "program":
                programs[item["programId"]] = tuple(part.record_id for part in parts)
            elif item.get("kind") == "invocation":
                invocations[item["invocationId"]] = item
                active[track] = owner = item["invocationId"]
            elif item.get("kind") == "source-read":
                rereads[(track, item["programId"])] = tuple(part.record_id for part in parts)
            elif item.get("kind") == "model-context" and "selectedState" in item:
                active[track] = owner = item["invocationId"]
                state_refresh = True
            if item.get("invocationId") in invocations:
                owner = item["invocationId"]
        realm = invocations[owner]["realmId"] if owner in invocations else None
        for part in parts:
            if part.policy_mask and not lifecycle:
                invocation = invocations[owner]
                source_ids = rereads.get(
                    (track, invocation["programId"]), programs[invocation["programId"]]
                )
                allowed = policy.visible(track, tracks, positions, identities, seen)
                available = {
                    identity
                    for identity, visible in zip(identities, allowed, strict=True)
                    if visible
                }
                if not set(source_ids) <= available:
                    raise ValueError(
                        "Generated native action cannot see all of its exact source context"
                    )
                checkpoint = snapshots.get(owner)
                required = set(checkpoint["records"]) if checkpoint is not None else set()
                required.update(
                    identity
                    for identity, position in realm_history.get(realm, ())
                    if checkpoint is None or position > checkpoint["cutoff"]
                )
                if not required <= available:
                    raise ValueError(
                        "Generated native action cannot see its earlier committed realm state"
                    )
            position = seen
            append(part, track)
            if realm is not None and not context:
                realm_history.setdefault(realm, []).append((part.record_id, position))
        if state_refresh:
            snapshots[owner] = {
                "records": tuple(part.record_id for part in parts),
                "cutoff": seen - 1,
            }
    return transport


def native_admission(
    document, *, codec, consumer_revision: str, max_positions=1024, max_history=1024
):
    """Verify the native-consumer gate; never certify the independent JS oracle.

    The producer calls this before marking the complete example admitted. Its
    private source/reference certificates remain separate from prediction data.
    """
    if not isinstance(consumer_revision, str) or not consumer_revision:
        raise ValueError("Native admission requires exact consumer revision provenance")
    if codec.fingerprint == ByteCodec.byte_fallback().fingerprint:
        raise ValueError("Production native admission requires the pinned production tokenizer")
    episode, receipt = adapt_selected_harness_v4(
        document,
        codec=codec,
        max_positions=max_positions,
        max_history=max_history,
        allow_local=True,
        # This verifier admits native geometry only. Source/reference authority
        # is deliberately not inferred from a producer's status string.
        admission_verifier=lambda _: True,
    )
    encoded = tuple(encode_record(record, codec) for record in episode.records)
    if any(record.P for record in encoded if not record.record.policy_mask):
        raise ValueError("Observed source, effects, events or receipts acquired generation loss")
    if any(record.tokens > max_positions for record in encoded):
        raise ValueError("Native physical framing exceeded its tokenizer-position budget")
    transport_records = validate_native_visibility(episode)
    logical = [
        parts
        for parts in episode.iter_logical_records()
        if not parts[0].logical_id.startswith("semantic:context:")
    ]
    timeline = document["public"]["timeline"]
    if len(logical) != len(timeline):
        raise ValueError("Physical fragmentation changed logical semantic boundaries")
    for item, parts in zip(timeline, logical, strict=True):
        expected = (
            item["line"].encode()
            if item["kind"] == "semantic"
            else json.dumps(
                item, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        if b"".join(part.payload for part in parts) != expected:
            raise ValueError("Native transport changed exact selected or observed bytes")
        if any(part.policy_mask != (item["kind"] == "semantic") for part in parts):
            raise ValueError("Logical action ownership changed across physical fragments")
    positions = {record.record_id: index for index, record in enumerate(episode.records)}
    for edge in episode.dependencies:
        if positions[edge.writer_id] >= positions[edge.reader_id]:
            raise ValueError("Future observation became visible to an earlier prediction")
    for action in episode.actions:
        last = max(positions[name] for name in action.record_ids)
        if any(
            positions[record.record_id] <= last
            for record in episode.records
            if record.result_of == action.action_id
        ):
            raise ValueError("An action can see its own uncommitted result")
    return {
        "schema": "semantic-native-admission/4",
        "componentGate": "native-consumer-only",
        "episodeDigest": hashlib.sha256(canonical_json(document["public"]).encode()).hexdigest(),
        "consumerRevision": consumer_revision,
        "contractDigest": CONTRACT_DIGEST,
        "schemaDigest": SCHEMA_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "tokenizerIdentity": codec.fingerprint,
        "tokenizerSha256": LING_TOKENIZER_SHA256,
        "generatedTokens": sum(record.tokens for record in encoded if record.record.policy_mask),
        "observedTokens": sum(record.tokens for record in encoded if not record.record.policy_mask),
        "physicalRecords": len(episode.records),
        "contextTransportRecords": transport_records,
        "logicalRecords": len(timeline),
        "actions": len(episode.actions),
        "lossMaskValidated": True,
        "visibilityValidated": True,
        "fragmentationValidated": True,
        "admitted": True,
        "optimizerSteps": 0,
        "importReceipt": asdict(receipt),
    }
