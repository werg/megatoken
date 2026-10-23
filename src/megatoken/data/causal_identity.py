"""Training equivalence is stricter than transcript equality or family overlap."""

from __future__ import annotations

import json

from megatoken.io import content_hash


def training_contract_hash(episode):
    """Normalize opaque identities and physical cuts, retain causal differences.

    Transcript hashes remain separate contamination keys. This key distinguishes
    co-selected peers from successive outputs, changed controls, evidence links,
    and loss masks without making arbitrary addresses into semantic features.
    """
    logical = list(episode.iter_logical_records())
    record_indices = {r.record_id: i for i, records in enumerate(logical) for r in records}
    tracks, purposes = {}, {t.track_id: t.description for t in episode.tracks}

    def track(address):
        return tracks.setdefault(address, len(tracks))

    groups, event_contracts = [], {}
    if episode.meridians:
        for group in episode.meridians:
            members = []
            for event in sorted(group.events, key=lambda e: record_indices[e.record_id]):
                index = record_indices[event.record_id]
                members.append(index)
                contract = (
                    track(event.track_id),
                    event.modality,
                    event.role,
                    event.instruction,
                    json.loads(event.settings_json),
                    tuple(record_indices[ref] for ref in event.reference_event_ids),
                )
                earlier = event_contracts.setdefault(index, [])
                if not earlier or earlier[-1] != contract:
                    earlier.append(contract)
            members = tuple(sorted(set(members)))
            # Consecutive physical parts of one logical event do not make a
            # different task. Multiple distinct logical events always do.
            if not (len(members) == 1 and groups and groups[-1] == members):
                groups.append(members)
    else:
        for index, records in enumerate(logical):
            record = records[0]
            groups.append((index,))
            event_contracts[index] = [(track(record.role), "text", record.role, "", {}, ())]

    scoring = []
    for records in logical:
        offset, spans = 0, []
        emitted = []
        for record in records:
            if record.policy_mask:
                for left, right in (
                    [(0, len(record.payload))]
                    if record.free_spans is None
                    else [(span.start, span.end) for span in record.free_spans]
                ):
                    left, right = offset + left, offset + right
                    if spans and spans[-1][1] == left:
                        spans[-1] = (spans[-1][0], right)
                    else:
                        spans.append((left, right))
            if record.emitted_ids is not None:
                emitted.extend(
                    zip(
                        record.emitted_ids,
                        record.emitted_free_mask or (record.policy_mask,) * len(record.emitted_ids),
                        strict=True,
                    )
                )
            offset += len(record.payload)
        scoring.append(
            (
                records[0].role,
                "all" if spans == [(0, offset)] else spans,
                emitted,
                records[0].media_type.partition(";")[0],
            )
        )

    # Dependencies duplicated by physical splitting collapse to their logical
    # endpoints. An internal physical continuation is not a new dependency.
    dependencies = sorted(
        {
            (record_indices[d.writer_id], record_indices[d.reader_id], d.kind)
            for d in episode.dependencies
            if record_indices[d.writer_id] != record_indices[d.reader_id]
        }
    )
    return content_hash(
        {
            "format": "causal-training-equivalence/v1",
            "groups": groups,
            "events": sorted(event_contracts.items()),
            "track_purposes": sorted(
                (index, purposes.get(address, "")) for address, index in tracks.items()
            ),
            "scoring": scoring,
            "dependencies": dependencies,
            "revisions": [
                (record_indices[r.record_id], track(r.track_id), r.purpose, r.revision)
                for r in episode.track_revisions
            ],
            "seed_hex": episode.control_seed.hex(),
            "reference_hex": episode.control_reference.hex(),
            "tools": json.loads(episode.tools_json),
            "relations": [
                (r.kind, record_indices[r.left], record_indices[r.right], r.weight)
                for r in episode.omni_relations
            ],
            "annotated_relations": [
                {
                    **r,
                    "left": record_indices.get(r.get("left"), r.get("left")),
                    "right": record_indices.get(r.get("right"), r.get("right")),
                }
                for r in json.loads(episode.supervision.annotations_json).get("relations", [])
            ],
            "native_selected_decisions": json.loads(episode.supervision.annotations_json).get(
                "native_action_tapes", []
            ),
        }
    )


def equivalence_key(variant):
    if (
        not isinstance(variant, str)
        or not variant.startswith("causal:")
        or ":variant:" not in variant
    ):
        raise ValueError("Missing current causal training identity; reprepare the source")
    return variant.split(":variant:", 1)[0]
