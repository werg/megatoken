"""Admission of locally supplied, executed full-duplex traces.

Inputs are native ``Episode`` JSONL plus the immutable tensor-asset root that
owns their media descriptors. This importer verifies real execution results,
clocked overlapping speaker audio, source readiness, and every supplied native
action tape before using the regular family split, deduplication, and export
path. It does not approve source terms or fetch source data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from megatoken.io import content_hash

from .dedup import DedupConfig, DedupStore
from .media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAsset, TensorAssetStore
from .schema import Episode, canonical_json
from .sources import TermsReview
from .storage import PreparedDataset, export_prepared


def _reviewed_terms(value: TermsReview | dict[str, Any]) -> TermsReview:
    terms = value if isinstance(value, TermsReview) else TermsReview(**value)
    terms.require_approval()
    if not terms.license or terms.license == "unreviewed":
        raise ValueError("Execution-trace terms need an approved, recorded source license")
    return terms


def _copy_asset(asset: TensorAsset, source: TensorAssetStore, target: TensorAssetStore) -> None:
    source_root = source.root.resolve()
    path = source.root / asset.relative_path
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(source_root):
        raise ValueError("Referenced media tensor escapes the supplied source asset root")
    values = source.load(asset)
    metadata = asdict(asset)
    metadata.pop("sha256")
    metadata.pop("shape")
    copied = target.put(values, **metadata)
    if copied != asset:
        raise ValueError("Copied media tensor changed its checksum or source-family identity")


def _copy_episode_assets(
    episode: Episode, source: TensorAssetStore, target: TensorAssetStore
) -> None:
    copied: set[tuple[str, str]] = set()
    for record in episode.records:
        if record.media_type != MEDIA_RECORD_TYPE:
            continue
        media = MediaRecord.from_bytes(record.payload)
        for asset in (media.source, media.embedding, media.target, media.rendered):
            if asset is None:
                continue
            key = (asset.sha256, asset.kind)
            if key not in copied:
                _copy_asset(asset, source, target)
                copied.add(key)


def _validate_action_results(episode: Episode) -> None:
    records = {record.record_id: record for record in episode.records}
    executed = {action.action_id: action for action in episode.actions if action.execution_id}
    if not executed:
        raise ValueError("Duplex trace needs at least one tool action with an execution receipt")
    results = [
        record
        for record in episode.records
        if record.result_of in executed
        and record.execution_id == executed[record.result_of].execution_id
        and record.role in {"tool", "environment"}
        and record.provenance == "executed"
    ]
    if not results:
        raise ValueError(
            "Duplex trace needs an executed tool/environment result linked to its real action"
        )
    for action in executed.values():
        if not any(record_id in records for record_id in action.record_ids):
            raise ValueError("Executed action has no physical call record")


def _validate_duplex_media(episode: Episode) -> None:
    records = {record.record_id: record for record in episode.records}
    audio_events = []
    ready = False
    for meridian in episode.meridians:
        for event in meridian.events:
            if event.modality != "audio" or not (
                event.direction == "incoming" or event.role == "user"
            ):
                continue
            if event.source_revision and event.available_at_ns is not None:
                ready = True
            record = records[event.record_id]
            if record.media_type != MEDIA_RECORD_TYPE:
                raise ValueError("Duplex audio events need canonical pinned media descriptors")
            media = MediaRecord.from_bytes(record.payload)
            if media.modality != "audio":
                raise ValueError("Audio event modality differs from its checked media descriptor")
            if event.asset_sha256 is not None and event.asset_sha256 != media.source.sha256:
                raise ValueError("Audio event asset digest differs from its checked descriptor")
            span = event.media_span
            if (
                not event.speaker
                or span is None
                or span.start_tick is None
                or span.end_tick is None
                or span.ticks_per_second is None
                or span.start_tick >= span.end_tick
                or not event.source_revision
                or event.available_at_ns is None
            ):
                raise ValueError(
                    "Duplex audio needs a speaker, exact source interval, revision, and readiness time"
                )
            audio_events.append(event)
    if not ready:
        raise ValueError("Duplex trace needs a versioned source publication readiness event")
    for index, left in enumerate(audio_events):
        for right in audio_events[index + 1 :]:
            a, b = left.media_span, right.media_span
            if left.speaker == right.speaker or a.clock_id != b.clock_id:
                continue
            if Fraction(a.start_tick, a.ticks_per_second) < Fraction(
                b.end_tick, b.ticks_per_second
            ) and Fraction(b.start_tick, b.ticks_per_second) < Fraction(
                a.end_tick, a.ticks_per_second
            ):
                return
    raise ValueError("Duplex trace needs overlapping exact audio intervals from distinct speakers")


def _validate_native_tapes(episode: Episode, codec) -> None:
    from megatoken.contracts import ROLE_TO_ID
    from megatoken.model.multitrack import GroupActionTape
    from megatoken.training.outcome import OutcomeCredit

    annotations = json.loads(episode.supervision.annotations_json)
    entries = annotations.get("native_action_tapes", ())
    if not isinstance(entries, list):
        raise ValueError("Native actor tapes must be an ordered list")
    positions = {record.record_id: index for index, record in enumerate(episode.records)}
    events = {event.record_id: event for group in episode.meridians for event in group.events}
    known_tracks = {track.track_id for track in episode.tracks}
    minimum_cursor = 0
    bound_positions: set[int] = set()
    for index, entry in enumerate(entries):
        try:
            tape = GroupActionTape.from_json(canonical_json(entry["tape"]))
            if tape.codec_fingerprint != codec.fingerprint:
                raise ValueError("native tape codec differs from the supplied codec")
            if known_tracks and not set(tape.tracks) <= known_tracks:
                raise ValueError("native tape names an actor track absent from the episode")
            for member, modality in enumerate(tape.modalities):
                if (
                    type(tape.roles[member]) is not int
                    or tape.roles[member] not in ROLE_TO_ID.values()
                    or type(tape.valid[member]) is not bool
                    or type(tape.terminal[member]) is not bool
                ):
                    raise ValueError("native tape has invalid role or outcome decisions")
                if modality == "text" and tape.valid[member]:
                    if codec.decode(tape.emitted_ids[member]) != tape.payloads[member]:
                        raise ValueError("native tape IDs do not reproduce its exact action bytes")
                elif tape.render_status == "rendered" and tape.valid[member]:
                    receipt = json.loads(tape.payloads[member])
                    if (
                        receipt.get("kind") != "media-sha256-v1"
                        or receipt.get("modality") != modality
                        or receipt.get("realizer") != tape.pack_fingerprints[member]
                    ):
                        raise ValueError("native media receipt differs from its selected decoder")
            cursor = entry["record_cursor"]
            if type(cursor) is not int or not 0 <= cursor <= len(episode.records):
                raise ValueError("native tape has an invalid physical record cursor")
            if cursor < minimum_cursor:
                raise ValueError("native tape reorders or repeats a consumed physical record")
            bindings = entry.get("record_ids")
            if bindings is None:
                if len(tape.tracks) != 1:
                    raise ValueError("multi-event native tapes require explicit record_ids")
                owns = tape.render_status == "rendered" and tape.valid[0] and bool(tape.payloads[0])
                if owns and cursor == len(episode.records):
                    raise ValueError("valid native output has no physical action record")
                bindings = [episode.records[cursor].record_id if owns else None]
            if not isinstance(bindings, (list, tuple)) or len(bindings) != len(tape.tracks):
                raise ValueError("native record bindings must identify every selected member")
            bound = []
            for member, record_id in enumerate(bindings):
                if record_id is None:
                    continue
                if tape.render_status != "rendered" or not tape.valid[member]:
                    raise ValueError("invalid or unrendered native output cannot own a record")
                position = positions[record_id]
                if position in bound_positions or position != cursor + len(bound):
                    raise ValueError(
                        "native tape records must be unique and contiguous in member order"
                    )
                record = episode.records[position]
                event = events.get(record_id)
                if (
                    event is None
                    or event.track_id != tape.tracks[member]
                    or event.modality != tape.modalities[member]
                    or tape.roles[member] != ROLE_TO_ID[record.role]
                    or record.role != "assistant"
                    or record.part_index != 0
                    or record.part_count != 1
                ):
                    raise ValueError("native tape member differs from its physical event")
                if tape.modalities[member] == "text":
                    if (
                        record.media_type == MEDIA_RECORD_TYPE
                        or record.payload != tape.payloads[member]
                        or record.emitted_ids != tape.emitted_ids[member]
                        or record.codec_fingerprint != codec.fingerprint
                    ):
                        raise ValueError("native text action differs from its physical record")
                else:
                    if record.media_type != MEDIA_RECORD_TYPE:
                        raise ValueError(
                            "native media action needs its checked physical descriptor"
                        )
                    media = MediaRecord.from_bytes(record.payload)
                    receipt = json.loads(tape.payloads[member])
                    asset = media.rendered
                    if (
                        asset is None
                        or receipt.get("kind") != "media-sha256-v1"
                        or receipt.get("modality") != tape.modalities[member]
                        or receipt.get("realizer") != tape.pack_fingerprints[member]
                        or asset.sha256 != receipt.get("sha256")
                        or list(asset.shape) != receipt.get("shape")
                        or asset.extractor != receipt.get("realizer")
                    ):
                        raise ValueError(
                            "native media tape differs from its actual decoded descriptor"
                        )
                bound.append(position)
                bound_positions.add(position)
            minimum_cursor = cursor + len(bound)
            if entry.get("outcome_credit") is not None:
                credit = OutcomeCredit(**entry["outcome_credit"])
                if tape.policy_context is None or len(tape.policy_context) != tape.meridian_id:
                    raise ValueError("native outcome credit lacks its complete actor context")
                if credit.selected_events != len(tape.tracks):
                    raise ValueError("native outcome work count differs from its selected group")
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError(f"Native action tape {index}: {error}") from error


def _validate_trace(episode: Episode, codec, target_stream: str) -> Episode:
    if episode.stream != target_stream and not episode.supervision.verified_success:
        raise ValueError(
            f"Cannot remap stream {episode.stream!r} to {target_stream!r} without verified success"
        )
    _validate_action_results(episode)
    _validate_duplex_media(episode)
    _validate_native_tapes(episode, codec)
    return episode if episode.stream == target_stream else replace(episode, stream=target_stream)


def prepare_traces(
    manifest: str | Path,
    *,
    source_assets: str | Path,
    output: str | Path,
    source_terms: TermsReview | dict[str, Any],
    codec,
    stream: str = "duplex_execution",
    source_id: str | None = None,
    dedup: DedupConfig | None = None,
    benchmark_quarantine: dict[str, Any] | None = None,
    shard_bytes: int = 128 << 20,
) -> dict[str, Any]:
    """Validate and export real full-duplex executions from native Episode JSONL.

    A source episode already in ``stream`` is admitted only with executed call
    results, overlapping distinct-speaker audio intervals, and versioned media
    readiness. Moving an episode from another stream additionally requires its
    independent verified-success label. Every referenced media tensor is read,
    checksum-verified, and copied into ``output/media`` before export.
    """
    terms = _reviewed_terms(source_terms)
    manifest_path = Path(manifest).resolve()
    source_root, output_root = Path(source_assets), Path(output)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source asset root does not exist: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_store, output_store = TensorAssetStore(source_root), TensorAssetStore(output_root)
    episodes = []
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for line_no, line in enumerate(manifest_path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            episode = Episode.from_dict(json.loads(line))
            episode = _validate_trace(episode, codec, stream)
            if source_id is not None and episode.lineage.source_id != source_id:
                raise ValueError("episode lineage source ID differs from the reviewed source ID")
            _copy_episode_assets(episode, source_store, output_store)
            episodes.append(episode)
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(f"Execution trace {manifest_path}:{line_no}: {error}") from error
    if not episodes:
        raise ValueError("Execution-trace manifest contains no episodes")

    dedup = dedup or DedupConfig()
    quarantine = benchmark_quarantine or {"family_keys": [], "content_sha256": []}
    pipeline_fingerprint = content_hash(
        {
            "manifest_sha256": manifest_sha,
            "codec": codec.fingerprint,
            "terms": asdict(terms),
            "stream": stream,
            "source_id": source_id,
            "dedup": asdict(dedup),
            "benchmark_quarantine": quarantine,
        }
    )
    manifest_receipt = {
        "format": "native-execution-traces/v1",
        "manifest_sha256": manifest_sha,
        "source_id": source_id,
        "stream": stream,
        "terms": asdict(terms),
        "episode_count": len(episodes),
        "source_families": sorted({episode.lineage.family_id for episode in episodes}),
        "asset_checksums": sorted(
            {
                asset.sha256
                for episode in episodes
                for record in episode.records
                if record.media_type == MEDIA_RECORD_TYPE
                for media in (MediaRecord.from_bytes(record.payload),)
                for asset in (media.source, media.embedding, media.target, media.rendered)
                if asset is not None
            }
        ),
    }
    prepared_path = output_root / "manifest.json"
    if prepared_path.exists():
        with PreparedDataset(output_root, split=None) as prepared:
            if prepared.manifest["pipeline_fingerprint"] != pipeline_fingerprint:
                raise ValueError("Execution-trace inputs changed; use a new output root")
            return prepared.manifest

    with DedupStore(output_root / "execution-traces.sqlite", dedup) as store:
        store.quarantine_benchmarks(
            family_keys=quarantine.get("family_keys", ()),
            content_sha256=quarantine.get("content_sha256", ()),
        )
        for episode in episodes:
            store.add(episode, codec)
        store.freeze()
        return export_prepared(
            store,
            output_root,
            codec=codec,
            source_receipts=[manifest_receipt],
            pipeline_fingerprint=pipeline_fingerprint,
            shard_bytes=shard_bytes,
            sidecars={"execution-traces-report.json": manifest_receipt},
        )
